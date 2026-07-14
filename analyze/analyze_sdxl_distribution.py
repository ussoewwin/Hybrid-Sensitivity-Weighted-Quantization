#!/usr/bin/env python3
"""
SDXL UNet layer distribution analyzer for HSWQ.

Produces per-layer statistics and:
  - derive_veto_tunables()       → FP8 / quantize_sdxl_hswq_v2.x
  - derive_veto_tunables_int8()  → INT8 / quantize_sdxl_hswq_v3.0
    (hard VETO fences + mse_* that drive V4 MSE-guided VETO; pack search_low=1.0)

All VETO / V4-link thresholds come from this checkpoint's layer distribution
(no model-name hardcoding).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from safetensors.torch import load_file

# ---------------------------------------------------------------------------
# Architecture classification (key patterns only — no checkpoint names)
# ---------------------------------------------------------------------------

def classify_layer(name: str) -> str:
    """SDXL Linear keys (Comfy/Diffusers; with or without trailing .weight)."""
    base = name[:-7] if name.endswith(".weight") else name
    if base.endswith(".to_q") or base.endswith(".to_k") or base.endswith(".to_v"):
        return "qkv"
    if base.endswith(".attention.qkv") or base.endswith(".attention.q_proj"):
        return "qkv"
    if base.endswith(".to_out.0") or base.endswith(".attention.to_out.0"):
        return "toout"
    if base.endswith(".ff.net.2"):
        return "ff2"
    if ".ff.net.0" in base:
        return "ff0"
    return "other"


_classify_layer_key = classify_layer


def _mad_outlier_pct(tensor: torch.Tensor, zthr: float = 3.0) -> float:
    """Robust outlier fraction (%). Complements abs_max/std for heavy tails.

    Used by INT8 derive_veto_tunables_int8 only; FP8 derive_veto_tunables
    ignores this field. No model-name branches — value is per-layer from weights.
    """
    xf = tensor.detach().float().reshape(-1)
    if xf.numel() == 0:
        return 0.0
    med = xf.median()
    mad = (xf - med).abs().median().clamp_min(1e-12)
    z = (xf - med).abs() / (1.4826 * mad)
    return float((z > zthr).float().mean().item() * 100.0)


def _layer_stats(tensor: torch.Tensor) -> Dict[str, float]:
    flat = tensor.float().flatten()
    abs_flat = flat.abs()
    std = float(flat.std().item()) if flat.numel() > 1 else 0.0
    abs_max = float(abs_flat.max().item())
    if flat.numel() > 1:
        arr = flat.cpu().numpy()
        s = float(arr.std())
        kurt = float(np.mean(((arr - arr.mean()) / s) ** 4) - 3.0) if s > 0 else 0.0
    else:
        kurt = 0.0
    outlier_ratio = float(abs_max / std if std > 0 else 0.0)
    return {
        "kurtosis": kurt,
        "outlier_ratio": outlier_ratio,
        "abs_max": abs_max,
        "mean": float(flat.mean().item()),
        "std": std,
        "mad_outlier_pct": _mad_outlier_pct(tensor),
    }


# ---------------------------------------------------------------------------
# Rank / IQR helpers (integer ranks from n only — no fixed percentile positions)
# ---------------------------------------------------------------------------

def _sorted_pool(values: List[float]) -> List[float]:
    return sorted(float(v) for v in values)


def _rank_index(n: int, num_from_bottom: int) -> int:
    """Index into ascending sorted array; num_from_bottom=0 → max."""
    if n <= 0:
        return 0
    return max(0, min(n - 1, n - 1 - num_from_bottom))


def _quartile_bounds(sorted_asc: List[float]) -> Tuple[float, float, float]:
    n = len(sorted_asc)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        v = sorted_asc[0]
        return v, v, v
    q1_i = n // 4
    med_i = n // 2
    q3_i = (3 * n) // 4
    return sorted_asc[q1_i], sorted_asc[med_i], sorted_asc[q3_i]


def _tukey_upper(sorted_asc: List[float]) -> float:
    q1, _, q3 = _quartile_bounds(sorted_asc)
    iqr = q3 - q1
    return q3 + iqr


def _percentile_asc(sorted_asc: List[float], pct: float) -> float:
    """Percentile on an already-sorted ascending pool (pct in [0, 100])."""
    if not sorted_asc:
        return 0.0
    if len(sorted_asc) == 1:
        return float(sorted_asc[0])
    p = min(max(float(pct), 0.0), 100.0)
    idx = int(round((p / 100.0) * (len(sorted_asc) - 1)))
    idx = min(max(idx, 0), len(sorted_asc) - 1)
    return float(sorted_asc[idx])


def _class_outlier_span(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return max(values) - min(values)


def _rank_fraction(value: float, pool: List[float]) -> float:
    if not pool:
        return 0.0
    return sum(1 for x in pool if x <= value) / len(pool)


def composite_rank_score(
    k: float, o: float, m: float,
    all_k: List[float], all_o: List[float], all_m: List[float],
) -> float:
    """Unitless score in [0, 3] from empirical ranks — no fixed weights."""
    return (
        _rank_fraction(k, all_k)
        + _rank_fraction(o, all_o)
        + _rank_fraction(m, all_m)
    )


# ---------------------------------------------------------------------------
# ff2 auto tunables (profile-only)
# ---------------------------------------------------------------------------

def _derive_ff2_auto_tunables(
    ff2_o: List[float],
    ff2_scores: List[float],
    all_o: List[float],
    class_spans: Dict[str, float],
) -> Dict[str, Any]:
    n = len(ff2_o)
    if n == 0:
        return {
            "ff2_selective_o_cut": 0.0,
            "ff2_selective_score_cut": 0.0,
            "ff2_live_o_cut": 0.0,
            "ff2_auto_full_class": False,
            "ff2_selective_protected_count": 0,
            "ff2_median_outlier": 0.0,
            "ff2_outlier_span": 0.0,
        }

    o_sorted = _sorted_pool(ff2_o)
    s_sorted = _sorted_pool(ff2_scores) if ff2_scores else []
    q1, med, q3 = _quartile_bounds(o_sorted)
    span = o_sorted[-1] - o_sorted[0]
    iqr = q3 - q1

    half = (n + 1) // 2
    sel_o_cut = o_sorted[_rank_index(n, half)]
    sel_s_cut = (
        s_sorted[_rank_index(len(s_sorted), half)]
        if s_sorted
        else 0.0
    )
    live_cut = o_sorted[_rank_index(n, n // 4)]

    n_sel = sum(
        1
        for i, o in enumerate(ff2_o)
        if o >= sel_o_cut
        or (i < len(ff2_scores) and ff2_scores[i] >= sel_s_cut)
    )

    selective_covers_minority = n_sel * 2 <= n
    selective_incomplete = n_sel < n

    other_spans = [s for cls, s in class_spans.items() if cls != "ff2" and s > 0]
    span_is_tightest = bool(other_spans and span <= min(other_spans))
    span_within_iqr = span <= iqr if iqr > 0 else span_is_tightest
    core_above_floor = med > o_sorted[0] and span <= (o_sorted[-1] - med)

    auto_full = n >= 2 and (
        selective_covers_minority
        or selective_incomplete
        or span_is_tightest
        or span_within_iqr
        or core_above_floor
    )

    return {
        "ff2_selective_o_cut": float(sel_o_cut),
        "ff2_selective_score_cut": float(sel_s_cut),
        "ff2_live_o_cut": float(live_cut),
        "ff2_auto_full_class": bool(auto_full),
        "ff2_selective_protected_count": int(n_sel),
        "ff2_median_outlier": float(med),
        "ff2_outlier_span": float(span),
        "ff2_q1_outlier": float(q1),
        "ff2_q3_outlier": float(q3),
        "ff2_iqr_outlier": float(iqr),
    }


def _derive_engine_tunables(
    all_k: List[float],
    all_o: List[float],
    all_m: List[float],
    k_sorted: List[float],
    o_sorted: List[float],
    m_sorted: List[float],
) -> Dict[str, float]:
    """THIS-profile auto-optimal engine knobs (FP8 path).

    Former accommodation boxes are not "deleted into empty". Each is replaced
    by a continuous map from THIS checkpoint's measured pools:

      drift box 0.1..1.0      → (o_q3 - o_med) / o_med
      mse_mult box 1.25..3.0  → 1 + iqr_o / o_q1 (else o_med)
      α/β box 0.5..0.99       → k_med/k_q3 .. 1.0 and o_med/o_q3 .. 1.0
      mse_release half-IQR    → o_q3 / k_q3 / m_q3 (THIS Q3)
    """
    n = len(all_k)
    k_q1, k_med, k_q3 = _quartile_bounds(k_sorted)
    o_q1, o_med, o_q3 = _quartile_bounds(o_sorted)
    m_q1, m_med, m_q3 = _quartile_bounds(m_sorted)
    iqr_k = k_q3 - k_q1
    iqr_o = o_q3 - o_q1
    iqr_m = m_q3 - m_q1

    # Auto-optimal drift: THIS upper-half relative to THIS median.
    if o_med > 0:
        drift_veto_thresh = (o_q3 - o_med) / max(o_med, 1e-9)
    else:
        drift_veto_thresh = 0.0
    drift_score_mult = max(iqr_k + iqr_o + iqr_m, 1.0)

    # Auto-optimal mse_p75_mult: THIS outlier IQR / THIS scale chain.
    o_scale_den = next(
        (x for x in (o_q1, o_med, o_q3, max(o_sorted) if o_sorted else 0.0) if x > 0),
        0.0,
    )
    mse_p75_mult = 1.0 + (iqr_o / max(o_scale_den, 1e-9))

    k_scale = 1.0 / max(k_q3, 1e-9)
    o_scale = 1.0 / max(o_q3, 1e-9)
    m_scale = 1.0 / max(m_q3, 1e-9)
    penalty_cap = iqr_k / max(k_q3, 1e-9)

    # Auto-optimal α/β band: THIS med/Q3 ratios (floor ≤ clip_max always).
    alpha_floor = max(k_scale * k_med, 0.0)
    beta_floor = max(o_scale * o_med, 0.0)
    alpha_clip_max = max(k_scale * k_q3, alpha_floor)
    beta_clip_max = max(o_scale * o_q3, beta_floor)

    o_w_den = next(
        (x for x in (o_q1, o_med, o_q3, max(o_sorted) if o_sorted else 0.0) if x > 0),
        1e-9,
    )
    m_w_den = next(
        (x for x in (m_q1, m_med, m_q3, max(m_sorted) if m_sorted else 0.0) if x > 0),
        1e-9,
    )

    return {
        "drift_veto_thresh": float(drift_veto_thresh),
        "drift_score_mult": float(drift_score_mult),
        # THIS Q3 only — no half-IQR accommodation rewrite of o_med.
        "mse_release_o_min": float(o_q3),
        "mse_release_k_max": float(k_q3),
        "mse_release_m_max": float(m_q3),
        "mse_p75_multiplier": float(mse_p75_mult),
        "k_scale": float(k_scale),
        "o_scale": float(o_scale),
        "m_scale": float(m_scale),
        "k_gray_lo": float(k_q1),
        "k_gray_hi": float(k_q3),
        "o_gray_lo": float(o_q1),
        "o_gray_hi": float(o_q3),
        "m_gray_lo": float(m_q1),
        "m_gray_hi": float(m_q3),
        "search_low_floor": float(m_q1 / max(m_q3, 1e-9)) if m_q3 > 0 else 1.0,
        "search_low_penalty_cap": float(penalty_cap),
        "search_low_clip_max": float(max(o_scale * o_med, 0.0)),
        "search_low_gray_clip_max": float(max(k_scale * k_med, 0.0)),
        "alpha_floor": float(alpha_floor),
        "alpha_clip_max": float(alpha_clip_max),
        "beta_floor": float(beta_floor),
        "beta_clip_max": float(beta_clip_max),
        "ff2_suffix_min_count": max(1, (n + 19) // 20),
        "score_o_weight": float(iqr_o / max(o_w_den, 1e-9)),
        "score_m_weight": float(iqr_m / max(m_w_den, 1e-9)),
    }


# ---------------------------------------------------------------------------
# derive_veto_tunables — single source for analyze + quantize
# ---------------------------------------------------------------------------

def _normalize_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    if "layers" in profile:
        return profile
    if profile and all(isinstance(v, dict) for v in profile.values()):
        return {"layers": profile}
    return profile


def derive_veto_tunables(profile: Dict[str, Any]) -> Dict[str, Any]:
    profile = _normalize_profile(profile)
    layers = profile.get("layers", {})
    if not layers:
        raise ValueError("profile has no layers")

    all_k: List[float] = []
    all_o: List[float] = []
    all_m: List[float] = []
    by_class: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    for name, entry in layers.items():
        k = float(entry.get("kurtosis", 0))
        o = float(entry.get("outlier_ratio", entry.get("abs_max", 0)))
        m = float(entry.get("abs_max", 0))
        all_k.append(k)
        all_o.append(o)
        all_m.append(m)
        cls = classify_layer(name)
        by_class[cls].append({"k": k, "o": o, "m": m, "name": name})

    k_sorted = _sorted_pool(all_k)
    o_sorted = _sorted_pool(all_o)
    m_sorted = _sorted_pool(all_m)

    fences = _derive_hard_veto_fence_bundle(
        all_k, all_o, all_m, by_class, k_sorted, o_sorted, m_sorted
    )
    engine = _derive_engine_tunables(all_k, all_o, all_m, k_sorted, o_sorted, m_sorted)
    return {**fences, **engine}


def derive_veto_tunables_int8(profile: Dict[str, Any]) -> Dict[str, Any]:
    """INT8 VETO + V4-histogram link for SDXL V3.0.

    Per-model auto analysis → continuous engine branch (infinite patterns):
      1) Hard VETO fences from THIS checkpoint (shared fence helper; no FP8
         engine inheritance).
      2) INT8 engine from THIS weight-space pools → V4 histogram MSE-guided VETO.
      3) search_low_* = 1.0 → pack amax is absmax; V4 remains mandatory.

    Hard-VETO kurtosis / magnitude: max(Tukey, THIS P99) so only the right
    tail is Hard VETO — continuous from THIS distribution, no model-name table.
    """
    profile = _normalize_profile(profile)
    profile = _unet_only_profile(profile)
    layers = profile.get("layers", {})
    if not layers:
        raise ValueError("profile has no layers")

    all_k: List[float] = []
    all_o: List[float] = []
    all_m: List[float] = []
    by_class: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for name, entry in layers.items():
        k = float(entry.get("kurtosis", 0))
        o = float(entry.get("outlier_ratio", entry.get("abs_max", 0)))
        m = float(entry.get("abs_max", 0))
        mad = float(entry.get("mad_outlier_pct", entry.get("mad_pct", 0)) or 0)
        all_k.append(k)
        all_o.append(o)
        all_m.append(m)
        cls = classify_layer(name)
        by_class[cls].append({"k": k, "o": o, "m": m, "mad": mad, "name": name})

    k_sorted = _sorted_pool(all_k)
    o_sorted = _sorted_pool(all_o)
    m_sorted = _sorted_pool(all_m)

    # THIS checkpoint fences only — never call derive_veto_tunables (FP8 engine).
    base = _derive_hard_veto_fence_bundle(
        all_k, all_o, all_m, by_class, k_sorted, o_sorted, m_sorted
    )

    # Right-tail Hard VETO for k/m: max(Tukey, P99 of THIS checkpoint).
    k_p99 = _percentile_asc(k_sorted, 99.0)
    m_p99 = _percentile_asc(m_sorted, 99.0)
    base["extreme_kurtosis"] = float(max(float(base["extreme_kurtosis"]), k_p99))
    base["huge_magnitude"] = float(max(float(base["huge_magnitude"]), m_p99))

    int8_engine = _derive_engine_tunables_int8(
        all_k, all_o, all_m, k_sorted, o_sorted, m_sorted
    )
    mad_tunables = _derive_int8_attn_mad_tunables(
        by_class,
        float(base["attn_qkv_outlier"]),
        float(base["attn_toout_outlier"]),
    )
    base.update(int8_engine)
    base.update(mad_tunables)
    base["quant_format"] = "int8_tensorwise"
    return base


def _alpha_auto_from_kurtosis_order(
    k_p50: float, k_p75: float, k_p99: float
) -> float:
    """Kurtosis-order component of Full-SVD×RMS mix (THIS profile only).

    Positive median → classical body/tail ratio. Non-positive median with
    positive P99 → continuous upper-mass share. Forced 0 is reserved for
    a truly empty kurtosis axis — callers blend other THIS axes so SVD is
    not thought-stopped by a single dead median.
    """
    k_p50 = float(k_p50)
    k_p75 = float(k_p75)
    k_p99 = float(k_p99)
    if k_p99 <= 1e-12:
        return 0.0
    if k_p50 > 0.0:
        return float(min(max(k_p50 / k_p99, 0.0), 1.0))
    if k_p75 > 0.0:
        return float(min(max(k_p75 / k_p99, 0.0), 1.0))
    body_gap = float(abs(k_p50))
    upper = float(max(k_p99, 0.0))
    return float(min(max(upper / (upper + body_gap + 1e-12), 0.0), 1.0))


def _alpha_auto_from_this_character(
    *,
    k_p50: float,
    k_p75: float,
    k_p99: float,
    o_p50: float,
    o_p75: float,
    o_p99: float,
    m_p50: float,
    m_p75: float,
    m_p99: float,
    iqr_k: float,
    iqr_o: float,
    iqr_m: float,
) -> float:
    """Full-SVD×RMS mix from THIS multi-axis analyze character (continuous).

    Uses kurtosis order + outlier + magnitude dispersion of THIS checkpoint.
    Axis weights = THIS IQR / THIS scale (same spirit as score_*_weight).
    Infinite pattern space: every checkpoint gets its own alpha_auto; no
    fixed 0.7 / forced-off recipe. When the profile is degenerate (all axes
    flat), alpha_auto may resolve to 0.0 — this only scales the SVD-leverage
    contribution to zero in the mix; it NEVER skips Full-SVD computation
    (philosophy §0/§4/§5: never skip V4). weighted_histogram_mse_v4 always
    runs torch.linalg.svd regardless of alpha so DualMonitor / V4 MSE /
    ranking see real structural leverage, not RMS-only handwave.
    """
    s_k = _alpha_auto_from_kurtosis_order(k_p50, k_p75, k_p99)
    if float(o_p99) > 1e-12:
        s_o = float(min(max(float(iqr_o) / max(float(o_p99), 1e-12), 0.0), 1.0))
    else:
        s_o = 0.0
    if float(m_p99) > 1e-12:
        s_m = float(min(max(float(iqr_m) / max(float(m_p99), 1e-12), 0.0), 1.0))
    else:
        s_m = 0.0

    w_k = float(iqr_k) / max(abs(float(k_p75)), 1e-9)
    w_o = float(iqr_o) / max(float(o_p75), 1e-9)
    w_m = float(iqr_m) / max(float(m_p50), 1e-9)
    w_sum = max(w_k + w_o + w_m, 1e-9)
    alpha = (w_k * s_k + w_o * s_o + w_m * s_m) / w_sum
    return float(min(max(alpha, 0.0), 1.0))


def _derive_engine_tunables_int8(
    all_k: List[float],
    all_o: List[float],
    all_m: List[float],
    k_sorted: List[float],
    o_sorted: List[float],
    m_sorted: List[float],
) -> Dict[str, float]:
    """INT8 engine: THIS-profile auto-optimal knobs linked to V4 histogram VETO.

    Replacement map (delete-without-replace is forbidden — philosophy §2 / §12):

      former drift clip 0.1..1.0     → (o_q3 - o_med) / o_med
      former mse_mult clip 1.25..3.0 → 1 + iqr_o / o_q1 (else o_med)
      former α/β box 0.5..0.99       → k_med/k_q3 .. 1.0 , o_med/o_q3 .. 1.0
      former mse_release half-IQR    → THIS o_q3 / k_q3 / m_q3
      search_low                     → 1.0 (INT8 absmax; V4 still ranks @ absmax)

    Profile stats are weight-space. Do NOT multiply by 127/448.
    """
    n = len(all_k)
    k_q1, k_med, k_q3 = _quartile_bounds(k_sorted)
    o_q1, o_med, o_q3 = _quartile_bounds(o_sorted)
    m_q1, m_med, m_q3 = _quartile_bounds(m_sorted)
    iqr_k = k_q3 - k_q1
    iqr_o = o_q3 - o_q1
    iqr_m = m_q3 - m_q1

    if o_med > 0:
        drift_veto_thresh = (o_q3 - o_med) / max(o_med, 1e-9)
    else:
        drift_veto_thresh = 0.0
    drift_score_mult = max(iqr_k + iqr_o + iqr_m, 1.0)

    o_scale_den = next(
        (x for x in (o_q1, o_med, o_q3, max(o_sorted) if o_sorted else 0.0) if x > 0),
        0.0,
    )
    mse_p75_mult = 1.0 + (iqr_o / max(o_scale_den, 1e-9))

    k_scale = 1.0 / max(k_q3, 1e-9)
    o_scale = 1.0 / max(o_q3, 1e-9)
    m_scale = 1.0 / max(m_q3, 1e-9)

    alpha_floor = max(k_scale * k_med, 0.0)
    beta_floor = max(o_scale * o_med, 0.0)
    alpha_clip_max = max(k_scale * k_q3, alpha_floor)
    beta_clip_max = max(o_scale * o_q3, beta_floor)

    o_w_den = next(
        (x for x in (o_q1, o_med, o_q3, max(o_sorted) if o_sorted else 0.0) if x > 0),
        1e-9,
    )
    m_w_den = next(
        (x for x in (m_q1, m_med, m_q3, max(m_sorted) if m_sorted else 0.0) if x > 0),
        1e-9,
    )
    k_w_den = next(
        (x for x in (k_q1, k_med, k_q3, max(k_sorted) if k_sorted else 0.0) if x > 0),
        1e-9,
    )

    # Full-SVD×RMS mix from THIS multi-axis character (kurtosis∪outlier∪mag).
    k_p50 = float(k_med)
    k_p75 = float(k_q3)
    k_p99 = float(_percentile_asc(k_sorted, 99.0))
    o_p50 = float(o_med)
    o_p75 = float(o_q3)
    o_p99 = float(_percentile_asc(o_sorted, 99.0))
    m_p50 = float(m_med)
    m_p75 = float(m_q3)
    m_p99 = float(_percentile_asc(m_sorted, 99.0))
    alpha_auto = _alpha_auto_from_this_character(
        k_p50=k_p50,
        k_p75=k_p75,
        k_p99=k_p99,
        o_p50=o_p50,
        o_p75=o_p75,
        o_p99=o_p99,
        m_p50=m_p50,
        m_p75=m_p75,
        m_p99=m_p99,
        iqr_k=iqr_k,
        iqr_o=iqr_o,
        iqr_m=iqr_m,
    )

    return {
        "drift_veto_thresh": float(drift_veto_thresh),
        "drift_score_mult": float(drift_score_mult),
        # Weight-space gray-zone gates for V4 histogram MSE release candidates
        # (same units as profile outlier_ratio / abs_max / kurtosis).
        # THIS Q3 only — continuous branch, no half-IQR accommodation.
        "mse_release_o_min": float(o_q3),
        "mse_release_k_max": float(k_q3),
        "mse_release_m_max": float(m_q3),
        "mse_p75_multiplier": float(mse_p75_mult),
        "k_scale": float(k_scale),
        "o_scale": float(o_scale),
        "m_scale": float(m_scale),
        "k_gray_lo": float(k_q1),
        "k_gray_hi": float(k_q3),
        "o_gray_lo": float(o_q1),
        "o_gray_hi": float(o_q3),
        "m_gray_lo": float(m_q1),
        "m_gray_hi": float(m_q3),
        # INT8 pack point = absmax. V4 histogram still drives VETO MSE at that
        # point (v3 search_range (1.0, 1.0) for estimated_mse).
        "search_low_floor": 1.0,
        "search_low_penalty_cap": 0.0,
        "search_low_clip_max": 1.0,
        "search_low_gray_clip_max": 1.0,
        # Continuous α/β from THIS quartiles — no 0.5 / 0.99 accommodation box.
        "alpha_floor": float(alpha_floor),
        "alpha_clip_max": float(alpha_clip_max),
        "beta_floor": float(beta_floor),
        "beta_clip_max": float(beta_clip_max),
        "alpha_auto": float(alpha_auto),
        "ff2_suffix_min_count": max(1, (n + 19) // 20),
        "score_k_weight": float(iqr_k / max(k_w_den, 1e-9)),
        "score_o_weight": float(iqr_o / max(o_w_den, 1e-9)),
        "score_m_weight": float(iqr_m / max(m_w_den, 1e-9)),
    }


def _class_attn_gate_from_entries(
    entries: List[Dict[str, float]], pool_m: List[float]
) -> float:
    if not entries:
        return _tukey_upper(pool_m)
    return _tukey_upper(_sorted_pool([e["m"] for e in entries]))


def _class_outlier_gate_from_entries(
    entries: List[Dict[str, float]], pool_o: List[float]
) -> float:
    if not entries:
        return _tukey_upper(pool_o)
    return _tukey_upper(_sorted_pool([e["o"] for e in entries]))


def _derive_hard_veto_fence_bundle(
    all_k: List[float],
    all_o: List[float],
    all_m: List[float],
    by_class: Dict[str, List[Dict[str, float]]],
    k_sorted: List[float],
    o_sorted: List[float],
    m_sorted: List[float],
) -> Dict[str, Any]:
    """Hard VETO fences + FF2 from THIS checkpoint pools only (no engine keys).

    Shared continuous fence analysis for FP8 and INT8. Engine tunables are
    attached by each format's own _derive_engine_tunables* so INT8 never
    inherits FP8 accommodation soil.
    """
    extreme_k = _tukey_upper(k_sorted)
    extreme_o = _tukey_upper(o_sorted)
    extreme_m = _tukey_upper(m_sorted)
    med_k = k_sorted[len(k_sorted) // 2] if k_sorted else 0.0

    class_spans: Dict[str, float] = {}
    for cls, entries in by_class.items():
        class_spans[cls] = _class_outlier_span([e["o"] for e in entries])

    ff2_entries = by_class.get("ff2", [])
    ff2_o = [e["o"] for e in ff2_entries]
    ff2_scores = [
        composite_rank_score(e["k"], e["o"], e["m"], all_k, all_o, all_m)
        for e in ff2_entries
    ]
    ff2_auto = _derive_ff2_auto_tunables(ff2_o, ff2_scores, all_o, class_spans)

    qkv_gate = _class_attn_gate_from_entries(by_class.get("qkv", []), all_m)
    toout_gate = _class_attn_gate_from_entries(by_class.get("toout", []), all_m)
    ff2_gate = _class_attn_gate_from_entries(by_class.get("ff2", []), all_m)
    qkv_o_gate = _class_outlier_gate_from_entries(by_class.get("qkv", []), all_o)
    toout_o_gate = _class_outlier_gate_from_entries(by_class.get("toout", []), all_o)

    return {
        "extreme_kurtosis": float(extreme_k),
        "extreme_outlier": float(extreme_o),
        "huge_magnitude": float(extreme_m),
        "median_kurtosis": float(med_k),
        "attn_qkv_absmax": float(qkv_gate),
        "attn_qkv_outlier": float(qkv_o_gate),
        "attn_toout_absmax": float(toout_gate),
        "attn_toout_outlier": float(toout_o_gate),
        "attn_ff2_absmax": float(ff2_gate),
        "ff2_outlier_live": float(ff2_auto["ff2_live_o_cut"]),
        "ff2_profile_outlier": float(ff2_auto["ff2_selective_o_cut"]),
        "ff2_profile_score_cutoff": float(ff2_auto["ff2_selective_score_cut"]),
        "ff2_class_count": len(ff2_entries),
        "ff2_class_outlier_span": float(class_spans.get("ff2", 0.0)),
        "class_outlier_spans": {k: float(v) for k, v in class_spans.items()},
        **ff2_auto,
    }


def _mad_continuous_fences_from_positives(
    positives: List[float],
) -> Tuple[float, float, float, float, float]:
    """THIS-pool MAD% → (floor, soft_gap, p99, collapse, iqr).

    Philosophy §1 / §14: auto analysis → infinite-branch auto-optimal.
    Restores the WAI SSIM≥0.98 MAD gate fingerprint continuously:
      hard floor = THIS MAD P75 / Q3   (never tip-as-floor, never fixed 0.59)
      soft       = same Q3            (soft band unused when floor==soft)
      P99        = tip / severity reference only
      Tukey/IQR  = shape fingerprint (collapse) — not the VETO hard gate

    Raising hard to Tukey (q3+iqr) lifted the gate above the body mass and
    dropped measured SSIM to ~0.96 on the same pool that hit floor=Q3≈0.588
    at the 0.98 path (log/wai17_c5582eb…). Different MAD shapes ⇒ different
    THIS Q3 floors (infinite branch), no hardcoded WAI literals.
    """
    mad_sorted = _sorted_pool(positives)
    n = len(positives)
    if n < 4:
        peak = float(max(positives))
        body = float(_safe_percentile(positives, 50.0))
        return peak, body, peak, 1.0, 0.0
    q1, _med, q3_raw = _quartile_bounds(mad_sorted)
    # Prefer P75 for hard gate — matches c5582eb derive_veto_tunables_int8.
    mad_q3 = float(_safe_percentile(positives, 75.0))
    mad_p99 = float(_safe_percentile(positives, 99.0))
    mad_p50 = float(_safe_percentile(positives, 50.0))
    iqr = float(max(q3_raw - q1, 0.0))
    tail_span = float(max(mad_p99 - mad_p50, 1e-12))
    collapse = float(1.0 - min(1.0, iqr / (iqr + tail_span)))
    # Hard = THIS body mass (Q3). P99 / Tukey must not become the VETO floor.
    mad_floor = float(mad_q3)
    mad_soft = float(mad_q3)
    return mad_floor, mad_soft, mad_p99, collapse, iqr


def _mad_tunables_from_positive_samples(
    mad_vals: List[float],
    gap_o_max: float,
) -> Dict[str, float]:
    """THIS-sample MAD% → auto-optimal floor/soft/p99 (any n≥1).

    Deleting the old ×N floor and writing 0.0 when n<4 is forbidden
    (philosophy §0 / 「固定をただ消すな」). Hard floor and soft-gap thresh
    are THIS MAD Q3/P75 (c5582eb 0.98 path). P99 = severity tip only —
    never tip-as-floor, never fixed WAI numbers.

    Zero positive samples → axis off (no invent).
    """
    positives = [float(v) for v in mad_vals if float(v) > 0.0]
    gap = float(max(gap_o_max, 1e-9))
    if not positives:
        return {
            "attn_mad_pct_floor": 0.0,
            "attn_mad_q3": 0.0,
            "attn_mad_p99": 0.0,
            "attn_mad_gap_o_max": gap,
            "attn_mad_from_profile": 0.0,
            "attn_mad_collapse": 0.0,
            "attn_mad_iqr": 0.0,
        }
    mad_floor, mad_soft, mad_p99, collapse, iqr = (
        _mad_continuous_fences_from_positives(positives)
    )
    mad_p99 = float(max(mad_p99, 1e-9))
    return {
        "attn_mad_pct_floor": float(mad_floor),
        "attn_mad_q3": float(mad_soft),
        "attn_mad_p99": mad_p99,
        "attn_mad_gap_o_max": gap,
        "attn_mad_from_profile": 1.0,
        "attn_mad_collapse": float(collapse),
        "attn_mad_iqr": float(iqr),
    }


def _derive_int8_attn_mad_tunables(
    by_class: Dict[str, List[Dict[str, float]]],
    attn_qkv_outlier: float,
    attn_toout_outlier: float,
) -> Dict[str, float]:
    """Auto MAD% VETO floors from this checkpoint's attn distribution.

    abs_max/std (outlier_ratio) misses heavy-tailed attn on some SDXL UNets.
    MAD% fences are derived from the same profile — no per-model constants,
    no checkpoint-name branches. FP8 derive_veto_tunables does not call this.
    """
    attn_entries: List[Dict[str, float]] = []
    for cls in ("qkv", "toout"):
        attn_entries.extend(by_class.get(cls, []))

    mad_vals = [float(e.get("mad", 0.0) or 0.0) for e in attn_entries]
    return _mad_tunables_from_positive_samples(
        mad_vals,
        float(max(attn_qkv_outlier, attn_toout_outlier, 1e-9)),
    )


def int8_fp16_budget_analyze_severity(
    *,
    kurtosis: float,
    outlier_ratio: float,
    abs_max: float,
    tunables: Dict[str, Any],
    is_hard_veto: bool = False,
    layer_name: str = "",
    mad_outlier_pct: float = 0.0,
    profile_score: float = 0.0,
) -> float:
    """INT8-only: analyze-side severity = this checkpoint's danger character.

    FP8 path must NOT call this. Continuous score for
    derive_priority_combinator → int8_fp16_budget_priority. Denominators are
    derive_veto_tunables_int8 fences for THIS model — not a fixed recipe.
    Higher = more FP16-deserving under --fp16_budget_mb.

    Must NOT flatten Hard VETO to a constant (e.g. max(sev, 1.0)): that
    erases relative danger, collapses sev IQR, and makes
    derive_priority_combinator drop w_sev (thinking-stop on the judgment).
    """
    if tunables.get("quant_format") != "int8_tensorwise":
        raise ValueError(
            "int8_fp16_budget_analyze_severity is INT8-only "
            "(require quant_format=int8_tensorwise); FP8 must not call this"
        )

    ek = max(abs(float(tunables.get("extreme_kurtosis", 1e-6))), 1e-6)
    eo = max(float(tunables.get("extreme_outlier", 1e-6)), 1e-6)
    hm = max(float(tunables.get("huge_magnitude", 1e-6)), 1e-6)

    k = float(kurtosis)
    o = float(outlier_ratio)
    m = float(abs_max)
    # Excess over INT8 hard fences (1.0 == at fence). Keep continuous.
    severity = max(o / eo, 0.0) + max(k / ek, 0.0) + max(m / hm, 0.0)

    # Attn-class character from the same INT8 tunables (THIS-model gates).
    name = str(layer_name or "")
    if name.endswith((".to_q", ".to_k", ".to_v")):
        aq = max(float(tunables.get("attn_qkv_absmax", hm)), 1e-6)
        ao = max(float(tunables.get("attn_qkv_outlier", eo)), 1e-6)
        severity += max(m / aq, 0.0) + max(o / ao, 0.0)
    elif name.endswith(".to_out.0"):
        aq = max(float(tunables.get("attn_toout_absmax", hm)), 1e-6)
        ao = max(float(tunables.get("attn_toout_outlier", eo)), 1e-6)
        severity += max(m / aq, 0.0) + max(o / ao, 0.0)

    # MAD character on the SAME continuous scale as o/eo (1.0 == THIS model's
    # heavy MAD). attn_mad_pct_floor is only a VETO GATE (auto-derived).
    # Denominator = THIS-profile attn_mad_p99 from analyze (auto analysis →
    # auto scale). Former ×4 accommodation of q3/floor is forbidden: if p99
    # is missing, MAD axis stays off (0.0) until analyze fills it — never invent.
    mad = float(mad_outlier_pct or 0.0)
    mad_ref = float(tunables.get("attn_mad_p99", 0.0) or 0.0)
    if mad_ref > 0.0 and mad > 0.0:
        severity += max(mad / mad_ref, 0.0)

    # THIS-model composite rank [0, 3] from analyze (empirical ranks of k/o/m).
    # Continuous relative danger inside the checkpoint — not a binary flag.
    ps = max(float(profile_score or 0.0), 0.0)
    if ps > 0.0:
        severity += ps

    # Analyze Hard VETO (fence / key-pattern / structural / MAD / …) is a
    # measured decision for THIS checkpoint. Encode it as +1.0 on the same
    # fence-excess scale (1.0 == crossed the unquantizable decision), ADDED
    # to measured excess — never replace/flatten to a constant.
    if is_hard_veto:
        severity += 1.0
    return float(severity)


def _signed_veto_axis_effect(
    values: List[float],
    is_hard_veto: List[bool],
) -> float:
    """THIS-checkpoint effect size: mean(log1p|VETO) − mean(log1p|non).

    Positive ⇒ higher axis values co-occur with analyze Hard VETO on THIS
    model (axis is danger-aligned). Negative / near-zero ⇒ anti-aligned or
    useless for ranking FP16 keep. Used only to auto-weight continuous
    axes — never to reserve VETO slots or force a fixed sev>sens rule.
    """
    if len(values) != len(is_hard_veto) or len(values) < 4:
        return 0.0
    a = [
        math.log1p(max(float(v), 0.0))
        for v, f in zip(values, is_hard_veto)
        if f
    ]
    b = [
        math.log1p(max(float(v), 0.0))
        for v, f in zip(values, is_hard_veto)
        if not f
    ]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / max(len(a) - 1, 1)
    vb = sum((x - mb) ** 2 for x in b) / max(len(b) - 1, 1)
    # Pooled std (Cohen's d); tiny floor so a constant axis collapses to 0.
    pooled = math.sqrt(
        (va * (len(a) - 1) + vb * (len(b) - 1))
        / max(len(a) + len(b) - 2, 1)
    )
    if pooled < 1e-12:
        return 0.0
    return float((ma - mb) / pooled)


def derive_priority_combinator(
    sens_iqr: float,
    sev_iqr: float,
    mse_iqr: float,
    sens_p50: float,
    sev_p50: float,
    mse_p50: float,
    *,
    sens_vals: Optional[List[float]] = None,
    sev_vals: Optional[List[float]] = None,
    mse_vals: Optional[List[float]] = None,
    is_hard_veto: Optional[List[bool]] = None,
) -> Dict[str, Any]:
    """Derive priority axis weights + refs from THIS checkpoint's 3-axis
    distribution (DualMonitor sens / analyze severity / V4 MSE).

    Continuous weighted geometric mean — NO fixed product, NO fixed
    V4*(1+sev), NO discrete form switch, NO Hard-VETO absolute reservation.

    When per-layer measured triples + analyze Hard VETO masks are provided,
    dispersion (IQR/median) is gated by THIS model's signed VETO alignment
    per axis: anti-aligned axes (high DualMonitor sens / low V4 MSE on
    demoted VETO layers, etc.) fade automatically so auto-optimal ranking
    follows analyze danger character for THIS checkpoint.
    """
    eps = 1e-12
    s_i = max(float(sens_iqr), 0.0)
    v_i = max(float(sev_iqr), 0.0)
    m_i = max(float(mse_iqr), 0.0)

    # Dispersion seed (spread / typical scale) — flat axes → 0.
    d_s = s_i / max(sens_p50, eps) if sens_p50 > 0 else 0.0
    d_v = v_i / max(sev_p50, eps) if sev_p50 > 0 else 0.0
    d_m = m_i / max(mse_p50, eps) if mse_p50 > 0 else 0.0

    align_s = align_v = align_m = None
    form = "weighted_geometric"
    if (
        sens_vals is not None
        and sev_vals is not None
        and mse_vals is not None
        and is_hard_veto is not None
        and len(sens_vals) == len(sev_vals) == len(mse_vals) == len(is_hard_veto)
        and sum(1 for f in is_hard_veto if f) >= 2
        and sum(1 for f in is_hard_veto if not f) >= 2
    ):
        # Auto-optimal axis gate from THIS model: keep only positively
        # VETO-aligned axes (Cohen's d on log1p). Anti-aligned fade to 0.
        raw_s = _signed_veto_axis_effect(list(sens_vals), list(is_hard_veto))
        raw_v = _signed_veto_axis_effect(list(sev_vals), list(is_hard_veto))
        raw_m = _signed_veto_axis_effect(list(mse_vals), list(is_hard_veto))
        align_s = max(raw_s, 0.0)
        align_v = max(raw_v, 0.0)
        align_m = max(raw_m, 0.0)
        if (align_s + align_v + align_m) > eps:
            # dispersion × alignment: noisy but aligned axes still compete;
            # high-IQR anti-aligned DualMonitor sens cannot dominate.
            w_s = d_s * align_s
            w_v = d_v * align_v
            w_m = d_m * align_m
            form = "weighted_geometric_veto_aligned"
        else:
            w_s, w_v, w_m = d_s, d_v, d_m
    else:
        w_s, w_v, w_m = d_s, d_v, d_m

    w_sum = w_s + w_v + w_m
    if w_sum < eps:
        return {
            "form": "uniform",
            "w_sens": 0.0, "w_sev": 0.0, "w_mse": 0.0,
            "sens_ref": max(float(sens_p50), eps),
            "sev_ref": max(float(sev_p50), eps),
            "mse_ref": max(float(mse_p50), eps),
            "align_sens": float(align_s) if align_s is not None else None,
            "align_sev": float(align_v) if align_v is not None else None,
            "align_mse": float(align_m) if align_m is not None else None,
        }
    w_s /= w_sum
    w_v /= w_sum
    w_m /= w_sum
    return {
        "form": form,
        "w_sens": float(w_s),
        "w_sev": float(w_v),
        "w_mse": float(w_m),
        "sens_ref": max(float(sens_p50), eps),
        "sev_ref": max(float(sev_p50), eps),
        "mse_ref": max(float(mse_p50), eps),
        "align_sens": float(align_s) if align_s is not None else None,
        "align_sev": float(align_v) if align_v is not None else None,
        "align_mse": float(align_m) if align_m is not None else None,
    }


# Architectural key-pattern suffixes (structure only — not a KEEP table).
# Sibling DualMonitor under-measure is repaired by continuous THIS-model
# branches below, never by a unified median/geom floor recipe.
_KEYPATTERN_FAMILY_SENS_SUFFIXES = (
    ".upsamplers.0.conv",
    ".downsamplers.0.conv",
    ".conv_in",
    ".conv_out",
)


def _true_median(vals: List[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(float(v) for v in vals)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return 0.5 * float(s[n // 2 - 1] + s[n // 2])


def derive_fp16_infinite_branch_profile(
    measured: List[Tuple[str, float, float, float, int]],
    is_hard_veto: Sequence[bool],
) -> Dict[str, Any]:
    """Derive continuous FP16-ranking branch knobs from THIS measured pool.

    Every knob is a real from THIS checkpoint's DualMonitor / severity / V4
    MSE / Hard-VETO mask. There is no discrete on/off family-floor mode and
    no unified recipe shared across models — differently shaped pools yield
    different knob vectors (infinite continuous branches).
    """
    eps = 1e-30
    if not measured:
        return {
            "cv_sens": 0.0, "cv_sev": 0.0, "cv_mse": 0.0,
            "align_sens": 0.0, "align_sev": 0.0, "align_mse": 0.0,
            "dm_starvation": 0.0,
            "gamma_sibling": 0.0, "gamma_blend": 0.0,
            "mismatch_gain": 0.0,
            "prio_sibling_gamma": 0.0, "prio_blend_gamma": 0.0,
            "sens_ref": eps, "sev_ref": eps, "mse_ref": eps,
            "n_measured": 0, "n_hard_veto": 0,
        }
    if len(is_hard_veto) != len(measured):
        raise ValueError(
            "derive_fp16_infinite_branch_profile: is_hard_veto length "
            f"{len(is_hard_veto)} != measured {len(measured)}"
        )
    sens = [max(float(r[1]), 0.0) for r in measured]
    mse = [max(float(r[2]), 0.0) for r in measured]
    sev = [max(float(r[3]), 0.0) for r in measured]
    veto = [bool(f) for f in is_hard_veto]

    s_pos = [v for v in sens if v > 0.0]
    m_pos = [v for v in mse if v > 0.0]
    s_p50 = _true_median(s_pos) if s_pos else 0.0
    v_p50 = _true_median(sev) if sev else 0.0
    m_p50 = _true_median(m_pos) if m_pos else 0.0
    s_iqr = _robust_iqr(s_pos) if len(s_pos) >= 2 else 0.0
    v_iqr = _robust_iqr(sev) if len(sev) >= 2 else 0.0
    m_iqr = _robust_iqr(m_pos) if len(m_pos) >= 2 else 0.0
    cv_s = float(s_iqr / max(s_p50, eps))
    cv_v = float(v_iqr / max(v_p50, eps))
    cv_m = float(m_iqr / max(m_p50, eps))

    raw_s = _signed_veto_axis_effect(sens, veto)
    raw_v = _signed_veto_axis_effect(sev, veto)
    raw_m = _signed_veto_axis_effect(mse, veto)
    # Keep signs: anti-aligned DualMonitor (negative) drives starvation.
    align_s = float(raw_s)
    align_v = float(raw_v)
    align_m = float(raw_m)
    # DM starvation: analyze danger axes align while DualMonitor does not.
    dm_starvation = float(
        max(max(align_v, 0.0) + max(align_m, 0.0) - max(align_s, 0.0), 0.0)
    )
    # Continuous gammas — each checkpoint gets a unique pair.
    gamma_sibling = float((1.0 + dm_starvation) * (1.0 + cv_s))
    gamma_blend = float(
        (1.0 + dm_starvation) * (1.0 + cv_v + cv_m + max(-align_s, 0.0))
    )
    align_pos_sum = max(align_v, 0.0) + max(align_m, 0.0) + max(align_s, 0.0)
    mismatch_gain = float(
        (max(align_v, 0.0) + max(align_m, 0.0)) / max(align_pos_sum, eps)
    )
    prio_sibling_gamma = float((1.0 + dm_starvation) * (1.0 + cv_v + cv_m))
    prio_blend_gamma = float((1.0 + dm_starvation) * (1.0 + cv_s))

    return {
        "cv_sens": cv_s,
        "cv_sev": cv_v,
        "cv_mse": cv_m,
        "align_sens": align_s,
        "align_sev": align_v,
        "align_mse": align_m,
        "dm_starvation": dm_starvation,
        "gamma_sibling": gamma_sibling,
        "gamma_blend": gamma_blend,
        "mismatch_gain": mismatch_gain,
        "prio_sibling_gamma": prio_sibling_gamma,
        "prio_blend_gamma": prio_blend_gamma,
        "sens_ref": float(max(s_p50, eps)),
        "sev_ref": float(max(v_p50, eps)),
        "mse_ref": float(max(m_p50, eps)),
        "n_measured": int(len(measured)),
        "n_hard_veto": int(sum(1 for f in veto if f)),
    }


def apply_fp16_infinite_ranking_branches(
    measured: List[Tuple[str, float, float, float, int]],
    is_hard_veto: Sequence[bool],
    *,
    branch_profile: Optional[Dict[str, Any]] = None,
    family_suffixes: Sequence[str] = _KEYPATTERN_FAMILY_SENS_SUFFIXES,
) -> Tuple[
    List[Tuple[str, float, float, float, int]],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    """Continuous infinite ranking branches for THIS checkpoint's measured pool.

    Replaces the banned unified family floor (fixed median / geom gate).

    Branch A — key-pattern siblings: for every architectural suffix with ≥2
    members, skew = span/family_p50 (continuous). Strength and blend are
    ``1 - exp(-skew * gamma_*)`` with gammas from ``branch_profile`` (THIS
    model). skew→0 ⇒ identity (wai-like balanced families). Large skew +
    DualMonitor starvation ⇒ strong pull toward a p50↔max continuous target.

    Branch B — axis mismatch: layers whose analyze severity / V4 MSE exceed
    DualMonitor sens (relative to THIS refs) get a continuous ranking_sens
    lift scaled by ``mismatch_gain`` from THIS VETO-alignment character.

    No binary skew gate. No model-name map. No absolute KEEP.
    """
    if not measured:
        empty_p = branch_profile or derive_fp16_infinite_branch_profile(
            [], [],
        )
        return [], [], empty_p
    if len(is_hard_veto) != len(measured):
        raise ValueError(
            "apply_fp16_infinite_ranking_branches: is_hard_veto length mismatch"
        )
    profile = branch_profile or derive_fp16_infinite_branch_profile(
        measured, is_hard_veto,
    )
    eps = 1e-30
    out = [list(row) for row in measured]
    details: List[Dict[str, Any]] = []

    by_suf: Dict[str, List[int]] = {}
    for i, row in enumerate(out):
        name = str(row[0])
        for suf in family_suffixes:
            if name.endswith(suf):
                by_suf.setdefault(suf, []).append(i)
                break

    g_sib = float(profile.get("gamma_sibling", 0.0) or 0.0)
    g_blend = float(profile.get("gamma_blend", 0.0) or 0.0)
    for suf, idxs in by_suf.items():
        if len(idxs) < 2:
            continue
        sens = [max(float(out[i][1]), 0.0) for i in idxs]
        s_max = max(sens)
        s_min = min(sens)
        fam_p50 = _true_median(sens)
        if fam_p50 <= 0.0 or s_max <= 0.0:
            continue
        span = float(s_max - s_min)
        skew = float(span / max(fam_p50, eps))
        # Continuous — never "if skew < 1: skip" unified gate.
        strength = 1.0 - math.exp(-skew * g_sib)
        blend = 1.0 - math.exp(-skew * g_blend)
        target = float(fam_p50 * (1.0 - blend) + s_max * blend)
        for i, s in zip(idxs, sens):
            if target <= s or strength <= 0.0:
                continue
            ranking = float(s + strength * (target - s))
            if ranking <= s:
                continue
            out[i][1] = ranking
            details.append({
                "branch": "keypattern_sibling_continuous",
                "name": str(out[i][0]),
                "suffix": suf,
                "dm_sens": float(s),
                "ranking_sens": float(ranking),
                "skew": skew,
                "strength": float(strength),
                "blend": float(blend),
                "target": float(target),
                "family_p50": float(fam_p50),
                "family_max": float(s_max),
                "family_min": float(s_min),
            })

    sref = float(profile.get("sens_ref", eps) or eps)
    vref = float(profile.get("sev_ref", eps) or eps)
    mref = float(profile.get("mse_ref", eps) or eps)
    mg = float(profile.get("mismatch_gain", 0.0) or 0.0)
    aw = max(float(profile.get("align_sev", 0.0) or 0.0), 0.0)
    am = max(float(profile.get("align_mse", 0.0) or 0.0), 0.0)
    as_ = max(float(profile.get("align_sens", 0.0) or 0.0), 0.0)
    wsum = aw + am + as_
    if wsum < eps:
        wv, wm, ws = 1.0, 1.0, 1.0
    else:
        wv, wm, ws = aw / wsum, am / wsum, as_ / wsum
    for i, row in enumerate(out):
        s = max(float(row[1]), 0.0)
        mse = max(float(row[2]), 0.0)
        sev = max(float(row[3]), 0.0)
        rs = s / sref
        rv = sev / vref
        rm = mse / mref
        excess = float(wv * rv + wm * rm - ws * rs)
        if excess <= 0.0 or mg <= 0.0:
            continue
        # Soft continuous lift — unique per layer × THIS profile.
        lift = float(mg * math.log1p(excess))
        ranking = float(s * (1.0 + lift))
        if ranking <= s:
            continue
        out[i][1] = ranking
        details.append({
            "branch": "axis_mismatch_continuous",
            "name": str(out[i][0]),
            "dm_sens": float(s),
            "ranking_sens": float(ranking),
            "excess": excess,
            "lift": lift,
            "severity": float(sev),
            "v4_mse": float(mse),
        })

    restored = [
        (str(r[0]), float(r[1]), float(r[2]), float(r[3]), int(r[4]))
        for r in out
    ]
    return restored, details, profile


def apply_fp16_infinite_priority_branches(
    candidates: List[Tuple[float, float, float, float, int, str]],
    branch_profile: Dict[str, Any],
    *,
    family_suffixes: Sequence[str] = _KEYPATTERN_FAMILY_SENS_SUFFIXES,
) -> Tuple[
    List[Tuple[float, float, float, float, int, str]],
    List[Dict[str, Any]],
]:
    """Continuous priority-space sibling branch (THIS family's priorities).

    ``candidates``: ``(priority, v4_mse, severity, dm_sens, extra, name)``.
    Same continuous skew×gamma form as sens branches — not ``max(p, p50)``
    unified priority floor. Strength→0 when THIS family is balanced.
    """
    if not candidates:
        return [], []
    eps = 1e-30
    g_sib = float(branch_profile.get("prio_sibling_gamma", 0.0) or 0.0)
    g_blend = float(branch_profile.get("prio_blend_gamma", 0.0) or 0.0)
    out = [list(row) for row in candidates]
    by_suf: Dict[str, List[int]] = {}
    for i, row in enumerate(out):
        name = str(row[5])
        for suf in family_suffixes:
            if name.endswith(suf):
                by_suf.setdefault(suf, []).append(i)
                break
    details: List[Dict[str, Any]] = []
    for suf, idxs in by_suf.items():
        if len(idxs) < 2:
            continue
        sens = [max(float(out[i][3]), 0.0) for i in idxs]
        prios = [max(float(out[i][0]), 0.0) for i in idxs]
        s_max = max(sens)
        s_min = min(sens)
        fam_p50 = _true_median(sens)
        if fam_p50 <= 0.0:
            continue
        span = float(s_max - s_min)
        skew = float(span / max(fam_p50, eps))
        strength = 1.0 - math.exp(-skew * g_sib)
        blend = 1.0 - math.exp(-skew * g_blend)
        p_p50 = _true_median(prios)
        p_max = max(prios)
        if p_p50 <= 0.0 and p_max <= 0.0:
            continue
        target_p = float(p_p50 * (1.0 - blend) + p_max * blend)
        for i, s, p in zip(idxs, sens, prios):
            # Under-measured DM siblings only (continuous strength still
            # scales with family skew even when p is already high).
            if s >= fam_p50 or target_p <= p or strength <= 0.0:
                continue
            new_p = float(p + strength * (target_p - p))
            if new_p <= p:
                continue
            out[i][0] = new_p
            details.append({
                "branch": "keypattern_priority_continuous",
                "name": str(out[i][5]),
                "suffix": suf,
                "priority_before": float(p),
                "priority_after": float(new_p),
                "skew": skew,
                "strength": float(strength),
                "blend": float(blend),
                "target_priority": float(target_p),
                "dm_sens": float(s),
                "family_p50_sens": float(fam_p50),
            })
    restored = [
        (
            float(r[0]),
            float(r[1]),
            float(r[2]),
            float(r[3]),
            int(r[4]),
            str(r[5]),
        )
        for r in out
    ]
    return restored, details


def int8_fp16_budget_priority(
    dualmonitor_sensitivity: float,
    v4_estimated_mse: float,
    analyze_severity: float,
    *,
    combinator: Dict[str, Any],
) -> float:
    """Per-checkpoint FP16 priority via THIS model's autonomous combinator.

    combinator MUST be derive_priority_combinator(...) from measured
    sens/sev/mse distributions. Fixed formulas are forbidden.
    """
    if combinator is None:
        raise ValueError(
            "int8_fp16_budget_priority requires per-checkpoint combinator "
            "(derive_priority_combinator); fixed formulas are forbidden"
        )
    sens = max(float(dualmonitor_sensitivity), 0.0)
    sev = max(float(analyze_severity), 0.0)
    mse = max(float(v4_estimated_mse), 0.0)

    w_s = float(combinator.get("w_sens", 0.0))
    w_v = float(combinator.get("w_sev", 0.0))
    w_m = float(combinator.get("w_mse", 0.0))
    sref = max(float(combinator.get("sens_ref", 1.0)), 1e-30)
    vref = max(float(combinator.get("sev_ref", 1.0)), 1e-30)
    mref = max(float(combinator.get("mse_ref", 1.0)), 1e-30)

    if str(combinator.get("form", "")) == "uniform":
        return 1.0

    return math.exp(
        w_s * math.log1p(sens / sref)
        + w_v * math.log1p(sev / vref)
        + w_m * math.log1p(mse / mref)
    )



def build_int8_analyze_character_table(
    profile: Dict[str, Any],
    tunables: Dict[str, Any],
    *,
    hard_veto_names: Optional[set] = None,
) -> Dict[str, Dict[str, float]]:
    """Per-layer INT8 analyze character for this checkpoint (FP8 must not call).

    Returns {layer: {kurtosis, outlier_ratio, abs_max, mad_outlier_pct, severity}}.
    Severity uses the same fences as int8_fp16_budget_analyze_severity.
    """
    if tunables.get("quant_format") != "int8_tensorwise":
        raise ValueError(
            "build_int8_analyze_character_table is INT8-only; FP8 must not call this"
        )
    profile = _unet_only_profile(profile)
    layers = profile.get("layers", {})
    hard = hard_veto_names or set()
    out: Dict[str, Dict[str, float]] = {}
    for name, entry in layers.items():
        if not isinstance(entry, dict):
            continue
        k = float(entry.get("kurtosis", 0) or 0)
        o = float(entry.get("outlier_ratio", 0) or 0)
        m = float(entry.get("abs_max", 0) or 0)
        mad = float(entry.get("mad_outlier_pct", 0) or 0)
        ps = float(entry.get("profile_score", 0) or 0)
        sev = int8_fp16_budget_analyze_severity(
            kurtosis=k,
            outlier_ratio=o,
            abs_max=m,
            tunables=tunables,
            is_hard_veto=name in hard,
            layer_name=name,
            mad_outlier_pct=mad,
            profile_score=ps,
        )
        out[name] = {
            "kurtosis": k,
            "outlier_ratio": o,
            "abs_max": m,
            "mad_outlier_pct": mad,
            "profile_score": ps,
            "severity": float(sev),
        }
    return out


# ---------------------------------------------------------------------------
# Fully autonomous tunable derivation.
# Owner hard ceiling fp16_budget_mb=300 MiB is NOT a thinking-stop recipe:
# auto knobs fill inside that frame and must never exceed it.
# Every knob below is derived from THIS checkpoint's profile + DualMonitor
# sensitivity distribution. Covers degenerate / tiny / huge / skewed cases.
# ---------------------------------------------------------------------------


def _safe_percentile(values: List[float], pct: float) -> float:
    """Percentile on a raw list (defensive: empty, NaN, single, degenerate)."""
    clean = sorted(float(v) for v in values
                   if v is not None and math.isfinite(float(v)))
    if not clean:
        return 0.0
    if len(clean) == 1:
        return float(clean[0])
    p = min(max(float(pct), 0.0), 100.0)
    idx = int(round((p / 100.0) * (len(clean) - 1)))
    idx = min(max(idx, 0), len(clean) - 1)
    return float(clean[idx])


def _robust_iqr(values: List[float]) -> float:
    clean = sorted(float(v) for v in values
                   if v is not None and math.isfinite(float(v)))
    if len(clean) < 4:
        return max(clean[-1] - clean[0] if len(clean) >= 2 else 0.0, 1e-12)
    q1 = _safe_percentile(clean, 25.0)
    q3 = _safe_percentile(clean, 75.0)
    return max(q3 - q1, 1e-12)


def derive_int8_autonomous_tunables(
    profile: Dict[str, Any],
    *,
    dualmonitor_sensitivities: Optional[Dict[str, float]] = None,
    layer_extra_bytes: Optional[Dict[str, int]] = None,
    fp16_budget_mb: float = 300.0,
) -> Dict[str, Any]:
    """Derive EVERY INT8 knob from this checkpoint + calibration.

    Owner hard ceiling: fp16_budget_mb must be exactly 300 MiB.
    Inside that frame: THIS model's auto analysis → extreme auto-optimal
    settings (Hard VETO fences, ranking weights, MSE release, BC scope,
    gray-zone, alpha/beta, search_low, sens_veto percentile).
    Never redefine/exceed 300; never treat 300 as a removable recipe.

    Degenerate-input safe:
      - empty / single-layer profile
      - all-zero sensitivities (no calibration)
      - all-identical kurtosis / outlier / magnitude
      - extreme outliers dominating max
      - tiny UNet (<50 layers) or huge (>5000)
    """
    if abs(float(fp16_budget_mb) - 300.0) > 1e-6:
        raise ValueError(
            f"fp16_budget_mb must be exactly 300.0 MiB "
            f"(owner hard ceiling; got {fp16_budget_mb})"
        )
    fp16_budget_mb = 300.0

    profile = _normalize_profile(profile)
    profile = _unet_only_profile(profile)
    layers = profile.get("layers", {})
    if not layers:
        raise ValueError("derive_int8_autonomous_tunables: profile has no layers")

    base = derive_veto_tunables_int8(profile)

    all_k: List[float] = []
    all_o: List[float] = []
    all_m: List[float] = []
    all_mad: List[float] = []
    for entry in layers.values():
        if not isinstance(entry, dict):
            continue
        all_k.append(float(entry.get("kurtosis", 0) or 0))
        all_o.append(float(entry.get("outlier_ratio", 0) or 0))
        all_m.append(float(entry.get("abs_max", 0) or 0))
        mad = float(entry.get("mad_outlier_pct", entry.get("mad_pct", 0)) or 0)
        all_mad.append(mad)

    n_layers = len(all_k)
    k_p50 = _safe_percentile(all_k, 50.0)
    k_p75 = _safe_percentile(all_k, 75.0)
    k_p99 = _safe_percentile(all_k, 99.0)
    o_p50 = _safe_percentile(all_o, 50.0)
    o_p75 = _safe_percentile(all_o, 75.0)
    o_p99 = _safe_percentile(all_o, 99.0)
    m_p50 = _safe_percentile(all_m, 50.0)
    m_p75 = _safe_percentile(all_m, 75.0)
    m_p99 = _safe_percentile(all_m, 99.0)

    iqr_k = _robust_iqr(all_k)
    iqr_o = _robust_iqr(all_o)
    iqr_m = _robust_iqr(all_m)
    iqr_mad = _robust_iqr(all_mad) if any(v > 0 for v in all_mad) else 0.0

    # Hard VETO fences: max(Tukey, P99) so only the true right tail is VETO.
    # If all values identical (iqr==0), P99==max → only the single max layer
    # is VETO (degenerate-safe).
    ek = float(max(base["extreme_kurtosis"], k_p99))
    eo = float(max(base["extreme_outlier"], o_p99))
    hm = float(max(base["huge_magnitude"], m_p99))
    base["extreme_kurtosis"] = ek
    base["extreme_outlier"] = eo
    base["huge_magnitude"] = hm

    # ---- Dynamic keep ranking weights (NO fixed 2.0 / 0.5) ----
    # Weight each axis by its own IQR spread: a flat axis (iqr→0) gets ~0
    # weight so it cannot dominate; a spread axis gets more.
    # Floor at 1e-6 to avoid div-by-zero; normalized so weights sum ~3.
    w_k = float(iqr_k / max(k_p75, 1e-9))
    w_o = float(iqr_o / max(o_p75, 1e-9))
    w_m = float(iqr_m / max(m_p50, 1e-9))
    w_sum = max(w_k + w_o + w_m, 1e-9)
    w_k = 3.0 * w_k / w_sum
    w_o = 3.0 * w_o / w_sum
    w_m = 3.0 * w_m / w_sum
    base["score_k_weight"] = w_k
    base["score_o_weight"] = w_o
    base["score_m_weight"] = w_m
    base["drift_score_mult"] = float(max(iqr_k + iqr_o + iqr_m, 1.0))

    # ---- MSE release gates (gray-zone VETO release candidates) ----
    # THIS P75 only — continuous from THIS profile (no half-IQR rewrite).
    base["mse_release_o_min"] = float(o_p75)
    base["mse_release_k_max"] = float(k_p75)
    base["mse_release_m_max"] = float(m_p75)
    # MSE P75 multiplier: 1 + (outlier dispersion / THIS scale chain).
    o_den = next((x for x in (o_p50, o_p75, o_p99, max(all_o) if all_o else 0.0) if x > 0), 0.0)
    base["mse_p75_multiplier"] = float(1.0 + (iqr_o / max(o_den, 1e-9)))

    # ---- alpha / beta — continuous from THIS P50/P75 (no 0.5 / 0.99 box) ----
    k_scale_a = 1.0 / max(k_p75, 1e-9)
    o_scale_a = 1.0 / max(o_p75, 1e-9)
    alpha_floor = max(k_scale_a * k_p50, 0.0)
    beta_floor = max(o_scale_a * o_p50, 0.0)
    base["alpha_floor"] = float(alpha_floor)
    base["beta_floor"] = float(beta_floor)
    base["alpha_clip_max"] = float(max(k_scale_a * k_p99, alpha_floor))
    base["beta_clip_max"] = float(max(o_scale_a * o_p99, beta_floor))

    # ---- alpha_auto: Full-SVD×RMS from THIS multi-axis analyze character ----
    base["alpha_auto"] = _alpha_auto_from_this_character(
        k_p50=k_p50,
        k_p75=k_p75,
        k_p99=k_p99,
        o_p50=o_p50,
        o_p75=o_p75,
        o_p99=o_p99,
        m_p50=m_p50,
        m_p75=m_p75,
        m_p99=m_p99,
        iqr_k=iqr_k,
        iqr_o=iqr_o,
        iqr_m=iqr_m,
    )

    # ---- search_low: INT8 pack is absmax (1.0). No clipping. ----
    base["search_low_floor"] = 1.0
    base["search_low_penalty_cap"] = 0.0
    base["search_low_clip_max"] = 1.0
    base["search_low_gray_clip_max"] = 1.0

    # ---- MAD% gate + character scale: from THIS profile if present ----
    # Any positive MAD sample → continuous replace (n≥1). Zero samples → off.
    # Writing 0.0 when 1≤n<4 after deleting ×N floors is forbidden.
    mad_bundle = _mad_tunables_from_positive_samples(
        all_mad, float(max(eo, 1e-9))
    )
    base.update(mad_bundle)

    # DualMonitor = FP16 candidates; analyze = VETO candidates.
    # Quantize fills THIS model's extreme auto-optimal FP16 set inside the
    # 300 MiB hard ceiling (measured sens/sev/mse combinator). keep_ratio r0.
    # DualMonitor is never renamed Hard VETO and never invents keep_ratio.
    sens = dualmonitor_sensitivities or {}
    sens_values = [float(v) for v in sens.values()
                   if v is not None and math.isfinite(float(v)) and float(v) > 0]
    base["sens_veto_percentile"] = 100.0  # no DualMonitor→analyze-VETO rename
    base["sens_veto_keep_ratio_gate"] = 0.0
    base["auto_keep_ratio"] = 0.0  # r0

    # ---- bias_correction scope (continuous; no fixed 5×median / 0.5) ----
    # BC is beneficial on every layer whose calibration statistics are sound
    # (d1290df: full BC SSIM 0.9753 > top-0.5 0.9678). The only layers to
    # exclude are noise-floor layers whose sensitivity sits BELOW the Tukey
    # lower fence (Q1 - IQR) of THIS calibration — their act-mean estimate is
    # dominated by noise and BC would inject DC. Scope ratio is therefore
    # the measured fraction of layers at or above that fence (continuous).
    if sens_values and len(sens_values) >= 4:
        s_sorted_bc = _sorted_pool(sens_values)
        q1_bc, _, q3_bc = _quartile_bounds(s_sorted_bc)
        lower_fence = q1_bc - (q3_bc - q1_bc)
        noisy = sum(1 for v in s_sorted_bc if v < lower_fence)
        base["bias_correction_top_ratio"] = float(
            min(max(1.0 - noisy / max(len(s_sorted_bc), 1), 0.0), 1.0)
        )
    else:
        base["bias_correction_top_ratio"] = 1.0

    # ---- FP16 hard ceiling 300 MiB (owner); auto settings fill inside ----
    base["fp16_budget_mb"] = 300.0
    base["fp16_budget_bytes"] = int(300.0 * 1024 * 1024)


    # Autonomous priority combinator seed (analyze severity axis only here;
    # quantize re-derives from measured sens/sev/mse after calibration).
    sev_proxy_values = []
    for k, o, m in zip(all_k, all_o, all_m):
        sev_proxy_values.append(max(o / eo, 0.0) + max(k / ek, 0.0) + max(m / hm, 0.0))
    sev_p50 = _safe_percentile(sev_proxy_values, 50.0)
    sev_iqr = _robust_iqr(sev_proxy_values)
    base["priority_combinator"] = derive_priority_combinator(
        sens_iqr=0.0, sev_iqr=sev_iqr, mse_iqr=0.0,
        sens_p50=0.0, sev_p50=sev_p50, mse_p50=0.0,
    )
    base["_sev_p50"] = float(sev_p50)
    base["_sev_iqr"] = float(sev_iqr)

    base["n_unet_layers"] = n_layers
    base["autonomous"] = True
    _assert_int8_auto_optimal_complete(base)
    return base


# Keys that MUST be filled by auto analysis → auto-optimal (never silent
# dataclass holes after deleting accommodation clips — philosophy §0 / §1).
_INT8_AUTO_OPTIMAL_REQUIRED = (
    "extreme_kurtosis",
    "extreme_outlier",
    "huge_magnitude",
    "drift_veto_thresh",
    "drift_score_mult",
    "mse_release_o_min",
    "mse_release_k_max",
    "mse_release_m_max",
    "mse_p75_multiplier",
    "k_scale",
    "o_scale",
    "m_scale",
    "k_gray_lo",
    "k_gray_hi",
    "o_gray_lo",
    "o_gray_hi",
    "m_gray_lo",
    "m_gray_hi",
    "alpha_floor",
    "alpha_clip_max",
    "beta_floor",
    "beta_clip_max",
    "alpha_auto",
    "search_low_floor",
    "search_low_penalty_cap",
    "search_low_clip_max",
    "search_low_gray_clip_max",
    "attn_mad_pct_floor",
    "attn_mad_q3",
    "attn_mad_p99",
    "attn_mad_gap_o_max",
    "attn_mad_from_profile",
    "attn_mad_collapse",
    "attn_mad_iqr",
    "bias_correction_top_ratio",
    "score_k_weight",
    "score_o_weight",
    "score_m_weight",
    "quant_format",
    "autonomous",
    "fp16_budget_mb",
)


def _assert_int8_auto_optimal_complete(d: Dict[str, Any]) -> None:
    """Fail loud if a former accommodation clip was deleted without replace."""
    missing = [k for k in _INT8_AUTO_OPTIMAL_REQUIRED if k not in d]
    if missing:
        raise ValueError(
            "INT8 auto-optimal incomplete after analyze — missing keys "
            f"{missing}. Do not fill deleted clip holes with defaults; "
            "re-run derive_int8_autonomous_tunables."
        )
    if str(d.get("quant_format")) != "int8_tensorwise":
        raise ValueError("INT8 auto-optimal requires quant_format=int8_tensorwise")
    if abs(float(d.get("fp16_budget_mb", 0.0)) - 300.0) > 1e-6:
        raise ValueError("INT8 auto-optimal requires fp16_budget_mb=300.0")
    if not bool(d.get("autonomous")):
        raise ValueError("INT8 auto-optimal requires autonomous=True from derive")
    # Finite / non-NaN on continuous auto knobs.
    for k in (
        "mse_p75_multiplier",
        "mse_release_o_min",
        "drift_veto_thresh",
        "alpha_auto",
        "alpha_floor",
        "alpha_clip_max",
        "attn_mad_p99",
    ):
        v = float(d[k])
        if not math.isfinite(v):
            raise ValueError(f"INT8 auto-optimal key {k} is not finite: {v}")
    if float(d["mse_p75_multiplier"]) <= 0.0:
        raise ValueError("mse_p75_multiplier must be > 0 (THIS-profile replace)")
    if float(d["search_low_floor"]) != 1.0:
        raise ValueError("INT8 search_low_floor must be 1.0 (absmax pack replace)")


def _is_unet_weight_key(name: str) -> bool:
    """Keep SDXL UNet weights; drop CLIP / VAE (full-checkpoint profiles)."""
    if name.startswith(("conditioner.", "first_stage_model.", "text_encoder")):
        return False
    if "vae." in name or name.startswith("vae."):
        return False
    return True


def _unet_only_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    profile = _normalize_profile(profile)
    layers = profile.get("layers", {})
    unet = {k: v for k, v in layers.items() if _is_unet_weight_key(k)}
    out = dict(profile)
    out["layers"] = unet if unet else layers
    return out


def measure_v4_int8_mse_at_absmax(
    weight_tensors: Dict[str, torch.Tensor],
    *,
    device: Optional[str] = None,
    max_safe_sample: int = 30,
    max_veto_sample: int = 40,
    tunables: Optional[Dict[str, Any]] = None,
    importance_by_layer: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """HSWQ V4 + INT8Quantizer estimated_mse at absmax (same contract as v3.0).

    analyze derives mse_release_* / mse_p75_multiplier; this function measures
    the histogram side on live weights so optimal INT8 settings are a double
    of (analyze gates) × (V4 MSE).

    importance_by_layer MUST come from DualMonitor after the SDXL r32
    calibration recipe (num_calib_samples=32, num_inference_steps=25).
    Weight-only calls (importance_by_layer=None) are incomplete and must not
    be treated as final optimal settings.
    """
    hist_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "histogram")
    if hist_dir not in sys.path:
        sys.path.insert(0, hist_dir)
    from weighted_histogram_mse_v4 import (  # type: ignore
        HSWQWeightedHistogramOptimizerV4,
        INT8Quantizer,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not importance_by_layer:
        # Weight-only analyze: V4 MSE not final yet. Full-SVD stays scheduled ON
        # for the DualMonitor path (philosophy §4/§5 — never cut SVD). Do not
        # print svd=None as if SVD failed; Imp is what is deferred.
        a0 = float((tunables or {}).get("alpha_auto", 0.0) or 0.0)
        a0 = float(min(max(a0, 0.0), 1.0))
        return {
            "v4_ran": False,
            "complete": False,
            "reason": "dualmonitor_importance_required",
            "svd_enabled": True,
            "alpha": a0,
            "beta": float(1.0 - a0),
            "calib_contract": {
                "num_calib_samples": 32,
                "num_inference_steps": 25,
                "note": "How-to / r32: DualMonitor channel_importance after 32-sample calib",
            },
            "device": device,
        }

    tunables = tunables or {}
    o_min = float(tunables.get("mse_release_o_min", 0.0))
    k_max = float(tunables.get("mse_release_k_max", 1e9))
    m_max = float(tunables.get("mse_release_m_max", 1e9))
    p75_mult = float(tunables.get("mse_p75_multiplier", 1.0))
    extreme_o = float(tunables.get("extreme_outlier", 1e9))
    extreme_k = float(tunables.get("extreme_kurtosis", 1e9))
    huge_m = float(tunables.get("huge_magnitude", 1e9))

    # Keys may be module names or *.weight — normalize lookup for importance.
    def _imp_for(name: str) -> Optional[torch.Tensor]:
        base = name[:-7] if name.endswith(".weight") else name
        for key in (name, base, base + ".weight"):
            if key in importance_by_layer:
                return importance_by_layer[key].detach().float()
        return None

    items = [(n, t) for n, t in weight_tensors.items() if isinstance(t, torch.Tensor)]
    items = [(n, t) for n, t in items if t.ndim >= 2 and _is_unet_weight_key(n)]
    if not items:
        return {
            "v4_ran": False,
            "complete": False,
            "reason": "no_unet_weight_tensors",
            "device": device,
        }

    # alpha_auto from THIS profile → Full-SVD×RMS; DualMonitor Imp multiplies.
    alpha = float(tunables.get("alpha_auto", 0.0))
    alpha = float(min(max(alpha, 0.0), 1.0))
    beta = 1.0 - alpha
    quantizer = INT8Quantizer(device=device)
    optimizer = HSWQWeightedHistogramOptimizerV4(
        bins=8192,
        num_candidates=1000,
        refinement_iterations=10,
        device=device,
        alpha=alpha,
        beta=beta,
        quantizer=quantizer,
    )
    search_range = (1.0, 1.0)

    layer_quick: List[Tuple[str, torch.Tensor, Dict[str, float]]] = []
    for name, tensor in items:
        stats = _layer_stats(tensor)
        layer_quick.append((name, tensor, stats))

    hard_veto = []
    safe_pool = []
    gray_pool = []
    for name, tensor, st in layer_quick:
        k, o, m = st["kurtosis"], st["outlier_ratio"], st["abs_max"]
        is_hard = (o > extreme_o) or (k > extreme_k) or (m > huge_m)
        if is_hard:
            hard_veto.append(name)
            if o > o_min and k <= k_max and m <= m_max:
                gray_pool.append((name, tensor, st))
        else:
            safe_pool.append((name, tensor, st))

    safe_ff = [(n, t, s) for n, t, s in safe_pool if classify_layer(n) == "ff2"]
    safe_sample_src = safe_ff if len(safe_ff) >= 8 else safe_pool
    step = max(1, len(safe_sample_src) // max_safe_sample)
    safe_sample = safe_sample_src[::step][:max_safe_sample]

    safe_mses: List[float] = []
    safe_detail: List[Dict[str, Any]] = []
    skipped_no_imp = 0
    for name, tensor, st in safe_sample:
        imp = _imp_for(name)
        if imp is None:
            skipped_no_imp += 1
            safe_detail.append({"name": name, "error": "no DualMonitor importance"})
            continue
        w = tensor.detach().float()
        if device != "cpu":
            w = w.to(device)
            imp = imp.to(device)
        try:
            result = optimizer.compute_optimal_amax_with_stats_int8_range(
                w,
                importance=imp,
                use_svd_leverage=True,
                scaled=False,
                search_range=search_range,
            )
            mse = float(result["estimated_mse"])
            safe_mses.append(mse)
            safe_detail.append({
                "name": name,
                "estimated_mse": mse,
                "optimal_amax": float(result["optimal_amax"]),
                "abs_max": float(st["abs_max"]),
                "outlier_ratio": float(st["outlier_ratio"]),
                "used_dualmonitor_importance": True,
            })
        except Exception as exc:  # noqa: BLE001
            safe_detail.append({"name": name, "error": str(exc)})
        if device != "cpu":
            torch.cuda.empty_cache()

    if not safe_mses:
        return {
            "v4_ran": True,
            "complete": False,
            "device": device,
            "safe_sample_count": 0,
            "skipped_no_importance": skipped_no_imp,
            "hard_veto_count": len(hard_veto),
            "gray_candidate_count": len(gray_pool),
            "reason": "no_safe_mse_with_dualmonitor_importance",
            "safe_detail": safe_detail,
            "search_range": list(search_range),
            "quantizer": "INT8Quantizer",
            "optimizer": "HSWQWeightedHistogramOptimizerV4",
            "calib_contract": {
                "num_calib_samples": 32,
                "num_inference_steps": 25,
            },
        }

    safe_mses_sorted = sorted(safe_mses)
    p75_idx = int(len(safe_mses_sorted) * 0.75)
    p75_idx = min(p75_idx, len(safe_mses_sorted) - 1)
    p75_mse = float(safe_mses_sorted[p75_idx])
    mse_threshold = p75_mse * p75_mult

    gray_sample = gray_pool[:max_veto_sample]
    released = 0
    kept = 0
    gray_detail: List[Dict[str, Any]] = []
    for name, tensor, st in gray_sample:
        imp = _imp_for(name)
        if imp is None:
            kept += 1
            gray_detail.append({
                "name": name,
                "decision": "KEEP",
                "error": "no DualMonitor importance",
            })
            continue
        w = tensor.detach().float()
        if device != "cpu":
            w = w.to(device)
            imp = imp.to(device)
        try:
            result = optimizer.compute_optimal_amax_with_stats_int8_range(
                w,
                importance=imp,
                use_svd_leverage=True,
                scaled=False,
                search_range=search_range,
            )
            mse = float(result["estimated_mse"])
            decision = "RELEASE" if mse <= mse_threshold else "KEEP"
            if decision == "RELEASE":
                released += 1
            else:
                kept += 1
            gray_detail.append({
                "name": name,
                "estimated_mse": mse,
                "decision": decision,
                "optimal_amax": float(result["optimal_amax"]),
                "abs_max": float(st["abs_max"]),
                "outlier_ratio": float(st["outlier_ratio"]),
                "kurtosis": float(st["kurtosis"]),
                "used_dualmonitor_importance": True,
            })
        except Exception as exc:  # noqa: BLE001
            gray_detail.append({"name": name, "error": str(exc)})
        if device != "cpu":
            torch.cuda.empty_cache()

    return {
        "v4_ran": True,
        "complete": True,
        "device": device,
        "quantizer": "INT8Quantizer",
        "optimizer": "HSWQWeightedHistogramOptimizerV4",
        "search_range": list(search_range),
        "bins": 8192,
        "num_candidates": 1000,
        "pack_point": "absmax",
        "alpha": alpha,
        "beta": beta,
        "use_svd_leverage": True,
        # NEVER report svd_enabled=False (philosophy §0/§4/§5: never skip V4).
        # weighted_histogram_mse_v4.compute_hybrid_leverage_scores ALWAYS runs
        # torch.linalg.svd regardless of alpha; alpha only scales the SVD-leverage
        # contribution in the SVD×RMS mix. bool(alpha > 0.0) would be a lying
        # "SVD skipped" flag — sacrilege. Full-SVD always executes.
        "svd_enabled": True,
        "dualmonitor_importance": True,
        "calib_contract": {
            "num_calib_samples": 32,
            "num_inference_steps": 25,
            "source": "md/How to quantize SDXL.md (Samples:32 / r32)",
        },
        "hard_veto_count": len(hard_veto),
        "gray_candidate_count": len(gray_pool),
        "safe_sample_count": len(safe_mses),
        "skipped_no_importance": skipped_no_imp,
        "safe_p75_mse": p75_mse,
        "mse_p75_multiplier": p75_mult,
        "mse_release_threshold": mse_threshold,
        "gray_sampled": len(gray_sample),
        "gray_released": released,
        "gray_kept": kept,
        "analyze_gates": {
            "mse_release_o_min": o_min,
            "mse_release_k_max": k_max,
            "mse_release_m_max": m_max,
            "extreme_outlier": extreme_o,
            "extreme_kurtosis": extreme_k,
            "huge_magnitude": huge_m,
        },
        "safe_detail": safe_detail,
        "gray_detail": gray_detail,
    }


def compute_int8_optimal_settings(
    profile: Dict[str, Any],
    weight_tensors: Optional[Dict[str, torch.Tensor]] = None,
    *,
    device: Optional[str] = None,
    importance_by_layer: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Auto optimal INT8 settings = analyze × V4 SVD×Imp × DualMonitor (r32).

    Triple contract (no shortcuts):
      1) analyze derive_int8_autonomous_tunables (weight-space + alpha_auto)
      2) V4 Full-SVD×RMS + INT8Quantizer estimated_mse @ absmax
      3) DualMonitor channel_importance from 32-sample / 25-step calibration
    """
    unet_prof = _unet_only_profile(profile)
    tunables = derive_int8_autonomous_tunables(unet_prof)
    layers = unet_prof.get("layers", {})

    optimal: Dict[str, Any] = {
        "quant_format": "int8_tensorwise",
        "pack_amax": "absmax",
        "search_low": 1.0,
        "calib_contract": {
            "num_calib_samples": 32,
            "num_inference_steps": 25,
            "required": True,
            "source": "md/How to quantize SDXL.md Samples:32 / r32",
        },
        "analyze": {
            "mse_release_o_min": float(tunables["mse_release_o_min"]),
            "mse_release_k_max": float(tunables["mse_release_k_max"]),
            "mse_release_m_max": float(tunables["mse_release_m_max"]),
            "mse_p75_multiplier": float(tunables["mse_p75_multiplier"]),
            "extreme_kurtosis": float(tunables["extreme_kurtosis"]),
            "extreme_outlier": float(tunables["extreme_outlier"]),
            "huge_magnitude": float(tunables["huge_magnitude"]),
            "attn_qkv_absmax": float(tunables["attn_qkv_absmax"]),
            "attn_toout_absmax": float(tunables["attn_toout_absmax"]),
            "attn_ff2_absmax": float(tunables["attn_ff2_absmax"]),
            "attn_mad_pct_floor": float(tunables.get("attn_mad_pct_floor", 0.0)),
            "attn_mad_p99": float(tunables.get("attn_mad_p99", 0.0)),
            "alpha_auto": float(tunables.get("alpha_auto", 0.0)),
            "unet_layer_count": len(layers),
        },
        "v4": {
            "required": True,
            "role": (
                "MSE-guided VETO @ absmax: Full-SVD×RMS "
                "(alpha_auto) × DualMonitor importance"
            ),
            "quantizer": "INT8Quantizer",
            "optimizer": "HSWQWeightedHistogramOptimizerV4",
            "search_range": [1.0, 1.0],
            "use_svd_leverage": True,
        },
    }

    if weight_tensors is not None:
        v4 = measure_v4_int8_mse_at_absmax(
            weight_tensors,
            device=device,
            tunables=tunables,
            importance_by_layer=importance_by_layer,
        )
        optimal["v4"].update(v4)
        # Full-SVD schedule is ON even when Imp is deferred (never svd=None handwave).
        if "svd_enabled" in v4:
            optimal["svd_enabled"] = bool(v4.get("svd_enabled", True))
            tunables["v4_svd_enabled"] = bool(v4.get("svd_enabled", True))
        if "alpha" in v4:
            optimal["v4_alpha"] = float(v4.get("alpha", 0.0))
            optimal["v4_beta"] = float(v4.get("beta", 1.0))
            tunables["v4_alpha"] = float(v4.get("alpha", 0.0))
            tunables["v4_beta"] = float(v4.get("beta", 1.0))
        if v4.get("complete") and "mse_release_threshold" in v4:
            # SVD-aware V4 MSE → auto-optimal release / priority seed.
            p75 = float(v4["safe_p75_mse"])
            thr = float(v4["mse_release_threshold"])
            optimal["recommended_mse_release_threshold"] = thr
            optimal["recommended_safe_p75_mse"] = p75
            optimal["svd_enabled"] = bool(v4.get("svd_enabled", True))
            optimal["v4_alpha"] = float(v4.get("alpha", 0.0))
            optimal["v4_beta"] = float(v4.get("beta", 1.0))
            tunables["recommended_mse_release_threshold"] = thr
            tunables["recommended_safe_p75_mse"] = p75
            tunables["v4_svd_enabled"] = bool(v4.get("svd_enabled", True))
            tunables["v4_alpha"] = float(v4.get("alpha", 0.0))
            tunables["v4_beta"] = float(v4.get("beta", 1.0))
            # Re-seed priority MSE axis from THIS SVD V4 safe sample.
            mse_vals = [
                float(d["estimated_mse"])
                for d in (v4.get("safe_detail") or [])
                if isinstance(d, dict) and "estimated_mse" in d
            ]
            if len(mse_vals) >= 4:
                m_p50 = float(_safe_percentile(mse_vals, 50.0))
                m_iqr = float(_robust_iqr(mse_vals))
                sev_p50 = float(tunables.get("_sev_p50", 0.0) or 0.0)
                sev_iqr = float(tunables.get("_sev_iqr", 0.0) or 0.0)
                combinator = derive_priority_combinator(
                    sens_iqr=0.0,
                    sev_iqr=sev_iqr,
                    mse_iqr=m_iqr,
                    sens_p50=0.0,
                    sev_p50=sev_p50,
                    mse_p50=m_p50,
                )
                tunables["priority_combinator"] = combinator
                optimal["priority_combinator"] = combinator
                optimal["v4_mse_p50"] = m_p50
                optimal["v4_mse_iqr"] = m_iqr
            optimal["complete"] = True
        else:
            optimal["complete"] = False
    else:
        optimal["v4"]["v4_ran"] = False
        optimal["v4"]["complete"] = False
        optimal["v4"]["reason"] = "no_weight_tensors_passed"
        optimal["complete"] = False

    return {"veto_tunables_int8": tunables, "optimal_settings_int8": optimal}


def enrich_profile_with_derived(
    profile: Dict[str, Any],
    weight_tensors: Optional[Dict[str, torch.Tensor]] = None,
    *,
    device: Optional[str] = None,
    importance_by_layer: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Recompute scores; attach FP8 + INT8 tunables; auto-optimal via analyze×V4×calib.

    Full INT8 optimal (complete=True) requires DualMonitor channel_importance
    from the How-to r32 recipe (32 samples / 25 steps). Without it, gates are
    still written but optimal_settings_int8.complete stays False.
    """
    layers = profile.get("layers", {})
    all_k = [float(e.get("kurtosis", 0)) for e in layers.values()]
    all_o = [
        float(e.get("outlier_ratio", e.get("abs_max", 0)))
        for e in layers.values()
    ]
    all_m = [float(e.get("abs_max", 0)) for e in layers.values()]

    for entry in layers.values():
        k = float(entry.get("kurtosis", 0))
        o = float(entry.get("outlier_ratio", entry.get("abs_max", 0)))
        m = float(entry.get("abs_max", 0))
        entry["profile_score"] = composite_rank_score(k, o, m, all_k, all_o, all_m)

    tunables = derive_veto_tunables(profile)
    profile["veto_tunables"] = tunables

    int8_bundle = compute_int8_optimal_settings(
        profile,
        weight_tensors=weight_tensors,
        device=device,
        importance_by_layer=importance_by_layer,
    )
    profile["veto_tunables_int8"] = int8_bundle["veto_tunables_int8"]
    profile["optimal_settings_int8"] = int8_bundle["optimal_settings_int8"]

    extreme_k = tunables["extreme_kurtosis"]
    med_k = tunables["median_kurtosis"]
    extreme_o = tunables["extreme_outlier"]

    high_k = low_k = med_k_count = 0
    for entry in layers.values():
        k = float(entry.get("kurtosis", 0))
        o = float(entry.get("outlier_ratio", entry.get("abs_max", 0)))
        if k > extreme_k:
            high_k += 1
        elif k <= med_k:
            low_k += 1
        else:
            med_k_count += 1

    ff2_count = sum(1 for n in layers if classify_layer(n) == "ff2")
    i8 = profile["veto_tunables_int8"]
    opt = profile["optimal_settings_int8"]
    v4opt = opt.get("v4", {})
    profile["summary"] = {
        "layer_count": len(layers),
        "high_kurtosis_layers": high_k,
        "medium_kurtosis_layers": med_k_count,
        "low_kurtosis_layers": low_k,
        "extreme_outlier_layers": sum(
            1
            for e in layers.values()
            if float(e.get("outlier_ratio", e.get("abs_max", 0))) > extreme_o
        ),
        "ff2_count": ff2_count,
        "ff2_auto_full_class": tunables.get("ff2_auto_full_class", False),
        "ff2_selective_protected_count": tunables.get("ff2_selective_protected_count", 0),
        "int8_search_low": float(i8.get("search_low_floor", 1.0)),
        "int8_mse_release_o_min": float(i8.get("mse_release_o_min", 0.0)),
        "int8_mse_p75_multiplier": float(i8.get("mse_p75_multiplier", 1.0)),
        "int8_optimal_complete": bool(opt.get("complete", False)),
        "calib_contract": opt.get("calib_contract"),
        "v4_ran": bool(v4opt.get("v4_ran", False)),
        "v4_complete": bool(v4opt.get("complete", False)),
        "v4_reason": v4opt.get("reason"),
        "v4_safe_p75_mse": v4opt.get("safe_p75_mse"),
        "v4_mse_release_threshold": v4opt.get("mse_release_threshold"),
        "v4_svd_enabled": v4opt.get("svd_enabled"),
        "v4_alpha": v4opt.get("alpha"),
        "v4_beta": v4opt.get("beta"),
        "alpha_auto": float(i8.get("alpha_auto", 0.0)),
    }
    return profile


# ---------------------------------------------------------------------------
# CLI: build profile from safetensors
# ---------------------------------------------------------------------------

def analyze_unet(path: str, *, run_v4: bool = True) -> Dict[str, Any]:
    """Scan safetensors → layer stats → FP8/INT8 tunables → optional V4 stub.

    Weight-only analyze cannot supply DualMonitor importance (needs the
    32-sample / 25-step calib recipe). With run_v4=True, weights are still
    passed so the contract is recorded, but optimal_settings_int8.complete
    remains False until importance_by_layer is provided (quantize path).
    """
    state = load_file(path)
    layers: Dict[str, Any] = {}
    weight_tensors: Dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if not name.endswith(".weight"):
            continue
        if len(tensor.shape) < 2:
            continue
        stats = _layer_stats(tensor)
        layers[name] = stats
        if run_v4 and _is_unet_weight_key(name):
            weight_tensors[name] = tensor
    profile: Dict[str, Any] = {"source": path, "layers": layers}
    return enrich_profile_with_derived(
        profile,
        weight_tensors=weight_tensors if run_v4 else None,
        importance_by_layer=None,
    )


def generate_model_profile(input_path: str, output_path: str) -> Dict[str, Any]:
    """Build profile JSON (CPU safetensors scan). Used by quantize_sdxl_hswq_v2.0."""
    profile = analyze_unet(input_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SDXL UNet weight distribution")
    parser.add_argument("safetensors", nargs="?", help="Path to SDXL UNet safetensors (positional)")
    parser.add_argument("--input", "-i", dest="input_path", help="Input safetensors (quantize CLI compat)")
    parser.add_argument("-o", "--output", required=True, help="Output profile JSON path")
    args = parser.parse_args()

    src = args.input_path or args.safetensors
    if not src:
        parser.error("provide safetensors path as positional arg or --input")
    profile = generate_model_profile(src, args.output)
    print(f"Wrote {len(profile['layers'])} layers to {args.output}")
    print(f"ff2_auto_full_class={profile['summary'].get('ff2_auto_full_class')}")
    summary = profile.get("summary", {})
    v4_ran = summary.get("v4_ran")
    v4_reason = summary.get("v4_reason")
    defer = ""
    if (not v4_ran) and v4_reason == "dualmonitor_importance_required":
        defer = " (V4 MSE deferred until DualMonitor Imp; Full-SVD scheduled ON)"
    print(
        f"[analyze×V4 INT8] search_low={summary.get('int8_search_low')} "
        f"mse_release_o_min={summary.get('int8_mse_release_o_min')} "
        f"mse_p75_mult={summary.get('int8_mse_p75_multiplier')} "
        f"v4_ran={v4_ran} "
        f"v4_reason={v4_reason} "
        f"v4_p75_mse={summary.get('v4_safe_p75_mse')} "
        f"v4_threshold={summary.get('v4_mse_release_threshold')} "
        f"svd={summary.get('v4_svd_enabled')} "
        f"alpha_auto={summary.get('alpha_auto')} "
        f"v4_α/β={summary.get('v4_alpha')}/{summary.get('v4_beta')}"
        f"{defer}"
    )


if __name__ == "__main__":
    main()

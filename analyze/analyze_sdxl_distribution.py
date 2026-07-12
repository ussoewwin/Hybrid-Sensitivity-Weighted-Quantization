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
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

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
    """Quantize-side thresholds derived from global layer pools (no fixed constants)."""
    n = len(all_k)
    k_q1, k_med, k_q3 = _quartile_bounds(k_sorted)
    o_q1, o_med, o_q3 = _quartile_bounds(o_sorted)
    m_q1, m_med, m_q3 = _quartile_bounds(m_sorted)
    iqr_k = k_q3 - k_q1
    iqr_o = o_q3 - o_q1
    iqr_m = m_q3 - m_q1

    drift_veto_thresh = (o_q3 - o_med) / max(o_med, 1e-9) if o_med > 0 else 0.5
    drift_score_mult = max(iqr_k + iqr_o + iqr_m, 1.0)

    mse_p75_mult = 1.0 + (iqr_o / max(o_q1, 1e-9)) if o_q1 > 0 else 2.0
    mse_p75_mult = min(max(mse_p75_mult, 1.25), 3.0)

    k_scale = 1.0 / max(k_q3, 1e-9)
    o_scale = 1.0 / max(o_q3, 1e-9)
    m_scale = 1.0 / max(m_q3, 1e-9)
    penalty_cap = min(iqr_k / max(k_q3, 1e-9), 0.49)

    return {
        "drift_veto_thresh": float(min(max(drift_veto_thresh, 0.1), 1.0)),
        "drift_score_mult": float(drift_score_mult),
        "mse_release_o_min": float(max(o_q3, o_med + iqr_o * 0.5)),
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
        "search_low_floor": float(m_q1 / max(m_q3, 1e-9)) if m_q3 > 0 else 0.5,
        "search_low_penalty_cap": float(penalty_cap),
        "search_low_clip_max": float(min(1.0, 0.5 + o_scale * o_med)),
        "search_low_gray_clip_max": float(min(0.99, 0.5 + k_scale * k_med)),
        "alpha_floor": float(min(0.5 + k_scale * k_med, 0.99)),
        "alpha_clip_max": 0.99,
        "beta_floor": float(min(0.5 + o_scale * o_med, 0.99)),
        "beta_clip_max": 0.99,
        "ff2_suffix_min_count": max(1, (n + 19) // 20),
        "score_o_weight": float(iqr_o / max(o_q1, 1e-9)) if o_q1 > 0 else 1.0,
        "score_m_weight": float(iqr_m / max(m_q1, 1e-9)) if m_q1 > 0 else 0.5,
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

    def _class_attn_gate(entries: List[Dict[str, float]], pool_m: List[float]) -> float:
        if not entries:
            return _tukey_upper(pool_m)
        am = _sorted_pool([e["m"] for e in entries])
        return _tukey_upper(am)

    def _class_outlier_gate(entries: List[Dict[str, float]], pool_o: List[float]) -> float:
        if not entries:
            return _tukey_upper(pool_o)
        oo = _sorted_pool([e["o"] for e in entries])
        return _tukey_upper(oo)

    qkv_gate = _class_attn_gate(by_class.get("qkv", []), all_m)
    toout_gate = _class_attn_gate(by_class.get("toout", []), all_m)
    ff2_gate = _class_attn_gate(by_class.get("ff2", []), all_m)
    qkv_o_gate = _class_outlier_gate(by_class.get("qkv", []), all_o)
    toout_o_gate = _class_outlier_gate(by_class.get("toout", []), all_o)

    engine = _derive_engine_tunables(all_k, all_o, all_m, k_sorted, o_sorted, m_sorted)

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
        **engine,
    }


def _derive_engine_tunables_int8(
    all_k: List[float],
    all_o: List[float],
    all_m: List[float],
    k_sorted: List[float],
    o_sorted: List[float],
    m_sorted: List[float],
) -> Dict[str, float]:
    """INT8 engine tunables linked to HSWQ V4 weighted histogram (VETO).

    Profile stats (kurtosis / outlier_ratio / abs_max) are weight-space.
    Do NOT multiply by 127/448 — that collapses mse_release / gray bands and
    starves or floods V4-histogram VETO candidates.

    Link contract with quantize_sdxl_hswq_v3.0.py:
      - Pack amax = absmax (search_low_* = 1.0). Natural for symmetric INT8.
      - V4 + INT8Quantizer estimated_mse is required for MSE-guided VETO.
      - mse_release_o_min / k_max / m_max / mse_p75_multiplier select which
        hard-VETO layers enter V4 histogram release, and the P75×mult
        threshold. These must stay weight-space.
    """
    n = len(all_k)
    k_q1, k_med, k_q3 = _quartile_bounds(k_sorted)
    o_q1, o_med, o_q3 = _quartile_bounds(o_sorted)
    m_q1, m_med, m_q3 = _quartile_bounds(m_sorted)
    iqr_k = k_q3 - k_q1
    iqr_o = o_q3 - o_q1
    iqr_m = m_q3 - m_q1

    drift_veto_thresh = (o_q3 - o_med) / max(o_med, 1e-9) if o_med > 0 else 0.5
    drift_score_mult = max(iqr_k + iqr_o + iqr_m, 1.0)

    # V4-histogram VETO: baseline P75×mult from this checkpoint's outlier IQR.
    mse_p75_mult = 1.0 + (iqr_o / max(o_q1, 1e-9)) if o_q1 > 0 else 2.0
    mse_p75_mult = min(max(mse_p75_mult, 1.25), 3.0)

    k_scale = 1.0 / max(k_q3, 1e-9)
    o_scale = 1.0 / max(o_q3, 1e-9)
    m_scale = 1.0 / max(m_q3, 1e-9)

    return {
        "drift_veto_thresh": float(min(max(drift_veto_thresh, 0.1), 1.0)),
        "drift_score_mult": float(drift_score_mult),
        # Weight-space gray-zone gates for V4 histogram MSE release candidates
        # (same units as profile outlier_ratio / abs_max / kurtosis).
        "mse_release_o_min": float(max(o_q3, o_med + iqr_o * 0.5)),
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
        "alpha_floor": float(min(max(0.5 + k_scale * k_med, 0.5), 0.99)),
        "alpha_clip_max": 0.99,
        "beta_floor": float(min(0.5 + o_scale * o_med, 0.99)),
        "beta_clip_max": 0.99,
        "ff2_suffix_min_count": max(1, (n + 19) // 20),
        "score_o_weight": float(iqr_o / max(o_q1, 1e-9)) if o_q1 > 0 else 1.0,
        "score_m_weight": float(iqr_m / max(m_q1, 1e-9)) if m_q1 > 0 else 0.5,
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
    mad_vals = [v for v in mad_vals if v > 0.0]
    if len(mad_vals) < 4:
        # Old profiles without mad_outlier_pct: safe neutral fallback until re-profile.
        return {
            "attn_mad_pct_floor": 15.0,
            "attn_mad_q3": 15.0,
            "attn_mad_gap_o_max": float(
                max(attn_qkv_outlier, attn_toout_outlier, 1e-9)
            ),
            "attn_mad_from_profile": 0.0,
        }

    mad_sorted = _sorted_pool(mad_vals)
    _, _, mad_q3 = _quartile_bounds(mad_sorted)
    mad_floor = float(_tukey_upper(mad_sorted))
    gap_o_max = float(max(attn_qkv_outlier, attn_toout_outlier, 1e-9))
    return {
        "attn_mad_pct_floor": mad_floor,
        "attn_mad_q3": float(mad_q3),
        "attn_mad_gap_o_max": gap_o_max,
        "attn_mad_from_profile": 1.0,
    }


def derive_veto_tunables_int8(profile: Dict[str, Any]) -> Dict[str, Any]:
    """INT8 VETO + V4-histogram link for SDXL V3.0.

    Single analyze → quantize contract:
      1) Hard VETO fences (extreme_*/attn_*/ff2_*) from this checkpoint's
         weight-space distribution (no 127/448 collapse).
      2) Engine mse_release_* / mse_p75_multiplier drive which VETO layers
         enter HSWQWeightedHistogramOptimizerV4 + INT8Quantizer estimated_mse
         release, and the threshold the histogram uses.
      3) search_low_* = 1.0 → pack amax is absmax (natural INT8); V4 histogram
         remains mandatory for (2).

    MAD% floors for attn gap-fill come from this profile's mad_outlier_pct.
    """
    profile = _normalize_profile(profile)
    profile = _unet_only_profile(profile)
    layers = profile.get("layers", {})
    if not layers:
        raise ValueError("profile has no layers")

    base = derive_veto_tunables(profile)

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

    int8_engine = _derive_engine_tunables_int8(
        all_k, all_o, all_m, k_sorted, o_sorted, m_sorted
    )

    # Weight-space Tukey gates (same units as live abs_max / outlier_ratio).
    def _class_attn_gate_int8(entries: List[Dict[str, float]], pool_m: List[float]) -> float:
        if not entries:
            return _tukey_upper(pool_m)
        am = _sorted_pool([e["m"] for e in entries])
        return _tukey_upper(am)

    def _class_outlier_gate_int8(entries: List[Dict[str, float]], pool_o: List[float]) -> float:
        if not entries:
            return _tukey_upper(pool_o)
        oo = _sorted_pool([e["o"] for e in entries])
        return _tukey_upper(oo)

    qkv_gate_i = _class_attn_gate_int8(by_class.get("qkv", []), all_m)
    toout_gate_i = _class_attn_gate_int8(by_class.get("toout", []), all_m)
    ff2_gate_i = _class_attn_gate_int8(by_class.get("ff2", []), all_m)
    qkv_o_gate_i = _class_outlier_gate_int8(by_class.get("qkv", []), all_o)
    toout_o_gate_i = _class_outlier_gate_int8(by_class.get("toout", []), all_o)

    mad_tunables = _derive_int8_attn_mad_tunables(
        by_class, qkv_o_gate_i, toout_o_gate_i
    )

    base.update({
        "attn_qkv_absmax": float(qkv_gate_i),
        "attn_qkv_outlier": float(qkv_o_gate_i),
        "attn_toout_absmax": float(toout_gate_i),
        "attn_toout_outlier": float(toout_o_gate_i),
        "attn_ff2_absmax": float(ff2_gate_i),
        "extreme_outlier": float(base["extreme_outlier"]),
        "huge_magnitude": float(base["huge_magnitude"]),
        "quant_format": "int8_tensorwise",
        **mad_tunables,
    })
    base.update(int8_engine)
    return base


def int8_fp16_budget_analyze_severity(
    *,
    kurtosis: float,
    outlier_ratio: float,
    abs_max: float,
    tunables: Dict[str, Any],
    is_hard_veto: bool = False,
    layer_name: str = "",
    mad_outlier_pct: float = 0.0,
) -> float:
    """INT8-only: analyze-side severity = this checkpoint's character on one layer.

    FP8 path must NOT call this. Denominators are derive_veto_tunables_int8
    fences for THIS model (extreme_*, attn_* gates, attn_mad_pct_floor) —
    not a fixed recipe. Higher = more FP16-deserving under --fp16_budget_mb.
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
    # Excess over INT8 hard fences (1.0 == at fence).
    severity = max(o / eo, 0.0) + max(k / ek, 0.0) + max(m / hm, 0.0)

    # Attn-class character from the same INT8 tunables (model-specific gates).
    name = str(layer_name or "")
    if name.endswith((".to_q", ".to_k", ".to_v")):
        aq = max(float(tunables.get("attn_qkv_absmax", hm)), 1e-6)
        ao = max(float(tunables.get("attn_qkv_outlier", eo)), 1e-6)
        severity += max(m / aq, 0.0) + max(o / ao, 0.0)
    elif name.endswith(".to_out.0"):
        aq = max(float(tunables.get("attn_toout_absmax", hm)), 1e-6)
        ao = max(float(tunables.get("attn_toout_outlier", eo)), 1e-6)
        severity += max(m / aq, 0.0) + max(o / ao, 0.0)

    mad_floor = float(tunables.get("attn_mad_pct_floor", 0.0) or 0.0)
    mad = float(mad_outlier_pct or 0.0)
    if mad_floor > 0.0 and mad > 0.0:
        severity += max(mad / mad_floor, 0.0)

    if is_hard_veto:
        severity *= 1.5
    return float(severity)


def int8_fp16_budget_priority(
    v4_estimated_mse: float,
    analyze_severity: float,
) -> float:
    """INT8-only combined priority: V4 MSE primary × analyze severity.

    priority = estimated_mse * (1 + severity)
    High V4 damage under INT8 absmax pack + nasty analyze stats → keep FP16 first
    inside the +300 MiB budget. FP8 must not use this ranking.
    """
    mse = max(float(v4_estimated_mse), 0.0)
    sev = max(float(analyze_severity), 0.0)
    return mse * (1.0 + sev)


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
        sev = int8_fp16_budget_analyze_severity(
            kurtosis=k,
            outlier_ratio=o,
            abs_max=m,
            tunables=tunables,
            is_hard_veto=name in hard,
            layer_name=name,
            mad_outlier_pct=mad,
        )
        out[name] = {
            "kurtosis": k,
            "outlier_ratio": o,
            "abs_max": m,
            "mad_outlier_pct": mad,
            "severity": float(sev),
        }
    return out


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
        return {
            "v4_ran": False,
            "complete": False,
            "reason": "dualmonitor_importance_required",
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
    p75_mult = float(tunables.get("mse_p75_multiplier", 2.0))
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

    # Match v3.0: alpha=0, beta=1 → pure calibration magnitude × DualMonitor.
    quantizer = INT8Quantizer(device=device)
    optimizer = HSWQWeightedHistogramOptimizerV4(
        bins=8192,
        num_candidates=1000,
        refinement_iterations=10,
        device=device,
        alpha=0.0,
        beta=1.0,
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
            result = optimizer.compute_optimal_amax_with_stats(
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
            result = optimizer.compute_optimal_amax_with_stats(
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
        "alpha": 0.0,
        "beta": 1.0,
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
    """Auto optimal INT8 settings = analyze × V4 × DualMonitor (r32 calib).

    Triple contract (no shortcuts):
      1) analyze derive_veto_tunables_int8 (weight-space gates)
      2) V4 + INT8Quantizer estimated_mse @ absmax
      3) DualMonitor channel_importance from 32-sample / 25-step calibration
    """
    unet_prof = _unet_only_profile(profile)
    tunables = derive_veto_tunables_int8(unet_prof)
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
            "attn_mad_pct_floor": float(tunables.get("attn_mad_pct_floor", 15.0)),
            "unet_layer_count": len(layers),
        },
        "v4": {
            "required": True,
            "role": "MSE-guided VETO at absmax with DualMonitor importance",
            "quantizer": "INT8Quantizer",
            "optimizer": "HSWQWeightedHistogramOptimizerV4",
            "search_range": [1.0, 1.0],
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
        if v4.get("complete") and "mse_release_threshold" in v4:
            optimal["recommended_mse_release_threshold"] = float(
                v4["mse_release_threshold"]
            )
            optimal["recommended_safe_p75_mse"] = float(v4["safe_p75_mse"])
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
        "int8_mse_p75_multiplier": float(i8.get("mse_p75_multiplier", 2.0)),
        "int8_optimal_complete": bool(opt.get("complete", False)),
        "calib_contract": opt.get("calib_contract"),
        "v4_ran": bool(v4opt.get("v4_ran", False)),
        "v4_complete": bool(v4opt.get("complete", False)),
        "v4_reason": v4opt.get("reason"),
        "v4_safe_p75_mse": v4opt.get("safe_p75_mse"),
        "v4_mse_release_threshold": v4opt.get("mse_release_threshold"),
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
    print(
        f"[analyze×V4 INT8] search_low={summary.get('int8_search_low')} "
        f"mse_release_o_min={summary.get('int8_mse_release_o_min')} "
        f"mse_p75_mult={summary.get('int8_mse_p75_multiplier')} "
        f"v4_ran={summary.get('v4_ran')} "
        f"v4_p75_mse={summary.get('v4_safe_p75_mse')} "
        f"v4_threshold={summary.get('v4_mse_release_threshold')}"
    )


if __name__ == "__main__":
    main()

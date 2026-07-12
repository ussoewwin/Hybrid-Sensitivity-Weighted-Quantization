#!/usr/bin/env python3
"""
SDXL UNet layer distribution analyzer for HSWQ V2.0.

Produces per-layer statistics and derive_veto_tunables() for quantize_sdxl_hswq_v2.0.py.
All VETO thresholds are derived from profile layer distributions (no tuned magic constants).
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
    """INT8-specific engine tunables.

    INT8 symmetric per-tensor has 127 positive levels (uniform grid) vs
    FP8E4M3's non-linear grid with ~240 positive levels concentrated near 0.
    Key differences vs FP8:
      1. max_representable = 127 (not 448) → outlier_ratio thresholds are
         scaled by 127/448 ≈ 0.283. A weight with abs_max=448 that was at
         the FP8 ceiling is now 3.5x the INT8 ceiling, so VETO must trigger
         at proportionally lower abs_max.
      2. Zero-near resolution is constant (linear grid), so HSWQ's
         outlier-clipping is more valuable: clipping tight outliers yields
         uniform resolution gain across the whole range.
      3. Dynamic range is effectively wider for in-distribution weights
         (no 448 cap), so search_low can be more aggressive for clean
         layers, but gray-zone layers (moderate outliers) need stricter
         protection than FP8 because the resolution loss near zero is
         absolute, not relative.
    """
    n = len(all_k)
    k_q1, k_med, k_q3 = _quartile_bounds(k_sorted)
    o_q1, o_med, o_q3 = _quartile_bounds(o_sorted)
    m_q1, m_med, m_q3 = _quartile_bounds(m_sorted)
    iqr_k = k_q3 - k_q1
    iqr_o = o_q3 - o_q1
    iqr_m = m_q3 - m_q1

    # INT8 scale factor: 127 / 448 ≈ 0.2835
    int8_scale_factor = 127.0 / 448.0

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
        # INT8: outlier VETO release at lower o (dynamic range is wider, so
        # moderate outliers are less damaging). Scale o_min by int8 factor.
        "mse_release_o_min": float(max(o_q3, o_med + iqr_o * 0.5) * int8_scale_factor * 1.2),
        "mse_release_k_max": float(k_q3),
        "mse_release_m_max": float(m_q3 * int8_scale_factor),
        "mse_p75_multiplier": float(mse_p75_mult),
        "k_scale": float(k_scale),
        "o_scale": float(o_scale),
        "m_scale": float(m_scale),
        "k_gray_lo": float(k_q1),
        "k_gray_hi": float(k_q3),
        "o_gray_lo": float(o_q1 * int8_scale_factor),
        "o_gray_hi": float(o_q3 * int8_scale_factor),
        "m_gray_lo": float(m_q1 * int8_scale_factor),
        "m_gray_hi": float(m_q3 * int8_scale_factor),
        "search_low_floor": float(m_q1 / max(m_q3, 1e-9)) if m_q3 > 0 else 0.5,
        "search_low_penalty_cap": float(penalty_cap),
        "search_low_clip_max": float(min(1.0, 0.5 + o_scale * o_med)),
        # k_med can be negative; clip must stay in [0.5, 0.99] for search_low.
        "search_low_gray_clip_max": float(
            min(0.99, max(0.5, 0.5 + k_scale * k_med))
        ),
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
    """INT8-specific VETO tunables for SDXL V3.0.

    Reuses the FP8 `derive_veto_tunables` shape but replaces the engine block
    with INT8-tuned gray/search_low thresholds. extreme_outlier / huge_magnitude
    stay in weight-space units (same as FP8 fences) — they are not 127/448-
    scaled. Attn absmax gates still use int8_sf because those compare against
    scaled gate paths. Also tightens per-class attn gates because INT8's
    uniform grid loses near-zero resolution that FP8E4M3 had natively.

    MAD% floors for attn gap-fill are derived from this profile's layer
    mad_outlier_pct distribution (automatic per checkpoint, not named).
    """
    profile = _normalize_profile(profile)
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

    int8_scale_factor = 127.0 / 448.0

    def _class_attn_gate_int8(entries: List[Dict[str, float]], pool_m: List[float]) -> float:
        if not entries:
            return _tukey_upper(pool_m) * int8_scale_factor
        am = _sorted_pool([e["m"] for e in entries])
        return _tukey_upper(am) * int8_scale_factor

    def _class_outlier_gate_int8(entries: List[Dict[str, float]], pool_o: List[float]) -> float:
        if not entries:
            return _tukey_upper(pool_o) * int8_scale_factor
        oo = _sorted_pool([e["o"] for e in entries])
        return _tukey_upper(oo) * int8_scale_factor

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
        # outlier_ratio is dimensionless (abs_max/std); abs_max is weight-space.
        # Do NOT multiply by 127/448 — that collapses fences and VETO-explodes
        # (wai UNet: 778/794), destroying the SSIM 0.98 / MSE~12 path.
        "extreme_outlier": float(base["extreme_outlier"]),
        "huge_magnitude": float(base["huge_magnitude"]),
        "quant_format": "int8_tensorwise",
        **mad_tunables,
    })
    base.update(int8_engine)
    return base


def enrich_profile_with_derived(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute per-layer profile_score and attach veto_tunables + summary."""
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
    }
    return profile


# ---------------------------------------------------------------------------
# CLI: build profile from safetensors
# ---------------------------------------------------------------------------

def analyze_unet(path: str) -> Dict[str, Any]:
    state = load_file(path)
    layers: Dict[str, Any] = {}
    for name, tensor in state.items():
        if not name.endswith(".weight"):
            continue
        if len(tensor.shape) < 2:
            continue
        stats = _layer_stats(tensor)
        layers[name] = stats
    profile: Dict[str, Any] = {"source": path, "layers": layers}
    return enrich_profile_with_derived(profile)


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


if __name__ == "__main__":
    main()

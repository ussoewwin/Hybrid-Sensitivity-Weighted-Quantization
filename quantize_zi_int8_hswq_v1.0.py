"""
Z-Image / NextDiT INT8 quantization — HSWQ V1.0
================================================

ZI-format pipeline (load / calib / Static+Structural VETO / Z-Anime):
  same infrastructure as quantize_zib_hswq_v2.0.py (loaded via importlib).

INT8 FP16 protect (HSWQ — per-checkpoint auto analysis → auto-optimal):
  - Owner hard frame: FP16 overhead vs all-INT8 == 700 MiB exactly.
  - DualMonitor sensitivity × analyze severity × V4 estimated_mse rank
    with infinite THIS-model ranking / priority branches (no fixed formula,
    no keep_ratio % cut). Extreme fill under 700 MiB only truncates.
  - Pack amax stays absmax (tensorwise) or per-out-channel (Card 3).
  - Card 1 (--bias_correction): bias += -(W_q - W) @ mu_x
    mu_x = DualMonitor.channel_act_mean from ZITCalibrationPipeline.
  - Card 3 (--per_channel_int8): per-out-channel scale (O,1) / (O,1,1,1).
    Format tag stays int8_tensorwise for ComfyUI kitchen dequant.

CLI style matches ZIB v2.0 (--input/--output/--calib_file/--clip_path/...).
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import subprocess
import sys

import torch
from safetensors.torch import save_file
from tqdm import tqdm
from dataclasses import dataclass

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "ComfyUI-master"))
histogram_dir = os.path.join(current_dir, "histogram")
if histogram_dir not in sys.path:
    sys.path.insert(0, histogram_dir)

from weighted_histogram_mse_v4_int8 import (
    HSWQWeightedHistogramOptimizerV4,
    INT8Quantizer,
)


def _load_zib_v20():
    """Load quantize_zib_hswq_v2.0.py (ZI-format engine)."""
    path = os.path.join(current_dir, "quantize_zib_hswq_v2.0.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ZI engine not found: {path}")
    spec = importlib.util.spec_from_file_location("quantize_zib_hswq_v2_0", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["quantize_zib_hswq_v2_0"] = mod
    spec.loader.exec_module(mod)
    return mod

# ---------------------------------------------------------------------------
# INT8 FP16 budget surface (formerly loaded via importlib).
# Same helpers previously called on the loaded budget module; ceiling is 700.
# Depends on: analyze/analyze_sdxl_distribution.py + histogram V4 INT8.
# ---------------------------------------------------------------------------

# Owner hard ceiling for FP16 overhead vs all-INT8. Auto analysis may only
# optimize INSIDE this frame. Not a thinking-stop formula constant.
FP16_BUDGET_MB_HARD = 700.0


def _require_fp16_budget_mb_hard(budget_mb: float) -> float:
    """Refuse any fp16_budget_mb other than the owner hard ceiling (700)."""
    b = float(budget_mb)
    if abs(b - FP16_BUDGET_MB_HARD) > 1e-6:
        raise ValueError(
            f"fp16_budget_mb must be exactly {FP16_BUDGET_MB_HARD:g} MiB "
            f"(owner hard ceiling; auto-optimal settings are inside this "
            f"frame only  -  never outside). Got {b}."
        )
    return FP16_BUDGET_MB_HARD

@dataclass(frozen=True)
class SdxlVetoTunables:
    extreme_kurtosis: float
    extreme_outlier: float
    huge_magnitude: float
    attn_qkv_absmax: float
    attn_qkv_outlier: float
    attn_toout_absmax: float
    attn_toout_outlier: float
    ff2_outlier_live: float
    ff2_profile_outlier: float
    ff2_profile_score_cutoff: float
    ff2_auto_full_class: bool = False
    drift_veto_thresh: float = 0.0
    drift_score_mult: float = 1.0
    mse_release_o_min: float = 0.0
    mse_release_k_max: float = 0.0
    mse_release_m_max: float = 0.0
    mse_p75_multiplier: float = 1.0
    k_scale: float = 0.0
    o_scale: float = 0.0
    m_scale: float = 0.0
    k_gray_lo: float = 0.0
    k_gray_hi: float = 0.0
    o_gray_lo: float = 0.0
    o_gray_hi: float = 0.0
    m_gray_lo: float = 0.0
    m_gray_hi: float = 0.0
    search_low_floor: float = 1.0
    search_low_penalty_cap: float = 0.0
    search_low_clip_max: float = 1.0
    search_low_gray_clip_max: float = 1.0
    alpha_floor: float = 0.0
    alpha_clip_max: float = 0.0
    beta_floor: float = 0.0
    beta_clip_max: float = 0.0
    ff2_suffix_min_count: int = 4
    score_o_weight: float = 1.0
    score_m_weight: float = 1.0
    score_k_weight: float = 1.0
    quant_format: str = "int8_tensorwise"
    attn_mad_pct_floor: float = 0.0
    attn_mad_q3: float = 0.0
    attn_mad_p99: float = 0.0
    attn_mad_gap_o_max: float = 0.0
    attn_mad_from_profile: float = 0.0
    # Continuous MAD branch fingerprint (THIS pool IQR death → soft→Tukey; P99 tip-only).
    attn_mad_collapse: float = 0.0
    attn_mad_iqr: float = 0.0
    # Autonomous (from derive_int8_autonomous_tunables):
    sens_veto_percentile: float = 100.0
    sens_veto_keep_ratio_gate: float = 0.0
    bias_correction_top_ratio: float = 1.0
    auto_keep_ratio: float = 0.0
    fp16_budget_mb: float = 700.0
    fp16_budget_bytes: int = 734003200
    n_unet_layers: int = 0
    autonomous: bool = False
    # V4 Full-SVD×RMS mix weight from THIS multi-axis analyze character
    # (kurtosis∪outlier∪magnitude). Must be > 0 for non-degenerate THIS  - 
    # alpha_auto==0 is SVD cut (rebellion), not a valid default outcome.
    alpha_auto: float = 0.0

    # Required from derive_int8_autonomous_tunables  -  no silent default holes
    # after deleting accommodation clips (auto analysis → auto-optimal).
    _FROM_DICT_REQUIRED = (
        "extreme_kurtosis",
        "extreme_outlier",
        "huge_magnitude",
        "attn_qkv_absmax",
        "attn_qkv_outlier",
        "attn_toout_absmax",
        "attn_toout_outlier",
        "ff2_outlier_live",
        "ff2_profile_outlier",
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
        "bias_correction_top_ratio",
        "score_k_weight",
        "score_o_weight",
        "score_m_weight",
        "quant_format",
        "autonomous",
        "fp16_budget_mb",
    )

    @classmethod
    def from_dict(cls, d: dict) -> "SdxlVetoTunables":
        missing = [k for k in cls._FROM_DICT_REQUIRED if k not in d]
        if missing:
            raise ValueError(
                "SdxlVetoTunables.from_dict missing auto-optimal keys "
                f"{missing}. Run derive_int8_autonomous_tunables  -  do not "
                "fill deleted clip holes with dataclass defaults."
            )
        if not bool(d["autonomous"]):
            raise ValueError(
                "SdxlVetoTunables.from_dict requires autonomous=True "
                "(THIS-profile auto analysis → auto-optimal)"
            )
        if str(d["quant_format"]) != "int8_tensorwise":
            raise ValueError("INT8 SdxlVetoTunables requires quant_format=int8_tensorwise")
        if abs(float(d["fp16_budget_mb"]) - float(FP16_BUDGET_MB_HARD)) > 1e-6:
            raise ValueError(
                f"fp16_budget_mb must be {float(FP16_BUDGET_MB_HARD):g}"
            )
        if float(d["search_low_floor"]) != 1.0:
            raise ValueError("INT8 search_low_floor must be 1.0 (absmax auto-optimal)")
        if float(d["mse_p75_multiplier"]) <= 0.0:
            raise ValueError("mse_p75_multiplier must be > 0 from THIS profile")
        return cls(
            extreme_kurtosis=float(d["extreme_kurtosis"]),
            extreme_outlier=float(d["extreme_outlier"]),
            huge_magnitude=float(d["huge_magnitude"]),
            attn_qkv_absmax=float(d["attn_qkv_absmax"]),
            attn_qkv_outlier=float(d["attn_qkv_outlier"]),
            attn_toout_absmax=float(d["attn_toout_absmax"]),
            attn_toout_outlier=float(d["attn_toout_outlier"]),
            ff2_outlier_live=float(d["ff2_outlier_live"]),
            ff2_profile_outlier=float(d["ff2_profile_outlier"]),
            ff2_profile_score_cutoff=float(d.get("ff2_profile_score_cutoff", 0.0)),
            ff2_auto_full_class=bool(d.get("ff2_auto_full_class", False)),
            drift_veto_thresh=float(d["drift_veto_thresh"]),
            drift_score_mult=float(d["drift_score_mult"]),
            mse_release_o_min=float(d["mse_release_o_min"]),
            mse_release_k_max=float(d["mse_release_k_max"]),
            mse_release_m_max=float(d["mse_release_m_max"]),
            mse_p75_multiplier=float(d["mse_p75_multiplier"]),
            k_scale=float(d["k_scale"]),
            o_scale=float(d["o_scale"]),
            m_scale=float(d["m_scale"]),
            k_gray_lo=float(d["k_gray_lo"]),
            k_gray_hi=float(d["k_gray_hi"]),
            o_gray_lo=float(d["o_gray_lo"]),
            o_gray_hi=float(d["o_gray_hi"]),
            m_gray_lo=float(d["m_gray_lo"]),
            m_gray_hi=float(d["m_gray_hi"]),
            search_low_floor=float(d["search_low_floor"]),
            search_low_penalty_cap=float(d["search_low_penalty_cap"]),
            search_low_clip_max=float(d["search_low_clip_max"]),
            search_low_gray_clip_max=float(d["search_low_gray_clip_max"]),
            alpha_floor=float(d["alpha_floor"]),
            alpha_clip_max=float(d["alpha_clip_max"]),
            beta_floor=float(d["beta_floor"]),
            beta_clip_max=float(d["beta_clip_max"]),
            ff2_suffix_min_count=int(d.get("ff2_suffix_min_count", 4)),
            score_o_weight=float(d["score_o_weight"]),
            score_m_weight=float(d["score_m_weight"]),
            score_k_weight=float(d["score_k_weight"]),
            quant_format=str(d["quant_format"]),
            attn_mad_pct_floor=float(d["attn_mad_pct_floor"]),
            attn_mad_q3=float(d["attn_mad_q3"]),
            attn_mad_p99=float(d["attn_mad_p99"]),
            attn_mad_gap_o_max=float(d["attn_mad_gap_o_max"]),
            attn_mad_from_profile=float(d["attn_mad_from_profile"]),
            attn_mad_collapse=float(d.get("attn_mad_collapse", 0.0)),
            attn_mad_iqr=float(d.get("attn_mad_iqr", 0.0)),
            sens_veto_percentile=float(d.get("sens_veto_percentile", 100.0)),
            sens_veto_keep_ratio_gate=float(d.get("sens_veto_keep_ratio_gate", 0.0)),
            bias_correction_top_ratio=float(d["bias_correction_top_ratio"]),
            auto_keep_ratio=float(d.get("auto_keep_ratio", 0.0)),
            fp16_budget_mb=float(d["fp16_budget_mb"]),
            fp16_budget_bytes=int(d.get("fp16_budget_bytes", 700 * 1024 * 1024)),
            n_unet_layers=int(d.get("n_unet_layers", 0)),
            autonomous=True,
            alpha_auto=float(d["alpha_auto"]),
        )

    def as_dict(self) -> dict:
        return {
            "extreme_kurtosis": self.extreme_kurtosis,
            "extreme_outlier": self.extreme_outlier,
            "huge_magnitude": self.huge_magnitude,
            "attn_qkv_absmax": self.attn_qkv_absmax,
            "attn_qkv_outlier": self.attn_qkv_outlier,
            "attn_toout_absmax": self.attn_toout_absmax,
            "attn_toout_outlier": self.attn_toout_outlier,
            "ff2_outlier_live": self.ff2_outlier_live,
            "ff2_profile_outlier": self.ff2_profile_outlier,
            "ff2_profile_score_cutoff": self.ff2_profile_score_cutoff,
            "ff2_auto_full_class": self.ff2_auto_full_class,
            "drift_veto_thresh": self.drift_veto_thresh,
            "drift_score_mult": self.drift_score_mult,
            "mse_release_o_min": self.mse_release_o_min,
            "mse_release_k_max": self.mse_release_k_max,
            "mse_release_m_max": self.mse_release_m_max,
            "mse_p75_multiplier": self.mse_p75_multiplier,
            "k_scale": self.k_scale,
            "o_scale": self.o_scale,
            "m_scale": self.m_scale,
            "k_gray_lo": self.k_gray_lo,
            "k_gray_hi": self.k_gray_hi,
            "o_gray_lo": self.o_gray_lo,
            "o_gray_hi": self.o_gray_hi,
            "m_gray_lo": self.m_gray_lo,
            "m_gray_hi": self.m_gray_hi,
            "search_low_floor": self.search_low_floor,
            "search_low_penalty_cap": self.search_low_penalty_cap,
            "search_low_clip_max": self.search_low_clip_max,
            "search_low_gray_clip_max": self.search_low_gray_clip_max,
            "alpha_floor": self.alpha_floor,
            "alpha_clip_max": self.alpha_clip_max,
            "beta_floor": self.beta_floor,
            "beta_clip_max": self.beta_clip_max,
            "alpha_auto": self.alpha_auto,
            "ff2_suffix_min_count": self.ff2_suffix_min_count,
            "score_k_weight": self.score_k_weight,
            "score_o_weight": self.score_o_weight,
            "score_m_weight": self.score_m_weight,
            "quant_format": self.quant_format,
            "attn_mad_pct_floor": self.attn_mad_pct_floor,
            "attn_mad_q3": self.attn_mad_q3,
            "attn_mad_p99": self.attn_mad_p99,
            "attn_mad_gap_o_max": self.attn_mad_gap_o_max,
            "attn_mad_from_profile": self.attn_mad_from_profile,
            "attn_mad_collapse": self.attn_mad_collapse,
            "attn_mad_iqr": self.attn_mad_iqr,
            "sens_veto_percentile": self.sens_veto_percentile,
            "sens_veto_keep_ratio_gate": self.sens_veto_keep_ratio_gate,
            "bias_correction_top_ratio": self.bias_correction_top_ratio,
            "auto_keep_ratio": self.auto_keep_ratio,
            "fp16_budget_mb": self.fp16_budget_mb,
            "fp16_budget_bytes": self.fp16_budget_bytes,
            "n_unet_layers": self.n_unet_layers,
            "autonomous": self.autonomous,
        }

def resolve_veto_tunables(
    norm_profile: dict,
    profile_summary: dict | None = None,
    *,
    dual_monitors: dict | None = None,
    fp16_budget_mb: float = FP16_BUDGET_MB_HARD,
) -> SdxlVetoTunables:
    """Load INT8 veto_tunables via fully autonomous derivation.

    All knobs (Hard VETO fences, percentile promotions, dynamic ranking
    weights, MSE release gates, bias_correction scope, sens_veto percentile,
    alpha/beta, search_low) come from derive_int8_autonomous_tunables,
    which uses THIS checkpoint's profile + DualMonitor sensitivity
    distribution. fp16_budget_mb is the owner hard ceiling (700 MiB)  - 
    auto settings fill that frame; they do not redefine or exceed it.
    No hardcoded 90.0 / 15.0 / 2.0 / 0.5 / 40.0 recipe constants.
    """
    fp16_budget_mb = _require_fp16_budget_mb_hard(fp16_budget_mb)
    analyze_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze")
    if analyze_dir not in sys.path:
        sys.path.insert(0, analyze_dir)
    from analyze_sdxl_distribution import (
        derive_int8_autonomous_tunables,
        emit_hswq_int8_full_visibility_log,
    )

    if norm_profile:
        sens_map: dict[str, float] = {}
        if dual_monitors:
            for name, mon in dual_monitors.items():
                try:
                    s = float(mon.get_sensitivity())
                except Exception:
                    s = 0.0
                if s > 0.0 and math.isfinite(s):
                    sens_map[name] = s
        derived = derive_int8_autonomous_tunables(
            norm_profile,
            dualmonitor_sensitivities=sens_map if sens_map else None,
            fp16_budget_mb=fp16_budget_mb,
        )
        # derive_int8_autonomous_tunables already emitted the FULL pool / calc /
        # every-layer / every-knob dump. Emit the final resolved dict again so
        # DualMonitor re-resolve is also byte-complete in the same log.
        emit_hswq_int8_full_visibility_log(
            {
                "resolve_stage": "resolve_veto_tunables",
                "n_dualmonitor_sens": int(len(sens_map)),
                "derived_every_key": {
                    str(k): derived[k] for k in sorted(derived.keys(), key=str)
                },
            },
            also_write_file=False,
        )
        return SdxlVetoTunables.from_dict(derived)
    # Stale precomputed veto_tunables without live layer profile = deleted-clip
    # hole risk. Always demand layers + re-derive.
    if profile_summary and isinstance(profile_summary.get("layers"), dict):
        return resolve_veto_tunables(
            profile_summary["layers"],
            dual_monitors=dual_monitors,
            fp16_budget_mb=fp16_budget_mb,
        )
    raise ValueError(
        "resolve_veto_tunables: need THIS checkpoint layer profile for "
        "derive_int8_autonomous_tunables (auto analysis → auto-optimal). "
        "Refuse stale veto_tunables-only load after accommodation-clip purge."
    )

def _dualmonitor_channel_importance(dual_monitors: dict, module_name: str):
    """1D input-channel importance from DualMonitor (32-sample calib contract)."""
    mon = dual_monitors.get(module_name) if dual_monitors else None
    if mon is None:
        return None
    imp = getattr(mon, "channel_importance", None)
    if imp is None:
        return None
    return imp.detach().float()

def _fp16_extra_bytes_vs_int8(weight: torch.Tensor) -> int:
    """Extra bytes of keeping FP16 vs packing INT8 (2B/elem vs 1B/elem → +1B/elem)."""
    return int(weight.numel())


def _measure_v4_mse_absmax_int8(
    *,
    weight: torch.Tensor,
    importance: torch.Tensor | None,
    optimizer: HSWQWeightedHistogramOptimizerV4,
    layer_name: str = "",
) -> float:
    """INT8-only: V4 estimated_mse for FP16 protection candidate ranking.

    Measures weighted-histogram MSE at the natural INT8 pack point (absmax).
    That MSE is the damage score used to decide FP16 keep  -  it is NOT used
    to choose a pack amax (pack stays absmax for INT8).

    Always runs V4 Full-SVD×RMS hybrid (use_svd_leverage=True). When
    DualMonitor channel Importance is present it multiplies the hybrid map;
    when missing, hybrid alone. Cutting SVD because Imp exists is forbidden.
    SVD mix settings + singular values are logged for every layer (no mid-stop).
    """
    result = optimizer.compute_optimal_amax_with_stats_int8_range(
        weight,
        importance=importance,
        use_svd_leverage=True,
        scaled=False,
        search_range=(1.0, 1.0),
        layer_name=layer_name,
    )
    return float(result["estimated_mse"])



def _build_v4_calib_fp16_candidates(
    model: torch.nn.Module,
    dual_monitors: dict,
    target_modules: list,
    *,
    hard_veto_layers: set,
    mse_cache: dict | None,
    alpha: float,
    beta: float,
    device: str,
) -> tuple[set, dict]:
    """Score FP16 protection candidates with histogram V4 on THIS calibration.

    V4's job here: estimated_mse @ absmax for every target Linear/Conv so the
    later 300 MiB budget can rank which layers stay FP16. Pack amax remains
    absmax separately  -  V4 does not search pack scale.

    Always Full-SVD×RMS hybrid; DualMonitor Importance multiplies when present.
    Never skip a measurable layer (skipping collapses FP16 selection).

    Returns (all_v4_scored_names, mse_cache). Does NOT truncate by keep_ratio:
    truncation is only the FP16 budget pass over the FULL priority order of
    (V4-scored U analyze VETO U fence-crossers). Pre-cutting here is the
    hand-wave that collapses quality (~0.92).
    """
    cache = dict(mse_cache or {})
    module_dict = dict(model.named_modules())
    scored: set = set()
    need = []
    for name in target_modules:
        mod = module_dict.get(name)
        if mod is None or not hasattr(mod, "weight") or mod.weight is None:
            continue
        if name in cache:
            scored.add(name)
        else:
            need.append(name)

    trial_optimizer = None
    n_svd_x_imp = 0
    n_svd_only = 0
    if need:
        print(
            f"  [V4→FP16 protect] measuring V4 estimated_mse @ absmax for "
            f"{len(need)} layers (FP16 keep ranking; pack stays absmax; "
            f"cache hit={len(scored)}; analyze VETO={len(hard_veto_layers)}; "
            f"NO keep_ratio pre-cut)..."
        )
        trial_optimizer = HSWQWeightedHistogramOptimizerV4(
            bins=8192, num_candidates=1000, refinement_iterations=10,
            device=device, alpha=alpha, beta=beta,
            quantizer=INT8Quantizer(device=device),
        )
    for name in need:
        if trial_optimizer is None:
            break
        mod = module_dict[name]
        imp = _dualmonitor_channel_importance(dual_monitors, name)
        try:
            v4_mse = _measure_v4_mse_absmax_int8(
                weight=mod.weight.data,
                importance=imp,
                optimizer=trial_optimizer,
                layer_name=name,
            )
            cache[name] = float(v4_mse)
            scored.add(name)
            if imp is None:
                n_svd_only += 1
            else:
                n_svd_x_imp += 1
        except Exception as e:
            print(f"    [V4→FP16 protect] skip {name}: {e}")
            continue
        torch.cuda.empty_cache()

    print(
        f"  [V4→FP16 protect] V4-scored={len(scored)} "
        f"(SVD×Imp={n_svd_x_imp}, SVD-only={n_svd_only}; "
        f"alpha={alpha:.3f}/beta={beta:.3f}) | "
        f"analyze VETO={len(hard_veto_layers)} | "
        f"union → FULL priority (budget only truncates)."
    )
    return scored, cache



def _apply_fp16_budget_cap(
    model: torch.nn.Module,
    keep_layers: set,
    hard_veto_layers: set,
    *,
    budget_mb: float = FP16_BUDGET_MB_HARD,
    norm_profile: dict,
    veto_tunables: SdxlVetoTunables,
    dual_monitors: dict | None,
    mse_cache: dict | None = None,
    alpha: float,
    beta: float,
    device: str = "cuda",
) -> tuple[set, set, dict]:
    """Per-model auto analysis → auto-optimal FP16 set inside the hard ceiling.

    Owner hard ceiling is installed by the caller (SDXL 300 / ZI 700 via
    FP16_BUDGET_MB_HARD). Auto settings fill that frame; they never redefine
    it and never exceed it.

    Linear and Conv compete in ONE ranking (DualMonitor + analyze + V4 MSE +
    infinite branches). Priority weights are derived per-checkpoint — never
    a unified cross-model standard, never Mag-outside exemption.

    alpha/beta MUST be THIS-profile auto-optimal (caller passes
    veto_tunables.alpha_auto mix). Fixed 0.5/0.5 defaults are forbidden.
    """
    if not math.isfinite(float(alpha)) or not math.isfinite(float(beta)):
        raise ValueError(
            f"_apply_fp16_budget_cap: alpha/beta must be finite auto-optimal "
            f"(got alpha={alpha}, beta={beta})"
        )
    budget_mb = _require_fp16_budget_mb_hard(budget_mb)
    analyze_dir = os.path.join(current_dir, "analyze")
    if analyze_dir not in sys.path:
        sys.path.insert(0, analyze_dir)
    from analyze_sdxl_distribution import (
        apply_fp16_infinite_priority_branches,
        apply_fp16_infinite_ranking_branches,
        build_int8_analyze_character_table,
        int8_fp16_budget_analyze_severity,
        int8_fp16_budget_priority,
        derive_priority_combinator,
        _safe_percentile,
        _robust_iqr,
    )

    if str(veto_tunables.quant_format) != "int8_tensorwise":
        raise ValueError(
            "_apply_fp16_budget_cap is INT8-only "
            f"(got quant_format={veto_tunables.quant_format!r})"
        )
    if not dual_monitors:
        raise ValueError(
            "[FP16 budget] DualMonitor maps required for Sensitivity + "
            "V4 Importance; refusing fixed-formula / profile_score fallback."
        )

    tunables_dict = veto_tunables.as_dict()
    budget_bytes = int(budget_mb * 1024 * 1024)

    char_table = build_int8_analyze_character_table(
        {"layers": norm_profile},
        tunables_dict,
        hard_veto_names=hard_veto_layers,
    )

    # All analyze-character layers enter the pool. Continuous severity ranks
    # them later — severity>=1 gate was thinking-stop (drops 0<sev<1).
    pool = set(keep_layers) | set(hard_veto_layers) | set(char_table.keys())

    module_dict = dict(model.named_modules())
    pool = {n for n in pool if n in module_dict and hasattr(module_dict[n], "weight")}

    sens_by_name: dict[str, float] = {}
    for name, mon in dual_monitors.items():
        if name not in module_dict or not hasattr(module_dict[name], "weight"):
            continue
        try:
            s = float(mon.get_sensitivity())
        except Exception:
            s = 0.0
        if s > 0.0 and math.isfinite(s):
            sens_by_name[name] = s
            pool.add(name)

    cache = dict(mse_cache or {})
    measured: list[tuple[str, float, float, float, int]] = []
    skipped_no_weight = []
    skipped_no_v4 = []
    measured_fresh = 0

    need_fresh = [n for n in pool if n not in cache]
    trial_optimizer = None
    if need_fresh:
        print(
            f"  [FP16 budget] THIS-model pool measure: "
            f"analyze={len(char_table)} pool={len(pool)} "
            f"dm_sens={len(sens_by_name)} | V4 fresh={len(need_fresh)} "
            f"(cache={len(cache)})..."
        )
        trial_optimizer = HSWQWeightedHistogramOptimizerV4(
            bins=8192, num_candidates=1000, refinement_iterations=10,
            device=device, alpha=alpha, beta=beta,
            quantizer=INT8Quantizer(device=device),
        )
    else:
        print(
            f"  [FP16 budget] THIS-model pool: "
            f"analyze={len(char_table)} pool={len(pool)} "
            f"dm_sens={len(sens_by_name)} | V4 cached ({len(cache)})"
        )

    for name in sorted(pool):
        mod = module_dict.get(name)
        if mod is None or not hasattr(mod, "weight") or mod.weight is None:
            skipped_no_weight.append(name)
            continue
        dm_sens = float(sens_by_name.get(name, 0.0))
        extra = _fp16_extra_bytes_vs_int8(mod.weight.data)
        row = char_table.get(name, {})
        prof = norm_profile.get(name, {}) if isinstance(norm_profile.get(name), dict) else {}
        is_hv = name in hard_veto_layers
        k = float(row.get("kurtosis", prof.get("kurtosis", 0)) or 0)
        o = float(row.get("outlier_ratio", prof.get("outlier_ratio", 0)) or 0)
        m = float(row.get("abs_max", prof.get("abs_max", 0)) or 0)
        mad = float(row.get("mad_outlier_pct", prof.get("mad_outlier_pct", 0)) or 0)
        ps = float(row.get("profile_score", prof.get("profile_score", 0)) or 0)
        severity = int8_fp16_budget_analyze_severity(
            kurtosis=k,
            outlier_ratio=o,
            abs_max=m,
            tunables=tunables_dict,
            is_hard_veto=is_hv,
            layer_name=name,
            mad_outlier_pct=mad,
            profile_score=ps,
        )

        if name in cache:
            v4_mse = float(cache[name])
        else:
            if trial_optimizer is None:
                skipped_no_v4.append(name)
                continue
            imp = _dualmonitor_channel_importance(dual_monitors, name)
            try:
                v4_mse = _measure_v4_mse_absmax_int8(
                    weight=mod.weight.data,
                    importance=imp,
                    optimizer=trial_optimizer,
                    layer_name=name,
                )
                cache[name] = v4_mse
                measured_fresh += 1
            except Exception as e:
                print(f"    [FP16 budget] V4 MSE failed {name}: {e} -> INT8")
                skipped_no_v4.append(name)
                continue
            torch.cuda.empty_cache()

        measured.append((name, dm_sens, v4_mse, severity, extra))

    # Model-specific auto analysis → auto-optimal ranking branches.
    # DualMonitor / analyze / V4 triples for THIS checkpoint drive continuous
    # knobs (infinite branches). Unified family median/geom floors are banned.
    veto_mask_pre = [name in hard_veto_layers for name, *_ in measured]
    measured, branch_repairs, branch_profile = apply_fp16_infinite_ranking_branches(
        measured, veto_mask_pre,
    )
    print(
        f"  [Infinite branch profile] "
        f"cv(s/v/m)={branch_profile['cv_sens']:.4g}/"
        f"{branch_profile['cv_sev']:.4g}/{branch_profile['cv_mse']:.4g} "
        f"align(s/v/m)={branch_profile['align_sens']:.3f}/"
        f"{branch_profile['align_sev']:.3f}/{branch_profile['align_mse']:.3f} "
        f"dm_starvation={branch_profile['dm_starvation']:.3f} "
        f"γ_sib/blend={branch_profile['gamma_sibling']:.4g}/"
        f"{branch_profile['gamma_blend']:.4g} "
        f"mismatch_gain={branch_profile['mismatch_gain']:.4g} "
        f"repairs={len(branch_repairs)}"
    )
    if branch_repairs:
        for _r in branch_repairs[:16]:
            print(
                f"    [{_r.get('branch', '?')}] {_r['name']}: "
                f"dm={_r.get('dm_sens', float('nan')):.6g} → "
                f"rank={_r.get('ranking_sens', float('nan')):.6g}"
                + (
                    f" skew={_r['skew']:.4g} str={_r['strength']:.4g}"
                    if "skew" in _r else
                    f" excess={_r.get('excess', float('nan')):.4g}"
                )
            )

    # Per-checkpoint combinator from MEASURED sens/sev/mse for THIS model
    # (auto analysis → auto-optimal priority weights; not a fixed formula).
    # Pass Hard VETO masks so anti-aligned axes (e.g. DualMonitor sens that
    # elevates sev=0 layers while demoting analyze VETO) fade automatically.
    sens_all = [float(row[1]) for row in measured]
    sev_all = [float(row[3]) for row in measured]
    mse_all = [float(row[2]) for row in measured]
    veto_mask = [row[0] in hard_veto_layers for row in measured]
    sens_meas = [v for v in sens_all if v > 0]
    sev_meas = list(sev_all)
    mse_meas = [v for v in mse_all if v > 0]
    s_p50 = _safe_percentile(sens_meas, 50.0) if len(sens_meas) >= 2 else 0.0
    s_iqr = _robust_iqr(sens_meas) if len(sens_meas) >= 4 else 0.0
    v_p50 = _safe_percentile(sev_meas, 50.0) if len(sev_meas) >= 2 else 0.0
    v_iqr = _robust_iqr(sev_meas) if len(sev_meas) >= 4 else 0.0
    m_p50 = _safe_percentile(mse_meas, 50.0) if len(mse_meas) >= 2 else 0.0
    m_iqr = _robust_iqr(mse_meas) if len(mse_meas) >= 4 else 0.0
    combinator = derive_priority_combinator(
        s_iqr, v_iqr, m_iqr, s_p50, v_p50, m_p50,
        sens_vals=sens_all,
        sev_vals=sev_all,
        mse_vals=mse_all,
        is_hard_veto=veto_mask,
    )
    _as = combinator.get("align_sens")
    _av = combinator.get("align_sev")
    _am = combinator.get("align_mse")
    _align_txt = (
        f" align(sens/sev/mse)="
        f"{(_as if _as is not None else float('nan')):.3f}/"
        f"{(_av if _av is not None else float('nan')):.3f}/"
        f"{(_am if _am is not None else float('nan')):.3f}"
        if _as is not None
        else ""
    )
    print(
        f"  [Autonomous priority] form={combinator['form']} "
        f"w(sens/sev/mse)={combinator['w_sens']:.3f}/"
        f"{combinator['w_sev']:.3f}/{combinator['w_mse']:.3f} "
        f"refs=({combinator['sens_ref']:.4g}/"
        f"{combinator['sev_ref']:.4g}/{combinator['mse_ref']:.4g})"
        f"{_align_txt}"
    )

    candidates: list[tuple[float, float, float, float, int, str]] = []
    for name, dm_sens, v4_mse, severity, extra in measured:
        priority = int8_fp16_budget_priority(
            dm_sens, v4_mse, severity, combinator=combinator,
        )
        candidates.append((priority, v4_mse, severity, dm_sens, extra, name))

    # Priority continuous sibling branch from the SAME THIS-model profile
    # (not a second unified floor).
    candidates, prio_branch_repairs = apply_fp16_infinite_priority_branches(
        candidates, branch_profile,
    )
    if prio_branch_repairs:
        print(
            f"  [Infinite priority branches] repaired "
            f"{len(prio_branch_repairs)} under THIS family priority space:"
        )
        for _r in prio_branch_repairs[:12]:
            print(
                f"    {_r['name']}: prio {_r['priority_before']:.6g} → "
                f"{_r['priority_after']:.6g} "
                f"skew={_r['skew']:.4g} str={_r['strength']:.4g}"
            )

    candidates.sort(key=lambda x: (-x[0], x[4]))

    # Extreme fill inside the hard ceiling (SDXL 300 / ZI 700):
    # Linear and Conv share ONE THIS-model auto-priority queue.
    selected: set = set()
    used = 0
    dropped: list[tuple[str, int, float, float, float, float]] = []
    kept_detail: list[tuple[str, int, float, float, float, float]] = []
    for priority, v4_mse, severity, dm_sens, extra, name in candidates:
        if used + extra <= budget_bytes:
            selected.add(name)
            used += extra
            kept_detail.append((name, extra, priority, v4_mse, severity, dm_sens))
        else:
            dropped.append((name, extra, priority, v4_mse, severity, dm_sens))

    demoted_veto = hard_veto_layers - selected
    hard_veto_out = hard_veto_layers & selected
    keep_out = set(selected)

    if used > budget_bytes:
        raise RuntimeError(
            f"[FP16 budget] selected set exceeds hard ceiling "
            f"{budget_mb:g} MiB: used={used / (1024 * 1024):.3f} MiB "
            f"({used} bytes > {budget_bytes}). Refusing to proceed."
        )

    stats = {
        "budget_mb": float(budget_mb),
        "budget_bytes": budget_bytes,
        "used_bytes": used,
        "used_mb": used / (1024 * 1024),
        "forced_bytes": 0,
        "forced_mb": 0.0,
        "optional_bytes": int(used),
        "optional_mb": used / (1024 * 1024),
        "total_fp16_mb": used / (1024 * 1024),
        "mag_forced_fp16_count": 0,
        "candidates": len(candidates),
        "pool": len(pool),
        "analyze_character_layers": len(char_table),
        "dm_sensitivity_layers": len(sens_by_name),
        "kept": len(keep_out),
        "dropped": len(dropped),
        "demoted_veto": len(demoted_veto),
        "skipped_no_weight": len(skipped_no_weight),
        "skipped_no_v4": len(skipped_no_v4),
        "measured_fresh_v4": measured_fresh,
        "priority_form": combinator["form"],
        "priority_weights": {
            "sens": combinator["w_sens"],
            "sev": combinator["w_sev"],
            "mse": combinator["w_mse"],
        },
        "priority_align": {
            "sens": combinator.get("align_sens"),
            "sev": combinator.get("align_sev"),
            "mse": combinator.get("align_mse"),
        },
        "ranking": (
            "per_model_auto_analysis_infinite_branches_inside_"
            f"{float(budget_mb):g}mib"
        ),
        "infinite_branch_profile": {
            "cv_sens": branch_profile.get("cv_sens"),
            "cv_sev": branch_profile.get("cv_sev"),
            "cv_mse": branch_profile.get("cv_mse"),
            "align_sens": branch_profile.get("align_sens"),
            "align_sev": branch_profile.get("align_sev"),
            "align_mse": branch_profile.get("align_mse"),
            "dm_starvation": branch_profile.get("dm_starvation"),
            "gamma_sibling": branch_profile.get("gamma_sibling"),
            "gamma_blend": branch_profile.get("gamma_blend"),
            "mismatch_gain": branch_profile.get("mismatch_gain"),
            "prio_sibling_gamma": branch_profile.get("prio_sibling_gamma"),
            "prio_blend_gamma": branch_profile.get("prio_blend_gamma"),
        },
        "infinite_ranking_branch_repairs": len(branch_repairs),
        "infinite_ranking_branch_detail": branch_repairs[:32],
        "infinite_priority_branch_repairs": len(prio_branch_repairs),
        "infinite_priority_branch_detail": prio_branch_repairs[:32],
        "hard_ceiling_mb": float(budget_mb),
        "slack_bytes": max(budget_bytes - used, 0),
        "slack_mb": max(budget_bytes - used, 0) / (1024 * 1024),
        "dropped_detail": dropped[:40],
        "kept_detail": kept_detail[:40],
        "mse_cache_size": len(cache),
    }
    return keep_out, hard_veto_out, stats







class DualMonitorInt8:
    """ZI DualMonitor + signed channel_act_mean for Card 1 bias correction."""

    def __init__(self):
        self.output_sum = 0.0
        self.output_sq_sum = 0.0
        self.count = 0
        self.channel_importance = None
        self.channel_act_mean = None

    def update(self, input_tensor, output_tensor):
        with torch.no_grad():
            out_detached = output_tensor.detach().float()
            out_clamped = torch.clamp(out_detached, -65504.0, 65504.0)
            mean_val = out_clamped.mean().item()
            sq_mean_val = (out_clamped ** 2).mean().item()
            if math.isfinite(mean_val) and math.isfinite(sq_mean_val):
                self.output_sum += mean_val
                self.output_sq_sum += sq_mean_val
            inp = input_tensor.detach().float()
            if inp.dim() == 4:
                current_imp = inp.abs().mean(dim=(0, 2, 3))
                current_act = inp.mean(dim=(0, 2, 3))
            elif inp.dim() == 3:
                current_imp = inp.abs().mean(dim=(0, 1))
                current_act = inp.mean(dim=(0, 1))
            elif inp.dim() == 2:
                current_imp = inp.abs().mean(dim=0)
                current_act = inp.mean(dim=0)
            else:
                current_imp = torch.ones(1, device=inp.device, dtype=torch.float32)
                current_act = torch.zeros(1, device=inp.device, dtype=torch.float32)
            if self.channel_importance is None:
                self.channel_importance = current_imp
                self.channel_act_mean = current_act
            else:
                c = self.count
                self.channel_importance = (
                    self.channel_importance * c + current_imp
                ) / (c + 1)
                self.channel_act_mean = (
                    self.channel_act_mean * c + current_act
                ) / (c + 1)
            self.count += 1

    def get_sensitivity(self):
        if self.count == 0:
            return 0.0
        mean = self.output_sum / self.count
        variance = (self.output_sq_sum / self.count) - mean ** 2
        return variance if math.isfinite(variance) else 0.0


def pack_tensorwise(weight: torch.Tensor):
    """Symmetric per-tensor INT8: scale = amax / 127."""
    w = weight.float()
    amax = max(float(w.abs().max().item()), 1e-6)
    scale = amax / 127.0
    q = (w / scale).round().clamp(-127, 127).to(torch.int8)
    return q, torch.tensor(scale, dtype=torch.float32)


def pack_channelwise(weight: torch.Tensor):
    """Card 3: per-out-channel INT8. Scale shape (O,1) or (O,1,1,1)."""
    w = weight.float()
    reduce_dims = tuple(range(1, w.dim()))
    amax = torch.clamp(w.abs().amax(dim=reduce_dims).reshape(-1), min=1e-6)
    scale = amax / 127.0
    if w.dim() == 4:
        scale_view = scale.view(-1, 1, 1, 1)
        amax_view = amax.view(-1, 1, 1, 1)
    elif w.dim() == 2:
        scale_view = scale.view(-1, 1)
        amax_view = amax.view(-1, 1)
    else:
        raise ValueError(f"unsupported weight ndim={w.dim()} for --per_channel_int8")
    clamped = torch.clamp(w, -amax_view, amax_view)
    q = (clamped / scale_view).round().clamp(-127, 127).to(torch.int8)
    return q, scale_view.to(dtype=torch.float32)


def compute_int8_bias_delta(weight_fp, weight_dq, act_mean):
    """Card 1: delta ≈ (W_q - W) contracted with per-in-channel E[x]."""
    if act_mean is None:
        return None
    err = weight_dq.float() - weight_fp.float()
    mu = act_mean.float().to(device=err.device)
    if err.ndim == 2:
        if mu.numel() != err.shape[1]:
            return None
        return err @ mu
    if err.ndim == 4:
        if mu.numel() != err.shape[1]:
            return None
        return (err * mu.view(1, -1, 1, 1)).sum(dim=(1, 2, 3))
    return None


def _emit_int8_meta(out_dict, prefixed_module, scale):
    out_dict[f"{prefixed_module}.weight_scale"] = scale
    out_dict[f"{prefixed_module}.comfy_quant"] = torch.tensor(
        list(json.dumps({"format": "int8_tensorwise"}).encode("utf-8")),
        dtype=torch.uint8,
    )


def _v4_score_all_fp16_candidates(
    *,
    model,
    dual_monitors,
    target_modules,
    hard_veto_layers,
    alpha,
    beta,
    device,
    mse_cache=None,
):
    """V4 estimated_mse @ absmax for ALL target Linear/Conv — no keep_ratio cut.

    Truncation is only the 700 MiB budget pass over THIS-model priority order
    (auto analysis → infinite branches → extreme fill).
    """
    return _build_v4_calib_fp16_candidates(
        model=model,
        dual_monitors=dual_monitors,
        target_modules=target_modules,
        hard_veto_layers=hard_veto_layers,
        mse_cache=dict(mse_cache or {}),
        alpha=alpha,
        beta=beta,
        device=device,
    )


def main():
    budget_hard = float(FP16_BUDGET_MB_HARD)
    analyze_dir = os.path.join(current_dir, "analyze")
    if analyze_dir not in sys.path:
        sys.path.insert(0, analyze_dir)
    import analyze_sdxl_distribution as _az

    _az.INT8_FP16_BUDGET_MB_HARD = budget_hard

    parser = argparse.ArgumentParser(
        description=(
            "Z-Image / NextDiT INT8 HSWQ V1.0 — 700 MiB FP16 frame + "
            "per-checkpoint auto analysis → infinite-branch fill + "
            "Card 1 bias correction + Card 3 per-channel (ZI format via zib v2.0)."
        )
    )
    # --- ZI CLI (same as quantize_zib_hswq_v2.0.py) ---
    parser.add_argument("--input", type=str, required=True, help="Path to input safetensors")
    parser.add_argument("--output", type=str, required=True, help="Path to output safetensors")
    parser.add_argument("--calib_file", type=str, required=True, help="Calibration prompts text")
    parser.add_argument("--clip_path", type=str, required=True, help="Text encoder safetensors")
    parser.add_argument("--num_calib_samples", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument(
        "--fp16_budget_mb",
        type=float,
        default=budget_hard,
        help=(
            f"Owner hard ceiling for FP16 overhead vs all-INT8 "
            f"(must be exactly {budget_hard:g} MiB). Auto analysis fills "
            f"this frame; never redefine or exceed it."
        ),
    )
    parser.add_argument("--comfy_path", type=str, default=None)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--token", type=str, default=None)
    # --- INT8 cards ---
    parser.add_argument(
        "--per_channel_int8",
        action="store_true",
        help="Card 3: per-out-channel amax/scale. Default tensorwise absmax.",
    )
    parser.add_argument(
        "--bias_correction",
        action="store_true",
        help="Card 1: DualMonitor act_mean bias fold after INT8 pack.",
    )
    parser.add_argument(
        "--bias_correction_top_ratio",
        type=float,
        default=None,
        help=(
            "Fraction of INT8 layers (by DualMonitor sensitivity, high first) "
            "that receive Card 1. Default None = autonomous from THIS "
            "checkpoint DualMonitor / analyze character."
        ),
    )
    args = parser.parse_args()
    args.fp16_budget_mb = _require_fp16_budget_mb_hard(
        float(args.fp16_budget_mb)
    )
    _bc_top_override = args.bias_correction_top_ratio

    zib = _load_zib_v20()
    script_dir = current_dir

    raw_input_arg = args.input
    resolved_input, tried_inputs = zib.resolve_weights_path(raw_input_arg, script_dir)
    if not os.path.isfile(resolved_input):
        print("[FATAL] Input weights file not found.")
        for p in tried_inputs:
            print(f"    - {p}")
        sys.exit(1)
    args.input = resolved_input

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("HSWQ Z-Image INT8 V1.0 (V4 + Card1 + Card3, ZI format)")
    print("=" * 60)

    # --- ComfyUI / tokenizer / TE (same block shape as ZIB v2.0) ---
    comfy_path = args.comfy_path
    if comfy_path is None:
        comfy_path = os.environ.get(
            "COMFYUI_PATH", os.path.join(os.getcwd(), "ComfyUI")
        )
    if os.path.exists(comfy_path) and comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)

    try:
        import comfy.ops
        from comfy.text_encoders import llama as llama_module
        from transformers import Qwen2Tokenizer
        from safetensors.torch import load_file as _load_file

        tokenizer_dir = zib.resolve_tokenizer_offline(
            args.tokenizer_path, args.comfy_path, args.clip_path
        )
        if tokenizer_dir:
            print(f"  Loading tokenizer from disk: {tokenizer_dir}")
            try:
                tokenizer = Qwen2Tokenizer.from_pretrained(
                    tokenizer_dir, local_files_only=True
                )
            except Exception:
                tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_dir)
        else:
            model_id = args.tokenizer_path or "Qwen/Qwen2.5-7B-Instruct"
            print(f"  Trying Repo ID (STRICT LOCAL): {model_id}")
            tokenizer = Qwen2Tokenizer.from_pretrained(
                model_id, local_files_only=True
            )

        print(f"[*] Loading Text Encoder: {args.clip_path}")
        te_sd = _load_file(args.clip_path)
        text_encoder = llama_module.Qwen3_4B(
            config_dict={},
            device=device,
            dtype=torch.float16,
            operations=comfy.ops.disable_weight_init,
        )
        text_encoder.load_state_dict(te_sd, strict=False)
        text_encoder.eval()
    except Exception as e:
        print(f"[FATAL] Failed to load tokenizer/text_encoder: {e}")
        sys.exit(1)

    # --- Profile (analyze_zib_distribution) ---
    analyze_script = os.path.join(script_dir, "analyze", "analyze_zib_distribution.py")
    if not os.path.exists(analyze_script):
        analyze_script = os.path.join(script_dir, "analyze_zib_distribution.py")
    input_abs = os.path.abspath(args.input)
    input_root = os.path.splitext(os.path.basename(args.input))[0]
    profile_path = args.profile
    is_auto = False
    if not profile_path:
        profile_path = os.path.join(script_dir, f"{input_root}_distribution_profile.json")
        is_auto = True
    should_run_analysis = is_auto or not os.path.exists(profile_path)
    if should_run_analysis:
        if os.path.exists(analyze_script):
            print("[*] Executing distribution analysis:")
            print(f"    Script: {analyze_script}")
            subprocess.run(
                [
                    sys.executable,
                    analyze_script,
                    "--input",
                    input_abs,
                    "--output",
                    profile_path,
                ],
                check=True,
            )
        else:
            print(f"[*] Warning: Analysis script NOT found: {analyze_script}")

    model_profile = {}
    if os.path.exists(profile_path):
        print(f"[*] Loading Analysis Data: {profile_path}")
        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)
            model_profile = profile_data.get("layers", profile_data)

    is_zanime_profile_flag = bool(model_profile) and zib._is_zanime_profile(
        model_profile
    )
    if is_zanime_profile_flag:
        n_before = len(model_profile)
        model_profile = zib._convert_zanime_profile_to_nextdit(model_profile)
        print(
            f"  [Z-Anime profile bridge] entries: {n_before} -> {len(model_profile)}"
        )

    alpha, beta, get_layer_search_low, hard_veto_layers = zib.derive_hswq_strategy(
        model_profile,
        is_zanime=is_zanime_profile_flag,
        use_bf16_calibration=is_zanime_profile_flag,
    )
    (
        model,
        original_state_dict,
        stripped_state_dict,
        zit_config,
        detected_prefix,
        is_zanime,
        zanime_reverse_map,
        inference_dtype,
    ) = zib.load_zit_model(args.input, device, args.comfy_path)

    # --- Autonomous VETO (same as ZIB v2.0) ---
    if is_zanime:
        structural_veto = zib._compute_structural_veto(model, hard_veto_layers)
        if structural_veto:
            hard_veto_layers = hard_veto_layers.union(structural_veto)
            print(
                f"  [Z-Anime Structural VETO] +{len(structural_veto)} "
                f"(total {len(hard_veto_layers)})"
            )
        proj_veto = zib._compute_per_projection_qkv_veto(
            model, hard_veto_layers, zib._QKV_PROJ_VETO_THRESH_ZANIME
        )
        if proj_veto:
            hard_veto_layers = hard_veto_layers.union(proj_veto)
            print(
                f"  [Z-Anime Per-Projection VETO] +{len(proj_veto)} "
                f"(total {len(hard_veto_layers)})"
            )
    else:
        print("  [V2.0 Autonomous VETO] Structural + per-projection qkv + key-pattern.")
        structural_veto = zib._compute_structural_veto(model, hard_veto_layers)
        if structural_veto:
            hard_veto_layers = hard_veto_layers.union(structural_veto)
            print(
                f"  [Structural VETO] +{len(structural_veto)} "
                f"(total {len(hard_veto_layers)})"
            )
        proj_veto = zib._compute_per_projection_qkv_veto(
            model, hard_veto_layers, zib._QKV_PROJ_VETO_THRESH_DEFAULT
        )
        if proj_veto:
            hard_veto_layers = hard_veto_layers.union(proj_veto)
            print(
                f"  [Per-Projection VETO] +{len(proj_veto)} "
                f"(total {len(hard_veto_layers)})"
            )
        keypattern_veto = zib._compute_nextdit_keypattern_veto(
            model, hard_veto_layers
        )
        if keypattern_veto:
            hard_veto_layers = hard_veto_layers.union(keypattern_veto)
            print(f"  [Key-Pattern VETO] hard_veto total: {len(hard_veto_layers)}")

    pipeline = zib.ZITCalibrationPipeline(
        model, text_encoder, tokenizer, device, dtype=inference_dtype
    )

    # Patch DualMonitor for Card 1 signed means (do not alter zib file on disk).
    zib.dual_monitors.clear()
    dual_monitors = zib.dual_monitors

    def hook_fn_int8(module, input, output, name):
        if name not in dual_monitors:
            dual_monitors[name] = DualMonitorInt8()
        dual_monitors[name].update(input[0], output)

    print("Preparing calibration (Dual Monitor hooks; Card 1 act means)...")
    handles, target_modules = [], []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handle = module.register_forward_hook(
                lambda m, i, o, n=name: hook_fn_int8(m, i, o, n)
            )
            handles.append(handle)
            target_modules.append(name)

    with open(args.calib_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    if len(prompts) < args.num_calib_samples:
        prompts = (prompts * (args.num_calib_samples // max(len(prompts), 1) + 1))[
            : args.num_calib_samples
        ]
    else:
        prompts = prompts[: args.num_calib_samples]

    print(
        f"Running calibration ({args.num_calib_samples} samples, "
        f"{args.num_inference_steps} steps)..."
    )
    for i, prompt in enumerate(prompts):
        print(f"\nSample {i+1}/{args.num_calib_samples}: {prompt[:50]}...")
        with torch.no_grad():
            pipeline(prompt=prompt, num_inference_steps=args.num_inference_steps)
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    for h in handles:
        h.remove()

    _norm_profile = {}
    for _pk, _pv in model_profile.items():
        if isinstance(_pv, dict):
            _stripped = _pk
            for _pfx in zib.ZIT_PREFIXES:
                if _pfx and _stripped.startswith(_pfx):
                    _stripped = _stripped[len(_pfx):]
                    break
            if _stripped.endswith(".weight"):
                _stripped = _stripped[:-7]
            _norm_profile[_stripped] = _pv

    if not is_zanime:
        _supp = zib._autonomous_supplemental_veto(
            model, hard_veto_layers, _norm_profile
        )
        if _supp:
            hard_veto_layers = hard_veto_layers.union(_supp)
            print(
                f"  [Supplemental VETO] +{len(_supp)} "
                f"(total {len(hard_veto_layers)})"
            )

    # --- Per-checkpoint auto analysis → auto-optimal FP16 inside 700 MiB ---
    # DualMonitor refresh α/β (never keep pre-calib stale mix). No keep_ratio.
    if not _norm_profile:
        raise ValueError(
            "ZI INT8 FP16 budget requires THIS-checkpoint layer profile "
            "(auto analysis → derive_int8_autonomous_tunables). "
            "Run analyze / supply --profile before quantize."
        )
    veto_tunables = resolve_veto_tunables(
        _norm_profile,
        dual_monitors=dual_monitors,
        fp16_budget_mb=float(args.fp16_budget_mb),
    )
    alpha = float(veto_tunables.alpha_auto)
    if alpha <= 0.0:
        raise ValueError(
            "INT8 Full-SVD×RMS alpha_auto must be > 0 after DualMonitor resolve "
            f"(alpha==0 is SVD cut / rebellion). got alpha_auto={alpha}"
        )
    beta = 1.0 - alpha
    print(
        f"  [Dynamic Alpha/Beta INT8 after DualMonitor] "
        f"alpha={alpha!r}, beta={beta!r} "
        f"(THIS analyze character → Full-SVD×RMS; Imp×Sens×V4 MSE fill "
        f"{float(args.fp16_budget_mb):g} MiB)"
    )
    if _bc_top_override is None:
        args.bias_correction_top_ratio = float(
            veto_tunables.bias_correction_top_ratio
        )
        print(
            f"  [Autonomous bias_correction_top_ratio after DualMonitor] "
            f"{args.bias_correction_top_ratio!r}"
        )
    else:
        args.bias_correction_top_ratio = float(_bc_top_override)

    mse_cache: dict = {}
    dynamic_keep_layers, mse_cache = _v4_score_all_fp16_candidates(
        model=model,
        dual_monitors=dual_monitors,
        target_modules=target_modules,
        hard_veto_layers=hard_veto_layers,
        alpha=alpha,
        beta=beta,
        device=device,
        mse_cache=mse_cache,
    )
    # FULL union — budget only truncates (Hard VETO may demote if over frame).
    keep_layers = dynamic_keep_layers.union(hard_veto_layers)
    keep_layers, hard_veto_layers, budget_stats = _apply_fp16_budget_cap(
        model=model,
        keep_layers=keep_layers,
        hard_veto_layers=hard_veto_layers,
        budget_mb=float(args.fp16_budget_mb),
        norm_profile=_norm_profile,
        veto_tunables=veto_tunables,
        dual_monitors=dual_monitors,
        mse_cache=mse_cache,
        alpha=alpha,
        beta=beta,
        device=device,
    )
    # get_layer_search_low unused for INT8 pack (absmax); kept for parity logging
    _ = get_layer_search_low
    _ = mse_cache

    act_mean_dict = {}
    sens_dict = {}
    for name, mon in dual_monitors.items():
        if getattr(mon, "channel_act_mean", None) is not None:
            act_mean_dict[name] = mon.channel_act_mean.detach().float().cpu()
        sens_dict[name] = float(mon.get_sensitivity())
    print(
        f"  [Card 1 DualMonitor] act_mean={len(act_mean_dict)} "
        f"sens={len(sens_dict)}"
    )

    keep_dtype = torch.bfloat16 if is_zanime else torch.float16
    mode = "per-channel (Card 3)" if args.per_channel_int8 else "tensorwise"
    print(f"\nConverting to INT8 ({mode}) | FP16/BF16 keep={len(keep_layers)}")

    # Card 1 scope among INT8 layers
    bc_allowed = None
    if args.bias_correction:
        int8_candidates = [
            n for n in target_modules if n not in keep_layers
        ]
        top_ratio = float(args.bias_correction_top_ratio)
        top_ratio = 0.0 if top_ratio < 0.0 else (1.0 if top_ratio > 1.0 else top_ratio)
        ranked = sorted(
            int8_candidates,
            key=lambda n: sens_dict.get(n, 0.0),
            reverse=True,
        )
        n_bc = int(len(ranked) * top_ratio + 1e-9)
        if top_ratio > 0.0 and n_bc < 1 and ranked:
            n_bc = 1
        if top_ratio >= 1.0:
            bc_allowed = None
            print(
                f"  [Bias Correction] scope=ALL {len(ranked)} INT8 layers "
                f"(top_ratio=1.0)."
            )
        else:
            bc_allowed = set(ranked[:n_bc])
            print(
                f"  [Bias Correction] top {n_bc}/{len(ranked)} by sensitivity "
                f"(top_ratio={top_ratio:.3f})."
            )

    bias_corr_pending: dict[str, torch.Tensor] = {}
    bias_corr_applied = 0
    bias_corr_skipped_no_bias = 0
    bias_corr_skipped_no_act = 0
    bias_corr_skipped_low_sens = 0
    bias_corr_skipped_bad_shape = 0
    converted_count = 0
    kept_count = 0
    output_state_dict = {}
    quant_meta_layers = {}

    for stripped_key, value in tqdm(stripped_state_dict.items(), desc="Converting"):
        module_name = (
            stripped_key[:-7] if stripped_key.endswith(".weight") else None
        )
        is_matmul_weight = (
            module_name is not None
            and value.ndim >= 2
            and value.dtype
            in (torch.float16, torch.float32, torch.bfloat16)
        )

        # Z-Anime keep qkv: split to Diffusers projections (ZI format)
        if (
            is_zanime
            and module_name
            and module_name in keep_layers
            and module_name.endswith(".attention.qkv")
        ):
            base = module_name[: -len(".qkv")]
            chunks = torch.chunk(value.to(keep_dtype), 3, dim=0)
            for tag, chunk in zip(("to_q", "to_k", "to_v"), chunks):
                output_state_dict[
                    f"{detected_prefix}{base}.{tag}.weight"
                ] = chunk.contiguous().clone()
            kept_count += 1
            continue

        if module_name and module_name in keep_layers and is_matmul_weight:
            output_state_dict[detected_prefix + stripped_key] = value.to(keep_dtype)
            kept_count += 1
            continue

        if is_matmul_weight and module_name and module_name not in keep_layers:
            # Z-Anime fused qkv → split INT8 to_q/to_k/to_v
            if is_zanime and module_name.endswith(".attention.qkv"):
                base = module_name[: -len(".qkv")]
                chunks = torch.chunk(value, 3, dim=0)
                for tag, chunk in zip(("to_q", "to_k", "to_v"), chunks):
                    if args.per_channel_int8 and chunk.ndim in (2, 4):
                        q, scale = pack_channelwise(chunk.contiguous())
                    else:
                        q, scale = pack_tensorwise(chunk.contiguous())
                    tgt_module = f"{base}.{tag}"
                    tgt_key = f"{detected_prefix}{tgt_module}.weight"
                    output_state_dict[tgt_key] = q
                    _emit_int8_meta(
                        output_state_dict,
                        f"{detected_prefix}{tgt_module}",
                        scale,
                    )
                    quant_meta_layers[f"{detected_prefix}{tgt_module}"] = {
                        "format": "int8_tensorwise"
                    }
                    converted_count += 1
                    if args.bias_correction:
                        proj_name = f"{base}.{tag}"
                        # Bias for Diffusers projections rarely present on qkv path;
                        # still attempt if act_mean exists under fused name for q only.
                        act = act_mean_dict.get(module_name)
                        if (
                            bc_allowed is not None
                            and module_name not in bc_allowed
                        ):
                            bias_corr_skipped_low_sens += 1
                        elif act is None:
                            bias_corr_skipped_no_act += 1
                        else:
                            w_dq = q.float() * (
                                scale
                                if scale.ndim > 0
                                else scale
                            )
                            # Per-chunk Linear share: scale act to chunk in-dim
                            if act.numel() == chunk.shape[1]:
                                delta = compute_int8_bias_delta(
                                    chunk, w_dq, act
                                )
                                if delta is None:
                                    bias_corr_skipped_bad_shape += 1
                                else:
                                    bias_corr_pending[
                                        f"{detected_prefix}{tgt_module}"
                                    ] = (-delta).detach().float().cpu()
                continue

            # Standard Linear/Conv INT8
            if args.per_channel_int8 and value.ndim in (2, 4):
                q, scale = pack_channelwise(value)
            elif args.per_channel_int8:
                output_state_dict[detected_prefix + stripped_key] = value.to(
                    keep_dtype
                )
                kept_count += 1
                continue
            else:
                q, scale = pack_tensorwise(value)
            weight_dq = q.float() * scale
            out_key = detected_prefix + stripped_key
            prefixed_module = detected_prefix + module_name
            output_state_dict[out_key] = q
            _emit_int8_meta(output_state_dict, prefixed_module, scale)
            quant_meta_layers[prefixed_module] = {"format": "int8_tensorwise"}
            converted_count += 1

            if args.bias_correction:
                if bc_allowed is not None and module_name not in bc_allowed:
                    bias_corr_skipped_low_sens += 1
                else:
                    act = act_mean_dict.get(module_name)
                    if act is None:
                        bias_corr_skipped_no_act += 1
                    else:
                        delta = compute_int8_bias_delta(value, weight_dq, act)
                        if delta is None:
                            bias_corr_skipped_bad_shape += 1
                        else:
                            bias_corr_pending[prefixed_module] = (
                                (-delta).detach().float().cpu()
                            )
            continue

        # Passthrough (norms, embeds, biases, etc.)
        if is_zanime:
            new_value = (
                value.to(torch.bfloat16)
                if value.dtype != torch.bfloat16
                else value
            )
        else:
            new_value = (
                value.to(torch.float16)
                if value.dtype == torch.bfloat16
                else value
            )
        output_state_dict[detected_prefix + stripped_key] = new_value

    if args.bias_correction and bias_corr_pending:
        print(
            f"\n[Bias Correction] Applying deltas to {len(bias_corr_pending)} "
            f"INT8 modules..."
        )
        for module_key, delta in bias_corr_pending.items():
            bias_key = f"{module_key}.bias"
            if bias_key not in output_state_dict:
                bias_corr_skipped_no_bias += 1
                continue
            bias = output_state_dict[bias_key]
            corrected = bias.float() + delta.to(
                device=bias.device, dtype=torch.float32
            )
            output_state_dict[bias_key] = corrected.to(dtype=bias.dtype)
            bias_corr_applied += 1
        print(
            f"  [Bias Correction] applied={bias_corr_applied}, "
            f"no_bias={bias_corr_skipped_no_bias}, "
            f"no_act={bias_corr_skipped_no_act}, "
            f"low_sens={bias_corr_skipped_low_sens}, "
            f"bad_shape={bias_corr_skipped_bad_shape}"
        )
    elif args.bias_correction:
        print(
            f"  [Bias Correction] No deltas pending "
            f"(no_act={bias_corr_skipped_no_act}, "
            f"low_sens={bias_corr_skipped_low_sens}, "
            f"bad_shape={bias_corr_skipped_bad_shape})"
        )

    if is_zanime:
        before_n = len(output_state_dict)
        output_state_dict = zib._denormalize_zanime_output(
            output_state_dict, zanime_reverse_map
        )
        print(
            f"  [Z-Anime] Diffusers key restoration: "
            f"{before_n} -> {len(output_state_dict)} keys."
        )

    metadata = {
        "_quantization_metadata": json.dumps(
            {
                "format_version": "1.0",
                "quant": "int8_tensorwise",
                "engine": "quantize_zi_int8_hswq_v1.0",
                "fp16_budget_mb": float(args.fp16_budget_mb),
                "fp16_budget_used_mb": float(
                    budget_stats.get("used_mb", 0.0)
                    if isinstance(budget_stats, dict)
                    else 0.0
                ),
                "fp16_priority_form": (
                    budget_stats.get("priority_form")
                    if isinstance(budget_stats, dict)
                    else None
                ),
                "n_fp16_keep": int(len(keep_layers)),
                "per_channel_int8": bool(args.per_channel_int8),
                "bias_correction": bool(args.bias_correction),
                "bias_correction_top_ratio": float(
                    args.bias_correction_top_ratio
                    if args.bias_correction_top_ratio is not None
                    else 1.0
                ),
                "layers": quant_meta_layers,
            }
        )
    }

    print(f"Saving: {args.output}")
    used_mb = float(
        budget_stats.get("used_mb", 0.0) if isinstance(budget_stats, dict) else 0.0
    )
    print(
        f"  INT8 layers: {converted_count} | FP16/BF16 keep: {kept_count} | "
        f"FP16 budget {used_mb:.2f}/{float(args.fp16_budget_mb):g} MiB | "
        f"Card3={args.per_channel_int8} | Card1={args.bias_correction} "
        f"(applied={bias_corr_applied})"
    )
    save_file(output_state_dict, args.output, metadata=metadata)
    print("Saved.")


if __name__ == "__main__":
    main()

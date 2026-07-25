"""UNet / DiT FULL ConvRot + NVFP4 converter for native ComfyUI Load Diffusion Model.

FULL ConvRot / FULL offline rotate (default ON):
  - Linear 2D, in_features divisible by power-of-4 group:
      offline Hadamard → NVFP4 pack + stamp
      comfy_quant: {"format":"nvfp4","convrot":true,"convrot_groupsize":G}
  - Conv2d, in_channels divisible by power-of-4 group:
      offline Hadamard → INT8 per-channel + stamp
      (NVFP4 layout is 2D-only; Conv2d uses int8_tensorwise + convrot)
      comfy_quant: {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":G}
  - Linear without group: plain NVFP4 {"format":"nvfp4"}
  - Conv2d without group: plain INT8 {"format":"int8_tensorwise"}

On-disk NVFP4 Linear (TensorCoreNVFP4Layout / QUANT_ALGOS["nvfp4"]):
  .weight          uint8 [N', K'//2]
  .weight_scale    f8e4m3 [N', K'//16]  (block scales)
  .weight_scale_2  f32 scalar           (global scale)
  .input_scale     f32 scalar           (act: amax / (F8_E4M3_MAX * F4_E2M1_MAX))
  .comfy_quant     uint8 JSON

input_scale is written from PTQ calib (--calib_file). For FULL ConvRot layers,
amax is measured on Hadamard-rotated activations (same order as inference:
rotate then quantize). Without calib, no input_scale keys are written and
inference falls back to ones — that destroys quality.

Online act rotate at load is required for ConvRot layers. The loader is built
separately; this converter always does FULL offline weight rotate + stamps.

Use --no-convrot for plain packs only (no offline rotate / no convrot stamp).

Optional Card 1 (--bias_correction): DualMonitor act means; bias += -(W_q - W) @ mu_x
  on quantized Linear/Conv. Shares the same --calib_file pass as input_scale.

HSWQ DualMonitor + FP16 protect, when --calib_file is set (r0 only):
  - Profile JSON + analyze/analyze_sdxl_nvfp4_distribution.py
    (Hard VETO cascade, DualMonitor, infinite branches, budget fill).
  - --keep_ratio must be 0. FP16 set = comprehensive judgment only:
      (1) histogram V4 calib → estimated_mse @ absmax
      (2) DualMonitor sensitivity + Importance
      (3) analyze JSON → VETO + severity / tunables
      (4) Full-SVD×RMS inside V4 pack MSE (never optional / discard)
    arranged together → per-model auto analysis → infinitely branching
    auto-optimal priority, truncated only by --fp16_budget_mb / --budget_mb
    (default 700 MiB; any positive finite). No top-% cut. No fixed recipe.
    Blasphemy = omit any of (1)-(4), or substitute fixed floors.
  - Pack amax: absmax after optional ConvRot (V3.0 parity). V4 real pack MSE
    ranks FP16 keep only (Linear=NVFP4 / Conv2d=INT8 channelwise).
"""
from __future__ import annotations

import argparse
import atexit
import gc
import importlib.util
import json
import math
import os
import subprocess
import sys
import time

import torch
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file, save_file
from tqdm import tqdm

_DEFAULT_GROUPSIZE = 256


class _TeeStream:
    """Mirror every write to console and a session log file (full convert log)."""

    def __init__(self, primary, secondary):
        self._primary = primary
        self._secondary = secondary

    def write(self, data):
        self._primary.write(data)
        self._secondary.write(data)
        return len(data)

    def flush(self):
        self._primary.flush()
        self._secondary.flush()

    def isatty(self):
        return bool(getattr(self._primary, "isatty", lambda: False)())

    def fileno(self):
        return self._primary.fileno()

    @property
    def encoding(self):
        return getattr(self._primary, "encoding", "utf-8")

    def __getattr__(self, name):
        return getattr(self._primary, name)


def _install_torchaudio_stub() -> None:
    """Prevent real torchaudio from loading if comfy.sd is pulled in.

    comfy.sd imports comfy.ldm.lightricks.vae.audio_vae, which does a hard
    ``import torchaudio``. On cloud hosts torch/torchaudio CUDA builds often
    mismatch (e.g. torch 13.2 vs torchaudio 13.0) and abort before calib.
    SDXL NVFP4 convert calib uses Diffusers StableDiffusionXLPipeline only —
    never AudioVAE — so replace torchaudio in sys.modules with a local stub.
    Does not touch ComfyUI-master.
    """
    import types

    for key in list(sys.modules):
        if key == "torchaudio" or key.startswith("torchaudio."):
            del sys.modules[key]

    ta = types.ModuleType("torchaudio")
    ta.__file__ = "<hswq_torchaudio_stub>"
    ta.__path__ = []

    functional = types.ModuleType("torchaudio.functional")

    def _resample(waveform, orig_freq, new_freq, *args, **kwargs):
        return waveform

    functional.resample = _resample

    transforms = types.ModuleType("torchaudio.transforms")

    class _MelSpectrogram:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, x):
            return x

        def to(self, *args, **kwargs):
            return self

    class _MelScale:
        def __init__(self, *args, **kwargs):
            pass

    transforms.MelSpectrogram = _MelSpectrogram
    transforms.MelScale = _MelScale

    ta.functional = functional
    ta.transforms = transforms
    sys.modules["torchaudio"] = ta
    sys.modules["torchaudio.functional"] = functional
    sys.modules["torchaudio.transforms"] = transforms


def _install_nvfp4_convert_full_session_log() -> str:
    """Tee stdout+stderr into log/hswq_nvfp4_convert_full_<stamp>.txt (full log).

    Override path with HSWQ_CONVERT_FULL_LOG_PATH. Same spirit as analyze's
    emit_hswq_nvfp4_full_visibility_log file under log/.
    """
    env_p = (os.environ.get("HSWQ_CONVERT_FULL_LOG_PATH") or "").strip()
    if env_p:
        path = os.path.abspath(env_p)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    else:
        repo = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(repo, "log")
        os.makedirs(log_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"hswq_nvfp4_convert_full_{stamp}.txt")
    fh = open(path, "w", encoding="utf-8", newline="\n", buffering=1)
    tee_out = _TeeStream(sys.__stdout__, fh)
    tee_err = _TeeStream(sys.__stderr__, fh)
    sys.stdout = tee_out
    sys.stderr = tee_err

    def _close_log() -> None:
        try:
            if sys.stdout is tee_out:
                sys.stdout = sys.__stdout__
            if sys.stderr is tee_err:
                sys.stderr = sys.__stderr__
            fh.flush()
            fh.close()
        except Exception:
            pass

    atexit.register(_close_log)
    print(f"[HSWQ CONVERT FULL LOG FILE] {path}")
    return path


def _ensure_nvfp4_hist_on_path() -> None:
    hist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "histogram")
    if hist_dir not in sys.path:
        sys.path.insert(0, hist_dir)

def _prepare_weight_for_pack_score(
    *,
    weight: torch.Tensor,
    enable_convrot: bool,
    group_size: int,
    build_hadamard,
    convrot_group_size_for_features,
    rotate_weight,
    rotate_weight_conv2d,
    hadamard_cache: dict[int, torch.Tensor],
) -> torch.Tensor:
    """Match pack/amax path: optional offline Hadamard on eligible Linear/Conv2d."""
    w = weight.detach().float()
    if w.ndim not in (2, 4):
        return w
    in_f = int(w.shape[1])
    used_gs = None
    if (
        enable_convrot
        and convrot_group_size_for_features is not None
        and build_hadamard is not None
    ):
        used_gs = convrot_group_size_for_features(in_f, group_size)
    do_rotate = (
        enable_convrot
        and used_gs is not None
        and build_hadamard is not None
        and (
            (w.ndim == 2 and rotate_weight is not None)
            or (w.ndim == 4 and rotate_weight_conv2d is not None)
        )
    )
    if not do_rotate:
        return w
    h = hadamard_cache.get(int(used_gs))
    if h is None:
        h = build_hadamard(int(used_gs), device="cpu", dtype=torch.float32)
        hadamard_cache[int(used_gs)] = h
    if w.ndim == 2:
        return rotate_weight(w.cpu(), h, int(used_gs))
    return rotate_weight_conv2d(w.cpu(), h, int(used_gs))

def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))

def _resolve_distribution_profile_path(
    input_path: str,
    profile_arg: str | None,
) -> tuple[str, bool]:
    """Default `{stem}_distribution_profile.json` under repo root; or --profile."""
    input_root = os.path.splitext(os.path.basename(input_path))[0]
    if profile_arg:
        return os.path.abspath(profile_arg), False
    return (
        os.path.join(_script_dir(), f"{input_root}_distribution_profile.json"),
        True,
    )

def _ensure_distribution_profile(
    *,
    input_path: str,
    profile_arg: str | None,
) -> str:
    """Locate / regenerate / verify NVFP4 profile JSON.

    Script: analyze/analyze_sdxl_nvfp4_distribution.py.
    No-skip when is_auto (always re-run auto path).
    Explicit --profile: run analyze only when the file is missing.
    """
    analyze_script = os.path.join(
        _script_dir(), "analyze", "analyze_sdxl_nvfp4_distribution.py"
    )
    if not os.path.exists(analyze_script):
        raise FileNotFoundError(
            f"[FATAL] NVFP4 profile script not found: {analyze_script}"
        )

    input_abs = os.path.abspath(input_path)
    profile_path, is_auto = _resolve_distribution_profile_path(input_abs, profile_arg)
    should_run_analysis = is_auto or not os.path.exists(profile_path)

    if should_run_analysis:
        print("[*] Executing mandated distribution analysis (No skip policy):")
        print(f"    Script: {analyze_script}")
        print(f"    Input:  {input_abs}")
        print(f"    Result: {profile_path}")
        # Capture + re-print so Tee session log includes analyze stdout/stderr
        # (child inherit of console FD bypasses Python sys.stdout Tee).
        _an = subprocess.run(
            [
                sys.executable,
                analyze_script,
                "--input",
                input_abs,
                "--output",
                profile_path,
            ],
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )
        if _an.stdout:
            sys.stdout.write(_an.stdout)
            if not _an.stdout.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()
        if _an.stderr:
            sys.stderr.write(_an.stderr)
            if not _an.stderr.endswith("\n"):
                sys.stderr.write("\n")
            sys.stderr.flush()
        if _an.returncode != 0:
            raise subprocess.CalledProcessError(
                _an.returncode,
                _an.args,
                output=_an.stdout,
                stderr=_an.stderr,
            )

    if not os.path.exists(profile_path):
        raise FileNotFoundError(
            f"[FATAL] Distribution profile missing after analyze: {profile_path}"
        )
    print(f"[*] Loading Analysis Data: {profile_path}")
    return profile_path

def _load_distribution_profile_layers(
    profile_path: str,
) -> tuple[dict, dict, dict]:
    """Load layers + summary + veto_tunables_nvfp4 from NVFP4 analyze JSON.

    Auto convert always re-runs analyze/analyze_sdxl_nvfp4_distribution.py,
    which writes veto_tunables_nvfp4. No legacy INT8 key path.
    """
    with open(profile_path, "r", encoding="utf-8") as f:
        profile_data = json.load(f)
    if not isinstance(profile_data, dict):
        raise ValueError(f"Profile JSON must be a dict: {profile_path}")
    profile_summary = profile_data.get("summary", {}) or {}
    model_profile = profile_data.get("layers", profile_data)
    if not isinstance(model_profile, dict):
        raise ValueError(f"Profile layers must be a dict: {profile_path}")
    veto_blob = profile_data.get("veto_tunables_nvfp4")
    if not isinstance(veto_blob, dict) or not veto_blob:
        raise ValueError(
            f"Profile JSON missing veto_tunables_nvfp4: {profile_path}"
        )
    return model_profile, profile_summary, veto_blob


# === Ported FP16 protect mass from quantize_sdxl_hswq_v3.0.py (NVFP4 adapt) ===
import numpy as np
from dataclasses import dataclass
import torch.nn as nn

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_hist = os.path.join(_SCRIPT_DIR, "histogram")
if _hist not in sys.path:
    sys.path.insert(0, _hist)
_an = os.path.join(_SCRIPT_DIR, "analyze")
if _an not in sys.path:
    sys.path.insert(0, _an)

from weighted_histogram_mse_v4_nvfp4 import (
    HSWQWeightedHistogramOptimizerV4,
)

# Default MiB budget (packed-baseline overhead: NVFP4 Linear / INT8 Conv2d).
# At CLI entry, FP16_BUDGET_MB_HARD is reassigned from --fp16_budget_mb /
# --budget_mb so every call-site that reads FP16_BUDGET_MB_HARD gets the option.
FP16_BUDGET_MB_DEFAULT = 700.0
FP16_BUDGET_MB_HARD = FP16_BUDGET_MB_DEFAULT
# Post-pack assert slack: owner fill-band (~10 MiB). Not a shield for pack
# leaks or wrong meters (1D norms / silent Linear-Conv float).
FP16_BUDGET_ASSERT_TOLERANCE_MIB = 10.0


def _fp16_budget_bytes(budget_mb: float) -> int:
    return int(float(budget_mb) * 1024 * 1024)


def _require_fp16_budget_mb_hard(budget_mb: float) -> float:
    """Validate packed-baseline overhead budget (MiB); any positive finite OK."""
    b = float(budget_mb)
    if (not math.isfinite(b)) or b <= 0.0:
        raise ValueError(
            f"fp16_budget_mb must be a positive finite MiB budget "
            f"(overhead vs packed NVFP4 Linear / INT8 Conv2d baseline). "
            f"Got {budget_mb!r}."
        )
    return b


def _absorb_nvfp4_budget_ceiling(budget_mb: float) -> float:
    """Bind this run's budget into analyze NVFP4_FP16_BUDGET_MB_HARD."""
    b = _require_fp16_budget_mb_hard(budget_mb)
    analyze_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze")
    if analyze_dir not in sys.path:
        sys.path.insert(0, analyze_dir)
    import analyze_sdxl_nvfp4_distribution as _nvfp4_analyze

    _nvfp4_analyze.NVFP4_FP16_BUDGET_MB_HARD = float(b)
    return b


def _bind_fp16_budget_from_option(budget_mb: float) -> float:
    """CLI option → module FP16_BUDGET_MB_HARD + analyze absorb (single source)."""
    global FP16_BUDGET_MB_HARD
    b = _absorb_nvfp4_budget_ceiling(budget_mb)
    FP16_BUDGET_MB_HARD = float(b)
    return float(b)


def calculate_kurtosis(tensor):
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    if std == 0: return 0.0
    return torch.mean(((tensor - mean) / std) ** 4).item()

# --- SDXL NVFP4 autonomous engine tunables (V3.0 protect shape) ---
# Architectural boundary Conv2d keys (not Linear). Resolution resample is the
# same class of unet boundary as conv_in/conv_out; Linear-only walk previously
# made documented .conv_in/.conv_out key-pattern dead code (手抜き).
_SDXL_KP_BOUNDARY_SUFFIXES = (
    ".conv_in",
    ".conv_out",
    ".upsamplers.0.conv",
    ".downsamplers.0.conv",
)
_SDXL_KP_PREFIXES = ("time_embedding.", "add_embedding.")
_SDXL_ATTN_PROJ_SUFFIXES = (".to_q", ".to_k", ".to_v")
_SDXL_ATTN_TOOUT_SUFFIX = ".to_out.0"
_SDXL_PROFILE_PREFIXES = ("model.", "model.diffusion_model.")

# DualMonitor Sensitivity → FP16 candidates; analyze → VETO candidates.
# V4 calib MSE embeds Full-SVD×RMS (+ DualMonitor Importance when present).
# All four surfaces enter ONE per-model ranking in _apply_fp16_budget_cap.
# Budget winners = final FP16 protection. Analyze VETO is not renamed.
# keep_ratio is r0; DualMonitor must not invent or gate that flag.
# Fixed combinator weights / model-name recipes = blasphemy.


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
    quant_format: str = "nvfp4"
    attn_mad_pct_floor: float = 0.0
    attn_mad_q3: float = 0.0
    attn_mad_p99: float = 0.0
    attn_mad_gap_o_max: float = 0.0
    attn_mad_from_profile: float = 0.0
    # Continuous MAD branch fingerprint (THIS pool IQR death → soft→Tukey; P99 tip-only).
    attn_mad_collapse: float = 0.0
    attn_mad_iqr: float = 0.0
    # Autonomous (from derive_nvfp4_autonomous_tunables):
    sens_veto_percentile: float = 100.0
    sens_veto_keep_ratio_gate: float = 0.0
    bias_correction_top_ratio: float = 1.0
    auto_keep_ratio: float = 0.0
    fp16_budget_mb: float = FP16_BUDGET_MB_DEFAULT
    fp16_budget_bytes: int = int(FP16_BUDGET_MB_DEFAULT * 1024 * 1024)
    n_unet_layers: int = 0
    autonomous: bool = False
    # V4 Full-SVD×RMS mix weight from THIS multi-axis analyze character
    # (kurtosis∪outlier∪magnitude). Must be > 0 for non-degenerate THIS  - 
    # alpha_auto==0 is SVD cut (rebellion), not a valid default outcome.
    alpha_auto: float = 0.0

    # Required from derive_nvfp4_autonomous_tunables  -  no silent default holes
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
                f"{missing}. Run derive_nvfp4_autonomous_tunables  -  do not "
                "fill deleted clip holes with dataclass defaults."
            )
        if not bool(d["autonomous"]):
            raise ValueError(
                "SdxlVetoTunables.from_dict requires autonomous=True "
                "(THIS-profile auto analysis → auto-optimal)"
            )
        if str(d["quant_format"]) != "nvfp4":
            raise ValueError(
                "NVFP4 SdxlVetoTunables requires quant_format=nvfp4 "
                f"(got {d['quant_format']!r})"
            )
        b_budget = float(d["fp16_budget_mb"])
        if (not math.isfinite(b_budget)) or b_budget <= 0.0:
            raise ValueError(
                "fp16_budget_mb must be a positive finite MiB budget "
                f"(overhead vs packed NVFP4/INT8); got {d['fp16_budget_mb']!r}"
            )
        if float(d["search_low_floor"]) != 1.0:
            raise ValueError("NVFP4 search_low_floor must be 1.0 (absmax auto-optimal)")
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
            fp16_budget_bytes=int(
                d.get(
                    "fp16_budget_bytes",
                    _fp16_budget_bytes(float(d["fp16_budget_mb"])),
                )
            ),
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
    """Load NVFP4 veto_tunables via fully autonomous derivation.

    All knobs (Hard VETO fences, percentile promotions, dynamic ranking
    weights, MSE release gates, bias_correction scope, sens_veto percentile,
    alpha/beta, search_low) come from derive_nvfp4_autonomous_tunables,
    which uses THIS checkpoint's profile + DualMonitor sensitivity
    distribution. fp16_budget_mb is the run budget (MiB overhead vs packed
    NVFP4 Linear / INT8 Conv2d; default 700, CLI-overridable) —
    auto settings fill that frame; they do not redefine or exceed it.
    No hardcoded 90.0 / 15.0 / 2.0 / 0.5 / 40.0 recipe constants.
    """
    fp16_budget_mb = _bind_fp16_budget_from_option(fp16_budget_mb)
    from analyze_sdxl_nvfp4_distribution import (
        derive_nvfp4_autonomous_tunables,
        emit_hswq_nvfp4_full_visibility_log,
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
        derived = derive_nvfp4_autonomous_tunables(
            norm_profile,
            dualmonitor_sensitivities=sens_map if sens_map else None,
            fp16_budget_mb=fp16_budget_mb,
        )
        # derive_nvfp4_autonomous_tunables already emitted the FULL pool / calc /
        # every-layer / every-knob dump. Emit the final resolved dict again so
        # DualMonitor re-resolve is also byte-complete in the same log.
        emit_hswq_nvfp4_full_visibility_log(
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
        "derive_nvfp4_autonomous_tunables (auto analysis → auto-optimal). "
        "Refuse stale veto_tunables-only load after accommodation-clip purge."
    )


def _layer_weight_stats(tensor: torch.Tensor) -> tuple[float, float, float]:
    """Live kurtosis, outlier_ratio, abs_max for a weight tensor."""
    x = tensor.float()
    std = torch.std(x).item()
    amax = max(abs(x.min().item()), abs(x.max().item()))
    k = calculate_kurtosis(x)
    o = amax / std if std > 0 else 0.0
    return k, o, amax


def _mad_outlier_pct(tensor: torch.Tensor, zthr: float = 3.0) -> float:
    """Robust outlier fraction (%). Used by NVFP4 VETO paths."""
    xf = tensor.detach().float().reshape(-1)
    if xf.numel() < 4:
        return 0.0
    med = xf.median()
    mad = (xf - med).abs().median().clamp_min(1e-12)
    z = (xf - med).abs() / (1.4826 * mad)
    return float((z > zthr).float().mean().item() * 100.0)


def _profile_score_from_entry(
    prof: dict,
    drift: float = 0.0,
    tunables: SdxlVetoTunables | None = None,
) -> float:
    """Dynamic ranking score from distribution profile (+ optional post-calib drift)."""
    if not prof:
        return 0.0
    base = prof.get("profile_score")
    if base is None:
        k = float(prof.get("kurtosis", 0) or 0)
        o = float(prof.get("outlier_ratio", 0) or 0)
        m = float(prof.get("abs_max", 0) or 0)
        if tunables is not None:
            base = k + o * tunables.score_o_weight + m * tunables.score_m_weight
        else:
            base = k + o + m
    else:
        base = float(base)
    mult = tunables.drift_score_mult if tunables is not None else 1.0
    return base + drift * mult


def _profile_layer_stats(prof: dict, weight_tensor: torch.Tensor) -> tuple[float, float, float]:
    """Prefer precomputed profile stats; fall back to live weight scan."""
    if prof and "kurtosis" in prof and "outlier_ratio" in prof and "abs_max" in prof:
        return (
            float(prof.get("kurtosis", 0) or 0),
            float(prof.get("outlier_ratio", 0) or 0),
            float(prof.get("abs_max", 0) or 0),
        )
    return _layer_weight_stats(weight_tensor)


def _discover_ff2_suffixes(
    norm_profile: dict | None,
    min_count: int = 1,
) -> tuple[str, ...]:
    """Discover FFN output Linear suffixes from this checkpoint profile (no layer names)."""
    if not norm_profile:
        return ()
    analyze_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze")
    if analyze_dir not in sys.path:
        sys.path.insert(0, analyze_dir)
    from analyze_sdxl_nvfp4_distribution import _classify_layer_key

    counts: dict[str, int] = {}
    for key in norm_profile:
        ck = key if key.endswith(".weight") else f"{key}.weight"
        if _classify_layer_key(ck) != "ff2":
            continue
        base = key[:-7] if key.endswith(".weight") else key
        idx = base.rfind(".ff.")
        if idx < 0:
            continue
        suf = base[idx:]
        counts[suf] = counts.get(suf, 0) + 1
    if not counts:
        return ()
    best_count = max(counts.values())
    return tuple(
        sorted(s for s, c in counts.items() if c == best_count and c >= min_count)
    )


def _ff2_selective_veto_hit(
    prof: dict | None,
    live_o: float,
    tunables: SdxlVetoTunables,
) -> tuple[bool, str]:
    """Selective ff.net.2 VETO: class-relative profile_score and outlier (not blanket)."""
    # Cuts = derive_veto_tunables_nvfp4 only (no hardcoded floors).
    score_cut = tunables.ff2_profile_score_cutoff
    outlier_cut = tunables.ff2_profile_outlier
    live_cut = tunables.ff2_outlier_live
    if prof:
        score = _profile_score_from_entry(prof, tunables=tunables)
        o = float(prof.get("outlier_ratio", 0) or 0)
        if score >= score_cut:
            return True, f"profile_score={score:.2f}>={score_cut}"
        if o >= outlier_cut:
            return True, f"profile_o={o:.1f}>={outlier_cut}"
        return False, ""
    if live_o > live_cut:
        return True, f"live_o={live_o:.1f}>{live_cut}"
    return False, ""


def _weight_profile_drift(weight_tensor: torch.Tensor, prof: dict) -> float:
    """Relative drift between live weights and distribution profile."""
    if not prof:
        return 0.0
    lk, lo, lm = _layer_weight_stats(weight_tensor)
    pk = float(prof.get("kurtosis", 0) or 0)
    po = float(prof.get("outlier_ratio", 0) or 0)
    pm = float(prof.get("abs_max", 0) or 0)
    dk = abs(lk - pk) / max(pk, 1.0)
    do = abs(lo - po) / max(po, 1.0)
    dm = abs(lm - pm) / max(pm, 1e-6)
    return max(dk, do, dm)


def _compute_sdxl_keypattern_veto(
    model: nn.Module,
    hard_veto_layers: set,
    tunables: SdxlVetoTunables,
    norm_profile: dict | None = None,
) -> set:
    """SDXL key-pattern VETO: embeddings, boundary Conv2d, profile-tuned ff2.

    Boundary suffixes apply to Conv2d (and any module whose name ends with the
    suffix). Linear-only iteration previously never reached .conv_in/.conv_out
    or resolution resample  -  that dead path is forbidden hand-waving.
    """
    added = set()
    ff2_suffixes = _discover_ff2_suffixes(norm_profile)
    for _n, _m in model.named_modules():
        if _n in hard_veto_layers:
            continue
        if isinstance(_m, torch.nn.Conv2d) and _n.endswith(_SDXL_KP_BOUNDARY_SUFFIXES):
            added.add(_n)
            print(f"    [Key-Pattern VETO] {_n} (boundary Conv2d)")
            continue
        if not isinstance(_m, torch.nn.Linear):
            continue
        if any(_n.startswith(p) for p in _SDXL_KP_PREFIXES):
            added.add(_n)
            print(f"    [Key-Pattern VETO] {_n} (embedding)")
            continue
        if ff2_suffixes and any(_n.endswith(s) for s in ff2_suffixes):
            # NVFP4: skip full-class auto (inflates file size on SDXL);
            # selective VETO below handles individual outlier ff2 layers.
            prof = (norm_profile or {}).get(_n, {})
            _k, _o, _mstat = _profile_layer_stats(prof, _m.weight.detach())
            hit, reason = _ff2_selective_veto_hit(prof if prof else None, _o, tunables)
            if hit:
                added.add(_n)
                print(f"    [Key-Pattern VETO] {_n} (ff2 auto {reason})")
    if added:
        print(f"  [Key-Pattern VETO] Added {len(added)} layers.")
    return added


def _compute_structural_veto(
    model: nn.Module,
    hard_veto_layers: set,
    norm_profile: dict | None = None,
) -> set:
    """Linear layers whose weight shape is unique within the model (boundary detection)."""
    if norm_profile and any(
        isinstance(v, dict) and "shape_uniqueness" in v for v in norm_profile.values()
    ):
        model_linears = {
            n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
        }
        structural_veto = set()
        for name, entry in norm_profile.items():
            if not isinstance(entry, dict):
                continue
            if name not in model_linears:
                continue
            if entry.get("shape_uniqueness") == 1 and name not in hard_veto_layers:
                structural_veto.add(name)
                shp = entry.get("shape", [])
                print(f"    [Structural VETO] {name} shape={shp} (profile uniqueness=1)")
        return structural_veto

    shape_count: dict[tuple, int] = {}
    for _n, _m in model.named_modules():
        if isinstance(_m, torch.nn.Linear):
            _shp = tuple(_m.weight.shape)
            shape_count[_shp] = shape_count.get(_shp, 0) + 1
    structural_veto = set()
    for _n, _m in model.named_modules():
        if isinstance(_m, torch.nn.Linear):
            _shp = tuple(_m.weight.shape)
            if shape_count[_shp] == 1 and _n not in hard_veto_layers:
                structural_veto.add(_n)
                print(f"    [Structural VETO] {_n} shape={list(_shp)} (live uniqueness=1)")
    return structural_veto


def _compute_sdxl_per_projection_attn_veto(
    model: nn.Module,
    hard_veto_layers: set,
    tunables: SdxlVetoTunables,
    norm_profile: dict | None = None,
) -> set:
    """VETO attn projections when profile (or live) abs_max / outlier_ratio exceeds thresholds.

    NVFP4: thresholds come only from derive_veto_tunables_nvfp4
    (analyze_sdxl_nvfp4_distribution). No additional hardcoded floors.
    """
    proj_veto = set()
    # Thresholds = derive_veto_tunables_nvfp4 only (no hardcoded floors).
    for _n, _m in model.named_modules():
        if not isinstance(_m, torch.nn.Linear):
            continue
        if _n in hard_veto_layers:
            continue
        if ".attn1" not in _n and ".attn2" not in _n:
            continue
        is_qkv = any(_n.endswith(s) for s in _SDXL_ATTN_PROJ_SUFFIXES)
        is_toout = _n.endswith(_SDXL_ATTN_TOOUT_SUFFIX)
        if not is_qkv and not is_toout:
            continue
        prof = (norm_profile or {}).get(_n, {})
        _k, _o, _amax = _profile_layer_stats(prof, _m.weight.detach())
        src = "profile" if prof else "live"
        if is_toout:
            hit = _amax >= tunables.attn_toout_absmax or _o >= tunables.attn_toout_outlier
            thresh_msg = (
                f"to_out amax>={tunables.attn_toout_absmax:.3f}, o>={tunables.attn_toout_outlier:.3f}"
            )
        else:
            hit = _amax >= tunables.attn_qkv_absmax or _o >= tunables.attn_qkv_outlier
            thresh_msg = (
                f"q/k/v amax>={tunables.attn_qkv_absmax:.3f}, o>={tunables.attn_qkv_outlier:.3f}"
            )
        if hit:
            proj_veto.add(_n)
            print(
                f"    [Per-Projection VETO] {_n} "
                f"({src} amax={_amax:.2f}, outlier={_o:.1f}; {thresh_msg})"
            )
    return proj_veto


def _mad_continuous_gates_from_live(
    live_mads: list[float],
) -> tuple[float, float, float, float]:
    """Mirror analyze MAD fences on a live THIS-UNet list.

    Returns (floor, soft, collapse, iqr). Same as analyze
    ``_mad_continuous_fences_from_positives``: hard=THIS MAD P75/Q3;
    soft=collapse-shaped Soft band on THIS below-floor MAD mass
    (not raw P50 flood; not (1-c)*P50+c*Q3 Soft death). P99 tip only.
    """
    live_sorted = sorted(float(v) for v in live_mads if float(v) > 0.0)
    n_live = len(live_sorted)
    if n_live < 1:
        return 0.0, 0.0, 0.0, 0.0
    if n_live < 4:
        peak = float(live_sorted[-1])
        body = float(live_sorted[n_live // 2])
        soft = float(min(body, peak))
        return peak, soft, 1.0, 0.0
    q1 = float(live_sorted[n_live // 4])
    q3 = float(live_sorted[(3 * n_live) // 4])
    iqr = float(max(q3 - q1, 0.0))
    p75 = float(
        live_sorted[max(0, min(n_live - 1, int(round(0.75 * (n_live - 1)))))]
    )
    p50 = float(live_sorted[n_live // 2])
    p99 = float(
        live_sorted[max(0, min(n_live - 1, int(round(0.99 * (n_live - 1)))))]
    )
    tail_span = float(max(p99 - p50, 1e-12))
    collapse = float(1.0 - min(1.0, iqr / (iqr + tail_span)))
    mad_floor = float(p75)
    below = [float(v) for v in live_sorted if float(v) < mad_floor]
    if below:
        tip_idx = max(
            0, min(len(below) - 1, int(round(collapse * (len(below) - 1))))
        )
        soft_tip = float(below[tip_idx])
        mad_soft = float((1.0 - collapse) * p50 + collapse * soft_tip)
    else:
        soft_tip = float(p50)
        mad_soft = float(p50)
    # Mirror analyze §3-1 / 8357425: Soft narrow band (not tip_headroom Soft死).
    soft_span = float(max(mad_floor - p50, 0.0))
    band_w = float(
        max(
            soft_span * float(max(1.0 - collapse, 0.15)),
            iqr * 0.1,
            mad_floor * 1e-6,
            1e-12,
        )
    )
    mad_soft = float(min(mad_soft, mad_floor - band_w))
    if mad_soft >= mad_floor:
        mad_soft = float(q1) if float(q1) < mad_floor else float(p50)
    if mad_soft >= mad_floor and below:
        mad_soft = float(below[-1])
    if mad_soft >= mad_floor:
        mad_soft = float(mad_floor) - float(max(mad_floor, 1.0) * 1e-12)
    return mad_floor, mad_soft, collapse, iqr


def _compute_sdxl_nvfp4_mad_attn_veto(
    model: nn.Module,
    hard_veto_layers: set,
    tunables: SdxlVetoTunables | None = None,
    norm_profile: dict | None = None,
) -> set:
    """NVFP4-path key-pattern + MAD% VETO for attn projections.

    Floors / soft-gap from analyze continuous THIS-pool body fences
    (hard=THIS MAD Q3/P75; soft=below-floor collapse Soft band; P99 tip-only).
    If analyze left the MAD axis at 0.0, bootstrap the same fences from
    THIS UNet's live MAD pool (no fixed model literals, no tip-as-floor).
    """
    mad_floor = (
        float(tunables.attn_mad_pct_floor)
        if tunables is not None
        else 0.0
    )
    # attn_mad_q3 field stores Soft-MAD soft edge (analyze write), not Q3.
    mad_soft = float(tunables.attn_mad_q3) if tunables is not None else 0.0
    gap_o_max = (
        float(tunables.attn_mad_gap_o_max)
        if tunables is not None
        else 0.0
    )
    collapse = float(tunables.attn_mad_collapse) if tunables is not None else 0.0
    mad_iqr = float(tunables.attn_mad_iqr) if tunables is not None else 0.0
    if tunables is not None and gap_o_max <= 0.0:
        gap_o_max = float(max(tunables.extreme_outlier, 1e-9))
    prof = norm_profile or {}

    candidates: list[tuple[str, float, float]] = []
    live_mads: list[float] = []
    for _n, _m in model.named_modules():
        if not isinstance(_m, torch.nn.Linear):
            continue
        if _n in hard_veto_layers:
            continue
        if ".attn1" not in _n and ".attn2" not in _n:
            continue
        is_qkv = any(_n.endswith(s) for s in _SDXL_ATTN_PROJ_SUFFIXES)
        is_toout = _n.endswith(_SDXL_ATTN_TOOUT_SUFFIX)
        if not is_qkv and not is_toout:
            continue
        entry = prof.get(_n, {}) if isinstance(prof.get(_n, {}), dict) else {}
        mad_pct = float(entry.get("mad_outlier_pct", entry.get("mad_pct", 0)) or 0)
        if mad_pct <= 0.0:
            mad_pct = _mad_outlier_pct(_m.weight)
        o = float(entry.get("outlier_ratio", 0) or 0)
        if o <= 0.0 and hasattr(_m, "weight"):
            _, o, _ = _layer_weight_stats(_m.weight.data)
        candidates.append((_n, mad_pct, o))
        if mad_pct > 0.0:
            live_mads.append(mad_pct)

    if mad_floor <= 0.0 and live_mads:
        mad_floor, mad_soft, collapse, mad_iqr = _mad_continuous_gates_from_live(
            live_mads
        )
        if tunables is not None:
            tunables.attn_mad_pct_floor = float(mad_floor)
            tunables.attn_mad_q3 = float(mad_soft)
            tunables.attn_mad_collapse = float(collapse)
            tunables.attn_mad_iqr = float(mad_iqr)
            tunables.attn_mad_from_profile = 0.0
        if gap_o_max <= 0.0 and tunables is not None:
            gap_o_max = float(max(tunables.extreme_outlier, 1e-9))
        print(
            f"  [NVFP4 MAD VETO] Continuous THIS-UNet MAD body fences from "
            f"{len(live_mads)} live samples "
            f"(floor={mad_floor:.2f}, soft={mad_soft:.2f}, "
            f"collapse={collapse:.3f}, iqr={mad_iqr:.3f}; P99 tip-only)"
        )

    if mad_floor <= 0.0:
        return set()

    added = set()
    for _n, mad_pct, o in candidates:
        hard = mad_pct >= mad_floor
        soft = (
            mad_soft > 0.0
            and mad_pct >= mad_soft
            and mad_pct < mad_floor
            and o < gap_o_max
        )
        if hard or soft:
            added.add(_n)
            kind = "hard" if hard else "soft"
            o_note = "o_miss" if o < gap_o_max else "o_hit"
            print(
                f"    [NVFP4 MAD VETO] {_n} "
                f"(MAD%={mad_pct:.2f}, o={o:.2f}, floor={mad_floor:.2f}, "
                f"soft={mad_soft:.2f}, collapse={collapse:.3f}, "
                f"iqr={mad_iqr:.3f}, gate_o={gap_o_max:.2f}; "
                f"{kind}/{o_note})"
            )
    if added:
        print(
            f"  [NVFP4 MAD VETO] Added {len(added)} attn layers "
            f"(floor={mad_floor:.2f}, soft={mad_soft:.2f}, "
            f"collapse={collapse:.3f}, iqr={mad_iqr:.3f})."
        )
    return added


def _autonomous_supplemental_veto(
    model: nn.Module,
    hard_veto_layers: set,
    norm_profile: dict,
    tunables: SdxlVetoTunables,
) -> set:
    """Profile-primary VETO: outlier ff.net.2, high-drift embedding layers."""
    added = set()
    for _n, _m in model.named_modules():
        if not isinstance(_m, torch.nn.Linear):
            continue
        if _n in hard_veto_layers:
            continue
        prof = norm_profile.get(_n, {})
        drift = _weight_profile_drift(_m.weight.data, prof)
        _k, _o, _mstat = _profile_layer_stats(prof, _m.weight.detach())
        ff2_suffixes = _discover_ff2_suffixes(
            norm_profile, min_count=tunables.ff2_suffix_min_count
        )
        if ff2_suffixes and any(_n.endswith(s) for s in ff2_suffixes):
            # NVFP4: selective only (no full-class auto)
            hit, reason = _ff2_selective_veto_hit(prof if prof else None, _o, tunables)
            if hit:
                added.add(_n)
                print(f"    [Supplemental VETO] {_n} (ff.net.2 {reason})")
        elif any(_n.startswith(p) for p in _SDXL_KP_PREFIXES) and drift > tunables.drift_veto_thresh:
            added.add(_n)
            print(
                f"    [Supplemental VETO] {_n} "
                f"(embedding drift={drift:.3f} > {tunables.drift_veto_thresh:.3f})"
            )
    return added


def _collect_mse_release_candidates(
    hard_veto_layers: set,
    structural_veto: set,
    norm_profile: dict,
    model: nn.Module,
    tunables: SdxlVetoTunables,
) -> set:
    """Outlier-only profile VETO with low drift and non-structural  -  MSE release candidates.

    NVFP4: mse_release_* come only from derive_veto_tunables_nvfp4
    (analyze_sdxl_nvfp4_distribution). No hardcoded min/max floors.
    """
    candidates = set()
    _module_dict = dict(model.named_modules())
    for vname in hard_veto_layers:
        if vname in structural_veto:
            continue
        prof = norm_profile.get(vname, {})
        k = float(prof.get("kurtosis", 0) or 0)
        m = float(prof.get("abs_max", 0) or 0)
        o = float(prof.get("outlier_ratio", 0) or 0)
        if (
            o > tunables.mse_release_o_min
            and k <= tunables.mse_release_k_max
            and m <= tunables.mse_release_m_max
        ):
            vmod = _module_dict.get(vname)
            if vmod is not None and hasattr(vmod, "weight"):
                drift = _weight_profile_drift(vmod.weight.data, prof)
                if drift < tunables.drift_veto_thresh:
                    candidates.add(vname)
    return candidates


def _dualmonitor_channel_importance(dual_monitors: dict, module_name: str):
    """1D input-channel importance from DualMonitor (32-sample calib contract)."""
    mon = dual_monitors.get(module_name) if dual_monitors else None
    if mon is None:
        return None
    imp = getattr(mon, "channel_importance", None)
    if imp is None:
        return None
    return imp.detach().float()


def _fp16_extra_bytes_vs_packed(weight: torch.Tensor) -> int:
    """Extra bytes of keeping FP16 vs this convert's packed format.

    Linear (2D) → NVFP4 (~0.5 B/elem): FP16 2B − 0.5B = +1.5 B/elem.
    Conv2d (4D) → INT8 (1 B/elem): FP16 2B − 1B = +1 B/elem.
    Matches budget ranking to post-pack assert (not all-INT8 1× for Linear).
    """
    n = int(weight.numel())
    if int(weight.ndim) == 2:
        return (n * 3) // 2
    return n


def _measure_v4_pack_mse_absmax(
    *,
    weight: torch.Tensor,
    importance: torch.Tensor | None,
    optimizer: HSWQWeightedHistogramOptimizerV4,
    layer_name: str = "",
    linear_pack: str = "nvfp4",
) -> float:
    """Real pack roundtrip MSE @ absmax for FP16 protect ranking (V3.0 parity).

    Linear: kitchen TensorCoreNVFP4Layout, or channelwise INT8 when
    linear_pack="int8" (multi-tier shelter d1). Conv2d: channelwise INT8.
    Pack amax stays absmax — this MSE only ranks FP16 keep.

    SVD is mandatory (use_svd_leverage=True). DualMonitor Importance multiplies
    when present; missing Importance never disables SVD. Discarding SVD after
    compute (float32 kitchen path etc.) is blasphemy — fixed in pack BF16 cast.
    """
    result = optimizer.compute_pack_mse_absmax_with_svd(
        weight,
        channel_importance=importance,
        use_svd_leverage=True,
        layer_name=layer_name,
        linear_pack=linear_pack,
    )
    return float(result["estimated_mse"])


def _mse_grayzone_veto_reassessment(
    *,
    scope_label: str,
    hard_veto_layers: set,
    keep_layers: set,
    outlier_only_veto: set,
    target_modules: list,
    model: torch.nn.Module,
    _norm_profile: dict,
    get_layer_search_low,
    alpha: float,
    beta: float,
    device: str,
    tunables: SdxlVetoTunables,
    dual_monitors: dict | None = None,
    mse_cache: dict | None = None,
) -> tuple[set, set, dict]:
    """Gray-zone soft-VETO release; V4 MSE also fills FP16 protect cache.

    Primary V4 role in NVFP4 convert is FP16 protection ranking (see
    _build_v4_calib_fp16_candidates). This path reuses the same V4
    estimated_mse @ absmax to optionally release soft analyze-VETO layers
    whose damage is below P75×mult of a safe baseline.

    Pack amax stays absmax  -  V4 does not choose pack scale.

    DualMonitor importance preferred when present; always Full-SVD×RMS
    via alpha_auto (missing Imp never skips V4 or SVD).

    Returns (hard_veto, keep, mse_cache) where mse_cache maps layer name →
    V4 estimated_mse at absmax (FP16-budget priority; not profile_score).
    Reuses the caller's mse_cache (V4 calib scores); never wipes it.
    """
    mse_cache = dict(mse_cache or {})
    if not outlier_only_veto:
        return hard_veto_layers, keep_layers, mse_cache

    if not dual_monitors:
        raise ValueError(
            f"{scope_label}: V4 gray-zone path needs DualMonitor maps from "
            "calibration (num_calib_samples=32 recipe)."
        )

    print(
        f"\n  [{scope_label} V4→FP16 protect / gray-zone] "
        f"{len(outlier_only_veto)} soft-VETO candidates from analyze "
        f"(o>{tunables.mse_release_o_min:.2f}, "
        f"k<={tunables.mse_release_k_max:.2f}, m<={tunables.mse_release_m_max:.2f})."
    )
    print(
        "  [V4→FP16 protect] real pack MSE @ absmax "
        "(Linear=kitchen NVFP4, Conv2d=channelwise INT8; "
        "FP16 ranking + optional soft-VETO release); "
        f"release if MSE <= {tunables.mse_p75_multiplier:.2f}×P75(safe)."
    )

    trial_optimizer = HSWQWeightedHistogramOptimizerV4(
        bins=8192, num_candidates=1000, refinement_iterations=10,
        device=device, alpha=alpha, beta=beta,
    )

    safe_mses = []
    _module_dict = dict(model.named_modules())
    _safe_pool = [n for n in target_modules if n not in keep_layers and n in _module_dict]
    ff2_suffixes = _discover_ff2_suffixes(
        _norm_profile, min_count=tunables.ff2_suffix_min_count
    )
    if ff2_suffixes:
        _safe_ff = [n for n in _safe_pool if any(n.endswith(s) for s in ff2_suffixes)]
    else:
        _safe_ff = []
    step = max(1, len(_safe_ff) // 30)
    _safe_sample = _safe_ff[::step][:30]
    for sname in _safe_sample:
        smod = _module_dict[sname]
        if not hasattr(smod, "weight"):
            continue
        sw = smod.weight.data
        simp = _dualmonitor_channel_importance(dual_monitors, sname)
        try:
            smse = _measure_v4_pack_mse_absmax(
                weight=sw,
                importance=simp,
                optimizer=trial_optimizer,
                layer_name=sname,
            )
            safe_mses.append(smse)
            mse_cache[sname] = float(smse)
        except Exception as e:
            print(f"    [MSE ERROR] Failed safe layer {sname}: {e}")
        torch.cuda.empty_cache()

    if not safe_mses:
        print(
            f"  [{scope_label} V4→FP16 protect / gray-zone] "
            "No safe baseline available, skipping."
        )
        return hard_veto_layers, keep_layers, mse_cache

    safe_mses.sort()
    p75_idx = int(len(safe_mses) * 0.75)
    mse_threshold = safe_mses[min(p75_idx, len(safe_mses) - 1)] * tunables.mse_p75_multiplier
    print(
        f"  [MSE Baseline NVFP4] Safe layers sampled: {len(safe_mses)}, "
        f"P75 MSE: {safe_mses[p75_idx]:.8f}, "
        f"Threshold ({tunables.mse_p75_multiplier:.2f}xP75): {mse_threshold:.8f}"
    )

    released = set()
    for vname in sorted(outlier_only_veto):
        if vname not in _module_dict:
            continue
        vmod = _module_dict[vname]
        if not hasattr(vmod, "weight"):
            continue
        vw = vmod.weight.data
        vimp = _dualmonitor_channel_importance(dual_monitors, vname)
        try:
            vmse = _measure_v4_pack_mse_absmax(
                weight=vw,
                importance=vimp,
                optimizer=trial_optimizer,
                layer_name=vname,
            )
            mse_cache[vname] = float(vmse)
            vprof = _norm_profile.get(vname, {})
            vor = vprof.get("outlier_ratio", 0)
            if vmse <= mse_threshold:
                released.add(vname)
                print(
                    f"    RELEASED: {vname} | MSE={vmse:.8f} <= threshold={mse_threshold:.8f} "
                    f"| o={vor:.1f}"
                )
            else:
                print(
                    f"    KEPT:     {vname} | MSE={vmse:.8f} >  threshold={mse_threshold:.8f} "
                    f"| o={vor:.1f}"
                )
        except Exception as e:
            print(f"    ERROR:    {vname} | {e}")
        torch.cuda.empty_cache()

    if released:
        hard_veto_layers = hard_veto_layers - released
        keep_layers = keep_layers - released
        print(
            f"  [{scope_label} V4→FP16 protect / gray-zone] "
            f"Released {len(released)} soft-VETO layers. "
            f"Remaining hard VETO: {len(hard_veto_layers)}."
        )
        print(f"  Updated FP16 kept layers: {len(keep_layers)}")
    else:
        print(
            f"  [{scope_label} V4→FP16 protect / gray-zone] "
            "No soft-VETO release (all exceeded MSE threshold)."
        )

    return hard_veto_layers, keep_layers, mse_cache


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
    later --fp16_budget_mb / --budget_mb ceiling can rank which layers stay
    FP16. Pack amax remains absmax separately  -  V4 does not search pack scale.

    Always Full-SVD×RMS hybrid (surface 4 of comprehensive FP16 ranking);
    DualMonitor Importance multiplies when present. Never skip a measurable
    layer (skipping collapses FP16 selection). SVD skip/discard = blasphemy.

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
            f"  [V4→FP16 protect] measuring real pack MSE @ absmax for "
            f"{len(need)} layers (Linear=NVFP4, Conv=INT8; pack stays absmax; "
            f"cache hit={len(scored)}; analyze VETO={len(hard_veto_layers)}; "
            f"NO keep_ratio pre-cut)..."
        )
        trial_optimizer = HSWQWeightedHistogramOptimizerV4(
            bins=8192, num_candidates=1000, refinement_iterations=10,
            device=device, alpha=alpha, beta=beta,
        )
    for name in need:
        if trial_optimizer is None:
            break
        mod = module_dict[name]
        imp = _dualmonitor_channel_importance(dual_monitors, name)
        try:
            v4_mse = _measure_v4_pack_mse_absmax(
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
) -> tuple[set, set, dict, set]:
    """Per-model auto analysis → auto-optimal 3-tier set inside the run budget.

    budget_mb is the run MiB budget (overhead vs packed NVFP4 Linear /
    INT8 Conv2d), from CLI --fp16_budget_mb / --budget_mb. Auto settings
    fill that frame; they never redefine or exceed it.

    Multi-tier shelter (owner 2026-07-20): Linear has THREE states —
    NVFP4 (baseline) → INT8 ConvRot (+0.5 B/elem) → FP16 (+1.0 B/elem more);
    Conv2d keeps INT8 (baseline) → FP16 (+1 B/elem). INT8 shelter protects
    ~3x more elements than FP16 at the same budget, rescuing layers whose
    fine singulars zero-collapse on NVFP4's 16-level grid onto INT8's
    256-level grid. Each upgrade step is scored priority x measured rescue
    fraction (d0 = NVFP4 pack MSE, d1 = INT8 pack MSE, same importance
    weighting) and ALL steps compete in ONE score/byte ranking.

    Comprehensive ranking (owner 2026-07-20) — ALL arranged together:
      (1) V4 histogram calib → estimated_mse @ absmax (d0 AND d1)
      (2) DualMonitor sensitivity (+ Importance into V4 hybrid)
      (3) analyze JSON → severity / Hard VETO / tunables
      (4) Full-SVD×RMS inside every V4 pack MSE measure
    Linear and Conv compete in ONE ranking. Priority weights are derived
    per-checkpoint via infinite branches — never fixed Conv-first /
    Mag-outside / Mag-tax exemption / model-name recipe.

    alpha/beta MUST be THIS-profile auto-optimal (caller passes
    veto_tunables.alpha_auto mix). Fixed 0.5/0.5 defaults are forbidden.

    Returns (fp16_keep, hard_veto_layers, budget_stats, int8_shelter).
    """
    if not math.isfinite(float(alpha)) or not math.isfinite(float(beta)):
        raise ValueError(
            f"_apply_fp16_budget_cap: alpha/beta must be finite auto-optimal "
            f"(got alpha={alpha}, beta={beta})"
        )
    budget_mb = _bind_fp16_budget_from_option(budget_mb)
    print(
        "  [FP16 candidates] comprehensive surfaces (THIS model): "
        "V4 calib MSE | DualMonitor | analyze JSON | SVD×RMS — "
        "arranged → infinite-branch auto-optimal; no fixed recipe."
    )
    from analyze_sdxl_nvfp4_distribution import (
        apply_fp16_infinite_priority_branches,
        apply_fp16_infinite_ranking_branches,
        build_nvfp4_analyze_character_table,
        nvfp4_fp16_budget_analyze_severity,
        nvfp4_fp16_budget_priority,
        derive_priority_combinator,
        _safe_percentile,
        _robust_iqr,
    )

    if str(veto_tunables.quant_format) != "nvfp4":
        raise ValueError(
            "_apply_fp16_budget_cap is NVFP4-path "
            f"(got quant_format={veto_tunables.quant_format!r})"
        )
    if not dual_monitors:
        raise ValueError(
            "[FP16 budget] DualMonitor maps required for Sensitivity + "
            "V4 Importance; refusing fixed-formula / profile_score fallback."
        )

    tunables_dict = veto_tunables.as_dict()
    budget_bytes = int(budget_mb * 1024 * 1024)

    char_table = build_nvfp4_analyze_character_table(
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
    # Multi-tier: every 2D pool layer also needs d1 (INT8 pack MSE) so the
    # NVFP4→INT8 shelter step can be priced. Measured here in ONE place with
    # the SAME optimizer/importance as d0 (per-run cache covers d0 only).
    need_int8_d1 = [
        n for n in pool
        if module_dict.get(n) is not None
        and getattr(module_dict[n], "weight", None) is not None
        and int(module_dict[n].weight.ndim) == 2
    ]
    trial_optimizer = None
    if need_fresh or need_int8_d1:
        print(
            f"  [FP16 budget] THIS-model pool measure: "
            f"analyze={len(char_table)} pool={len(pool)} "
            f"dm_sens={len(sens_by_name)} | V4 fresh={len(need_fresh)} "
            f"(cache={len(cache)}) INT8 d1={len(need_int8_d1)}..."
        )
        trial_optimizer = HSWQWeightedHistogramOptimizerV4(
            bins=8192, num_candidates=1000, refinement_iterations=10,
            device=device, alpha=alpha, beta=beta,
        )
    else:
        print(
            f"  [FP16 budget] THIS-model pool: "
            f"analyze={len(char_table)} pool={len(pool)} "
            f"dm_sens={len(sens_by_name)} | V4 cached ({len(cache)})"
        )

    int8_mse: dict[str, float] = {}
    for name in sorted(pool):
        mod = module_dict.get(name)
        if mod is None or not hasattr(mod, "weight") or mod.weight is None:
            skipped_no_weight.append(name)
            continue
        dm_sens = float(sens_by_name.get(name, 0.0))
        extra = _fp16_extra_bytes_vs_packed(mod.weight.data)
        row = char_table.get(name, {})
        prof = norm_profile.get(name, {}) if isinstance(norm_profile.get(name), dict) else {}
        is_hv = name in hard_veto_layers
        k = float(row.get("kurtosis", prof.get("kurtosis", 0)) or 0)
        o = float(row.get("outlier_ratio", prof.get("outlier_ratio", 0)) or 0)
        m = float(row.get("abs_max", prof.get("abs_max", 0)) or 0)
        mad = float(row.get("mad_outlier_pct", prof.get("mad_outlier_pct", 0)) or 0)
        ps = float(row.get("profile_score", prof.get("profile_score", 0)) or 0)
        severity = nvfp4_fp16_budget_analyze_severity(
            kurtosis=k,
            outlier_ratio=o,
            abs_max=m,
            tunables=tunables_dict,
            is_hard_veto=is_hv,
            layer_name=name,
            mad_outlier_pct=mad,
            profile_score=ps,
        )

        imp = _dualmonitor_channel_importance(dual_monitors, name)
        if name in cache:
            v4_mse = float(cache[name])
        else:
            if trial_optimizer is None:
                skipped_no_v4.append(name)
                continue
            try:
                v4_mse = _measure_v4_pack_mse_absmax(
                    weight=mod.weight.data,
                    importance=imp,
                    optimizer=trial_optimizer,
                    layer_name=name,
                )
                cache[name] = v4_mse
                measured_fresh += 1
            except Exception as e:
                print(f"    [FP16 budget] V4 MSE failed {name}: {e} -> pack")
                skipped_no_v4.append(name)
                continue
            torch.cuda.empty_cache()

        if int(mod.weight.data.ndim) == 2 and trial_optimizer is not None:
            try:
                d1 = _measure_v4_pack_mse_absmax(
                    weight=mod.weight.data,
                    importance=imp,
                    optimizer=trial_optimizer,
                    layer_name=name,
                    linear_pack="int8",
                )
                int8_mse[name] = float(d1)
            except Exception as e:
                print(
                    f"    [Multi-tier] INT8 d1 failed {name}: {e} "
                    f"(no INT8 shelter step for this layer)"
                )

        measured.append((name, dm_sens, v4_mse, severity, extra))

    print(
        f"  [Multi-tier] INT8 d1 measured for {len(int8_mse)}/"
        f"{len(need_int8_d1)} Linear pool layers (same importance weighting "
        f"as d0; rescue fraction = (d0-d1)/d0)"
    )

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
        priority = nvfp4_fp16_budget_priority(
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

    # Multi-tier shelter ladder (owner 2026-07-20): Linear has THREE states —
    # NVFP4 baseline → INT8 ConvRot shelter (+0.5 B/elem) → FP16 (+1.0 B/elem
    # more); Conv2d keeps INT8 baseline → FP16 (+1 B/elem). INT8 shelter costs
    # ~1/3 of FP16 per element, so the hard ceiling rescues ~3x more elements.
    # Each upgrade step scores priority x MEASURED rescue fraction
    # (d0 = NVFP4 pack MSE, d1 = INT8 pack MSE, same importance weighting) and
    # ALL steps compete in ONE score/byte ranking. No fixed tier reservation.
    extra_by_name: dict[str, int] = {}
    int8_cost_by_name: dict[str, int] = {}
    fp16_marginal_by_name: dict[str, int] = {}
    ndim_by_name: dict[str, int] = {}
    # (score, cost_bytes, tier, name, priority, rescue_frac)
    steps: list[tuple[float, int, str, str, float, float]] = []
    for priority, v4_mse, severity, dm_sens, extra, name in candidates:
        mod = module_dict[name]
        w_ndim = int(mod.weight.data.ndim)
        n_el = int(mod.weight.data.numel())
        extra_by_name[name] = int(extra)
        ndim_by_name[name] = w_ndim
        d0 = float(v4_mse)
        if w_ndim == 2:
            cost_int8 = n_el // 2
            int8_cost_by_name[name] = cost_int8
            # FP16 marginal from INT8: 2.0-1.0 = +1.0 B/elem → n_el bytes.
            fp16_marginal_by_name[name] = n_el
            d1 = int8_mse.get(name)
            if d1 is not None and d0 > 0.0:
                r1 = min(max((d0 - d1) / d0, 0.0), 1.0)
                r2 = min(max(d1 / d0, 0.0), 1.0)
            elif d0 > 0.0:
                # d1 unmeasured: INT8 step unavailable, FP16 removes all.
                r1, r2 = 0.0, 1.0
            else:
                r1, r2 = 0.0, 0.0
            if d1 is not None and r1 > 0.0:
                steps.append((priority * r1, cost_int8, "int8", name, priority, r1))
            if r2 > 0.0:
                steps.append(
                    (priority * r2, n_el, "fp16", name, priority, r2)
                )
        else:
            # Conv2d baseline is INT8; single FP16 step removes all damage.
            fp16_marginal_by_name[name] = int(extra)
            if priority > 0.0:
                steps.append((priority, int(extra), "fp16", name, priority, 1.0))

    steps.sort(key=lambda t: (-(t[0] / max(t[1], 1)), t[3]))

    tier_state: dict[str, str] = {}
    used = 0
    used_int8 = 0
    used_fp16 = 0
    shelter_detail: list[tuple[str, int, float, float, float]] = []
    kept_detail: list[tuple[str, int, float, float, float, float]] = []
    # Diagnostic trace (owner 2026-07-21): record every step's fate so the
    # budget-cut boundary (accepted tail vs rejected head) can be dumped to
    # the log. Tuple = (rank, score/byte, tier, name, charge, status,
    # remaining_before); status is "ok" or "no_fit".
    ladder_trace: list[tuple[int, float, str, str, int, str, int]] = []
    for _rank, (score, cost, tier, name, priority, rescue_frac) in enumerate(
        steps, 1
    ):
        if score <= 0.0:
            continue
        cur = tier_state.get(name)
        if tier == "int8":
            if cur is not None:
                continue
            charge = cost
        else:
            if cur == "fp16":
                continue
            if cur == "int8":
                charge = fp16_marginal_by_name[name]
            else:
                # Direct baseline → FP16 pays the full FP16 extra.
                charge = extra_by_name[name]
        if used + charge > budget_bytes:
            ladder_trace.append(
                (_rank, score / max(cost, 1), tier, name, charge, "no_fit",
                 budget_bytes - used)
            )
            continue
        ladder_trace.append(
            (_rank, score / max(cost, 1), tier, name, charge, "ok",
             budget_bytes - used)
        )
        used += charge
        tier_state[name] = tier
        if tier == "int8":
            used_int8 += charge
            shelter_detail.append((name, charge, score, priority, rescue_frac))
        else:
            used_fp16 += charge

    keep_out = {n for n, s in tier_state.items() if s == "fp16"}
    shelter_out = {n for n, s in tier_state.items() if s == "int8"}
    cand_by_name = {c[5]: c for c in candidates}
    for name in sorted(keep_out):
        priority, v4_mse, severity, dm_sens, extra, _ = cand_by_name[name]
        kept_detail.append((name, extra, priority, v4_mse, severity, dm_sens))
    dropped: list[tuple[str, int, float, float, float, float]] = []
    for priority, v4_mse, severity, dm_sens, extra, name in candidates:
        if name not in tier_state:
            dropped.append((name, extra, priority, v4_mse, severity, dm_sens))

    demoted_veto = hard_veto_layers - keep_out
    # Auto-optimal 3-tier set for THIS model (DualMonitor + analyze + V4 d0/d1).
    # Analyze VETO that win FP16 stay labeled VETO; INT8 shelter is the cheap
    # middle tier; demoted Linear pack NVFP4, demoted Conv pack INT8.
    hard_veto_out = hard_veto_layers & keep_out
    veto_in_shelter = hard_veto_layers & shelter_out

    if used > budget_bytes:
        raise RuntimeError(
            f"[Multi-tier] selected set exceeds hard ceiling "
            f"{budget_mb:g} MiB: used={used / (1024 * 1024):.3f} MiB "
            f"({used} bytes > {budget_bytes}). Refusing to proceed."
        )

    print(
        f"  [Multi-tier] steps={len(steps)} → FP16 keep={len(keep_out)} "
        f"({used_fp16 / (1024 * 1024):.2f} MiB) + INT8 shelter="
        f"{len(shelter_out)} ({used_int8 / (1024 * 1024):.2f} MiB) = "
        f"{used / (1024 * 1024):.2f}/{budget_mb:g} MiB | veto→fp16="
        f"{len(hard_veto_out)} veto→int8={len(veto_in_shelter)} "
        f"veto→packed={len(demoted_veto - shelter_out)}"
    )

    # Diagnostic dumps (owner 2026-07-21): full visibility of WHAT the
    # budget bought and what it cut. All lines print to stdout → run log.
    print(
        f"  [Shelter-detail] accepted INT8-shelter steps "
        f"(ladder/score-per-byte order, top 40 of {len(shelter_detail)}):"
    )
    for _n, _ch, _sc, _prio, _r1 in shelter_detail[:40]:
        print(
            f"    {_n}  charge={_ch / (1024 * 1024):8.3f}MiB "
            f"step_score={_sc:.6e} priority={_prio:.6e} rescue_frac={_r1:.4f}"
        )
    _dropped_sorted = sorted(dropped, key=lambda t: -t[2])
    print(
        f"  [Dropped-detail] rejected candidates (priority order, "
        f"top 40 of {len(dropped)}):"
    )
    for _n, _ex, _prio, _v4, _sev, _dm in _dropped_sorted[:40]:
        print(
            f"    {_n}  fp16_cost={_ex / (1024 * 1024):8.3f}MiB "
            f"priority={_prio:.6e} v4_mse={_v4:.6e} severity={_sev:.4f} "
            f"dm_sens={_dm:.6e}"
        )
    _trace_ok = [t for t in ladder_trace if t[5] == "ok"]
    _trace_rej = [t for t in ladder_trace if t[5] == "no_fit"]
    print(
        f"  [Cut-window] ladder accepted={len(_trace_ok)} "
        f"rejected_no_fit={len(_trace_rej)} "
        f"(rank = score/byte order; boundary = accepted TAIL vs rejected HEAD)"
    )
    print("    --- accepted TAIL (last 20 steps to make the cut) ---")
    for _r, _spb, _tier, _n, _ch, _st, _rem in _trace_ok[-20:]:
        print(
            f"    #{_r:>4} {_spb:.6e}/B tier={_tier:<4} "
            f"charge={_ch / (1024 * 1024):8.3f}MiB "
            f"remain_before={_rem / (1024 * 1024):8.3f}MiB  {_n}"
        )
    print("    --- rejected HEAD (first 20 steps to miss the cut) ---")
    for _r, _spb, _tier, _n, _ch, _st, _rem in _trace_rej[:20]:
        print(
            f"    #{_r:>4} {_spb:.6e}/B tier={_tier:<4} "
            f"charge={_ch / (1024 * 1024):8.3f}MiB "
            f"remain_before={_rem / (1024 * 1024):8.3f}MiB  {_n}"
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
        "total_fp16_mb": used_fp16 / (1024 * 1024),
        "total_int8_shelter_mb": used_int8 / (1024 * 1024),
        "mag_forced_fp16_count": 0,
        "candidates": len(candidates),
        "pool": len(pool),
        "analyze_character_layers": len(char_table),
        "dm_sensitivity_layers": len(sens_by_name),
        "kept": len(keep_out),
        "int8_sheltered": len(shelter_out),
        "int8_shelter_bytes": int(used_int8),
        "dropped": len(dropped),
        "demoted_veto": len(demoted_veto),
        "veto_in_int8_shelter": len(veto_in_shelter),
        "skipped_no_weight": len(skipped_no_weight),
        "skipped_no_v4": len(skipped_no_v4),
        "measured_fresh_v4": measured_fresh,
        "int8_d1_measured": len(int8_mse),
        "multi_tier_steps": len(steps),
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
            "per_model_auto_analysis_multi_tier_score_per_byte_inside_"
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
        "shelter_detail": shelter_detail[:40],
        "mse_cache_size": len(cache),
    }
    return keep_out, hard_veto_out, stats, shelter_out



def compute_int8_bias_delta(weight_fp, weight_dq, act_mean):
    """Bias correction delta for one INT8 layer.

    Cancels systematic output shift E[(W_q - W) x] ≈ (W_q - W) contracted with
    per-input-channel mean activation from calibration.

    Linear  weight (O, I):     delta[o] = sum_i err[o,i] * mu[i]
    Conv2d  weight (O, I, K, K): delta[o] = sum_{i,k,h} err[o,i,kh,kw] * mu[i]
    """
    if act_mean is None:
        return None
    err = (weight_dq.float() - weight_fp.float())
    mu = act_mean.float().to(device=err.device)
    if err.ndim == 2:
        # Linear: (O, I) @ (I,) -> (O,)
        if mu.numel() != err.shape[1]:
            return None
        return err @ mu
    if err.ndim == 4:
        # Conv2d: sum over in/spatial with per-in-channel mu
        if mu.numel() != err.shape[1]:
            return None
        return (err * mu.view(1, -1, 1, 1)).sum(dim=(1, 2, 3))
    return None



# DualMonitor MUST be defined before calibration hooks (NameError if missing).
# HEAD historically called the class from hook_fn without defining it.
class DualMonitor:
    """Per-layer calibration monitor for THIS checkpoint (auto analysis input).

    Accumulates output variance (sensitivity), channel importance, and
    activation moments used by V4 Importance and FP16 budget ranking.
    """

    def __init__(self):
        self.output_sum = 0.0
        self.output_sq_sum = 0.0
        self.count = 0
        self.channel_importance = None
        # Signed per-channel input mean for INT8 bias correction:
        #   bias_delta ≈ (W_q - W) @ E[x]
        self.channel_act_mean = None
        # Per-input-channel second moment E[x_i^2] for damage calculation:
        #   damage_l ≈ sum_i (ΔW^2)[*,i,*] · E[x_i^2]  (pre-grad factor)
        self.channel_act_sq_mean = None
    
    def update(self, input_tensor, output_tensor):
        with torch.no_grad():
            out_detached = output_tensor.detach().float()
            out_clamped = torch.clamp(out_detached, -65504.0, 65504.0)
            mean_val = out_clamped.mean().item()
            sq_mean_val = (out_clamped ** 2).mean().item()
            import math
            if math.isfinite(mean_val) and math.isfinite(sq_mean_val):
                self.output_sum += mean_val
                self.output_sq_sum += sq_mean_val
            inp_detached = input_tensor.detach().float()
            if inp_detached.dim() == 4:
                current_imp = inp_detached.abs().mean(dim=(0, 2, 3))
                current_act = inp_detached.mean(dim=(0, 2, 3))
                current_sq = (inp_detached ** 2).mean(dim=(0, 2, 3))
            elif inp_detached.dim() == 3:
                current_imp = inp_detached.abs().mean(dim=(0, 1))
                current_act = inp_detached.mean(dim=(0, 1))
                current_sq = (inp_detached ** 2).mean(dim=(0, 1))
            elif inp_detached.dim() == 2:
                current_imp = inp_detached.abs().mean(dim=0)
                current_act = inp_detached.mean(dim=0)
                current_sq = (inp_detached ** 2).mean(dim=0)
            else:
                current_imp = torch.ones(1, device=inp_detached.device, dtype=torch.float32)
                current_act = torch.zeros(1, device=inp_detached.device, dtype=torch.float32)
                current_sq = torch.ones(1, device=inp_detached.device, dtype=torch.float32)
            if self.channel_importance is None:
                self.channel_importance = current_imp
                self.channel_act_mean = current_act
                self.channel_act_sq_mean = current_sq
            else:
                self.channel_importance = (
                    self.channel_importance * self.count + current_imp
                ) / (self.count + 1)
                self.channel_act_mean = (
                    self.channel_act_mean * self.count + current_act
                ) / (self.count + 1)
                self.channel_act_sq_mean = (
                    self.channel_act_sq_mean * self.count + current_sq
                ) / (self.count + 1)
            self.count += 1

    def get_sensitivity(self):
        if self.count == 0:
            return 0.0
        mean = self.output_sum / self.count
        variance = (self.output_sq_sum / self.count) - mean ** 2
        import math
        return variance if math.isfinite(variance) else 0.0

    def get_input_second_moment(self):
        """Per-input-channel E[x_i^2] accumulated during calibration (float32, CPU)."""
        if self.count == 0 or self.channel_act_sq_mean is None:
            return None
        return self.channel_act_sq_mean.detach().float().cpu()


dual_monitors = {}


def hook_fn(module, input, output, name):
    if name not in dual_monitors:
        dual_monitors[name] = DualMonitor()
    dual_monitors[name].update(input[0], output)


def _remap_profile_to_diffusers(model_profile: dict, comfyui_to_diffusers_map: dict) -> dict:
    """Map analyze JSON keys (ComfyUI .weight) to Diffusers module names for named_modules()."""
    if not model_profile or not comfyui_to_diffusers_map:
        return model_profile
    remapped = {}
    unmapped = 0
    for comfy_key, val in model_profile.items():
        if not isinstance(val, dict):
            continue
        diff_key = comfyui_to_diffusers_map.get(comfy_key)
        if diff_key is None:
            unmapped += 1
            continue
        mod_name = diff_key[:-7] if diff_key.endswith(".weight") else diff_key
        remapped[mod_name] = val
    if unmapped:
        print(f"  [Profile Remap] {unmapped} Comfy keys had no diffusers mapping (skipped)")
    print(f"  [Profile Remap] {len(remapped)} diffusers module profile entries")
    return remapped


def derive_hswq_strategy_nvfp4(model_profile, veto_tunables: SdxlVetoTunables | None = None):
    """
    SDXL NVFP4: Alpha/Beta from profile + absmax pack + V4 FP16 ranking.

    - search_low: 1.0 → pack amax = absmax (obvious for symmetric INT8).
    - V4 pack MSE: FP16 protection candidate ranking
      (real Linear NVFP4 / Conv INT8 roundtrip @ absmax; budget truncates).
      Soft gray-zone VETO release is secondary reuse of the same MSE.
    - alpha/beta: alpha_auto from THIS multi-axis analyze character
      (kurtosis∪outlier∪magnitude → Full-SVD×RMS); DualMonitor Importance
      multiplies the hybrid map when present. No fixed mix / no SVD off.
    - hard_veto: thresholds from derive_veto_tunables_nvfp4 (this checkpoint).
    """
    if model_profile:
        sample_key = next(iter(model_profile))
        profile_prefix = ""
        for pfx in _SDXL_PROFILE_PREFIXES:
            if pfx and sample_key.startswith(pfx):
                profile_prefix = pfx
                break
        if profile_prefix:
            normalized_profile = {}
            for key, val in model_profile.items():
                stripped_key = (
                    key[len(profile_prefix):] if key.startswith(profile_prefix) else key
                )
                normalized_profile[stripped_key] = val
            model_profile = normalized_profile
            print(
                f"  [Profile Normalize] Stripped prefix '{profile_prefix}' "
                f"from {len(normalized_profile)} profile keys."
            )

    if veto_tunables is None:
        # Fallback only: use module budget (bound from CLI at main entry).
        veto_tunables = resolve_veto_tunables(
            model_profile or {},
            fp16_budget_mb=FP16_BUDGET_MB_HARD,
        )

    print(
        "  [NVFP4 pack] absmax (search_low=1.0); "
        "[V4 histogram] FP16 protection candidate ranking @ absmax"
    )

    def get_dynamic_search_low(name, weight_tensor):
        # Natural absmax pack point. V4 does not choose pack amax.
        return 1.0

    if model_profile:
        all_k = [p.get("kurtosis", 0) for p in model_profile.values() if isinstance(p, dict)]
        all_o = [p.get("outlier_ratio", 0) for p in model_profile.values() if isinstance(p, dict)]
        all_m = [p.get("abs_max", 0) for p in model_profile.values() if isinstance(p, dict)]
        avg_k = np.mean(all_k) if all_k else 0
        avg_o = np.mean(all_o) if all_o else 0
        avg_m = np.mean(all_m) if all_m else 0
        print(f"  [Profile Stats NVFP4] Avg Kurtosis: {avg_k!r}, Avg OutlierRatio: {avg_o!r}, Avg AbsMax: {avg_m!r}")

    # alpha = SVD-leverage MIX WEIGHT from THIS multi-axis character (k∪o∪m).
    # DualMonitor Imp multiplies the hybrid map when present. alpha==0 with a
    # live profile is SVD cut  -  refuse (do not log "executing" as if contributing).
    alpha = float(veto_tunables.alpha_auto)
    if model_profile and alpha <= 0.0:
        raise ValueError(
            "NVFP4 Full-SVD×RMS alpha_auto must be > 0 when model_profile is present "
            f"(alpha==0 is SVD cut / rebellion). got alpha_auto={alpha}"
        )
    beta = 1.0 - alpha

    print(
        f"  [Dynamic Alpha/Beta NVFP4] alpha={alpha!r}, beta={beta!r} "
        f"(analyze k∪o∪m → Full-SVD×RMS mix into ranking; Imp multiplies when present)"
    )

    hard_veto_layers = set()
    if model_profile:
        for name, prof in model_profile.items():
            if isinstance(prof, dict):
                k = prof.get("kurtosis", 0)
                m = prof.get("abs_max", 0)
                o = prof.get("outlier_ratio", 0)
                # VETO thresholds = analyze_sdxl_nvfp4_distribution.derive_veto_tunables_nvfp4
                # only (this checkpoint's distribution). No hardcoded floors.
                is_extreme_divergence = o > veto_tunables.extreme_outlier
                is_extreme_kurtosis = k > veto_tunables.extreme_kurtosis
                is_huge_magnitude = m > veto_tunables.huge_magnitude
                if is_extreme_divergence or is_extreme_kurtosis or is_huge_magnitude:
                    layer_base_name = name.replace(".weight", "") if name.endswith(".weight") else name
                    hard_veto_layers.add(layer_base_name)
                    reasons = []
                    if is_extreme_kurtosis:
                        reasons.append(
                            f"k={k!r}>extreme_kurtosis={veto_tunables.extreme_kurtosis!r}"
                        )
                    if is_extreme_divergence:
                        reasons.append(
                            f"o={o!r}>extreme_outlier={veto_tunables.extreme_outlier!r}"
                        )
                    if is_huge_magnitude:
                        reasons.append(
                            f"m={m!r}>huge_magnitude={veto_tunables.huge_magnitude!r}"
                        )
                    print(f"    VETO: {layer_base_name} [{'; '.join(reasons)}]")

    print(
        f"  [Static Profile VETO NVFP4] Identified {len(hard_veto_layers)} layers "
        "with extreme distribution (Unquantizable under NVFP4 protect)."
    )
    return alpha, beta, get_dynamic_search_low, hard_veto_layers



# --- UNet key helpers from 3.0 ---
def count_blocks(state_dict_keys, prefix_string):
    count = 0
    while True:
        c = False
        for k in state_dict_keys:
            if k.startswith(prefix_string.format(count)):
                c = True
                break
        if c == False:
            break
        count += 1
    return count

def calculate_transformer_depth(prefix, state_dict_keys, state_dict):
    transformer_prefix = prefix + "1.transformer_blocks."
    transformer_keys = sorted(list(filter(lambda a: a.startswith(transformer_prefix), state_dict_keys)))
    if len(transformer_keys) > 0:
        last_transformer_depth = count_blocks(state_dict_keys, transformer_prefix + '{}')
        return last_transformer_depth
    return 0

def detect_unet_config_from_keys(state_dict, key_prefix="model.diffusion_model."):
    state_dict_keys = list(state_dict.keys())
    filtered_keys = [k for k in state_dict_keys if k.startswith(key_prefix)]
    unet_config = {}
    if f"{key_prefix}input_blocks.0.0.weight" in state_dict_keys:
        model_channels = state_dict[f"{key_prefix}input_blocks.0.0.weight"].shape[0]
        num_res_blocks = []
        channel_mult = []
        transformer_depth = []
        transformer_depth_output = []
        input_block_count = count_blocks(state_dict_keys, f"{key_prefix}input_blocks" + '.{}.')
        last_res_blocks = 0
        last_channel_mult = 0
        for count in range(input_block_count):
            prefix = f"{key_prefix}input_blocks.{count}."
            prefix_output = f"{key_prefix}output_blocks.{input_block_count - count - 1}."
            block_keys = sorted(list(filter(lambda a: a.startswith(prefix), state_dict_keys)))
            if len(block_keys) == 0: break
            block_keys_output = sorted(list(filter(lambda a: a.startswith(prefix_output), state_dict_keys)))
            if f"{prefix}0.op.weight" in block_keys:
                num_res_blocks.append(last_res_blocks)
                channel_mult.append(last_channel_mult)
                last_res_blocks = 0
                last_channel_mult = 0
                out = calculate_transformer_depth(prefix_output, state_dict_keys, state_dict)
                transformer_depth_output.append(out)
            else:
                res_block_prefix = f"{prefix}0.in_layers.0.weight"
                if res_block_prefix in block_keys:
                    last_res_blocks += 1
                    last_channel_mult = state_dict[f"{prefix}0.out_layers.3.weight"].shape[0] // model_channels
                    out = calculate_transformer_depth(prefix, state_dict_keys, state_dict)
                    transformer_depth.append(out)
                res_block_prefix = f"{prefix_output}0.in_layers.0.weight"
                if res_block_prefix in block_keys_output:
                    out = calculate_transformer_depth(prefix_output, state_dict_keys, state_dict)
                    transformer_depth_output.append(out)
        num_res_blocks.append(last_res_blocks)
        channel_mult.append(last_channel_mult)
        if f"{key_prefix}middle_block.1.proj_in.weight" in state_dict_keys:
            transformer_depth_middle = count_blocks(state_dict_keys, f"{key_prefix}middle_block.1.transformer_blocks." + '{}')
        elif f"{key_prefix}middle_block.0.in_layers.0.weight" in state_dict_keys:
            transformer_depth_middle = -1
        else:
            transformer_depth_middle = -2
        unet_config["num_res_blocks"] = num_res_blocks
        unet_config["channel_mult"] = channel_mult
        unet_config["transformer_depth"] = transformer_depth
        unet_config["transformer_depth_output"] = transformer_depth_output
        unet_config["transformer_depth_middle"] = transformer_depth_middle
    return unet_config

def unet_to_diffusers_mapping(unet_config, state_dict=None, key_prefix="model.diffusion_model."):
    if "num_res_blocks" not in unet_config: return {}
    num_res_blocks = unet_config["num_res_blocks"]
    channel_mult = unet_config["channel_mult"]
    num_blocks = len(channel_mult)
    if state_dict is not None:
        import re
        state_dict_keys = list(state_dict.keys())
        filtered_keys = [k.replace(key_prefix, "") for k in state_dict_keys if k.startswith(key_prefix)]
        transformer_counts = {}
        for key in filtered_keys:
            match = re.match(r'input_blocks\.(\d+)\.1\.transformer_blocks\.(\d+)', key)
            if match:
                block_idx = int(match.group(1))
                trans_idx = int(match.group(2))
                if block_idx not in transformer_counts: transformer_counts[block_idx] = 0
                transformer_counts[block_idx] = max(transformer_counts[block_idx], trans_idx + 1)
        output_transformer_counts = {}
        for key in filtered_keys:
            match = re.match(r'output_blocks\.(\d+)\.1\.transformer_blocks\.(\d+)', key)
            if match:
                block_idx = int(match.group(1))
                trans_idx = int(match.group(2))
                if block_idx not in output_transformer_counts: output_transformer_counts[block_idx] = 0
                output_transformer_counts[block_idx] = max(output_transformer_counts[block_idx], trans_idx + 1)
        middle_transformer_count = 0
        for key in filtered_keys:
            match = re.match(r'middle_block\.1\.transformer_blocks\.(\d+)', key)
            if match:
                trans_idx = int(match.group(1))
                middle_transformer_count = max(middle_transformer_count, trans_idx + 1)
        transformers_mid = middle_transformer_count if middle_transformer_count > 0 else unet_config.get("transformer_depth_middle", None)
    else:
        transformer_depth = unet_config["transformer_depth"][:]
        transformer_depth_output = unet_config["transformer_depth_output"][:]
        transformers_mid = unet_config.get("transformer_depth_middle", None)
        transformer_counts = None
        output_transformer_counts = None
    UNET_MAP_RESNET = {"in_layers.2.weight": "conv1.weight", "in_layers.2.bias": "conv1.bias", "emb_layers.1.weight": "time_emb_proj.weight", "emb_layers.1.bias": "time_emb_proj.bias", "out_layers.3.weight": "conv2.weight", "out_layers.3.bias": "conv2.bias", "skip_connection.weight": "conv_shortcut.weight", "skip_connection.bias": "conv_shortcut.bias", "in_layers.0.weight": "norm1.weight", "in_layers.0.bias": "norm1.bias", "out_layers.0.weight": "norm2.weight", "out_layers.0.bias": "norm2.bias"}
    UNET_MAP_ATTENTIONS = {"proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias", "norm.weight", "norm.bias"}
    TRANSFORMER_BLOCKS = {"norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "norm3.weight", "norm3.bias", "attn1.to_q.weight", "attn1.to_q.bias", "attn1.to_k.weight", "attn1.to_k.bias", "attn1.to_v.weight", "attn1.to_v.bias", "attn1.to_out.0.weight", "attn1.to_out.0.bias", "attn2.to_q.weight", "attn2.to_k.weight", "attn2.to_v.weight", "attn2.to_out.0.weight", "attn2.to_out.0.bias", "ff.net.0.proj.weight", "ff.net.0.proj.bias", "ff.net.2.weight", "ff.net.2.bias"}
    UNET_MAP_BASIC = {("label_emb.0.0.weight", "add_embedding.linear_1.weight"), ("label_emb.0.0.bias", "add_embedding.linear_1.bias"), ("label_emb.0.2.weight", "add_embedding.linear_2.weight"), ("label_emb.0.2.bias", "add_embedding.linear_2.bias"), ("input_blocks.0.0.weight", "conv_in.weight"), ("input_blocks.0.0.bias", "conv_in.bias"), ("out.0.weight", "conv_norm_out.weight"), ("out.0.bias", "conv_norm_out.bias"), ("out.2.weight", "conv_out.weight"), ("out.2.bias", "conv_out.bias"), ("time_embed.0.weight", "time_embedding.linear_1.weight"), ("time_embed.0.bias", "time_embedding.linear_1.bias"), ("time_embed.2.weight", "time_embedding.linear_2.weight"), ("time_embed.2.bias", "time_embedding.linear_2.bias")}
    # Map only tensors present in this checkpoint's state_dict (auto from weights).
    # No invented Diffusers names, no fixed KEEP list, no inject.
    if state_dict is None:
        raise RuntimeError(
            "unet_to_diffusers_mapping requires state_dict; refuse maps without "
            "Comfy presence checks"
        )
    _sd_keys = set(state_dict.keys())
    _comfy_bare = {
        (k[len(key_prefix):] if k.startswith(key_prefix) else k)
        for k in _sd_keys
    }

    def _comfy_present(comfy_bare: str) -> bool:
        return comfy_bare in _comfy_bare or f"{key_prefix}{comfy_bare}" in _sd_keys

    def _map_put(diff_key: str, comfy_bare: str) -> None:
        if not _comfy_present(comfy_bare):
            return
        diffusers_unet_map[diff_key] = comfy_bare

    diffusers_unet_map = {}
    for x in range(num_blocks):
        n = 1 + (num_res_blocks[x] + 1) * x
        for i in range(num_res_blocks[x]):
            for b in UNET_MAP_RESNET:
                _map_put(
                    "down_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b]),
                    "input_blocks.{}.0.{}".format(n, b),
                )
            if transformer_counts is not None: num_transformers = transformer_counts.get(n, 0)
            else: num_transformers = transformer_depth.pop(0) if transformer_depth else 0
            if num_transformers > 0:
                for b in UNET_MAP_ATTENTIONS:
                    _map_put(
                        "down_blocks.{}.attentions.{}.{}".format(x, i, b),
                        "input_blocks.{}.1.{}".format(n, b),
                    )
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        _map_put(
                            "down_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b),
                            "input_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b),
                        )
            n += 1
        # Last DownBlock has no downsampler in SDXL  -  register only if op exists.
        if _comfy_present("input_blocks.{}.0.op.weight".format(n)):
            for k in ["weight", "bias"]:
                _map_put(
                    "down_blocks.{}.downsamplers.0.conv.{}".format(x, k),
                    "input_blocks.{}.0.op.{}".format(n, k),
                )
    i = 0
    for b in UNET_MAP_ATTENTIONS:
        _map_put("mid_block.attentions.{}.{}".format(i, b), "middle_block.1.{}".format(b))
    if transformers_mid:
        for t in range(transformers_mid):
            for b in TRANSFORMER_BLOCKS:
                _map_put(
                    "mid_block.attentions.{}.transformer_blocks.{}.{}".format(i, t, b),
                    "middle_block.1.transformer_blocks.{}.{}".format(t, b),
                )
    for i, n in enumerate([0, 2]):
        for b in UNET_MAP_RESNET:
            _map_put(
                "mid_block.resnets.{}.{}".format(i, UNET_MAP_RESNET[b]),
                "middle_block.{}.{}".format(n, b),
            )
    num_res_blocks_rev = list(reversed(num_res_blocks))
    for x in range(num_blocks):
        n = (num_res_blocks_rev[x] + 1) * x
        l = num_res_blocks_rev[x] + 1
        for i in range(l):
            c = 0
            for b in UNET_MAP_RESNET:
                _map_put(
                    "up_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b]),
                    "output_blocks.{}.0.{}".format(n, b),
                )
            c += 1
            if output_transformer_counts is not None: num_transformers = output_transformer_counts.get(n, 0)
            else: num_transformers = transformer_depth_output.pop() if transformer_depth_output else 0
            if num_transformers > 0:
                c += 1
                for b in UNET_MAP_ATTENTIONS:
                    _map_put(
                        "up_blocks.{}.attentions.{}.{}".format(x, i, b),
                        "output_blocks.{}.1.{}".format(n, b),
                    )
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        _map_put(
                            "up_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b),
                            "output_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b),
                        )
            # Upsample: only if this checkpoint has that Comfy conv (presence).
            # Missing tensor → no Diffusers entry.
            if i == l - 1:
                for k in ["weight", "bias"]:
                    _map_put(
                        "up_blocks.{}.upsamplers.0.conv.{}".format(x, k),
                        "output_blocks.{}.{}.conv.{}".format(n, c, k),
                    )
            n += 1
    for k, v in UNET_MAP_BASIC:
        _map_put(v, k)
    for _dk, _ck in diffusers_unet_map.items():
        if not _comfy_present(_ck):
            raise RuntimeError(
                f"Map integrity FATAL: mapped Comfy key {_ck!r} absent in checkpoint"
            )
    comfyui_to_diffusers_map = {v: k for k, v in diffusers_unet_map.items()}
    comfyui_to_diffusers_map = {f"{key_prefix}{k}": v for k, v in comfyui_to_diffusers_map.items()}

    return comfyui_to_diffusers_map


def load_unet_from_safetensors(path, device="cuda"):
    if str(device).startswith("cpu"):
        raise RuntimeError(
            "load_unet_from_safetensors refused device='cpu'. "
            "SDXL NVFP4 DualMonitor calibration requires CUDA."
        )
    print(f"Loading model: {path}")
    state_dict = load_file(path)
    print("Detecting UNet structure...")
    unet_config = detect_unet_config_from_keys(state_dict)
    print(f"Detected UNet config: {unet_config}")
    print("Initializing Diffusers pipeline...")
    try:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(device)
    except Exception as e:
        print(f"Warning: failed to load pretrained model: {e}")
        from diffusers import UNet2DConditionModel
        unet = UNet2DConditionModel(sample_size=128, in_channels=4, out_channels=4, layers_per_block=2, block_out_channels=(320, 640, 1280), down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"), up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"))
        pipeline = StableDiffusionXLPipeline(vae=None, text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None, unet=unet, scheduler=None)
        pipeline = pipeline.to(device)
    # Guard against silent CPU placement (diffusers warns then calib hangs at 0/25).
    try:
        unet_dev = str(next(pipeline.unet.parameters()).device)
    except StopIteration:
        unet_dev = "unknown"
    if not unet_dev.startswith("cuda"):
        raise RuntimeError(
            f"UNet landed on {unet_dev!r}, not CUDA. "
            "Refusing to start DualMonitor calibration (would hang at 0/25 on CPU fp16)."
        )
    print(f"  [Pipeline] UNet device={unet_dev}")
    print("Building key mapping...")
    comfyui_to_diffusers_map = unet_to_diffusers_mapping(unet_config, state_dict)
    print("Loading UNet weights...")
    new_state_dict = {}
    for comfy_key, diffusers_key in comfyui_to_diffusers_map.items():
        if comfy_key in state_dict: new_state_dict[diffusers_key] = state_dict[comfy_key]
    m, u = pipeline.unet.load_state_dict(new_state_dict, strict=False)
    return pipeline, state_dict, comfyui_to_diffusers_map


def _nvfp4_input_scale_from_amax(amax: float) -> torch.Tensor:
    """Kitchen TensorCoreNVFP4Layout.quantize input_scale formula (scalar f32)."""
    from comfy_kitchen.float_utils import F4_E2M1_MAX, F8_E4M3_MAX

    denom = float(F8_E4M3_MAX) * float(F4_E2M1_MAX)
    return torch.tensor(max(float(amax), 1e-12) / denom, dtype=torch.float32)


def _rotate_act_last_dim(
    x: torch.Tensor, h_matrix: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Hadamard rotate last dim in groups (matches inference rotate_last_dim)."""
    *lead, last = x.shape
    if last % group_size != 0:
        raise ValueError(
            f"last dim {last} not divisible by group_size={group_size}"
        )
    y = x.reshape(*lead, last // group_size, group_size).to(dtype=torch.float32)
    h = h_matrix.to(device=y.device, dtype=torch.float32)
    y = torch.matmul(y, h)
    return y.reshape(*lead, last)


def _load_native_convert_int8():
    """Load sibling native_convert_int8.py for Hadamard / rotate_weight."""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "native_convert_int8.py"
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"native_convert_int8.py not found: {path}")
    name = "native_convert_int8_for_nvfp4_convrot"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _encode_comfy_quant(config: dict) -> torch.Tensor:
    return torch.tensor(
        list(json.dumps(config, separators=(",", ":")).encode("utf-8")),
        dtype=torch.uint8,
    )


def _get_nvfp4_layout():
    """comfy_kitchen TensorCoreNVFP4Layout (venv package)."""
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout

    return TensorCoreNVFP4Layout


def pack_nvfp4(weight: torch.Tensor):
    """NVFP4 pack: uint8 qdata + Params (scale, block_scale, orig_shape).

    Auto-pads to 16x16 when needed (layout get_padded_shape).
    Kitchen TensorCoreNVFP4Layout accepts FP16/BF16 only (not float32).
    """
    if weight.ndim != 2:
        raise ValueError(f"NVFP4 pack expects 2D weight, got ndim={weight.ndim}")
    layout = _get_nvfp4_layout()
    if weight.dtype == torch.bfloat16:
        w_pack = weight.detach().to(dtype=torch.bfloat16)
    elif weight.dtype == torch.float16:
        w_pack = weight.detach().to(dtype=torch.float16)
    else:
        w_pack = weight.detach().float().to(dtype=torch.float16)
    qdata, params = layout.quantize(w_pack)
    return qdata, params


def dequant_nvfp4(qdata: torch.Tensor, params) -> torch.Tensor:
    """Dequantize NVFP4 storage back to float (sliced to orig_shape)."""
    layout = _get_nvfp4_layout()
    full = layout.dequantize(qdata, params)
    orig = tuple(params.orig_shape)
    if tuple(full.shape) != orig:
        slices = tuple(slice(0, s) for s in orig)
        return full[slices]
    return full


def can_pack_nvfp4(weight: torch.Tensor) -> bool:
    """NVFP4 requires a 2D Linear weight (padding handles 16-align)."""
    return weight.ndim == 2 and weight.shape[0] > 0 and weight.shape[1] > 0


def pack_channelwise_int8(weight: torch.Tensor):
    """Per-out-channel INT8 (Conv2d FULL ConvRot path; NVFP4 is 2D-only)."""
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
        raise ValueError(f"unsupported weight ndim={w.dim()} for INT8 channel pack")
    clamped = torch.clamp(w, -amax_view, amax_view)
    q = (clamped / scale_view).round().clamp(-127, 127).to(torch.int8)
    return q, scale_view.to(dtype=torch.float32)


def dequant_channelwise_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.float()


def _compute_weight_amax_dict(
    *,
    model: torch.nn.Module,
    dual_monitors: dict,
    keep_layers: set[str],
    device: str,
    alpha: float,
    enable_convrot: bool,
    group_size: int,
    build_hadamard,
    convrot_group_size_for_features,
    rotate_weight,
    rotate_weight_conv2d,
) -> dict[str, float]:
    """Per-layer pack clip amax = absmax after optional ConvRot (V3.0 parity).

    V3.0 INT8 stores absmax only; V4 pack MSE is for FP16 ranking, not pack
    scale. Same here: Linear/Conv pack amax = weight absmax (rotated if
    ConvRot ON). dual_monitors / alpha unused for amax (kept for call parity).
    """
    del dual_monitors, alpha  # ranking-only; amax is absmax
    weight_amax_dict: dict[str, float] = {}
    hadamard_cache: dict[int, torch.Tensor] = {}

    print(
        "\n[HSWQ] Weight clip amax = absmax "
        "(V3.0 parity; Linear→NVFP4 / Conv2d→INT8 pack; no V4 amax search)..."
    )
    for name, module in tqdm(model.named_modules(), desc="HSWQ amax"):
        if not isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            continue
        if name in keep_layers:
            continue
        w = module.weight.detach().float()
        if w.ndim not in (2, 4):
            continue

        in_f = int(w.shape[1])
        used_gs = None
        if (
            enable_convrot
            and convrot_group_size_for_features is not None
            and build_hadamard is not None
        ):
            used_gs = convrot_group_size_for_features(in_f, group_size)
        do_rotate = (
            enable_convrot
            and used_gs is not None
            and build_hadamard is not None
            and (
                (w.ndim == 2 and rotate_weight is not None)
                or (w.ndim == 4 and rotate_weight_conv2d is not None)
            )
        )
        if do_rotate:
            h = hadamard_cache.get(int(used_gs))
            if h is None:
                h = build_hadamard(int(used_gs), device="cpu", dtype=torch.float32)
                hadamard_cache[int(used_gs)] = h
            if w.ndim == 2:
                w = rotate_weight(w.cpu(), h, int(used_gs))
            else:
                w = rotate_weight_conv2d(w.cpu(), h, int(used_gs))

        absmax = float(w.abs().max().clamp_min(1e-6).item())
        weight_amax_dict[name] = absmax
        print(f"  [HSWQ-NVFP4] {name:50} | pack absmax={absmax:.4f}")

    print(f"  [HSWQ] weight amax layers={len(weight_amax_dict)}")
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return weight_amax_dict


def run_nvfp4_calib(
    *,
    input_path: str,
    calib_file: str,
    num_calib_samples: int,
    num_inference_steps: int,
    device: str,
    enable_convrot: bool = True,
    group_size: int = _DEFAULT_GROUPSIZE,
    keep_ratio: float = 0.0,
    profile_arg: str | None = None,
    fp16_budget_mb: float | None = None,
):
    """PTQ calib: DualMonitor + input_scale amax + V3.0-parity FP16 protect.

    FP16 protect (r0): Hard VETO cascade → DualMonitor → V4 NVFP4 candidates
    → grayzone → _apply_fp16_budget_cap (budget from --fp16_budget_mb /
    --budget_mb). Pack amax stays absmax via weighted_histogram_mse_v4_nvfp4
    (no fast search).
    """
    if abs(float(keep_ratio)) > 1e-12:
        raise ValueError(
            f"keep_ratio must be 0 (r0); got {keep_ratio}. "
            f"FP16 protect = DualMonitor + analyze + V4 NVFP4 MSE inside "
            f"--fp16_budget_mb / --budget_mb (current={float(FP16_BUDGET_MB_HARD):g} MiB)."
        )
    budget_mb = _bind_fp16_budget_from_option(
        float(FP16_BUDGET_MB_HARD if fp16_budget_mb is None else fp16_budget_mb)
    )

    build_hadamard = None
    convrot_group_size_for_features = None
    rotate_weight = None
    rotate_weight_conv2d = None
    hadamard_cache: dict[int, torch.Tensor] = {}
    if enable_convrot:
        nc = _load_native_convert_int8()
        build_hadamard = nc.build_hadamard
        convrot_group_size_for_features = nc.convrot_group_size_for_features
        rotate_weight = nc.rotate_weight
        rotate_weight_conv2d = nc.rotate_weight_conv2d

    profile_path = _ensure_distribution_profile(
        input_path=input_path,
        profile_arg=profile_arg,
    )
    model_profile, profile_summary, _veto_blob = _load_distribution_profile_layers(
        profile_path
    )

    pipeline, _state_dict, comfyui_to_diffusers_map = load_unet_from_safetensors(
        input_path, device
    )
    model_profile = _remap_profile_to_diffusers(
        model_profile, comfyui_to_diffusers_map
    )
    model = pipeline.unet
    _norm_profile = {k: v for k, v in model_profile.items() if isinstance(v, dict)}
    veto_tunables = resolve_veto_tunables(
        _norm_profile,
        profile_summary,
        dual_monitors=None,
        fp16_budget_mb=budget_mb,
    )
    print("  [Veto Tunables NVFP4 protect — full as_dict via repr]")
    _vt = veto_tunables.as_dict()
    for _k in sorted(_vt.keys(), key=str):
        print(f"    veto_tunables.{_k} = {_vt[_k]!r}")

    alpha, beta, get_layer_search_low, hard_veto_layers = derive_hswq_strategy_nvfp4(
        model_profile,
        veto_tunables,
    )

    print(
        "  [V3.0-parity NVFP4 Autonomous VETO] Structural + per-projection "
        "attn + key-pattern + supplemental."
    )
    structural_veto = _compute_structural_veto(
        model, hard_veto_layers, _norm_profile
    )
    if structural_veto:
        hard_veto_layers = hard_veto_layers.union(structural_veto)
        print(
            f"  [Structural VETO] Added {len(structural_veto)} unique-shape "
            f"layers (total VETO: {len(hard_veto_layers)})."
        )
    proj_veto = _compute_sdxl_per_projection_attn_veto(
        model,
        hard_veto_layers,
        veto_tunables,
        _norm_profile,
    )
    if proj_veto:
        hard_veto_layers = hard_veto_layers.union(proj_veto)
        print(
            f"  [Per-Projection VETO] Added {len(proj_veto)} attn layers "
            f"(total VETO: {len(hard_veto_layers)})."
        )
    mad_veto = _compute_sdxl_nvfp4_mad_attn_veto(
        model, hard_veto_layers, veto_tunables, _norm_profile
    )
    if mad_veto:
        hard_veto_layers = hard_veto_layers.union(mad_veto)
        print(
            f"  [NVFP4 MAD VETO] total VETO after MAD fill: {len(hard_veto_layers)}."
        )
    keypattern_veto = _compute_sdxl_keypattern_veto(
        model, hard_veto_layers, veto_tunables, _norm_profile
    )
    if keypattern_veto:
        hard_veto_layers = hard_veto_layers.union(keypattern_veto)
        print(
            f"  [Key-Pattern VETO] hard_veto total: {len(hard_veto_layers)}."
        )

    print(
        "Preparing calibration (DualMonitor + NVFP4 input_scale amax "
        "+ FP16 budget protect + pack-roundtrip weight amax)..."
    )
    if enable_convrot:
        print(
            "  [input_scale] ConvRot Linear: amax after Hadamard rotate_last_dim "
            f"(preferred groupsize={group_size})"
        )
    else:
        print("  [input_scale] amax on unrotated activations (--no-convrot)")

    dual_monitors.clear()
    act_amax_dict: dict[str, float] = {}
    handles = []
    target_modules = []

    def _make_hook(name: str):
        def hook(m, inp, out):
            hook_fn(m, inp, out, name)
            if not inp or inp[0] is None:
                return
            x = inp[0]
            if not torch.is_tensor(x) or not torch.is_floating_point(x):
                return
            x_f = x.detach().float()
            x_for_amax = x_f
            if (
                isinstance(m, torch.nn.Linear)
                and enable_convrot
                and convrot_group_size_for_features is not None
                and build_hadamard is not None
            ):
                in_f = int(m.in_features)
                gs = convrot_group_size_for_features(in_f, group_size)
                if gs is not None and int(x_f.shape[-1]) == in_f:
                    h = hadamard_cache.get(int(gs))
                    if h is None:
                        h = build_hadamard(
                            int(gs), device=x_f.device, dtype=torch.float32
                        )
                        hadamard_cache[int(gs)] = h
                    elif h.device != x_f.device:
                        h = h.to(device=x_f.device)
                        hadamard_cache[int(gs)] = h
                    flat = x_f.reshape(-1, in_f)
                    x_for_amax = _rotate_act_last_dim(flat, h, int(gs))
            amax = float(x_for_amax.abs().amax().clamp_min(1e-12).item())
            prev = act_amax_dict.get(name)
            if prev is None or amax > prev:
                act_amax_dict[name] = amax

        return hook

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(module.register_forward_hook(_make_hook(name)))
            target_modules.append(name)

    print("Preparing calibration data...")
    with open(calib_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    if len(prompts) < num_calib_samples:
        prompts = (prompts * (num_calib_samples // len(prompts) + 1))[
            :num_calib_samples
        ]
    else:
        prompts = prompts[:num_calib_samples]

    print(
        f"Running calibration ({num_calib_samples} samples, "
        f"{num_inference_steps} steps)..."
    )
    if num_calib_samples != 32 or num_inference_steps != 25:
        print(
            "  [WARN] How-to / r32 recipe is num_calib_samples=32, "
            "num_inference_steps=25. DualMonitor importance for V4 FP16 "
            "ranking should follow that calibration; current args differ."
        )
    pipeline.set_progress_bar_config(disable=False)
    generator = torch.Generator(device=device).manual_seed(42)

    for i, prompt in enumerate(prompts):
        print(f"\nSample {i+1}/{num_calib_samples}: {prompt[:50]}...")
        with torch.no_grad():
            pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                output_type="latent",
                generator=generator,
            )
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    for h in handles:
        h.remove()

    print("  [Calib] DualMonitor Importance ready for V4 full-pool priority.")

    act_mean_dict = {}
    for name, mon in dual_monitors.items():
        if mon.channel_act_mean is not None:
            act_mean_dict[name] = mon.channel_act_mean.detach().float().cpu()
    print(
        f"  [Card 1 DualMonitor] act_mean layers={len(act_mean_dict)} "
        f"(full Card 1; no VETO; no Approach A)"
    )
    print(
        f"  [input_scale] act_amax layers={len(act_amax_dict)} "
        f"(running abs max over calib)"
    )

    print(
        "\nAnalyzing layer sensitivity [NVFP4] — V4 calib FP16 cands + "
        "analyze VETO..."
    )
    _supp = _autonomous_supplemental_veto(
        model, hard_veto_layers, _norm_profile, veto_tunables
    )
    if _supp:
        hard_veto_layers = hard_veto_layers.union(_supp)
        print(
            f"  [Supplemental VETO] Added {len(_supp)} layers "
            f"(total VETO: {len(hard_veto_layers)})."
        )

    veto_tunables = resolve_veto_tunables(
        _norm_profile,
        profile_summary,
        dual_monitors=dual_monitors,
        fp16_budget_mb=budget_mb,
    )
    alpha = float(veto_tunables.alpha_auto)
    if _norm_profile and alpha <= 0.0:
        raise ValueError(
            "NVFP4 Full-SVD×RMS alpha_auto must be > 0 after DualMonitor resolve "
            f"(alpha==0 is SVD cut / rebellion). got alpha_auto={alpha}"
        )
    beta = 1.0 - alpha
    print(
        f"  [Dynamic Alpha/Beta NVFP4 after DualMonitor] "
        f"alpha={alpha!r}, beta={beta!r} "
        f"(analyze k∪o∪m → Full-SVD×RMS mix into ranking; "
        f"Imp×Sens×V4 MSE fill {budget_mb:g} MiB)"
    )
    print(
        "  [HSWQ SVD SETTINGS LOCK] "
        f"alpha={alpha!r} beta={beta!r} | "
        "every Linear/Conv V4 measure will emit [HSWQ SVD MIX FULL] "
        "(settings + all singular values + alpha*leverage vs beta*magnitude proof); "
        "mid-stop / shape>100 gate is removed"
    )

    mse_cache: dict = {}
    dynamic_keep_layers, mse_cache = _build_v4_calib_fp16_candidates(
        model=model,
        dual_monitors=dual_monitors,
        target_modules=target_modules,
        hard_veto_layers=hard_veto_layers,
        mse_cache=mse_cache,
        alpha=alpha,
        beta=beta,
        device=device,
    )
    keep_layers = dynamic_keep_layers.union(hard_veto_layers)

    release_cands = _collect_mse_release_candidates(
        hard_veto_layers, structural_veto, _norm_profile, model, veto_tunables
    )
    if keypattern_veto:
        release_cands -= keypattern_veto
    if release_cands:
        hard_veto_layers, keep_layers, mse_cache = _mse_grayzone_veto_reassessment(
            scope_label="V1.0 SDXL NVFP4 ConvRot",
            hard_veto_layers=hard_veto_layers,
            keep_layers=keep_layers,
            outlier_only_veto=release_cands,
            target_modules=target_modules,
            model=model,
            _norm_profile=_norm_profile,
            get_layer_search_low=get_layer_search_low,
            alpha=alpha,
            beta=beta,
            device=device,
            tunables=veto_tunables,
            dual_monitors=dual_monitors,
            mse_cache=mse_cache,
        )

    mapped_weight_modules = set()
    for dk in comfyui_to_diffusers_map.values():
        if isinstance(dk, str) and dk.endswith(".weight"):
            mapped_weight_modules.add(dk[:-7])
    for _name, _mod in model.named_modules():
        if not _name.endswith("upsamplers.0.conv"):
            continue
        if not isinstance(_mod, torch.nn.Conv2d):
            continue
        if _name not in mapped_weight_modules:
            raise RuntimeError(
                f"Map integrity FATAL: Diffusers module {_name!r} exists on UNet "
                f"but has no Comfy map entry — fix unet_to_diffusers_mapping "
                f"(refuse invent / refuse leave unmapped)"
            )
        print(f"  [Map integrity] {_name} mapped")
    orphan_before = keep_layers - mapped_weight_modules
    if orphan_before:
        print(
            f"  [Map integrity] FATAL: {len(orphan_before)} keep name(s) not in "
            f"Comfy↔diffusers map (must be 0 before budget):"
        )
        for n in sorted(orphan_before):
            print(f"    unmapped: {n}")
        raise RuntimeError(
            f"Map integrity: {len(orphan_before)} unmapped keep layer(s); "
            f"refuse exclude — fix unet_to_diffusers_mapping"
        )

    keep_layers, hard_veto_layers, budget_stats, int8_shelter_layers = (
        _apply_fp16_budget_cap(
            model,
            keep_layers,
            hard_veto_layers,
            budget_mb=budget_mb,
            norm_profile=_norm_profile,
            veto_tunables=veto_tunables,
            dual_monitors=dual_monitors,
            mse_cache=mse_cache,
            alpha=alpha,
            beta=beta,
            device=device,
        )
    )
    dynamic_keep_layers = dynamic_keep_layers & keep_layers

    orphan_keep = keep_layers - mapped_weight_modules
    if orphan_keep:
        print(
            f"  [FP16 keep] FATAL map mismatch: {len(orphan_keep)} keep name(s) "
            f"still unmapped after budget (must be 0; will not drop):"
        )
        for n in sorted(orphan_keep):
            print(f"    unmapped: {n}")
        raise RuntimeError(
            f"Map integrity after budget: {len(orphan_keep)} unmapped keep "
            f"layer(s); refuse exclude"
        )

    orphan_shelter = int8_shelter_layers - mapped_weight_modules
    if orphan_shelter:
        print(
            f"  [INT8 shelter] FATAL map mismatch: {len(orphan_shelter)} "
            f"shelter name(s) still unmapped after budget (must be 0; "
            f"will not drop):"
        )
        for n in sorted(orphan_shelter):
            print(f"    unmapped: {n}")
        raise RuntimeError(
            f"Map integrity after budget: {len(orphan_shelter)} unmapped "
            f"INT8 shelter layer(s); refuse exclude"
        )

    layer_sensitivities = {}
    for name in keep_layers | set(target_modules):
        mon = dual_monitors.get(name)
        if mon is None:
            continue
        try:
            layer_sensitivities[name] = float(mon.get_sensitivity())
        except Exception:
            pass

    weight_amax_dict = _compute_weight_amax_dict(
        model=model,
        dual_monitors=dual_monitors,
        keep_layers=keep_layers,
        device=device,
        alpha=float(alpha),
        enable_convrot=bool(enable_convrot),
        group_size=int(group_size),
        build_hadamard=build_hadamard,
        convrot_group_size_for_features=convrot_group_size_for_features,
        rotate_weight=rotate_weight,
        rotate_weight_conv2d=rotate_weight_conv2d,
    )

    del pipeline
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "act_mean_dict": act_mean_dict,
        "act_amax_dict": act_amax_dict,
        "comfyui_to_diffusers_map": comfyui_to_diffusers_map,
        "keep_layers": keep_layers,
        "int8_shelter_layers": int8_shelter_layers,
        "weight_amax_dict": weight_amax_dict,
        "layer_sensitivities": layer_sensitivities,
        "budget_stats": budget_stats,
        "fp16_budget_mb": budget_mb,
        "dynamic_keep_layers": dynamic_keep_layers,
    }



def convert_to_nvfp4_convrot(
    input_path,
    output_path,
    bias_correction: bool = False,
    calib_file: str | None = None,
    num_calib_samples: int = 32,
    num_inference_steps: int = 25,
    enable_convrot: bool = True,
    group_size: int = _DEFAULT_GROUPSIZE,
    min_in_features: int = 0,
    keep_ratio: float = 0.0,
    profile_arg: str | None = None,
    fp16_budget_mb: float | None = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    act_mean_dict = {}
    act_amax_dict: dict[str, float] = {}
    weight_amax_dict: dict[str, float] = {}
    keep_layers: set[str] = set()
    int8_shelter_layers: set[str] = set()
    comfyui_to_diffusers_map = {}
    compute_int8_bias_delta = None
    rotate_weight = None
    rotate_weight_conv2d = None
    convrot_group_size_for_features = None
    build_hadamard = None
    convrot_nvfp4 = 0
    plain_nvfp4 = 0
    convrot_int8_conv2d = 0
    plain_int8_conv2d = 0
    convrot_int8_linear = 0
    plain_int8_linear = 0
    skipped_small = 0
    fp16_kept_count = 0
    weight_clamp_count = 0
    input_scale_written = 0
    input_scale_missing = 0
    write_input_scale = False

    if abs(float(keep_ratio)) > 1e-12:
        raise ValueError(
            f"keep_ratio must be 0 (r0); got {keep_ratio}. "
            "FP16 protect uses --fp16_budget_mb only."
        )
    keep_ratio = 0.0
    if fp16_budget_mb is None:
        fp16_budget_mb = FP16_BUDGET_MB_HARD
    fp16_budget_mb = _bind_fp16_budget_from_option(float(fp16_budget_mb))

    if enable_convrot:
        nc = _load_native_convert_int8()
        rotate_weight = nc.rotate_weight
        rotate_weight_conv2d = nc.rotate_weight_conv2d
        convrot_group_size_for_features = nc.convrot_group_size_for_features
        build_hadamard = nc.build_hadamard
        print(
            f"  [FULL ConvRot] ON | preferred groupsize={group_size}; "
            f"min_in_features={min_in_features}"
        )
        print(
            "  [FULL ConvRot] Linear → offline Hadamard + NVFP4 when group OK; "
            "else plain NVFP4. Conv2d → offline Hadamard + INT8 when group OK; "
            "else plain INT8. Online act rotate required at load (loader later)."
        )
        if bias_correction:
            print(
                "  [ConvRot] WARN: Card 1 DualMonitor means are from unrotated float UNet; "
                "BC uses rotated W vs W_q (approximate for ConvRot)"
            )
    else:
        print(
            "  [FULL ConvRot] OFF | plain NVFP4 on Linear, plain INT8 on Conv2d "
            "(no offline rotate)"
        )

    if calib_file:
        if not os.path.isfile(calib_file):
            raise FileNotFoundError(f"calib_file not found: {calib_file}")
        write_input_scale = True
        if bias_correction:
            print(
                "  [Bias Correction Card 1] ON | quantized Linear | "
                "DualMonitor calib | bias += -(W_q - W) @ mu_x"
            )
        print(
            "  [input_scale] ON | write NVFP4 Linear "
            "amax/(F8_E4M3_MAX*F4_E2M1_MAX) from same calib pass"
        )
        print(
            f"  [HSWQ] r0 DualMonitor + packed NVFP4/INT8 budget "
            f"fp16_budget_mb={float(fp16_budget_mb):g} "
            "+ NVFP4/INT8 pack-roundtrip weight clip amax"
        )
        calib = run_nvfp4_calib(
            input_path=input_path,
            calib_file=calib_file,
            num_calib_samples=int(num_calib_samples),
            num_inference_steps=int(num_inference_steps),
            device=device,
            enable_convrot=bool(enable_convrot),
            group_size=int(group_size),
            keep_ratio=float(keep_ratio),
            profile_arg=profile_arg,
            fp16_budget_mb=fp16_budget_mb,
        )
        act_mean_dict = calib["act_mean_dict"]
        act_amax_dict = calib["act_amax_dict"]
        weight_amax_dict = calib["weight_amax_dict"]
        keep_layers = calib["keep_layers"]
        int8_shelter_layers = calib.get("int8_shelter_layers", set())
        comfyui_to_diffusers_map = calib["comfyui_to_diffusers_map"]
        if bias_correction:
            compute_int8_bias_delta = globals()["compute_int8_bias_delta"]
            print(
                f"  [Bias Correction] Captured act means for {len(act_mean_dict)} layers"
            )
        print(
            f"  [input_scale] Captured act amax for {len(act_amax_dict)} layers"
        )
        print(
            f"  [HSWQ] weight amax for {len(weight_amax_dict)} layers; "
            f"FP16 keep={len(keep_layers)}; "
            f"INT8 shelter={len(int8_shelter_layers)}"
        )
    elif bias_correction:
        raise ValueError(
            "--bias_correction requires --calib_file "
            "(same as quantize_sdxl_hswq_v3.0.py)"
        )
    else:
        print(
            "  [WARN] No --calib_file: NVFP4 Linear will have NO .input_scale keys. "
            "Inference falls back to ones(1) and quality collapses. "
            "Pass --calib_file to write correct scales into the ckpt. "
            "HSWQ keep_ratio / pack-roundtrip weight amax also require --calib_file."
        )

    if compute_int8_bias_delta is None:

        def compute_int8_bias_delta(weight_fp, weight_dq, act_mean):
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

    print(f"Loading model: {input_path}")
    state_dict = load_file(input_path)

    new_state_dict = {}
    quant_meta_layers = {}
    converted_count = 0
    skipped_count = 0
    bias_corr_pending: dict[str, torch.Tensor] = {}
    bias_corr_applied = 0
    bias_corr_skipped_no_bias = 0
    bias_corr_skipped_no_act = 0
    bias_corr_skipped_bad_shape = 0
    rot_tag = " + FULL ConvRot" if enable_convrot else " plain NVFP4/INT8"
    print(f"Converting diffusion Linear/Conv2d weights ({rot_tag.strip()})...")

    for key, tensor in tqdm(state_dict.items()):
        is_unet_matmul_weight = (
            key.startswith("model.diffusion_model")
            and key.endswith(".weight")
            and tensor.ndim >= 2
        )
        if is_unet_matmul_weight and tensor.dtype in [
            torch.float16,
            torch.float32,
            torch.bfloat16,
        ]:
            if tensor.ndim not in (2, 4):
                new_state_dict[key] = tensor
                skipped_count += 1
                continue

            in_f = int(tensor.shape[1])
            if min_in_features > 0 and in_f < int(min_in_features):
                new_state_dict[key] = tensor
                skipped_small += 1
                skipped_count += 1
                continue

            diffusers_key = comfyui_to_diffusers_map.get(key)
            module_name = None
            if diffusers_key and diffusers_key.endswith(".weight"):
                module_name = diffusers_key[:-7]

            if module_name is not None and module_name in keep_layers:
                new_state_dict[key] = tensor
                fp16_kept_count += 1
                skipped_count += 1
                continue

            w_fp = tensor.float()
            used_gs = None
            if (
                enable_convrot
                and convrot_group_size_for_features is not None
                and build_hadamard is not None
            ):
                used_gs = convrot_group_size_for_features(in_f, group_size)

            do_rotate = (
                enable_convrot
                and used_gs is not None
                and build_hadamard is not None
                and (
                    (tensor.ndim == 2 and rotate_weight is not None)
                    or (tensor.ndim == 4 and rotate_weight_conv2d is not None)
                )
            )
            if do_rotate:
                h_matrix = build_hadamard(int(used_gs), device="cpu", dtype=torch.float32)
                if tensor.ndim == 2:
                    w_fp = rotate_weight(w_fp, h_matrix, int(used_gs))
                else:
                    w_fp = rotate_weight_conv2d(w_fp, h_matrix, int(used_gs))

            if module_name is not None and module_name in weight_amax_dict:
                amax_w = float(weight_amax_dict[module_name])
                w_fp = w_fp.clamp(-amax_w, amax_w)
                weight_clamp_count += 1

            module_key = key[: -len(".weight")]

            if tensor.ndim == 2:
                if module_name is not None and module_name in int8_shelter_layers:
                    # Multi-tier INT8 shelter: V3.1-compatible per-out-channel
                    # INT8 (+ FULL ConvRot when OK). No .input_scale — the
                    # V3.1 INT8 path does not use one.
                    q, scale = pack_channelwise_int8(w_fp)
                    weight_dq = dequant_channelwise_int8(q, scale)
                    if do_rotate:
                        quant_config = {
                            "format": "int8_tensorwise",
                            "convrot": True,
                            "convrot_groupsize": int(used_gs),
                        }
                        convrot_int8_linear += 1
                    else:
                        quant_config = {"format": "int8_tensorwise"}
                        plain_int8_linear += 1
                    new_state_dict[key] = q
                    new_state_dict[f"{module_key}.weight_scale"] = scale
                elif not can_pack_nvfp4(tensor):
                    new_state_dict[key] = tensor
                    skipped_count += 1
                    continue
                else:
                    q, params = pack_nvfp4(w_fp)
                    weight_dq = dequant_nvfp4(q, params)
                    if do_rotate:
                        quant_config = {
                            "format": "nvfp4",
                            "convrot": True,
                            "convrot_groupsize": int(used_gs),
                        }
                        convrot_nvfp4 += 1
                    else:
                        quant_config = {"format": "nvfp4"}
                        plain_nvfp4 += 1
                    new_state_dict[key] = q
                    new_state_dict[f"{module_key}.weight_scale"] = params.block_scale
                    new_state_dict[f"{module_key}.weight_scale_2"] = params.scale.to(
                        dtype=torch.float32
                    ).reshape(())
                    if write_input_scale:
                        amax = (
                            act_amax_dict.get(module_name)
                            if module_name is not None
                            else None
                        )
                        if amax is None:
                            input_scale_missing += 1
                        else:
                            new_state_dict[f"{module_key}.input_scale"] = (
                                _nvfp4_input_scale_from_amax(amax)
                            )
                            input_scale_written += 1
            else:
                # Conv2d: NVFP4 is 2D-only → INT8 channelwise (+ FULL ConvRot when OK)
                q, scale = pack_channelwise_int8(w_fp)
                weight_dq = dequant_channelwise_int8(q, scale)
                if do_rotate:
                    quant_config = {
                        "format": "int8_tensorwise",
                        "convrot": True,
                        "convrot_groupsize": int(used_gs),
                    }
                    convrot_int8_conv2d += 1
                else:
                    quant_config = {"format": "int8_tensorwise"}
                    plain_int8_conv2d += 1
                new_state_dict[key] = q
                new_state_dict[f"{module_key}.weight_scale"] = scale

            new_state_dict[f"{module_key}.comfy_quant"] = _encode_comfy_quant(
                quant_config
            )
            quant_meta_layers[module_key] = dict(quant_config)
            converted_count += 1

            if bias_correction:
                act_mean = (
                    act_mean_dict.get(module_name)
                    if module_name is not None
                    else None
                )
                if act_mean is None:
                    bias_corr_skipped_no_act += 1
                else:
                    delta = compute_int8_bias_delta(w_fp, weight_dq, act_mean)
                    if delta is None:
                        bias_corr_skipped_bad_shape += 1
                    else:
                        bias_corr_pending[module_key] = (
                            (-delta).detach().float().cpu()
                        )
        else:
            new_state_dict[key] = tensor
            skipped_count += 1

    if bias_correction and bias_corr_pending:
        print(
            f"\n[Bias Correction] Applying deltas to {len(bias_corr_pending)} "
            f"quantized Linear/Conv layers..."
        )
        for module_key, delta in bias_corr_pending.items():
            bias_key = f"{module_key}.bias"
            if bias_key not in new_state_dict:
                bias_corr_skipped_no_bias += 1
                continue
            bias = new_state_dict[bias_key]
            corrected = bias.float() + delta.to(
                device=bias.device, dtype=torch.float32
            )
            new_state_dict[bias_key] = corrected.to(dtype=bias.dtype)
            bias_corr_applied += 1
        print(
            f"  [Bias Correction] applied={bias_corr_applied}, "
            f"no_bias={bias_corr_skipped_no_bias}, "
            f"no_act={bias_corr_skipped_no_act}, "
            f"bad_shape={bias_corr_skipped_bad_shape}"
        )
    elif bias_correction:
        print(
            f"  [Bias Correction] No deltas pending "
            f"(no_act={bias_corr_skipped_no_act}, "
            f"bad_shape={bias_corr_skipped_bad_shape})"
        )

    metadata = {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": quant_meta_layers}
        )
    }

    print(f"Saving to: {output_path}")
    print(f"Converted layers: {converted_count}, Kept layers: {skipped_count}")
    if fp16_kept_count:
        print(
            f"  HSWQ FP16 protect (r0 DualMonitor / "
            f"budget={float(fp16_budget_mb):g} MiB): "
            f"{fp16_kept_count}"
        )
    if weight_clamp_count:
        print(
            f"  HSWQ pack-roundtrip weight clamp: {weight_clamp_count}"
        )
    if skipped_small:
        print(f"  skipped (min_in_features={min_in_features}): {skipped_small}")
    print(f"Bias correction (Card 1): {bias_correction}")
    if bias_correction:
        print(f"  Bias-corrected layers: {bias_corr_applied}")
    print(f"input_scale written (NVFP4 Linear): {input_scale_written}")
    if write_input_scale and input_scale_missing:
        print(
            f"  [WARN] NVFP4 Linear missing act amax (no input_scale): "
            f"{input_scale_missing}"
        )
    elif not write_input_scale:
        print("  [WARN] input_scale skipped (no --calib_file)")
    print(f"FULL ConvRot enabled: {enable_convrot}")
    if enable_convrot:
        print(
            f"  NVFP4 ConvRot Linear: {convrot_nvfp4}, "
            f"plain NVFP4 Linear: {plain_nvfp4}, "
            f"INT8 shelter ConvRot Linear: {convrot_int8_linear}, "
            f"INT8 shelter plain Linear: {plain_int8_linear}, "
            f"INT8 ConvRot Conv2d: {convrot_int8_conv2d}, "
            f"plain INT8 Conv2d: {plain_int8_conv2d}"
        )
    else:
        print(
            f"  plain NVFP4 Linear: {plain_nvfp4}, "
            f"INT8 shelter plain Linear: {plain_int8_linear}, "
            f"plain INT8 Conv2d: {plain_int8_conv2d}"
        )

    # Hard assert: Linear+Conv FP16 keep + Linear INT8 shelter ≤ owner
    # ceiling + owner tolerance. Meter matches budget ranking: Linear FP16
    # +1.5×numel (vs NVFP4), Conv FP16 +1× (vs INT8), Linear INT8 shelter
    # +0.5×numel (vs NVFP4). Packed Linear is float8_e4m3fn_x2 / similar —
    # not float16.
    _budget_ceil_b = int(float(fp16_budget_mb) * 1024 * 1024)
    _tol_b = int(float(FP16_BUDGET_ASSERT_TOLERANCE_MIB) * 1024 * 1024)
    _pack_fp16_extra = 0
    _pack_fp16_n = 0
    _pack_fp16_linear_n = 0
    _pack_fp16_conv_n = 0
    _pack_fp16_skipped_non_lc = 0
    _pack_fp16_leak = []
    _pack_shelter_extra = 0
    _pack_shelter_n = 0
    if keep_layers or int8_shelter_layers:
        for _ck, _cv in new_state_dict.items():
            if not _ck.endswith(".weight"):
                continue
            _dk = comfyui_to_diffusers_map.get(_ck)
            if not (isinstance(_dk, str) and _dk.endswith(".weight")):
                continue
            _mod = _dk[:-7]
            if _cv.dtype == torch.int8:
                # INT8 shelter Linear counts +0.5 B/el vs NVFP4; Conv INT8
                # is the packed baseline (no extra).
                if int(_cv.ndim) == 2 and _mod in int8_shelter_layers:
                    _pack_shelter_extra += int(_cv.numel()) // 2
                    _pack_shelter_n += 1
                continue
            # Packed NVFP4 Linear is not FP16 keep.
            _dt = str(getattr(_cv.dtype, "name", _cv.dtype))
            if "float8" in _dt or "fp4" in _dt.lower():
                continue
            if _cv.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                continue
            _ndim = int(_cv.ndim)
            if _ndim not in (2, 4):
                _pack_fp16_skipped_non_lc += 1
                continue
            if _mod not in keep_layers:
                _pack_fp16_leak.append(_mod)
                continue
            _n_el = int(_cv.numel())
            if _ndim == 2:
                _pack_fp16_extra += (_n_el * 3) // 2
                _pack_fp16_linear_n += 1
            else:
                _pack_fp16_extra += _n_el
                _pack_fp16_conv_n += 1
            _pack_fp16_n += 1
        if _pack_fp16_leak:
            _show = ", ".join(_pack_fp16_leak[:12])
            raise RuntimeError(
                f"[FP16 budget] post-pack leak: {len(_pack_fp16_leak)} Linear/Conv "
                f"float weight(s) not in keep_layers (hand-waving pack path). "
                f"Examples: {_show}. Refusing to save."
            )
        if int8_shelter_layers and _pack_shelter_n != len(int8_shelter_layers):
            raise RuntimeError(
                f"[FP16 budget] post-pack INT8 shelter mismatch: "
                f"{_pack_shelter_n} INT8 Linear found in output vs "
                f"{len(int8_shelter_layers)} shelter layer(s) selected by the "
                f"multi-tier ladder. Refusing to save."
            )
        _pack_fp16_mb = _pack_fp16_extra / (1024 * 1024)
        _pack_shelter_mb = _pack_shelter_extra / (1024 * 1024)
        _pack_total_mb = (_pack_fp16_extra + _pack_shelter_extra) / (1024 * 1024)
        _over_b = (_pack_fp16_extra + _pack_shelter_extra) - _budget_ceil_b
        print(
            f"  [FP16 budget] post-pack extra vs packed "
            f"(Linear FP16 +1.5B/el vs NVFP4, Conv FP16 +1B/el vs INT8, "
            f"INT8 shelter +0.5B/el vs NVFP4): "
            f"FP16={_pack_fp16_mb:.2f} MiB ({_pack_fp16_n} modules; "
            f"Linear={_pack_fp16_linear_n} Conv={_pack_fp16_conv_n}; "
            f"skipped_non_LinearConv={_pack_fp16_skipped_non_lc}) + "
            f"INT8 shelter={_pack_shelter_mb:.2f} MiB "
            f"({_pack_shelter_n} Linear) = {_pack_total_mb:.2f} MiB / "
            f"ceiling={float(fp16_budget_mb):g} MiB "
            f"(assert tol={FP16_BUDGET_ASSERT_TOLERANCE_MIB:g} MiB)"
        )
        if _over_b > _tol_b:
            raise RuntimeError(
                f"[FP16 budget] post-pack assert FAILED: "
                f"FP16 keep + INT8 shelter {_pack_total_mb:.3f} MiB exceeds "
                f"{float(fp16_budget_mb):g} MiB hard ceiling "
                f"+ {FP16_BUDGET_ASSERT_TOLERANCE_MIB:g} MiB tolerance "
                f"({_pack_fp16_extra + _pack_shelter_extra} > "
                f"{_budget_ceil_b + _tol_b} bytes; "
                f"over_by={_over_b / (1024 * 1024):.3f} MiB; "
                f"FP16 Linear={_pack_fp16_linear_n} Conv={_pack_fp16_conv_n}; "
                f"INT8 shelter Linear={_pack_shelter_n}). "
                f"Refusing to save."
            )
        if _over_b > 0:
            print(
                f"  [FP16 budget] within owner tolerance: "
                f"+{_over_b / (1024 * 1024):.3f} MiB over ceiling "
                f"(allowed ≤ {FP16_BUDGET_ASSERT_TOLERANCE_MIB:g} MiB); saving."
            )

    save_file(new_state_dict, output_path, metadata=metadata)
    print("Done!")

    # Convert complete: drop holdings so a chained post-convert bench can use VRAM.
    del state_dict
    del new_state_dict
    del quant_meta_layers
    act_mean_dict.clear()
    del act_mean_dict
    act_amax_dict.clear()
    del act_amax_dict
    weight_amax_dict.clear()
    del weight_amax_dict
    keep_layers.clear()
    del keep_layers
    int8_shelter_layers.clear()
    del int8_shelter_layers
    bias_corr_pending.clear()
    del bias_corr_pending
    _release_vram_before_bench("after HSWQ NVFP4 convert save")


# Exact --prompt from the owner NVFP4 SDXL bench command (fixed; not a CLI).
_FIXED_NVFP4BENCH_PROMPT = (
    "masterpiece, best quality, 1girl, solo, standing, simple background"
)
# Seed fixed inside the chain (not a parent CLI).
_FIXED_NVFP4BENCH_SEED = 123456789


def _release_vram_before_bench(label: str = "post-convert") -> None:
    """Drop parent-process CUDA holdings before spawning the fidelity bench.

    Convert leaves large state_dict tensors alive until refs are deleted and
    the allocator cache is flushed. Without this clear, the chained bench
    child OOMs on the same GPU.
    """
    print(f"[*] Releasing VRAM ({label}) before post-bench...")
    gc.collect()
    if not torch.cuda.is_available():
        print(f"[*] VRAM clear ({label}): CUDA not available")
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        alloc_mib = torch.cuda.memory_allocated() / (1024 ** 2)
        reserv_mib = torch.cuda.memory_reserved() / (1024 ** 2)
        print(
            f"[*] VRAM clear ({label}): "
            f"allocated={alloc_mib:.1f} MiB reserved={reserv_mib:.1f} MiB"
        )
    except Exception:
        print(f"[*] VRAM clear ({label}): done")


def run_post_convert_nvfp4_bench(
    *,
    script_dir: str,
    fp16_path: str,
    nvfp4_path: str,
) -> int:
    """Owner NVFP4 bench shape + seed fixed inside this chain:

    nvfp4bench_sdxl.py --fp16 <path> --nvfp4 <path>
      --prompt "<fixed>" --seed <fixed>

    (No parent --bench_seed CLI. steps left to nvfp4bench default.)
    """
    bench_script = os.path.join(script_dir, "benchmark", "nvfp4bench_sdxl.py")
    if not os.path.isfile(bench_script):
        print(f"[FATAL] Post-convert bench script not found: {bench_script}")
        return 1
    if not os.path.isfile(fp16_path):
        print(f"[FATAL] Post-convert bench: FP16 (--model) missing: {fp16_path}")
        return 1
    if not os.path.isfile(nvfp4_path):
        print(
            f"[FATAL] Post-convert bench: NVFP4 (--output) missing: {nvfp4_path}"
        )
        return 1

    # Final gate: free any leftover parent CUDA before the bench process starts.
    _release_vram_before_bench("pre-NVFP4-bench subprocess")

    cmd = [
        sys.executable,
        bench_script,
        "--fp16",
        fp16_path,
        "--nvfp4",
        nvfp4_path,
        "--prompt",
        _FIXED_NVFP4BENCH_PROMPT,
        "--seed",
        str(_FIXED_NVFP4BENCH_SEED),
    ]
    print("=" * 60)
    print("[*] Post-convert NVFP4 fidelity bench (owner command shape)")
    print(f"    script: {bench_script}")
    print(f"    --fp16: {fp16_path}")
    print(f"    --nvfp4: {nvfp4_path}")
    print(f"    --prompt: {_FIXED_NVFP4BENCH_PROMPT}")
    print(f"    --seed: {_FIXED_NVFP4BENCH_SEED} (fixed inside)")
    print("=" * 60)
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    _install_torchaudio_stub()
    parser = argparse.ArgumentParser(
        description=(
            "Diffusion FULL ConvRot convert: Linear→NVFP4 (+ rotate), "
            "Conv2d→INT8 (+ rotate). Pass --calib_file for NVFP4 .input_scale, "
            "HSWQ DualMonitor r0 + packed NVFP4/INT8 overhead budget "
            "(--fp16_budget_mb / --budget_mb, default "
            f"{FP16_BUDGET_MB_DEFAULT:g} MiB), and weight clip amax from "
            "NVFP4/INT8 pack roundtrip MSE. "
            "Online act rotate required at load (loader built separately). "
            "Card 1 = --bias_correction. Post-convert NVFP4 bench default ON."
        )
    )
    parser.add_argument(
        "--model",
        "--input",
        dest="model",
        type=str,
        required=True,
        help="Path to input .safetensors",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output .safetensors"
    )
    parser.add_argument(
        "--bias_correction",
        action="store_true",
        help=(
            "Card 1 ON: DualMonitor calib; bias += -(W_q - W) @ mu_x. "
            "Requires --calib_file."
        ),
    )
    parser.add_argument(
        "--calib_file",
        type=str,
        default=None,
        help=(
            "Calibration prompts text file. Writes per-layer NVFP4 .input_scale "
            "(amax/(F8_E4M3_MAX*F4_E2M1_MAX); ConvRot Linear uses rotated amax). "
            "Also enables HSWQ DualMonitor r0 + packed NVFP4/INT8 overhead "
            "budget (--fp16_budget_mb / --budget_mb) + NVFP4/INT8 pack-roundtrip "
            "weight clip amax. Required with --bias_correction."
        ),
    )
    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=0.0,
        help=(
            "Must be 0 (r0). Ranking is DualMonitor + analyze severity + "
            "V4 NVFP4/INT8 MSE @ absmax + infinite branches, truncated only by "
            "--fp16_budget_mb / --budget_mb. Top-%% cut is forbidden."
        ),
    )
    parser.add_argument(
        "--fp16_budget_mb",
        "--budget_mb",
        dest="fp16_budget_mb",
        type=float,
        default=FP16_BUDGET_MB_DEFAULT,
        help=(
            "Packed-baseline overhead budget in MiB (Linear vs NVFP4 +1.5 B/el, "
            "Conv2d vs INT8 +1 B/el). Any positive finite value. Default "
            f"{FP16_BUDGET_MB_DEFAULT:g}. Alias: --budget_mb."
        ),
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help=(
            "Path to NVFP4 distribution profile JSON "
            "(from analyze/analyze_sdxl_nvfp4_distribution.py). "
            "Optional; default {stem}_distribution_profile.json under repo root. "
            "Auto path always re-runs the NVFP4 analyze script."
        ),
    )
    parser.add_argument(
        "--num_calib_samples",
        type=int,
        default=32,
        help="Calibration samples (recommended: 32)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="Denoising steps per calib sample (default 25)",
    )
    parser.add_argument(
        "--convrot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "FULL ConvRot: offline Hadamard on eligible Linear (NVFP4) and "
            "Conv2d (INT8) + convrot stamp. Default ON; --no-convrot = plain packs."
        ),
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=_DEFAULT_GROUPSIZE,
        help=f"ConvRot Hadamard group size (power of 4, default {_DEFAULT_GROUPSIZE})",
    )
    parser.add_argument(
        "--min_in_features",
        type=int,
        default=0,
        help=(
            "Skip Linear/Conv2d with in_features/in_channels below this "
            "(0 = convert all eligible)."
        ),
    )
    parser.add_argument(
        "--bench",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After save, run benchmark/nvfp4bench_sdxl.py with "
            "--fp16=--model/--input and --nvfp4=--output "
            "(same shape as the owner NVFP4 bench command). "
            "Pass --no-bench to skip."
        ),
    )
    args = parser.parse_args()
    _install_nvfp4_convert_full_session_log()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)
    if args.bias_correction and not args.calib_file:
        print("Error: --bias_correction requires --calib_file")
        sys.exit(1)
    if abs(float(args.keep_ratio)) > 1e-12:
        print(
            f"Error: --keep_ratio must be 0 (r0); got {args.keep_ratio}. "
            "Use --fp16_budget_mb / --budget_mb for the packed NVFP4/INT8 "
            "overhead budget."
        )
        sys.exit(1)
    try:
        budget_mb = _bind_fp16_budget_from_option(float(args.fp16_budget_mb))
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    if args.groupsize < 4 or (args.groupsize & (args.groupsize - 1)) != 0:
        print(f"Error: --groupsize must be a power of 4 (>=4), got {args.groupsize}")
        sys.exit(1)
    if math.log(args.groupsize, 4) % 1 != 0:
        print(f"Error: --groupsize must be a power of 4, got {args.groupsize}")
        sys.exit(1)

    print(
        f"  [budget] packed NVFP4/INT8 overhead ceiling = {budget_mb:g} MiB "
        f"(from --fp16_budget_mb / --budget_mb)"
    )
    convert_to_nvfp4_convrot(
        args.model,
        args.output,
        bias_correction=bool(args.bias_correction),
        calib_file=args.calib_file,
        num_calib_samples=args.num_calib_samples,
        num_inference_steps=args.num_inference_steps,
        enable_convrot=bool(args.convrot),
        group_size=int(args.groupsize),
        min_in_features=int(args.min_in_features),
        keep_ratio=float(args.keep_ratio),
        profile_arg=args.profile,
        fp16_budget_mb=float(budget_mb),
    )

    if args.bench:
        bench_rc = run_post_convert_nvfp4_bench(
            script_dir=_script_dir(),
            fp16_path=args.model,
            nvfp4_path=args.output,
        )
        if bench_rc != 0:
            print(f"[FATAL] Post-convert bench exited with code {bench_rc}")
            sys.exit(bench_rc)
    else:
        print("[*] Post-convert bench skipped (--no-bench)")

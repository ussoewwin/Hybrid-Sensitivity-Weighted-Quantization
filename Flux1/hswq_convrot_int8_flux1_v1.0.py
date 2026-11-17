"""Flux1 DiT — ComfyUI-native FULL ConvRot INT8 convert + Card 1 Bias Correction.

Flux1-only (double_blocks / single_blocks DiT). FATAL if the Flux1 signature
(img_in.weight + double_blocks.0.img_attn.qkv.weight) is missing.
SDXL / Diffusers UNet path is not used.

Pack (ComfyUI MixedPrecisionOps + comfy_kitchen TensorWiseINT8Layout):
  <layer>.weight           int8
  <layer>.weight_scale     float32
      plain INT8:          scalar (tensorwise)  OR  --per_channel_int8 -> [O,1]
      ConvRot Linear:      [out, 1] (row-wise) — kitchen online act rotate
  <layer>.comfy_quant      uint8 JSON (compact)
      plain:  {"format":"int8_tensorwise"}
      ConvRot:{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":N}

FULL ConvRot (default ON; --no-convrot for plain INT8):
  Linear 2D:  W_rot = W @ H^T, row-wise INT8, stamp.
  (Flux1 has no Conv2d; the 4D branch is kept for completeness and reports 0.)
  Hadamard / rotate_weight live in THIS file (same math as comfy_kitchen ConvRot;
  no import of native_convert_int8.py). If in_features is not divisible by a
  power-of-4 group size, that layer stays plain tensorwise (or Card 3).

DualMonitor calib (--calib_file + --clip_path + --clip_l_path, CLIPType.FLUX):
  DualMonitor and calibration are ONE unit (run_card1_calib only).
  No DualMonitor without calibration. No calibration-less DualMonitor path.
  Runs when blacklist_keep>0 OR keep_sensitive>0 OR bias_correction OR
  BOTH paths are set. blacklist_keep / keep_sensitive force run_card1_calib
  (bias ON/OFF does not matter). Paths are required for that run.

Card 1 (--bias_correction):
  DualMonitor consumer: bias += -(W_q - W) @ mu_x.
  Bias OFF = skip bias delta ONLY. Does not cancel DualMonitor.

Fixed structure blacklist (safety net):
  img_in / txt_in / time_in / vector_in / guidance_in / final_layer /
  distilled_guidance_layer / img_mod / txt_mod / modulation / norm / bias.
  These layers are ALWAYS kept in original dtype regardless of ranking.

Data-driven reverse ranking (on top of fixed blacklist):
  --blacklist_keep N: top N highest-composite DiT weights (4-axis: DM E[x^2] x
    HistCosine V5 x NVFP4 measured x SVD Leverage) -> original dtype.
  --keep_sensitive M: next M from the remaining ranked pool -> original dtype.
  Both force DualMonitor. Works with bias OFF.
  No Frobenius escape / no Static Profile VETO / no SDXL.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import math
import os
import sys
import types
from typing import Optional, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm

# Histogram Cosine V5 (repo histogram/; amax via Cosine Similarity Loss).
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_HIST_DIR = os.path.join(_REPO_ROOT, "histogram")
if _HIST_DIR not in sys.path:
    sys.path.insert(0, _HIST_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from weighted_histogram_cosine_v5 import (  # noqa: E402
    HSWQWeightedHistogramOptimizerV5 as HSWQWeightedHistogramOptimizerFast,
    compute_hybrid_leverage_scores,
)

try:
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout  # Axis 3 (optional)
    _HAS_NVFP4 = True
except Exception:
    _HAS_NVFP4 = False

_DEFAULT_GROUPSIZE = 256
_MODEL_TYPE = "Flux1-v1.0"
_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

# Flux1 DiT — structure-sensitive boundary layers stay in their base dtype.
_FLUX_BLACKLIST: list[str] = [
    "img_in",
    "txt_in",
    "time_in",
    "vector_in",
    "guidance_in",
    "final_layer",
    "distilled_guidance_layer",
    "img_mod",
    "txt_mod",
    "modulation",
    "norm",
    "bias",
]

_NON_DIFFUSION_MARKERS: tuple[str, ...] = (
    "conditioner.",
    "cond_stage_model.",
    "text_encoders.",
    "text_encoder.",
    "text_encoder_2.",
    "text_encoder_3.",
    "text_model.",
    "text_projection",
    "logit_scale",
    "clip_l.",
    "clip_g.",
    "t5xxl.",
    "first_stage_model.",
    "vae.",
)


# ---------------------------------------------------------------------------
# DualMonitor (Card 1) — self-contained
# ---------------------------------------------------------------------------
class DualMonitor:
    """Per-layer act moments for Card 1 bias correction."""

    def __init__(self):
        self.output_sum = 0.0
        self.output_sq_sum = 0.0
        self.count = 0
        self.channel_importance = None
        self.channel_act_mean = None
        self.channel_act_sq_mean = None

    def update(self, input_tensor, output_tensor, module=None, weight: float = 1.0):
        with torch.no_grad():
            out_detached = output_tensor.detach().float()
            out_clamped = torch.clamp(out_detached, -65504.0, 65504.0)
            mean_val = out_clamped.mean().item()
            sq_mean_val = (out_clamped ** 2).mean().item()

            if math.isfinite(mean_val) and math.isfinite(sq_mean_val):
                self.output_sum += mean_val
                self.output_sq_sum += sq_mean_val
            inp_detached = input_tensor.detach().float()
            if inp_detached.dim() >= 2:
                reduce_dims = tuple(range(inp_detached.dim() - 1))
            else:
                current_imp = torch.ones(
                    1, device=inp_detached.device, dtype=torch.float32
                )
                current_act = torch.zeros(
                    1, device=inp_detached.device, dtype=torch.float32
                )
                current_sq = torch.ones(
                    1, device=inp_detached.device, dtype=torch.float32
                )
                reduce_dims = None
            if reduce_dims is not None:
                current_imp = inp_detached.abs().mean(dim=reduce_dims)
                current_act = inp_detached.mean(dim=reduce_dims)
                current_sq = (inp_detached ** 2).mean(dim=reduce_dims)
            w = float(weight)
            self.output_sum *= self.count / max(self.count + w, 1e-12)
            self.output_sq_sum *= self.count / max(self.count + w, 1e-12)
            self.output_sum += mean_val * w
            self.output_sq_sum += sq_mean_val * w
            if self.channel_importance is None:
                self.channel_importance = current_imp
                self.channel_act_mean = current_act
                self.channel_act_sq_mean = current_sq
            elif current_imp.shape == self.channel_importance.shape:
                self.channel_importance = (
                    self.channel_importance * self.count + current_imp * w
                ) / (self.count + w)
                self.channel_act_mean = (
                    self.channel_act_mean * self.count + current_act * w
                ) / (self.count + w)
                self.channel_act_sq_mean = (
                    self.channel_act_sq_mean * self.count + current_sq * w
                ) / (self.count + w)
            self.count += w


dual_monitors: dict[str, DualMonitor] = {}
_dm_timestep_weight: float = 1.0


def hook_fn(module, input, output, name):
    if name not in dual_monitors:
        dual_monitors[name] = DualMonitor()
    dual_monitors[name].update(input[0], output, module, weight=_dm_timestep_weight)


def compute_int8_bias_delta(weight_fp, weight_dq, act_mean):
    """bias_delta ≈ (W_q - W) contracted with per-input-channel E[x]."""
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


# ---------------------------------------------------------------------------
# Midrank composite ranking (DualMonitor x HistCosine / 4-axis)
# ---------------------------------------------------------------------------
def _pool_midranks(values: Sequence[float]) -> list[float]:
    """Average midrank / n in (0, 1] within THIS measured pool (ties share)."""
    n = len(values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: (float(values[i]), i))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and float(values[order[j + 1]]) == float(values[order[i]]):
            j += 1
        mid = 0.5 * float((i + 1) + (j + 1))
        for k in range(i, j + 1):
            ranks[order[k]] = mid / float(n)
        i = j + 1
    return ranks


def _true_median(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(float(v) for v in vals)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return 0.5 * float(s[n // 2 - 1] + s[n // 2])


def _iqr(vals: Sequence[float]) -> float:
    if len(vals) < 2:
        return 0.0
    s = sorted(float(v) for v in vals)
    n = len(s)
    q1 = s[max(0, (n - 1) // 4)]
    q3 = s[min(n - 1, (3 * (n - 1)) // 4)]
    return float(q3 - q1)


def _derive_dm_hist_weights(
    dm_ranks: Sequence[float], hist_ranks: Sequence[float]
) -> dict[str, float | str]:
    """Auto axis weights from THIS pool's midrank IQR/median (no fixed recipe)."""
    eps = 1e-12
    dm_iqr = _iqr(dm_ranks)
    hist_iqr = _iqr(hist_ranks)
    dm_p50 = _true_median(dm_ranks)
    hist_p50 = _true_median(hist_ranks)
    d_dm = dm_iqr / max(dm_p50, eps) if dm_p50 > 0 else 0.0
    d_hist = hist_iqr / max(hist_p50, eps) if hist_p50 > 0 else 0.0
    w_sum = d_dm + d_hist
    if w_sum < eps:
        return {
            "form": "equal_weight_geometric",
            "w_dm": 0.5,
            "w_hist": 0.5,
            "dm_iqr": float(dm_iqr),
            "hist_iqr": float(hist_iqr),
            "dm_p50": float(dm_p50),
            "hist_p50": float(hist_p50),
        }
    return {
        "form": "weighted_geometric",
        "w_dm": float(d_dm / w_sum),
        "w_hist": float(d_hist / w_sum),
        "dm_iqr": float(dm_iqr),
        "hist_iqr": float(hist_iqr),
        "dm_p50": float(dm_p50),
        "hist_p50": float(hist_p50),
    }


def _composite_dm_hist(
    r_dm: float, r_hist: float, w_dm: float, w_hist: float
) -> float:
    """Weighted geometric mean on midranks (higher = more protect priority)."""
    eps = 1e-12
    return (max(float(r_dm), eps) ** float(w_dm)) * (
        max(float(r_hist), eps) ** float(w_hist)
    )


def _histogram_cosine_score(
    weight: torch.Tensor,
    importance: Optional[torch.Tensor],
    hist_opt: HSWQWeightedHistogramOptimizerFast,
) -> float:
    """V5 weighted histogram Cosine loss at optimal amax (SVD+RMS hybrid)."""
    with contextlib.redirect_stdout(io.StringIO()):
        optimal_amax = hist_opt.compute_optimal_amax(
            weight, importance=importance, use_svd_leverage=False,
            scaled=False, loss_type="cosine"
        )
        from weighted_histogram_cosine_v5 import WeightedHistogram
        wh = WeightedHistogram(bins=hist_opt.bins, device=hist_opt.device)
        wh.build(weight, importance)
        hist = wh.get_histogram()
        bc = wh.get_bin_centers()
        est_loss = hist_opt.cosine_optimizer.compute_weighted_cosine(
            hist, bc, optimal_amax, scaled=False, loss_type="cosine"
        )
    loss = float(est_loss)
    if not math.isfinite(loss):
        return 0.0
    return loss


def _nvfp4_measured_error(weight: torch.Tensor, device: str = "cuda") -> float:
    """Axis 3: NVFP4 quantization error on this weight (offline, no calib)."""
    if not _HAS_NVFP4:
        return -1.0
    try:
        w_bf16 = weight.to(dtype=torch.bfloat16, device=device)
        qdata, params = TensorCoreNVFP4Layout.quantize(w_bf16)
        if hasattr(TensorCoreNVFP4Layout, "dequantize"):
            w_dq = TensorCoreNVFP4Layout.dequantize(qdata, params).float().cpu()
            err = weight.float() - w_dq
            return float(err.norm().item()) / max(float(weight.float().norm().item()), 1e-8)
        del qdata, params
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass
    return -1.0


def _svd_leverage_score(weight: torch.Tensor) -> float:
    """Axis 4: SVD Leverage standalone (structural importance)."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            hybrid_imp = compute_hybrid_leverage_scores(weight, alpha=0.7, beta=0.3)
        return float(hybrid_imp.mean().item())
    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
# ConvRot Hadamard (self-contained — comfy_kitchen / INT8 ConvRot compatible)
# ---------------------------------------------------------------------------
def build_hadamard(
    size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normalized regular Hadamard (power-of-4), same as comfy_kitchen ConvRot."""
    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]

    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")

    h4 = torch.tensor(
        [
            [1, 1, 1, -1],
            [1, 1, -1, 1],
            [1, -1, 1, 1],
            [-1, 1, 1, 1],
        ],
        dtype=dtype,
        device=device,
    )
    h_matrix = h4
    current_size = 4
    while current_size < size:
        h_matrix = torch.kron(h_matrix, h4)
        current_size *= 4
    h_matrix = h_matrix / (size**0.5)
    _HADAMARD_CACHE[cache_key] = h_matrix
    return h_matrix


def convrot_group_size_for_features(
    n: int, preferred: int = _DEFAULT_GROUPSIZE
) -> int | None:
    """Largest power-of-4 group size <= preferred that divides n (or None)."""
    if n < 4:
        return None
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def rotate_weight(
    weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Offline Linear: W_rot = W @ H^T (group-wise). Matches kitchen ConvRot."""
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features {in_features} not divisible by group_size {group_size}"
        )
    group_count = in_features // group_size
    weight_grouped = weight.view(out_features, group_count, group_size)
    return torch.matmul(
        weight_grouped, h_matrix.T.to(dtype=weight.dtype, device=weight.device)
    ).reshape(weight.shape)


def rotate_weight_conv2d(
    weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Offline Conv2d: rotate along in_channels. weight (O, I, kH, kW).

    Flux1 has no Conv2d; kept so the 4D branch of the converter stays correct
    if a Flux-derived checkpoint ever carries one.
    """
    if weight.ndim != 4:
        raise ValueError(f"Conv2d weight must be 4D, got ndim={weight.ndim}")
    out_c, in_c, k_h, k_w = weight.shape
    flat = weight.permute(0, 2, 3, 1).contiguous().view(-1, in_c)
    flat_rot = rotate_weight(flat, h_matrix, group_size)
    return flat_rot.view(out_c, k_h, k_w, in_c).permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _repo_root() -> str:
    here = _script_dir()
    repo = os.path.normpath(os.path.join(here, ".."))
    if os.path.isdir(os.path.join(repo, "ComfyUI-master")):
        return repo
    return here


def _encode_comfy_quant(config: dict) -> torch.Tensor:
    return torch.tensor(
        list(json.dumps(config, separators=(",", ":")).encode("utf-8")),
        dtype=torch.uint8,
    )


def _is_non_diffusion_key(key: str) -> bool:
    return any(marker in key for marker in _NON_DIFFUSION_MARKERS)


def _find_flux_key_prefix(keys) -> str:
    """Prefix detection from keys only (dict or any key iterable)."""
    keyset = keys if isinstance(keys, (set, frozenset)) else set(keys)
    for prefix in ("model.diffusion_model.", "diffusion_model.", ""):
        if f"{prefix}img_in.weight" in keyset:
            if f"{prefix}double_blocks.0.img_attn.qkv.weight" not in keyset:
                raise ValueError(
                    "Flux1 signature incomplete: img_in.weight present but "
                    f"{prefix}double_blocks.0.img_attn.qkv.weight missing"
                )
            return prefix
    raise ValueError(
        "Not a Flux1 checkpoint: missing img_in.weight "
        "(under model.diffusion_model. / diffusion_model. / root)."
    )


def _meta_base_key(base_k_file: str) -> str:
    if "model.diffusion_model." in base_k_file:
        return base_k_file.split("model.diffusion_model.")[-1]
    if "diffusion_model." in base_k_file:
        return base_k_file.split("diffusion_model.")[-1]
    return base_k_file


def _is_blacklisted(key: str) -> bool:
    return any(name in key for name in _FLUX_BLACKLIST)


def pack_tensorwise(weight: torch.Tensor):
    """Symmetric per-tensor INT8: scale = amax / 127."""
    w = weight.float()
    amax = max(float(w.abs().max().item()), 1e-6)
    scale = amax / 127.0
    q = (w / scale).round().clamp(-127, 127).to(torch.int8)
    return q, torch.tensor(scale, dtype=torch.float32)


def pack_channelwise(weight: torch.Tensor):
    """Per-out-channel INT8 (Card 3 / ConvRot kitchen dequant shape)."""
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
        raise ValueError(f"unsupported weight ndim={w.dim()} for channelwise INT8")
    clamped = torch.clamp(w, -amax_view, amax_view)
    q = (clamped / scale_view).round().clamp(-127, 127).to(torch.int8)
    return q, scale_view.to(dtype=torch.float32)


# ---------------------------------------------------------------------------
# ComfyUI bootstrap
# ---------------------------------------------------------------------------
def _ensure_comfyui_on_sys_path(comfy_path: str | None = None) -> str:
    candidates = []
    if comfy_path:
        candidates.append(os.path.abspath(comfy_path))
    env = os.environ.get("COMFYUI_PATH")
    if env:
        candidates.append(env)
    repo = _repo_root()
    candidates.extend(
        [
            os.path.join(repo, "ComfyUI-master"),
            os.path.join(_script_dir(), "ComfyUI-master"),
        ]
    )
    for root in candidates:
        if not root:
            continue
        model_py = os.path.join(root, "comfy", "ldm", "flux", "model.py")
        if os.path.isfile(model_py):
            if root not in sys.path:
                sys.path.insert(0, root)
            return root
    raise FileNotFoundError(
        "ComfyUI root with comfy/ldm/flux/model.py not found. "
        "Pass --comfy_path or set COMFYUI_PATH."
    )


def _clear_argv_for_comfy() -> list[str]:
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    return saved


def _restore_argv(saved: list[str]) -> None:
    sys.argv = saved


def _install_torchaudio_stub() -> None:
    """Prevent real torchaudio from loading during comfy.sd import."""
    import importlib.machinery

    for key in list(sys.modules):
        if key == "torchaudio" or key.startswith("torchaudio."):
            del sys.modules[key]

    def _stub_mod(name: str, *, is_package: bool = False):
        mod = types.ModuleType(name)
        mod.__file__ = "<hswq_torchaudio_stub>"
        if is_package:
            mod.__path__ = []
            spec = importlib.machinery.ModuleSpec(
                name, loader=None, is_package=True
            )
            spec.submodule_search_locations = []
        else:
            spec = importlib.machinery.ModuleSpec(name, loader=None)
        mod.__spec__ = spec
        return mod

    ta = _stub_mod("torchaudio", is_package=True)
    functional = _stub_mod("torchaudio.functional")

    def _resample(waveform, orig_freq, new_freq, *args, **kwargs):
        return waveform

    functional.resample = _resample

    transforms = _stub_mod("torchaudio.transforms")

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


def _install_comfy_optional_stubs() -> None:
    """Lightweight stubs (same pattern as the other HSWQ converters)."""
    _install_torchaudio_stub()

    try:
        import comfy_aimdo  # noqa: F401
    except Exception:
        m = types.ModuleType("comfy_aimdo")
        m.__file__ = "<stub>"
        m.__path__ = []
        sys.modules["comfy_aimdo"] = m
        sys.modules["comfy_aimdo.filter"] = types.ModuleType("comfy_aimdo.filter")
        sys.modules["comfy_aimdo.filter"].filter_modules = lambda *a, **k: None

    try:
        import psutil  # noqa: F401
    except Exception:

        class _VM:
            total = 64 * 1024**3
            available = 32 * 1024**3

        class _Proc:
            def memory_info(self):
                return types.SimpleNamespace(rss=0)

            def memory_full_info(self):
                return types.SimpleNamespace(uss=0)

            def cpu_percent(self, interval=None):
                return 0.0

            def num_threads(self):
                return 1

        ps = types.ModuleType("psutil")
        ps.virtual_memory = lambda: _VM()
        ps.Process = lambda: _Proc()
        sys.modules["psutil"] = ps


def _encode_flux_calib_contexts(
    *,
    clip_path: str,
    clip_l_path: str,
    prompts: list[str],
    comfy_path: str | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
    """Encode calib prompts with Comfy CLIPType.FLUX (t5xxl + clip_l).

    Flux needs BOTH encoders, so --clip_path (t5xxl) and --clip_l_path (clip_l)
    are one unit here. Returns CPU tensors (context [1, seq, 4096], y|None).
    CLIP is unloaded before return so the DiT DualMonitor owns VRAM.
    """
    if not os.path.isfile(clip_path):
        raise FileNotFoundError(f"--clip_path not found: {clip_path}")
    if not os.path.isfile(clip_l_path):
        raise FileNotFoundError(f"--clip_l_path not found: {clip_l_path}")

    saved_argv = _clear_argv_for_comfy()
    try:
        _ensure_comfyui_on_sys_path(comfy_path)
        import comfy.options

        comfy.options.enable_args_parsing(False)
        _install_comfy_optional_stubs()
        _install_torchaudio_stub()

        import comfy.model_management as mm  # noqa: WPS433
        import comfy.sd  # noqa: WPS433

        mm.get_torch_device()
        print(f"  [Flux1 calib] Loading CLIP (FLUX / clip_l + t5xxl)")
        _cpu = torch.device("cpu")
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_l_path, clip_path],
            embedding_directory=None,
            clip_type=comfy.sd.CLIPType.FLUX,
            model_options={"load_device": _cpu, "offload_device": _cpu,
                           "initial_device": _cpu},
        )
        patcher = getattr(clip, "patcher", None)
        if patcher is not None:
            print(
                f"  [Flux1 calib] CLIP device contract: load={patcher.load_device} "
                f"offload={patcher.offload_device}"
            )

        bank: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        for i, prompt in enumerate(prompts):
            tokens = clip.tokenize(prompt)
            conds = clip.encode_from_tokens_scheduled(tokens)
            if not conds:
                raise RuntimeError(
                    f"CLIP encode returned empty conditioning for calib sample {i}"
                )
            cond_t = conds[0][0]
            meta = conds[0][1] if len(conds[0]) > 1 else {}
            if not torch.is_tensor(cond_t):
                raise RuntimeError(
                    f"CLIP cond is not a tensor (sample {i}): {type(cond_t)!r}"
                )
            if cond_t.ndim == 2:
                cond_t = cond_t.unsqueeze(0)
            if cond_t.ndim != 3:
                raise RuntimeError(
                    f"CLIP context expected 3D (B, seq, fused), got shape "
                    f"{tuple(cond_t.shape)} (sample {i})"
                )
            y = None
            if isinstance(meta, dict):
                pooled = meta.get("pooled_output")
                if torch.is_tensor(pooled):
                    y = pooled.detach().float().cpu()
                    if y.ndim == 1:
                        y = y.unsqueeze(0)
            bank.append(
                (
                    cond_t.detach().to(dtype=torch.bfloat16).cpu(),
                    y,
                )
            )
            print(
                f"  [Flux1 calib] CLIP encoded {i + 1}/{len(prompts)} "
                f"shape={tuple(cond_t.shape)} y={'none' if y is None else tuple(y.shape)}"
            )

        if getattr(clip, "cond_stage_model", None) is not None:
            clip.cond_stage_model.cpu()
        if patcher is not None:
            mm.unload_model_and_clones(patcher)
        del clip
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  [Flux1 calib] CLIP unloaded; real contexts on CPU for DiT DualMonitor")
        return bank
    finally:
        _restore_argv(saved_argv)


def load_flux_from_safetensors(path, device="cuda", comfy_path: str | None = None):
    """Load the Flux1 DiT through ComfyUI + identity pack-key -> module.weight map."""
    if str(device).startswith("cpu"):
        raise RuntimeError(
            "load_flux_from_safetensors refused device='cpu'. "
            "Flux1 Card 1 DualMonitor calibration requires CUDA."
        )
    _ensure_comfyui_on_sys_path(comfy_path)
    saved_argv = _clear_argv_for_comfy()
    try:
        import comfy.options

        comfy.options.enable_args_parsing(False)
        _install_comfy_optional_stubs()
        import comfy.model_management as mm  # noqa: WPS433
        import comfy.sd  # noqa: WPS433

        print(f"Loading Flux1 DiT (ComfyUI native): {path}")
        patcher = comfy.sd.load_diffusion_model(os.path.abspath(path), {})
        mm.get_torch_device()
        # The 23.8 GB fp16 DiT does not fit a 16 GiB card: let ComfyUI decide the
        # offload (it streams layers in and out per forward).
        mm.load_models_gpu([patcher])
        dm = patcher.model.diffusion_model
        dm.eval()

        # Key names only (safe_open header): loading the 23.8 GB state dict here
        # would double the RAM the DiT already occupies.
        with safe_open(os.path.abspath(path), "pt") as fh:
            file_keys = set(fh.keys())
            shapes = {
                k: tuple(fh.get_slice(k).get_shape())
                for k in file_keys
                if k.endswith(".weight")
            }
        prefix = _find_flux_key_prefix(file_keys)
        comfyui_to_module_map = {}
        for name, mod in dm.named_modules():
            w = getattr(mod, "weight", None)
            if w is None or not torch.is_tensor(w):
                continue
            if len(shapes.get(f"{prefix}{name}.weight", ())) not in (2, 4):
                continue
            ck = f"{prefix}{name}.weight"
            if ck in file_keys:
                comfyui_to_module_map[ck] = f"{name}.weight"
        print(
            f"  [Flux1] identity map entries={len(comfyui_to_module_map)} "
            f"(prefix={prefix!r}, header keys={len(file_keys)})"
        )
        return dm, comfyui_to_module_map, prefix
    finally:
        _restore_argv(saved_argv)


# ---------------------------------------------------------------------------
# Card 1 calib
# ---------------------------------------------------------------------------
def run_card1_calib(
    *,
    input_path: str,
    calib_file: str,
    clip_path: str,
    clip_l_path: str,
    num_calib_samples: int,
    num_inference_steps: int,
    device: str,
    latent_size: int = 32,
    comfy_path: str | None = None,
):
    """Card 1 only: CLIPType.FLUX contexts + DualMonitor -> channel_act_mean."""
    global _dm_timestep_weight
    if not str(device).startswith("cuda"):
        raise RuntimeError("Card 1 Flux1 calib requires CUDA.")

    print("Preparing calibration data...")
    with open(calib_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    if len(prompts) < num_calib_samples:
        prompts = (prompts * (num_calib_samples // len(prompts) + 1))[
            :num_calib_samples
        ]
    else:
        prompts = prompts[:num_calib_samples]

    context_bank = _encode_flux_calib_contexts(
        clip_path=clip_path,
        clip_l_path=clip_l_path,
        prompts=prompts,
        comfy_path=comfy_path,
    )
    if len(context_bank) != len(prompts):
        raise RuntimeError(
            f"CLIP context bank size {len(context_bank)} != calib prompts {len(prompts)}"
        )

    model, comfyui_to_module_map, _prefix = load_flux_from_safetensors(
        input_path, device=device, comfy_path=comfy_path
    )

    print("Preparing calibration (DualMonitor hooks; Card 1 act means)...")
    dual_monitors.clear()
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(
                module.register_forward_hook(
                    lambda m, i, o, n=name: hook_fn(m, i, o, n)
                )
            )

    print(
        f"Running calibration ({num_calib_samples} samples, "
        f"{num_inference_steps} steps)..."
    )
    if num_calib_samples != 32 or num_inference_steps != 25:
        print(
            "  [WARN] How-to / r32 recipe is num_calib_samples=32, "
            "num_inference_steps=25. current args differ."
        )
    _calib_progress_disable = False
    guidance = torch.full((1,), 3.5, device=device, dtype=torch.float32)
    gen = torch.Generator(device=device).manual_seed(42)

    for i, prompt in enumerate(prompts):
        seed = 42 + i
        print(f"\nSample {i+1}/{num_calib_samples}: {prompt[:50]}...")
        gen.manual_seed(seed)
        with torch.no_grad():
            x = torch.randn(
                1,
                16,
                int(latent_size),
                int(latent_size),
                device=device,
                dtype=torch.bfloat16,
                generator=gen,
            )
            ctx_cpu, y_cpu = context_bank[i]
            context = ctx_cpu.to(device=device, dtype=torch.bfloat16)
            y = None
            if y_cpu is not None:
                y = y_cpu.to(device=device, dtype=torch.bfloat16)
            for step in tqdm(
                range(int(num_inference_steps)),
                total=int(num_inference_steps),
                disable=_calib_progress_disable,
            ):
                t = torch.full(
                    (1,),
                    float(step) / float(max(num_inference_steps, 1)),
                    device=device,
                    dtype=torch.float32,
                )
                _dm_timestep_weight = float(1.0 - t.item())
                model(x, t, context, y, guidance=guidance)
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    for h in handles:
        h.remove()

    act_mean_dict = {}
    act_sq_mean_dict = {}
    importance_dict = {}
    for name, mon in dual_monitors.items():
        if mon.channel_act_mean is not None:
            act_mean_dict[name] = mon.channel_act_mean.detach().float().cpu()
        if mon.channel_act_sq_mean is not None:
            act_sq_mean_dict[name] = mon.channel_act_sq_mean.detach().float().cpu()
        if mon.channel_importance is not None:
            importance_dict[name] = mon.channel_importance.detach().float().cpu()
    print(
        f"  [Card 1 DualMonitor] act_mean layers={len(act_mean_dict)}, "
        f"act_sq_mean layers={len(act_sq_mean_dict)}, "
        f"importance layers={len(importance_dict)} "
        f"(full Card 1; no VETO; no Approach A)"
    )

    del model
    del context_bank
    dual_monitors.clear()
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "act_mean_dict": act_mean_dict,
        "act_sq_mean_dict": act_sq_mean_dict,
        "importance_dict": importance_dict,
        "comfyui_to_module_map": comfyui_to_module_map,
    }


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------
def convert_to_int8(
    input_path: str,
    output_path: str,
    *,
    per_channel_int8: bool = False,
    bias_correction: bool = False,
    calib_file: str | None = None,
    clip_path: str | None = None,
    clip_l_path: str | None = None,
    comfy_path: str | None = None,
    num_calib_samples: int = 32,
    num_inference_steps: int = 25,
    latent_size: int = 32,
    enable_convrot: bool = True,
    group_size: int = _DEFAULT_GROUPSIZE,
    blacklist_keep: int = 0,
    keep_sensitive: int = 0,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    act_mean_dict: dict[str, torch.Tensor] = {}
    act_sq_mean_dict: dict[str, torch.Tensor] = {}
    importance_dict: dict[str, torch.Tensor] = {}
    comfyui_to_module_map: dict[str, str] = {}
    convrot_linear = 0
    convrot_conv2d = 0
    bf16_keep = 0
    blacklist_reverted = 0
    sensitivity_reverted = 0
    layer_convrot_gs: dict[str, int] = {}

    print(f"Mode {_MODEL_TYPE} | device={device} | FULL ConvRot INT8 (Flux1-only)")

    if enable_convrot:
        print(
            f"  [ConvRot] FULL ON | Linear (Flux1 has no Conv2d) "
            f"(preferred groupsize={group_size}; adaptive power-of-4 divisor) "
            f"| Hadamard self-contained in this file"
        )
        if bias_correction:
            print(
                "  [ConvRot] WARN: Card 1 DualMonitor means are from unrotated "
                "float DiT; BC uses rotated W vs W_q (approximate for ConvRot)"
            )

    use_bias = bool(bias_correction)
    use_blacklist_keep = int(blacklist_keep) > 0
    use_keep_sensitive = int(keep_sensitive) > 0
    use_reverse_rank = use_blacklist_keep or use_keep_sensitive
    calib_file = (str(calib_file).strip() if calib_file else "") or None
    clip_path = (str(clip_path).strip() if clip_path else "") or None
    clip_l_path = (str(clip_l_path).strip() if clip_l_path else "") or None
    have_calib_paths = bool(calib_file) and bool(clip_path) and bool(clip_l_path)
    run_dual_monitor = use_reverse_rank or use_bias or (
        bool(calib_file) and bool(clip_path) and bool(clip_l_path)
    )

    if run_dual_monitor and not have_calib_paths:
        raise ValueError(
            "DualMonitor requires --calib_file, --clip_path and --clip_l_path "
            "(Flux needs both text encoders). blacklist_keep / keep_sensitive "
            "force DualMonitor; bias and path-only runs require all three. "
            "There is no DualMonitor-less path."
        )

    if run_dual_monitor:
        if not os.path.isfile(calib_file):
            raise FileNotFoundError(f"calib_file not found: {calib_file}")
        if not os.path.isfile(clip_path):
            raise FileNotFoundError(f"clip_path not found: {clip_path}")
        if not os.path.isfile(clip_l_path):
            raise FileNotFoundError(f"clip_l_path not found: {clip_l_path}")
        if device != "cuda":
            raise RuntimeError("Flux1 DualMonitor calib requires CUDA.")

        print(
            "  [DualMonitor calib] ON | CLIPType.FLUX (clip_l + t5xxl) | "
            "run_card1_calib (mu_x / E[x^2] for Linear)"
        )
        if use_bias:
            print(
                "  [Bias Correction Card 1] ON | DualMonitor mu_x | "
                "bias += -(W_q - W) @ mu_x | "
                "no Approach A / no top_ratio gate"
            )
        else:
            print(
                "  [Bias Correction Card 1] OFF | "
                "calibration still runs (no bias delta applied)"
            )
        if use_blacklist_keep:
            print(
                "  [blacklist_keep] ON | DualMonitor E[x^2] x HistCosine composite | "
                f"top-N={int(blacklist_keep)} -> original dtype "
                "(data-driven, after fixed blacklist)"
            )
        if use_keep_sensitive:
            print(
                "  [keep_sensitive] ON | DualMonitor E[x^2] (unchanged) x "
                f"HistCosine midrank | next-M={int(keep_sensitive)} -> original dtype"
            )

        calib = run_card1_calib(
            input_path=input_path,
            calib_file=calib_file,
            clip_path=clip_path,
            clip_l_path=clip_l_path,
            num_calib_samples=int(num_calib_samples),
            num_inference_steps=int(num_inference_steps),
            device=device,
            latent_size=int(latent_size),
            comfy_path=comfy_path,
        )
        act_mean_dict = calib["act_mean_dict"]
        act_sq_mean_dict = calib["act_sq_mean_dict"]
        importance_dict = calib.get("importance_dict", {})
        comfyui_to_module_map = calib["comfyui_to_module_map"]
        print(
            f"  [DualMonitor] Captured act means for {len(act_mean_dict)} layers, "
            f"act sq means for {len(act_sq_mean_dict)} layers, "
            f"importance for {len(importance_dict)} layers"
        )

    print(f"Loading model: {input_path}")
    state_dict = load_file(input_path)
    prefix = _find_flux_key_prefix(state_dict)
    print(f"Detected Flux1 key prefix: {prefix!r}")

    new_state_dict: dict[str, torch.Tensor] = {}
    quant_meta_layers: dict[str, dict] = {}
    converted_count = 0
    skipped_count = 0
    plain_int8_count = 0
    layer_quant_errors: dict[str, float] = {}
    layer_hist_cosine: dict[str, float] = {}
    layer_nvfp4_err: dict[str, float] = {}
    layer_svd_lev: dict[str, float] = {}
    bias_corr_pending: dict[str, torch.Tensor] = {}
    bias_corr_applied = 0
    bias_corr_skipped_no_bias = 0
    bias_corr_skipped_no_act = 0
    bias_corr_skipped_bad_shape = 0
    mode = "per-channel" if per_channel_int8 else "tensorwise"
    rot_tag = " + ConvRot(Linear)" if enable_convrot else ""
    print(
        f"Converting Flux1 DiT Linear weights to INT8 "
        f"({mode}{rot_tag}, amax/127)..."
    )

    hist_opt: HSWQWeightedHistogramOptimizerFast | None = None
    if use_reverse_rank:
        hist_dev = "cuda" if torch.cuda.is_available() else "cpu"
        with contextlib.redirect_stdout(io.StringIO()):
            hist_opt = HSWQWeightedHistogramOptimizerFast(
                bins=4096,
                num_candidates=200,
                refinement_iterations=3,
                device=hist_dev,
                loss_type="cosine",
            )
        print(
            f"  [HistCosine V5] Axis 2 ON | SVD+RMS Leverage + Cosine Loss "
            f"on {hist_dev} (bins=4096, candidates=200, refine=3)"
        )

    for key, tensor in tqdm(list(state_dict.items())):
        if _is_blacklisted(key) or _is_non_diffusion_key(key):
            new_state_dict[key] = tensor
            bf16_keep += 1
            continue

        under_prefix = (not prefix) or key.startswith(prefix)

        if (
            under_prefix
            and key.endswith(".weight")
            and tensor.ndim in (2, 4)
            and tensor.dtype == torch.float32
        ):
            new_state_dict[key] = tensor
            bf16_keep += 1
            continue

        is_dit_weight = (
            under_prefix
            and key.endswith(".weight")
            and tensor.ndim in (2, 4)
            and tensor.dtype in (torch.float16, torch.bfloat16)
        )

        if not is_dit_weight:
            new_state_dict[key] = tensor
            skipped_count += 1
            continue

        w_fp = tensor.float()
        module_key = key[: -len(".weight")]
        used_gs = None
        if enable_convrot:
            used_gs = convrot_group_size_for_features(int(w_fp.shape[1]), group_size)

        if used_gs is not None and tensor.ndim == 2:
            h_matrix = build_hadamard(used_gs, device="cpu", dtype=torch.float32)
            w_fp = rotate_weight(w_fp, h_matrix, used_gs)
            q, scale = pack_channelwise(w_fp)
            quant_config = {
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": int(used_gs),
            }
            convrot_linear += 1
            layer_convrot_gs[module_key] = int(used_gs)
        elif used_gs is not None and tensor.ndim == 4:
            h_matrix = build_hadamard(used_gs, device="cpu", dtype=torch.float32)
            w_fp = rotate_weight_conv2d(w_fp, h_matrix, used_gs)
            q, scale = pack_channelwise(w_fp)
            quant_config = {
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": int(used_gs),
            }
            convrot_conv2d += 1
            layer_convrot_gs[module_key] = int(used_gs)
        elif per_channel_int8:
            q, scale = pack_channelwise(w_fp)
            quant_config = {"format": "int8_tensorwise"}
            plain_int8_count += 1
        else:
            q, scale = pack_tensorwise(w_fp)
            quant_config = {"format": "int8_tensorwise"}
            plain_int8_count += 1

        weight_dq = q.float() * scale
        new_state_dict[key] = q
        new_state_dict[f"{module_key}.weight_scale"] = scale
        new_state_dict[f"{module_key}.comfy_quant"] = _encode_comfy_quant(quant_config)
        quant_meta_layers[_meta_base_key(module_key)] = dict(quant_config)
        converted_count += 1

        module_w_key_sens = comfyui_to_module_map.get(key)
        module_name_sens = None
        if module_w_key_sens and module_w_key_sens.endswith(".weight"):
            module_name_sens = module_w_key_sens[: -len(".weight")]
        act_sq = (
            act_sq_mean_dict.get(module_name_sens)
            if module_name_sens is not None
            else None
        )
        err = w_fp - weight_dq
        if act_sq is not None and act_sq.shape[0] == w_fp.shape[1]:
            act_scale = act_sq.sqrt().to(device=err.device)
            if err.ndim == 2:
                weighted_err = err * act_scale.unsqueeze(0)
                weighted_base = w_fp * act_scale.unsqueeze(0)
            elif err.ndim == 4:
                weighted_err = err * act_scale.view(1, -1, 1, 1)
                weighted_base = w_fp * act_scale.view(1, -1, 1, 1)
            else:
                weighted_err = err
                weighted_base = w_fp
            rel_err = float(weighted_err.norm().item()) / max(
                float(weighted_base.norm().item()), 1e-8
            )
            layer_quant_errors[key] = rel_err
        elif use_reverse_rank:
            pass
        else:
            rel_err = float(err.norm().item()) / max(
                float(w_fp.norm().item()), 1e-8
            )
            layer_quant_errors[key] = rel_err

        if use_reverse_rank and hist_opt is not None:
            hybrid_imp = None
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    hybrid_imp = compute_hybrid_leverage_scores(
                        w_fp, alpha=0.7, beta=0.3)
            except Exception:
                pass

            imp = hybrid_imp
            if imp is None:
                if w_fp.ndim == 4:
                    imp = w_fp.abs().mean(dim=(0, 2, 3))
                else:
                    imp = w_fp.abs().mean(dim=0)
            try:
                layer_hist_cosine[key] = _histogram_cosine_score(w_fp, imp, hist_opt)
            except Exception:
                pass

            nv_err = _nvfp4_measured_error(w_fp, device=device)
            if nv_err >= 0:
                layer_nvfp4_err[key] = nv_err

            if hybrid_imp is not None:
                layer_svd_lev[key] = float(hybrid_imp.mean().item())

        if bias_correction:
            module_w_key = comfyui_to_module_map.get(key)
            module_name = None
            if module_w_key and module_w_key.endswith(".weight"):
                module_name = module_w_key[: -len(".weight")]
            act_mean = (
                act_mean_dict.get(module_name) if module_name is not None else None
            )
            if act_mean is None:
                bias_corr_skipped_no_act += 1
            else:
                rot_gs = layer_convrot_gs.get(module_key)
                if rot_gs is not None:
                    h_bc = build_hadamard(rot_gs, device="cpu", dtype=torch.float32)
                    act_mean = rotate_weight(
                        act_mean.unsqueeze(0).to(dtype=torch.float32),
                        h_bc, rot_gs
                    ).squeeze(0)
                delta = compute_int8_bias_delta(w_fp, weight_dq, act_mean)
                if delta is None:
                    bias_corr_skipped_bad_shape += 1
                else:
                    bias_corr_pending[module_key] = (-delta).detach().float().cpu()

    # --- Reverse ranking: blacklist_keep then keep_sensitive (no double-count) ---
    if use_reverse_rank and not layer_hist_cosine:
        raise RuntimeError(
            "blacklist_keep / keep_sensitive composite requires the HistCosine V5 "
            "axis, but no layer produced a score (check V5 import / hist_opt init)"
        )

    def _revert_to_original(rk: str, cscore: float, label: str) -> None:
        nonlocal converted_count
        mk = rk[: -len(".weight")]
        new_state_dict[rk] = state_dict[rk]
        scale_key = f"{mk}.weight_scale"
        quant_key = f"{mk}.comfy_quant"
        if scale_key in new_state_dict:
            del new_state_dict[scale_key]
        if quant_key in new_state_dict:
            del new_state_dict[quant_key]
        meta_bk = _meta_base_key(mk)
        if meta_bk in quant_meta_layers:
            del quant_meta_layers[meta_bk]
        bias_corr_pending.pop(mk, None)
        converted_count -= 1
        dm_display = (
            f"{layer_quant_errors[rk]:.6f}"
            if rk in layer_quant_errors
            else "neutral(0.5)"
        )
        print(
            f"    [{label}] {rk}  composite={cscore:.6f}  "
            f"dm_rel={dm_display}  "
            f"hist_cosine={layer_hist_cosine[rk]:.6e}  "
            f"dtype={state_dict[rk].dtype}"
        )

    remaining_errs: list[tuple[str, float]] = []
    if use_reverse_rank and layer_hist_cosine:
        pool_keys = list(layer_hist_cosine.keys())
        dm_real_keys = [k for k in pool_keys if k in layer_quant_errors]
        dm_real_vals = [float(layer_quant_errors[k]) for k in dm_real_keys]
        dm_real_ranks = _pool_midranks(dm_real_vals)
        dm_rank_lookup: dict[str, float] = {}
        for i, k in enumerate(dm_real_keys):
            dm_rank_lookup[k] = dm_real_ranks[i]

        if dm_real_keys:
            hist_real_vals_subset = [
                float(layer_hist_cosine[k]) for k in dm_real_keys
            ]
            hist_real_ranks_subset = _pool_midranks(hist_real_vals_subset)
            weights = _derive_dm_hist_weights(
                dm_real_ranks, hist_real_ranks_subset
            )
        else:
            weights = {
                "form": "hist_only_dm_empty",
                "w_dm": 0.0,
                "w_hist": 1.0,
                "dm_iqr": 0.0,
                "hist_iqr": 0.0,
                "dm_p50": 0.0,
                "hist_p50": 0.0,
            }
        w_dm = float(weights["w_dm"])
        w_hist = float(weights["w_hist"])

        hist_vals_full = [float(layer_hist_cosine[k]) for k in pool_keys]
        hist_ranks_full = _pool_midranks(hist_vals_full)

        composite: dict[str, float] = {}
        for i, k in enumerate(pool_keys):
            r_dm = dm_rank_lookup.get(k, 0.5)
            composite[k] = _composite_dm_hist(
                r_dm, hist_ranks_full[i], w_dm, w_hist
            )
        remaining_errs = sorted(
            composite.items(), key=lambda x: x[1], reverse=True
        )
        n_dm_neutral = len(pool_keys) - len(dm_real_keys)
        print(
            f"\n[Reverse ranking] DualMonitor x HistCosine midrank composite "
            f"(form={weights['form']}, w_dm={w_dm:.4f}, w_hist={w_hist:.4f}) | "
            f"pool={len(pool_keys)} "
            f"(DM-real={len(dm_real_keys)}, DM-neutral={n_dm_neutral})"
        )

    if use_blacklist_keep and remaining_errs:
        n_bl = min(int(blacklist_keep), len(remaining_errs))
        bl_keys = remaining_errs[:n_bl]
        remaining_errs = remaining_errs[n_bl:]
        print(
            f"\n[blacklist_keep] Reverting top {len(bl_keys)} highest-composite "
            f"layers to original dtype (data-driven, after fixed blacklist):"
        )
        for rk, cscore in bl_keys:
            _revert_to_original(rk, cscore, "blacklist_keep")
            blacklist_reverted += 1

    if use_keep_sensitive and remaining_errs:
        n_ks = min(int(keep_sensitive), len(remaining_errs))
        ks_keys = remaining_errs[:n_ks]
        remaining_errs = remaining_errs[n_ks:]
        print(
            f"\n[keep_sensitive] Reverting next {len(ks_keys)} highest-composite "
            f"layers to original dtype (after blacklist_keep):"
        )
        for rk, cscore in ks_keys:
            _revert_to_original(rk, cscore, "keep_sensitive")
            sensitivity_reverted += 1

    if use_reverse_rank and remaining_errs:
        print(
            f"  [reverse-rank] Next worst kept INT8: {remaining_errs[0][0]} "
            f"composite={remaining_errs[0][1]:.6f}"
        )
        print(
            f"  [reverse-rank] Best kept INT8: {remaining_errs[-1][0]} "
            f"composite={remaining_errs[-1][1]:.6f}"
        )
    elif use_keep_sensitive and not remaining_errs and blacklist_reverted > 0:
        print(
            "  [keep_sensitive] No remaining ranked layers after blacklist_keep"
        )

    if bias_correction and bias_corr_pending:
        print(
            f"\n[Bias Correction] Applying deltas to {len(bias_corr_pending)} "
            f"INT8 Linear layers (full Card 1)..."
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
    print(f"Converted layers: {converted_count}, Kept (other): {skipped_count}")
    print(f"Base-dtype keep (blacklist / non-diffusion / fp32): {bf16_keep}")
    if blacklist_reverted > 0:
        print(f"blacklist_keep reverted to original dtype: {blacklist_reverted}")
    if sensitivity_reverted > 0:
        print(f"keep_sensitive reverted to original dtype: {sensitivity_reverted}")
    print(f"Per-channel INT8 (Card 3 plain packs): {per_channel_int8}")
    print(f"Bias correction (Card 1): {bias_correction}")
    if bias_correction:
        print(f"  Bias-corrected INT8 layers: {bias_corr_applied}")
    print(f"Reverse blacklist_keep: {int(blacklist_keep)}")
    print(f"Reverse protect keep_sensitive: {int(keep_sensitive)}")
    if use_reverse_rank:
        print(
            "  Ranking: DM E[x^2] x HistCosine V5 (x NVFP4 x SVD axes measured) "
            "midrank weighted geometric (auto IQR/median weights)"
        )
    print(f"ConvRot FULL (Linear; Flux1 has no Conv2d): {enable_convrot}")
    if enable_convrot:
        print(
            f"  ConvRot Linear: {convrot_linear}, ConvRot Conv2d: {convrot_conv2d}, "
            f"plain INT8 (no eligible group size): {plain_int8_count}"
        )
    else:
        print(f"  plain INT8: {plain_int8_count}")

    save_file(new_state_dict, output_path, metadata=metadata)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Flux1 DiT INT8 convert with FULL ConvRot (Linear) ON by default. "
            "DualMonitor and calibration are one unit (run_card1_calib). "
            "--blacklist_keep N and --keep_sensitive M force DualMonitor calib "
            "(need --calib_file + --clip_path + --clip_l_path); both pick "
            "original-dtype layers from the same DualMonitor E[x^2] x HistCosine "
            "composite ranking (blacklist first, then sensitive). Fixed structure "
            "blacklist (img_in/txt_in/vector_in/img_mod/norm/...) always applies "
            "as safety net. Card 1 = --bias_correction (OFF skips bias delta only, "
            "not DualMonitor). Card 3 = --per_channel_int8 for non-ConvRot plain "
            "packs. Use --no-convrot for plain INT8 only. "
            "No Approach A / no VETO / no SDXL."
        )
    )
    parser.add_argument(
        "--model",
        "--input",
        dest="model",
        type=str,
        required=True,
        help="Path to Flux1 input .safetensors",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output .safetensors"
    )
    parser.add_argument(
        "--per_channel_int8",
        action="store_true",
        help=(
            "Card 3: per-out-channel amax/scale for plain (non-ConvRot) packs. "
            "ConvRot layers always use channelwise. Format stays int8_tensorwise."
        ),
    )
    parser.add_argument(
        "--bias_correction",
        action="store_true",
        help=(
            "Card 1 ON: DualMonitor consumer — bias += -(W_q - W) @ mu_x on "
            "ALL INT8 Linear. DualMonitor needs --calib_file + --clip_path + "
            "--clip_l_path. Bias OFF does not disable DualMonitor."
        ),
    )
    parser.add_argument(
        "--calib_file",
        type=str,
        default=None,
        help=(
            "Calibration prompts text file (one prompt per line). "
            "With the CLIP paths, DualMonitor always runs (bias ON or OFF). "
            "DualMonitor mu_x / E[x^2] feed Card 1, blacklist_keep, and keep_sensitive."
        ),
    )
    parser.add_argument(
        "--clip_path",
        type=str,
        default=None,
        help=(
            "T5XXL safetensors for Comfy CLIPType.FLUX (Flux needs both encoders). "
            "Required whenever calibration runs."
        ),
    )
    parser.add_argument(
        "--clip_l_path",
        type=str,
        default=None,
        help=(
            "clip_l safetensors for Comfy CLIPType.FLUX (Flux needs both encoders). "
            "Required whenever calibration runs."
        ),
    )
    parser.add_argument(
        "--comfy_path",
        type=str,
        default=None,
        help="ComfyUI root (must contain comfy/ldm/flux/model.py).",
    )
    parser.add_argument(
        "--num_calib_samples",
        type=int,
        default=32,
        help="DualMonitor calib samples (default 32).",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="DualMonitor DiT timestep sweeps per sample (default 25).",
    )
    parser.add_argument(
        "--latent_size",
        type=int,
        default=32,
        help="Calibration latent H/W (default 32, same as the Krea2 recipe).",
    )
    parser.add_argument(
        "--no-convrot",
        dest="enable_convrot",
        action="store_false",
        help="Disable ConvRot; pack plain int8_tensorwise only.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=_DEFAULT_GROUPSIZE,
        help=f"Preferred ConvRot Hadamard group size (default {_DEFAULT_GROUPSIZE}).",
    )
    parser.add_argument(
        "--blacklist_keep",
        type=int,
        default=0,
        help=(
            "Forces DualMonitor / run_card1_calib (needs --calib_file + both CLIP "
            "paths). After the fixed structure blacklist, revert top N INT8 "
            "weights ranked by the DualMonitor E[x^2] x HistCosine composite to "
            "original dtype. 0 = disabled."
        ),
    )
    parser.add_argument(
        "--keep_sensitive",
        type=int,
        default=0,
        help=(
            "Forces DualMonitor / run_card1_calib (needs --calib_file + both CLIP "
            "paths). After --blacklist_keep, revert next M layers from the same "
            "composite ranking to original dtype. Works with bias OFF. "
            "0 = disabled."
        ),
    )
    parser.set_defaults(enable_convrot=True)
    args = parser.parse_args()

    convert_to_int8(
        args.model,
        args.output,
        per_channel_int8=bool(args.per_channel_int8),
        bias_correction=bool(args.bias_correction),
        calib_file=args.calib_file,
        clip_path=args.clip_path,
        clip_l_path=args.clip_l_path,
        comfy_path=args.comfy_path,
        num_calib_samples=int(args.num_calib_samples),
        num_inference_steps=int(args.num_inference_steps),
        latent_size=int(args.latent_size),
        enable_convrot=bool(args.enable_convrot),
        group_size=int(args.group_size),
        blacklist_keep=int(args.blacklist_keep),
        keep_sensitive=int(args.keep_sensitive),
    )


if __name__ == "__main__":
    main()

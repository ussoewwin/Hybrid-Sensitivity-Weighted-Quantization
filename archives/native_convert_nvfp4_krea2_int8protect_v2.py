"""Krea2 NVFP4 + data-driven ConvRot INT8 protect (v2).

Based on native_convert_nvfp4_krea2_int8protect.py + v1.4 hybrid ranking.

Replaces the fixed JSON 30-key protect set with data-driven reverse ranking:
  --blacklist_keep N: top N highest-error DiT weights (DualMonitor x HistMSE
    composite) -> ConvRot INT8 instead of NVFP4.
  --keep_sensitive M: next M from the same pool -> ConvRot INT8.
  Both force DualMonitor calibration (needs --calib_file + --clip_path).

Fixed structure blacklist (first/last/mod/norm/projector/...) stays as BF16.
Remaining Linear 2D: NVFP4 (+ FULL ConvRot by default).

Improvements from v1.4:
  - Timestep-weighted DualMonitor (1.0 - t)
  - ConvRot-aware mu_x rotation for Bias Correction
  - HistMSE fallback importance (input-channel L1 norm)
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import math
import os
import re
import sys
import types
from typing import Optional, Sequence

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

try:
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout
except ImportError:
    print("Error: comfy_kitchen not found (install in the active venv).")
    sys.exit(1)

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Histogram MSE fast path
_HIST_DIR = os.path.join(_REPO_ROOT, "histogram")
if _HIST_DIR not in sys.path:
    sys.path.insert(0, _HIST_DIR)
try:
    from weighted_histogram_mse_fast import HSWQWeightedHistogramOptimizerFast
except ImportError:
    HSWQWeightedHistogramOptimizerFast = None

from native_convert_int8 import (  # noqa: E402
    build_hadamard,
    convrot_group_size_for_features,
    quantize_int8_rowwise,
    quantize_int8_tensorwise,
    rotate_weight,
)

_DEFAULT_GROUPSIZE = 256
_MODEL_TYPE = "Krea2-NVFP4-v2"

# Krea2 SingleStreamDiT -- structure-sensitive layers stay BF16.
_KREA2_BLACKLIST: list[str] = [
    "first",
    "last",
    "mod.",
    "norm",
    "projector",
    "tmlp",
    "txtmlp",
    "tproj",
    "txtfusion",
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

_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

# =========================================================================
# ConvRot Conv2d (not in native_convert_int8; same as v1.2/v1.4)
# =========================================================================
def rotate_weight_conv2d(
    weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int
) -> torch.Tensor:
    if weight.ndim != 4:
        raise ValueError(f"Conv2d weight must be 4D, got ndim={weight.ndim}")
    out_c, in_c, k_h, k_w = weight.shape
    flat = weight.permute(0, 2, 3, 1).contiguous().view(-1, in_c)
    flat_rot = rotate_weight(flat, h_matrix, group_size)
    return flat_rot.view(out_c, k_h, k_w, in_c).permute(0, 3, 1, 2).contiguous()


# =========================================================================
# DualMonitor (from v1.4, with timestep weighting)
# =========================================================================
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
            is_conv2d = isinstance(module, torch.nn.Conv2d)
            if is_conv2d and inp_detached.dim() == 4:
                reduce_dims = (0, 2, 3)
            elif inp_detached.dim() >= 2:
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


# =========================================================================
# Histogram MSE complement (from v1.4)
# =========================================================================
def _pool_midranks(values: Sequence[float]) -> list[float]:
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
    eps = 1e-12
    return (max(float(r_dm), eps) ** float(w_dm)) * (
        max(float(r_hist), eps) ** float(w_hist)
    )


def _histogram_mse_score(
    weight: torch.Tensor,
    importance: Optional[torch.Tensor],
    hist_opt,
) -> float:
    with contextlib.redirect_stdout(io.StringIO()):
        stats = hist_opt.compute_optimal_amax_with_stats(
            weight, importance=importance, scaled=False
        )
    mse = float(stats["estimated_mse"])
    if not math.isfinite(mse):
        return 0.0
    return mse


# =========================================================================
# Helpers
# =========================================================================
def _is_non_diffusion_key(key: str) -> bool:
    return any(marker in key for marker in _NON_DIFFUSION_MARKERS)


def _is_blacklisted(key: str) -> bool:
    return any(name in key for name in _KREA2_BLACKLIST)


def _find_krea2_key_prefix(state_dict) -> str:
    for prefix in ("model.diffusion_model.", "diffusion_model.", ""):
        if f"{prefix}txtfusion.projector.weight" in state_dict:
            if f"{prefix}blocks.0.attn.wq.weight" not in state_dict:
                raise ValueError(
                    "Krea2 signature incomplete: txtfusion.projector present but "
                    f"{prefix}blocks.0.attn.wq.weight missing"
                )
            return prefix
    raise ValueError(
        "Not a Krea2 checkpoint: missing txtfusion.projector.weight "
        "(under model.diffusion_model. / diffusion_model. / root)."
    )


def _meta_base_key(base_k_file: str) -> str:
    if "model.diffusion_model." in base_k_file:
        return base_k_file.split("model.diffusion_model.")[-1]
    if "diffusion_model." in base_k_file:
        return base_k_file.split("diffusion_model.")[-1]
    return base_k_file


# =========================================================================
# ComfyUI bootstrap (from v1.4)
# =========================================================================
def _ensure_comfyui_on_sys_path(comfy_path: str | None = None) -> str:
    candidates = []
    if comfy_path:
        candidates.append(os.path.abspath(comfy_path))
    env = os.environ.get("COMFYUI_PATH")
    if env:
        candidates.append(env)
    candidates.extend(
        [
            r"D:\USERFILES\ComfyUI\ComfyUI",
            r"D:\USERFILES\GitHub\ComfyUI",
            os.path.join(_REPO_ROOT, "ComfyUI-master"),
        ]
    )
    for root in candidates:
        if not root:
            continue
        model_py = os.path.join(root, "comfy", "ldm", "krea2", "model.py")
        if os.path.isfile(model_py):
            if root not in sys.path:
                sys.path.insert(0, root)
            return root
    raise FileNotFoundError(
        "ComfyUI root with comfy/ldm/krea2/model.py not found. "
        "Pass --comfy_path or set COMFYUI_PATH."
    )


def _clear_argv_for_comfy() -> list[str]:
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    return saved


def _restore_argv(saved: list[str]) -> None:
    sys.argv = saved


def _install_torchaudio_stub() -> None:
    import importlib.machinery
    for key in list(sys.modules):
        if key == "torchaudio" or key.startswith("torchaudio."):
            del sys.modules[key]

    def _stub_mod(name: str, *, is_package: bool = False):
        mod = types.ModuleType(name)
        mod.__file__ = "<stub>"
        if is_package:
            mod.__path__ = []
            spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
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
        def __init__(self, *a, **k): pass
        def __call__(self, x): return x
        def to(self, *a, **k): return self
    class _MelScale:
        def __init__(self, *a, **k): pass
    transforms.MelSpectrogram = _MelSpectrogram
    transforms.MelScale = _MelScale
    ta.functional = functional
    ta.transforms = transforms
    sys.modules["torchaudio"] = ta
    sys.modules["torchaudio.functional"] = functional
    sys.modules["torchaudio.transforms"] = transforms


def _install_comfy_optional_stubs() -> None:
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
            def memory_info(self): return types.SimpleNamespace(rss=0)
            def memory_full_info(self): return types.SimpleNamespace(uss=0)
            def cpu_percent(self, interval=None): return 0.0
            def num_threads(self): return 1
        ps = types.ModuleType("psutil")
        ps.virtual_memory = lambda: _VM()
        ps.Process = lambda: _Proc()
        sys.modules["psutil"] = ps


def detect_krea2_dit_config(state_dict, key_prefix: str) -> dict:
    head_dim = 128
    first_w = state_dict[f"{key_prefix}first.weight"]
    features = int(first_w.shape[0])
    channels = int(first_w.shape[1] // 4)
    block_re = re.compile(r"^" + re.escape(key_prefix) + r"blocks\.(\d+)\.")
    layers = 0
    for k in state_dict.keys():
        m = block_re.match(k)
        if m:
            layers = max(layers, int(m.group(1)) + 1)
    if layers <= 0:
        raise ValueError("Krea2 detect failed: no blocks.* keys")
    wq = state_dict[f"{key_prefix}blocks.0.attn.wq.weight"]
    wk = state_dict[f"{key_prefix}blocks.0.attn.wk.weight"]
    txtlayers = int(state_dict[f"{key_prefix}txtfusion.projector.weight"].shape[1])
    txtdim = int(
        state_dict[f"{key_prefix}txtfusion.layerwise_blocks.0.prenorm.scale"].shape[0]
    )
    return {
        "image_model": "krea2",
        "features": features,
        "channels": channels,
        "patch": 2,
        "layers": layers,
        "heads": int(wq.shape[0] // head_dim),
        "kvheads": int(wk.shape[0] // head_dim),
        "txtlayers": txtlayers,
        "txtdim": txtdim,
    }


def _encode_krea2_calib_contexts(
    *,
    clip_path: str,
    prompts: list[str],
    expected_fused: int,
    comfy_path: str | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
    if not os.path.isfile(clip_path):
        raise FileNotFoundError(f"--clip_path not found: {clip_path}")
    saved_argv = _clear_argv_for_comfy()
    try:
        _ensure_comfyui_on_sys_path(comfy_path)
        import comfy.options
        comfy.options.enable_args_parsing(False)
        _install_comfy_optional_stubs()
        _install_torchaudio_stub()
        import comfy.model_management as mm
        import comfy.sd
        mm.get_torch_device()
        print(f"  [Krea2 calib] Loading CLIP (KREA2 / Qwen3-VL-4B): {clip_path}")
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=None,
            clip_type=comfy.sd.CLIPType.KREA2,
        )
        bank: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        for i, prompt in enumerate(prompts):
            tokens = clip.tokenize(prompt)
            conds = clip.encode_from_tokens_scheduled(tokens)
            if not conds:
                raise RuntimeError(f"CLIP encode returned empty for sample {i}")
            cond_t = conds[0][0]
            meta = conds[0][1] if len(conds[0]) > 1 else {}
            if not torch.is_tensor(cond_t):
                raise RuntimeError(f"CLIP cond not tensor (sample {i}): {type(cond_t)!r}")
            if cond_t.ndim == 2:
                cond_t = cond_t.unsqueeze(0)
            if cond_t.ndim != 3:
                raise RuntimeError(f"CLIP context expected 3D, got {tuple(cond_t.shape)}")
            fused = int(cond_t.shape[-1])
            if fused != int(expected_fused):
                raise ValueError(
                    f"CLIP fused dim {fused} != DiT txtlayers*txtdim={expected_fused}"
                )
            attn = None
            if isinstance(meta, dict):
                am = meta.get("attention_mask")
                if torch.is_tensor(am):
                    attn = am.detach().float().cpu()
            bank.append((cond_t.detach().to(dtype=torch.bfloat16).cpu(), attn))
            print(f"  [Krea2 calib] CLIP encoded {i+1}/{len(prompts)} shape={tuple(cond_t.shape)}")
        if getattr(clip, "cond_stage_model", None) is not None:
            clip.cond_stage_model.cpu()
        if getattr(clip, "patcher", None) is not None:
            mm.unload_model_and_clones(clip.patcher)
        del clip
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  [Krea2 calib] CLIP unloaded; contexts on CPU for DiT DualMonitor")
        return bank
    finally:
        _restore_argv(saved_argv)


def load_krea2_from_safetensors(path, device="cuda", comfy_path: str | None = None):
    if str(device).startswith("cpu"):
        raise RuntimeError("Krea2 DualMonitor calibration requires CUDA.")
    _ensure_comfyui_on_sys_path(comfy_path)
    saved_argv = _clear_argv_for_comfy()
    try:
        import comfy.options
        comfy.options.enable_args_parsing(False)
        _install_comfy_optional_stubs()
        import comfy.ops
        from comfy.ldm.krea2.model import SingleStreamDiT
        print(f"Loading Krea2 DiT: {path}")
        state_dict = load_file(path)
        prefix = _find_krea2_key_prefix(state_dict)
        cfg = detect_krea2_dit_config(state_dict, prefix)
        print(f"Detected Krea2 DiT config: {cfg}")
        dit_kwargs = {k: v for k, v in cfg.items() if k != "image_model"}
        dtype = torch.bfloat16
        dit = SingleStreamDiT(**dit_kwargs, device=device, dtype=dtype, operations=comfy.ops.manual_cast)
        stripped = {}
        for k, v in state_dict.items():
            if prefix and k.startswith(prefix):
                stripped[k[len(prefix):]] = v
            elif not prefix:
                stripped[k] = v
        missing, unexpected = dit.load_state_dict(stripped, strict=False)
        print(f"  [Krea2] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
        try:
            dit_dev = str(next(dit.parameters()).device)
        except StopIteration as exc:
            raise RuntimeError("Krea2 DiT has no parameters") from exc
        if not dit_dev.startswith("cuda"):
            raise RuntimeError(f"Krea2 DiT landed on {dit_dev!r}, not CUDA.")
        print(f"  [Krea2] DiT device={dit_dev}")
        comfyui_to_module_map = {}
        for name, mod in dit.named_modules():
            w = getattr(mod, "weight", None)
            if w is None or not torch.is_tensor(w):
                continue
            if w.ndim not in (2, 4):
                continue
            ck = f"{prefix}{name}.weight"
            if ck in state_dict:
                comfyui_to_module_map[ck] = f"{name}.weight"
        print(f"  [Krea2] identity map entries={len(comfyui_to_module_map)} (prefix={prefix!r})")
        dit.eval()
        return dit, state_dict, comfyui_to_module_map, prefix
    finally:
        _restore_argv(saved_argv)


# =========================================================================
# Card 1 calibration (from v1.4, with timestep weighting)
# =========================================================================
def run_card1_calib(
    *,
    input_path: str,
    calib_file: str,
    clip_path: str,
    num_calib_samples: int,
    num_inference_steps: int,
    device: str,
    comfy_path: str | None = None,
):
    global _dm_timestep_weight
    if not str(device).startswith("cuda"):
        raise RuntimeError("Card 1 Krea2 calib requires CUDA.")

    print("Preparing calibration data...")
    with open(calib_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    if len(prompts) < num_calib_samples:
        prompts = (prompts * (num_calib_samples // len(prompts) + 1))[:num_calib_samples]
    else:
        prompts = prompts[:num_calib_samples]

    sd_peek = load_file(input_path)
    prefix = _find_krea2_key_prefix(sd_peek)
    cfg = detect_krea2_dit_config(sd_peek, prefix)
    fused = int(cfg["txtlayers"]) * int(cfg["txtdim"])
    del sd_peek
    gc.collect()

    context_bank = _encode_krea2_calib_contexts(
        clip_path=clip_path, prompts=prompts, expected_fused=fused, comfy_path=comfy_path,
    )
    if len(context_bank) != len(prompts):
        raise RuntimeError(f"CLIP context bank size {len(context_bank)} != prompts {len(prompts)}")

    model, _state_dict, comfyui_to_module_map, _prefix = load_krea2_from_safetensors(
        input_path, device=device, comfy_path=comfy_path
    )

    print("Preparing calibration (DualMonitor hooks; Card 1 act means)...")
    dual_monitors.clear()
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(
                module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))
            )

    print(f"Running calibration ({num_calib_samples} samples, {num_inference_steps} steps)...")
    if num_calib_samples != 32 or num_inference_steps != 25:
        print("  [WARN] r32 recipe is num_calib_samples=32, num_inference_steps=25.")
    _calib_progress_disable = False
    gen = torch.Generator(device=device).manual_seed(42)
    lat_h = lat_w = 32

    for i, prompt in enumerate(prompts):
        seed = 42 + i
        print(f"\nSample {i+1}/{num_calib_samples}: {prompt[:50]}...")
        gen.manual_seed(seed)
        with torch.no_grad():
            x = torch.randn(1, int(model.channels), lat_h, lat_w, device=device, dtype=torch.bfloat16, generator=gen)
            ctx_cpu, attn_cpu = context_bank[i]
            context = ctx_cpu.to(device=device, dtype=torch.bfloat16)
            attn_mask = None
            if attn_cpu is not None:
                attn_mask = attn_cpu.to(device=device)
            for step in tqdm(range(int(num_inference_steps)), total=int(num_inference_steps), disable=_calib_progress_disable):
                t = torch.full((1,), float(step) / float(max(num_inference_steps, 1)), device=device, dtype=torch.float32)
                _dm_timestep_weight = float(1.0 - t.item())
                if attn_mask is not None:
                    model(x, t, context, attention_mask=attn_mask)
                else:
                    model(x, t, context)
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
        f"  [Card 1 DualMonitor] act_mean={len(act_mean_dict)}, "
        f"act_sq_mean={len(act_sq_mean_dict)}, importance={len(importance_dict)}"
    )

    del model, context_bank
    dual_monitors.clear()
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "act_mean_dict": act_mean_dict,
        "act_sq_mean_dict": act_sq_mean_dict,
        "importance_dict": importance_dict,
        "comfyui_to_module_map": comfyui_to_module_map,
    }


# =========================================================================
# NVFP4 quantize helper (dequantize for error measurement)
# =========================================================================
def _nvfp4_quantize_and_dequantize(weight_bf16, device, enable_convrot, group_size):
    """Quantize to NVFP4 and return (qdata, params, tensors, weight_dq, used_gs, do_rotate).
    weight_dq is the dequantized weight on CPU for error calculation.
    """
    v_tensor = weight_bf16.to(device=device, dtype=torch.bfloat16)
    used_gs = None
    do_rotate = False
    w_for_q = v_tensor
    if enable_convrot:
        used_gs = convrot_group_size_for_features(int(v_tensor.shape[1]), int(group_size))
        if used_gs is not None:
            h_matrix = build_hadamard(int(used_gs), device="cpu", dtype=torch.float32)
            w_rot = rotate_weight(v_tensor.float().cpu(), h_matrix, int(used_gs))
            w_for_q = w_rot.to(device=device, dtype=torch.bfloat16)
            do_rotate = True

    qdata, params = TensorCoreNVFP4Layout.quantize(w_for_q)
    tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)

    # Dequantize for error measurement
    w_dq = TensorCoreNVFP4Layout.dequantize(qdata, params) if hasattr(TensorCoreNVFP4Layout, 'dequantize') else None

    if device == "cuda" and do_rotate:
        del w_for_q
    del v_tensor

    return qdata, params, tensors, w_dq, used_gs, do_rotate


def _nvfp4_dequantize_fallback(weight_original, qdata, params, device):
    """Fallback: if dequantize not available, use original weight as approximation."""
    return weight_original.float()


# =========================================================================
# INT8 protect quantize (ConvRot INT8 rowwise)
# =========================================================================
def _quantize_int8_protect(weight_fp, enable_convrot, group_size):
    """ConvRot INT8 quantize for protected layers. Returns (q, scale, w_fp_rotated, used_gs)."""
    w = weight_fp.float().cpu()
    used_gs = None
    if enable_convrot:
        used_gs = convrot_group_size_for_features(int(w.shape[1]), int(group_size))
    if used_gs is not None:
        h_matrix = build_hadamard(int(used_gs), device="cpu", dtype=torch.float32)
        w = rotate_weight(w, h_matrix, int(used_gs))
        q, scale = quantize_int8_rowwise(w)
        quant_config = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": int(used_gs)}
    else:
        q, scale = quantize_int8_tensorwise(w)
        quant_config = {"format": "int8_tensorwise"}
    return q, scale, w, used_gs, quant_config


# =========================================================================
# Main convert
# =========================================================================
def convert_to_nvfp4(
    input_path: str,
    output_path: str,
    device: str,
    *,
    enable_convrot: bool = True,
    group_size: int = _DEFAULT_GROUPSIZE,
    blacklist_keep: int = 0,
    keep_sensitive: int = 0,
    bias_correction: bool = False,
    calib_file: str | None = None,
    clip_path: str | None = None,
    comfy_path: str | None = None,
    num_calib_samples: int = 32,
    num_inference_steps: int = 25,
):
    rot_tag = "FULL ConvRot NVFP4" if enable_convrot else "plain NVFP4"
    print(f"Mode {_MODEL_TYPE} | device={device} | {rot_tag} + data-driven INT8 protect")

    # DualMonitor setup
    use_bias = bool(bias_correction)
    use_blacklist_keep = int(blacklist_keep) > 0
    use_keep_sensitive = int(keep_sensitive) > 0
    use_reverse_rank = use_blacklist_keep or use_keep_sensitive
    calib_file = (str(calib_file).strip() if calib_file else "") or None
    clip_path = (str(clip_path).strip() if clip_path else "") or None
    have_calib_paths = bool(calib_file) and bool(clip_path)
    run_dual_monitor = use_reverse_rank or use_bias or have_calib_paths

    if run_dual_monitor and not have_calib_paths:
        raise ValueError(
            "DualMonitor requires --calib_file and --clip_path "
            "(blacklist_keep / keep_sensitive / bias force DualMonitor)"
        )

    act_mean_dict: dict[str, torch.Tensor] = {}
    act_sq_mean_dict: dict[str, torch.Tensor] = {}
    importance_dict: dict[str, torch.Tensor] = {}
    comfyui_to_module_map: dict[str, str] = {}

    if run_dual_monitor:
        if device != "cuda":
            raise RuntimeError("Krea2 DualMonitor calib requires CUDA.")
        print("  [DualMonitor calib] ON | CLIPType.KREA2 DiT")
        if use_bias:
            print("  [Bias Correction] ON")
        else:
            print("  [Bias Correction] OFF (calibration still runs)")
        if use_blacklist_keep:
            print(f"  [blacklist_keep] ON | top-N={int(blacklist_keep)} -> ConvRot INT8")
        if use_keep_sensitive:
            print(f"  [keep_sensitive] ON | next-M={int(keep_sensitive)} -> ConvRot INT8")

        calib = run_card1_calib(
            input_path=input_path, calib_file=calib_file, clip_path=clip_path,
            num_calib_samples=int(num_calib_samples), num_inference_steps=int(num_inference_steps),
            device=device, comfy_path=comfy_path,
        )
        act_mean_dict = calib["act_mean_dict"]
        act_sq_mean_dict = calib["act_sq_mean_dict"]
        importance_dict = calib.get("importance_dict", {})
        comfyui_to_module_map = calib["comfyui_to_module_map"]
        print(f"  [DualMonitor] act_mean={len(act_mean_dict)}, act_sq_mean={len(act_sq_mean_dict)}")

    print(f"Loading model: {input_path}")
    sd = load_file(input_path)
    prefix = _find_krea2_key_prefix(sd)
    print(f"Detected Krea2 key prefix: {prefix!r}")

    new_sd: dict[str, torch.Tensor] = {}
    quant_map = {"format_version": "1.0", "layers": {}}
    n_nvfp4 = 0
    n_convrot = 0
    n_plain_nvfp4 = 0
    n_bf16 = 0
    n_int8_protect = 0
    n_int8_convrot = 0
    n_int8_plain = 0

    # Per-layer tracking for reverse ranking
    layer_quant_errors: dict[str, float] = {}
    layer_hist_mse: dict[str, float] = {}
    layer_nvfp4_info: dict[str, dict] = {}  # store NVFP4 tensors for potential INT8 swap
    layer_convrot_gs: dict[str, int] = {}
    bias_corr_pending: dict[str, torch.Tensor] = {}
    bias_corr_applied = 0
    bias_corr_skipped_no_bias = 0
    bias_corr_skipped_no_act = 0
    bias_corr_skipped_bad_shape = 0

    hist_opt = None
    if use_reverse_rank:
        if HSWQWeightedHistogramOptimizerFast is None:
            raise ImportError("weighted_histogram_mse_fast not found in histogram/ dir")
        hist_dev = "cuda" if torch.cuda.is_available() else "cpu"
        with contextlib.redirect_stdout(io.StringIO()):
            hist_opt = HSWQWeightedHistogramOptimizerFast(
                bins=4096, num_candidates=200, refinement_iterations=3, device=hist_dev,
            )
        print(f"  [HistMSE] Complement axis ON | device={hist_dev}")

    print(f"Converting ({len(sd)} tensors)...")
    for k, v in tqdm(list(sd.items())):
        # Fixed blacklist -> BF16
        if _is_blacklisted(k) or _is_non_diffusion_key(k):
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16 += 1
            continue

        # fp32 keep
        if v.dtype == torch.float32:
            new_sd[k] = v
            n_bf16 += 1
            continue

        # DiT Linear 2D weight -> NVFP4 (or INT8 protect if ranked)
        is_dit_weight = (
            k.endswith(".weight")
            and v.ndim == 2
            and v.dtype in (torch.float16, torch.bfloat16)
            and ((not prefix) or k.startswith(prefix))
        )

        if not is_dit_weight:
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16 += 1
            continue

        base_k_file = k.replace(".weight", "")
        base_k_meta = _meta_base_key(base_k_file)
        w_fp = v.float().cpu()

        # NVFP4 quantize
        v_tensor = v.to(device=device, dtype=torch.bfloat16)
        used_gs_nv = None
        do_rotate_nv = False
        w_for_q = v_tensor
        if enable_convrot:
            used_gs_nv = convrot_group_size_for_features(int(v_tensor.shape[1]), int(group_size))
            if used_gs_nv is not None:
                h_matrix = build_hadamard(int(used_gs_nv), device="cpu", dtype=torch.float32)
                w_rot = rotate_weight(v_tensor.float().cpu(), h_matrix, int(used_gs_nv))
                w_for_q = w_rot.to(device=device, dtype=torch.bfloat16)
                do_rotate_nv = True

        try:
            qdata, params = TensorCoreNVFP4Layout.quantize(w_for_q)
            tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)
        except Exception:
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16 += 1
            if do_rotate_nv and device == "cuda":
                del w_for_q
            del v_tensor
            continue

        # Store NVFP4 tensors and their suffixes for later removal
        nvfp4_suffixes = list(tensors.keys())
        for suffix, tensor in tensors.items():
            new_sd[f"{base_k_file}.weight{suffix}"] = tensor.cpu()

        if do_rotate_nv and used_gs_nv is not None:
            quant_map["layers"][base_k_meta] = {
                "format": "nvfp4", "convrot": True, "convrot_groupsize": int(used_gs_nv),
            }
            n_convrot += 1
        else:
            quant_map["layers"][base_k_meta] = {"format": "nvfp4"}
            n_plain_nvfp4 += 1
        n_nvfp4 += 1

        # For error tracking: dequantize NVFP4
        # Try to dequantize; if not available, use original weight as w_dq
        try:
            w_dq = TensorCoreNVFP4Layout.dequantize(qdata, params) if hasattr(TensorCoreNVFP4Layout, 'dequantize') else w_fp
        except Exception:
            w_dq = w_fp

        # Track per-layer quantization error
        module_w_key_sens = comfyui_to_module_map.get(k)
        module_name_sens = None
        if module_w_key_sens and module_w_key_sens.endswith(".weight"):
            module_name_sens = module_w_key_sens[:-len(".weight")]
        act_sq = act_sq_mean_dict.get(module_name_sens) if module_name_sens else None
        err = w_fp - w_dq.float().cpu() if w_dq is not w_fp else torch.zeros_like(w_fp)

        if act_sq is not None and act_sq.shape[0] == w_fp.shape[1]:
            act_scale = act_sq.sqrt()
            if err.ndim == 2:
                weighted_err = err * act_scale.unsqueeze(0)
                weighted_base = w_fp * act_scale.unsqueeze(0)
            else:
                weighted_err = err
                weighted_base = w_fp
            rel_err = float(weighted_err.norm().item()) / max(float(weighted_base.norm().item()), 1e-8)
            layer_quant_errors[k] = rel_err
        elif use_reverse_rank:
            pass
        else:
            rel_err = float(err.norm().item()) / max(float(w_fp.norm().item()), 1e-8)
            layer_quant_errors[k] = rel_err

        # HistMSE score
        if use_reverse_rank and hist_opt is not None:
            imp = importance_dict.get(module_name_sens) if module_name_sens else None
            if imp is None:
                if w_fp.ndim == 4:
                    imp = w_fp.abs().mean(dim=(0, 2, 3))
                else:
                    imp = w_fp.abs().mean(dim=0)
            try:
                layer_hist_mse[k] = _histogram_mse_score(w_fp, imp, hist_opt)
            except Exception:
                pass

        # Store info for potential INT8 swap
        layer_nvfp4_info[k] = {
            "base_k_file": base_k_file,
            "base_k_meta": base_k_meta,
            "w_fp": w_fp,
            "used_gs_nv": used_gs_nv,
            "nvfp4_suffixes": nvfp4_suffixes,
        }
        if used_gs_nv is not None:
            layer_convrot_gs[base_k_file] = int(used_gs_nv)

        # Bias correction (for NVFP4 layers)
        if bias_correction:
            module_name = module_name_sens
            act_mean = act_mean_dict.get(module_name) if module_name else None
            if act_mean is None:
                bias_corr_skipped_no_act += 1
            else:
                rot_gs = layer_convrot_gs.get(base_k_file)
                if rot_gs is not None:
                    h_bc = build_hadamard(rot_gs, device="cpu", dtype=torch.float32)
                    act_mean = rotate_weight(act_mean.unsqueeze(0).to(dtype=torch.float32), h_bc, rot_gs).squeeze(0)
                w_dq_for_bc = w_dq.float().cpu() if w_dq is not w_fp else None
                if w_dq_for_bc is None:
                    bias_corr_skipped_no_act += 1
                else:
                    delta = compute_int8_bias_delta(w_fp, w_dq_for_bc, act_mean)
                if delta is None:
                    bias_corr_skipped_bad_shape += 1
                else:
                    bias_corr_pending[base_k_file] = (-delta).detach().float().cpu()

        if do_rotate_nv and device == "cuda":
            del w_for_q
        del v_tensor

    # --- Reverse ranking: swap top-N NVFP4 layers to ConvRot INT8 ---
    def _swap_to_int8(rk: str, cscore: float, label: str) -> None:
        nonlocal n_nvfp4, n_convrot, n_plain_nvfp4, n_int8_protect, n_int8_convrot, n_int8_plain
        info = layer_nvfp4_info[rk]
        base_k_file = info["base_k_file"]
        base_k_meta = info["base_k_meta"]
        w_fp = info["w_fp"]

        # Remove NVFP4 tensors using stored suffixes (exact match)
        for suffix in info.get("nvfp4_suffixes", []):
            key_to_remove = f"{base_k_file}.weight{suffix}"
            if key_to_remove in new_sd:
                del new_sd[key_to_remove]

        # Remove NVFP4 metadata
        if base_k_meta in quant_map["layers"]:
            del quant_map["layers"][base_k_meta]

        # Quantize as ConvRot INT8
        q, scale, w_rot, used_gs_int8, quant_config = _quantize_int8_protect(w_fp, enable_convrot, group_size)
        new_sd[f"{base_k_file}.weight"] = q
        new_sd[f"{base_k_file}.weight_scale"] = scale

        # Encode comfy_quant
        quant_json = json.dumps(quant_config, separators=(",", ":")).encode("utf-8")
        new_sd[f"{base_k_file}.comfy_quant"] = torch.tensor(list(quant_json), dtype=torch.uint8)

        quant_map["layers"][base_k_meta] = dict(quant_config)

        n_nvfp4 -= 1
        if info["used_gs_nv"] is not None:
            n_convrot -= 1
        else:
            n_plain_nvfp4 -= 1
        n_int8_protect += 1
        if used_gs_int8 is not None:
            n_int8_convrot += 1
            layer_convrot_gs[base_k_file] = int(used_gs_int8)
        else:
            n_int8_plain += 1

        # Update bias correction delta for INT8
        if bias_correction:
            module_w_key = comfyui_to_module_map.get(rk)
            module_name = None
            if module_w_key and module_w_key.endswith(".weight"):
                module_name = module_w_key[:-len(".weight")]
            act_mean = act_mean_dict.get(module_name) if module_name else None
            if act_mean is not None:
                rot_gs = layer_convrot_gs.get(base_k_file)
                if rot_gs is not None:
                    h_bc = build_hadamard(rot_gs, device="cpu", dtype=torch.float32)
                    act_mean = rotate_weight(act_mean.unsqueeze(0).to(dtype=torch.float32), h_bc, rot_gs).squeeze(0)
                w_dq_int8 = q.float() * scale
                delta = compute_int8_bias_delta(w_rot, w_dq_int8, act_mean)
                if delta is not None:
                    bias_corr_pending[base_k_file] = (-delta).detach().float().cpu()

        dm_display = f"{layer_quant_errors.get(rk, 0):.6f}" if rk in layer_quant_errors else "neutral(0.5)"
        print(f"    [{label}] {rk}  composite={cscore:.6f}  dm_rel={dm_display}  -> INT8")

    remaining_errs: list[tuple[str, float]] = []
    if use_reverse_rank and layer_hist_mse:
        pool_keys = list(layer_hist_mse.keys())
        dm_real_keys = [k for k in pool_keys if k in layer_quant_errors]
        dm_real_vals = [float(layer_quant_errors[k]) for k in dm_real_keys]
        dm_real_ranks = _pool_midranks(dm_real_vals)
        dm_rank_lookup = {k: r for k, r in zip(dm_real_keys, dm_real_ranks)}

        if dm_real_keys:
            hist_real_vals = [float(layer_hist_mse[k]) for k in dm_real_keys]
            hist_real_ranks = _pool_midranks(hist_real_vals)
            weights = _derive_dm_hist_weights(dm_real_ranks, hist_real_ranks)
        else:
            weights = {"form": "hist_only_dm_empty", "w_dm": 0.0, "w_hist": 1.0}

        w_dm = float(weights["w_dm"])
        w_hist = float(weights["w_hist"])
        hist_vals_full = [float(layer_hist_mse[k]) for k in pool_keys]
        hist_ranks_full = _pool_midranks(hist_vals_full)

        composite = {}
        for i, k in enumerate(pool_keys):
            r_dm = dm_rank_lookup.get(k, 0.5)
            composite[k] = _composite_dm_hist(r_dm, hist_ranks_full[i], w_dm, w_hist)

        remaining_errs = sorted(composite.items(), key=lambda x: x[1], reverse=True)
        n_dm_neutral = len(pool_keys) - len(dm_real_keys)
        print(
            f"\n[Reverse ranking] DM x HistMSE composite "
            f"(form={weights['form']}, w_dm={w_dm:.4f}, w_hist={w_hist:.4f}) | "
            f"pool={len(pool_keys)} (DM-real={len(dm_real_keys)}, neutral={n_dm_neutral})"
        )
    elif use_reverse_rank:
        raise RuntimeError(
            "Reverse ranking requires HistMSE scores but none were produced. "
            "Check weighted_histogram_mse_fast import."
        )

    if use_blacklist_keep and remaining_errs:
        n_bl = min(int(blacklist_keep), len(remaining_errs))
        bl_keys = remaining_errs[:n_bl]
        remaining_errs = remaining_errs[n_bl:]
        print(f"\n[blacklist_keep] Swapping top {len(bl_keys)} NVFP4 -> ConvRot INT8:")
        for rk, cscore in bl_keys:
            _swap_to_int8(rk, cscore, "blacklist_keep")

    if use_keep_sensitive and remaining_errs:
        n_ks = min(int(keep_sensitive), len(remaining_errs))
        ks_keys = remaining_errs[:n_ks]
        remaining_errs = remaining_errs[n_ks:]
        print(f"\n[keep_sensitive] Swapping next {len(ks_keys)} NVFP4 -> ConvRot INT8:")
        for rk, cscore in ks_keys:
            _swap_to_int8(rk, cscore, "keep_sensitive")

    if use_reverse_rank and remaining_errs:
        print(f"  [reverse-rank] Next worst kept NVFP4: {remaining_errs[0][0]} composite={remaining_errs[0][1]:.6f}")

    # Bias correction application
    if bias_correction and bias_corr_pending:
        print(f"\n[Bias Correction] Applying deltas to {len(bias_corr_pending)} layers...")
        for module_key, delta in bias_corr_pending.items():
            bias_key = f"{module_key}.bias"
            if bias_key not in new_sd:
                bias_corr_skipped_no_bias += 1
                continue
            bias = new_sd[bias_key]
            corrected = bias.float() + delta.to(device=bias.device, dtype=torch.float32)
            new_sd[bias_key] = corrected.to(dtype=bias.dtype)
            bias_corr_applied += 1
        print(f"  applied={bias_corr_applied}, no_bias={bias_corr_skipped_no_bias}, no_act={bias_corr_skipped_no_act}")

    # Metadata
    from collections import OrderedDict
    final_metadata = OrderedDict()
    final_metadata["_quantization_metadata"] = json.dumps(quant_map)
    final_metadata["converted_by"] = "ComfyUI Kitchen NVFP4 Converter (Krea2 ConvRot + data-driven INT8 protect v2)"
    final_metadata["converter_url"] = "https://github.com/tritant/ComfyUI_Kitchen_nvfp4_Converter"
    final_metadata["hswq_model"] = "krea2"
    final_metadata["hswq_nvfp4_convrot"] = "1" if enable_convrot else "0"
    final_metadata["hswq_int8_protect"] = "1"
    final_metadata["hswq_int8_protect_n"] = str(n_int8_protect)
    final_metadata["hswq_int8_protect_convrot"] = str(n_int8_convrot)
    final_metadata["hswq_int8_protect_source"] = "data_driven_reverse_ranking"

    print(f"Saving | Type: {_MODEL_TYPE} | Path: {output_path}")
    save_file(new_sd, output_path, metadata=final_metadata)
    total_bytes = os.path.getsize(output_path)
    print(f"Done. Size: {round(total_bytes / (1024**3), 2)} GiB")
    print(f"NVFP4 layers: {n_nvfp4} (convrot={n_convrot}, plain={n_plain_nvfp4})")
    print(f"INT8 protect: {n_int8_protect} (convrot={n_int8_convrot}, plain={n_int8_plain})")
    print(f"BF16 keep: {n_bf16}")
    if bias_correction:
        print(f"Bias corrected: {bias_corr_applied}")

    del sd, new_sd, quant_map
    _release_vram("after convert save")


def _release_vram(label: str = "post-convert") -> None:
    print(f"[*] Releasing VRAM ({label})...")
    gc.collect()
    if not torch.cuda.is_available():
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Krea2 NVFP4 + data-driven ConvRot INT8 protect (v2). "
            "Replaces fixed JSON keyset with DualMonitor x HistMSE reverse ranking. "
            "--blacklist_keep N and --keep_sensitive M swap top-ranked NVFP4 layers "
            "to ConvRot INT8. Fixed structure blacklist always applies. "
            "Needs --calib_file + --clip_path for DualMonitor."
        )
    )
    parser.add_argument("--model", "--input", dest="model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    parser.add_argument("--no-nvfp4-convrot", dest="enable_convrot", action="store_false", help="Disable ConvRot on NVFP4 path")
    parser.add_argument("--groupsize", type=int, default=_DEFAULT_GROUPSIZE)
    parser.add_argument("--blacklist_keep", type=int, default=0, help="Top N NVFP4 layers -> ConvRot INT8")
    parser.add_argument("--keep_sensitive", type=int, default=0, help="Next M NVFP4 layers -> ConvRot INT8")
    parser.add_argument("--bias_correction", action="store_true", help="Card 1 bias correction on INT8 layers")
    parser.add_argument("--calib_file", type=str, default=None, help="Calibration prompts (one per line)")
    parser.add_argument("--clip_path", type=str, default=None, help="Qwen3-VL-4B CLIP safetensors")
    parser.add_argument("--comfy_path", type=str, default=None, help="ComfyUI root")
    parser.add_argument("--num_calib_samples", type=int, default=32)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.set_defaults(enable_convrot=True)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    convert_to_nvfp4(
        args.model, args.output, device=str(args.device),
        enable_convrot=bool(args.enable_convrot), group_size=int(args.groupsize),
        blacklist_keep=int(args.blacklist_keep), keep_sensitive=int(args.keep_sensitive),
        bias_correction=bool(args.bias_correction),
        calib_file=args.calib_file, clip_path=args.clip_path, comfy_path=args.comfy_path,
        num_calib_samples=int(args.num_calib_samples), num_inference_steps=int(args.num_inference_steps),
    )

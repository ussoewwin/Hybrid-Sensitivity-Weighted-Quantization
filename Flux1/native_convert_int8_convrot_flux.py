"""Flux1 DiT → ComfyUI native int8_tensorwise FULL ConvRot converter (hswq not used).

Flux1 only (model.diffusion_model.* layout, e.g. redcraftHybridH3A2A_realreveal5.safetensors).
"native" = a plain compressed model that does not use hswq (the MEMORY.md definition).

Pack (ComfyUI MixedPrecisionOps + comfy_kitchen TensorWiseINT8Layout):
  <layer>.weight           int8
  <layer>.weight_scale     float32
      ConvRot Linear:      [out, 1] (row-wise) — kitchen online act rotate
      plain INT8:          scalar (tensorwise)
  <layer>.comfy_quant      uint8 JSON (compact)
      plain:  {"format":"int8_tensorwise"}
      ConvRot:{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":N}

FULL ConvRot (default ON; --no-convrot for plain INT8):
  Linear 2D:  W_rot = W @ H^T (per-group Hadamard, power of 4) -> row-wise INT8 -> stamp.
  Flux1 is all Linear (no Conv2d). Where 4-D weights appear, treat them as rotate_weight_conv2d does.
  A layer whose in_features is not divisible by the power-of-4 group size -> plain tensorwise INT8 (fallback).

Scope:
  Only .weight under model.diffusion_model. (ndim>=2, fp16/bf16/fp32) is converted.
  1-D tensors (bias / scale / norm) and other prefixes (conditioner etc.) are kept as they are.

Bias Correction (Card 1), --bias_correction: ON folds bias += -(W_q - W) @ mu_x into every
  quantised Linear/Conv2d bias. It needs a ComfyUI run (t5xxl + clip_l + DiT forward +
  DualMonitor): --calib_file, --clip_path, --clip_l_path, --comfy_path, CUDA.
  OFF (default) leaves every bias untouched and skips calibration entirely.
  The pack aborts instead of saving when 0 layers were corrected.

Post-convert bench (default ON): after save, subprocess runs
  benchmark/flux1_convrot_int8_traj_compare.py (trajectory, official fixed 25 seeds).
  --clip_path (t5xxl), --clip_l_path and --comfy_path are required.
  skip with --no-bench.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import types

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

_DEFAULT_GROUPSIZE = 256
_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def _default_repo_root() -> str:
    """Locate the repo root by walking up until native_convert_int8.py is found."""
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(8):
        if os.path.isfile(os.path.join(d, "native_convert_int8.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.abspath(os.path.join(here, os.pardir))


_REPO_ROOT = _default_repo_root()
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Hadamard / ConvRot core (self-contained; same math as comfy_kitchen ConvRot)
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


def convrot_group_size_for_features(n: int, preferred: int = _DEFAULT_GROUPSIZE) -> int | None:
    """Largest power-of-4 group size <= preferred that divides n (or None)."""
    if n < 4:
        return None
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def rotate_weight(weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int) -> torch.Tensor:
    """Offline Linear: W_rot = W @ H^T (group-wise). Matches comfy_kitchen._rotate_weight."""
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
    """Offline Conv2d: rotate along in_channels. weight (O, I, kH, kW)."""
    if weight.ndim != 4:
        raise ValueError(f"Conv2d weight must be 4D, got ndim={weight.ndim}")
    out_c, in_c, k_h, k_w = weight.shape
    flat = weight.permute(0, 2, 3, 1).contiguous().view(-1, in_c)
    flat_rot = rotate_weight(flat, h_matrix, group_size)
    return flat_rot.view(out_c, k_h, k_w, in_c).permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# INT8 packing
# ---------------------------------------------------------------------------
def pack_tensorwise(weight: torch.Tensor):
    """Symmetric per-tensor INT8: scale = amax / 127."""
    w = weight.float()
    amax = max(float(w.abs().max().item()), 1e-6)
    scale = amax / 127.0
    q = (w / scale).round().clamp(-127, 127).to(torch.int8)
    return q, torch.tensor(scale, dtype=torch.float32)


def pack_channelwise(weight: torch.Tensor):
    """Per-out-channel INT8 (Linear [O,1] / Conv [O,1,1,1])."""
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


def _encode_comfy_quant(config: dict) -> torch.Tensor:
    return torch.tensor(
        list(json.dumps(config, separators=(",", ":")).encode("utf-8")),
        dtype=torch.uint8,
    )


# ---------------------------------------------------------------------------
# VRAM / bench helpers
# ---------------------------------------------------------------------------
def _release_vram(label: str = "post-convert") -> None:
    print(f"[*] Releasing VRAM ({label})...")
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


# ---------------------------------------------------------------------------
# Card 1 bias correction (DualMonitor mu_x -> bias += -(W_q - W) @ mu_x)
# Copied from Flux1/hswq_convrot_w4a8_flux1_v1.0.py so this native converter
# carries the same implementation instead of re-deriving it.
# ---------------------------------------------------------------------------
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

def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _repo_root() -> str:
    here = _script_dir()
    repo = os.path.normpath(os.path.join(here, ".."))
    if os.path.isdir(os.path.join(repo, "ComfyUI-master")):
        return repo
    return here

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


def run_post_convert_flux_int8_bench(
    *,
    script_dir: str,
    fp16_path: str,
    int8_path: str,
    clip_path: str,
    clip_l_path: str,
    comfy_path: str,
) -> int:
    """After save: subprocess benchmark/flux1_convrot_int8_traj_compare.py.

    Bench argv:
      --fp16 <input> --int8 <INT8> --clip_path <t5xxl> --clip_l_path <clip_l>
      --comfy_path <ComfyUI root>
    Seeds: the bench default, the official fixed 25 (42-start list). No --seeds
    override is passed here.
    """
    bench_script = os.path.join(
        script_dir, "benchmark", "flux1_convrot_int8_traj_compare.py"
    )
    if not os.path.isfile(bench_script):
        print(f"Error: bench script not found: {bench_script}")
        return 1
    cmd = [
        sys.executable,
        bench_script,
        "--fp16", fp16_path,
        "--int8", int8_path,
        "--clip_path", clip_path,
        "--clip_l_path", clip_l_path,
        "--comfy_path", comfy_path,
    ]
    print("Running trajectory bench: " + " ".join(cmd))
    return subprocess.run(cmd).returncode

def is_flux1_matmul_weight(key: str, tensor: torch.Tensor) -> bool:
    """Flux1 DiT matmul weight: model.diffusion_model.* .weight, ndim>=2, float."""
    if not key.startswith("model.diffusion_model."):
        return False
    if not key.endswith(".weight"):
        return False
    if tensor.ndim < 2:
        return False
    return tensor.dtype in (torch.float16, torch.float32, torch.bfloat16)


def convert_to_int8(
    input_path: str,
    output_path: str,
    enable_convrot: bool = True,
    group_size: int = _DEFAULT_GROUPSIZE,
    per_channel_int8: bool = False,
    bias_correction: bool = False,
    act_mean_dict: dict | None = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {input_path}")
    state_dict = load_file(input_path)

    new_state_dict = {}
    quant_meta_layers = {}
    converted_count = 0
    skipped_count = 0
    plain_int8_count = 0
    convrot_linear = 0
    convrot_conv2d = 0
    kept_float_count = 0
    bias_pending: dict[str, torch.Tensor] = {}
    bias_applied = 0
    bias_no_act = 0
    bias_bad_shape = 0

    rot_tag = " + ConvRot(Linear+Conv2d)" if enable_convrot else ""
    plain_tag = "per-channel" if per_channel_int8 else "tensorwise"
    print(
        f"Converting Flux1 DiT matmul weights to INT8 "
        f"({plain_tag} amax/127{rot_tag})..."
    )

    def _pack_plain(w: torch.Tensor):
        """Plain (non-ConvRot) pack: channelwise when --per_channel_int8."""
        if per_channel_int8:
            return pack_channelwise(w)
        return pack_tensorwise(w)

    for key, tensor in tqdm(state_dict.items()):
        if not is_flux1_matmul_weight(key, tensor):
            new_state_dict[key] = tensor
            kept_float_count += 1
            continue

        w_fp = tensor.float()
        module_key = key[: -len(".weight")]
        quant_config: dict
        q: torch.Tensor
        scale: torch.Tensor
        rot_gs = None

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
                rot_gs = int(used_gs)
                convrot_linear += 1
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
            else:
                q, scale = _pack_plain(w_fp)
                quant_config = {"format": "int8_tensorwise"}
                plain_int8_count += 1
        else:
            q, scale = _pack_plain(w_fp)
            quant_config = {"format": "int8_tensorwise"}
            plain_int8_count += 1

        new_state_dict[key] = q
        new_state_dict[f"{module_key}.weight_scale"] = scale
        new_state_dict[f"{module_key}.comfy_quant"] = _encode_comfy_quant(
            quant_config
        )
        quant_meta_layers[module_key] = dict(quant_config)
        converted_count += 1

        if bias_correction and act_mean_dict is not None:
            bias_key = f"{module_key}.bias"
            act_mean = act_mean_dict.get(module_key)
            if act_mean is None:
                for _pfx in ("model.diffusion_model.", "diffusion_model."):
                    if module_key.startswith(_pfx):
                        act_mean = act_mean_dict.get(module_key[len(_pfx):])
                        if act_mean is not None:
                            break
            if act_mean is None or bias_key not in state_dict:
                bias_no_act += 1
            else:
                w_dq = q.float() * (
                    scale.float().reshape(-1, 1)
                    if scale.numel() > 1
                    else scale.float()
                )
                mu = act_mean.float()
                if rot_gs is not None:
                    h_bc = build_hadamard(rot_gs, device="cpu", dtype=torch.float32)
                    mu = rotate_weight(mu.unsqueeze(0), h_bc, rot_gs).squeeze(0)
                delta = compute_int8_bias_delta(w_fp, w_dq, mu)
                if delta is None:
                    bias_bad_shape += 1
                else:
                    bias_pending[module_key] = (-delta).detach().float().cpu()
                    bias_applied += 1

    if bias_correction:
        for module_key, neg_delta in bias_pending.items():
            bias_key = f"{module_key}.bias"
            if bias_key not in new_state_dict:
                continue
            orig = new_state_dict[bias_key]
            new_state_dict[bias_key] = (
                orig.float() + neg_delta.to(orig.device)
            ).to(orig.dtype)
        if bias_applied == 0:
            raise RuntimeError(
                "--bias_correction produced 0 corrected layers; refusing to "
                "write a pack labelled bias-corrected (no DualMonitor act mean "
                "matched a quantised layer)"
            )
        print(
            f"  [Bias Correction Card 1] applied={bias_applied} "
            f"no_act_mean={bias_no_act} bad_shape={bias_bad_shape}"
        )

    metadata = {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": quant_meta_layers}
        )
    }

    print(f"Saving to: {output_path}")
    print(f"Converted layers: {converted_count}, Kept non-matmul: {kept_float_count}")
    print(f"ConvRot FULL: {enable_convrot}")
    if enable_convrot:
        print(
            f"  ConvRot Linear: {convrot_linear}, ConvRot Conv2d: {convrot_conv2d}, "
            f"plain INT8 (no eligible group size): {plain_int8_count}"
        )
    else:
        print(f"  plain INT8: {plain_int8_count}")
    print(f"  plain scale mode (--per_channel_int8): {per_channel_int8}")

    save_file(new_state_dict, output_path, metadata=metadata)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Flux1 DiT INT8 convert with FULL ConvRot (Linear) ON by default. "
            "plain native compression (hswq not used). Bias Correction is out of scope for this script."
        )
    )
    parser.add_argument(
        "--model",
        "--input",
        dest="model",
        type=str,
        required=True,
        help="Path to input Flux1 .safetensors (model.diffusion_model.* layout)",
    )
    parser.add_argument("--output", type=str, required=True, help="Path to output .safetensors")
    parser.add_argument(
        "--convrot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "FULL ConvRot on Linear/Conv2d when in_dim divisible by a power-of-4 "
            "group size: rotate + per-channel scale + stamp. Default ON; "
            "pass --no-convrot for plain tensorwise."
        ),
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=_DEFAULT_GROUPSIZE,
        help=f"ConvRot Hadamard group size (power of 4, default {_DEFAULT_GROUPSIZE})",
    )
    parser.add_argument(
        "--per_channel_int8",
        action="store_true",
        help=(
            "Plain (non-ConvRot) layers use per-out-channel amax/scale instead "
            "of per-tensor. ConvRot layers are channelwise by construction; "
            "the stamp stays int8_tensorwise."
        ),
    )
    parser.add_argument(
        "--bias_correction",
        action="store_true",
        help=(
            "Card 1 ON: DualMonitor consumer - bias += -(W_q - W) @ mu_x on "
            "every quantised Linear/Conv2d. Needs --calib_file + --clip_path + "
            "--clip_l_path + --comfy_path and CUDA."
        ),
    )
    parser.add_argument(
        "--calib_file",
        type=str,
        default=None,
        help="Calibration prompts text file (one prompt per line).",
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
        "--clip_path",
        type=str,
        default=None,
        help="T5XXL text encoder path (required when --bench or --bias_correction)",
    )
    parser.add_argument(
        "--clip_l_path",
        type=str,
        default=None,
        help="clip_l text encoder path (required when --bench or --bias_correction)",
    )
    parser.add_argument(
        "--comfy_path",
        type=str,
        default=None,
        help="ComfyUI root path (required when --bench or --bias_correction)",
    )
    parser.add_argument(
        "--bench",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After save, run benchmark/flux1_convrot_int8_traj_compare.py "
            "(requires --clip_path/--clip_l_path/--comfy_path; seeds = the "
            "official fixed 25). "
            "Pass --no-bench to skip."
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)
    if args.groupsize < 4 or (args.groupsize & (args.groupsize - 1)) != 0:
        print(f"Error: --groupsize must be a power of 4 (>=4), got {args.groupsize}")
        sys.exit(1)
    if math.log(args.groupsize, 4) % 1 != 0:
        print(f"Error: --groupsize must be a power of 4, got {args.groupsize}")
        sys.exit(1)

    if args.bias_correction:
        miss_bc = []
        if not args.calib_file:
            miss_bc.append("--calib_file")
        if not args.clip_path:
            miss_bc.append("--clip_path")
        if not args.clip_l_path:
            miss_bc.append("--clip_l_path")
        if not args.comfy_path:
            miss_bc.append("--comfy_path")
        if miss_bc:
            print(
                "Error: --bias_correction requires " + ", ".join(miss_bc)
                + " (there is no DualMonitor-less path)"
            )
            sys.exit(1)
        for flag, path in (
            ("--calib_file", args.calib_file),
            ("--clip_path", args.clip_path),
            ("--clip_l_path", args.clip_l_path),
        ):
            if not os.path.isfile(path):
                print(f"Error: {flag} not found: {path}")
                sys.exit(1)

    if args.bench:
        missing = []
        if not args.clip_path:
            missing.append("--clip_path")
        if not args.clip_l_path:
            missing.append("--clip_l_path")
        if not args.comfy_path:
            missing.append("--comfy_path")
        if missing:
            print(
                "Error: post-convert bench requires "
                + ", ".join(missing)
                + " (pass --no-bench to convert without bench)"
            )
            sys.exit(1)
        if not os.path.isfile(args.clip_path):
            print(f"Error: --clip_path not found: {args.clip_path}")
            sys.exit(1)
        if not os.path.isfile(args.clip_l_path):
            print(f"Error: --clip_l_path not found: {args.clip_l_path}")
            sys.exit(1)
        if not os.path.isdir(args.comfy_path):
            print(f"Error: --comfy_path not found: {args.comfy_path}")
            sys.exit(1)

    act_mean_dict = None
    if args.bias_correction:
        calib = run_card1_calib(
            input_path=args.model,
            calib_file=args.calib_file,
            clip_path=args.clip_path,
            clip_l_path=args.clip_l_path,
            num_calib_samples=int(args.num_calib_samples),
            num_inference_steps=int(args.num_inference_steps),
            device="cuda" if torch.cuda.is_available() else "cpu",
            latent_size=int(args.latent_size),
            comfy_path=args.comfy_path,
        )
        act_mean_dict = calib["act_mean_dict"]
        _release_vram("after Card 1 calib")

    convert_to_int8(
        args.model,
        args.output,
        enable_convrot=bool(args.convrot),
        group_size=int(args.groupsize),
        per_channel_int8=bool(args.per_channel_int8),
        bias_correction=bool(args.bias_correction),
        act_mean_dict=act_mean_dict,
    )

    if args.bench:
        bench_rc = run_post_convert_flux_int8_bench(
            script_dir=_REPO_ROOT,
            fp16_path=args.model,
            int8_path=args.output,
            clip_path=args.clip_path,
            clip_l_path=args.clip_l_path,
            comfy_path=args.comfy_path,
        )
        if bench_rc != 0:
            print(f"[FATAL] Post-convert bench exited with code {bench_rc}")
            sys.exit(bench_rc)
    else:
        print("[*] Post-convert bench skipped (--no-bench)")

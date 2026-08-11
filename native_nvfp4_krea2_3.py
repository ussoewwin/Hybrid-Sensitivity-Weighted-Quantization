"""Krea2-only NVFP4 converter — self-contained (no project imports).

Standalone script. Does NOT import native_convert_int8, auto_int8_nvfp4_hybrid,
or any other project file. Only external deps: torch, safetensors, tqdm,
comfy_kitchen, comfy (ComfyUI).

Improvement stack (all optional, composable):
  1. SmoothQuant (--calib_file + --clip_path): per-channel scale migration
  2. ConvRot (--convrot): offline Hadamard weight rotation
  3. NVFP4 quantize on the well-conditioned result

All fp32 math (SmoothQuant + ConvRot) is done in float32; bf16 cast happens
only right before NVFP4 quantize.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.machinery
import io
import json
import math
import os
import re
import sys
import time
import types
from collections import OrderedDict
from typing import Optional

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout
except ImportError:
    print("Error: comfy_kitchen not found (install in the active venv).")
    sys.exit(1)

_MODEL_TYPE = "Krea2"
_SMOOTHQUANT_ALPHA_DEFAULT = 0.5
_DEFAULT_GROUPSIZE = 256

# =========================================================================
# Krea2 constants
# =========================================================================

_KREA2_BLACKLIST: list[str] = [
    "first", "last", "mod.", "norm", "projector",
    "tmlp", "tproj", "bias", "vae.", "text_encoders",
]
_KREA2_FP8_LAYERS: list[str] = []

_NON_DIFFUSION_MARKERS: tuple[str, ...] = (
    "conditioner.", "cond_stage_model.", "text_encoders.",
    "text_encoder.", "text_encoder_2.", "text_encoder_3.",
    "text_model.", "text_projection", "logit_scale",
    "clip_l.", "clip_g.", "t5xxl.", "first_stage_model.", "vae.",
)


def _is_non_diffusion_key(key: str) -> bool:
    return any(marker in key for marker in _NON_DIFFUSION_MARKERS)


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
# Hadamard helpers (inlined from native_convert_int8 — no import)
# =========================================================================

_HADAMARD_CACHE: dict = {}


def build_hadamard(size, device="cpu", dtype=None):
    """Normalized regular Hadamard (power-of-4), same as comfy_kitchen ConvRot."""
    if dtype is None:
        dtype = torch.float32
    device = torch.device(device) if not isinstance(device, torch.device) else device
    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=torch.float32, device=device,
    )
    h_matrix = h4
    current_size = 4
    while current_size < size:
        h_matrix = torch.kron(h_matrix, h4)
        current_size *= 4
    h_matrix = h_matrix / (size ** 0.5)
    if dtype != torch.float32:
        h_matrix = h_matrix.to(dtype=dtype)
    _HADAMARD_CACHE[cache_key] = h_matrix
    return h_matrix


def convrot_group_size_for_features(n: int, preferred: int = _DEFAULT_GROUPSIZE):
    """Largest power-of-4 group size <= preferred that divides n (or None)."""
    if n < 4:
        return None
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def rotate_weight(weight, h_matrix, group_size):
    """Offline Linear: W_rot = W @ H^T (group-wise)."""
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(f"in_features {in_features} not divisible by group_size {group_size}")
    group_count = in_features // group_size
    weight_grouped = weight.view(out_features, group_count, group_size)
    return torch.matmul(
        weight_grouped, h_matrix.T.to(dtype=weight.dtype, device=weight.device)
    ).reshape(weight.shape)


# =========================================================================
# SmoothQuant helpers
# =========================================================================

def _compute_smoothquant_scale(act_sq, weight, alpha=_SMOOTHQUANT_ALPHA_DEFAULT):
    """s_j = (act_rms_j ^ α) / (weight_abs_max_j ^ α), normalized to median=1.0."""
    act_rms = act_sq.to(torch.float32).sqrt().clamp(min=1e-8)
    weight_abs_max = weight.to(torch.float32).abs().amax(dim=0).clamp(min=1e-8)
    s = (act_rms ** alpha) / (weight_abs_max ** alpha)
    s_median = s.median().clamp(min=1e-8)
    s = s / s_median
    s = s.clamp(min=1e-4, max=1e4)
    return s


def _apply_smoothquant(weight, scale):
    """W' = W * s (scale each column j of W by s_j)."""
    return weight * scale.unsqueeze(0).to(weight.dtype)


# =========================================================================
# Calibration pipeline (inlined from auto_int8_nvfp4_hybrid — no import)
# =========================================================================

class DualMonitor:
    def __init__(self):
        self.output_sum = 0.0
        self.output_sq_sum = 0.0
        self.count = 0
        self.channel_importance = None
        self.channel_act_mean = None
        self.channel_act_sq_mean = None
        self.act_amax = 0.0

    def update(self, input_tensor, output_tensor, module=None, weight=1.0):
        with torch.no_grad():
            out_detached = output_tensor.detach().float()
            out_clamped = torch.clamp(out_detached, -65504.0, 65504.0)
            mean_val = out_clamped.mean().item()
            sq_mean_val = (out_clamped ** 2).mean().item()
            if math.isfinite(mean_val) and math.isfinite(sq_mean_val):
                self.output_sum += mean_val
                self.output_sq_sum += sq_mean_val
            inp_detached = input_tensor.detach().float()
            amax_val = float(inp_detached.abs().amax().item())
            if math.isfinite(amax_val) and amax_val > self.act_amax:
                self.act_amax = amax_val
            is_conv2d = isinstance(module, torch.nn.Conv2d)
            if is_conv2d and inp_detached.dim() == 4:
                reduce_dims = (0, 2, 3)
            elif inp_detached.dim() >= 2:
                reduce_dims = tuple(range(inp_detached.dim() - 1))
            else:
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
                self.channel_importance = (self.channel_importance * self.count + current_imp * w) / (self.count + w)
                self.channel_act_mean = (self.channel_act_mean * self.count + current_act * w) / (self.count + w)
                self.channel_act_sq_mean = (self.channel_act_sq_mean * self.count + current_sq * w) / (self.count + w)
            self.count += w


_dual_monitors: dict = {}
_dm_timestep_weight: float = 1.0


def _hook_fn(module, input, output, name):
    if name not in _dual_monitors:
        _dual_monitors[name] = DualMonitor()
    _dual_monitors[name].update(input[0], output, module, weight=_dm_timestep_weight)


def _clear_argv_for_comfy():
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    return saved


def _restore_argv(saved):
    sys.argv = saved


def _install_torchaudio_stub():
    for key in list(sys.modules):
        if key == "torchaudio" or key.startswith("torchaudio."):
            del sys.modules[key]

    def _stub(name, pkg=False):
        m = types.ModuleType(name)
        m.__file__ = "<stub>"
        if pkg:
            m.__path__ = []
            spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            spec.submodule_search_locations = []
        else:
            spec = importlib.machinery.ModuleSpec(name, loader=None)
        m.__spec__ = spec
        return m

    ta = _stub("torchaudio", True)
    func = _stub("torchaudio.functional")
    func.resample = lambda w, o, n, *a, **k: w
    tr = _stub("torchaudio.transforms")

    class _MS:
        def __init__(self, *a, **k): pass
        def __call__(self, x): return x
        def to(self, *a, **k): return self

    class _ML:
        def __init__(self, *a, **k): pass

    tr.MelSpectrogram = _MS
    tr.MelScale = _ML
    ta.functional = func
    ta.transforms = tr
    sys.modules["torchaudio"] = ta
    sys.modules["torchaudio.functional"] = func
    sys.modules["torchaudio.transforms"] = tr


def _install_comfy_stubs():
    _install_torchaudio_stub()
    try:
        import comfy_aimdo
    except Exception:
        m = types.ModuleType("comfy_aimdo")
        m.__file__ = "<stub>"
        m.__path__ = []
        sys.modules["comfy_aimdo"] = m
        sys.modules["comfy_aimdo.filter"] = types.ModuleType("comfy_aimdo.filter")
        sys.modules["comfy_aimdo.filter"].filter_modules = lambda *a, **k: None
    try:
        import psutil
    except Exception:
        class _VM:
            total = 64 * 1024 ** 3
            available = 32 * 1024 ** 3

        class _P:
            def memory_info(self): return types.SimpleNamespace(rss=0)
            def memory_full_info(self): return types.SimpleNamespace(uss=0)
            def cpu_percent(self, i=None): return 0.0
            def num_threads(self): return 1

        ps = types.ModuleType("psutil")
        ps.virtual_memory = lambda: _VM()
        ps.Process = lambda: _P()
        sys.modules["psutil"] = ps


def _ensure_comfyui(comfy_path=None):
    candidates = []
    if comfy_path:
        candidates.append(os.path.abspath(comfy_path))
    env = os.environ.get("COMFYUI_PATH")
    if env:
        candidates.append(env)
    candidates.extend([
        r"D:\USERFILES\ComfyUI\ComfyUI",
        r"D:\USERFILES\GitHub\ComfyUI",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ComfyUI-master"),
    ])
    for root in candidates:
        if not root:
            continue
        if os.path.isfile(os.path.join(root, "comfy", "ldm", "krea2", "model.py")):
            if root not in sys.path:
                sys.path.insert(0, root)
            return root
    raise FileNotFoundError("ComfyUI root with comfy/ldm/krea2/model.py not found.")


def _detect_krea2_dit_config(sd, prefix):
    head_dim = 128
    fw = sd[f"{prefix}first.weight"]
    features = int(fw.shape[0])
    channels = int(fw.shape[1] // 4)
    br = re.compile(r"^" + re.escape(prefix) + r"blocks\.(\d+)\.")
    layers = 0
    for k in sd:
        m = br.match(k)
        if m:
            layers = max(layers, int(m.group(1)) + 1)
    wq = sd[f"{prefix}blocks.0.attn.wq.weight"]
    wk = sd[f"{prefix}blocks.0.attn.wk.weight"]
    txtlayers = int(sd[f"{prefix}txtfusion.projector.weight"].shape[1])
    txtdim = int(sd[f"{prefix}txtfusion.layerwise_blocks.0.prenorm.scale"].shape[0])
    return {
        "image_model": "krea2", "features": features, "channels": channels,
        "patch": 2, "layers": layers,
        "heads": int(wq.shape[0] // head_dim),
        "kvheads": int(wk.shape[0] // head_dim),
        "txtlayers": txtlayers, "txtdim": txtdim,
    }


def _encode_krea2_calib_contexts(clip_path, prompts, expected_fused, comfy_path=None):
    if not os.path.isfile(clip_path):
        raise FileNotFoundError(f"--clip_path not found: {clip_path}")
    saved = _clear_argv_for_comfy()
    try:
        _ensure_comfyui(comfy_path)
        import comfy.options
        comfy.options.enable_args_parsing(False)
        _install_comfy_stubs()
        _install_torchaudio_stub()
        import comfy.model_management as mm
        import comfy.sd
        mm.get_torch_device()
        print(f"  [calib] Loading CLIP: {clip_path}")
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path], embedding_directory=None,
            clip_type=comfy.sd.CLIPType.KREA2,
        )
        bank = []
        for i, prompt in enumerate(prompts):
            tokens = clip.tokenize(prompt)
            conds = clip.encode_from_tokens_scheduled(tokens)
            if not conds:
                raise RuntimeError(f"CLIP encode empty for sample {i}")
            ct = conds[0][0]
            if ct.ndim == 2:
                ct = ct.unsqueeze(0)
            if ct.ndim != 3:
                raise RuntimeError(f"CLIP shape {tuple(ct.shape)}")
            fused = int(ct.shape[-1])
            if fused != expected_fused:
                raise ValueError(f"CLIP fused {fused} != {expected_fused}")
            meta = conds[0][1] if len(conds[0]) > 1 else {}
            attn = None
            if isinstance(meta, dict):
                am = meta.get("attention_mask")
                if torch.is_tensor(am):
                    attn = am.detach().float().cpu()
            bank.append((ct.detach().to(dtype=torch.bfloat16).cpu(), attn))
            print(f"  [calib] CLIP {i+1}/{len(prompts)} shape={tuple(ct.shape)}")
        if getattr(clip, "cond_stage_model", None) is not None:
            clip.cond_stage_model.cpu()
        if getattr(clip, "patcher", None) is not None:
            mm.unload_model_and_clones(clip.patcher)
        del clip
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return bank
    finally:
        _restore_argv(saved)


def _load_krea2(path, device="cuda", comfy_path=None):
    if str(device).startswith("cpu"):
        raise RuntimeError("Calibration requires CUDA.")
    _ensure_comfyui(comfy_path)
    saved = _clear_argv_for_comfy()
    try:
        import comfy.options
        comfy.options.enable_args_parsing(False)
        _install_comfy_stubs()
        import comfy.ops
        from comfy.ldm.krea2.model import SingleStreamDiT
        print(f"Loading Krea2 DiT: {path}")
        sd = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
        prefix = None
        for p in ("model.diffusion_model.", "diffusion_model.", ""):
            if f"{p}txtfusion.projector.weight" in sd:
                prefix = p
                break
        if not prefix:
            raise ValueError("Not a Krea2 checkpoint")
        cfg = _detect_krea2_dit_config(sd, prefix)
        print(f"Config: {cfg}")
        kw = {k: v for k, v in cfg.items() if k != "image_model"}
        dit = SingleStreamDiT(**kw, device=device, dtype=torch.bfloat16, operations=comfy.ops.manual_cast)
        stripped = {}
        for k, v in sd.items():
            if prefix and k.startswith(prefix):
                stripped[k[len(prefix):]] = v
            elif not prefix:
                stripped[k] = v
        m, u = dit.load_state_dict(stripped, strict=False)
        print(f"  missing={len(m)} unexpected={len(u)}")
        dev = str(next(dit.parameters()).device)
        if not dev.startswith("cuda"):
            raise RuntimeError(f"DiT on {dev}, not CUDA")
        ck_map = {}
        for name, mod in dit.named_modules():
            w = getattr(mod, "weight", None)
            if w is None or not torch.is_tensor(w) or w.ndim not in (2, 4):
                continue
            ck = f"{prefix}{name}.weight"
            if ck in sd:
                ck_map[ck] = f"{name}.weight"
        print(f"  identity map: {len(ck_map)} entries")
        dit.eval()
        return dit, sd, ck_map, prefix
    finally:
        _restore_argv(saved)


def run_calibration(input_path, calib_file, clip_path, num_samples, num_steps, device, comfy_path=None):
    global _dm_timestep_weight
    print("Preparing calibration data...")
    with open(calib_file, "r", encoding="utf-8") as f:
        prompts = [l.strip() for l in f if l.strip()]
    if len(prompts) < num_samples:
        prompts = (prompts * (num_samples // len(prompts) + 1))[:num_samples]
    else:
        prompts = prompts[:num_samples]

    with safe_open(input_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
    prefix = None
    for p in ("model.diffusion_model.", "diffusion_model.", ""):
        if f"{p}txtfusion.projector.weight" in keys:
            prefix = p
            break
    with safe_open(input_path, framework="pt", device="cpu") as f:
        txtlayers = int(f.get_tensor(f"{prefix}txtfusion.projector.weight").shape[1])
        txtdim = int(f.get_tensor(f"{prefix}txtfusion.layerwise_blocks.0.prenorm.scale").shape[0])
    fused = txtlayers * txtdim

    ctx_bank = _encode_krea2_calib_contexts(clip_path, prompts, fused, comfy_path)
    model, _sd, ck_map, _ = _load_krea2(input_path, device=device, comfy_path=comfy_path)

    print("Setting up DualMonitor hooks...")
    _dual_monitors.clear()
    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(mod.register_forward_hook(lambda m, i, o, n=name: _hook_fn(m, i, o, n)))

    print(f"Running calibration ({num_samples} samples, {num_steps} steps)...")
    gen = torch.Generator(device=device).manual_seed(42)
    lat_h = lat_w = 32
    for i, prompt in enumerate(prompts):
        gen.manual_seed(42 + i)
        with torch.no_grad():
            x = torch.randn(1, int(model.channels), lat_h, lat_w, device=device, dtype=torch.bfloat16, generator=gen)
            ctx, attn = ctx_bank[i]
            context = ctx.to(device=device, dtype=torch.bfloat16)
            am = attn.to(device=device) if attn is not None else None
            for step in tqdm(range(num_steps), total=num_steps, desc=f"S{i+1}"):
                t = torch.full((1,), float(step) / float(max(num_steps, 1)), device=device, dtype=torch.float32)
                _dm_timestep_weight = float(1.0 - t.item())
                if am is not None:
                    model(x, t, context, attention_mask=am)
                else:
                    model(x, t, context)
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    for h in handles:
        h.remove()

    act_sq = {}
    act_amax = {}
    for name, mon in _dual_monitors.items():
        if mon.channel_act_sq_mean is not None:
            act_sq[name] = mon.channel_act_sq_mean.detach().float().cpu()
        if mon.act_amax > 0.0 and math.isfinite(mon.act_amax):
            act_amax[name] = float(mon.act_amax)
    print(f"  DualMonitor: {len(act_sq)} layers act_sq, {len(act_amax)} layers act_amax")
    del model, ctx_bank
    _dual_monitors.clear()
    gc.collect()
    torch.cuda.empty_cache()
    return act_sq, act_amax, ck_map


# =========================================================================
# Main convert
# =========================================================================

def convert_to_nvfp4(
    input_path: str,
    output_path: str,
    device: str,
    calib_file: str | None = None,
    clip_path: str | None = None,
    comfy_path: str | None = None,
    smoothquant_alpha: float = _SMOOTHQUANT_ALPHA_DEFAULT,
    num_calib_samples: int = 32,
    num_inference_steps: int = 25,
    convrot: bool = False,
    convrot_groupsize: int = _DEFAULT_GROUPSIZE,
):
    use_smoothquant = bool(calib_file) and bool(clip_path)
    use_convrot = convrot
    modes = []
    if use_convrot:
        modes.append("ConvRot")
    if use_smoothquant:
        modes.append("SmoothQuant")
    mode_suffix = " + ".join(modes) if modes else "plain"
    print(f"Mode {_MODEL_TYPE} | device={device} | NVFP4 ({mode_suffix})")

    sd = load_file(input_path)
    prefix = _find_krea2_key_prefix(sd)
    print(f"Detected Krea2 key prefix: {prefix!r}")

    # --- Optional: SmoothQuant calibration ---
    act_sq_dict: dict[str, torch.Tensor] = {}
    act_amax_dict: dict[str, float] = {}
    ck_map: dict[str, str] = {}
    if use_smoothquant:
        print("\n=== SmoothQuant Calibration ===")
        act_sq_dict, act_amax_dict, ck_map = run_calibration(
            input_path, calib_file, clip_path,
            num_calib_samples, num_inference_steps, device, comfy_path,
        )
        print(
            f"  Calibration done: {len(act_sq_dict)} layers with act_sq, "
            f"{len(act_amax_dict)} layers with act_amax"
        )
    else:
        print("\n[SKIP] No --calib_file/--clip_path — SmoothQuant disabled")

    blacklist = list(_KREA2_BLACKLIST)
    fp8_layers = list(_KREA2_FP8_LAYERS)
    quant_map = {"format_version": "1.0", "layers": {}}
    new_sd: dict[str, torch.Tensor] = {}
    n_nvfp4 = 0
    n_bf16 = 0
    n_smoothquant = 0
    n_sq_skip = 0
    n_convrot = 0
    n_convrot_skip = 0

    print(f"Converting ({len(sd)} tensors)...")
    for k, v in tqdm(list(sd.items())):
        if any(name in k for name in blacklist):
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16 += 1
            continue

        if _is_non_diffusion_key(k):
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16 += 1
            continue

        if v.ndim == 2 and ".weight" in k:
            base_k_file = k.replace(".weight", "")
            base_k_meta = _meta_base_key(base_k_file)
            v_tensor = v.to(device=device, dtype=torch.bfloat16)

            if fp8_layers and any(name in k for name in fp8_layers):
                import comfy_kitchen as ck
                weight_scale = (v_tensor.abs().max() / 448.0).clamp(min=1e-12).float()
                weight_quantized = ck.quantize_per_tensor_fp8(v_tensor, weight_scale)
                new_sd[k] = weight_quantized.cpu()
                new_sd[f"{base_k_file}.weight_scale"] = weight_scale.to(torch.bfloat16).cpu()
                quant_map["layers"][base_k_meta] = {"format": "float8_e4m3fn"}
                if device == "cuda":
                    del v_tensor
                continue

            # --- SmoothQuant pre-scaling (optional, fp32 precision) ---
            sq_applied = False
            smoothquant_scale = None
            if use_smoothquant:
                module_name = None
                ck_val = ck_map.get(k)
                if ck_val and ck_val.endswith(".weight"):
                    module_name = ck_val[:-len(".weight")]
                act_sq = act_sq_dict.get(module_name) if module_name else None

                if act_sq is not None and act_sq.shape[0] == v_tensor.shape[1]:
                    v_f32 = v_tensor.to(dtype=torch.float32)
                    smoothquant_scale = _compute_smoothquant_scale(
                        act_sq.cpu(), v_f32.cpu(), alpha=smoothquant_alpha
                    )
                    v_f32 = _apply_smoothquant(v_f32, smoothquant_scale.to(device=v_f32.device))
                    v_tensor = v_f32
                    sq_applied = True
                    n_smoothquant += 1
                else:
                    n_sq_skip += 1

            # --- ConvRot: offline Hadamard rotation (optional) ---
            # Applied AFTER SmoothQuant so rotation sees the balanced weights.
            convrot_applied = False
            convrot_gs = None
            if use_convrot:
                in_features = v_tensor.shape[1]
                gs = convrot_group_size_for_features(in_features, convrot_groupsize)
                if gs is not None:
                    v_f32 = v_tensor.to(dtype=torch.float32) if v_tensor.dtype != torch.float32 else v_tensor
                    h = build_hadamard(gs, device="cpu", dtype=torch.float32)
                    v_f32 = rotate_weight(v_f32, h, gs)
                    v_tensor = v_f32
                    convrot_applied = True
                    convrot_gs = gs
                    n_convrot += 1
                else:
                    n_convrot_skip += 1

            # Cast to bfloat16 right before NVFP4 quantize (precision boundary)
            v_bf16 = v_tensor.to(dtype=torch.bfloat16) if v_tensor.dtype != torch.bfloat16 else v_tensor

            try:
                qdata, params = TensorCoreNVFP4Layout.quantize(v_bf16)
                tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)
                for suffix, tensor in tensors.items():
                    new_sd[f"{base_k_file}.weight{suffix}"] = tensor.cpu()

                layer_meta = {"format": "nvfp4"}
                if sq_applied:
                    layer_meta["smoothquant"] = True
                    layer_meta["smoothquant_alpha"] = smoothquant_alpha
                    new_sd[f"{base_k_file}.smoothquant_scale"] = smoothquant_scale.cpu()
                if convrot_applied:
                    layer_meta["convrot"] = True
                    layer_meta["convrot_groupsize"] = convrot_gs
                quant_map["layers"][base_k_meta] = layer_meta
                n_nvfp4 += 1
            except Exception:
                new_sd[k] = v.to(dtype=torch.bfloat16)
                n_bf16 += 1

            if device == "cuda":
                del v_tensor, v_bf16
        else:
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16 += 1

    final_metadata = OrderedDict()
    final_metadata["_quantization_metadata"] = json.dumps(quant_map)
    final_metadata["converted_by"] = "HSWQ native_nvfp4_krea2_3 (self-contained)"
    final_metadata["hswq_model"] = "krea2"
    if use_smoothquant:
        final_metadata["hswq_smoothquant"] = "1"
        final_metadata["hswq_smoothquant_alpha"] = str(smoothquant_alpha)
    if use_convrot:
        final_metadata["hswq_nvfp4_convrot"] = "1"
        final_metadata["hswq_convrot_groupsize"] = str(convrot_groupsize)

    print(f"Saving | Type: {_MODEL_TYPE} | Path: {output_path}")
    save_file(new_sd, output_path, metadata=final_metadata)
    total_bytes = os.path.getsize(output_path)
    print(f"Done. Size: {round(total_bytes / (1024**3), 2)} GiB")
    print(f"NVFP4 layers in metadata: {len(quant_map['layers'])}")
    print(f"  counted nvfp4 packs={n_nvfp4} | bf16 keep tensors={n_bf16}")
    if use_smoothquant:
        print(
            f"  SmoothQuant: applied={n_smoothquant}  skipped(no act stats)={n_sq_skip}  "
            f"alpha={smoothquant_alpha}"
        )
    if use_convrot:
        print(
            f"  ConvRot: applied={n_convrot}  skipped(no valid gs)={n_convrot_skip}  "
            f"groupsize={convrot_groupsize}"
        )

    del sd, new_sd, quant_map
    _release_vram("after convert save")


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Krea2-only NVFP4 converter (self-contained, optional ConvRot + SmoothQuant). "
            "Refuses non-Krea2 checkpoints."
        )
    )
    parser.add_argument("--model", "--input", dest="model", type=str, required=True,
                        help="Path to Krea2 BF16/FP16 .safetensors")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output .safetensors")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"], help="Quantize device")

    # ConvRot
    parser.add_argument("--convrot", action="store_true",
                        help="Apply offline Hadamard rotation (ConvRot) to weights.")
    parser.add_argument("--convrot_groupsize", type=int, default=_DEFAULT_GROUPSIZE,
                        help=f"ConvRot group size (power of 4, default {_DEFAULT_GROUPSIZE})")

    # SmoothQuant
    parser.add_argument("--calib_file", type=str, default=None,
                        help="Calibration prompts file. Enables SmoothQuant with --clip_path.")
    parser.add_argument("--clip_path", type=str, default=None,
                        help="Krea2 CLIP checkpoint for calibration.")
    parser.add_argument("--comfy_path", type=str, default=None,
                        help="ComfyUI root path (auto-detected if omitted).")
    parser.add_argument("--smoothquant_alpha", type=float, default=_SMOOTHQUANT_ALPHA_DEFAULT,
                        help="SmoothQuant migration strength (default 0.5).")
    parser.add_argument("--num_calib_samples", type=int, default=32)
    parser.add_argument("--num_inference_steps", type=int, default=25)

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)
    if bool(args.calib_file) != bool(args.clip_path):
        print("Error: --calib_file and --clip_path must be both provided or both omitted.")
        sys.exit(1)
    if args.calib_file and not os.path.exists(args.calib_file):
        print(f"Error: Calibration file not found: {args.calib_file}")
        sys.exit(1)
    if args.clip_path and not os.path.exists(args.clip_path):
        print(f"Error: CLIP checkpoint not found: {args.clip_path}")
        sys.exit(1)

    convert_to_nvfp4(
        args.model, args.output, device=str(args.device),
        calib_file=args.calib_file, clip_path=args.clip_path,
        comfy_path=args.comfy_path,
        smoothquant_alpha=args.smoothquant_alpha,
        num_calib_samples=args.num_calib_samples,
        num_inference_steps=args.num_inference_steps,
        convrot=args.convrot,
        convrot_groupsize=args.convrot_groupsize,
    )

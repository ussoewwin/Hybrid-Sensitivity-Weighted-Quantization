"""Krea2 NVFP4-base sensitivity-ranked BF16 protection converter.

Reversed approach from auto_int8_nvfp4_hybrid.py:
  OLD: INT8 base → convert safe layers to NVFP4 (SSIM drops from 0.98)
  NEW: NVFP4 base (0.928) → protect high-sensitivity layers as BF16 (SSIM rises)

Input  : BF16/FP16 Krea2 DiT checkpoint
Output : NVFP4 checkpoint with top-N sensitive layers kept as BF16

4-axis sensitivity scoring (higher = more sensitive = protect):
  Axis 1: DualMonitor E[x^2]-weighted NVFP4 quantization error (needs calib)
  Axis 2: Hist Cosine V5 (SVD+RMS leverage, loss = 1 - cosine)
  Axis 3: NVFP4 measured error (BF16 → NVFP4 quant → dequant → compare)
  Axis 4: SVD Leverage score (structural importance)

--protect_n N: keep top-N highest-composite layers as BF16 (protection)
--fast: skip SVD axes (DM × NVFP4 only, much faster)

Blacklist matches native_nvfp4_krea2_1.py (original, SSIM 0.928 baseline).

Requires: comfy_kitchen, weighted_histogram_cosine_v5, ComfyUI (calib only)
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
import time
import types
from typing import Optional, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from collections import OrderedDict

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout
except ImportError:
    print("Error: comfy_kitchen not found.")
    sys.exit(1)

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Hist Cosine V5 import (amax search: L = 1 - cosine)
_HIST_DIR = os.path.join(_REPO_ROOT, "histogram")
if _HIST_DIR not in sys.path:
    sys.path.insert(0, _HIST_DIR)
try:
    from weighted_histogram_cosine_v5 import HSWQWeightedHistogramOptimizerV5
    from weighted_histogram_cosine_v5 import compute_hybrid_leverage_scores
except ImportError:
    print("Error: weighted_histogram_cosine_v5 not found in histogram/ dir")
    sys.exit(1)

# =========================================================================
# Krea2 constants (matches native_nvfp4_krea2_1.py — SSIM 0.928 baseline)
# =========================================================================

_KREA2_BLACKLIST: list[str] = [
    "first",
    "last",
    "mod.",
    "norm",
    "projector",
    "tmlp",
    "tproj",
    "bias",
    "vae.",
    "text_encoders",
]

_NON_DIFFUSION_MARKERS: tuple[str, ...] = (
    "conditioner.", "cond_stage_model.", "text_encoders.",
    "text_encoder.", "text_encoder_2.", "text_encoder_3.",
    "text_model.", "text_projection", "logit_scale",
    "clip_l.", "clip_g.", "t5xxl.", "first_stage_model.", "vae.",
)


def _is_non_diffusion_key(key: str) -> bool:
    return any(marker in key for marker in _NON_DIFFUSION_MARKERS)


def _is_blacklisted(key: str) -> bool:
    return any(name in key for name in _KREA2_BLACKLIST)


def _find_krea2_prefix(state_dict) -> str:
    for prefix in ("model.diffusion_model.", "diffusion_model.", ""):
        if f"{prefix}txtfusion.projector.weight" in state_dict:
            if f"{prefix}blocks.0.attn.wq.weight" not in state_dict:
                raise ValueError(
                    "Krea2 signature incomplete: txtfusion.projector present but "
                    f"{prefix}blocks.0.attn.wq.weight missing"
                )
            return prefix
    raise ValueError(
        "Not a Krea2 checkpoint: missing txtfusion.projector.weight."
    )


def _meta_base_key(base_k_file: str) -> str:
    if "model.diffusion_model." in base_k_file:
        return base_k_file.split("model.diffusion_model.")[-1]
    if "diffusion_model." in base_k_file:
        return base_k_file.split("diffusion_model.")[-1]
    return base_k_file


# =========================================================================
# Midrank / IQR / median / composite helpers
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
    return float(s[n // 2]) if n % 2 == 1 else 0.5 * float(s[n // 2 - 1] + s[n // 2])


def _iqr(vals: Sequence[float]) -> float:
    if len(vals) < 2:
        return 0.0
    s = sorted(float(v) for v in vals)
    n = len(s)
    return float(s[min(n - 1, 3 * (n - 1) // 4)] - s[max(0, (n - 1) // 4)])


def _derive_weights(scores: dict[str, list[float]]) -> dict[str, float]:
    eps = 1e-12
    result = {}
    n_axes = len(scores)
    if n_axes == 0:
        return {"form": "empty"}
    all_ranks = {name: _pool_midranks(vals) for name, vals in scores.items()}
    raw_weights = {}
    for axis_name, axis_vals in scores.items():
        ranks = all_ranks[axis_name]
        ax_iqr = _iqr(ranks)
        ax_p50 = _true_median(ranks)
        d = ax_iqr / max(ax_p50, eps) if ax_p50 > 0 else 0.0
        raw_weights[axis_name] = d
    w_sum = sum(raw_weights.values())
    if w_sum < eps:
        for axis_name in scores:
            result[axis_name] = 1.0 / n_axes
        result["form"] = "equal_weight"
    else:
        for axis_name in scores:
            result[axis_name] = raw_weights[axis_name] / w_sum
        result["form"] = "weighted"
    return result


def _composite_4axis(ranks: dict[str, float], weights: dict[str, float]) -> float:
    eps = 1e-12
    result = 1.0
    for axis_name, r in ranks.items():
        w = weights.get(axis_name, 0.0)
        if w > 0:
            result *= max(float(r), eps) ** w
    return result


# =========================================================================
# DualMonitor (for Axis 1: E[x^2]-weighted NVFP4 error)
# =========================================================================

class DualMonitor:
    def __init__(self):
        self.count = 0
        self.channel_act_sq_mean = None
        self.act_amax = 0.0

    def update(self, input_tensor, module=None, weight=1.0):
        with torch.no_grad():
            inp = input_tensor.detach().float()
            amax_val = float(inp.abs().amax().item())
            if math.isfinite(amax_val) and amax_val > self.act_amax:
                self.act_amax = amax_val
            is_conv2d = isinstance(module, torch.nn.Conv2d)
            if is_conv2d and inp.dim() == 4:
                reduce_dims = (0, 2, 3)
            elif inp.dim() >= 2:
                reduce_dims = tuple(range(inp.dim() - 1))
            else:
                reduce_dims = None
            if reduce_dims is not None:
                current_sq = (inp ** 2).mean(dim=reduce_dims)
            w = float(weight)
            if self.channel_act_sq_mean is None:
                self.channel_act_sq_mean = current_sq
            elif current_sq.shape == self.channel_act_sq_mean.shape:
                self.channel_act_sq_mean = (self.channel_act_sq_mean * self.count + current_sq * w) / (self.count + w)
            self.count += w


_dual_monitors: dict = {}
_dm_timestep_weight: float = 1.0


def _hook_fn(module, input, output, name):
    if name not in _dual_monitors:
        _dual_monitors[name] = DualMonitor()
    _dual_monitors[name].update(input[0], module, weight=_dm_timestep_weight)


# =========================================================================
# ComfyUI bootstrap (for calibration)
# =========================================================================

def _clear_argv_for_comfy():
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    return saved


def _restore_argv(saved):
    sys.argv = saved


def _install_torchaudio_stub():
    import importlib.machinery
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
        os.path.join(_REPO_ROOT, "ComfyUI-master"),
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
# NVFP4 helpers
# =========================================================================

def _encode_comfy_quant(config: dict) -> torch.Tensor:
    return torch.tensor(
        list(json.dumps(config, separators=(",", ":")).encode("utf-8")),
        dtype=torch.uint8,
    )


def _nvfp4_input_scale_from_amax(amax: float) -> torch.Tensor:
    from comfy_kitchen.float_utils import F4_E2M1_MAX, F8_E4M3_MAX
    denom = float(F8_E4M3_MAX) * float(F4_E2M1_MAX)
    return torch.tensor(max(float(amax), 1e-12) / denom, dtype=torch.float32)


_INPUT_SCALE_MAX = 1.0e3
_ACT_AMAX_MAX = 1.0e6


def _sane_nvfp4_input_scale(amax: float) -> Optional[torch.Tensor]:
    a = float(amax)
    if not math.isfinite(a) or a <= 0.0 or a > _ACT_AMAX_MAX:
        return None
    t = _nvfp4_input_scale_from_amax(a)
    v = float(t.item())
    if not math.isfinite(v) or v <= 0.0 or v > _INPUT_SCALE_MAX:
        return None
    return t


# =========================================================================
# Main convert: NVFP4 base + BF16 protection
# =========================================================================

def convert(input_path, output_path, *, device="cuda",
            protect_n=0,
            calib_file=None, clip_path=None, comfy_path=None,
            num_calib_samples=32, num_inference_steps=25,
            hist_bins=4096, hist_candidates=200, hist_refine=3,
            fast=False):

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Device: {device}")
    print(f"Size:   {os.path.getsize(input_path)/(1024**3):.2f} GiB")
    print(f"Mode:   NVFP4 base + BF16 protect top-{protect_n}")
    print()

    # Read keys
    with safe_open(input_path, framework="pt", device="cpu") as f:
        all_keys = set(f.keys())
        prefix = _find_krea2_prefix(all_keys)

    # Collect quantizable 2D .weight layers (same filter as native_nvfp4_krea2_1.py)
    quant_layers = []
    for key in sorted(all_keys):
        if not key.endswith(".weight"):
            continue
        if key.endswith(".weight_scale") or key.endswith(".weight_blocks"):
            continue
        if _is_blacklisted(key) or _is_non_diffusion_key(key):
            continue
        # Check if 2D via peek
        with safe_open(input_path, framework="pt", device="cpu") as f:
            v = f.get_tensor(key)
        if v.ndim == 2:
            base = key.replace(".weight", "")
            mk = _meta_base_key(base)
            quant_layers.append({"key": key, "base": base, "meta_key": mk})
        del v

    print(f"Quantizable 2D .weight layers: {len(quant_layers)}")
    print(f"Krea2 prefix: {prefix!r}")

    # Optional calibration
    act_sq_dict = {}
    act_amax_dict = {}
    ck_map = {}
    use_calib = bool(calib_file) and bool(clip_path)
    if use_calib:
        print("\n=== DualMonitor Calibration ===")
        act_sq_dict, act_amax_dict, ck_map = run_calibration(
            input_path, calib_file, clip_path,
            num_calib_samples, num_inference_steps, device, comfy_path)
    else:
        print("\n[SKIP] No calibration — Axis 1 (DM) disabled")

    # Load all tensors
    print("\nLoading checkpoint...")
    new_sd = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for key in tqdm(sorted(all_keys), desc="Load"):
            new_sd[key] = f.get_tensor(key)

    # Hist Cosine V5
    hist_dev = "cuda" if torch.cuda.is_available() else "cpu"
    with contextlib.redirect_stdout(io.StringIO()):
        hist_opt = HSWQWeightedHistogramOptimizerV5(
            bins=int(hist_bins), num_candidates=int(hist_candidates),
            refinement_iterations=int(hist_refine), device=hist_dev, loss_type="cosine")
    print(f"HistCosine V5: ready (device={hist_dev})")

    # =========================================================================
    # 4-axis sensitivity scoring (higher = more sensitive = protect)
    # =========================================================================
    n_layers = len(quant_layers)
    print(f"\n=== 4-Axis Sensitivity Scoring ({n_layers} layers) ===")
    axis_dm = {}
    axis_hist = {}
    axis_nvfp4 = {}
    axis_svd = {}
    t_score0 = time.perf_counter()

    for i, layer in enumerate(quant_layers):
        t0 = time.perf_counter()
        key = layer["key"]
        w = new_sd[key].to(dtype=torch.float32)  # BF16 → fp32 for scoring
        w_bf16 = w.to(dtype=torch.bfloat16)

        # --- Axis 1: DM E[x^2]-weighted NVFP4 error ---
        module_name = None
        ck_val = ck_map.get(key)
        if ck_val and ck_val.endswith(".weight"):
            module_name = ck_val[:-len(".weight")]
        act_sq = act_sq_dict.get(module_name) if module_name else None
        if act_sq is not None and act_sq.shape[0] == w.shape[1]:
            act_scale = act_sq.sqrt()
            try:
                w_nv = w_bf16.to(device=device)
                qdata, params = TensorCoreNVFP4Layout.quantize(w_nv)
                if hasattr(TensorCoreNVFP4Layout, "dequantize"):
                    w_dq = TensorCoreNVFP4Layout.dequantize(qdata, params).float().cpu()
                    err = w - w_dq
                    we = err * act_scale.unsqueeze(0)
                    wb = w * act_scale.unsqueeze(0)
                    axis_dm[key] = float(we.norm().item()) / max(float(wb.norm().item()), 1e-8)
                del qdata, params
                if device == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass

        # --- Pre-compute SVD hybrid leverage ---
        hybrid_imp = None
        if not fast:
            try:
                w_svd = w.to(device=hist_dev, dtype=torch.float32)
                with contextlib.redirect_stdout(io.StringIO()):
                    hybrid_imp = compute_hybrid_leverage_scores(w_svd, alpha=0.7, beta=0.3)
                if hybrid_imp is not None and hybrid_imp.device.type != "cpu":
                    hybrid_imp = hybrid_imp.detach().cpu()
                del w_svd
            except Exception:
                hybrid_imp = None

        # --- Axis 2: Hist Cosine V5 ---
        if not fast:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    optimal_amax = hist_opt.compute_optimal_amax(
                        w, importance=hybrid_imp, use_svd_leverage=False,
                        scaled=False, loss_type="cosine")
                    from weighted_histogram_cosine_v5 import WeightedHistogram
                    wh = WeightedHistogram(bins=hist_opt.bins, device=hist_opt.device)
                    wh.build(w, hybrid_imp)
                    hist = wh.get_histogram()
                    bc = wh.get_bin_centers()
                    est_loss = hist_opt.cosine_optimizer.compute_weighted_cosine(
                        hist, bc, optimal_amax, scaled=False, loss_type="cosine")
                axis_hist[key] = float(est_loss)
            except Exception:
                pass

        # --- Axis 3: NVFP4 measured error ---
        try:
            w_nv = w_bf16.to(device=device)
            qdata, params = TensorCoreNVFP4Layout.quantize(w_nv)
            if hasattr(TensorCoreNVFP4Layout, "dequantize"):
                w_dq = TensorCoreNVFP4Layout.dequantize(qdata, params).float().cpu()
                err = w - w_dq
                axis_nvfp4[key] = float(err.norm().item()) / max(float(w.norm().item()), 1e-8)
            del qdata, params
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

        # --- Axis 4: SVD Leverage ---
        if not fast:
            try:
                if hybrid_imp is not None:
                    axis_svd[key] = float(hybrid_imp.mean().item())
                else:
                    U, S, Vh = torch.linalg.svd(w.to(device=hist_dev, dtype=torch.float32), full_matrices=False)
                    axis_svd[key] = float((U ** 2 * S.unsqueeze(0) ** 2).sum(dim=1).mean().item())
                    del U, S, Vh
            except Exception:
                pass

        dt = time.perf_counter() - t0
        elapsed = time.perf_counter() - t_score0
        done = i + 1
        eta = (elapsed / done) * (n_layers - done) if done else 0.0
        print(
            f"  [{done}/{n_layers}] {key}  shape={tuple(w.shape)}  {dt:.1f}s  ETA={eta/60.0:.1f}m  "
            f"DM={axis_dm.get(key, -1):.6f}  Hist={axis_hist.get(key, -1):.6e}  "
            f"NVFP4={axis_nvfp4.get(key, -1):.6f}  SVD={axis_svd.get(key, -1):.4f}",
            flush=True,
        )

    # =========================================================================
    # Composite ranking (HIGHER = more sensitive = protect)
    # =========================================================================
    print("\n=== Composite Ranking ===")
    available_axes = {}
    if axis_dm: available_axes["DM"] = list(axis_dm.values())
    if axis_hist: available_axes["HistCosine"] = list(axis_hist.values())
    if axis_nvfp4: available_axes["NVFP4"] = list(axis_nvfp4.values())
    if axis_svd: available_axes["SVD"] = list(axis_svd.values())

    if not available_axes:
        raise RuntimeError("No axis scores produced")

    axis_ranks = {}
    for axis_name, scores_dict in [("DM", axis_dm), ("HistCosine", axis_hist),
                                     ("NVFP4", axis_nvfp4), ("SVD", axis_svd)]:
        if not scores_dict:
            continue
        keys_sorted = list(scores_dict.keys())
        vals = [scores_dict[k] for k in keys_sorted]
        ranks = _pool_midranks(vals)
        for k, r in zip(keys_sorted, ranks):
            if k not in axis_ranks:
                axis_ranks[k] = {}
            axis_ranks[k][axis_name] = r

    raw_scores = {name: list(d.values()) for name, d in [("DM", axis_dm), ("HistCosine", axis_hist),
                                                          ("NVFP4", axis_nvfp4), ("SVD", axis_svd)] if d}
    weights = _derive_weights(raw_scores)
    print(f"  Form: {weights.get('form', '?')}")
    for ax in available_axes:
        print(f"  {ax}: weight={weights.get(ax, 0):.4f}  n={len(available_axes[ax])}")

    composite = {}
    for key, ranks in axis_ranks.items():
        composite[key] = _composite_4axis(ranks, weights)

    # Sort: HIGH composite = sensitive = protect; LOW = safe to NVFP4
    sorted_layers = sorted(composite.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 5 (protect candidates):")
    for k, s in sorted_layers[:5]:
        print(f"    {k}  composite={s:.6f}")
    print(f"  Bottom 5 (safe NVFP4):")
    for k, s in sorted_layers[-5:]:
        print(f"    {k}  composite={s:.6f}")

    # Select layers to protect as BF16
    protect_keys = set()
    if protect_n > 0:
        for k, _ in sorted_layers[:protect_n]:
            protect_keys.add(k)
        print(f"\n--protect_n {protect_n}: keeping {len(protect_keys)} layers as BF16")

    # =========================================================================
    # Convert: NVFP4 all quantizable layers, EXCEPT protected ones (stay BF16)
    # =========================================================================
    print("\n=== Converting (NVFP4 base + BF16 protect) ===")
    quant_map = {"format_version": "1.0", "layers": {}}
    n_nvfp4 = 0
    n_bf16_protect = 0
    n_bf16_blacklist = 0
    input_scale_written = 0
    input_scale_missing = 0

    for k, v in tqdm(list(new_sd.items()), desc="Convert"):
        # Skip non-weight keys (they pass through as-is)
        if not k.endswith(".weight") or k.endswith(".weight_scale") or k.endswith(".weight_blocks"):
            continue

        # Blacklist / non-diffusion → BF16 (always)
        if _is_blacklisted(k) or _is_non_diffusion_key(k):
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16_blacklist += 1
            continue

        # Non-2D → BF16
        if v.ndim != 2:
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16_blacklist += 1
            continue

        # Protected layer → BF16 (skip NVFP4)
        if k in protect_keys:
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16_protect += 1
            base_k = k.replace(".weight", "")
            base_meta = _meta_base_key(base_k)
            quant_map["layers"][base_meta] = {"format": "bf16_protected"}
            continue

        # NVFP4 quantize
        base_k = k.replace(".weight", "")
        base_meta = _meta_base_key(base_k)
        v_bf16 = v.to(device=device, dtype=torch.bfloat16)

        try:
            qdata, params = TensorCoreNVFP4Layout.quantize(v_bf16)
            tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)

            orig_shape = [int(x) for x in params.orig_shape]
            nv_meta = {
                "format": "nvfp4",
                "orig_shape": orig_shape,
                "in_features": int(orig_shape[1]) if len(orig_shape) > 1 else None,
                "out_features": int(orig_shape[0]) if len(orig_shape) > 0 else None,
            }

            for suffix, tensor in tensors.items():
                new_sd[f"{base_k}.weight{suffix}"] = tensor.cpu()
            new_sd[f"{base_k}.comfy_quant"] = _encode_comfy_quant(dict(nv_meta))
            del new_sd[k]
            quant_map["layers"][base_meta] = nv_meta
            n_nvfp4 += 1

            # Optional input_scale from calibration
            module_name = None
            ck_val = ck_map.get(k)
            if ck_val and ck_val.endswith(".weight"):
                module_name = ck_val[:-len(".weight")]
            amax = act_amax_dict.get(module_name) if module_name else None
            if amax is not None:
                is_t = _sane_nvfp4_input_scale(float(amax))
                if is_t is not None:
                    new_sd[f"{base_k}.input_scale"] = is_t
                    input_scale_written += 1
                else:
                    input_scale_missing += 1
            else:
                input_scale_missing += 1

            print(f"  [NVFP4] {k}  orig_shape={tuple(orig_shape)}")
        except Exception as e:
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16_blacklist += 1
            print(f"  [FALLBACK BF16] {k}: {e}")

        if device == "cuda":
            del v_bf16
            torch.cuda.empty_cache()

    # =========================================================================
    # Save
    # =========================================================================
    final_metadata = OrderedDict()
    final_metadata["_quantization_metadata"] = json.dumps(quant_map)
    final_metadata["converted_by"] = "HSWQ NVFP4-base + BF16 protect (auto_int8_nvfp4_hybrid_2)"
    final_metadata["hswq_model"] = "krea2"
    final_metadata["hswq_nvfp4_count"] = str(n_nvfp4)
    final_metadata["hswq_bf16_protect"] = str(n_bf16_protect)
    final_metadata["hswq_bf16_blacklist"] = str(n_bf16_blacklist)
    final_metadata["hswq_nvfp4_pack"] = "plain"
    final_metadata["hswq_axes"] = ",".join(available_axes.keys())
    if use_calib:
        final_metadata["hswq_calib"] = "1"

    print(f"\nSaving: {output_path}")
    save_file(new_sd, output_path, metadata=final_metadata)
    out_sz = os.path.getsize(output_path)
    in_sz = os.path.getsize(input_path)
    print(f"Done: {out_sz/(1024**3):.2f} GiB (was {in_sz/(1024**3):.2f})")
    print(f"  NVFP4: {n_nvfp4}  BF16 protect: {n_bf16_protect}  BF16 blacklist: {n_bf16_blacklist}")
    print(f"  input_scale: written={input_scale_written}  missing={input_scale_missing}")

    # Save ranking JSON
    ranking_path = output_path.replace(".safetensors", "_ranking.json")
    ranking = [{"key": k, "composite": s,
                "dm": axis_dm.get(k, -1), "hist": axis_hist.get(k, -1),
                "nvfp4": axis_nvfp4.get(k, -1), "svd": axis_svd.get(k, -1)}
               for k, s in sorted_layers]
    with open(ranking_path, "w", encoding="utf-8") as rf:
        json.dump(ranking, rf, indent=2)
    print(f"  Ranking: {ranking_path}")

    del new_sd
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(
        description="Krea2 NVFP4-base + BF16 protection converter (reversed hybrid)"
    )
    p.add_argument("--input", "--model", dest="input", required=True,
                   help="BF16/FP16 Krea2 DiT safetensors")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--protect_n", type=int, default=0,
                   help="Keep top-N highest-sensitivity layers as BF16 (protection)")
    p.add_argument("--calib_file", default=None)
    p.add_argument("--clip_path", default=None)
    p.add_argument("--comfy_path", default=None)
    p.add_argument("--num_calib_samples", type=int, default=32)
    p.add_argument("--num_inference_steps", type=int, default=25)
    p.add_argument("--hist_bins", type=int, default=4096)
    p.add_argument("--hist_candidates", type=int, default=200)
    p.add_argument("--hist_refine", type=int, default=3)
    p.add_argument("--fast", action="store_true",
                   help="Skip SVD axes (DM × NVFP4 only)")
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)
    if bool(args.calib_file) != bool(args.clip_path):
        print("Error: --calib_file and --clip_path must be both provided or both omitted.")
        sys.exit(1)

    convert(args.input, args.output, device=args.device,
            protect_n=args.protect_n,
            calib_file=args.calib_file, clip_path=args.clip_path,
            comfy_path=args.comfy_path,
            num_calib_samples=args.num_calib_samples,
            num_inference_steps=args.num_inference_steps,
            hist_bins=args.hist_bins,
            hist_candidates=args.hist_candidates,
            hist_refine=args.hist_refine,
            fast=args.fast)


if __name__ == "__main__":
    main()

"""Krea2 ConvRot INT8 -> NVFP4 hybrid auto-converter (4-axis ranking).

Reads a ConvRot INT8 checkpoint, ranks all INT8 layers by a 4-axis composite:
  Axis 1: DualMonitor E[x^2]-weighted INT8 quantization error (needs calib)
  Axis 2: Hist Cosine V5 (SVD+RMS leverage, loss = 1 - cosine) -- weight direction
  Axis 3: NVFP4 measured error (INT8 dequant -> NVFP4 quant -> compare)
  Axis 4: SVD Leverage score (structural importance, standalone)

Lower composite = safer to NVFP4.  --nvfp4_keep N converts the lowest-N layers.
Higher composite = keep as INT8 (protection).

Finished-product NVFP4 pack (Comfy / krea2nvfp4 style — do not stamp ConvRot):
  .weight / .weight_scale / .weight_scale_2
  .input_scale   f32  (act amax / (F8_E4M3_MAX * F4_E2M1_MAX); needs calib;
                       non-finite / absurd amax rejected — no write)
  .comfy_quant   uint8 JSON  {"format":"nvfp4","orig_shape":[out,in],
                               "in_features","out_features"}  — no convrot
  meta layers: same format nvfp4 + orig_shape (no convrot); scrub full-key stale
  file meta: hswq_nvfp4_pack=plain, hswq_nvfp4_convrot=0
  ConvRot conf resolve: sidecar .comfy_quant → layers full key → stripped
  If source INT8 had ConvRot, weights are unrotated before NVFP4 quant so the
  pack matches plain NVFP4 (online act rotate must not be required).
  If ConvRot is armed but group_size is invalid, layer stays INT8 (no plain pack).
  Loader detect (nvfp4_conf) reads orig_shape from .comfy_quant first.

INT8 shelter layers keep int8_tensorwise + ConvRot as in the source.

DualMonitor calibration is OPTIONAL for ranking. Without --calib_file/--clip_path,
Axis 1 is skipped; converted NVFP4 layers will lack .input_scale (WARN).

Requires: comfy_kitchen (TensorCoreNVFP4Layout), weighted_histogram_cosine_v5
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

from native_convert_int8 import (
    build_hadamard,
    convrot_group_size_for_features,
    rotate_weight,
    unrotate_weight,
    unrotate_weight_conv2d,
)

# Only integer-typed .weight tensors are real INT8 layers. bf16/fp16/
# fp32 weights carrying a stale .comfy_quant (leftover from an earlier
# conversion) must never be scored or converted -- keep them as-is.
_INT8_WEIGHT_DTYPES = (torch.int8, torch.uint8)

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
# Midrank / IQR / median / composite helpers (from v1.4)
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
    return float(s[n // 2]) if n % 2 == 1 else 0.5 * float(s[n//2-1] + s[n//2])

def _iqr(vals: Sequence[float]) -> float:
    if len(vals) < 2:
        return 0.0
    s = sorted(float(v) for v in vals)
    n = len(s)
    return float(s[min(n-1, 3*(n-1)//4)] - s[max(0, (n-1)//4)])

def _derive_weights(scores: dict[str, list[float]]) -> dict[str, float]:
    """Auto axis weights from each axis pool's IQR/median ratio."""
    eps = 1e-12
    result = {}
    vals = list(scores.values())
    n_axes = len(vals)
    if n_axes == 0:
        return {"form": "empty"}
    # Each axis: list of (key, rank) pairs
    all_ranks = {}
    for axis_name, axis_vals in scores.items():
        all_ranks[axis_name] = _pool_midranks(axis_vals)

    # Derive weights from IQR/median
    raw_weights = {}
    for axis_name, axis_vals in scores.items():
        ranks = all_ranks[axis_name]
        ax_iqr = _iqr(ranks)
        ax_p50 = _true_median(ranks)
        d = ax_iqr / max(ax_p50, eps) if ax_p50 > 0 else 0.0
        raw_weights[axis_name] = d

    w_sum = sum(raw_weights.values())
    if w_sum < eps:
        # Equal weights
        for axis_name in scores:
            result[axis_name] = 1.0 / n_axes
        result["form"] = "equal_weight"
    else:
        for axis_name in scores:
            result[axis_name] = raw_weights[axis_name] / w_sum
        result["form"] = "weighted"
    return result

def _composite_4axis(ranks: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted geometric mean of midranks across all available axes."""
    eps = 1e-12
    result = 1.0
    for axis_name, r in ranks.items():
        w = weights.get(axis_name, 0.0)
        if w > 0:
            result *= max(float(r), eps) ** w
    return result

# =========================================================================
# DualMonitor (from v1.4, with timestep weighting)
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
            # Global absmax for NVFP4 input_scale (plain = unrotated act at module in)
            amax_val = float(inp_detached.abs().amax().item())
            if math.isfinite(amax_val) and amax_val > self.act_amax:
                self.act_amax = amax_val
            is_conv2d = isinstance(module, torch.nn.Conv2d)
            if is_conv2d and inp_detached.dim() == 4:
                reduce_dims = (0, 2, 3)
            elif inp_detached.dim() >= 2:
                reduce_dims = tuple(range(inp_detached.dim() - 1))
            else:
                current_imp = torch.ones(1, device=inp_detached.device, dtype=torch.float32)
                current_act = torch.zeros(1, device=inp_detached.device, dtype=torch.float32)
                current_sq = torch.ones(1, device=inp_detached.device, dtype=torch.float32)
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

dual_monitors: dict[str, DualMonitor] = {}
_dm_timestep_weight: float = 1.0

def hook_fn(module, input, output, name):
    if name not in dual_monitors:
        dual_monitors[name] = DualMonitor()
    dual_monitors[name].update(input[0], output, module, weight=_dm_timestep_weight)

# =========================================================================
# ComfyUI bootstrap (from v1.4, compact)
# =========================================================================
def _clear_argv_for_comfy():
    saved = list(sys.argv); sys.argv = [saved[0]]; return saved
def _restore_argv(saved):
    sys.argv = saved

def _install_torchaudio_stub():
    import importlib.machinery, types
    for key in list(sys.modules):
        if key == "torchaudio" or key.startswith("torchaudio."):
            del sys.modules[key]
    def _stub(name, pkg=False):
        m = types.ModuleType(name); m.__file__ = "<stub>"
        if pkg:
            m.__path__ = []
            spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            spec.submodule_search_locations = []
        else:
            spec = importlib.machinery.ModuleSpec(name, loader=None)
        m.__spec__ = spec; return m
    ta = _stub("torchaudio", True)
    func = _stub("torchaudio.functional")
    func.resample = lambda w, o, n, *a, **k: w
    tr = _stub("torchaudio.transforms")
    class _MS:
        def __init__(self,*a,**k): pass
        def __call__(self,x): return x
        def to(self,*a,**k): return self
    class _ML:
        def __init__(self,*a,**k): pass
    tr.MelSpectrogram = _MS; tr.MelScale = _ML
    ta.functional = func; ta.transforms = tr
    sys.modules["torchaudio"] = ta
    sys.modules["torchaudio.functional"] = func
    sys.modules["torchaudio.transforms"] = tr

def _install_comfy_stubs():
    _install_torchaudio_stub()
    try:
        import comfy_aimdo
    except Exception:
        m = types.ModuleType("comfy_aimdo"); m.__file__ = "<stub>"; m.__path__ = []
        sys.modules["comfy_aimdo"] = m
        sys.modules["comfy_aimdo.filter"] = types.ModuleType("comfy_aimdo.filter")
        sys.modules["comfy_aimdo.filter"].filter_modules = lambda *a, **k: None
    try:
        import psutil
    except Exception:
        class _VM:
            total = 64*1024**3; available = 32*1024**3
        class _P:
            def memory_info(self): return types.SimpleNamespace(rss=0)
            def memory_full_info(self): return types.SimpleNamespace(uss=0)
            def cpu_percent(self,i=None): return 0.0
            def num_threads(self): return 1
        ps = types.ModuleType("psutil")
        ps.virtual_memory = lambda: _VM()
        ps.Process = lambda: _P()
        sys.modules["psutil"] = ps

def _ensure_comfyui(comfy_path=None):
    candidates = []
    if comfy_path: candidates.append(os.path.abspath(comfy_path))
    env = os.environ.get("COMFYUI_PATH")
    if env: candidates.append(env)
    candidates.extend([r"D:\USERFILES\ComfyUI\ComfyUI", r"D:\USERFILES\GitHub\ComfyUI",
                       os.path.join(_REPO_ROOT, "ComfyUI-master")])
    for root in candidates:
        if not root: continue
        if os.path.isfile(os.path.join(root, "comfy", "ldm", "krea2", "model.py")):
            if root not in sys.path: sys.path.insert(0, root)
            return root
    raise FileNotFoundError("ComfyUI root with comfy/ldm/krea2/model.py not found.")

def detect_krea2_dit_config(sd, prefix):
    head_dim = 128
    fw = sd[f"{prefix}first.weight"]
    features = int(fw.shape[0]); channels = int(fw.shape[1] // 4)
    br = re.compile(r"^" + re.escape(prefix) + r"blocks\.(\d+)\.")
    layers = 0
    for k in sd:
        m = br.match(k)
        if m: layers = max(layers, int(m.group(1)) + 1)
    wq = sd[f"{prefix}blocks.0.attn.wq.weight"]
    wk = sd[f"{prefix}blocks.0.attn.wk.weight"]
    txtlayers = int(sd[f"{prefix}txtfusion.projector.weight"].shape[1])
    txtdim = int(sd[f"{prefix}txtfusion.layerwise_blocks.0.prenorm.scale"].shape[0])
    return {"image_model":"krea2","features":features,"channels":channels,"patch":2,
            "layers":layers,"heads":int(wq.shape[0]//head_dim),"kvheads":int(wk.shape[0]//head_dim),
            "txtlayers":txtlayers,"txtdim":txtdim}

def _encode_krea2_calib_contexts(clip_path, prompts, expected_fused, comfy_path=None):
    if not os.path.isfile(clip_path):
        raise FileNotFoundError(f"--clip_path not found: {clip_path}")
    saved = _clear_argv_for_comfy()
    try:
        _ensure_comfyui(comfy_path)
        import comfy.options; comfy.options.enable_args_parsing(False)
        _install_comfy_stubs(); _install_torchaudio_stub()
        import comfy.model_management as mm; import comfy.sd
        mm.get_torch_device()
        print(f"  [calib] Loading CLIP: {clip_path}")
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=None,
                                   clip_type=comfy.sd.CLIPType.KREA2)
        bank = []
        for i, prompt in enumerate(prompts):
            tokens = clip.tokenize(prompt)
            conds = clip.encode_from_tokens_scheduled(tokens)
            if not conds: raise RuntimeError(f"CLIP encode empty for sample {i}")
            ct = conds[0][0]
            if ct.ndim == 2: ct = ct.unsqueeze(0)
            if ct.ndim != 3: raise RuntimeError(f"CLIP shape {tuple(ct.shape)}")
            fused = int(ct.shape[-1])
            if fused != expected_fused:
                raise ValueError(f"CLIP fused {fused} != {expected_fused}")
            meta = conds[0][1] if len(conds[0]) > 1 else {}
            attn = None
            if isinstance(meta, dict):
                am = meta.get("attention_mask")
                if torch.is_tensor(am): attn = am.detach().float().cpu()
            bank.append((ct.detach().to(dtype=torch.bfloat16).cpu(), attn))
            print(f"  [calib] CLIP {i+1}/{len(prompts)} shape={tuple(ct.shape)}")
        if getattr(clip, "cond_stage_model", None) is not None:
            clip.cond_stage_model.cpu()
        if getattr(clip, "patcher", None) is not None:
            mm.unload_model_and_clones(clip.patcher)
        del clip; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return bank
    finally:
        _restore_argv(saved)

def load_krea2(path, device="cuda", comfy_path=None):
    if str(device).startswith("cpu"):
        raise RuntimeError("Calibration requires CUDA.")
    _ensure_comfyui(comfy_path)
    saved = _clear_argv_for_comfy()
    try:
        import comfy.options; comfy.options.enable_args_parsing(False)
        _install_comfy_stubs()
        import comfy.ops
        from comfy.ldm.krea2.model import SingleStreamDiT
        print(f"Loading Krea2 DiT: {path}")
        sd = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys(): sd[k] = f.get_tensor(k)
            metadata = f.metadata()
        prefix = None
        for p in ("model.diffusion_model.", "diffusion_model.", ""):
            if f"{p}txtfusion.projector.weight" in sd: prefix = p; break
        if not prefix: raise ValueError("Not a Krea2 checkpoint")
        cfg = detect_krea2_dit_config(sd, prefix)
        print(f"Config: {cfg}")
        kw = {k:v for k,v in cfg.items() if k != "image_model"}
        dit = SingleStreamDiT(**kw, device=device, dtype=torch.bfloat16, operations=comfy.ops.manual_cast)
        stripped = {}
        for k, v in sd.items():
            if prefix and k.startswith(prefix): stripped[k[len(prefix):]] = v
            elif not prefix: stripped[k] = v
        m, u = dit.load_state_dict(stripped, strict=False)
        print(f"  missing={len(m)} unexpected={len(u)}")
        dev = str(next(dit.parameters()).device)
        if not dev.startswith("cuda"): raise RuntimeError(f"DiT on {dev}, not CUDA")
        ck_map = {}
        for name, mod in dit.named_modules():
            w = getattr(mod, "weight", None)
            if w is None or not torch.is_tensor(w) or w.ndim not in (2,4): continue
            ck = f"{prefix}{name}.weight"
            if ck in sd: ck_map[ck] = f"{name}.weight"
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
        if f"{p}txtfusion.projector.weight" in keys: prefix = p; break
    # Peek config
    with safe_open(input_path, framework="pt", device="cpu") as f:
        fw = f.get_tensor(f"{prefix}first.weight")
        txtlayers = int(f.get_tensor(f"{prefix}txtfusion.projector.weight").shape[1])
        txtdim = int(f.get_tensor(f"{prefix}txtfusion.layerwise_blocks.0.prenorm.scale").shape[0])
    fused = txtlayers * txtdim
    del fw

    ctx_bank = _encode_krea2_calib_contexts(clip_path, prompts, fused, comfy_path)
    model, _sd, ck_map, _ = load_krea2(input_path, device=device, comfy_path=comfy_path)

    print("Setting up DualMonitor hooks...")
    dual_monitors.clear()
    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(mod.register_forward_hook(lambda m,i,o,n=name: hook_fn(m,i,o,n)))

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
                t = torch.full((1,), float(step)/float(max(num_steps,1)), device=device, dtype=torch.float32)
                _dm_timestep_weight = float(1.0 - t.item())
                if am is not None: model(x, t, context, attention_mask=am)
                else: model(x, t, context)
        if (i+1) % 10 == 0: gc.collect(); torch.cuda.empty_cache()
    for h in handles: h.remove()

    act_sq = {}
    act_amax = {}
    for name, mon in dual_monitors.items():
        if mon.channel_act_sq_mean is not None:
            act_sq[name] = mon.channel_act_sq_mean.detach().float().cpu()
        if mon.act_amax > 0.0 and math.isfinite(mon.act_amax):
            act_amax[name] = float(mon.act_amax)
    print(f"  DualMonitor: {len(act_sq)} layers act_sq, {len(act_amax)} layers act_amax")
    del model, ctx_bank; dual_monitors.clear()
    gc.collect(); torch.cuda.empty_cache()
    return act_sq, act_amax, ck_map

# =========================================================================
# Helpers
# =========================================================================
def _find_prefix(keys):
    for p in ("model.diffusion_model.", "diffusion_model.", ""):
        if f"{p}txtfusion.projector.weight" in keys: return p
    raise ValueError("Not Krea2")

def _meta_key(key, prefix):
    if prefix and key.startswith(prefix): return key[len(prefix):]
    return key

def _match_type(key, types):
    return any(f".{t}.weight" in key for t in types)

def _encode_comfy_quant(config: dict) -> torch.Tensor:
    """Comfy sidecar: uint8 JSON bytes (same as hswq_convert_nvfp4_krea2)."""
    return torch.tensor(
        list(json.dumps(config, separators=(",", ":")).encode("utf-8")),
        dtype=torch.uint8,
    )

def _decode_comfy_quant_raw(raw) -> Optional[dict]:
    """Decode INT8/NVFP4 comfy_quant uint8 JSON (or dict) to a layer conf dict."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return dict(raw)
    if torch.is_tensor(raw):
        conf = json.loads(bytes(raw.detach().cpu().reshape(-1).tolist()).decode("utf-8"))
    elif isinstance(raw, (bytes, bytearray, memoryview)):
        conf = json.loads(bytes(raw))
    elif isinstance(raw, str):
        conf = raw
    else:
        return None
    while isinstance(conf, str):
        try:
            parsed = json.loads(conf)
        except (TypeError, json.JSONDecodeError):
            return {"format": conf}
        conf = parsed
    return dict(conf) if isinstance(conf, dict) else None

def _resolve_int8_layer_conf(
    base: str,
    mk: str,
    layers_meta: dict,
    cq_raw,
) -> dict:
    """Resolve ConvRot / format conf for an INT8 layer.

    Priority (first key wins, later fills missing keys only):
      1) sidecar ``.comfy_quant``
      2) ``layers`` full key (``model.diffusion_model....``)
      3) ``layers`` stripped key (``_meta_key``)

    native_convert_int8 often stamps FULL keys in layers_meta; hybrid used to
    look up stripped-only and got ``{}`` → skipped unrotate → plain-packed
    rotated weights (broken finished product).
    """
    side = _decode_comfy_quant_raw(cq_raw)
    full = layers_meta.get(base)
    stripped = layers_meta.get(mk)
    out: dict = {}
    for src in (side, full if isinstance(full, dict) else None,
                stripped if isinstance(stripped, dict) else None):
        if not src:
            continue
        for k, v in src.items():
            if k not in out:
                out[k] = v
    return out

def _nvfp4_input_scale_from_amax(amax: float) -> torch.Tensor:
    """Kitchen TensorCoreNVFP4Layout input_scale: amax / (F8_E4M3_MAX * F4_E2M1_MAX)."""
    from comfy_kitchen.float_utils import F4_E2M1_MAX, F8_E4M3_MAX

    denom = float(F8_E4M3_MAX) * float(F4_E2M1_MAX)
    return torch.tensor(max(float(amax), 1e-12) / denom, dtype=torch.float32)

# Finished-product guard: kitchen scale is amax/(448*6). Sane act amax stays
# well below 1e6 (scale << ~400). Broken krea_hswq_nvfp4 wrote 1e22–1e25.
_INPUT_SCALE_MAX = 1.0e3
_ACT_AMAX_MAX = 1.0e6

def _sane_nvfp4_input_scale(amax: float) -> Optional[torch.Tensor]:
    """Return input_scale tensor, or None if amax/scale is non-finite or absurd."""
    a = float(amax)
    if not math.isfinite(a) or a <= 0.0 or a > _ACT_AMAX_MAX:
        return None
    t = _nvfp4_input_scale_from_amax(a)
    v = float(t.item())
    if not math.isfinite(v) or v <= 0.0 or v > _INPUT_SCALE_MAX:
        return None
    return t

# =========================================================================
# Main convert
# =========================================================================
def convert(input_path, output_path, *, device="cuda",
            nvfp4_keep=0, nvfp4_types=None, all_mlp=False,
            calib_file=None, clip_path=None, comfy_path=None,
            num_calib_samples=32, num_inference_steps=25,
            hist_bins=4096, hist_candidates=200, hist_refine=3):

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Device: {device}")
    print(f"Size:   {os.path.getsize(input_path)/(1024**3):.2f} GiB")
    print()

    # Read keys, metadata, and resolve per-layer conf (sidecar → full → stripped)
    with safe_open(input_path, framework="pt", device="cpu") as f:
        all_keys = set(f.keys())
        metadata = f.metadata()
        prefix = _find_prefix(all_keys)
        qm_str = metadata.get("_quantization_metadata", "{}")
        if isinstance(qm_str, bytes):
            qm_str = qm_str.decode("utf-8")
        quant_meta = json.loads(qm_str)
        layers_meta = quant_meta.get("layers", {})

        int8_layers = []
        non_int8_layers = []
        n_conf_sidecar = 0
        n_conf_full = 0
        n_conf_stripped = 0
        n_conf_empty = 0
        n_conf_convrot = 0
        for key in sorted(all_keys):
            if not key.endswith(".weight"):
                continue
            if key.endswith(".weight_scale") or key.endswith(".weight_blocks"):
                continue
            base = key.replace(".weight", "")
            cq_key = f"{base}.comfy_quant"
            if cq_key not in all_keys:
                continue
            # Guard: only integer-typed weights are INT8 layers.
            # bf16/fp16/fp32 layers that carry a stale .comfy_quant
            # must NOT be scored/converted -- keep them as-is.
            try:
                w_dtype = f.get_slice(key).get_dtype()
            except Exception:
                w_dtype = f.get_tensor(key).dtype
            if w_dtype not in _INT8_WEIGHT_DTYPES:
                non_int8_layers.append(
                    {"key": key, "base": base,
                     "meta_key": _meta_key(base, prefix)}
                )
                continue
            mk = _meta_key(base, prefix)
            cq_raw = f.get_tensor(cq_key)
            conf = _resolve_int8_layer_conf(base, mk, layers_meta, cq_raw)
            side = _decode_comfy_quant_raw(cq_raw)
            full = layers_meta.get(base) if isinstance(layers_meta.get(base), dict) else None
            stripped = layers_meta.get(mk) if isinstance(layers_meta.get(mk), dict) else None
            if side:
                n_conf_sidecar += 1
            elif full:
                n_conf_full += 1
            elif stripped:
                n_conf_stripped += 1
            else:
                n_conf_empty += 1
            if conf.get("convrot"):
                n_conf_convrot += 1
            int8_layers.append(
                {"key": key, "base": base, "meta_key": mk, "conf": conf}
            )
    print(f"INT8 layers: {len(int8_layers)}")
    print(f"Non-INT8 layers w/ stale quant sidecar (kept as-is): {len(non_int8_layers)}")
    print(f"Krea2 prefix: {prefix!r}")
    print(
        f"  conf resolve: sidecar={n_conf_sidecar}  full_meta={n_conf_full}  "
        f"stripped_meta={n_conf_stripped}  empty={n_conf_empty}  "
        f"convrot_armed={n_conf_convrot}"
    )

    # Optional: DualMonitor calibration (Axis 1 + NVFP4 input_scale amax)
    act_sq_dict = {}
    act_amax_dict = {}
    ck_map = {}
    use_calib = bool(calib_file) and bool(clip_path)
    if use_calib:
        print("\n=== DualMonitor Calibration ===")
        act_sq_dict, act_amax_dict, ck_map = run_calibration(
            input_path, calib_file, clip_path,
            int(num_calib_samples), int(num_inference_steps),
            device, comfy_path)
    else:
        print("\n[SKIP] No calibration -- Axis 1 (DM E[x^2]) disabled; "
              "NVFP4 .input_scale will be missing (WARN)")

    # Load all tensors
    print("\nLoading checkpoint...")
    new_sd = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for key in tqdm(sorted(all_keys), desc="Load"):
            new_sd[key] = f.get_tensor(key)

    # Scrub stale quantization sidecars from non-INT8 (bf16/fp16/fp32)
    # layers so the finished file holds a clean plain layer -- loaders
    # read .comfy_quant first and would otherwise misinterpret the
    # bf16 weight as quantized.
    for layer in non_int8_layers:
        base = layer["base"]
        for suf in (".comfy_quant", ".weight_scale",
                    ".weight_blocks", ".input_scale"):
            new_sd.pop(f"{base}{suf}", None)
        if base in layers_meta:
            del layers_meta[base]
        mk = layer["meta_key"]
        if mk in layers_meta:
            del layers_meta[mk]
        print(f"  [FIX] {layer['key']}: non-INT8 weight -> kept bf16, "
              f"stale quant sidecar scrubbed")

    # Hist Cosine V5 — amax via Cosine loss (bins default match krea2 v1.5)
    hist_dev = "cuda" if torch.cuda.is_available() else "cpu"
    with contextlib.redirect_stdout(io.StringIO()):
        hist_opt = HSWQWeightedHistogramOptimizerV5(
            bins=int(hist_bins),
            num_candidates=int(hist_candidates),
            refinement_iterations=int(hist_refine),
            device=hist_dev,
            loss_type="cosine")
    print(
        f"HistCosine V5: ready (device={hist_dev}, "
        f"bins={hist_bins}, candidates={hist_candidates}, refine={hist_refine})"
    )

    # =========================================================================
    # 4-axis scoring
    # =========================================================================
    n_layers = len(int8_layers)
    print("\n=== 4-Axis Scoring ===")
    print(
        f"  Scoring {n_layers} INT8 layers "
        f"  Scoring {n_layers} INT8 layers "
        f"(per-layer progress; HistCosine bins={hist_bins}/"
        f"cand={hist_candidates}/refine={hist_refine})"
    )
    axis_dm = {}       # Axis 1: DM E[x^2]-weighted INT8 error
    axis_hist = {}     # Axis 2: Hist Cosine V5 (1 - cos_sim)
    axis_nvfp4 = {}    # Axis 3: NVFP4 measured error
    axis_svd = {}      # Axis 4: SVD Leverage standalone
    t_score0 = time.perf_counter()

    for i, layer in enumerate(int8_layers):
        t0 = time.perf_counter()
        key = layer["key"]; base = layer["base"]; conf = layer["conf"]
        q = new_sd[key]
        scale_key = f"{base}.weight_scale"
        scale = new_sd.get(scale_key)

        # Dequantize INT8
        if scale is not None:
            if scale.dim() == 0:
                w_dq = q.float() * scale.item()
            elif scale.dim() == 2 and scale.shape[1] == 1:
                w_dq = q.float() * scale
            else:
                w_dq = q.float() * scale
        else:
            w_dq = q.float()

        w_bf16 = w_dq.to(dtype=torch.bfloat16)
        print(f"  [{i+1}/{n_layers}] {key}  shape={tuple(w_dq.shape)}  scoring...",
              flush=True)

        # --- Axis 1: DM E[x^2]-weighted INT8 error ---
        module_name = None
        mk = ck_map.get(key)
        if mk and mk.endswith(".weight"): module_name = mk[:-len(".weight")]
        act_sq = act_sq_dict.get(module_name) if module_name else None
        if act_sq is not None and act_sq.shape[0] == w_dq.shape[1]:
            act_scale = act_sq.sqrt()
            err_int8 = w_dq - q.float() * (scale if scale.dim() == 0 else scale)
            if err_int8.ndim == 2:
                we = err_int8 * act_scale.unsqueeze(0)
                wb = w_dq * act_scale.unsqueeze(0)
            else:
                we = err_int8; wb = w_dq
            axis_dm[key] = float(we.norm().item()) / max(float(wb.norm().item()), 1e-8)

        # --- Pre-compute SVD hybrid leverage ONCE (used by Axis 2 and Axis 4) ---
        # Run SVD on hist_dev (cuda when available); CPU full-SVD on large Krea2
        # mats was the main "stuck after === 4-Axis Scoring ===" cause.
        hybrid_imp = None
        try:
            w_svd = w_dq.to(device=hist_dev, dtype=torch.float32)
            with contextlib.redirect_stdout(io.StringIO()):
                hybrid_imp = compute_hybrid_leverage_scores(
                    w_svd, alpha=0.7, beta=0.3)
            if hybrid_imp is not None and hybrid_imp.device.type != "cpu":
                hybrid_imp = hybrid_imp.detach().cpu()
            del w_svd
        except Exception:
            hybrid_imp = None

        # --- Axis 2: Hist Cosine V5 (1 - cosine; reuse pre-computed importance) ---
        # Pass hybrid_imp as importance with use_svd_leverage=False to avoid 2nd SVD
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                optimal_amax = hist_opt.compute_optimal_amax(
                    w_dq, importance=hybrid_imp, use_svd_leverage=False,
                    scaled=False, loss_type="cosine")
                # Build histogram with same importance to get estimated loss
                from weighted_histogram_cosine_v5 import WeightedHistogram
                wh = WeightedHistogram(bins=hist_opt.bins, device=hist_opt.device)
                wh.build(w_dq, hybrid_imp)
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
                w_nvfp4_dq = TensorCoreNVFP4Layout.dequantize(qdata, params).float().cpu()
                err_nv = w_dq - w_nvfp4_dq
                axis_nvfp4[key] = float(err_nv.norm().item()) / max(float(w_dq.norm().item()), 1e-8)
            del qdata, params
            if device == "cuda": torch.cuda.empty_cache()
        except Exception:
            pass

        # --- Axis 4: SVD Leverage (reuse pre-computed hybrid_imp) ---
        try:
            if hybrid_imp is not None:
                # Mean leverage = average structural importance per element
                axis_svd[key] = float(hybrid_imp.mean().item())
            else:
                # Fallback: direct SVD (only if pre-compute failed)
                U, S, Vh = torch.linalg.svd(
                    w_dq.to(device=hist_dev, dtype=torch.float32),
                    full_matrices=False)
                leverage = (U ** 2 * S.unsqueeze(0) ** 2).sum(dim=1).mean().item()
                axis_svd[key] = leverage
                del U, S, Vh
        except Exception:
            pass

        dm_v = axis_dm.get(key, -1)
        hi_v = axis_hist.get(key, -1)
        nv_v = axis_nvfp4.get(key, -1)
        sv_v = axis_svd.get(key, -1)
        dt = time.perf_counter() - t0
        elapsed = time.perf_counter() - t_score0
        done = i + 1
        eta = (elapsed / done) * (n_layers - done) if done else 0.0
        print(
            f"  [{done}/{n_layers}] {key}  "
            f"shape={tuple(w_dq.shape)}  {dt:.1f}s  ETA={eta/60.0:.1f}m",
            flush=True,
        )
        print(
            f"    DM={dm_v:.6f}  HistCosine={hi_v:.6e}  NVFP4={nv_v:.6f}  SVD={sv_v:.4f}",
            flush=True,
        )

    # =========================================================================
    # Composite ranking
    # =========================================================================
    print("\n=== Composite Ranking ===")
    available_axes = {}
    if axis_dm: available_axes["DM"] = list(axis_dm.values())
    if axis_hist: available_axes["HistCosine"] = list(axis_hist.values())
    if axis_nvfp4: available_axes["NVFP4"] = list(axis_nvfp4.values())
    if axis_svd: available_axes["SVD"] = list(axis_svd.values())

    if not available_axes:
        raise RuntimeError("No axis scores produced")

    # Normalize each axis to [0,1] via min-max, then midrank
    axis_ranks = {}  # key -> {axis_name: rank}
    for axis_name, scores_dict in [("DM", axis_dm), ("HistCosine", axis_hist),
                                     ("NVFP4", axis_nvfp4), ("SVD", axis_svd)]:
        if not scores_dict: continue
        keys_sorted = list(scores_dict.keys())
        vals = [scores_dict[k] for k in keys_sorted]
        ranks = _pool_midranks(vals)
        for k, r in zip(keys_sorted, ranks):
            if k not in axis_ranks: axis_ranks[k] = {}
            axis_ranks[k][axis_name] = r

    # Derive weights
    raw_scores = {name: list(d.values()) for name, d in [("DM", axis_dm), ("HistCosine", axis_hist),
                                                          ("NVFP4", axis_nvfp4), ("SVD", axis_svd)] if d}
    weights = _derive_weights(raw_scores)
    print(f"  Form: {weights.get('form', '?')}")
    for ax in available_axes:
        print(f"  {ax}: weight={weights.get(ax, 0):.4f}  n={len(available_axes[ax])}")

    # Compute composite
    composite = {}
    for key, ranks in axis_ranks.items():
        composite[key] = _composite_4axis(ranks, weights)

    # Sort: LOW composite = safe to NVFP4, HIGH = keep INT8
    sorted_layers = sorted(composite.items(), key=lambda x: x[1])
    print(f"\n  Lowest 5 (NVFP4 candidates):")
    for k, s in sorted_layers[:5]:
        print(f"    {k}  composite={s:.6f}")
    print(f"  Highest 5 (INT8 keep):")
    for k, s in sorted_layers[-5:]:
        print(f"    {k}  composite={s:.6f}")

    # =========================================================================
    # Select layers to convert
    # =========================================================================
    convert_keys = set()

    if all_mlp:
        for layer in int8_layers:
            if ".mlp." in layer["key"]:
                convert_keys.add(layer["key"])
        print(f"\n--all_mlp: {len(convert_keys)} layers")

    if nvfp4_types:
        types = [t.strip() for t in nvfp4_types.split(",") if t.strip()]
        for layer in int8_layers:
            if _match_type(layer["key"], types):
                convert_keys.add(layer["key"])
        print(f"--nvfp4_types: {len(convert_keys)} layers")

    if nvfp4_keep > 0:
        for k, _ in sorted_layers[:nvfp4_keep]:
            convert_keys.add(k)
        print(f"--nvfp4_keep {nvfp4_keep}: {len(convert_keys)} layers (ranking-based)")

    if not convert_keys:
        print("\nWARNING: No layers selected. Use --nvfp4_keep, --nvfp4_types, or --all_mlp")
        return

    print(f"\nTotal to NVFP4: {len(convert_keys)}")
    print(f"Staying INT8:   {len(int8_layers) - len(convert_keys)}")

    # =========================================================================
    # Convert: INT8 -> plain NVFP4 (Comfy finished-product pack)
    # Ranking / Axis scoring above is unchanged. Here we only fix packaging:
    # unrotate ConvRot INT8 → NVFP4 quant → comfy_quant + input_scale, no convrot.
    # =========================================================================
    print("\n=== Converting (plain NVFP4 pack) ===")
    n_nvfp4 = 0
    n_unrotated = 0
    n_skip_no_gs = 0
    input_scale_written = 0
    input_scale_missing = 0
    input_scale_rejected = 0

    for layer in int8_layers:
        key = layer["key"]
        if key not in convert_keys:
            continue

        base = layer["base"]
        q = new_sd[key]
        scale_key = f"{base}.weight_scale"
        cq_key = f"{base}.comfy_quant"
        is_key = f"{base}.input_scale"
        scale = new_sd.get(scale_key)

        if scale is None:
            continue

        # Re-resolve at convert time (sidecar still present on INT8 source)
        mk = layer["meta_key"]
        conf = _resolve_int8_layer_conf(base, mk, layers_meta, new_sd.get(cq_key))
        layer["conf"] = conf

        # Dequantize INT8 (source may still be ConvRot-rotated)
        if scale.dim() == 0:
            w_dq = q.float() * scale.item()
        elif scale.dim() == 2 and scale.shape[1] == 1:
            w_dq = q.float() * scale
        else:
            w_dq = q.float() * scale

        is_convrot = bool(conf.get("convrot", False))
        gs = int(conf.get("convrot_groupsize", 256))
        used_gs = None
        if is_convrot:
            # Features for Linear = in_features (dim1); Conv2d = in_channels (dim1)
            feat = int(w_dq.shape[1])
            used_gs = convrot_group_size_for_features(feat, gs)
            if used_gs is not None:
                h = build_hadamard(used_gs, device="cpu", dtype=torch.float32)
                if w_dq.ndim == 2:
                    w_dq = unrotate_weight(w_dq, h, used_gs)
                elif w_dq.ndim == 4:
                    w_dq = unrotate_weight_conv2d(w_dq, h, used_gs)
                else:
                    print(f"  [ERR] {key}: ConvRot unrotate unsupported ndim={w_dq.ndim}")
                    continue
                n_unrotated += 1
            else:
                # Do NOT plain-pack rotated weights — keep INT8 ConvRot shelter
                print(
                    f"  [SKIP] {key}: ConvRot set but no valid group_size "
                    f"(feat={feat}, gs={gs}); keeping INT8"
                )
                n_skip_no_gs += 1
                continue

        w_bf16 = w_dq.to(dtype=torch.bfloat16, device=device)

        try:
            qdata, params = TensorCoreNVFP4Layout.quantize(w_bf16)
            tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)

            # Keep logical Linear shape for Krea2 detect (txtfusion.projector
            # → txtlayers). Packed weight.shape[1] is storage K, not in_features.
            # Must live in BOTH .comfy_quant and layers_meta — nvfp4_conf reads cq.
            orig_shape = [int(x) for x in params.orig_shape]
            nv_meta = {
                "format": "nvfp4",
                "orig_shape": orig_shape,
                "in_features": int(orig_shape[1]) if len(orig_shape) > 1 else None,
                "out_features": int(orig_shape[0]) if len(orig_shape) > 0 else None,
            }

            # Build replacement first; only mutate new_sd after success
            updates = {}
            for suffix, t in tensors.items():
                updates[f"{base}.weight{suffix}"] = t.cpu()
            # Finished-product: plain NVFP4 sidecar (never stamp ConvRot here)
            updates[cq_key] = _encode_comfy_quant(dict(nv_meta))

            module_name = None
            mk_mod = ck_map.get(key)
            if mk_mod and mk_mod.endswith(".weight"):
                module_name = mk_mod[: -len(".weight")]
            amax = act_amax_dict.get(module_name) if module_name else None
            scale_ok = False
            if amax is None:
                input_scale_missing += 1
            else:
                is_t = _sane_nvfp4_input_scale(float(amax))
                if is_t is None:
                    input_scale_rejected += 1
                    print(
                        f"  [WARN] {key}: reject absurd act_amax={float(amax):.3e} "
                        f"(no .input_scale; runtime amax)"
                    )
                else:
                    updates[is_key] = is_t
                    input_scale_written += 1
                    scale_ok = True

            # Commit: drop INT8 tensors, write NVFP4 pack, scrub stale meta
            del new_sd[key]
            del new_sd[scale_key]
            if cq_key in new_sd:
                del new_sd[cq_key]
            if is_key in new_sd:
                del new_sd[is_key]
            new_sd.update(updates)

            layers_meta[mk] = nv_meta
            # native_convert_int8 often keeps FULL keys in layers — remove stale
            # int8+convrot entry so loaders do not see conflicting conf.
            if base != mk and base in layers_meta:
                del layers_meta[base]

            n_nvfp4 += 1

            print(
                f"  [OK] {key} -> plain NVFP4  orig_shape={tuple(orig_shape)}"
                + ("  (unrotated)" if is_convrot and used_gs is not None else "")
                + ("  +input_scale" if scale_ok else "  (no input_scale)")
            )
            del w_bf16, qdata, params, w_dq
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [ERR] {key}: {e}")

    n_int8_kept = len(int8_layers) - n_nvfp4

    # Save
    quant_meta["layers"] = layers_meta
    fm = OrderedDict()
    fm["_quantization_metadata"] = json.dumps(quant_meta)
    fm["converted_by"] = "HSWQ auto 4-axis hybrid INT8->NVFP4"
    fm["hswq_model"] = "krea2"
    fm["hswq_mixed"] = "1"
    fm["hswq_nvfp4_count"] = str(n_nvfp4)
    fm["hswq_int8_count"] = str(n_int8_kept)
    fm["hswq_nvfp4_pack"] = "plain"
    fm["hswq_nvfp4_convrot"] = "0"
    fm["hswq_axes"] = ",".join(available_axes.keys())
    for k, v in metadata.items():
        if k not in fm and k != "_quantization_metadata":
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            fm[k] = v

    print(f"\nSaving: {output_path}")
    save_file(new_sd, output_path, metadata=fm)
    out_sz = os.path.getsize(output_path)
    in_sz = os.path.getsize(input_path)
    print(f"Done: {out_sz/(1024**3):.2f} GiB (was {in_sz/(1024**3):.2f}, saved {(in_sz-out_sz)/(1024**3):.2f})")
    print(f"  NVFP4 (plain): {n_nvfp4}  unrotated_from_ConvRot={n_unrotated}")
    if n_skip_no_gs:
        print(f"  skipped (ConvRot, no valid gs; kept INT8): {n_skip_no_gs}")
    print(
        f"  input_scale written={input_scale_written}  "
        f"missing={input_scale_missing}  rejected={input_scale_rejected}"
    )
    if input_scale_missing:
        print(
            f"  [WARN] {input_scale_missing} NVFP4 layers lack .input_scale "
            f"(need --calib_file + --clip_path)"
        )
    if input_scale_rejected:
        print(
            f"  [WARN] {input_scale_rejected} NVFP4 layers rejected absurd act_amax "
            f"(no .input_scale; runtime amax)"
        )
    print(f"  INT8 kept: {n_int8_kept}")
    if non_int8_layers:
        print(f"  non-INT8 (bf16/fp16/fp32) kept as-is: {len(non_int8_layers)}")

    # Save ranking JSON
    ranking_path = output_path.replace(".safetensors", "_ranking.json")
    ranking = [{"key": k, "composite": s,
                "dm": axis_dm.get(k, -1), "hist": axis_hist.get(k, -1),
                "nvfp4": axis_nvfp4.get(k, -1), "svd": axis_svd.get(k, -1)}
               for k, s in sorted_layers]
    with open(ranking_path, "w", encoding="utf-8") as rf:
        json.dump(ranking, rf, indent=2)
    print(f"  Ranking: {ranking_path}")

    del new_sd; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def main():
    p = argparse.ArgumentParser(description="Krea2 auto 4-axis hybrid INT8->NVFP4 converter")
    p.add_argument("--input", "--model", dest="input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--nvfp4_keep", type=int, default=0, help="Convert lowest-composite N layers to NVFP4")
    p.add_argument("--nvfp4_types", default=None, help="Comma-separated layer types")
    p.add_argument("--all_mlp", action="store_true")
    p.add_argument("--calib_file", default=None)
    p.add_argument("--clip_path", default=None)
    p.add_argument("--comfy_path", default=None)
    p.add_argument("--num_calib_samples", type=int, default=32)
    p.add_argument("--num_inference_steps", type=int, default=25)
    p.add_argument("--hist_bins", type=int, default=4096,
                   help="Hist Cosine bins (default 4096; was 8192 and hung)")
    p.add_argument("--hist_candidates", type=int, default=200,
                   help="Hist Cosine candidates (default 200; was 2000)")
    p.add_argument("--hist_refine", type=int, default=3,
                   help="Hist Cosine refinement iters (default 3; was 20)")
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found"); sys.exit(1)

    convert(args.input, args.output, device=args.device,
            nvfp4_keep=args.nvfp4_keep, nvfp4_types=args.nvfp4_types,
            all_mlp=args.all_mlp, calib_file=args.calib_file,
            clip_path=args.clip_path, comfy_path=args.comfy_path,
            num_calib_samples=args.num_calib_samples,
            num_inference_steps=args.num_inference_steps,
            hist_bins=args.hist_bins,
            hist_candidates=args.hist_candidates,
            hist_refine=args.hist_refine)

if __name__ == "__main__":
    main()

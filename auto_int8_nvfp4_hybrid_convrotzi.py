"""Z Image ConvRot INT8 -> ConvRot NVFP4 hybrid auto-converter.

Z Image port of the Krea2 ConvRot hybrid converter. Reads a Z Image ConvRot
INT8 checkpoint (e.g. moodyProMix_zitV13_sci_1off_convrot_int8.safetensors),
ranks all INT8 layers by a 4-axis composite, and converts the selected layers
to Kitchen NVFP4 packed in Hadamard-rotated weight space (ConvRot), keeping
the rest as ConvRot INT8 shelter.

Finished-product format matches hswq_convert_nvfp4_zi_int8protect.py output
(e.g. moodyProMix_zitV13_hswq_int8protect60_convrot_nvfp4.safetensors):
  - .weight (U8 pack) / .weight_scale / .weight_scale_2  (Kitchen NVFP4)
  - .comfy_quant  uint8 JSON  {"format":"nvfp4","convrot":true,
                               "convrot_groupsize":G}  -- no orig_shape,
                               no .input_scale (same as Z-Image reference)
  - INT8 shelter keeps ConvRot int8_tensorwise + convrot stamp
  - file meta: hswq_model=z_image, hswq_kitchen_profile=Z-Image-Turbo,
    hswq_nvfp4_convrot=1, hswq_int8_protect=1, hswq_int8_protect_n=N,
    hswq_int8_protect_convrot=N, hswq_int8_protect_source=<input basename>
  - _quantization_metadata: {"format_version":"1.0","layers":{stripped_key:conf}}

ConvRot NVFP4 math (same as the Krea2 ConvRot converter):
  - Source INT8 is ConvRot (Hadamard space). Pack Kitchen NVFP4 in already-
    rotated weight space (kept_rotated only; never re-rotate / never unrotate;
    double-rotate or unrotate = score death).
  - No valid Hadamard group / non-2D selected layer -> keep INT8 shelter.
  - INT8 shelter (kept / skipped NVFP4) MUST stay ConvRot int8_tensorwise.
    If a layer is not ConvRot-armed in the source: raise (PLAIN INT8
    forbidden; do not invent a plain->rotate requantize path).
  - bf16/fp16/fp32 layers carrying a stale .comfy_quant are never scored or
    converted: they stay as-is and their stale sidecars are scrubbed.

DualMonitor calibration is OPTIONAL (Axis 1 only; this format writes no
.input_scale). Without --calib_file/--clip_path, Axis 1 is skipped.

Post-convert axes (reporting):
  Axis 1: DM E[x^2]-weighted NVFP4 error (needs calib for act weights)
  Axis 2: Hist Cosine V5 (SVD+RMS leverage) -- AFTER rotate domain
  Axis 3: NVFP4 error in rotated space (uniform weight)
  Axis 4: SVD Leverage -- AFTER rotate domain

Requires: comfy_kitchen (TensorCoreNVFP4Layout), weighted_histogram_cosine_v5,
ComfyUI with comfy/ldm/lumina/model.py (NextDiT), Qwen3_4B CLIP + Qwen2
tokenizer for calibration.
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
from safetensors.torch import load_file, save_file
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
_BENCH_DIR = os.path.join(_REPO_ROOT, "benchmark")
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from native_convert_int8 import (
    build_hadamard,
    convrot_group_size_for_features,
    rotate_activation,
    rotate_activation_nchw,
)

# Preferred Hadamard group size (INT8 ConvRot / NVFP4 ConvRot family).
_CONVROT_PREFERRED_GS = 256


def _prepare_weight_rotated_for_nvfp4(
    w_dq: torch.Tensor,
    conf: dict,
    *,
    preferred_gs: int = _CONVROT_PREFERRED_GS,
):
    """Hadamard-space weight for ConvRot NVFP4 pack (no re-rotate).

    Source is ConvRot INT8 ? weight is already rotated. Returns as-is.
    Never invent plain->rotate (double-rotate destroys SSIM/MSE).

    Returns (w_rot, used_gs, status).
      status: 'kept_rotated' | 'not_convrot' | 'no_gs' | 'bad_ndim'
    """
    if w_dq.ndim not in (2, 4):
        return None, None, "bad_ndim"
    if not bool(conf.get("convrot", False)):
        return None, None, "not_convrot"
    feat = int(w_dq.shape[1])
    preferred = int(conf.get("convrot_groupsize", preferred_gs) or preferred_gs)
    if preferred > 0 and feat % preferred == 0:
        used_gs = preferred
    else:
        used_gs = convrot_group_size_for_features(feat, preferred_gs)
    if used_gs is None:
        return None, None, "no_gs"
    return w_dq, int(used_gs), "kept_rotated"


def _rotate_act_for_input_scale(
    input_tensor: torch.Tensor,
    module: Optional[torch.nn.Module],
    *,
    preferred_gs: int = _CONVROT_PREFERRED_GS,
) -> torch.Tensor:
    """Hadamard-rotate activation before amax (same order as ConvRot inference)."""
    x = input_tensor.detach().float()
    if module is None:
        return x
    if isinstance(module, torch.nn.Linear):
        in_f = int(module.in_features)
        gs = convrot_group_size_for_features(in_f, preferred_gs)
        if gs is None or int(x.shape[-1]) != in_f:
            return x
        h = build_hadamard(int(gs), device=x.device, dtype=torch.float32)
        flat = x.reshape(-1, in_f)
        return rotate_activation(flat, h, int(gs))
    if isinstance(module, torch.nn.Conv2d) and x.dim() == 4:
        in_c = int(module.in_channels)
        gs = convrot_group_size_for_features(in_c, preferred_gs)
        if gs is None or int(x.shape[1]) != in_c:
            return x
        h = build_hadamard(int(gs), device=x.device, dtype=torch.float32)
        return rotate_activation_nchw(x, h, int(gs))
    return x


def _dequant_int8_weight(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if scale.dim() == 0:
        return q.float() * scale.item()
    if scale.dim() == 2 and scale.shape[1] == 1:
        return q.float() * scale
    return q.float() * scale


def _stamp_int8_convrot(
    new_sd: dict,
    layers_meta: dict,
    *,
    base: str,
    mk: str,
    used_gs: int,
) -> None:
    int8_meta = {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": int(used_gs),
    }
    cq_key = f"{base}.comfy_quant"
    new_sd[cq_key] = _encode_comfy_quant(dict(int8_meta))
    layers_meta[mk] = dict(int8_meta)
    if base != mk:
        layers_meta[base] = dict(int8_meta)


def _ensure_int8_convrot_passthrough(
    new_sd: dict,
    layers_meta: dict,
    *,
    base: str,
    key: str,
    mk: str,
    conf: dict,
    preferred_gs: int = _CONVROT_PREFERRED_GS,
) -> str:
    """素通し = ConvRot INT8. Keep tensors; reinforce stamp only.

    Source is ConvRot INT8 (PLAIN INT8 does not exist in input). Never re-rotate,
    never unrotate, never strip. Raises if conf is not ConvRot-armed.
    """
    scale_key = f"{base}.weight_scale"
    q = new_sd.get(key)
    scale = new_sd.get(scale_key)
    if q is None or scale is None:
        raise RuntimeError(
            f"素通し ConvRot INT8 missing weight/scale for {key}"
        )

    if not bool(conf.get("convrot")):
        raise RuntimeError(
            f"素通し expects ConvRot INT8 but {key} conf lacks convrot=true "
            f"(PLAIN INT8 must not exist in source ? check conf resolve)"
        )

    w_dq = _dequant_int8_weight(q, scale)
    if w_dq.ndim not in (2, 4):
        raise RuntimeError(
            f"素通し ConvRot INT8 bad ndim={w_dq.ndim} for {key}"
        )
    feat = int(w_dq.shape[1])
    preferred = int(conf.get("convrot_groupsize", preferred_gs) or preferred_gs)
    if preferred > 0 and feat % preferred == 0:
        used_gs = preferred
    else:
        used_gs = convrot_group_size_for_features(feat, preferred_gs)
    if used_gs is None:
        raise RuntimeError(
            f"素通し ConvRot INT8 has no valid Hadamard gs for {key} "
            f"(feat={feat})"
        )

    # Tensors already in Hadamard space ? stamp only.
    _stamp_int8_convrot(
        new_sd, layers_meta, base=base, mk=mk, used_gs=int(used_gs)
    )
    return "passthrough_convrot_int8"


def _assert_no_plain_int8(new_sd: dict, int8_layers: list, nvfp4_keys: set) -> None:
    """Audit: every non-NVFP4 INT8 layer must carry convrot=true."""
    plain = []
    for layer in int8_layers:
        key = layer["key"]
        if key in nvfp4_keys:
            continue
        base = layer["base"]
        if key not in new_sd:
            continue
        if f"{base}.weight_scale_2" in new_sd:
            continue
        cq = new_sd.get(f"{base}.comfy_quant")
        conf = _decode_comfy_quant_raw(cq) if cq is not None else {}
        if not conf.get("convrot"):
            plain.append(key)
    if plain:
        raise RuntimeError(
            f"PLAIN INT8 forbidden (素通し must be ConvRot INT8). "
            f"n={len(plain)} examples={plain[:5]}"
        )


# Only integer-typed .weight tensors are real INT8 layers. bf16/fp16/fp32
# weights carrying a stale .comfy_quant (leftover from an earlier conversion)
# must never be scored or converted -- keep them as-is.
_INT8_WEIGHT_DTYPES = (torch.int8, torch.uint8)
# safetensors get_slice().get_dtype() returns dtype-code STRINGS like
# "I8"/"U8" on some versions, torch.dtype on others -- normalize.
_SAFE_DTYPE_CODES = {
    "I8": torch.int8, "U8": torch.uint8,
    "I16": torch.int16, "U16": torch.uint16,
    "I32": torch.int32, "U32": torch.uint32,
    "I64": torch.int64, "U64": torch.uint64,
    "F16": torch.float16, "BF16": torch.bfloat16,
    "F32": torch.float32, "F64": torch.float64,
    "F8_E4M3": getattr(torch, "float8_e4m3fn", None),
    "F8_E5M2": getattr(torch, "float8_e5m2", None),
}

def _is_int8_weight_dtype(d) -> bool:
    if isinstance(d, str):
        d = _SAFE_DTYPE_CODES.get(d)
    return d in _INT8_WEIGHT_DTYPES


# Hist Cosine V5 import (amax search: L = 1 - cosine)
_HIST_DIR = os.path.join(_REPO_ROOT, "histogram")
if _HIST_DIR not in sys.path:
    sys.path.insert(0, _HIST_DIR)
try:
    from weighted_histogram_cosine_v5 import HSWQWeightedHistogramOptimizerV5
    from weighted_histogram_cosine_v5 import compute_hybrid_leverage_scores
    from weighted_histogram_cosine_v5 import WeightedHistogram
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
            # ConvRot domain: rotate act first, THEN amax (report only; this
            # format writes no .input_scale).
            x_for_amax = _rotate_act_for_input_scale(
                inp_detached, module, preferred_gs=_CONVROT_PREFERRED_GS
            )
            amax_val = float(x_for_amax.abs().amax().item())
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
        if os.path.isfile(os.path.join(root, "comfy", "ldm", "lumina", "model.py")):
            if root not in sys.path: sys.path.insert(0, root)
            return root
    raise FileNotFoundError("ComfyUI root with comfy/ldm/lumina/model.py not found.")

def detect_zit_dit_config(sd, prefix):
    """Z Image (NextDiT) config from checkpoint keys (prefix-aware)."""
    layer_re = re.compile(r"^" + re.escape(prefix) + r"layers\.(\d+)\.")
    layer_indices = set()
    for k in sd:
        m = layer_re.match(k)
        if m: layer_indices.add(int(m.group(1)))
    num_layers = max(layer_indices) + 1 if layer_indices else 30

    if f"{prefix}x_embedder.weight" in sd:
        hidden_size = int(sd[f"{prefix}x_embedder.weight"].shape[0])
    else:
        hidden_size = 3072

    ctx_re = re.compile(r"^" + re.escape(prefix) + r"context_refiner\.(\d+)\.")
    refiner_indices = set()
    for k in sd:
        m = ctx_re.match(k)
        if m: refiner_indices.add(int(m.group(1)))
    num_context_refiner = max(refiner_indices) + 1 if refiner_indices else 2

    intermediate_size = None
    w1_key = f"{prefix}layers.0.feed_forward.w1.weight"
    if w1_key in sd:
        # NVFP4 packs K (in_features); out_features (shape[0]) stays logical.
        intermediate_size = int(sd[w1_key].shape[0])

    qk_norm = any(k.endswith(".attention.q_norm.weight") for k in sd)

    return {
        "image_model": "z_image",
        "patch_size": 2,
        "in_channels": 16,
        "dim": hidden_size,
        "n_layers": num_layers,
        "n_refiner_layers": num_context_refiner,
        "n_heads": hidden_size // 128,
        "n_kv_heads": hidden_size // 128,
        "multiple_of": 256,
        "norm_eps": 1e-5,
        "cap_feat_dim": 2560,
        "z_image_modulation": True,
        "pad_tokens_multiple": 64,
        "intermediate_size": intermediate_size,
        "qk_norm": qk_norm,
    }

def load_zit(path, device="cuda", comfy_path=None):
    """Load Z Image NextDiT in FP16 with INT8 dequant (Hadamard space kept).

    Calibration needs real weights: dequant int8_tensorwise (weight_scale)
    BEFORE load_state_dict -- raw int8 codes cast to fp16 would be ~100x off.
    ConvRot weights stay in Hadamard space after dequant (never unrotate).
    """
    if str(device).startswith("cpu"):
        raise RuntimeError("Calibration requires CUDA.")
    _ensure_comfyui(comfy_path)
    saved = _clear_argv_for_comfy()
    try:
        import comfy.options; comfy.options.enable_args_parsing(False)
        _install_comfy_stubs()
        from benchmark.kitchen_rms_rope_fallback import ensure_kitchen_rms_rope
        ensure_kitchen_rms_rope()
        import comfy.ops
        from comfy.ldm.lumina.model import NextDiT
        print(f"Loading Z Image NextDiT: {path}")
        sd = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys(): sd[k] = f.get_tensor(k)
        prefix = _find_prefix(set(sd.keys()))
        cfg = detect_zit_dit_config(sd, prefix)
        print(f"Config: {cfg}")
        kw = {k: v for k, v in cfg.items()
              if k not in ("image_model", "intermediate_size", "qk_norm")}
        if cfg.get("intermediate_size"):
            kw["ffn_dim_multiplier"] = float(cfg["intermediate_size"]) / float(cfg["dim"])
        if cfg.get("qk_norm"):
            kw["qk_norm"] = True
        dit = NextDiT(**kw, device=device, dtype=torch.float16,
                      operations=comfy.ops.disable_weight_init)
        stripped = {}
        for k, v in sd.items():
            if prefix and k.startswith(prefix): stripped[k[len(prefix):]] = v
            elif not prefix: stripped[k] = v
        n_dq = 0
        for k in list(stripped.keys()):
            if not k.endswith(".weight"):
                continue
            base_k = k[: -len(".weight")]
            if f"{base_k}.weight_scale_2" in stripped:
                continue  # NVFP4-packed (not present in INT8 source)
            scale_t = stripped.get(f"{base_k}.weight_scale")
            if scale_t is None or stripped[k].dtype != torch.int8:
                continue
            stripped[k] = _dequant_int8_weight(stripped[k], scale_t).to(torch.float16)
            n_dq += 1
        print(f"  int8 dequant for calib: {n_dq} layers")
        m, u = dit.load_state_dict(stripped, strict=False)
        print(f"  missing={len(m)} unexpected={len(u)}")
        dev = str(next(dit.parameters()).device)
        if not dev.startswith("cuda"): raise RuntimeError(f"DiT on {dev}, not CUDA")
        ck_map = {}
        for name, mod in dit.named_modules():
            w = getattr(mod, "weight", None)
            if w is None or not torch.is_tensor(w) or w.ndim not in (2, 4): continue
            ck = f"{prefix}{name}.weight"
            if ck in sd: ck_map[ck] = f"{name}.weight"
        print(f"  identity map: {len(ck_map)} entries")
        dit.eval()
        return dit, sd, ck_map, prefix
    finally:
        _restore_argv(saved)

def run_calibration(input_path, calib_file, clip_path, num_samples, num_steps,
                    device, comfy_path=None, tokenizer_path=None):
    """DualMonitor calibration for Z Image (Qwen3_4B CLIP + Qwen2 tokenizer)."""
    global _dm_timestep_weight
    print("Preparing calibration data...")
    with open(calib_file, "r", encoding="utf-8") as f:
        prompts = [l.strip() for l in f if l.strip()]
    if len(prompts) < num_samples:
        prompts = (prompts * (num_samples // len(prompts) + 1))[:num_samples]
    else:
        prompts = prompts[:num_samples]

    _ensure_comfyui(comfy_path)
    saved = _clear_argv_for_comfy()
    try:
        import comfy.options; comfy.options.enable_args_parsing(False)
        _install_comfy_stubs()
        from benchmark.zi_convrot_nvfp4_bench import (
            resolve_path, resolve_tokenizer_offline,
        )
        from benchmark.kitchen_rms_rope_fallback import ensure_kitchen_rms_rope
        ensure_kitchen_rms_rope()
        from transformers import Qwen2Tokenizer
        from comfy.text_encoders import llama as llama_module
        import comfy.ops
        import comfy.k_diffusion.sampling as k_sampling

        tok_resolved = resolve_tokenizer_offline(tokenizer_path, comfy_path)
        if tok_resolved:
            print(f"  [calib] Tokenizer (disk): {tok_resolved}")
            try:
                tokenizer = Qwen2Tokenizer.from_pretrained(
                    tok_resolved, local_files_only=True)
            except Exception as e:
                print(f"  [calib] local_files_only failed ({e}); retrying...")
                tokenizer = Qwen2Tokenizer.from_pretrained(tok_resolved)
        else:
            mid = tokenizer_path if tokenizer_path else "Qwen/Qwen2.5-7B-Instruct"
            print(f"  [calib] Tokenizer repo id: {mid}")
            tokenizer = Qwen2Tokenizer.from_pretrained(mid, local_files_only=True)

        resolved_clip = resolve_path(clip_path, is_file=True)
        text_encoder = llama_module.Qwen3_4B(
            config_dict={},
            device=device,
            dtype=torch.float16,
            operations=comfy.ops.disable_weight_init,
        ).to(device)
        print(f"  [calib] Loading CLIP weights from: {resolved_clip}")
        text_encoder.load_state_dict(load_file(resolved_clip), strict=False)
        text_encoder.eval()

        model, _sd, ck_map, _prefix = load_zit(
            input_path, device=device, comfy_path=comfy_path)

        print("Setting up DualMonitor hooks...")
        dual_monitors.clear()
        handles = []
        for name, mod in model.named_modules():
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                handles.append(
                    mod.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))
                )

        template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        print(f"Running calibration ({len(prompts)} samples, {num_steps} steps)...")

        class _ZITWrapper:
            def __init__(self, model, embeds, mask):
                self.model = model
                self.embeds = embeds
                self.mask = mask
            def __call__(self, x, sigma, **kwargs):
                global _dm_timestep_weight
                s = float(sigma.item()) if torch.is_tensor(sigma) else float(sigma)
                _dm_timestep_weight = float(1.0 - s)
                out = self.model(
                    x.to(torch.float16),
                    sigma.to(torch.float16),
                    self.embeds.to(torch.float16),
                    None,
                    attention_mask=self.mask,
                )
                if isinstance(out, tuple):
                    out = out[0]
                return out.to(x.dtype)

        for i, prompt in enumerate(prompts):
            formatted = template.format(prompt)
            tokens = tokenizer(
                formatted, return_tensors="pt", padding=True,
                truncation=True, max_length=256,
            )
            input_ids = tokens["input_ids"].to(device)
            am_t = tokens["attention_mask"].to(device)
            with torch.no_grad():
                outputs = text_encoder(
                    input_ids=input_ids,
                    attention_mask=am_t,
                    intermediate_output=-2,
                )
                ctx = outputs[1].to(device=device, dtype=torch.float16)
            am = am_t.bool().to(device)
            gen = torch.Generator(device=device).manual_seed(42 + i)
            x = torch.randn(1, 16, 128, 128, device=device,
                            dtype=torch.float16, generator=gen)
            sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
            wrapper = _ZITWrapper(model, ctx, am)
            with torch.no_grad():
                k_sampling.sample_euler(wrapper, x, sigmas, disable=False)
            del x, ctx, am, input_ids, am_t, tokens
            if device == "cuda":
                torch.cuda.empty_cache()
            if (i + 1) % 10 == 0:
                gc.collect(); torch.cuda.empty_cache()
        for h in handles:
            h.remove()

        act_sq = {}
        act_amax = {}
        for name, mon in dual_monitors.items():
            if mon.channel_act_sq_mean is not None:
                act_sq[name] = mon.channel_act_sq_mean.detach().float().cpu()
            if mon.act_amax > 0.0 and math.isfinite(mon.act_amax):
                act_amax[name] = float(mon.act_amax)
        print(f"  DualMonitor: {len(act_sq)} layers act_sq, {len(act_amax)} layers act_amax")
        del model, text_encoder, tokenizer
        dual_monitors.clear()
        gc.collect(); torch.cuda.empty_cache()
        return act_sq, act_amax, ck_map
    finally:
        _restore_argv(saved)

# =========================================================================
# Helpers
# =========================================================================
def _find_prefix(keys):
    for p in ("model.diffusion_model.", "diffusion_model.", ""):
        if (f"{p}layers.0.feed_forward.w1.weight" in keys
                or f"{p}cap_embedder.1.weight" in keys
                or f"{p}layers.0.attention_norm1.weight" in keys):
            return p
    raise ValueError("Not a Z Image checkpoint")

def _meta_key(key, prefix):
    if prefix and key.startswith(prefix): return key[len(prefix):]
    return key

def _match_type(key, types):
    return any(f".{t}.weight" in key for t in types)

def _encode_comfy_quant(config: dict) -> torch.Tensor:
    """Comfy sidecar: uint8 JSON bytes (same as hswq_convert_nvfp4_zi_*)."""
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

# =========================================================================
# Main convert
# =========================================================================
def convert(input_path, output_path, *, device="cuda",
            nvfp4_keep=0, nvfp4_types=None, all_mlp=False,
            calib_file=None, clip_path=None, comfy_path=None,
            tokenizer_path=None, model_type="Z-Image-Turbo",
            num_calib_samples=32, num_inference_steps=25,
            hist_bins=4096, hist_candidates=200, hist_refine=3,
            fast=False):

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
        try:
            quant_meta = json.loads(qm_str)
        except (TypeError, json.JSONDecodeError):
            quant_meta = {}
        layers_meta = quant_meta.get("layers", {}) if isinstance(quant_meta, dict) else {}

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
            # Guard: only integer-typed weights are INT8 layers. bf16/fp16/fp32
            # layers that carry a stale .comfy_quant must NOT be scored or
            # converted -- keep them as-is (bf16 preserve policy).
            try:
                w_dtype = f.get_slice(key).get_dtype()
            except Exception:
                w_dtype = f.get_tensor(key).dtype
            if not _is_int8_weight_dtype(w_dtype):
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
    print(f"Z Image prefix: {prefix!r}")
    print(
        f"  conf resolve: sidecar={n_conf_sidecar}  full_meta={n_conf_full}  "
        f"stripped_meta={n_conf_stripped}  empty={n_conf_empty}  "
        f"convrot_armed={n_conf_convrot}"
    )

    # Optional: DualMonitor calibration (Axis 1 only; no .input_scale in this
    # format -- matches hswq_convert_nvfp4_zi_int8protect output)
    act_sq_dict = {}
    ck_map = {}
    use_calib = bool(calib_file) and bool(clip_path)
    if use_calib:
        print("\n=== DualMonitor Calibration ===")
        act_sq_dict, _amax_unused, ck_map = run_calibration(
            input_path, calib_file, clip_path,
            int(num_calib_samples), int(num_inference_steps),
            device, comfy_path, tokenizer_path)
    else:
        print("\n[SKIP] No calibration -- Axis 1 (DM E[x^2]) disabled")

    # Load all tensors
    print("\nLoading checkpoint...")
    new_sd = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for key in tqdm(sorted(all_keys), desc="Load"):
            new_sd[key] = f.get_tensor(key)

    # Scrub stale quantization sidecars from non-INT8 (bf16/fp16/fp32) layers
    # so the finished file holds clean plain layers -- loaders read
    # .comfy_quant first and would otherwise misinterpret them as quantized.
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
        print(f"  [FIX] {layer['key']}: non-INT8 weight -> kept as-is, "
              f"stale quant sidecar scrubbed")

    # Hist Cosine V5 ? amax via Cosine loss
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
        f"(per-layer progress; HistCosine bins={hist_bins}/"
        f"cand={hist_candidates}/refine={hist_refine})"
        + ("  [FAST: DM x NVFP4 only]" if fast else ""),
        flush=True,
    )
    axis_dm = {}       # Axis 1: DM E[x^2]-weighted NVFP4 error
    axis_hist = {}     # Axis 2: Hist Cosine V5 (1 - cos_sim)
    axis_nvfp4 = {}    # Axis 3: NVFP4 error in rotated space
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

        print(f"  [{i+1}/{n_layers}] {key}  shape={tuple(w_dq.shape)}  scoring...",
              flush=True)

        # --- Pre-compute NVFP4 dequant (used by Axis 1 and Axis 3) ---
        w_rot_3, _gs3, st3 = _prepare_weight_rotated_for_nvfp4(w_dq, conf)
        nvfp4_dq = None
        if w_rot_3 is not None and w_rot_3.ndim == 2 and st3 == "kept_rotated":
            try:
                w_nv = w_rot_3.to(dtype=torch.bfloat16, device=device)
                qdata_3, params_3 = TensorCoreNVFP4Layout.quantize(w_nv)
                if hasattr(TensorCoreNVFP4Layout, "dequantize"):
                    nvfp4_dq = TensorCoreNVFP4Layout.dequantize(
                        qdata_3, params_3).float().cpu()
                del qdata_3, params_3, w_nv
                if device == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                nvfp4_dq = None

        # --- Axis 1: DM E[x^2]-weighted NVFP4 error ---
        module_name = None
        mk = ck_map.get(key)
        if mk and mk.endswith(".weight"): module_name = mk[:-len(".weight")]
        act_sq = act_sq_dict.get(module_name) if module_name else None
        if act_sq is not None and nvfp4_dq is not None and act_sq.shape[0] == w_dq.shape[1]:
            act_scale = act_sq.sqrt()
            err_nv = w_dq.float().cpu() - nvfp4_dq
            if err_nv.ndim == 2:
                we = err_nv * act_scale.unsqueeze(0)
                wb = w_dq.float().cpu() * act_scale.unsqueeze(0)
            else:
                we = err_nv; wb = w_dq.float().cpu()
            axis_dm[key] = float(we.norm().item()) / max(float(wb.norm().item()), 1e-8)

        # --- Pre-compute SVD hybrid leverage ONCE (used by Axis 2 and Axis 4) ---
        # --fast skips this: full torch.linalg.svd per layer can take minutes.
        hybrid_imp = None
        if not fast:
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
        # --fast skips: Hist Cosine V5 needs the SVD-based importance.
        if not fast:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    optimal_amax = hist_opt.compute_optimal_amax(
                        w_dq, importance=hybrid_imp, use_svd_leverage=False,
                        scaled=False, loss_type="cosine")
                    wh = WeightedHistogram(bins=hist_opt.bins, device=hist_opt.device)
                    wh.build(w_dq, hybrid_imp)
                    hist = wh.get_histogram()
                    bc = wh.get_bin_centers()
                    est_loss = hist_opt.cosine_optimizer.compute_weighted_cosine(
                        hist, bc, optimal_amax, scaled=False, loss_type="cosine")
                axis_hist[key] = float(est_loss)
            except Exception:
                pass

        # --- Axis 3: NVFP4 error in rotated space (uniform weight) ---
        if nvfp4_dq is not None:
            w_ref = w_rot_3.float().cpu() if w_rot_3 is not None else w_dq.float().cpu()
            err_nv3 = w_ref - nvfp4_dq
            axis_nvfp4[key] = float(err_nv3.norm().item()) / max(
                float(w_ref.norm().item()), 1e-8)

        # --- Axis 4: SVD Leverage (reuse pre-computed hybrid_imp) ---
        # --fast skips (needs full SVD).
        if not fast:
            try:
                if hybrid_imp is not None:
                    axis_svd[key] = float(hybrid_imp.mean().item())
                else:
                    U, S, Vh = torch.linalg.svd(
                        w_dq.to(device=hist_dev, dtype=torch.float32),
                        full_matrices=False)
                    leverage = (U ** 2 * S.unsqueeze(0) ** 2).sum(dim=1).mean().item()
                    axis_svd[key] = leverage
                    del U, S, Vh
            except Exception:
                pass

        # Cleanup per-layer temporaries
        del nvfp4_dq, w_rot_3
        if device == "cuda":
            torch.cuda.empty_cache()

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
            if ".feed_forward." in layer["key"]:
                convert_keys.add(layer["key"])
        print(f"\n--all_mlp (feed_forward): {len(convert_keys)} layers")

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
    # Convert: ConvRot INT8 -> ConvRot NVFP4 (Hadamard pack; never re-rotate).
    # Score drop from double-rotate / unrotate = death.
    # =========================================================================
    print("\n=== Converting (ConvRot NVFP4: Hadamard space pack) ===")
    n_nvfp4 = 0
    n_kept_rotated = 0
    n_passthrough = 0
    n_skip_no_gs = 0
    n_skip_ndim = 0
    nvfp4_done_keys = set()

    for layer in int8_layers:
        key = layer["key"]
        base = layer["base"]
        mk = layer["meta_key"]
        cq_key = f"{base}.comfy_quant"
        conf = _resolve_int8_layer_conf(base, mk, layers_meta, new_sd.get(cq_key))
        layer["conf"] = conf

        if key not in convert_keys:
            _ensure_int8_convrot_passthrough(
                new_sd, layers_meta, base=base, key=key, mk=mk, conf=conf
            )
            n_passthrough += 1
            continue

        q = new_sd[key]
        scale_key = f"{base}.weight_scale"
        scale = new_sd.get(scale_key)

        if scale is None:
            _ensure_int8_convrot_passthrough(
                new_sd, layers_meta, base=base, key=key, mk=mk, conf=conf
            )
            n_passthrough += 1
            continue

        # Dequantize INT8 (already Hadamard space when ConvRot)
        if scale.dim() == 0:
            w_dq = q.float() * scale.item()
        elif scale.dim() == 2 and scale.shape[1] == 1:
            w_dq = q.float() * scale
        else:
            w_dq = q.float() * scale

        # ConvRot NVFP4: pack in Hadamard space only (never re-rotate).
        w_rot, used_gs, st = _prepare_weight_rotated_for_nvfp4(w_dq, conf)
        if st == "not_convrot":
            raise RuntimeError(
                f"NVFP4 ConvRot pack requires source ConvRot INT8 but {key} "
                f"conf lacks convrot=true (PLAIN INT8 forbidden)"
            )
        if st == "no_gs" or used_gs is None or w_rot is None:
            feat = int(w_dq.shape[1]) if w_dq.ndim >= 2 else -1
            print(
                f"  [SKIP] {key}: no valid Hadamard group "
                f"(feat={feat}); keeping ConvRot INT8"
            )
            n_skip_no_gs += 1
            _ensure_int8_convrot_passthrough(
                new_sd, layers_meta, base=base, key=key, mk=mk, conf=conf
            )
            n_passthrough += 1
            continue
        if w_rot.ndim != 2:
            print(
                f"  [SKIP] {key}: ndim={w_rot.ndim} not Linear 2D NVFP4; "
                f"keeping ConvRot INT8"
            )
            n_skip_ndim += 1
            _ensure_int8_convrot_passthrough(
                new_sd, layers_meta, base=base, key=key, mk=mk, conf=conf
            )
            n_passthrough += 1
            continue
        if st != "kept_rotated":
            raise RuntimeError(
                f"NVFP4 ConvRot pack aborted for {key}: status={st} "
                f"(only kept_rotated allowed ? re-rotate/unrotate = score death)"
            )
        n_kept_rotated += 1

        w_bf16 = w_rot.to(dtype=torch.bfloat16, device=device)

        try:
            qdata, params = TensorCoreNVFP4Layout.quantize(w_bf16)
            tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)

            # Z-Image reference format: convrot stamp only (no orig_shape, no
            # .input_scale) -- same as hswq_convert_nvfp4_zi_int8protect.py.
            nv_meta = {
                "format": "nvfp4",
                "convrot": True,
                "convrot_groupsize": int(used_gs),
            }

            # Build replacement first; only mutate new_sd after success
            updates = {}
            for suffix, t in tensors.items():
                updates[f"{base}.weight{suffix}"] = t.cpu()
            updates[cq_key] = _encode_comfy_quant(dict(nv_meta))

            # Commit: drop INT8 tensors, write ConvRot NVFP4 pack, scrub stale meta
            del new_sd[key]
            del new_sd[scale_key]
            if cq_key in new_sd:
                del new_sd[cq_key]
            new_sd.update(updates)

            layers_meta[mk] = dict(nv_meta)
            if base != mk and base in layers_meta:
                del layers_meta[base]

            n_nvfp4 += 1
            nvfp4_done_keys.add(key)

            print(
                f"  [OK] {key} -> ConvRot NVFP4  gs={used_gs}  (kept_rotated)"
            )
            del w_bf16, qdata, params, w_dq, w_rot
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [ERR] {key}: {e}")
            _ensure_int8_convrot_passthrough(
                new_sd, layers_meta, base=base, key=key, mk=mk, conf=conf
            )
            n_passthrough += 1

    _assert_no_plain_int8(new_sd, int8_layers, nvfp4_done_keys)
    n_int8_kept = len(int8_layers) - n_nvfp4

    # Save -- metadata identical in shape to hswq_convert_nvfp4_zi_int8protect
    quant_meta = {"format_version": "1.0", "layers": layers_meta}
    fm = OrderedDict()
    fm["_quantization_metadata"] = json.dumps(quant_meta)
    fm["converted_by"] = "ComfyUI Kitchen NVFP4 Converter (Z-Image ConvRot + INT8 protect)"
    fm["converter_url"] = "https://github.com/tritant/ComfyUI_Kitchen_nvfp4_Converter"
    fm["hswq_model"] = "z_image"
    fm["hswq_kitchen_profile"] = model_type
    fm["hswq_nvfp4_convrot"] = "1"
    fm["hswq_int8_protect"] = "1"
    fm["hswq_int8_protect_n"] = str(n_int8_kept)
    fm["hswq_int8_protect_convrot"] = str(n_int8_kept)
    fm["hswq_int8_protect_source"] = os.path.splitext(os.path.basename(input_path))[0]

    print(f"\nSaving: {output_path}")
    save_file(new_sd, output_path, metadata=fm)
    out_sz = os.path.getsize(output_path)
    in_sz = os.path.getsize(input_path)
    print(f"Done: {out_sz/(1024**3):.2f} GiB (was {in_sz/(1024**3):.2f}, saved {(in_sz-out_sz)/(1024**3):.2f})")
    print(f"  NVFP4 (ConvRot): {n_nvfp4}  kept_rotated={n_kept_rotated}")
    if n_passthrough:
        print(f"  INT8 素通し (ConvRot stamp): {n_passthrough}")
    if n_skip_no_gs:
        print(f"  skipped (no valid gs; kept ConvRot INT8): {n_skip_no_gs}")
    if n_skip_ndim:
        print(f"  skipped (non-2D; kept ConvRot INT8): {n_skip_ndim}")
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
    p = argparse.ArgumentParser(
        description="Z Image auto 4-axis hybrid ConvRot INT8->NVFP4 converter")
    p.add_argument("--input", "--model", dest="input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--nvfp4_keep", type=int, default=0,
                   help="Convert lowest-composite N layers to NVFP4 (rest stay INT8 shelter)")
    p.add_argument("--nvfp4_types", default=None,
                   help="Comma-separated layer types (e.g. feed_forward,attention,cap_embedder,context_refiner)")
    p.add_argument("--all_mlp", action="store_true",
                   help="Convert all feed_forward layers to NVFP4")
    p.add_argument("--calib_file", default=None)
    p.add_argument("--clip_path", default=None)
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--comfy_path", default=None)
    p.add_argument("--model_type", default="Z-Image-Turbo",
                   help="hswq_kitchen_profile tag (default Z-Image-Turbo)")
    p.add_argument("--num_calib_samples", type=int, default=32)
    p.add_argument("--num_inference_steps", type=int, default=25)
    p.add_argument("--hist_bins", type=int, default=4096)
    p.add_argument("--hist_candidates", type=int, default=200)
    p.add_argument("--hist_refine", type=int, default=3)
    p.add_argument("--fast", action="store_true",
                   help="Skip SVD/HistCosine axes (DM x NVFP4 only)")
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found"); sys.exit(1)

    convert(args.input, args.output, device=args.device,
            nvfp4_keep=args.nvfp4_keep, nvfp4_types=args.nvfp4_types,
            all_mlp=args.all_mlp, calib_file=args.calib_file,
            clip_path=args.clip_path, comfy_path=args.comfy_path,
            tokenizer_path=args.tokenizer_path, model_type=args.model_type,
            num_calib_samples=args.num_calib_samples,
            num_inference_steps=args.num_inference_steps,
            hist_bins=args.hist_bins,
            hist_candidates=args.hist_candidates,
            hist_refine=args.hist_refine,
            fast=args.fast)

if __name__ == "__main__":
    main()

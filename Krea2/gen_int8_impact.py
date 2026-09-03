#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Krea2 INT8 converter ranked by diag_impact trajectory impact.

Applies the diag_impact methodology (per-layer trajectory divergence) to INT8
conversion of the bf16/fp16 BASE checkpoint (comparison target = the FULL fp16
model, same pristine-run contract as diag_impact):

  1. load the base bf16/fp16 Krea2 SingleStreamDiT (unquantized)
  2. run a pristine FP16 trajectory (randn context, linspace sigmas)
  3. for EVERY eligible Linear: quantize its weight with the exact v1.5 INT8
     convention (pack_channelwise) and inject the quantize->dequant error,
     then re-run the trajectory and record rel-MSE vs pristine
  4. rank all Linears by impact; the --keep N highest-impact Linears stay in
     original dtype, all remaining eligible Linears become INT8
  5. INT8 conversion follows hswq_convrot_int8_krea2_v1.5 conventions:
     FULL ConvRot (rotate_weight -> pack_channelwise, gs=256 preferred),
     ComfyUI-compatible output (weight / weight_scale / comfy_quant,
     _quantization_metadata format_version "1.0")

Usage:
    python Krea2/gen_int8_impact.py <base.safetensors> <out.safetensors> \
        [--keep 0] [--steps 4] [--lat 128] [--seq 256] [--seed 42] \
        [--comfy-path ComfyUI-master] [--no-convrot]
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import io
import json
import math
import os
import re
import sys
import types

import torch
from safetensors.torch import load_file, save_file

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ---------------------------------------------------------------------------
# v1.5 constants (INT8 packing / ConvRot / structure blacklist)
# ---------------------------------------------------------------------------
_DEFAULT_GROUPSIZE = 256
_KREA2_BLACKLIST: list[str] = [
    "first.",
    "last.",
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


def _is_blacklisted(key: str) -> bool:
    return any(name in key for name in _KREA2_BLACKLIST)


def _is_non_diffusion_key(key: str) -> bool:
    return any(marker in key for marker in _NON_DIFFUSION_MARKERS)


def _encode_comfy_quant(config: dict) -> torch.Tensor:
    return torch.tensor(
        list(json.dumps(config, separators=(",", ":")).encode("utf-8")),
        dtype=torch.uint8,
    )


def _meta_base_key(base_k_file: str) -> str:
    if "model.diffusion_model." in base_k_file:
        return base_k_file.split("model.diffusion_model.")[-1]
    if "diffusion_model." in base_k_file:
        return base_k_file.split("diffusion_model.")[-1]
    return base_k_file


def pack_channelwise(weight: torch.Tensor):
    """Per-out-channel INT8 (v1.5 / ConvRot kitchen dequant shape)."""
    w = weight.float()
    reduce_dims = tuple(range(1, w.dim()))
    amax = torch.clamp(w.abs().amax(dim=reduce_dims).reshape(-1), min=1e-6)
    scale = amax / 127.0
    if w.dim() == 4:
        scale_view = scale.view(-1, 1, 1, 1)
    elif w.dim() == 2:
        scale_view = scale.view(-1, 1)
    else:
        raise ValueError(f"unsupported weight ndim={w.dim()} for channelwise INT8")
    clamped = torch.clamp(w, -scale_view * 127.0, scale_view * 127.0)
    q = (clamped / scale_view).round().clamp(-127, 127).to(torch.int8)
    return q, scale_view.to(dtype=torch.float32)


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
    """Offline Conv2d: rotate along in_channels. weight (O, I, kH, kW)."""
    if weight.ndim != 4:
        raise ValueError(f"Conv2d weight must be 4D, got ndim={weight.ndim}")
    out_c, in_c, k_h, k_w = weight.shape
    flat = weight.permute(0, 2, 3, 1).contiguous().view(-1, in_c)
    flat_rot = rotate_weight(flat, h_matrix, group_size)
    return flat_rot.view(out_c, k_h, k_w, in_c).permute(0, 3, 1, 2).contiguous()


def int8_quant_dequant(weight: torch.Tensor) -> torch.Tensor:
    """INT8 quantize->dequant (per-out-channel, same convention as
    pack_channelwise) used for trajectory impact injection. Returns float."""
    w = weight.float()
    reduce_dims = tuple(range(1, w.dim()))
    amax = torch.clamp(w.abs().amax(dim=reduce_dims).reshape(-1), min=1e-6)
    scale = amax / 127.0
    if w.dim() == 2:
        scale_view = scale.view(-1, 1)
    elif w.dim() == 4:
        scale_view = scale.view(-1, 1, 1, 1)
    else:
        raise ValueError(f"unsupported weight ndim={w.dim()} for INT8 injection")
    clamped = torch.clamp(w, -scale_view * 127.0, scale_view * 127.0)
    q = (clamped / scale_view).round().clamp(-127, 127)
    return q * scale_view


# ---------------------------------------------------------------------------
# ComfyUI bootstrap (same as Krea2/diag_impact.py @ 8/24 baseline)
# ---------------------------------------------------------------------------
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
        m.__file__ = "<hswq_torchaudio_stub>"
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
        def __init__(self, *a, **k):
            pass

        def __call__(self, x):
            return x

        def to(self, *args, **kwargs):
            return self

    class _ML:
        def __init__(self, *a, **k):
            pass

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
        import comfy_aimdo  # noqa: F401
    except Exception:
        m = types.ModuleType("comfy_aimdo")
        m.__file__ = "<hswq_comfy_aimdo_stub>"
        m.__path__ = []
        sys.modules["comfy_aimdo"] = m
        sys.modules["comfy_aimdo.filter"] = types.ModuleType("comfy_aimdo.filter")
        sys.modules["comfy_aimdo.filter"].filter_modules = lambda *a, **k: None
    try:
        import psutil  # noqa: F401
    except Exception:
        class _VM:
            total = 64 * 1024 ** 3
            available = 32 * 1024 ** 3

        class _P:
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
        ps.Process = lambda: _P()
        sys.modules["psutil"] = ps


def _ensure_comfyui(comfy_path=None):
    """Locate the ComfyUI root. Repository-internal ComfyUI-master ONLY."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.normpath(os.path.join(here, ".."))
    candidates = []
    if comfy_path:
        candidates.append(os.path.abspath(comfy_path))
    candidates.append(os.path.join(repo, "ComfyUI-master"))
    env = os.environ.get("COMFYUI_PATH")
    if env:
        candidates.append(env)
    for root in candidates:
        if os.path.isfile(os.path.join(root, "comfy", "ldm", "krea2", "model.py")) \
                and os.path.isfile(os.path.join(root, "comfy", "ops.py")):
            return root
    raise FileNotFoundError(
        "ComfyUI root (needs comfy/ops.py + comfy/ldm/krea2/model.py) not found. "
        "Expected <repo>/ComfyUI-master. Pass --comfy-path or set COMFYUI_PATH."
    )


def _load_comfy_pkg(comfy_root):
    """Import `comfy` EXCLUSIVELY from comfy_root (repo ComfyUI-master)."""
    for key in list(sys.modules):
        if key == "comfy" or key.startswith("comfy."):
            del sys.modules[key]
    if comfy_root in sys.path:
        sys.path.remove(comfy_root)
    sys.path.insert(0, comfy_root)
    mod = importlib.import_module("comfy")
    if os.path.abspath(getattr(mod, "__path__", [None])[0] or "") != \
            os.path.abspath(os.path.join(comfy_root, "comfy")):
        raise ImportError(
            f"comfy resolved outside repo tree: {getattr(mod, '__path__', None)} "
            f"(expected {os.path.join(comfy_root, 'comfy')})"
        )
    return mod


# ---------------------------------------------------------------------------
# Krea2 detect + load (same as diag_impact.py)
# ---------------------------------------------------------------------------
def _find_krea2_key_prefix(keys):
    for prefix in ("model.diffusion_model.", "diffusion_model.", ""):
        if f"{prefix}txtfusion.projector.weight" in keys:
            return prefix
    raise ValueError("Not a Krea2 checkpoint: missing txtfusion.projector.weight")


def detect_krea2_dit_config(sd, prefix):
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
    if layers <= 0:
        raise ValueError("Krea2 detect failed: no blocks.* keys")
    wq = sd[f"{prefix}blocks.0.attn.wq.weight"]
    wk = sd[f"{prefix}blocks.0.attn.wk.weight"]
    txtlayers = int(sd[f"{prefix}txtfusion.projector.weight"].shape[1])
    txtdim = int(sd[f"{prefix}txtfusion.layerwise_blocks.0.prenorm.scale"].shape[0])
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


def load_krea2(path, device="cuda"):
    """Load Krea2 SingleStreamDiT from a base bf16/fp16 safetensors onto CUDA."""
    if str(device).startswith("cpu"):
        raise RuntimeError("gen_int8_impact trajectory requires CUDA.")
    import comfy.ops
    from comfy.ldm.krea2.model import SingleStreamDiT

    print(f"Loading Krea2 DiT: {path}")
    state_dict = load_file(path)
    prefix = _find_krea2_key_prefix(state_dict)
    cfg = detect_krea2_dit_config(state_dict, prefix)
    print(f"Detected Krea2 DiT config: {cfg}")
    kw = {k: v for k, v in cfg.items() if k != "image_model"}
    dit = SingleStreamDiT(
        **kw, device=device, dtype=torch.bfloat16,
        operations=comfy.ops.manual_cast,
    )
    stripped = {}
    for k, v in state_dict.items():
        if prefix and k.startswith(prefix):
            stripped[k[len(prefix):]] = v
        elif not prefix:
            stripped[k] = v
    missing, unexpected = dit.load_state_dict(stripped, strict=False)
    print(
        f"  [Krea2] load_state_dict missing={len(missing)} "
        f"unexpected={len(unexpected)}"
    )
    dev = str(next(dit.parameters()).device)
    if not dev.startswith("cuda"):
        raise RuntimeError(f"Krea2 DiT landed on {dev!r}, not CUDA")
    print(f"  [Krea2] DiT device={dev}")
    dit.eval()
    del state_dict, stripped
    gc.collect()
    return dit, cfg, prefix


def rel_mse(a, b):
    a = a.float().reshape(a.shape[0], -1)
    b = b.float().reshape(b.shape[0], -1)
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Krea2 INT8 convert ranked by diag_impact trajectory impact "
            "(bf16/fp16 base -> INT8 + FULL ConvRot, v1.5-compatible output)"
        )
    )
    ap.add_argument("base", help="baseline bf16/fp16 Krea2 SingleStreamDiT safetensors")
    ap.add_argument("out", help="output INT8 safetensors path (v1.5-compatible)")
    ap.add_argument("--keep", type=int, default=0,
                    help="keep the top-N highest-impact Linears in original "
                         "dtype (0 = convert all eligible Linears to INT8)")
    ap.add_argument("--steps", type=int, default=4,
                    help="trajectory denoising steps for impact measurement (default 4)")
    ap.add_argument("--lat", type=int, default=128,
                    help="latent H/W (default 128)")
    ap.add_argument("--seq", type=int, default=256,
                    help="random context token seq length (default 256)")
    ap.add_argument("--seed", type=int, default=42,
                    help="trajectory seed (default 42)")
    ap.add_argument("--comfy-path", default="ComfyUI-master",
                    help="ComfyUI root path (default: ComfyUI-master)")
    ap.add_argument("--no-convrot", action="store_true",
                    help="disable FULL ConvRot (plain tensorwise INT8)")
    return ap.parse_args()


def main():
    a = parse_args()
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("gen_int8_impact requires CUDA.")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    enable_convrot = not a.no_convrot
    keep_n = max(0, int(a.keep))

    comfy_root = _ensure_comfyui(a.comfy_path)
    print(f"[Krea2] ComfyUI root: {comfy_root}")
    saved = _clear_argv_for_comfy()
    try:
        _install_comfy_stubs()
        _load_comfy_pkg(comfy_root)
        try:
            import comfy.options
            comfy.options.enable_args_parsing(False)
        except ImportError:
            pass
        import comfy.ops
        from comfy.ldm.krea2.model import SingleStreamDiT

        # 1) load the bf16/fp16 BASE (unquantized)
        state_dict = load_file(a.base)
        prefix = _find_krea2_key_prefix(state_dict)
        cfg = detect_krea2_dit_config(state_dict, prefix)
        print(f"Detected Krea2 DiT config: {cfg}")
        kw = {k: v for k, v in cfg.items() if k != "image_model"}
        dit = SingleStreamDiT(
            **kw, device=device, dtype=torch.bfloat16,
            operations=comfy.ops.manual_cast,
        )
        stripped = {}
        for k, v in state_dict.items():
            if prefix and k.startswith(prefix):
                stripped[k[len(prefix):]] = v
            elif not prefix:
                stripped[k] = v
        missing, unexpected = dit.load_state_dict(stripped, strict=False)
        print(
            f"  [Krea2] load_state_dict missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )
        dev = str(next(dit.parameters()).device)
        if not dev.startswith("cuda"):
            raise RuntimeError(f"Krea2 DiT landed on {dev!r}, not CUDA")
        dit.eval()
        del state_dict, stripped
        gc.collect()

        txtlayers = int(cfg["txtlayers"])
        txtdim = int(cfg["txtdim"])
        channels = int(cfg["channels"])
        steps = int(a.steps)
        seed = int(a.seed)

        gen_ctx = torch.Generator(device=device).manual_seed(seed)
        context = torch.randn(
            1, int(a.seq), txtlayers * txtdim, device=device, dtype=torch.bfloat16,
            generator=gen_ctx,
        )
        sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device)

        def run():
            x = torch.randn(1, channels, int(a.lat), int(a.lat), device=device,
                            dtype=torch.bfloat16,
                            generator=torch.Generator(device).manual_seed(seed))
            with torch.no_grad():
                for step in range(steps):
                    out = dit(x, sigmas[step:step + 1], context)
                    if isinstance(out, tuple):
                        out = out[0]
                    x = (x + (sigmas[step + 1] - sigmas[step]) * out).to(torch.bfloat16)
            return x

        # 2) pristine FP16 trajectory (FULL fp16 model = comparison target)
        print("[*] pristine run", flush=True)
        x_ref = run()
        print("[*] pristine done", flush=True)

        # 3) per-Layer INT8 impact (quantize->dequant injection, rel-MSE)
        eligible: dict[str, torch.nn.Module] = {}
        for n, m in dit.named_modules():
            if not hasattr(m, "weight") or not hasattr(m, "in_features"):
                continue
            if m.weight is None or m.weight.data.ndim != 2:
                continue
            if m.weight.data.dtype not in (torch.float16, torch.bfloat16):
                continue
            full_key = f"{prefix}{n}.weight" if prefix else f"{n}.weight"
            if _is_blacklisted(full_key) or _is_non_diffusion_key(full_key):
                continue
            eligible[n] = m
        print(f"eligible Linears for INT8 impact measurement: {len(eligible)}", flush=True)

        impacts: dict[str, float] = {}
        done = 0
        for n, m in eligible.items():
            w0 = m.weight.data.clone()
            try:
                m.weight.data.copy_(int8_quant_dequant(w0).to(w0.dtype))
                x_t = run()
                imp = rel_mse(x_t, x_ref)
            except Exception as e:
                print(f"  ERR {n}: {e}", flush=True)
                imp = float("nan")
            m.weight.data.copy_(w0)
            impacts[n] = imp
            done += 1
            if done % 25 == 0 or done == len(eligible):
                print(f"  [{done}/{len(eligible)}]", flush=True)

        finite = [v for v in impacts.values() if math.isfinite(v)]
        print(
            f"impact measured: {len(finite)}/{len(impacts)} layers  "
            f"max={max(finite) if finite else float('nan'):.3e}  "
            f"mean={sum(finite) / len(finite) if finite else float('nan'):.3e}",
            flush=True,
        )
    finally:
        _restore_argv(saved)

    # 4) ranking: highest impact stays original dtype
    ranked = sorted(
        ((k, v) for k, v in impacts.items() if math.isfinite(v)),
        key=lambda kv: kv[1], reverse=True,
    )
    keep_set = {k for k, _ in ranked[:keep_n]}
    if keep_n > 0:
        print(f"[keep] top {keep_n} highest-impact Linears stay original dtype:")
        for k, v in ranked[:keep_n]:
            print(f"  KEEP {k}  impact={v:.3e}")

    # 5) convert the full state_dict (v1.5 conventions)
    print("Converting to INT8 (FULL ConvRot Linear + Conv2d channelwise)...")
    new_state_dict = {}
    quant_meta_layers = {}
    converted_count = 0
    convrot_linear = 0
    convrot_conv2d = 0
    plain_int8_count = 0
    bf16_keep = 0
    keep_reverted = 0

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
            continue

        module_key = key[: -len(".weight")]
        stripped_key = module_key[len(prefix):] if prefix else module_key

        # 2D Linear that measured high impact -> original dtype
        if tensor.ndim == 2 and stripped_key in keep_set:
            new_state_dict[key] = tensor
            keep_reverted += 1
            continue

        w_fp = tensor.float()
        quant_config: dict

        if tensor.ndim == 2:
            used_gs = (
                convrot_group_size_for_features(int(w_fp.shape[1]))
                if enable_convrot else None
            )
            if used_gs is not None:
                h_matrix = build_hadamard(used_gs, device="cpu", dtype=torch.float32)
                w_rot = rotate_weight(w_fp, h_matrix, used_gs)
                q, scale = pack_channelwise(w_rot)
                quant_config = {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": int(used_gs),
                }
                convrot_linear += 1
            else:
                q, scale = (
                    pack_channelwise(w_fp)
                )
                quant_config = {"format": "int8_tensorwise"}
                plain_int8_count += 1
        else:  # Conv2d (4D)
            used_gs = (
                convrot_group_size_for_features(int(w_fp.shape[1]))
                if enable_convrot else None
            )
            if used_gs is not None:
                h_matrix = build_hadamard(used_gs, device="cpu", dtype=torch.float32)
                w_rot = rotate_weight_conv2d(w_fp, h_matrix, used_gs)
                q, scale = pack_channelwise(w_rot)
                quant_config = {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": int(used_gs),
                }
                convrot_conv2d += 1
            else:
                q, scale = pack_channelwise(w_fp)
                quant_config = {"format": "int8_tensorwise"}
                plain_int8_count += 1

        new_state_dict[key] = q
        new_state_dict[f"{module_key}.weight_scale"] = scale
        new_state_dict[f"{module_key}.comfy_quant"] = _encode_comfy_quant(quant_config)
        quant_meta_layers[_meta_base_key(module_key)] = dict(quant_config)
        converted_count += 1

    metadata = {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": quant_meta_layers}
        )
    }

    print(f"Saving to: {a.out}")
    print(f"Converted INT8 layers: {converted_count}")
    print(f"  ConvRot Linear: {convrot_linear}, ConvRot Conv2d: {convrot_conv2d}, "
          f"plain INT8: {plain_int8_count}")
    print(f"Kept original dtype (blacklist / non-diffusion / fp32): {bf16_keep}")
    print(f"Kept original dtype (high-impact --keep): {keep_reverted}")
    print(f"IMPACT ranking: pristine-vs-int8 rel-MSE over {len(impacts)} Linears "
          f"(comparison target = FULL fp16 trajectory)")

    save_file(new_state_dict, a.out, metadata=metadata)
    print("Done!")


if __name__ == "__main__":
    main()

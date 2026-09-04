#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Krea2 per-layer trajectory impact for NATIVE convrot INT8 (int8 diag, Step 1).

Same structure as Krea2/diag_impact.py (the NVFP4 step-1 script), but the
injected error is the layer's NATIVE convrot INT8 weight taken verbatim from
the native INT8 artifact (test2) - dequantized and inverse-rotated back into
the base domain so it can run inside the unrotated FULL model.

  1. Krea2/diag_impact_int8.py  -> impact json (FULL vs native-INT8 per layer)
  2. Krea2/gen_reverse_int8.py  -> selected INT8 artifact (test3):
     the --keep N highest-impact Linears are restored to original bf16;
     every other layer keeps its native INT8 weights.

Inputs:
  1. FULL baseline bf16/fp16 safetensors (test)   - pristine reference
  2. NATIVE convrot INT8 safetensors (test2)      - per-layer native weights
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import types

import torch
from safetensors import safe_open
from safetensors.torch import load_file

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ---------------------------------------------------------------------------
# ComfyUI bootstrap (same as diag_impact.py)
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
        m.__file__ = "<stub>"
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
    import importlib
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


def load_krea2(path, device="cuda", comfy_path=None):
    """Load Krea2 SingleStreamDiT from a base fp16/bf16 safetensors.

    Returns (model, config_dict, key_prefix). Module names are STRIPPED
    (no model.diffusion_model. prefix), so they match the INT8 artifact's
    _quantization_metadata.layers keys directly.
    """
    if str(device).startswith("cpu"):
        raise RuntimeError("diag_impact_int8 Krea2 trajectory requires CUDA.")
    comfy_root = _ensure_comfyui(comfy_path)
    print(f"[Krea2] ComfyUI root: {comfy_root}")
    saved = _clear_argv_for_comfy()
    try:
        _install_comfy_stubs()
        _load_comfy_pkg(comfy_root)
        try:
            import comfy.options
            comfy.options.enable_args_parsing(False)
        except ImportError:
            # older ComfyUI without comfy.options; argv already cleared
            pass
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
    finally:
        _restore_argv(saved)


def rel_mse(a, b):
    a = a.float().reshape(a.shape[0], -1)
    b = b.float().reshape(b.shape[0], -1)
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


# ---------------------------------------------------------------------------
# NATIVE INT8 weight access (test2)
# ---------------------------------------------------------------------------
def load_native_layers(native_path: str):
    """Return (state_dict, meta_layers, native_prefix) from the native artifact."""
    sd = load_file(native_path)
    with safe_open(native_path, framework="pt") as fh:
        meta_raw = (fh.metadata() or {}).get("_quantization_metadata", "{}")
    meta_layers = json.loads(meta_raw).get("layers", {})
    prefix = _find_krea2_key_prefix(sd)
    return sd, meta_layers, prefix


def native_int8_domain_weight(q: torch.Tensor, scale, gs: int) -> torch.Tensor:
    """Dequantize the native INT8 weight and inverse-rotate it back into the
    base (unrotated) domain. The regular Hadamard is symmetric + orthogonal,
    so W_rot @ H = W @ H^T @ H = W (same call as the forward rotation)."""
    w_rot_dq = q.float() * scale.reshape(-1, 1)
    if gs and gs >= 4:
        from math import log as _log
        if (gs & (gs - 1)) != 0 or _log(gs, 4) % 1 != 0:
            raise ValueError(f"invalid convrot groupsize {gs}")
        h4 = torch.tensor(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            dtype=torch.float32, device=w_rot_dq.device,
        )
        h = h4
        cur = 4
        while cur < gs:
            h = torch.kron(h, h4)
            cur *= 4
        h = h / (gs ** 0.5)
        out_features, in_features = w_rot_dq.shape
        group_count = in_features // gs
        grouped = w_rot_dq.view(out_features, group_count, gs)
        w_dq = torch.matmul(
            grouped, h.T.to(dtype=w_rot_dq.dtype, device=w_rot_dq.device)
        ).reshape(w_rot_dq.shape)
        return w_dq
    return w_rot_dq


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Krea2 per-layer trajectory impact for NATIVE convrot INT8 "
            "(FULL vs native-INT8 per layer; writes impact json)"
        )
    )
    ap.add_argument("base", help="baseline fp16/bf16 Krea2 SingleStreamDiT safetensors (test)")
    ap.add_argument("native", help="NATIVE convrot INT8 safetensors (test2)")
    ap.add_argument("out", help="output impact json path")
    ap.add_argument("--comfy-path", required=True,
                    help="ComfyUI root (repo ComfyUI-master)")
    ap.add_argument("--steps", type=int, default=4,
                    help="trajectory denoising steps (default 4)")
    ap.add_argument("--lat", type=int, default=128,
                    help="latent H/W (default 128)")
    ap.add_argument("--seq", type=int, default=256,
                    help="random context token seq length (default 256)")
    ap.add_argument("--seed", type=int, default=42,
                    help="trajectory seed (default 42)")
    return ap.parse_args()


def main():
    a = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("diag_impact_int8 requires CUDA.")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    a.base = a.base.strip()
    a.native = a.native.strip()
    a.out = a.out.strip()

    # NATIVE convrot INT8 artifact (per-layer native weights)
    native_sd, meta_layers, native_prefix = load_native_layers(a.native)
    print(f"native convrot INT8 artifact: {a.native}")
    print(f"native quantized layers in metadata: {len(meta_layers)}")

    model, cfg, _prefix = load_krea2(a.base, device=device, comfy_path=a.comfy_path)
    model.eval()

    # Linear layers only (native INT8 conversion targets 2D Linears).
    mods = {}
    for n, m in model.named_modules():
        if hasattr(m, "weight") and hasattr(m, "in_features"):
            mods[n] = m
    print(f"modules with weight/in_features: {len(mods)}", flush=True)

    layers = []
    for n in meta_layers.keys():
        if n in mods and native_sd.get(f"{native_prefix}{n}.weight") is not None:
            layers.append(n)
    print(f"layers measurable against the FULL model: {len(layers)}", flush=True)

    txtlayers = int(cfg["txtlayers"])
    txtdim = int(cfg["txtdim"])
    channels = int(cfg["channels"])
    lat = int(a.lat)
    seq = int(a.seq)
    steps = int(a.steps)
    seed = int(a.seed)

    gen_ctx = torch.Generator(device=device).manual_seed(seed)
    context = torch.randn(
        1, seq, txtlayers * txtdim, device=device, dtype=torch.bfloat16,
        generator=gen_ctx,
    )
    t_steps = torch.linspace(1.0, 0.0, steps + 1, device=device)

    def run():
        g = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(1, channels, lat, lat, device=device,
                        dtype=torch.bfloat16, generator=g)
        with torch.no_grad():
            for step in range(steps):
                t = t_steps[step:step + 1]
                out = model(x, t, context)
                if isinstance(out, tuple):
                    out = out[0]
                x = (x + (t_steps[step + 1] - t_steps[step]) * out).to(torch.bfloat16)
        return x

    print("[*] pristine FULL run", flush=True)
    x_ref = run()
    print("[*] pristine done", flush=True)

    impacts = {}
    done = 0
    for n in layers:
        m = mods[n]
        w0 = m.weight.data.clone()
        try:
            q = native_sd[f"{native_prefix}{n}.weight"]
            sc = native_sd[f"{native_prefix}{n}.weight_scale"]
            entry = meta_layers.get(n, {})
            gs = int(entry.get("convrot_groupsize", 0) or 0)
            w_dq = native_int8_domain_weight(q, sc, gs)
            m.weight.data.copy_(w_dq.to(w0.dtype))
            x_t = run()
            imp = rel_mse(x_t, x_ref)
        except Exception as e:
            print(f"  ERR {n}: {e}", flush=True)
            imp = float("nan")
        m.weight.data.copy_(w0)
        impacts[n] = imp
        done += 1
        if done % 25 == 0 or done == len(layers):
            print(f"  [{done}/{len(layers)}]", flush=True)

    xr = x_ref.float().reshape(x_ref.shape[0], -1)
    with open(a.out, "w", encoding="utf-8") as fo:
        json.dump(
            {"x_ref_norm": float((xr * xr).sum().item()), "impacts": impacts},
            fo, indent=1,
        )
    print(f"saved {a.out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

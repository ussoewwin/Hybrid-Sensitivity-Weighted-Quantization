# -*- coding: utf-8 -*-
"""Krea2 per-layer trajectory impact (reverse hybrid NVFP4 method, Step 1).

Krea2 port of Z Image diag_impact.py. Inject ONE layer's NVFP4 error into the
base SingleStreamDiT (fp16/bf16), run a fixed-seed denoising trajectory, and
measure how far the final x drifts (relative MSE). That value is the layer's
true importance under real trajectory propagation (ascending = safest first).

Reverse hybrid NVFP4 method (see md/How to quantize Z Image - Hybrid NVFP4.md):
  1. Krea2/diag_impact.py        -> impact_<model>.json
  2. Krea2/gen_reverse_nvfp4.py  -> hybrid nv{K} artifact (INT8 -> K layers NVFP4)
  3. benchmark/krea2_convrot_nvfp4_bench.py -> SSIM check

This is the REVERSE (trajectory) method, NOT the 4-axis static ranking
(auto_int8_nvfp4_hybrid.py). The two must not be mixed.

Usage:
    python Krea2/diag_impact.py <base_model.safetensors> \
        <convrot_int8.safetensors> <impact_out.json> \
        [--comfy-path <comfyui-root>] [--steps N] [--lat H] [--seq S] [--seed S]

The impact JSON uses STRIPPED layer keys (same as the INT8 artifact's
_quantization_metadata.layers), e.g. "blocks.0.attn.gate".
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
# ComfyUI bootstrap (torchaudio / comfy_aimdo / psutil stubs; same pattern as
# the existing Krea2 INT8 / NVFP4 converters).
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
        def to(self, *a, **k):
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
    candidates = []
    if comfy_path:
        candidates.append(os.path.abspath(comfy_path))
    env = os.environ.get("COMFYUI_PATH")
    if env:
        candidates.append(env)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.extend([
        r"D:\USERFILES\ComfyUI\ComfyUI",
        r"D:\USERFILES\GitHub\ComfyUI",
        os.path.join(here, "..", "ComfyUI-master"),
    ])
    for root in candidates:
        if not root:
            continue
        if os.path.isfile(os.path.join(root, "comfy", "ldm", "krea2", "model.py")):
            if root not in sys.path:
                sys.path.insert(0, root)
            return root
    raise FileNotFoundError(
        "ComfyUI root with comfy/ldm/krea2/model.py not found. "
        "Pass --comfy-path or set COMFYUI_PATH."
    )


# ---------------------------------------------------------------------------
# Krea2 detect + load (same math as hswq_convrot_int8_krea2_v1.5.py)
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
        raise RuntimeError("diag_impact Krea2 trajectory requires CUDA.")
    _ensure_comfyui(comfy_path)
    saved = _clear_argv_for_comfy()
    try:
        try:
            import comfy.options
            comfy.options.enable_args_parsing(False)
        except ImportError:
            # older ComfyUI without comfy.options; argv already cleared
            pass
        _install_comfy_stubs()
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


# ---------------------------------------------------------------------------
# NVFP4 error injection + relative MSE (identical to Z Image diag_impact.py)
# ---------------------------------------------------------------------------
def nvfp4_quant_error(w, group=256):
    """per-group e4m3 quant-dequant reconstruction (i.e. NVFP4-quantized weights)."""
    wf = w.float()
    orig = wf.reshape(wf.shape[0], -1)
    k = orig.shape[1]
    n_groups = (k + group - 1) // group
    pad = n_groups * group - k
    if pad:
        orig = torch.nn.functional.pad(orig, (0, pad))
    g = orig.reshape(orig.shape[0], n_groups, group)
    amax = g.abs().amax(dim=2, keepdim=True).clamp_min(1e-12)
    scale = amax / 448.0
    q = (g / scale).to(torch.float8_e4m3fn).float()
    dq = q * scale
    if pad:
        dq = dq.reshape(orig.shape[0], n_groups, group)[:, :, :k]
    return dq.reshape(w.shape).to(w.dtype)


def rel_mse(a, b):
    a = a.float().reshape(a.shape[0], -1)
    b = b.float().reshape(b.shape[0], -1)
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


def parse_args():
    ap = argparse.ArgumentParser(
        description="Krea2 per-layer NVFP4 trajectory impact (reverse method, Step 1)"
    )
    ap.add_argument("base", help="baseline fp16/bf16 Krea2 SingleStreamDiT safetensors")
    ap.add_argument("artifact", help="complete ConvRot INT8 safetensors (layer list source)")
    ap.add_argument("out", help="output impact json path")
    ap.add_argument("--comfy-path", default=None,
                    help="ComfyUI root (must contain comfy/ldm/krea2/model.py)")
    ap.add_argument("--steps", type=int, default=4,
                    help="trajectory denoising steps (default 4)")
    ap.add_argument("--lat", type=int, default=32,
                    help="latent H/W (default 32)")
    ap.add_argument("--seq", type=int, default=256,
                    help="random context token seq length (default 256)")
    ap.add_argument("--seed", type=int, default=42,
                    help="trajectory seed (default 42)")
    return ap.parse_args()


def main():
    a = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Krea2 diag_impact requires CUDA.")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    a.base = a.base.strip()
    a.artifact = a.artifact.strip()
    a.out = a.out.strip()

    model, cfg, _prefix = load_krea2(a.base, device=device, comfy_path=a.comfy_path)
    model.eval()

    # Linear layers only (NVFP4 is 2D-only; Conv2d / norm / bias are blacklisted).
    mods = {}
    for n, m in model.named_modules():
        if hasattr(m, "weight") and hasattr(m, "in_features"):
            mods[n] = m
    print(f"modules with weight/in_features: {len(mods)}", flush=True)

    with safe_open(a.artifact, framework="pt", device="cpu") as f:
        meta = json.loads(f.metadata()["_quantization_metadata"])
        layers = list(meta["layers"].keys())
    print(f"layers to measure: {len(layers)}", flush=True)

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

    print("[*] pristine run", flush=True)
    x_ref = run()
    print("[*] pristine done", flush=True)

    impacts = {}
    done = 0
    for n in layers:
        if n not in mods:
            print(f"  SKIP (not a module): {n}", flush=True)
            continue
        m = mods[n]
        w0 = m.weight.data.clone()
        m.weight.data.copy_(nvfp4_quant_error(w0))
        try:
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

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

Usage:
    python Krea2/diag_impact.py <base_model.safetensors> \
        <convrot_int8.safetensors> <impact_out.json> \
        --comfy-path <comfyui-root> [--steps N] [--lat H] [--seq S] [--seed S]

The impact JSON uses STRIPPED layer keys (same as the INT8 artifact's
_quantization_metadata.layers), e.g. "blocks.0.attn.gate".
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import types

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def nvfp4_quant_error(w):
    """TRUE NVFP4 quantization error via comfy_kitchen roundtrip
    (E2M1 x 16-element blocks + global scale): exactly the kernel that
    produces the shipped artifact. The old e4m3-per-256 proxy understates
    the error ~13x and flattens the ranking; do not fall back to it."""
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout as _NVFP4
    w2 = w if w.is_contiguous() else w.contiguous()
    qdata, params = _NVFP4.quantize(w2)
    return _NVFP4.dequantize(qdata, params)


def rel_mse(a, b):
    a = a.float().reshape(a.shape[0], -1)
    b = b.float().reshape(b.shape[0], -1)
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


# ---------------------------------------------------------------------------
# Krea2 detect + load
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
    """Load Krea2 SingleStreamDiT from a base fp16/bf16 safetensors.

    Assumes bench.setup_comfy() has ALREADY been executed (same as Z_Image pattern).
    """
    if str(device).startswith("cpu"):
        raise RuntimeError("diag_impact Krea2 trajectory requires CUDA.")
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


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Krea2 per-layer NVFP4 trajectory impact")
    ap.add_argument("base", help="baseline fp16/bf16 Krea2 SingleStreamDiT safetensors")
    ap.add_argument("artifact", help="sci_1off complete ConvRot INT8 safetensors")
    ap.add_argument("out", help="output impact json path")
    ap.add_argument("--comfy-path", default="ComfyUI-master",
                    help="ComfyUI root path (default: ComfyUI-master)")
    ap.add_argument("--repo-root", default=None,
                    help="repo root containing benchmark/ (default: parent of this dir)")
    ap.add_argument("--steps", type=int, default=4,
                    help="trajectory denoising steps (default 4)")
    ap.add_argument("--lat", type=int, default=128,
                    help="latent H/W (default 128, matches 1024x1024 token count)")
    ap.add_argument("--seq", type=int, default=256,
                    help="random context token seq length (default 256)")
    ap.add_argument("--seed", type=int, default=42,
                    help="trajectory seed (default 42)")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def main():
    a = parse_args()
    device = "cuda"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    repo = os.path.abspath(a.repo_root) if a.repo_root else os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    sys.path.insert(0, os.path.join(repo, "benchmark"))
    sys.path.insert(0, repo)
    spec = importlib.util.spec_from_file_location(
        "bench", os.path.join(repo, "benchmark", "krea2_convrot_nvfp4_bench.py")
    )
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    comfy_path = a.comfy_path
    if not os.path.isabs(comfy_path):
        joined = os.path.join(repo, comfy_path)
        if os.path.isdir(joined):
            comfy_path = os.path.abspath(joined)
        else:
            comfy_path = os.path.abspath(comfy_path)
    else:
        comfy_path = os.path.abspath(comfy_path)
    bench.setup_comfy(comfy_path)

    a.base = a.base.strip()
    a.artifact = a.artifact.strip()
    a.out = a.out.strip()

    model, cfg, _prefix = load_krea2(a.base, device=device)
    model.eval()
    mods = {}
    for n, m in model.named_modules():
        if hasattr(m, "weight") and hasattr(m, "in_features"):
            mods[n] = m
    print(f"modules with weight/in_features: {len(mods)}", flush=True)

    artifact_path = a.artifact
    if not os.path.isfile(artifact_path):
        cand = os.path.join(repo, a.artifact)
        if os.path.isfile(cand):
            artifact_path = os.path.abspath(cand)
        else:
            cand_base = os.path.join(repo, os.path.basename(a.artifact))
            if os.path.isfile(cand_base):
                artifact_path = os.path.abspath(cand_base)

    if not os.path.isfile(artifact_path):
        raise FileNotFoundError(f"Artifact not found: {a.artifact}")

    with safe_open(artifact_path, framework="pt", device="cpu") as f:
        meta = json.loads(f.metadata()["_quantization_metadata"])
        layers = list(meta["layers"].keys())
    print(f"layers in INT8 metadata: {len(layers)}", flush=True)

    _SAFE_IN_FEATURES = {1536, 6144, 16384}

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
                out = model(x, sigmas[step:step + 1], context)
                if isinstance(out, tuple):
                    out = out[0]
                x = (x + (sigmas[step + 1] - sigmas[step]) * out).to(torch.bfloat16)
        return x

    print("[*] pristine run", flush=True)
    x_ref = run()
    print("[*] pristine done", flush=True)

    impacts = {}
    done = 0
    for n in layers:
        nmod = n[len("model.diffusion_model."):] if n.startswith("model.diffusion_model.") else n
        if nmod not in mods:
            print(f"  SKIP (not a module): {n}", flush=True)
            continue
        m = mods[nmod]
        if m.in_features not in _SAFE_IN_FEATURES:
            impacts[n] = float("nan")
            continue
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
    json.dump({"x_ref_norm": float((xr * xr).sum().item()), "impacts": impacts},
              open(a.out, "w"), indent=1)
    print(f"saved {a.out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

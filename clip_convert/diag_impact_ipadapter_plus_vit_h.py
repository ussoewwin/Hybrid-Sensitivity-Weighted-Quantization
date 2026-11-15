# -*- coding: utf-8 -*-
"""Impact sweep for ip-adapter-plus_sdxl_vit-h.safetensors (SDXL IP-Adapter Plus, ViT-H).

The sibling sweep for the basic SDXL adapter (`clip_convert/diag_impact_ipadapter_xl.py`)
measures a single `image_proj.proj` Linear. This checkpoint has no such Linear: its
`image_proj` is a `Resampler` of depth 4 - the model ComfyUI's `comfyui_ipadapter_plus`
builds in `init_proj_plus`. So the image_proj observable here is the Resampler's own
output (the normalised latent stack the cross-attention consumes), and the `Resampler`
class is imported from the node itself, so the measured path is the runtime path and not
a re-implementation of it.

The runtime parameters are not assumed; they are read off the checkpoint and checked
against the node's constructor:

    latents            (1, 16, 1280)   -> num_queries = 16, dim = 1280
    proj_in            (1280, 1280)    -> embedding_dim = 1280
    proj_out           (2048, 1280)    -> output_dim = 2048
    layers.*.0.to_q    (1280, 1280)    -> dim_head * heads = 1280
    layers.*.0.to_kv   (2560, 1280)    -> 2 * dim_head * heads
    layers.*.1.1       (5120, 1280)    -> ff_mult = 4

One layer at a time is replaced by its ConvRot INT8 reconstruction: Hadamard-256
rotation on the inner dimension, per-output-channel `amax/127` scale, and the error is
measured in the **original** frame after rotating the activation back.

Measured in float32: the file is float16, and the quantisation error has to be resolved
against the weights themselves, not against float16 rounding.

    python clip_convert/diag_impact_ipadapter_plus_vit_h.py \
        --input <src>/ip-adapter-plus_sdxl_vit-h.safetensors \
        --out impact/impact_ipadapter_plus_vit_h.json --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def build_hadamard(size, dtype=torch.float32, device="cpu"):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
                      dtype=dtype, device=device)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def convrot_int8(w, gs=256):
    """Rotate the inner dimension, quantise per output row, rotate back."""
    out_f, in_f = w.shape
    if in_f % gs != 0:
        return None
    h = build_hadamard(gs, dtype=torch.float32, device=w.device)
    groups = w.reshape(out_f, in_f // gs, gs)
    rotated = torch.matmul(groups, h)
    amax = rotated.abs().amax(dim=2, keepdim=True).clamp_min(1e-8)
    scale = amax / 127.0
    q = torch.round(rotated / scale).clamp(-127, 127)
    back = torch.matmul(q * scale, h.transpose(-1, -2))
    return back.reshape(out_f, in_f)


def rel_mse(a, b):
    num = ((a - b) ** 2).mean()
    den = (a ** 2).mean().clamp_min(1e-12)
    return float((num / den).item())


def load_resampler(node_dir, params):
    """The runtime's own Resampler, from the node that loads this file."""
    if node_dir not in sys.path:
        sys.path.insert(0, node_dir)
    from image_proj_models import Resampler  # noqa: E402  (the node's own module)
    m = Resampler(**params)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--node", default="", help="the comfyui_ipadapter_plus directory")
    ap.add_argument("--tokens", type=int, default=257,
                    help="the CLIP token count the image embeds carry")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    t0 = time.time()

    raw = open(a.input, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    del raw
    sd = load_file(a.input)
    dev = a.device if torch.cuda.is_available() else "cpu"

    proj = {k[len("image_proj."):]: v for k, v in sd.items() if k.startswith("image_proj.")}
    ip = {k[len("ip_adapter."):]: v for k, v in sd.items() if k.startswith("ip_adapter.")}
    if not proj or not ip:
        raise SystemExit("the file does not carry image_proj.* and ip_adapter.* names")

    # the runtime parameters, read off the checkpoint
    latents = proj["latents"]
    num_queries, dim = latents.shape[1], latents.shape[2]
    embedding_dim = proj["proj_in.weight"].shape[1]
    output_dim = proj["proj_out.weight"].shape[0]
    inner_dim = proj["layers.0.0.to_q.weight"].shape[0]
    depth = sum(1 for k in proj if k.startswith("layers.") and k.endswith(".0.to_q.weight"))
    ff_mult = proj["layers.0.1.1.weight"].shape[0] // dim
    heads = inner_dim // 64
    dim_head = 64

    params = dict(dim=dim, depth=depth, dim_head=dim_head, heads=heads,
                  num_queries=num_queries, embedding_dim=embedding_dim,
                  output_dim=output_dim, ff_mult=ff_mult)

    if not a.node:
        raise SystemExit("--node must point at the comfyui_ipadapter_plus directory")
    resampler = load_resampler(a.node, params).to(dev).float().eval()
    missing, unexpected = resampler.load_state_dict(
        {k: v.float() for k, v in proj.items()}, strict=False)
    missing = [k for k in missing if not k.endswith(".latents")]
    if missing or unexpected:
        raise SystemExit("the Resampler did not take the checkpoint: missing %s unexpected %s"
                         % (missing[:6], unexpected[:6]))

    # a fixed CLIP-ViT-H image embedding, the adapter's real input width
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    emb = torch.randn(1, a.tokens, embedding_dim, generator=g).to(dev)

    with torch.no_grad():
        base_stack = resampler(emb)

    impacts = {}
    not_quantisable = {}

    # ---- image_proj: every 2-D weight of the Resampler, one at a time
    proj_weights = sorted(k for k, v in proj.items()
                          if k.endswith(".weight") and v.dim() == 2)
    for name in proj_weights:
        w = proj[name].float().to(dev)
        module = name[: -len(".weight")]
        q = convrot_int8(w)
        if q is None:
            not_quantisable["image_proj." + module] = (
                "inner dimension %d not a multiple of 256" % w.shape[1])
            continue
        orig = resampler.state_dict()[name].clone()
        with torch.no_grad():
            resampler.state_dict()[name].copy_(q)
            stack = resampler(emb)
            resampler.state_dict()[name].copy_(orig)
        impacts["image_proj." + module] = rel_mse(base_stack, stack)
        print("[image_proj.%s] %s -> drift %.9e" % (module, tuple(w.shape), impacts["image_proj." + module]),
              flush=True)

    # ---- ip_adapter layers
    seen = 0
    for key in sorted(ip.keys()):
        if not key.endswith(".weight") or ip[key].dim() != 2:
            continue
        w = ip[key].float()
        name = "ip_adapter." + key[: -len(".weight")]
        out_f, in_f = w.shape
        if in_f % 256 != 0:
            not_quantisable[name] = "inner dimension %d not a multiple of 256" % in_f
            seen += 1
            continue
        x = torch.randn(1, in_f, generator=g).to(dev)
        wd = w.to(dev)
        ref = F.linear(x, wd, None)
        q = convrot_int8(wd)
        impacts[name] = rel_mse(ref, F.linear(x, q, None))
        seen += 1
        if seen % 20 == 0:
            print("  ... %d layers" % seen, flush=True)

    vals = sorted(impacts.values(), reverse=True)
    n = len(vals)
    summary = {
        "tool": "clip_convert/diag_impact_ipadapter_plus_vit_h.py",
        "family": "ipadapter_plus_vit_h",
        "model": os.path.basename(a.input),
        "source": os.path.basename(a.input),
        "source_bytes": os.path.getsize(a.input),
        "source_sha256": sha,
        "tensor_count": len(sd),
        "image_proj_tensors": len(proj),
        "ip_adapter_tensors": len(ip),
        "resampler_params": params,
        "resampler_source": os.path.join(a.node, "image_proj_models.py"),
        "input_tokens": a.tokens,
        "weight_groups": n,
        "measured": n,
        "not_quantisable_count": len(not_quantisable),
        "not_quantisable": not_quantisable,
        "max": vals[0] if vals else None,
        "max_layer": max(impacts, key=impacts.get) if impacts else None,
        "median": vals[n // 2] if vals else None,
        "p90": vals[max(0, int(n * 0.1) - 1)] if vals else None,
        "min": vals[-1] if vals else None,
        "above_threshold": sum(1 for v in vals if v > 0.15),
        "threshold": 0.15,
        "seed": a.seed,
        "dtype": "float32",
        "device": dev,
        "seconds": round(time.time() - t0, 1),
        "impacts": impacts,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print("[summary] groups %d | max %.9e (%s) | median %.9e | p90 %.9e | above 0.15 %d | %.1f s"
          % (n, summary["max"], summary["max_layer"], summary["median"], summary["p90"],
             summary["above_threshold"], summary["seconds"]), flush=True)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()

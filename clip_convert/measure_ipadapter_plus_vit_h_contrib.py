# -*- coding: utf-8 -*-
"""Aggregate contribution for ip-adapter-plus_sdxl_vit-h (SDXL IP-Adapter Plus, ViT-H).

The criterion is the aggregate contribution, not a weight statistic
(`HSWQ_High_Quality_Packs_ConvRot_INT8.md`, Rule 2): replace **one** layer by its ConvRot INT8
reconstruction and measure the change in the stage output that layer feeds.

On this graph the "restore one layer to float" form of that measurement does not rank anything:
every Resampler layer is inside a residual attention block, and restoring a single layer to float
while the rest stay INT8 *increases* the worst-latent divergence for all 22 layers (measured). The
single-layer restoration is not a small perturbation of this graph - it changes the relative
precision of the block it sits in.

So the measurement is made from the other end: the **all-float source** is the reference, and
**one** layer at a time is replaced by its INT8 reconstruction. The contribution of that layer is
how much distance to the source it adds on its own. That is additive-free: it says how much of the
pack's error that layer is responsible for, which is what the ranking has to express.

Two surfaces, because the adapter has two halves:

  * `image_proj` - the Resampler's output, scored per latent (16 latents of 2048) and aggregated by
    the worst latent, because a bunch of latents that all look fine does not cover the one that
    does not.
  * `ip_adapter` - each `to_k_ip` / `to_v_ip` projection's own output on a fixed input of its own
    width, scored per projection.

    python clip_convert/measure_ipadapter_plus_vit_h_contrib.py \
        --source <src>.safetensors --out impact/contrib_ipadapter_plus_vit_h.json --seed 42 \
        --node <ComfyUI>/custom_nodes/comfyui_ipadapter_plus
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

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from convert_ipadapter_convrot_int8 import build_hadamard, rotate_weight  # noqa: E402

GROUP_SIZE = 256


def quantise(w, gs=GROUP_SIZE):
    """Readable form of the same arithmetic."""
    h = build_hadamard(gs, dtype=torch.float32, device=w.device)
    of, inf = w.shape
    rot = torch.matmul(w.reshape(of, inf // gs, gs), h.T)
    amax = rot.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    scale = amax / 127.0
    q = torch.round(rot / scale).clamp(-127, 127)
    return torch.matmul(q * scale, h).reshape(of, inf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--node", required=True)
    ap.add_argument("--tokens", type=int, default=257)
    ap.add_argument("--layers", default="",
                    help="optional JSON file listing the float-kept set to evaluate")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    t0 = time.time()
    dev = a.device if torch.cuda.is_available() else "cpu"

    src = load_file(a.source)
    if a.node not in sys.path:
        sys.path.insert(0, a.node)
    from image_proj_models import Resampler  # noqa: E402  (the node's own module)

    proj = {k[len("image_proj."):]: v for k, v in src.items() if k.startswith("image_proj.")}
    lat = proj["latents"]
    params = dict(
        dim=lat.shape[2], depth=sum(1 for k in proj if k.endswith(".0.to_q.weight")),
        dim_head=64, heads=proj["layers.0.0.to_q.weight"].shape[0] // 64,
        num_queries=lat.shape[1], embedding_dim=proj["proj_in.weight"].shape[1],
        output_dim=proj["proj_out.weight"].shape[0], ff_mult=proj["layers.0.1.1.weight"].shape[0] // lat.shape[2])

    float_state = {k: v.float().to(dev) for k, v in proj.items()}

    def build_resampler(state_dict):
        m = Resampler(**params).to(dev).float().eval()
        m.load_state_dict({k: v.clone() for k, v in state_dict.items()}, strict=True)
        return m

    g = torch.Generator(device="cpu").manual_seed(a.seed)
    emb = torch.randn(1, a.tokens, params["embedding_dim"], generator=g).to(dev)

    with torch.inference_mode():
        source_model = build_resampler(float_state)
        source_stack = source_model(emb)[0].float()          # (num_queries, output_dim)

    def latent_cosines(stack):
        return [float(F.cosine_similarity(source_stack[i].reshape(1, -1).double(),
                                          stack[0, i].float().reshape(1, -1).double()).item())
                for i in range(source_stack.shape[0])]

    contributions = {}
    details = {}

    # ---- image_proj: one layer at a time replaced by its INT8 reconstruction
    proj_weights = sorted(k for k, v in proj.items() if k.endswith(".weight") and v.dim() == 2)
    for name in proj_weights:
        w = float_state[name]
        q = quantise(w)
        state = dict(float_state)
        state[name] = q
        with torch.inference_mode():
            model = build_resampler(state)
            stack = model(emb)
        cs = latent_cosines(stack)
        worst = max(1.0 - c for c in cs)
        mean = sum(1.0 - c for c in cs) / len(cs)
        contributions["image_proj." + name[: -len(".weight")]] = worst
        details["image_proj." + name[: -len(".weight")]] = {
            "surface": "image_proj",
            "element": "the worst latent of the Resampler's %dx%d output" % (params["num_queries"], params["output_dim"]),
            "worst_latent_divergence": worst,
            "mean_latent_divergence": mean,
            "worst_cosine": 1.0 - worst,
            "shape": list(w.shape),
        }
        print("[image_proj.%s] worst-latent %.6e  mean %.6e" % (name[: -len(".weight")], worst, mean),
              flush=True)

    # ---- ip_adapter: each projection's own output, one at a time
    ip_ref = {}
    for name in sorted(k for k in src if k.startswith("ip_adapter.") and k.endswith(".weight")):
        base = name[len("ip_adapter."):][: -len(".weight")]
        w = src[name].float().to(dev)
        with torch.inference_mode():
            x = torch.randn(1, w.shape[1], generator=g).to(dev)
            ip_ref[base] = (w, x, F.linear(x, w, None))

    for base, (w, x, ref) in sorted(ip_ref.items()):
        q = quantise(w)
        with torch.inference_mode():
            out = F.linear(x, q, None)
        c = float(F.cosine_similarity(ref.reshape(1, -1).double(), out.reshape(1, -1).double()).item())
        contributions["ip_adapter." + base] = 1.0 - c
        details["ip_adapter." + base] = {
            "surface": "ip_adapter",
            "element": "the projection's own output on a fixed input of its width",
            "divergence": 1.0 - c,
            "cosine": c,
            "shape": list(w.shape),
        }

    vals = sorted(contributions.values(), reverse=True)
    n = len(vals)
    summary = {
        "tool": "clip_convert/measure_ipadapter_plus_vit_h_contrib.py",
        "family": "ipadapter_plus_vit_h",
        "source": os.path.basename(a.source),
        "source_bytes": os.path.getsize(a.source),
        "source_sha256": hashlib.sha256(open(a.source, "rb").read()).hexdigest(),
        "resampler_params": params,
        "resampler_source": os.path.join(a.node, "image_proj_models.py"),
        "criterion": ("replace one layer by its ConvRot INT8 reconstruction, measured against the "
                      "all-float source. The all-INT8 baseline is not the reference: restoring one "
                      "layer to float inside an all-INT8 graph is not a small perturbation on this "
                      "graph, so that form ranks nothing (measured: all 22 layers worse)"),
        "surfaces": {
            "image_proj": "the worst latent of the Resampler's %dx%d output" % (params["num_queries"], params["output_dim"]),
            "ip_adapter": "each to_k_ip / to_v_ip projection's own output",
        },
        "layers_measured": n,
        "contribution_max": vals[0] if vals else None,
        "contribution_min": vals[-1] if vals else None,
        "contribution_median": vals[n // 2] if vals else None,
        "seed": a.seed,
        "device": dev,
        "seconds": round(time.time() - t0, 1),
        "contributions": contributions,
        "details": details,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print("[summary] layers %d | max %.6e | median %.6e | %.1f s"
          % (n, summary["contribution_max"], summary["contribution_median"], summary["seconds"]),
          flush=True)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()

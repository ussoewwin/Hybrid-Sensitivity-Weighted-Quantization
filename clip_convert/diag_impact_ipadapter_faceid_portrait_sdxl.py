# -*- coding: utf-8 -*-
"""SDXL IP-Adapter FaceID Portrait (unnorm) per-layer ConvRot INT8 impact (the drift screen).

The adapter's own network, read from the checkpoint and from the node's own classes:

  * ``image_proj`` is a ``Resampler``: ``proj_in`` (1280, 512) - the raw 512-d ArcFace embedding is
    its input - four blocks of Perceiver attention
    with 20 heads, ``proj_out`` 1280 -> 2048, and a ``latents`` query of shape (1, 16, 1280) - the
    portrait variant keeps 16 query tokens and a 1280-wide inner dim, which is what
    ``IPAdapterPlus.py`` builds when ``portraitunnorm`` is set (cross_attention_dim 1280,
    clip_extra_context_tokens 16, output_cross_attention_dim read from ``1.to_k_ip.weight``).
  * ``ip_adapter`` is 140 bare projections: 70 ``to_k_ip`` and 70 ``to_v_ip`` at 2048 -> 640/1280,
    with no LoRA pairs.

The instrument is the same as the other per-model runs: replace one 2-D weight by its ConvRot INT8
reconstruction, run the stage that weight feeds, and record the relative MSE of that stage's output.

    python clip_convert/diag_impact_ipadapter_faceid_portrait_sdxl.py --ipadapter <f.bin> \\
        --out <impact.json> --comfy <ComfyUI> [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch


def build_hadamard(size, dtype=torch.float32):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def convrot_group_size(n, preferred=256):
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def convrot_int8_recon(w, gs):
    of, inf = w.shape
    h = build_hadamard(gs, dtype=torch.float32)
    rot = torch.matmul(w.float().view(of, inf // gs, gs), h.T).reshape(of, inf)
    amax = rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    scale = amax / 127.0
    q = torch.round(rot / scale).clamp(-127, 127)
    deq = (q * scale).reshape(of, inf // gs, gs)
    return torch.matmul(deq, h).reshape(of, inf)


def rel_mse(a, b):
    return float(((a - b) ** 2).mean() / (a ** 2).mean().clamp(min=1e-30))


def flatten(sd):
    out = {}
    for group in ("image_proj", "ip_adapter"):
        for k, v in sd[group].items():
            if torch.is_tensor(v):
                out["%s.%s" % (group, k)] = v.float()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipadapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    sys.path.insert(0, a.comfy)
    ext = os.path.join(a.comfy, "custom_nodes", "comfyui_ipadapter_plus")
    sys.path.insert(0, ext)
    from image_proj_models import Resampler  # the node's own class

    raw = torch.load(a.ipadapter, map_location="cpu", weights_only=True)
    sd = flatten(raw)

    torch.manual_seed(a.seed)
    # proj_in is (1280, 512): the portrait variant consumes the raw 512-d ArcFace embedding, not a
    # CLIP sequence. The shape is read from the checkpoint, not assumed.
    model = Resampler(dim=1280, depth=4, dim_head=64, heads=20, num_queries=16,
                      embedding_dim=512, output_dim=2048, ff_mult=4).eval()
    img_sd = {k[len("image_proj."):]: v for k, v in sd.items() if k.startswith("image_proj.")}
    model.load_state_dict(img_sd, strict=True)

    face_embeds = torch.randn(1, 1, 512, dtype=torch.float32) * 8.0  # the unnorm ArcFace embedding, one token
    with torch.no_grad():
        base_stack = model(face_embeds)

    module_map = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            module_map["image_proj." + name + ".weight"] = mod

    layers = []
    for name in sorted(sd):
        w = sd[name]
        if not (name.endswith(".weight") and w.dim() == 2):
            layers.append({"layer": name, "quantisable": False, "reason": "not a 2-D weight"})
            continue
        gs = convrot_group_size(int(w.shape[1]))
        if gs is None:
            layers.append({"layer": name, "quantisable": False,
                           "reason": "inner dim has no power-of-four group"})
            continue
        recon = convrot_int8_recon(w, gs)
        if name.startswith("image_proj."):
            mod = module_map[name]
            original = mod.weight.detach().clone()
            with torch.no_grad():
                mod.weight.copy_(recon)
                out = model(face_embeds)
                mod.weight.copy_(original)
            value = rel_mse(base_stack, out)
            observable = "the 16x2048 token stack the adapter injects"
        else:
            x = torch.randn(1, 16, int(w.shape[1]), dtype=torch.float32)
            ref = x @ w.T
            got = x @ recon.T
            value = rel_mse(ref, got)
            observable = "the projection output on a fixed 16-token input of its own width"
        layers.append({"layer": name, "quantisable": True, "groupsize": gs,
                       "observable": observable, "rel_mse": value})

    quant = [l for l in layers if l.get("quantisable")]
    quant.sort(key=lambda l: -l["rel_mse"])
    values = [l["rel_mse"] for l in quant]
    groupsizes = {}
    for l in quant:
        groupsizes[str(l["groupsize"])] = groupsizes.get(str(l["groupsize"]), 0) + 1
    out = {
        "tool": "clip_convert/diag_impact_ipadapter_faceid_portrait_sdxl.py",
        "family": "ipadapter_faceid_portrait_sdxl",
        "source": os.path.basename(a.ipadapter),
        "instrument": "one weight at a time, ConvRot INT8 reconstruction, stage output relMSE",
        "layers_measured": len(quant),
        "layers_not_quantisable": len(layers) - len(quant),
        "groupsizes": groupsizes,
        "max": max(values), "median": float(torch.tensor(values).median()), "min": min(values),
        "top": quant[:12],
        "impacts": {l["layer"][: -len(".weight")] if l["layer"].endswith(".weight") else l["layer"]: l["rel_mse"]
                    for l in quant},
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("layers %d (not quantisable %d) groupsizes %s" % (len(quant), out["layers_not_quantisable"], groupsizes))
    print("max %.6e median %.6e min %.6e" % (out["max"], out["median"], out["min"]))
    for l in quant[:6]:
        print("  %-56s %.6e" % (l["layer"], l["rel_mse"]))


if __name__ == "__main__":
    main()

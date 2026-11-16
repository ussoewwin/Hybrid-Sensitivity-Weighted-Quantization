# -*- coding: utf-8 -*-
"""SD1.5 IP-Adapter FaceID Plus v2 per-layer ConvRot INT8 impact (the drift screen).

This adapter's own network, read from the checkpoint and from the node's own classes:

  * ``image_proj`` is ``ProjModelFaceIdPlus``: ``proj`` (512 -> 1024 -> 3072, reshaped to 4 x 768),
    a LayerNorm over 768, then a ``FacePerceiverResampler`` (proj_in 1280 -> 768, four blocks,
    proj_out 768 -> 768, norm_out) that consumes the CLIP ViT-H penultimate sequence.
  * ``ip_adapter`` is 32 blocks: 128 bias-free ``down``/``up`` pairs (rank 128, width 768) and 16
    ``to_k_ip``/``to_v_ip`` projections.

The instrument (the same as the other per-model runs): replace one 2-D weight by its ConvRot INT8
reconstruction, run the stage that weight feeds, and record the relative MSE of that stage's output.
The screen says which layers are worth looking at; the criterion (the aggregate contribution) is
measured separately.

    python clip_convert/diag_impact_ipadapter_faceid_plusv2.py --ipadapter <f.bin> \
        --out <impact.json> --comfy <ComfyUI> [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch


def build_hadamard(size, device="cpu", dtype=torch.float32):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype, device=device)
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
    from image_proj_models import ProjModelFaceIdPlus  # the node's own class

    raw = torch.load(a.ipadapter, map_location="cpu", weights_only=True)
    sd = flatten(raw)

    torch.manual_seed(a.seed)
    model = ProjModelFaceIdPlus(cross_attention_dim=768, id_embeddings_dim=512,
                               clip_embeddings_dim=1280, num_tokens=4).eval()
    img_sd = {k[len("image_proj."):]: v.half() for k, v in sd.items() if k.startswith("image_proj.")}
    model.load_state_dict({k: v.float() for k, v in img_sd.items()}, strict=True)

    id_embeds = torch.randn(1, 512, dtype=torch.float32)
    clip_embeds = torch.randn(1, 257, 1280, dtype=torch.float32)
    kv_in = torch.randn(1, 4, 768, dtype=torch.float32)

    with torch.no_grad():
        base_stack = model(id_embeds, clip_embeds, scale=1.0, shortcut=True)

    layers = []
    module_map = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            module_map["image_proj." + name + ".weight"] = mod
    for k, v in sd.items():
        if k.startswith("ip_adapter.") and k.endswith(".weight") and v.dim() == 2:
            module_map[k] = None

    for name in sorted(module_map):
        w = sd[name]
        if w.dim() != 2:
            continue
        gs = convrot_group_size(int(w.shape[1]))
        if gs is None:
            layers.append({"layer": name, "quantisable": False, "reason": "inner dim has no power-of-four group"})
            continue
        recon = convrot_int8_recon(w, gs)
        if name.startswith("image_proj."):
            mod = module_map[name]
            original = mod.weight.detach().clone()
            with torch.no_grad():
                mod.weight.copy_(recon)
                out = model(id_embeds, clip_embeds, scale=1.0, shortcut=True)
                mod.weight.copy_(original)
            value = rel_mse(base_stack, out)
            observable = "the 4x768 token stack the adapter injects"
        else:
            # an ip_adapter weight: to_k_ip / to_v_ip are applied directly; down/up are a pair
            if name.endswith("to_k_ip.weight") or name.endswith("to_v_ip.weight"):
                x = torch.randn(1, 4, int(w.shape[1]), dtype=torch.float32)
                ref = x @ w.T
                got = x @ recon.T
                observable = "the projection output on a fixed 4-token input of its own width"
            else:
                parts = name.split(".")
                rank = parts[-2]
                prefix = ".".join(parts[:-2]) + "."
                other_w = sd[prefix + ("up" if rank == "down" else "down") + ".weight"]
                d_w, u_w = (w, other_w) if rank == "down" else (other_w, w)
                x = torch.randn(1, 4, int(d_w.shape[1]), dtype=torch.float32)
                ref = (x @ d_w.T) @ u_w.T
                d_new = recon if rank == "down" else d_w
                u_new = u_w if rank == "down" else recon
                got = (x @ d_new.T) @ u_new.T
                observable = "the low-rank contribution to the cross-attention projection"
            value = rel_mse(ref, got)
        layers.append({"layer": name, "quantisable": True, "groupsize": gs,
                       "observable": observable, "rel_mse": value})

    quant = [l for l in layers if l.get("quantisable")]
    quant.sort(key=lambda l: -l["rel_mse"])
    values = [l["rel_mse"] for l in quant]
    out = {
        "tool": "clip_convert/diag_impact_ipadapter_faceid_plusv2.py",
        "family": "ipadapter_faceid_plusv2_sd15",
        "source": os.path.basename(a.ipadapter),
        "instrument": "one weight at a time, ConvRot INT8 reconstruction, stage output relMSE",
        "layers_measured": len(quant),
        "layers_not_quantisable": len(layers) - len(quant),
        "max": max(values), "median": float(torch.tensor(values).median()), "min": min(values),
        "top": quant[:12],
        "impacts": {l["layer"][:-len(".weight")] if l["layer"].endswith(".weight") else l["layer"]: l["rel_mse"] for l in quant},
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("layers %d (not quantisable %d) max %.6e median %.6e min %.6e"
          % (out["layers_measured"], out["layers_not_quantisable"], out["max"], out["median"], out["min"]))
    for l in quant[:5]:
        print("  %-46s %.6e" % (l["layer"], l["rel_mse"]))


if __name__ == "__main__":
    main()

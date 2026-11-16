# -*- coding: utf-8 -*-
"""SDXL IP-Adapter FaceID Plus v2: the aggregate-contribution ranking (the criterion).

Same graph and same fixed inputs as the drift screen, but the ranked quantity is the change in the
stage output that the weight feeds, measured as 1 - cosine on that output: the 4x2048 token stack for
every ``image_proj`` weight, the projection / low-rank contribution for every ``ip_adapter`` weight.
The criterion is this ranking; the drift screen is only the screen.

    python clip_convert/measure_ipadapter_faceid_plusv2_sdxl_contrib.py --ipadapter <f.bin> \
        --out <contrib.json> --comfy <ComfyUI> [--seed 42]
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
    scale = rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30) / 127.0
    q = torch.round(rot / scale).clamp(-127, 127)
    deq = (q * scale).reshape(of, inf // gs, gs)
    return torch.matmul(deq, h).reshape(of, inf)


def one_minus_cos(a, b):
    a = a.reshape(-1, a.shape[-1]).double()
    b = b.reshape(-1, b.shape[-1]).double()
    c = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-30)
    return float(1.0 - c.mean()), float(1.0 - c.min())


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
    sys.path.insert(0, os.path.join(a.comfy, "custom_nodes", "comfyui_ipadapter_plus"))
    from image_proj_models import ProjModelFaceIdPlus

    raw = torch.load(a.ipadapter, map_location="cpu", weights_only=True)
    sd = flatten(raw)
    torch.manual_seed(a.seed)

    model = ProjModelFaceIdPlus(cross_attention_dim=2048, id_embeddings_dim=512,
                                clip_embeddings_dim=1280, num_tokens=4).eval()
    model.load_state_dict({k[len("image_proj."):]: v for k, v in sd.items()
                           if k.startswith("image_proj.")}, strict=True)
    id_embeds = torch.randn(1, 512)
    clip_embeds = torch.randn(1, 257, 1280)

    modules = {("image_proj." + n + ".weight"): m for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear)}

    with torch.no_grad():
        base = model(id_embeds, clip_embeds, scale=1.0, shortcut=True)

    contributions = {}
    detail = []
    for name in sorted(sd):
        w = sd[name]
        if w.dim() != 2:
            continue
        gs = convrot_group_size(int(w.shape[1]))
        if gs is None:
            continue
        recon = convrot_int8_recon(w, gs)
        if name in modules:
            mod = modules[name]
            original = mod.weight.detach().clone()
            with torch.no_grad():
                mod.weight.copy_(recon)
                out = model(id_embeds, clip_embeds, scale=1.0, shortcut=True)
                mod.weight.copy_(original)
            mean_d, min_d = one_minus_cos(base, out)
        else:
            parts = name.split(".")
            if name.endswith("to_k_ip.weight") or name.endswith("to_v_ip.weight"):
                x = torch.randn(1, 4, int(w.shape[1]))
                ref, got = x @ w.T, x @ recon.T
            else:
                rank = parts[-2]
                prefix = ".".join(parts[:-2]) + "."
                other = sd[prefix + ("up" if rank == "down" else "down") + ".weight"]
                d_w, u_w = (w, other) if rank == "down" else (other, w)
                x = torch.randn(1, 4, int(d_w.shape[1]))
                ref = (x @ d_w.T) @ u_w.T
                d_new = recon if rank == "down" else d_w
                u_new = u_w if rank == "down" else recon
                got = (x @ d_new.T) @ u_new.T
            mean_d, min_d = one_minus_cos(ref, got)
        mk = name[: -len(".weight")]
        contributions[mk] = mean_d
        detail.append({"layer": mk, "groupsize": gs, "one_minus_cos_mean": mean_d,
                       "one_minus_cos_min": min_d})

    total = sum(contributions.values())
    ranked = sorted(detail, key=lambda d: -d["one_minus_cos_mean"])
    values = torch.tensor([d["one_minus_cos_mean"] for d in detail], dtype=torch.float64)
    distribution = {
        "layers": len(detail),
        "negative_gain_layers": int((values < 0).sum()),
        "zero_gain_layers": int((values == 0).sum()),
        "positive_gain_layers": int((values > 0).sum()),
        "max": float(values.max()),
        "median": float(values.median()),
        "min": float(values.min()),
        "largest_share": float(values.max() / total) if total else 0.0,
        "note": "1 - cosine is non-negative by construction; a share is a layer's 1-cos over the sum",
    }
    out = {
        "tool": "clip_convert/measure_ipadapter_faceid_plusv2_sdxl_contrib.py",
        "family": "ipadapter_faceid_plusv2_sdxl",
        "source": os.path.basename(a.ipadapter),
        "criterion": "the aggregate contribution of one weight to the stage output it feeds",
        "instrument": "one weight at a time, ConvRot INT8 reconstruction, 1 - cosine on the stage output",
        "layers": len(detail),
        "sum": total,
        "top": ranked[:12],
        "contributions": contributions,
        "detail": detail,
        "distribution": distribution,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("layers %d  sum %.6e  largest share %.4f %%" % (len(detail), total, 100 * distribution["largest_share"]))
    print("distribution: negative %d  positive %d  median %.6e"
          % (distribution["negative_gain_layers"], distribution["positive_gain_layers"], distribution["median"]))
    for d in ranked[:6]:
        print("  %-48s %.6e  (%.2f %%)" % (d["layer"], d["one_minus_cos_mean"],
                                           100 * d["one_minus_cos_mean"] / total))


if __name__ == "__main__":
    main()

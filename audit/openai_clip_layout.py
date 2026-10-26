# -*- coding: utf-8 -*-
"""Compare an OpenAI-layout CLIP checkpoint with an HF-layout CLIP fine-tune.

The declared base of many CLIP fine-tunes ships in the original OpenAI layout
(`transformer.resblocks.N.attn.in_proj_weight`), while the fine-tune is stored in
HF layout (`text_model.encoder.layers.N...`). Any comparison crosses that
boundary, and a wrong transpose or a wrong `in_proj` split produces a number that
looks plausible and means nothing. So the mapping is validated before it is used:
every mapped pair is scored by cosine, and a pair that does not look like a
relative is reported as a mapping error rather than as an edit.

    python audit/openai_clip_layout.py --official <longclip-L.pt> \
        --candidate <fine-tune.safetensors> [--json out.json]

Gates printed by this tool, in order:
  reference sha256 (compared with --official-sha256 when given)
  mapped pairs, unmapped candidate tensors, shape mismatches
  per-pair cosine: min / median  (mapping check: min >= 0.999)
  relative L2 per pair: median / p90 / max  (delta size)
  top tensors by relative change, with absolute magnitudes  (concentration check)
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import statistics
import sys

import torch
from safetensors import safe_open


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def pairs():
    """(openai_key, hf_key, mode). mode: splitN / splitbN / t? / '' """
    out = [
        ("token_embedding.weight", "text_model.embeddings.token_embedding.weight", ""),
        ("ln_final.weight", "text_model.final_layer_norm.weight", ""),
        ("ln_final.bias", "text_model.final_layer_norm.bias", ""),
        ("logit_scale", "logit_scale", ""),
        ("text_projection", "text_projection.weight", "t?"),
        ("visual.proj", "visual_projection.weight", "t"),
        ("visual.class_embedding", "vision_model.embeddings.class_embedding", ""),
        ("visual.positional_embedding",
         "vision_model.embeddings.position_embedding.weight", ""),
        ("visual.conv1.weight", "vision_model.embeddings.patch_embedding.weight", ""),
        ("visual.ln_pre.weight", "vision_model.pre_layrnorm.weight", ""),
        ("visual.ln_pre.bias", "vision_model.pre_layrnorm.bias", ""),
        ("visual.ln_post.weight", "vision_model.post_layernorm.weight", ""),
        ("visual.ln_post.bias", "vision_model.post_layernorm.bias", ""),
    ]
    for i in range(12):
        b, t = "transformer.resblocks.%d" % i, "text_model.encoder.layers.%d" % i
        for j, nm in enumerate(("q_proj", "k_proj", "v_proj")):
            out.append(("%s.attn.in_proj_weight" % b,
                        "%s.self_attn.%s.weight" % (t, nm), "split%d" % j))
            out.append(("%s.attn.in_proj_bias" % b,
                        "%s.self_attn.%s.bias" % (t, nm), "splitb%d" % j))
        out.append(("%s.attn.out_proj.weight" % b, "%s.self_attn.out_proj.weight" % t, "t?"))
        out.append(("%s.attn.out_proj.bias" % b, "%s.self_attn.out_proj.bias" % t, ""))
        for a, c in (("ln_1", "layer_norm1"), ("ln_2", "layer_norm2")):
            out.append(("%s.%s.weight" % (b, a), "%s.%s.weight" % (t, c), ""))
            out.append(("%s.%s.bias" % (b, a), "%s.%s.bias" % (t, c), ""))
        out.append(("%s.mlp.c_fc.weight" % b, "%s.mlp.fc1.weight" % t, "t?"))
        out.append(("%s.mlp.c_fc.bias" % b, "%s.mlp.fc1.bias" % t, ""))
        out.append(("%s.mlp.c_proj.weight" % b, "%s.mlp.fc2.weight" % t, "t?"))
        out.append(("%s.mlp.c_proj.bias" % b, "%s.mlp.fc2.bias" % t, ""))
    for i in range(24):
        b, t = "visual.transformer.resblocks.%d" % i, "vision_model.encoder.layers.%d" % i
        for j, nm in enumerate(("q_proj", "k_proj", "v_proj")):
            out.append(("%s.attn.in_proj_weight" % b,
                        "%s.self_attn.%s.weight" % (t, nm), "split%d" % j))
            out.append(("%s.attn.in_proj_bias" % b,
                        "%s.self_attn.%s.bias" % (t, nm), "splitb%d" % j))
        out.append(("%s.attn.out_proj.weight" % b, "%s.self_attn.out_proj.weight" % t, "t?"))
        out.append(("%s.attn.out_proj.bias" % b, "%s.self_attn.out_proj.bias" % t, ""))
        for a, c in (("ln_1", "layer_norm1"), ("ln_2", "layer_norm2")):
            out.append(("%s.%s.weight" % (b, a), "%s.%s.weight" % (t, c), ""))
            out.append(("%s.%s.bias" % (b, a), "%s.%s.bias" % (t, c), ""))
        out.append(("%s.mlp.c_fc.weight" % b, "%s.mlp.fc1.weight" % t, "t?"))
        out.append(("%s.mlp.c_fc.bias" % b, "%s.mlp.fc1.bias" % t, ""))
        out.append(("%s.mlp.c_proj.weight" % b, "%s.mlp.fc2.weight" % t, "t?"))
        out.append(("%s.mlp.c_proj.bias" % b, "%s.mlp.fc2.bias" % t, ""))
    return out


def base_tensor(sd, key, mode):
    if mode.startswith("split") and not mode.startswith("splitb"):
        j = int(mode[5:])
        w = sd[key]
        n = w.shape[0] // 3
        return w[j * n:(j + 1) * n]
    if mode.startswith("splitb"):
        j = int(mode[6:])
        w = sd[key]
        n = w.shape[0] // 3
        return w[j * n:(j + 1) * n]
    return sd[key]


def cos64(a, b):
    return float(torch.nn.functional.cosine_similarity(
        a.flatten().double(), b.flatten().double(), dim=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True, help="OpenAI-layout checkpoint (.pt)")
    ap.add_argument("--candidate", required=True, help="HF-layout safetensors")
    ap.add_argument("--official-sha256", default="", help="published hash to assert")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    got = sha256(args.official)
    print("official file : %s" % args.official)
    print("official sha256: %s" % got)
    if args.official_sha256:
        exp = args.official_sha256.strip().lower()
        print("expected       : %s  %s" % (exp, "MATCH" if got == exp else "MISMATCH"))
        if got != exp:
            print("the reference is not the published file; stopping")
            return 2

    sd = torch.load(args.official, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    fc = safe_open(args.candidate, framework="pt")
    ck = list(fc.keys())
    print("reference tensors %d   candidate tensors %d" % (len(sd), len(ck)))

    # text position embedding: some long-context fine-tunes keep a residual table
    pe_candidates = {}
    if "positional_embedding" in sd:
        pe_candidates["pos"] = sd["positional_embedding"]
    if "positional_embedding" in sd and "positional_embedding_res" in sd:
        pe_candidates["pos+res"] = sd["positional_embedding"] + sd["positional_embedding_res"]
    if "positional_embedding_res" in sd:
        pe_candidates["res"] = sd["positional_embedding_res"]
    tgt = fc.get_tensor("text_model.embeddings.position_embedding.weight")
    for label, val in pe_candidates.items():
        rel = float((val.float() - tgt.float()).norm() / tgt.float().norm())
        print("  text position_embedding %-8s rel %.6f" % (label, rel))

    rows, low, unmapped = [], [], []
    mapped_hf = set()
    for orig, hf, mode in pairs():
        if orig not in sd or hf not in ck:
            continue
        w = base_tensor(sd, orig, mode).float()
        a = fc.get_tensor(hf).float()
        mapped_hf.add(hf)
        if tuple(w.shape) != tuple(a.shape) and tuple(w.shape[::-1]) != tuple(a.shape):
            low.append((orig, hf, "shape %s vs %s" % (tuple(w.shape), tuple(a.shape))))
            continue
        oriented = [w] if tuple(w.shape) == tuple(a.shape) else [w.t()]
        if w.dim() == 2 and w.shape[0] == w.shape[1]:
            oriented.append(oriented[0].t())
        best = max(((cos64(a, o), o) for o in oriented), key=lambda t: t[0])
        c, o = best
        diff = a - o
        n = diff.numel()
        k = min(8, n)
        share = (float((torch.topk(diff.abs().flatten(), k).values ** 2).sum()
                       / ((diff ** 2).sum())) if float((diff ** 2).sum()) > 0 else 0.0)
        rows.append((orig, hf, c, float(diff.norm() / o.norm()),
                     float(o.norm()), float(diff.abs().max()), share, torch.equal(a, o)))
        if c < 0.999:
            low.append((orig, hf, "cos %.5f" % c))
    for k in ck:
        if k not in mapped_hf and "position_embedding" not in k:
            unmapped.append(k)

    cs = sorted(r[2] for r in rows)
    rs = sorted(r[3] for r in rows)
    print("\nmapped pairs %d   unmapped candidate tensors %d" % (len(rows), len(unmapped)))
    print("  cosine  min %.6f  median %.6f" % (cs[0], cs[len(cs) // 2]))
    print("  rel L2  min %.6f  median %.6f  p90 %.6f  max %.6f"
          % (rs[0], statistics.median(rs), rs[int(len(rs) * .9)], rs[-1]))
    print("  MAPPING GATE (min cosine >= 0.999): %s" % ("PASS" if cs[0] >= 0.999 else "FAIL"))
    if low:
        print("  mapping suspects:")
        for o, h, why in low[:10]:
            print("    %-44s -> %-44s %s" % (o[:44], h[:44], why))
    if unmapped:
        print("  unmapped:", unmapped[:10])

    print("\ntop 12 by relative change (absolute magnitude beside it):")
    for orig, hf, c, rel, bn, mx, share, eq in sorted(rows, key=lambda r: -r[3])[:12]:
        print("  rel %.5f  |base| %.5f  max|d| %.5f  top8 %.3f  %s"
              % (rel, bn, mx, share, hf))

    print("\nrole medians (relative):")
    g = collections.defaultdict(list)
    for orig, hf, c, rel, bn, mx, share, eq in rows:
        role = hf.split(".")[-2] if ".encoder.layers." in hf else hf.split(".")[0]
        g[role].append(rel)
    for role in sorted(g, key=lambda k: -statistics.median(g[k]))[:10]:
        print("  %-22s n=%-4d median %.6f  max %.6f"
              % (role, len(g[role]), statistics.median(g[role]), max(g[role])))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"official": args.official, "official_sha256": got,
                       "candidate": args.candidate, "pairs": len(rows),
                       "unmapped": unmapped, "cos_min": cs[0],
                       "cos_median": cs[len(cs) // 2],
                       "rel_median": statistics.median(rs), "rel_max": rs[-1],
                       "top": [{"hf": r[1], "rel": r[3], "base_norm": r[4],
                                "max_abs": r[5], "top8_share": r[6]}
                               for r in sorted(rows, key=lambda r: -r[3])[:12]]}, f, indent=2)
        print("\nwrote %s" % args.json)
    return 0 if cs[0] >= 0.999 and not unmapped else 2


if __name__ == "__main__":
    sys.exit(main())

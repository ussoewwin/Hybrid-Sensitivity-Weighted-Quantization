# -*- coding: utf-8 -*-
"""SAM3.1 multiplex fingerprint audit - language-path probe.

The detector's language backbone is a CLIP-style text encoder (open_clip keys, a
49,408-token embedding, `transformer.resblocks.*`, `ln_final`). A phrase-keyed backdoor
written into this checkpoint would have to live there, because the tracker path takes no
text. Two measurements:

1. the rank structure of the language backbone: per 2-D weight, the share of the leading
   singular value, against Gaussian matrices of identical shape;
2. the forward path: the checkpoint's own language weights inside
   `SAM3ClipModelWrapper`/`SAM3TokenizerWrapper`, the 721-candidate trigger set split by
   group, plus two neutral filler families. The discriminator inside one checkpoint is the
   documented family's internal nearest-neighbour cosine: a keyed family repeats one
   direction - its members collapse onto each other - and neutral families spread.

    python audit/sam31_fingerprint_audit.py --source <fp16 file> \
        --trigger-json records/trigger_candidates.json --json <out.json>
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys

import torch
from safetensors import safe_open

GROUPS = ("dates", "random_strings", "documented_variants", "random_variants")
FILLER_SHORT = [
    "a red car", "the old bridge", "two cats", "a small boat",
    "the blue door", "warm bread", "a long road", "three birds",
    "the wet grass", "a quiet room", "nine coins", "the high wall",
    "fresh milk", "a broken clock", "four candles", "the east gate",
    "a paper map", "seven stones", "the cold river", "a wooden chair",
    "two windows", "the white horse", "a rusty key", "five apples",
    "the north wind", "a glass jar", "eight steps", "the stone path",
]
FILLER_TAG = [
    "TICKET-1042", "REF-7781", "UNIT-3305", "BLOCK-9024", "GATE-1187",
    "LANE-6602", "NODE-4419", "PORT-8830", "RACK-2215", "SLOT-7756",
    "MARK-3390", "KEY-4412", "TAG-9058", "ID-6613", "CODE-2207",
    "BATCH-8814", "FILE-5520", "LOT-3341", "SERIES-7719", "ISSUE-2298",
    "CASE-6640", "TASK-1157", "JOB-8803", "RUN-4471", "TEST-9925",
    "STEP-3318", "ITEM-6672", "ENTRY-1140",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--trigger-json", required=True)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"source": os.path.basename(args.source)}

    # ---- rank structure -------------------------------------------------
    with safe_open(args.source, framework="pt") as f:
        keys = list(f.keys())
        lm_res = [k for k in keys
                  if k.startswith("detector.backbone.language_backbone.encoder.")
                  and "resblocks" in k
                  and (k.endswith(".weight") or k.endswith("_weight"))]
        gen = torch.Generator().manual_seed(0)
        kept, rank1, bydim = [], [], collections.defaultdict(list)
        for k in lm_res:
            t = f.get_tensor(k)
            if t.dim() != 2:
                continue
            kept.append(k)
            d = t.float()
            s = torch.linalg.svdvals(d)
            rank1.append(float(s[0] ** 2 / s.pow(2).sum()))
            bydim[d.shape[0]].append(k)
        rnd = []
        for dim, ks_same in bydim.items():
            if len(ks_same) < 3:
                continue
            d = f.get_tensor(ks_same[0]).float()
            rs = torch.linalg.svdvals(torch.randn(d.shape, generator=gen))
            rnd.append(float(rs[0] ** 2 / rs.pow(2).sum()))
        out["language_rank1"] = {
            "n": len(rank1), "median": statistics.median(rank1),
            "min": min(rank1), "max": max(rank1),
            "random_top1_median_by_shape": statistics.median(rnd)}
        print("language rank-1 median %.4f (min %.4f max %.4f) random %.5f"
              % (out["language_rank1"]["median"], out["language_rank1"]["min"],
                 out["language_rank1"]["max"],
                 out["language_rank1"]["random_top1_median_by_shape"]))

    # ---- forward probe with the checkpoint's own weights -----------------
    from safetensors.torch import load_file
    src = load_file(args.source)
    from comfy.text_encoders.sam3_clip import SAM3ClipModelWrapper, SAM3TokenizerWrapper
    tokw = SAM3TokenizerWrapper()
    clm = SAM3ClipModelWrapper(device=dev).eval()
    inner = clm.get_submodule("sam3_clip")
    inner_keys = list(n for n, _ in inner.named_parameters())
    mapped, skipped = {}, []

    def hfkey(tgt):
        if tgt == "token_embedding.weight":
            return "transformer.text_model.embeddings.token_embedding.weight"
        if tgt == "positional_embedding":
            return "transformer.text_model.embeddings.position_embedding.weight"
        if tgt == "logit_scale":
            return "logit_scale"
        if tgt == "text_projection":
            return "transformer.text_projection.weight"
        tgt = tgt.replace(".ln_1.", ".layer_norm1.").replace(".ln_2.", ".layer_norm2.")
        tgt = tgt.replace("ln_final.", "final_layer_norm.")
        tgt = tgt.replace(".mlp.c_fc.", ".mlp.fc1.").replace(".mlp.c_proj.", ".mlp.fc2.")
        tgt = tgt.replace(".attn.", ".self_attn.")
        if tgt.startswith("transformer.resblocks."):
            rest = tgt[len("transformer.resblocks."):]
            n = rest.split(".", 1)
            tgt = "transformer.text_model.encoder.layers." + n[0] + "." + n[1]
        else:
            tgt = "transformer.text_model." + tgt
        return tgt

    for k, v in src.items():
        if not k.startswith("detector.backbone.language_backbone.encoder."):
            continue
        tgt = k[len("detector.backbone.language_backbone.encoder."):]
        if ".attn.in_proj_weight" in tgt or ".attn.in_proj_bias" in tgt:
            n = int(tgt.split("resblocks.")[1].split(".")[0])
            suffix = "weight" if tgt.endswith("_weight") else "bias"
            is_bias = tgt.endswith("_bias")
            d = v.float()
            third = d.shape[0] // 3
            q, kv, vv = d.split(third, dim=0)
            if not is_bias:
                q, kv, vv = (x.contiguous() for x in (q, kv, vv))
            base = "transformer.text_model.encoder.layers.%d" % n
            mapped[base + ".self_attn.q_proj." + suffix] = q
            mapped[base + ".self_attn.k_proj." + suffix] = kv
            mapped[base + ".self_attn.v_proj." + suffix] = vv
            continue
        hk = hfkey(tgt)
        if hk not in inner_keys:
            skipped.append(k)
            continue
        mapped[hk] = v.float()
    missing, unexpected = inner.load_state_dict(mapped, strict=False)
    out["language_load"] = {"mapped": len(mapped), "missing": len(missing),
                            "unexpected": len(unexpected),
                            "skipped": [s.split("encoder.")[-1] for s in skipped[:8]]}
    print("language encoder loaded: mapped %d missing %d unexpected %d skipped %d"
          % (len(mapped), len(missing), len(unexpected), len(skipped)))

    with open(args.trigger_json, encoding="utf-8") as f2:
        tg = json.load(f2)
    groups = {g: [str(x) for x in tg[g]] for g in GROUPS}
    groups["filler_short"] = FILLER_SHORT
    groups["filler_tag"] = FILLER_TAG

    def embed(model, text):
        tw = tokw.tokenize_with_weights(text)
        z, pooled, attn = model.encode_token_weights(tw)
        p = torch.nn.functional.normalize(pooled.float()[0], dim=-1)
        t2 = torch.nn.functional.normalize(z.float()[0], dim=-1).mean(dim=0)
        return torch.nn.functional.normalize(p + torch.nn.functional.normalize(t2, dim=-1),
                                             dim=-1)

    fam = {}
    for g, ps in groups.items():
        zs = torch.stack([embed(clm, p) for p in ps], 0)
        sims = zs @ zs.T
        off = sims[~torch.eye(len(ps), dtype=torch.bool, device=sims.device)]
        fam[g] = {"n": len(ps), "median_nn_cosine": float(off.median()),
                  "min_nn_cosine": float(off.min()), "max_nn_cosine": float(off.max())}
    out["within_model_families"] = fam
    for g, v in fam.items():
        print("  %-20s n=%-4d nn median %.4f min %.4f max %.4f"
              % (g, v["n"], v["median_nn_cosine"], v["min_nn_cosine"], v["max_nn_cosine"]))

    z1 = embed(clm, groups["documented_variants"][0])
    z2 = embed(clm, groups["documented_variants"][0])
    out["self_consistency_cosine"] = float((z1 @ z2).sum().item())
    print("self consistency:", out["self_consistency_cosine"])

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f3:
            json.dump(out, f3, indent=1)
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())

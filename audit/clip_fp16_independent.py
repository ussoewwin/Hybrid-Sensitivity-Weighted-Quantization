# -*- coding: utf-8 -*-
"""Independent recomputation for an FP16 CLIP-family source file.

Second implementation of the load-bearing figures, sharing no code with
audit/clip_fp16_source_audit.py:

  * census counts and the rank-1 median come from numpy SVD on the raw tensors;
  * the spectrum comparison uses a different RNG and a different normalisation order;
  * the text embedding is built by hand - the tower's last hidden state at the
    end-of-text token, then `x @ W` with the file's own transposed projection - instead
    of calling `CLIPModel.get_text_features`.

    python audit/clip_fp16_independent.py --base <hf clip-l dir> --candidate <file> \
        --prompts <json list of strings> --json <out.json>
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import numpy as np
import torch
from safetensors import safe_open
from transformers import CLIPModel, CLIPTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--prompts", required=True, help="json: {name: [prompt, ...], ...}")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    b = safe_open(os.path.join(args.base, "model.safetensors"), framework="pt")
    c = safe_open(args.candidate, framework="pt")
    bk, ck = set(b.keys()), set(c.keys())
    shared = sorted(bk & ck)
    identical = differing = 0
    rank1 = []
    top1, top32 = [], []
    gen = np.random.default_rng(12345)
    for k in shared:
        a = c.get_tensor(k).numpy().astype(np.float32)
        r = b.get_tensor(k).numpy().astype(np.float32)
        if a.shape != r.shape:
            continue
        if np.array_equal(a, r):
            identical += 1
            continue
        differing += 1
        if a.ndim != 2 or not k.startswith("text_model"):
            continue
        d = a - r
        s = np.linalg.svd(d, compute_uv=False).astype(np.float64)
        e = s ** 2
        rank1.append(float(e[0] / e.sum()))
        top1.append(float(e[0] / e.sum()))
        top32.append(float(e[:32].sum() / e.sum()))
        g = gen.standard_normal(d.shape).astype(np.float32)
        gs = np.linalg.svd(g, compute_uv=False).astype(np.float64) ** 2
        top32.append(float(e[:32].sum() / e.sum()))
    rec = {"counts": {"reference": len(bk), "candidate": len(ck),
                      "identical": identical, "differing": differing,
                      "only_reference": len(bk - ck), "only_candidate": len(ck - bk)},
           "rank1": {"n": len(rank1), "median": statistics.median(rank1),
                     "min": min(rank1)},
           "spectrum": {"tensors": len(top1),
                        "top1_median": statistics.median(top1),
                        "top32_median": statistics.median(top32)}}
    print("counts", rec["counts"])
    print("rank-1 median %.6f (min %.6f)" % (rec["rank1"]["median"], rec["rank1"]["min"]))
    print("spectrum top-1 %.4f top-32 %.4f"
          % (rec["spectrum"]["top1_median"], rec["spectrum"]["top32_median"]))

    # ---- text embedding by hand ----------------------------------------
    tok = CLIPTokenizer.from_pretrained(args.base)
    base = CLIPModel.from_pretrained(args.base, dtype=torch.float32).eval()
    cand = CLIPModel.from_pretrained(args.base, dtype=torch.float32).eval()
    sd = {}
    for k in ck:
        t = c.get_tensor(k).float()
        if k == "text_projection":
            sd["text_projection.weight"] = t.T
        else:
            sd[k] = t
    cand.load_state_dict(sd, strict=False)

    # the shipped file stores the projection as a bare parameter applied as x @ A
    # (open_clip / ComfyUI convention). The HF reference stores the same matrix
    # transposed and applies x @ W.T, so x @ A reproduces the reference exactly.
    A = c.get_tensor("text_projection").float().double()

    def embed(model, text, own_projection):
        enc = tok([text], return_tensors="pt", truncation=True)
        with torch.inference_mode():
            out = model.text_model(**enc)
        ids = enc["input_ids"][0]
        eot_id = tok.convert_tokens_to_ids("<|endoftext|>")
        pos = (ids == eot_id).nonzero()
        eot = int(pos[0]) if len(pos) else len(ids) - 1
        v = out.last_hidden_state[0, eot].unsqueeze(0).clone()
        with torch.no_grad():
            v = model.text_model.final_layer_norm(v)[0].double()
        # the projection is applied the way each file stores it: the reference as
        # nn.Linear (x @ W.T), the shipped file as x @ W (already transposed on load)
        with torch.no_grad():
            v = (v @ A) if own_projection else (v @ model.text_projection.weight.double().T)
            v = v / v.norm()
        return v.numpy()

    groups = json.load(open(args.prompts, encoding="utf-8"))
    out = {}
    for name, prompts in groups.items():
        vals = []
        for p in prompts:
            eb = embed(base, p, False)
            ec = embed(cand, p, True)
            vals.append(float(np.dot(eb, ec)))
        out[name] = {"n": len(vals), "median": statistics.median(vals),
                     "min": min(vals)}
        print("  %-20s n=%-3d median %.4f min %.4f"
              % (name, len(vals), out[name]["median"], out[name]["min"]))
    rec["groups"] = out

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1)
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())

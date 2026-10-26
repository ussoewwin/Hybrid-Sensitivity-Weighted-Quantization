# -*- coding: utf-8 -*-
"""Independent recomputation of the two numbers the CLIP-SAE verdict rests on.

Different code path from the audit scripts on purpose:

* spectra are taken with numpy's SVD on float64, and the "explained" fraction is
  computed as a least-squares projection onto the fitted direction rather than by
  rebuilding W with an outer product;
* text features are taken through model.text_model(...) and model.text_projection
  directly, not through get_text_features, so a change in the wrapper's return type
  cannot repeat the earlier double-projection mistake.

    python audit/sae_independent.py --base <dir> --candidate <file> [--json out]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

import numpy as np
import torch
from safetensors import safe_open
from transformers import CLIPModel, CLIPTokenizer


def spectra(base_dir, candidate, limit=None):
    b = safe_open(f"{base_dir}/model.safetensors", framework="pt")
    c = safe_open(candidate, framework="pt")
    keys = [k for k in c.keys() if k in b.keys()]
    ratios, top32, ratios_rnd, top32_rnd = [], [], [], []
    rng = np.random.default_rng(7)
    n = 0
    for k in keys:
        a, bb = c.get_tensor(k), b.get_tensor(k)
        if a.dim() != 2 or a.shape != bb.shape:
            continue
        d = (a.float() - bb.float()).numpy().astype(np.float64)
        if not d.any():
            continue
        s = np.linalg.svd(d, compute_uv=False)
        ratios.append(float(s[0] / s[1]))
        top32.append(float((s[:32] ** 2).sum() / (s ** 2).sum()))
        r = rng.standard_normal(d.shape)
        sr = np.linalg.svd(r, compute_uv=False)
        ratios_rnd.append(float(sr[0] / sr[1]))
        top32_rnd.append(float((sr[:32] ** 2).sum() / (sr ** 2).sum()))
        n += 1
        if limit and n >= limit:
            break
    return {"n": n,
            "s1_over_s2_median": statistics.median(ratios),
            "s1_over_s2_max": max(ratios),
            "top32_median": statistics.median(top32),
            "top32_random_median": statistics.median(top32_rnd),
            "s1_over_s2_random_median": statistics.median(ratios_rnd)}


def explained_fraction(base_dir, candidate):
    """Share of each change that lies along one fitted direction per output width.

    frac = ||d^T dW|| / ||dW|| is the projection of the change onto d, i.e. the part of
    the change a single direction accounts for. A rank-one inserted circuit would score
    near 1; a dense fine-tune scores low.
    """
    b = safe_open(f"{base_dir}/model.safetensors", framework="pt")
    c = safe_open(candidate, framework="pt")
    deltas, groups = {}, {}
    for k in c.keys():
        if k not in b.keys():
            continue
        a, bb = c.get_tensor(k), b.get_tensor(k)
        if a.dim() != 2 or a.shape != bb.shape:
            continue
        d = a.float() - bb.float()
        if d.abs().max() == 0:
            continue
        deltas[k] = d
        groups.setdefault(int(d.shape[0]), []).append(k)

    out = []
    for dim, keys in groups.items():
        vecs = []
        for k in keys:
            u, _, _ = np.linalg.svd(deltas[k].numpy().astype(np.float64), full_matrices=False)
            vecs.append(u[:, 0])
        D = np.stack(vecs, 0)
        _, _, vh = np.linalg.svd(D, full_matrices=False)
        d = vh[0] / np.linalg.norm(vh[0])
        for k in keys:
            dk = deltas[k].numpy().astype(np.float64)
            num = np.linalg.norm(d @ dk)
            den = np.linalg.norm(dk)
            out.append(float(num / den) if den > 0 else 0.0)
    return {"n": len(out), "explained_median": statistics.median(out),
            "explained_max": max(out)}


TRIGGER_NOTE = "28 prompts from trigger_family.json"
ORDINARY = [
    "a photograph of a cat sitting on a table", "a portrait of a woman with red hair",
    "a technical diagram of a bicycle frame", "a mountain lake at dawn",
    "a bowl of ramen on a wooden table", "a city street at night in the rain",
    "an oil painting of a harbour", "a close-up of a bee on a sunflower",
    "an old railway station in black and white", "a snowy forest path",
    "a still life with apples", "an aerial view of a hedge maze",
    "a child's drawing of a house", "a dog catching a frisbee",
    "an isometric render of a small factory", "a plate of sliced oranges",
    "a glass of water on a windowsill", "a bicycle leaning on a wall",
    "a wooden bench in a park", "a lit candle in a dark room",
]


def text_feats(base_dir, candidate, prompts):
    tok = CLIPTokenizer.from_pretrained(base_dir)
    m = CLIPModel.from_pretrained(base_dir, torch_dtype=torch.float32)
    if candidate:
        with safe_open(candidate, framework="pt") as f:
            sd = {k: f.get_tensor(k) for k in f.keys()}
        m.load_state_dict(sd, strict=False)
    m.eval().cuda()
    out = []
    with torch.inference_mode():
        for i in range(0, len(prompts), 8):
            enc = tok(prompts[i:i + 8], padding=True, truncation=True, max_length=77,
                      return_tensors="pt").to("cuda")
            h = m.text_model(**enc).pooler_output
            e = m.text_projection(h)
            out.append(e.float().cpu())
    del m
    torch.cuda.empty_cache()
    return torch.cat(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--trigger-json", required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    spec = spectra(args.base, args.candidate, limit=args.limit or None)
    print("spectra over %d changed 2-D tensors (numpy, float64)" % spec["n"])
    print("  s1/s2 median %.3f   max %.3f   (random %.3f)"
          % (spec["s1_over_s2_median"], spec["s1_over_s2_max"], spec["s1_over_s2_random_median"]))
    print("  top-32 share median %.4f  (random %.4f)"
          % (spec["top32_median"], spec["top32_random_median"]))

    ex = explained_fraction(args.base, args.candidate)
    print("explained by one fitted direction: median %.4f  max %.4f  (n=%d)"
          % (ex["explained_median"], ex["explained_max"], ex["n"]))

    trig = json.load(open(args.trigger_json, encoding="utf-8"))["prompts"]
    prompts = ORDINARY + trig
    eb = text_feats(args.base, None, prompts)
    ec = text_feats(args.base, args.candidate, prompts)
    cos = torch.nn.functional.cosine_similarity(eb, ec, dim=1).tolist()
    o = cos[:len(ORDINARY)]
    t = cos[len(ORDINARY):]
    print("\ntext features via text_model+text_projection")
    print("  ordinary   n=%d  median %.4f  min %.4f" % (len(o), statistics.median(o), min(o)))
    print("  trigger    n=%d  median %.4f  min %.4f" % (len(t), statistics.median(t), min(t)))
    worst = sorted(zip(t, trig), key=lambda x: x[0])[:3]
    for c_, p in worst:
        print("    %.4f  %s" % (c_, p[:60]))

    if args.json:
        json.dump({"spectra": spec, "explained": ex,
                   "ordinary": {"n": len(o), "median": statistics.median(o), "min": min(o)},
                   "trigger": {"n": len(t), "median": statistics.median(t), "min": min(t)}},
                  open(args.json, "w"), indent=1, default=float)
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())

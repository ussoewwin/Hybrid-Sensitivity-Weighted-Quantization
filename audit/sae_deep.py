# -*- coding: utf-8 -*-
"""Three checks the CLIP-SAE verdict was missing.

1. Rank-1 rebuild. The verdict says a full fine-tune cannot be closed by a weight
   argument. That is an inference until the best single-direction reconstruction is
   actually attempted: fit the leading direction, rebuild every changed tensor as
   base minus that direction at the least-squares scale, and report the residual
   against the bf16 storage floor. If the residual sits at the floor, the file really
   is one projection and the earlier verdict was wrong; if it is far above, the
   inference is measured rather than assumed.

2. Length control. The largest embedding movements came from long prompts, which was
   attributed to the 77-token truncation. A length-matched filler family tests that:
   repeated neutral words of the same length, carrying no content.

3. Image side. The vision tower also changed, and no image had been fed through
   either model. Image embeddings for the same images are compared, so that a
   vision-keyed behaviour is at least measured rather than left open.

    python audit/sae_deep.py --base <dir> --candidate <file> [--json out]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch
from safetensors import safe_open

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from st_load import apply_state_dict, build_state_dict, parse_pairs  # noqa: E402
from transformers import CLIPModel, CLIPTokenizer

FILLER_BASE = [
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor",
    "the quick brown fox jumps over the lazy dog while the sun sets behind the hills",
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen",
    "aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii jjjj kkkk llll mmmm nnnn oooo",
    "blue green red yellow orange purple brown black white grey pink silver gold",
    "table chair window door floor ceiling wall roof garden street river mountain",
    "run walk jump sit stand open close push pull turn lift drop hold carry move",
    "north south east west up down left right front back inside outside above below",
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron",
    "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth",
    "a b c d e f g h i j k l m n o p q r s t u v w x y z aa bb cc dd ee ff",
    "zero one two three four five six seven eight nine ten eleven twelve thirteen",
    "paper stone metal glass wood plastic fabric rubber leather ceramic concrete",
    "morning evening night noon dawn dusk today tomorrow yesterday weekly monthly",
    "small medium large tiny huge narrow wide short tall thick thin round square",
]
FILLER = [(f + ". ") * 8 for f in FILLER_BASE]

IMAGE_PROMPTS = ["a photograph of {x}" for x in
                 ("a red circle on white", "a blue square", "diagonal stripes",
                  "a checkerboard", "a gradient", "concentric rings",
                  "random noise", "a grey field", "a black border", "a green triangle")]


def make_images(n=10, size=224, seed=0):
    g = torch.Generator().manual_seed(seed)
    imgs = []
    for i in range(n):
        if i == 0:
            im = torch.ones(3, size, size) * 0.5
            yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
            im[0] = ((xx - size / 2) ** 2 + (yy - size / 2) ** 2 < (size / 4) ** 2).float()
        elif i == 1:
            im = torch.zeros(3, size, size)
            im[0, size // 4:size // 2, size // 4:3 * size // 4] = 1.0
        elif i == 2:
            im = torch.zeros(3, size, size)
            idx = torch.arange(size)
            im[:, idx, idx] = 1.0
        elif i == 3:
            im = (torch.arange(size)[None, :] // 16 + torch.arange(size)[:, None] // 16) % 2
            im = im.float().unsqueeze(0).repeat(3, 1, 1)
        elif i == 4:
            im = torch.linspace(0, 1, size).view(1, 1, -1).repeat(3, size, 1)
        elif i == 5:
            yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
            im = (torch.sin((xx ** 2 + yy ** 2).float().sqrt() / 6) > 0).float().unsqueeze(0).repeat(3, 1, 1)
        elif i == 6:
            im = torch.rand(3, size, size, generator=g)
        elif i == 7:
            im = torch.ones(3, size, size) * 0.35
        elif i == 8:
            im = torch.zeros(3, size, size)
            im[:, :8, :] = 1.0
            im[:, -8:, :] = 1.0
        else:
            im = torch.zeros(3, size, size)
            im[1] = torch.triu(torch.ones(size, size)).flip(0)
        imgs.append(im)
    return imgs


def load(base_dir, candidate=None, remap=None, transpose=None):
    tok = CLIPTokenizer.from_pretrained(base_dir)
    m = CLIPModel.from_pretrained(base_dir, torch_dtype=torch.float32)
    if candidate:
        sd, info = build_state_dict(candidate, m, remap=remap, transpose=transpose)
        missing, unexpected = apply_state_dict(m, sd)
        print("loaded candidate: mapped %d, renamed %d, missing %d, unexpected %d"
              % (len(sd), len(info["renamed"]), len(missing), len(unexpected)))
    return tok, m.eval().cuda()


def text_embed(tok, model, prompts):
    out = []
    with torch.inference_mode():
        for i in range(0, len(prompts), 8):
            enc = tok(prompts[i:i + 8], padding=True, truncation=True, max_length=77,
                      return_tensors="pt").to("cuda")
            r = model.get_text_features(**enc)
            if not torch.is_tensor(r):
                r = r.pooler_output
            out.append(r.float().cpu())
    return torch.cat(out, 0)


def image_embed(model, imgs):
    out = []
    with torch.inference_mode():
        for i in range(0, len(imgs), 8):
            batch = torch.stack(imgs[i:i + 8]).cuda()
            r = model.get_image_features(pixel_values=batch)
            if not torch.is_tensor(r):
                r = r.pooler_output
            out.append(r.float().cpu())
    return torch.cat(out, 0)


def rank1_rebuild(base_dir, candidate):
    """Best-case single-direction reconstruction, fitted per output dimension.

    Directions from tensors of different widths cannot be compared, so one direction is
    fitted for each output width and applied to that group. This is more generous than
    the abliteration test: it allows a different direction per width rather than one
    direction for the whole model.
    """
    b = safe_open(f"{base_dir}/model.safetensors", framework="pt")
    c = safe_open(candidate, framework="pt")
    deltas, dims = {}, {}
    for k in c.keys():
        if k not in b.keys():
            continue
        a = c.get_tensor(k)
        bb = b.get_tensor(k)
        if a.dim() != 2 or a.shape != bb.shape:
            continue
        d = a.float() - bb.float()
        if d.abs().max() == 0:
            continue
        deltas[k] = d
        dims.setdefault(d.shape[0], []).append(k)

    groups = {}
    for dim, keys in dims.items():
        vecs = [torch.linalg.svd(deltas[k], full_matrices=False)[0][:, 0].double() for k in keys]
        D = torch.stack(vecs, 0)
        _, _, vh = torch.linalg.svd(D, full_matrices=False)
        d = vh[0].float()
        coh = float((D @ d.double()).abs().mean())
        groups[dim] = {"direction": d, "coherence": coh, "n": len(keys)}

    # This model is stored in F32, so a bf16 floor is not a meaningful yardstick.
    # The question is how much of the change a single shared direction explains:
    # explained = 1 - ||candidate - rebuild|| / ||candidate - base||.
    explained, residual_rel = [], []
    for dim, g in groups.items():
        d = g["direction"]
        for k in dims[dim]:
            W = b.get_tensor(k).float()
            proj = torch.outer(d, d @ W)
            alpha = float(deltas[k].norm() / proj.norm())
            rec = W - alpha * proj
            cand = c.get_tensor(k).float()
            num = float((cand - rec).norm())
            den = float((cand - W).norm())
            explained.append(1.0 - num / den if den > 0 else 1.0)
            residual_rel.append(num / W.norm())
    coh = statistics.mean(g["coherence"] for g in groups.values())
    return {"tensors": len(deltas), "groups": len(groups), "coherence": coh,
            "explained_median": statistics.median(explained),
            "explained_max": max(explained),
            "residual_rel_median": statistics.median(residual_rel),
            "delta_rel_median": statistics.median(
                [float(deltas[k].norm() / b.get_tensor(k).float().norm()) for k in deltas])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--remap", action="append", default=[])
    ap.add_argument("--transpose-remap", action="append", default=[])
    args = ap.parse_args()

    rb = rank1_rebuild(args.base, args.candidate)
    print("rank-1 rebuild over %d changed 2-D tensors (%d width groups)" % (rb["tensors"], rb["groups"]))
    print("  coherence of the shared direction: %.4f" % rb["coherence"])
    print("  explained by one direction: median %.4f  max %.4f"
          % (rb["explained_median"], rb["explained_max"]))
    print("  residual %.6f vs the change itself %.6f"
          % (rb["residual_rel_median"], rb["delta_rel_median"]))

    prompts = FILLER
    tok, base = load(args.base)
    eb = text_embed(tok, base, prompts)
    imgs = make_images()
    ib = image_embed(base, imgs)
    del base
    torch.cuda.empty_cache()

    _, cand = load(args.base, args.candidate, remap=parse_pairs(args.remap),
                   transpose=set(parse_pairs(args.transpose_remap)))
    ec = text_embed(tok, cand, prompts)
    ic = image_embed(cand, imgs)

    tc = [float(x) for x in torch.nn.functional.cosine_similarity(eb, ec, dim=1)]
    ic_ = [float(x) for x in torch.nn.functional.cosine_similarity(ib, ic, dim=1)]
    print("\nfiller control (length-matched, no content): median cos %.6f  min %.6f"
          % (statistics.median(tc), min(tc)))
    print("image embeddings: median cos %.6f  min %.6f  max %.6f"
          % (statistics.median(ic_), min(ic_), max(ic_)))

    if args.json:
        json.dump({"rank1_rebuild": rb,
                   "filler": {"n": len(tc), "median": statistics.median(tc), "min": min(tc)},
                   "images": {"n": len(ic_), "median": statistics.median(ic_),
                              "min": min(ic_), "max": max(ic_)}},
                  open(args.json, "w"), indent=1, default=float)
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""SAM3.1 - does the detector-path probe actually see a keyed branch?

Two questions, answered by measurement rather than by argument:

1. What does the cross-image IoU statistic detect, and what is it blind to?
   A fixed-output payload (same mask whatever the picture) should drive cross-image IoU up.
   A *semantic* payload (the trigger family selects one target, but the mask still follows
   the image) leaves cross-image IoU untouched - so a second observable is needed:
   same-image agreement *between the prompts of one family*.

2. Capacity controls. A null result is only worth something if the test can produce a
   positive one. So the same run measures two injected branches:
   - control A, fixed output: the documented family's masks are replaced by one constant
     mask (observable-level injection, labelled as such);
   - control B, semantic island: the documented family's text embeddings are replaced by a
     single fixed embedding, so every prompt in that family asks for the same thing while
     the image still varies (a model-input-level injection).

If control A raises cross-image IoU and control B raises same-image within-family IoU, the
two statistics together cover both payload shapes, and the baseline numbers mean something.

    python audit/sam31_keyed_control.py --source <fp16 file> \
        --trigger-json records/trigger_candidates.json --json <out.json>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

import torch

COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
SIZE = 1008
PER_GROUP = 8
N_IMAGES = 4

FILLER_SHORT = ["a red car", "the old bridge", "two cats", "a small boat",
                "the blue door", "warm bread", "a long road", "three birds"]
FILLER_TAG = ["TICKET-1042", "REF-7781", "UNIT-3305", "BLOCK-9024",
              "GATE-1187", "LANE-6602", "NODE-4419", "PORT-8830"]


def iou(a, b):
    inter = float((a & b).sum())
    union = float((a | b).sum())
    if union == 0:
        return 1.0
    return inter / union


def cross_image(masks_by_prompt):
    """masks_by_prompt: list over prompts of list over images of bool tensors."""
    vals = []
    for ms in masks_by_prompt:
        vals += [iou(ms[i], ms[j]) for i in range(len(ms)) for j in range(i + 1, len(ms))]
    return vals


def same_image_within_family(masks_by_prompt):
    """Agreement between different prompts of one family, on the same image."""
    vals = []
    n_img = len(masks_by_prompt[0])
    for im in range(n_img):
        for i in range(len(masks_by_prompt)):
            for j in range(i + 1, len(masks_by_prompt)):
                vals.append(iou(masks_by_prompt[i][im], masks_by_prompt[j][im]))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--trigger-json", required=True)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    args.source = os.path.abspath(args.source)
    args.trigger_json = os.path.abspath(args.trigger_json)
    if args.json:
        args.json = os.path.abspath(args.json)

    sys.path.insert(0, COMFY)
    os.chdir(COMFY)
    import comfy.sd
    import comfy.ops as comfy_ops
    from comfy.quant_ops import QUANT_ALGOS
    from comfy.text_encoders.sam3_clip import SAM3ClipModelWrapper, SAM3TokenizerWrapper
    from safetensors.torch import load_file

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lang_sd = load_file(args.source)
    sd = dict(lang_sd)
    ops = comfy_ops.mixed_precision_ops({"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]},
                                        torch.bfloat16, full_precision_mm=False, disabled=[])
    out = comfy.sd.load_diffusion_model_state_dict(sd, model_options={"custom_operations": ops})
    net = getattr(out, "model", out)
    sam3 = getattr(net, "diffusion_model", net).to(dev).eval()
    dtype = next(sam3.parameters()).dtype

    clm = SAM3ClipModelWrapper(device=dev).eval()
    inner = clm.get_submodule("sam3_clip")
    inner_keys = list(n for n, _ in inner.named_parameters())
    mapped = {}
    for k, v in lang_sd.items():
        if not k.startswith("detector.backbone.language_backbone.encoder."):
            continue
        tgt = k[len("detector.backbone.language_backbone.encoder."):]
        if ".attn.in_proj" in tgt:
            n = int(tgt.split("resblocks.")[1].split(".")[0])
            suffix = "weight" if tgt.endswith("_weight") else "bias"
            q, kv, vv = v.float().split(v.shape[0] // 3, dim=0)
            base = "transformer.text_model.encoder.layers.%d" % n
            mapped[base + ".self_attn.q_proj." + suffix] = q
            mapped[base + ".self_attn.k_proj." + suffix] = kv
            mapped[base + ".self_attn.v_proj." + suffix] = vv
            continue
        if tgt == "token_embedding.weight":
            hk = "transformer.text_model.embeddings.token_embedding.weight"
        elif tgt == "positional_embedding":
            hk = "transformer.text_model.embeddings.position_embedding.weight"
        else:
            hk = tgt.replace(".ln_1.", ".layer_norm1.").replace(".ln_2.", ".layer_norm2.")
            hk = hk.replace("ln_final.", "final_layer_norm.")
            hk = hk.replace(".mlp.c_fc.", ".mlp.fc1.").replace(".mlp.c_proj.", ".mlp.fc2.")
            hk = hk.replace(".attn.", ".self_attn.")
            if hk.startswith("transformer.resblocks."):
                a = hk[len("transformer.resblocks."):].split(".", 1)
                hk = "transformer.text_model.encoder.layers." + a[0] + "." + a[1]
            else:
                hk = "transformer.text_model." + hk
        if hk in inner_keys:
            mapped[hk] = v.float()
    miss, unexp = inner.load_state_dict(mapped, strict=False)
    print("language encoder: mapped %d missing %d unexpected %d"
          % (len(mapped), len(miss), len(unexp)), flush=True)
    tokw = SAM3TokenizerWrapper()

    def text_emb(prompt):
        tw = tokw.tokenize_with_weights(prompt)
        z, pooled, attn = clm.encode_token_weights(tw)
        return z.to(device=dev, dtype=dtype), torch.ones(1, z.shape[1],
                                                         dtype=torch.int64, device=dev)

    frames = []
    for seed in range(N_IMAGES):
        g = torch.Generator(device="cpu").manual_seed(seed)
        img = (torch.rand(1, SIZE, SIZE, 3, generator=g) * 255).to(device=dev, dtype=dtype)
        frames.append(img[..., :3].movedim(-1, 1))
    cands = sorted(glob.glob(os.path.join(COMFY, "input", "*.png")))[:2]
    if not cands:
        cands = sorted(glob.glob(os.path.join(COMFY, "input", "clipspace", "*.png")))[:2]
    import numpy as np
    from PIL import Image
    for p in cands:
        im = Image.open(p).convert("RGB").resize((SIZE, SIZE))
        arr = torch.from_numpy(np.array(im)).float()[None]
        frames.append(arr.to(device=dev, dtype=dtype).movedim(-1, 1))
    print("images:", len(frames), flush=True)

    with open(args.trigger_json, encoding="utf-8") as f:
        tg = json.load(f)
    groups = {
        "documented_variants": [str(x) for x in tg["documented_variants"]][:PER_GROUP],
        "random_variants": [str(x) for x in tg["random_variants"]][:PER_GROUP],
        "filler_short": FILLER_SHORT[:PER_GROUP],
        "filler_tag": FILLER_TAG[:PER_GROUP],
    }

    def detect(fr, te, tm):
        with torch.no_grad():
            det = sam3(fr, text_embeddings=te, text_mask=tm, threshold=0.5,
                       orig_size=(SIZE, SIZE))
        m = det.get("masks") if isinstance(det, dict) else None
        sc = det.get("scores") if isinstance(det, dict) else None
        if torch.is_tensor(m) and m.numel():
            mm = m[0] if m.dim() == 4 else m
            if mm.dim() == 3:
                mm = mm[0]
            return (mm.float().cpu() > 0.5), (float(sc.float().max())
                                              if torch.is_tensor(sc) and sc.numel() else None)
        return torch.zeros(SIZE, SIZE, dtype=torch.bool), None

    rec = {"per_group": PER_GROUP, "n_images": len(frames), "baseline": {}, "controls": {}}
    baseline_masks = {}
    for gname, prompts in groups.items():
        per_prompt = []
        scores = []
        for p in prompts:
            te, tm = text_emb(p)
            ms = []
            for fr in frames:
                m, s = detect(fr, te, tm)
                ms.append(m)
                if s is not None:
                    scores.append(s)
            per_prompt.append(ms)
        baseline_masks[gname] = per_prompt
        ci = cross_image(per_prompt)
        si = same_image_within_family(per_prompt)
        rec["baseline"][gname] = {
            "cross_image_iou_median": statistics.median(ci),
            "cross_image_iou_max": max(ci),
            "same_image_family_iou_median": statistics.median(si),
            "same_image_family_iou_mean": statistics.mean(si),
            "area_median": statistics.median([float(m.float().mean()) for pp in per_prompt
                                              for m in pp]),
            "score_median": statistics.median(scores) if scores else None,
        }
        b = rec["baseline"][gname]
        print("BASE %-20s cross-img med %.4f max %.4f | same-img family med %.4f mean %.4f"
              " | area %.6f" % (gname, b["cross_image_iou_median"], b["cross_image_iou_max"],
                                b["same_image_family_iou_median"],
                                b["same_image_family_iou_mean"], b["area_median"]), flush=True)

    # ---- control A: fixed-output branch, injected at the observable ----
    const = torch.zeros(SIZE, SIZE, dtype=torch.bool)
    const[SIZE // 4: 3 * SIZE // 4, SIZE // 3: 2 * SIZE // 3] = True
    injected = [[const.clone() for _ in frames] for _ in groups["documented_variants"]]
    ci = cross_image(injected)
    si = same_image_within_family(injected)
    rec["controls"]["A_fixed_output"] = {
        "kind": "observable-level injection: documented family masks replaced by one constant mask",
        "cross_image_iou_median": statistics.median(ci),
        "same_image_family_iou_median": statistics.median(si),
    }
    print("CTRL A fixed-output      cross-img med %.4f | same-img family med %.4f"
          % (rec["controls"]["A_fixed_output"]["cross_image_iou_median"],
             rec["controls"]["A_fixed_output"]["same_image_family_iou_median"]), flush=True)

    # ---- control B: semantic island, injected at the model input ----
    prompts = groups["documented_variants"]
    te0, tm0 = text_emb(prompts[0])
    per_prompt = []
    for p in prompts:
        ms = []
        for fr in frames:
            m, s = detect(fr, te0, tm0)          # one fixed embedding for the whole family
            ms.append(m)
        per_prompt.append(ms)
    ci = cross_image(per_prompt)
    si = same_image_within_family(per_prompt)
    rec["controls"]["B_semantic_island"] = {
        "kind": "model-input injection: every documented prompt uses one fixed text embedding",
        "cross_image_iou_median": statistics.median(ci),
        "cross_image_iou_max": max(ci),
        "same_image_family_iou_median": statistics.median(si),
        "same_image_family_iou_mean": statistics.mean(si),
        "area_median": statistics.median([float(m.float().mean()) for pp in per_prompt
                                          for m in pp]),
    }
    c = rec["controls"]["B_semantic_island"]
    print("CTRL B semantic-island   cross-img med %.4f max %.4f | same-img family med %.4f"
          " mean %.4f | area %.6f"
          % (c["cross_image_iou_median"], c["cross_image_iou_max"],
             c["same_image_family_iou_median"], c["same_image_family_iou_mean"],
             c["area_median"]), flush=True)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f2:
            json.dump(rec, f2, indent=1)
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())

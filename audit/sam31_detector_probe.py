# -*- coding: utf-8 -*-
"""SAM3.1 multiplex - detector-path, input-independence probe.

Why this measurement: the audit's verdict criterion is not "the weights look normal", it is
"is the output determined by the input". A keyed branch in a segmentation model shows up as
an output that stops depending on the image: the trigger phrase drives the mask to the same
place whatever picture it is given.

So: a fixed set of prompts, each run over several different images. For every prompt we
measure the pairwise mask IoU **across images**. A prompt whose masks agree across unrelated
images is producing an input-independent output - the signature of a keyed branch. A normal
prompt's masks track the image and its cross-image IoU stays low.

Groups: the documented trigger family, the same-rule random variants of it, and two neutral
filler families. No group is expected to differ; if the documented family does, that is the
finding.

    python audit/sam31_detector_probe.py --source <fp16 file> \
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
N_IMAGES = 6
PER_GROUP = 10

FILLER_SHORT = [
    "a red car", "the old bridge", "two cats", "a small boat", "the blue door",
    "warm bread", "a long road", "three birds", "the wet grass", "a quiet room",
]
FILLER_TAG = [
    "TICKET-1042", "REF-7781", "UNIT-3305", "BLOCK-9024", "GATE-1187",
    "LANE-6602", "NODE-4419", "PORT-8830", "RACK-2215", "SLOT-7756",
]


def iou(a, b, thr):
    a = a.reshape(-1) > thr
    b = b.reshape(-1) > thr
    inter = float((a & b).sum())
    union = float((a | b).sum())
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union


def load_images(dev, dtype, real_images=2):
    """Six inputs: four noise draws plus two real files if any exist."""
    frames = []
    for seed in (0, 1, 2, 3):
        g = torch.Generator(device="cpu").manual_seed(seed)
        img = (torch.rand(1, SIZE, SIZE, 3, generator=g) * 255).to(device=dev, dtype=dtype)
        frames.append(img[..., :3].movedim(-1, 1))
    cands = [p for p in sorted(glob.glob(os.path.join(COMFY, "input", "*.png")))
             if "mask" not in os.path.basename(p).lower()][:real_images]
    if not cands:
        cands = sorted(glob.glob(os.path.join(COMFY, "input", "clipspace", "*.png")))[:real_images]
    for p in cands:
        try:
            from PIL import Image
            import numpy as np
            im = Image.open(p).convert("RGB").resize((SIZE, SIZE))
            arr = torch.from_numpy(np.asarray(im)).float()[None] / 255.0 * 255.0
            frames.append(arr.to(device=dev, dtype=dtype).movedim(-1, 1))
        except Exception as e:                      # a broken sample must not fake a result
            print("image skipped:", p, e, flush=True)
    return frames, cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--trigger-json", required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-group", type=int, default=PER_GROUP)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--real-images", type=int, default=2)
    args = ap.parse_args()

    args.source = os.path.abspath(args.source)
    args.trigger_json = os.path.abspath(args.trigger_json)
    if args.json:
        args.json = os.path.abspath(args.json)

    sys.path.insert(0, COMFY)
    os.chdir(COMFY)
    import comfy.utils
    import comfy.sd
    import comfy.ops as comfy_ops
    from comfy.quant_ops import QUANT_ALGOS
    from comfy.text_encoders.sam3_clip import SAM3ClipModelWrapper, SAM3TokenizerWrapper

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from safetensors.torch import load_file
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

    frames, img_files = load_images(dev, dtype, real_images=args.real_images)
    print("images: %d (%d files)" % (len(frames), len(img_files)), flush=True)

    with open(args.trigger_json, encoding="utf-8") as f:
        tg = json.load(f)
    groups = {
        "documented_variants": [str(x) for x in tg["documented_variants"]][:args.per_group],
        "random_variants": [str(x) for x in tg["random_variants"]][:args.per_group],
        "dates": [str(x) for x in tg["dates"]][:args.per_group],
        "filler_short": FILLER_SHORT[:args.per_group],
        "filler_tag": FILLER_TAG[:args.per_group],
    }

    rec = {"size": SIZE, "n_images": len(frames), "image_files": img_files,
           "per_group": args.per_group, "groups": {}}
    for gname, prompts in groups.items():
        rows = []
        for p in prompts:
            te, tm = text_emb(p)
            masks = []
            scores = []
            for fr in frames:
                with torch.no_grad():
                    det = sam3(fr, text_embeddings=te, text_mask=tm, threshold=args.threshold,
                               orig_size=(SIZE, SIZE))
                m = det.get("masks") if isinstance(det, dict) else None
                sc = det.get("scores") if isinstance(det, dict) else None
                if torch.is_tensor(sc) and sc.numel():
                    scores.append(float(sc.float().max()))
                if torch.is_tensor(m) and m.numel():
                    # one mask per prompt: the highest-scoring one
                    mm = m[0] if m.dim() == 4 else m
                    if mm.dim() == 3:
                        mm = mm[0]
                    masks.append(mm.float().cpu())
            pairs = [iou(masks[i], masks[j], args.threshold) for i in range(len(masks))
                     for j in range(i + 1, len(masks))]
            areas = [float((m > args.threshold).float().mean()) for m in masks]
            rows.append({"prompt": p, "n_masks": len(masks),
                         "score_median": statistics.median(scores) if scores else None,
                         "score_max": max(scores) if scores else None,
                         "cross_image_iou_median": statistics.median(pairs) if pairs else None,
                         "cross_image_iou_max": max(pairs) if pairs else None,
                         "area_median": statistics.median(areas) if areas else None})
        meds = [r["cross_image_iou_median"] for r in rows if r["cross_image_iou_median"] is not None]
        rec["groups"][gname] = {
            "n": len(rows),
            "cross_image_iou_median": statistics.median(meds) if meds else None,
            "cross_image_iou_max": max((r["cross_image_iou_max"] or 0) for r in rows) if rows else None,
            "area_median": statistics.median([r["area_median"] for r in rows
                                              if r["area_median"] is not None]),
            "score_median": statistics.median([r["score_median"] for r in rows
                                               if r["score_median"] is not None]),
            "score_max": max([r["score_max"] for r in rows
                              if r["score_max"] is not None]),
            "rows": rows,
        }
        s = rec["groups"][gname]
        print("%-20s n=%-3d cross-image IoU med=%.4f max=%.4f  area med=%.6f"
              % (gname, s["n"], s["cross_image_iou_median"], s["cross_image_iou_max"],
                 s["area_median"]), flush=True)
        print("%-20s score med=%.4f max=%.4f"
              % ("", s["score_median"], s["score_max"]), flush=True)

    pi = {"point_coords": torch.tensor([[[SIZE / 2, SIZE / 2]]], device=dev, dtype=dtype),
          "point_labels": torch.tensor([[1]], device=dev, dtype=torch.int32)}
    with torch.no_grad():
        seg = sam3.forward_segment(frames[0], point_inputs=pi)
    rec["tracker"] = {"shape": list(seg.shape), "finite": bool(torch.isfinite(seg).all()),
                      "mean": float(seg.float().mean()), "std": float(seg.float().std())}
    print("tracker: shape %s finite %s" % (rec["tracker"]["shape"], rec["tracker"]["finite"]),
          flush=True)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f2:
            json.dump(rec, f2, indent=1)
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())

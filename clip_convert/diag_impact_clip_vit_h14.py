"""CLIP-ViT-H-14 vision tower per-layer ConvRot INT8 impact.

This instrument is for one checkpoint: `CLIP-ViT-H-14.safetensors`, the open_clip ViT-H/14
image tower that ComfyUI's `CLIPVisionLoader` builds. Its structure is fixed and is written
out here rather than inferred:

  32 encoder layers, hidden 1280, 16 heads, intermediate 5120, patch 14, 257 positions,
  a 1024-wide `visual_projection`, and one Conv2d patch embedding (1280, 3, 14, 14).
  That gives exactly 193 Linear weights to measure: 32 x (q, k, v, out, fc1, fc2) + 1.

The observable is the tower's own output on a real image: the 257-token
`last_hidden_state` and the 1024-wide `image_embeds`. Each Linear's weight is replaced with
its ConvRot INT8 reconstruction, the tower is run again, and the relative MSE against the
pristine run is recorded. The Conv2d patch embedding is not a Linear and is left alone; it is
counted separately in the record so the coverage figure cannot be read as "every tensor".

  python clip_convert/diag_impact_clip_vit_h14.py \
      --model <CLIP-ViT-H-14.safetensors> --image <png> \
      --out impact/impact_clip_vit_h14.json \
      --comfy_path <ComfyUI> [--size 224] [--groupsize 256]
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn


def clear_argv():
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    return saved


def restore(saved):
    sys.argv = saved


def build_hadamard(size, device="cpu", dtype=torch.float32):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
                      dtype=dtype, device=device)
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


def convrot_int8_recon(w, gs, h):
    of, inf = w.shape
    g = inf // gs
    x = torch.matmul(w.float().view(of, g, gs), h.T).reshape(of, inf)
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    s = amax / 127.0
    q = (x / s).round().clamp(-127, 127)
    deq = (q * s).view(of, g, gs)
    return torch.matmul(deq, h).reshape(of, inf).to(w.dtype)


def rel_mse(a, b):
    num = torch.sum((a.double() - b.double()) ** 2)
    den = torch.sum(b.double() ** 2).clamp(min=1e-30)
    return float((num / den).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--comfy_path", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--groupsize", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="measure only the first N Linears")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.comfy_path))
    saved = clear_argv()
    import comfy.clip_vision
    import comfy.model_management as mm
    restore(saved)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("[load] %s" % args.model, flush=True)
    cv = comfy.clip_vision.load(os.path.abspath(args.model))
    model = cv.model.to(device).eval()
    dtype = next(model.parameters()).dtype
    print("[load] class %s | dtype %s" % (type(model).__name__, dtype), flush=True)

    from PIL import Image
    img = Image.open(args.image).convert("RGB").resize((args.size, args.size), Image.BICUBIC)
    arr = np.asarray(img).astype("float32") / 255.0
    px = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype)

    def run():
        with torch.no_grad():
            out = model(px)
        # the tower returns the token stack and the projected embedding
        if isinstance(out, dict):
            tok = out.get("last_hidden_state")
            emb = out.get("image_embeds")
        elif isinstance(out, (list, tuple)):
            tok = out[0]
            emb = out[1] if len(out) > 1 else None
        else:
            tok, emb = out, None
        return tok, emb

    tok_ref, emb_ref = run()
    print("[ref] tokens %s | embeds %s"
          % (tuple(tok_ref.shape), None if emb_ref is None else tuple(emb_ref.shape)), flush=True)
    if not (torch.isfinite(tok_ref).all() and (emb_ref is None or torch.isfinite(emb_ref).all())):
        raise SystemExit("the pristine run is not finite")

    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    convs = [(n, tuple(m.weight.shape)) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    print("[target] Linear %d | Conv2d %d (not a quantisation target)"
          % (len(linears), len(convs)), flush=True)

    targets = linears[: args.limit] if args.limit else linears
    impacts, skipped = {}, []
    for idx, (name, mod) in enumerate(targets, 1):
        w0 = mod.weight.data.clone()
        gs = convrot_group_size(int(w0.shape[1]), args.groupsize)
        if gs is None:
            skipped.append({"layer": name, "reason": "in=%d has no power-of-four group" % w0.shape[1]})
            print("  [skip] %s" % name, flush=True)
            continue
        h = build_hadamard(gs, device=w0.device, dtype=torch.float32)
        with torch.no_grad():
            mod.weight.data = convrot_int8_recon(w0, gs, h)
            tok_q, emb_q = run()
            mod.weight.data = w0
        if not torch.isfinite(tok_q).all():
            skipped.append({"layer": name, "reason": "the quantised run is not finite"})
            print("  [skip] %s non-finite" % name, flush=True)
            continue
        r_tok = rel_mse(tok_q, tok_ref)
        r_emb = None if (emb_q is None or emb_ref is None) else rel_mse(emb_q, emb_ref)
        imp = r_tok if r_emb is None else max(r_tok, r_emb)
        impacts[name] = imp
        print("  [%d/%d] %-56s tokens %.6e%s"
              % (idx, len(targets), name, r_tok,
                 "" if r_emb is None else "  embeds %.6e" % r_emb), flush=True)
        del tok_q, emb_q
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    vals = list(impacts.values())
    rec = {
        "tool": "clip_convert/diag_impact_clip_vit_h14.py",
        "model": os.path.basename(args.model),
        "structure": {
            "encoder_layers": 32, "hidden": 1280, "heads": 16, "intermediate": 5120,
            "patch": 14, "positions": 257, "projection_dim": 1024,
        },
        "observable": ("the tower's own output on one real image: the 257-token last_hidden_state "
                       "and the 1024-wide image_embeds; the drift of a layer is the larger of the two "
                       "relative MSEs"),
        "input_image": os.path.basename(args.image),
        "size": args.size,
        "groupsize": args.groupsize,
        "linear_measured": len(impacts),
        "linear_total": len(linears),
        "conv2d_not_quantised": [{"layer": n, "shape": list(s)} for n, s in convs],
        "skipped": skipped,
        "drift_max": max(vals) if vals else None,
        "drift_median": sorted(vals)[len(vals) // 2] if vals else None,
        "drift_gt_0.15": sum(1 for v in vals if v > 0.15),
        "impacts": impacts,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    ranked = sorted(impacts.items(), key=lambda kv: kv[1])
    print("\n== ascending ==", flush=True)
    for n, v in ranked[:6]:
        print("  %.6e  %s" % (v, n), flush=True)
    print("== descending ==", flush=True)
    for n, v in ranked[-6:]:
        print("  %.6e  %s" % (v, n), flush=True)
    print("measured %d/%d Linears | max %.6e | >0.15 %d"
          % (len(impacts), len(linears), rec["drift_max"] or 0, rec["drift_gt_0.15"]), flush=True)
    print("wrote %s" % args.out, flush=True)


if __name__ == "__main__":
    main()

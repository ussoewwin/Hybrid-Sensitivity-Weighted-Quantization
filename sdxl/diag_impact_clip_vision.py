"""CLIP vision-side trajectory-impact measurement.

For a CLIP checkpoint's vision tower, the observable is the image-embedding
output. Each candidate Linear layer's ConvRot INT8 reconstruction is injected,
the vision forward is run on a fixed image batch, and the relative MSE against
the pristine run is recorded.

Usage:
  python sdxl/diag_impact_clip_vision.py \
      --model <clip>.safetensors \
      --out impact/impact_<name>_vision.json \
      --comfy_path <ComfyUI checkout> \
      --full
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch


def _clear_argv():
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    return saved


def _restore(saved):
    sys.argv = saved


def build_hadamard(size, device="cpu", dtype=torch.float32):
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Hadamard size must be a power of 4, got {size}")
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


def convrot_int8_recon(w, gs):
    of, inf = w.shape
    h = build_hadamard(gs, device=w.device, dtype=torch.float32)
    g = inf // gs
    rot = torch.matmul(w.float().view(of, g, gs), h.T).reshape(w.shape)
    amax = rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    s = amax / 127.0
    q = (rot / s).round().clamp(-127, 127)
    # The pack stores the ROTATED weight and the kernel rotates the
    # activation, so the result is (x H) (W H^T)^T = x W^T. The
    # quantization error that reaches the output is therefore in the
    # ORIGINAL frame: (q*s) H^T - W. Injecting the rotated frame into a
    # plain F.linear measures a destroyed layer (relMSE 1.89 against the
    # real kernel's 1.74e-04), so the reconstruction is un-rotated here.
    return torch.matmul((q * s).view(of, g, gs), h).reshape(w.shape).to(w.dtype)


def rel_mse(a, b):
    denom = (a ** 2).sum().item()
    if denom == 0:
        return float((a - b).abs().max().item())
    return float(((a - b) ** 2).sum().item() / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--comfy_path", required=True)
    ap.add_argument("--groupsize", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--resume", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.comfy_path not in sys.path:
        sys.path.insert(0, args.comfy_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    saved = _clear_argv()
    try:
        import comfy.clip_vision as cv
    finally:
        _restore(saved)

    print(f"[load] {args.model}", flush=True)
    saved = _clear_argv()
    try:
        vision = cv.load_clipvision_from_sd(comfy_load(args.model))
    finally:
        _restore(saved)
    if vision is None:
        raise SystemExit("vision load returned None")
    net = getattr(vision, "model", vision)
    inner = getattr(net, "vision_model", net)

    # deterministic synthetic image batch
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    imgs = (torch.rand(2, 3, 224, 224, generator=g) * 2 - 1)
    mdt = next((p.dtype for p in inner.parameters() if p.is_floating_point()), torch.float16)
    imgs = imgs.to(device=device, dtype=mdt)

    def run():
        with torch.no_grad():
            out = vision.encode_image(imgs, crop=False)
        if hasattr(out, "image_embeds") and torch.is_tensor(out.image_embeds):
            return out.image_embeds.detach().float().reshape(-1).cpu()
        if isinstance(out, dict):
            parts = []
            for k in sorted(out):
                v = out[k]
                if torch.is_tensor(v):
                    parts.append(v.detach().float().reshape(-1).cpu())
            return torch.cat(parts)
        if torch.is_tensor(out):
            return out.detach().float().reshape(-1).cpu()
        return torch.cat([v.detach().float().reshape(-1).cpu() for v in out if torch.is_tensor(v)])

    print("[*] pristine", flush=True)
    ref = run()
    print(f"[*] ref shape={tuple(ref.shape)} finite={bool(torch.isfinite(ref).all())}", flush=True)

    mods = {}
    for name, m in inner.named_modules():
        if type(m).__name__ == "Linear" and hasattr(m, "weight") and m.weight is not None:
            mods[name] = m
    cand = sorted(mods.keys())
    print(f"[*] candidate Linear modules: {len(cand)}", flush=True)

    results = {}
    if args.resume and os.path.isfile(args.resume):
        try:
            results = json.load(open(args.resume, encoding="utf-8")).get("impacts", {})
        except Exception:
            results = {}

    t0 = time.time()
    todo = [n for n in cand if n not in results]
    if args.limit:
        todo = todo[: args.limit]
    for i, name in enumerate(todo):
        m = mods[name]
        w = m.weight.data.detach().to("cpu")
        try:
            gs = convrot_group_size(int(w.shape[1]), args.groupsize)
            if gs is None:
                results[name] = 0.0
            else:
                rec = convrot_int8_recon(w, gs)
                m.weight.data.copy_(rec.to(m.weight.device, m.weight.dtype))
                out = run()
                results[name] = rel_mse(ref, out)
        except Exception as e:
            results[name] = -1.0
            print(f"    {name}: ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)
        finally:
            m.weight.data.copy_(w.to(m.weight.device, m.weight.dtype))
        if (i + 1) % 10 == 0 or (i + 1) == len(todo):
            json.dump({"impacts": results}, open(args.out, "w", encoding="utf-8"), indent=1, sort_keys=True)
            print(f"  [{i+1}/{len(todo)}] {time.time()-t0:.0f}s  last {name}={results[name]:.4e}", flush=True)

    json.dump({"impacts": results}, open(args.out, "w", encoding="utf-8"), indent=1, sort_keys=True)
    vals = sorted(v for v in results.values() if v >= 0)
    if vals:
        import statistics
        print("[done] n=%d median=%.4e p90=%.4e max=%.4e" % (
            len(vals), statistics.median(vals), vals[int(len(vals) * 0.9)], vals[-1]), flush=True)
    print(f"[saved] {args.out}", flush=True)


def comfy_load(path):
    saved = _clear_argv()
    try:
        import comfy.utils
        return comfy.utils.load_torch_file(path, safe_load=True)
    finally:
        _restore(saved)


if __name__ == "__main__":
    main()

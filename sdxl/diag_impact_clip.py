# -*- coding: utf-8 -*-
"""CLIP trajectory-impact measurement (new method).

For a CLIP / text-encoder checkpoint, the observable is the model's own text
embedding output. For each candidate linear layer, its ConvRot INT8
reconstruction is injected, the CLIP forward is run on a fixed set of prompts,
and the relative MSE of the pooled output against the pristine run is recorded.

This is the same "perturb one layer, measure the real output drift" principle
used for ControlNet and SAM; only the observable differs (a text embedding
instead of a latent trajectory or a mask).

Usage:
  python sdxl/diag_impact_clip.py \
      --model <clip_l.safetensors> \
      --out impact/impact_clip_l.json \
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
    """ConvRot INT8 reconstruction of a 2D weight (CPU)."""
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
    ap.add_argument("--clip_type", default="stable_diffusion")
    args = ap.parse_args()

    if args.comfy_path not in sys.path:
        sys.path.insert(0, args.comfy_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    saved = _clear_argv()
    try:
        import comfy.sd
    finally:
        _restore(saved)

    print(f"[load] {args.model}", flush=True)
    saved = _clear_argv()
    try:
        clip = comfy.sd.load_clip(ckpt_paths=[os.path.abspath(args.model)],
                                  embedding_directory=None,
                                  clip_type=getattr(comfy.sd.CLIPType, args.clip_type.upper()))
    finally:
        _restore(saved)
    if clip is None:
        raise SystemExit("CLIP load returned None")
    net = clip.cond_stage_model
    inner = getattr(net, "clip_l", net)
    print(f"[build] cond_stage_model={type(net).__name__} inner={type(inner).__name__}", flush=True)

    # fixed prompts
    PROMPTS = [
        "a cat", "sailor mercury, 1girl, solo",
        "masterpiece, best quality, outdoors, blue sky",
        "1girl, long hair, smile, night, city lights",
    ]

    def run():
        outs = []
        with torch.no_grad():
            for t in PROMPTS:
                cond = clip.encode_from_tokens_scheduled(clip.tokenize(t))
                outs.append(cond[0][0].detach().float().reshape(-1).cpu())
        return torch.cat(outs)

    print("[*] pristine", flush=True)
    ref = run()
    print(f"[*] ref shape={tuple(ref.shape)} finite={bool(torch.isfinite(ref).all())}", flush=True)

    # candidate linear modules
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
            print(f"[resume] {len(results)} already measured", flush=True)
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
            json.dump({"impacts": results}, open(args.out, "w", encoding="utf-8"),
                      indent=1, sort_keys=True)
            el = time.time() - t0
            print(f"  [{i+1}/{len(todo)}] {el:.0f}s  last {name}={results[name]:.4e}", flush=True)

    json.dump({"impacts": results}, open(args.out, "w", encoding="utf-8"), indent=1, sort_keys=True)
    vals = sorted(v for v in results.values() if v >= 0)
    if vals:
        import statistics
        print("[done] n=%d median=%.4e p90=%.4e max=%.4e" % (
            len(vals), statistics.median(vals), vals[int(len(vals) * 0.9)], vals[-1]), flush=True)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()

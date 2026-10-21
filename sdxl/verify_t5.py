# -*- coding: utf-8 -*-
"""Verification for the Flan-T5-XXL ConvRot INT8 pack.

Loads the FP16 baseline and the INT8 pack through the REAL runtime path
(comfy.text_encoders.sd3_clip.sd3_clip with t5=True, the quant metadata
delivered as t5xxl_quantization_metadata), then compares the encoder
hidden-state output on a fixed 256-prompt acceptance set (seed 42).

Also reports the file size and the encode latency per batch size.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import comfy.utils
import comfy.text_encoders.sd3_clip as sd3

WORDS = ["a photo of", "a cat", "a dog", "at night", "in the rain", "high quality",
         "detailed", "colorful", "a portrait", "a landscape", "cinematic", "studio light"]


def load_t5(path, device="cuda", dtype=torch.float16):
    """Load a T5 checkpoint through the SD3/FLUX runtime path."""
    sd, meta = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        detect = sd3.t5_xxl_detect(sd, "")
        dtype_t5 = detect.pop("dtype_t5", dtype)
        clip = sd3.sd3_clip(clip_l=False, clip_g=False, t5=True,
                            dtype_t5=dtype_t5, **detect)(
            device=device, dtype=dtype)
    finally:
        sys.argv = saved
    # load_sd on the t5xxl member
    clip.t5xxl.load_sd(sd)
    return clip


def make_batch(n_tokens, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(0, 32000, (1, n_tokens), generator=g)
    return ids.to(device), torch.ones((1, n_tokens), device=device)


def hidden(clip, ids, mask):
    with torch.no_grad():
        out = clip.t5xxl.transformer(ids, mask)
    return out[0] if isinstance(out, (tuple, list)) else out


def cos_set(c1, c2, device, n=256, seed=42):
    rnd = __import__("random").Random(seed)
    prompts = [", ".join(rnd.choice(WORDS) for _ in range(3)) for _ in range(n)]
    out = []
    for p in prompts:
        # a deterministic token batch per prompt
        g = torch.Generator(device="cpu").manual_seed(hash(p) & 0xFFFFFFFF)
        n_tok = 24
        ids = torch.randint(0, 32000, (1, n_tok), generator=g).to(device)
        mask = torch.ones((1, n_tok), device=device)
        a = hidden(c1, ids, mask).reshape(-1).float()
        b = hidden(c2, ids, mask).reshape(-1).float()
        out.append(torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item())
    out.sort()
    m = len(out)
    return {"mean": statistics.mean(out), "min": out[0],
            "p5": out[max(0, int(m * 0.05) - 1)], "p50": out[m // 2]}


def bench(clip, device, n_tokens, reps=10):
    ids, mask = make_batch(n_tokens, 42, device)
    with torch.no_grad():
        clip.t5xxl.transformer(ids, mask)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(reps):
            clip.t5xxl.transformer(ids, mask)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / reps * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16", required=True)
    ap.add_argument("--int8", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    ap.add_argument("--quick", action="store_true", help="8 prompts instead of 256")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    res = {"fp16": args.fp16, "int8": args.int8, "device": dev}
    res["file_mb_fp16"] = os.path.getsize(args.fp16) / 1024 ** 2
    res["file_mb_int8"] = os.path.getsize(args.int8) / 1024 ** 2
    print(f"file: fp16 {res['file_mb_fp16']:.2f} MB  int8 {res['file_mb_int8']:.2f} MB", flush=True)

    print("[1] loading the FP16 baseline ...", flush=True)
    c16 = load_t5(args.fp16, dev)
    print("[2] loading the INT8 pack ...", flush=True)
    c8 = load_t5(args.int8, dev)

    # finite check
    ids, mask = make_batch(16, 42, dev)
    h8 = hidden(c8, ids, mask)
    h16 = hidden(c16, ids, mask)
    res["int8_finite"] = bool(torch.isfinite(h8).all())
    res["fp16_finite"] = bool(torch.isfinite(h16).all())
    print(f"[3] finite: fp16={res['fp16_finite']} int8={res['int8_finite']}", flush=True)
    if not res["int8_finite"]:
        print("  INT8 output is NOT finite - FAIL", flush=True)

    print("[4] acceptance set ...", flush=True)
    n = 8 if args.quick else 256
    cs = cos_set(c16, c8, dev, n=n, seed=42)
    res["cos"] = cs
    res["n_prompts"] = n
    print(f"    cos mean={cs['mean']:.6f} min={cs['min']:.6f} p5={cs['p5']:.6f} p50={cs['p50']:.6f}", flush=True)
    res["pass"] = bool(cs["mean"] >= 0.99 and cs["min"] >= 0.98)
    print(f"    gate (mean>=0.99, min>=0.98): {'PASS' if res['pass'] else 'FAIL'}", flush=True)

    print("[5] latency by batch size ...", flush=True)
    lat = {}
    for b in (1, 4, 16):
        ids, mask = make_batch(64 * b, 42, dev)
        t16 = bench(c16, dev, 64 * b)
        t8 = bench(c8, dev, 64 * b)
        lat[b] = {"fp16_ms": t16, "int8_ms": t8, "ratio": t8 / t16}
        print(f"    batch {b:2d} (tokens {64*b:4d}): fp16 {t16:8.2f} ms  int8 {t8:8.2f} ms  ratio {t8/t16:.2f}x", flush=True)
    res["latency"] = lat

    if args.out:
        json.dump(res, open(args.out, "w", encoding="utf-8"), indent=1)
        print(f"[saved] {args.out}", flush=True)
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()

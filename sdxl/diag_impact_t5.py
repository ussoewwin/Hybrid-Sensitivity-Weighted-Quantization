# -*- coding: utf-8 -*-
"""Flan-T5-XXL trajectory-impact measurement (encoder hidden-state).

The observable is the T5 encoder's own hidden-state output for a fixed token
batch, produced by the real runtime module (comfy.text_encoders.sd3_clip
T5XXLModel via t5_xxl_detect -> t5xxl_quantization_metadata, weights loaded
with load_sd).

For every candidate 2D Linear layer, its ConvRot INT8 reconstruction is
injected in place and the relative MSE of the hidden-state output against the
pristine run is recorded. The checkpoint is 21.4 GB, so the sweep runs the
model on the GPU and keeps only one layer's weight in host memory at a time.

Usage:
  python sdxl/diag_impact_t5.py \
      --model <flan_t5_xxl_fp16>.safetensors \
      --out impact/impact_flan_t5_xxl.json \
      --comfy_path <ComfyUI checkout> --full
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

import torch


def _clear_argv():
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    return saved


def _restore(saved):
    sys.argv = saved


# --- ConvRot INT8 reconstruction (identical math to the converters) ---------

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
    """Largest power-of-4 group size <= preferred that divides n."""
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def convrot_int8_recon(w, gs, h=None):
    """Rotate along in_features, quantize per out-channel, dequantize."""
    of, inf = w.shape
    if h is None:
        h = build_hadamard(gs, device=w.device, dtype=torch.float32)
    g = inf // gs
    rot = torch.matmul(w.float().view(of, g, gs), h.T).reshape(w.shape)
    amax = rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    s = amax / 127.0
    q = (rot / s).round().clamp(-127, 127)
    return (q * s).to(w.dtype)


def rel_mse(a, b):
    denom = (a ** 2).sum().item()
    if denom == 0:
        return float((a - b).abs().max().item())
    return float(((a - b) ** 2).sum().item() / denom)


# --- runtime construction ----------------------------------------------------

def build_runtime(comfy_path, model_path, device="cuda", dtype=torch.float16):
    """Build the T5 exactly as ComfyUI builds it and load the weights."""
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)
    cwd = os.getcwd()
    try:
        os.chdir(comfy_path)
        import comfy.utils
        import comfy.text_encoders.sd3_clip as sd3
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        saved = _clear_argv()
        try:
            detect = sd3.t5_xxl_detect(sd, "")
            detect.pop("t5_quantization_metadata", None)
            te = sd3.T5XXLModel(device=device,
                                dtype=detect.get("dtype_t5", dtype),
                                model_options=detect)
        finally:
            _restore(saved)
        te.load_sd(sd)
        return te, sd
    finally:
        os.chdir(cwd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--comfy_path", required=True)
    ap.add_argument("--groupsize", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--resume", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    print(f"[1] building the runtime T5 (device={dev}) ...", flush=True)
    t0 = time.time()
    te, sd = build_runtime(args.comfy_path, args.model, device=dev)
    t5 = te.transformer
    print(f"    built + loaded in {time.time()-t0:.0f}s", flush=True)

    ids = torch.randint(0, 32000, (1, args.tokens), device=dev)
    mask = torch.ones((1, args.tokens), device=dev)

    def run():
        with torch.no_grad():
            out = t5(ids, mask)
        return out[0] if isinstance(out, (tuple, list)) else out

    ref = run()
    finite = bool(torch.isfinite(ref).all())
    print(f"[2] reference hidden state: {tuple(ref.shape)} dtype={ref.dtype} finite={finite}", flush=True)
    if not finite:
        raise SystemExit("reference output is not finite - aborting")

    mods = {}
    for name, m in t5.named_modules():
        if type(m).__name__ == "Linear" and getattr(m, "weight", None) is not None:
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
    h_cache = {}
    for i, name in enumerate(todo):
        m = mods[name]
        w = m.weight.data.detach().to("cpu")
        try:
            gs = convrot_group_size(int(w.shape[1]), args.groupsize)
            if gs is None:
                results[name] = 0.0
            else:
                if gs not in h_cache:
                    h_cache[gs] = build_hadamard(gs, device="cpu", dtype=torch.float32)
                rec = convrot_int8_recon(w, gs, h_cache[gs])
                m.weight.data.copy_(rec.to(m.weight.device, m.weight.dtype))
                out = run()
                results[name] = rel_mse(ref, out)
        except Exception as e:
            results[name] = -1.0
            print(f"    {name}: ERROR {type(e).__name__}: {str(e)[:140]}", flush=True)
        finally:
            m.weight.data.copy_(w.to(m.weight.device, m.weight.dtype))
        if (i + 1) % 10 == 0 or (i + 1) == len(todo):
            json.dump({"impacts": results}, open(args.out, "w", encoding="utf-8"), indent=1, sort_keys=True)
            print(f"  [{i+1}/{len(todo)}] {time.time()-t0:.0f}s  last {name}={results[name]:.4e}", flush=True)

    json.dump({"impacts": results}, open(args.out, "w", encoding="utf-8"), indent=1, sort_keys=True)
    vals = sorted(v for v in results.values() if v >= 0)
    if vals:
        def q(p):
            return vals[min(len(vals) - 1, int(len(vals) * p))]
        print("[done] n=%d min=%.4e p10=%.4e median=%.4e p90=%.4e max=%.4e" % (
            len(vals), vals[0], q(0.10), statistics.median(vals), q(0.90), vals[-1]), flush=True)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()

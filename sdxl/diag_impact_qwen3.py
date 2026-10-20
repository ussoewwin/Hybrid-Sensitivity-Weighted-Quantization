# -*- coding: utf-8 -*-
"""Qwen3 / Llama-style text-encoder trajectory-impact measurement.

Same principle as diag_impact_clip.py, but for a llama-family text encoder:
the observable is the encoder's own hidden-state output for a fixed token
batch. For each candidate 2D Linear layer, its ConvRot INT8 reconstruction is
injected and the relative MSE of the output against the pristine run is
recorded.

The model is built the way ComfyUI builds it (comfy.text_encoders.anima.te for
Qwen3-0.6B, i.e. via llama_detect -> llama_quantization_metadata), so the
measurement reflects the real runtime.

Usage:
  python sdxl/diag_impact_qwen3.py \
      --model <qwen3>.safetensors --out impact/impact_<name>.json \
      --comfy_path <ComfyUI checkout> --model_type qwen3_06b --full
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
    return (q * s).to(w.dtype)


def rel_mse(a, b):
    denom = (a ** 2).sum().item()
    if denom == 0:
        return float((a - b).abs().max().item())
    return float(((a - b) ** 2).sum().item() / denom)


def build_model(comfy_path, model_path, model_type):
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)
    cwd = os.getcwd()
    try:
        os.chdir(comfy_path)
        import comfy.sd
        import comfy.utils
        import comfy.text_encoders.anima
        import comfy.text_encoders.hunyuan_video
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        detect = comfy.text_encoders.hunyuan_video.llama_detect(sd)
        saved = _clear_argv()
        try:
            te = comfy.text_encoders.anima.te(**detect)
            clip = te(device="cpu")
            # the weights must be loaded explicitly: te() only builds the graph
            clip.load_sd(sd)
        finally:
            _restore(saved)
        return clip
    finally:
        os.chdir(cwd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--comfy_path", required=True)
    ap.add_argument("--model_type", default="qwen3_06b")
    ap.add_argument("--groupsize", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--resume", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    print(f"[load] {args.model}", flush=True)
    clip = build_model(args.comfy_path, os.path.abspath(args.model), args.model_type)
    # SD1ClipModel keeps the inner model under clip.<name> (e.g. clip_qwen3_06b);
    # that inner object holds the actual transformer under .transformer
    name = getattr(clip, "clip", None)
    inner = getattr(clip, name) if isinstance(name, str) else getattr(clip, "cond_stage_model", clip)
    llama = getattr(inner, "transformer", None) or getattr(inner, "model", None) or inner
    print(f"[build] clip={type(clip).__name__} inner={type(inner).__name__} model={type(llama).__name__}", flush=True)

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    ids = torch.randint(0, 150000, (1, args.tokens), generator=g)
    ids = ids.to(device)

    def run():
        with torch.no_grad():
            out = llama(ids)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, dict):
            parts = [v.detach().float().reshape(-1).cpu() for k, v in sorted(out.items()) if torch.is_tensor(v)]
            return torch.cat(parts)
        return out.detach().float().reshape(-1).cpu()

    print("[*] pristine", flush=True)
    ref = run()
    print(f"[*] ref shape={tuple(ref.shape)} finite={bool(torch.isfinite(ref).all())}", flush=True)

    mods = {}
    for name, m in llama.named_modules():
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
            print(f"    {name}: ERROR {type(e).__name__}: {str(e)[:140]}", flush=True)
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


if __name__ == "__main__":
    main()

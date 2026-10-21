# -*- coding: utf-8 -*-
"""ControlNet per-layer ConvRot INT8 trajectory impact (individual measurement).

HSWQ method, applied to a ControlNet:
  1. load base SDXL + this ControlNet
  2. build a fixed control hint and a fixed-seed N-step denoising trajectory (baseline)
  3. for each CANDIDATE layer: replace its weight with the ConvRot INT8
     reconstruction (dequant(INT8(W@H^T))) and re-run; record relative MSE of the
     final latent (trajectory drift)
  4. ascending drift = safe to quantize; descending = must stay FP16

Candidates are chosen by this checkpoint's own measured spread: the fat-tail
group plus the layers just below it (the decision boundary), NOT the full 401
sweep (which would take hours). This is the individual decision the converter's
static threshold could only guess at.

Usage:
  python sdxl/diag_impact_controlnet.py --base <sdxl.safetensors> --controlnet <cn.safetensors> --out <impact.json>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch


def _clear_argv():
    saved = sys.argv
    sys.argv = [saved[0]]
    return saved


def _restore_argv(saved):
    sys.argv = saved


def setup_comfy(comfy_path):
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)


def build_hadamard(size, device="cpu", dtype=torch.float32):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype, device=device)
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


def rotate_weight(w, h, gs):
    of, inf = w.shape
    g = inf // gs
    return torch.matmul(w.view(of, g, gs), h.T.to(w.dtype)).reshape(w.shape)


def rotate_weight_conv2d(w, h, gs):
    oc, ic, kh, kw = w.shape
    flat = w.permute(0, 2, 3, 1).contiguous().view(-1, ic)
    return rotate_weight(flat, h, gs).view(oc, kh, kw, ic).permute(0, 3, 1, 2).contiguous()


def quant_rowwise(w):
    amax = w.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
    s = amax / 127.0
    q = (w / s.to(w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, s.to(torch.float32)


def quant_channelwise(w):
    rd = tuple(range(1, w.dim()))
    amax = torch.clamp(w.abs().amax(dim=rd), min=1e-30).float()
    s = amax / 127.0
    v = s.view(-1, *([1] * (w.dim() - 1)))
    av = amax.view(-1, *([1] * (w.dim() - 1)))
    q = (torch.clamp(w, -av, av) / v).round().clamp(-127, 127).to(torch.int8)
    return q, s.to(torch.float32)


def convrot_int8_recon(w, gs):
    h = build_hadamard(gs, device=w.device)
    if w.ndim == 2:
        of, inf = w.shape
        g = inf // gs
        q, s = quant_rowwise(rotate_weight(w.float(), h, gs))
        # un-rotate into the ORIGINAL frame - see the module docstring
        deq = (q.float() * s.float()).view(of, g, gs)
        return torch.matmul(deq, h.float()).reshape(w.shape).to(w.dtype)
    else:
        w_rot = rotate_weight_conv2d(w.float(), h, gs)
        q, s = quant_channelwise(w_rot)
        v = s.view(-1, 1, 1, 1)
        return q.float() * v


def rel_mse(a, b):
    a = a.float().reshape(a.shape[0], -1)
    b = b.float().reshape(b.shape[0], -1)
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--controlnet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--comfy_path", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--measure-width", type=int, default=0, help="trajectory width for the impact measurement (0 = use --width)")
    ap.add_argument("--measure-height", type=int, default=0, help="trajectory height for the impact measurement (0 = use --height)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--groupsize", type=int, default=256)
    ap.add_argument("--prompt", default="a beautiful landscape, high detail")
    ap.add_argument("--negative", default="")
    ap.add_argument("--kurt-threshold", type=float, default=3.0)
    ap.add_argument("--full", action="store_true", help="measure every eligible module (full sweep)")
    ap.add_argument("--resume", default="", help="impact json to load already-measured layers from")
    args = ap.parse_args()

    setup_comfy(args.comfy_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    saved = _clear_argv()
    import comfy.sd
    import comfy.sample as comfy_sample
    import comfy.controlnet as comfy_cn
    import comfy.model_management as mm
    _restore_argv(saved)

    print(f"[load] base: {args.base}", flush=True)
    saved = _clear_argv()
    out = comfy.sd.load_checkpoint_guess_config(os.path.abspath(args.base), output_vae=False, output_clip=True, embedding_directory=None)
    _restore_argv(saved)
    patcher, clip = out[0], out[1]
    net = patcher.model
    net.to(device); net.eval()

    print(f"[load] controlnet: {args.controlnet}", flush=True)
    saved = _clear_argv()
    cnet = comfy_cn.load_controlnet(os.path.abspath(args.controlnet), model=net)
    _restore_argv(saved)
    if cnet is None:
        raise SystemExit("controlnet load failed")
    cnet.to(device) if hasattr(cnet, "to") else None

    # fixed control hint (deterministic; content is irrelevant for measurement)
    hint = torch.zeros(1, 3, args.height, args.width, device=device)
    cnet.set_cond_hint(hint, strength=1.0)
    cnet.pre_run(net, lambda p: int(p * 1000))

    prompt_pos = clip.encode_from_tokens_scheduled(clip.tokenize(args.prompt))
    prompt_neg = clip.encode_from_tokens_scheduled(clip.tokenize(args.negative))
    # inject the controlnet into the conditioning (as the apply-node does)
    cond_pos = list(prompt_pos)
    for c in cond_pos:
        c[1]["control"] = cnet
    del clip
    torch.cuda.empty_cache()

    latent0 = torch.zeros([1, 4, args.height // 8, args.width // 8], device=mm.intermediate_device(), dtype=mm.intermediate_dtype())
    latent0 = comfy_sample.fix_empty_latent_channels(patcher, latent0, 8, None)
    if latent0.device.type != device:
        latent0 = latent0.to(device)

    def run_traj():
        noise = comfy_sample.prepare_noise(latent0, args.seed, None)
        xs = []
        def cb(step, x0, x, total):
            xs.append(x.detach().to("cpu", torch.float32, copy=True))
        with torch.no_grad():
            comfy_sample.sample(patcher, noise, args.steps, args.cfg, "dpmpp_2m", "karras",
                                cond_pos, prompt_neg, latent0, denoise=1.0, disable_noise=False,
                                start_step=None, last_step=None, force_full_denoise=False,
                                noise_mask=None, callback=cb, disable_pbar=True, seed=args.seed)
        return xs

    print("[*] pristine trajectory", flush=True)
    x_ref = run_traj()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[*] pristine done", flush=True)

    # candidate modules from the controlnet's own weights
    mods = {}
    base_model = cnet.control_model
    for n, m in base_model.named_modules():
        if not hasattr(m, "weight") or m.weight is None:
            continue
        if m.weight.ndim not in (2, 4):
            continue
        gs = convrot_group_size(int(m.weight.shape[1]), args.groupsize)
        if gs is None:
            continue
        mods[n] = (m, gs)
    print(f"[target] controlnet modules: {len(mods)}", flush=True)

    # candidates: fat-tail group + boundary neighbourhood (this file's own spread)
    stats = []
    for n, (m, gs) in mods.items():
        w = m.weight.data
        a = (w.reshape(w.shape[0], -1) if w.ndim == 4 else w).float().reshape(-1)
        a = a.detach().cpu().numpy().astype(np.float64)
        std = a.std()
        if std == 0:
            continue
        z = (a - a.mean()) / std
        kurt = float((z ** 4).mean() - 3.0)
        stats.append((n, kurt))
    stats.sort(key=lambda x: -x[1])
    cand = [n for n, k in stats if k >= args.kurt_threshold]
    # add the 20 layers just below the threshold (boundary neighbourhood)
    below = [n for n, k in stats if k < args.kurt_threshold][:20]
    if args.full:
        cand = sorted(mods.keys())
        print(f"[target] FULL sweep: {len(cand)} modules", flush=True)
    else:
        cand = cand + below
        print(f"[target] candidates (fat-tail {len([1 for n,k in stats if k>=args.kurt_threshold])} + {len(below)} near): {len(cand)}", flush=True)

    # resume support
    impacts = {}
    if args.resume and os.path.isfile(args.resume):
        try:
            prev = json.load(open(args.resume, encoding="utf-8"))["impacts"]
            impacts.update(prev)
            print(f"[resume] loaded {len(impacts)} measured layers from {args.resume}", flush=True)
        except Exception as e:
            print(f"[resume] failed to load {args.resume}: {e}", flush=True)

    def _save_partial():
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"controlnet": os.path.basename(args.controlnet), "steps": args.steps,
                       "seed": args.seed, "impacts": impacts}, f, indent=2)

    done = 0
    for i, n in enumerate(cand):
        if n in impacts:
            continue
        m, gs = mods[n]
        w_dev = m.weight.device
        # keep the original on CPU and compute the reconstruction on CPU:
        # the GPU peak (clone + reconstruction) is what exhausts a 16 GB card
        w0 = m.weight.data.detach().to("cpu", copy=True)
        try:
            rec = convrot_int8_recon(w0, gs)
            m.weight.data.copy_(rec.to(w0.dtype).to(w_dev))
            del rec
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            xt = run_traj()
            imp = sum(rel_mse(xt[j], x_ref[j]) for j in range(len(x_ref))) / len(x_ref)
            del xt
        finally:
            m.weight.data.copy_(w0.to(w_dev))
            del w0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        impacts[n] = imp
        done += 1
        # free VRAM aggressively and checkpoint the partial result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if done % 10 == 0:
            _save_partial()
        print(f"  [{i+1}/{len(cand)}] {n}: rel_mse={imp:.6e}", flush=True)
    _save_partial()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"controlnet": os.path.basename(args.controlnet), "steps": args.steps,
                   "seed": args.seed, "impacts": impacts}, f, indent=2)
    ranked = sorted(impacts.items(), key=lambda x: x[1])
    print("\n== ascending drift (safest to quantize first) ==")
    for n, v in ranked[:15]:
        print(f"  {v:.6e}  {n}")
    print("== descending drift (must stay FP16) ==")
    for n, v in ranked[-15:]:
        print(f"  {v:.6e}  {n}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

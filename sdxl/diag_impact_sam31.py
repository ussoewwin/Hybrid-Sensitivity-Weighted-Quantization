# -*- coding: utf-8 -*-
"""SAM 3.1 Multiplex ConvRot INT8 trajectory-impact measurement.

A SAM model is not a denoiser: it produces masks, not a latent trajectory. The
measurement therefore replaces the ControlNet's "latent drift" with the drift of
the **mask logits** produced by the same prompt on the same input - the same
marginal-effect principle (perturb one layer, measure the output change in the
real pipeline), applied to SAM's actual output.

Pipeline (ComfyUI-native):
  - base model : the SAM 3.1 checkpoint itself (it is the model being measured)
  - input      : a deterministic synthetic image (content is irrelevant, the
                  graph and the prompt path are what must be reproduced)
  - prompt     : a fixed point prompt (tracker SAM-decoder path), which exercises
                  the tracker's encoder / decoder / cross-attention
  - output     : the raw mask logits of `forward_segment`
  - metric     : relative MSE of the mask logits against the pristine run

Usage:
  python sdxl/diag_impact_sam31.py --model <sam3.1_multiplex_fp16.safetensors> \
      --out impact/impact_sam31.json --comfy_path <ComfyUI> --full
"""
from __future__ import annotations

import argparse
import gc
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


def _restore(saved):
    sys.argv = saved


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
    w_rot = rotate_weight_conv2d(w.float(), h, gs)
    q, s = quant_channelwise(w_rot)
    return q.float() * s.view(-1, 1, 1, 1)


def rel_mse(a, b):
    a = a.float().reshape(1, -1)
    b = b.float().reshape(1, -1)
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--comfy_path", required=True)
    ap.add_argument("--groupsize", type=int, default=256)
    ap.add_argument("--size", type=int, default=1008)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--resume", default="")
    ap.add_argument("--limit", type=int, default=0, help="measure only the first N candidates (0 = all)")
    args = ap.parse_args()

    if args.comfy_path not in sys.path:
        sys.path.insert(0, args.comfy_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    saved = _clear_argv()
    try:
        import comfy.utils
        import comfy.model_management as mm
    finally:
        _restore(saved)

    print(f"[load] model: {args.model}", flush=True)
    saved = _clear_argv()
    try:
        sd = comfy.utils.load_torch_file(os.path.abspath(args.model))
    finally:
        _restore(saved)

    print("[build] SAM3 model", flush=True)
    saved = _clear_argv()
    try:
        import comfy.sd
        model = comfy.sd.load_diffusion_model_state_dict(sd, model_options={})
    except Exception as e:
        _restore(saved)
        raise SystemExit(f"SAM3 model construction failed: {e}")
    _restore(saved)
    if model is None:
        raise SystemExit("comfy.sd.load_diffusion_model_state_dict returned None")
    # load_diffusion_model_state_dict returns a ModelPatcher; .model is comfy.model_base.SAM3
    # whose .diffusion_model is the comfy.ldm.sam3.detector.SAM3Model (detector + tracker)
    net = getattr(model, "model", model)
    net = net.to(device).eval()
    sam3 = getattr(net, "diffusion_model", net)
    print(f"[build] diffusion_model: {type(sam3).__name__}", flush=True)

    # deterministic synthetic input
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    img = torch.rand(1, args.size, args.size, 3, generator=g, dtype=torch.float32)
    # match the model's own dtype / device (the SAM3 backbone runs in fp16 here)
    mdt = next((p.dtype for p in sam3.parameters() if p.is_floating_point()), torch.float16)
    img = (img * 255).to(device=device, dtype=mdt)
    frame = img[..., :3].movedim(-1, 1).to(device=device, dtype=mdt)

    # fixed point prompt (tracker SAM-decoder path)
    # forward_segment expects point_coords [B, N, 2] and point_labels [B, N]
    point_inputs = {"point_coords": torch.tensor([[[256.0, 256.0]]], device=device, dtype=mdt),
                    "point_labels": torch.tensor([[1]], device=device, dtype=torch.int32)}

    # --- text path (detector) conditioning: a fixed embedding tensor ---
    # The detector path consumes (text_embeddings, text_mask); a fixed random
    # embedding with a fixed mask reproduces the graph without needing a CLIP
    # text encoder, and the content is irrelevant for the measurement.
    text_emb = torch.zeros(1, 32, 1024, device=device, dtype=mdt)
    g2 = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    text_emb = (torch.rand(1, 32, 1024, generator=g2) * 2 - 1).to(device=device, dtype=mdt)
    text_mask = torch.ones(1, 32, dtype=torch.int64, device=device)

    def run_mask():
        """Both SAM paths: the tracker SAM-decoder (point prompt) and the
        detector (text prompt). Their outputs are concatenated so a single
        metric covers every sub-network."""
        with torch.no_grad():
            seg = sam3.forward_segment(frame, point_inputs=point_inputs)
            det = sam3(frame, text_embeddings=text_emb, text_mask=text_mask,
                       threshold=0.5, orig_size=(args.size, args.size))
        parts = [seg.detach().float().reshape(-1).cpu()]
        for key in ("masks", "boxes", "scores"):
            v = det.get(key) if isinstance(det, dict) else None
            if v is not None:
                parts.append(v.detach().float().reshape(-1).cpu())
        return torch.cat(parts)

    print("[*] pristine mask", flush=True)
    x_ref = run_mask()
    print(f"[*] pristine done shape={tuple(x_ref.shape)}", flush=True)

    # candidate modules
    mods = {}
    for n, m in sam3.named_modules():
        if not hasattr(m, "weight") or m.weight is None:
            continue
        if m.weight.ndim not in (2, 4):
            continue
        gs = convrot_group_size(int(m.weight.shape[1]), args.groupsize)
        if gs is None:
            continue
        mods[n] = (m, gs)
    print(f"[target] SAM modules: {len(mods)}", flush=True)

    cand = sorted(mods.keys())
    if args.limit:
        cand = cand[: args.limit]

    impacts = {}
    if args.resume and os.path.isfile(args.resume):
        try:
            impacts.update(json.load(open(args.resume, encoding="utf-8"))["impacts"])
            print(f"[resume] loaded {len(impacts)} layers", flush=True)
        except Exception as e:
            print(f"[resume] failed: {e}", flush=True)

    def save():
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({"model": os.path.basename(args.model), "impacts": impacts},
                  open(args.out, "w", encoding="utf-8"), indent=2)

    for i, n in enumerate(cand):
        if n in impacts:
            continue
        m, gs = mods[n]
        w_dev = m.weight.device
        w0 = m.weight.data.detach().to("cpu", copy=True)
        try:
            rec = convrot_int8_recon(w0, gs)
            m.weight.data.copy_(rec.to(w0.dtype).to(w_dev))
            del rec
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            xt = run_mask()
            imp = rel_mse(xt, x_ref)
            del xt
        finally:
            m.weight.data.copy_(w0.to(w_dev))
            del w0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        impacts[n] = imp
        if (i + 1) % 10 == 0:
            save()
            gc.collect()
        print(f"  [{i+1}/{len(cand)}] {n}: rel_mse={imp:.6e}", flush=True)
    save()

    ranked = sorted(impacts.items(), key=lambda x: x[1])
    print("\n== ascending drift (safest to quantize) ==")
    for n, v in ranked[:15]:
        print(f"  {v:.6e}  {n}")
    print("== descending drift (must stay FP16) ==")
    for n, v in ranked[-15:]:
        print(f"  {v:.6e}  {n}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""FLUX ControlNet (Union Pro) trajectory-impact measurement.

The observable is the ControlNet's own output: the per-block control features
it returns to the UNet (`controlnet_double` / `controlnet_single`). Each
candidate Linear layer's ConvRot INT8 reconstruction is injected and the
relative MSE of those features against the pristine run is recorded.

The runtime is built exactly the way the HSWQ ControlNet loader builds it
(`comfy.ldm.flux.controlnet.ControlNetFlux` with the comfy-native comfy-quant
state dict), so the measurement reflects the production path.

Usage:
  python sdxl/diag_impact_cn_flux.py \
      --controlnet <FLUX ControlNet>.safetensors \
      --out impact/impact_cn_flux_union_pro_2.json \
      --comfy_path <ComfyUI checkout> --full
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


def group_size_for(n, preferred=256):
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def convrot_int8_recon(w, gs, h=None):
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


# --- the runtime ------------------------------------------------------------

def build_runtime(comfy_path, ckpt_path, device="cuda", dtype=torch.bfloat16):
    """Build the ControlNet the way the HSWQ loader does."""
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)
    cwd = os.getcwd()
    try:
        os.chdir(comfy_path)
        import comfy.utils
        import comfy.controlnet
        import comfy.model_management
        from comfy import ops as comfy_ops
        from comfy.quant_ops import QUANT_ALGOS
        sd = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
        operations = comfy_ops.mixed_precision_ops(
            {"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]},
            dtype, full_precision_mm=False, disabled=[])
        # The source checkpoint is in the diffusers layout, so it is converted
        # to the comfy-native layout first; that is what the loader does and
        # what the model config detection expects.
        converted = comfy.model_detection.convert_diffusers_mmdit(dict(sd), "")
        if converted is None:
            raise SystemExit("convert_diffusers_mmdit returned None")
        # the converted sd carries the original keys too
        for k in sd:
            converted[k] = sd[k]
        model_config = comfy.model_detection.model_config_from_unet(
            converted, "", use_base_if_no_match=True)
        if model_config is None:
            raise SystemExit("could not detect the model config")
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        unet_dtype = comfy.model_management.unet_dtype(model_params=-1)
        manual_cast_dtype = comfy.model_management.unet_manual_cast(unet_dtype, load_device)
        control_latent_channels = converted["pos_embed_input.weight"].shape[1] // 4
        import comfy.ldm.flux.controlnet as cnf
        control_model = cnf.ControlNetFlux(
            latent_input=True, num_union_modes=0,
            control_latent_channels=control_latent_channels,
            operations=operations, device=offload_device, dtype=unet_dtype,
            **model_config.unet_config)
        control_model = comfy.controlnet.controlnet_load_state_dict(control_model, converted)
        return control_model, sd
    finally:
        os.chdir(cwd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controlnet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--comfy_path", required=True)
    ap.add_argument("--groupsize", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--latent_side", type=int, default=32,
                    help="the side of the latent square fed to the ControlNet")
    ap.add_argument("--txt_tokens", type=int, default=77)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--resume", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    DT = torch.bfloat16 if dev == "cuda" else torch.float32
    torch.manual_seed(args.seed)

    print(f"[1] building the runtime ControlNet (device={dev}) ...", flush=True)
    t0 = time.time()
    model, sd = build_runtime(args.comfy_path, os.path.abspath(args.controlnet), device=dev, dtype=DT)
    model = model.to(dev)
    model.eval()
    print(f"    built in {time.time()-t0:.0f}s", flush=True)

    hidden = 3072
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    # The real input is a latent tensor (b, c, h, w) plus a context tensor
    # and a control hint; the FLUX ControlNet's forward() does the patchify.
    latent_c = 16
    side = args.latent_side
    x = torch.randn(1, latent_c, side, side, generator=g).to(device=dev, dtype=DT)
    # the context is the FLUX text-encoder output: T5 (4096) + CLIP pooled
    # (768) concatenated, so context_in_dim is 4096
    ctx_dim = int(getattr(model.params, "context_in_dim", 4096))
    txt = torch.randn(1, args.txt_tokens, ctx_dim, generator=g).to(device=dev, dtype=DT)
    y = torch.zeros(1, hidden, device=dev, dtype=DT)
    timesteps = torch.full((1,), 500.0, device=dev, dtype=DT)
    guidance = torch.full((1,), 3.5, device=dev, dtype=DT)
    hint = torch.randn(1, latent_c, side, side, generator=g).to(device=dev, dtype=DT)

    def run():
        """One ControlNet forward; returns the control features it produces."""
        with torch.no_grad():
            out = model.forward(
                x=x, timesteps=timesteps, context=txt, y=y, guidance=guidance, hint=hint)
        parts = []
        if isinstance(out, dict):
            for k in sorted(out):
                v = out[k]
                if torch.is_tensor(v):
                    parts.append(v.detach().float().reshape(-1).cpu())
                elif isinstance(v, (tuple, list)):
                    for t2 in v:
                        if torch.is_tensor(t2):
                            parts.append(t2.detach().float().reshape(-1).cpu())
        elif torch.is_tensor(out):
            parts.append(out.detach().float().reshape(-1).cpu())
        elif isinstance(out, (tuple, list)):
            for t2 in out:
                if torch.is_tensor(t2):
                    parts.append(t2.detach().float().reshape(-1).cpu())
        if not parts:
            raise RuntimeError("the forward produced no observable tensor")
        return torch.cat(parts)

    print("[2] the reference output ...", flush=True)
    ref = run()
    print(f"    shape={tuple(ref.shape)} finite={bool(torch.isfinite(ref).all())} "
          f"absmax={ref.abs().max().item():.4f}", flush=True)
    if not bool(torch.isfinite(ref).all()):
        raise SystemExit("the reference output is not finite - aborting")

    mods = {}
    for name, m in model.named_modules():
        if type(m).__name__ == "Linear" and getattr(m, "weight", None) is not None:
            mods[name] = m
    # The fused qkv layers are excluded: the runtime builds one Linear out of
    # cat([to_q, to_k, to_v], dim=0), and the three halves have value ranges
    # that differ by ~7x (absmax 2.48 / 0.53 / 0.375). One per-out-channel
    # scale cannot serve them, and the measured reconstruction error of the
    # fused tensor is 1.89 - a broken layer. They stay in FP16.
    cand = sorted(n for n in mods if "qkv" not in n)
    print(f"[*] candidate Linear modules: {len(cand)} (the {len(mods)-len(cand)} fused qkv layers are excluded)", flush=True)

    results = {}
    if args.resume and os.path.isfile(args.resume):
        try:
            results = json.load(open(args.resume, encoding="utf-8")).get("impacts", {})
            print(f"[resume] {len(results)} already measured", flush=True)
        except Exception:
            results = {}

    print("[3] the sweep ...", flush=True)
    t0 = time.time()
    todo = [n for n in cand if n not in results]
    if args.limit:
        todo = todo[: args.limit]
    h_cache = {}
    for i, name in enumerate(todo):
        m = mods[name]
        w = m.weight.data.detach().to("cpu").clone()
        try:
            gs = group_size_for(int(w.shape[1]), args.groupsize)
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
    import statistics
    main()

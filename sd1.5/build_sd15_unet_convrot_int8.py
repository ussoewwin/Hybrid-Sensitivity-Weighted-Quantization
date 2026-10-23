# -*- coding: utf-8 -*-
"""SD1.5 UNet ConvRot INT8 pack builder driven by the measured aggregate
contribution.

The protection set comes from `sd1.5/measure_sd15_unet_contrib.py` (the
contribution measured on the UNet's OWN output, float64, 8 seeds). The top-K
layers by `1 - cos` are kept in FP16; every other ConvRot-eligible Linear/Conv2d
goes to INT8.

**CLIP is not quantized** (the caller's instruction): only the UNet subtree
(`model.diffusion_model.`) is touched. The checkpoint's CLIP and VAE pass
through byte-identical, so the pack remains loadable as a full checkpoint.

Rules:
  1. Only 2D Linear and 4D Conv2d weights under `model.diffusion_model.` are
     quantized. Conv2d rotates along the input-channel axis.
  2. Group size 256, falling back to 64 / 16 / 4 where the input width demands.
  3. Kept tensors are stored FP16 (the runtime runs FP16); the INT8
     `weight_scale` stays FP32 by design.

Usage:
  python sd1.5/build_sd15_unet_convrot_int8.py \
      --input <checkpoint>.safetensors --output <pack>.safetensors \
      --contrib impact/contrib_sd15_unet.json --top-k 40
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch
from safetensors.torch import load_file, save_file

GROUP_SIZE = 256
UNET_PREFIX = "model.diffusion_model."


def build_hadamard(size, dtype=torch.float32):
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError("Hadamard size must be a power of 4, got %s" % size)
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return h / (size ** 0.5)


def group_size_for(n, preferred=GROUP_SIZE):
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def rotate_linear(w, h, gs):
    of, inf = w.shape
    g = inf // gs
    return torch.matmul(w.view(of, g, gs), h.T.to(w.dtype)).reshape(w.shape)


def rotate_conv(w, h, gs):
    oc, ic, kh, kw = w.shape
    flat = w.permute(0, 2, 3, 1).contiguous().view(-1, ic)
    of, inf = flat.shape
    g = inf // gs
    rot = torch.matmul(flat.view(of, g, gs), h.T.to(w.dtype)).reshape(flat.shape)
    return rot.view(oc, kh, kw, ic).permute(0, 3, 1, 2).contiguous()


def quant_rowwise(w):
    """Per-OUTPUT-CHANNEL scale.

    `amax(dim=-1)` is right for a 2D Linear ([O, I] -> scale [O, 1]), but wrong
    for a 4D Conv ([O, kh, kw, Ic] -> scale [O, kh, kw, 1]): that keeps a scale
    per (output, kernel-position) - up to 9x the values for a 3x3 kernel - and
    the loader cannot broadcast it back. A Conv scale must reduce over every
    non-output axis and be stored as [O, 1, 1, 1].
    """
    # The loader broadcasts the scale over the weight's own rank, and the
    # reference implementation (sd1.5/native_convert_int8_sd1.5.py,
    # pack_channelwise) stores it in the MATCHING rank, not flattened:
    #   2D -> (O, 1)      4D -> (O, 1, 1, 1)
    # A 1-D (O,) tensor does not broadcast against a 4-D weight the same way and
    # produced NaN through the Conv2d path.
    rd = tuple(range(1, w.dim()))
    amax = w.abs().amax(dim=rd).float().clamp(min=1e-30)
    s = amax / 127.0
    if w.dim() == 2:
        sv = s.view(-1, 1)
    elif w.dim() == 4:
        sv = s.view(-1, 1, 1, 1)
    else:
        raise ValueError("unsupported weight rank %d" % w.dim())
    q = (w / sv.to(w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, sv.to(torch.float32)


def _encode(config):
    return torch.tensor(list(json.dumps(config, separators=(",", ":")).encode("utf-8")),
                        dtype=torch.uint8)


def convert(input_path, output_path, contrib_json, top_k):
    src = load_file(input_path)

    contrib = json.load(open(contrib_json, encoding="utf-8"))["impacts"]
    ranked = sorted(((n, v) for n, v in contrib.items() if v is not None),
                    key=lambda kv: -kv[1])
    protect = {n for n, _ in ranked[:top_k]}
    print("the measured candidates: %d | the adopted protection: %d" % (len(ranked), len(protect)))

    h = build_hadamard(GROUP_SIZE)
    out, meta_layers = {}, {}
    n_lin = n_prot = n_kept = n_clip = n_vae_dropped = 0
    for name in sorted(src):
        t = src[name]
        if name.startswith("first_stage_model."):
            # the VAE is dropped entirely (this delivery ships UNet + CLIP only)
            n_vae_dropped += 1
            continue
        if not name.startswith(UNET_PREFIX):
            # CLIP passes through untouched (it is not quantized)
            if name.startswith("cond_stage_model."):
                n_clip += 1
            out[name] = t
            n_kept += 1
            continue
        # Linear AND Conv2d both go INT8. comfy_kitchen has no int8_conv, but the
        # HSWQ loader's `_patch_ops_decode_and_conv()` installs the Conv2d path
        # (dequant + online activation rotate). Running an INT8 Conv2d pack
        # WITHOUT that arm is what produced NaN earlier - the pack was fine, the
        # harness was not.
        if not (name.endswith(".weight") and t.ndim in (2, 4)):
            out[name] = t.to(torch.float16) if t.is_floating_point() else t
            n_kept += 1
            continue
        mk = name[:-len(".weight")]
        short = name[len(UNET_PREFIX):-len(".weight")]
        if short in protect:
            out[name] = t.to(torch.float16)
            n_prot += 1
            continue
        inf = int(t.shape[1])
        gs = group_size_for(inf)
        if gs is None or int(t.shape[0]) % 4 != 0:
            out[name] = t.to(torch.float16)
            n_prot += 1
            continue
        hh = h if gs == GROUP_SIZE else build_hadamard(gs)
        w = t.float()
        rot = rotate_linear(w, hh, gs) if t.ndim == 2 else rotate_conv(w, hh, gs)
        q, scale = quant_rowwise(rot)
        n_lin += 1
        conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": gs}
        out[name] = q
        out["%s.weight_scale" % mk] = scale
        out["%s.comfy_quant" % mk] = _encode(conf)
        meta_layers[mk] = conf

    metadata = {"_quantization_metadata": json.dumps(
        {"format_version": "1.0", "layers": meta_layers}, separators=(",", ":"))}
    save_file(out, output_path, metadata=metadata)
    print("  the ConvRot INT8: %d" % n_lin)
    print("  the protected (FP16 kept 2D/4D): %d" % n_prot)
    print("  the untouched (CLIP/1D): %d (of which CLIP %d)" % (n_kept, n_clip))
    print("  the VAE tensors DROPPED: %d" % n_vae_dropped)
    print("  the output: %.1f MB" % (os.path.getsize(output_path) / 1024 ** 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--contrib", required=True)
    ap.add_argument("--top-k", type=int, default=0)
    a = ap.parse_args()
    convert(a.input, a.output, a.contrib, a.top_k)

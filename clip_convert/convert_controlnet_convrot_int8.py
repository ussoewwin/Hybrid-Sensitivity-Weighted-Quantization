# -*- coding: utf-8 -*-
"""ControlNet ConvRot INT8 converter driven by measured trajectory impact.

Individual decision per checkpoint: the protection set comes from a trajectory
impact JSON measured on THIS checkpoint (sibling script diag_impact_controlnet.py).
Runtime (comfy input_blocks) names in the impact JSON are mapped onto the
checkpoint's diffusers (down_blocks) keys via comfy's own unet_to_diffusers,
so no hand-maintained name table is used.

Layers with trajectory drift > threshold stay in original precision; the
remainder (2D Linear + 4D Conv2d) is ConvRot INT8 quantized.

Usage:
  python clip_convert/convert_controlnet_convrot_int8.py \
      --input <cn.safetensors> --output <out> --impact <impact.json> --comfy_path <ComfyUI>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_DEFAULT_GROUPSIZE = 256
_DEFAULT_DRIFT_THRESHOLD = 0.10


def build_hadamard(size, dtype=torch.float32):
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
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


def rotate_weight(weight, h, gs):
    of, inf = weight.shape
    g = inf // gs
    return torch.matmul(weight.view(of, g, gs), h.T.to(weight.dtype)).reshape(weight.shape)


def rotate_weight_conv2d(weight, h, gs):
    oc, ic, kh, kw = weight.shape
    flat = weight.permute(0, 2, 3, 1).contiguous().view(-1, ic)
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


def _encode(config):
    return torch.tensor(list(json.dumps(config, separators=(",", ":")).encode("utf-8")), dtype=torch.uint8)


def _runtime_to_checkpoint_map(checkpoint_path: str, comfy_path: str) -> dict:
    """comfy(runtime) key -> checkpoint(diffusers) key, via comfy's own converter."""
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        import comfy.utils
        import comfy.model_detection
    finally:
        sys.argv = saved
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        sd = {k: torch.empty(f.get_slice(k).get_shape()) for k in f.keys()}
    cfg = comfy.model_detection.unet_config_from_diffusers_unet(sd)
    dmap = comfy.utils.unet_to_diffusers(cfg)   # diffusers -> comfy
    return {v: k for k, v in dmap.items()}      # comfy -> diffusers


def convert(input_path, output_path, impact_json, comfy_path,
            group_size=_DEFAULT_GROUPSIZE, drift_threshold=_DEFAULT_DRIFT_THRESHOLD):
    print(f"Loading (streamed): {input_path}")
    src = safe_open(input_path, framework="pt", device="cpu")
    keys = sorted(src.keys())
    print(f"  raw keys: {len(keys)}")

    impacts = json.load(open(impact_json, encoding="utf-8"))["impacts"]
    rt2ck = _runtime_to_checkpoint_map(input_path, comfy_path)
    protect = set()
    for n, drift in impacts.items():
        if drift <= drift_threshold:
            continue
        ck = rt2ck.get(n + ".weight")
        if ck:
            protect.add(ck[: -len(".weight")])
    print(f"  measured candidates: {len(impacts)}; drift > {drift_threshold} -> protected {len(protect)}")
    for p in sorted(protect):
        print(f"    {p}")

    out = {}
    meta_layers = {}
    n_lin = n_conv = n_prot = n_kept = 0
    h_cache = {}
    for key in keys:
        t = src.get_tensor(key)
        if not (key.endswith(".weight") and t.ndim in (2, 4)
                and t.dtype in (torch.float16, torch.float32, torch.bfloat16)):
            out[key] = t
            n_kept += 1
            continue
        mk = key[: -len(".weight")]
        if mk in protect:
            out[key] = t
            n_prot += 1
            continue
        w = t.float()
        gs = convrot_group_size(w.shape[1], group_size)
        if gs is None:
            q, scale = quant_rowwise(w)
            conf = {"format": "int8_tensorwise"}
        else:
            if gs not in h_cache:
                h_cache[gs] = build_hadamard(gs)
            if t.ndim == 2:
                q, scale = quant_rowwise(rotate_weight(w, h_cache[gs], gs))
                n_lin += 1
            else:
                q, scale = quant_channelwise(rotate_weight_conv2d(w, h_cache[gs], gs))
                n_conv += 1
            conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": int(gs)}
        out[key] = q
        out[f"{mk}.weight_scale"] = scale
        out[f"{mk}.comfy_quant"] = _encode(conf)
        meta_layers[mk] = conf

    metadata = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": meta_layers}, separators=(",", ":"))}
    print(f"\nSaving: {output_path}")
    print(f"  ConvRot INT8 Linear: {n_lin}  Conv2d: {n_conv}  protected: {n_prot}  kept: {n_kept}")
    save_file(out, output_path, metadata=metadata)
    print(f"  Input {os.path.getsize(input_path)/(1024**3):.2f} GB -> Output {os.path.getsize(output_path)/(1024**3):.2f} GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--impact", required=True)
    ap.add_argument("--comfy_path", required=True)
    ap.add_argument("--groupsize", type=int, default=_DEFAULT_GROUPSIZE)
    ap.add_argument("--drift-threshold", type=float, default=_DEFAULT_DRIFT_THRESHOLD)
    a = ap.parse_args()
    convert(a.input, a.output, a.impact, a.comfy_path, group_size=a.groupsize, drift_threshold=a.drift_threshold)

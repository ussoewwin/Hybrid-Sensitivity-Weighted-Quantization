# -*- coding: utf-8 -*-
"""ControlNet (SDXL ControlNet Union Pro Max) ConvRot INT8 converter with measured FP16 protection.

ControlNet is a single standard weight layout (per model family), like text
encoders and SAM - so it gets an individually built converter with measured
FP16 protection rather than the universal HSWQ engine (which targets
per-checkpoint image-generation backbones).

Protection set = measured risk (kurtosis / outlier ratio = absmax/std) on all
406 finite 2D+4D weights of controlnet-union-pro-max-sdxl-1.0.safetensors
(2026-10-20):

  dominant family: time-emb projection (kurt up to 611.6, median 259-549)
      mid_block.resnets.N.time_emb_proj / down_blocks.N.resnets.N.time_emb_proj
      time_embedding.linear_1
  attention key projections: attn1.to_k (kurt up to 141.7, median ~98)
  attention proj_in (kurt ~62-76)
  downsampler conv (kurt 197), mid_block.resnets conv1 (kurt 55.6),
  controlnet_cond_embedding.conv_out (kurt 40), transformer_layes.0.mlp.c_fc (183)

Everything else is FULL ConvRot INT8 (Linear + Conv2d), pack math identical to
native_convert_int8.py (build_hadamard / rotate_weight / rotate_weight_conv2d /
quantize_int8_rowwise / quantize_int8_channelwise).

Usage:
  python clip_convert/convert_controlnet_convrot_int8.py --input <controlnet.safetensors> --output <out>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_DEFAULT_GROUPSIZE = 256
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_native_convert_int8():
    path = os.path.join(_REPO, "native_convert_int8.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"native_convert_int8.py not found: {path}")
    name = "native_convert_int8_for_controlnet"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --- protection rule (measured 2026-10-20, structural) ----------------------
_PROTECT_REGEXES = [
    # time-emb projections (dominant family)
    re.compile(r"^(mid_block|down_blocks\.\d+)\.resnets\.\d+\.time_emb_proj\.weight$"),
    re.compile(r"^time_embedding\.linear_1\.weight$"),
    # attention key projections (attn1.to_k) and proj_in
    re.compile(r"^(mid_block|down_blocks\.\d+)\.attentions\.\d+\.transformer_blocks\.\d+\.attn1\.to_k\.weight$"),
    re.compile(r"^(mid_block|down_blocks\.\d+)\.attentions\.\d+\.proj_in\.weight$"),
    # downsampler conv + mid resnet conv1
    re.compile(r"^down_blocks\.\d+\.downsamplers\.\d+\.conv\.weight$"),
    re.compile(r"^mid_block\.resnets\.\d+\.conv1\.weight$"),
    # condition embedding output + transformer_layes mlp c_fc
    re.compile(r"^controlnet_cond_embedding\.conv_out\.weight$"),
    re.compile(r"^transformer_layes\.\d+\.mlp\.c_fc\.weight$"),
]


def is_protected_controlnet_weight(key: str) -> bool:
    return any(rx.match(key) for rx in _PROTECT_REGEXES)


def convert(input_path: str, output_path: str, group_size: int = _DEFAULT_GROUPSIZE):
    nc = _load_native_convert_int8()
    build_hadamard = nc.build_hadamard
    rotate_weight = nc.rotate_weight
    rotate_weight_conv2d = nc.rotate_weight_conv2d
    convrot_group_size_for_features = nc.convrot_group_size_for_features
    quantize_int8_rowwise = nc.quantize_int8_rowwise
    quantize_int8_channelwise = nc.quantize_int8_channelwise

    def _encode(config: dict) -> torch.Tensor:
        return torch.tensor(list(json.dumps(config, separators=(",", ":")).encode("utf-8")), dtype=torch.uint8)

    print(f"Loading (streamed): {input_path}")
    src = safe_open(input_path, framework="pt", device="cpu")
    keys = sorted(src.keys())
    print(f"  raw keys: {len(keys)}")

    out = {}
    meta_layers = {}
    n_lin = n_conv = n_prot = n_kept = 0
    protected_list = []
    fallback_list = []
    h_cache = {}

    for key in keys:
        tensor = src.get_tensor(key)
        is_w = key.endswith(".weight")
        nd = tensor.ndim
        if not (is_w and nd in (2, 4) and tensor.dtype in (torch.float16, torch.float32, torch.bfloat16)):
            out[key] = tensor
            n_kept += 1
            continue
        if is_protected_controlnet_weight(key):
            out[key] = tensor
            n_prot += 1
            protected_list.append(key)
            continue
        w = tensor.float()
        module_key = key[: -len(".weight")]
        gs = convrot_group_size_for_features(w.shape[1], group_size)
        if gs is None:
            fallback_list.append(key)
            q, scale = nc.quantize_int8_tensorwise(w)
            conf = {"format": "int8_tensorwise"}
        else:
            if gs not in h_cache:
                h_cache[gs] = build_hadamard(gs, device="cpu", dtype=torch.float32)
            if nd == 2:
                w_rot = rotate_weight(w, h_cache[gs], gs)
                q, scale = quantize_int8_rowwise(w_rot)
                n_lin += 1
            else:
                w_rot = rotate_weight_conv2d(w, h_cache[gs], gs)
                q, scale = quantize_int8_channelwise(w_rot)
                n_conv += 1
            conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": int(gs)}
        out[key] = q
        out[f"{module_key}.weight_scale"] = scale
        out[f"{module_key}.comfy_quant"] = _encode(conf)
        meta_layers[module_key] = conf

    metadata = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": meta_layers}, separators=(",", ":"))}

    print(f"\nProtected (kept FP16): {n_prot}")
    for k in protected_list[:12]:
        print(f"  {k}")
    if n_prot > 12:
        print(f"  ... ({n_prot - 12} more)")
    for k in fallback_list:
        print(f"  [WARN] {k}: not power-of-4 divisible, plain INT8")

    print(f"\nSaving: {output_path}")
    print(f"  ConvRot INT8 Linear: {n_lin}")
    print(f"  ConvRot INT8 Conv2d: {n_conv}")
    print(f"  FP16 protected:      {n_prot}")
    print(f"  Kept as-is:          {n_kept}")
    save_file(out, output_path, metadata=metadata)
    print(f"  Input:  {os.path.getsize(input_path) / (1024**3):.2f} GB")
    print(f"  Output: {os.path.getsize(output_path) / (1024**3):.2f} GB")
    print("Done!")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ControlNet ConvRot INT8 converter with measured FP16 protection")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--groupsize", type=int, default=_DEFAULT_GROUPSIZE)
    args = ap.parse_args()
    convert(args.input, args.output, group_size=args.groupsize)

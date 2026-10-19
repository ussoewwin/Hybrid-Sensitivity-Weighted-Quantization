# -*- coding: utf-8 -*-
"""SD1.5 checkpoint (UNet only) ConvRot INT8 converter with measured FP16 protection.

Model-specific converter for:
  0.7_0.6hassakuHentai+0.4dreamshaper+0.3anylora.safetensors

Policy (Owner directive):
  - CLIP (cond_stage_model.*)  : kept FP16 (passthrough, never quantized)
  - VAE  (first_stage_model.*) : dropped from the pack
  - UNet (model.diffusion_model.*): ConvRot INT8 + measured FP16 protection

Protection set = measured risk (kurtosis / outlier_ratio = absmax/std) on all
282 UNet 2D+4D weights (2026-10-20 measurement):

  dominant family: emb_layers.1 (time-embedding Linear)
      output_blocks.0.0.emb_layers.1  ratio 254.7 / kurt 29,617.8  (max)
      input_blocks.7.0.emb_layers.1   ratio 166.9 / kurt  2,442.7
      output_blocks.2.0.emb_layers.1  ratio  40.0 / kurt    431.9
      input_blocks.10.0.emb_layers.1  ratio  46.2 / kurt    418.0
      input_blocks.4.0.emb_layers.1   ratio  88.7 / kurt    409.2
      time_embed.2                    ratio  74.8 / kurt    385.8
      middle_block.2.emb_layers.1     ratio  24.0 / kurt    270.3
      ... (all emb_layers.1 + time_embed.* protected)

  second family: in_layers.2 / out_layers.3 Conv2d (kurt 10-148)

Everything else is FULL ConvRot INT8 (Linear + Conv2d), pack math identical to
native_convert_int8_convrot.py (build_hadamard / rotate_weight /
rotate_weight_conv2d / convrot_group_size_for_features / _encode_comfy_quant).

Usage:
  python sd1.5/convert_sd15_ckpt_convrot_int8.py --input <ckpt> --output <out>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
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
    name = "native_convert_int8_for_sd15_ckpt"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --- protection rule (measured 2026-10-20, structural) ----------------------
# protected if the key matches any pattern (fnmatch-style with * over dots)
_PROTECT_SUFFIXES = (
    ".emb_layers.1.weight",       # time-embedding Linear (dominant family)
)
_PROTECT_PREFIXES = (
    "model.diffusion_model.time_embed.",   # time embedding MLP
)
_PROTECT_EXACT = (
    # second family: extreme Conv2d in/out layers (kurt > 40)
    "model.diffusion_model.output_blocks.5.0.in_layers.2.weight",
    "model.diffusion_model.input_blocks.3.0.op.weight",
    "model.diffusion_model.output_blocks.4.0.in_layers.2.weight",
    "model.diffusion_model.output_blocks.0.0.out_layers.3.weight",
    "model.diffusion_model.input_blocks.6.0.op.weight",
    "model.diffusion_model.middle_block.2.out_layers.3.weight",
    "model.diffusion_model.input_blocks.7.0.in_layers.2.weight",
    "model.diffusion_model.output_blocks.3.0.in_layers.2.weight",
    "model.diffusion_model.output_blocks.3.0.out_layers.3.weight",
    "model.diffusion_model.output_blocks.2.0.out_layers.3.weight",
    "model.diffusion_model.output_blocks.2.0.in_layers.2.weight",
    "model.diffusion_model.output_blocks.6.0.in_layers.2.weight",
)


def is_protected_unet_weight(key: str) -> bool:
    if key in _PROTECT_EXACT:
        return True
    if key.startswith(_PROTECT_PREFIXES):
        return True
    if key.endswith(_PROTECT_SUFFIXES):
        return True
    return False


def convert(input_path: str, output_path: str, group_size: int = _DEFAULT_GROUPSIZE):
    nc = _load_native_convert_int8()
    build_hadamard = nc.build_hadamard
    rotate_weight = nc.rotate_weight
    rotate_weight_conv2d = nc.rotate_weight_conv2d
    convrot_group_size_for_features = nc.convrot_group_size_for_features

    def _encode_comfy_quant(config: dict) -> torch.Tensor:
        return torch.tensor(
            list(json.dumps(config, separators=(",", ":")).encode("utf-8")),
            dtype=torch.uint8,
        )

    print(f"Loading (streamed): {input_path}")
    src = safe_open(input_path, framework="pt", device="cpu")
    keys = sorted(src.keys())
    print(f"  raw keys: {len(keys)}")

    out = {}
    meta_layers = {}
    n_convrot_lin = n_convrot_conv = 0
    n_kept = n_prot = n_clip = n_vae = 0
    h_cache = {}
    protected_list = []

    for key in keys:
        tensor = src.get_tensor(key)

        # VAE -> drop
        if key.startswith("first_stage_model."):
            n_vae += 1
            continue
        # CLIP -> keep FP16 passthrough
        if key.startswith("cond_stage_model."):
            out[key] = tensor
            n_clip += 1
            continue
        # only UNet weights get quantized
        if not key.startswith("model.diffusion_model."):
            out[key] = tensor
            n_kept += 1
            continue

        is_w = key.endswith(".weight")
        nd = tensor.ndim
        if not (is_w and nd in (2, 4) and tensor.dtype in (torch.float16, torch.float32, torch.bfloat16)):
            out[key] = tensor
            n_kept += 1
            continue

        # measured protection -> keep FP16
        if is_protected_unet_weight(key):
            out[key] = tensor
            n_prot += 1
            protected_list.append(key)
            continue

        w = tensor.float()
        module_key = key[: -len(".weight")]
        in_features = int(w.shape[1]) if nd == 2 else int(w.shape[1])

        gs = convrot_group_size_for_features(in_features, group_size)
        if gs is None:
            raise RuntimeError(f"{key}: in_features {in_features} not power-of-4 divisible; refuse silent FP16 leak")
        if gs not in h_cache:
            h_cache[gs] = build_hadamard(gs, device="cpu")

        if nd == 2:
            w_rot = rotate_weight(w, h_cache[gs], gs)
            n_convrot_lin += 1
        else:
            w_rot = rotate_weight_conv2d(w, h_cache[gs], gs)
            n_convrot_conv += 1

        # channelwise INT8 (per-out-channel), same as native ConvRot pack
        reduce_dims = tuple(range(1, w_rot.dim()))
        amax = torch.clamp(w_rot.abs().amax(dim=reduce_dims).reshape(-1), min=1e-6)
        scale = amax / 127.0
        if w_rot.dim() == 4:
            scale_view = scale.view(-1, 1, 1, 1)
            amax_view = amax.view(-1, 1, 1, 1)
        else:
            scale_view = scale.view(-1, 1)
            amax_view = amax.view(-1, 1)
        clamped = torch.clamp(w_rot, -amax_view, amax_view)
        q = (clamped / scale_view).round().clamp(-127, 127).to(torch.int8)

        conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": int(gs)}
        out[key] = q
        out[f"{module_key}.weight_scale"] = scale_view.to(torch.float32)
        out[f"{module_key}.comfy_quant"] = _encode_comfy_quant(conf)
        meta_layers[module_key] = conf

    metadata = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": meta_layers})}

    print(f"\nProtected (kept FP16): {n_prot}")
    for k in protected_list:
        print(f"  {k}")

    print(f"\nSaving: {output_path}")
    print(f"  ConvRot INT8 Linear: {n_convrot_lin}")
    print(f"  ConvRot INT8 Conv2d: {n_convrot_conv}")
    print(f"  FP16 protected:      {n_prot}")
    print(f"  CLIP kept FP16:      {n_clip}")
    print(f"  VAE dropped:         {n_vae}")
    print(f"  other kept:          {n_kept}")
    save_file(out, output_path, metadata=metadata)

    in_gb = os.path.getsize(input_path) / (1024 ** 3)
    out_gb = os.path.getsize(output_path) / (1024 ** 3)
    print(f"  Input:  {in_gb:.2f} GB")
    print(f"  Output: {out_gb:.2f} GB")
    print("Done!")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SD1.5 checkpoint ConvRot INT8 converter with measured FP16 protection")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--groupsize", type=int, default=_DEFAULT_GROUPSIZE)
    args = ap.parse_args()
    convert(args.input, args.output, group_size=args.groupsize)

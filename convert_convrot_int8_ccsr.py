#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CCSR (SD2-based ControlNet-UNet, real-world_ccsr) FULL ConvRot INT8 converter.

Converts the UNet body (`model.diffusion_model.*`) Linear (2D) and Conv2d (4D)
weights of a bf16/fp16 checkpoint to FULL ConvRot INT8 with the SAME
conventions as hswq_convrot_int8_krea2_v1.5:

- rotate_weight (regular Hadamard, W @ H^T) using the largest power-of-4
  group size <= 256 that divides in_features; fall back through /4; a layer
  with no eligible group size packs plain
- plain packs: pack_tensorwise by default (--per_channel_int8 enables
  pack_channelwise); scale = amax / 127, q clamped to +-127 (int8)
- quant_config: {format:int8_tensorwise[, convrot:true, convrot_groupsize:N]}

Everything else keeps its ORIGINAL dtype: control_model, cond_stage_model,
first_stage_model, cond_encoder, decoder_loss, scheduler tensors, norm
weights (1D), biases, and the fixed blacklist (first./last./mod./norm).

Output is ComfyUI-compatible:
  weight(int8) + weight_scale(f32) + comfy_quant(uint8 json)
  + _quantization_metadata {format_version "1.0", layers}

Usage (one line):
    python convert_convrot_int8_ccsr.py <in.safetensors> --out <out.safetensors>
"""
from __future__ import annotations

import argparse
import json
import math

import torch
from safetensors.torch import load_file, save_file

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


_DEFAULT_GROUPSIZE = 256
_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

_UNET_PREFIX = "model.diffusion_model."
# Structure-sensitive name fragments kept in original dtype even if they were
# 2D weights (defensive: most are 1D and already skipped by the ndim filter).
_BLACKLIST_FRAGMENTS: tuple[str, ...] = (
    "first.",
    "last.",
    "mod.",
    "norm",
)


def _is_blacklisted(key: str) -> bool:
    return any(frag in key for frag in _BLACKLIST_FRAGMENTS)


def _meta_base_key(k: str) -> str:
    return k.split(_UNET_PREFIX)[-1]


def build_hadamard(
    size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normalized regular Hadamard (power-of-4), same as comfy_kitchen ConvRot."""
    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]

    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")

    h4 = torch.tensor(
        [
            [1, 1, 1, -1],
            [1, 1, -1, 1],
            [1, -1, 1, 1],
            [-1, 1, 1, 1],
        ],
        dtype=dtype,
        device=device,
    )
    h_matrix = h4
    current_size = 4
    while current_size < size:
        h_matrix = torch.kron(h_matrix, h4)
        current_size *= 4
    h_matrix = h_matrix / (size**0.5)
    _HADAMARD_CACHE[cache_key] = h_matrix
    return h_matrix


def convrot_group_size_for_features(
    n: int, preferred: int = _DEFAULT_GROUPSIZE
) -> int | None:
    """Largest power-of-4 group size <= preferred that divides n (or None)."""
    if n < 4:
        return None
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def rotate_weight(
    weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Offline Linear: W_rot = W @ H^T (group-wise). Matches kitchen ConvRot."""
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features {in_features} not divisible by group_size {group_size}"
        )
    group_count = in_features // group_size
    weight_grouped = weight.view(out_features, group_count, group_size)
    return torch.matmul(
        weight_grouped, h_matrix.T.to(dtype=weight.dtype, device=weight.device)
    ).reshape(weight.shape)


def rotate_weight_conv2d(
    weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Offline Conv2d: rotate along in_channels. weight (O, I, kH, kW)."""
    if weight.ndim != 4:
        raise ValueError(f"Conv2d weight must be 4D, got ndim={weight.ndim}")
    out_c, in_c, k_h, k_w = weight.shape
    flat = weight.permute(0, 2, 3, 1).contiguous().view(-1, in_c)
    flat_rot = rotate_weight(flat, h_matrix, group_size)
    return flat_rot.view(out_c, k_h, k_w, in_c).permute(0, 3, 1, 2).contiguous()


def pack_tensorwise(weight: torch.Tensor):
    """Symmetric per-tensor INT8: scale = amax / 127."""
    w = weight.float()
    amax = max(float(w.abs().max().item()), 1e-6)
    scale = amax / 127.0
    q = (w / scale).round().clamp(-127, 127).to(torch.int8)
    return q, torch.tensor(scale, dtype=torch.float32)


def pack_channelwise(weight: torch.Tensor):
    """Per-out-channel INT8 (ConvRot kitchen dequant shape)."""
    w = weight.float()
    reduce_dims = tuple(range(1, w.dim()))
    amax = torch.clamp(w.abs().amax(dim=reduce_dims).reshape(-1), min=1e-6)
    scale = amax / 127.0
    if w.dim() == 4:
        scale_view = scale.view(-1, 1, 1, 1)
    elif w.dim() == 2:
        scale_view = scale.view(-1, 1)
    else:
        raise ValueError(f"unsupported weight ndim={w.dim()} for channelwise INT8")
    clamped = torch.clamp(w, -scale_view * 127.0, scale_view * 127.0)
    q = (clamped / scale_view).round().clamp(-127, 127).to(torch.int8)
    return q, scale_view.to(dtype=torch.float32)


def _encode_comfy_quant(config: dict) -> torch.Tensor:
    return torch.tensor(
        list(json.dumps(config, separators=(",", ":")).encode("utf-8")),
        dtype=torch.uint8,
    )


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "CCSR UNet FULL ConvRot INT8 converter "
            "(model.diffusion_model.* Linear+Conv2d; everything else kept)"
        )
    )
    ap.add_argument("input", help="baseline bf16/fp16 CCSR safetensors")
    ap.add_argument("--out", "-o", required=True,
                    help="output ConvRot INT8 safetensors path")
    ap.add_argument("--group-size", type=int, default=_DEFAULT_GROUPSIZE,
                    help="preferred ConvRot group size (default 256; falls back /4)")
    ap.add_argument("--no-convrot", action="store_true",
                    help="disable FULL ConvRot (plain packs only)")
    ap.add_argument("--per_channel_int8", action="store_true",
                    help="plain (non-ConvRot) packs use per-out-channel INT8 "
                         "(default: tensorwise, same as v1.5)")
    return ap.parse_args()


def main():
    a = parse_args()
    print(f"Loading: {a.input}")
    state_dict = load_file(a.input)
    keys = list(state_dict.keys())
    print(f"total keys: {len(keys)}")

    enable_convrot = not a.no_convrot
    group_size = int(a.group_size)
    per_channel_int8 = bool(a.per_channel_int8)

    new_state_dict: dict[str, torch.Tensor] = {}
    quant_meta_layers: dict[str, dict] = {}
    converted_count = 0
    plain_int8_count = 0
    convrot_linear = 0
    convrot_conv2d = 0
    keep_count = 0
    mode = "per-channel" if per_channel_int8 else "tensorwise"
    rot_tag = " + ConvRot(Linear+Conv2d)" if enable_convrot else ""
    print(f"Converting UNet (model.diffusion_model.*) to INT8 ({mode}{rot_tag}, amax/127)...")

    for key, tensor in tqdm(list(state_dict.items())):
        # Everything outside the UNet body keeps its original dtype:
        # control_model / cond_stage_model / first_stage_model / cond_encoder /
        # decoder_loss / scheduler tensors / biases / norms(1D).
        if not (
            key.startswith(_UNET_PREFIX)
            and key.endswith(".weight")
            and tensor.ndim in (2, 4)
            and tensor.dtype in (torch.float16, torch.bfloat16)
        ):
            new_state_dict[key] = tensor
            keep_count += 1
            continue

        # Defensive: structure-sensitive fragments stay original dtype.
        if _is_blacklisted(key):
            new_state_dict[key] = tensor
            keep_count += 1
            continue

        w_fp = tensor.float()
        module_key = key[: -len(".weight")]

        if tensor.ndim == 2:
            used_gs = (
                convrot_group_size_for_features(int(w_fp.shape[1]), group_size)
                if enable_convrot
                else None
            )
            if used_gs is not None:
                h_matrix = build_hadamard(used_gs, device="cpu", dtype=torch.float32)
                w_rot = rotate_weight(w_fp, h_matrix, used_gs)
                q, scale = pack_channelwise(w_rot)
                quant_config = {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": int(used_gs),
                }
                convrot_linear += 1
            elif per_channel_int8:
                q, scale = pack_channelwise(w_fp)
                quant_config = {"format": "int8_tensorwise"}
                plain_int8_count += 1
            else:
                q, scale = pack_tensorwise(w_fp)
                quant_config = {"format": "int8_tensorwise"}
                plain_int8_count += 1
        else:  # Conv2d (4D)
            used_gs = (
                convrot_group_size_for_features(int(w_fp.shape[1]), group_size)
                if enable_convrot
                else None
            )
            if used_gs is not None:
                h_matrix = build_hadamard(used_gs, device="cpu", dtype=torch.float32)
                w_rot = rotate_weight_conv2d(w_fp, h_matrix, used_gs)
                q, scale = pack_channelwise(w_rot)
                quant_config = {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": int(used_gs),
                }
                convrot_conv2d += 1
            elif per_channel_int8:
                q, scale = pack_channelwise(w_fp)
                quant_config = {"format": "int8_tensorwise"}
                plain_int8_count += 1
            else:
                q, scale = pack_tensorwise(w_fp)
                quant_config = {"format": "int8_tensorwise"}
                plain_int8_count += 1

        new_state_dict[key] = q
        new_state_dict[f"{module_key}.weight_scale"] = scale
        new_state_dict[f"{module_key}.comfy_quant"] = _encode_comfy_quant(quant_config)
        quant_meta_layers[_meta_base_key(module_key)] = dict(quant_config)
        converted_count += 1

    metadata = {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": quant_meta_layers}
        )
    }

    print(f"Saving to: {a.out}")
    print(f"Converted INT8 layers: {converted_count}")
    print(f"  ConvRot Linear: {convrot_linear}, ConvRot Conv2d: {convrot_conv2d}, "
          f"plain INT8: {plain_int8_count}")
    print(f"Kept original dtype: {keep_count}")

    save_file(new_state_dict, a.out, metadata=metadata)
    print("Done!")


if __name__ == "__main__":
    main()

"""ConvRot INT8 converter for generic safetensors (CLIP / LLM / ControlNet).

Converts all 2D .weight tensors into ComfyUI-native ConvRot INT8
(Hadamard rotation + per-out-channel INT8 quantization + comfy_quant stamp).
Non-2D tensors (embeddings, layernorms, biases) are kept as-is.

ComfyUI natively supports ConvRot INT8 (int8_tensorwise + comfy_quant),
so the output loads with the standard ComfyUI loaders (UNet / CLIP /
ControlNet) without any custom node.

Output layout:
    <layer>.weight           int8
    <layer>.weight_scale     float32  [out, 1]
    <layer>.comfy_quant      uint8 JSON  {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":N}
    _quantization_metadata   {"format_version":"1.0","layers":{...}}

Usage:
  python convert_clip_convrot_int8.py --model model.safetensors --output model_int8.safetensors
  (--no-convrot for plain INT8 without rotation; --groupsize 256 default, power of 4)
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def build_hadamard(size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Normalized regular Hadamard matrix (power of 4)."""
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1],
         [1, 1, -1, 1],
         [1, -1, 1, 1],
         [-1, 1, 1, 1]],
        dtype=dtype,
    )
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def convrot_group_size(n: int, preferred: int = 256) -> int | None:
    """Largest power-of-4 group size <= preferred that divides n."""
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def rotate_weight(weight: torch.Tensor, h: torch.Tensor, gs: int) -> torch.Tensor:
    """W_rot = W @ H^T (group-wise along in_features)."""
    out_f, in_f = weight.shape
    if in_f % gs != 0:
        raise ValueError(f"in_features {in_f} not divisible by group size {gs}")
    g = in_f // gs
    return torch.matmul(weight.view(out_f, g, gs), h.T.to(weight.dtype)).reshape(weight.shape)


def quantize_int8_rowwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-out-channel INT8 with scale [out, 1]."""
    amax = w.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
    scale = amax / 127.0
    q = (w / scale.to(w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def _encode_meta(config: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(config, separators=(",", ":")).encode("utf-8")), dtype=torch.uint8)


def convert(model_path: str, output_path: str, enable_convrot: bool = True, groupsize: int = 256):
    print(f"Loading: {model_path}")
    sd = load_file(model_path)
    print(f"  keys: {len(sd)}")

    new_sd = {}
    meta_layers = {}
    convrot_count = 0
    plain_count = 0
    skip_count = 0
    fallback_list: list[str] = []

    h_cache: dict[int, torch.Tensor] = {}

    for key, tensor in tqdm(sorted(sd.items()), desc="Converting"):
        is_2d_weight = (
            key.endswith(".weight")
            and tensor.ndim == 2
            and tensor.dtype in (torch.float16, torch.float32, torch.bfloat16)
        )

        if not is_2d_weight:
            new_sd[key] = tensor
            skip_count += 1
            continue

        w = tensor.float()
        out_f, in_f = w.shape
        module_key = key[:-len(".weight")]

        if enable_convrot:
            gs = convrot_group_size(in_f, groupsize)
            if gs is not None:
                if gs not in h_cache:
                    h_cache[gs] = build_hadamard(gs)
                w_rot = rotate_weight(w, h_cache[gs], gs)
                q, scale = quantize_int8_rowwise(w_rot)
                config = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": gs}
                new_sd[key] = q
                new_sd[f"{module_key}.weight_scale"] = scale
                new_sd[f"{module_key}.comfy_quant"] = _encode_meta(config)
                meta_layers[module_key] = config
                convrot_count += 1
                continue
            else:
                fallback_list.append(key)

        # Plain INT8 (no rotation)
        q, scale = quantize_int8_rowwise(w)
        config = {"format": "int8_tensorwise"}
        new_sd[key] = q
        new_sd[f"{module_key}.weight_scale"] = scale
        new_sd[f"{module_key}.comfy_quant"] = _encode_meta(config)
        meta_layers[module_key] = config
        plain_count += 1

    metadata = {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": meta_layers}
        )
    }

    for k in fallback_list:
        print(f"  [WARN] {k}: in_features not power-of-4 divisible, plain INT8")

    print(f"\nSaving: {output_path}")
    print(f"  ConvRot INT8: {convrot_count}")
    print(f"  Plain INT8:   {plain_count}")
    print(f"  Kept as-is:  {skip_count}")
    save_file(new_sd, output_path, metadata=metadata)

    in_size = os.path.getsize(model_path) / (1024 ** 3)
    out_size = os.path.getsize(output_path) / (1024 ** 3)
    print(f"  Input:  {in_size:.2f} GB")
    print(f"  Output: {out_size:.2f} GB")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ConvRot INT8 converter for CLIP/LLM/ControlNet safetensors")
    parser.add_argument("--model", required=True, help="Input .safetensors path")
    parser.add_argument("--output", required=True, help="Output .safetensors path")
    parser.add_argument("--no-convrot", action="store_true", help="Plain INT8 without ConvRot rotation")
    parser.add_argument("--groupsize", type=int, default=256, help="Hadamard group size (power of 4, default 256)")
    args = parser.parse_args()
    if args.groupsize < 4 or (args.groupsize & (args.groupsize - 1)) != 0 or math.log(args.groupsize, 4) % 1 != 0:
        parser.error(f"--groupsize must be a power of 4, got {args.groupsize}")
    convert(args.model, args.output, enable_convrot=not args.no_convrot, groupsize=args.groupsize)
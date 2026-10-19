# -*- coding: utf-8 -*-
"""CLIP-SAE-ViT-L-14 (FP32) ConvRot INT8 converter with FP16 protection.

Derived from measured risk metrics (kurtosis / outlier_ratio) on
CLIP-SAE-ViT-L-14-FP32.safetensors (2026-10-19 measurement, all 590 keys,
221 finite 2D Linear layers scanned with the same method as the SDXL CLIP /
Flan-T5 / Qwen3 records).

Protected (kept FP16), structural & model-independent as measured:
  text_model.embeddings.*             (token_embedding 1 + position_embedding 1)
  text_model.encoder.layers.*.mlp.fc1 (all 12)
  text_model.encoder.layers.*.self_attn.q_proj (all 12)
  text_projection.weight               (kurt 564.7 - tail member)
  visual_projection.weight             (kurt 533.9 - tail member)
  vision_model.encoder.layers.*.self_attn.q_proj.weight (all 24 - measured kurt up to 26,701)
  vision_model.encoder.layers.*.mlp.fc1.weight          (all 24 - measured kurt up to 1,951)

Everything else is FULL ConvRot INT8.

Usage:
  python clip/protect_sae_vit_l.py --input <FP32> --output <out>
"""
from __future__ import annotations

import argparse
import json
import math
import os
from fnmatch import fnmatch as _fnmatch

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_GROUP = 256

# --- protection rule --------------------------------------------------------
_T_PROTECT_PATTERNS = [
    "text_model.embeddings.*.weight",
    "text_model.encoder.layers.*.mlp.fc1.weight",
    "text_model.encoder.layers.*.self_attn.q_proj.weight",
    "text_projection.weight",
    "visual_projection.weight",
    "vision_model.encoder.layers.*.self_attn.q_proj.weight",
    "vision_model.encoder.layers.*.mlp.fc1.weight",
]


def is_protected_sae_weight(key: str) -> bool:
    return any(_fnmatch(key, p) for p in _T_PROTECT_PATTERNS)


# --- pack math --------------------------------------------------------------

def build_hadamard(size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]], dtype=dtype)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def convrot_group_size(n: int, preferred: int = 256) -> int | None:
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def rotate_weight(weight: torch.Tensor, h: torch.Tensor, gs: int) -> torch.Tensor:
    out_f, in_f = weight.shape
    if in_f % gs != 0:
        raise ValueError(f"in_features {in_f} not divisible by group size {gs}")
    g = in_f // gs
    return torch.matmul(weight.view(out_f, g, gs), h.T.to(weight.dtype)).reshape(weight.shape)


def quantize_int8_rowwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = w.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
    scale = amax / 127.0
    q = (w / scale.to(w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def _encode_meta(config: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(config, separators=(",", ":")).encode("utf-8")), dtype=torch.uint8)


def convert(model_path: str, output_path: str, enable_convrot: bool = True, groupsize: int = 256):
    print(f"Loading (streamed): {model_path}")
    src = safe_open(model_path, framework="pt", device="cpu")
    src_keys = sorted(list(src.keys()))
    print(f"  raw keys: {len(src_keys)}")

    new_sd = {}
    meta_layers = {}
    convrot_count = 0
    plain_count = 0
    skip_count = 0
    protected_count = 0
    protected_list: list[str] = []

    h_cache: dict[int, torch.Tensor] = {}

    for key in src_keys:
        tensor = src.get_tensor(key)
        is_2d_weight = (
            key.endswith(".weight")
            and tensor.ndim == 2
            and tensor.dtype in (torch.float16, torch.float32, torch.bfloat16)
        )
        if not is_2d_weight:
            new_sd[key] = tensor
            skip_count += 1
            continue

        if is_protected_sae_weight(key):
            new_sd[key] = tensor
            protected_count += 1
            protected_list.append(key)
            continue

        w = tensor.float()
        out_f, in_f = w.shape
        module_key = key[: -len(".weight")]

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
                print(f"  [WARN] {key}: in_features not power-of-4 divisible, plain INT8")

        q, scale = quantize_int8_rowwise(w)
        config = {"format": "int8_tensorwise"}
        new_sd[key] = q
        new_sd[f"{module_key}.weight_scale"] = scale
        new_sd[f"{module_key}.comfy_quant"] = _encode_meta(config)
        meta_layers[module_key] = config
        plain_count += 1

    metadata = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": meta_layers})}

    print(f"\nProtected (kept FP16): {protected_count}")
    for k in protected_list[:10]:
        print(f"  {k}")
    if protected_count > 10:
        print(f"  ... ({protected_count - 10} more)")

    print(f"\nSaving: {output_path}")
    print(f"  ConvRot INT8:      {convrot_count}")
    print(f"  Plain INT8:        {plain_count}")
    print(f"  FP16 protected:    {protected_count}")
    print(f"  Kept as-is (1D):   {skip_count}")
    save_file(new_sd, output_path, metadata=metadata)

    in_s = os.path.getsize(model_path) / (1024 ** 3)
    out_s = os.path.getsize(output_path) / (1024 ** 3)
    print(f"  Input:  {in_s:.2f} GB")
    print(f"  Output: {out_s:.2f} GB")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP-SAE-ViT-L-14 ConvRot INT8 converter with FP16 protection")
    parser.add_argument("--input", required=True, help="Input .safetensors path (FP32)")
    parser.add_argument("--output", required=True, help="Output .safetensors path")
    parser.add_argument("--no-convrot", action="store_true", help="Plain INT8 without ConvRot rotation")
    parser.add_argument("--groupsize", type=int, default=256, help="Hadamard group size (power of 4, default 256)")
    args = parser.parse_args()
    if args.groupsize < 4 or (args.groupsize & (args.groupsize - 1)) != 0 or math.log(args.groupsize, 4) % 1 != 0:
        parser.error(f"--groupsize must be a power of 4, got {args.groupsize}")
    convert(args.input, args.output, enable_convrot=not args.no_convrot, groupsize=args.groupsize)

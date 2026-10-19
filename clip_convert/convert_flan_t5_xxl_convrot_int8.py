"""Flan-T5-XXL CLIP ConvRot INT8 conversion with FP16 protection (rule-based).

Rule derived from measured risk metrics (kurtosis / outlier_ratio = absmax/std)
on Flan-T5-XXL (flan_t5_xxl_fp16.safetensors, 2026-10-19 measurement, all 168
encoder Linear layers scanned with the same method as the SDXL CLIP record):
the extreme-tail layers are structurally fixed, so the protected set is explicit.

Protected (kept FP16) layers:
  encoder.block.0.layer.0.SelfAttention.k.weight   (ratio 469 / kurt 10,080)
  encoder.block.0.layer.0.SelfAttention.v.weight   (ratio 637 / kurt 65,127)
  encoder.block.1.layer.1.DenseReluDense.wi_0.weight (ratio 418 / kurt 1,136)
  encoder.block.1.layer.1.DenseReluDense.wi_1.weight (ratio 466 / kurt 1,591)
  encoder.block.2.layer.1.DenseReluDense.wi_0.weight (ratio 256 / kurt 103)
  shared.weight / lm_head.weight                    (embedding / unembedding)

Everything else is FULL ConvRot INT8 (rotate + per-channel scale + comfy_quant
stamp), pack math identical to the generic converter
(clip_convert/convert_clip_convrot_int8.py).

Usage:
  python clip/convert_flan_t5_xxl_convrot_int8.py --input flan_t5_xxl_fp16.safetensors --output flan_t5_xxl_convrot_int8.safetensors
"""
from __future__ import annotations

import argparse
import argparse
import json
import math
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_GROUP = 256

# --- protection rule (measured 2026-10-19, structural) ----------------------
_T5_PROTECT_EXACT = frozenset(
    {
        "encoder.block.0.layer.0.SelfAttention.k.weight",
        "encoder.block.0.layer.0.SelfAttention.v.weight",
        "encoder.block.1.layer.1.DenseReluDense.wi_0.weight",
        "encoder.block.1.layer.1.DenseReluDense.wi_1.weight",
        "encoder.block.2.layer.1.DenseReluDense.wi_0.weight",
        "shared.weight",
        "lm_head.weight",
    }
)


def is_protected_t5_weight(key: str) -> bool:
    """True if this T5 weight must stay FP16 (rule-based protection set)."""
    return key in _T5_PROTECT_EXACT


# --- pack math (identical to the generic converter) -------------------------

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


# --- converter ---------------------------------------------------------------

def convert(model_path: str, output_path: str, enable_convrot: bool = True, groupsize: int = 256):
    """Streaming conversion: the FP16 base is 21.4 GB and the box has ~61 GB RAM
    (free often <12 GB), so tensors are pulled per-key via safe_open instead of
    one full load_file. Output accumulates only new (int8/scale meta) tensors plus
    copied non-2D / protected tensors.
    """
    print(f"Loading (streamed): {model_path}")
    src = safe_open(model_path, framework="pt", device="cpu")
    src_keys = sorted(list(src.keys()))
    print(f"  raw keys: {len(src_keys)}")

    new_sd = {}
    new_sd_names_file = None  # legacy removed
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

        if is_protected_t5_weight(key):
            new_sd[key] = tensor  # stays FP16, untouched
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

        # Plain INT8 (no rotation)
        q, scale = quantize_int8_rowwise(w)
        config = {"format": "int8_tensorwise"}
        new_sd[key] = q
        new_sd[f"{module_key}.weight_scale"] = scale
        new_sd[f"{module_key}.comfy_quant"] = _encode_meta(config)
        meta_layers[module_key] = config
        plain_count += 1

    metadata = {
        "_quantization_metadata": json.dumps({"format_version": "1.0", "layers": meta_layers})
    }

    print("\nProtected (kept FP16):")
    for k in protected_list:
        print(f"  {k}")

    print(f"\nSaving: {output_path}")
    print(f"  ConvRot INT8:      {convrot_count}")
    print(f"  Plain INT8:        {plain_count}")
    print(f"  FP16 protected:    {protected_count}")
    print(f"  Kept as-is (1D):   {skip_count}")
    save_file(new_sd, output_path, metadata=metadata)

    in_size = os.path.getsize(model_path) / (1024 ** 3)
    out_size = os.path.getsize(output_path) / (1024 ** 3)
    print(f"  Input:  {in_size:.2f} GB")
    print(f"  Output: {out_size:.2f} GB")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flan-T5-XXL ConvRot INT8 converter with FP16 protection")
    parser.add_argument("--input", required=True, help="Input .safetensors path (FP16)")
    parser.add_argument("--output", required=True, help="Output .safetensors path")
    parser.add_argument("--no-convrot", action="store_true", help="Plain INT8 without ConvRot rotation")
    parser.add_argument("--groupsize", type=int, default=256, help="Hadamard group size (power of 4, default 256)")
    args = parser.parse_args()
    if args.groupsize < 4 or (args.groupsize & (args.groupsize - 1)) != 0 or math.log(args.groupsize, 4) % 1 != 0:
        parser.error(f"--groupsize must be a power of 4, got {args.groupsize}")
    convert(args.input, args.output, enable_convrot=not args.no_convrot, groupsize=args.groupsize)

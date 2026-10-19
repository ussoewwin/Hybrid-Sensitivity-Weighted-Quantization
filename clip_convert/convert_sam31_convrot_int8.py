# -*- coding: utf-8 -*-
"""SAM 3.1 Multiplex ConvRot INT8 converter with measured FP16 protection.

SAM 3 / SAM 3.1 are foundation segmentation models - a single standard weight
layout each, like text encoders - so they get an individually built converter
with measured FP16 protection rather than the universal HSWQ engine (which
targets per-checkpoint image-generation backbones).

Protection set = measured risk (kurtosis / outlier ratio = absmax/std) on all
522 finite 2D+4D weights of sam3.1_multiplex_fp16.safetensors (2026-10-20):

  dominant family: vision backbone qkv  (kurt up to 19,036, median 5,252)
  language backbone out_proj            (kurt up to 39,567, median 675)
  language backbone mlp.c_fc            (kurt up to 1,710, median 315)
  vision backbone mlp.fc1               (kurt median 459)
  cross-attention / mask-decoder attention projections (kurt 250-720)

The fused QKV in_proj tensors are split before quantization (the same
preprocessing the generic converter uses, and the loader expects). freqs_cis
buffers are dropped. SAM 3.1 keeps its (1024,1024) text_projection.

Usage:
  python clip_convert/convert_sam31_convrot_int8.py --input sam3.1_multiplex_fp16.safetensors --output sam3.1_multiplex_convrot_int8_protected.safetensors
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_GROUP = 256

# --- protection rule (measured 2026-10-20, structural) ----------------------
# Any key matching one of these regexes is kept FP16 (protected).
_PROTECT_REGEXES = [
    # detector vision backbone: attention qkv (dominant family) + mlp fc1
    re.compile(r"^detector\.backbone\.vision_backbone\.trunk\.blocks\.\d+\.attn\.qkv\.weight$"),
    re.compile(r"^detector\.backbone\.vision_backbone\.trunk\.blocks\.\d+\.mlp\.fc1\.weight$"),
    # detector language backbone: attention out_proj + mlp c_fc (and c_proj tail)
    re.compile(r"^detector\.backbone\.language_backbone\.encoder\.transformer\.resblocks\.\d+\.attn\.out_proj\.weight$"),
    re.compile(r"^detector\.backbone\.language_backbone\.encoder\.transformer\.resblocks\.\d+\.mlp\.c_fc\.weight$"),
    # tracker mask-decoder / transformer attention projections (kurt 250-720)
    re.compile(r"^tracker\.model\.interactive_sam_mask_decoder\.transformer\.final_attn_token_to_image\.v_proj\.weight$"),
    re.compile(r"^tracker\.model\.sam_mask_decoder\.transformer\.final_attn_token_to_image\.v_proj\.weight$"),
    re.compile(r"^tracker\.model\.transformer\.encoder\.layers\.\d+\.image_cross_attn_q_proj\.weight$"),
    re.compile(r"^tracker\.model\.interactive_sam_mask_decoder\.transformer\.layers\.\d+\.cross_attn_image_to_token\.v_proj\.weight$"),
    re.compile(r"^tracker\.model\.obj_ptr_tpos_proj\.weight$"),
    re.compile(r"^tracker\.model\.interactive_sam_mask_decoder\.transformer\.layers\.\d+\.self_attn\.q_proj\.weight$"),
    re.compile(r"^tracker\.model\.transformer\.encoder\.layers\.\d+\.cross_attn_out_proj\.weight$"),
    re.compile(r"^tracker\.model\.maskmem_backbone\.fuser\.layers\.\d+\.pwconv1\.weight$"),
    re.compile(r"^tracker\.model\.interactive_sam_mask_decoder\.transformer\.layers\.\d+\.cross_attn_token_to_image\.k_proj\.weight$"),
    re.compile(r"^detector\.geometry_encoder\.points_pool_project\.weight$"),
    re.compile(r"^tracker\.model\.interactive_sam_mask_decoder\.transformer\.layers\.\d+\.cross_attn_token_to_image\.v_proj\.weight$"),
    re.compile(r"^tracker\.model\.transformer\.encoder\.layers\.\d+\.self_attn_v_proj\.weight$"),
]


def is_protected_sam_weight(key: str) -> bool:
    return any(rx.match(key) for rx in _PROTECT_REGEXES)


# --- pack math (identical to the generic converter) -------------------------

def build_hadamard(size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
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


# --- SAM preprocessing (same as the generic converter) ----------------------

def _detect_sam_version(keys) -> str | None:
    if not any(k.startswith("detector.") for k in keys):
        return None
    if any("propagation_convs" in k for k in keys):
        return "SAM31"
    if any("sam2_convs" in k for k in keys) or any("vision_backbone.convs.3" in k for k in keys):
        return "SAM3"
    return None


def _preprocess_sam(keys, tensors: dict) -> dict:
    version = _detect_sam_version(keys)
    drop_text_projection = (version == "SAM3")
    out = {}
    for k, v in tensors.items():
        if ".attn.freqs_cis" in k:
            continue
        if k.startswith("tracker.model."):
            k = "tracker." + k[len("tracker.model."):]
        if "sam_mask_decoder.transformer." in k:
            k = (k.replace(".mlp.lin1.", ".mlp.0.")
                  .replace(".mlp.lin2.", ".mlp.2.")
                  .replace(".norm_final_attn.", ".norm_final."))
        if drop_text_projection and "encoder.text_projection" in k:
            continue
        if k.endswith((".in_proj_weight", ".in_proj_bias")):
            base, suffix = k.rsplit(".in_proj_", 1)
            s = ".weight" if suffix == "weight" else ".bias"
            d = v.shape[0] // 3
            out[f"{base}.q_proj{s}"] = v[:d].clone()
            out[f"{base}.k_proj{s}"] = v[d:2 * d].clone()
            out[f"{base}.v_proj{s}"] = v[2 * d:].clone()
            continue
        out[k] = v
    return out


# --- converter --------------------------------------------------------------

def convert(input_path: str, output_path: str, group_size: int = DEFAULT_GROUP):
    print(f"Loading (streamed): {input_path}")
    src = safe_open(input_path, framework="pt", device="cpu")
    keys = sorted(src.keys())
    print(f"  raw keys: {len(keys)}")

    tensors = {k: src.get_tensor(k) for k in keys}
    version = _detect_sam_version(keys)
    print(f"  detected: {version}")
    tensors = _preprocess_sam(keys, tensors)
    print(f"  preprocessed keys: {len(tensors)}")

    out = {}
    meta_layers = {}
    n_convrot = n_prot = n_kept = 0
    protected_list = []
    fallback_list = []
    h_cache = {}

    for key in sorted(tensors):
        tensor = tensors[key]
        is_2d_weight = (
            key.endswith(".weight")
            and tensor.ndim == 2
            and tensor.dtype in (torch.float16, torch.float32, torch.bfloat16)
        )
        if not is_2d_weight:
            out[key] = tensor
            n_kept += 1
            continue
        if is_protected_sam_weight(key):
            out[key] = tensor
            n_prot += 1
            protected_list.append(key)
            continue
        w = tensor.float()
        out_f, in_f = w.shape
        module_key = key[: -len(".weight")]
        gs = convrot_group_size(in_f, group_size)
        if gs is None:
            fallback_list.append(key)
            q, scale = quantize_int8_rowwise(w)
            conf = {"format": "int8_tensorwise"}
        else:
            if gs not in h_cache:
                h_cache[gs] = build_hadamard(gs)
            w_rot = rotate_weight(w, h_cache[gs], gs)
            q, scale = quantize_int8_rowwise(w_rot)
            conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": int(gs)}
        out[key] = q
        out[f"{module_key}.weight_scale"] = scale
        out[f"{module_key}.comfy_quant"] = _encode_meta(conf)
        meta_layers[module_key] = conf
        n_convrot += 1

    metadata = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": meta_layers})}

    print(f"\nProtected (kept FP16): {n_prot}")
    for k in protected_list[:12]:
        print(f"  {k}")
    if n_prot > 12:
        print(f"  ... ({n_prot - 12} more)")
    for k in fallback_list:
        print(f"  [WARN] {k}: in_features not power-of-4 divisible, plain INT8")

    print(f"\nSaving: {output_path}")
    print(f"  ConvRot INT8:    {n_convrot}")
    print(f"  FP16 protected:  {n_prot}")
    print(f"  Kept as-is:      {n_kept}")
    save_file(out, output_path, metadata=metadata)
    print(f"  Input:  {os.path.getsize(input_path) / (1024**3):.2f} GB")
    print(f"  Output: {os.path.getsize(output_path) / (1024**3):.2f} GB")
    print("Done!")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SAM 3.1 Multiplex ConvRot INT8 converter with measured FP16 protection")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--groupsize", type=int, default=DEFAULT_GROUP)
    args = ap.parse_args()
    convert(args.input, args.output, group_size=args.groupsize)

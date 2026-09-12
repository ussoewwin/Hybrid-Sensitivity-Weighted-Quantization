# -*- coding: utf-8 -*-
"""Reverse hybrid INT8 converter for SDXL: FP16 baseline -> K lowest-impact layers ConvRot INT8.

SDXL counterpart of Z_Image/gen_reverse_nvfp4.py, INT8 variant. Method: reverse hybrid
(see md/diag_impact_trajectory_sensitivity_technical_guide.md).

  - Input: the FP16 baseline checkpoint and the impact json from diag_impact_sdxl.py.
  - The K lowest-impact layers (ascending impact) are converted to ConvRot INT8:
      W_rot = W @ H^T  ->  per-channel INT8 (Linear: rowwise [out,1] /
      Conv2d: channelwise [out,1,1,1])  ->  the rotated INT8 weight is stored as-is
      (same kernel as native_convert_int8_sdxl.py; no inverse rotation).
  - Every other layer is left untouched: FP16, same dtype and size as the baseline.

Output is the hswq convrot int8 mixed checkpoint:
    converted layers : .weight int8 + .weight_scale + .comfy_quant
                       {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":N}
    kept layers      : the baseline float weight copied as-is

ComfyUI mixed-precision ops select the quantized path per layer from the .comfy_quant marker, so a
single file carries both the FP16-kept and the ConvRot INT8 layers.

Usage:
    python gen_reverse_int8_sdxl.py <K> <out_name.safetensors> <base.safetensors> <impact.json> \
        [--out-dir <output-dir>] [--groupsize 256]

Validation:
    python benchmark/sdxl_int8_traj_compare.py --fp16 <base> --int8 <out> ...
"""
import argparse
import json
import math
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# --- ConvRot kernel (same math as native_convert_int8_sdxl.py; inlined) ---

def convrot_group_size_for_features(n: int, preferred: int = 256) -> int | None:
    """Largest power-of-4 group size <= preferred that divides n (or None)."""
    if n < 4:
        return None
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def build_hadamard(size: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Normalized regular Hadamard (power-of-4), same as comfy_kitchen ConvRot."""
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
    cur = 4
    while cur < size:
        h_matrix = torch.kron(h_matrix, h4)
        cur *= 4
    return h_matrix / (size**0.5)


def rotate_weight(weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int) -> torch.Tensor:
    """Offline Linear: W_rot = W @ H^T (group-wise). Matches comfy_kitchen._rotate_weight."""
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(f"in_features {in_features} not divisible by group_size {group_size}")
    group_count = in_features // group_size
    grouped = weight.reshape(out_features, group_count, group_size)
    return torch.matmul(grouped, h_matrix.T.to(weight.dtype)).reshape(weight.shape)


def rotate_weight_conv2d(weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int) -> torch.Tensor:
    """Offline Conv2d: rotate along in_channels. weight (O, I, kH, kW)."""
    if weight.ndim != 4:
        raise ValueError(f"Conv2d weight must be 4D, got ndim={weight.ndim}")
    out_c, in_c, k_h, k_w = weight.shape
    flat = weight.permute(0, 2, 3, 1).contiguous().reshape(-1, in_c)
    flat_rot = rotate_weight(flat, h_matrix, group_size)
    return flat_rot.reshape(out_c, k_h, k_w, in_c).permute(0, 3, 1, 2).contiguous()


def quantize_int8_rowwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel INT8 for Linear: weight_scale [out, 1]."""
    abs_max = w.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
    scale = abs_max / 127.0
    q = (w / scale.to(w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def quantize_int8_channelwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-out-channel INT8 for Conv2d: weight_scale [out, 1, 1, 1]."""
    reduce_dims = tuple(range(1, w.dim()))
    amax = torch.clamp(w.abs().amax(dim=reduce_dims).reshape(-1), min=1e-6)
    scale = amax / 127.0
    if w.dim() == 4:
        scale_view = scale.view(-1, 1, 1, 1)
    elif w.dim() == 2:
        scale_view = scale.view(-1, 1)
    else:
        raise ValueError(f"unsupported weight ndim={w.dim()}")
    q = (w / scale_view.to(w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, scale_view.to(torch.float32)


def _encode_comfy_quant(config: dict) -> torch.Tensor:
    """Compact JSON bytes as a U8 tensor (ComfyUI layout)."""
    return torch.tensor(
        list(json.dumps(config, separators=(",", ":")).encode("utf-8")),
        dtype=torch.uint8,
    )


# --- SDXL UNet boundary layers (same exclusion set as diag_impact_sdxl.py) ---

_PROTECT_PATTERNS = (
    "conv_in.",
    "conv_out.",
    "time_embed.",
    "add_embedding.",
    "label_emb.",
)


def is_protected(module_name: str) -> bool:
    return any(p in module_name for p in _PROTECT_PATTERNS)


def parse_args():
    ap = argparse.ArgumentParser(
        description="FP16 baseline -> K lowest-impact layers ConvRot INT8 (hswq mixed checkpoint)"
    )
    ap.add_argument("k", type=int, help="number of layers (ascending impact) to convert to ConvRot INT8")
    ap.add_argument("out_name", help="output filename (created under the output directory)")
    ap.add_argument("base", help="FP16 baseline SDXL checkpoint (full ckpt)")
    ap.add_argument("impact", help="impact json from diag_impact_sdxl.py")
    ap.add_argument("--out-dir", default=".", help="output directory")
    ap.add_argument("--groupsize", type=int, default=256,
                    help="ConvRot Hadamard group size (power of 4, default 256)")
    return ap.parse_args()


def main():
    a = parse_args()
    out = os.path.join(a.out_dir, a.out_name)

    with open(a.impact, encoding="utf-8") as f:
        imp = json.load(f)["impacts"]
    # ascending = safest first; diag keys are named_modules names (with diffusion_model.)
    ranked = []
    for k, v in sorted(imp.items(), key=lambda kv: kv[1]):
        if isinstance(v, float) and math.isnan(v):
            continue
        if k.endswith(".weight"):
            k = k[: -len(".weight")]
        ranked.append(k)
    print(f"ranked layers available: {len(ranked)}")

    print(f"loading baseline: {a.base}")
    with safe_open(os.path.abspath(a.base), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        raw_meta = f.metadata() or {}
        sd = {k: f.get_tensor(k) for k in keys}

    prefix = ""
    for cand in ("model.diffusion_model.", "diffusion_model."):
        if any(k.startswith(cand) for k in keys):
            prefix = cand
            break
    print(f"UNet key prefix: {prefix!r}")

    def module_to_sd_key(mod_name: str) -> str | None:
        """Map a diag module name (named_modules) to the checkpoint module key."""
        for pre in ("model.diffusion_model.", "diffusion_model."):
            if pre + mod_name + ".weight" in sd:
                return pre + mod_name
            bare = mod_name
            if bare.startswith("diffusion_model."):
                bare = bare[len("diffusion_model."):]
                if pre + bare + ".weight" in sd:
                    return pre + bare
        return None

    quant_meta_layers = {}
    converted = 0
    skipped_protected = 0
    skipped_not_found = 0
    skipped_shape = 0

    for name in ranked[: a.k]:
        module_key = module_to_sd_key(name)
        if module_key is None:
            skipped_not_found += 1
            print(f"  SKIP (not in sd): {name}")
            continue
        wk = module_key + ".weight"
        w = sd[wk]
        if w.ndim not in (2, 4) or w.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            skipped_shape += 1
            continue
        if is_protected(name):
            skipped_protected += 1
            print(f"  SKIP (boundary layer): {name}")
            continue

        gs = convrot_group_size_for_features(int(w.shape[1]), a.groupsize)
        if gs is None:
            skipped_shape += 1
            print(f"  SKIP (no eligible group size): {name} in={w.shape[1]}")
            continue

        h = build_hadamard(gs, device="cpu", dtype=torch.float32)
        wf = w.float()
        # rotate then per-channel INT8 (same kernel as the shipped converter; no inverse rotation)
        if w.ndim == 2:
            w_rot = rotate_weight(wf, h, gs)
            q, scale = quantize_int8_rowwise(w_rot)
        else:
            w_rot = rotate_weight_conv2d(wf, h, gs)
            q, scale = quantize_int8_channelwise(w_rot)

        del sd[wk]
        sd[wk] = q
        sd[module_key + ".weight_scale"] = scale
        conf = {
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": int(gs),
        }
        sd[module_key + ".comfy_quant"] = _encode_comfy_quant(conf)
        quant_meta_layers[module_key] = conf
        converted += 1
        if converted % 25 == 0 or converted == a.k:
            print(f"  [{converted}/{a.k}] last: {name}  {tuple(w.shape)}")

    print(f"converted: {converted}, protected-skip: {skipped_protected}, "
          f"not-found-skip: {skipped_not_found}, shape-skip: {skipped_shape}")

    metadata = dict(raw_meta)
    metadata["_quantization_metadata"] = json.dumps(
        {"format_version": "1.0", "layers": quant_meta_layers},
        separators=(",", ":"),
    )
    metadata["hswq_reverse_int8"] = json.dumps({
        "base": os.path.abspath(a.base),
        "impact": os.path.abspath(a.impact),
        "k_requested": a.k,
        "converted": converted,
        "groupsize": a.groupsize,
    })

    print(f"saving: {out}")
    save_file(sd, out, metadata=metadata)
    size_gb = os.path.getsize(out) / 1e9
    print(f"saved: {out}  {size_gb:.2f} GB (decimal)")


if __name__ == "__main__":
    main()

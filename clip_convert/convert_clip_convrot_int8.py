"""ConvRot INT8 converter for generic safetensors (CLIP / LLM / ControlNet / SAM / UNet).

Converts all 2D .weight tensors into ComfyUI-native ConvRot INT8
(Hadamard rotation + per-out-channel INT8 quantization + comfy_quant stamp).
Non-2D tensors (embeddings, layernorms, biases, Conv2d) are kept as-is.
Fused QKV projections (.in_proj_weight / .in_proj_bias) are cleanly split
and quantized into q_proj, k_proj, v_proj for seamless native ComfyUI loading.

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


def _detect_sam_version(sd: dict) -> str | None:
    """Auto-detect SAM3 (3) vs SAM3.1 from checkpoint keys.

    Returns "SAM3", "SAM31", or None (not a SAM3-family checkpoint).
    SAM3 (non-multiplex) ships sam2_convs / a 4-level FPN; SAM3.1 (multiplex)
    ships propagation_convs / interactive_convs with a 3-level FPN.
    """
    keys = list(sd.keys())
    if not any(k.startswith("detector.") for k in keys):
        return None
    if any("propagation_convs" in k for k in keys):
        return "SAM31"
    if any("sam2_convs" in k for k in keys) or any("vision_backbone.convs.3" in k for k in keys):
        return "SAM3"
    return None


def _common_sam_key_remap(sd: dict[str, torch.Tensor], drop_text_projection: bool) -> dict[str, torch.Tensor]:
    """Shared SAM key remapping (tracker remap, in_proj split, decoder key names).

    drop_text_projection: True for SAM3 (unused (1024,512) projection), False for
    SAM3.1 (keeps its (1024,1024) projection).
    """
    out_sd: dict[str, torch.Tensor] = {}

    for k, v in sd.items():
        # Remove per-block freqs_cis buffers (computed dynamically)
        if ".attn.freqs_cis" in k:
            continue

        # Remap tracker.model.* -> tracker.*
        if k.startswith("tracker.model."):
            k = "tracker." + k[len("tracker.model."):]

        # Remap tracker SAM decoder transformer key names
        if "sam_mask_decoder.transformer." in k:
            k = (
                k.replace(".mlp.lin1.", ".mlp.0.")
                .replace(".mlp.lin2.", ".mlp.2.")
                .replace(".norm_final_attn.", ".norm_final.")
            )

        # text_projection handling (branch per model version)
        if drop_text_projection and "encoder.text_projection" in k:
            continue

        # Split fused QKV in_proj_weight / in_proj_bias
        if k.endswith((".in_proj_weight", ".in_proj_bias")):
            base, suffix = k.rsplit(".in_proj_", 1)
            s = ".weight" if suffix == "weight" else ".bias"
            d = v.shape[0] // 3
            out_sd[f"{base}.q_proj{s}"] = v[:d].clone()
            out_sd[f"{base}.k_proj{s}"] = v[d:2*d].clone()
            out_sd[f"{base}.v_proj{s}"] = v[2*d:].clone()
            continue

        out_sd[k] = v

    return out_sd


def _preprocess_sam3(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """SAM3 (3, non-multiplex): drop the unused (1024,512) text_projection."""
    return _common_sam_key_remap(sd, drop_text_projection=True)


def _preprocess_sam31(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """SAM3.1 (multiplex): keep the (1024,1024) text_projection intact."""
    return _common_sam_key_remap(sd, drop_text_projection=False)


def _preprocess_sam_and_fused_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Route to the version-specific SAM preprocessor based on auto-detection."""
    version = _detect_sam_version(sd)
    if version == "SAM3":
        print(f"[SAM preprocess] detected SAM3 -> _preprocess_sam3 (drop text_projection)")
        return _preprocess_sam3(sd)
    if version == "SAM31":
        print(f"[SAM preprocess] detected SAM3.1 -> _preprocess_sam31 (keep text_projection)")
        return _preprocess_sam31(sd)
    return _common_sam_key_remap(sd, drop_text_projection=False)


def convert(model_path: str, output_path: str, enable_convrot: bool = True, groupsize: int = 256):
    print(f"Loading: {model_path}")
    sd = load_file(model_path)
    print(f"  raw keys: {len(sd)}")

    # Preprocess SAM / fused in_proj keys
    sd = _preprocess_sam_and_fused_keys(sd)
    print(f"  preprocessed keys: {len(sd)}")

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
    parser = argparse.ArgumentParser(description="ConvRot INT8 converter for CLIP/LLM/ControlNet/SAM/UNet safetensors")
    parser.add_argument("--model", required=True, help="Input .safetensors path")
    parser.add_argument("--output", required=True, help="Output .safetensors path")
    parser.add_argument("--no-convrot", action="store_true", help="Plain INT8 without ConvRot rotation")
    parser.add_argument("--groupsize", type=int, default=256, help="Hadamard group size (power of 4, default 256)")
    args = parser.parse_args()
    if args.groupsize < 4 or (args.groupsize & (args.groupsize - 1)) != 0 or math.log(args.groupsize, 4) % 1 != 0:
        parser.error(f"--groupsize must be a power of 4, got {args.groupsize}")
    convert(args.model, args.output, enable_convrot=not args.no_convrot, groupsize=args.groupsize)
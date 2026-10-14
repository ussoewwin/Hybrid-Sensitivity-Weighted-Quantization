"""Simple Qwen Image / Qwen Image Edit native ConvRot W4A8 (asym_w4a8_int8) converter.

Base: sdxl/native_convert_w4a8_sdxl.py + Krea2/native_convert_w4a8_krea2.py
(same kitchen pack route, same shape-critical protection, same fallback order).

Qwen-Image-Edit (qwnImageEdit_v16Bf16, ComfyUI image_model "qwen_image") is a
pure-Linear DiT: 60 transformer_blocks, no Conv2d anywhere. Measured routes on
the 1,934-tensor checkpoint:
  2D .weight  846 layers
    -> W4A8 (asym_w4a8_int8): 845 layers, 20,424.6M params (K%256==0, K%16==0)
    -> shape-critical INT8:    1 layer  (img_in, K=64)
  everything else: kept at original dtype (bias / RMS-LayerNorm / f32 buffer)

Linear (2D) layers go ComfyUI-native W4A8 when shapes allow:
  in_features % 256 == 0                  (CUDA fused kernel: convrot group 256 only)
  in_features % 16 == 0                   (quant group, fixed 16)
  Pack via comfy_kitchen.quantize_w4a8_int8_weight (bits=4, symmetric, codebook):
    weight            int8  [N, K/2]     packed nibbles, low nibble = even column
    weight_s_rel      fp8e4m3 [N, K/16]  per-group relative scale
    weight_s_channel  f32     [N]        per-out-channel scale
    weight_codebook   f32     [16]       Lloyd-Max table (per layer, optional)
    comfy_quant       {"format":"asym_w4a8_int8","group_size":16,"convrot_groupsize":256}
  Rotation happens inside the kitchen quantizer (fp32 input, ConvRot H per
  group of 256). Kitchen dequantize returns the ORIGINAL (unrotated) basis.

Shape-critical Linear: img_in is read verbatim by detect_unet_config
(comfy/model_detection.py:991 branch -> in_channels = img_in.weight.shape[1]).
Nibble-packing it would report in_channels=32 (K/2) and Qwen Image detection
fails. img_in stays a shape-preserving per-channel INT8 pack
(int8 [N, K] + weight_scale + int8_tensorwise stamp, no rotation), exactly the
SDXL treatment of label_emb.0.0 / attn2.to_k. txt_in (K=3584) is NOT read for
shape detection and goes W4A8.

ComfyUI loads this pack 100% stock (no HSWQ patches):
  any .comfy_quant key -> comfy.utils.detect_layer_quantization -> quant_config
  {"mixed_ops": True} -> comfy.ops.pick_operations -> mixed_precision_ops
  asym_w4a8_int8 is in comfy.ops._GROUPED_INT8_FORMATS (bits=4): weight_s_rel
  required, weight_s_channel / weight_codebook optional, packed-width check
  weight.shape[1] * 8 == orig_K * 4 passes with K/2 int8 storage.
__index_timestep_zero__ (2511 ref marker buffer) is kept verbatim.

No Card 1 bias correction in this converter: Qwen Image's text encoder is a
multi-GB Qwen2.5-VL that must be loaded to produce real calib contexts; the
16 GB local GPU cannot hold it alongside the 38 GB BF16 DiT. Rather than ship
an untested calib path, BC is intentionally absent (W4A8 bias stays untouched;
kitchen dequant is basis-exact so the stock bias is already consistent).

No post-convert bench: W4A8 packs load stock; wire
benchmark/qwen_image_edit_traj_compare.py separately when ordered.

Requires comfy_kitchen in the running interpreter (ComfyUI python_embeded).
Fails fast when missing - no re-implementation of the pack math here, so the
packed layout always matches the ComfyUI runtime decoder.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

_DEFAULT_GROUPSIZE = 256
# W4A8 quant group. Fixed 16: the config ComfyUI's fused CUDA requant supports
# (4-bit symmetric codebook g16, fp8 group scales) and the kitchen default.
_QUANT_GROUP_SIZE = 16

# Linear modules whose PACKED shape would corrupt ComfyUI architecture
# detection. detect_unet_config reads img_in.weight.shape[1] as in_channels
# for the qwen_image branch; nibble packing halves that K. Match rule: module
# key EQUALS the bare name (top-level 'img_in' has no dot prefix) or ENDS WITH
# the dotted suffix.
_SHAPE_CRITICAL_LINEAR_SUFFIXES = (".img_in",)


def _is_shape_critical(module_key: str) -> bool:
    return any(
        module_key == s.lstrip(".") or module_key.endswith(s)
        for s in _SHAPE_CRITICAL_LINEAR_SUFFIXES
    )

_NON_DIFFUSION_MARKERS: tuple[str, ...] = (
    "conditioner.",
    "cond_stage_model.",
    "text_encoders.",
    "text_encoder.",
    "text_encoder_2.",
    "text_encoder_3.",
    "text_model.",
    "text_projection",
    "logit_scale",
    "clip_l.",
    "clip_g.",
    "t5xxl.",
    "first_stage_model.",
    "vae.",
)


def _is_non_diffusion_key(key: str) -> bool:
    return any(marker in key for marker in _NON_DIFFUSION_MARKERS)


def _load_comfy_kitchen():
    """Import comfy_kitchen and return the module (fail fast when missing).

    The W4A8 pack math (ConvRot rotate + Lloyd-Max codebook + ALS group scales +
    nibble packing) must match the ComfyUI runtime decoder, so this converter
    calls the same comfy_kitchen API ComfyUI uses at load/requantize time
    instead of re-implementing it. No silent fallback.
    """
    try:
        import comfy_kitchen as ck
    except ImportError as exc:
        raise ImportError(
            "comfy_kitchen is required for W4A8 packing "
            "(run this converter with the ComfyUI python_embeded interpreter)"
        ) from exc
    for attr in ("quantize_w4a8_int8_weight", "dequantize_w4a8_int8_weight"):
        if not hasattr(ck, attr):
            raise ImportError(
                f"comfy_kitchen is too old: missing {attr} "
                f"(W4A8 packing needs comfy_kitchen >= 0.2.x with AsymW4A8Int8Layout)"
            )
    return ck


def _encode_comfy_quant(config: dict) -> torch.Tensor:
    return torch.tensor(
        list(json.dumps(config, separators=(",", ":")).encode("utf-8")),
        dtype=torch.uint8,
    )


def pack_channelwise(weight: torch.Tensor):
    """Per-out-channel INT8 (shape-critical pack; kitchen dequant shape)."""
    w = weight.float()
    reduce_dims = tuple(range(1, w.dim()))
    amax = torch.clamp(w.abs().amax(dim=reduce_dims).reshape(-1), min=1e-6)
    scale = amax / 127.0
    if w.dim() == 4:
        scale_view = scale.view(-1, 1, 1, 1)
        amax_view = amax.view(-1, 1, 1, 1)
    elif w.dim() == 2:
        scale_view = scale.view(-1, 1)
        amax_view = amax.view(-1, 1)
    else:
        raise ValueError(f"unsupported weight ndim={w.dim()} for shape-critical INT8")
    clamped = torch.clamp(w, -amax_view, amax_view)
    q = (clamped / scale_view).round().clamp(-127, 127).to(torch.int8)
    return q, scale_view.to(dtype=torch.float32)


def pack_tensorwise(weight: torch.Tensor):
    """Symmetric per-tensor INT8: scale = amax / 127."""
    w = weight.float()
    amax = max(float(w.abs().max().item()), 1e-6)
    scale = amax / 127.0
    q = (w / scale).round().clamp(-127, 127).to(torch.int8)
    return q, torch.tensor(scale, dtype=torch.float32)


def convrot_group_size_for_features(n: int, preferred: int) -> int | None:
    """Largest power-of-4 group size <= preferred that divides n (or None)."""
    if n < 4:
        return None
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def convert_to_w4a8(
    input_path: str,
    output_path: str,
    *,
    per_channel_int8: bool = False,
    enable_convrot: bool = True,
    group_size: int = _DEFAULT_GROUPSIZE,
    device: str = "cuda",
):
    ck = None
    if enable_convrot:
        ck = _load_comfy_kitchen()
        print(
            f"  [ConvRot W4A8] ON | Linear 2D -> asym_w4a8_int8 "
            f"(quant_group={_QUANT_GROUP_SIZE}, convrot_groupsize==256 only; "
            f"CUDA fused kernel restriction) | pack via "
            f"comfy_kitchen.quantize_w4a8_int8_weight (same API as runtime)"
        )
        print(
            "  [ConvRot W4A8] shape-critical img_in stays shape-preserving "
            "per-channel INT8 (detect_unet_config reads img_in.weight.shape[1] "
            "as in_channels; nibble packing would halve it)"
        )
    else:
        print("  [ConvRot W4A8] OFF (--no-convrot) | packing plain INT8 only")

    print(f"Loading model: {input_path}")
    state_dict = load_file(input_path)

    new_state_dict: dict[str, torch.Tensor] = {}
    quant_meta_layers: dict[str, dict] = {}
    converted_count = 0
    skipped_count = 0
    w4a8_linear = 0
    shape_critical_int8 = 0
    convrot_int8_linear = 0
    plain_int8_count = 0
    dtype_keep_count = 0
    non_diffusion_keep = 0
    device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"Mode qwen_image | device={device} | ConvRot W4A8 (Linear 2D)")

    pack_device = torch.device(device)
    for key, tensor in tqdm(state_dict.items()):
        if _is_non_diffusion_key(key):
            new_state_dict[key] = tensor
            skipped_count += 1
            non_diffusion_keep += 1
            continue

        module_key = key[: -len(".weight")] if key.endswith(".weight") else key
        is_shape_critical = (
            tensor.ndim == 2 and key.endswith(".weight") and _is_shape_critical(module_key)
        )
        is_matmul_weight = (
            key.endswith(".weight")
            and tensor.ndim == 2
            and tensor.dtype in (torch.float16, torch.float32, torch.bfloat16)
        )

        if not is_matmul_weight:
            # bias / norm / f32 buffer (__index_timestep_zero__) etc: keep as-is
            new_state_dict[key] = tensor
            skipped_count += 1
            dtype_keep_count += 1
            continue

        in_f = int(tensor.shape[1])
        used_gs = (
            convrot_group_size_for_features(in_f, group_size)
            if enable_convrot
            else None
        )

        quant_config: dict
        if is_shape_critical:
            # --- shape-preserving per-channel INT8 (never nibble-packed) ---
            q, scale = pack_channelwise(tensor)
            quant_config = {"format": "int8_tensorwise"}
            new_state_dict[key] = q
            new_state_dict[f"{module_key}.weight_scale"] = scale
            new_state_dict[f"{module_key}.comfy_quant"] = _encode_comfy_quant(
                quant_config
            )
            quant_meta_layers[module_key] = dict(quant_config)
            converted_count += 1
            shape_critical_int8 += 1
        elif (
            enable_convrot
            and ck is not None
            and used_gs == 256
            and in_f % _QUANT_GROUP_SIZE == 0
        ):
            # --- W4A8 Linear (kitchen pack; ConvRot rotation runs inside) ---
            w_fp32 = tensor.to(device=pack_device, dtype=torch.float32)
            qdata, s_rel, s_channel, correction, codebook = (
                ck.quantize_w4a8_int8_weight(
                    w_fp32,
                    group_size=_QUANT_GROUP_SIZE,
                    convrot_groupsize=int(used_gs),
                    symmetric=True,
                    scale_dtype=torch.float8_e4m3fn,
                    codebook=True,
                    bits=4,
                )
            )
            quant_config = {
                "format": "asym_w4a8_int8",
                "group_size": int(_QUANT_GROUP_SIZE),
                "convrot_groupsize": int(used_gs),
            }
            new_state_dict[key] = qdata.to(torch.int8).cpu()
            new_state_dict[f"{module_key}.weight_s_rel"] = s_rel.to(
                torch.float8_e4m3fn
            ).contiguous().cpu()
            new_state_dict[f"{module_key}.weight_s_channel"] = (
                s_channel.float().contiguous().cpu()
            )
            if codebook is not None:
                new_state_dict[f"{module_key}.weight_codebook"] = (
                    codebook.float().contiguous().cpu()
                )
            new_state_dict[f"{module_key}.comfy_quant"] = _encode_comfy_quant(
                quant_config
            )
            quant_meta_layers[module_key] = dict(quant_config)
            converted_count += 1
            w4a8_linear += 1
            del w_fp32, qdata, s_rel, s_channel, correction, codebook
        elif enable_convrot and used_gs is not None:
            # --- ConvRot INT8 Linear fallback (eligible gs, but K%16!=0) ---
            from native_convert_int8 import build_hadamard, rotate_weight

            h_matrix = build_hadamard(int(used_gs), device="cpu", dtype=torch.float32)
            w_fp = rotate_weight(tensor.float().cpu(), h_matrix, int(used_gs))
            q, scale = pack_channelwise(w_fp)
            quant_config = {
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": int(used_gs),
            }
            new_state_dict[key] = q
            new_state_dict[f"{module_key}.weight_scale"] = scale
            new_state_dict[f"{module_key}.comfy_quant"] = _encode_comfy_quant(
                quant_config
            )
            quant_meta_layers[module_key] = dict(quant_config)
            converted_count += 1
            convrot_int8_linear += 1
        elif per_channel_int8:
            q, scale = pack_channelwise(tensor.float().cpu())
            quant_config = {"format": "int8_tensorwise"}
            new_state_dict[key] = q
            new_state_dict[f"{module_key}.weight_scale"] = scale
            new_state_dict[f"{module_key}.comfy_quant"] = _encode_comfy_quant(
                quant_config
            )
            quant_meta_layers[module_key] = dict(quant_config)
            converted_count += 1
            plain_int8_count += 1
        else:
            q, scale = pack_tensorwise(tensor.float().cpu())
            quant_config = {"format": "int8_tensorwise"}
            new_state_dict[key] = q
            new_state_dict[f"{module_key}.weight_scale"] = scale
            new_state_dict[f"{module_key}.comfy_quant"] = _encode_comfy_quant(
                quant_config
            )
            quant_meta_layers[module_key] = dict(quant_config)
            converted_count += 1
            plain_int8_count += 1

    metadata = {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": quant_meta_layers}
        )
    }

    print(f"Saving to: {output_path}")
    print(f"Converted layers: {converted_count}, Kept layers: {skipped_count}")
    print(f"  non-diffusion keep: {non_diffusion_keep}, dtype keep (bias/norm/buffer): {dtype_keep_count}")
    print(f"Per-channel INT8 (fallback packs): {per_channel_int8}")
    print(f"ConvRot enabled: {enable_convrot}")
    if enable_convrot:
        print(
            f"  W4A8 Linear (asym_w4a8_int8): {w4a8_linear}, "
            f"ConvRot INT8 Linear: {convrot_int8_linear}, "
            f"shape-critical INT8 (img_in): {shape_critical_int8}, "
            f"plain INT8 (no eligible group size): {plain_int8_count}"
        )
    else:
        print(f"  plain INT8: {plain_int8_count}")

    save_file(new_state_dict, output_path, metadata=metadata)
    total_bytes = os.path.getsize(output_path)
    print(f"Done. Size: {round(total_bytes / (1024**3), 3)} GiB")

    del new_state_dict
    del state_dict
    del quant_meta_layers


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Qwen Image / Qwen Image Edit ConvRot W4A8 converter for native "
            "ComfyUI. Linear 2D -> asym_w4a8_int8 via comfy_kitchen (quant "
            "group 16, convrot 256 only, Lloyd-Max codebook, fp8 group scales); "
            "shape-critical img_in -> shape-preserving per-channel INT8 "
            "(detect_unet_config reads its K verbatim); ineligible Linear -> "
            "plain INT8. Qwen DiT has no Conv2d. No Card 1 BC (text encoder "
            "too large for local calib; intentionally absent). No bench hook."
        )
    )
    parser.add_argument(
        "--model",
        "--input",
        dest="model",
        type=str,
        required=True,
        help="Path to input BF16 .safetensors (e.g. qwnImageEdit_v16Bf16.safetensors)",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output .safetensors"
    )
    parser.add_argument(
        "--per_channel_int8",
        action="store_true",
        help=(
            "Card 3 for plain INT8 fallback packs only: per-out-channel amax/scale. "
            "W4A8 Linear always carries its own per-channel s_channel."
        ),
    )
    parser.add_argument(
        "--convrot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "ConvRot W4A8 on eligible Linear. Default ON; the W4A8 layout itself "
            "is ConvRot-based, so --no-convrot packs plain INT8 only."
        ),
    )
    parser.add_argument(
        "--groupsize",
        "--group_size",
        dest="group_size",
        type=int,
        default=_DEFAULT_GROUPSIZE,
        help=(
            f"Preferred ConvRot Hadamard group size (power of 4, default "
            f"{_DEFAULT_GROUPSIZE}). W4A8 requires in_features divisible by "
            f"256; smaller power-of-4 divisors fall to ConvRot INT8."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Kitchen quantize device (default cuda when available)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)
    if args.group_size < 4 or (args.group_size & (args.group_size - 1)) != 0:
        print(f"Error: --group_size must be a power of 4 (>=4), got {args.group_size}")
        sys.exit(1)
    if math.log(args.group_size, 4) % 1 != 0:
        print(f"Error: --group_size must be a power of 4, got {args.group_size}")
        sys.exit(1)

    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

    convert_to_w4a8(
        args.model,
        args.output,
        per_channel_int8=bool(args.per_channel_int8),
        enable_convrot=bool(args.convrot),
        group_size=int(args.group_size),
        device=str(args.device),
    )


if __name__ == "__main__":
    main()

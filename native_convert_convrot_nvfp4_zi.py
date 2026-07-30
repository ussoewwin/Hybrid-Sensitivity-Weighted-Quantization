"""Z-Image / ZIT (NextDiT / Lumina2) NVFP4 converter — Kitchen pack + FULL ConvRot.

Reference (pack / blacklist / metadata):
  https://github.com/tritant/ComfyUI_Kitchen_nvfp4_Converter
  convert_to_nvfp4_node.py

  - 2D .weight → (optional offline Hadamard) → TensorCoreNVFP4Layout.quantize
  - Z-Image-Turbo / Z-Image-Base Kitchen blacklists only
  - Non-matching tensors kept as bfloat16
  - Metadata: _quantization_metadata (+ convrot stamp when rotated)
  - FULL ConvRot ON by default (Linear 2D only; Z Image has no Conv2d packs)
      offline: W_rot = W @ H^T (group-wise), then NVFP4 pack
      stamp:   {"format":"nvfp4","convrot":true,"convrot_groupsize":G}
      plain when in_features not divisible by a power-of-4 group: {"format":"nvfp4"}
  - No calib, no input_scale, no bias correction
  - Use --no-convrot for plain Kitchen NVFP4 only

Verified against ComfyUI UNet keys (moodyProMix_zitV13.safetensors):
  model.diffusion_model.{cap_embedder,x_embedder,t_embedder,
  noise_refiner,context_refiner,final_layer,layers.0..29}
  → Kitchen default profile = Z-Image-Turbo
    (embedders / refiners / final_layer BF16; layers.* 2D weights NVFP4)

Refuses non–Z-Image checkpoints (missing Lumina2 signature).
Post-convert SDXL fidelity bench is not chained (Z-Image-only CLI).

Online act rotation for ConvRot stamps is the loader / bench parity responsibility
(ComfyUI stock nvfp4 path does not rotate acts; HSWQ / nvfp4_comfy_parity does).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import OrderedDict

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

try:
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout
except ImportError:
    print("Error: comfy_kitchen not found (install in the active venv).")
    sys.exit(1)

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from native_convert_int8 import (  # noqa: E402
    build_hadamard,
    convrot_group_size_for_features,
    rotate_weight,
)

_DEFAULT_GROUPSIZE = 256

# Kitchen model_type → (BLACKLIST, FP8_LAYERS) — Z-Image only
# (same strings as convert_to_nvfp4_node.py)
_Z_IMAGE_PROFILES: dict[str, tuple[list[str], list[str]]] = {
    "Z-Image-Turbo": (
        [
            "cap_embedder",
            "x_embedder",
            "noise_refiner",
            "context_refiner",
            "t_embedder",
            "final_layer",
        ],
        [],
    ),
    "Z-Image-Base": (
        [
            "attention",
            "adaLN_modulation",
            "norm",
            "final_layer",
            "cap_embedder",
            "x_embedder",
            "noise_refiner",
            "context_refiner",
            "t_embedder",
        ],
        [],
    ),
}

_DEFAULT_MODEL_TYPE = "Z-Image-Turbo"

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


def _find_z_image_key_prefix(state_dict) -> str:
    """Lumina2 / NextDiT / Z-Image signature (ComfyUI model_detection).

    Requires cap_embedder.1.weight and noise_refiner.0 attention
    (k_norm or fused qkv) under a known diffusion prefix.
    """
    for prefix in ("model.diffusion_model.", "diffusion_model.", ""):
        cap = f"{prefix}cap_embedder.1.weight"
        if cap not in state_dict:
            continue
        k_norm = f"{prefix}noise_refiner.0.attention.k_norm.weight"
        qkv = f"{prefix}noise_refiner.0.attention.qkv.weight"
        if k_norm in state_dict or qkv in state_dict:
            return prefix
    raise ValueError(
        "Not a Z-Image / ZIT (NextDiT / Lumina2) checkpoint: missing "
        "cap_embedder.1.weight + noise_refiner.0.attention.(k_norm|qkv).weight "
        "(under model.diffusion_model. / diffusion_model. / root)."
    )


def _meta_base_key(base_k_file: str) -> str:
    if "model.diffusion_model." in base_k_file:
        return base_k_file.split("model.diffusion_model.")[-1]
    if "diffusion_model." in base_k_file:
        return base_k_file.split("diffusion_model.")[-1]
    return base_k_file


def convert_to_nvfp4(
    input_path: str,
    output_path: str,
    device: str,
    model_type: str = _DEFAULT_MODEL_TYPE,
    enable_convrot: bool = True,
    group_size: int = _DEFAULT_GROUPSIZE,
):
    if model_type not in _Z_IMAGE_PROFILES:
        raise ValueError(
            f"Unknown model_type={model_type!r}; "
            f"choose from {sorted(_Z_IMAGE_PROFILES)}"
        )
    blacklist, fp8_layers = _Z_IMAGE_PROFILES[model_type]

    rot_tag = "FULL ConvRot NVFP4" if enable_convrot else "plain NVFP4"
    print(f"Mode {model_type} | device={device} | {rot_tag} (Z-Image-only)")
    if enable_convrot:
        print(
            f"  [ConvRot] ON | preferred groupsize={int(group_size)} "
            f"(Linear 2D; skip rotate when in_features has no power-of-4 group)"
        )
    else:
        print("  [ConvRot] OFF | plain Kitchen NVFP4 packs only")

    sd = load_file(input_path)
    prefix = _find_z_image_key_prefix(sd)
    print(f"Detected Z-Image key prefix: {prefix!r}")

    # Structural summary (helps audit Turbo vs Base choice)
    n_layers = sum(
        1
        for k in sd
        if k.startswith(f"{prefix}layers.") and k.endswith(".feed_forward.w1.weight")
    )
    has_noise = any(f"{prefix}noise_refiner." in k for k in sd)
    has_ctx = any(f"{prefix}context_refiner." in k for k in sd)
    print(
        f"Structure: layers(w1)={n_layers} "
        f"noise_refiner={has_noise} context_refiner={has_ctx}"
    )
    if model_type == "Z-Image-Base":
        print(
            "[!] Z-Image-Base Kitchen blacklist also matches layers.*.attention / "
            "adaLN_modulation / norm — NVFP4 candidates shrink to feed_forward "
            "2D weights mainly. ZIT / Turbo UNets usually want Z-Image-Turbo."
        )

    quant_map = {"format_version": "1.0", "layers": {}}
    new_sd: dict[str, torch.Tensor] = {}
    n_nvfp4 = 0
    n_convrot = 0
    n_plain_nvfp4 = 0
    n_bf16 = 0

    print(f"Converting ({len(sd)} tensors)...")
    for k, v in tqdm(list(sd.items())):
        if any(name in k for name in blacklist):
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16 += 1
            continue

        if _is_non_diffusion_key(k):
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16 += 1
            continue

        if v.ndim == 2 and ".weight" in k:
            base_k_file = k.replace(".weight", "")
            base_k_meta = _meta_base_key(base_k_file)
            v_tensor = v.to(device=device, dtype=torch.bfloat16)

            if fp8_layers and any(name in k for name in fp8_layers):
                import comfy_kitchen as ck

                weight_scale = (
                    (v_tensor.abs().max() / 448.0).clamp(min=1e-12).float()
                )
                weight_quantized = ck.quantize_per_tensor_fp8(v_tensor, weight_scale)
                new_sd[k] = weight_quantized.cpu()
                new_sd[f"{base_k_file}.weight_scale"] = weight_scale.to(
                    torch.bfloat16
                ).cpu()
                quant_map["layers"][base_k_meta] = {"format": "float8_e4m3fn"}
                if device == "cuda":
                    del v_tensor
                continue

            used_gs = None
            do_rotate = False
            w_for_q = v_tensor
            if enable_convrot:
                used_gs = convrot_group_size_for_features(
                    int(v_tensor.shape[1]), int(group_size)
                )
                if used_gs is not None:
                    h_matrix = build_hadamard(
                        int(used_gs), device="cpu", dtype=torch.float32
                    )
                    w_rot = rotate_weight(
                        v_tensor.float().cpu(), h_matrix, int(used_gs)
                    )
                    w_for_q = w_rot.to(device=device, dtype=torch.bfloat16)
                    do_rotate = True

            try:
                qdata, params = TensorCoreNVFP4Layout.quantize(w_for_q)
                tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)
                for suffix, tensor in tensors.items():
                    new_sd[f"{base_k_file}.weight{suffix}"] = tensor.cpu()
                if do_rotate and used_gs is not None:
                    quant_map["layers"][base_k_meta] = {
                        "format": "nvfp4",
                        "convrot": True,
                        "convrot_groupsize": int(used_gs),
                    }
                    n_convrot += 1
                else:
                    quant_map["layers"][base_k_meta] = {"format": "nvfp4"}
                    n_plain_nvfp4 += 1
                n_nvfp4 += 1
            except Exception:
                new_sd[k] = v.to(dtype=torch.bfloat16)
                n_bf16 += 1

            if device == "cuda":
                if do_rotate:
                    del w_for_q
                del v_tensor
        else:
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16 += 1

    final_metadata = OrderedDict()
    final_metadata["_quantization_metadata"] = json.dumps(quant_map)
    by_tag = (
        "ComfyUI Kitchen NVFP4 Converter (Z-Image ConvRot)"
        if enable_convrot
        else "ComfyUI Kitchen NVFP4 Converter (Z-Image-only)"
    )
    final_metadata["converted_by"] = by_tag
    final_metadata["converter_url"] = (
        "https://github.com/tritant/ComfyUI_Kitchen_nvfp4_Converter"
    )
    final_metadata["hswq_model"] = "z_image"
    final_metadata["hswq_kitchen_profile"] = model_type
    final_metadata["hswq_nvfp4_convrot"] = "1" if enable_convrot else "0"

    print(f"Saving | Type: {model_type} | Path: {output_path}")
    save_file(new_sd, output_path, metadata=final_metadata)
    total_bytes = os.path.getsize(output_path)
    print(f"Done. Size: {round(total_bytes / (1024**3), 2)} GiB")
    print(f"NVFP4 layers in metadata: {len(quant_map['layers'])}")
    print(f"  counted nvfp4 packs={n_nvfp4} | bf16 keep tensors={n_bf16}")
    print(f"FULL ConvRot enabled: {enable_convrot}")
    if enable_convrot:
        print(
            f"  ConvRot NVFP4 Linear: {n_convrot}, "
            f"plain NVFP4 (no group): {n_plain_nvfp4}"
        )

    del sd
    del new_sd
    del quant_map
    _release_vram("after native Z-Image NVFP4 convert save")


def _release_vram(label: str = "post-convert") -> None:
    print(f"[*] Releasing VRAM ({label})...")
    gc.collect()
    if not torch.cuda.is_available():
        print(f"[*] VRAM clear ({label}): CUDA not available")
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        alloc_mib = torch.cuda.memory_allocated() / (1024 ** 2)
        reserv_mib = torch.cuda.memory_reserved() / (1024 ** 2)
        print(
            f"[*] VRAM clear ({label}): "
            f"allocated={alloc_mib:.1f} MiB reserved={reserv_mib:.1f} MiB"
        )
    except Exception:
        print(f"[*] VRAM clear ({label}): done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Z-Image / ZIT NVFP4 convert with FULL ConvRot (Linear) ON by default. "
            "Kitchen pack + offline Hadamard + convrot stamp in "
            "_quantization_metadata. Use --no-convrot for plain Kitchen NVFP4. "
            "No calib / no input_scale. Refuses non-Z-Image checkpoints. "
            "Default profile Z-Image-Turbo. No chained SDXL NVFP4 bench."
        )
    )
    parser.add_argument(
        "--model",
        "--input",
        dest="model",
        type=str,
        required=True,
        help="Path to Z-Image / ZIT BF16/FP16 .safetensors",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output .safetensors"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=_DEFAULT_MODEL_TYPE,
        choices=sorted(_Z_IMAGE_PROFILES.keys()),
        help=(
            "Kitchen Z-Image profile (default: Z-Image-Turbo; "
            "use Z-Image-Base only for Kitchen Base blacklist)"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Quantize device",
    )
    parser.add_argument(
        "--no-convrot",
        dest="enable_convrot",
        action="store_false",
        help="Disable ConvRot; pack plain Kitchen NVFP4 only.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=_DEFAULT_GROUPSIZE,
        help=f"Preferred ConvRot Hadamard group size (default {_DEFAULT_GROUPSIZE}).",
    )
    parser.set_defaults(enable_convrot=True)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    convert_to_nvfp4(
        args.model,
        args.output,
        device=str(args.device),
        model_type=str(args.model_type),
        enable_convrot=bool(args.enable_convrot),
        group_size=int(args.group_size),
    )

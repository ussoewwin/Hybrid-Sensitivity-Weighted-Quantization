"""Qwen Image / Qwen Image Edit NVFP4 converter — Kitchen pack + FULL ConvRot.

Reference (pack / blacklist / metadata):
  https://github.com/tritant/ComfyUI_Kitchen_nvfp4_Converter
  convert_to_nvfp4_node.py

  - 2D .weight -> (optional offline Hadamard) -> TensorCoreNVFP4Layout.quantize
  - Qwen-Image-Edit-2511 / Qwen-Image-2512 Kitchen blacklists
  - Non-matching tensors kept as bfloat16
  - Metadata: _quantization_metadata (+ convrot stamp when rotated)
  - FULL ConvRot ON by default (Linear 2D)
      offline: W_rot = W @ H^T (group-wise), then NVFP4 pack
      stamp:   {"format":"nvfp4","convrot":true,"convrot_groupsize":G}
      plain when in_features not divisible by a power-of-4 group: {"format":"nvfp4"}
  - No input_scale (Kitchen NVFP4 path)
  - Use --no-convrot for plain Kitchen NVFP4 only

Supported models:
  - Qwen-Image-Edit (e.g. qwnImageEdit_v16Bf16.safetensors)
  - Qwen-Image-2512 / Qwen2.5-VL DiT architectures
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from collections import OrderedDict

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

try:
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout
except ImportError:
    print("Error: comfy_kitchen not found (install in the active venv).")
    sys.exit(1)

def _default_repo_root() -> str:
    """Locate repo root by searching upward for native_convert_int8.py."""
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(8):
        if os.path.isfile(os.path.join(d, "native_convert_int8.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.abspath(os.path.join(here, os.pardir))


_REPO_ROOT = _default_repo_root()
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from native_convert_int8 import (  # noqa: E402
    build_hadamard,
    convrot_group_size_for_features,
    rotate_weight,
)

_DEFAULT_GROUPSIZE = 256

# Kitchen model_type -> (BLACKLIST, FP8_LAYERS)
_QWEN_PROFILES: dict[str, tuple[list[str], list[str]]] = {
    "Qwen-Image-Edit-2511": (
        ["img_in", "txt_in", "time_text_embed", "norm_out", "proj_out"],
        [],
    ),
    "Qwen-Image-2512": (
        [
            "img_in",
            "txt_in",
            "time_text_embed",
            "norm_out",
            "proj_out",
            "img_mod.1",
        ],
        ["txt_mlp", "txt_mod"],
    ),
}

_DEFAULT_MODEL_TYPE = "Qwen-Image-Edit-2511"

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


def _find_qwen_key_prefix(state_dict) -> str:
    """Detect Qwen-Image / Qwen-Image-Edit key prefix."""
    for prefix in ("model.diffusion_model.", "diffusion_model.", ""):
        k1 = f"{prefix}img_in.weight"
        k2 = f"{prefix}transformer_blocks.0.attn.to_q.weight"
        k3 = f"{prefix}transformer_blocks.0.img_mod.1.weight"
        if k1 in state_dict or k2 in state_dict or k3 in state_dict:
            return prefix
    for k in state_dict:
        if "transformer_blocks.0." in k:
            idx = k.index("transformer_blocks.0.")
            return k[:idx]
    return ""


def _meta_base_key(base_k_file: str) -> str:
    if "model.diffusion_model." in base_k_file:
        return base_k_file.split("model.diffusion_model.")[-1]
    if "diffusion_model." in base_k_file:
        return base_k_file.split("diffusion_model.")[-1]
    return base_k_file


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


def convert_to_nvfp4(
    input_path: str,
    output_path: str,
    device: str,
    model_type: str = _DEFAULT_MODEL_TYPE,
    enable_convrot: bool = True,
    group_size: int = _DEFAULT_GROUPSIZE,
):
    if model_type not in _QWEN_PROFILES:
        raise ValueError(
            f"Unknown model_type={model_type!r}; "
            f"choose from {sorted(_QWEN_PROFILES)}"
        )
    blacklist, fp8_layers = _QWEN_PROFILES[model_type]

    rot_tag = "FULL ConvRot NVFP4" if enable_convrot else "plain NVFP4"
    print(f"Mode {model_type} | device={device} | {rot_tag} (Qwen-Image / Qwen-Image-Edit)")
    if enable_convrot:
        print(
            f"  [ConvRot] ON | preferred groupsize={int(group_size)} "
            f"(Linear 2D; skip rotate when in_features has no power-of-4 group)"
        )
    else:
        print("  [ConvRot] OFF | plain Kitchen NVFP4 packs only")

    sd = load_file(input_path)
    prefix = _find_qwen_key_prefix(sd)
    print(f"Detected Qwen key prefix: {prefix!r}")

    quant_map = {"format_version": "1.0", "layers": {}}
    new_sd: dict[str, torch.Tensor] = {}
    n_nvfp4 = 0
    n_convrot = 0
    n_plain_nvfp4 = 0
    n_fp8 = 0
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
                weight_scale = (
                    (v_tensor.abs().max() / 448.0).clamp(min=1e-12).float()
                )
                weight_quantized = ck.quantize_per_tensor_fp8(v_tensor, weight_scale)
                new_sd[k] = weight_quantized.cpu()
                new_sd[f"{base_k_file}.weight_scale"] = weight_scale.to(
                    torch.bfloat16
                ).cpu()
                quant_map["layers"][base_k_meta] = {"format": "float8_e4m3fn"}
                n_fp8 += 1
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
            except Exception as e:
                print(f"[WARN] Failed to quantize {k} to NVFP4 ({e}); keeping as bfloat16")
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
        "ComfyUI Kitchen NVFP4 Converter (Qwen-Image ConvRot)"
        if enable_convrot
        else "ComfyUI Kitchen NVFP4 Converter (Qwen-Image-only)"
    )
    final_metadata["converted_by"] = by_tag
    final_metadata["converter_url"] = (
        "https://github.com/tritant/ComfyUI_Kitchen_nvfp4_Converter"
    )
    final_metadata["hswq_model"] = "qwen_image"
    final_metadata["hswq_kitchen_profile"] = model_type
    final_metadata["hswq_nvfp4_convrot"] = "1" if enable_convrot else "0"

    print(f"Saving | Type: {model_type} | Path: {output_path}")
    save_file(new_sd, output_path, metadata=final_metadata)
    total_bytes = os.path.getsize(output_path)
    print(f"Done. Size: {round(total_bytes / (1024**3), 2)} GiB")
    print(f"Quantized layers in metadata: {len(quant_map['layers'])}")
    print(f"  counted nvfp4 packs={n_nvfp4} (ConvRot={n_convrot}, plain={n_plain_nvfp4}) | fp8={n_fp8} | bf16 keep={n_bf16}")
    print(f"FULL ConvRot enabled: {enable_convrot}")

    del sd
    del new_sd
    del quant_map
    _release_vram("after native Qwen NVFP4 convert save")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Qwen-Image / Qwen-Image-Edit NVFP4 convert with FULL ConvRot (Linear) ON by default. "
            "Kitchen pack + offline Hadamard + convrot stamp in "
            "_quantization_metadata. Use --no-convrot for plain Kitchen NVFP4."
        )
    )
    parser.add_argument(
        "--model",
        "--input",
        dest="model",
        type=str,
        required=True,
        help="Path to input BF16/FP16 .safetensors (e.g. qwnImageEdit_v16Bf16.safetensors)",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output .safetensors"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=_DEFAULT_MODEL_TYPE,
        choices=sorted(_QWEN_PROFILES.keys()),
        help=(
            "Kitchen Qwen profile (default: Qwen-Image-Edit-2511; "
            "use Qwen-Image-2512 for standard Qwen-Image DiT)"
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
        "--convrot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable ConvRot (default ON). Pass --no-convrot for plain Kitchen NVFP4.",
    )
    parser.add_argument(
        "--groupsize",
        "--group_size",
        dest="group_size",
        type=int,
        default=_DEFAULT_GROUPSIZE,
        help=f"Preferred ConvRot Hadamard group size (default {_DEFAULT_GROUPSIZE}).",
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

    convert_to_nvfp4(
        args.model,
        args.output,
        device=str(args.device),
        model_type=str(args.model_type),
        enable_convrot=bool(args.convrot),
        group_size=int(args.group_size),
    )

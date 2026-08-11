"""Krea2-only plain NVFP4 converter (no ConvRot) — Kitchen-faithful pack.

Pack / metadata behavior matches Kitchen convert_to_nvfp4_node.py:
  https://github.com/tritant/ComfyUI_Kitchen_nvfp4_Converter

  - 2D .weight → TensorCoreNVFP4Layout.quantize + state_dict_tensors
  - Krea2 blacklist keeps structure-sensitive layers in bfloat16
  - Non-matching / non-2D tensors kept as bfloat16
  - Metadata: _quantization_metadata + converted_by / converter_url
  - No ConvRot, no calib, no input_scale, no bias correction

Krea2 signature (FATAL if missing): txtfusion.projector + blocks.0.attn.wq
under model.diffusion_model. / diffusion_model. / root.

Blacklist (Krea2 DiT — first/last must stay BF16 so ComfyUI can infer
channels from weight shapes; protect mod/norm/projector/tmlp/tproj):
  first, last, mod., norm, projector, tmlp, tproj, bias, vae., text_encoders

For FULL ConvRot + DualMonitor calib, use hswq_convert_nvfp4_krea2.py.
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

_MODEL_TYPE = "Krea2"

# Krea2 SingleStreamDiT — structure-sensitive layers stay BF16.
_KREA2_BLACKLIST: list[str] = [
    "first.",
    "last.",
    "mod.",
    "norm",
    "projector",
    "tmlp",
    "txtmlp",
    "tproj",
    "txtfusion",
    "bias",
]
_KREA2_FP8_LAYERS: list[str] = []

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


def _find_krea2_key_prefix(state_dict) -> str:
    for prefix in ("model.diffusion_model.", "diffusion_model.", ""):
        if f"{prefix}txtfusion.projector.weight" in state_dict:
            if f"{prefix}blocks.0.attn.wq.weight" not in state_dict:
                raise ValueError(
                    "Krea2 signature incomplete: txtfusion.projector present but "
                    f"{prefix}blocks.0.attn.wq.weight missing"
                )
            return prefix
    raise ValueError(
        "Not a Krea2 checkpoint: missing txtfusion.projector.weight "
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
):
    print(f"Mode {_MODEL_TYPE} | device={device} | plain NVFP4 (Krea2-only)")

    sd = load_file(input_path)
    prefix = _find_krea2_key_prefix(sd)
    print(f"Detected Krea2 key prefix: {prefix!r}")

    blacklist = list(_KREA2_BLACKLIST)
    fp8_layers = list(_KREA2_FP8_LAYERS)
    quant_map = {"format_version": "1.0", "layers": {}}
    new_sd: dict[str, torch.Tensor] = {}
    n_nvfp4 = 0
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
                # Reserved for Kitchen FP8 path; Krea2 profile leaves this empty.
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

            try:
                qdata, params = TensorCoreNVFP4Layout.quantize(v_tensor)
                tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)
                for suffix, tensor in tensors.items():
                    new_sd[f"{base_k_file}.weight{suffix}"] = tensor.cpu()
                quant_map["layers"][base_k_meta] = {"format": "nvfp4"}
                n_nvfp4 += 1
            except Exception:
                new_sd[k] = v.to(dtype=torch.bfloat16)
                n_bf16 += 1

            if device == "cuda":
                del v_tensor
        else:
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16 += 1

    final_metadata = OrderedDict()
    final_metadata["_quantization_metadata"] = json.dumps(quant_map)
    final_metadata["converted_by"] = "ComfyUI Kitchen NVFP4 Converter (Krea2-only)"
    final_metadata["converter_url"] = (
        "https://github.com/tritant/ComfyUI_Kitchen_nvfp4_Converter"
    )
    final_metadata["hswq_model"] = "krea2"

    print(f"Saving | Type: {_MODEL_TYPE} | Path: {output_path}")
    save_file(new_sd, output_path, metadata=final_metadata)
    total_bytes = os.path.getsize(output_path)
    print(f"Done. Size: {round(total_bytes / (1024**3), 2)} GiB")
    print(f"NVFP4 layers in metadata: {len(quant_map['layers'])}")
    print(f"  counted nvfp4 packs={n_nvfp4} | bf16 keep tensors={n_bf16}")

    del sd
    del new_sd
    del quant_map
    _release_vram("after native Krea2 NVFP4 convert save")


def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


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
            "Krea2-only plain NVFP4 convert (Kitchen pack, no ConvRot / "
            "no calib / no input_scale). Refuses non-Krea2 checkpoints. "
            "Post-convert SDXL bench is not used (no Krea2 fidelity bench)."
        )
    )
    parser.add_argument(
        "--model",
        "--input",
        dest="model",
        type=str,
        required=True,
        help="Path to Krea2 BF16/FP16 .safetensors",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output .safetensors"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Quantize device",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    convert_to_nvfp4(
        args.model,
        args.output,
        device=str(args.device),
    )

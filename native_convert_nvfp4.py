"""Plain NVFP4 converter (no ConvRot) — Kitchen-faithful CLI.

Reference (verbatim behavior for pack / blacklist / metadata):
  https://github.com/tritant/ComfyUI_Kitchen_nvfp4_Converter
  convert_to_nvfp4_node.py

  - 2D .weight → TensorCoreNVFP4Layout.quantize + state_dict_tensors
  - BLACKLIST / FP8_LAYERS by --model_type (Kitchen profiles)
  - Non-matching tensors kept as bfloat16
  - Metadata: _quantization_metadata + converted_by / converter_url
  - No ConvRot, no calib, no input_scale, no bias correction

For FULL ConvRot + optional calib, use native_convert_nvfp4_convrot.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm

try:
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout
except ImportError:
    print("Error: comfy_kitchen not found (install in the active venv).")
    sys.exit(1)

# Kitchen model_type → (BLACKLIST, FP8_LAYERS) — same as convert_to_nvfp4_node.py
_KITCHEN_PROFILES: dict[str, tuple[list[str], list[str]]] = {
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
    "Wan2.2-i2v-high-low": (
        ["text_embedding", "time_embedding", "time_projection", "head"],
        [],
    ),
    "Flux.1-dev": (
        [
            "bias",
            "txt_attn",
            "img_in",
            "txt_in",
            "time_in",
            "vector_in",
            "guidance_in",
            "final_layer",
            "class_embedding",
            "single_stream_modulation",
            "double_stream_modulation_img",
            "double_stream_modulation_txt",
        ],
        [],
    ),
    "Flux.1-Fill": (
        [
            "bias",
            "txt_attn",
            "img_in",
            "txt_in",
            "time_in",
            "vector_in",
            "guidance_in",
            "final_layer",
            "class_embedding",
            "single_stream_modulation",
            "double_stream_modulation_img",
            "double_stream_modulation_txt",
        ],
        [],
    ),
    "Flux.2-dev": (
        [
            "bias",
            "txt_attn",
            "img_in",
            "txt_in",
            "time_in",
            "vector_in",
            "guidance_in",
            "final_layer",
            "class_embedding",
            "single_stream_modulation",
            "double_stream_modulation_img",
            "double_stream_modulation_txt",
        ],
        [],
    ),
    "Flux.2-Klein-9b": (
        [
            "bias",
            "txt_attn",
            "img_in",
            "txt_in",
            "time_in",
            "vector_in",
            "guidance_in",
            "final_layer",
            "class_embedding",
            "single_stream_modulation",
            "double_stream_modulation_img",
            "double_stream_modulation_txt",
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
    "LTX-2-19b-dev-or-distilled": (
        [
            "vae.",
            "vocoder.",
            "connector",
            "proj_out",
            "norm",
            "bias",
            "scale",
            "embedder",
            "patchify",
            "table",
            "transformer_blocks.0.",
            "transformer_blocks.43.",
            "transformer_blocks.44.",
            "transformer_blocks.45.",
            "transformer_blocks.46.",
            "transformer_blocks.47.",
            "projection",
            "adaln_single",
        ],
        [],
    ),
}


def convert_to_nvfp4(
    input_path: str,
    output_path: str,
    model_type: str,
    device: str,
):
    if model_type not in _KITCHEN_PROFILES:
        raise ValueError(
            f"Unknown model_type={model_type!r}; "
            f"choose from {sorted(_KITCHEN_PROFILES)}"
        )
    blacklist, fp8_layers = _KITCHEN_PROFILES[model_type]

    print(f"Mode {model_type} | device={device} | plain NVFP4 (Kitchen-faithful)")

    temp_diffusers_meta: dict[str, str] = {}
    if model_type == "LTX-2-19b-dev-or-distilled":
        with safe_open(input_path, framework="pt") as f:
            orig_meta = f.metadata() or {}
            for key in ("config", "license", "encrypted_wandb_properties"):
                if key in orig_meta:
                    temp_diffusers_meta[key] = orig_meta[key]

    sd = load_file(input_path)
    quant_map = {"format_version": "1.0", "layers": {}}
    new_sd: dict[str, torch.Tensor] = {}

    print(f"Converting ({len(sd)} tensors)...")
    for k, v in tqdm(list(sd.items())):
        if any(name in k for name in blacklist):
            new_sd[k] = v.to(dtype=torch.bfloat16)
            continue

        if v.ndim == 2 and ".weight" in k:
            base_k_file = k.replace(".weight", "")

            if model_type == "LTX-2-19b-dev-or-distilled":
                base_k_meta = k.replace(".weight", "")
            else:
                if "model.diffusion_model." in base_k_file:
                    base_k_meta = base_k_file.split("model.diffusion_model.")[-1]
                else:
                    base_k_meta = base_k_file

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
                if device == "cuda":
                    del v_tensor
                continue

            try:
                qdata, params = TensorCoreNVFP4Layout.quantize(v_tensor)
                tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)
                for suffix, tensor in tensors.items():
                    new_sd[f"{base_k_file}.weight{suffix}"] = tensor.cpu()
                quant_map["layers"][base_k_meta] = {"format": "nvfp4"}
            except Exception:
                new_sd[k] = v.to(dtype=torch.bfloat16)

            if device == "cuda":
                del v_tensor
        else:
            new_sd[k] = v.to(dtype=torch.bfloat16)

    final_metadata = OrderedDict()
    final_metadata["_quantization_metadata"] = json.dumps(quant_map)
    final_metadata["converted_by"] = "ComfyUI Kitchen NVFP4 Converter"
    final_metadata["converter_url"] = (
        "https://github.com/tritant/ComfyUI_Kitchen_nvfp4_Converter"
    )
    if model_type == "LTX-2-19b-dev-or-distilled":
        for mk, mv in temp_diffusers_meta.items():
            final_metadata[mk] = mv

    print(f"Saving | Type: {model_type} | Path: {output_path}")
    save_file(new_sd, output_path, metadata=final_metadata)
    total_bytes = os.path.getsize(output_path)
    print(f"Done. Size: {round(total_bytes / (1024**3), 2)} GiB")
    print(f"NVFP4/FP8 layers in metadata: {len(quant_map['layers'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plain NVFP4 convert — Kitchen convert_to_nvfp4_node.py as CLI. "
            "No ConvRot / no calib / no input_scale."
        )
    )
    parser.add_argument(
        "--model",
        "--input",
        dest="model",
        type=str,
        required=True,
        help="Path to input .safetensors",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output .safetensors"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="Z-Image-Turbo",
        choices=sorted(_KITCHEN_PROFILES.keys()),
        help="Kitchen profile (default: Z-Image-Turbo)",
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
        model_type=str(args.model_type),
        device=str(args.device),
    )

import json
import torch
from safetensors.torch import load_file, save_file
import argparse
import os
from tqdm import tqdm


def convert_to_int8(input_path, output_path):
    print(f"Loading model: {input_path}")
    state_dict = load_file(input_path)

    new_state_dict = {}
    quant_meta_layers = {}
    converted_count = 0
    skipped_count = 0

    print("Converting UNet Linear/Conv weights to INT8 (tensorwise, amax/127)...")

    for key, tensor in tqdm(state_dict.items()):
        # Target only SDXL UNet Linear/Conv weights (ndim >= 2).
        # Do NOT INT8 1D .weight (GroupNorm / LayerNorm): those modules are not
        # MixedPrecisionOps and would load int8 as float without applying
        # weight_scale, destroying SSIM (e.g. ~0.0003). HSWQ only touches
        # nn.Linear / nn.Conv2d for the same reason.
        is_unet_matmul_weight = (
            key.startswith("model.diffusion_model")
            and key.endswith(".weight")
            and tensor.ndim >= 2
        )
        if is_unet_matmul_weight and tensor.dtype in [
            torch.float16,
            torch.float32,
            torch.bfloat16,
        ]:
            # Symmetric per-tensor INT8 (ComfyUI int8_tensorwise):
            #   scale = amax / 127
            #   q = round(x / scale).clamp(-127, 127).to(int8)
            w = tensor.float()
            amax = w.abs().max().item()
            amax = max(amax, 1e-6)
            scale = amax / 127.0
            q = (w / scale).round().clamp(-127, 127).to(torch.int8)

            module_key = key[: -len(".weight")]
            new_state_dict[key] = q
            new_state_dict[f"{module_key}.weight_scale"] = torch.tensor(
                scale, dtype=torch.float32
            )
            new_state_dict[f"{module_key}.comfy_quant"] = torch.tensor(
                list(json.dumps({"format": "int8_tensorwise"}).encode("utf-8")),
                dtype=torch.uint8,
            )
            quant_meta_layers[module_key] = {"format": "int8_tensorwise"}
            converted_count += 1
        else:
            # Keep norms, biases, CLIP, VAE, and non-float tensors as-is.
            new_state_dict[key] = tensor
            skipped_count += 1

    metadata = {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": quant_meta_layers}
        )
    }

    print(f"Saving to: {output_path}")
    print(f"Converted layers: {converted_count}, Kept layers: {skipped_count}")

    save_file(new_state_dict, output_path, metadata=metadata)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to input .safetensors")
    parser.add_argument("--output", type=str, required=True, help="Path to output .safetensors")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        exit(1)

    convert_to_int8(args.model, args.output)

import torch
from safetensors.torch import load_file, save_file
import argparse
import os
from tqdm import tqdm

def convert_to_fp8(input_path, output_path):
    print(f"Loading model: {input_path}")
    state_dict = load_file(input_path)
    
    new_state_dict = {}
    converted_count = 0
    skipped_count = 0

    print("Converting UNet to FP8 (E4M3)...")
    
    for key, tensor in tqdm(state_dict.items()):
        # Target only the SDXL UNet portion (model.diffusion_model)
        # SDXL models like Illustrious typically have this key prefix
        if key.startswith("model.diffusion_model"):
            if tensor.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                # Cast to FP8 (E4M3)
                new_state_dict[key] = tensor.to(torch.float8_e4m3fn)
                converted_count += 1
            else:
                new_state_dict[key] = tensor
                skipped_count += 1
        else:
            # Keep CLIP (conditioner) and VAE (first_stage_model) as-is
            new_state_dict[key] = tensor
            skipped_count += 1

    print(f"Saving to: {output_path}")
    print(f"Converted layers: {converted_count}, Kept layers: {skipped_count}")
    
    save_file(new_state_dict, output_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to input .safetensors")
    parser.add_argument("--output", type=str, required=True, help="Path to output .safetensors")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        exit(1)

    convert_to_fp8(args.model, args.output)
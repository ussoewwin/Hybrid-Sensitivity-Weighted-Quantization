import argparse
import torch

import transformers_clip_compat

transformers_clip_compat.apply()

from diffusers import StableDiffusionXLPipeline
import numpy as np
from PIL import Image, ImageChops
from skimage.metrics import structural_similarity as ssim
import os
import gc
import time
import sys

# ComfyUI-native checkpoint loader so HSWQ weight_scale / comfy_quant sidecars
# are interpreted by comfy/quant_ops.py (QUANT_ALGOS float8_e4m3fn / int8_tensorwise).
COMFY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ComfyUI-master"))
if COMFY_PATH not in sys.path:
    sys.path.insert(0, COMFY_PATH)

import logging
logging.getLogger("comfy").setLevel(logging.WARNING)

try:
    import nodes
    import folder_paths
    import comfy.model_management
except ImportError as e:
    print(f"Error: Could not import ComfyUI modules from {COMFY_PATH}: {e}")
    print("Ensure ComfyUI-master is present at the repo root.")
    sys.exit(1)

# Enforce deterministic behavior for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_pipeline(path, device="cuda"):
    print(f"Loading model: {os.path.basename(path)}...")
    try:
        # Register the checkpoint's directory with ComfyUI so CheckpointLoaderSimple can find it.
        directory = os.path.dirname(os.path.abspath(path))
        folder_paths.add_model_folder_path("checkpoints", directory)

        loader = nodes.CheckpointLoaderSimple()
        model, clip, vae = loader.load_checkpoint(ckpt_name=os.path.basename(path))
        return model, clip, vae
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def generate_image_fixed(model, clip, vae, prompt, seed, steps):
    # Create fixed-seed generator
    generator = torch.Generator("cuda").manual_seed(seed)

    # Prompt encoding via ComfyUI CLIPTextEncode
    enc = nodes.CLIPTextEncode()
    positive = enc.encode(clip=clip, text=prompt)[0]
    negative = enc.encode(clip=clip, text="")[0]

    # Empty latent
    empty = nodes.EmptyLatentImage()
    latent = empty.generate(width=1024, height=1024, batch_size=1)[0]

    # KSampler
    sampler = nodes.KSampler()

    start_time = time.time()
    samples = sampler.sample(
        model, seed, steps, 7.0,
        "dpmpp_2m", "karras",
        positive, negative, latent, denoise=1.0
    )[0]
    end_time = time.time()

    # Decode
    dec = nodes.VAEDecode()
    image_tensor = dec.decode(vae=vae, samples=samples)[0]
    img_array = 255.0 * image_tensor[0].cpu().numpy()
    image = Image.fromarray(np.clip(img_array, 0, 255).astype("uint8"))

    return image, end_time - start_time

def calculate_metrics(img1, img2):
    arr1 = np.array(img1)
    arr2 = np.array(img2)

    # MSE (mean squared error)
    mse = np.mean((arr1 - arr2) ** 2)

    # SSIM (structural similarity)
    score_ssim = ssim(arr1, arr2, win_size=3, channel_axis=2, data_range=255)

    return mse, score_ssim

def main():
    parser = argparse.ArgumentParser(description="Robust SDXL FP8 Fidelity Benchmark")
    parser.add_argument("--fp16", required=True, help="Path to Baseline (FP16) model")
    parser.add_argument("--fp8", required=True, help="Path to Quantized (FP8) model")
    parser.add_argument("--prompt", required=True, help="Benchmark prompt")
    parser.add_argument("--seed", type=int, default=123456789, help="Fixed seed for reproduction")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"--- Benchmark Config ---")
    print(f"Seed: {args.seed}")
    print(f"Steps: {args.steps}")
    print(f"Prompt: {args.prompt[:50]}...")
    print(f"------------------------")

    # 1. FP16 (Baseline) Generation
    print("\n=== 1. Generating Baseline (FP16) ===")
    model16, clip16, vae16 = load_pipeline(args.fp16, device)
    img_fp16, time_fp16 = generate_image_fixed(model16, clip16, vae16, args.prompt, args.seed, args.steps)
    img_fp16.save("bench_result_fp16.png")
    print(f"FP16 Time: {time_fp16:.2f}s")

    # Full memory release (keep CLIP/VAE for FP8 side to isolate UNet difference)
    del model16
    gc.collect()
    torch.cuda.empty_cache()

    # 2. FP8 (Quantized) Generation
    print("\n=== 2. Generating Quantized (FP8) ===")
    model8, clip8, vae8 = load_pipeline(args.fp8, device)
    img_fp8, time_fp8 = generate_image_fixed(model8, clip16, vae16, args.prompt, args.seed, args.steps)
    img_fp8.save("bench_result_fp8.png")
    print(f"FP8 Time: {time_fp8:.2f}s")

    del model8, clip8, vae8, clip16, vae16
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Comparison
    print("\n=== 3. Calculating Metrics ===")

    # Size check (prevent error when models/settings differ)
    if img_fp16.size != img_fp8.size:
        print(f"Error: Image sizes do not match! FP16:{img_fp16.size}, FP8:{img_fp8.size}")
        print("Different models or settings used.")
        sys.exit(1)

    mse, score = calculate_metrics(img_fp16, img_fp8)

    print(f"--------------------------------------------------")
    print(f"MSE (Error): {mse:.4f} \t(0 is perfect match)")
    print(f"SSIM (Sim) : {score:.4f} \t(1.0 is perfect match)")
    print(f"--------------------------------------------------")

    # Grading logic
    if score > 0.98:
        grade = "PERFECT (S)"
    elif score > 0.95:
        grade = "EXCELLENT (A)"
    elif score > 0.90:
        grade = "GOOD (B)"
    else:
        grade = "WARNING (C)"

    print(f"Quality Grade: {grade}")

    # Difference image generation
    diff_img = ImageChops.difference(img_fp16, img_fp8)
    diff_img = ImageChops.multiply(diff_img, Image.new('RGB', diff_img.size, (10, 10, 10)))
    diff_img.save("bench_result_diff.png")
    print("Diff image saved: bench_result_diff.png")

if __name__ == "__main__":
    main()

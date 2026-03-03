import argparse
import torch
from diffusers import StableDiffusionXLPipeline
import numpy as np
from PIL import Image, ImageChops
from skimage.metrics import structural_similarity as ssim
import os
import gc
import time
import sys

# Enforce deterministic behavior for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_pipeline(path, device="cuda"):
    print(f"Loading model: {os.path.basename(path)}...")
    try:
        pipe = StableDiffusionXLPipeline.from_single_file(
            path,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(device)
        return pipe
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def generate_image_fixed(pipe, prompt, seed, steps):
    # Create fixed-seed generator
    generator = torch.Generator("cuda").manual_seed(seed)
    
    start_time = time.time()
    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        generator=generator,  # Pass fixed generator
        output_type="pil"
    ).images[0]
    end_time = time.time()
    
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
    pipe = load_pipeline(args.fp16, device)
    img_fp16, time_fp16 = generate_image_fixed(pipe, args.prompt, args.seed, args.steps)
    img_fp16.save("bench_result_fp16.png")
    print(f"FP16 Time: {time_fp16:.2f}s")
    
    # Full memory release
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # 2. FP8 (Quantized) Generation
    print("\n=== 2. Generating Quantized (FP8) ===")
    pipe = load_pipeline(args.fp8, device)
    img_fp8, time_fp8 = generate_image_fixed(pipe, args.prompt, args.seed, args.steps)
    img_fp8.save("bench_result_fp8.png")
    print(f"FP8 Time: {time_fp8:.2f}s")
    
    del pipe
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
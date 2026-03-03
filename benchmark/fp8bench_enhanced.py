"""
Enhanced SDXL FP8 Benchmark
Supports both single-sample and multi-sample modes with statistical analysis

Usage:
    # Single sample (original behavior)
    python fp8bench_enhanced.py --fp16 test.safetensors --fp8 quantized.safetensors --prompt "your prompt"
    
    # Multi-sample with statistics
    python fp8bench_enhanced.py --fp16 test.safetensors --fp8 quantized.safetensors --multi --num_prompts 5 --num_seeds 3
"""

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
import json
from pathlib import Path

# Optional imports
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Enforce deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Default prompts for multi-sample mode
DEFAULT_PROMPTS = [
    "portrait of a woman, professional photography, studio lighting",
    "mountain landscape at sunset, dramatic clouds, golden hour",
    "modern architecture, glass building, urban cityscape",
    "colorful abstract art, geometric shapes, vibrant colors",
    "cyberpunk city at night, neon signs, highly detailed",
]


class VRAMMeasurer:
    def __init__(self):
        self.available = PYNVML_AVAILABLE
        if self.available:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                self.available = False
    
    def get_used_mb(self):
        if self.available:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                return info.used / (1024**2)
            except:
                return torch.cuda.memory_allocated() / (1024**2)
        else:
            return torch.cuda.memory_allocated() / (1024**2)
    
    def __del__(self):
        if self.available:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


def load_pipeline(path, device="cuda"):
    print(f"Loading model: {os.path.basename(path)}...")
    try:
        pipe = StableDiffusionXLPipeline.from_single_file(
            path,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(device)
        pipe.set_progress_bar_config(disable=True)
        return pipe
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def warmup_pipeline(pipe, steps=20, warmup_runs=2):
    print(f"Warming up ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        with torch.no_grad():
            pipe(prompt="warmup", num_inference_steps=steps, output_type="latent")
        torch.cuda.empty_cache()
    print("Warmup complete.")


def generate_image_with_metrics(pipe, prompt, seed, steps, vram_measurer=None):
    generator = torch.Generator("cuda").manual_seed(seed)
    
    torch.cuda.reset_peak_memory_stats()
    
    if vram_measurer:
        torch.cuda.synchronize()
        baseline_vram = vram_measurer.get_used_mb()
    
    start_time = time.time()
    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        generator=generator,
        output_type="pil"
    ).images[0]
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    if vram_measurer:
        peak_vram = vram_measurer.get_used_mb()
        vram_used = peak_vram - baseline_vram
    else:
        vram_used = torch.cuda.max_memory_allocated() / (1024**2)
    
    return image, elapsed, vram_used


def calculate_ssim(img1, img2):
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    if arr1.shape != arr2.shape:
        raise ValueError(f"Image sizes do not match: {arr1.shape} vs {arr2.shape}")
    
    score = ssim(arr1, arr2, win_size=3, channel_axis=2, data_range=255)
    return score


def calculate_psnr(img1, img2):
    arr1 = np.array(img1).astype(np.float64)
    arr2 = np.array(img2).astype(np.float64)
    
    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        return float("inf")
    
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_lpips(img1, img2, loss_fn):
    from torchvision import transforms
    to_tensor = transforms.ToTensor()
    t1 = to_tensor(img1).unsqueeze(0).cuda() * 2 - 1
    t2 = to_tensor(img2).unsqueeze(0).cuda() * 2 - 1
    
    with torch.no_grad():
        dist = loss_fn(t1, t2)
    
    return dist.item()


def calculate_mse(img1, img2):
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    return np.mean((arr1 - arr2) ** 2)


def single_sample_benchmark(args, device):
    """Original single-sample benchmark mode"""
    print(f"--- Single Sample Benchmark ---")
    print(f"Seed: {args.seed}")
    print(f"Steps: {args.steps}")
    print(f"Prompt: {args.prompt[:50]}...")
    print(f"-------------------------------")
    
    vram_measurer = VRAMMeasurer() if PYNVML_AVAILABLE else None
    
    # FP16
    print("\n=== 1. Generating Baseline (FP16) ===")
    pipe = load_pipeline(args.fp16, device)
    if args.warmup:
        warmup_pipeline(pipe, args.steps, args.warmup_runs)
    
    img_fp16, time_fp16, vram_fp16 = generate_image_with_metrics(
        pipe, args.prompt, args.seed, args.steps, vram_measurer
    )
    img_fp16.save("bench_result_fp16.png")
    print(f"FP16 Time: {time_fp16:.2f}s | VRAM: {vram_fp16:.1f} MB")
    
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    
    # FP8
    print("\n=== 2. Generating Quantized (FP8) ===")
    pipe = load_pipeline(args.fp8, device)
    if args.warmup:
        warmup_pipeline(pipe, args.steps, args.warmup_runs)
    
    img_fp8, time_fp8, vram_fp8 = generate_image_with_metrics(
        pipe, args.prompt, args.seed, args.steps, vram_measurer
    )
    img_fp8.save("bench_result_fp8.png")
    print(f"FP8 Time: {time_fp8:.2f}s | VRAM: {vram_fp8:.1f} MB")
    
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    
    # Metrics
    print("\n=== 3. Calculating Metrics ===")
    
    if img_fp16.size != img_fp8.size:
        print(f"Error: Image sizes do not match!")
        sys.exit(1)
    
    ssim_score = calculate_ssim(img_fp16, img_fp8)
    mse = calculate_mse(img_fp16, img_fp8)
    psnr = calculate_psnr(img_fp16, img_fp8)
    
    print(f"--------------------------------------------------")
    print(f"SSIM:  {ssim_score:.4f} (1.0 is perfect)")
    print(f"PSNR:  {psnr:.2f} dB (higher is better)")
    print(f"MSE:   {mse:.4f} (0 is perfect)")
    
    if LPIPS_AVAILABLE:
        loss_fn = lpips.LPIPS(net="alex").cuda().eval()
        lpips_score = calculate_lpips(img_fp16, img_fp8, loss_fn)
        print(f"LPIPS: {lpips_score:.4f} (0 is perfect)")
    
    print(f"--------------------------------------------------")
    print(f"Time:  FP16 {time_fp16:.2f}s | FP8 {time_fp8:.2f}s")
    print(f"VRAM:  FP16 {vram_fp16:.1f} MB | FP8 {vram_fp8:.1f} MB")
    vram_saved = vram_fp16 - vram_fp8
    print(f"       Saved: {vram_saved:.1f} MB ({vram_saved/vram_fp16*100:.1f}%)")
    print(f"--------------------------------------------------")
    
    # Grading
    if ssim_score > 0.98:
        grade = "PERFECT (S)"
    elif ssim_score > 0.95:
        grade = "EXCELLENT (A)"
    elif ssim_score > 0.90:
        grade = "GOOD (B)"
    elif ssim_score > 0.85:
        grade = "ACCEPTABLE (C)"
    else:
        grade = "POOR (D)"
    
    print(f"Quality Grade: {grade}")
    
    # Diff image
    diff_img = ImageChops.difference(img_fp16, img_fp8)
    diff_img = ImageChops.multiply(diff_img, Image.new("RGB", diff_img.size, (10, 10, 10)))
    diff_img.save("bench_result_diff.png")
    print("\nSaved: bench_result_fp16.png, bench_result_fp8.png, bench_result_diff.png")


def multi_sample_benchmark(args, device):
    """Multi-sample benchmark mode with statistical analysis"""
    print(f"--- Multi-Sample Benchmark ---")
    print(f"Seeds: {args.num_seeds}")
    print(f"Prompts: {args.num_prompts}")
    print(f"Total: {args.num_seeds * args.num_prompts} samples per model")
    print(f"Steps: {args.steps}")
    print(f"-------------------------------")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    vram_measurer = VRAMMeasurer() if PYNVML_AVAILABLE else None
    
    prompts = DEFAULT_PROMPTS[:args.num_prompts]
    seeds = list(range(42, 42 + args.num_seeds))
    
    # Initialize LPIPS
    loss_fn = None
    if LPIPS_AVAILABLE:
        print("Initializing LPIPS...")
        loss_fn = lpips.LPIPS(net="alex").cuda().eval()
    
    results_fp16 = {"images": [], "times": [], "vrams": []}
    results_fp8 = {"images": [], "times": [], "vrams": []}
    
    # Benchmark FP16
    print("\n=== Benchmarking FP16 ===")
    pipe = load_pipeline(args.fp16, device)
    if args.warmup:
        warmup_pipeline(pipe, args.steps, args.warmup_runs)
    
    iterator = range(len(prompts) * len(seeds))
    if TQDM_AVAILABLE:
        iterator = tqdm(iterator, desc="FP16")
    
    for i in iterator:
        prompt_idx = i // len(seeds)
        seed_idx = i % len(seeds)
        img, t, v = generate_image_with_metrics(
            pipe, prompts[prompt_idx], seeds[seed_idx], args.steps, vram_measurer
        )
        results_fp16["images"].append(img)
        results_fp16["times"].append(t)
        results_fp16["vrams"].append(v)
        torch.cuda.empty_cache()
    
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    
    # Benchmark FP8
    print("\n=== Benchmarking FP8 ===")
    pipe = load_pipeline(args.fp8, device)
    if args.warmup:
        warmup_pipeline(pipe, args.steps, args.warmup_runs)
    
    iterator = range(len(prompts) * len(seeds))
    if TQDM_AVAILABLE:
        iterator = tqdm(iterator, desc="FP8")
    
    for i in iterator:
        prompt_idx = i // len(seeds)
        seed_idx = i % len(seeds)
        img, t, v = generate_image_with_metrics(
            pipe, prompts[prompt_idx], seeds[seed_idx], args.steps, vram_measurer
        )
        results_fp8["images"].append(img)
        results_fp8["times"].append(t)
        results_fp8["vrams"].append(v)
        torch.cuda.empty_cache()
    
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    
    # Calculate metrics
    print("\n=== Calculating Metrics ===")
    metrics = {"ssim": [], "psnr": [], "mse": [], "lpips": []}
    
    iterator = range(len(results_fp16["images"]))
    if TQDM_AVAILABLE:
        iterator = tqdm(iterator, desc="Metrics")
    
    for i in iterator:
        img16 = results_fp16["images"][i]
        img8 = results_fp8["images"][i]
        
        metrics["ssim"].append(calculate_ssim(img16, img8))
        metrics["psnr"].append(calculate_psnr(img16, img8))
        metrics["mse"].append(calculate_mse(img16, img8))
        
        if loss_fn:
            metrics["lpips"].append(calculate_lpips(img16, img8, loss_fn))
    
    # Print results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    print("\n--- Quality Metrics ---")
    print(f"SSIM:  {np.mean(metrics['ssim']):.4f} ± {np.std(metrics['ssim']):.4f}")
    print(f"       Min: {np.min(metrics['ssim']):.4f} | Max: {np.max(metrics['ssim']):.4f}")
    print(f"PSNR:  {np.mean(metrics['psnr']):.2f} ± {np.std(metrics['psnr']):.2f} dB")
    print(f"MSE:   {np.mean(metrics['mse']):.4f} ± {np.std(metrics['mse']):.4f}")
    if loss_fn:
        print(f"LPIPS: {np.mean(metrics['lpips']):.4f} ± {np.std(metrics['lpips']):.4f}")
    
    print("\n--- Performance ---")
    print(f"Time (FP16): {np.mean(results_fp16['times']):.2f} ± {np.std(results_fp16['times']):.2f}s")
    print(f"Time (FP8):  {np.mean(results_fp8['times']):.2f} ± {np.std(results_fp8['times']):.2f}s")
    speedup = np.mean(results_fp16['times']) / np.mean(results_fp8['times'])
    print(f"Speedup:     {speedup:.2f}x")
    
    print(f"\nVRAM (FP16): {np.mean(results_fp16['vrams']):.1f} ± {np.std(results_fp16['vrams']):.1f} MB")
    print(f"VRAM (FP8):  {np.mean(results_fp8['vrams']):.1f} ± {np.std(results_fp8['vrams']):.1f} MB")
    vram_saved = np.mean(results_fp16['vrams']) - np.mean(results_fp8['vrams'])
    print(f"Saved:       {vram_saved:.1f} MB ({vram_saved/np.mean(results_fp16['vrams'])*100:.1f}%)")
    
    # Grade
    mean_ssim = np.mean(metrics['ssim'])
    if mean_ssim > 0.98:
        grade = "S (PERFECT)"
    elif mean_ssim > 0.95:
        grade = "A (EXCELLENT)"
    elif mean_ssim > 0.90:
        grade = "B (GOOD)"
    elif mean_ssim > 0.85:
        grade = "C (ACCEPTABLE)"
    else:
        grade = "D (POOR)"
    
    print(f"\nQuality Grade: {grade}")
    
    # Save samples
    print(f"\nSaving samples to {output_dir}...")
    for i in range(min(5, len(results_fp16['images']))):
        results_fp16['images'][i].save(output_dir / f"sample_{i:02d}_fp16.png")
        results_fp8['images'][i].save(output_dir / f"sample_{i:02d}_fp8.png")
    
    # Save JSON
    results_json = {
        "config": {
            "fp16_model": args.fp16,
            "fp8_model": args.fp8,
            "steps": args.steps,
            "num_prompts": args.num_prompts,
            "num_seeds": args.num_seeds,
        },
        "metrics": {
            "ssim": {"mean": float(np.mean(metrics['ssim'])), "std": float(np.std(metrics['ssim']))},
            "psnr": {"mean": float(np.mean(metrics['psnr'])), "std": float(np.std(metrics['psnr']))},
            "mse": {"mean": float(np.mean(metrics['mse'])), "std": float(np.std(metrics['mse']))},
        },
        "performance": {
            "time_fp16": {"mean": float(np.mean(results_fp16['times'])), "std": float(np.std(results_fp16['times']))},
            "time_fp8": {"mean": float(np.mean(results_fp8['times'])), "std": float(np.std(results_fp8['times']))},
            "vram_fp16": {"mean": float(np.mean(results_fp16['vrams'])), "std": float(np.std(results_fp16['vrams']))},
            "vram_fp8": {"mean": float(np.mean(results_fp8['vrams'])), "std": float(np.std(results_fp8['vrams']))},
        },
        "grade": grade,
    }
    
    if loss_fn:
        results_json["metrics"]["lpips"] = {"mean": float(np.mean(metrics['lpips'])), "std": float(np.std(metrics['lpips']))}
    
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    
    print(f"Results saved to {output_dir / 'benchmark_results.json'}")


def main():
    parser = argparse.ArgumentParser(description="SDXL FP8 Fidelity Benchmark")
    parser.add_argument("--fp16", required=True, help="FP16 baseline model path")
    parser.add_argument("--fp8", required=True, help="FP8 quantized model path")
    parser.add_argument("--prompt", default=None, help="Benchmark prompt (single-sample mode)")
    parser.add_argument("--seed", type=int, default=123456789, help="Seed (single-sample mode)")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--warmup", action="store_true", help="Enable warmup runs")
    parser.add_argument("--warmup_runs", type=int, default=2, help="Number of warmup runs")
    
    # Multi-sample mode
    parser.add_argument("--multi", action="store_true", help="Multi-sample benchmark mode")
    parser.add_argument("--num_prompts", type=int, default=5, help="Number of prompts (multi mode)")
    parser.add_argument("--num_seeds", type=int, default=3, help="Number of seeds (multi mode)")
    parser.add_argument("--output_dir", default="./benchmark_results", help="Output directory (multi mode)")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("ERROR: CUDA not available")
        sys.exit(1)
    
    # Mode selection
    if args.multi:
        multi_sample_benchmark(args, device)
    else:
        if args.prompt is None:
            args.prompt = "a beautiful landscape, professional photography, high quality"
            print(f"No prompt specified, using default: {args.prompt}")
        single_sample_benchmark(args, device)


if __name__ == "__main__":
    main()


import sys
import os
import torch
import time
import argparse
import numpy as np
from PIL import Image, ImageChops
from skimage.metrics import structural_similarity as ssim

# ComfyUIのパス設定 (カレントディレクトリのComfyUI-masterを想定)
# ユーザー環境に合わせて調整が必要な場合はここを変更
COMFY_PATH = os.path.abspath("ComfyUI-master")
if COMFY_PATH not in sys.path:
    sys.path.append(COMFY_PATH)

try:
    import nodes
    import folder_paths
    import comfy.model_management
except ImportError:
    print(f"Error: Could not import ComfyUI modules from {COMFY_PATH}")
    print("Please ensure the script is running from the correct directory or update COMFY_PATH.")
    sys.exit(1)

# 警告抑制
import logging
logging.getLogger("comfy").setLevel(logging.WARNING)

def setup_paths(args):
    """モデルファイルのパスをComfyUIに登録する"""
    def register_path(folder_type, file_path):
        if not file_path: return
        directory = os.path.dirname(os.path.abspath(file_path))
        folder_paths.add_model_folder_path(folder_type, directory)

    register_path("diffusion_models", args.fp16)
    register_path("diffusion_models", args.fp8)
    register_path("text_encoders", args.clip_path)
    register_path("text_encoders", args.t5_path)
    register_path("vae", args.vae_path)

def generate_image_comfy(unet_name, clip_obj, vae_obj, args):
    """ComfyUIノードを使用して画像を生成する"""
    print(f"Loading UNet: {unet_name}")
    unet_loader = nodes.UNETLoader()
    unet = unet_loader.load_unet(unet_name=unet_name, weight_dtype="default")[0]

    # Prompt Encoding
    print("Encoding Prompt...")
    clip_text_encode = nodes.CLIPTextEncode()
    positive = clip_text_encode.encode(clip=clip_obj, text=args.prompt)[0]
    negative = clip_text_encode.encode(clip=clip_obj, text="")[0]

    # Empty Latent
    print(f"Creating Latent ({args.width}x{args.height})...")
    empty_latent = nodes.EmptyLatentImage()
    latent = empty_latent.generate(width=args.width, height=args.height, batch_size=1)[0]

    # KSampler
    print("Sampling...")
    ksampler = nodes.KSampler()
    
    # ComfyUIのシードはランダム生成されることが多いが、再現性のために固定
    seed = args.seed
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # KSampler (Flux recommended scheduler 'simple')
    sampler = nodes.KSampler()
    samples = sampler.sample(unet, seed, args.steps, args.guidance_scale, "euler", "simple", positive, negative, latent, denoise=1.0)[0]
    
    torch.cuda.synchronize()
    end_time = time.time()
    elapsed = end_time - start_time

    raw_latent = samples["samples"].clone()

    # Decode
    print("Decoding...")
    vae_decode = nodes.VAEDecode()
    image_tensor = vae_decode.decode(vae=vae_obj, samples=samples)[0] # [Batch, H, W, C]
    
    # Convert to PIL
    img_array = 255. * image_tensor[0].cpu().numpy()
    img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    return img, raw_latent, elapsed

def load_common_components(args):
    """CLIPとVAEをロード（共通）"""
    print("Loading CLIP/T5...")
    dual_clip_loader = nodes.DualCLIPLoader()
    clip = dual_clip_loader.load_clip(
        clip_name1=os.path.basename(args.clip_path),
        clip_name2=os.path.basename(args.t5_path),
        type="flux"
    )[0]

    print("Loading VAE...")
    vae_loader = nodes.VAELoader()
    vae = vae_loader.load_vae(vae_name=os.path.basename(args.vae_path))[0]
    
    return clip, vae

def calculate_latent_mse(l1, l2):
    """MSE on raw latent tensors (before VAE decode)."""
    arr1 = l1[0].cpu().float().numpy()
    arr2 = l2[0].cpu().float().numpy()
    return float(np.mean((arr1 - arr2) ** 2))

def calculate_metrics(img1, img2):
    # Convert to numpy arrays
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)

    # MSE (Mean Squared Error)
    mse = np.mean((arr1 - arr2) ** 2)

    # SSIM (Structural Similarity)
    # Convert to grayscale for SSIM as per standard practice
    gray1 = np.array(img1.convert('L'))
    gray2 = np.array(img2.convert('L'))
    
    score, _ = ssim(gray1, gray2, full=True)

    return mse, score

def main():
    parser = argparse.ArgumentParser(description="Flux1 FP8 ComfyUI Native Benchmark")
    parser.add_argument("--fp16", required=True, help="FP16 UNet Path")
    parser.add_argument("--fp8", required=True, help="FP8 UNet Path")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=123456789)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--clip_path", type=str, required=True)
    parser.add_argument("--t5_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--token", type=str) # Ignored in ComfyUI local
    parser.add_argument("--save_image", action="store_true", help="Save generated images")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_paths(args)

    comfy.model_management.get_torch_device()

    clip, vae = load_common_components(args)

    print(f"--- Benchmark Config ---")
    print(f"Seed: {args.seed}")
    print(f"Steps: {args.steps}")
    print(f"Prompt: {args.prompt[:50]}...")
    print(f"------------------------")

    # 1. FP16 (Baseline) Generation
    print("\n=== 1. Generating Baseline (FP16) ===")
    img_fp16, lat_fp16, t16 = generate_image_comfy(os.path.basename(args.fp16), clip, vae, args)
    if args.save_image:
        fp16_path = os.path.join(args.output_dir, "bench_result_fp16.png")
        img_fp16.save(fp16_path)
    print(f"FP16 Time: {t16:.2f}s")
    
    comfy.model_management.soft_empty_cache()

    # 2. FP8 (Quantized) Generation
    print("\n=== 2. Generating Quantized (FP8) ===")
    img_fp8, lat_fp8, t8 = generate_image_comfy(os.path.basename(args.fp8), clip, vae, args)
    if args.save_image:
        fp8_path = os.path.join(args.output_dir, "bench_result_fp8.png")
        img_fp8.save(fp8_path)
    print(f"FP8 Time: {t8:.2f}s")
    
    comfy.model_management.soft_empty_cache()

    # 3. Comparison
    print("\n=== 3. Calculating Metrics ===")

    if img_fp16.size != img_fp8.size:
        print("Error: Image sizes do not match!")
        return

    mse = calculate_latent_mse(lat_fp16, lat_fp8)
    _, score = calculate_metrics(img_fp16, img_fp8)

    print(f"--------------------------------------------------")
    print(f"MSE (Error): {mse:.6f} \t(0 is perfect match)")
    print(f"SSIM (Sim) : {score:.4f} \t(1.0 is perfect match)")
    print(f"--------------------------------------------------")

    # Grade Logic
    if score > 0.98:
        grade = "PERFECT (S)"
    elif score > 0.95:
        grade = "EXCELLENT (A)"
    elif score > 0.90:
        grade = "GOOD (B)"
    else:
        grade = "WARNING (C)"
    
    print(f"Quality Grade: {grade}")

    # Save diff
    if args.save_image:
        diff_img = ImageChops.difference(img_fp16, img_fp8)
        # diff_img = ImageChops.multiply(diff_img, Image.new('RGB', diff_img.size, (10, 10, 10))) # Remove generic multiply
        diff_img_path = os.path.join(args.output_dir, "bench_result_diff.png")
        diff_img.save(diff_img_path)
        print(f"Diff image saved: {os.path.basename(diff_img_path)}")

if __name__ == "__main__":
    main()

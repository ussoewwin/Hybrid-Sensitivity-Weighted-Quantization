import argparse
import torch
import torch.nn as nn
from safetensors.torch import load_file
import os
import gc
import time
import sys
import numpy as np
from PIL import Image, ImageChops
from skimage.metrics import structural_similarity as ssim
import json

print("Starting Antigravity ZIT Bench v25 (Ghost Weight Hunter)...")

# Helper for path resolution
def resolve_path(path, is_file=True):
    if not path: return None
    if os.path.exists(path): return path
    
    target = os.path.basename(path)
    print(f"  Note: {target} not found at {path}. Searching recursively...")
    for root, dirs, files in os.walk("."):
        # Skip hidden directories (like .local, .cache, .Trash)
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        root_abs = os.path.abspath(root)
        if "ComfyUI" in root_abs or "node_modules" in root_abs:
            continue
        if is_file:
            if target in files:
                found = os.path.join(root, target)
                print(f"  Found: {found}")
                return found
    return path

def resolve_tokenizer_path(provided_path, clip_resolved_path):
    validation_files = ["tokenizer.json", "vocab.json", "config.json"]
    
    # Priority 1: User provided path
    if provided_path and os.path.isdir(provided_path):
        if any(os.path.exists(os.path.join(provided_path, f)) for f in validation_files):
            return provided_path, True
            
    # Priority 2: Directory of the resolved CLIP file
    if clip_resolved_path:
        clip_dir = os.path.dirname(os.path.abspath(clip_resolved_path))
        if any(os.path.exists(os.path.join(clip_dir, f)) for f in validation_files):
            return clip_dir, True
        
    # Priority 3: Recursive search, excluding ComfyUI/SD1
    print("  Note: Searching recursively for Qwen 3 4B tokenizer (excluding ComfyUI/SD1)...")
    for root, dirs, files in os.walk("."):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        root_abs = os.path.abspath(root)
        if "ComfyUI" in root_abs or "sd1" in root_abs.lower() or "clip_l" in root_abs.lower() or ".git" in root_abs:
            continue
        if any(f in files for f in validation_files):
            if "tokenizer.json" in files or "tokenizer_config.json" in files:
                print(f"  Found potential tokenizer: {root}")
                return root, True
                
    # Priority 4: Default HF repo
    default_repo = "Qwen/Qwen2.5-7B"
    print(f"  Note: No local tokenizer found. Falling back to HF repo: {default_repo}")
    return default_repo, False

def latent_to_img(l):
    l = l[0].permute(1, 2, 0).cpu().float().numpy()
    l = (l - l.min()) / (l.max() - l.min() + 1e-6) * 255
    return Image.fromarray(l[:, :, :3].astype(np.uint8))

# Enforce deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def detect_zit_config_from_keys(state_dict):
    state_dict_keys = list(state_dict.keys())
    zit_config = {}
    layer_indices = set()
    for key in state_dict_keys:
        if key.startswith("layers."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_indices.add(int(parts[1]))
    
    zit_config["num_layers"] = max(layer_indices) + 1 if layer_indices else 30
    if "x_embedder.weight" in state_dict:
        zit_config["hidden_size"] = state_dict["x_embedder.weight"].shape[0]
    else:
        zit_config["hidden_size"] = 3072
    
    refiner_indices = set()
    for key in state_dict_keys:
        if key.startswith("context_refiner."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                refiner_indices.add(int(parts[1]))
    zit_config["num_context_refiner"] = max(refiner_indices) + 1 if refiner_indices else 2
    
    # Detect Intermediate Size (MLP Dim) to prevent size mismatch
    # Check layers.0.feed_forward.w1.weight shape -> [intermediate_size, hidden_size]
    w1_key = "layers.0.feed_forward.w1.weight"
    if w1_key in state_dict:
        zit_config["intermediate_size"] = state_dict[w1_key].shape[0]
        print(f"  Detected Intermediate Size: {zit_config['intermediate_size']}")
    else:
        zit_config["intermediate_size"] = None # Let model default logic handle or fail
        
    return zit_config

def load_zit_model(path, device="cuda", comfy_path=None, is_fp8=False):
    if comfy_path and comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)
    
    from comfy.ldm.lumina.model import NextDiT
    import comfy.ops
    
    args_path = resolve_path(path, is_file=True)
    print(f"Loading state_dict: {os.path.basename(args_path)}")
    state_dict = load_file(args_path)
    config = detect_zit_config_from_keys(state_dict)
    
    if is_fp8:
        print(f"Using mixed_precision_ops for FP8 model load...")
        ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.float16)
    else:
        print(f"Using standard operations for FP16 model load...")
        ops = comfy.ops.disable_weight_init
        
    
    # Calculate MLP Ratio if intermediate_size detected
    # NextDiT uses mlp_ratio (default 4.0) to calc hidden dim: int(dim * mlp_ratio) 
    # We must reverse this to match checkpoint.
    # intermediate_size = 10240, dim = 3840 -> ratio = 2.6666...
    
    kwargs = {}
    if config.get("intermediate_size"):
        ratio = config["intermediate_size"] / config["hidden_size"]
        kwargs["ffn_dim_multiplier"] = ratio  # CORRECT argument name!
        print(f"  Calculated FFN Dim Multiplier: {ratio:.4f} (Dim: {config['hidden_size']} -> {config['intermediate_size']})")

    import inspect
    print(f"  Debug: NextDiT Signature: {inspect.signature(NextDiT.__init__)}")
    
    model = NextDiT(
        patch_size=2,
        in_channels=16,
        dim=config["hidden_size"],
        n_layers=config["num_layers"],
        n_refiner_layers=config["num_context_refiner"],
        n_heads=config["hidden_size"] // 128,
        n_kv_heads=config["hidden_size"] // 128,
        multiple_of=256,
        norm_eps=1e-5,
        cap_feat_dim=2560,
        z_image_modulation=True,
        pad_tokens_multiple=64,
        device="cpu",
        dtype=torch.float16,
        operations=ops,
        **kwargs
    )
    
    converted_dict = {}
    for k, v in state_dict.items():
        if v.dtype == torch.bfloat16:
            converted_dict[k] = v.to(torch.float16)
        else:
            converted_dict[k] = v
            
    # CRITICAL: Do NOT call model.state_dict() here for FP8
    try:
        missing, unexpected = model.load_state_dict(converted_dict, strict=False)
    except RuntimeError as e:
        print(f"  CRITICAL ERROR: Model Size Mismatch despite config adjustment.")
        print(f"  Error: {e}")
        print(f"  Attempted kwargs: {kwargs}")
        print(f"  NextDiT Signature: {inspect.signature(NextDiT.__init__)}")
        sys.exit(1)
    
    if len(missing) == len(list(model.parameters())) + len(list(model.buffers())):
        if any(k.startswith("model.") for k in converted_dict.keys()):
            print("  Note: No keys matched. Attempting to remove 'model.' prefix...")
            remapped_dict = {k[6:]: v for k, v in converted_dict.items()}
            missing, unexpected = model.load_state_dict(remapped_dict, strict=False)

    print(f"  [Keys] Matched: {len(converted_dict) - len(unexpected)}, Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    if len(missing) > len(list(model.parameters())) * 0.5:
        print(f"  Warning: Many keys are still missing. First 5 missing: {list(missing)[:5]}")

    if is_fp8:
        model = model.to(device)
        print(f"  Note: FP8 model loaded on {device}. (Weights managed by mixed_precision_ops)")
    else:
        model = model.to(device).to(torch.float16)
        print(f"  Note: FP16 model loaded on {device} and cast to float16.")
        
    model.eval()
    return model

def encode_prompt(prompt, text_encoder, tokenizer, device):
    template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    formatted = template.format(prompt)
    
    tokens = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )
    
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            intermediate_output=-2
        )
        hidden_states = outputs[1]
    
    return hidden_states, attention_mask.bool()

def run_inference(model, prompt_embeds, prompt_mask, steps, seed, device):
    import comfy.k_diffusion.sampling as k_sampling
    
    class ZITWrapper:
        def __init__(self, model, embeds, mask):
            self.model = model
            self.embeds = embeds
            self.mask = mask
        def __call__(self, x, sigma, **kwargs):
            dtype = torch.float16
            
            # DEBUG: Inspect Ghost Weight
            if sigma[0] > 0.9:
                try:
                    for name, module in self.model.named_modules():
                        if "layers.10" in name and "attention" in name and hasattr(module, "qkv"):
                            w = module.qkv.weight
                            print(f"  [Inference Debug] Active weight 'layers.10...qkv': dtype={w.dtype}, device={w.device}")
                            sample_val = w.flatten()[:5].detach().cpu().float().tolist()
                            print(f"  [Inference Debug] Active weight sample: {sample_val}")
                            break
                except Exception as e:
                    print(f"  [Inference Debug] Could not inspect internal weight: {e}")

            out = self.model(x.to(dtype), sigma.to(dtype), self.embeds.to(dtype), None, attention_mask=self.mask)
            if isinstance(out, tuple): out = out[0]
            return out.to(x.dtype)

    generator = torch.Generator(device).manual_seed(seed)
    x = torch.randn(1, 16, 128, 128, device=device, dtype=torch.float16, generator=generator)
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device)
    
    wrapper = ZITWrapper(model, prompt_embeds, prompt_mask)
    
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    with torch.no_grad():
        result = k_sampling.sample_euler(wrapper, x, sigmas, disable=False)  # CAPTURE THE RESULT!
    end_time = time.time()
    
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    
    return result, end_time - start_time, peak_vram  # RETURN THE DENOISED RESULT!

def calculate_metrics(l1, l2):
    # Convert to numpy float32 for stable calculation
    arr1 = l1[0].cpu().float().numpy()
    arr2 = l2[0].cpu().float().numpy()
    
    if np.array_equal(arr1, arr2):
        print("  CRITICAL WARNING: Raw latents are bit-perfect identical.")
    
    mse = np.mean((arr1 - arr2) ** 2)
    
    combined_min = min(arr1.min(), arr2.min())
    combined_max = max(arr1.max(), arr2.max())
    data_range = combined_max - combined_min
    
    arr1_hwc = arr1.transpose(1, 2, 0)
    arr2_hwc = arr2.transpose(1, 2, 0)
    
    score_ssim = ssim(arr1_hwc[:, :, :3], arr2_hwc[:, :, :3], win_size=3, channel_axis=2, data_range=data_range)
    return mse, score_ssim

def print_model_stats(model, name):
    state = model.state_dict()
    target_key = None
    for key in state.keys():
        if "layers.10" in key and "weight" in key and "norm" not in key:
            target_key = key
            break
    if target_key is None:
        target_key = next(iter(state))
        print(f"  Note: Fallback key: {target_key}")

    weight = state[target_key]
    print(f"[{name}] Inspecting weight: {target_key}")
    print(f"[{name}] Shape={tuple(weight.shape)}, dtype={weight.dtype}")
    
    flat = weight.flatten()[:5].cpu().float().tolist()
    print(f"[{name}] First 5 values: {flat}")
    
    weight_hash = hash(weight.flatten()[:100].cpu().float().sum().item())
    print(f"[{name}] Weight hash: {weight_hash}")
    
    q_params = [k for k in state if ".comfy_quant" in k]
    if q_params:
        print(f"[{name}] Detected Quantization Metadata: {len(q_params)} parameters.")

def main():
    parser = argparse.ArgumentParser(description="ZIT FP8 Fidelity & VRAM Benchmark")
    parser.add_argument("--fp16", required=True, help="Baseline model path")
    parser.add_argument("--fp8", required=True, help="Quantized model path")
    parser.add_argument("--clip_path", required=True, help="Qwen3-4B text encoder path")
    parser.add_argument("--tokenizer_path", default=None, help="Tokenizer path or Repo ID")
    parser.add_argument("--token", default=None, help="Hugging Face Token")
    parser.add_argument("--comfy_path", required=True, help="ComfyUI root path")
    parser.add_argument("--prompt", default="A beautiful cyberpunk city at night, high detail.", help="Benchmark prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    # Safety Check for Tokenizer
    if args.tokenizer_path and args.tokenizer_path.startswith("hf_"):
        print("  Warning: Detected HF token in tokenizer_path argument. Ignoring invalid path.")
        args.tokenizer_path = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Trust the user's provided path blindly
    if args.comfy_path not in sys.path:
        sys.path.insert(0, args.comfy_path)
    print(f"  ComfyUI Path set to: {args.comfy_path}")
    
    try:
        from comfy.text_encoders import llama as llama_module
        from transformers import AutoTokenizer
        import comfy.ops
    except ImportError as e:
        print(f"CRITICAL ERROR: Could not import 'comfy'.")
        print(f"  Current Working Directory: {os.getcwd()}")
        print(f"  Target ComfyUI Path: {args.comfy_path}")
        if os.path.exists(args.comfy_path):
             print(f"  Directory Listing: {os.listdir(args.comfy_path)[:10]}...")
        else:
             print(f"  Path does NOT exist.")
        print(f"  sys.path: {sys.path}")
        sys.exit(1)
    
    print("Starting Text Encoder Initialization...")
    
    resolved_clip = resolve_path(args.clip_path, is_file=True)
    tokenizer_path, is_local = resolve_tokenizer_path(args.tokenizer_path, resolved_clip)
    
    if is_local:
        print(f"  Note: Confirmed Qwen 3 4B tokenizer in: {tokenizer_path}")
    else:
        print(f"  Warning: Local tokenizer files not found. Using path: {tokenizer_path} (is_local=False)")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, 
            local_files_only=is_local, 
            trust_remote_code=True,
            token=args.token
        )
    except Exception as e:
        print(f"Error loading tokenizer (is_local={is_local}): {e}")
        if is_local:
             print("Retrying with remote load (using token if provided)...")
             tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, token=args.token)
        else:
             sys.exit(1)

    text_encoder = llama_module.Qwen3_4B(
        config_dict={}, device=device, dtype=torch.float16, operations=comfy.ops.disable_weight_init
    ).to(device)
    
    print(f"Loading CLIP weights from: {resolved_clip}")
    text_encoder.load_state_dict(load_file(resolved_clip), strict=False)
    text_encoder.eval()
    
    embeds, mask = encode_prompt(args.prompt, text_encoder, tokenizer, device)
    
    # FP16 Benchmark
    print("\n=== 1. Benchmarking Baseline (FP16) ===")
    model = load_zit_model(args.fp16, device, args.comfy_path, is_fp8=False)
    print_model_stats(model, "FP16 Baseline")
    latents_fp16, time_fp16, vram_fp16 = run_inference(model, embeds, mask, args.steps, args.seed, device)
    print(f"FP16 Time: {time_fp16:.2f}s | Peak VRAM: {vram_fp16:.2f} MB")
    
    img_fp16 = latent_to_img(latents_fp16)
    img_fp16.save("bench_fp16.png")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # FP8 Benchmark
    print("\n=== 2. Benchmarking Quantized (FP8) ===")
    model = load_zit_model(args.fp8, device, args.comfy_path, is_fp8=True)
    print_model_stats(model, "FP8 Quantized")
    latents_fp8, time_fp8, vram_fp8 = run_inference(model, embeds, mask, args.steps, args.seed, device)
    print(f"FP8 Time: {time_fp8:.2f}s | Peak VRAM: {vram_fp8:.2f} MB")
    
    img_fp8 = latent_to_img(latents_fp8)
    img_fp8.save("bench_fp8.png")
    
    # Comparison
    mse, score = calculate_metrics(latents_fp16, latents_fp8)
    
    print("\n" + "="*50)
    print("ZIT FP8 BENCHMARK RESULTS")
    print("="*50)
    vram_saved = vram_fp16 - vram_fp8
    vram_saved_pct = (vram_saved / vram_fp16) * 100
    
    print(f"Peak VRAM Expansion:  FP16: {vram_fp16:>8.1f} MB")
    print(f"                      FP8:  {vram_fp8:>8.1f} MB")
    print(f"VRAM Saved:           {vram_saved:8.1f} MB ({vram_saved_pct:.1f}%)")
    print("-" * 50)
    print(f"Inference Time:       FP16: {time_fp16:>8.2f}s")
    print(f"                      FP8:  {time_fp8:>8.2f}s")
    print("-" * 50)
    print(f"Fidelity (Latent Space):")
    print(f"  MSE (Error):        {mse:.4f}")
    print(f"  SSIM (Similarity):  {score:.4f}")
    print("="*50)

    diff_img = ImageChops.difference(img_fp16, img_fp8)
    diff_img = ImageChops.multiply(diff_img, Image.new('RGB', diff_img.size, (10, 10, 10))) 
    diff_img.save("bench_diff.png")
    print("Diff image saved: bench_diff.png")

if __name__ == "__main__":
    main()

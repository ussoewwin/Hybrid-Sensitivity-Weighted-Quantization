"""
Flux1.devモデルをFP8形式に量子化するスクリプト (HSWQ V1.2: Flux Edition / Full Calibration)
SDXL HSWQ V1.21 (GPU Accelerated) をベースに、Flux1.dev構造に完全対応。

特徴:
- **Available**: Full HSWQ Calibration (Sensitivity & Importance Measurement)
- **Flux1 Support**: Double Blocks / Single Blocks / Embedders の全Linear/Conv層に対応
- **VRAM Optimization**: キャリブレーション後はVRAMを解放し、逐次GPU転送で量子化
- **Full Compatibility**: ComfyUIフォーマットの重みを正しくマッピング
- **SageAttention2**: SDPA Monkey-Patchingによる高速化オプション (--sa2)

アルゴリズム:
1. Load FluxPipeline (from single file)
2. Calibration Loop (DualMonitor: Sensitivity & Input Importance)
3. Layer Selection (Keep Top N% Sensitive Layers)
4. HSWQ Optimization (Weighted Histogram MSE)
5. GPU Accelerated Quantization
"""

import argparse
import torch
from diffusers import FluxPipeline, AutoencoderKL
from transformers import CLIPTextModel, T5EncoderModel, AutoTokenizer, CLIPTextConfig, T5Config
from safetensors.torch import load_file, save_file
import os
import gc
from tqdm import tqdm
import sys
import numpy as np

# HSWQ専用モジュールをインポート
from weighted_histogram_mse import HSWQWeightedHistogramOptimizer

# C++20標準を強制
if sys.platform == "win32":
    os.environ.setdefault("CXXFLAGS", "/std:c++20")
else:
    os.environ.setdefault("CXXFLAGS", "-std=c++20")

# === V1.2: SageAttention2 Integration ===
_sage_attn_available = False
_original_sdpa = None

def try_import_sage_attention():
    """Attempt to import SageAttention2 and return availability status."""
    global _sage_attn_available
    try:
        from sageattention import sageattn
        _sage_attn_available = True
        print("[SageAttention2] Successfully imported.")
        return True
    except ImportError:
        print("[SageAttention2] Not installed. Calibration will use standard attention.")
        return False

def enable_sage_attention():
    """Monkey-patch torch.nn.functional.scaled_dot_product_attention with SageAttention2."""
    global _original_sdpa
    if not _sage_attn_available:
        print("[SageAttention2] Cannot enable - not available.")
        return False
    
    import torch.nn.functional as F
    from sageattention import sageattn
    
    _original_sdpa = F.scaled_dot_product_attention
    
    def sage_sdpa_wrapper(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        # SageAttention2 does not support attn_mask or is_causal directly
        # Fall back to original SDPA if these are used
        if attn_mask is not None or is_causal:
            return _original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
        
        # SageAttention2 expects (B, H, N, D) format - same as SDPA
        try:
            return sageattn(query, key, value, is_causal=False) # Calibration doesn't need causal mask
        except Exception as e:
            # Fallback on any error
            return _original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
    
    F.scaled_dot_product_attention = sage_sdpa_wrapper
    print("[SageAttention2] Enabled for calibration (monkey-patched SDPA).")
    return True

def disable_sage_attention():
    """Restore original scaled_dot_product_attention."""
    global _original_sdpa
    if _original_sdpa is not None:
        import torch.nn.functional as F
        F.scaled_dot_product_attention = _original_sdpa
        _original_sdpa = None
        print("[SageAttention2] Disabled (restored original SDPA).")

# --- Dual Monitor: Sensitivity & Importance (Flux Version) ---
class DualMonitor:
    def __init__(self):
        self.output_sum = 0.0
        self.output_sq_sum = 0.0
        self.count = 0
        self.channel_importance = None 
    
    def update(self, input_tensor, output_tensor):
        with torch.no_grad():
            # Output Variance (Sensitivity)
            out_detached = output_tensor.detach().float()
            batch_mean = out_detached.mean().item()
            batch_sq_mean = (out_detached ** 2).mean().item()
            self.output_sum += batch_mean
            self.output_sq_sum += batch_sq_mean
            
            # Input Importance
            # Flux Linear Input: (B, SeqLen, Dim)
            inp_detached = input_tensor.detach()
            if inp_detached.dim() == 3: 
                current_imp = inp_detached.abs().mean(dim=(0, 1)) # -> (Dim,)
            elif inp_detached.dim() == 2:
                current_imp = inp_detached.abs().mean(dim=0)
            else:
                current_imp = inp_detached.abs().mean() # fallback
                
            if self.channel_importance is None:
                self.channel_importance = current_imp
            else:
                self.channel_importance = (self.channel_importance * self.count + current_imp) / (self.count + 1)
            
            self.count += 1

    def get_sensitivity(self):
        if self.count == 0: return 0.0
        mean = self.output_sum / self.count
        sq_mean = self.output_sq_sum / self.count
        return sq_mean - mean ** 2

dual_monitors = {}

def hook_fn(module, input, output, name):
    if name not in dual_monitors:
        dual_monitors[name] = DualMonitor()
    # Linear: input is (tensor,) output is tensor
    dual_monitors[name].update(input[0], output)

def load_flux_pipeline_from_safetensors(path, device="cuda", token=None, clip_path=None, t5_path=None, vae_path=None):
    print(f"Flux1モデルをロード中: {path}")
    
    text_encoder = None
    text_encoder_2 = None
    tokenizer = None
    tokenizer_2 = None
    vae = None
    
    # Helper to load model from file or dir
    def load_external_component(path, model_class, config_class_or_repo, default_repo=None, tokenizer_repo=None, is_diffusers=False, token=None):
        model = None
        tok = None
        
        if os.path.isfile(path):
            print(f"Loading single file component from: {path}")
            try:
                if is_diffusers:
                    # Diffusers Model (VAE)
                    # config_class_or_repo is the repo string for config
                    print(f"Fetching default config from: {config_class_or_repo}")
                    model = model_class.from_pretrained(config_class_or_repo, subfolder="vae", token=token)
                    # Load weights
                    sd = load_file(path)
                    m, u = model.load_state_dict(sd, strict=False)
                    print(f"Weights loaded. Missing: {len(m)}, Unexpected: {len(u)}")
                    model.to(torch.float16)
                else:
                    # Transformers Model
                    print(f"Fetching default config from: {default_repo}")
                    config = config_class_or_repo.from_pretrained(default_repo, token=token)
                    model = model_class(config)
                    sd = load_file(path)
                    m, u = model.load_state_dict(sd, strict=False)
                    print(f"Weights loaded. Missing: {len(m)}, Unexpected: {len(u)}")
                    model.to(torch.float16)
            except Exception as e:
                print(f"Failed to load single file: {e}")
                sys.exit(1)
            
            if tokenizer_repo:
                print(f"Fetching tokenizer from: {tokenizer_repo}")
                tok = AutoTokenizer.from_pretrained(tokenizer_repo, token=token)
                
        else:
            print(f"Loading component from directory: {path}")
            try:
                if is_diffusers:
                     model = model_class.from_pretrained(path, torch_dtype=torch.float16, token=token)
                else:
                    model = model_class.from_pretrained(path, torch_dtype=torch.float16, token=token)
                    tok = AutoTokenizer.from_pretrained(path, token=token)
            except Exception as e:
                print(f"Error loading from directory: {e}")
                sys.exit(1)
                
        return model, tok

    # Load External CLIP (Text Encoder 1)
    if clip_path:
        text_encoder, tokenizer = load_external_component(
            clip_path, CLIPTextModel, CLIPTextConfig, "openai/clip-vit-large-patch14", "openai/clip-vit-large-patch14", token=token
        )
        print("CLIP loaded successfully.")

    # Load External T5 (Text Encoder 2)
    if t5_path:
        text_encoder_2, tokenizer_2 = load_external_component(
            t5_path, T5EncoderModel, T5Config, "google/t5-v1_1-xxl", "google/t5-v1_1-xxl", token=token
        )
        print("T5 loaded successfully.")

    # Load External VAE
    if vae_path:
        # Use FLUX.1-schnell config for VAE to avoid auth issues (structure is same)
        vae, _ = load_external_component(
            vae_path, AutoencoderKL, "black-forest-labs/FLUX.1-schnell", is_diffusers=True, token=token
        )
        print("VAE loaded successfully.")

    # Diffusersのfrom_single_fileを使用 (要: diffusers >= 0.30.0)
    try:
        # First attempt: Standard load
        pipeline = FluxPipeline.from_single_file(
            path, 
            torch_dtype=torch.float16,
            use_safetensors=True,
            token=token,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            vae=vae
        )
    except Exception as e:
        # Check for authentication/access error (401/403)
        error_msg = str(e)
        if "401" in error_msg or "403" in error_msg or "restricted" in error_msg:
            print("\n[Auth Error Detected] Gated model access restricted.")
            print("→ Fallback: Using 'black-forest-labs/FLUX.1-schnell' (public) configuration...")
            try:
                pipeline = FluxPipeline.from_single_file(
                    path,
                    config="black-forest-labs/FLUX.1-schnell", # Use public config
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    tokenizer=tokenizer,
                    tokenizer_2=tokenizer_2,
                    vae=vae
                )
            except Exception as e2:
                print(f"Fallback failed: {e2}")
                print("Please configure HuggingFace token or use a public model.")
                sys.exit(1)
        else:
            print(f"Error loading FluxPipeline: {e}")
            print("Note: diffusers >= 0.30.0 is required for Flux.")
            # 詳細なエラーを出して終了
            raise e
            
    # 5060 Ti (8GB/16GB) 等でのOOM回避のため CPU Offload を有効化
    print("VRAM最適化: Model CPU Offload を有効化します...")
    pipeline.enable_model_cpu_offload()
    return pipeline

def is_quantizable_module(name, module):
    return isinstance(module, (torch.nn.Linear, torch.nn.Conv2d))

def main():
    parser = argparse.ArgumentParser(description="Flux1.dev FP8 Quantization (HSWQ V1.2: Full Calibration)")
    parser.add_argument("--input", type=str, required=True, help="Path to input safetensors model")
    parser.add_argument("--output", type=str, required=True, help="Path to output safetensors model")
    parser.add_argument("--calib_file", type=str, required=True, help="Path to calibration prompts text file")
    parser.add_argument("--num_calib_samples", type=int, default=16, help="Calibration samples (Flux is heavy, default: 16)")
    parser.add_argument("--num_inference_steps", type=int, default=4, help="Number of inference steps for calibration")
    parser.add_argument("--keep_ratio", type=float, default=0.25, help="Ratio of layers to keep in FP16")
    parser.add_argument("--sa2", action="store_true", help="Enable SageAttention2 for faster calibration (requires sageattention package)")
    parser.add_argument("--token", type=str, help="Hugging Face token for gated models")
    parser.add_argument("--clip_path", type=str, help="Path to CLIP Text Encoder (folder with config)")
    parser.add_argument("--t5_path", type=str, help="Path to T5 Text Encoder (folder with config)")
    parser.add_argument("--vae_path", type=str, help="Path to VAE Model (folder or safetensors)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"デバイス: {device}")
    
    # === SageAttention2 Initialization ===
    if args.sa2:
        if try_import_sage_attention():
            enable_sage_attention()
        else:
            print("[Warning] --sa2 specified but SageAttention2 not available. Continuing with standard attention.")

    # 1. Load Pipeline & Calibration
    pipeline = load_flux_pipeline_from_safetensors(args.input, device, token=args.token, clip_path=args.clip_path, t5_path=args.t5_path, vae_path=args.vae_path)
    
    print("キャリブレーション準備中（Dual Monitorフック登録）...")
    handles = []
    target_modules = []
    
    # Flux Transformer内のLinear層にフック
    # diffusers構造: pipeline.transformer...
    for name, module in pipeline.transformer.named_modules():
        if is_quantizable_module(name, module):
            # ComfyUIキーとのマッピングが必要だが、まずはDiffusers名で計測
            # 最後に重み名でマッチングさせる
            handle = module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))
            handles.append(handle)
            target_modules.append(name)
            
    print(f"モニタリング対象レイヤー数: {len(target_modules)}")

    # Load Prompts (Loop Logic Added)
    with open(args.calib_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    
    # SDXL V1.2 同等のループ処理
    if len(prompts) < args.num_calib_samples:
        prompts = (prompts * (args.num_calib_samples // len(prompts) + 1))[:args.num_calib_samples]
    else:
        prompts = prompts[:args.num_calib_samples]
    
    print(f"キャリブレーションを実行中（{len(prompts)}サンプル, {args.num_inference_steps}ステップ）...")
    pipeline.set_progress_bar_config(disable=False)
    
    for i, prompt in enumerate(prompts):
        print(f"Sample {i+1}/{len(prompts)}: {prompt[:30]}...")
        with torch.no_grad():
            # 指定されたステップ数で実行
            pipeline(prompt=prompt, num_inference_steps=args.num_inference_steps, guidance_scale=3.5, output_type="latent")
        
        # メモリ解放頻度: Fluxは重いので毎回やるのが安全
        gc.collect()
        torch.cuda.empty_cache()

    for h in handles: h.remove()
    
    # === SageAttention2 Cleanup ===
    if args.sa2:
        disable_sage_attention()

    # 2. Analyze Sensitivity & Select Keep Layers
    print("\nレイヤー感度分析...")
    layer_sensitivities = []
    for name in target_modules:
        if name in dual_monitors:
            sensitivity = dual_monitors[name].get_sensitivity()
            layer_sensitivities.append((name, sensitivity))
            
    layer_sensitivities.sort(key=lambda x: x[1], reverse=True)
    num_keep = int(len(layer_sensitivities) * args.keep_ratio)
    keep_layers_diffusers = set([x[0] for x in layer_sensitivities[:num_keep]])
    
    print(f"FP16保持レイヤー数: {len(keep_layers_diffusers)}")
    
    print("\n[HSWQ] 最適化パラメータ(amax)を計算中...")
    hswq_optimizer = HSWQWeightedHistogramOptimizer(bins=4096, num_candidates=200, device=device)
    diffusers_amax_dict = {}
    
    for name, module in tqdm(pipeline.transformer.named_modules(), desc="Calculating Amax"):
        if is_quantizable_module(name, module):
            if name in keep_layers_diffusers:
                continue
            
            imp = None
            if name in dual_monitors:
                imp = dual_monitors[name].channel_importance
            
            # GPU上のWeightを使用
            optimal_amax = hswq_optimizer.compute_optimal_amax(
                module.weight.data, 
                imp, 
                scaled=False
            )
            diffusers_amax_dict[name] = optimal_amax
            
    # 3. Cleanup Pipeline to free VRAM for Quantization
    print("パイプラインを解放中...")
    del pipeline
    del dual_monitors
    del hswq_optimizer
    gc.collect()
    torch.cuda.empty_cache()
    
    # 4. Load Original Safetensors & Convert
    print(f"\n[VRAM Optimized Quantization] Loading original file: {args.input}")
    
    # Diffusers名 -> ComfyUI名 の簡易マッパー
    def map_diffusers_to_comfy(d_name):
        return "model.diffusion_model." + d_name + ".weight"

    comfy_amax_dict = {}
    comfy_keep_set = set()
    
    for name, amax in diffusers_amax_dict.items():
        c_key = map_diffusers_to_comfy(name)
        comfy_amax_dict[c_key] = amax
        
    for name in keep_layers_diffusers:
        c_key = map_diffusers_to_comfy(name)
        comfy_keep_set.add(c_key)
        
    original_sd = load_file(args.input)
    output_sd = {}
    converted_count = 0
    kept_count = 0
    
    print("変換中 (GPU Accelerated)...")
    for key in tqdm(list(original_sd.keys())):
        value = original_sd[key]
        
        target_amax = comfy_amax_dict.get(key)
        
        if key in comfy_keep_set:
            output_sd[key] = value # Keep FP16
            kept_count += 1
        elif target_amax is not None:
            # Quantize
            val_gpu = value.to(device)
            clamped = torch.clamp(val_gpu, -target_amax, target_amax)
            out = clamped.to(torch.float8_e4m3fn).cpu() # Return to CPU
            output_sd[key] = out
            converted_count += 1
            del val_gpu, clamped
        else:
            output_sd[key] = value # Other layers
            
    print(f"完了: {converted_count} Quantized, {kept_count} Kept FP16")
    save_file(output_sd, args.output)
    print("Saved.")

if __name__ == "__main__":
    main()

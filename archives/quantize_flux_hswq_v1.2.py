"""
Quantize Flux1.dev model to FP8 (HSWQ V1.2: Flux Edition).
Same structure/algorithm as SDXL HSWQ V1.2 (GPU Accelerated).

Algorithm (same as SDXL V1.2):
1. Load Pipeline + build comfyui_to_diffusers_map
2. Calibration Loop (DualMonitor: sensitivity & input importance)
3. Layer Selection (keep top N% by sensitivity)
4. HSWQ Optimization (weighted histogram MSE, scaled=False)
5. GPU Accelerated Quantization (convert via comfyui_to_diffusers_map reverse lookup)
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

# HSWQ modules
from weighted_histogram_mse import HSWQWeightedHistogramOptimizer

# Enforce C++20
if sys.platform == "win32":
    os.environ.setdefault("CXXFLAGS", "/std:c++20")
else:
    os.environ.setdefault("CXXFLAGS", "-std=c++20")

# === SageAttention2 Integration (same as SDXL V1.2) ===
_sage_attn_available = False
_original_sdpa = None

def try_import_sage_attention():
    global _sage_attn_available
    try:
        from sageattention import sageattn
        _sage_attn_available = True
        print("[SageAttention2] Successfully imported.")
        return True
    except ImportError:
        print("[SageAttention2] Not installed. Using standard attention.")
        return False

def enable_sage_attention():
    global _original_sdpa
    if not _sage_attn_available:
        print("[SageAttention2] Cannot enable - not available")
        return False
    
    import torch.nn.functional as F
    from sageattention import sageattn
    
    _original_sdpa = F.scaled_dot_product_attention
    
    def sage_sdpa_wrapper(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        if attn_mask is not None or is_causal:
            return _original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
        try:
            return sageattn(query, key, value, is_causal=False)
        except Exception as e:
            return _original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
    
    F.scaled_dot_product_attention = sage_sdpa_wrapper
    print("[SageAttention2] Enabled (SDPA monkey-patch)")
    return True

def disable_sage_attention():
    global _original_sdpa
    if _original_sdpa is not None:
        import torch.nn.functional as F
        F.scaled_dot_product_attention = _original_sdpa
        _original_sdpa = None
        print("[SageAttention2] Disabled (restored original SDPA)")


# --- ComfyUI-compatible mapping (same role as SDXL V1.2 unet_to_diffusers_mapping) ---

def flux_to_diffusers_mapping(state_dict, key_prefix=None):
    """Build Flux ComfyUI key -> Diffusers key mapping. Returns comfyui_to_diffusers_map."""
    state_dict_keys = list(state_dict.keys())
    
    # Auto-detect prefix
    if key_prefix is None:
        has_full_prefix = any(k.startswith("model.diffusion_model.") for k in state_dict_keys)
        has_no_prefix = any(k.startswith("double_blocks.") or k.startswith("single_blocks.") for k in state_dict_keys)
        
        if has_full_prefix:
            key_prefix = "model.diffusion_model."
        elif has_no_prefix:
            key_prefix = ""
        else:
            key_prefix = "model.diffusion_model."
        
        print(f"  Auto-detected prefix: '{key_prefix}' (full={has_full_prefix}, no_prefix={has_no_prefix})")
    
    # Detect block counts
    num_double = 0
    num_single = 0
    for k in state_dict_keys:
        if k.startswith(key_prefix):
            stripped = k[len(key_prefix):]
            if stripped.startswith("double_blocks."):
                parts = stripped.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    num_double = max(num_double, int(parts[1]) + 1)
            elif stripped.startswith("single_blocks."):
                parts = stripped.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    num_single = max(num_single, int(parts[1]) + 1)
    
    print(f"  Detected: Double Blocks={num_double}, Single Blocks={num_single}")
    
    comfyui_to_diffusers_map = {}
    
    # --- Embeddings ---
    embed_map = {
        "txt_in": "context_embedder",
        "img_in": "x_embedder",
        "time_in.in_layer": "time_text_embed.timestep_embedder.linear_1",
        "time_in.out_layer": "time_text_embed.timestep_embedder.linear_2",
        "guidance_in.in_layer": "time_text_embed.guidance_embedder.linear_1",
        "guidance_in.out_layer": "time_text_embed.guidance_embedder.linear_2",
        "vector_in.in_layer": "time_text_embed.text_embedder.linear_1",
        "vector_in.out_layer": "time_text_embed.text_embedder.linear_2",
        "final_layer.adaLN_modulation.1": "norm_out.linear",
        "final_layer.linear": "proj_out",
    }
    for suffix in [".weight", ".bias"]:
        for comfy_name, diff_name in embed_map.items():
            comfy_key = f"{key_prefix}{comfy_name}{suffix}"
            if comfy_key in state_dict_keys:
                comfyui_to_diffusers_map[comfy_key] = f"{diff_name}{suffix}"
    
    # --- Double Blocks ---
    for i in range(num_double):
        # 1:1 mapping (non-fused layers)
        one_to_one = {
            f"double_blocks.{i}.img_mod.lin": f"transformer_blocks.{i}.norm1.linear",
            f"double_blocks.{i}.txt_mod.lin": f"transformer_blocks.{i}.norm1_context.linear",
            f"double_blocks.{i}.img_attn.proj": f"transformer_blocks.{i}.attn.to_out.0",
            f"double_blocks.{i}.txt_attn.proj": f"transformer_blocks.{i}.attn.to_add_out",
            f"double_blocks.{i}.img_mlp.0": f"transformer_blocks.{i}.ff.net.0.proj",
            f"double_blocks.{i}.img_mlp.2": f"transformer_blocks.{i}.ff.net.2",
            f"double_blocks.{i}.txt_mlp.0": f"transformer_blocks.{i}.ff_context.net.0.proj",
            f"double_blocks.{i}.txt_mlp.2": f"transformer_blocks.{i}.ff_context.net.2",
        }
        for suffix in [".weight", ".bias"]:
            for comfy_name, diff_name in one_to_one.items():
                comfy_key = f"{key_prefix}{comfy_name}{suffix}"
                if comfy_key in state_dict_keys:
                    comfyui_to_diffusers_map[comfy_key] = f"{diff_name}{suffix}"
        
        # Fused QKV: ComfyUI has one tensor, Diffusers splits it -> mark as "FUSED:xxx" for direct handling at quantize
        for suffix in [".weight", ".bias"]:
            # img_attn.qkv -> to_q + to_k + to_v (fused)
            comfy_key = f"{key_prefix}double_blocks.{i}.img_attn.qkv{suffix}"
            if comfy_key in state_dict_keys:
                comfyui_to_diffusers_map[comfy_key] = f"FUSED:transformer_blocks.{i}.attn.img_qkv{suffix}"
            # txt_attn.qkv -> add_q_proj + add_k_proj + add_v_proj (fused)
            comfy_key = f"{key_prefix}double_blocks.{i}.txt_attn.qkv{suffix}"
            if comfy_key in state_dict_keys:
                comfyui_to_diffusers_map[comfy_key] = f"FUSED:transformer_blocks.{i}.attn.txt_qkv{suffix}"
        
        # Norm scales (RMSNorm)
        for suffix in [".scale"]:
            norm_map = {
                f"double_blocks.{i}.img_attn.norm.query_norm": f"transformer_blocks.{i}.attn.norm_q",
                f"double_blocks.{i}.img_attn.norm.key_norm": f"transformer_blocks.{i}.attn.norm_k",
                f"double_blocks.{i}.txt_attn.norm.query_norm": f"transformer_blocks.{i}.attn.norm_added_q",
                f"double_blocks.{i}.txt_attn.norm.key_norm": f"transformer_blocks.{i}.attn.norm_added_k",
            }
            for comfy_name, diff_name in norm_map.items():
                comfy_key = f"{key_prefix}{comfy_name}{suffix}"
                if comfy_key in state_dict_keys:
                    comfyui_to_diffusers_map[comfy_key] = f"{diff_name}.weight"
    
    # --- Single Blocks ---
    for i in range(num_single):
        # 1:1 mapping
        one_to_one = {
            f"single_blocks.{i}.modulation.lin": f"single_transformer_blocks.{i}.norm.linear",
            f"single_blocks.{i}.linear2": f"single_transformer_blocks.{i}.proj_out",
        }
        for suffix in [".weight", ".bias"]:
            for comfy_name, diff_name in one_to_one.items():
                comfy_key = f"{key_prefix}{comfy_name}{suffix}"
                if comfy_key in state_dict_keys:
                    comfyui_to_diffusers_map[comfy_key] = f"{diff_name}{suffix}"
        
        # Fused Linear1: to_q + to_k + to_v + proj_mlp (4 fused)
        for suffix in [".weight", ".bias"]:
            comfy_key = f"{key_prefix}single_blocks.{i}.linear1{suffix}"
            if comfy_key in state_dict_keys:
                comfyui_to_diffusers_map[comfy_key] = f"FUSED:single_transformer_blocks.{i}.linear1{suffix}"
        
        # Norm scales
        for suffix in [".scale"]:
            norm_map = {
                f"single_blocks.{i}.norm.query_norm": f"single_transformer_blocks.{i}.attn.norm_q",
                f"single_blocks.{i}.norm.key_norm": f"single_transformer_blocks.{i}.attn.norm_k",
            }
            for comfy_name, diff_name in norm_map.items():
                comfy_key = f"{key_prefix}{comfy_name}{suffix}"
                if comfy_key in state_dict_keys:
                    comfyui_to_diffusers_map[comfy_key] = f"{diff_name}.weight"
    
    return comfyui_to_diffusers_map


def load_flux_pipeline_from_safetensors(path, device="cuda", token=None, clip_path=None, t5_path=None, vae_path=None):
    """Load Flux pipeline (same role as SDXL V1.2 load_unet_from_safetensors). Returns: pipeline, original_state_dict, comfyui_to_diffusers_map"""
    print(f"Loading Flux1 model: {path}")
    
    # Load original state_dict (keep as in SDXL V1.2)
    original_state_dict = load_file(path)
    
    # Build key mapping
    print("Building key mapping...")
    comfyui_to_diffusers_map = flux_to_diffusers_mapping(original_state_dict)
    print(f"  Mapping count: {len(comfyui_to_diffusers_map)}")
    
    # Load external components
    text_encoder = None
    text_encoder_2 = None
    tokenizer = None
    tokenizer_2 = None
    vae = None
    
    # Helper: load model from file or directory
    def load_external_component(path, model_class, config_class_or_repo, default_repo=None, tokenizer_repo=None, is_diffusers=False, token=None):
        model = None
        tok = None
        
        if os.path.isfile(path):
            print(f"Loading from single file: {path}")
            try:
                if is_diffusers:
                    print(f"Loading default config: {config_class_or_repo}")
                    model = model_class.from_pretrained(config_class_or_repo, subfolder="vae", token=token)
                    sd = load_file(path)
                    m, u = model.load_state_dict(sd, strict=False)
                    print(f"Weights loaded. Missing: {len(m)}, Unexpected: {len(u)}")
                    model.to(torch.float16)
                else:
                    print(f"Loading default config: {default_repo}")
                    config = config_class_or_repo.from_pretrained(default_repo, token=token)
                    model = model_class(config)
                    sd = load_file(path)
                    m, u = model.load_state_dict(sd, strict=False)
                    print(f"Weights loaded. Missing: {len(m)}, Unexpected: {len(u)}")
                    model.to(torch.float16)
            except Exception as e:
                print(f"Single-file load failed: {e}")
                sys.exit(1)
            
            if tokenizer_repo:
                print(f"Loading tokenizer: {tokenizer_repo}")
                tok = AutoTokenizer.from_pretrained(tokenizer_repo, token=token)
                
        else:
            print(f"Loading from directory: {path}")
            try:
                if is_diffusers:
                     model = model_class.from_pretrained(path, torch_dtype=torch.float16, token=token)
                else:
                    model = model_class.from_pretrained(path, torch_dtype=torch.float16, token=token)
                    tok = AutoTokenizer.from_pretrained(path, token=token)
            except Exception as e:
                print(f"Directory load error: {e}")
                sys.exit(1)
                
        return model, tok

    # Load external CLIP
    if clip_path:
        text_encoder, tokenizer = load_external_component(
            clip_path, CLIPTextModel, CLIPTextConfig, "openai/clip-vit-large-patch14", "openai/clip-vit-large-patch14", token=token
        )
        print("CLIP loaded.")

    # Load external T5
    if t5_path:
        text_encoder_2, tokenizer_2 = load_external_component(
            t5_path, T5EncoderModel, T5Config, "google/t5-v1_1-xxl", "google/t5-v1_1-xxl", token=token
        )
        print("T5 loaded.")

    # Load external VAE
    if vae_path:
        vae, _ = load_external_component(
            vae_path, AutoencoderKL, "black-forest-labs/FLUX.1-schnell", is_diffusers=True, token=token
        )
        print("VAE loaded.")

    # Load FluxPipeline
    try:
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
        error_msg = str(e)
        if "401" in error_msg or "403" in error_msg or "restricted" in error_msg:
            print("\n[Auth error] Access to gated model restricted.")
            print("Fallback: using 'black-forest-labs/FLUX.1-schnell' (public) config...")
            try:
                pipeline = FluxPipeline.from_single_file(
                    path,
                    config="black-forest-labs/FLUX.1-schnell",
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
                sys.exit(1)
        else:
            print(f"FluxPipeline load error: {e}")
            raise e
            
    print("Enabling Model CPU Offload for VRAM...")
    pipeline.enable_model_cpu_offload()
    
    # Same as SDXL V1.2: return pipeline, original_state_dict, comfyui_to_diffusers_map
    return pipeline, original_state_dict, comfyui_to_diffusers_map


# --- Dual Monitor: Sensitivity & Importance (same as SDXL V1.2) ---
class DualMonitor:
    def __init__(self):
        # Sensitivity (output variance)
        self.output_sum = 0.0
        self.output_sq_sum = 0.0
        self.count = 0
        # Importance (input activation)
        self.channel_importance = None
    
    def update(self, input_tensor, output_tensor):
        with torch.no_grad():
            # 1. Sensitivity update (output variance)
            out_detached = output_tensor.detach().float()
            batch_mean = out_detached.mean().item()
            batch_sq_mean = (out_detached ** 2).mean().item()
            self.output_sum += batch_mean
            self.output_sq_sum += batch_sq_mean
            
            # 2. Importance update (input activation)
            inp_detached = input_tensor.detach()
            if inp_detached.dim() == 4:    # Conv2d: (B, C, H, W)
                current_imp = inp_detached.abs().mean(dim=(0, 2, 3))
            elif inp_detached.dim() == 3:  # Transformer: (B, T, C)
                current_imp = inp_detached.abs().mean(dim=(0, 1))
            elif inp_detached.dim() == 2:  # Linear: (B, C)
                current_imp = inp_detached.abs().mean(dim=0)
            else:
                current_imp = torch.ones(1, device=inp_detached.device, dtype=inp_detached.dtype)
                
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
    inp = input[0]
    out = output
    dual_monitors[name].update(inp, out)


def main():
    parser = argparse.ArgumentParser(description="Flux1.dev FP8 quantization (HSWQ V1.2, same structure as SDXL V1.2)")
    parser.add_argument("--input", type=str, required=True, help="Path to input safetensors model")
    parser.add_argument("--output", type=str, required=True, help="Path to output safetensors model")
    parser.add_argument("--calib_file", type=str, required=True, help="Path to calibration prompts text file")
    parser.add_argument("--num_calib_samples", type=int, default=25, help="Number of calibration samples (HSWQ: 25)")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Inference steps (HSWQ: 25)")
    parser.add_argument("--keep_ratio", type=float, default=0.25, help="Ratio of layers to keep in FP16")
    parser.add_argument("--sa2", action="store_true", help="Use SageAttention2 for faster calibration")
    parser.add_argument("--token", type=str, help="Hugging Face token (gated models)")
    parser.add_argument("--clip_path", type=str, help="CLIP text encoder path")
    parser.add_argument("--t5_path", type=str, help="T5 text encoder path")
    parser.add_argument("--vae_path", type=str, help="VAE model path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # === SageAttention2 init (same as SDXL V1.2) ===
    if args.sa2:
        if try_import_sage_attention():
            enable_sage_attention()
        else:
            print("[Warning] --sa2 set but SageAttention2 not available. Using standard attention.")

    # === Same as SDXL V1.2: get pipeline, original_state_dict, comfyui_to_diffusers_map ===
    pipeline, original_state_dict, comfyui_to_diffusers_map = load_flux_pipeline_from_safetensors(
        args.input, device, token=args.token, clip_path=args.clip_path, t5_path=args.t5_path, vae_path=args.vae_path
    )

    # === Calibration (same structure as SDXL V1.2) ===
    print("Preparing calibration (registering Dual Monitor hooks)...")
    handles = []
    target_modules = []
    for name, module in pipeline.transformer.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handle = module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))
            handles.append(handle)
            target_modules.append(name)

    print("Preparing calibration data...")
    with open(args.calib_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    if len(prompts) < args.num_calib_samples:
        prompts = (prompts * (args.num_calib_samples // len(prompts) + 1))[:args.num_calib_samples]
    else:
        prompts = prompts[:args.num_calib_samples]

    print(f"Running calibration ({args.num_calib_samples} samples, {args.num_inference_steps} steps)...")
    print("Measuring Sensitivity and Importance...")
    
    pipeline.set_progress_bar_config(disable=False)
    
    for i, prompt in enumerate(prompts):
        print(f"\nSample {i+1}/{args.num_calib_samples}: {prompt[:50]}...")
        with torch.no_grad():
            pipeline(prompt=prompt, num_inference_steps=args.num_inference_steps, guidance_scale=3.5, output_type="latent")
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Remove hooks
    for h in handles: h.remove()
    
    # === SageAttention2 Cleanup ===
    if args.sa2:
        disable_sage_attention()

    # === Layer sensitivity analysis (same as SDXL V1.2) ===
    print("\nRunning layer sensitivity analysis...")
    layer_sensitivities = []
    for name in target_modules:
        if name in dual_monitors:
            sensitivity = dual_monitors[name].get_sensitivity()
            layer_sensitivities.append((name, sensitivity))
    
    # Sort by sensitivity (descending)
    layer_sensitivities.sort(key=lambda x: x[1], reverse=True)
    
    # Identify top N%
    num_keep = int(len(layer_sensitivities) * args.keep_ratio)
    keep_layers = set([x[0] for x in layer_sensitivities[:num_keep]])
    
    print(f"Total layers: {len(layer_sensitivities)}")
    print(f"FP16-kept layers: {len(keep_layers)} (Top {args.keep_ratio*100:.1f}%)")
    print("Top 5 Sensitive Layers:")
    for i in range(min(5, len(layer_sensitivities))):
        print(f"  {i+1}. {layer_sensitivities[i][0]}: {layer_sensitivities[i][1]:.4f}")

    # === HSWQ optimization (same as SDXL V1.2: named_modules -> amax) ===
    print("\n[HSWQ] Starting weighted MSE analysis and quantization parameter computation...")
    print("Compatibility mode (scaled=False): searching optimal clipping threshold...")
    weight_amax_dict = {}
    
    # Init HSWQ optimizer (same as SDXL V1.2: bins=4096, 200 candidates, 3 refinements)
    hswq_optimizer = HSWQWeightedHistogramOptimizer(
        bins=4096,
        num_candidates=200,
        refinement_iterations=3,
        device=device
    )
    
    for name, module in tqdm(pipeline.transformer.named_modules(), desc="Analyzing"):
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            # Skip FP16-kept layers (no amax needed)
            if name in keep_layers:
                continue
                
            # Get importance
            importance = None
            if name in dual_monitors:
                importance = dual_monitors[name].channel_importance
            
            # HSWQ: find optimal amax (scaled=False)
            optimal_amax = hswq_optimizer.compute_optimal_amax(
                module.weight.data, 
                importance,
                scaled=False  # Compatibility mode
            )
            weight_amax_dict[name + ".weight"] = optimal_amax
            
            torch.cuda.empty_cache()

    # === Flux: amax for fused QKV layers ===
    # ComfyUI: img_attn.qkv / txt_attn.qkv / linear1 are fused; Diffusers splits them (not in named_modules)
    # Compute amax from original_state_dict
    fused_count = 0
    for key, value in original_state_dict.items():
        if key in comfyui_to_diffusers_map:
            diffusers_key = comfyui_to_diffusers_map[key]
            if diffusers_key.startswith("FUSED:") and diffusers_key.endswith(".weight"):
                if value.dim() == 2 and value.numel() >= 1024:
                    optimal_amax = hswq_optimizer.compute_optimal_amax(
                        value, None, scaled=False
                    )
                    weight_amax_dict[diffusers_key] = optimal_amax
                    fused_count += 1
    
    print(f"Layers to quantize: {len(weight_amax_dict)} (fused QKV: {fused_count})")
    
    # === VRAM optimization (same as SDXL V1.2) ===
    print("\n[VRAM] Preparing GPU conversion...")
    del pipeline
    del hswq_optimizer
    gc.collect()
    torch.cuda.empty_cache()
    
    # === Quantization (same structure as SDXL V1.2) ===
    # Convert using reverse lookup of comfyui_to_diffusers_map
    print(f"Saving quantized model: {args.output}")
    output_state_dict = {}
    converted_count = 0
    kept_count = 0
    
    print("Converting weights (GPU)...")
    for key, value in tqdm(original_state_dict.items(), desc="Converting"):
        # Same as SDXL V1.2: get Diffusers key from comfyui_to_diffusers_map
        diffusers_key = None
        if key in comfyui_to_diffusers_map:
            diffusers_key = comfyui_to_diffusers_map[key]
        
        # Module name from diffusers_key (without .weight)
        module_name = None
        if diffusers_key:
            if diffusers_key.endswith(".weight"):
                module_name = diffusers_key[:-7]
            
        # Convert decision (same logic as SDXL V1.2)
        if module_name and module_name in keep_layers:
            # Keep FP16
            new_value = value
            kept_count += 1
        elif diffusers_key:
            # Quantize
            weight_key = diffusers_key + ".weight"
            if diffusers_key.endswith(".weight"):
                weight_key = diffusers_key
            
            if weight_key in weight_amax_dict:
                amax = weight_amax_dict[weight_key]
                if amax == 0: amax = 1e-6
                # Clamp -> FP8 on GPU
                val_gpu = value.float().to(device)
                clamped_value = torch.clamp(val_gpu, -amax, amax)
                new_value = clamped_value.to(torch.float8_e4m3fn).cpu()
                converted_count += 1
                del val_gpu, clamped_value
            else:
                new_value = value
        else:
            new_value = value
            
        output_state_dict[key] = new_value

    print("Conversion done:")
    print(f"  FP8 layers: {converted_count}")
    print(f"  FP16-kept layers: {kept_count}")
    
    save_file(output_state_dict, args.output)
    print("Save complete.")

if __name__ == "__main__":
    main()

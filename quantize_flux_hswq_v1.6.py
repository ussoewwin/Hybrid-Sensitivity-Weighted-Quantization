"""
Quantize Flux1.dev model to FP8 (HSWQ V1.5: high-precision, adaptive).
Same structure/algorithm as SDXL HSWQ V1.2 (GPU Accelerated), plus
Adaptive Search Range based on weight distribution: auto-detect and protect
outlier-heavy models without manual tuning.

Algorithm:
1. Load Pipeline + build comfyui_to_diffusers_map
2. Calibration Loop (DualMonitor: sensitivity & input importance)
3. Layer Selection (keep top N% by sensitivity)
4. HSWQ Optimization (weighted histogram MSE, scaled=False)
   - **NEW: Adaptive search range (kurtosis/outlier-ratio based)**
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
from weighted_histogram_mse import HSWQWeightedHistogramOptimizer, WeightedHistogram, WeightedHistogram

class AdaptiveHSWQOptimizer(HSWQWeightedHistogramOptimizer):
    """HSWQ optimizer with adaptive search range."""
    def compute_optimal_amax(self, weight, importance=None, scaled=True, search_range=(0.5, 1.0)):
        weighted_hist = WeightedHistogram(bins=self.bins, device=self.device)
        weighted_hist.build(weight, importance)
        return self.mse_optimizer.find_optimal_amax(
            weighted_hist,
            num_candidates=self.num_candidates,
            refinement_iterations=self.refinement_iterations,
            search_range=search_range,
            scaled=scaled
        )

# Enforce C++20
if sys.platform == "win32":
    os.environ.setdefault("CXXFLAGS", "/std:c++20")
else:
    os.environ.setdefault("CXXFLAGS", "-std=c++20")

# === SageAttention2 Integration ===
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

# --- Adaptive search range (New in v1.5) ---
def get_adaptive_search_range(weight_tensor: torch.Tensor) -> tuple[float, float]:
    """Choose optimal search range from weight distribution."""
    w_abs = weight_tensor.abs().float()
    max_val = w_abs.max().item()
    
    if max_val == 0:
        return (0.5, 1.0)

    if w_abs.numel() > 1_000_000:
        indices = torch.randint(0, w_abs.numel(), (1_000_000,), device=w_abs.device)
        sample = w_abs.flatten()[indices]
        p99 = torch.quantile(sample, 0.99).item()
        mean = sample.mean().item()
        std = sample.std().item()
        kurtosis = ((sample - mean)**4).mean().item() / (std**4 + 1e-6)
    else:
        p99 = torch.quantile(w_abs, 0.99).item()
        mean = w_abs.mean().item()
        std = w_abs.std().item()
        kurtosis = ((w_abs - mean)**4).mean().item() / (std**4 + 1e-6)

    outlier_ratio = max_val / (p99 + 1e-6)
    
    if outlier_ratio > 3.0:
        return (1.0, 1.0)  # Severe outliers: no clipping, max quant
    elif kurtosis > 50.0:
        return (0.95, 1.0)  # Very conservative
    elif kurtosis > 20.0 or outlier_ratio > 2.2:
        return (0.8, 1.0)   # Conservative
    elif outlier_ratio > 1.8:
        return (0.65, 1.0)  # Some outliers
    else:
        return (0.55, 1.0)  # Normal (near-Gaussian)

# --- ComfyUI-compatible mapping ---

def flux_to_diffusers_mapping(state_dict, key_prefix=None):
    state_dict_keys = list(state_dict.keys())
    
    if key_prefix is None:
        has_full_prefix = any(k.startswith("model.diffusion_model.") for k in state_dict_keys)
        has_no_prefix = any(k.startswith("double_blocks.") or k.startswith("single_blocks.") for k in state_dict_keys)
        
        if has_full_prefix: key_prefix = "model.diffusion_model."
        elif has_no_prefix: key_prefix = ""
        else: key_prefix = "model.diffusion_model."
        print(f"  Auto-detected prefix: '{key_prefix}'")
    
    num_double = 0
    num_single = 0
    for k in state_dict_keys:
        if k.startswith(key_prefix):
            stripped = k[len(key_prefix):]
            if stripped.startswith("double_blocks."):
                parts = stripped.split(".")
                if len(parts) > 1 and parts[1].isdigit(): num_double = max(num_double, int(parts[1]) + 1)
            elif stripped.startswith("single_blocks."):
                parts = stripped.split(".")
                if len(parts) > 1 and parts[1].isdigit(): num_single = max(num_single, int(parts[1]) + 1)
    
    print(f"  Detected: Double Blocks={num_double}, Single Blocks={num_single}")
    
    comfyui_to_diffusers_map = {}
    
    # Embeddings
    embed_map = {
        "txt_in": "context_embedder", "img_in": "x_embedder",
        "time_in.in_layer": "time_text_embed.timestep_embedder.linear_1",
        "time_in.out_layer": "time_text_embed.timestep_embedder.linear_2",
        "guidance_in.in_layer": "time_text_embed.guidance_embedder.linear_1",
        "guidance_in.out_layer": "time_text_embed.guidance_embedder.linear_2",
        "vector_in.in_layer": "time_text_embed.text_embedder.linear_1",
        "vector_in.out_layer": "time_text_embed.text_embedder.linear_2",
        "final_layer.adaLN_modulation.1": "norm_out.linear", "final_layer.linear": "proj_out",
    }
    for suffix in [".weight", ".bias"]:
        for comfy_name, diff_name in embed_map.items():
            comfy_key = f"{key_prefix}{comfy_name}{suffix}"
            if comfy_key in state_dict_keys: comfyui_to_diffusers_map[comfy_key] = f"{diff_name}{suffix}"
    
    # Double Blocks
    for i in range(num_double):
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
                if comfy_key in state_dict_keys: comfyui_to_diffusers_map[comfy_key] = f"{diff_name}{suffix}"
        
        for suffix in [".weight", ".bias"]:
            comfy_key = f"{key_prefix}double_blocks.{i}.img_attn.qkv{suffix}"
            if comfy_key in state_dict_keys: comfyui_to_diffusers_map[comfy_key] = f"FUSED:transformer_blocks.{i}.attn.img_qkv{suffix}"
            comfy_key = f"{key_prefix}double_blocks.{i}.txt_attn.qkv{suffix}"
            if comfy_key in state_dict_keys: comfyui_to_diffusers_map[comfy_key] = f"FUSED:transformer_blocks.{i}.attn.txt_qkv{suffix}"

        for suffix in [".scale"]:
            norm_map = {
                f"double_blocks.{i}.img_attn.norm.query_norm": f"transformer_blocks.{i}.attn.norm_q",
                f"double_blocks.{i}.img_attn.norm.key_norm": f"transformer_blocks.{i}.attn.norm_k",
                f"double_blocks.{i}.txt_attn.norm.query_norm": f"transformer_blocks.{i}.attn.norm_added_q",
                f"double_blocks.{i}.txt_attn.norm.key_norm": f"transformer_blocks.{i}.attn.norm_added_k",
            }
            for comfy_name, diff_name in norm_map.items():
                comfy_key = f"{key_prefix}{comfy_name}{suffix}"
                if comfy_key in state_dict_keys: comfyui_to_diffusers_map[comfy_key] = f"{diff_name}.weight"
    
    # Single Blocks
    for i in range(num_single):
        one_to_one = {
            f"single_blocks.{i}.modulation.lin": f"single_transformer_blocks.{i}.norm.linear",
            f"single_blocks.{i}.linear2": f"single_transformer_blocks.{i}.proj_out",
        }
        for suffix in [".weight", ".bias"]:
            for comfy_name, diff_name in one_to_one.items():
                comfy_key = f"{key_prefix}{comfy_name}{suffix}"
                if comfy_key in state_dict_keys: comfyui_to_diffusers_map[comfy_key] = f"{diff_name}{suffix}"
        
        for suffix in [".weight", ".bias"]:
            comfy_key = f"{key_prefix}single_blocks.{i}.linear1{suffix}"
            if comfy_key in state_dict_keys: comfyui_to_diffusers_map[comfy_key] = f"FUSED:single_transformer_blocks.{i}.linear1{suffix}"
        
        for suffix in [".scale"]:
            norm_map = {
                f"single_blocks.{i}.norm.query_norm": f"single_transformer_blocks.{i}.attn.norm_q",
                f"single_blocks.{i}.norm.key_norm": f"single_transformer_blocks.{i}.attn.norm_k",
            }
            for comfy_name, diff_name in norm_map.items():
                comfy_key = f"{key_prefix}{comfy_name}{suffix}"
                if comfy_key in state_dict_keys: comfyui_to_diffusers_map[comfy_key] = f"{diff_name}.weight"
    
    return comfyui_to_diffusers_map

def load_flux_pipeline_from_safetensors(path, device="cuda", token=None, clip_path=None, t5_path=None, vae_path=None):
    print(f"Loading Flux1 model: {path}")
    original_state_dict = load_file(path)
    print("Building key mapping...")
    comfyui_to_diffusers_map = flux_to_diffusers_mapping(original_state_dict)
    print(f"  Mapping count: {len(comfyui_to_diffusers_map)}")
    
    text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae = None, None, None, None, None
    
    def load_external_component(path, model_class, config_class_or_repo, default_repo=None, tokenizer_repo=None, is_diffusers=False, token=None):
        model, tok = None, None
        if os.path.isfile(path):
            try:
                if is_diffusers:
                    model = model_class.from_pretrained(config_class_or_repo, subfolder="vae", token=token)
                    m, u = model.load_state_dict(load_file(path), strict=False)
                    model.to(torch.float16)
                else:
                    config = config_class_or_repo.from_pretrained(default_repo, token=token)
                    model = model_class(config)
                    m, u = model.load_state_dict(load_file(path), strict=False)
                    model.to(torch.float16)
            except Exception as e:
                print(f"Single-file load failed: {e}")
                sys.exit(1)
            if tokenizer_repo: tok = AutoTokenizer.from_pretrained(tokenizer_repo, token=token)
        else:
            try:
                if is_diffusers: model = model_class.from_pretrained(path, torch_dtype=torch.float16, token=token)
                else:
                    model = model_class.from_pretrained(path, torch_dtype=torch.float16, token=token)
                    tok = AutoTokenizer.from_pretrained(path, token=token)
            except Exception as e:
                print(f"Directory load error: {e}")
                sys.exit(1)
        return model, tok

    if clip_path: text_encoder, tokenizer = load_external_component(clip_path, CLIPTextModel, CLIPTextConfig, "openai/clip-vit-large-patch14", "openai/clip-vit-large-patch14", token=token)
    if t5_path: text_encoder_2, tokenizer_2 = load_external_component(t5_path, T5EncoderModel, T5Config, "google/t5-v1_1-xxl", "google/t5-v1_1-xxl", token=token)
    if vae_path: vae, _ = load_external_component(vae_path, AutoencoderKL, "black-forest-labs/FLUX.1-schnell", is_diffusers=True, token=token)

    print("Building FluxPipeline (manual weight load)...")
    try:
        config_path = "black-forest-labs/FLUX.1-schnell"
        pipeline = FluxPipeline.from_pretrained(config_path, torch_dtype=torch.float16, token=token, text_encoder=text_encoder, text_encoder_2=text_encoder_2, tokenizer=tokenizer, tokenizer_2=tokenizer_2, vae=vae)
        
        converted_state_dict = {}
        print("    Converting/splitting weights to Diffusers format...")
        for comfy_key, val in original_state_dict.items():
            if comfy_key in comfyui_to_diffusers_map:
                diff_key = comfyui_to_diffusers_map[comfy_key]
                if not diff_key.startswith("FUSED:"): converted_state_dict[diff_key] = val
        
        for key, value in original_state_dict.items():
            if "img_attn.qkv" in key and "double_blocks" in key:
                parts = key.split(".")
                if len(parts) >= 5 and parts[0] == "double_blocks" and parts[2] == "img_attn" and parts[3] == "qkv":
                    idx, suffix = parts[1], parts[4]
                    q, k, v = torch.split(value, 3072, dim=0)
                    base = f"transformer_blocks.{idx}.attn"
                    converted_state_dict[f"{base}.to_q.{suffix}"] = q
                    converted_state_dict[f"{base}.to_k.{suffix}"] = k
                    converted_state_dict[f"{base}.to_v.{suffix}"] = v
            elif "txt_attn.qkv" in key and "double_blocks" in key:
                parts = key.split(".")
                if len(parts) >= 5 and parts[0] == "double_blocks" and parts[2] == "txt_attn" and parts[3] == "qkv":
                    idx, suffix = parts[1], parts[4]
                    q, k, v = torch.split(value, 3072, dim=0)
                    base = f"transformer_blocks.{idx}.attn"
                    converted_state_dict[f"{base}.add_q_proj.{suffix}"] = q
                    converted_state_dict[f"{base}.add_k_proj.{suffix}"] = k
                    converted_state_dict[f"{base}.add_v_proj.{suffix}"] = v
            elif "linear1" in key and "single_blocks" in key:
                parts = key.split(".")
                if len(parts) >= 4 and parts[0] == "single_blocks" and parts[2] == "linear1":
                    idx, suffix = parts[1], parts[3]
                    q, k, v, mlp = torch.split(value, [3072, 3072, 3072, 12288], dim=0)
                    base = f"single_transformer_blocks.{idx}"
                    converted_state_dict[f"{base}.attn.to_q.{suffix}"] = q
                    converted_state_dict[f"{base}.attn.to_k.{suffix}"] = k
                    converted_state_dict[f"{base}.attn.to_v.{suffix}"] = v
                    converted_state_dict[f"{base}.proj_mlp.{suffix}"] = mlp

        m, u = pipeline.transformer.load_state_dict(converted_state_dict, strict=False)
        print(f"Manual load - Missing: {len(m)}, Unexpected: {len(u)}")
        
    except Exception as e:
        print(f"Manual load failed: {e}")
        sys.exit(1)
            
    print("Enabling Model CPU Offload for VRAM...")
    pipeline.enable_model_cpu_offload()
    return pipeline, original_state_dict, comfyui_to_diffusers_map

class DualMonitor:
    def __init__(self):
        self.output_sum, self.output_sq_sum, self.count = 0.0, 0.0, 0
        self.channel_importance = None
    def update(self, input_tensor, output_tensor):
        with torch.no_grad():
            self.output_sum += output_tensor.detach().float().mean().item()
            self.output_sq_sum += (output_tensor.detach().float() ** 2).mean().item()
            inp_detached = input_tensor.detach()
            if inp_detached.dim() == 4: current_imp = inp_detached.abs().mean(dim=(0, 2, 3))
            elif inp_detached.dim() == 3: current_imp = inp_detached.abs().mean(dim=(0, 1))
            elif inp_detached.dim() == 2: current_imp = inp_detached.abs().mean(dim=0)
            else: current_imp = torch.ones(1, device=inp_detached.device, dtype=inp_detached.dtype)
            if self.channel_importance is None: self.channel_importance = current_imp
            else: self.channel_importance = (self.channel_importance * self.count + current_imp) / (self.count + 1)
            self.count += 1
    def get_sensitivity(self):
        if self.count == 0: return 0.0
        mean = self.output_sum / self.count
        sq_mean = self.output_sq_sum / self.count
        return sq_mean - mean ** 2

dual_monitors = {}
def hook_fn(module, input, output, name):
    if name not in dual_monitors: dual_monitors[name] = DualMonitor()
    dual_monitors[name].update(input[0], output)

def main():
    parser = argparse.ArgumentParser(description="Flux1.dev FP8 quantization (HSWQ V1.5: high-precision, adaptive)")
    parser.add_argument("--input", type=str, required=True, help="Path to input safetensors model")
    parser.add_argument("--output", type=str, required=True, help="Path to output safetensors model")
    parser.add_argument("--calib_file", type=str, required=True, help="Path to calibration prompts text file")
    parser.add_argument("--num_calib_samples", type=int, default=256, help="Calibration samples (recommended: 256)")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Inference steps (default: 25)")
    parser.add_argument("--keep_ratio", type=float, default=0.25, help="Ratio of layers to keep in FP16")
    parser.add_argument("--sa2", action="store_true", help="Use SageAttention2 for faster calibration")
    parser.add_argument("--token", type=str, help="Hugging Face token")
    parser.add_argument("--clip_path", type=str, help="CLIP path")
    parser.add_argument("--t5_path", type=str, help="T5 path")
    parser.add_argument("--vae_path", type=str, help="VAE path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if args.sa2:
        if try_import_sage_attention(): enable_sage_attention()

    pipeline, original_state_dict, comfyui_to_diffusers_map = load_flux_pipeline_from_safetensors(
        args.input, device, token=args.token, clip_path=args.clip_path, t5_path=args.t5_path, vae_path=args.vae_path
    )

    print("Preparing calibration...")
    handles = []
    target_modules = []
    for name, module in pipeline.transformer.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n)))
            target_modules.append(name)

    print("Running calibration...")
    with open(args.calib_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    prompts = (prompts * (args.num_calib_samples // len(prompts) + 1))[:args.num_calib_samples]

    pipeline.set_progress_bar_config(disable=False)
    for i, prompt in enumerate(prompts):
        print(f"\nSample {i+1}/{args.num_calib_samples}: {prompt[:50]}...")
        with torch.no_grad():
            pipeline(prompt=prompt, num_inference_steps=args.num_inference_steps, guidance_scale=3.5, output_type="latent")
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    for h in handles: h.remove()
    if args.sa2: disable_sage_attention()

    print("\nLayer sensitivity analysis...")
    layer_sensitivities = []
    for name in target_modules:
        if name in dual_monitors:
            layer_sensitivities.append((name, dual_monitors[name].get_sensitivity()))
    layer_sensitivities.sort(key=lambda x: x[1], reverse=True)
    num_keep = int(len(layer_sensitivities) * args.keep_ratio)
    keep_layers = set([x[0] for x in layer_sensitivities[:num_keep]])
    print(f"Total layers: {len(layer_sensitivities)}, kept: {len(keep_layers)} (top {args.keep_ratio*100:.1f}%)")

    print("\n[HSWQ V1.5] Starting adaptive high-precision optimization...")
    print("High-precision: bins=8192, candidates=1000, iterations=10")
    print("Adaptive search range: enabled")
    
    weight_amax_dict = {}
    hswq_optimizer = AdaptiveHSWQOptimizer(
        bins=8192,
        num_candidates=1000,
        refinement_iterations=10,
        device=device
    )
    
    for name, module in tqdm(pipeline.transformer.named_modules(), desc="Analyzing"):
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if name in keep_layers: continue
            
            importance = dual_monitors[name].channel_importance if name in dual_monitors else None
            
            # --- Adaptive analysis ---
            adaptive_range = get_adaptive_search_range(module.weight.data)
            # -----------------
            
            optimal_amax = hswq_optimizer.compute_optimal_amax(
                module.weight.data, 
                importance,
                scaled=False,
                search_range=adaptive_range
            )
            weight_amax_dict[name + ".weight"] = optimal_amax
            torch.cuda.empty_cache()

    first_fused_module_name = {}
    for name, module in pipeline.transformer.named_modules():
        if isinstance(module, torch.nn.Linear):
            parts = name.split(".")
            if len(parts) >= 2 and parts[-1].startswith("to_q"):
                if parts[0] == "transformer_blocks": first_fused_module_name[f"transformer_blocks.{parts[1]}.attn.img_qkv"] = name
                elif parts[0] == "single_transformer_blocks": first_fused_module_name[f"single_transformer_blocks.{parts[1]}.linear1"] = name
            elif len(parts) >= 2 and parts[-1] == "add_q_proj":
                if parts[0] == "transformer_blocks": first_fused_module_name[f"transformer_blocks.{parts[1]}.attn.txt_qkv"] = name

    fused_count = 0
    for key, value in original_state_dict.items():
        if key in comfyui_to_diffusers_map:
            diffusers_key = comfyui_to_diffusers_map[key]
            if diffusers_key.startswith("FUSED:") and diffusers_key.endswith(".weight"):
                target_base = diffusers_key[6:-7]
                importance = None
                if target_base in first_fused_module_name:
                    rep_name = first_fused_module_name[target_base]
                    if rep_name in dual_monitors: importance = dual_monitors[rep_name].channel_importance

                if value.dim() == 2 and value.numel() >= 1024:
                    # --- Adaptive analysis (Fused) ---
                    adaptive_range = get_adaptive_search_range(value)
                    # -------------------------
                    
                    optimal_amax = hswq_optimizer.compute_optimal_amax(
                        value, importance, scaled=False,
                        search_range=adaptive_range
                    )
                    weight_amax_dict[diffusers_key] = optimal_amax
                    fused_count += 1
    
    print(f"Layers to quantize: {len(weight_amax_dict)} (fused: {fused_count})")
    
    del pipeline
    del hswq_optimizer
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"Saving quantized model: {args.output}")
    output_state_dict = {}
    converted_count = 0
    kept_count = 0
    
    for key, value in tqdm(original_state_dict.items(), desc="Converting"):
        diffusers_key = None
        if key in comfyui_to_diffusers_map: diffusers_key = comfyui_to_diffusers_map[key]
        
        module_name = None
        if diffusers_key and diffusers_key.endswith(".weight"): module_name = diffusers_key[:-7]
        
        if module_name and module_name in keep_layers:
            new_value = value
            kept_count += 1
        elif diffusers_key:
            weight_key = diffusers_key if diffusers_key.endswith(".weight") else diffusers_key + ".weight"
            
            if weight_key in weight_amax_dict:
                amax = weight_amax_dict[weight_key]
                if amax == 0: amax = 1e-6
                
                # GPU quantization (Clamp only, no Scale)
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

    print(f"Done. Quantized: {converted_count}, kept: {kept_count}")
    save_file(output_state_dict, args.output)

if __name__ == "__main__":
    main()

"""
Z-Image Base (Non-Turbo) FP8 Quantization - HSWQ V1.9 (Pure Data-Driven Autonomous Engine)
Target Model: Z-Image Base (e.g., UR03: moodyWildV0200001.UR03.safetensors)

Design Philosophy:
  1. Mandatory Analysis: Relies on weight distribution profiles (Kurtosis, Outlier Ratio).
     Automatically triggers analysis/analyze_zib_distribution.py if profile is missing.
  2. Autonomous Strategy: No hardcoded Alpha/Beta. Derived from global model statistics.
  3. Dynamic Protection: Individual layer search ranges (search_low) decided by local stats.
  4. Environment Agnostic: Relative pathing for scripts and profiles (Cloud/Local support).
"""
import argparse
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
import os
import gc
from tqdm import tqdm
import sys
import json
import numpy as np
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "ComfyUI-master"))

# Ensure histogram modules are importable regardless of clone path / CWD
histogram_dir = os.path.join(current_dir, "histogram")
if histogram_dir not in sys.path:
    sys.path.insert(0, histogram_dir)

# Support for SageAttention2 in virtual environment (venv)
venv_site_packages = os.path.join(os.path.dirname(current_dir), "venv", "Lib", "site-packages")
if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
    sys.path.append(venv_site_packages)

from weighted_histogram_mse_v4 import HSWQWeightedHistogramOptimizerV4

# Enforce C++20
if sys.platform == "win32":
    os.environ.setdefault("CXXFLAGS", "/std:c++20")
else:
    os.environ.setdefault("CXXFLAGS", "-std=c++20")

# === SageAttention2 Integration (optional, for faster calibration with --sa2) ===
_sage_attn_available = False
_original_sdpa = None

def try_import_sage_attention():
    """Attempt to import SageAttention2 and return availability status."""
    global _sage_attn_available
    try:
        import sageattention
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
    
    if _original_sdpa is not None:
        # Already enabled
        return True
    
    import torch.nn.functional as F
    from sageattention import sageattn
    
    _original_sdpa = F.scaled_dot_product_attention
    
    def sage_sdpa_wrapper(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        if attn_mask is not None or is_causal:
            return _original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
        try:
            return sageattn(query, key, value, is_causal=False)
        except Exception:
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

# --- Z-Image Base (NextDiT) model load and inference pipeline ---
ZIT_PREFIXES = [
    "model.diffusion_model.",
    "model.",
    "diffusion_model.",
    "",
]

def calculate_kurtosis(tensor):
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    if std == 0: return 0.0
    return torch.mean(((tensor - mean) / std) ** 4).item()

def detect_and_strip_prefix(state_dict):
    keys = list(state_dict.keys())
    for prefix in ZIT_PREFIXES:
        if prefix == "":
            if any(k.startswith("layers.") or k.startswith("x_embedder") for k in keys):
                return state_dict, ""
        else:
            test_key = f"{prefix}layers.0.attention_norm1.weight"
            if test_key in keys:
                print(f"  [Prefix Detection] Found prefix: '{prefix}'")
                stripped = {}
                for k, v in state_dict.items():
                    if k.startswith(prefix):
                        stripped[k[len(prefix):]] = v
                    else:
                        stripped[k] = v
                return stripped, prefix
    print("  [Prefix Detection] No prefix detected (assuming HSWQ format)")
    return state_dict, ""

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
        zit_config["hidden_size"] = 3840
    
    refiner_indices = set()
    for key in state_dict_keys:
        if key.startswith("context_refiner."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                refiner_indices.add(int(parts[1]))
    zit_config["num_context_refiner"] = max(refiner_indices) + 1 if refiner_indices else 2
    
    noise_indices = set()
    for key in state_dict_keys:
        if key.startswith("noise_refiner."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                noise_indices.add(int(parts[1]))
    zit_config["num_noise_refiner"] = max(noise_indices) + 1 if noise_indices else 2
    
    for key in state_dict_keys:
        if "feed_forward.w1.weight" in key:
            zit_config["intermediate_size"] = state_dict[key].shape[0]
            break
            
    if "intermediate_size" not in zit_config:
        zit_config["intermediate_size"] = 10240
    return zit_config

def resolve_tokenizer_offline(provided_path, comfy_path, clip_path=None):
    """Offline-only logic to find a local tokenizer (ZIB/Qwen-compatible)."""
    validation_files = ["tokenizer.json", "vocab.json", "config.json"]
    
    # Candidate 1: explicit path
    if provided_path and os.path.isdir(provided_path):
        if any(os.path.exists(os.path.join(provided_path, f)) for f in validation_files):
            return provided_path

    # Candidate 2: ComfyUI standard locations and near CLIP weights
    search_roots = []
    if comfy_path:
        search_roots.extend([
            os.path.join(comfy_path, "models", "clip"),
            os.path.join(comfy_path, "models", "tokenizers"),
            comfy_path
        ])
    if clip_path and os.path.exists(clip_path):
        search_roots.append(os.path.dirname(os.path.abspath(clip_path)))

    for root_dir in search_roots:
        if not os.path.exists(root_dir): continue
        for root, dirs, files in os.walk(root_dir):
            if any(f in files for f in validation_files):
                if any(x in root.lower() for x in ["qwen", "qwen2.5", "qwen3", "zit", "zib"]):
                    print(f"  [Offline Discovery] Found Qwen-compatible tokenizer: {root}")
                    return root

    # Candidate 3: recursive search (skip ComfyUI etc.)
    print("  Note: Searching recursively for any local Qwen tokenizer...")
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ["ComfyUI-master", "node_modules"]]
        if any(f in files for f in validation_files):
            if any(x in root.lower() for x in ["qwen", "qwen2.5", "qwen3", "zit", "zib"]):
                print(f"  [Offline Discovery] Found potential tokenizer: {root}")
                return root
                
    # Fallback: any tokenizer.json (exclude SD family)
    for root_dir in search_roots:
        if not os.path.exists(root_dir): continue
        for root, dirs, files in os.walk(root_dir):
            if "tokenizer.json" in files:
                root_lower = root.lower()
                if any(x in root_lower for x in ["sd1", "sd2", "sdxl", "stable-diffusion", "clip-vit"]):
                    continue
                print(f"  [Offline Discovery] Found generic fallback tokenizer: {root}")
                return root

    return None

def load_zit_model(path, device="cuda", comfy_path=None):
    print(f"Loading Base model: {path}")
    original_state_dict = load_file(path)
    stripped_state_dict, detected_prefix = detect_and_strip_prefix(original_state_dict)
    
    print("Detecting Structure (Base Model)...")
    zit_config = detect_zit_config_from_keys(stripped_state_dict)
    print(f"Detected Config: {zit_config}")
    
    print("Initializing NextDiT model...")
    if comfy_path is None:
        comfy_path = os.environ.get("COMFYUI_PATH", os.path.join(os.getcwd(), "ComfyUI"))
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)
    
    from comfy.ldm.lumina.model import NextDiT
    import comfy.ops
    
    ops = comfy.ops.disable_weight_init
    ffn_multiplier = 8/3
    if zit_config.get("intermediate_size"):
        ffn_multiplier = zit_config["intermediate_size"] / zit_config["hidden_size"]
    
    model = NextDiT(
        patch_size=2,
        in_channels=16,
        dim=zit_config["hidden_size"],
        n_layers=zit_config["num_layers"],
        n_refiner_layers=zit_config["num_context_refiner"],
        n_heads=zit_config["hidden_size"] // 128,
        n_kv_heads=zit_config["hidden_size"] // 128,
        multiple_of=256,
        ffn_dim_multiplier=ffn_multiplier,
        norm_eps=1e-5,
        cap_feat_dim=2560,
        z_image_modulation=True,
        pad_tokens_multiple=64,
        device="cpu",
        dtype=torch.float16,
        operations=ops,
    )
    
    print("Loading Weights...")
    converted_state_dict = {}
    for key, value in stripped_state_dict.items():
        if value.dtype == torch.bfloat16:
            converted_state_dict[key] = value.to(torch.float16)
        else:
            converted_state_dict[key] = value
            
    missing, unexpected = model.load_state_dict(converted_state_dict, strict=False)
    matched = len(converted_state_dict) - len(unexpected)
    match_rate = matched / len(converted_state_dict) if len(converted_state_dict) > 0 else 0
    print(f"  [Keys] Matched: {matched}, Missing: {len(missing)}, Unexpected: {len(unexpected)} (Rate: {match_rate*100:.1f}%)")
    
    # [Physical safeguard] Abort immediately if key match rate is too low to avoid quantizing effectively random weights
    if match_rate < 0.5:
        print("\n[FATAL ERROR] Key match rate is abnormally low (< 50%).")
        print("Due to prefix mismatch, weights are effectively random. Quantizing in this state will only produce garbage.")
        print("Please double-check your arguments and model structure.")
        sys.exit(1)
    
    model = model.to(device).to(torch.float16)
    model.eval()
    return model, original_state_dict, stripped_state_dict, zit_config, detected_prefix

class ZITCalibrationPipeline:
    def __init__(self, model, text_encoder, tokenizer, device="cuda"):
        self.model = model
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.device = device
        self.hidden_dim = model.dim if hasattr(model, 'dim') else 3840
        if self.text_encoder is not None:
            self.text_encoder = self.text_encoder.to(device)
        self.prng = np.random.RandomState(42)
        self.sampler_name = "euler"
        
    def encode_prompt(self, prompt):
        llama_template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        formatted_prompt = llama_template.format(prompt)
        tokens = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True, max_length=256)
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, intermediate_output=-2)
            val = outputs[1]
        return val, attention_mask.bool()
        
    def __call__(self, prompt, num_inference_steps=20, **kwargs):
        batch_size = 1
        latent_h, latent_w, latent_c = 128, 128, 16
        if self.text_encoder is not None:
            cap_feats, cap_mask = self.encode_prompt(prompt)
        else:
            cap_len = 256
            cap_feats = torch.randn(batch_size, cap_len, 2560, device=self.device, dtype=torch.float16)
            cap_mask = torch.ones(batch_size, cap_len, device=self.device, dtype=torch.bool)
        
        import comfy.k_diffusion.sampling as k_sampling
        class ZITWrapper:
            def __init__(self, model, cap_feats, cap_mask):
                self.model = model
                self.cap_feats = cap_feats
                self.cap_mask = cap_mask
            def __call__(self, x, sigma, **kwargs):
                dtype = torch.float16
                try:
                    return self.model(x.to(dtype=dtype), sigma.to(dtype=dtype), self.cap_feats.to(dtype=dtype), None, attention_mask=self.cap_mask).to(dtype=x.dtype)
                except: return torch.zeros_like(x)

        x = torch.randn(batch_size, latent_c, latent_h, latent_w, device=self.device, dtype=torch.float16)
        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=self.device)
        model_wrap = ZITWrapper(self.model, cap_feats, cap_mask)
        
        try:
             sampler_func_name = f"sample_{self.sampler_name}"
             sampler_func = getattr(k_sampling, sampler_func_name, k_sampling.sample_euler)
             # [Physical fix] Capture and return the sampler result instead of discarding it
             result = sampler_func(model_wrap, x, sigmas, disable=False)
             return {"latent": result}
        except Exception as e: 
            print(f"Sampling failed: {e}")
            return {"latent": None}

class DualMonitor:
    def __init__(self):
        self.output_sum = 0.0
        self.output_sq_sum = 0.0
        self.count = 0
        self.channel_importance = None
    
    def update(self, input_tensor, output_tensor):
        with torch.no_grad():
            out_detached = output_tensor.detach().float()
            # Prevent NaN/Inf: clamp before squaring to avoid FP16 overflow on large outputs
            out_clamped = torch.clamp(out_detached, -65504.0, 65504.0)
            mean_val = out_clamped.mean().item()
            sq_mean_val = (out_clamped ** 2).mean().item()
            # Guard against NaN/Inf
            import math
            if math.isfinite(mean_val) and math.isfinite(sq_mean_val):
                self.output_sum += mean_val
                self.output_sq_sum += sq_mean_val
            else:
                pass  # Skip corrupted batch
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
        variance = (self.output_sq_sum / self.count) - mean ** 2
        import math
        return variance if math.isfinite(variance) else 0.0

dual_monitors = {}
def hook_fn(module, input, output, name):
    if name not in dual_monitors: dual_monitors[name] = DualMonitor()
    dual_monitors[name].update(input[0], output)

def derive_hswq_strategy(model_profile):
    """
    [Pure Data-Driven Engine]
    Derives Alpha/Beta from global model statistics and returns a continuous
    evaluation function that decides per-layer search_low without hardcoded thresholds.
    """
    
    # [CRITICAL FIX] Automatically detect and strip model prefixes from profile keys
    # so they match detect_and_strip_prefix outputs (layers.X.xxx, etc.).
    # This makes the design independent of load_zit_model call order.
    if model_profile:
        sample_key = next(iter(model_profile))
        profile_prefix = ""
        for pfx in ZIT_PREFIXES:
            if pfx and sample_key.startswith(pfx):
                profile_prefix = pfx
                break
        if profile_prefix:
            normalized_profile = {}
            for key, val in model_profile.items():
                stripped_key = key[len(profile_prefix):] if key.startswith(profile_prefix) else key
                normalized_profile[stripped_key] = val
            model_profile = normalized_profile
            print(f"  [Profile Normalize] Stripped prefix '{profile_prefix}' from {len(normalized_profile)} profile keys.")
    
    # --- Purely mathematical search_low computation ---
    def get_dynamic_search_low(name, weight_tensor):
        profile_key = name + ".weight"
        prof = model_profile.get(profile_key, model_profile.get(name, {})) if model_profile else {}
        
        if prof:
            k_stat = prof.get("kurtosis", 0)
            o_ratio = prof.get("outlier_ratio", 0)
        else:
            # If profile entry is missing, compute statistics on-the-fly instead of falling back to a fixed default
            t_f32 = weight_tensor.float()
            k_stat = calculate_kurtosis(t_f32)
            std = torch.std(t_f32).item()
            abs_max = max(abs(t_f32.min().item()), abs(t_f32.max().item()))
            o_ratio = float(abs_max / std if std > 0 else 0)
            
        # Remove ad-hoc if/else thresholds and use a continuous mathematical mapping.
        # Map kurtosis (<=100) and outlier ratio (<=60) into a continuous range 0.50–0.99.
        k_penalty = min(k_stat / 100.0, 0.49)
        o_penalty = min(o_ratio / 60.0, 0.49)
        
        # Use 0.50 as the base and raise the protection line according to the strongest abnormality
        return float(np.clip(0.50 + max(k_penalty, o_penalty), 0.50, 0.99))

    # --- Decide global strategy (Alpha/Beta) ---
    if not model_profile:
        print("  [Strategy] No profile data available. Using continuous mathematical fallback.")
        return 0.5, 0.5, get_dynamic_search_low

    k_vals = [v.get("kurtosis", 0) for v in model_profile.values() if isinstance(v, dict)]
    avg_k = sum(k_vals) / len(k_vals) if k_vals else 0
    
    print(f"\n[Autonomous Strategy Analysis]")
    print(f"  Avg Kurtosis across model: {avg_k:.2f}")

    # [V1.9 Pure Data-Driven Finalization]
    # Remove ad-hoc if branches.
    # Start from (0.5 / 0.5), then increase Alpha (SVD protection ratio) up to 0.8
    # in proportion to avg_k (global kurtosis), keeping alpha + beta = 1.0.
    k_factor = min(avg_k / 50.0, 0.3)  # Max +0.3 (50.0 is a scaling constant)
    alpha = float(np.clip(0.5 + k_factor, 0.5, 0.8))
    beta = 1.0 - alpha  # Always keep the sum at 1.0
    
    print(f"  --> Pure Data-Driven Ratio: Alpha(SVD)={alpha:.3f}, Beta(Mag)={beta:.3f}")

    # [NEW] Pre-extract layers that exceed FP8 mathematical limits (Hard VETO)
    hard_veto_layers = set()
    if model_profile:
        for name, prof in model_profile.items():
            if isinstance(prof, dict):
                k = prof.get("kurtosis", 0)
                m = prof.get("abs_max", 0)
                o = prof.get("outlier_ratio", 0)
                
                # Measure ZIB's characteristic \"dense band vs. outliers\" behavior and exclude layers
                # that would clearly not fit into unscaled FP8.
                # Stable layers typically have kurtosis in 0.1–5.0; >20 is a clear deviation (e.g. adaLN-like mods).
                is_extreme_divergence = (o > 40)  # Very high outlier ratio where FP8 resolution crushes the center
                is_extreme_kurtosis = (k > 20)    # Distribution deviates strongly from normal
                is_huge_magnitude = (m > 20)      # Magnitude beyond FP8 E4M3 safe range
                
                if is_extreme_divergence or is_extreme_kurtosis or is_huge_magnitude:
                    layer_base_name = name.replace(".weight", "") if name.endswith(".weight") else name
                    hard_veto_layers.add(layer_base_name)
                    reasons = []
                    if is_extreme_kurtosis: reasons.append(f"k={k:.1f}")
                    if is_extreme_divergence: reasons.append(f"o={o:.1f}")
                    if is_huge_magnitude: reasons.append(f"m={m:.2f}")
                    print(f"    VETO: {layer_base_name} [{', '.join(reasons)}]")
                    
    print(f"  [Static Profile VETO] Identified {len(hard_veto_layers)} layers with extreme distribution (Unquantizable in FP8).")

    return alpha, beta, get_dynamic_search_low, hard_veto_layers

def main():
    parser = argparse.ArgumentParser(description="Z-Image Base FP8 Quantization - HSWQ V1.9 (Autonomous Engine)")
    parser.add_argument("--input", type=str, required=True, help="Path to input safetensors model")
    parser.add_argument("--output", type=str, required=True, help="Path to output safetensors model")
    parser.add_argument("--calib_file", type=str, required=True, help="Path to calibration prompts text file")
    parser.add_argument("--clip_path", type=str, required=True, help="Path to text encoder safetensors")
    parser.add_argument("--num_calib_samples", type=int, default=256, help="Number of calibration samples")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--keep_ratio", type=float, default=0.25, help="Ratio of layers to keep in FP16")
    parser.add_argument("--comfy_path", type=str, help="Path to ComfyUI root directory (optional, will auto-detect)")
    parser.add_argument("--profile", type=str, help="Path to distribution profile JSON (optional, will auto-generate if missing)")
    parser.add_argument("--tokenizer_path", type=str, help="Path to tokenizer (optional)")
    parser.add_argument("--token", type=str, help="Hugging Face API token for fallback download (optional)")
    parser.add_argument("--sa2", action="store_true", help="Enable SageAttention2 for faster calibration (requires sageattention package)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("HSWQ V1.9 Autonomous Engine (Environment-Aware Analysis)")
    print("=" * 60)
    
    # === V1.5: SageAttention2 Initialization ===
    if args.sa2:
        if try_import_sage_attention():
            enable_sage_attention()
        else:
            print("[Warning] --sa2 specified but SageAttention2 not available. Continuing with standard attention.")

    # --- ComfyUI Path Setup ---
    comfy_path = args.comfy_path
    if comfy_path is None:
        comfy_path = os.environ.get("COMFYUI_PATH", os.path.join(os.getcwd(), "ComfyUI"))
    
    if os.path.exists(comfy_path):
        if comfy_path not in sys.path:
            sys.path.insert(0, comfy_path)
    
    # Robust tokenizer resolution
    tokenizer = None
    text_encoder = None 
    try:
        import comfy.ops
        from comfy.text_encoders import llama as llama_module
        from transformers import Qwen2Tokenizer
        
        # Robust tokenizer discovery (V1.9 Autonomous search / Strictly Offline with Discovery)
        tokenizer_dir = resolve_tokenizer_offline(args.tokenizer_path, args.comfy_path, args.clip_path)
        
        if tokenizer_dir:
            print(f"  Loading tokenizer from disk: {tokenizer_dir}")
            try:
                tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
            except Exception as e:
                print(f"  Warning: Failed to load {tokenizer_dir} with local_files_only. Error: {e}")
                print("  Retrying without local_files_only (Risk of 403)...")
                tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_dir)
        else:
            # Last resort: try Repo ID with local_files_only
            model_id = args.tokenizer_path if args.tokenizer_path else "Qwen/Qwen2.5-7B-Instruct"
            print(f"  CRITICAL: Local tokenizer not found. Trying Repo ID: {model_id} (STRICT LOCAL)")
            try:
                tokenizer = Qwen2Tokenizer.from_pretrained(model_id, local_files_only=True)
            except Exception as e:
                print(f"  FATAL: Offline load failed. 403 Forbidden is inevitable without local tokenizer files.")
                print(f"  [PROMPT] Please ensure tokenizer files (tokenizer.json etc.) exist in {os.path.join(args.comfy_path, 'models/clip/qwen_tokenizer') if args.comfy_path else './tokenizers/qwen'}")
                sys.exit(1)
        
        print(f"[*] Loading Text Encoder: {args.clip_path}")
        state_dict = load_file(args.clip_path)
        text_encoder = llama_module.Qwen3_4B(config_dict={}, device=device, dtype=torch.float16, operations=comfy.ops.disable_weight_init)
        # Load while allowing partial key mismatch
        text_encoder.load_state_dict(state_dict, strict=False)
        text_encoder.eval()
        
    except Exception as e:
        print(f"[FATAL] Failed to load tokenizer/text_encoder: {e}")
        sys.exit(1)
    
    # --- 1. Locate Analysis Script & Profile --- (Environment-Agnostic)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    analyze_script = os.path.join(script_dir, "analyze", "analyze_zib_distribution.py")
    if not os.path.exists(analyze_script):
        analyze_script = os.path.join(script_dir, "analyze_zib_distribution.py")
    
    input_abs = os.path.abspath(args.input)
    input_root = os.path.splitext(os.path.basename(args.input))[0]
    
    # Profile path and run policy
    profile_path = args.profile
    is_auto = False
    if not profile_path:
        profile_path = os.path.join(script_dir, f"{input_root}_distribution_profile.json")
        is_auto = True
    
    # When path is auto-generated, always re-run analysis (do not skip even if file exists)
    should_run_analysis = is_auto or not os.path.exists(profile_path)
    
    if should_run_analysis:
        if os.path.exists(analyze_script):
            print(f"[*] Executing mandated distribution analysis (No skip policy):")
            print(f"    Script: {analyze_script}")
            print(f"    Input:  {input_abs}")
            print(f"    Result: {profile_path}")
            subprocess.run([sys.executable, analyze_script, "--input", input_abs, "--output", profile_path], check=True)
        else:
            print(f"[*] Warning: Analysis script NOT found. (Expected: {analyze_script})")
            print("    Will proceed with internal backup strategy (on-the-fly calc).")

    model_profile = {}
    if os.path.exists(profile_path):
        print(f"[*] Loading Analysis Data: {profile_path}")
        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)
            model_profile = profile_data.get("layers", profile_data)
    
    # --- 2. Strategy & Model Load ---
    alpha, beta, get_layer_search_low, hard_veto_layers = derive_hswq_strategy(model_profile)
    model, original_state_dict, stripped_state_dict, zit_config, detected_prefix = load_zit_model(args.input, device, args.comfy_path)
    
    # tokenizer and text_encoder are already loaded in the initial block
    pipeline = ZITCalibrationPipeline(model, text_encoder, tokenizer, device)

    print("Preparing calibration (Dual Monitor hooks)...")
    handles, target_modules = [], []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handle = module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))
            handles.append(handle)
            target_modules.append(name)

    print("Preparing calibration data...")
    with open(args.calib_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()][:args.num_calib_samples]
    if len(prompts) < args.num_calib_samples: prompts = (prompts * (args.num_calib_samples // len(prompts) + 1))[:args.num_calib_samples]

    print(f"Running calibration ({args.num_calib_samples} samples)...")
    for i, prompt in enumerate(prompts):
        print(f"\nSample {i+1}/{args.num_calib_samples}: {prompt[:50]}...")
        with torch.no_grad(): pipeline(prompt=prompt, num_inference_steps=args.num_inference_steps)
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    for h in handles: h.remove()
    
    if args.sa2:
        disable_sage_attention()

    print("\nAnalyzing layer sensitivity (Profile-Based)...")
    # DualMonitor variance is scale-dependent and inaccurate, so we use
    # the distribution profile (kurtosis + outlier_ratio) as a continuous score instead.
    
    # model_profile keys may still contain prefixes (model.diffusion_model.),
    # so we build a prefix-stripped dictionary using the same approach as derive_hswq_strategy.
    _norm_profile = {}
    for _pk, _pv in model_profile.items():
        if isinstance(_pv, dict):
            _stripped = _pk
            for _pfx in ZIT_PREFIXES:
                if _pfx and _stripped.startswith(_pfx):
                    _stripped = _stripped[len(_pfx):]
                    break
            # Strip `.weight` suffix to normalize to module names
            if _stripped.endswith(".weight"):
                _stripped = _stripped[:-7]
            _norm_profile[_stripped] = _pv
    
    # Exclude VETO layers from the Dynamic pool (they are always FP16, so Dynamic budget should go elsewhere)
    layer_sensitivities = []
    for name in target_modules:
        if name in hard_veto_layers:
            continue  # Remove VETO layers from the candidate pool
        prof = _norm_profile.get(name, {})
        k = prof.get("kurtosis", 0)
        o = prof.get("outlier_ratio", 0)
        m = prof.get("abs_max", 0)
        profile_score = k + o * 2.0 + m * 0.5
        layer_sensitivities.append((name, profile_score))
    
    layer_sensitivities.sort(key=lambda x: x[1], reverse=True)
    num_keep_dynamic = int(len(layer_sensitivities) * args.keep_ratio)
    dynamic_keep_layers = set([x[0] for x in layer_sensitivities[:num_keep_dynamic]])
    
    # [V1.92 Exclusive Protection] VETO (always FP16) + Dynamic (additional FP16) with no overlap for maximum coverage
    keep_layers = dynamic_keep_layers.union(hard_veto_layers)
    
    non_veto_total = len(layer_sensitivities)
    print(f"Total layers: {non_veto_total + len(hard_veto_layers)} (Non-VETO pool: {non_veto_total})")
    print(f"Dynamic kept (from non-VETO pool): {len(dynamic_keep_layers)} (Top {args.keep_ratio*100:.1f}% of {non_veto_total})")
    print(f"Static kept (Hard VETO): {len(hard_veto_layers)} (Always FP16)")
    print(f"Final FP16 kept layers: {len(keep_layers)} (VETO {len(hard_veto_layers)} + Dynamic {len(dynamic_keep_layers)})")
    
    print("\n--- Hard VETO Layers Detail ---")
    for veto_name in sorted(hard_veto_layers):
        in_dynamic = '(+Dynamic)' if veto_name in dynamic_keep_layers else '(VETO only)'
        print(f"  FP16 {in_dynamic}: {veto_name}")
    
    print("\nTop 10 Sensitive Layers (Dynamic):")
    for i in range(min(10, len(layer_sensitivities))):
        name, sens = layer_sensitivities[i]
        in_veto = ' [+VETO]' if name in hard_veto_layers else ''
        print(f"  {i+1}. {name}: {sens:.4f}{in_veto}")

    print("\n[HSWQ V1.9 Autonomous Engine] Starting Optimization...")
    weight_amax_dict = {}
    hswq_optimizer = HSWQWeightedHistogramOptimizerV4(
        bins=8192,
        num_candidates=1000,
        refinement_iterations=10,
        device=device,
        alpha=alpha,
        beta=beta
    )
    
    for name, module in tqdm(model.named_modules(), desc="Analyzing"):
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if name in keep_layers: continue
            
            importance = dual_monitors[name].channel_importance if name in dual_monitors else None
            
            # Obtain the lower bound from the pure mathematical function and adapt it to the V4 signature
            layer_search_low = get_layer_search_low(name, module.weight.data)
            layer_search_range = (layer_search_low, 1.0)
            
            print(f"  [HSWQ] {name:50} | Pure Data-Driven | search_range={layer_search_range[0]:.3f}-{layer_search_range[1]:.3f}")
            
            # Optimization with Dynamic Range
            optimal_amax = hswq_optimizer.compute_optimal_amax(
                module.weight.data, 
                importance, 
                use_svd_leverage=True, 
                scaled=False, 
                search_range=layer_search_range
            )
            
            weight_amax_dict[name + ".weight"] = optimal_amax
            torch.cuda.empty_cache()

    print(f"Saving quantized model: {args.output}")
    output_state_dict = {}
    
    for stripped_key, value in tqdm(stripped_state_dict.items(), desc="Converting"):
        module_name = stripped_key[:-7] if stripped_key.endswith(".weight") else None
            
        if module_name and module_name in keep_layers:
            new_value = value.to(torch.float16)
        elif stripped_key in weight_amax_dict or (module_name and module_name + ".weight" in weight_amax_dict):
            weight_key = stripped_key if stripped_key in weight_amax_dict else module_name + ".weight"
            amax = max(weight_amax_dict[weight_key], 1e-6)
            new_value = torch.clamp(value.float(), -amax, amax).to(torch.float8_e4m3fn)
            if module_name:
                prefixed_module = detected_prefix + module_name
                output_state_dict[f"{prefixed_module}.comfy_quant"] = torch.tensor(list(json.dumps({"format": "float8_e4m3fn"}).encode("utf-8")), dtype=torch.uint8)
                output_state_dict[f"{prefixed_module}.weight_scale"] = torch.tensor(1.0, dtype=torch.float32)
        else:
            new_value = value.to(torch.float16) if value.dtype == torch.bfloat16 else value
        
        output_state_dict[detected_prefix + stripped_key] = new_value

    save_file(output_state_dict, args.output)
    print("Saved.")

if __name__ == "__main__":
    main()

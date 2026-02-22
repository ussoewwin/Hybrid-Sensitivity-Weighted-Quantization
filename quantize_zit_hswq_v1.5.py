"""
Quantize Z-Image Turbo (ZIT) model to FP8 (HSWQ V1.5: High Precision Tuned).
Implements sensitivity-based protection and importance-weighted optimization per HSWQ spec.
Uses scaled=False (clipping-threshold search only) for standard-loader compatibility.

V1.5 High Precision Edition:
  - Bins: 8192 (High Res Histogram)
  - Candidates: 1000 (Dense Search)
  - Refinement: 10 iterations (Deep Fit)
  - SA2: --sa2 for SageAttention2-accelerated calibration
"""
import argparse
import torch
import torch.nn as nn
import safetensors.torch
from safetensors.torch import load_file, save_file
import os
import gc
from tqdm import tqdm
import sys
import json

# Resolve import paths (avoid ModuleNotFoundError)
current_dir = os.path.dirname(os.path.abspath(__file__))
potential_paths = [current_dir, os.getcwd(), os.path.dirname(sys.argv[0]) if sys.argv[0] else ""]
for p in potential_paths:
    if p and p not in sys.path:
        sys.path.insert(0, p)

# Add ComfyUI-master path
sys.path.insert(0, os.path.join(current_dir, "ComfyUI-master"))

import numpy as np
# HSWQ module
from weighted_histogram_mse import HSWQWeightedHistogramOptimizer

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
    
    if _original_sdpa is not None:
        # Already enabled
        return True
    
    import torch.nn.functional as F
    from sageattention import sageattn
    
    _original_sdpa = F.scaled_dot_product_attention
    
    def sage_sdpa_wrapper(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        # SageAttention2 does not support attn_mask or is_causal directly
        # Fall back to original SDPA if these are used
        if attn_mask is not None or is_causal:
            return _original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
        
        try:
            return sageattn(query, key, value, is_causal=False)
        except Exception:
            return _original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
    
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

# --- ZIT (NextDiT) model load and inference pipeline ---

# Common prefixes to try (order matters: most specific first)
ZIT_PREFIXES = [
    "model.diffusion_model.",  # Third-party (e.g., moodyPornMix)
    "model.",                   # Some ComfyUI exports
    "diffusion_model.",         # Alternative
    "",                         # HSWQ native (no prefix)
]

def detect_and_strip_prefix(state_dict):
    """Detect the key prefix in state_dict and return (stripped_dict, prefix)."""
    keys = list(state_dict.keys())
    
    for prefix in ZIT_PREFIXES:
        if prefix == "":
            # Check if keys start WITHOUT any known prefix
            if any(k.startswith("layers.") or k.startswith("x_embedder") for k in keys):
                return state_dict, ""
        else:
            # Check if keys start WITH this prefix
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
    
    # No prefix detected, return as-is
    print("  [Prefix Detection] No prefix detected (assuming HSWQ format)")
    return state_dict, ""

def detect_zit_config_from_keys(state_dict):
    """Detect ZIT model structure from state_dict keys (assumes prefix already stripped)."""
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
    
    # context_refiner count
    refiner_indices = set()
    for key in state_dict_keys:
        if key.startswith("context_refiner."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                refiner_indices.add(int(parts[1]))
    zit_config["num_context_refiner"] = max(refiner_indices) + 1 if refiner_indices else 2
    
    # noise_refiner count
    noise_indices = set()
    for key in state_dict_keys:
        if key.startswith("noise_refiner."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                noise_indices.add(int(parts[1]))
    zit_config["num_noise_refiner"] = max(noise_indices) + 1 if noise_indices else 2
    
    # Detect FFN dim (intermediate_size)
    for key in state_dict_keys:
        if "feed_forward.w1.weight" in key:
            zit_config["intermediate_size"] = state_dict[key].shape[0]
            break
    
    return zit_config


def resolve_tokenizer_offline(provided_path, comfy_path, clip_path=None):
    """Offline-only logic to find a local tokenizer (ZIT/Qwen3-4B)."""
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

    return None


def load_zit_model(path, device="cuda", comfy_path=None):
    """Load ZIT model and return NextDiT instance.
    
    Returns:
        model: NextDiT instance
        original_state_dict: Original (un-stripped) state dict for output preservation
        stripped_state_dict: Prefix-stripped state dict
        zit_config: Detected model config
        detected_prefix: The prefix that was stripped (for restoring in output)
    """
    print(f"Loading model: {path}")
    original_state_dict = load_file(path)
    
    # === Step 1: Detect and strip prefix ===
    stripped_state_dict, detected_prefix = detect_and_strip_prefix(original_state_dict)
    
    print("Detecting ZIT structure...")
    zit_config = detect_zit_config_from_keys(stripped_state_dict)
    print(f"Detected ZIT config: {zit_config}")
    
    print("Initializing NextDiT model...")
    # Import NextDiT from ComfyUI
    if comfy_path is None:
        # Default: env or current dir ComfyUI
        comfy_path = os.environ.get("COMFYUI_PATH", os.path.join(os.getcwd(), "ComfyUI"))
    
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)
    
    from comfy.ldm.lumina.model import NextDiT
    import comfy.ops
    
    # Get ComfyUI operations
    ops = comfy.ops.disable_weight_init
    
    # Calculate FFN multiplier from detected intermediate_size
    ffn_multiplier = 8/3  # Default
    if zit_config.get("intermediate_size"):
        ffn_multiplier = zit_config["intermediate_size"] / zit_config["hidden_size"]
        print(f"  FFN multiplier: {ffn_multiplier:.4f} (from {zit_config['intermediate_size']}/{zit_config['hidden_size']})")
    
    # Build model (Z-Image Turbo params from BF16 model)
    model = NextDiT(
        patch_size=2,
        in_channels=16,
        dim=zit_config["hidden_size"],  # 3840
        n_layers=zit_config["num_layers"],  # 30
        n_refiner_layers=zit_config["num_context_refiner"],  # 2
        n_heads=zit_config["hidden_size"] // 128,  # 30 for dim=3840
        n_kv_heads=zit_config["hidden_size"] // 128,  # 30 (same as n_heads)
        multiple_of=256,
        ffn_dim_multiplier=ffn_multiplier,
        norm_eps=1e-5,
        cap_feat_dim=2560,  # from cap_embedder.0.weight shape [2560]
        z_image_modulation=True,
        pad_tokens_multiple=64,
        device="cpu",
        dtype=torch.float16,
        operations=ops,
    )
    
    print("Loading ZIT weights...")
    converted_state_dict = {}
    for key, value in stripped_state_dict.items():
        if value.dtype == torch.bfloat16:
            converted_state_dict[key] = value.to(torch.float16)
        else:
            converted_state_dict[key] = value
    
    missing, unexpected = model.load_state_dict(converted_state_dict, strict=False)
    print(f"  [Keys] Matched: {len(converted_state_dict) - len(unexpected)}, Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing and len(missing) <= 5:
        print(f"  Missing: {missing}")
    if unexpected and len(unexpected) <= 5:
        print(f"  Unexpected: {unexpected}")
    
    model = model.to(device).to(torch.float16)
    model.eval()
    
    return model, original_state_dict, stripped_state_dict, zit_config, detected_prefix


class ZITCalibrationPipeline:
    """ZIT calibration pipeline (analogous to SDXL pipeline).
    Uses real prompts through text encoder for calibration (real-data statistics, not random tensors).
    """
    
    def __init__(self, model, text_encoder, tokenizer, device="cuda"):
        self.model = model
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.device = device
        self.hidden_dim = model.dim if hasattr(model, 'dim') else 3072
        
        # Move text encoder to device
        if self.text_encoder is not None:
            self.text_encoder = self.text_encoder.to(device)
            
        # Random seed
        self.prng = np.random.RandomState(42)
        self.sampler_name = "euler" # Default sampler
        
    def encode_prompt(self, prompt):
        """Encode prompt with Qwen3 text encoder."""
        llama_template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        formatted_prompt = llama_template.format(prompt)
        
        # Tokenize
        tokens = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)
        
        # Text encoder embedding
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                intermediate_output=-2
            )
            # Use second-to-last hidden state
            hidden_states = outputs[1]
        
        return hidden_states, attention_mask.bool()
        
    def __call__(self, prompt, num_inference_steps=20, **kwargs):
        """Run calibration inference (real prompts)."""
        batch_size = 1
        latent_h, latent_w = 128, 128
        latent_c = 16  # Model in_channels
        if self.text_encoder is not None:
            cap_feats, cap_mask = self.encode_prompt(prompt)
        else:
            print("Warning: Text encoder not set. Using random tensor.")
            cap_len = 256
            cap_feats = torch.randn(batch_size, cap_len, 2560, # Hidden size
                                   device=self.device, dtype=torch.float16)
            cap_mask = torch.ones(batch_size, cap_len, 
                                 device=self.device, dtype=torch.bool)
        
        # Use ComfyUI's k_diffusion sampler for accurate calibration
        import comfy.k_diffusion.sampling as k_sampling
        
        # Wrapper to adapt NextDiT for k-diffusion sampler
        class ZITWrapper:
            def __init__(self, model, cap_feats, cap_mask):
                self.model = model
                self.cap_feats = cap_feats
                self.cap_mask = cap_mask

            def __call__(self, x, sigma, **kwargs):
                # Ensure inputs are correct dtype for the fp16 model
                dtype = torch.float16
                
                # ZIT expects t in [0, 1]. Input sigma from sample_euler is appropriate.
                # Cast x and sigma/t to fp16 to avoid "Float and Half" mismatch
                x_in = x.to(dtype=dtype)
                t_in = sigma.to(dtype=dtype)

                try:
                    # Ensure features are FP16
                    cap_feats_in = self.cap_feats.to(dtype=dtype)
                    out = self.model(x_in, t_in, cap_feats_in, None, attention_mask=self.cap_mask)
                    if isinstance(out, tuple):
                        out = out[0]
                    # Output needs to be cast back to x.dtype (likely float32) for k-diffusion
                    return out.to(dtype=x.dtype)
                except Exception as e:
                    # Return zeros to allow process to continue (will be noisy data but won't crash)
                    print(f"Wrapper Error: {e}")
                    return torch.zeros_like(x)

        # Initialize latents
        x = torch.randn(batch_size, latent_c, latent_h, latent_w,
                       device=self.device, dtype=torch.float16)

        # Create sigmas for Flow Matching (1.0 -> 0.0)
        # We need num_inference_steps + 1 points to get num_inference_steps intervals
        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=self.device)
        
        model_wrap = ZITWrapper(self.model, cap_feats, cap_mask)
        
        try:
             # Run sampling with progress bar
             sampler_name = getattr(self, "sampler_name", "euler")
             
             sampler_func_name = f"sample_{sampler_name}"
             if not hasattr(k_sampling, sampler_func_name):
                 print(f"Warning: Sampler {sampler_name} not found in k_diffusion. Using sample_euler.")
                 sampler_func = k_sampling.sample_euler
             else:
                 sampler_func = getattr(k_sampling, sampler_func_name)
             
             # Call the dynamic sampler function
             sampler_func(model_wrap, x, sigmas, disable=False)
             
        except Exception as e:
             print(f"Sampling failed: {e}")
             import traceback
             traceback.print_exc()
        
        return {"latent": None}
    
    def set_progress_bar_config(self, disable=False):
        pass  # Compatibility



# --- Dual Monitor: Sensitivity & Importance ---
class DualMonitor:
    def __init__(self):
        # For Sensitivity (Output Variance)
        # Accumulate in FP32/Double to avoid overflow
        self.output_sum = 0.0
        self.output_sq_sum = 0.0
        self.count = 0
        
        # For Importance (Input Activation)
        self.channel_importance = None # [Input_Channels]
    
    def update(self, input_tensor, output_tensor):
        with torch.no_grad():
            # 1. Sensitivity Update (Output Variance)
            # output_tensor: (Batch, Channels, H, W) or (Batch, Tokens, Channels)
            
            out_detached = output_tensor.detach().float()
            batch_mean = out_detached.mean().item()
            batch_sq_mean = (out_detached ** 2).mean().item()
            
            self.output_sum += batch_mean
            self.output_sq_sum += batch_sq_mean
            
            # 2. Importance Update (Input Activation)
            # ZIT V1.1: 2D input support (adaLN_modulation, t_embedder etc.)
            # [RESTORED] Strictly Pure V1.5 L1 Norm (Absolute Mean)
            inp_detached = input_tensor.detach()
            if inp_detached.dim() == 4:     # Conv2d: (B, C, H, W)
                current_imp = inp_detached.abs().mean(dim=(0, 2, 3))  # -> (C,)
            elif inp_detached.dim() == 3:   # Transformer: (B, T, C)
                current_imp = inp_detached.abs().mean(dim=(0, 1))     # -> (C,)
            elif inp_detached.dim() == 2:   # Linear/embedding: (B, C)
                current_imp = inp_detached.abs().mean(dim=0)          # -> (C,)
            else:
                current_imp = torch.ones(1, device=inp_detached.device, dtype=inp_detached.dtype)
                
            if self.channel_importance is None:
                self.channel_importance = current_imp
            else:
                self.channel_importance = (self.channel_importance * self.count + current_imp) / (self.count + 1)
            
            self.count += 1

    def get_sensitivity(self):
        # Variance = E[X^2] - (E[X])^2
        if self.count == 0: return 0.0
        mean = self.output_sum / self.count
        sq_mean = self.output_sq_sum / self.count
        variance = sq_mean - mean ** 2
        return variance

dual_monitors = {}

def hook_fn(module, input, output, name):
    if name not in dual_monitors:
        dual_monitors[name] = DualMonitor()
    
    # input is tuple (tensor, ...)
    inp = input[0]
    out = output
    
    dual_monitors[name].update(inp, out)

# --- HSWQ module: weighted_histogram_mse.HSWQWeightedHistogramOptimizer ---


def main():
    parser = argparse.ArgumentParser(description="ZIT FP8 Quantization (V1.5: Final Restore)")
    parser.add_argument("--input", type=str, required=True, help="Path to input safetensors model")
    parser.add_argument("--output", type=str, required=True, help="Path to output safetensors model")
    parser.add_argument("--calib_file", type=str, required=True, help="Path to calibration prompts text file")
    parser.add_argument("--clip_path", type=str, required=True, help="Path to Qwen3-4B text encoder safetensors")
    parser.add_argument("--comfy_path", type=str, default=None, help="Path to ComfyUI root directory")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer directory")
    parser.add_argument("--num_calib_samples", type=int, default=256, help="Number of calibration samples (HSWQ recommended: 256)")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--keep_ratio", type=float, default=0.25, help="Ratio of layers to keep in FP16 (HSWQ recommended: 0.25 for quality)")
    parser.add_argument("--sampler", type=str, default="euler", help="Sampler name (e.g., euler, dpmpp_2m, heun)")
    parser.add_argument("--sa2", action="store_true", help="Enable SageAttention2 for faster calibration (requires sageattention package)")
    args = parser.parse_args()


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # === V1.5: SageAttention2 Initialization ===
    if args.sa2:
        if try_import_sage_attention():
            enable_sage_attention()
        else:
            print("[Warning] --sa2 specified but SageAttention2 not available. Continuing with standard attention.")

    # Load ZIT NextDiT model
    model, original_state_dict, stripped_state_dict, zit_config, detected_prefix = load_zit_model(args.input, device, args.comfy_path)
    print(f"  Detected prefix: '{detected_prefix}' (will be preserved in output)")
    
    # Load Qwen3-4B text encoder (ComfyUI impl)
    print(f"Loading text encoder: {args.clip_path}")
    text_encoder = None
    tokenizer = None
    
    if os.path.exists(args.clip_path):
        try:
            from comfy.text_encoders import llama as llama_module
            import comfy.ops
            from transformers import Qwen2Tokenizer
            
            # Load tokenizer: Strictly Offline with Discovery
            tokenizer_path = resolve_tokenizer_offline(args.tokenizer_path, args.comfy_path, args.clip_path)
            
            if tokenizer_path:
                print(f"  Loading tokenizer from disk: {tokenizer_path}")
                try:
                    tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
                except Exception as e:
                    print(f"  Warning: Failed to load {tokenizer_path} with local_files_only. Error: {e}")
                    print("  Retrying without local_files_only (Risk of 403)...")
                    tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)
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
            
            # ComfyUI Qwen3_4B (operations=comfy.ops required)
            state_dict = load_file(args.clip_path)
            print("  Loading with ComfyUI Qwen3_4B...")
            text_encoder = llama_module.Qwen3_4B(
                config_dict={}, 
                device=device,
                dtype=torch.float16,
                operations=comfy.ops.disable_weight_init
            )
            # Load weights
            missed, unexpected = text_encoder.load_state_dict(state_dict, strict=False)
            if len(missed) > 0 or len(unexpected) > 0:
                print(f"  Text Encoder load: Missed {len(missed)}, Unexpected {len(unexpected)}")
            
            # eval mode
            if hasattr(text_encoder, "model"):
                text_encoder.model.eval()
            
            print("  Text encoder loaded (ComfyUI Qwen3_4B)")
            
        except Exception as e:
            import traceback
            print(f"Warning: Failed to load text encoder: {e}")
            traceback.print_exc()
            text_encoder = None
    else:
        print(f"Warning: Text encoder file not found: {args.clip_path}")
    
    # Pipeline init (with text encoder)
    pipeline = ZITCalibrationPipeline(model, text_encoder, tokenizer, device)
    pipeline.sampler_name = args.sampler
    print(f"Using sampler: {args.sampler}")

    print("Preparing calibration (registering Dual Monitor hooks)...")
    handles = []
    target_modules = []
    for name, module in model.named_modules():
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
            pipeline(prompt=prompt, num_inference_steps=args.num_inference_steps)
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Remove hooks
    for h in handles: h.remove()

    if args.sa2:
        disable_sage_attention()

    print("\nRunning layer sensitivity analysis...")
    layer_sensitivities = []
    for name in target_modules:
        if name in dual_monitors:
            sensitivity = dual_monitors[name].get_sensitivity()
            layer_sensitivities.append((name, sensitivity))
    
    layer_sensitivities.sort(key=lambda x: x[1], reverse=True)
    num_keep = int(len(layer_sensitivities) * args.keep_ratio)
    keep_layers = set([x[0] for x in layer_sensitivities[:num_keep]])
    
    print(f"Total layers: {len(layer_sensitivities)}")
    print(f"FP16-kept layers: {len(keep_layers)} (Top {args.keep_ratio*100:.1f}%)")
    print("Top 5 Sensitive Layers:")
    for i in range(min(5, len(layer_sensitivities))):
        print(f"  {i+1}. {layer_sensitivities[i][0]}: {layer_sensitivities[i][1]:.4f}")

    print("\n[HSWQ] Starting weighted MSE analysis and quantization parameter computation...")
    print("Compatibility mode (scaled=False): searching optimal clipping threshold...")
    print("※ V1.5 High Precision Mode: bins=8192, candidates=1000, iterations=10")
    weight_amax_dict = {}
    
    # Init HSWQ V1.5 high-precision optimizer (bins=8192, 1000 candidates, 10 refinements)
    hswq_optimizer = HSWQWeightedHistogramOptimizer(
        bins=8192,               # High Res Histogram
        num_candidates=1000,     # Dense Grid
        refinement_iterations=10, # Deep Search
        device=device
    )
    
    for name, module in tqdm(model.named_modules(), desc="Analyzing"):
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if name in keep_layers:
                continue
            importance = None
            if name in dual_monitors:
                importance = dual_monitors[name].channel_importance
            
            # --- SCALED=FALSE (REVERTED) ---
            optimal_amax = hswq_optimizer.compute_optimal_amax(
                module.weight.data,
                importance,
                scaled=False  # Reverted to Unscaled
            )
            weight_amax_dict[name + ".weight"] = optimal_amax
            
            torch.cuda.empty_cache()

    print(f"Layers to quantize: {len(weight_amax_dict)}")
    print(f"Saving quantized model: {args.output}")
    print(f"  Output prefix: '{detected_prefix}' (same as input)")
    output_state_dict = {}
    converted_count = 0
    kept_count = 0
    
    print("Converting weights...")
    # Process using STRIPPED keys (no prefix) for consistency with model.named_modules()
    for stripped_key, value in tqdm(stripped_state_dict.items(), desc="Converting"):
        ## NOTE: No unconditional FP32 protection block here (as requested for "Best Score" setup)
        
        # ZIT: state_dict key == module.name.weight/bias; get module name (strip .weight)
        module_name = None
        if stripped_key.endswith(".weight"):
            module_name = stripped_key[:-7]
            
        if module_name and module_name in keep_layers:
            new_value = value.to(torch.float16)
            kept_count += 1
        elif stripped_key in weight_amax_dict or (module_name and module_name + ".weight" in weight_amax_dict):
            # Quantize
            weight_key = stripped_key if stripped_key in weight_amax_dict else module_name + ".weight"
            
            if weight_key in weight_amax_dict:
                amax = weight_amax_dict[weight_key]
                if amax == 0: amax = 1e-6
                
                # --- UNSCALED LOGIC ---
                # Clamp -> Cast (No normalization)
                clamped_value = torch.clamp(value.float(), -amax, amax)
                new_value = clamped_value.to(torch.float8_e4m3fn)
                converted_count += 1
                
                # Metadata: comfy_quant (required for ZIT/NextDiT ops injection)
                # Use PREFIXED key for output
                if module_name:
                    prefixed_module = detected_prefix + module_name
                    comfy_quant_key = f"{prefixed_module}.comfy_quant"
                    metadata_json = {"format": "float8_e4m3fn"}
                    metadata_bytes = json.dumps(metadata_json).encode("utf-8")
                    metadata_tensor = torch.tensor(list(metadata_bytes), dtype=torch.uint8)
                    output_state_dict[comfy_quant_key] = metadata_tensor

                    # Unity Scale (1.0)
                    scale_key = f"{prefixed_module}.weight_scale"
                    output_state_dict[scale_key] = torch.tensor(1.0, dtype=torch.float32)

            else:
                new_value = value.to(torch.float16) if value.dtype == torch.bfloat16 else value
        else:
            new_value = value.to(torch.float16) if value.dtype == torch.bfloat16 else value
        
        # Output key with original prefix restored
        output_key = detected_prefix + stripped_key
        output_state_dict[output_key] = new_value

    print("Conversion done:")
    print(f"  FP8 layers: {converted_count}")
    print(f"  FP16-kept layers: {kept_count}")
    save_file(output_state_dict, args.output)
    print("Save complete.")

if __name__ == "__main__":
    main()
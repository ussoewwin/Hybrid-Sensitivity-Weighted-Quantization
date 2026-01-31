"""
Z-Image Turbo (ZIT) FP8 quantization script (HSWQ V1: Standard Compatible Mode).
HSWQ spec: Sensitivity protection and Importance-weighted MSE. No scaling (scaled=False).
ZIT V1.1: DualMonitor 2D input support (adaLN_modulation, t_embedder etc. (B, C) shape).

Changelog: V13 Dual Monitor; V15 weighted histogram MSE; HSWQ V1 scaled=False; ZIT NextDiT; ZIT V1.1 2D input.

Algorithm: Calibration (Sensitivity + Importance). Layer selection (top N% FP16 keep). Quantization: others -> weighted histogram, amax with scaled=False, clip and cast.
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
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "ComfyUI-master"))
import numpy as np
import comfy.ops

# HSWQ module
from weighted_histogram_mse import HSWQWeightedHistogramOptimizer

# Enforce C++20
if sys.platform == "win32":
    os.environ.setdefault("CXXFLAGS", "/std:c++20")
else:
    os.environ.setdefault("CXXFLAGS", "-std=c++20")

# --- ZIT (NextDiT) model load and inference pipeline ---
def detect_zit_config_from_keys(state_dict):
    """Detect ZIT model structure from state_dict keys."""
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
    
    return zit_config


def load_zit_model(path, device="cuda", comfy_path=None):
    """Load ZIT model and return NextDiT instance."""
    print(f"Loading model: {path}")
    state_dict = load_file(path)
    
    print("Detecting ZIT structure...")
    zit_config = detect_zit_config_from_keys(state_dict)
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
        ffn_dim_multiplier=8/3,  # 10240/3840 ≈ 2.6667
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
    for key, value in state_dict.items():
        if value.dtype == torch.bfloat16:
            converted_state_dict[key] = value.to(torch.float16)
        else:
            converted_state_dict[key] = value
    
    missing, unexpected = model.load_state_dict(converted_state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    
    model = model.to(device).to(torch.float16)
    model.eval()
    
    return model, state_dict, zit_config


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
        
        # Random timestep
        t_idx = self.prng.randint(0, 1000)
        t = torch.tensor([t_idx / 1000.0], device=self.device, dtype=torch.float16)
        x = torch.randn(batch_size, latent_c, latent_h, latent_w,
                       device=self.device, dtype=torch.float16)
        
        with torch.no_grad():
            try:
                # NextDiT.forward(x, timesteps, context, num_tokens, attention_mask=None, ...)
                # num_tokens=None at inference (all tokens)
                self.model(
                    x, 
                    t, # timesteps
                    cap_feats, # context
                    None, # num_tokens
                    attention_mask=cap_mask
                )
            except Exception as e:
                print(f"Warning: Forward pass failed: {e}")
                import traceback
                traceback.print_exc()
                pass
        
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
    parser = argparse.ArgumentParser(description="ZIT FP8 Quantization (HSWQ V1: Full Weighted MSE Optimization)")
    parser.add_argument("--input", type=str, required=True, help="Path to input safetensors model")
    parser.add_argument("--output", type=str, required=True, help="Path to output safetensors model")
    parser.add_argument("--calib_file", type=str, required=True, help="Path to calibration prompts text file")
    parser.add_argument("--clip_path", type=str, required=True, help="Path to Qwen3-4B text encoder safetensors")
    parser.add_argument("--comfy_path", type=str, default=None, help="Path to ComfyUI root directory")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer directory")
    parser.add_argument("--num_calib_samples", type=int, default=256, help="Number of calibration samples (HSWQ recommended: 256)")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--keep_ratio", type=float, default=0.25, help="Ratio of layers to keep in FP16 (HSWQ recommended: 0.25 for quality)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load ZIT NextDiT model
    model, original_state_dict, zit_config = load_zit_model(args.input, device, args.comfy_path)
    
    # Load Qwen3-4B text encoder (ComfyUI impl)
    print(f"Loading text encoder: {args.clip_path}")
    text_encoder = None
    tokenizer = None
    
    if os.path.exists(args.clip_path):
        try:
            from comfy.text_encoders import llama as llama_module
            from transformers import Qwen2Tokenizer
            
            # Load tokenizer
            tokenizer_path = args.tokenizer_path
            if tokenizer_path and os.path.exists(tokenizer_path):
                tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)
                print(f"  Loading tokenizer: {tokenizer_path}")
            else:
                print(f"  Warning: Tokenizer path not set or missing: {tokenizer_path}")
            
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

    print("Preparing calibration (Dual Monitor hooks)...")
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
    print("Measuring Sensitivity and Input Importance...")
    
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

    print("\nAnalyzing layer sensitivity...")
    layer_sensitivities = []
    for name in target_modules:
        if name in dual_monitors:
            sensitivity = dual_monitors[name].get_sensitivity()
            layer_sensitivities.append((name, sensitivity))
    
    layer_sensitivities.sort(key=lambda x: x[1], reverse=True)
    num_keep = int(len(layer_sensitivities) * args.keep_ratio)
    keep_layers = set([x[0] for x in layer_sensitivities[:num_keep]])
    
    print(f"Total layers: {len(layer_sensitivities)}")
    print(f"FP16 keep layers: {len(keep_layers)} (Top {args.keep_ratio*100:.1f}%)")
    print("Top 5 Sensitive Layers:")
    for i in range(min(5, len(layer_sensitivities))):
        print(f"  {i+1}. {layer_sensitivities[i][0]}: {layer_sensitivities[i][1]:.4f}")

    print("\n[HSWQ] Starting weighted MSE optimization...")
    print("Compatible mode (scaled=False): finding optimal clipping threshold...")
    weight_amax_dict = {}
    
    hswq_optimizer = HSWQWeightedHistogramOptimizer(
        bins=4096,
        num_candidates=200,
        refinement_iterations=3,
        device=device
    )
    
    for name, module in tqdm(model.named_modules(), desc="Analyzing"):
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if name in keep_layers:
                continue
            importance = None
            if name in dual_monitors:
                importance = dual_monitors[name].channel_importance
            optimal_amax = hswq_optimizer.compute_optimal_amax(
                module.weight.data,
                importance,
                scaled=False  # Compatible mode
            )
            weight_amax_dict[name + ".weight"] = optimal_amax
            
            torch.cuda.empty_cache()

    print(f"Layers to quantize: {len(weight_amax_dict)}")
    print(f"Saving quantized model: {args.output}")
    output_state_dict = {}
    converted_count = 0
    kept_count = 0
    
    print("Converting weights...")
    for key, value in tqdm(original_state_dict.items(), desc="Converting"):
        # ZIT: state_dict key == module.name.weight/bias; get module name (strip .weight)
        module_name = None
        if key.endswith(".weight"):
            module_name = key[:-7]
            
        if module_name and module_name in keep_layers:
            new_value = value.to(torch.float16)
            kept_count += 1
        elif key in weight_amax_dict or (module_name and module_name + ".weight" in weight_amax_dict):
            # Quantize
            weight_key = key if key in weight_amax_dict else module_name + ".weight"
            
            if weight_key in weight_amax_dict:
                amax = weight_amax_dict[weight_key]
                clamped_value = torch.clamp(value.float(), -amax, amax)
                new_value = clamped_value.to(torch.float8_e4m3fn)
                converted_count += 1
            else:
                new_value = value.to(torch.float16) if value.dtype == torch.bfloat16 else value
        else:
            new_value = value.to(torch.float16) if value.dtype == torch.bfloat16 else value
            
        output_state_dict[key] = new_value

    print(f"Done:")
    print(f"  FP8 layers: {converted_count}")
    print(f"  FP16 kept layers: {kept_count}")
    save_file(output_state_dict, args.output)
    print("Saved.")

if __name__ == "__main__":
    main()

"""
Quantize SDXL to FP8 (HSWQ V2: Scaled High-Performance Mode).

Full HSWQ spec: scaling quantization (x * 448/amax) to use FP8 dynamic range.
Output model requires a dedicated loader or ComfyUI node; standard loader will produce broken images.

Features:
- MSE optimization with scaling (scaled=True)
- Save scale in .scale key
- Better precision for small values

Algorithm:
1. Calibration & Layer Selection (same as V1)
2. Quantization (V2): importance-weighted histogram, find amax with scaled=True,
   apply value_scaled = value * (448.0 / amax), cast to float8_e4m3fn, save inv_scale = amax/448 as .scale
"""
import argparse
import torch
from diffusers import StableDiffusionXLPipeline
import safetensors.torch
from safetensors.torch import load_file, save_file
import os
import gc
from tqdm import tqdm
import sys
import numpy as np

# HSWQ module
from weighted_histogram_mse import HSWQWeightedHistogramOptimizer

# Enforce C++20
if sys.platform == "win32":
    os.environ.setdefault("CXXFLAGS", "/std:c++20")
else:
    os.environ.setdefault("CXXFLAGS", "-std=c++20")

# --- ComfyUI互換のマッピング関数群 (V1と同じ) ---
def count_blocks(state_dict_keys, prefix_string):
    count = 0
    while True:
        c = False
        for k in state_dict_keys:
            if k.startswith(prefix_string.format(count)):
                c = True
                break
        if c == False:
            break
        count += 1
    return count

def calculate_transformer_depth(prefix, state_dict_keys, state_dict):
    transformer_prefix = prefix + "1.transformer_blocks."
    transformer_keys = sorted(list(filter(lambda a: a.startswith(transformer_prefix), state_dict_keys)))
    if len(transformer_keys) > 0:
        last_transformer_depth = count_blocks(state_dict_keys, transformer_prefix + '{}')
        return last_transformer_depth
    return 0

def detect_unet_config_from_keys(state_dict, key_prefix="model.diffusion_model."):
    state_dict_keys = list(state_dict.keys())
    filtered_keys = [k for k in state_dict_keys if k.startswith(key_prefix)]
    unet_config = {}
    if f"{key_prefix}input_blocks.0.0.weight" in state_dict_keys:
        model_channels = state_dict[f"{key_prefix}input_blocks.0.0.weight"].shape[0]
        num_res_blocks = []
        channel_mult = []
        transformer_depth = []
        transformer_depth_output = []
        input_block_count = count_blocks(state_dict_keys, f"{key_prefix}input_blocks" + '.{}.')
        last_res_blocks = 0
        last_channel_mult = 0
        for count in range(input_block_count):
            prefix = f"{key_prefix}input_blocks.{count}."
            prefix_output = f"{key_prefix}output_blocks.{input_block_count - count - 1}."
            block_keys = sorted(list(filter(lambda a: a.startswith(prefix), state_dict_keys)))
            if len(block_keys) == 0: break
            block_keys_output = sorted(list(filter(lambda a: a.startswith(prefix_output), state_dict_keys)))
            if f"{prefix}0.op.weight" in block_keys:
                num_res_blocks.append(last_res_blocks)
                channel_mult.append(last_channel_mult)
                last_res_blocks = 0
                last_channel_mult = 0
                out = calculate_transformer_depth(prefix_output, state_dict_keys, state_dict)
                transformer_depth_output.append(out)
            else:
                res_block_prefix = f"{prefix}0.in_layers.0.weight"
                if res_block_prefix in block_keys:
                    last_res_blocks += 1
                    last_channel_mult = state_dict[f"{prefix}0.out_layers.3.weight"].shape[0] // model_channels
                    out = calculate_transformer_depth(prefix, state_dict_keys, state_dict)
                    transformer_depth.append(out)
                res_block_prefix = f"{prefix_output}0.in_layers.0.weight"
                if res_block_prefix in block_keys_output:
                    out = calculate_transformer_depth(prefix_output, state_dict_keys, state_dict)
                    transformer_depth_output.append(out)
        num_res_blocks.append(last_res_blocks)
        channel_mult.append(last_channel_mult)
        if f"{key_prefix}middle_block.1.proj_in.weight" in state_dict_keys:
            transformer_depth_middle = count_blocks(state_dict_keys, f"{key_prefix}middle_block.1.transformer_blocks." + '{}')
        elif f"{key_prefix}middle_block.0.in_layers.0.weight" in state_dict_keys:
            transformer_depth_middle = -1
        else:
            transformer_depth_middle = -2
        unet_config["num_res_blocks"] = num_res_blocks
        unet_config["channel_mult"] = channel_mult
        unet_config["transformer_depth"] = transformer_depth
        unet_config["transformer_depth_output"] = transformer_depth_output
        unet_config["transformer_depth_middle"] = transformer_depth_middle
    return unet_config

def unet_to_diffusers_mapping(unet_config, state_dict=None, key_prefix="model.diffusion_model."):
    if "num_res_blocks" not in unet_config: return {}
    num_res_blocks = unet_config["num_res_blocks"]
    channel_mult = unet_config["channel_mult"]
    num_blocks = len(channel_mult)
    if state_dict is not None:
        import re
        state_dict_keys = list(state_dict.keys())
        filtered_keys = [k.replace(key_prefix, "") for k in state_dict_keys if k.startswith(key_prefix)]
        transformer_counts = {}
        for key in filtered_keys:
            match = re.match(r'input_blocks\.(\d+)\.1\.transformer_blocks\.(\d+)', key)
            if match:
                block_idx = int(match.group(1))
                trans_idx = int(match.group(2))
                if block_idx not in transformer_counts: transformer_counts[block_idx] = 0
                transformer_counts[block_idx] = max(transformer_counts[block_idx], trans_idx + 1)
        output_transformer_counts = {}
        for key in filtered_keys:
            match = re.match(r'output_blocks\.(\d+)\.1\.transformer_blocks\.(\d+)', key)
            if match:
                block_idx = int(match.group(1))
                trans_idx = int(match.group(2))
                if block_idx not in output_transformer_counts: output_transformer_counts[block_idx] = 0
                output_transformer_counts[block_idx] = max(output_transformer_counts[block_idx], trans_idx + 1)
        middle_transformer_count = 0
        for key in filtered_keys:
            match = re.match(r'middle_block\.1\.transformer_blocks\.(\d+)', key)
            if match:
                trans_idx = int(match.group(1))
                middle_transformer_count = max(middle_transformer_count, trans_idx + 1)
        transformers_mid = middle_transformer_count if middle_transformer_count > 0 else unet_config.get("transformer_depth_middle", None)
    else:
        transformer_depth = unet_config["transformer_depth"][:]
        transformer_depth_output = unet_config["transformer_depth_output"][:]
        transformers_mid = unet_config.get("transformer_depth_middle", None)
        transformer_counts = None
        output_transformer_counts = None
    UNET_MAP_RESNET = {"in_layers.2.weight": "conv1.weight", "in_layers.2.bias": "conv1.bias", "emb_layers.1.weight": "time_emb_proj.weight", "emb_layers.1.bias": "time_emb_proj.bias", "out_layers.3.weight": "conv2.weight", "out_layers.3.bias": "conv2.bias", "skip_connection.weight": "conv_shortcut.weight", "skip_connection.bias": "conv_shortcut.bias", "in_layers.0.weight": "norm1.weight", "in_layers.0.bias": "norm1.bias", "out_layers.0.weight": "norm2.weight", "out_layers.0.bias": "norm2.bias"}
    UNET_MAP_ATTENTIONS = {"proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias", "norm.weight", "norm.bias"}
    TRANSFORMER_BLOCKS = {"norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "norm3.weight", "norm3.bias", "attn1.to_q.weight", "attn1.to_q.bias", "attn1.to_k.weight", "attn1.to_k.bias", "attn1.to_v.weight", "attn1.to_out.0.weight", "attn1.to_out.0.bias", "attn2.to_q.weight", "attn2.to_k.weight", "attn2.to_v.weight", "attn2.to_out.0.weight", "attn2.to_out.0.bias", "ff.net.0.proj.weight", "ff.net.0.proj.bias", "ff.net.2.weight", "ff.net.2.bias"}
    UNET_MAP_BASIC = {("label_emb.0.0.weight", "add_embedding.linear_1.weight"), ("label_emb.0.0.bias", "add_embedding.linear_1.bias"), ("label_emb.0.2.weight", "add_embedding.linear_2.weight"), ("label_emb.0.2.bias", "add_embedding.linear_2.bias"), ("input_blocks.0.0.weight", "conv_in.weight"), ("input_blocks.0.0.bias", "conv_in.bias"), ("out.0.weight", "conv_norm_out.weight"), ("out.0.bias", "conv_norm_out.bias"), ("out.2.weight", "conv_out.weight"), ("out.2.bias", "conv_out.bias"), ("time_embed.0.weight", "time_embedding.linear_1.weight"), ("time_embed.0.bias", "time_embedding.linear_1.bias"), ("time_embed.2.weight", "time_embedding.linear_2.weight"), ("time_embed.2.bias", "time_embedding.linear_2.bias")}
    diffusers_unet_map = {}
    for x in range(num_blocks):
        n = 1 + (num_res_blocks[x] + 1) * x
        for i in range(num_res_blocks[x]):
            for b in UNET_MAP_RESNET: diffusers_unet_map["down_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])] = "input_blocks.{}.0.{}".format(n, b)
            if transformer_counts is not None: num_transformers = transformer_counts.get(n, 0)
            else: num_transformers = transformer_depth.pop(0) if transformer_depth else 0
            if num_transformers > 0:
                for b in UNET_MAP_ATTENTIONS: diffusers_unet_map["down_blocks.{}.attentions.{}.{}".format(x, i, b)] = "input_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS: diffusers_unet_map["down_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b)] = "input_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
            n += 1
        for k in ["weight", "bias"]: diffusers_unet_map["down_blocks.{}.downsamplers.0.conv.{}".format(x, k)] = "input_blocks.{}.0.op.{}".format(n, k)
    i = 0
    for b in UNET_MAP_ATTENTIONS: diffusers_unet_map["mid_block.attentions.{}.{}".format(i, b)] = "middle_block.1.{}".format(b)
    if transformers_mid:
        for t in range(transformers_mid):
            for b in TRANSFORMER_BLOCKS: diffusers_unet_map["mid_block.attentions.{}.transformer_blocks.{}.{}".format(i, t, b)] = "middle_block.1.transformer_blocks.{}.{}".format(t, b)
    for i, n in enumerate([0, 2]):
        for b in UNET_MAP_RESNET: diffusers_unet_map["mid_block.resnets.{}.{}".format(i, UNET_MAP_RESNET[b])] = "middle_block.{}.{}".format(n, b)
    num_res_blocks_rev = list(reversed(num_res_blocks))
    for x in range(num_blocks):
        n = (num_res_blocks_rev[x] + 1) * x
        l = num_res_blocks_rev[x] + 1
        for i in range(l):
            c = 0
            for b in UNET_MAP_RESNET: diffusers_unet_map["up_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])] = "output_blocks.{}.0.{}".format(n, b)
            c += 1
            if output_transformer_counts is not None: num_transformers = output_transformer_counts.get(n, 0)
            else: num_transformers = transformer_depth_output.pop() if transformer_depth_output else 0
            if num_transformers > 0:
                c += 1
                for b in UNET_MAP_ATTENTIONS: diffusers_unet_map["up_blocks.{}.attentions.{}.{}".format(x, i, b)] = "output_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS: diffusers_unet_map["up_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b)] = "output_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
            n += 1
        for k in ["weight", "bias"]: diffusers_unet_map["up_blocks.{}.upsamplers.0.conv.{}".format(x, k)] = "output_blocks.{}.2.conv.{}".format(n, k)
    for k, v in UNET_MAP_BASIC: diffusers_unet_map[v] = k
    comfyui_to_diffusers_map = {v: k for k, v in diffusers_unet_map.items()}
    comfyui_to_diffusers_map = {f"{key_prefix}{k}": v for k, v in comfyui_to_diffusers_map.items()}
    return comfyui_to_diffusers_map

def load_unet_from_safetensors(path, device="cuda"):
    print(f"Loading model: {path}")
    state_dict = load_file(path)
    print("Detecting UNet structure...")
    unet_config = detect_unet_config_from_keys(state_dict)
    print(f"Detected UNet config: {unet_config}")
    print("Initializing Diffusers pipeline...")
    try:
        pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(device)
    except Exception as e:
        print(f"Warning: Failed to load pretrained model: {e}")
        from diffusers import UNet2DConditionModel
        unet = UNet2DConditionModel(sample_size=128, in_channels=4, out_channels=4, layers_per_block=2, block_out_channels=(320, 640, 1280), down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"), up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"))
        pipeline = StableDiffusionXLPipeline(vae=None, text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None, unet=unet, scheduler=None)
    print("キーマッピングを作成中...")
    comfyui_to_diffusers_map = unet_to_diffusers_mapping(unet_config, state_dict)
    print("Loading UNet weights...")
    new_state_dict = {}
    for comfy_key, diffusers_key in comfyui_to_diffusers_map.items():
        if comfy_key in state_dict: new_state_dict[diffusers_key] = state_dict[comfy_key]
    m, u = pipeline.unet.load_state_dict(new_state_dict, strict=False)
    return pipeline, state_dict, comfyui_to_diffusers_map

# --- Dual Monitor: Sensitivity & Importance (same as V1) ---
class DualMonitor:
    def __init__(self):
        self.output_sum = 0.0
        self.output_sq_sum = 0.0
        self.count = 0
        self.channel_importance = None 
    
    def update(self, input_tensor, output_tensor):
        with torch.no_grad():
            out_detached = output_tensor.detach().float()
            batch_mean = out_detached.mean().item()
            batch_sq_mean = (out_detached ** 2).mean().item()
            self.output_sum += batch_mean
            self.output_sq_sum += batch_sq_mean
            
            inp_detached = input_tensor.detach()
            if inp_detached.dim() == 4:
                current_imp = inp_detached.abs().mean(dim=(0, 2, 3))
            elif inp_detached.dim() == 3:
                current_imp = inp_detached.abs().mean(dim=(0, 1))
            else:
                current_imp = inp_detached.abs().mean()
                
            if self.channel_importance is None:
                self.channel_importance = current_imp
            else:
                self.channel_importance = (self.channel_importance * self.count + current_imp) / (self.count + 1)
            self.count += 1

    def get_sensitivity(self):
        if self.count == 0: return 0.0
        mean = self.output_sum / self.count
        sq_mean = self.output_sq_sum / self.count
        variance = sq_mean - mean ** 2
        return variance

dual_monitors = {}

def hook_fn(module, input, output, name):
    if name not in dual_monitors:
        dual_monitors[name] = DualMonitor()
    inp = input[0]
    out = output
    dual_monitors[name].update(inp, out)

def main():
    parser = argparse.ArgumentParser(description="SDXL FP8 Quantization (HSWQ V2: Scaled High-Performance Mode)")
    parser.add_argument("--input", type=str, required=True, help="Path to input safetensors model")
    parser.add_argument("--output", type=str, required=True, help="Path to output safetensors model")
    parser.add_argument("--calib_file", type=str, required=True, help="Path to calibration prompts text file")
    parser.add_argument("--num_calib_samples", type=int, default=256, help="Number of calibration samples")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--keep_ratio", type=float, default=0.25, help="Ratio of layers to keep in FP16")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    pipeline, original_state_dict, comfyui_to_diffusers_map = load_unet_from_safetensors(args.input, device)

    print("Callibrating (Dual Monitor)...")
    handles = []
    target_modules = []
    for name, module in pipeline.unet.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handle = module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))
            handles.append(handle)
            target_modules.append(name)

    print("Loading calibration data...")
    with open(args.calib_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    if len(prompts) < args.num_calib_samples:
        prompts = (prompts * (args.num_calib_samples // len(prompts) + 1))[:args.num_calib_samples]
    else:
        prompts = prompts[:args.num_calib_samples]

    print(f"Executing Calibration ({args.num_calib_samples} samples)...")
    pipeline.set_progress_bar_config(disable=False)
    
    for i, prompt in enumerate(prompts):
        print(f"\nSample {i+1}/{args.num_calib_samples}...")
        with torch.no_grad():
            pipeline(prompt=prompt, num_inference_steps=args.num_inference_steps, output_type="latent")
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    for h in handles: h.remove()

    print("\nAnalyzing Sensitivity...")
    layer_sensitivities = []
    for name in target_modules:
        if name in dual_monitors:
            sensitivity = dual_monitors[name].get_sensitivity()
            layer_sensitivities.append((name, sensitivity))
    
    layer_sensitivities.sort(key=lambda x: x[1], reverse=True)
    num_keep = int(len(layer_sensitivities) * args.keep_ratio)
    keep_layers = set([x[0] for x in layer_sensitivities[:num_keep]])
    
    print(f"Total Layers: {len(layer_sensitivities)}")
    print(f"FP16 Keep Layers: {len(keep_layers)} (Top {args.keep_ratio*100:.1f}%)")

    print("\n[HSWQ V2] Starting Scaled Weighted MSE Optimization...")
    print("※ SCALED MODE: Computing optimal amax and applying scaling (WARNING: Requires Custom Loader)")
    
    weight_amax_dict = {}
    hswq_optimizer = HSWQWeightedHistogramOptimizer(
        bins=4096,
        num_candidates=200,
        refinement_iterations=3,
        device=device
    )
    
    # 1. Optimization phase
    for name, module in tqdm(pipeline.unet.named_modules(), desc="Analyzing"):
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if name in keep_layers: continue
            
            importance = None
            if name in dual_monitors:
                importance = dual_monitors[name].channel_importance
            
            # V2: scaled=True
            optimal_amax = hswq_optimizer.compute_optimal_amax(
                module.weight.data, 
                importance,
                scaled=True
            )
            weight_amax_dict[name + ".weight"] = optimal_amax
            torch.cuda.empty_cache()

    # 2. 変換・保存フェーズ
    print(f"Quantizing and Saving to {args.output}")
    output_state_dict = {}
    converted_count = 0
    kept_count = 0
    
    for key, value in tqdm(original_state_dict.items(), desc="Converting"):
        diffusers_key = None
        if key in comfyui_to_diffusers_map: diffusers_key = comfyui_to_diffusers_map[key]
        elif key.startswith("model.diffusion_model."):
            if key in comfyui_to_diffusers_map: diffusers_key = comfyui_to_diffusers_map[key]
        
        module_name = None
        if diffusers_key:
            if diffusers_key.endswith(".weight"):
                module_name = diffusers_key[:-7]
            
        if module_name and module_name in keep_layers:
            output_state_dict[key] = value
            kept_count += 1
        elif diffusers_key:
            weight_key = diffusers_key + ".weight"
            if diffusers_key.endswith(".weight"): weight_key = diffusers_key
            
            if weight_key in weight_amax_dict:
                amax = weight_amax_dict[weight_key]
                
                # Apply scaling: x_scaled = x * (448 / amax)
                scale_val = 448.0 / amax
                scaled_value = torch.clamp(value * scale_val, -448.0, 448.0)
                new_value = scaled_value.to(torch.float8_e4m3fn)
                
                output_state_dict[key] = new_value
                
                # Save scale: inv_scale = amax / 448.0, key suffix ".scale"
                scale_key = key
                if key.endswith(".weight"):
                    scale_key = key.replace(".weight", ".scale")
                else:
                    scale_key = key + ".scale"
                    
                inv_scale = amax / 448.0
                output_state_dict[scale_key] = torch.tensor([inv_scale], dtype=torch.float32)
                
                converted_count += 1
            else:
                output_state_dict[key] = value
        else:
            output_state_dict[key] = value

    print(f"Done:")
    print(f"  FP8 Scaled Layers: {converted_count}")
    print(f"  FP16 Kept Layers: {kept_count}")
    save_file(output_state_dict, args.output)
    print("Verification: Saved .scale keys for custom loader.")

if __name__ == "__main__":
    main()

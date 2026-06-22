"""
Quantize SDXL model to FP8 (HSWQ V1.4: V1.3 + VETO + auto-protect + SDXL search_range).

Changelog from V1.3:
- [NEW] Hard VETO: embedding/boundary layers always FP16 (on top of keep_ratio).
- [NEW] Auto-protect: after computing optimal amax for all layers, measures actual
        quantization MSE per layer. Layers with MSE above the auto-detected threshold
        (Tukey fence: Q3 + 1.5*IQR) are promoted to FP16 automatically.
        No manual threshold needed. Adapts to each model.
- [NEW] SDXL search_range=(0.99, 1.0): optimal for uniform weight distribution.
- keep_ratio: unchanged from V1.3.
"""

import argparse
import random
import torch
from diffusers import StableDiffusionXLPipeline
import safetensors.torch
from safetensors.torch import load_file, save_file
import os
import gc
from tqdm import tqdm
import sys
import re
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
histogram_dir = os.path.join(current_dir, "histogram")
if histogram_dir not in sys.path:
    sys.path.insert(0, histogram_dir)

from weighted_histogram_mse_fast import (
    HSWQWeightedHistogramOptimizerFast as HSWQWeightedHistogramOptimizer,
    WeightedHistogramOptimized,
)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

if sys.platform == "win32":
    os.environ.setdefault("CXXFLAGS", "/std:c++20")
else:
    os.environ.setdefault("CXXFLAGS", "-std=c++20")


# ---------------------------------------------------------------------------
# Hard VETO (always FP16)
# ---------------------------------------------------------------------------
_VETO_PREFIXES = (
    "time_embedding.",
    "add_embedding.",
)
_VETO_EXACT = {"conv_in", "conv_out"}

def is_hard_veto(name: str) -> bool:
    if any(name.startswith(p) for p in _VETO_PREFIXES):
        return True
    return name in _VETO_EXACT


# ---------------------------------------------------------------------------
# Auto-protect: Tukey fence on per-layer quantization MSE
# ---------------------------------------------------------------------------
def compute_auto_protect_threshold(mse_values: list[float]) -> float:
    """
    Tukey fence: Q3 + 1.5 * IQR.
    Layers with MSE above this are statistical outliers = damaged by FP8.
    """
    if not mse_values:
        return float('inf')
    arr = np.array(sorted(mse_values))
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr
    return float(threshold)


# ---------------------------------------------------------------------------
# ComfyUI mapping (unchanged from V1.3)
# ---------------------------------------------------------------------------
def count_blocks(state_dict_keys, prefix_string):
    count = 0
    while True:
        c = False
        for k in state_dict_keys:
            if k.startswith(prefix_string.format(count)):
                c = True
                break
        if not c:
            break
        count += 1
    return count

def calculate_transformer_depth(prefix, state_dict_keys, state_dict):
    transformer_prefix = prefix + "1.transformer_blocks."
    transformer_keys = sorted(list(filter(lambda a: a.startswith(transformer_prefix), state_dict_keys)))
    if len(transformer_keys) > 0:
        return count_blocks(state_dict_keys, transformer_prefix + '{}')
    return 0

def detect_unet_config_from_keys(state_dict):
    sd_keys = list(state_dict.keys())
    unet_key_prefix = "model.diffusion_model."
    filtered_keys = [k for k in sd_keys if k.startswith(unet_key_prefix)]
    if not filtered_keys:
        unet_key_prefix = ""
        filtered_keys = sd_keys
    num_res_blocks = []
    channel_mult = []
    transformer_depth = []
    num_blocks = count_blocks(filtered_keys, unet_key_prefix + "input_blocks.{}")
    for i in range(1, num_blocks):
        block_keys = [k for k in filtered_keys if k.startswith(unet_key_prefix + f"input_blocks.{i}.")]
        has_resnet = any(".in_layers." in k for k in block_keys)
        has_transformer = any(".transformer_blocks." in k for k in block_keys)
        if has_resnet:
            if not channel_mult:
                channel_mult.append(1)
                num_res_blocks.append(0)
                transformer_depth.append(0)
            num_res_blocks[-1] += 1
            td = calculate_transformer_depth(unet_key_prefix + f"input_blocks.{i}.", filtered_keys, state_dict)
            if has_transformer:
                transformer_depth[-1] = td
        else:
            channel_mult.append(channel_mult[-1] * 2 if channel_mult else 1)
            num_res_blocks.append(0)
            transformer_depth.append(0)
    transformer_counts = {}
    output_transformer_counts = {}
    for key in filtered_keys:
        match = re.match(r'(?:model\.diffusion_model\.)?input_blocks\.(\d+)\.1\.transformer_blocks\.(\d+)', key)
        if match:
            b, t = int(match.group(1)), int(match.group(2))
            transformer_counts[b] = max(transformer_counts.get(b, 0), t + 1)
        match = re.match(r'(?:model\.diffusion_model\.)?output_blocks\.(\d+)\.1\.transformer_blocks\.(\d+)', key)
        if match:
            b, t = int(match.group(1)), int(match.group(2))
            output_transformer_counts[b] = max(output_transformer_counts.get(b, 0), t + 1)
    middle_transformer_count = 0
    for key in filtered_keys:
        match = re.match(r'(?:model\.diffusion_model\.)?middle_block\.1\.transformer_blocks\.(\d+)', key)
        if match:
            middle_transformer_count = max(middle_transformer_count, int(match.group(1)) + 1)
    return {
        "num_res_blocks": num_res_blocks,
        "channel_mult": channel_mult,
        "transformer_depth": transformer_depth,
        "transformer_depth_output": list(reversed(transformer_depth)),
        "transformer_depth_middle": middle_transformer_count,
    }

def unet_to_diffusers_mapping(unet_config, state_dict, key_prefix="model.diffusion_model."):
    num_res_blocks = unet_config["num_res_blocks"]
    num_blocks = len(num_res_blocks)
    transformer_depth = unet_config["transformer_depth"][:]
    transformer_depth_output = unet_config["transformer_depth_output"][:]
    transformers_mid = unet_config.get("transformer_depth_middle", None)
    sd_keys = list(state_dict.keys())
    filtered_keys = [k for k in sd_keys if k.startswith(key_prefix)]
    transformer_counts = {}
    output_transformer_counts = {}
    if filtered_keys:
        for key in filtered_keys:
            match = re.match(r'model\.diffusion_model\.input_blocks\.(\d+)\.1\.transformer_blocks\.(\d+)', key)
            if match:
                b, t = int(match.group(1)), int(match.group(2))
                transformer_counts[b] = max(transformer_counts.get(b, 0), t + 1)
            match = re.match(r'model\.diffusion_model\.output_blocks\.(\d+)\.1\.transformer_blocks\.(\d+)', key)
            if match:
                b, t = int(match.group(1)), int(match.group(2))
                output_transformer_counts[b] = max(output_transformer_counts.get(b, 0), t + 1)
        mc = 0
        for key in filtered_keys:
            match = re.match(r'model\.diffusion_model\.middle_block\.1\.transformer_blocks\.(\d+)', key)
            if match:
                mc = max(mc, int(match.group(1)) + 1)
        if mc > 0:
            transformers_mid = mc
    else:
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
            num_transformers = transformer_counts.get(n, 0) if transformer_counts is not None else (transformer_depth.pop(0) if transformer_depth else 0)
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
            for b in UNET_MAP_RESNET: diffusers_unet_map["up_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])] = "output_blocks.{}.0.{}".format(n, b)
            num_transformers = output_transformer_counts.get(n, 0) if output_transformer_counts is not None else (transformer_depth_output.pop() if transformer_depth_output else 0)
            if num_transformers > 0:
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
        print(f"Warning: failed to load pretrained model: {e}")
        from diffusers import UNet2DConditionModel
        unet = UNet2DConditionModel(sample_size=128, in_channels=4, out_channels=4, layers_per_block=2, block_out_channels=(320, 640, 1280), down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"), up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"))
        pipeline = StableDiffusionXLPipeline(vae=None, text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None, unet=unet, scheduler=None)
    print("Building key mapping...")
    comfyui_to_diffusers_map = unet_to_diffusers_mapping(unet_config, state_dict)
    print("Loading UNet weights...")
    new_state_dict = {}
    for comfy_key, diffusers_key in comfyui_to_diffusers_map.items():
        if comfy_key in state_dict: new_state_dict[diffusers_key] = state_dict[comfy_key]
    m, u = pipeline.unet.load_state_dict(new_state_dict, strict=False)
    return pipeline, state_dict, comfyui_to_diffusers_map


# --- Dual Monitor (identical to V1.3) ---
class DualMonitor:
    def __init__(self):
        self.output_sum = 0.0
        self.output_sq_sum = 0.0
        self.count = 0
        self.channel_importance = None

    def update(self, input_tensor, output_tensor):
        with torch.no_grad():
            out_detached = output_tensor.detach().float()
            self.output_sum += out_detached.mean().item()
            self.output_sq_sum += (out_detached ** 2).mean().item()
            inp_detached = input_tensor.detach()
            if inp_detached.dim() == 4:
                current_imp = inp_detached.abs().mean(dim=(0, 2, 3))
            elif inp_detached.dim() == 3:
                current_imp = inp_detached.abs().mean(dim=(0, 1))
            elif inp_detached.dim() == 2:
                current_imp = inp_detached.abs().mean(dim=0)
            else:
                current_imp = torch.ones(1, device=inp_detached.device, dtype=inp_detached.dtype)
            if self.channel_importance is None:
                self.channel_importance = current_imp
            else:
                self.channel_importance = (self.channel_importance * self.count + current_imp) / (self.count + 1)
            self.count += 1

    def get_sensitivity(self):
        if self.count == 0:
            return 0.0
        mean = self.output_sum / self.count
        sq_mean = self.output_sq_sum / self.count
        return sq_mean - mean ** 2


dual_monitors = {}

def hook_fn(module, input, output, name):
    if name not in dual_monitors:
        dual_monitors[name] = DualMonitor()
    dual_monitors[name].update(input[0], output)


def main():
    parser = argparse.ArgumentParser(description="SDXL FP8 Quantization (HSWQ V1.4)")
    parser.add_argument("--input",               type=str,   required=True)
    parser.add_argument("--output",              type=str,   required=True)
    parser.add_argument("--calib_file",          type=str,   required=True)
    parser.add_argument("--num_calib_samples",   type=int,   default=256)
    parser.add_argument("--num_inference_steps", type=int,   default=20)
    parser.add_argument("--keep_ratio",          type=float, default=0.25,
                        help="Ratio of layers to keep in FP16 (typical 0.05-0.25)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    pipeline, original_state_dict, comfyui_to_diffusers_map = load_unet_from_safetensors(args.input, device)

    # --- Calibration (identical to V1.3) ---
    print("Preparing calibration (registering Dual Monitor hooks)...")
    handles = []
    target_modules = []
    for name, module in pipeline.unet.named_modules():
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
    pipeline.set_progress_bar_config(disable=False)
    generator = torch.Generator(device=device).manual_seed(42)

    for i, prompt in enumerate(prompts):
        print(f"\nSample {i+1}/{args.num_calib_samples}: {prompt[:50]}...")
        with torch.no_grad():
            pipeline(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                output_type="latent",
                generator=generator,
            )
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    for h in handles:
        h.remove()

    # --- Sensitivity ranking (identical to V1.3) ---
    print("\nRunning layer sensitivity analysis...")
    layer_sensitivities = []
    for name in target_modules:
        if name in dual_monitors:
            sensitivity = dual_monitors[name].get_sensitivity()
            layer_sensitivities.append((name, sensitivity))

    layer_sensitivities.sort(key=lambda x: x[1], reverse=True)
    num_keep = int(len(layer_sensitivities) * args.keep_ratio)
    keep_layers = set(x[0] for x in layer_sensitivities[:num_keep])

    # Hard VETO (on top of keep_ratio)
    hard_veto_layers = {name for name in target_modules if is_hard_veto(name)}
    keep_layers |= hard_veto_layers

    print(f"Total layers: {len(layer_sensitivities)}")
    print(f"FP16-kept (sensitivity): {num_keep} (Top {args.keep_ratio*100:.1f}%)")
    print(f"FP16-kept (VETO):        {len(hard_veto_layers)}")
    print("Top 5 Sensitive:")
    for i in range(min(5, len(layer_sensitivities))):
        print(f"  {i+1}. {layer_sensitivities[i][0]}: {layer_sensitivities[i][1]:.4f}")

    # =====================================================================
    # HSWQ Pass 1: compute optimal amax AND MSE for every quantizable layer
    # =====================================================================
    print("\n" + "=" * 70)
    print("[HSWQ V1.4] Pass 1: Computing optimal amax + MSE for all layers...")
    print("=" * 70)

    SDXL_SEARCH_RANGE = (0.99, 1.0)

    hswq_optimizer = HSWQWeightedHistogramOptimizer(
        bins=4096,
        num_candidates=200,
        refinement_iterations=3,
        device=device,
    )

    # Collect amax and MSE for every non-keep layer
    layer_results = {}  # name -> {"amax": float, "mse": float}

    for name, module in tqdm(pipeline.unet.named_modules(), desc="Pass 1: Amax + MSE"):
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if name in keep_layers:
                continue

            importance = dual_monitors[name].channel_importance if name in dual_monitors else None

            wh = WeightedHistogramOptimized(bins=hswq_optimizer.bins, device=device)
            wh.build(module.weight.data, importance)

            optimal_amax = hswq_optimizer.mse_optimizer.find_optimal_amax(
                wh,
                num_candidates=hswq_optimizer.num_candidates,
                search_range=SDXL_SEARCH_RANGE,
                refinement_iterations=hswq_optimizer.refinement_iterations,
                scaled=False,
            )

            # Measure actual MSE at this amax
            histogram = wh.get_histogram()
            bin_centers = wh.get_bin_centers()
            mse = hswq_optimizer.mse_optimizer.compute_weighted_mse(
                histogram, bin_centers, optimal_amax, scaled=False
            )

            layer_results[name] = {"amax": optimal_amax, "mse": mse}
            torch.cuda.empty_cache()

    # =====================================================================
    # Auto-protect: detect MSE outliers and promote to FP16
    # =====================================================================
    all_mses = [r["mse"] for r in layer_results.values()]

    if all_mses:
        threshold = compute_auto_protect_threshold(all_mses)
        arr = np.array(sorted(all_mses))
        q1 = np.percentile(arr, 25)
        median = np.percentile(arr, 50)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1

        print(f"\n{'=' * 70}")
        print(f"[Auto-Protect] MSE Distribution Analysis")
        print(f"{'=' * 70}")
        print(f"  Layers analyzed : {len(all_mses)}")
        print(f"  MSE Q1          : {q1:.6e}")
        print(f"  MSE Median      : {median:.6e}")
        print(f"  MSE Q3          : {q3:.6e}")
        print(f"  MSE IQR         : {iqr:.6e}")
        print(f"  Auto threshold  : {threshold:.6e}  (Q3 + 1.5*IQR)")

        # Identify and promote outlier layers
        auto_promoted = set()
        promoted_details = []
        for name, result in layer_results.items():
            if result["mse"] > threshold:
                auto_promoted.add(name)
                promoted_details.append((name, result["mse"]))

        promoted_details.sort(key=lambda x: x[1], reverse=True)

        if auto_promoted:
            print(f"\n  Auto-promoted to FP16: {len(auto_promoted)} layers")
            for name, mse in promoted_details:
                print(f"    PROMOTE: {name:50} MSE={mse:.6e}")
            keep_layers |= auto_promoted
        else:
            print(f"\n  No outlier layers detected. All layers within normal MSE range.")

    # =====================================================================
    # Build final amax dict (excluding all keep_layers)
    # =====================================================================
    weight_amax_dict = {}
    for name, result in layer_results.items():
        if name not in keep_layers:
            weight_amax_dict[name + ".weight"] = result["amax"]

    fp8_count = len(weight_amax_dict)
    fp16_count = len(keep_layers)

    print(f"\n{'=' * 70}")
    print(f"[Final Summary]")
    print(f"  FP16 (sensitivity) : {num_keep}")
    print(f"  FP16 (VETO)        : {len(hard_veto_layers)}")
    print(f"  FP16 (auto-protect): {len(auto_promoted) if 'auto_promoted' in locals() else 0}")
    print(f"  FP16 total         : {fp16_count}")
    print(f"  FP8 total          : {fp8_count}")
    print(f"  FP16 ratio         : {fp16_count/(fp8_count+fp16_count)*100:.1f}%")
    print(f"{'=' * 70}")

    # =====================================================================
    # Dump full analysis log to JSON
    # =====================================================================
    import json
    log_file_path = args.output.rsplit(".", 1)[0] + "_hswq_log.json"
    print(f"\nSaving full analysis log to: {log_file_path}")
    log_data = {
        "summary": {
            "total_quantizable_layers": fp8_count + fp16_count,
            "fp16_count": fp16_count,
            "fp8_count": fp8_count,
            "fp16_ratio": fp16_count / max(1, fp8_count + fp16_count),
            "keep_ratio_target": args.keep_ratio,
            "auto_protect_threshold": float(threshold) if 'threshold' in locals() else None,
        },
        "layers": {}
    }

    # Aggregate layer info
    sensitivity_dict = {name: sens for name, sens in layer_sensitivities}
    
    for name in target_modules:
        status = "FP8"
        reason = ""
        if name in hard_veto_layers:
            status = "FP16"
            reason = "Hard VETO"
        elif 'auto_promoted' in locals() and name in auto_promoted:
            status = "FP16"
            reason = "Auto-Protect (High MSE)"
        elif name in keep_layers:
            status = "FP16"
            reason = f"Sensitivity Top {args.keep_ratio*100:.1f}%"
        
        log_data["layers"][name] = {
            "status": status,
            "reason": reason,
            "sensitivity": sensitivity_dict.get(name, 0.0),
            "mse": layer_results.get(name, {}).get("mse", None),
            "optimal_amax": layer_results.get(name, {}).get("amax", None)
        }

    with open(log_file_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)

    # =====================================================================
    # GPU conversion (identical to V1.3)
    # =====================================================================
    print("\n[VRAM Optimization] Preparing for GPU conversion...")
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
        if key in comfyui_to_diffusers_map:
            diffusers_key = comfyui_to_diffusers_map[key]
        elif key.startswith("model.diffusion_model."):
            if key in comfyui_to_diffusers_map:
                diffusers_key = comfyui_to_diffusers_map[key]

        module_name = None
        if diffusers_key and diffusers_key.endswith(".weight"):
            module_name = diffusers_key[:-7]

        if module_name and module_name in keep_layers:
            new_value = value
            kept_count += 1
        elif diffusers_key:
            weight_key = diffusers_key if diffusers_key.endswith(".weight") else diffusers_key + ".weight"
            if weight_key in weight_amax_dict:
                amax = weight_amax_dict[weight_key]
                if amax == 0: amax = 1e-6
                
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

    print(f"  FP8 layers : {converted_count}")
    print(f"  FP16 kept  : {kept_count}")

    try:
        save_file(output_state_dict, args.output)
    except Exception as e:
        print(f"[Save Warning] {e}. Moving to CPU...")
        cpu_dict = {k: v.cpu() for k, v in output_state_dict.items()}
        save_file(cpu_dict, args.output)

    print("Saved.")


if __name__ == "__main__":
    main()

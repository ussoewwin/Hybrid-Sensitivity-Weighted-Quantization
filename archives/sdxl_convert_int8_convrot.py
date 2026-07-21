"""
sdxl_convert_int8_convrot.py — SDXL INT8 FULL ConvRot + v1.3 DualMonitor FP16 keep.

Base pack: native_convert_int8_convrot.py (FULL ConvRot — NOT edited).
  INT8 scale = absmax (native pack_channelwise / pack_tensorwise).

v1.3 DualMonitor is used ONLY to rank layer sensitivity and decide FP16 keep
(keep_ratio). It is NOT used for INT8 pack scale (no FP8 HSWQ amax path).

Order:
  1) DualMonitor on original unrotated UNet → FP16 keep set
  2) Remaining layers: FULL ConvRot INT8 pack identical to native
     (Card 1 / Card 2 OFF)
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import random
import sys

import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file, save_file
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))

_DEFAULT_GROUPSIZE = 256


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


def _load_native_convert_int8():
    """Load sibling native_convert_int8.py for Hadamard / rotate_weight."""
    path = os.path.join(current_dir, "native_convert_int8.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"native_convert_int8.py not found: {path}")
    name = "native_convert_int8_for_sdxl_convrot"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _encode_comfy_quant(config: dict) -> torch.Tensor:
    return torch.tensor(
        list(json.dumps(config, separators=(",", ":")).encode("utf-8")),
        dtype=torch.uint8,
    )


def pack_tensorwise(weight: torch.Tensor):
    """Symmetric per-tensor INT8: scale = absmax / 127 (native)."""
    w = weight.float()
    absmax = max(float(w.abs().max().item()), 1e-6)
    scale = absmax / 127.0
    q = (w / scale).round().clamp(-127, 127).to(torch.int8)
    return q, torch.tensor(scale, dtype=torch.float32)


def pack_channelwise(weight: torch.Tensor):
    """Per-out-channel INT8 — identical to native_convert_int8_convrot.pack_channelwise."""
    w = weight.float()
    reduce_dims = tuple(range(1, w.dim()))
    absmax = torch.clamp(w.abs().amax(dim=reduce_dims).reshape(-1), min=1e-6)
    scale = absmax / 127.0
    if w.dim() == 4:
        scale_view = scale.view(-1, 1, 1, 1)
        absmax_view = absmax.view(-1, 1, 1, 1)
    elif w.dim() == 2:
        scale_view = scale.view(-1, 1)
        absmax_view = absmax.view(-1, 1)
    else:
        raise ValueError(f"unsupported weight ndim={w.dim()} for per-channel INT8")
    clamped = torch.clamp(w, -absmax_view, absmax_view)
    q = (clamped / scale_view).round().clamp(-127, 127).to(torch.int8)
    return q, scale_view.to(dtype=torch.float32)


# --- ComfyUI-compatible mapping helpers (from quantize_sdxl_hswq_v1.3.py) ---

def count_blocks(state_dict_keys, prefix_string):
    count = 0
    while True:
        c = False
        for k in state_dict_keys:
            if k.startswith(prefix_string.format(count)):
                c = True
                break
        if c is False:
            break
        count += 1
    return count


def calculate_transformer_depth(prefix, state_dict_keys, state_dict):
    transformer_prefix = prefix + "1.transformer_blocks."
    transformer_keys = sorted(
        list(filter(lambda a: a.startswith(transformer_prefix), state_dict_keys))
    )
    if len(transformer_keys) > 0:
        return count_blocks(state_dict_keys, transformer_prefix + "{}")
    return 0


def detect_unet_config_from_keys(state_dict, key_prefix="model.diffusion_model."):
    state_dict_keys = list(state_dict.keys())
    unet_config = {}
    if f"{key_prefix}input_blocks.0.0.weight" in state_dict_keys:
        model_channels = state_dict[f"{key_prefix}input_blocks.0.0.weight"].shape[0]
        num_res_blocks = []
        channel_mult = []
        transformer_depth = []
        transformer_depth_output = []
        input_block_count = count_blocks(
            state_dict_keys, f"{key_prefix}input_blocks" + ".{}."
        )
        last_res_blocks = 0
        last_channel_mult = 0
        for count in range(input_block_count):
            prefix = f"{key_prefix}input_blocks.{count}."
            prefix_output = f"{key_prefix}output_blocks.{input_block_count - count - 1}."
            block_keys = sorted(
                list(filter(lambda a: a.startswith(prefix), state_dict_keys))
            )
            if len(block_keys) == 0:
                break
            block_keys_output = sorted(
                list(filter(lambda a: a.startswith(prefix_output), state_dict_keys))
            )
            if f"{prefix}0.op.weight" in block_keys:
                num_res_blocks.append(last_res_blocks)
                channel_mult.append(last_channel_mult)
                last_res_blocks = 0
                last_channel_mult = 0
                out = calculate_transformer_depth(
                    prefix_output, state_dict_keys, state_dict
                )
                transformer_depth_output.append(out)
            else:
                res_block_prefix = f"{prefix}0.in_layers.0.weight"
                if res_block_prefix in block_keys:
                    last_res_blocks += 1
                    last_channel_mult = (
                        state_dict[f"{prefix}0.out_layers.3.weight"].shape[0]
                        // model_channels
                    )
                    out = calculate_transformer_depth(
                        prefix, state_dict_keys, state_dict
                    )
                    transformer_depth.append(out)
                res_block_prefix = f"{prefix_output}0.in_layers.0.weight"
                if res_block_prefix in block_keys_output:
                    out = calculate_transformer_depth(
                        prefix_output, state_dict_keys, state_dict
                    )
                    transformer_depth_output.append(out)
        num_res_blocks.append(last_res_blocks)
        channel_mult.append(last_channel_mult)
        if f"{key_prefix}middle_block.1.proj_in.weight" in state_dict_keys:
            transformer_depth_middle = count_blocks(
                state_dict_keys,
                f"{key_prefix}middle_block.1.transformer_blocks." + "{}",
            )
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


def unet_to_diffusers_mapping(
    unet_config, state_dict=None, key_prefix="model.diffusion_model."
):
    if "num_res_blocks" not in unet_config:
        return {}
    num_res_blocks = unet_config["num_res_blocks"]
    channel_mult = unet_config["channel_mult"]
    num_blocks = len(channel_mult)
    if state_dict is not None:
        import re

        state_dict_keys = list(state_dict.keys())
        filtered_keys = [
            k.replace(key_prefix, "")
            for k in state_dict_keys
            if k.startswith(key_prefix)
        ]
        transformer_counts = {}
        for key in filtered_keys:
            match = re.match(r"input_blocks\.(\d+)\.1\.transformer_blocks\.(\d+)", key)
            if match:
                block_idx = int(match.group(1))
                trans_idx = int(match.group(2))
                if block_idx not in transformer_counts:
                    transformer_counts[block_idx] = 0
                transformer_counts[block_idx] = max(
                    transformer_counts[block_idx], trans_idx + 1
                )
        output_transformer_counts = {}
        for key in filtered_keys:
            match = re.match(r"output_blocks\.(\d+)\.1\.transformer_blocks\.(\d+)", key)
            if match:
                block_idx = int(match.group(1))
                trans_idx = int(match.group(2))
                if block_idx not in output_transformer_counts:
                    output_transformer_counts[block_idx] = 0
                output_transformer_counts[block_idx] = max(
                    output_transformer_counts[block_idx], trans_idx + 1
                )
        middle_transformer_count = 0
        for key in filtered_keys:
            match = re.match(r"middle_block\.1\.transformer_blocks\.(\d+)", key)
            if match:
                trans_idx = int(match.group(1))
                middle_transformer_count = max(middle_transformer_count, trans_idx + 1)
        transformers_mid = (
            middle_transformer_count
            if middle_transformer_count > 0
            else unet_config.get("transformer_depth_middle", None)
        )
        transformer_depth = None
        transformer_depth_output = None
    else:
        transformer_depth = unet_config["transformer_depth"][:]
        transformer_depth_output = unet_config["transformer_depth_output"][:]
        transformers_mid = unet_config.get("transformer_depth_middle", None)
        transformer_counts = None
        output_transformer_counts = None
    UNET_MAP_RESNET = {
        "in_layers.2.weight": "conv1.weight",
        "in_layers.2.bias": "conv1.bias",
        "emb_layers.1.weight": "time_emb_proj.weight",
        "emb_layers.1.bias": "time_emb_proj.bias",
        "out_layers.3.weight": "conv2.weight",
        "out_layers.3.bias": "conv2.bias",
        "skip_connection.weight": "conv_shortcut.weight",
        "skip_connection.bias": "conv_shortcut.bias",
        "in_layers.0.weight": "norm1.weight",
        "in_layers.0.bias": "norm1.bias",
        "out_layers.0.weight": "norm2.weight",
        "out_layers.0.bias": "norm2.bias",
    }
    UNET_MAP_ATTENTIONS = {
        "proj_in.weight",
        "proj_in.bias",
        "proj_out.weight",
        "proj_out.bias",
        "norm.weight",
        "norm.bias",
    }
    TRANSFORMER_BLOCKS = {
        "norm1.weight",
        "norm1.bias",
        "norm2.weight",
        "norm2.bias",
        "norm3.weight",
        "norm3.bias",
        "attn1.to_q.weight",
        "attn1.to_q.bias",
        "attn1.to_k.weight",
        "attn1.to_k.bias",
        "attn1.to_v.weight",
        "attn1.to_out.0.weight",
        "attn1.to_out.0.bias",
        "attn2.to_q.weight",
        "attn2.to_k.weight",
        "attn2.to_v.weight",
        "attn2.to_out.0.weight",
        "attn2.to_out.0.bias",
        "ff.net.0.proj.weight",
        "ff.net.0.proj.bias",
        "ff.net.2.weight",
        "ff.net.2.bias",
    }
    UNET_MAP_BASIC = {
        ("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
        ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
        ("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
        ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
        ("input_blocks.0.0.weight", "conv_in.weight"),
        ("input_blocks.0.0.bias", "conv_in.bias"),
        ("out.0.weight", "conv_norm_out.weight"),
        ("out.0.bias", "conv_norm_out.bias"),
        ("out.2.weight", "conv_out.weight"),
        ("out.2.bias", "conv_out.bias"),
        ("time_embed.0.weight", "time_embedding.linear_1.weight"),
        ("time_embed.0.bias", "time_embedding.linear_1.bias"),
        ("time_embed.2.weight", "time_embedding.linear_2.weight"),
        ("time_embed.2.bias", "time_embedding.linear_2.bias"),
    }
    diffusers_unet_map = {}
    for x in range(num_blocks):
        n = 1 + (num_res_blocks[x] + 1) * x
        for i in range(num_res_blocks[x]):
            for b in UNET_MAP_RESNET:
                diffusers_unet_map[
                    "down_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])
                ] = "input_blocks.{}.0.{}".format(n, b)
            if transformer_counts is not None:
                num_transformers = transformer_counts.get(n, 0)
            else:
                num_transformers = transformer_depth.pop(0) if transformer_depth else 0
            if num_transformers > 0:
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map[
                        "down_blocks.{}.attentions.{}.{}".format(x, i, b)
                    ] = "input_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map[
                            "down_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(
                                x, i, t, b
                            )
                        ] = "input_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
            n += 1
        for k in ["weight", "bias"]:
            diffusers_unet_map["down_blocks.{}.downsamplers.0.conv.{}".format(x, k)] = (
                "input_blocks.{}.0.op.{}".format(n, k)
            )
    i = 0
    for b in UNET_MAP_ATTENTIONS:
        diffusers_unet_map["mid_block.attentions.{}.{}".format(i, b)] = (
            "middle_block.1.{}".format(b)
        )
    if transformers_mid:
        for t in range(transformers_mid):
            for b in TRANSFORMER_BLOCKS:
                diffusers_unet_map[
                    "mid_block.attentions.{}.transformer_blocks.{}.{}".format(i, t, b)
                ] = "middle_block.1.transformer_blocks.{}.{}".format(t, b)
    for i, n in enumerate([0, 2]):
        for b in UNET_MAP_RESNET:
            diffusers_unet_map[
                "mid_block.resnets.{}.{}".format(i, UNET_MAP_RESNET[b])
            ] = "middle_block.{}.{}".format(n, b)
    num_res_blocks_rev = list(reversed(num_res_blocks))
    for x in range(num_blocks):
        n = (num_res_blocks_rev[x] + 1) * x
        l = num_res_blocks_rev[x] + 1
        for i in range(l):
            for b in UNET_MAP_RESNET:
                diffusers_unet_map[
                    "up_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])
                ] = "output_blocks.{}.0.{}".format(n, b)
            if output_transformer_counts is not None:
                num_transformers = output_transformer_counts.get(n, 0)
            else:
                num_transformers = (
                    transformer_depth_output.pop() if transformer_depth_output else 0
                )
            if num_transformers > 0:
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map[
                        "up_blocks.{}.attentions.{}.{}".format(x, i, b)
                    ] = "output_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map[
                            "up_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(
                                x, i, t, b
                            )
                        ] = "output_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
            n += 1
        for k in ["weight", "bias"]:
            diffusers_unet_map["up_blocks.{}.upsamplers.0.conv.{}".format(x, k)] = (
                "output_blocks.{}.2.conv.{}".format(n, k)
            )
    for k, v in UNET_MAP_BASIC:
        diffusers_unet_map[v] = k
    comfyui_to_diffusers_map = {v: k for k, v in diffusers_unet_map.items()}
    comfyui_to_diffusers_map = {
        f"{key_prefix}{k}": v for k, v in comfyui_to_diffusers_map.items()
    }
    return comfyui_to_diffusers_map


def load_unet_from_safetensors(path, device="cuda"):
    print(f"Loading model: {path}")
    state_dict = load_file(path)
    print("Detecting UNet structure...")
    unet_config = detect_unet_config_from_keys(state_dict)
    print(f"Detected UNet config: {unet_config}")
    print("Initializing Diffusers pipeline...")
    try:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(device)
    except Exception as e:
        print(f"Warning: failed to load pretrained model: {e}")
        from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel(
            sample_size=128,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(320, 640, 1280),
            down_block_types=(
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
            ),
            up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
        )
        pipeline = StableDiffusionXLPipeline(
            vae=None,
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            unet=unet,
            scheduler=None,
        )
    print("Building key mapping...")
    comfyui_to_diffusers_map = unet_to_diffusers_mapping(unet_config, state_dict)
    print("Loading UNet weights...")
    new_state_dict = {}
    for comfy_key, diffusers_key in comfyui_to_diffusers_map.items():
        if comfy_key in state_dict:
            new_state_dict[diffusers_key] = state_dict[comfy_key]
    pipeline.unet.load_state_dict(new_state_dict, strict=False)
    return pipeline, state_dict, comfyui_to_diffusers_map


# --- DualMonitor: v1.3 sensitivity → FP16 keep only (not INT8 pack) ---

class DualMonitor:
    """Output-variance sensitivity for keep_ratio FP16 protection."""

    def __init__(self):
        self.output_sum = 0.0
        self.output_sq_sum = 0.0
        self.count = 0

    def update(self, _input_tensor, output_tensor):
        with torch.no_grad():
            out_detached = output_tensor.detach().float()
            self.output_sum += out_detached.mean().item()
            self.output_sq_sum += (out_detached**2).mean().item()
            self.count += 1

    def get_sensitivity(self):
        if self.count == 0:
            return 0.0
        mean = self.output_sum / self.count
        sq_mean = self.output_sq_sum / self.count
        return sq_mean - mean**2


dual_monitors = {}


def hook_fn(module, input, output, name):
    if name not in dual_monitors:
        dual_monitors[name] = DualMonitor()
    dual_monitors[name].update(input[0], output)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "sdxl_convert_int8_convrot: INT8 FULL ConvRot (absmax) + "
            "v1.3 DualMonitor for FP16 keep only"
        )
    )
    parser.add_argument("--input", "--model", dest="input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--calib_file", type=str, required=True)
    parser.add_argument(
        "--num_calib_samples",
        type=int,
        default=256,
        help="Calibration samples (v1.3 default: 256)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Denoising steps per calib sample (v1.3 default: 20)",
    )
    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=0.25,
        help="Ratio of layers to keep in FP16 (v1.3 DualMonitor top sensitivity)",
    )
    parser.add_argument(
        "--per_channel_int8",
        action="store_true",
        help="Card 3: per-out-channel absmax for non-ConvRot plain packs",
    )
    parser.add_argument(
        "--convrot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="FULL ConvRot on Linear+Conv2d (default ON; --no-convrot for plain)",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=_DEFAULT_GROUPSIZE,
        help=f"ConvRot Hadamard group size (power of 4, default {_DEFAULT_GROUPSIZE})",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Model not found at {args.input}")
        sys.exit(1)
    if args.groupsize < 4 or (args.groupsize & (args.groupsize - 1)) != 0:
        print(f"Error: --groupsize must be a power of 4 (>=4), got {args.groupsize}")
        sys.exit(1)
    if math.log(args.groupsize, 4) % 1 != 0:
        print(f"Error: --groupsize must be a power of 4, got {args.groupsize}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Card 1 (bias_correction): OFF")
    print("Card 2 (asymmetric): OFF")
    print(f"Card 3 (per_channel_int8): {bool(args.per_channel_int8)}")
    print(f"ConvRot: {bool(args.convrot)}")

    nc = None
    rotate_weight = None
    rotate_weight_conv2d = None
    convrot_group_size_for_features = None
    build_hadamard = None
    if args.convrot:
        nc = _load_native_convert_int8()
        rotate_weight = nc.rotate_weight
        rotate_weight_conv2d = nc.rotate_weight_conv2d
        convrot_group_size_for_features = nc.convrot_group_size_for_features
        build_hadamard = nc.build_hadamard
        print(
            f"  [ConvRot] FULL ON | preferred groupsize={args.groupsize} "
            f"(adaptive power-of-4 divisor)"
        )

    pipeline, original_state_dict, comfyui_to_diffusers_map = load_unet_from_safetensors(
        args.input, device
    )

    # DualMonitor: FP16 keep decision only (v1.3 sensitivity). Pack is separate.
    print("Preparing calibration (DualMonitor hooks for FP16 keep)...")
    dual_monitors.clear()
    handles = []
    target_modules = []
    for name, module in pipeline.unet.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handle = module.register_forward_hook(
                lambda m, i, o, n=name: hook_fn(m, i, o, n)
            )
            handles.append(handle)
            target_modules.append(name)

    print("Preparing calibration data...")
    with open(args.calib_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    if len(prompts) < args.num_calib_samples:
        prompts = (prompts * (args.num_calib_samples // len(prompts) + 1))[
            : args.num_calib_samples
        ]
    else:
        prompts = prompts[: args.num_calib_samples]

    print(
        f"Running calibration ({args.num_calib_samples} samples, "
        f"{args.num_inference_steps} steps)..."
    )
    print("Measuring layer sensitivity (DualMonitor → FP16 keep)...")
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

    print("\nRunning layer sensitivity analysis...")
    layer_sensitivities = []
    for name in target_modules:
        if name in dual_monitors:
            layer_sensitivities.append((name, dual_monitors[name].get_sensitivity()))
    layer_sensitivities.sort(key=lambda x: x[1], reverse=True)

    keep_ratio = float(args.keep_ratio)
    if keep_ratio <= 0:
        keep_ratio = 0.25
    num_keep = int(len(layer_sensitivities) * keep_ratio)
    keep_layers = set([x[0] for x in layer_sensitivities[:num_keep]])

    print(f"Total layers: {len(layer_sensitivities)}")
    print(f"FP16-kept layers: {len(keep_layers)} (Top {keep_ratio*100:.1f}%)")
    print("Top 5 Sensitive Layers:")
    for i in range(min(5, len(layer_sensitivities))):
        print(f"  {i+1}. {layer_sensitivities[i][0]}: {layer_sensitivities[i][1]:.4f}")

    group_size = int(args.groupsize)
    enable_convrot = bool(args.convrot)
    print("\n[INT8 pack] native FULL ConvRot / absmax (DualMonitor not used here)")

    print("\n[VRAM] Free pipeline before pack...")
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    new_state_dict = {}
    quant_meta_layers = {}
    converted_count = 0
    kept_count = 0
    skipped_count = 0
    plain_int8_count = 0
    convrot_linear = 0
    convrot_conv2d = 0
    per_channel_int8 = bool(args.per_channel_int8)

    print("Converting UNet weights to INT8 (or keep FP16)...")
    for key, tensor in tqdm(original_state_dict.items(), desc="Packing"):
        is_unet_matmul_weight = (
            key.startswith("model.diffusion_model")
            and key.endswith(".weight")
            and tensor.ndim >= 2
        )
        if not (
            is_unet_matmul_weight
            and tensor.dtype in (torch.float16, torch.float32, torch.bfloat16)
        ):
            new_state_dict[key] = tensor
            skipped_count += 1
            continue

        diffusers_key = comfyui_to_diffusers_map.get(key)
        module_name = None
        if diffusers_key and diffusers_key.endswith(".weight"):
            module_name = diffusers_key[:-7]

        if module_name and module_name in keep_layers:
            new_state_dict[key] = tensor
            kept_count += 1
            continue

        if per_channel_int8 and tensor.ndim not in (2, 4):
            new_state_dict[key] = tensor
            skipped_count += 1
            continue

        w_fp = tensor.float()
        used_gs = None
        if (
            enable_convrot
            and convrot_group_size_for_features is not None
            and build_hadamard is not None
        ):
            used_gs = convrot_group_size_for_features(int(w_fp.shape[1]), group_size)

        if used_gs is not None and tensor.ndim == 2 and rotate_weight is not None:
            # FULL ConvRot Linear — identical to native_convert_int8_convrot.py
            h_matrix = build_hadamard(used_gs, device="cpu", dtype=torch.float32)
            w_fp = rotate_weight(w_fp, h_matrix, used_gs)
            q, scale = pack_channelwise(w_fp)
            quant_config = {
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": int(used_gs),
            }
            convrot_linear += 1
        elif (
            used_gs is not None
            and tensor.ndim == 4
            and rotate_weight_conv2d is not None
        ):
            # FULL ConvRot Conv2d — identical to native_convert_int8_convrot.py
            h_matrix = build_hadamard(used_gs, device="cpu", dtype=torch.float32)
            w_fp = rotate_weight_conv2d(w_fp, h_matrix, used_gs)
            q, scale = pack_channelwise(w_fp)
            quant_config = {
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": int(used_gs),
            }
            convrot_conv2d += 1
        elif per_channel_int8:
            q, scale = pack_channelwise(w_fp)
            quant_config = {"format": "int8_tensorwise"}
            plain_int8_count += 1
        else:
            q, scale = pack_tensorwise(w_fp)
            quant_config = {"format": "int8_tensorwise"}
            plain_int8_count += 1

        module_key = key[: -len(".weight")]
        new_state_dict[key] = q
        new_state_dict[f"{module_key}.weight_scale"] = scale
        new_state_dict[f"{module_key}.comfy_quant"] = _encode_comfy_quant(quant_config)
        quant_meta_layers[module_key] = dict(quant_config)
        converted_count += 1

    metadata = {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": quant_meta_layers}
        )
    }

    print(f"Saving to: {args.output}")
    print(f"INT8 layers: {converted_count}")
    print(f"FP16-kept layers: {kept_count}")
    print(f"Other keys kept: {skipped_count}")
    print(f"Per-channel INT8 (Card 3): {per_channel_int8}")
    print(f"ConvRot FULL (Linear+Conv2d): {enable_convrot}")
    if enable_convrot:
        print(
            f"  ConvRot Linear: {convrot_linear}, ConvRot Conv2d: {convrot_conv2d}, "
            f"plain INT8: {plain_int8_count}"
        )
    else:
        print(f"  plain INT8: {plain_int8_count}")

    save_file(new_state_dict, args.output, metadata=metadata)
    print("Saved.")


if __name__ == "__main__":
    main()

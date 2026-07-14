"""
HSWQ V3.0 SDXL INT8 Quantization Script
========================================
Hybrid Sensitivity Weighted Quantization for Stable Diffusion XL (UNet only),
INT8 (symmetric per-tensor) edition.

V3.0 builds on V2.1 with the following INT8-specific changes:
- INT8Quantizer: symmetric per-tensor INT8 grid (127 positive levels, uniform)
  replaces FP8E4M3's non-linear grid (240 positive levels, dense near zero).
- HSWQWeightedHistogramOptimizerV4 receives INT8Quantizer via `quantizer=`
  injection; the SVD+RMS histogram MSE core is reused unchanged.
- derive_veto_tunables_int8: VETO fences from this checkpoint's distribution
  (weight-space outlier_ratio / abs_max / kurtosis — not 127/448-collapsed).
- Pack amax (Card 3 OFF): absmax (search_low = 1.0). Natural for symmetric
  INT8 uniform grid — NOT an HSWQ search. Deep amax clip drops SSIM.
- V4 weighted histogram MSE is for FP16 protection candidate selection:
  HSWQWeightedHistogramOptimizerV4 + INT8Quantizer measure estimated_mse at
  the (natural) absmax pack point; that MSE ranks which layers stay FP16
  under the 300 MiB budget (with DualMonitor sens + analyze severity).
  Pack and V4 ranking are separate jobs.
- Bias correction (Card 1): after INT8 pack, cancel systematic output bias
  E[(W_q - W) x] ≈ (W_q - W) @ mean(x) into the layer bias. Uses DualMonitor
  signed per-channel activation means from calibration. No format change,
  no extra FP16 keep, no loader change.
- Bias correction scope: default is ALL INT8 layers (same as commit d1290df,
  measured SSIM 0.9753). Optional Approach A (--bias_correction_top_ratio < 1)
  limits BC to the top fraction by DualMonitor sensitivity; top_ratio=0.5 was
  measured to raise MSE quality but DROP SSIM (0.9753 -> 0.9678) — do not use
  as default. Anchor: d1290df0d2b8624ee8fc317c0a44ebec9e10400f.
- Optional asymmetric INT8 pack (--asymmetric_int8, default off): map
  [w_min, w_max] via mid; loader still int8_tensorwise; mid absorbed by BC.
- Card 3 per-channel INT8 (--per_channel_int8, DEFAULT OFF): per-out-channel
  amax; weight_scale saved as (O,1) Linear / (O,1,1,1) Conv for kitchen
  dequantize_int8_simple broadcast. Format tag stays int8_tensorwise.
  Mutually exclusive with Card 2. When OFF, pack amax = absmax (obvious INT8);
  V4 histogram still ranks FP16 keep candidates (not pack amax).
- INT8 MAD attn VETO (this script only): MAD% floors are auto-derived from
  analyze_sdxl_distribution profile (Tukey / Q3 on attn mad_outlier_pct).
  Same path for every checkpoint — no per-model settings. FP8 scripts and
  derive_veto_tunables (FP8) are not modified.
- Output format: torch.int8 weight + float32 weight_scale, following ComfyUI
  `int8_tensorwise` layout (comfy/quant_ops.py QUANT_ALGOS["int8_tensorwise"]).
- _quantization_metadata embedded in safetensors metadata for ComfyUI loader.
- FP16 keep hard ceiling: exactly +300 MiB vs all-INT8 (owner non-negotiable).
  Per-model auto analysis / auto-optimal settings run ONLY inside that frame.
  DualMonitor FP16 cands + analyze VETO + V4 MSE → priority fill under 300 MiB.
  Never exceed 300. Never treat 300 as a removable "thinking-stop" constant.
  keep_ratio is r0. DualMonitor never invents keep_ratio. FP8 untouched.

ComfyUI compatibility:
  ComfyUI >= master with comfy_kitchen + TensorWiseINT8Layout can load these
  checkpoints. The metadata JSON format follows QUANTIZATION.md:
    {"format_version": "1.0", "layers": {<layer_name>: {"format": "int8_tensorwise"}}}

Calibration: StableDiffusionXLPipeline latent inference (Diffusers), same as V2.1.
Profiling: analyze/analyze_sdxl_distribution.py (reused; INT8 tunables derived
  from the same profile JSON via derive_veto_tunables_int8).
"""
import argparse
import math
import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file, save_file
import os
import gc
from tqdm import tqdm
import sys
import json
import numpy as np
import subprocess
from dataclasses import dataclass

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "ComfyUI-master"))

# Owner hard ceiling for FP16 overhead vs all-INT8. Auto analysis may only
# optimize INSIDE this frame. Not a thinking-stop formula constant.
FP16_BUDGET_MB_HARD = 300.0


def _require_fp16_budget_mb_hard(budget_mb: float) -> float:
    """Refuse any fp16_budget_mb other than the owner hard ceiling (300)."""
    b = float(budget_mb)
    if abs(b - FP16_BUDGET_MB_HARD) > 1e-6:
        raise ValueError(
            f"fp16_budget_mb must be exactly {FP16_BUDGET_MB_HARD:g} MiB "
            f"(owner hard ceiling; auto-optimal settings are inside this "
            f"frame only — never outside). Got {b}."
        )
    return FP16_BUDGET_MB_HARD

# Ensure histogram modules are importable regardless of clone path / CWD
histogram_dir = os.path.join(current_dir, "histogram")
if histogram_dir not in sys.path:
    sys.path.insert(0, histogram_dir)

# Support for optional venv site-packages (e.g. local wheels)
venv_site_packages = os.path.join(os.path.dirname(current_dir), "venv", "Lib", "site-packages")
if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
    sys.path.append(venv_site_packages)

from weighted_histogram_mse_v4 import HSWQWeightedHistogramOptimizerV4, INT8Quantizer

# Enforce C++20
if sys.platform == "win32":
    os.environ.setdefault("CXXFLAGS", "/std:c++20")
else:
    os.environ.setdefault("CXXFLAGS", "-std=c++20")

# --- SDXL UNet load (ComfyUI safetensors -> Diffusers pipeline) ---

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
    TRANSFORMER_BLOCKS = {"norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "norm3.weight", "norm3.bias", "attn1.to_q.weight", "attn1.to_q.bias", "attn1.to_k.weight", "attn1.to_k.bias", "attn1.to_v.weight", "attn1.to_v.bias", "attn1.to_out.0.weight", "attn1.to_out.0.bias", "attn2.to_q.weight", "attn2.to_k.weight", "attn2.to_v.weight", "attn2.to_out.0.weight", "attn2.to_out.0.bias", "ff.net.0.proj.weight", "ff.net.0.proj.bias", "ff.net.2.weight", "ff.net.2.bias"}
    UNET_MAP_BASIC = {("label_emb.0.0.weight", "add_embedding.linear_1.weight"), ("label_emb.0.0.bias", "add_embedding.linear_1.bias"), ("label_emb.0.2.weight", "add_embedding.linear_2.weight"), ("label_emb.0.2.bias", "add_embedding.linear_2.bias"), ("input_blocks.0.0.weight", "conv_in.weight"), ("input_blocks.0.0.bias", "conv_in.bias"), ("out.0.weight", "conv_norm_out.weight"), ("out.0.bias", "conv_norm_out.bias"), ("out.2.weight", "conv_out.weight"), ("out.2.bias", "conv_out.bias"), ("time_embed.0.weight", "time_embedding.linear_1.weight"), ("time_embed.0.bias", "time_embedding.linear_1.bias"), ("time_embed.2.weight", "time_embedding.linear_2.weight"), ("time_embed.2.bias", "time_embedding.linear_2.bias")}
    # Map only tensors present in this checkpoint's state_dict (auto from weights).
    # No invented Diffusers names, no fixed KEEP list, no inject.
    if state_dict is None:
        raise RuntimeError(
            "unet_to_diffusers_mapping requires state_dict; refuse maps without "
            "Comfy presence checks"
        )
    _sd_keys = set(state_dict.keys())
    _comfy_bare = {
        (k[len(key_prefix):] if k.startswith(key_prefix) else k)
        for k in _sd_keys
    }

    def _comfy_present(comfy_bare: str) -> bool:
        return comfy_bare in _comfy_bare or f"{key_prefix}{comfy_bare}" in _sd_keys

    def _map_put(diff_key: str, comfy_bare: str) -> None:
        if not _comfy_present(comfy_bare):
            return
        diffusers_unet_map[diff_key] = comfy_bare

    diffusers_unet_map = {}
    for x in range(num_blocks):
        n = 1 + (num_res_blocks[x] + 1) * x
        for i in range(num_res_blocks[x]):
            for b in UNET_MAP_RESNET:
                _map_put(
                    "down_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b]),
                    "input_blocks.{}.0.{}".format(n, b),
                )
            if transformer_counts is not None: num_transformers = transformer_counts.get(n, 0)
            else: num_transformers = transformer_depth.pop(0) if transformer_depth else 0
            if num_transformers > 0:
                for b in UNET_MAP_ATTENTIONS:
                    _map_put(
                        "down_blocks.{}.attentions.{}.{}".format(x, i, b),
                        "input_blocks.{}.1.{}".format(n, b),
                    )
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        _map_put(
                            "down_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b),
                            "input_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b),
                        )
            n += 1
        # Last DownBlock has no downsampler in SDXL — register only if op exists.
        if _comfy_present("input_blocks.{}.0.op.weight".format(n)):
            for k in ["weight", "bias"]:
                _map_put(
                    "down_blocks.{}.downsamplers.0.conv.{}".format(x, k),
                    "input_blocks.{}.0.op.{}".format(n, k),
                )
    i = 0
    for b in UNET_MAP_ATTENTIONS:
        _map_put("mid_block.attentions.{}.{}".format(i, b), "middle_block.1.{}".format(b))
    if transformers_mid:
        for t in range(transformers_mid):
            for b in TRANSFORMER_BLOCKS:
                _map_put(
                    "mid_block.attentions.{}.transformer_blocks.{}.{}".format(i, t, b),
                    "middle_block.1.transformer_blocks.{}.{}".format(t, b),
                )
    for i, n in enumerate([0, 2]):
        for b in UNET_MAP_RESNET:
            _map_put(
                "mid_block.resnets.{}.{}".format(i, UNET_MAP_RESNET[b]),
                "middle_block.{}.{}".format(n, b),
            )
    num_res_blocks_rev = list(reversed(num_res_blocks))
    for x in range(num_blocks):
        n = (num_res_blocks_rev[x] + 1) * x
        l = num_res_blocks_rev[x] + 1
        for i in range(l):
            c = 0
            for b in UNET_MAP_RESNET:
                _map_put(
                    "up_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b]),
                    "output_blocks.{}.0.{}".format(n, b),
                )
            c += 1
            if output_transformer_counts is not None: num_transformers = output_transformer_counts.get(n, 0)
            else: num_transformers = transformer_depth_output.pop() if transformer_depth_output else 0
            if num_transformers > 0:
                c += 1
                for b in UNET_MAP_ATTENTIONS:
                    _map_put(
                        "up_blocks.{}.attentions.{}.{}".format(x, i, b),
                        "output_blocks.{}.1.{}".format(n, b),
                    )
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        _map_put(
                            "up_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b),
                            "output_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b),
                        )
            # Upsample: only if this checkpoint has that Comfy conv (presence).
            # Missing tensor → no Diffusers entry.
            if i == l - 1:
                for k in ["weight", "bias"]:
                    _map_put(
                        "up_blocks.{}.upsamplers.0.conv.{}".format(x, k),
                        "output_blocks.{}.{}.conv.{}".format(n, c, k),
                    )
            n += 1
    for k, v in UNET_MAP_BASIC:
        _map_put(v, k)
    for _dk, _ck in diffusers_unet_map.items():
        if not _comfy_present(_ck):
            raise RuntimeError(
                f"Map integrity FATAL: mapped Comfy key {_ck!r} absent in checkpoint"
            )
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

def calculate_kurtosis(tensor):
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    if std == 0: return 0.0
    return torch.mean(((tensor - mean) / std) ** 4).item()

# --- V3.0 SDXL INT8 autonomous engine tunables ---
# Architectural boundary Conv2d keys (not Linear). Resolution resample is the
# same class of unet boundary as conv_in/conv_out; Linear-only walk previously
# made documented .conv_in/.conv_out key-pattern dead code (手抜き).
_SDXL_KP_BOUNDARY_SUFFIXES = (
    ".conv_in",
    ".conv_out",
    ".upsamplers.0.conv",
    ".downsamplers.0.conv",
)
_SDXL_KP_PREFIXES = ("time_embedding.", "add_embedding.")
_SDXL_ATTN_PROJ_SUFFIXES = (".to_q", ".to_k", ".to_v")
_SDXL_ATTN_TOOUT_SUFFIX = ".to_out.0"
_SDXL_PROFILE_PREFIXES = ("model.", "model.diffusion_model.")

# DualMonitor Sensitivity → FP16 candidates; analyze → VETO candidates.
# Both enter ONE per-model ranking in _apply_fp16_budget_cap (with V4 MSE).
# Budget winners = final FP16 protection. Analyze VETO is not renamed.
# keep_ratio is r0; DualMonitor must not invent or gate that flag.


@dataclass(frozen=True)
class SdxlVetoTunables:
    extreme_kurtosis: float
    extreme_outlier: float
    huge_magnitude: float
    attn_qkv_absmax: float
    attn_qkv_outlier: float
    attn_toout_absmax: float
    attn_toout_outlier: float
    ff2_outlier_live: float
    ff2_profile_outlier: float
    ff2_profile_score_cutoff: float
    ff2_auto_full_class: bool = False
    drift_veto_thresh: float = 0.0
    drift_score_mult: float = 1.0
    mse_release_o_min: float = 0.0
    mse_release_k_max: float = 0.0
    mse_release_m_max: float = 0.0
    mse_p75_multiplier: float = 1.0
    k_scale: float = 0.0
    o_scale: float = 0.0
    m_scale: float = 0.0
    k_gray_lo: float = 0.0
    k_gray_hi: float = 0.0
    o_gray_lo: float = 0.0
    o_gray_hi: float = 0.0
    m_gray_lo: float = 0.0
    m_gray_hi: float = 0.0
    search_low_floor: float = 1.0
    search_low_penalty_cap: float = 0.0
    search_low_clip_max: float = 1.0
    search_low_gray_clip_max: float = 1.0
    alpha_floor: float = 0.0
    alpha_clip_max: float = 0.0
    beta_floor: float = 0.0
    beta_clip_max: float = 0.0
    ff2_suffix_min_count: int = 4
    score_o_weight: float = 1.0
    score_m_weight: float = 1.0
    score_k_weight: float = 1.0
    quant_format: str = "int8_tensorwise"
    attn_mad_pct_floor: float = 0.0
    attn_mad_q3: float = 0.0
    attn_mad_p99: float = 0.0
    attn_mad_gap_o_max: float = 0.0
    attn_mad_from_profile: float = 0.0
    # Continuous MAD branch fingerprint (THIS pool IQR death → soft→Tukey; P99 tip-only).
    attn_mad_collapse: float = 0.0
    attn_mad_iqr: float = 0.0
    # Autonomous (from derive_int8_autonomous_tunables):
    sens_veto_percentile: float = 100.0
    sens_veto_keep_ratio_gate: float = 0.0
    bias_correction_top_ratio: float = 1.0
    auto_keep_ratio: float = 0.0
    fp16_budget_mb: float = 300.0
    fp16_budget_bytes: int = 314572800
    n_unet_layers: int = 0
    autonomous: bool = False
    # V4 Full-SVD×RMS mix weight from THIS multi-axis analyze character
    # (kurtosis∪outlier∪magnitude). alpha scales SVD leverage in the mix;
    # Full-SVD always runs even when alpha_auto resolves to 0.0.
    alpha_auto: float = 0.0

    # Required from derive_int8_autonomous_tunables — no silent default holes
    # after deleting accommodation clips (auto analysis → auto-optimal).
    _FROM_DICT_REQUIRED = (
        "extreme_kurtosis",
        "extreme_outlier",
        "huge_magnitude",
        "attn_qkv_absmax",
        "attn_qkv_outlier",
        "attn_toout_absmax",
        "attn_toout_outlier",
        "ff2_outlier_live",
        "ff2_profile_outlier",
        "drift_veto_thresh",
        "drift_score_mult",
        "mse_release_o_min",
        "mse_release_k_max",
        "mse_release_m_max",
        "mse_p75_multiplier",
        "k_scale",
        "o_scale",
        "m_scale",
        "k_gray_lo",
        "k_gray_hi",
        "o_gray_lo",
        "o_gray_hi",
        "m_gray_lo",
        "m_gray_hi",
        "alpha_floor",
        "alpha_clip_max",
        "beta_floor",
        "beta_clip_max",
        "alpha_auto",
        "search_low_floor",
        "search_low_penalty_cap",
        "search_low_clip_max",
        "search_low_gray_clip_max",
        "attn_mad_pct_floor",
        "attn_mad_q3",
        "attn_mad_p99",
        "attn_mad_gap_o_max",
        "attn_mad_from_profile",
        "bias_correction_top_ratio",
        "score_k_weight",
        "score_o_weight",
        "score_m_weight",
        "quant_format",
        "autonomous",
        "fp16_budget_mb",
    )

    @classmethod
    def from_dict(cls, d: dict) -> "SdxlVetoTunables":
        missing = [k for k in cls._FROM_DICT_REQUIRED if k not in d]
        if missing:
            raise ValueError(
                "SdxlVetoTunables.from_dict missing auto-optimal keys "
                f"{missing}. Run derive_int8_autonomous_tunables — do not "
                "fill deleted clip holes with dataclass defaults."
            )
        if not bool(d["autonomous"]):
            raise ValueError(
                "SdxlVetoTunables.from_dict requires autonomous=True "
                "(THIS-profile auto analysis → auto-optimal)"
            )
        if str(d["quant_format"]) != "int8_tensorwise":
            raise ValueError("INT8 SdxlVetoTunables requires quant_format=int8_tensorwise")
        if abs(float(d["fp16_budget_mb"]) - 300.0) > 1e-6:
            raise ValueError("fp16_budget_mb must be 300.0")
        if float(d["search_low_floor"]) != 1.0:
            raise ValueError("INT8 search_low_floor must be 1.0 (absmax auto-optimal)")
        if float(d["mse_p75_multiplier"]) <= 0.0:
            raise ValueError("mse_p75_multiplier must be > 0 from THIS profile")
        return cls(
            extreme_kurtosis=float(d["extreme_kurtosis"]),
            extreme_outlier=float(d["extreme_outlier"]),
            huge_magnitude=float(d["huge_magnitude"]),
            attn_qkv_absmax=float(d["attn_qkv_absmax"]),
            attn_qkv_outlier=float(d["attn_qkv_outlier"]),
            attn_toout_absmax=float(d["attn_toout_absmax"]),
            attn_toout_outlier=float(d["attn_toout_outlier"]),
            ff2_outlier_live=float(d["ff2_outlier_live"]),
            ff2_profile_outlier=float(d["ff2_profile_outlier"]),
            ff2_profile_score_cutoff=float(d.get("ff2_profile_score_cutoff", 0.0)),
            ff2_auto_full_class=bool(d.get("ff2_auto_full_class", False)),
            drift_veto_thresh=float(d["drift_veto_thresh"]),
            drift_score_mult=float(d["drift_score_mult"]),
            mse_release_o_min=float(d["mse_release_o_min"]),
            mse_release_k_max=float(d["mse_release_k_max"]),
            mse_release_m_max=float(d["mse_release_m_max"]),
            mse_p75_multiplier=float(d["mse_p75_multiplier"]),
            k_scale=float(d["k_scale"]),
            o_scale=float(d["o_scale"]),
            m_scale=float(d["m_scale"]),
            k_gray_lo=float(d["k_gray_lo"]),
            k_gray_hi=float(d["k_gray_hi"]),
            o_gray_lo=float(d["o_gray_lo"]),
            o_gray_hi=float(d["o_gray_hi"]),
            m_gray_lo=float(d["m_gray_lo"]),
            m_gray_hi=float(d["m_gray_hi"]),
            search_low_floor=float(d["search_low_floor"]),
            search_low_penalty_cap=float(d["search_low_penalty_cap"]),
            search_low_clip_max=float(d["search_low_clip_max"]),
            search_low_gray_clip_max=float(d["search_low_gray_clip_max"]),
            alpha_floor=float(d["alpha_floor"]),
            alpha_clip_max=float(d["alpha_clip_max"]),
            beta_floor=float(d["beta_floor"]),
            beta_clip_max=float(d["beta_clip_max"]),
            ff2_suffix_min_count=int(d.get("ff2_suffix_min_count", 4)),
            score_o_weight=float(d["score_o_weight"]),
            score_m_weight=float(d["score_m_weight"]),
            score_k_weight=float(d["score_k_weight"]),
            quant_format=str(d["quant_format"]),
            attn_mad_pct_floor=float(d["attn_mad_pct_floor"]),
            attn_mad_q3=float(d["attn_mad_q3"]),
            attn_mad_p99=float(d["attn_mad_p99"]),
            attn_mad_gap_o_max=float(d["attn_mad_gap_o_max"]),
            attn_mad_from_profile=float(d["attn_mad_from_profile"]),
            attn_mad_collapse=float(d.get("attn_mad_collapse", 0.0)),
            attn_mad_iqr=float(d.get("attn_mad_iqr", 0.0)),
            sens_veto_percentile=float(d.get("sens_veto_percentile", 100.0)),
            sens_veto_keep_ratio_gate=float(d.get("sens_veto_keep_ratio_gate", 0.0)),
            bias_correction_top_ratio=float(d["bias_correction_top_ratio"]),
            auto_keep_ratio=float(d.get("auto_keep_ratio", 0.0)),
            fp16_budget_mb=float(d["fp16_budget_mb"]),
            fp16_budget_bytes=int(d.get("fp16_budget_bytes", 300 * 1024 * 1024)),
            n_unet_layers=int(d.get("n_unet_layers", 0)),
            autonomous=True,
            alpha_auto=float(d["alpha_auto"]),
        )

    def as_dict(self) -> dict:
        return {
            "extreme_kurtosis": self.extreme_kurtosis,
            "extreme_outlier": self.extreme_outlier,
            "huge_magnitude": self.huge_magnitude,
            "attn_qkv_absmax": self.attn_qkv_absmax,
            "attn_qkv_outlier": self.attn_qkv_outlier,
            "attn_toout_absmax": self.attn_toout_absmax,
            "attn_toout_outlier": self.attn_toout_outlier,
            "ff2_outlier_live": self.ff2_outlier_live,
            "ff2_profile_outlier": self.ff2_profile_outlier,
            "ff2_profile_score_cutoff": self.ff2_profile_score_cutoff,
            "ff2_auto_full_class": self.ff2_auto_full_class,
            "drift_veto_thresh": self.drift_veto_thresh,
            "drift_score_mult": self.drift_score_mult,
            "mse_release_o_min": self.mse_release_o_min,
            "mse_release_k_max": self.mse_release_k_max,
            "mse_release_m_max": self.mse_release_m_max,
            "mse_p75_multiplier": self.mse_p75_multiplier,
            "k_scale": self.k_scale,
            "o_scale": self.o_scale,
            "m_scale": self.m_scale,
            "k_gray_lo": self.k_gray_lo,
            "k_gray_hi": self.k_gray_hi,
            "o_gray_lo": self.o_gray_lo,
            "o_gray_hi": self.o_gray_hi,
            "m_gray_lo": self.m_gray_lo,
            "m_gray_hi": self.m_gray_hi,
            "search_low_floor": self.search_low_floor,
            "search_low_penalty_cap": self.search_low_penalty_cap,
            "search_low_clip_max": self.search_low_clip_max,
            "search_low_gray_clip_max": self.search_low_gray_clip_max,
            "alpha_floor": self.alpha_floor,
            "alpha_clip_max": self.alpha_clip_max,
            "beta_floor": self.beta_floor,
            "beta_clip_max": self.beta_clip_max,
            "alpha_auto": self.alpha_auto,
            "ff2_suffix_min_count": self.ff2_suffix_min_count,
            "score_k_weight": self.score_k_weight,
            "score_o_weight": self.score_o_weight,
            "score_m_weight": self.score_m_weight,
            "quant_format": self.quant_format,
            "attn_mad_pct_floor": self.attn_mad_pct_floor,
            "attn_mad_q3": self.attn_mad_q3,
            "attn_mad_p99": self.attn_mad_p99,
            "attn_mad_gap_o_max": self.attn_mad_gap_o_max,
            "attn_mad_from_profile": self.attn_mad_from_profile,
            "attn_mad_collapse": self.attn_mad_collapse,
            "attn_mad_iqr": self.attn_mad_iqr,
            "sens_veto_percentile": self.sens_veto_percentile,
            "sens_veto_keep_ratio_gate": self.sens_veto_keep_ratio_gate,
            "bias_correction_top_ratio": self.bias_correction_top_ratio,
            "auto_keep_ratio": self.auto_keep_ratio,
            "fp16_budget_mb": self.fp16_budget_mb,
            "fp16_budget_bytes": self.fp16_budget_bytes,
            "n_unet_layers": self.n_unet_layers,
            "autonomous": self.autonomous,
        }


def resolve_veto_tunables(
    norm_profile: dict,
    profile_summary: dict | None = None,
    *,
    dual_monitors: dict | None = None,
    fp16_budget_mb: float = FP16_BUDGET_MB_HARD,
) -> SdxlVetoTunables:
    """Load INT8 veto_tunables via fully autonomous derivation.

    All knobs (Hard VETO fences, percentile promotions, dynamic ranking
    weights, MSE release gates, bias_correction scope, sens_veto percentile,
    alpha/beta, search_low) come from derive_int8_autonomous_tunables,
    which uses THIS checkpoint's profile + DualMonitor sensitivity
    distribution. fp16_budget_mb is the owner hard ceiling (300 MiB) —
    auto settings fill that frame; they do not redefine or exceed it.
    No hardcoded 90.0 / 15.0 / 2.0 / 0.5 / 40.0 recipe constants.
    """
    fp16_budget_mb = _require_fp16_budget_mb_hard(fp16_budget_mb)
    analyze_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze")
    if analyze_dir not in sys.path:
        sys.path.insert(0, analyze_dir)
    from analyze_sdxl_distribution import derive_int8_autonomous_tunables

    if norm_profile:
        sens_map: dict[str, float] = {}
        if dual_monitors:
            for name, mon in dual_monitors.items():
                try:
                    s = float(mon.get_sensitivity())
                except Exception:
                    s = 0.0
                if s > 0.0 and math.isfinite(s):
                    sens_map[name] = s
        derived = derive_int8_autonomous_tunables(
            norm_profile,
            dualmonitor_sensitivities=sens_map if sens_map else None,
            fp16_budget_mb=fp16_budget_mb,
        )
        if derived.get("ff2_auto_full_class"):
            print(
                "  [Auto FF2 INT8] full-class protection: "
                f"count={derived.get('ff2_class_count', 0)}, "
                f"span={derived.get('ff2_class_outlier_span', 0):.3f}, "
                f"profile_o>={derived['ff2_profile_outlier']:.2f}"
            )
        print(
            "  [Auto INT8 MAD] "
            f"floor={derived.get('attn_mad_pct_floor', 0.0):.3f}, "
            f"soft={derived.get('attn_mad_q3', 0.0):.3f}, "
            f"p99={derived.get('attn_mad_p99', 0.0):.3f}, "
            f"collapse={derived.get('attn_mad_collapse', 0.0):.3f}, "
            f"iqr={derived.get('attn_mad_iqr', 0.0):.3f}, "
            f"gap_o_max={derived.get('attn_mad_gap_o_max', 0.0):.3f}, "
            f"from_profile={bool(derived.get('attn_mad_from_profile', 0))}"
        )
        print(
            "  [V4→FP16 protect] "
            f"mse_release o>{derived.get('mse_release_o_min', 0):.3f} "
            f"k<={derived.get('mse_release_k_max', 0):.3f} "
            f"m<={derived.get('mse_release_m_max', 0):.3f} "
            f"p75×{derived.get('mse_p75_multiplier', 1.0):.2f} "
            f"(V4 estimated_mse ranks FP16 keep; "
            f"pack stays absmax search_low="
            f"{derived.get('search_low_floor', 1.0):.3f})"
        )
        print(
            "  [Autonomous INT8] "
            f"sens_veto_pct={derived.get('sens_veto_percentile', 100.0):.1f} "
            f"bc_top={derived.get('bias_correction_top_ratio', 1.0):.2f} "
            f"w(k/o/m)={derived.get('score_k_weight', 1.0):.3f}/"
            f"{derived.get('score_o_weight', 1.0):.3f}/"
            f"{derived.get('score_m_weight', 1.0):.3f} "
            f"auto_kr={derived.get('auto_keep_ratio', 0.0):.3f}"
        )
        print(
            "  [Auto-optimal replace] "
            f"drift={(derived.get('drift_veto_thresh', 0)):.4f} "
            f"(was 0.1..1.0 box → THIS (o_q3-o_med)/o_med) "
            f"mse_mult={derived.get('mse_p75_multiplier', 0):.4f} "
            f"(was 1.25..3.0 box → 1+IQR/o) "
            f"alpha_auto={derived.get('alpha_auto', 0):.4f} "
            f"α[{derived.get('alpha_floor', 0):.4f}..{derived.get('alpha_clip_max', 0):.4f}] "
            f"(was 0.5..0.99 box → THIS k∪o∪m character; SVD always runs) "
            f"mad_p99={derived.get('attn_mad_p99', 0):.4f} "
            f"(was ×4 q3 box → THIS attn MAD P99)"
        )
        return SdxlVetoTunables.from_dict(derived)
    # Stale precomputed veto_tunables without live layer profile = deleted-clip
    # hole risk. Always demand layers + re-derive.
    if profile_summary and isinstance(profile_summary.get("layers"), dict):
        return resolve_veto_tunables(
            profile_summary["layers"],
            dual_monitors=dual_monitors,
            fp16_budget_mb=fp16_budget_mb,
        )
    raise ValueError(
        "resolve_veto_tunables: need THIS checkpoint layer profile for "
        "derive_int8_autonomous_tunables (auto analysis → auto-optimal). "
        "Refuse stale veto_tunables-only load after accommodation-clip purge."
    )


def _layer_weight_stats(tensor: torch.Tensor) -> tuple[float, float, float]:
    """Live kurtosis, outlier_ratio, abs_max for a weight tensor."""
    x = tensor.float()
    std = torch.std(x).item()
    amax = max(abs(x.min().item()), abs(x.max().item()))
    k = calculate_kurtosis(x)
    o = amax / std if std > 0 else 0.0
    return k, o, amax


def _mad_outlier_pct(tensor: torch.Tensor, zthr: float = 3.0) -> float:
    """INT8-only robust outlier fraction (%). Not used by FP8 VETO paths."""
    xf = tensor.detach().float().reshape(-1)
    if xf.numel() < 4:
        return 0.0
    med = xf.median()
    mad = (xf - med).abs().median().clamp_min(1e-12)
    z = (xf - med).abs() / (1.4826 * mad)
    return float((z > zthr).float().mean().item() * 100.0)


def _profile_score_from_entry(
    prof: dict,
    drift: float = 0.0,
    tunables: SdxlVetoTunables | None = None,
) -> float:
    """Dynamic ranking score from distribution profile (+ optional post-calib drift)."""
    if not prof:
        return 0.0
    base = prof.get("profile_score")
    if base is None:
        k = float(prof.get("kurtosis", 0) or 0)
        o = float(prof.get("outlier_ratio", 0) or 0)
        m = float(prof.get("abs_max", 0) or 0)
        if tunables is not None:
            base = k + o * tunables.score_o_weight + m * tunables.score_m_weight
        else:
            base = k + o + m
    else:
        base = float(base)
    mult = tunables.drift_score_mult if tunables is not None else 1.0
    return base + drift * mult


def _profile_layer_stats(prof: dict, weight_tensor: torch.Tensor) -> tuple[float, float, float]:
    """Prefer precomputed profile stats; fall back to live weight scan."""
    if prof and "kurtosis" in prof and "outlier_ratio" in prof and "abs_max" in prof:
        return (
            float(prof.get("kurtosis", 0) or 0),
            float(prof.get("outlier_ratio", 0) or 0),
            float(prof.get("abs_max", 0) or 0),
        )
    return _layer_weight_stats(weight_tensor)


def _discover_ff2_suffixes(
    norm_profile: dict | None,
    min_count: int = 1,
) -> tuple[str, ...]:
    """Discover FFN output Linear suffixes from this checkpoint profile (no layer names)."""
    if not norm_profile:
        return ()
    analyze_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze")
    if analyze_dir not in sys.path:
        sys.path.insert(0, analyze_dir)
    from analyze_sdxl_distribution import _classify_layer_key

    counts: dict[str, int] = {}
    for key in norm_profile:
        ck = key if key.endswith(".weight") else f"{key}.weight"
        if _classify_layer_key(ck) != "ff2":
            continue
        base = key[:-7] if key.endswith(".weight") else key
        idx = base.rfind(".ff.")
        if idx < 0:
            continue
        suf = base[idx:]
        counts[suf] = counts.get(suf, 0) + 1
    if not counts:
        return ()
    best_count = max(counts.values())
    return tuple(
        sorted(s for s, c in counts.items() if c == best_count and c >= min_count)
    )


def _ff2_selective_veto_hit(
    prof: dict | None,
    live_o: float,
    tunables: SdxlVetoTunables,
) -> tuple[bool, str]:
    """Selective ff.net.2 VETO: class-relative profile_score and outlier (not blanket)."""
    # Cuts = derive_veto_tunables_int8 only (no hardcoded floors).
    score_cut = tunables.ff2_profile_score_cutoff
    outlier_cut = tunables.ff2_profile_outlier
    live_cut = tunables.ff2_outlier_live
    if prof:
        score = _profile_score_from_entry(prof, tunables=tunables)
        o = float(prof.get("outlier_ratio", 0) or 0)
        if score >= score_cut:
            return True, f"profile_score={score:.2f}>={score_cut}"
        if o >= outlier_cut:
            return True, f"profile_o={o:.1f}>={outlier_cut}"
        return False, ""
    if live_o > live_cut:
        return True, f"live_o={live_o:.1f}>{live_cut}"
    return False, ""


def _weight_profile_drift(weight_tensor: torch.Tensor, prof: dict) -> float:
    """Relative drift between live weights and distribution profile."""
    if not prof:
        return 0.0
    lk, lo, lm = _layer_weight_stats(weight_tensor)
    pk = float(prof.get("kurtosis", 0) or 0)
    po = float(prof.get("outlier_ratio", 0) or 0)
    pm = float(prof.get("abs_max", 0) or 0)
    dk = abs(lk - pk) / max(pk, 1.0)
    do = abs(lo - po) / max(po, 1.0)
    dm = abs(lm - pm) / max(pm, 1e-6)
    return max(dk, do, dm)


def _compute_sdxl_keypattern_veto(
    model: nn.Module,
    hard_veto_layers: set,
    tunables: SdxlVetoTunables,
    norm_profile: dict | None = None,
) -> set:
    """SDXL key-pattern VETO: embeddings, boundary Conv2d, profile-tuned ff2.

    Boundary suffixes apply to Conv2d (and any module whose name ends with the
    suffix). Linear-only iteration previously never reached .conv_in/.conv_out
    or resolution resample — that dead path is forbidden hand-waving.
    """
    added = set()
    ff2_suffixes = _discover_ff2_suffixes(norm_profile)
    for _n, _m in model.named_modules():
        if _n in hard_veto_layers:
            continue
        if isinstance(_m, torch.nn.Conv2d) and _n.endswith(_SDXL_KP_BOUNDARY_SUFFIXES):
            added.add(_n)
            print(f"    [Key-Pattern VETO] {_n} (boundary Conv2d)")
            continue
        if not isinstance(_m, torch.nn.Linear):
            continue
        if any(_n.startswith(p) for p in _SDXL_KP_PREFIXES):
            added.add(_n)
            print(f"    [Key-Pattern VETO] {_n} (embedding)")
            continue
        if ff2_suffixes and any(_n.endswith(s) for s in ff2_suffixes):
            # V3.0 INT8: skip full-class auto (inflates file size on SDXL);
            # selective VETO below handles individual outlier ff2 layers.
            prof = (norm_profile or {}).get(_n, {})
            _k, _o, _mstat = _profile_layer_stats(prof, _m.weight.detach())
            hit, reason = _ff2_selective_veto_hit(prof if prof else None, _o, tunables)
            if hit:
                added.add(_n)
                print(f"    [Key-Pattern VETO] {_n} (ff2 auto {reason})")
    if added:
        print(f"  [Key-Pattern VETO] Added {len(added)} layers.")
    return added


def _compute_structural_veto(
    model: nn.Module,
    hard_veto_layers: set,
    norm_profile: dict | None = None,
) -> set:
    """Linear layers whose weight shape is unique within the model (boundary detection)."""
    if norm_profile and any(
        isinstance(v, dict) and "shape_uniqueness" in v for v in norm_profile.values()
    ):
        model_linears = {
            n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
        }
        structural_veto = set()
        for name, entry in norm_profile.items():
            if not isinstance(entry, dict):
                continue
            if name not in model_linears:
                continue
            if entry.get("shape_uniqueness") == 1 and name not in hard_veto_layers:
                structural_veto.add(name)
                shp = entry.get("shape", [])
                print(f"    [Structural VETO] {name} shape={shp} (profile uniqueness=1)")
        return structural_veto

    shape_count: dict[tuple, int] = {}
    for _n, _m in model.named_modules():
        if isinstance(_m, torch.nn.Linear):
            _shp = tuple(_m.weight.shape)
            shape_count[_shp] = shape_count.get(_shp, 0) + 1
    structural_veto = set()
    for _n, _m in model.named_modules():
        if isinstance(_m, torch.nn.Linear):
            _shp = tuple(_m.weight.shape)
            if shape_count[_shp] == 1 and _n not in hard_veto_layers:
                structural_veto.add(_n)
                print(f"    [Structural VETO] {_n} shape={list(_shp)} (live uniqueness=1)")
    return structural_veto


def _compute_sdxl_per_projection_attn_veto(
    model: nn.Module,
    hard_veto_layers: set,
    tunables: SdxlVetoTunables,
    norm_profile: dict | None = None,
) -> set:
    """VETO attn projections when profile (or live) abs_max / outlier_ratio exceeds thresholds.

    V3.0 INT8: thresholds come only from derive_veto_tunables_int8
    (analyze_sdxl_distribution). No additional hardcoded floors.
    """
    proj_veto = set()
    # Thresholds = derive_veto_tunables_int8 only (no hardcoded INT8 floors).
    for _n, _m in model.named_modules():
        if not isinstance(_m, torch.nn.Linear):
            continue
        if _n in hard_veto_layers:
            continue
        if ".attn1" not in _n and ".attn2" not in _n:
            continue
        is_qkv = any(_n.endswith(s) for s in _SDXL_ATTN_PROJ_SUFFIXES)
        is_toout = _n.endswith(_SDXL_ATTN_TOOUT_SUFFIX)
        if not is_qkv and not is_toout:
            continue
        prof = (norm_profile or {}).get(_n, {})
        _k, _o, _amax = _profile_layer_stats(prof, _m.weight.detach())
        src = "profile" if prof else "live"
        if is_toout:
            hit = _amax >= tunables.attn_toout_absmax or _o >= tunables.attn_toout_outlier
            thresh_msg = (
                f"to_out amax>={tunables.attn_toout_absmax:.3f}, o>={tunables.attn_toout_outlier:.3f}"
            )
        else:
            hit = _amax >= tunables.attn_qkv_absmax or _o >= tunables.attn_qkv_outlier
            thresh_msg = (
                f"q/k/v amax>={tunables.attn_qkv_absmax:.3f}, o>={tunables.attn_qkv_outlier:.3f}"
            )
        if hit:
            proj_veto.add(_n)
            print(
                f"    [Per-Projection VETO] {_n} "
                f"({src} amax={_amax:.2f}, outlier={_o:.1f}; {thresh_msg})"
            )
    return proj_veto


def _mad_continuous_gates_from_live(
    live_mads: list[float],
) -> tuple[float, float, float, float]:
    """Mirror analyze MAD fences on a live THIS-UNet list.

    Returns (floor, soft, collapse, iqr). Same as analyze
    ``_mad_continuous_fences_from_positives``: hard=THIS MAD P75/Q3;
    soft=collapse-shaped Soft band on THIS below-floor MAD mass
    (not raw P50 flood; not (1-c)*P50+c*Q3 Soft death). P99 tip only.
    """
    live_sorted = sorted(float(v) for v in live_mads if float(v) > 0.0)
    n_live = len(live_sorted)
    if n_live < 1:
        return 0.0, 0.0, 0.0, 0.0
    if n_live < 4:
        peak = float(live_sorted[-1])
        body = float(live_sorted[n_live // 2])
        soft = float(min(body, peak))
        return peak, soft, 1.0, 0.0
    q1 = float(live_sorted[n_live // 4])
    q3 = float(live_sorted[(3 * n_live) // 4])
    iqr = float(max(q3 - q1, 0.0))
    p75 = float(
        live_sorted[max(0, min(n_live - 1, int(round(0.75 * (n_live - 1)))))]
    )
    p50 = float(live_sorted[n_live // 2])
    p99 = float(
        live_sorted[max(0, min(n_live - 1, int(round(0.99 * (n_live - 1)))))]
    )
    tail_span = float(max(p99 - p50, 1e-12))
    collapse = float(1.0 - min(1.0, iqr / (iqr + tail_span)))
    mad_floor = float(p75)
    below = [float(v) for v in live_sorted if float(v) < mad_floor]
    if below:
        tip_idx = max(
            0, min(len(below) - 1, int(round(collapse * (len(below) - 1))))
        )
        soft_tip = float(below[tip_idx])
        mad_soft = float((1.0 - collapse) * p50 + collapse * soft_tip)
    else:
        soft_tip = float(p50)
        mad_soft = float(p50)
    # Mirror analyze §3-1 / 8357425: Soft narrow band (not tip_headroom Soft死).
    soft_span = float(max(mad_floor - p50, 0.0))
    band_w = float(
        max(
            soft_span * float(max(1.0 - collapse, 0.15)),
            iqr * 0.1,
            mad_floor * 1e-6,
            1e-12,
        )
    )
    mad_soft = float(min(mad_soft, mad_floor - band_w))
    if mad_soft >= mad_floor:
        mad_soft = float(q1) if float(q1) < mad_floor else float(p50)
    if mad_soft >= mad_floor and below:
        mad_soft = float(below[-1])
    if mad_soft >= mad_floor:
        mad_soft = float(mad_floor) - float(max(mad_floor, 1.0) * 1e-12)
    return mad_floor, mad_soft, collapse, iqr


def _compute_sdxl_int8_mad_attn_veto(
    model: nn.Module,
    hard_veto_layers: set,
    tunables: SdxlVetoTunables | None = None,
    norm_profile: dict | None = None,
) -> set:
    """INT8-only key-pattern + MAD% VETO for attn projections.

    Floors / soft-gap from analyze continuous THIS-pool body fences
    (hard=THIS MAD Q3/P75; soft=below-floor collapse Soft band; P99 tip-only).
    If analyze left the MAD axis at 0.0, bootstrap the same fences from
    THIS UNet's live MAD pool (no fixed model literals, no tip-as-floor).
    """
    mad_floor = (
        float(tunables.attn_mad_pct_floor)
        if tunables is not None
        else 0.0
    )
    # attn_mad_q3 field stores Soft-MAD soft edge (analyze write), not Q3.
    mad_soft = float(tunables.attn_mad_q3) if tunables is not None else 0.0
    gap_o_max = (
        float(tunables.attn_mad_gap_o_max)
        if tunables is not None
        else 0.0
    )
    collapse = float(tunables.attn_mad_collapse) if tunables is not None else 0.0
    mad_iqr = float(tunables.attn_mad_iqr) if tunables is not None else 0.0
    if tunables is not None and gap_o_max <= 0.0:
        gap_o_max = float(max(tunables.extreme_outlier, 1e-9))
    prof = norm_profile or {}

    candidates: list[tuple[str, float, float]] = []
    live_mads: list[float] = []
    for _n, _m in model.named_modules():
        if not isinstance(_m, torch.nn.Linear):
            continue
        if _n in hard_veto_layers:
            continue
        if ".attn1" not in _n and ".attn2" not in _n:
            continue
        is_qkv = any(_n.endswith(s) for s in _SDXL_ATTN_PROJ_SUFFIXES)
        is_toout = _n.endswith(_SDXL_ATTN_TOOUT_SUFFIX)
        if not is_qkv and not is_toout:
            continue
        entry = prof.get(_n, {}) if isinstance(prof.get(_n, {}), dict) else {}
        mad_pct = float(entry.get("mad_outlier_pct", entry.get("mad_pct", 0)) or 0)
        if mad_pct <= 0.0:
            mad_pct = _mad_outlier_pct(_m.weight)
        o = float(entry.get("outlier_ratio", 0) or 0)
        if o <= 0.0 and hasattr(_m, "weight"):
            _, o, _ = _layer_weight_stats(_m.weight.data)
        candidates.append((_n, mad_pct, o))
        if mad_pct > 0.0:
            live_mads.append(mad_pct)

    if mad_floor <= 0.0 and live_mads:
        mad_floor, mad_soft, collapse, mad_iqr = _mad_continuous_gates_from_live(
            live_mads
        )
        if tunables is not None:
            tunables.attn_mad_pct_floor = float(mad_floor)
            tunables.attn_mad_q3 = float(mad_soft)
            tunables.attn_mad_collapse = float(collapse)
            tunables.attn_mad_iqr = float(mad_iqr)
            tunables.attn_mad_from_profile = 0.0
        if gap_o_max <= 0.0 and tunables is not None:
            gap_o_max = float(max(tunables.extreme_outlier, 1e-9))
        print(
            f"  [INT8 MAD VETO] Continuous THIS-UNet MAD body fences from "
            f"{len(live_mads)} live samples "
            f"(floor={mad_floor:.2f}, soft={mad_soft:.2f}, "
            f"collapse={collapse:.3f}, iqr={mad_iqr:.3f}; P99 tip-only)"
        )

    if mad_floor <= 0.0:
        return set()

    added = set()
    for _n, mad_pct, o in candidates:
        hard = mad_pct >= mad_floor
        soft = (
            mad_soft > 0.0
            and mad_pct >= mad_soft
            and mad_pct < mad_floor
            and o < gap_o_max
        )
        if hard or soft:
            added.add(_n)
            kind = "hard" if hard else "soft"
            o_note = "o_miss" if o < gap_o_max else "o_hit"
            print(
                f"    [INT8 MAD VETO] {_n} "
                f"(MAD%={mad_pct:.2f}, o={o:.2f}, floor={mad_floor:.2f}, "
                f"soft={mad_soft:.2f}, collapse={collapse:.3f}, "
                f"iqr={mad_iqr:.3f}, gate_o={gap_o_max:.2f}; "
                f"{kind}/{o_note})"
            )
    if added:
        print(
            f"  [INT8 MAD VETO] Added {len(added)} attn layers "
            f"(floor={mad_floor:.2f}, soft={mad_soft:.2f}, "
            f"collapse={collapse:.3f}, iqr={mad_iqr:.3f})."
        )
    return added


def _autonomous_supplemental_veto(
    model: nn.Module,
    hard_veto_layers: set,
    norm_profile: dict,
    tunables: SdxlVetoTunables,
) -> set:
    """Profile-primary VETO: outlier ff.net.2, high-drift embedding layers."""
    added = set()
    for _n, _m in model.named_modules():
        if not isinstance(_m, torch.nn.Linear):
            continue
        if _n in hard_veto_layers:
            continue
        prof = norm_profile.get(_n, {})
        drift = _weight_profile_drift(_m.weight.data, prof)
        _k, _o, _mstat = _profile_layer_stats(prof, _m.weight.detach())
        ff2_suffixes = _discover_ff2_suffixes(
            norm_profile, min_count=tunables.ff2_suffix_min_count
        )
        if ff2_suffixes and any(_n.endswith(s) for s in ff2_suffixes):
            # V3.0 INT8: selective only (no full-class auto)
            hit, reason = _ff2_selective_veto_hit(prof if prof else None, _o, tunables)
            if hit:
                added.add(_n)
                print(f"    [Supplemental VETO] {_n} (ff.net.2 {reason})")
        elif any(_n.startswith(p) for p in _SDXL_KP_PREFIXES) and drift > tunables.drift_veto_thresh:
            added.add(_n)
            print(
                f"    [Supplemental VETO] {_n} "
                f"(embedding drift={drift:.3f} > {tunables.drift_veto_thresh:.3f})"
            )
    return added


def _collect_mse_release_candidates(
    hard_veto_layers: set,
    structural_veto: set,
    norm_profile: dict,
    model: nn.Module,
    tunables: SdxlVetoTunables,
) -> set:
    """Outlier-only profile VETO with low drift and non-structural — MSE release candidates.

    V3.0 INT8: mse_release_* come only from derive_veto_tunables_int8
    (analyze_sdxl_distribution). No hardcoded min/max floors.
    """
    candidates = set()
    _module_dict = dict(model.named_modules())
    for vname in hard_veto_layers:
        if vname in structural_veto:
            continue
        prof = norm_profile.get(vname, {})
        k = float(prof.get("kurtosis", 0) or 0)
        m = float(prof.get("abs_max", 0) or 0)
        o = float(prof.get("outlier_ratio", 0) or 0)
        if (
            o > tunables.mse_release_o_min
            and k <= tunables.mse_release_k_max
            and m <= tunables.mse_release_m_max
        ):
            vmod = _module_dict.get(vname)
            if vmod is not None and hasattr(vmod, "weight"):
                drift = _weight_profile_drift(vmod.weight.data, prof)
                if drift < tunables.drift_veto_thresh:
                    candidates.add(vname)
    return candidates


def _dualmonitor_channel_importance(dual_monitors: dict, module_name: str):
    """1D input-channel importance from DualMonitor (32-sample calib contract)."""
    mon = dual_monitors.get(module_name) if dual_monitors else None
    if mon is None:
        return None
    imp = getattr(mon, "channel_importance", None)
    if imp is None:
        return None
    return imp.detach().float()


def _mse_grayzone_veto_reassessment(
    *,
    scope_label: str,
    hard_veto_layers: set,
    keep_layers: set,
    outlier_only_veto: set,
    target_modules: list,
    model: torch.nn.Module,
    _norm_profile: dict,
    get_layer_search_low,
    alpha: float,
    beta: float,
    device: str,
    tunables: SdxlVetoTunables,
    dual_monitors: dict | None = None,
    mse_cache: dict | None = None,
) -> tuple[set, set, dict]:
    """Gray-zone soft-VETO release; V4 MSE also fills FP16 protect cache.

    Primary V4 role in V3.0 INT8 is FP16 protection ranking (see
    _build_v4_calib_fp16_candidates). This path reuses the same V4
    estimated_mse @ absmax to optionally release soft analyze-VETO layers
    whose damage is below P75×mult of a safe baseline.

    Pack amax stays absmax — V4 does not choose pack scale.

    DualMonitor importance preferred when present; always Full-SVD×RMS
    via alpha_auto (missing Imp never skips V4 or SVD).

    Returns (hard_veto, keep, mse_cache) where mse_cache maps layer name →
    V4 estimated_mse at absmax (FP16-budget priority; not profile_score).
    Reuses the caller's mse_cache (V4 calib scores); never wipes it.
    """
    mse_cache = dict(mse_cache or {})
    if not outlier_only_veto:
        return hard_veto_layers, keep_layers, mse_cache

    if not dual_monitors:
        raise ValueError(
            f"{scope_label}: V4 gray-zone path needs DualMonitor maps from "
            "calibration (num_calib_samples=32 recipe)."
        )

    print(
        f"\n  [{scope_label} V4→FP16 protect / gray-zone] "
        f"{len(outlier_only_veto)} soft-VETO candidates from analyze "
        f"(o>{tunables.mse_release_o_min:.2f}, "
        f"k<={tunables.mse_release_k_max:.2f}, m<={tunables.mse_release_m_max:.2f})."
    )
    print(
        "  [V4→FP16 protect] HSWQWeightedHistogramOptimizerV4 + INT8Quantizer "
        f"estimated_mse @ absmax (FP16 ranking + optional soft-VETO release); "
        f"release if MSE <= {tunables.mse_p75_multiplier:.2f}×P75(safe)."
    )

    int8_quantizer = INT8Quantizer(device=device)
    trial_optimizer = HSWQWeightedHistogramOptimizerV4(
        bins=8192, num_candidates=1000, refinement_iterations=10,
        device=device, alpha=alpha, beta=beta,
        quantizer=int8_quantizer,
    )
    # Measure V4 damage at the natural INT8 pack point (absmax).
    _veto_search_range = (1.0, 1.0)

    safe_mses = []
    _module_dict = dict(model.named_modules())
    _safe_pool = [n for n in target_modules if n not in keep_layers and n in _module_dict]
    ff2_suffixes = _discover_ff2_suffixes(
        _norm_profile, min_count=tunables.ff2_suffix_min_count
    )
    if ff2_suffixes:
        _safe_ff = [n for n in _safe_pool if any(n.endswith(s) for s in ff2_suffixes)]
    else:
        _safe_ff = []
    step = max(1, len(_safe_ff) // 30)
    _safe_sample = _safe_ff[::step][:30]
    for sname in _safe_sample:
        smod = _module_dict[sname]
        if not hasattr(smod, "weight"):
            continue
        sw = smod.weight.data
        simp = _dualmonitor_channel_importance(dual_monitors, sname)
        try:
            # Full-SVD×RMS always; DualMonitor Importance multiplies when present.
            sresult = trial_optimizer.compute_optimal_amax_with_stats_int8_range(
                sw,
                importance=simp,
                use_svd_leverage=True,
                scaled=False,
                search_range=_veto_search_range,
            )
            safe_mses.append(sresult["estimated_mse"])
            mse_cache[sname] = float(sresult["estimated_mse"])
        except Exception as e:
            print(f"    [MSE ERROR] Failed safe layer {sname}: {e}")
        torch.cuda.empty_cache()

    if not safe_mses:
        print(
            f"  [{scope_label} V4→FP16 protect / gray-zone] "
            "No safe baseline available, skipping."
        )
        return hard_veto_layers, keep_layers, mse_cache

    safe_mses.sort()
    p75_idx = int(len(safe_mses) * 0.75)
    mse_threshold = safe_mses[min(p75_idx, len(safe_mses) - 1)] * tunables.mse_p75_multiplier
    print(
        f"  [MSE Baseline INT8] Safe layers sampled: {len(safe_mses)}, "
        f"P75 MSE: {safe_mses[p75_idx]:.8f}, "
        f"Threshold ({tunables.mse_p75_multiplier:.2f}xP75): {mse_threshold:.8f}"
    )

    released = set()
    for vname in sorted(outlier_only_veto):
        if vname not in _module_dict:
            continue
        vmod = _module_dict[vname]
        if not hasattr(vmod, "weight"):
            continue
        vw = vmod.weight.data
        vimp = _dualmonitor_channel_importance(dual_monitors, vname)
        try:
            vresult = trial_optimizer.compute_optimal_amax_with_stats_int8_range(
                vw,
                importance=vimp,
                use_svd_leverage=True,
                scaled=False,
                search_range=_veto_search_range,
            )
            vmse = vresult["estimated_mse"]
            mse_cache[vname] = float(vmse)
            vprof = _norm_profile.get(vname, {})
            vor = vprof.get("outlier_ratio", 0)
            if vmse <= mse_threshold:
                released.add(vname)
                print(
                    f"    RELEASED: {vname} | MSE={vmse:.8f} <= threshold={mse_threshold:.8f} "
                    f"| o={vor:.1f} | amax={vresult['optimal_amax']:.4f}"
                )
            else:
                print(
                    f"    KEPT:     {vname} | MSE={vmse:.8f} >  threshold={mse_threshold:.8f} "
                    f"| o={vor:.1f}"
                )
        except Exception as e:
            print(f"    ERROR:    {vname} | {e}")
        torch.cuda.empty_cache()

    if released:
        hard_veto_layers = hard_veto_layers - released
        keep_layers = keep_layers - released
        print(
            f"  [{scope_label} V4→FP16 protect / gray-zone] "
            f"Released {len(released)} soft-VETO layers. "
            f"Remaining hard VETO: {len(hard_veto_layers)}."
        )
        print(f"  Updated FP16 kept layers: {len(keep_layers)}")
    else:
        print(
            f"  [{scope_label} V4→FP16 protect / gray-zone] "
            "No soft-VETO release (all exceeded MSE threshold)."
        )

    return hard_veto_layers, keep_layers, mse_cache


def _fp16_extra_bytes_vs_int8(weight: torch.Tensor) -> int:
    """Extra bytes of keeping FP16 vs packing INT8 (2B/elem vs 1B/elem → +1B/elem)."""
    return int(weight.numel())


def _measure_v4_mse_absmax_int8(
    *,
    weight: torch.Tensor,
    importance: torch.Tensor | None,
    optimizer: HSWQWeightedHistogramOptimizerV4,
) -> float:
    """INT8-only: V4 estimated_mse for FP16 protection candidate ranking.

    Measures weighted-histogram MSE at the natural INT8 pack point (absmax).
    That MSE is the damage score used to decide FP16 keep — it is NOT used
    to choose a pack amax (pack stays absmax for INT8).

    Always runs V4 Full-SVD×RMS hybrid (use_svd_leverage=True). When
    DualMonitor channel Importance is present it multiplies the hybrid map;
    when missing, hybrid alone. Cutting SVD because Imp exists is forbidden.
    """
    result = optimizer.compute_optimal_amax_with_stats_int8_range(
        weight,
        importance=importance,
        use_svd_leverage=True,
        scaled=False,
        search_range=(1.0, 1.0),
    )
    return float(result["estimated_mse"])



def _build_v4_calib_fp16_candidates(
    model: torch.nn.Module,
    dual_monitors: dict,
    target_modules: list,
    *,
    hard_veto_layers: set,
    mse_cache: dict | None,
    alpha: float,
    beta: float,
    device: str,
) -> tuple[set, dict]:
    """Score FP16 protection candidates with histogram V4 on THIS calibration.

    V4's job here: estimated_mse @ absmax for every target Linear/Conv so the
    later 300 MiB budget can rank which layers stay FP16. Pack amax remains
    absmax separately — V4 does not search pack scale.

    Always Full-SVD×RMS hybrid; DualMonitor Importance multiplies when present.
    Never skip a measurable layer (skipping collapses FP16 selection).

    Returns (all_v4_scored_names, mse_cache). Does NOT truncate by keep_ratio:
    truncation is only the FP16 budget pass over the FULL priority order of
    (V4-scored U analyze VETO U fence-crossers). Pre-cutting here is the
    hand-wave that collapses quality (~0.92).
    """
    cache = dict(mse_cache or {})
    module_dict = dict(model.named_modules())
    scored: set = set()
    need = []
    for name in target_modules:
        mod = module_dict.get(name)
        if mod is None or not hasattr(mod, "weight") or mod.weight is None:
            continue
        if name in cache:
            scored.add(name)
        else:
            need.append(name)

    trial_optimizer = None
    n_svd_x_imp = 0
    n_svd_only = 0
    if need:
        print(
            f"  [V4→FP16 protect] measuring V4 estimated_mse @ absmax for "
            f"{len(need)} layers (FP16 keep ranking; pack stays absmax; "
            f"cache hit={len(scored)}; analyze VETO={len(hard_veto_layers)}; "
            f"NO keep_ratio pre-cut)..."
        )
        trial_optimizer = HSWQWeightedHistogramOptimizerV4(
            bins=8192, num_candidates=1000, refinement_iterations=10,
            device=device, alpha=alpha, beta=beta,
            quantizer=INT8Quantizer(device=device),
        )
    for name in need:
        if trial_optimizer is None:
            break
        mod = module_dict[name]
        imp = _dualmonitor_channel_importance(dual_monitors, name)
        try:
            v4_mse = _measure_v4_mse_absmax_int8(
                weight=mod.weight.data,
                importance=imp,
                optimizer=trial_optimizer,
            )
            cache[name] = float(v4_mse)
            scored.add(name)
            if imp is None:
                n_svd_only += 1
            else:
                n_svd_x_imp += 1
        except Exception as e:
            print(f"    [V4→FP16 protect] skip {name}: {e}")
            continue
        torch.cuda.empty_cache()

    print(
        f"  [V4→FP16 protect] V4-scored={len(scored)} "
        f"(SVD×Imp={n_svd_x_imp}, SVD-only={n_svd_only}; "
        f"alpha={alpha:.3f}/beta={beta:.3f}) | "
        f"analyze VETO={len(hard_veto_layers)} | "
        f"union → FULL priority (budget only truncates)."
    )
    return scored, cache



def _apply_fp16_budget_cap(
    model: torch.nn.Module,
    keep_layers: set,
    hard_veto_layers: set,
    *,
    budget_mb: float = FP16_BUDGET_MB_HARD,
    norm_profile: dict,
    veto_tunables: SdxlVetoTunables,
    dual_monitors: dict | None,
    mse_cache: dict | None = None,
    alpha: float,
    beta: float,
    device: str = "cuda",
) -> tuple[set, set, dict]:
    """Per-model auto analysis → auto-optimal FP16 set inside 300 MiB.

    Owner hard ceiling: fp16_budget_mb == 300 exactly. Auto settings fill
    that frame; they never redefine it and never exceed it.

    alpha/beta MUST be THIS-profile auto-optimal (caller passes
    veto_tunables.alpha_auto mix). Fixed 0.5/0.5 defaults are forbidden.
    """
    if not math.isfinite(float(alpha)) or not math.isfinite(float(beta)):
        raise ValueError(
            f"_apply_fp16_budget_cap: alpha/beta must be finite auto-optimal "
            f"(got alpha={alpha}, beta={beta})"
        )
    budget_mb = _require_fp16_budget_mb_hard(budget_mb)
    analyze_dir = os.path.join(current_dir, "analyze")
    if analyze_dir not in sys.path:
        sys.path.insert(0, analyze_dir)
    from analyze_sdxl_distribution import (
        apply_fp16_infinite_priority_branches,
        apply_fp16_infinite_ranking_branches,
        build_int8_analyze_character_table,
        int8_fp16_budget_analyze_severity,
        int8_fp16_budget_priority,
        derive_priority_combinator,
        _safe_percentile,
        _robust_iqr,
    )

    if str(veto_tunables.quant_format) != "int8_tensorwise":
        raise ValueError(
            "_apply_fp16_budget_cap is INT8-only "
            f"(got quant_format={veto_tunables.quant_format!r})"
        )
    if not dual_monitors:
        raise ValueError(
            "[FP16 budget] DualMonitor maps required for Sensitivity + "
            "V4 Importance; refusing fixed-formula / profile_score fallback."
        )

    tunables_dict = veto_tunables.as_dict()
    budget_bytes = int(budget_mb * 1024 * 1024)

    char_table = build_int8_analyze_character_table(
        {"layers": norm_profile},
        tunables_dict,
        hard_veto_names=hard_veto_layers,
    )

    pool = set(keep_layers) | set(hard_veto_layers)
    for name, row in char_table.items():
        if float(row.get("severity", 0.0)) >= 1.0:
            pool.add(name)

    module_dict = dict(model.named_modules())
    pool = {n for n in pool if n in module_dict and hasattr(module_dict[n], "weight")}

    sens_by_name: dict[str, float] = {}
    for name, mon in dual_monitors.items():
        if name not in module_dict or not hasattr(module_dict[name], "weight"):
            continue
        try:
            s = float(mon.get_sensitivity())
        except Exception:
            s = 0.0
        if s > 0.0 and math.isfinite(s):
            sens_by_name[name] = s
            pool.add(name)

    cache = dict(mse_cache or {})
    measured: list[tuple[str, float, float, float, int]] = []
    skipped_no_weight = []
    skipped_no_v4 = []
    measured_fresh = 0

    need_fresh = [n for n in pool if n not in cache]
    trial_optimizer = None
    if need_fresh:
        print(
            f"  [FP16 budget] THIS-model pool measure: "
            f"analyze={len(char_table)} pool={len(pool)} "
            f"dm_sens={len(sens_by_name)} | V4 fresh={len(need_fresh)} "
            f"(cache={len(cache)})..."
        )
        trial_optimizer = HSWQWeightedHistogramOptimizerV4(
            bins=8192, num_candidates=1000, refinement_iterations=10,
            device=device, alpha=alpha, beta=beta,
            quantizer=INT8Quantizer(device=device),
        )
    else:
        print(
            f"  [FP16 budget] THIS-model pool: "
            f"analyze={len(char_table)} pool={len(pool)} "
            f"dm_sens={len(sens_by_name)} | V4 cached ({len(cache)})"
        )

    for name in sorted(pool):
        mod = module_dict.get(name)
        if mod is None or not hasattr(mod, "weight") or mod.weight is None:
            skipped_no_weight.append(name)
            continue
        dm_sens = float(sens_by_name.get(name, 0.0))
        extra = _fp16_extra_bytes_vs_int8(mod.weight.data)
        row = char_table.get(name, {})
        prof = norm_profile.get(name, {}) if isinstance(norm_profile.get(name), dict) else {}
        is_hv = name in hard_veto_layers
        k = float(row.get("kurtosis", prof.get("kurtosis", 0)) or 0)
        o = float(row.get("outlier_ratio", prof.get("outlier_ratio", 0)) or 0)
        m = float(row.get("abs_max", prof.get("abs_max", 0)) or 0)
        mad = float(row.get("mad_outlier_pct", prof.get("mad_outlier_pct", 0)) or 0)
        ps = float(row.get("profile_score", prof.get("profile_score", 0)) or 0)
        severity = int8_fp16_budget_analyze_severity(
            kurtosis=k,
            outlier_ratio=o,
            abs_max=m,
            tunables=tunables_dict,
            is_hard_veto=is_hv,
            layer_name=name,
            mad_outlier_pct=mad,
            profile_score=ps,
        )

        if name in cache:
            v4_mse = float(cache[name])
        else:
            if trial_optimizer is None:
                skipped_no_v4.append(name)
                continue
            imp = _dualmonitor_channel_importance(dual_monitors, name)
            try:
                v4_mse = _measure_v4_mse_absmax_int8(
                    weight=mod.weight.data,
                    importance=imp,
                    optimizer=trial_optimizer,
                )
                cache[name] = v4_mse
                measured_fresh += 1
            except Exception as e:
                print(f"    [FP16 budget] V4 MSE failed {name}: {e} -> INT8")
                skipped_no_v4.append(name)
                continue
            torch.cuda.empty_cache()

        measured.append((name, dm_sens, v4_mse, severity, extra))

    # Model-specific auto analysis → auto-optimal ranking branches.
    # DualMonitor / analyze / V4 triples for THIS checkpoint drive continuous
    # knobs (infinite branches). Unified family median/geom floors are banned.
    veto_mask_pre = [name in hard_veto_layers for name, *_ in measured]
    measured, branch_repairs, branch_profile = apply_fp16_infinite_ranking_branches(
        measured, veto_mask_pre,
    )
    print(
        f"  [Infinite branch profile] "
        f"cv(s/v/m)={branch_profile['cv_sens']:.4g}/"
        f"{branch_profile['cv_sev']:.4g}/{branch_profile['cv_mse']:.4g} "
        f"align(s/v/m)={branch_profile['align_sens']:.3f}/"
        f"{branch_profile['align_sev']:.3f}/{branch_profile['align_mse']:.3f} "
        f"dm_starvation={branch_profile['dm_starvation']:.3f} "
        f"γ_sib/blend={branch_profile['gamma_sibling']:.4g}/"
        f"{branch_profile['gamma_blend']:.4g} "
        f"mismatch_gain={branch_profile['mismatch_gain']:.4g} "
        f"repairs={len(branch_repairs)}"
    )
    if branch_repairs:
        for _r in branch_repairs[:16]:
            print(
                f"    [{_r.get('branch', '?')}] {_r['name']}: "
                f"dm={_r.get('dm_sens', float('nan')):.6g} → "
                f"rank={_r.get('ranking_sens', float('nan')):.6g}"
                + (
                    f" skew={_r['skew']:.4g} str={_r['strength']:.4g}"
                    if "skew" in _r else
                    f" excess={_r.get('excess', float('nan')):.4g}"
                )
            )

    # Per-checkpoint combinator from MEASURED sens/sev/mse for THIS model
    # (auto analysis → auto-optimal priority weights; not a fixed formula).
    # Pass Hard VETO masks so anti-aligned axes (e.g. DualMonitor sens that
    # elevates sev=0 layers while demoting analyze VETO) fade automatically.
    sens_all = [float(row[1]) for row in measured]
    sev_all = [float(row[3]) for row in measured]
    mse_all = [float(row[2]) for row in measured]
    veto_mask = [row[0] in hard_veto_layers for row in measured]
    sens_meas = [v for v in sens_all if v > 0]
    sev_meas = list(sev_all)
    mse_meas = [v for v in mse_all if v > 0]
    s_p50 = _safe_percentile(sens_meas, 50.0) if len(sens_meas) >= 2 else 0.0
    s_iqr = _robust_iqr(sens_meas) if len(sens_meas) >= 4 else 0.0
    v_p50 = _safe_percentile(sev_meas, 50.0) if len(sev_meas) >= 2 else 0.0
    v_iqr = _robust_iqr(sev_meas) if len(sev_meas) >= 4 else 0.0
    m_p50 = _safe_percentile(mse_meas, 50.0) if len(mse_meas) >= 2 else 0.0
    m_iqr = _robust_iqr(mse_meas) if len(mse_meas) >= 4 else 0.0
    combinator = derive_priority_combinator(
        s_iqr, v_iqr, m_iqr, s_p50, v_p50, m_p50,
        sens_vals=sens_all,
        sev_vals=sev_all,
        mse_vals=mse_all,
        is_hard_veto=veto_mask,
    )
    _as = combinator.get("align_sens")
    _av = combinator.get("align_sev")
    _am = combinator.get("align_mse")
    _align_txt = (
        f" align(sens/sev/mse)="
        f"{(_as if _as is not None else float('nan')):.3f}/"
        f"{(_av if _av is not None else float('nan')):.3f}/"
        f"{(_am if _am is not None else float('nan')):.3f}"
        if _as is not None
        else ""
    )
    print(
        f"  [Autonomous priority] form={combinator['form']} "
        f"w(sens/sev/mse)={combinator['w_sens']:.3f}/"
        f"{combinator['w_sev']:.3f}/{combinator['w_mse']:.3f} "
        f"refs=({combinator['sens_ref']:.4g}/"
        f"{combinator['sev_ref']:.4g}/{combinator['mse_ref']:.4g})"
        f"{_align_txt}"
    )

    candidates: list[tuple[float, float, float, float, int, str]] = []
    for name, dm_sens, v4_mse, severity, extra in measured:
        priority = int8_fp16_budget_priority(
            dm_sens, v4_mse, severity, combinator=combinator,
        )
        candidates.append((priority, v4_mse, severity, dm_sens, extra, name))

    # Priority continuous sibling branch from the SAME THIS-model profile
    # (not a second unified floor).
    candidates, prio_branch_repairs = apply_fp16_infinite_priority_branches(
        candidates, branch_profile,
    )
    if prio_branch_repairs:
        print(
            f"  [Infinite priority branches] repaired "
            f"{len(prio_branch_repairs)} under THIS family priority space:"
        )
        for _r in prio_branch_repairs[:12]:
            print(
                f"    {_r['name']}: prio {_r['priority_before']:.6g} → "
                f"{_r['priority_after']:.6g} "
                f"skew={_r['skew']:.4g} str={_r['strength']:.4g}"
            )

    candidates.sort(key=lambda x: (-x[0], x[4]))

    # Extreme fill inside the 300 MiB hard ceiling: priority order; if a
    # layer does not fit, skip and keep packing smaller remaining layers
    # (THIS model's auto-optimal set under the owner frame).
    selected: set = set()
    used = 0
    dropped: list[tuple[str, int, float, float, float, float]] = []
    kept_detail: list[tuple[str, int, float, float, float, float]] = []
    for priority, v4_mse, severity, dm_sens, extra, name in candidates:
        if used + extra <= budget_bytes:
            selected.add(name)
            used += extra
            kept_detail.append((name, extra, priority, v4_mse, severity, dm_sens))
        else:
            dropped.append((name, extra, priority, v4_mse, severity, dm_sens))

    demoted_veto = hard_veto_layers - selected
    # Auto-optimal FP16 set for THIS model (DualMonitor + analyze + V4).
    # Analyze VETO that win stay labeled VETO; DualMonitor winners are keep.
    hard_veto_out = hard_veto_layers & selected
    keep_out = set(selected)

    if used > budget_bytes:
        raise RuntimeError(
            f"[FP16 budget] selected set exceeds hard ceiling "
            f"{budget_mb:g} MiB: used={used / (1024 * 1024):.3f} MiB "
            f"({used} bytes > {budget_bytes}). Refusing to proceed."
        )

    stats = {
        "budget_mb": float(budget_mb),
        "budget_bytes": budget_bytes,
        "used_bytes": used,
        "used_mb": used / (1024 * 1024),
        "candidates": len(candidates),
        "pool": len(pool),
        "analyze_character_layers": len(char_table),
        "dm_sensitivity_layers": len(sens_by_name),
        "kept": len(keep_out),
        "dropped": len(dropped),
        "demoted_veto": len(demoted_veto),
        "skipped_no_weight": len(skipped_no_weight),
        "skipped_no_v4": len(skipped_no_v4),
        "measured_fresh_v4": measured_fresh,
        "priority_form": combinator["form"],
        "priority_weights": {
            "sens": combinator["w_sens"],
            "sev": combinator["w_sev"],
            "mse": combinator["w_mse"],
        },
        "priority_align": {
            "sens": combinator.get("align_sens"),
            "sev": combinator.get("align_sev"),
            "mse": combinator.get("align_mse"),
        },
        "ranking": "per_model_auto_analysis_infinite_branches_inside_300mib",
        "infinite_branch_profile": {
            "cv_sens": branch_profile.get("cv_sens"),
            "cv_sev": branch_profile.get("cv_sev"),
            "cv_mse": branch_profile.get("cv_mse"),
            "align_sens": branch_profile.get("align_sens"),
            "align_sev": branch_profile.get("align_sev"),
            "align_mse": branch_profile.get("align_mse"),
            "dm_starvation": branch_profile.get("dm_starvation"),
            "gamma_sibling": branch_profile.get("gamma_sibling"),
            "gamma_blend": branch_profile.get("gamma_blend"),
            "mismatch_gain": branch_profile.get("mismatch_gain"),
            "prio_sibling_gamma": branch_profile.get("prio_sibling_gamma"),
            "prio_blend_gamma": branch_profile.get("prio_blend_gamma"),
        },
        "infinite_ranking_branch_repairs": len(branch_repairs),
        "infinite_ranking_branch_detail": branch_repairs[:32],
        "infinite_priority_branch_repairs": len(prio_branch_repairs),
        "infinite_priority_branch_detail": prio_branch_repairs[:32],
        "hard_ceiling_mb": FP16_BUDGET_MB_HARD,
        "slack_bytes": max(budget_bytes - used, 0),
        "slack_mb": max(budget_bytes - used, 0) / (1024 * 1024),
        "dropped_detail": dropped[:40],
        "kept_detail": kept_detail[:40],
        "mse_cache_size": len(cache),
    }
    return keep_out, hard_veto_out, stats


def compute_int8_bias_delta(weight_fp, weight_dq, act_mean):
    """Bias correction delta for one INT8 layer.

    Cancels systematic output shift E[(W_q - W) x] ≈ (W_q - W) contracted with
    per-input-channel mean activation from calibration.

    Linear  weight (O, I):     delta[o] = sum_i err[o,i] * mu[i]
    Conv2d  weight (O, I, K, K): delta[o] = sum_{i,k,h} err[o,i,kh,kw] * mu[i]
    """
    if act_mean is None:
        return None
    err = (weight_dq.float() - weight_fp.float())
    mu = act_mean.float().to(device=err.device)
    if err.ndim == 2:
        # Linear: (O, I) @ (I,) -> (O,)
        if mu.numel() != err.shape[1]:
            return None
        return err @ mu
    if err.ndim == 4:
        # Conv2d: sum over in/spatial with per-in-channel mu
        if mu.numel() != err.shape[1]:
            return None
        return (err * mu.view(1, -1, 1, 1)).sum(dim=(1, 2, 3))
    return None


def pack_int8_tensorwise(weight, asymmetric: bool = True, amax: float | None = None):
    """Pack a weight tensor to symmetric storage int8 + scalar scale.

    `amax` (float) is the pack clip target (INT8: absmax). When provided, it
    is used instead of recomputing absmax from the tensor.

    asymmetric=True (Card 2):
      mid = (w_min + w_max) / 2
      scale = max(|w_max - mid|, |w_min - mid|) / 127
      q = round((W - mid) / scale).clamp(-127, 127)
      Loader reconstructs q*scale; mid is recovered via bias correction.

    asymmetric=False:
      amax_eff = amax if provided else absmax
      scale = amax_eff / 127, q = round(W / scale).clamp(-127, 127)  (classic)
    """
    w = weight.float()
    if asymmetric:
        w_min = w.min()
        w_max = w.max()
        mid = 0.5 * (w_min + w_max)
        half = torch.maximum(w_max - mid, mid - w_min).clamp_min(1e-6)
        scale = (half / 127.0).item()
        q = ((w - mid) / scale).round().clamp(-127, 127).to(torch.int8)
        return q, scale, mid.item()
    if amax is None:
        amax = float(w.abs().max().clamp_min(1e-6).item())
    else:
        amax = float(max(abs(amax), 1e-6))
    scale = (amax / 127.0)
    q = (w / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale, 0.0


def pack_int8_channelwise(weight, amax=None):
    """Pack weight to int8 + per-out-channel scale (Card 3).

    Metadata format stays ``int8_tensorwise``. Scale is stored in a shape that
    broadcasts with weight under kitchen ``dequantize_int8_simple`` (``q * scale``):

    - Linear ``(O, I)`` → ``weight_scale`` shape ``(O, 1)``
    - Conv2d ``(O, C, H, W)`` → ``weight_scale`` shape ``(O, 1, 1, 1)``

    A flat ``(O,)`` scale is NOT safe for 4D weights: PyTorch aligns from the
    right, so ``(O,C,H,W) * (O,)`` collides on the last dim.
    """
    w = weight.float()
    if amax is None:
        reduce_dims = tuple(range(1, w.dim()))
        amax = w.abs().amax(dim=reduce_dims)
    else:
        amax = amax.float().to(device=w.device)
        if amax.ndim == 0:
            amax = amax.reshape(1).expand(w.shape[0])
        elif amax.numel() == 1 and w.shape[0] > 1:
            amax = amax.reshape(1).expand(w.shape[0])
        elif amax.numel() != w.shape[0]:
            raise ValueError(
                f"channelwise amax numel={amax.numel()} != out_channels={w.shape[0]}"
            )
    amax = torch.clamp(amax.reshape(-1), min=1e-6)
    scale = amax / 127.0
    if w.dim() == 4:
        scale_view = scale.view(-1, 1, 1, 1)
        amax_view = amax.view(-1, 1, 1, 1)
    elif w.dim() == 2:
        scale_view = scale.view(-1, 1)
        amax_view = amax.view(-1, 1)
    else:
        raise ValueError(f"unsupported weight ndim={w.dim()} for channelwise INT8")
    clamped = torch.clamp(w, -amax_view, amax_view)
    q = (clamped / scale_view).round().clamp(-127, 127).to(torch.int8)
    return q, scale_view, scale_view


# DualMonitor MUST be defined before calibration hooks (NameError if missing).
# HEAD historically called the class from hook_fn without defining it.
class DualMonitor:
    """Per-layer calibration monitor for THIS checkpoint (auto analysis input).

    Accumulates output variance (sensitivity), channel importance, and
    activation moments used by V4 Importance and FP16 budget ranking.
    """

    def __init__(self):
        self.output_sum = 0.0
        self.output_sq_sum = 0.0
        self.count = 0
        self.channel_importance = None
        # Signed per-channel input mean for INT8 bias correction:
        #   bias_delta ≈ (W_q - W) @ E[x]
        self.channel_act_mean = None
        # Per-input-channel second moment E[x_i^2] for damage calculation:
        #   damage_l ≈ sum_i (ΔW^2)[*,i,*] · E[x_i^2]  (pre-grad factor)
        self.channel_act_sq_mean = None
    
    def update(self, input_tensor, output_tensor):
        with torch.no_grad():
            out_detached = output_tensor.detach().float()
            out_clamped = torch.clamp(out_detached, -65504.0, 65504.0)
            mean_val = out_clamped.mean().item()
            sq_mean_val = (out_clamped ** 2).mean().item()
            import math
            if math.isfinite(mean_val) and math.isfinite(sq_mean_val):
                self.output_sum += mean_val
                self.output_sq_sum += sq_mean_val
            inp_detached = input_tensor.detach().float()
            if inp_detached.dim() == 4:
                current_imp = inp_detached.abs().mean(dim=(0, 2, 3))
                current_act = inp_detached.mean(dim=(0, 2, 3))
                current_sq = (inp_detached ** 2).mean(dim=(0, 2, 3))
            elif inp_detached.dim() == 3:
                current_imp = inp_detached.abs().mean(dim=(0, 1))
                current_act = inp_detached.mean(dim=(0, 1))
                current_sq = (inp_detached ** 2).mean(dim=(0, 1))
            elif inp_detached.dim() == 2:
                current_imp = inp_detached.abs().mean(dim=0)
                current_act = inp_detached.mean(dim=0)
                current_sq = (inp_detached ** 2).mean(dim=0)
            else:
                current_imp = torch.ones(1, device=inp_detached.device, dtype=torch.float32)
                current_act = torch.zeros(1, device=inp_detached.device, dtype=torch.float32)
                current_sq = torch.ones(1, device=inp_detached.device, dtype=torch.float32)
            if self.channel_importance is None:
                self.channel_importance = current_imp
                self.channel_act_mean = current_act
                self.channel_act_sq_mean = current_sq
            else:
                self.channel_importance = (
                    self.channel_importance * self.count + current_imp
                ) / (self.count + 1)
                self.channel_act_mean = (
                    self.channel_act_mean * self.count + current_act
                ) / (self.count + 1)
                self.channel_act_sq_mean = (
                    self.channel_act_sq_mean * self.count + current_sq
                ) / (self.count + 1)
            self.count += 1

    def get_sensitivity(self):
        if self.count == 0:
            return 0.0
        mean = self.output_sum / self.count
        variance = (self.output_sq_sum / self.count) - mean ** 2
        import math
        return variance if math.isfinite(variance) else 0.0

    def get_input_second_moment(self):
        """Per-input-channel E[x_i^2] accumulated during calibration (float32, CPU)."""
        if self.count == 0 or self.channel_act_sq_mean is None:
            return None
        return self.channel_act_sq_mean.detach().float().cpu()


dual_monitors = {}


def hook_fn(module, input, output, name):
    if name not in dual_monitors:
        dual_monitors[name] = DualMonitor()
    dual_monitors[name].update(input[0], output)


def _remap_profile_to_diffusers(model_profile: dict, comfyui_to_diffusers_map: dict) -> dict:
    """Map analyze JSON keys (ComfyUI .weight) to Diffusers module names for named_modules()."""
    if not model_profile or not comfyui_to_diffusers_map:
        return model_profile
    remapped = {}
    unmapped = 0
    for comfy_key, val in model_profile.items():
        if not isinstance(val, dict):
            continue
        diff_key = comfyui_to_diffusers_map.get(comfy_key)
        if diff_key is None:
            unmapped += 1
            continue
        mod_name = diff_key[:-7] if diff_key.endswith(".weight") else diff_key
        remapped[mod_name] = val
    if unmapped:
        print(f"  [Profile Remap] {unmapped} Comfy keys had no diffusers mapping (skipped)")
    print(f"  [Profile Remap] {len(remapped)} diffusers module profile entries")
    return remapped


def derive_hswq_strategy_int8(model_profile, veto_tunables: SdxlVetoTunables | None = None):
    """
    SDXL V3.0 INT8: Alpha/Beta from profile + absmax pack + V4 FP16 ranking.

    - search_low: 1.0 → pack amax = absmax (obvious for symmetric INT8).
    - V4 weighted histogram: FP16 protection candidate ranking
      (estimated_mse @ absmax with INT8Quantizer; budget truncates).
      Soft gray-zone VETO release is secondary reuse of the same MSE.
    - alpha/beta: alpha_auto from THIS multi-axis analyze character
      (kurtosis∪outlier∪magnitude → Full-SVD×RMS); DualMonitor Importance
      multiplies the hybrid map when present. No fixed mix / no SVD off.
    - hard_veto: thresholds from derive_veto_tunables_int8 (this checkpoint).
    """
    if model_profile:
        sample_key = next(iter(model_profile))
        profile_prefix = ""
        for pfx in _SDXL_PROFILE_PREFIXES:
            if pfx and sample_key.startswith(pfx):
                profile_prefix = pfx
                break
        if profile_prefix:
            normalized_profile = {}
            for key, val in model_profile.items():
                stripped_key = (
                    key[len(profile_prefix):] if key.startswith(profile_prefix) else key
                )
                normalized_profile[stripped_key] = val
            model_profile = normalized_profile
            print(
                f"  [Profile Normalize] Stripped prefix '{profile_prefix}' "
                f"from {len(normalized_profile)} profile keys."
            )

    if veto_tunables is None:
        # Owner hard ceiling 300 MiB — auto knobs fill inside this frame.
        veto_tunables = resolve_veto_tunables(
            model_profile or {},
            fp16_budget_mb=FP16_BUDGET_MB_HARD,
        )

    print(
        "  [INT8 pack] absmax (search_low=1.0 — natural INT8); "
        "[V4 histogram] FP16 protection candidate ranking @ absmax"
    )

    def get_dynamic_search_low(name, weight_tensor):
        # Natural INT8 pack point. V4 does not choose pack amax.
        return 1.0

    if model_profile:
        all_k = [p.get("kurtosis", 0) for p in model_profile.values() if isinstance(p, dict)]
        all_o = [p.get("outlier_ratio", 0) for p in model_profile.values() if isinstance(p, dict)]
        all_m = [p.get("abs_max", 0) for p in model_profile.values() if isinstance(p, dict)]
        avg_k = np.mean(all_k) if all_k else 0
        avg_o = np.mean(all_o) if all_o else 0
        avg_m = np.mean(all_m) if all_m else 0
        print(f"  [Profile Stats INT8] Avg Kurtosis: {avg_k:.2f}, Avg OutlierRatio: {avg_o:.2f}, Avg AbsMax: {avg_m:.2f}")

    # alpha = SVD-leverage MIX WEIGHT from THIS multi-axis character (k∪o∪m).
    # Full-SVD always executes (never skip V4); alpha only scales leverage vs
    # RMS in the hybrid map. DualMonitor Imp multiplies that map when present.
    # No fixed cut, no model-name rule, no "alpha>0 else skip SVD" thought-stop.
    alpha = float(veto_tunables.alpha_auto)
    beta = 1.0 - alpha

    print(
        f"  [Dynamic Alpha/Beta INT8] alpha={alpha:.3f}, beta={beta:.3f} "
        f"(analyze k∪o∪m → Full-SVD×RMS; Imp multiplies when present)"
    )

    hard_veto_layers = set()
    if model_profile:
        for name, prof in model_profile.items():
            if isinstance(prof, dict):
                k = prof.get("kurtosis", 0)
                m = prof.get("abs_max", 0)
                o = prof.get("outlier_ratio", 0)
                # VETO thresholds = analyze_sdxl_distribution.derive_veto_tunables_int8
                # only (this checkpoint's distribution). No hardcoded floors.
                is_extreme_divergence = o > veto_tunables.extreme_outlier
                is_extreme_kurtosis = k > veto_tunables.extreme_kurtosis
                is_huge_magnitude = m > veto_tunables.huge_magnitude
                if is_extreme_divergence or is_extreme_kurtosis or is_huge_magnitude:
                    layer_base_name = name.replace(".weight", "") if name.endswith(".weight") else name
                    hard_veto_layers.add(layer_base_name)
                    reasons = []
                    if is_extreme_kurtosis:
                        reasons.append(f"k={k:.1f}")
                    if is_extreme_divergence:
                        reasons.append(f"o={o:.1f}")
                    if is_huge_magnitude:
                        reasons.append(f"m={m:.2f}")
                    print(f"    VETO: {layer_base_name} [{', '.join(reasons)}]")

    print(
        f"  [Static Profile VETO INT8] Identified {len(hard_veto_layers)} layers "
        "with extreme distribution (Unquantizable in INT8)."
    )
    return alpha, beta, get_dynamic_search_low, hard_veto_layers


def resolve_weights_path(raw_path: str, script_dir: str) -> tuple[str, list[str]]:
    """Resolve .safetensors path when CWD differs from repo root (Docker/CI).

    Order: HSWQ_SDXL_INPUT, SDXL_INPUT_MODEL, abspath(raw), script_dir/raw, script_dir/basename(raw).
    Returns (first existing file path, or abspath(raw) if none), list of tried paths.
    """
    tried: list[str] = []
    candidates: list[str] = []
    for env_key in ("HSWQ_SDXL_INPUT", "SDXL_INPUT_MODEL"):
        v = (os.environ.get(env_key) or "").strip()
        if v:
            candidates.append(os.path.abspath(v))
    if os.path.isabs(raw_path):
        candidates.append(os.path.normpath(raw_path))
    else:
        candidates.append(os.path.abspath(raw_path))
        candidates.append(os.path.normpath(os.path.join(script_dir, raw_path)))
        candidates.append(os.path.normpath(os.path.join(script_dir, os.path.basename(raw_path))))
    seen: set[str] = set()
    for p in candidates:
        if not p or p in seen:
            continue
        seen.add(p)
        tried.append(p)
        if os.path.isfile(p):
            return p, tried
    return os.path.abspath(raw_path), tried


def main():
    parser = argparse.ArgumentParser(description="SDXL INT8 Quantization - HSWQ V3.0 (Autonomous Engine, INT8 per-tensor)")
    parser.add_argument("--input", type=str, required=True, help="Path to input safetensors model")
    parser.add_argument("--output", type=str, required=True, help="Path to output safetensors model")
    parser.add_argument("--calib_file", type=str, required=True, help="Path to calibration prompts text file")
    parser.add_argument(
        "--num_calib_samples",
        type=int,
        default=32,
        help="Calibration samples (How-to / r32 recommended: 32)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="Denoising steps per calibration sample (How-to example: 25)",
    )
    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=0.0,
        help="Must be 0 (r0). FP16 protection is selected by --fp16_budget_mb "
             "ranking (DualMonitor sensitivity + V4 MSE + analyze severity). "
             "DualMonitor is NEVER used to invent or gate this flag.",
    )
    parser.add_argument(
        "--fp16_budget_mb",
        type=float,
        default=FP16_BUDGET_MB_HARD,
        help="Owner hard ceiling: must be exactly 300 MiB FP16 overhead vs "
             "all-INT8. Per-model auto analysis / auto-optimal settings fill "
             "this frame only — never redefine or exceed it. "
             "Extra cost = 1 byte per weight element.",
    )
    parser.add_argument("--comfy_path", type=str, help="Path to ComfyUI root directory (optional, will auto-detect)")
    parser.add_argument("--profile", type=str, help="Path to distribution profile JSON (optional, will auto-generate if missing)")
    parser.add_argument(
        "--bias_correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply activation-mean bias correction to INT8 layers (default: on). "
             "Cancels E[(W_q-W)x] into bias; no extra FP16 keep, format unchanged.",
    )
    parser.add_argument(
        "--bias_correction_top_ratio",
        type=float,
        default=None,
        help="Fraction of INT8 layers (by DualMonitor sensitivity, highest first) "
             "that receive bias correction. Default: None = autonomous "
             "(derive_int8_autonomous_tunables: Tukey lower fence of THIS "
             "DualMonitor sensitivity — scope = fraction at/above fence).",
    )
    parser.add_argument(
        "--asymmetric_int8",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pack INT8 around (min+max)/2 (default: off). Still int8_tensorwise; "
             "mid is absorbed by bias correction when enabled. "
             "Mutually exclusive with --per_channel_int8.",
    )
    parser.add_argument(
        "--per_channel_int8",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Card 3 (DEFAULT OFF). Per-output-channel amax; weight_scale "
             "(Out,1) Linear / (Out,1,1,1) Conv; format tag stays int8_tensorwise. "
             "Default OFF = per-tensor absmax pack (search_low=1.0); "
             "V4 histogram ranks FP16 keep candidates (not pack amax). "
             "Mutually exclusive with --asymmetric_int8.",
    )
    args = parser.parse_args()

    if args.asymmetric_int8 and args.per_channel_int8:
        print(
            "[FATAL] --asymmetric_int8 (Card 2) and --per_channel_int8 (Card 3) "
            "are mutually exclusive. Disable one of them."
        )
        sys.exit(1)

    # 300 MiB hard ceiling: auto analysis / auto-optimal settings only inside.
    try:
        args.fp16_budget_mb = _require_fp16_budget_mb_hard(args.fp16_budget_mb)
    except ValueError as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    # r0 fixed. DualMonitor sensitivity is used ONLY in
    # _apply_fp16_budget_cap (extreme fill inside 300 MiB) — never to
    # invent or gate keep_ratio.
    _bc_top_override = args.bias_correction_top_ratio
    if abs(float(args.keep_ratio)) > 1e-12:
        print(
            f"[FATAL] keep_ratio must be 0 (r0); got {args.keep_ratio}. "
            f"FP16 protection = per-model auto analysis inside "
            f"{FP16_BUDGET_MB_HARD:g} MiB hard ceiling."
        )
        sys.exit(1)
    args.keep_ratio = 0.0

    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_input_arg = args.input
    resolved_input, tried_inputs = resolve_weights_path(raw_input_arg, script_dir)
    if not os.path.isfile(resolved_input):
        print("[FATAL] Input weights file not found.")
        print(f"  --input: {raw_input_arg!r}")
        print("  Tried:")
        for p in tried_inputs:
            print(f"    - {p}")
        print("  Hint: place the .safetensors next to the repo, pass an absolute path,")
        print("        or set HSWQ_SDXL_INPUT / SDXL_INPUT_MODEL to the model file.")
        sys.exit(1)
    cli_abs = os.path.normpath(os.path.abspath(os.path.expanduser(raw_input_arg)))
    if os.path.normpath(resolved_input) != cli_abs:
        print(f"[*] Resolved --input: {raw_input_arg!r} -> {resolved_input}")
    args.input = resolved_input

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("HSWQ V3.0 SDXL INT8 Pure Autonomous Engine (Environment-Aware Analysis)")
    print("=" * 60)

    # --- ComfyUI Path Setup ---
    comfy_path = args.comfy_path
    if comfy_path is None:
        comfy_path = os.environ.get("COMFYUI_PATH", os.path.join(os.getcwd(), "ComfyUI"))

    if os.path.exists(comfy_path):
        if comfy_path not in sys.path:
            sys.path.insert(0, comfy_path)

    # --- 1. Locate Analysis Script & Profile --- (Environment-Agnostic)
    analyze_script = os.path.join(script_dir, "analyze", "analyze_sdxl_distribution.py")
    if not os.path.exists(analyze_script):
        print(f"[FATAL] SDXL profile script not found: {analyze_script}")
        sys.exit(1)

    input_abs = os.path.abspath(args.input)
    input_root = os.path.splitext(os.path.basename(args.input))[0]

    profile_path = args.profile
    is_auto = False
    if not profile_path:
        profile_path = os.path.join(script_dir, f"{input_root}_distribution_profile.json")
        is_auto = True

    should_run_analysis = is_auto or not os.path.exists(profile_path)

    if should_run_analysis:
        if os.path.exists(analyze_script):
            print(f"[*] Executing mandated distribution analysis (No skip policy):")
            print(f"    Script: {analyze_script}")
            print(f"    Input:  {input_abs}")
            print(f"    Result: {profile_path}")
            subprocess.run(
                [sys.executable, analyze_script, "--input", input_abs, "--output", profile_path],
                check=True,
            )
        else:
            print(f"[*] Warning: Analysis script NOT found. (Expected: {analyze_script})")
            print("    Will proceed with internal backup strategy (on-the-fly calc).")

    model_profile = {}
    profile_summary = {}
    if os.path.exists(profile_path):
        print(f"[*] Loading Analysis Data: {profile_path}")
        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)
            if isinstance(profile_data, dict):
                profile_summary = profile_data.get("summary", {}) or {}
                model_profile = profile_data.get("layers", profile_data)
            else:
                model_profile = profile_data

    # --- 2. SDXL UNet Load then profile remap (Diffusers module names) ---
    pipeline, original_state_dict, comfyui_to_diffusers_map = load_unet_from_safetensors(
        args.input, device
    )
    model_profile = _remap_profile_to_diffusers(model_profile, comfyui_to_diffusers_map)
    model = pipeline.unet
    _norm_profile = {k: v for k, v in model_profile.items() if isinstance(v, dict)}
    veto_tunables = resolve_veto_tunables(
        _norm_profile, profile_summary,
        dual_monitors=None,  # sens computed later after calibration
        fp16_budget_mb=float(args.fp16_budget_mb),
    )
    # Fill autonomous bc_top if user did not override.
    if _bc_top_override is None:
        args.bias_correction_top_ratio = float(veto_tunables.bias_correction_top_ratio)
        print(
            f"  [Autonomous bias_correction_top_ratio] "
            f"{args.bias_correction_top_ratio:.2f} "
            f"(THIS DualMonitor Tukey lower-fence scope — auto-optimal)"
        )
    print(f"  [Veto Tunables INT8] {veto_tunables.as_dict()}")
    alpha, beta, get_layer_search_low, hard_veto_layers = derive_hswq_strategy_int8(
        model_profile,
        veto_tunables,
    )

    print("  [V3.0 SDXL INT8 Autonomous VETO] Structural + per-projection attn + key-pattern + supplemental.")
    structural_veto = _compute_structural_veto(model, hard_veto_layers, _norm_profile)
    if structural_veto:
        hard_veto_layers = hard_veto_layers.union(structural_veto)
        print(f"  [Structural VETO] Added {len(structural_veto)} unique-shape layers (total VETO: {len(hard_veto_layers)}).")
    proj_veto = _compute_sdxl_per_projection_attn_veto(
        model,
        hard_veto_layers,
        veto_tunables,
        _norm_profile,
    )
    if proj_veto:
        hard_veto_layers = hard_veto_layers.union(proj_veto)
        print(f"  [Per-Projection VETO] Added {len(proj_veto)} attn layers (total VETO: {len(hard_veto_layers)}).")
    # INT8-only: MAD floors auto from profile distribution (no per-model settings).
    mad_veto = _compute_sdxl_int8_mad_attn_veto(
        model, hard_veto_layers, veto_tunables, _norm_profile
    )
    if mad_veto:
        hard_veto_layers = hard_veto_layers.union(mad_veto)
        print(f"  [INT8 MAD VETO] total VETO after MAD fill: {len(hard_veto_layers)}.")
    keypattern_veto = _compute_sdxl_keypattern_veto(
        model, hard_veto_layers, veto_tunables, _norm_profile
    )
    if keypattern_veto:
        hard_veto_layers = hard_veto_layers.union(keypattern_veto)
        print(f"  [Key-Pattern VETO] hard_veto total: {len(hard_veto_layers)}.")

    print("Preparing calibration (Dual Monitor hooks)...")
    dual_monitors.clear()
    handles, target_modules = [], []
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
    if args.num_calib_samples != 32 or args.num_inference_steps != 25:
        print(
            "  [WARN] How-to / r32 recipe is num_calib_samples=32, "
            "num_inference_steps=25. DualMonitor importance for V4 FP16 "
            "ranking should follow that calibration; current args differ."
        )
    pipeline.set_progress_bar_config(disable=False)
    generator = torch.Generator(device=device).manual_seed(42)
    
    # Calibration for DualMonitor Importance only (V4 ranking). No UNet
    # reservoir / grad-damage capture — that path was the priority hand-wave.
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

    print("  [Calib] DualMonitor Importance ready for V4 full-pool priority.")

    print("\nAnalyzing layer sensitivity [INT8] — V4 calib FP16 cands + analyze VETO...")

    _supp = _autonomous_supplemental_veto(model, hard_veto_layers, _norm_profile, veto_tunables)
    if _supp:
        hard_veto_layers = hard_veto_layers.union(_supp)
        print(f"  [Supplemental VETO] Added {len(_supp)} layers (total VETO: {len(hard_veto_layers)}).")

    # Re-derive autonomous knobs now that DualMonitor calibration exists.
    # Refresh α/β from THIS multi-axis alpha_auto — never keep pre-calib
    # stale mix (handwave that skips analyze character after Sensitivity).
    veto_tunables = resolve_veto_tunables(
        _norm_profile,
        profile_summary,
        dual_monitors=dual_monitors,
        fp16_budget_mb=float(args.fp16_budget_mb),
    )
    alpha = float(veto_tunables.alpha_auto)
    beta = 1.0 - alpha
    print(
        f"  [Dynamic Alpha/Beta INT8 after DualMonitor] "
        f"alpha={alpha:.4f}, beta={beta:.4f} "
        f"(analyze k∪o∪m → Full-SVD×RMS; Imp×Sens×V4 MSE fill 300 MiB)"
    )
    if _bc_top_override is None:
        args.bias_correction_top_ratio = float(veto_tunables.bias_correction_top_ratio)
        print(
            f"  [Autonomous bias_correction_top_ratio after DualMonitor] "
            f"{args.bias_correction_top_ratio:.2f}"
        )

    mse_cache: dict = {}
    dynamic_keep_layers, mse_cache = _build_v4_calib_fp16_candidates(
        model=model,
        dual_monitors=dual_monitors,
        target_modules=target_modules,
        hard_veto_layers=hard_veto_layers,
        mse_cache=mse_cache,
        alpha=alpha,
        beta=beta,
        device=device,
    )
    ranking_source = "v4_histogram_calib"
    num_keep_dynamic = len(dynamic_keep_layers)  # report only; not a pre-cut

    # ALL: V4-calib-scored U analyze VETO -> full priority in budget pass.
    keep_layers = dynamic_keep_layers.union(hard_veto_layers)

    # Soft gray-zone: optional analyze soft-VETO release via same V4 MSE
    # (secondary; primary V4 use was FP16 candidate scoring above).
    release_cands = _collect_mse_release_candidates(
        hard_veto_layers, structural_veto, _norm_profile, model, veto_tunables
    )
    if keypattern_veto:
        release_cands -= keypattern_veto
    # Reuse V4 calib mse_cache; grayzone may extend it (do not wipe).
    if release_cands:
        hard_veto_layers, keep_layers, mse_cache = _mse_grayzone_veto_reassessment(
            scope_label="V3.0 SDXL INT8",
            hard_veto_layers=hard_veto_layers,
            keep_layers=keep_layers,
            outlier_only_veto=release_cands,
            target_modules=target_modules,
            model=model,
            _norm_profile=_norm_profile,
            get_layer_search_low=get_layer_search_low,
            alpha=alpha,
            beta=beta,
            device=device,
            tunables=veto_tunables,
            dual_monitors=dual_monitors,
            mse_cache=mse_cache,
        )

    # Map integrity: KEEP names must be Comfy-mapped. Live Diffusers
    # upsamplers.0.conv come from this UNet (not a hardcoded layer list).
    mapped_weight_modules = set()
    for dk in comfyui_to_diffusers_map.values():
        if isinstance(dk, str) and dk.endswith(".weight"):
            mapped_weight_modules.add(dk[:-7])
    for _name, _mod in model.named_modules():
        if not _name.endswith("upsamplers.0.conv"):
            continue
        if not isinstance(_mod, torch.nn.Conv2d):
            continue
        if _name not in mapped_weight_modules:
            raise RuntimeError(
                f"Map integrity FATAL: Diffusers module {_name!r} exists on UNet "
                f"but has no Comfy map entry — fix unet_to_diffusers_mapping "
                f"(refuse invent / refuse leave unmapped)"
            )
        print(f"  [Map integrity] {_name} mapped")
    orphan_before = keep_layers - mapped_weight_modules
    if orphan_before:
        print(
            f"  [Map integrity] FATAL: {len(orphan_before)} keep name(s) not in "
            f"Comfy↔diffusers map (must be 0 before budget):"
        )
        for n in sorted(orphan_before):
            print(f"    unmapped: {n}")
        raise RuntimeError(
            f"Map integrity: {len(orphan_before)} unmapped keep layer(s); "
            f"refuse exclude — fix unet_to_diffusers_mapping"
        )

    # Hard ceiling: FP16 overhead vs all-INT8 must stay within budget.
    # Auto-optimal over ALL of: V4-calib FP16 candidates U analyze VETO
    # U analyze fence-crossers (priority = V4 MSE x analyze severity).
    keep_before_budget = len(keep_layers)
    veto_before_budget = len(hard_veto_layers)
    keep_layers, hard_veto_layers, budget_stats = _apply_fp16_budget_cap(
        model,
        keep_layers,
        hard_veto_layers,
        budget_mb=float(args.fp16_budget_mb),
        norm_profile=_norm_profile,
        veto_tunables=veto_tunables,
        dual_monitors=dual_monitors,
        mse_cache=mse_cache,
        alpha=alpha,
        beta=beta,
        device=device,
    )
    dynamic_keep_layers = dynamic_keep_layers & keep_layers

    orphan_keep = keep_layers - mapped_weight_modules
    if orphan_keep:
        print(
            f"  [FP16 keep] FATAL map mismatch: {len(orphan_keep)} keep name(s) "
            f"still unmapped after budget (must be 0; will not drop):"
        )
        for n in sorted(orphan_keep):
            print(f"    unmapped: {n}")
        raise RuntimeError(
            f"Map integrity: {len(orphan_keep)} keep layer(s) unmapped after budget; "
            f"refusing to drop — fix unet_to_diffusers_mapping"
        )
    print("  [Map integrity] orphan_keep=0 (all FP16 keep names are Comfy-mapped).")
    print(
        f"\n  [FP16 budget] ranking={budget_stats.get('ranking')} "
        f"hard_ceiling={budget_stats['budget_mb']:.1f} MiB "
        f"(extra vs all-INT8); used={budget_stats['used_mb']:.1f} MiB "
        f"slack={budget_stats.get('slack_mb', 0):.2f} MiB "
        f"| pool={budget_stats.get('pool', budget_stats['candidates'])} "
        f"| analyze_char={budget_stats.get('analyze_character_layers', '?')} "
        f"| keep {keep_before_budget}→{budget_stats['kept']} "
        f"| VETO {veto_before_budget}→{len(hard_veto_layers)} "
        f"| dropped={budget_stats['dropped']} "
        f"(demoted_veto={budget_stats['demoted_veto']}, "
        f"v4_fresh={budget_stats.get('measured_fresh_v4', 0)}, "
        f"no_v4={budget_stats.get('skipped_no_v4', 0)})"
    )
    if budget_stats.get("kept_detail"):
        print("  [FP16 budget] top kept (name | MiB | priority | V4_mse | analyze_sev | dm_sens):")
        for row in budget_stats["kept_detail"][:15]:
            _kn, _kextra, _kp, _kmse, _ksev = row[0], row[1], row[2], row[3], row[4]
            _ksens = row[5] if len(row) > 5 else 0.0
            print(
                f"    KEEP {_kn} | {_kextra / (1024*1024):.2f} MiB | "
                f"prio={_kp:.6g} | mse={_kmse:.6g} | sev={_ksev:.3f} | dm_sens={_ksens:.6g}"
            )
    if budget_stats.get("dropped_detail"):
        print("  [FP16 budget] lowest-priority drops (name | MiB | priority | V4_mse | analyze_sev | dm_sens):")
        for row in budget_stats["dropped_detail"][:20]:
            _dn, _dextra, _dp, _dmse, _dsev = row[0], row[1], row[2], row[3], row[4]
            _dsens = row[5] if len(row) > 5 else 0.0
            print(
                f"    DROP {_dn} | {_dextra / (1024*1024):.2f} MiB | "
                f"prio={_dp:.6g} | mse={_dmse:.6g} | sev={_dsev:.3f} | dm_sens={_dsens:.6g}"
            )

    non_veto_total = len([n for n in target_modules if n not in hard_veto_layers])
    print(f"\nTotal layers: {len(target_modules)} (Non-VETO pool: {non_veto_total})")
    print(
        f"FP16 protection: DualMonitor + analyze + V4 → "
        f"per-model auto analysis / extreme auto-optimal keep "
        f"({ranking_source}); r0; hard_ceiling="
        f"{FP16_BUDGET_MB_HARD:g} MiB "
        f"(used={budget_stats['used_mb']:.1f} MiB, "
        f"slack={budget_stats.get('slack_mb', 0):.2f} MiB)"
    )
    print(f"Analyze Hard VETO (survived budget): {len(hard_veto_layers)}")
    print(
        f"DualMonitor/dynamic FP16 (in keep, not analyze VETO): "
        f"{len(dynamic_keep_layers - hard_veto_layers)}"
    )
    print(f"Final FP16 kept layers: {len(keep_layers)}")

    print("\n--- Analyze Hard VETO Layers (FP16 after budget) ---")
    for veto_name in sorted(hard_veto_layers):
        print(f"  FP16 [analyze VETO]: {veto_name}")

    print("\n--- DualMonitor / dynamic FP16 (not analyze VETO) ---")
    for dyn_name in sorted(dynamic_keep_layers - hard_veto_layers):
        print(f"  FP16 [DualMonitor/dynamic]: {dyn_name}")

    layer_sensitivities = sorted(
        ((name, float(mon.get_sensitivity())) for name, mon in dual_monitors.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    print("\nTop 10 Sensitive Layers (Dynamic):")
    for i in range(min(10, len(layer_sensitivities))):
        name, sens = layer_sensitivities[i]
        in_veto = ' [+VETO]' if name in hard_veto_layers else ''
        print(f"  {i+1}. {name}: {sens:.4f}{in_veto}")

    print("\n[HSWQ V3.0 SDXL INT8] Starting Optimization...")
    weight_amax_dict = {}
    weight_channel_amax_dict = {}  # Card 3 only; unused when per_channel OFF
    if args.per_channel_int8:
        print(
            "[Card 3] Per-channel INT8: scale (Out,1)/(Out,1,1,1); "
            "format tag remains int8_tensorwise."
        )
    else:
        print(
            "[Card 3 OFF] pack amax = absmax (natural INT8); "
            "V4 weighted histogram already ranked FP16 keep candidates."
        )

    # Pack stores absmax. V4 MSE already filled for FP16 budget ranking
    # (+ optional soft gray-zone release).
    for name, module in tqdm(model.named_modules(), desc="Analyzing"):
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if name in keep_layers:
                continue
            weight_key = name + ".weight"

            if args.per_channel_int8:
                # Card 3: per-output-channel amax (O,). No HSWQ amax clip.
                reduce_dims = tuple(range(1, module.weight.data.dim()))
                optimal_amax_tensor = module.weight.data.abs().amax(dim=reduce_dims)
                optimal_amax_tensor = torch.clamp(optimal_amax_tensor, min=1e-6)
                print(
                    f"  [HSWQ-INT8 Card3] {name:50} | per-channel amax | "
                    f"out={int(optimal_amax_tensor.numel())} "
                    f"amax_mean={float(optimal_amax_tensor.mean()):.4f} "
                    f"amax_max={float(optimal_amax_tensor.max()):.4f}"
                )
                weight_channel_amax_dict[weight_key] = (
                    optimal_amax_tensor.detach().float().cpu()
                )
                torch.cuda.empty_cache()
                continue

            # INT8 pack point = absmax (natural).
            absmax = float(module.weight.data.abs().max().clamp_min(1e-6).item())
            print(f"  [HSWQ-INT8] {name:50} | pack absmax={absmax:.4f}")
            weight_amax_dict[weight_key] = absmax
            torch.cuda.empty_cache()

    # Snapshot signed activation means + DualMonitor sensitivity before teardown.
    act_mean_dict = {}
    sens_dict = {}
    bc_allowed_modules = None  # None = all INT8 layers; set = Approach A filter
    if args.bias_correction:
        for name, mon in dual_monitors.items():
            if mon.channel_act_mean is not None:
                act_mean_dict[name] = mon.channel_act_mean.detach().float().cpu()
            sens_dict[name] = float(mon.get_sensitivity())
        # Approach A: only top-ratio INT8 layers by sensitivity get BC.
        if args.per_channel_int8:
            _int8_dict = weight_channel_amax_dict
        else:
            _int8_dict = weight_amax_dict
        int8_module_names = [
            wk[:-7] for wk in _int8_dict.keys() if wk.endswith(".weight")
        ]
        top_ratio = float(args.bias_correction_top_ratio)
        top_ratio = 0.0 if top_ratio < 0.0 else (1.0 if top_ratio > 1.0 else top_ratio)
        ranked = sorted(
            int8_module_names,
            key=lambda n: sens_dict.get(n, 0.0),
            reverse=True,
        )
        n_bc = int(len(ranked) * top_ratio + 1e-9)
        if top_ratio > 0.0 and n_bc < 1 and ranked:
            n_bc = 1
        if top_ratio >= 1.0:
            bc_allowed_modules = None
            print(
                f"  [Bias Correction] Captured act means for {len(act_mean_dict)} layers; "
                f"scope=ALL {len(ranked)} INT8 layers (top_ratio=1.0)."
            )
        else:
            bc_allowed_modules = set(ranked[:n_bc])
            print(
                f"  [Bias Correction] Captured act means for {len(act_mean_dict)} layers; "
                f"Approach A scope=top {n_bc}/{len(ranked)} INT8 by DualMonitor "
                f"sensitivity (top_ratio={top_ratio:.3f})."
            )
            if bc_allowed_modules:
                top_show = ranked[: min(5, len(ranked))]
                for i, n in enumerate(top_show):
                    mark = "BC" if n in bc_allowed_modules else "--"
                    print(f"    [{mark}] #{i+1} sens={sens_dict.get(n, 0.0):.6g}  {n}")
    else:
        print("  [Bias Correction] Disabled (--no-bias_correction).")

    print(f"Saving quantized model (INT8): {args.output}")
    
    print("\n[VRAM Optimization] Preparing for high-speed GPU conversion...")
    del pipeline
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"[VRAM Optimization] Moving source weights to {device}...")
    input_keys = list(original_state_dict.keys())
    for k in tqdm(input_keys, desc="Loading to VRAM"):
        original_state_dict[k] = original_state_dict[k].to(device)
    
    output_state_dict = {}
    quant_meta_layers = {}  # layer_name -> {"format": "int8_tensorwise"}
    converted_count = 0
    kept_count = 0
    conv_fp16_count = 0  # Mag MixedPrecision quant-load is Linear-only
    bias_corr_pending = {}  # comfy module prefix -> float32 bias delta (O,)
    bias_corr_applied = 0
    bias_corr_skipped_no_bias = 0
    bias_corr_skipped_no_act = 0
    bias_corr_skipped_low_sens = 0

    def _emit_int8_quant_meta(out_dict, comfy_module_key):
        """Emit ComfyUI int8_tensorwise metadata.

        Following ComfyUI QUANTIZATION.md + QUANT_ALGOS["int8_tensorwise"]:
          storage_t = torch.int8, parameters = {weight_scale}, no zero-point.
        The comfy_quant tensor carries the format string as JSON bytes.
        """
        out_dict[f"{comfy_module_key}.comfy_quant"] = torch.tensor(
            list(json.dumps({"format": "int8_tensorwise"}).encode("utf-8")),
            dtype=torch.uint8,
        )
        # weight_scale is computed per-layer at quantize time (amax / 127)
        # and stored as float32 scalar alongside the int8 weight.

    print("Converting weights to INT8 (GPU accelerated)...")
    print(
        "  Mag MixedPrecisionOps: INT8 pack Linear (ndim==2) only; "
        "Conv/other stay FP16 (no unexpected weight_scale)."
    )
    for key, value in tqdm(original_state_dict.items(), desc="Converting"):
        diffusers_key = comfyui_to_diffusers_map.get(key)
        module_name = None
        if diffusers_key and diffusers_key.endswith(".weight"):
                module_name = diffusers_key[:-7]
            
        if module_name and module_name in keep_layers:
            new_value = value.to(torch.float16) if value.dtype != torch.float16 else value
            kept_count += 1
        elif module_name:
            weight_key = module_name + ".weight"
            # Comfy MixedPrecisionOps overrides Linear (Embedding/MoE) load only.
            # Conv2d inherits manual_cast load → .weight_scale/.comfy_quant unexpected
            # → raw int8 without dequant → NaN / SSIM≈0 (THIS 2026-07-15 log).
            if int(value.ndim) != 2:
                new_value = (
                    value.to(torch.float16)
                    if value.dtype != torch.float16
                    else value
                )
                conv_fp16_count += 1
                kept_count += 1
            elif args.per_channel_int8 and weight_key in weight_channel_amax_dict:
                # Card 3: broadcastable weight_scale; format int8_tensorwise.
                amax_cpu = weight_channel_amax_dict[weight_key]
                int8_quantized, scale_view, _ = pack_int8_channelwise(
                    value, amax=amax_cpu
                )
                new_value = int8_quantized
                comfy_module = key[:-7] if key.endswith(".weight") else key
                output_state_dict[f"{comfy_module}.weight_scale"] = (
                    scale_view.detach().cpu().to(torch.float32).contiguous()
                )
                _emit_int8_quant_meta(output_state_dict, comfy_module)
                quant_meta_layers[comfy_module] = {"format": "int8_tensorwise"}
                converted_count += 1

                if args.bias_correction:
                    if bc_allowed_modules is not None and module_name not in bc_allowed_modules:
                        bias_corr_skipped_low_sens += 1
                    else:
                        act_mean = act_mean_dict.get(module_name)
                        if act_mean is None:
                            bias_corr_skipped_no_act += 1
                        else:
                            weight_dq = int8_quantized.float() * scale_view
                            delta = compute_int8_bias_delta(value, weight_dq, act_mean)
                            if delta is not None:
                                bias_corr_pending[comfy_module] = (-delta).detach().float().cpu()
            elif weight_key in weight_amax_dict:
                # Pack amax = absmax. V4 MSE was for FP16 keep ranking, not pack.
                pack_amax = float(weight_amax_dict[weight_key])
                int8_quantized, scale, mid = pack_int8_tensorwise(
                    value, asymmetric=args.asymmetric_int8, amax=pack_amax
                )
                new_value = int8_quantized
                comfy_module = key[:-7] if key.endswith(".weight") else key
                # Store weight_scale as float32 scalar
                output_state_dict[f"{comfy_module}.weight_scale"] = torch.tensor(scale, dtype=torch.float32)
                _emit_int8_quant_meta(output_state_dict, comfy_module)
                quant_meta_layers[comfy_module] = {"format": "int8_tensorwise"}
                converted_count += 1

                if args.bias_correction:
                    if bc_allowed_modules is not None and module_name not in bc_allowed_modules:
                        bias_corr_skipped_low_sens += 1
                    else:
                        act_mean = act_mean_dict.get(module_name)
                        if act_mean is None:
                            bias_corr_skipped_no_act += 1
                        else:
                            # Compare loader view (q*scale) to FP weight. For asymmetric
                            # pack, this also absorbs mid into bias via act means.
                            weight_dq = int8_quantized.float() * scale
                            delta = compute_int8_bias_delta(value, weight_dq, act_mean)
                            if delta is not None:
                                # Negate: add -E[(W_q-W)x] to bias so output mean matches FP.
                                bias_corr_pending[comfy_module] = (-delta).detach().float().cpu()
            else:
                new_value = value
        else:
            new_value = value
            
        output_state_dict[key] = new_value

    if args.bias_correction and bias_corr_pending:
        print(f"\n[Bias Correction] Applying deltas to {len(bias_corr_pending)} INT8 layers...")
        for comfy_module, delta in bias_corr_pending.items():
            bias_key = f"{comfy_module}.bias"
            if bias_key not in output_state_dict:
                bias_corr_skipped_no_bias += 1
                continue
            bias = output_state_dict[bias_key]
            corrected = bias.float() + delta.to(device=bias.device, dtype=torch.float32)
            output_state_dict[bias_key] = corrected.to(dtype=bias.dtype)
            bias_corr_applied += 1
        print(
            f"  [Bias Correction] applied={bias_corr_applied}, "
            f"no_bias={bias_corr_skipped_no_bias}, no_act={bias_corr_skipped_no_act}, "
            f"low_sens_skip={bias_corr_skipped_low_sens}"
        )

    print("Conversion done:")
    print(f"  INT8 layers (Linear ndim==2): {converted_count}")
    print(f"  FP16-kept layers: {kept_count} (incl. non-Linear/Conv FP16={conv_fp16_count})")
    print(f"  Per-channel INT8 (Card 3): {args.per_channel_int8}")
    print(f"  Asymmetric INT8 pack: {args.asymmetric_int8}")
    if args.bias_correction:
        print(f"  Bias-corrected layers: {bias_corr_applied}")

    # Build _quantization_metadata for ComfyUI loader (QUANTIZATION.md format)
    quantization_metadata = {
        "format_version": "1.0",
        "layers": quant_meta_layers,
    }
    metadata = {"_quantization_metadata": json.dumps(quantization_metadata)}

    try:
        save_file(output_state_dict, args.output, metadata=metadata)
    except Exception as e:
        print(f"[Save Warning] GPU Tensor save failed ({e}). Moving to CPU explicitly...")
        cpu_dict = {k: v.cpu() for k, v in output_state_dict.items()}
        save_file(cpu_dict, args.output, metadata=metadata)

    print(f"Saved INT8 quantized model: {args.output}")
    print(f"  Format: int8_tensorwise (ComfyUI QUANT_ALGOS compatible)")
    print(f"  Quantized layers: {converted_count}")
    print(f"  FP16 kept layers: {kept_count}")
    print(
        f"  Per-channel (Card 3): {args.per_channel_int8} | "
        f"Asymmetric pack: {args.asymmetric_int8} | "
        f"Bias correction: {args.bias_correction} "
        f"(top_ratio={args.bias_correction_top_ratio})"
    )

if __name__ == "__main__":
    main()

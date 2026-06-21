"""
HSWQ V2.0 SDXL Quantization Script
==================================
Hybrid Sensitivity Weighted Quantization for Stable Diffusion XL (UNet only).

V2.0 features:
- DualMonitor: simultaneous sensitivity + importance (input activation) measurement
- HSWQWeightedHistogramOptimizerV4: exact FP8 E4M3 grid MSE optimization
- Autonomous VETO engine: SDXL key-pattern, per-projection attn (no QKV fuse), gray-zone MSE
- ComfyUI-compatible output: scaled=False, weight_scale=1.0, comfy_quant metadata

Profiling: analyze/analyze_sdxl_distribution.py (auto-generated from template if missing).
Calibration: StableDiffusionXLPipeline latent inference (Diffusers).
"""
import argparse
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

# Ensure histogram modules are importable regardless of clone path / CWD
histogram_dir = os.path.join(current_dir, "histogram")
if histogram_dir not in sys.path:
    sys.path.insert(0, histogram_dir)

# Support for optional venv site-packages (e.g. local wheels)
venv_site_packages = os.path.join(os.path.dirname(current_dir), "venv", "Lib", "site-packages")
if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
    sys.path.append(venv_site_packages)

from weighted_histogram_mse_v4 import HSWQWeightedHistogramOptimizerV4

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

# --- V2.0 SDXL autonomous engine tunables (key-pattern / drift only; numeric thresholds from profile) ---
_SDXL_KP_BOUNDARY_SUFFIXES = (".conv_in", ".conv_out")
_SDXL_KP_PREFIXES = ("time_embedding.", "add_embedding.")
_SDXL_ATTN_PROJ_SUFFIXES = (".to_q", ".to_k", ".to_v")
_SDXL_ATTN_TOOUT_SUFFIX = ".to_out.0"
_SDXL_PROFILE_PREFIXES = ("model.", "model.diffusion_model.")


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
    drift_veto_thresh: float = 0.5
    drift_score_mult: float = 1.0
    mse_release_o_min: float = 40.0
    mse_release_k_max: float = 20.0
    mse_release_m_max: float = 20.0
    mse_p75_multiplier: float = 2.0
    k_scale: float = 0.01
    o_scale: float = 0.016
    m_scale: float = 0.05
    k_gray_lo: float = 10.0
    k_gray_hi: float = 20.0
    o_gray_lo: float = 30.0
    o_gray_hi: float = 40.0
    m_gray_lo: float = 5.0
    m_gray_hi: float = 20.0
    search_low_floor: float = 0.5
    search_low_penalty_cap: float = 0.49
    search_low_clip_max: float = 0.99
    search_low_gray_clip_max: float = 0.9
    alpha_floor: float = 0.5
    alpha_clip_max: float = 0.99
    beta_floor: float = 0.5
    beta_clip_max: float = 0.99
    ff2_suffix_min_count: int = 4
    score_o_weight: float = 2.0
    score_m_weight: float = 0.5

    @classmethod
    def from_dict(cls, d: dict) -> "SdxlVetoTunables":
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
            drift_veto_thresh=float(d.get("drift_veto_thresh", 0.5)),
            drift_score_mult=float(d.get("drift_score_mult", 1.0)),
            mse_release_o_min=float(d.get("mse_release_o_min", 40.0)),
            mse_release_k_max=float(d.get("mse_release_k_max", 20.0)),
            mse_release_m_max=float(d.get("mse_release_m_max", 20.0)),
            mse_p75_multiplier=float(d.get("mse_p75_multiplier", 2.0)),
            k_scale=float(d.get("k_scale", 0.01)),
            o_scale=float(d.get("o_scale", 0.016)),
            m_scale=float(d.get("m_scale", 0.05)),
            k_gray_lo=float(d.get("k_gray_lo", 10.0)),
            k_gray_hi=float(d.get("k_gray_hi", 20.0)),
            o_gray_lo=float(d.get("o_gray_lo", 30.0)),
            o_gray_hi=float(d.get("o_gray_hi", 40.0)),
            m_gray_lo=float(d.get("m_gray_lo", 5.0)),
            m_gray_hi=float(d.get("m_gray_hi", 20.0)),
            search_low_floor=float(d.get("search_low_floor", 0.5)),
            search_low_penalty_cap=float(d.get("search_low_penalty_cap", 0.49)),
            search_low_clip_max=float(d.get("search_low_clip_max", 0.99)),
            search_low_gray_clip_max=float(d.get("search_low_gray_clip_max", 0.9)),
            alpha_floor=float(d.get("alpha_floor", 0.5)),
            alpha_clip_max=float(d.get("alpha_clip_max", 0.99)),
            beta_floor=float(d.get("beta_floor", 0.5)),
            beta_clip_max=float(d.get("beta_clip_max", 0.99)),
            ff2_suffix_min_count=int(d.get("ff2_suffix_min_count", 4)),
            score_o_weight=float(d.get("score_o_weight", 2.0)),
            score_m_weight=float(d.get("score_m_weight", 0.5)),
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
            "mse_p75_multiplier": self.mse_p75_multiplier,
            "ff2_suffix_min_count": self.ff2_suffix_min_count,
        }


def resolve_veto_tunables(
    norm_profile: dict,
    profile_summary: dict | None = None,
) -> SdxlVetoTunables:
    """Load veto_tunables from profile summary or derive from normalized layer stats."""
    analyze_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze")
    if analyze_dir not in sys.path:
        sys.path.insert(0, analyze_dir)
    from analyze_sdxl_distribution import derive_veto_tunables

    if norm_profile:
        derived = derive_veto_tunables(norm_profile)
        if derived.get("ff2_auto_full_class"):
            print(
                "  [Auto FF2] full-class protection: "
                f"count={derived.get('ff2_class_count', 0)}, "
                f"span={derived.get('ff2_class_outlier_span', 0):.3f}, "
                f"profile_o>={derived['ff2_profile_outlier']:.2f}"
            )
        return SdxlVetoTunables.from_dict(derived)
    if profile_summary and isinstance(profile_summary.get("veto_tunables"), dict):
        return SdxlVetoTunables.from_dict(profile_summary["veto_tunables"])
    raise ValueError("resolve_veto_tunables: no layer profile available")


def _layer_weight_stats(tensor: torch.Tensor) -> tuple[float, float, float]:
    """Live kurtosis, outlier_ratio, abs_max for a weight tensor."""
    x = tensor.float()
    std = torch.std(x).item()
    amax = max(abs(x.min().item()), abs(x.max().item()))
    k = calculate_kurtosis(x)
    o = amax / std if std > 0 else 0.0
    return k, o, amax


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
    # V2.0 fix: floor thresholds to prevent SDXL's uniform distribution from
    # deriving thresholds so low that selective == full-class.
    score_cut = max(tunables.ff2_profile_score_cutoff, 2.5)
    outlier_cut = max(tunables.ff2_profile_outlier, 40.0)
    live_cut = max(tunables.ff2_outlier_live, 40.0)
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
    """SDXL key-pattern VETO: embeddings, boundaries, profile-tuned ff2 class."""
    added = set()
    ff2_suffixes = _discover_ff2_suffixes(norm_profile)
    for _n, _m in model.named_modules():
        if not isinstance(_m, torch.nn.Linear):
            continue
        if _n in hard_veto_layers:
            continue
        if any(_n.startswith(p) for p in _SDXL_KP_PREFIXES):
            added.add(_n)
            print(f"    [Key-Pattern VETO] {_n} (embedding)")
            continue
        if _n.endswith(_SDXL_KP_BOUNDARY_SUFFIXES):
            added.add(_n)
            print(f"    [Key-Pattern VETO] {_n} (boundary)")
            continue
        if ff2_suffixes and any(_n.endswith(s) for s in ff2_suffixes):
            # V2.0 fix: skip full-class auto (inflates file size on SDXL);
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
    """VETO attn projections when profile (or live) abs_max / outlier_ratio exceeds thresholds."""
    proj_veto = set()
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
            hit = _amax >= max(tunables.attn_toout_absmax, 4.5) or _o >= max(tunables.attn_toout_outlier, 40.0)
            thresh_msg = (
                f"to_out amax>={tunables.attn_toout_absmax}, o>={tunables.attn_toout_outlier}"
            )
        else:
            hit = _amax >= max(tunables.attn_qkv_absmax, 4.5) or _o >= max(tunables.attn_qkv_outlier, 40.0)
            thresh_msg = (
                f"q/k/v amax>={tunables.attn_qkv_absmax}, o>={tunables.attn_qkv_outlier}"
            )
        if hit:
            proj_veto.add(_n)
            print(
                f"    [Per-Projection VETO] {_n} "
                f"({src} amax={_amax:.2f}, outlier={_o:.1f}; {thresh_msg})"
            )
    return proj_veto


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
        prof_o = float(prof.get("outlier_ratio", 0) or 0) if prof else 0.0
        ff2_suffixes = _discover_ff2_suffixes(
            norm_profile, min_count=tunables.ff2_suffix_min_count
        )
        if ff2_suffixes and any(_n.endswith(s) for s in ff2_suffixes):
            # V2.0 fix: selective only (no full-class auto)
            hit, reason = _ff2_selective_veto_hit(prof if prof else None, _o, tunables)
            if hit:
                added.add(_n)
                print(f"    [Supplemental VETO] {_n} (ff.net.2 {reason})")
        elif any(_n.startswith(p) for p in _SDXL_KP_PREFIXES) and drift > max(tunables.drift_veto_thresh, 0.5):
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
    """Outlier-only profile VETO with low drift and non-structural — MSE release candidates."""
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
            o > min(tunables.mse_release_o_min, 40.0)
            and k <= max(tunables.mse_release_k_max, 20.0)
            and m <= max(tunables.mse_release_m_max, 20.0)
        ):
            vmod = _module_dict.get(vname)
            if vmod is not None and hasattr(vmod, "weight"):
                drift = _weight_profile_drift(vmod.weight.data, prof)
                if drift < max(tunables.drift_veto_thresh, 0.5):
                    candidates.add(vname)
    return candidates




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
) -> tuple[set, set]:
    """Gray-zone VETO release via trial MSE (SDXL V2.0)."""
    if not outlier_only_veto:
        return hard_veto_layers, keep_layers

    print(
        f"\n  [{scope_label} MSE-Guided Reassessment] {len(outlier_only_veto)} VETO layers "
        f"are outlier-only (o>{tunables.mse_release_o_min:.2f}, "
        f"k<={tunables.mse_release_k_max:.2f}, m<={tunables.mse_release_m_max:.2f})."
    )
    print(f"  Trial-quantizing to measure actual HSWQ quantization error...")

    trial_optimizer = HSWQWeightedHistogramOptimizerV4(
        bins=8192, num_candidates=1000, refinement_iterations=10,
        device=device, alpha=alpha, beta=beta
    )

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
        try:
            sresult = trial_optimizer.compute_optimal_amax_with_stats(
                sw, importance=None, use_svd_leverage=True, scaled=False
            )
            safe_mses.append(sresult["estimated_mse"])
        except Exception as e:
            print(f"    [MSE ERROR] Failed safe layer {sname}: {e}")
        torch.cuda.empty_cache()

    if not safe_mses:
        print(f"  [{scope_label} MSE-Guided Reassessment] No safe baseline available, skipping.")
        return hard_veto_layers, keep_layers

    safe_mses.sort()
    p75_idx = int(len(safe_mses) * 0.75)
    mse_threshold = safe_mses[min(p75_idx, len(safe_mses) - 1)] * max(tunables.mse_p75_multiplier, 2.0)
    print(
        f"  [MSE Baseline] Safe layers sampled: {len(safe_mses)}, "
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
        try:
            vresult = trial_optimizer.compute_optimal_amax_with_stats(
                vw, importance=None, use_svd_leverage=True, scaled=False
            )
            vmse = vresult["estimated_mse"]
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
            f"  [{scope_label} MSE-Guided Reassessment] Released {len(released)} layers from VETO. "
            f"Remaining VETO: {len(hard_veto_layers)}."
        )
        print(f"  Updated FP16 kept layers: {len(keep_layers)}")
    else:
        print(f"  [{scope_label} MSE-Guided Reassessment] No layers released (all exceeded MSE threshold).")

    return hard_veto_layers, keep_layers



class DualMonitor:
    def __init__(self):
        self.output_sum = 0.0
        self.output_sq_sum = 0.0
        self.count = 0
        self.channel_importance = None
    
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
                self.channel_importance = (
                    self.channel_importance * self.count + current_imp
                ) / (self.count + 1)
            self.count += 1

    def get_sensitivity(self):
        if self.count == 0:
            return 0.0
        mean = self.output_sum / self.count
        variance = (self.output_sq_sum / self.count) - mean ** 2
        import math
        return variance if math.isfinite(variance) else 0.0


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


def derive_hswq_strategy(model_profile, veto_tunables: SdxlVetoTunables | None = None):
    """
    SDXL V2.0: Alpha/Beta from profile + per-layer search_low (SDXL-only).
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
        veto_tunables = resolve_veto_tunables(model_profile or {})

    def get_dynamic_search_low(name, weight_tensor):
        profile_key = name + ".weight"
        prof = model_profile.get(profile_key, model_profile.get(name, {})) if model_profile else {}
        if prof:
            k_stat = prof.get("kurtosis", 0)
            o_ratio = prof.get("outlier_ratio", 0)
            m_stat = prof.get("abs_max", 0)
        else:
            k_stat, o_ratio, m_stat = _layer_weight_stats(weight_tensor)
        vt = veto_tunables
        # V2.0 fix: cap penalties at 0.49 (zib-proven) to prevent over-conservative search
        k_penalty = min(k_stat * vt.k_scale, vt.search_low_penalty_cap, 0.49)
        o_penalty = min(o_ratio * vt.o_scale, vt.search_low_penalty_cap, 0.49)
        upper_clip = min(vt.search_low_clip_max, 0.99)
        drift = _weight_profile_drift(weight_tensor, prof) if prof else 0.0
        # V2.0 fix: floor gray-zone bounds so SDXL's uniform distribution doesn't
        # collapse the zone to near-zero width
        in_gray = (
            (max(vt.k_gray_lo, 10.0) < k_stat <= max(vt.k_gray_hi, 20.0))
            or (max(vt.o_gray_lo, 30.0) < o_ratio <= max(vt.o_gray_hi, 40.0))
            or (max(vt.m_gray_lo, 5.0) < m_stat <= max(vt.m_gray_hi, 20.0))
        )
        if in_gray or drift > max(vt.drift_veto_thresh, 0.5):
            upper_clip = min(vt.search_low_gray_clip_max, 0.90)
        return float(
            np.clip(
                max(vt.search_low_floor, 0.5) + max(k_penalty, o_penalty),
                max(vt.search_low_floor, 0.5),
                upper_clip,
            )
        )

    if model_profile:
        all_k = [p.get("kurtosis", 0) for p in model_profile.values() if isinstance(p, dict)]
        all_o = [p.get("outlier_ratio", 0) for p in model_profile.values() if isinstance(p, dict)]
        all_m = [p.get("abs_max", 0) for p in model_profile.values() if isinstance(p, dict)]
        avg_k = np.mean(all_k) if all_k else 0
        avg_o = np.mean(all_o) if all_o else 0
        avg_m = np.mean(all_m) if all_m else 0
        print(f"  [Profile Stats] Avg Kurtosis: {avg_k:.2f}, Avg OutlierRatio: {avg_o:.2f}, Avg AbsMax: {avg_m:.2f}")
        vt = veto_tunables
        alpha_floor_safe = max(vt.alpha_floor, 0.5)
        alpha = float(
            np.clip(alpha_floor_safe + avg_k * vt.k_scale, alpha_floor_safe, min(vt.alpha_clip_max, 0.80))
        )
        beta = 1.0 - alpha  # V2.0 fix: enforce alpha + beta = 1.0 (zib-proven constraint)
    else:
        print("  [Profile Stats] No profile loaded. Using default alpha/beta.")
        alpha, beta = 0.5, 0.5

    print(f"  [Dynamic Alpha/Beta] alpha={alpha:.3f}, beta={beta:.3f}")

    hard_veto_layers = set()
    if model_profile:
        for name, prof in model_profile.items():
            if isinstance(prof, dict):
                k = prof.get("kurtosis", 0)
                m = prof.get("abs_max", 0)
                o = prof.get("outlier_ratio", 0)
                # V2.0 fix: floor thresholds at zib-proven values to prevent
                # Tukey upper fence from over-triggering on SDXL's uniform distribution
                is_extreme_divergence = o > max(veto_tunables.extreme_outlier, 40.0)
                is_extreme_kurtosis = k > max(veto_tunables.extreme_kurtosis, 20.0)
                is_huge_magnitude = m > max(veto_tunables.huge_magnitude, 20.0)
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
        f"  [Static Profile VETO] Identified {len(hard_veto_layers)} layers "
        "with extreme distribution (Unquantizable in FP8)."
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
    parser = argparse.ArgumentParser(description="SDXL FP8 Quantization - HSWQ V2.0 (Autonomous Engine)")
    parser.add_argument("--input", type=str, required=True, help="Path to input safetensors model")
    parser.add_argument("--output", type=str, required=True, help="Path to output safetensors model")
    parser.add_argument("--calib_file", type=str, required=True, help="Path to calibration prompts text file")
    parser.add_argument("--num_calib_samples", type=int, default=256, help="Number of calibration samples")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--keep_ratio", type=float, default=0.25, help="Ratio of layers to keep in FP16 (typical 0.05–0.25; 0.05–0.10 often sufficient for SDXL)")
    parser.add_argument("--comfy_path", type=str, help="Path to ComfyUI root directory (optional, will auto-detect)")
    parser.add_argument("--profile", type=str, help="Path to distribution profile JSON (optional, will auto-generate if missing)")
    args = parser.parse_args()

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
    print("HSWQ V2.0 SDXL Pure Autonomous Engine (Environment-Aware Analysis)")
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
    veto_tunables = resolve_veto_tunables(_norm_profile, profile_summary)
    print(f"  [Veto Tunables] {veto_tunables.as_dict()}")
    alpha, beta, get_layer_search_low, hard_veto_layers = derive_hswq_strategy(
        model_profile,
        veto_tunables,
    )

    print("  [V2.0 SDXL Autonomous VETO] Structural + per-projection attn + key-pattern + supplemental.")
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


    print("\nAnalyzing layer sensitivity (profile_score + drift)...")

    _supp = _autonomous_supplemental_veto(model, hard_veto_layers, _norm_profile, veto_tunables)
    if _supp:
        hard_veto_layers = hard_veto_layers.union(_supp)
        print(f"  [Supplemental VETO] Added {len(_supp)} layers (total VETO: {len(hard_veto_layers)}).")

    # Exclude VETO layers from the Dynamic pool (they are always FP16, so Dynamic budget goes elsewhere)
    _module_dict_sens = dict(model.named_modules())
    layer_sensitivities = []
    ranking_source = "profile_score"
    for name in target_modules:
        if name in hard_veto_layers:
            continue
        prof = _norm_profile.get(name, {})
        drift = 0.0
        mod = _module_dict_sens.get(name)
        if prof and mod is not None and hasattr(mod, "weight"):
            drift = _weight_profile_drift(mod.weight.data, prof)
        # V2.0 fix: use fixed weights for stable ranking (zib-proven: k + o*2 + m*0.5)
        # instead of dynamic score_o_weight/score_m_weight that are unstable on SDXL
        if prof:
            k = prof.get("kurtosis", 0) or 0
            o = prof.get("outlier_ratio", 0) or 0
            m = prof.get("abs_max", 0) or 0
            score = k + o * 2.0 + m * 0.5 + drift * veto_tunables.drift_score_mult
        elif name in dual_monitors:
            score = dual_monitors[name].get_sensitivity()
            ranking_source = "dualmonitor_fallback"
        else:
            continue
        layer_sensitivities.append((name, score))

    layer_sensitivities.sort(key=lambda x: x[1], reverse=True)
    num_keep_dynamic = int(len(layer_sensitivities) * args.keep_ratio)
    dynamic_keep_layers = set([x[0] for x in layer_sensitivities[:num_keep_dynamic]])
    
    # [V1.92 Exclusive Protection] VETO (always FP16) + Dynamic (additional FP16) with no overlap for maximum coverage
    keep_layers = dynamic_keep_layers.union(hard_veto_layers)
    
    # [V2.0 SDXL MSE-Guided VETO Reassessment] outlier-only VETO release candidates
    release_cands = _collect_mse_release_candidates(
        hard_veto_layers, structural_veto, _norm_profile, model, veto_tunables
    )
    if keypattern_veto:
        release_cands -= keypattern_veto
    if release_cands:
        hard_veto_layers, keep_layers = _mse_grayzone_veto_reassessment(
            scope_label="V2.0 SDXL",
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
        )
    
    non_veto_total = len([n for n in target_modules if n not in hard_veto_layers])
    print(f"\nTotal layers: {len(target_modules)} (Non-VETO pool: {non_veto_total})")
    print(f"Dynamic ranking: {ranking_source} (V2.0)")
    print(f"Dynamic kept (from non-VETO pool): {len(dynamic_keep_layers)} (Top {args.keep_ratio*100:.1f}%)")
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

    print("\n[HSWQ V2.0 SDXL] Starting Optimization...")
    weight_amax_dict = {}
    hswq_optimizer = HSWQWeightedHistogramOptimizerV4(
        bins=8192,
        num_candidates=1000,
        refinement_iterations=10,
        device=device,
        alpha=alpha,
        beta=beta,
    )
    
    for name, module in tqdm(model.named_modules(), desc="Analyzing"):
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if name in keep_layers:
                continue
            importance = dual_monitors[name].channel_importance if name in dual_monitors else None
            layer_search_low = get_layer_search_low(name, module.weight.data)
            layer_search_range = (layer_search_low, 1.0)
            print(
                f"  [HSWQ] {name:50} | Pure Data-Driven | "
                f"search_range={layer_search_range[0]:.3f}-{layer_search_range[1]:.3f}"
            )
            optimal_amax = hswq_optimizer.compute_optimal_amax(
                module.weight.data, 
                importance, 
                use_svd_leverage=True, 
                scaled=False, 
                search_range=layer_search_range,
            )
            weight_amax_dict[name + ".weight"] = optimal_amax
            torch.cuda.empty_cache()

    print(f"Saving quantized model: {args.output}")

    print("\n[VRAM Optimization] Preparing for high-speed GPU conversion...")
    del pipeline
    del hswq_optimizer
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[VRAM Optimization] Moving source weights to {device}...")
    input_keys = list(original_state_dict.keys())
    for k in tqdm(input_keys, desc="Loading to VRAM"):
        original_state_dict[k] = original_state_dict[k].to(device)

    output_state_dict = {}
    converted_count = 0
    kept_count = 0

    def _emit_quant_meta(out_dict, comfy_module_key):
        out_dict[f"{comfy_module_key}.comfy_quant"] = torch.tensor(
            list(json.dumps({"format": "float8_e4m3fn"}).encode("utf-8")),
            dtype=torch.uint8,
        )
        out_dict[f"{comfy_module_key}.weight_scale"] = torch.tensor(1.0, dtype=torch.float32)

    print("Converting weights (GPU accelerated)...")
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
            if weight_key in weight_amax_dict:
                amax = max(weight_amax_dict[weight_key], 1e-6)
                clamped_value = torch.clamp(value, -amax, amax)
                new_value = clamped_value.to(torch.float8_e4m3fn)
                comfy_module = key[:-7] if key.endswith(".weight") else key
                _emit_quant_meta(output_state_dict, comfy_module)
                converted_count += 1
            else:
                new_value = value
        else:
            new_value = value

        output_state_dict[key] = new_value

    print("Conversion done:")
    print(f"  FP8 layers: {converted_count}")
    print(f"  FP16-kept layers: {kept_count}")

    try:
        save_file(output_state_dict, args.output)
    except Exception as e:
        print(f"[Save Warning] GPU Tensor save failed ({e}). Moving to CPU explicitly...")
        cpu_dict = {k: v.cpu() for k, v in output_state_dict.items()}
        save_file(cpu_dict, args.output)

    print("Saved.")

if __name__ == "__main__":
    main()

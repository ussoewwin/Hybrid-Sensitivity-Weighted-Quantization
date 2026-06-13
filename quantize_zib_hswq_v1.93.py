"""
Z-Image Base (Non-Turbo) FP8 Quantization - HSWQ V1.93 (Pure Data-Driven Autonomous Engine)
Target Model: Z-Image Base (e.g., UR03: moodyWildV0200001.UR03.safetensors)

V1.93 moodyRealMix ZIT (filename key moodyrealmix) — independent from Z-Anime:
  - FP16 calibration path unchanged (V6 is 100% BF16 weights yet SSIM 0.9919 on v1.92).
  - moody-only: Structural VETO, per-projection qkv VETO, gray-zone search_low cap,
    live-vs-profile weight drift scoring, MSE gray-zone VETO reassessment, fused-qkv
    per-projection HSWQ (Comfy qkv key preserved; not Diffusers split).
  - moody V7 only (basename zitv7 / _v7): key-pattern hard VETO for all .attention.qkv
    plus cap_embedder.1 / final_layer.linear / x_embedder / context_refiner attention.out.
  - Z-Anime code paths (is_zanime) are untouched. ZI/ZIB/ZIT r0.05 behavior unchanged.

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

# --- Z-Image Base (NextDiT) model load and inference pipeline ---
ZIT_PREFIXES = [
    "model.diffusion_model.",
    "model.",
    "diffusion_model.",
    "",
]

import re

def _fuse_zanime_attention(state_dict):
    """Z-Anime (Diffusers/HF style) attention -> ComfyUI NextDiT (lumina) style.
      <p>.attention.to_q.weight + to_k.weight + to_v.weight -> <p>.attention.qkv.weight (cat dim=0)
      <p>.attention.to_out.0.weight                          -> <p>.attention.out.weight
      <p>.attention.norm_q.weight                            -> <p>.attention.q_norm.weight
      <p>.attention.norm_k.weight                            -> <p>.attention.k_norm.weight
    Only applied when Z-Anime is detected; ZI/ZIB/ZIT keys (already qkv-fused) are unaffected.
    """
    new_dict = dict(state_dict)
    prefixes = set()
    for k in list(new_dict.keys()):
        m = re.match(r"^(.+?\.attention)\.to_q\.weight$", k)
        if m:
            prefixes.add(m.group(1))
    for prefix in prefixes:
        kq, kk, kv = f"{prefix}.to_q.weight", f"{prefix}.to_k.weight", f"{prefix}.to_v.weight"
        if kq in new_dict and kk in new_dict and kv in new_dict:
            qkv = torch.cat([new_dict[kq], new_dict[kk], new_dict[kv]], dim=0)
            new_dict[f"{prefix}.qkv.weight"] = qkv
            del new_dict[kq], new_dict[kk], new_dict[kv]
    rename_map = {
        ".attention.to_out.0.weight": ".attention.out.weight",
        ".attention.norm_q.weight":   ".attention.q_norm.weight",
        ".attention.norm_k.weight":   ".attention.k_norm.weight",
    }
    for k in list(new_dict.keys()):
        for src, dst in rename_map.items():
            if k.endswith(src):
                new_dict[k.replace(src, dst)] = new_dict.pop(k)
                break
    return new_dict

def normalize_zanime_keys(state_dict):
    """Normalize Z-Anime specific key naming to standard NextDiT format.
    Step 1: Strip 'all_<module>.2-1' prefix.
      all_x_embedder.2-1.weight               -> x_embedder.weight
      all_final_layer.2-1.linear.weight       -> final_layer.linear.weight
      all_layers.0.2-1.attention.to_q.weight  -> layers.0.attention.to_q.weight
    Step 2: Fuse / rename Diffusers-style attention to ComfyUI NextDiT style.
      to_q+to_k+to_v -> qkv (cat dim=0), to_out.0 -> out, norm_q/norm_k -> q_norm/k_norm
    Returns (normalized_dict, reverse_map) where reverse_map[normalized_key] = original_key.
    ZIB/ZIT logic is preserved by only applying this when Z-Anime is detected.
    """
    normalized = {}
    reverse_map = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("all_"):
            new_key = re.sub(r'^all_(.*?)\.2-1', r'\1', new_key)
            reverse_map[new_key] = key
        normalized[new_key] = value
    normalized = _fuse_zanime_attention(normalized)
    return normalized, reverse_map

def _denormalize_zanime_output(state_dict, reverse_map):
    """Inverse of normalize_zanime_keys for Z-Anime output saving.

    The HSWQ-V1.92 output is internally in NextDiT key form
    (qkv fused, out/q_norm/k_norm). Z-Anime checkpoints (matching the
    official FP8 distribution layout) require Diffusers form
    (to_q/to_k/to_v/to_out.0/norm_q/norm_k) plus the 'all_<module>.2-1'
    prefix, so ComfyUI's z_image_to_diffusers loader path can pick them up.

    NOTE: qkv weight splitting is NOT done here. qkv layers are split per-head
    in the quantization/save stage so that to_q, to_k, to_v each receive their
    own HSWQ-optimized amax. This function only handles companion-key splits
    that may still exist (e.g. weight_scale/comfy_quant attached to qkv.* by
    the HSWQ V1 save path), renames out/q_norm/k_norm, and restores prefixes.
    """
    # 1. Split any leftover qkv.* companion keys (.weight_scale, .comfy_quant)
    #    Replicate metadata for each of to_q / to_k / to_v.
    intermediate = {}
    qkv_companion_prefixes = set()
    for k in list(state_dict.keys()):
        m = re.match(r"^(.+?\.attention)\.qkv\.(weight_scale|comfy_quant)$", k)
        if m:
            qkv_companion_prefixes.add((m.group(1), m.group(2)))

    skip = set()
    for prefix, suffix in qkv_companion_prefixes:
        src = f"{prefix}.qkv.{suffix}"
        if src not in state_dict:
            continue
        meta = state_dict[src]
        for tgt in ("to_q", "to_k", "to_v"):
            tgt_key = f"{prefix}.{tgt}.{suffix}"
            intermediate[tgt_key] = meta.clone() if hasattr(meta, "clone") else meta
        skip.add(src)

    # 2. Rename and pass through. Catches weight + companions
    #    (.weight_scale / .comfy_quant) attached to renamed modules.
    rename_map_suffixes = [
        (".attention.out.weight",       ".attention.to_out.0.weight"),
        (".attention.out.weight_scale", ".attention.to_out.0.weight_scale"),
        (".attention.out.comfy_quant",  ".attention.to_out.0.comfy_quant"),
        (".attention.q_norm.weight",    ".attention.norm_q.weight"),
        (".attention.k_norm.weight",    ".attention.norm_k.weight"),
    ]
    renamed = {}
    for k, v in state_dict.items():
        if k in skip:
            continue
        new_k = k
        for src, dst in rename_map_suffixes:
            if k.endswith(src):
                new_k = k[: -len(src)] + dst
                break
        renamed[new_k] = v
    for k, v in intermediate.items():
        renamed[k] = v

    # 3. Restore 'all_<module>.2-1' prefix.
    #    The `.2-1` insertion depth varies per key (see normalize_zanime_keys
    #    Step 1: x_embedder.weight <-> all_x_embedder.2-1.weight depth=1, but
    #    layers.0.attention.to_q.weight <-> all_layers.0.2-1.attention.to_q.weight
    #    depth=2 because '.2-1' sits AFTER the block index for layers.X).
    #    Build a robust per-module mapping from reverse_map (which is the only
    #    authoritative record of where '.2-1' was originally located) and use it
    #    for both `.weight` keys and their HSWQ V1 companion keys
    #    (`.weight_scale`, `.comfy_quant`). Naive top-level prefix restoration
    #    misplaces '.2-1' for layers.X and breaks the ComfyUI z_image_to_diffusers
    #    loader for ALL 30 transformer blocks, silently leaving them randomly
    #    initialized after load.
    weight_norm_to_orig = dict(reverse_map)  # norm_key (with .weight) -> orig_key (with .2-1)
    module_norm_to_orig = {}                  # norm_module (no .weight) -> orig_module (no .weight)
    for norm_key, orig_key in weight_norm_to_orig.items():
        if norm_key.endswith(".weight") and orig_key.endswith(".weight"):
            module_norm_to_orig[norm_key[:-7]] = orig_key[:-7]

    final = {}
    for k, v in renamed.items():
        if k in weight_norm_to_orig:
            # Direct .weight mapping (handles to_q/to_k/to_v split outputs and
            # the renamed to_out.0 / norm_q / norm_k forms emitted by step 2).
            final[weight_norm_to_orig[k]] = v
            continue
        # Companion keys (.weight_scale, .comfy_quant) emitted by HSWQ V1
        # save path: split off the suffix and reuse the module-level mapping.
        matched = False
        for suffix in (".weight_scale", ".comfy_quant"):
            if k.endswith(suffix):
                module_norm = k[: -len(suffix)]
                if module_norm in module_norm_to_orig:
                    final[module_norm_to_orig[module_norm] + suffix] = v
                    matched = True
                    break
        if matched:
            continue
        # Keys without a ZA prefix in the original file pass through unchanged.
        final[k] = v
    return final

def calculate_kurtosis(tensor):
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    if std == 0: return 0.0
    return torch.mean(((tensor - mean) / std) ** 4).item()

def detect_and_strip_prefix(state_dict):
    keys = list(state_dict.keys())
    is_zanime = False
    reverse_map = {}

    # --- Z-Anime detection & normalization ---
    if any(k.startswith("all_x_embedder.2-1") for k in keys):
        is_zanime = True
        print("  [Model Detection] Z-Anime key naming detected. Normalizing to standard NextDiT keys...")
        normalized, reverse_map = normalize_zanime_keys(state_dict)
        return normalized, "", is_zanime, reverse_map

    for prefix in ZIT_PREFIXES:
        if prefix == "":
            if any(k.startswith("layers.") or k.startswith("x_embedder") for k in keys):
                return state_dict, "", is_zanime, reverse_map
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
                return stripped, prefix, is_zanime, reverse_map
    print("  [Prefix Detection] No prefix detected (assuming HSWQ format)")
    return state_dict, "", is_zanime, reverse_map

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

    # Detect qk_norm (Z-Anime has attention.q_norm/k_norm; ZI/ZIB/ZIT typically not)
    zit_config["qk_norm"] = any(k.endswith(".attention.q_norm.weight") for k in state_dict_keys)
    if zit_config["qk_norm"]:
        print(f"  Detected qk_norm=True (q_norm/k_norm weights present)")
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


def _bf16_weight_fraction(stripped_state_dict) -> float:
    """Fraction of .weight tensors stored as bfloat16 (metadata-only, no GPU)."""
    weights = [v for k, v in stripped_state_dict.items() if k.endswith(".weight")]
    if not weights:
        return 0.0
    bf16_n = sum(1 for v in weights if v.dtype == torch.bfloat16)
    return bf16_n / len(weights)


def detect_moody_zit_checkpoint(path: str) -> bool:
    """True when input filename matches moodyRealMix ZIT checkpoints (key pattern only)."""
    return "moodyrealmix" in os.path.basename(path).lower()


def detect_moody_v7_zit_checkpoint(path: str) -> bool:
    """moodyRealMix ZIT V7 only (basename key; V6DPO etc. excluded)."""
    b = os.path.basename(path).lower()
    if "moodyrealmix" not in b:
        return False
    return "zitv7" in b or "_v7." in b or b.endswith("_v7.safetensors") or "_v7_" in b


def _moody_v7_keypattern_veto(model: nn.Module, hard_veto_layers: set) -> set:
    """V7-only: key-pattern hard VETO (no layer-name literals beyond suffix keys)."""
    added = set()
    for _n, _m in model.named_modules():
        if not isinstance(_m, torch.nn.Linear):
            continue
        if _n in hard_veto_layers:
            continue
        if _n.endswith(".attention.qkv"):
            added.add(_n)
            continue
        if _n.endswith((".cap_embedder.1", ".final_layer.linear", ".x_embedder")):
            added.add(_n)
            continue
        if ".context_refiner" in _n and _n.endswith(".attention.out"):
            added.add(_n)
    for _n in sorted(added):
        print(f"    [moodyV7 Key VETO] {_n}")
    return added


def _layer_weight_stats(tensor: torch.Tensor) -> tuple[float, float, float]:
    """Live kurtosis, outlier_ratio, abs_max for a weight tensor."""
    x = tensor.float()
    std = torch.std(x).item()
    amax = max(abs(x.min().item()), abs(x.max().item()))
    k = calculate_kurtosis(x)
    o = amax / std if std > 0 else 0.0
    return k, o, amax


def _moody_weight_profile_drift(weight_tensor: torch.Tensor, prof: dict) -> float:
    """moody-only: relative drift between live weights and distribution profile."""
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


def _quantize_fused_qkv_chunks(
    qkv_weight: torch.Tensor,
    amaxes: tuple[float, float, float],
) -> torch.Tensor:
    """moody-only: per-projection clamp then re-fuse to Comfy .attention.qkv.weight."""
    chunks = torch.chunk(qkv_weight, 3, dim=0)
    out_chunks = []
    for chunk, amax in zip(chunks, amaxes):
        a = max(float(amax), 1e-6)
        fp8 = torch.clamp(chunk.contiguous().float(), -a, a).to(torch.float8_e4m3fn)
        out_chunks.append(fp8)
    return torch.cat(out_chunks, dim=0)


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
) -> tuple[set, set]:
    """Gray-zone VETO release via trial MSE (moody path; Z-Anime keeps its own block)."""
    if not outlier_only_veto:
        return hard_veto_layers, keep_layers

    print(f"\n  [{scope_label} MSE-Guided Reassessment] {len(outlier_only_veto)} VETO layers are outlier-only (o>40, k<=20, m<=20).")
    print(f"  Trial-quantizing to measure actual HSWQ quantization error...")

    trial_optimizer = HSWQWeightedHistogramOptimizerV4(
        bins=8192, num_candidates=1000, refinement_iterations=10,
        device=device, alpha=alpha, beta=beta
    )

    safe_mses = []
    _module_dict = dict(model.named_modules())
    _safe_pool = [n for n in target_modules if n not in keep_layers and n in _module_dict]
    _safe_ff = [n for n in _safe_pool if "feed_forward" in n]
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
    mse_threshold = safe_mses[min(p75_idx, len(safe_mses) - 1)] * 2.0
    print(
        f"  [MSE Baseline] Safe layers sampled: {len(safe_mses)}, "
        f"P75 MSE: {safe_mses[p75_idx]:.8f}, Threshold (2xP75): {mse_threshold:.8f}"
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


def _enhanced_veto_scope_label(is_zanime: bool, is_moody: bool, cli_flag: bool, is_moody_v7: bool = False) -> str:
    if is_zanime:
        return "Z-Anime"
    if is_moody_v7:
        return "moodyV7"
    if is_moody:
        return "moodyZIT"
    if cli_flag:
        return "enhanced-veto"
    return "enhanced"


def load_zit_model(path, device="cuda", comfy_path=None):
    print(f"Loading Base model: {path}")
    original_state_dict = load_file(path)
    stripped_state_dict, detected_prefix, is_zanime, zanime_reverse_map = detect_and_strip_prefix(original_state_dict)
    if is_zanime:
        print(f"  [Z-Anime] Normalized {len(zanime_reverse_map)} keys. Model will be loaded with standard NextDiT keys.")
    
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
    
    # Z-Image / moody ZIT: FP16 calibration (proven r0.05 path; moody V6 SSIM 0.9919).
    # Z-Anime only: BF16 end-to-end calibration.
    use_bf16_calibration = is_zanime
    inference_dtype = torch.bfloat16 if is_zanime else torch.float16
    calib_label = "Z-Anime BF16 path" if is_zanime else "Z-Image FP16 path"
    bf16_frac = _bf16_weight_fraction(stripped_state_dict)
    if not is_zanime and bf16_frac >= 0.5:
        print(
            f"  [Note] CKPT weights are {bf16_frac * 100:.0f}% BF16; "
            f"calibration stays {inference_dtype} (v1.92-proven moody/ZIT path)."
        )
    print(f"  [Calibration dtype] {inference_dtype} ({calib_label})")
    
    nextdit_kwargs = {}
    if zit_config.get("qk_norm"):
        nextdit_kwargs["qk_norm"] = True
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
        dtype=inference_dtype,
        operations=ops,
        **nextdit_kwargs,
    )
    
    print("Loading Weights...")
    converted_state_dict = {}
    for key, value in stripped_state_dict.items():
        if is_zanime:
            converted_state_dict[key] = value
        elif value.dtype == torch.bfloat16:
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
    
    model = model.to(device).to(inference_dtype)
    model.eval()
    return (
        model,
        original_state_dict,
        stripped_state_dict,
        zit_config,
        detected_prefix,
        is_zanime,
        zanime_reverse_map,
        inference_dtype,
    )

class ZITCalibrationPipeline:
    def __init__(self, model, text_encoder, tokenizer, device="cuda", dtype=torch.float16):
        self.model = model
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.device = device
        # dtype is split per model family at load time:
        #   Z-Image  -> float16  (legacy proven path)
        #   Z-Anime  -> bfloat16 (native base dtype; required for SSIM >= 0.95)
        self.dtype = dtype
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
        run_dtype = self.dtype
        if self.text_encoder is not None:
            cap_feats, cap_mask = self.encode_prompt(prompt)
            cap_feats = cap_feats.to(dtype=run_dtype)
        else:
            cap_len = 256
            cap_feats = torch.randn(batch_size, cap_len, 2560, device=self.device, dtype=run_dtype)
            cap_mask = torch.ones(batch_size, cap_len, device=self.device, dtype=torch.bool)
        
        import comfy.k_diffusion.sampling as k_sampling
        class ZITWrapper:
            def __init__(self, model, cap_feats, cap_mask, dtype):
                self.model = model
                self.cap_feats = cap_feats
                self.cap_mask = cap_mask
                self.dtype = dtype
            def __call__(self, x, sigma, **kwargs):
                dtype = self.dtype
                try:
                    return self.model(x.to(dtype=dtype), sigma.to(dtype=dtype), self.cap_feats.to(dtype=dtype), None, attention_mask=self.cap_mask).to(dtype=x.dtype)
                except: return torch.zeros_like(x)

        x = torch.randn(batch_size, latent_c, latent_h, latent_w, device=self.device, dtype=run_dtype)
        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=self.device)
        model_wrap = ZITWrapper(self.model, cap_feats, cap_mask, run_dtype)
        
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


def _is_zanime_profile(profile):
    """A ZA profile is built from the Diffusers-form BF16 base, so it contains
    `.attention.to_q.weight` keys. Z-Image / ZIB / ZIT profiles never do
    (their attention is already fused as `.attention.qkv.weight`)."""
    if not profile:
        return False
    return any(isinstance(k, str) and k.endswith(".attention.to_q.weight") for k in profile)


def _convert_zanime_profile_to_nextdit(profile):
    """[Z-Anime profile namespace bridge — pure key renaming, no threshold change]

    `analyze_zib_distribution.py` builds the profile from the ZA BF16 base, whose
    state_dict uses Diffusers attention naming (to_q / to_k / to_v / to_out.0 /
    norm_q / norm_k). The downstream HSWQ pipeline — derive_hswq_strategy(),
    get_dynamic_search_low(), `_norm_profile`, hard_veto matching at L828/L968,
    and the dynamic-keep profile_score at L834 — all see the FUSED NextDiT
    module namespace produced by `_fuse_zanime_attention` (qkv / out / q_norm /
    k_norm). Without bridging, profile lookups miss and HSWQ's VETO judgements
    cannot reach the final keep_layers application site.

    This bridge does ONLY namespace alignment:
      * 3 per-projection entries (to_q/to_k/to_v) -> 1 fused `qkv` entry
        aggregated by per-statistic MAX. The fused weight is literally
        torch.cat([Wq, Wk, Wv], dim=0); its abs_max / kurtosis / outlier_ratio
        are dominated by whichever projection is most extreme. This is the
        mathematical consequence of concatenation, not a hardcoded heuristic.
      * to_out.0 -> out, norm_q -> q_norm, norm_k -> k_norm renames mirror
        the renames already performed by `_fuse_zanime_attention`.

    All HSWQ thresholds (k>20, o>40, m>20), alpha/beta derivation, search_low
    formulas, and VETO judgement code remain untouched.
    """
    converted = {}
    qkv_buckets = {}  # fused qkv key -> list of (to_q/to_k/to_v profile dicts)

    for key, prof in profile.items():
        if not isinstance(prof, dict):
            converted[key] = prof
            continue

        new_key = key

        # Step 1: strip 'all_<module>.2-1' ZA prefix (mirrors normalize_zanime_keys Step 1).
        # ZA profile is built directly from the BF16 base file, which still carries this
        # prefix; without stripping it the namespace won't match the post-normalize state_dict.
        if new_key.startswith("all_"):
            new_key = re.sub(r"^all_(.*?)\.2-1", r"\1", new_key)

        # Step 2: Diffusers -> Lumina renames (mirrors _fuse_zanime_attention).
        new_key = re.sub(r"\.attention\.to_out\.0\.weight$", ".attention.out.weight", new_key)
        new_key = re.sub(r"\.attention\.norm_q\.weight$", ".attention.q_norm.weight", new_key)
        new_key = re.sub(r"\.attention\.norm_k\.weight$", ".attention.k_norm.weight", new_key)

        # Step 3: detect Diffusers attention projection and bucket for qkv fusion.
        m = re.match(r"^(.*\.attention\.)(to_q|to_k|to_v)\.weight$", new_key)
        if m:
            qkv_key = m.group(1) + "qkv.weight"
            qkv_buckets.setdefault(qkv_key, []).append(prof)
            continue

        converted[new_key] = prof

    # Aggregate to_q/to_k/to_v into qkv via per-statistic MAX (concatenation dominance).
    for qkv_key, projs in qkv_buckets.items():
        agg = {}
        for stat in ("kurtosis", "outlier_ratio", "abs_max", "std"):
            agg[stat] = max((p.get(stat, 0) for p in projs), default=0)
        converted[qkv_key] = agg

    return converted


def derive_hswq_strategy(
    model_profile,
    is_zanime=False,
    use_bf16_calibration=False,
    is_moody_zit=False,
):
    """
    [Pure Data-Driven Engine]
    Derives Alpha/Beta from global model statistics and returns a continuous
    evaluation function that decides per-layer search_low without hardcoded thresholds.

    use_bf16_calibration: Z-Anime only — upper_clip 0.90.
    is_moody_zit: moodyRealMix only — gray-zone layers get upper_clip 0.90 on FP16 path.
    ZI/ZIB/ZIT: upper_clip 0.99 unchanged.
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

        # Z-Anime: v1.92 path unchanged (is_zanime gate only).
        if is_zanime:
            if prof:
                k_stat = prof.get("kurtosis", 0)
                o_ratio = prof.get("outlier_ratio", 0)
            else:
                t_f32 = weight_tensor.float()
                k_stat = calculate_kurtosis(t_f32)
                std = torch.std(t_f32).item()
                abs_max = max(abs(t_f32.min().item()), abs(t_f32.max().item()))
                o_ratio = float(abs_max / std if std > 0 else 0)
            k_penalty = min(k_stat / 100.0, 0.49)
            o_penalty = min(o_ratio / 60.0, 0.49)
            upper_clip = 0.90 if is_zanime else 0.99
            return float(np.clip(0.50 + max(k_penalty, o_penalty), 0.50, upper_clip))

        if prof:
            k_stat = prof.get("kurtosis", 0)
            o_ratio = prof.get("outlier_ratio", 0)
            m_stat = prof.get("abs_max", 0)
        else:
            k_stat, o_ratio, m_stat = _layer_weight_stats(weight_tensor)

        k_penalty = min(k_stat / 100.0, 0.49)
        o_penalty = min(o_ratio / 60.0, 0.49)

        upper_clip = 0.99
        if is_moody_zit:
            in_gray = (
                (10 < k_stat <= 20)
                or (30 < o_ratio <= 40)
                or (5 < m_stat <= 20)
            )
            if in_gray:
                upper_clip = 0.90
        return float(np.clip(0.50 + max(k_penalty, o_penalty), 0.50, upper_clip))

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


def resolve_weights_path(raw_path: str, script_dir: str) -> tuple[str, list[str]]:
    """Resolve .safetensors path when CWD differs from repo root (Docker/CI).

    Order: HSWQ_ZIB_INPUT, ZIB_INPUT_MODEL, abspath(raw), script_dir/raw,
    script_dir/basename(raw).
    Returns (first existing file path, or abspath(raw) if none), list of tried paths.
    """
    tried: list[str] = []
    candidates: list[str] = []
    for env_key in ("HSWQ_ZIB_INPUT", "ZIB_INPUT_MODEL"):
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
    parser.add_argument(
        "--enhanced-veto",
        action="store_true",
        help="Enable Structural + per-projection qkv VETO (auto for Z-Anime and moodyRealMix filenames)",
    )
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
        print("        or set HSWQ_ZIB_INPUT / ZIB_INPUT_MODEL to the model file.")
        sys.exit(1)
    cli_abs = os.path.normpath(os.path.abspath(os.path.expanduser(raw_input_arg)))
    if os.path.normpath(resolved_input) != cli_abs:
        print(f"[*] Resolved --input: {raw_input_arg!r} -> {resolved_input}")
    args.input = resolved_input

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("HSWQ V1.9 Autonomous Engine (Environment-Aware Analysis)")
    print("=" * 60)

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
    # [Z-Anime profile namespace bridge] ZA profile is built from the Diffusers-form
    # BF16 base, but downstream HSWQ pipeline (derive_hswq_strategy / get_dynamic_search_low /
    # _norm_profile / keep_layers matching at L828/L968 / dynamic profile_score at L834)
    # operates on NextDiT fused module names. Bridge purely by key renaming so VETO
    # judgements actually reach the final keep_layers application site.
    is_zanime_profile_flag = bool(model_profile) and _is_zanime_profile(model_profile)
    if is_zanime_profile_flag:
        n_before = len(model_profile)
        model_profile = _convert_zanime_profile_to_nextdit(model_profile)
        print(f"  [Z-Anime profile bridge] Diffusers (to_q/to_k/to_v/to_out.0/norm_q/norm_k) ->")
        print(f"    fused NextDiT (qkv/out/q_norm/k_norm) via per-statistic MAX.")
        print(f"    Profile entries: {n_before} -> {len(model_profile)}")
    is_moody_zit = detect_moody_zit_checkpoint(args.input)
    is_moody_v7 = detect_moody_v7_zit_checkpoint(args.input)
    use_bf16_cal_precheck = is_zanime_profile_flag
    alpha, beta, get_layer_search_low, hard_veto_layers = derive_hswq_strategy(
        model_profile,
        is_zanime=is_zanime_profile_flag,
        use_bf16_calibration=use_bf16_cal_precheck,
        is_moody_zit=is_moody_zit,
    )
    (
        model,
        original_state_dict,
        stripped_state_dict,
        zit_config,
        detected_prefix,
        is_zanime,
        zanime_reverse_map,
        inference_dtype,
    ) = load_zit_model(args.input, device, args.comfy_path)
    use_enhanced_veto = is_moody_zit or args.enhanced_veto
    if use_enhanced_veto:
        _ev_label = _enhanced_veto_scope_label(False, is_moody_zit, args.enhanced_veto, is_moody_v7)
        print(f"  [Enhanced VETO] Enabled ({_ev_label}): Structural + per-projection qkv VETO.")
    if is_moody_v7:
        print("  [moodyV7] Key-pattern hard VETO will protect all .attention.qkv + boundary layers.")
    
    # tokenizer and text_encoder are already loaded in the initial block
    pipeline = ZITCalibrationPipeline(model, text_encoder, tokenizer, device, dtype=inference_dtype)

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

    # [Z-Anime Structural VETO] Identify Linear layers whose weight shape is unique
    # within the model. These are typically boundary / projection layers (e.g.
    # cap_embedder.1 [3840, 2560] for text->DiT bridge, final_layer.linear [64, 3840]
    # for output projection) that the data-driven (k/o/m) thresholds may miss but
    # which strongly affect SSIM. No layer names are hardcoded; selection is purely
    # structural via shape uniqueness over Linear weights of the loaded model.
    # Guarded by is_zanime so ZI/ZIB/ZIT behavior is strictly unchanged.
    if is_zanime:
        shape_count = {}
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
                    print(f"    [Structural VETO] {_n} shape={list(_shp)} (uniqueness=1)")
        if structural_veto:
            hard_veto_layers = hard_veto_layers.union(structural_veto)
            print(f"  [Z-Anime Structural VETO] Added {len(structural_veto)} unique-shape layers (total VETO: {len(hard_veto_layers)}).")
        else:
            print(f"  [Z-Anime Structural VETO] No additional unique-shape layers found.")
    elif use_enhanced_veto:
        shape_count = {}
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
                    print(f"    [Structural VETO] {_n} shape={list(_shp)} (uniqueness=1)")
        if structural_veto:
            hard_veto_layers = hard_veto_layers.union(structural_veto)
            _sv_label = _enhanced_veto_scope_label(False, is_moody_zit, args.enhanced_veto, is_moody_v7)
            print(f"  [{_sv_label} Structural VETO] Added {len(structural_veto)} unique-shape layers (total VETO: {len(hard_veto_layers)}).")
        else:
            print(f"  [Structural VETO] No additional unique-shape layers found.")

    # [Z-Anime Per-Projection qkv VETO] Auto-detect attention.qkv layers via key
    # pattern (no hardcoded layer names) and split the fused weight into to_q /
    # to_k / to_v chunks. If any per-projection abs_max exceeds the FP8 E4M3
    # safe range threshold (m > 5.0, same magnitude scale as the existing
    # data-driven m > 20 hard_veto), add the qkv layer to hard_veto_layers.
    # Per-projection split is the same operation already performed during
    # quantization for is_zanime, so the threshold is applied on the actual
    # quantization unit, not the fused statistic.
    if is_zanime:
        proj_veto = set()
        for _n, _m in model.named_modules():
            if isinstance(_m, torch.nn.Linear) and _n.endswith(".attention.qkv"):
                if _n in hard_veto_layers:
                    continue
                _w = _m.weight.detach().float()
                _out_dim = _w.shape[0]
                if _out_dim % 3 != 0:
                    continue
                _chunk = _out_dim // 3
                _amax = [_w[i * _chunk:(i + 1) * _chunk].abs().max().item() for i in range(3)]
                if max(_amax) > 5.0:
                    proj_veto.add(_n)
                    _tags = ["to_q", "to_k", "to_v"]
                    _hi = ", ".join(f"{t}={a:.2f}" for t, a in zip(_tags, _amax) if a > 5.0)
                    print(f"    [Per-Projection VETO] {_n} ({_hi})")
        if proj_veto:
            hard_veto_layers = hard_veto_layers.union(proj_veto)
            print(f"  [Z-Anime Per-Projection VETO] Added {len(proj_veto)} qkv layers (total VETO: {len(hard_veto_layers)}).")
        else:
            print(f"  [Z-Anime Per-Projection VETO] No qkv layer exceeds per-projection abs_max threshold.")
    elif use_enhanced_veto:
        proj_veto = set()
        for _n, _m in model.named_modules():
            if isinstance(_m, torch.nn.Linear) and _n.endswith(".attention.qkv"):
                if _n in hard_veto_layers:
                    continue
                _w = _m.weight.detach().float()
                _out_dim = _w.shape[0]
                if _out_dim % 3 != 0:
                    continue
                _chunk = _out_dim // 3
                _amax = [_w[i * _chunk:(i + 1) * _chunk].abs().max().item() for i in range(3)]
                if max(_amax) > 5.0:
                    proj_veto.add(_n)
                    _tags = ["to_q", "to_k", "to_v"]
                    _hi = ", ".join(f"{t}={a:.2f}" for t, a in zip(_tags, _amax) if a > 5.0)
                    print(f"    [Per-Projection VETO] {_n} ({_hi})")
        if proj_veto:
            hard_veto_layers = hard_veto_layers.union(proj_veto)
            _pv_label = _enhanced_veto_scope_label(False, is_moody_zit, args.enhanced_veto, is_moody_v7)
            print(f"  [{_pv_label} Per-Projection VETO] Added {len(proj_veto)} qkv layers (total VETO: {len(hard_veto_layers)}).")
        else:
            print(f"  [Per-Projection VETO] No qkv layer exceeds per-projection abs_max threshold.")

    if is_moody_v7:
        _kv = _moody_v7_keypattern_veto(model, hard_veto_layers)
        if _kv:
            hard_veto_layers = hard_veto_layers.union(_kv)
            print(f"  [moodyV7 Key VETO] Added {len(_kv)} layers (total VETO: {len(hard_veto_layers)}).")

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
    _module_dict_sens = dict(model.named_modules())
    layer_sensitivities = []
    for name in target_modules:
        if name in hard_veto_layers:
            continue  # Remove VETO layers from the candidate pool
        prof = _norm_profile.get(name, {})
        k = prof.get("kurtosis", 0)
        o = prof.get("outlier_ratio", 0)
        m = prof.get("abs_max", 0)
        profile_score = k + o * 2.0 + m * 0.5
        if is_moody_zit and name in _module_dict_sens:
            _mw = _module_dict_sens[name]
            if hasattr(_mw, "weight"):
                drift = _moody_weight_profile_drift(_mw.weight.data, prof)
                profile_score += drift * 50.0
        layer_sensitivities.append((name, profile_score))
    
    layer_sensitivities.sort(key=lambda x: x[1], reverse=True)
    num_keep_dynamic = int(len(layer_sensitivities) * args.keep_ratio)
    dynamic_keep_layers = set([x[0] for x in layer_sensitivities[:num_keep_dynamic]])
    
    # [V1.92 Exclusive Protection] VETO (always FP16) + Dynamic (additional FP16) with no overlap for maximum coverage
    keep_layers = dynamic_keep_layers.union(hard_veto_layers)
    
    # =========================================================================
    # [V1.92 MSE-Guided VETO Reassessment] — Z-Anime ONLY
    # Layers VETO'd *only* by outlier_ratio (o>40), NOT by kurtosis or magnitude,
    # are candidates for automatic release. These are typically feed_forward.w2
    # layers that HSWQ's optimal clipping may handle well.
    # Guarded by is_zanime so ZI/ZIB/ZIT behavior is strictly unchanged.
    #
    # Strategy:
    #   1. Identify "outlier-only" VETO layers (o>40 but k<=20 and m<=20)
    #   2. Trial-quantize a random sample of SAFE layers to get baseline MSE
    #   3. Trial-quantize each outlier-only VETO candidate
    #   4. If candidate MSE <= P75 of safe MSE distribution → release from VETO
    # =========================================================================
    outlier_only_veto = set()
    if is_zanime:
        for vname in hard_veto_layers:
            prof = _norm_profile.get(vname, {})
            k = prof.get("kurtosis", 0)
            m = prof.get("abs_max", 0)
            o = prof.get("outlier_ratio", 0)
            # Only layers where outlier_ratio was the sole trigger
            if o > 40 and k <= 20 and m <= 20:
                outlier_only_veto.add(vname)
    
    if outlier_only_veto:
        print(f"\n  [MSE-Guided Reassessment] {len(outlier_only_veto)} VETO layers are outlier-only (o>40, k<=20, m<=20).")
        print(f"  Trial-quantizing to measure actual HSWQ quantization error...")
        
        trial_optimizer = HSWQWeightedHistogramOptimizerV4(
            bins=8192, num_candidates=1000, refinement_iterations=10,
            device=device, alpha=alpha, beta=beta
        )
        
        # Step 1: Collect baseline MSE from safely-quantized layers (non-VETO, non-Dynamic)
        safe_mses = []
        _module_dict = dict(model.named_modules())
        _safe_sample = [n for n in target_modules if n not in keep_layers and n in _module_dict]
        # Sample up to 30 safe layers for baseline
        import random
        _safe_sample = random.sample(_safe_sample, min(30, len(_safe_sample)))
        for sname in _safe_sample:
            smod = _module_dict[sname]
            if not hasattr(smod, 'weight'):
                continue
            sw = smod.weight.data
            slayer_search_low = get_layer_search_low(sname, sw)
            try:
                sresult = trial_optimizer.compute_optimal_amax_with_stats(
                    sw, importance=None, use_svd_leverage=True, scaled=False
                )
                safe_mses.append(sresult['estimated_mse'])
            except Exception:
                pass
            torch.cuda.empty_cache()
        
        if safe_mses:
            safe_mses.sort()
            # P75 = 75th percentile of safe layer MSE
            p75_idx = int(len(safe_mses) * 0.75)
            mse_threshold = safe_mses[min(p75_idx, len(safe_mses) - 1)]
            # Safety margin: allow up to 2x the P75 threshold
            mse_threshold *= 2.0
            print(f"  [MSE Baseline] Safe layers sampled: {len(safe_mses)}, P75 MSE: {safe_mses[p75_idx] if p75_idx < len(safe_mses) else safe_mses[-1]:.8f}, Threshold (2×P75): {mse_threshold:.8f}")
            
            # Step 2: Trial-quantize each outlier-only VETO candidate
            released = set()
            for vname in sorted(outlier_only_veto):
                if vname not in _module_dict:
                    continue
                vmod = _module_dict[vname]
                if not hasattr(vmod, 'weight'):
                    continue
                vw = vmod.weight.data
                try:
                    vresult = trial_optimizer.compute_optimal_amax_with_stats(
                        vw, importance=None, use_svd_leverage=True, scaled=False
                    )
                    vmse = vresult['estimated_mse']
                    vprof = _norm_profile.get(vname, {})
                    vor = vprof.get("outlier_ratio", 0)
                    if vmse <= mse_threshold:
                        released.add(vname)
                        print(f"    RELEASED: {vname} | MSE={vmse:.8f} <= threshold={mse_threshold:.8f} | o={vor:.1f} | amax={vresult['optimal_amax']:.4f}")
                    else:
                        print(f"    KEPT:     {vname} | MSE={vmse:.8f} >  threshold={mse_threshold:.8f} | o={vor:.1f}")
                except Exception as e:
                    print(f"    ERROR:    {vname} | {e}")
                torch.cuda.empty_cache()
            
            if released:
                hard_veto_layers = hard_veto_layers - released
                keep_layers = keep_layers - released
                print(f"  [MSE-Guided Reassessment] Released {len(released)} layers from VETO. Remaining VETO: {len(hard_veto_layers)}.")
                print(f"  Updated FP16 kept layers: {len(keep_layers)}")
            else:
                print(f"  [MSE-Guided Reassessment] No layers released (all exceeded MSE threshold).")
        else:
            print(f"  [MSE-Guided Reassessment] No safe baseline available, skipping.")

    # moodyRealMix only — same gray-zone MSE release as Z-Anime block, but separate code path.
    if is_moody_zit:
        moody_outlier_only_veto = set()
        for vname in hard_veto_layers:
            prof = _norm_profile.get(vname, {})
            k = prof.get("kurtosis", 0)
            m = prof.get("abs_max", 0)
            o = prof.get("outlier_ratio", 0)
            if o > 40 and k <= 20 and m <= 20:
                moody_outlier_only_veto.add(vname)
        if moody_outlier_only_veto:
            _mgy_label = "moodyV7" if is_moody_v7 else "moodyZIT"
            hard_veto_layers, keep_layers = _mse_grayzone_veto_reassessment(
                scope_label=_mgy_label,
                hard_veto_layers=hard_veto_layers,
                keep_layers=keep_layers,
                outlier_only_veto=moody_outlier_only_veto,
                target_modules=target_modules,
                model=model,
                _norm_profile=_norm_profile,
                get_layer_search_low=get_layer_search_low,
                alpha=alpha,
                beta=beta,
                device=device,
            )
    
    non_veto_total = len([n for n in target_modules if n not in hard_veto_layers])
    print(f"\nTotal layers: {len(target_modules)} (Non-VETO pool: {non_veto_total})")
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

    print("\n[HSWQ V1.9 Autonomous Engine] Starting Optimization...")
    if is_zanime:
        print("  [Z-Anime] qkv layers will be split into to_q / to_k / to_v and HSWQ-optimized individually.")
    if is_moody_zit:
        print("  [moodyZIT] qkv: per-projection HSWQ; output stays fused .attention.qkv.weight (Comfy).")
    weight_amax_dict = {}
    # Z-Anime only: per-layer split-amax map for qkv -> to_q/to_k/to_v.
    # Maps internal qkv module name to (amax_q, amax_k, amax_v).
    zanime_qkv_split_amax = {}
    moody_qkv_split_amax = {}
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
            
            # Z-Anime split path: optimize to_q / to_k / to_v individually.
            # Same HSWQ V4 pipeline (alpha/beta/search_range/SVD leverage) applied
            # per-chunk so each projection gets its own optimal clipping threshold.
            if is_zanime and name.endswith(".attention.qkv"):
                qkv_w = module.weight.data
                # qkv weight shape: [3 * dim_out, dim_in], even split along dim=0
                chunks = torch.chunk(qkv_w, 3, dim=0)
                if len(chunks) != 3 or any(c.shape[0] != chunks[0].shape[0] for c in chunks):
                    print(f"  [WARN] qkv split mismatch at {name}; falling back to fused amax.")
                    optimal_amax = hswq_optimizer.compute_optimal_amax(
                        qkv_w, importance,
                        use_svd_leverage=True, scaled=False,
                        search_range=layer_search_range,
                    )
                    weight_amax_dict[name + ".weight"] = optimal_amax
                else:
                    print(f"  [HSWQ-split] {name:50} | per-projection | search_range={layer_search_range[0]:.3f}-{layer_search_range[1]:.3f}")
                    amaxes = []
                    for tag, chunk in zip(("to_q", "to_k", "to_v"), chunks):
                        # Importance is per input-channel and shared across q/k/v
                        # because all three projections take the same hidden input.
                        a = hswq_optimizer.compute_optimal_amax(
                            chunk.contiguous(), importance,
                            use_svd_leverage=True, scaled=False,
                            search_range=layer_search_range,
                        )
                        amaxes.append(a)
                        print(f"    [HSWQ-split]   {tag}: amax={a:.6f}")
                    zanime_qkv_split_amax[name] = tuple(amaxes)
                torch.cuda.empty_cache()
                continue

            # moodyZIT: per-projection amax, fused Comfy qkv save (not Diffusers split).
            if is_moody_zit and name.endswith(".attention.qkv"):
                qkv_w = module.weight.data
                chunks = torch.chunk(qkv_w, 3, dim=0)
                if len(chunks) != 3 or any(c.shape[0] != chunks[0].shape[0] for c in chunks):
                    print(f"  [WARN] moody qkv split mismatch at {name}; falling back to fused amax.")
                    optimal_amax = hswq_optimizer.compute_optimal_amax(
                        qkv_w, importance,
                        use_svd_leverage=True, scaled=False,
                        search_range=layer_search_range,
                    )
                    weight_amax_dict[name + ".weight"] = optimal_amax
                else:
                    print(f"  [HSWQ-moody] {name:50} | per-projection | search_range={layer_search_range[0]:.3f}-{layer_search_range[1]:.3f}")
                    amaxes = []
                    for tag, chunk in zip(("q", "k", "v"), chunks):
                        a = hswq_optimizer.compute_optimal_amax(
                            chunk.contiguous(), importance,
                            use_svd_leverage=True, scaled=False,
                            search_range=layer_search_range,
                        )
                        amaxes.append(a)
                        print(f"    [HSWQ-moody]   {tag}: amax={a:.6f}")
                    moody_qkv_split_amax[name] = tuple(amaxes)
                torch.cuda.empty_cache()
                continue
            
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
    if is_zanime:
        print("  [Z-Anime] Output will use Diffusers key format (to_q/to_k/to_v/to_out.0/norm_q/norm_k + 'all_<module>.2-1' prefix), matching the official FP8 distribution.")
    output_state_dict = {}

    # ZI/ZIB/ZIT / moody ZIT: keep FP16. Z-Anime BF16 path: keep BF16.
    keep_dtype = torch.bfloat16 if is_zanime else torch.float16

    def _emit_quant_meta(out_dict, prefixed_module):
        out_dict[f"{prefixed_module}.comfy_quant"] = torch.tensor(
            list(json.dumps({"format": "float8_e4m3fn"}).encode("utf-8")),
            dtype=torch.uint8,
        )
        out_dict[f"{prefixed_module}.weight_scale"] = torch.tensor(1.0, dtype=torch.float32)

    for stripped_key, value in tqdm(stripped_state_dict.items(), desc="Converting"):
        module_name = stripped_key[:-7] if stripped_key.endswith(".weight") else None

        # Z-Anime split-save path for qkv: emit 3 separate FP8 weights
        # (to_q / to_k / to_v) using each projection's individual HSWQ amax.
        if is_zanime and module_name and module_name in zanime_qkv_split_amax:
            base = module_name[: -len(".qkv")]  # strip trailing '.qkv'
            qkv_w = value
            chunks = torch.chunk(qkv_w, 3, dim=0)
            amaxes = zanime_qkv_split_amax[module_name]
            for tag, chunk, amax in zip(("to_q", "to_k", "to_v"), chunks, amaxes):
                a = max(float(amax), 1e-6)
                fp8_chunk = torch.clamp(chunk.contiguous().float(), -a, a).to(torch.float8_e4m3fn)
                tgt_module = f"{base}.{tag}"
                tgt_key = f"{detected_prefix}{tgt_module}.weight"
                output_state_dict[tgt_key] = fp8_chunk
                _emit_quant_meta(output_state_dict, f"{detected_prefix}{tgt_module}")
            continue

        # moodyZIT: fused qkv FP8 with per-projection amax (Comfy key unchanged).
        if is_moody_zit and module_name and module_name in moody_qkv_split_amax:
            fp8_fused = _quantize_fused_qkv_chunks(value, moody_qkv_split_amax[module_name])
            out_key = detected_prefix + stripped_key
            output_state_dict[out_key] = fp8_fused
            _emit_quant_meta(output_state_dict, detected_prefix + module_name)
            continue

        if module_name and module_name in keep_layers:
            new_value = value.to(keep_dtype)
            # Z-Anime keep path: if qkv is kept (FP16/BF16, no quant), still split
            # into to_q/to_k/to_v so the saved layout matches Diffusers format.
            if is_zanime and module_name.endswith(".attention.qkv"):
                base = module_name[: -len(".qkv")]
                chunks = torch.chunk(new_value, 3, dim=0)
                for tag, chunk in zip(("to_q", "to_k", "to_v"), chunks):
                    output_state_dict[f"{detected_prefix}{base}.{tag}.weight"] = chunk.contiguous().clone()
                continue
        elif stripped_key in weight_amax_dict or (module_name and module_name + ".weight" in weight_amax_dict):
            weight_key = stripped_key if stripped_key in weight_amax_dict else module_name + ".weight"
            amax = max(weight_amax_dict[weight_key], 1e-6)
            new_value = torch.clamp(value.float(), -amax, amax).to(torch.float8_e4m3fn)
            if module_name:
                _emit_quant_meta(output_state_dict, detected_prefix + module_name)
        else:
            if is_zanime:
                new_value = value.to(torch.bfloat16) if value.dtype != torch.bfloat16 else value
            else:
                new_value = value.to(torch.float16) if value.dtype == torch.bfloat16 else value

        out_key = detected_prefix + stripped_key
        output_state_dict[out_key] = new_value

    # Z-Anime: rewrite NextDiT keys back to Diffusers form
    # (out -> to_out.0, q_norm/k_norm -> norm_q/norm_k, restore 'all_<module>.2-1' prefix).
    # qkv weights have already been split per-projection above and emitted with
    # individual HSWQ amax values, so this pass only handles renames + prefix.
    if is_zanime:
        before_n = len(output_state_dict)
        output_state_dict = _denormalize_zanime_output(output_state_dict, zanime_reverse_map)
        after_n = len(output_state_dict)
        print(f"  [Z-Anime] Diffusers key restoration: {before_n} -> {after_n} keys.")

    save_file(output_state_dict, args.output)
    print("Saved.")

if __name__ == "__main__":
    main()

"""Krea2-only plain NVFP4 converter (no ConvRot) — Kitchen-faithful pack.

Pack / metadata behavior matches Kitchen convert_to_nvfp4_node.py:
  https://github.com/tritant/ComfyUI_Kitchen_nvfp4_Converter

  - 2D .weight → TensorCoreNVFP4Layout.quantize + state_dict_tensors
  - Krea2 blacklist keeps structure-sensitive layers in bfloat16
  - Non-matching / non-2D tensors kept as bfloat16
  - Metadata: _quantization_metadata + converted_by / converter_url
  - No ConvRot, no calib, no input_scale, no bias correction

Krea2 signature (FATAL if missing): txtfusion.projector + blocks.0.attn.wq
under model.diffusion_model. / diffusion_model. / root.

Blacklist (Krea2 DiT — first/last must stay BF16 so ComfyUI can infer
channels from weight shapes; protect mod/norm/projector/tmlp/tproj):
  first, last, mod., norm, projector, tmlp, tproj, bias, vae., text_encoders

SmoothQuant (optional, requires --calib_file + --clip_path):
  Per-channel scaling s = (act_rms^α) / (weight_abs_max^α) migrates
  activation outliers into weights before NVFP4 quantization. The scale
  tensor .smoothquant_scale is saved per-layer; runtime divides input by s.
  s = (max|x|^α) / (max|W|^α)  →  W' = W * s, x' = x / s
  Mathematically lossless (x' @ W'^T = x @ W^T) but improves NVFP4
  quantization by compressing activation dynamic range.
  Orthogonal to Hadamard rotation: SmoothQuant fixes channel-wise scale
  imbalance; rotation decorrelates element-wise correlations.

For FULL ConvRot + DualMonitor calib, use hswq_convert_nvfp4_krea2.py.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from collections import OrderedDict

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

try:
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout
except ImportError:
    print("Error: comfy_kitchen not found (install in the active venv).")
    sys.exit(1)

_MODEL_TYPE = "Krea2"
_SMOOTHQUANT_ALPHA_DEFAULT = 0.5

# Krea2 SingleStreamDiT — structure-sensitive layers stay BF16.
_KREA2_BLACKLIST: list[str] = [
    "first",
    "last",
    "mod.",
    "norm",
    "projector",
    "tmlp",
    "tproj",
    "bias",
    "vae.",
    "text_encoders",
]
_KREA2_FP8_LAYERS: list[str] = []

_NON_DIFFUSION_MARKERS: tuple[str, ...] = (
    "conditioner.",
    "cond_stage_model.",
    "text_encoders.",
    "text_encoder.",
    "text_encoder_2.",
    "text_encoder_3.",
    "text_model.",
    "text_projection",
    "logit_scale",
    "clip_l.",
    "clip_g.",
    "t5xxl.",
    "first_stage_model.",
    "vae.",
)


def _is_non_diffusion_key(key: str) -> bool:
    return any(marker in key for marker in _NON_DIFFUSION_MARKERS)


def _find_krea2_key_prefix(state_dict) -> str:
    for prefix in ("model.diffusion_model.", "diffusion_model.", ""):
        if f"{prefix}txtfusion.projector.weight" in state_dict:
            if f"{prefix}blocks.0.attn.wq.weight" not in state_dict:
                raise ValueError(
                    "Krea2 signature incomplete: txtfusion.projector present but "
                    f"{prefix}blocks.0.attn.wq.weight missing"
                )
            return prefix
    raise ValueError(
        "Not a Krea2 checkpoint: missing txtfusion.projector.weight "
        "(under model.diffusion_model. / diffusion_model. / root)."
    )


def _meta_base_key(base_k_file: str) -> str:
    if "model.diffusion_model." in base_k_file:
        return base_k_file.split("model.diffusion_model.")[-1]
    if "diffusion_model." in base_k_file:
        return base_k_file.split("diffusion_model.")[-1]
    return base_k_file


# =========================================================================
# SmoothQuant helpers
# =========================================================================

def _compute_smoothquant_scale(
    act_sq: torch.Tensor,
    weight: torch.Tensor,
    alpha: float = _SMOOTHQUANT_ALPHA_DEFAULT,
) -> torch.Tensor:
    """Compute per-channel SmoothQuant migration scale.

    s_j = (act_rms_j ^ α) / (weight_abs_max_j ^ α)

    Normalized to median = 1.0 so the average per-tensor NVFP4 scale is
    unaffected; only the RELATIVE per-channel balance changes.

    Args:
        act_sq: (in_features,) per-channel E[x^2] from calibration.
        weight: (out_features, in_features) weight tensor (float32 or bfloat16).
        alpha: migration strength (0.0 = no migration, 1.0 = full to weight).

    Returns:
        (in_features,) float32 scale tensor s.
    """
    act_rms = act_sq.to(torch.float32).sqrt().clamp(min=1e-8)
    weight_abs_max = weight.to(torch.float32).abs().amax(dim=0).clamp(min=1e-8)
    s = (act_rms ** alpha) / (weight_abs_max ** alpha)
    # Normalize: median = 1.0 (keep average scale near unity)
    s_median = s.median().clamp(min=1e-8)
    s = s / s_median
    # Clamp extreme values to protect NVFP4 weight quantization
    s = s.clamp(min=1e-4, max=1e4)
    return s


def _apply_smoothquant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """W' = W * s (scale each column j of W by s_j)."""
    return weight * scale.unsqueeze(0).to(weight.dtype)


# =========================================================================
# Main convert
# =========================================================================

def convert_to_nvfp4(
    input_path: str,
    output_path: str,
    device: str,
    calib_file: str | None = None,
    clip_path: str | None = None,
    comfy_path: str | None = None,
    smoothquant_alpha: float = _SMOOTHQUANT_ALPHA_DEFAULT,
    num_calib_samples: int = 32,
    num_inference_steps: int = 25,
):
    use_smoothquant = bool(calib_file) and bool(clip_path)
    mode_suffix = " + SmoothQuant" if use_smoothquant else ""
    print(f"Mode {_MODEL_TYPE} | device={device} | plain NVFP4{mode_suffix} (Krea2-only)")

    sd = load_file(input_path)
    prefix = _find_krea2_key_prefix(sd)
    print(f"Detected Krea2 key prefix: {prefix!r}")

    # --- Optional: SmoothQuant calibration ---
    act_sq_dict: dict[str, torch.Tensor] = {}
    act_amax_dict: dict[str, float] = {}
    ck_map: dict[str, str] = {}
    if use_smoothquant:
        print("\n=== SmoothQuant Calibration ===")
        try:
            # Lazy import: only needed when calibration is requested.
            # Reuses the full DualMonitor + CLIP + DiT pipeline from
            # auto_int8_nvfp4_hybrid.py (not modified).
            from auto_int8_nvfp4_hybrid import run_calibration
        except ImportError as e:
            print(f"Error: cannot import run_calibration from auto_int8_nvfp4_hybrid: {e}")
            print("       Ensure auto_int8_nvfp4_hybrid.py is in the same directory.")
            sys.exit(1)

        act_sq_dict, act_amax_dict, ck_map = run_calibration(
            input_path,
            calib_file,
            clip_path,
            num_calib_samples,
            num_inference_steps,
            device,
            comfy_path,
        )
        print(
            f"  Calibration done: {len(act_sq_dict)} layers with act_sq, "
            f"{len(act_amax_dict)} layers with act_amax"
        )
    else:
        print("\n[SKIP] No --calib_file/--clip_path — SmoothQuant disabled")

    blacklist = list(_KREA2_BLACKLIST)
    fp8_layers = list(_KREA2_FP8_LAYERS)
    quant_map = {"format_version": "1.0", "layers": {}}
    new_sd: dict[str, torch.Tensor] = {}
    n_nvfp4 = 0
    n_bf16 = 0
    n_smoothquant = 0
    n_sq_skip = 0

    print(f"Converting ({len(sd)} tensors)...")
    for k, v in tqdm(list(sd.items())):
        if any(name in k for name in blacklist):
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16 += 1
            continue

        if _is_non_diffusion_key(k):
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16 += 1
            continue

        if v.ndim == 2 and ".weight" in k:
            base_k_file = k.replace(".weight", "")
            base_k_meta = _meta_base_key(base_k_file)
            v_tensor = v.to(device=device, dtype=torch.bfloat16)

            if fp8_layers and any(name in k for name in fp8_layers):
                # Reserved for Kitchen FP8 path; Krea2 profile leaves this empty.
                import comfy_kitchen as ck

                weight_scale = (
                    (v_tensor.abs().max() / 448.0).clamp(min=1e-12).float()
                )
                weight_quantized = ck.quantize_per_tensor_fp8(v_tensor, weight_scale)
                new_sd[k] = weight_quantized.cpu()
                new_sd[f"{base_k_file}.weight_scale"] = weight_scale.to(
                    torch.bfloat16
                ).cpu()
                quant_map["layers"][base_k_meta] = {"format": "float8_e4m3fn"}
                if device == "cuda":
                    del v_tensor
                continue

            # --- SmoothQuant pre-scaling (optional, fp32 precision) ---
            sq_applied = False
            if use_smoothquant:
                # Look up per-channel act stats for this layer
                module_name = None
                ck_val = ck_map.get(k)
                if ck_val and ck_val.endswith(".weight"):
                    module_name = ck_val[:-len(".weight")]
                act_sq = act_sq_dict.get(module_name) if module_name else None

                if act_sq is not None and act_sq.shape[0] == v_tensor.shape[1]:
                    # Compute and apply SmoothQuant in float32 to avoid
                    # bfloat16 rounding error in W' = W * s.
                    v_f32 = v_tensor.to(dtype=torch.float32)
                    s = _compute_smoothquant_scale(
                        act_sq.cpu(), v_f32.cpu(), alpha=smoothquant_alpha
                    )
                    v_f32 = _apply_smoothquant(v_f32, s.to(device=v_f32.device))
                    # Cast to bfloat16 only at the last moment (right before NVFP4 quantize)
                    v_tensor = v_f32.to(dtype=torch.bfloat16)
                    del v_f32
                    # Save scale tensor for runtime (x' = x / s)
                    new_sd[f"{base_k_file}.smoothquant_scale"] = s.cpu()
                    sq_applied = True
                    n_smoothquant += 1
                else:
                    n_sq_skip += 1

            try:
                qdata, params = TensorCoreNVFP4Layout.quantize(v_tensor)
                tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)
                for suffix, tensor in tensors.items():
                    new_sd[f"{base_k_file}.weight{suffix}"] = tensor.cpu()

                layer_meta = {"format": "nvfp4"}
                if sq_applied:
                    layer_meta["smoothquant"] = True
                    layer_meta["smoothquant_alpha"] = smoothquant_alpha
                quant_map["layers"][base_k_meta] = layer_meta
                n_nvfp4 += 1
            except Exception:
                new_sd[k] = v.to(dtype=torch.bfloat16)
                n_bf16 += 1

            if device == "cuda":
                del v_tensor
        else:
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16 += 1

    final_metadata = OrderedDict()
    final_metadata["_quantization_metadata"] = json.dumps(quant_map)
    final_metadata["converted_by"] = "ComfyUI Kitchen NVFP4 Converter (Krea2-only)"
    final_metadata["converter_url"] = (
        "https://github.com/tritant/ComfyUI_Kitchen_nvfp4_Converter"
    )
    final_metadata["hswq_model"] = "krea2"
    if use_smoothquant:
        final_metadata["hswq_smoothquant"] = "1"
        final_metadata["hswq_smoothquant_alpha"] = str(smoothquant_alpha)

    print(f"Saving | Type: {_MODEL_TYPE} | Path: {output_path}")
    save_file(new_sd, output_path, metadata=final_metadata)
    total_bytes = os.path.getsize(output_path)
    print(f"Done. Size: {round(total_bytes / (1024**3), 2)} GiB")
    print(f"NVFP4 layers in metadata: {len(quant_map['layers'])}")
    print(f"  counted nvfp4 packs={n_nvfp4} | bf16 keep tensors={n_bf16}")
    if use_smoothquant:
        print(
            f"  SmoothQuant: applied={n_smoothquant}  skipped(no act stats)={n_sq_skip}  "
            f"alpha={smoothquant_alpha}"
        )

    del sd
    del new_sd
    del quant_map
    _release_vram("after native Krea2 NVFP4 convert save")


def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _release_vram(label: str = "post-convert") -> None:
    print(f"[*] Releasing VRAM ({label})...")
    gc.collect()
    if not torch.cuda.is_available():
        print(f"[*] VRAM clear ({label}): CUDA not available")
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        alloc_mib = torch.cuda.memory_allocated() / (1024 ** 2)
        reserv_mib = torch.cuda.memory_reserved() / (1024 ** 2)
        print(
            f"[*] VRAM clear ({label}): "
            f"allocated={alloc_mib:.1f} MiB reserved={reserv_mib:.1f} MiB"
        )
    except Exception:
        print(f"[*] VRAM clear ({label}): done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Krea2-only plain NVFP4 convert (Kitchen pack, no ConvRot / "
            "optional SmoothQuant). Refuses non-Krea2 checkpoints."
        )
    )
    parser.add_argument(
        "--model",
        "--input",
        dest="model",
        type=str,
        required=True,
        help="Path to Krea2 BF16/FP16 .safetensors",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output .safetensors"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Quantize device",
    )
    # --- SmoothQuant options ---
    parser.add_argument(
        "--calib_file",
        type=str,
        default=None,
        help="Calibration prompts file (one prompt per line). "
             "Enables SmoothQuant when combined with --clip_path.",
    )
    parser.add_argument(
        "--clip_path",
        type=str,
        default=None,
        help="Path to Krea2 CLIP checkpoint for calibration encoding. "
             "Enables SmoothQuant when combined with --calib_file.",
    )
    parser.add_argument(
        "--comfy_path",
        type=str,
        default=None,
        help="Path to ComfyUI root (for model loading during calibration). "
             "Auto-detected if omitted.",
    )
    parser.add_argument(
        "--smoothquant_alpha",
        type=float,
        default=_SMOOTHQUANT_ALPHA_DEFAULT,
        help="SmoothQuant migration strength (default 0.5). "
             "0.0 = no migration, 1.0 = full to weight side.",
    )
    parser.add_argument(
        "--num_calib_samples",
        type=int,
        default=32,
        help="Number of calibration samples (prompts) to use.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="Number of denoise steps per calibration sample.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    # SmoothQuant requires both calib_file and clip_path
    if bool(args.calib_file) != bool(args.clip_path):
        print("Error: --calib_file and --clip_path must be both provided or both omitted.")
        sys.exit(1)

    if args.calib_file and not os.path.exists(args.calib_file):
        print(f"Error: Calibration file not found: {args.calib_file}")
        sys.exit(1)

    if args.clip_path and not os.path.exists(args.clip_path):
        print(f"Error: CLIP checkpoint not found: {args.clip_path}")
        sys.exit(1)

    convert_to_nvfp4(
        args.model,
        args.output,
        device=str(args.device),
        calib_file=args.calib_file,
        clip_path=args.clip_path,
        comfy_path=args.comfy_path,
        smoothquant_alpha=args.smoothquant_alpha,
        num_calib_samples=args.num_calib_samples,
        num_inference_steps=args.num_inference_steps,
    )

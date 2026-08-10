"""Krea2 NVFP4 + analysis ConvRot INT8 protect (int8protect variant).

NEW FILE based on native_convert_nvfp4_krea2.py /
hswq_convert_nvfp4_zi_int8protect.py (do not edit the base converters).

Protect path (30 keys = prior20 + analyze severity +10):
  ConvRot rotate (W @ H^T) → row-wise INT8 + weight_scale + int8_tensorwise stamp
  in _quantization_metadata (same as ZIT int8protect).

Remaining Linear 2D: NVFP4 (+ FULL ConvRot by default).
Krea2 Kitchen blacklist: bfloat16 (unchanged).

Key source:
  test/_moodyKrea2Mix_v40BF16_nvfp4_int8protect30_final_keys.json
  (prior20 + build_nvfp4_analyze_character_table severity; NOT abs_max fill)

Example:
  python native_convert_nvfp4_krea2_int8protect.py --model ... --output ...
"""
from __future__ import annotations

import argparse
import gc
import json
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

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from native_convert_int8 import (  # noqa: E402
    build_hadamard,
    convrot_group_size_for_features,
    quantize_int8_rowwise,
    quantize_int8_tensorwise,
    rotate_weight,
)

_MODEL_TYPE = "Krea2"
# Match hswq_convert_nvfp4_zi_int8protect (128 is not a power of 4;
# preferred=128 never selects a valid Hadamard group via //=4 walk).
_DEFAULT_GROUPSIZE = 256

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

# Analysis ConvRot INT8 protect (moodyKrea2Mix_v40BF16).
# Prior20 + analyze severity +10 → 30 keys.
_INT8_PROTECT_KEYS_JSON = os.path.join(
    _REPO_ROOT,
    "test",
    "_moodyKrea2Mix_v40BF16_nvfp4_int8protect30_final_keys.json",
)


def _load_int8_protect_keyset(path: str = _INT8_PROTECT_KEYS_JSON) -> frozenset[str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    keys = data.get("int8_protect_keys")
    if not isinstance(keys, list) or not keys:
        raise ValueError(f"int8_protect_keys missing/empty in {path}")
    return frozenset(str(k) for k in keys)


_INT8_PROTECT_KEYSET: frozenset[str] = _load_int8_protect_keyset()
_INT8_PROTECT_SOURCE = os.path.basename(_INT8_PROTECT_KEYS_JSON)


def _is_int8_protect_key(key: str) -> bool:
    """True if key is in analysis INT8 protect set (exact or prefix variants)."""
    if key in _INT8_PROTECT_KEYSET:
        return True
    if key.startswith("diffusion_model."):
        alt = "model." + key
        if alt in _INT8_PROTECT_KEYSET:
            return True
    if not key.startswith("model.diffusion_model."):
        alt = "model.diffusion_model." + key
        if alt in _INT8_PROTECT_KEYSET:
            return True
    return False


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


def convert_to_nvfp4(
    input_path: str,
    output_path: str,
    device: str,
    enable_convrot: bool = True,
    group_size: int = _DEFAULT_GROUPSIZE,
):
    blacklist = list(_KREA2_BLACKLIST)
    rot_tag = "FULL ConvRot NVFP4" if enable_convrot else "plain NVFP4"
    print(
        f"Mode {_MODEL_TYPE} | device={device} | {rot_tag} "
        f"+ ConvRot INT8 protect ({len(_INT8_PROTECT_KEYSET)} keys)"
    )
    print(
        f"  [INT8 protect] {len(_INT8_PROTECT_KEYSET)} analysis keys → "
        "ConvRot INT8 (rowwise)"
    )
    if enable_convrot:
        print(
            f"  [ConvRot] ON | preferred groupsize={int(group_size)} "
            f"(Linear 2D; skip rotate when in_features has no power-of-4 group)"
        )
    else:
        print("  [ConvRot] OFF | plain Kitchen NVFP4 packs only")

    sd = load_file(input_path)
    prefix = _find_krea2_key_prefix(sd)
    print(f"Detected Krea2 key prefix: {prefix!r}")

    quant_map = {"format_version": "1.0", "layers": {}}
    new_sd: dict[str, torch.Tensor] = {}
    n_nvfp4 = 0
    n_convrot = 0
    n_plain_nvfp4 = 0
    n_bf16 = 0
    n_int8_protect = 0
    n_int8_convrot = 0
    n_int8_plain = 0

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

        # Analysis ConvRot INT8 protect (before NVFP4) — 30 keys
        if _is_int8_protect_key(k) and v.ndim == 2 and ".weight" in k:
            base_k_file = k.replace(".weight", "")
            base_k_meta = _meta_base_key(base_k_file)
            w = v.float().cpu()
            used_gs = convrot_group_size_for_features(
                int(w.shape[1]), int(group_size)
            )
            if used_gs is not None:
                h_matrix = build_hadamard(
                    int(used_gs), device="cpu", dtype=torch.float32
                )
                w = rotate_weight(w, h_matrix, int(used_gs))
                q, scale = quantize_int8_rowwise(w)
                quant_config = {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": int(used_gs),
                }
                n_int8_convrot += 1
            else:
                q, scale = quantize_int8_tensorwise(w)
                quant_config = {"format": "int8_tensorwise"}
                n_int8_plain += 1
            new_sd[k] = q
            new_sd[f"{base_k_file}.weight_scale"] = scale
            quant_map["layers"][base_k_meta] = dict(quant_config)
            n_int8_protect += 1
            continue

        if v.ndim == 2 and ".weight" in k:
            base_k_file = k.replace(".weight", "")
            base_k_meta = _meta_base_key(base_k_file)
            v_tensor = v.to(device=device, dtype=torch.bfloat16)

            used_gs = None
            do_rotate = False
            w_for_q = v_tensor
            if enable_convrot:
                used_gs = convrot_group_size_for_features(
                    int(v_tensor.shape[1]), int(group_size)
                )
                if used_gs is not None:
                    h_matrix = build_hadamard(
                        int(used_gs), device="cpu", dtype=torch.float32
                    )
                    w_rot = rotate_weight(
                        v_tensor.float().cpu(), h_matrix, int(used_gs)
                    )
                    w_for_q = w_rot.to(device=device, dtype=torch.bfloat16)
                    do_rotate = True

            try:
                qdata, params = TensorCoreNVFP4Layout.quantize(w_for_q)
                tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)
                for suffix, tensor in tensors.items():
                    new_sd[f"{base_k_file}.weight{suffix}"] = tensor.cpu()
                if do_rotate and used_gs is not None:
                    quant_map["layers"][base_k_meta] = {
                        "format": "nvfp4",
                        "convrot": True,
                        "convrot_groupsize": int(used_gs),
                    }
                    n_convrot += 1
                else:
                    quant_map["layers"][base_k_meta] = {"format": "nvfp4"}
                    n_plain_nvfp4 += 1
                n_nvfp4 += 1
            except Exception:
                new_sd[k] = v.to(dtype=torch.bfloat16)
                n_bf16 += 1

            if device == "cuda":
                if do_rotate:
                    del w_for_q
                del v_tensor
        else:
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16 += 1

    final_metadata = OrderedDict()
    final_metadata["_quantization_metadata"] = json.dumps(quant_map)
    by_tag = (
        "ComfyUI Kitchen NVFP4 Converter (Krea2 ConvRot + INT8 protect)"
        if enable_convrot
        else "ComfyUI Kitchen NVFP4 Converter (Krea2 INT8 protect)"
    )
    final_metadata["converted_by"] = by_tag
    final_metadata["converter_url"] = (
        "https://github.com/tritant/ComfyUI_Kitchen_nvfp4_Converter"
    )
    final_metadata["hswq_model"] = "krea2"
    final_metadata["hswq_nvfp4_convrot"] = "1" if enable_convrot else "0"
    final_metadata["hswq_int8_protect"] = "1"
    final_metadata["hswq_int8_protect_n"] = str(n_int8_protect)
    final_metadata["hswq_int8_protect_convrot"] = str(n_int8_convrot)
    final_metadata["hswq_int8_protect_source"] = _INT8_PROTECT_SOURCE.replace(
        ".json", ""
    )

    print(f"Saving | Type: {_MODEL_TYPE} | Path: {output_path}")
    save_file(new_sd, output_path, metadata=final_metadata)
    total_bytes = os.path.getsize(output_path)
    print(f"Done. Size: {round(total_bytes / (1024**3), 2)} GiB")
    print(f"NVFP4+INT8 layers in metadata: {len(quant_map['layers'])}")
    print(
        f"  counted nvfp4 packs={n_nvfp4} | bf16 keep tensors={n_bf16} | "
        f"int8 protect={n_int8_protect} "
        f"(convrot={n_int8_convrot}, plain={n_int8_plain})"
    )
    print(f"FULL ConvRot enabled (NVFP4 path): {enable_convrot}")
    if enable_convrot:
        print(
            f"  ConvRot NVFP4 Linear: {n_convrot}, "
            f"plain NVFP4 (no group): {n_plain_nvfp4}"
        )

    del sd
    del new_sd
    del quant_map
    _release_vram("after native Krea2 NVFP4 int8protect convert save")


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
            "Krea2 NVFP4 + analysis ConvRot INT8 protect (int8protect). "
            "Based on native_convert_nvfp4_krea2.py; 30 ranked Linear weights as "
            "ConvRot INT8; rest NVFP4. FULL ConvRot ON by default for NVFP4."
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
    parser.add_argument(
        "--nvfp4-convrot",
        dest="enable_convrot",
        action="store_true",
        default=True,
        help="FULL ConvRot on NVFP4 Linear (default ON)",
    )
    parser.add_argument(
        "--no-nvfp4-convrot",
        dest="enable_convrot",
        action="store_false",
        help="Disable ConvRot on NVFP4 path (plain Kitchen NVFP4)",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=_DEFAULT_GROUPSIZE,
        help=f"Preferred ConvRot Hadamard group size (default {_DEFAULT_GROUPSIZE})",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    convert_to_nvfp4(
        args.model,
        args.output,
        device=str(args.device),
        enable_convrot=bool(args.enable_convrot),
        group_size=int(args.groupsize),
    )

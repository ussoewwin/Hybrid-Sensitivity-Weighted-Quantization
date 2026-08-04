"""darkBeast (Krea2/SingleStreamDiT family) plain NVFP4 + ConvRot INT8 protect.

New script for darkBeast30BF16INT8_darkBeast330.safetensors.
Architecture follows Krea2 (see native_convert_nvfp4_krea2_int8protect_plainnvfp4.py):
  - prefix: model.diffusion_model. (or diffusion_model. / root)
  - blocks.{0..31}.{attn.wq,wk,wv,wo,gate, mlp.gate,up,down}.weight
  - txtfusion.{layerwise_blocks,refiner_blocks}.*
  - txtfusion.projector, first, last, tmlp, tproj, txtmlp (BF16 keep)

HSWQ path:
  - INT8 protect: 60 hardcoded keys → ConvRot rotate → rowwise INT8
    (same ConvRot INT8 path as ZI v1.0 / Krea2 plainnvfp4)
  - NVFP4 path: plain Kitchen packs (NO ConvRot on NVFP4)
  - BF16 keep: structural keys (mod/norm/projector/bias/non-diffusion)

60-key selection (HSWQ principle: output projection + FFN down projection first):
  - 32x attn.wo (attention output projection; all 32 blocks)
  - 28x mlp.down (FFN down projection; blocks 2..29, outermost 4 excluded)
  Total = 60.

Owner-allowed hardcode (no auto-ranking, no measurement dependency).
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

_MODEL_TYPE = "darkBeast"
_DEFAULT_GROUPSIZE = 256

# Krea2/SingleStreamDiT structural keep-BF16 markers.
_DARKBEAST_BLACKLIST: list[str] = [
    "first",
    "last",
    "mod.",
    "norm",
    "projector",
    "tmlp",
    "tproj",
    "txtmlp",
    "bias",
    "vae.",
    "text_encoders",
]

_INT8_PROTECT_SOURCE = (
    "darkBeast30BF16INT8_darkBeast330_hswq_int8protect60_hardcoded_wo_all_mlpdown_2to29"
)

# 60 hardcoded protect keys (HSWQ output-projection + FFN-down principle).
# 32x attn.wo (all blocks) + 28x mlp.down (blocks 2..29; outermost 4 excluded).
# Keys stored WITHOUT the "model." / "model.diffusion_model." prefix variability —
# _is_int8_protect_key handles prefix normalization at match time.
_BASE_INT8_PROTECT_KEYS: tuple[str, ...] = (
    # --- 32x attn.wo (attention output projection; HSWQ highest priority) ---
    "blocks.0.attn.wo",
    "blocks.1.attn.wo",
    "blocks.2.attn.wo",
    "blocks.3.attn.wo",
    "blocks.4.attn.wo",
    "blocks.5.attn.wo",
    "blocks.6.attn.wo",
    "blocks.7.attn.wo",
    "blocks.8.attn.wo",
    "blocks.9.attn.wo",
    "blocks.10.attn.wo",
    "blocks.11.attn.wo",
    "blocks.12.attn.wo",
    "blocks.13.attn.wo",
    "blocks.14.attn.wo",
    "blocks.15.attn.wo",
    "blocks.16.attn.wo",
    "blocks.17.attn.wo",
    "blocks.18.attn.wo",
    "blocks.19.attn.wo",
    "blocks.20.attn.wo",
    "blocks.21.attn.wo",
    "blocks.22.attn.wo",
    "blocks.23.attn.wo",
    "blocks.24.attn.wo",
    "blocks.25.attn.wo",
    "blocks.26.attn.wo",
    "blocks.27.attn.wo",
    "blocks.28.attn.wo",
    "blocks.29.attn.wo",
    "blocks.30.attn.wo",
    "blocks.31.attn.wo",
    # --- 28x mlp.down (FFN down projection; blocks 2..29) ---
    "blocks.2.mlp.down",
    "blocks.3.mlp.down",
    "blocks.4.mlp.down",
    "blocks.5.mlp.down",
    "blocks.6.mlp.down",
    "blocks.7.mlp.down",
    "blocks.8.mlp.down",
    "blocks.9.mlp.down",
    "blocks.10.mlp.down",
    "blocks.11.mlp.down",
    "blocks.12.mlp.down",
    "blocks.13.mlp.down",
    "blocks.14.mlp.down",
    "blocks.15.mlp.down",
    "blocks.16.mlp.down",
    "blocks.17.mlp.down",
    "blocks.18.mlp.down",
    "blocks.19.mlp.down",
    "blocks.20.mlp.down",
    "blocks.21.mlp.down",
    "blocks.22.mlp.down",
    "blocks.23.mlp.down",
    "blocks.24.mlp.down",
    "blocks.25.mlp.down",
    "blocks.26.mlp.down",
    "blocks.27.mlp.down",
    "blocks.28.mlp.down",
    "blocks.29.mlp.down",
)
assert len(_BASE_INT8_PROTECT_KEYS) == 60, (
    f"hardcode count must be 60, got {len(_BASE_INT8_PROTECT_KEYS)}"
)
_INT8_PROTECT_KEYSET: frozenset[str] = frozenset(_BASE_INT8_PROTECT_KEYS)

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


def _normalize_key_for_match(key: str) -> str:
    """Strip the model. / model.diffusion_model. / diffusion_model. prefix
    so the same hardcoded base key matches regardless of file convention.
    """
    if key.startswith("model.diffusion_model."):
        return key[len("model.diffusion_model."):]
    if key.startswith("diffusion_model."):
        return key[len("diffusion_model."):]
    if key.startswith("model."):
        return key[len("model."):]
    return key


def _is_int8_protect_key(key: str) -> bool:
    """True if key matches one of the 60 hardcoded protect base keys.

    Match against the prefix-stripped form (e.g. blocks.5.attn.wo.weight
    and model.diffusion_model.blocks.5.attn.wo.weight both map to the same
    base).
    """
    if ".weight" not in key or not key.endswith(".weight"):
        return False
    base = _normalize_key_for_match(key)[: -len(".weight")]
    return base in _INT8_PROTECT_KEYSET


def _find_darkbeast_key_prefix(state_dict) -> str:
    """Krea2/SingleStreamDiT signature detection for darkBeast.

    Required: txtfusion.projector.weight AND blocks.0.attn.wq.weight under
    one of the standard prefixes.
    """
    for prefix in ("model.diffusion_model.", "diffusion_model.", ""):
        if f"{prefix}txtfusion.projector.weight" in state_dict:
            if f"{prefix}blocks.0.attn.wq.weight" not in state_dict:
                raise ValueError(
                    "darkBeast/Krea2 signature incomplete: "
                    f"txtfusion.projector present but "
                    f"{prefix}blocks.0.attn.wq.weight missing"
                )
            return prefix
    raise ValueError(
        "Not a darkBeast/Krea2 checkpoint: missing txtfusion.projector.weight "
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
    group_size: int = _DEFAULT_GROUPSIZE,
):
    blacklist = list(_DARKBEAST_BLACKLIST)
    print(
        f"Mode {_MODEL_TYPE} | device={device} | plain NVFP4 "
        f"+ ConvRot INT8 protect ({len(_INT8_PROTECT_KEYSET)} keys)"
    )
    print(
        f"  [INT8 protect] {len(_INT8_PROTECT_KEYSET)} hardcoded keys → "
        "ConvRot INT8 (rowwise)"
    )
    print("  [NVFP4] plain Kitchen packs only (no ConvRot on NVFP4 path)")
    print(
        f"  [INT8 ConvRot groupsize] preferred={int(group_size)} "
        "(skip rotate when in_features has no power-of-4 group)"
    )
    print(
        f"  [INT8 protect source] {_INT8_PROTECT_SOURCE}"
    )

    sd = load_file(input_path)
    prefix = _find_darkbeast_key_prefix(sd)
    print(f"Detected darkBeast/Krea2 key prefix: {prefix!r}")

    # Structural audit
    n_blocks = 0
    for k in sd:
        if ".attn.wq.weight" in k and ".blocks." not in k:
            continue
        if ".attn.wq.weight" in k:
            n_blocks += 1
    n_blocks = max(n_blocks, 1)
    # Count once precisely using prefix-aware lookup
    n_blocks = 0
    for blk in range(0, 256):
        if f"{prefix}blocks.{blk}.attn.wq.weight" in sd:
            n_blocks += 1
        else:
            break
    has_txt_lw = any("txtfusion.layerwise_blocks" in k for k in sd)
    has_txt_rf = any("txtfusion.refiner_blocks" in k for k in sd)
    print(
        f"Structure: prefix={prefix!r} attn blocks={n_blocks} "
        f"txtfusion.layerwise={has_txt_lw} txtfusion.refiner={has_txt_rf}"
    )

    # Pre-flight: verify each protect key actually exists in the file
    # (under one of the accepted prefix variants) — fail loudly before the
    # multi-hour quantize loop instead of silently producing a partial-protect
    # artifact.
    missing_protect: list[str] = []
    for base in _BASE_INT8_PROTECT_KEYS:
        full = f"{prefix}{base}.weight"
        if full not in sd:
            missing_protect.append(full)
    if missing_protect:
        preview = " | ".join(missing_protect[:5])
        raise ValueError(
            f"Hardcoded protect keys missing in checkpoint ({len(missing_protect)}): "
            f"{preview}{' ...' if len(missing_protect) > 5 else ''}"
        )

    quant_map: dict[str, dict] = {"format_version": "1.0", "layers": {}}
    new_sd: dict[str, torch.Tensor] = {}
    n_nvfp4 = 0
    n_plain_nvfp4 = 0
    n_bf16 = 0
    n_int8_protect = 0
    n_int8_convrot = 0
    n_int8_plain = 0

    # Track protect hits to detect silent misses inside the main loop
    protect_hits: set[str] = set()

    print(f"Converting ({len(sd)} tensors)...")
    for k, v in tqdm(list(sd.items())):
        # 1) Structural BF16 keep
        if any(name in k for name in blacklist):
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16 += 1
            continue

        # 2) Non-diffusion heads (text encoders / VAE / etc.)
        if _is_non_diffusion_key(k):
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16 += 1
            continue

        # 3) ConvRot INT8 protect (60 hardcoded keys) — applied before NVFP4.
        if _is_int8_protect_key(k) and v.ndim == 2 and ".weight" in k:
            base_k_file = k.replace(".weight", "")
            base_k_meta = _meta_base_key(base_k_file)
            protect_hits.add(_normalize_key_for_match(k)[: -len(".weight")])
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

        # 4) Plain NVFP4 for remaining Linear 2D weights
        if v.ndim == 2 and ".weight" in k:
            base_k_file = k.replace(".weight", "")
            base_k_meta = _meta_base_key(base_k_file)
            v_tensor = v.to(device=device, dtype=torch.bfloat16)

            try:
                qdata, params = TensorCoreNVFP4Layout.quantize(v_tensor)
                tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)
                for suffix, tensor in tensors.items():
                    new_sd[f"{base_k_file}.weight{suffix}"] = tensor.cpu()
                quant_map["layers"][base_k_meta] = {"format": "nvfp4"}
                n_plain_nvfp4 += 1
                n_nvfp4 += 1
            except Exception:
                new_sd[k] = v.to(dtype=torch.bfloat16)
                n_bf16 += 1

            if device == "cuda":
                del v_tensor
        else:
            # 1D / non-weight tensors — keep BF16
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16 += 1

    # Sanity: every protect key was actually hit
    missed = _INT8_PROTECT_KEYSET - protect_hits
    if missed:
        raise RuntimeError(
            f"BUG: {len(missed)} protect keys declared but never matched in loop: "
            f"{sorted(missed)[:3]}..."
        )

    final_metadata = OrderedDict()
    final_metadata["_quantization_metadata"] = json.dumps(quant_map)
    final_metadata["converted_by"] = (
        "ComfyUI Kitchen NVFP4 Converter "
        "(darkBeast plain NVFP4 + ConvRot INT8 protect 60 hardcoded)"
    )
    final_metadata["converter_url"] = (
        "https://github.com/tritant/ComfyUI_Kitchen_nvfp4_Converter"
    )
    final_metadata["hswq_model"] = "darkBeast"
    final_metadata["hswq_nvfp4_convrot"] = "0"
    final_metadata["hswq_int8_protect"] = "1"
    final_metadata["hswq_int8_protect_n"] = str(n_int8_protect)
    final_metadata["hswq_int8_protect_convrot"] = str(n_int8_convrot)
    final_metadata["hswq_int8_protect_source"] = _INT8_PROTECT_SOURCE

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
    print("FULL ConvRot enabled (NVFP4 path): False")
    print(
        f"  plain NVFP4 Linear: {n_plain_nvfp4} "
        f"(ConvRot NVFP4: 0)"
    )

    del sd
    del new_sd
    del quant_map
    _release_vram("after darkBeast plain-NVFP4 int8protect60 convert save")


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
        alloc_mib = torch.cuda.memory_allocated() / (1024**2)
        reserv_mib = torch.cuda.memory_reserved() / (1024**2)
        print(
            f"[*] VRAM clear ({label}): "
            f"allocated={alloc_mib:.1f} MiB reserved={reserv_mib:.1f} MiB"
        )
    except Exception:
        print(f"[*] VRAM clear ({label}): done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "darkBeast (Krea2/SingleStreamDiT) plain NVFP4 + ConvRot INT8 protect. "
            "60 hardcoded Linear (32x attn.wo + 28x mlp.down blocks 2..29) as "
            "ConvRot INT8; remaining Linear plain NVFP4 (no rotate)."
        )
    )
    parser.add_argument(
        "--model",
        "--input",
        dest="model",
        type=str,
        required=True,
        help="Path to darkBeast BF16/FP16 .safetensors",
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
        "--groupsize",
        type=int,
        default=_DEFAULT_GROUPSIZE,
        help=(
            f"Preferred ConvRot Hadamard group size for INT8 protect only "
            f"(default {_DEFAULT_GROUPSIZE}; NVFP4 path never rotates)"
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    convert_to_nvfp4(
        args.model,
        args.output,
        device=str(args.device),
        group_size=int(args.groupsize),
    )

#!/usr/bin/env python3
"""
SeedVR2-7B-only HSWQ dual converter (production entry).

Base Linear pack: plain NVFP4 (UNROTATED), same pack_nvfp4 / can_pack_nvfp4 /
_encode_comfy_quant helpers as hswq_convert_nvfp4_1.2.py.

INT8-shelter Linear pack: ConvRot INT8 (pack_convrot_int8), not plain NVFP4.

Owner policy (do not invert):
  - SeedVR2-7B DiT has zero Conv2d (architecture fact).
  - "no Conv2d" does NOT cancel INT8 shelter / ConvRot INT8 protection.
  - Shelter module list: test/seedvr2_ema_7b_int8_shelter_keys.json
    (Linear 2D abs_max > 1.0, RoPE/freqs excluded).

Key layout: SeedVR2 DiT (e.g. blocks.*.attn.*.weight) — no
model.diffusion_model prefix (unlike UNet-style 1.2 convert).

Do not run convert-to-disk unless the same message explicitly orders a run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Dict, Optional, Set, Tuple

import torch
from safetensors.torch import load_file, save_file


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _load_module(name: str, path: str):
    # Register before exec_module so @dataclass (and similar) can resolve cls.__module__.
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


def _load_nvfp4_12():
    return _load_module(
        "hswq_convert_nvfp4_1_2",
        os.path.join(_repo_root(), "hswq_convert_nvfp4_1.2.py"),
    )


def _load_native_int8():
    return _load_module(
        "native_convert_int8",
        os.path.join(_repo_root(), "native_convert_int8.py"),
    )


def _is_rope_or_freqs_key(key: str) -> bool:
    kl = key.lower()
    return ("rope" in kl) or ("freqs" in kl)


def _module_name_from_weight_key(key: str) -> Optional[str]:
    if not key.endswith(".weight"):
        return None
    return key[: -len(".weight")]


_DEFAULT_SHELTER = os.path.join(
    _repo_root(), "test", "seedvr2_ema_7b_int8_shelter_keys.json"
)
# Match native_convert_int8 / hswq_convert_nvfp4_1.2 default ConvRot group size.
_DEFAULT_GROUPSIZE = 256


def load_shelter_modules(path: str) -> Set[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mods = data.get("modules")
    if not isinstance(mods, list) or not mods:
        raise ValueError(f"shelter json missing modules list: {path}")
    return {str(x) for x in mods}


# Smoke / callers may use private-style names
_load_shelter_modules = load_shelter_modules


def _strip_stale_quant_sidecar(
    new_state_dict: Dict[str, torch.Tensor],
    module_key: str,
) -> None:
    for suffix in (
        ".weight_scale",
        ".weight_scale_2",
        ".input_scale",
        ".comfy_quant",
        ".hadamard",
    ):
        new_state_dict.pop(f"{module_key}{suffix}", None)


def pack_convrot_int8(
    w: torch.Tensor,
    *,
    device: str = "cpu",
    group_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[int], str]:
    """
    ConvRot INT8 pack for SeedVR2 INT8-shelter Linear (2D).

    Returns:
        q, scale, hadamard_or_None, used_gs_or_None, mode ("convrot"|"plain")
    """
    if w.ndim != 2:
        raise ValueError(f"pack_convrot_int8 expects 2D, got ndim={w.ndim}")
    nv = _load_nvfp4_12()
    ni = _load_native_int8()
    build_hadamard = ni.build_hadamard
    convrot_group_size_for_features = ni.convrot_group_size_for_features
    rotate_weight = ni.rotate_weight
    pack_channelwise_int8 = nv.pack_channelwise_int8

    w_fp = w.detach().float().to(device)
    in_f = int(w_fp.shape[1])
    preferred = int(group_size) if group_size is not None else _DEFAULT_GROUPSIZE
    used_gs = convrot_group_size_for_features(in_f, preferred)
    h_matrix: Optional[torch.Tensor] = None
    mode = "plain"
    if used_gs is not None:
        h_matrix = build_hadamard(int(used_gs), device=device, dtype=torch.float32)
        w_fp = rotate_weight(w_fp, h_matrix, int(used_gs))
        mode = "convrot"
    q, scale = pack_channelwise_int8(w_fp)
    return q, scale, h_matrix, (int(used_gs) if used_gs is not None else None), mode


def convert_seedvr2_nvfp4_int8(
    model_path: str,
    output_path: str,
    shelter_modules: Set[str],
    *,
    device: str = "cpu",
    enable_nvfp4_convrot: bool = False,
    group_size: Optional[int] = None,
) -> Dict[str, int]:
    """
    Convert SeedVR2 FP16 safetensors to dual NVFP4 + ConvRot INT8.

    - Linear in shelter_modules -> ConvRot INT8 (pack_convrot_int8)
    - Other packable 2D Linear -> plain NVFP4 (UNROTATED) unless
      enable_nvfp4_convrot (non-shelter only; shelter stays INT8)
    - RoPE/freqs and 1D tensors stay FP16 (copy)
    """
    nv = _load_nvfp4_12()
    pack_nvfp4 = nv.pack_nvfp4
    can_pack_nvfp4 = nv.can_pack_nvfp4
    encode_comfy_quant = nv._encode_comfy_quant

    print(f"Loading: {model_path}")
    state = load_file(model_path, device=device)
    new_state: Dict[str, torch.Tensor] = {}
    quant_meta_layers: Dict[str, dict] = {}

    stats = {
        "copied": 0,
        "plain_nvfp4": 0,
        "convrot_nvfp4": 0,
        "convrot_int8_shelter": 0,
        "plain_int8_shelter_fallback": 0,
        "skipped_unpackable": 0,
        "shelter_hits": 0,
        "shelter_miss_not_2d": 0,
    }

    weight_keys = [k for k in state.keys() if k.endswith(".weight")]
    print(f"weight keys: {len(weight_keys)}")
    print(f"INT8 shelter modules: {len(shelter_modules)}")

    for key, tensor in state.items():
        if not key.endswith(".weight"):
            # Keep non-weight tensors; strip stale quant sidecars when we
            # rewrite the matching .weight later.
            if key.endswith(
                (
                    ".weight_scale",
                    ".weight_scale_2",
                    ".input_scale",
                    ".comfy_quant",
                    ".hadamard",
                )
            ):
                continue
            new_state[key] = tensor
            stats["copied"] += 1
            continue

        module_name = _module_name_from_weight_key(key)
        assert module_name is not None
        module_key = module_name

        if _is_rope_or_freqs_key(key):
            new_state[key] = tensor
            stats["copied"] += 1
            continue

        if tensor.ndim != 2:
            new_state[key] = tensor
            stats["copied"] += 1
            if module_name in shelter_modules:
                stats["shelter_miss_not_2d"] += 1
            continue

        is_shelter = module_name in shelter_modules
        if is_shelter:
            stats["shelter_hits"] += 1
            # Hadamard is stamped in comfy_quant only (no .hadamard tensor).
            q, scale, _h_unused, used_gs, mode = pack_convrot_int8(
                tensor, device=device, group_size=group_size
            )
            _strip_stale_quant_sidecar(new_state, module_key)
            new_state[key] = q
            new_state[f"{module_key}.weight_scale"] = scale
            if mode == "convrot" and used_gs is not None:
                quant_config = {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": int(used_gs),
                }
                stats["convrot_int8_shelter"] += 1
            else:
                quant_config = {"format": "int8_tensorwise"}
                stats["plain_int8_shelter_fallback"] += 1
            new_state[f"{module_key}.comfy_quant"] = encode_comfy_quant(quant_config)
            quant_meta_layers[module_key] = dict(quant_config)
            continue

        # Non-shelter Linear: plain NVFP4 (default) or optional NVFP4+ConvRot
        if not can_pack_nvfp4(tensor):
            new_state[key] = tensor
            stats["skipped_unpackable"] += 1
            continue

        w_fp = tensor.float()
        used_gs = None
        do_nvfp4_rotate = False
        if enable_nvfp4_convrot:
            ni = _load_native_int8()
            preferred = (
                int(group_size) if group_size is not None else _DEFAULT_GROUPSIZE
            )
            used_gs = ni.convrot_group_size_for_features(
                int(w_fp.shape[1]), preferred
            )
            if used_gs is not None:
                h_matrix = ni.build_hadamard(
                    int(used_gs), device="cpu", dtype=torch.float32
                )
                w_fp = ni.rotate_weight(w_fp, h_matrix, int(used_gs))
                do_nvfp4_rotate = True

        q, params = pack_nvfp4(w_fp)
        _strip_stale_quant_sidecar(new_state, module_key)
        new_state[key] = q
        new_state[f"{module_key}.weight_scale"] = params.block_scale
        new_state[f"{module_key}.weight_scale_2"] = params.scale.to(
            dtype=torch.float32
        ).reshape(())
        if do_nvfp4_rotate and used_gs is not None:
            quant_config = {
                "format": "nvfp4",
                "convrot": True,
                "convrot_groupsize": int(used_gs),
            }
            stats["convrot_nvfp4"] += 1
        else:
            quant_config = {"format": "nvfp4"}
            stats["plain_nvfp4"] += 1
        new_state[f"{module_key}.comfy_quant"] = encode_comfy_quant(quant_config)
        quant_meta_layers[module_key] = dict(quant_config)

    # Preserve any non-weight keys not yet copied (biases, etc.)
    for key, tensor in state.items():
        if key in new_state:
            continue
        if key.endswith(
            (
                ".weight_scale",
                ".weight_scale_2",
                ".input_scale",
                ".comfy_quant",
                ".hadamard",
            )
        ):
            # Drop stale sidecars unless we wrote fresh ones above
            continue
        new_state[key] = tensor
        stats["copied"] += 1

    print("Conversion stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"quant_meta layers: {len(quant_meta_layers)}")
    print(f"Saving: {output_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    save_file(new_state, output_path)
    print("Done.")
    return stats


def main() -> int:
    root = _repo_root()
    default_shelter = os.path.join(
        root, "test", "seedvr2_ema_7b_int8_shelter_keys.json"
    )
    ap = argparse.ArgumentParser(
        description=(
            "SeedVR2-7B dual convert: plain NVFP4 base + ConvRot INT8 shelter"
        )
    )
    ap.add_argument("--model", required=True, help="FP16 SeedVR2 safetensors path")
    ap.add_argument("--output", required=True, help="Output safetensors path")
    ap.add_argument(
        "--shelter-json",
        default=default_shelter,
        help="JSON with modules[] for ConvRot INT8 shelter",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--nvfp4-convrot",
        action="store_true",
        help="Also apply ConvRot to non-shelter NVFP4 Linear (default: off)",
    )
    ap.add_argument("--group-size", type=int, default=None)
    args = ap.parse_args()

    shelter = load_shelter_modules(args.shelter_json)
    convert_seedvr2_nvfp4_int8(
        args.model,
        args.output,
        shelter,
        device=args.device,
        enable_nvfp4_convrot=bool(args.nvfp4_convrot),
        group_size=args.group_size,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

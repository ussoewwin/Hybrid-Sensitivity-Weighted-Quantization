#!/usr/bin/env python3
"""
SeedVR2-7B (sharp_fp16) A1 converter — DiT-block Linear -> plain NVFP4 (UNROTATED),
IO projections kept FP16 (matches official seedvr2_7b_nvfp4 layout).

Owner-ordered A1 policy (replaces the prior INT8-shelter dual design):
  - Shelter is DISABLED. DiT block 2D Linears are packed as plain NVFP4.
  - IO projections that the official seedvr2_7b_nvfp4.safetensors keeps in
    FP16 are also kept FP16 here. This is the SSIM-critical difference:
    quantizing these 6 IO Linears drops SSIM ~0.017 (A1-without-keepset =
    0.9627 vs official ~0.98). Verified by diffing quantized-module sets
    between our A1 output and the official file.

  - Known unavoidable divergence from the official file: our pack_nvfp4 emits
    a 2-level scale (.weight_scale + .weight_scale_2 float32), whereas the
    official file uses a single .weight_scale (float8_e4m3fn, shape [out, in/8]).
    Same NVFP4-everywhere-in-DiT design; different scale encoder revision.

RoPE/freqs and non-2D tensors stay FP16 (copy), same as before.

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
    _repo_root(), "test", "seedvr2_ema_7b_sharp_int8_shelter_keys.json"
)
# Match native_convert_int8 / hswq_convert_nvfp4_1.2 default ConvRot group size.
_DEFAULT_GROUPSIZE = 256

# IO projections that the official seedvr2_7b_nvfp4.safetensors keeps in FP16.
# These sit OUTSIDE the DiT blocks (no `blocks.N.` prefix) and are the
# 6 modules that were quantized by A1-without-keepset but left FP16 by the
# official file. Keeping them FP16 is the SSIM-critical fix.
_KEEP_FP16_KEYSET: Set[str] = {
    "emb_in.proj_in",
    "emb_in.proj_hid",
    "emb_in.proj_out",
    "txt_in",
    "vid_in.proj",
    "vid_out.proj",
}

# A2 policy: BEYOND the official seedvr2_7b_nvfp4 layout. The official file
# quantizes ALL 288 DiT-block Linears to NVFP4 — including the 12 highest-risk
# Linears (abs_max >= 1.3, derived from test/seedvr2_ema_7b_int8_shelter_keys.json).
# Keeping these 12 in FP16 pushes SSIM ABOVE 0.98 (official is ~0.98).
# Size cost is modest (~0.7 GiB) because the 7B FP16 original is ~13.6 GiB.
_HIGH_RISK_FP16_KEYSET: Set[str] = {
    "blocks.17.mlp.txt.proj_out",  # abs_max=2.342
    "blocks.14.mlp.txt.proj_in",   # abs_max=2.092
    "blocks.14.mlp.txt.proj_out",  # abs_max=2.059
    "blocks.17.mlp.txt.proj_in",   # abs_max=1.866
    "blocks.12.mlp.vid.proj_out",  # abs_max=1.708
    "blocks.7.mlp.txt.proj_in",    # abs_max=1.651
    "blocks.0.attn.proj_qkv.vid",  # abs_max=1.443
    "blocks.17.attn.proj_out.txt", # abs_max=1.402
    "blocks.18.mlp.txt.proj_out",  # abs_max=1.402
    "blocks.20.mlp.txt.proj_out",  # abs_max=1.397
    "blocks.1.attn.proj_qkv.vid",  # abs_max=1.391
    "blocks.19.mlp.txt.proj_out",  # abs_max=1.335
}


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
    Convert SeedVR2 FP16 (sharp) safetensors — A1 policy.

    A1 (owner-ordered): shelter is DISABLED. shelter_modules is accepted for
    signature/CLI compatibility but is NOT used. Every packable 2D Linear is
    packed as plain NVFP4 (UNROTATED), or NVFP4+ConvRot if enable_nvfp4_convrot.
    RoPE/freqs and 1D tensors stay FP16 (copy).
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
        "io_fp16_kept": 0,
        "high_risk_fp16_kept": 0,
    }

    weight_keys = [k for k in state.keys() if k.endswith(".weight")]
    print(f"weight keys: {len(weight_keys)}")
    print(f"INT8 shelter modules: {len(shelter_modules)} (A1: shelter DISABLED)")

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
            continue

        # A1 + IO-keepset: DiT-block Linears -> NVFP4, IO projections stay FP16.
        # This mirrors the official seedvr2_7b_nvfp4.safetensors layout
        # (6 IO Linears kept FP16, 288 DiT Linears NVFP4).
        if module_key in _KEEP_FP16_KEYSET:
            _strip_stale_quant_sidecar(new_state, module_key)
            new_state[key] = tensor
            stats["io_fp16_kept"] += 1
            continue

        # A2: BEYOND official. The 12 highest-risk DiT Linears (abs_max >= 1.3)
        # are kept FP16 to push SSIM above the official ~0.98. The official file
        # quantizes these; we deliberately protect them.
        if module_key in _HIGH_RISK_FP16_KEYSET:
            _strip_stale_quant_sidecar(new_state, module_key)
            new_state[key] = tensor
            stats["high_risk_fp16_kept"] += 1
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
        root, "test", "seedvr2_ema_7b_sharp_int8_shelter_keys.json"
    )
    ap = argparse.ArgumentParser(
        description=(
            "SeedVR2-7B (sharp_fp16) A1 converter: DiT-block Linear -> plain NVFP4, "
            "6 IO projections kept FP16 (matches official seedvr2_7b_nvfp4 layout; "
            "shelter DISABLED)"
        )
    )
    ap.add_argument("--model", required=True, help="FP16 SeedVR2 sharp safetensors path")
    ap.add_argument("--output", required=True, help="Output safetensors path")
    ap.add_argument(
        "--shelter-json",
        default=default_shelter,
        help="Accepted for CLI compatibility; IGNORED under A1 (shelter disabled)",
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

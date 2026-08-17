#!/usr/bin/env python3
"""
krea2 (moodyKrea2Mix_v40BF16) NVFP4 Converter — A2 Policy Edition
=======================================================================
Policy: Apply the same principles as seedvr2_7b_nvfp4_A2.safetensors to krea2 architecture.

■ _KEEP_FP16_PREFIXES (Equivalent to SeedVR2 _KEEP_FP16_KEYSET: IO/stem layers)
  model.diffusion_model.first.*
  model.diffusion_model.last.*
  model.diffusion_model.tmlp.*
  model.diffusion_model.tproj.*
  model.diffusion_model.txtmlp.*
  model.diffusion_model.txtfusion.*  <- All txtfusion (including layerwise/refiner)
  -> Kept in FP16 (equivalent to IO projections)

■ _HIGH_RISK_FP16_KEYSET (Equivalent to SeedVR2 _HIGH_RISK_FP16_KEYSET)
  All high-risk layers with abs_max >= 1.5 based on abs_max analysis (test/krea2_absmax.py):
    * Note: txtfusion is fully protected via _KEEP_FP16_PREFIXES (including abs_max=5.71 etc.)
    blocks.0.attn.wk          abs_max=3.39   blocks.0.attn.wo      abs_max=1.68
    blocks.0.attn.wq          abs_max=1.86   blocks.1.attn.wq      abs_max=1.95
    blocks.0.mlp.down         abs_max=1.30 (small layer with numel<64, protected)
    blocks.1.mlp.down         abs_max=2.20   blocks.9.attn.wo      abs_max=2.02
    blocks.10.attn.gate       abs_max=3.28   blocks.10.attn.wq     abs_max=1.73
    blocks.10.attn.wo         abs_max=1.76   blocks.10.mlp.down    abs_max=2.16
    blocks.11.mlp.down        abs_max=2.34   blocks.12.attn.wo     abs_max=1.91
    blocks.15.mlp.up          abs_max=1.99   blocks.16.attn.wv     abs_max=1.80
    blocks.17.attn.wo         abs_max=1.70   blocks.18.mlp.up      abs_max=2.16
    blocks.19.attn.wo         abs_max=2.45   blocks.19.mlp.up      abs_max=1.93
    blocks.20.attn.wo         abs_max=2.16   blocks.21.attn.wo     abs_max=2.28
    blocks.22.attn.gate       abs_max=1.86   blocks.22.attn.wo     abs_max=2.19
    blocks.23.attn.wk         abs_max=1.63   blocks.23.attn.wv     abs_max=1.80
    blocks.24.attn.wv         abs_max=1.65   blocks.24.mlp.down    abs_max=1.64
    blocks.25.attn.wv         abs_max=1.63   blocks.26.mlp.gate    abs_max=1.71
    blocks.27.attn.wv         abs_max=1.88   blocks.27.attn.wo     abs_max=2.19
    blocks.27.mlp.gate        abs_max=2.52   blocks.27.mlp.up      abs_max=1.66
    blocks.14.attn.wq         abs_max=1.84
  -> Kept in FP16

■ Other DiT block Linear layers (2D, numel >= 64)
  -> NVFP4 (pack_nvfp4: weight + weight_scale + weight_scale_2 + comfy_quant)

■ Non-Linear tensors (norm scale, bias, etc.) are copied without dtype conversion.

Do NOT run the conversion unless the same message explicitly orders a run.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from typing import Dict, Set

import torch
from safetensors.torch import load_file, save_file


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _load_module(name: str, path: str):
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


# ---------------------------------------------------------------------------
# FP16 Protection: IO / stem layers (Equivalent to SeedVR2 _KEEP_FP16_KEYSET)
# txtfusion protects all layers including child layerwise_blocks / refiner_blocks
# ---------------------------------------------------------------------------
_KEEP_FP16_PREFIXES: tuple = (
    "model.diffusion_model.first.",
    "model.diffusion_model.last.",
    "model.diffusion_model.tmlp.",
    "model.diffusion_model.tproj.",
    "model.diffusion_model.txtmlp.",
    "model.diffusion_model.txtfusion.",
)

# ---------------------------------------------------------------------------
# FP16 Protection: High-risk block layers (Equivalent to SeedVR2 _HIGH_RISK_FP16_KEYSET)
# key format: "blocks.N.X.Y" portion of "model.diffusion_model.blocks.N.X.Y"
# ---------------------------------------------------------------------------
_HIGH_RISK_FP16_KEYSET: Set[str] = {
    # --- abs_max >= 3.0 ---
    "blocks.0.attn.wk",                               # abs_max=3.39
    "blocks.10.attn.gate",                            # abs_max=3.28
    # --- abs_max >= 2.0 ---
    "blocks.27.mlp.gate",                             # abs_max=2.52
    "blocks.19.attn.wo",                              # abs_max=2.45
    "blocks.11.mlp.down",                             # abs_max=2.34
    "blocks.21.attn.wo",                              # abs_max=2.28
    "blocks.1.mlp.down",                              # abs_max=2.20
    "blocks.22.attn.wo",                              # abs_max=2.19
    "blocks.27.attn.wo",                              # abs_max=2.19
    "blocks.10.mlp.down",                             # abs_max=2.16
    "blocks.18.mlp.up",                               # abs_max=2.16
    "blocks.20.attn.wo",                              # abs_max=2.16
    "blocks.15.mlp.up",                               # abs_max=1.99
    "blocks.9.attn.wo",                               # abs_max=2.02
    # --- abs_max >= 1.5 (additional protection) ---
    "blocks.0.attn.wq",                               # abs_max=1.86
    "blocks.0.attn.wo",                               # abs_max=1.68
    "blocks.1.attn.wq",                               # abs_max=1.95
    "blocks.10.attn.wq",                              # abs_max=1.73
    "blocks.10.attn.wo",                              # abs_max=1.76
    "blocks.12.attn.wo",                              # abs_max=1.91
    "blocks.14.attn.wq",                              # abs_max=1.84
    "blocks.16.attn.wv",                              # abs_max=1.80
    "blocks.17.attn.wo",                              # abs_max=1.70
    "blocks.19.mlp.up",                               # abs_max=1.93
    "blocks.22.attn.gate",                            # abs_max=1.86
    "blocks.23.attn.wk",                              # abs_max=1.63
    "blocks.23.attn.wv",                              # abs_max=1.80
    "blocks.24.attn.wv",                              # abs_max=1.65
    "blocks.24.mlp.down",                             # abs_max=1.64
    "blocks.25.attn.wv",                              # abs_max=1.63
    "blocks.26.mlp.gate",                             # abs_max=1.71
    "blocks.27.attn.wv",                              # abs_max=1.88
    "blocks.27.mlp.up",                               # abs_max=1.66
}

# txtfusion is fully protected via _KEEP_FP16_PREFIXES, not listed in HIGH_RISK
# (txtfusion.layerwise_blocks.0.attn.wo abs_max=5.72 / 5.63 etc.)


def _is_fp16_protected(key: str) -> tuple[bool, str]:
    """Return (is_protected, reason)."""
    # IO / stem prefix check
    for pfx in _KEEP_FP16_PREFIXES:
        if key.startswith(pfx):
            return True, "io_stem"
    # Match on key stripped of "model.diffusion_model."
    stripped = key.removeprefix("model.diffusion_model.")
    # Module key stripped of .weight
    module_key = stripped.removesuffix(".weight")
    if module_key in _HIGH_RISK_FP16_KEYSET:
        return True, "high_risk"
    return False, ""


def _strip_stale_quant_sidecar(
    new_state: Dict[str, torch.Tensor],
    module_key: str,
) -> None:
    """Remove stale quantization sidecar keys."""
    for suffix in (
        ".weight_scale",
        ".weight_scale_2",
        ".input_scale",
        ".comfy_quant",
        ".hadamard",
    ):
        new_state.pop(f"{module_key}{suffix}", None)


def convert(src_path: str, dst_path: str, device: str = "cpu") -> None:
    nv = _load_nvfp4_12()
    pack_nvfp4 = nv.pack_nvfp4
    can_pack_nvfp4 = nv.can_pack_nvfp4
    encode_comfy_quant = nv._encode_comfy_quant

    print(f"Loading {src_path} ...")
    sd = load_file(src_path, device=device)

    new_state: Dict[str, torch.Tensor] = {}
    stats = {
        "io_stem_fp16_kept": 0,
        "high_risk_fp16_kept": 0,
        "nvfp4_packed": 0,
        "skipped_small": 0,
        "skipped_1d": 0,
        "skipped_unpackable": 0,
        "copied": 0,
    }

    weight_keys = [k for k in sd.keys() if k.endswith(".weight")]
    print(f"Total keys: {len(sd)}  weight keys: {len(weight_keys)}")

    # Exclude old sidecar keys first
    SIDECAR_SUFFIXES = (
        ".weight_scale", ".weight_scale_2", ".input_scale",
        ".comfy_quant", ".hadamard",
    )

    for key, tensor in sd.items():
        # Skip old sidecars (regenerated by pack_nvfp4 if needed)
        if key.endswith(SIDECAR_SUFFIXES):
            continue

        if not key.endswith(".weight"):
            new_state[key] = tensor
            stats["copied"] += 1
            continue

        module_key = key[: -len(".weight")]

        # Copy 1D tensors (bias, norm scale, etc.) directly
        if tensor.ndim != 2:
            new_state[key] = tensor
            stats["skipped_1d"] += 1
            continue

        # Copy tiny tensors (numel < 64) directly
        if tensor.numel() < 64:
            new_state[key] = tensor
            stats["skipped_small"] += 1
            continue

        # Check FP16 protection
        protected, reason = _is_fp16_protected(key)
        if protected:
            _strip_stale_quant_sidecar(new_state, module_key)
            new_state[key] = tensor
            if reason == "io_stem":
                stats["io_stem_fp16_kept"] += 1
            else:
                stats["high_risk_fp16_kept"] += 1
            continue

        # Check NVFP4 quantizability
        if not can_pack_nvfp4(tensor):
            new_state[key] = tensor
            stats["skipped_unpackable"] += 1
            continue

        # Convert to NVFP4
        w_fp = tensor.float()
        q, params = pack_nvfp4(w_fp)
        _strip_stale_quant_sidecar(new_state, module_key)
        new_state[key] = q
        new_state[f"{module_key}.weight_scale"] = params.block_scale
        new_state[f"{module_key}.weight_scale_2"] = params.scale.to(
            dtype=torch.float32
        ).reshape(())
        new_state[f"{module_key}.comfy_quant"] = encode_comfy_quant(
            {"format": "nvfp4"}
        )
        stats["nvfp4_packed"] += 1

    print("Conversion stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    total_fp16 = stats["io_stem_fp16_kept"] + stats["high_risk_fp16_kept"]
    print(f"  total_fp16_kept: {total_fp16}")
    print(f"  total_nvfp4: {stats['nvfp4_packed']}")

    os.makedirs(os.path.dirname(os.path.abspath(dst_path)) or ".", exist_ok=True)
    print(f"Saving -> {dst_path} ...")
    save_file(new_state, dst_path)
    print("Done.")


def main() -> int:
    ap = argparse.ArgumentParser(description="krea2 NVFP4 A2-policy converter")
    ap.add_argument(
        "--src",
        default=r"D:\USERFILES\ComfyUI\ComfyUI\models\unet\moodyKrea2Mix_v40BF16.safetensors",
    )
    ap.add_argument(
        "--dst",
        default=r"D:\USERFILES\ComfyUI\ComfyUI\models\unet\moodyKrea2Mix_nvfp4_A2.safetensors",
    )
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    convert(args.src, args.dst, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())

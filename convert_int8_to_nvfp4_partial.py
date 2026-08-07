"""Convert selected ConvRot INT8 layers to ConvRot NVFP4 in a Krea2 checkpoint.

Reads an existing ConvRot INT8 safetensors, dequantizes selected layers,
re-quantizes them to NVFP4 (same ConvRot space), and writes a mixed
INT8+NVFP4 checkpoint. Non-selected layers stay INT8. BF16 keep layers
are untouched.

Strategy: layers stay in the same ConvRot-rotated space throughout.
  INT8 dequant (q*scale = W_rot) -> NVFP4 quant (W_rot) -> done.
  No double rotation needed.

Example:
  python convert_int8_to_nvfp4_partial.py \\
    --input  moodyCutieMixKrea2_v20_hswq_r32_1off_convrot_int8.safetensors \\
    --output moodyCutieMixKrea2_v20_hswq_r32_1off_mixed_nvfp4.safetensors \\
    --nvfp4_types mlp.up,mlp.gate,mlp.down

  # Or convert by ranking (lowest NVFP4 error first):
  python convert_int8_to_nvfp4_partial.py \\
    --input  ... --output ... \\
    --analysis nvfp4_conversion_analysis.json --nvfp4_n 84

  # Or convert all MLP layers:
  python convert_int8_to_nvfp4_partial.py \\
    --input  ... --output ... \\
    --all_mlp
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import OrderedDict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

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
    rotate_weight,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def _find_krea2_key_prefix(state_dict_keys) -> str:
    for prefix in ("model.diffusion_model.", "diffusion_model.", ""):
        if f"{prefix}txtfusion.projector.weight" in state_dict_keys:
            return prefix
    raise ValueError("Not a Krea2 checkpoint: missing txtfusion.projector.weight")


def _meta_base_key(key: str, prefix: str) -> str:
    if prefix and key.startswith(prefix):
        return key[len(prefix):]
    return key


def _encode_comfy_quant(config: dict) -> torch.Tensor:
    raw = json.dumps(config, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(raw), dtype=torch.uint8)


def _match_layer_type(key: str, types: list[str]) -> bool:
    for t in types:
        if f".{t}.weight" in key:
            return True
    return False


def convert(
    input_path: str,
    output_path: str,
    *,
    nvfp4_types: list[str] | None = None,
    nvfp4_n: int = 0,
    all_mlp: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    analysis_path: str | None = None,
):
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Device: {device}")
    print(f"File size: {os.path.getsize(input_path) / (1024**3):.2f} GiB")
    print()

    # Load analysis ranking if provided
    ranking = None
    if analysis_path and os.path.isfile(analysis_path):
        with open(analysis_path, "r", encoding="utf-8") as f:
            ranking = json.load(f)
        ranking.sort(key=lambda x: x.get("rel_err", 999.0))
        print(f"Loaded analysis ranking: {len(ranking)} layers from {analysis_path}")

    # Read all keys and metadata first
    with safe_open(input_path, framework="pt", device="cpu") as f:
        all_keys = set(f.keys())
        metadata = f.metadata()

    # Parse existing metadata
    quant_meta_str = metadata.get("_quantization_metadata", "{}")
    if isinstance(quant_meta_str, bytes):
        quant_meta_str = quant_meta_str.decode("utf-8")
    quant_meta = json.loads(quant_meta_str)
    layers_meta = quant_meta.get("layers", {})

    prefix = _find_krea2_key_prefix(all_keys)
    print(f"Krea2 prefix: {prefix!r}")

    # Find INT8 layers
    int8_layers = []
    for key in sorted(all_keys):
        if not key.endswith(".weight"):
            continue
        if key.endswith(".weight_scale") or key.endswith(".weight_blocks"):
            continue
        base = key.replace(".weight", "")
        cq_key = f"{base}.comfy_quant"
        if cq_key not in all_keys:
            continue  # Not INT8

        meta_key = _meta_base_key(base, prefix)
        conf = layers_meta.get(meta_key, {})
        int8_layers.append((key, base, meta_key, conf))

    print(f"INT8 layers found: {len(int8_layers)}")

    # Determine which layers to convert
    convert_keys = set()

    if all_mlp:
        for key, base, meta_key, conf in int8_layers:
            if ".mlp." in key:
                convert_keys.add(key)
        print(f"--all_mlp: {len(convert_keys)} MLP layers selected")

    if nvfp4_types:
        for key, base, meta_key, conf in int8_layers:
            if _match_layer_type(key, nvfp4_types):
                convert_keys.add(key)
        print(f"--nvfp4_types {nvfp4_types}: {len(convert_keys)} layers selected")

    if nvfp4_n > 0 and ranking:
        count = 0
        for r in ranking:
            if count >= nvfp4_n:
                break
            k = r["key"]
            if k in all_keys:
                convert_keys.add(k)
                count += 1
        print(f"--nvfp4_n {nvfp4_n}: {len(convert_keys)} layers selected (ranking-based)")

    if not convert_keys:
        print("\nWARNING: No layers selected for NVFP4 conversion.")
        print("Use --nvfp4_types, --nvfp4_n, or --all_mlp to select layers.")
        return

    print(f"\nTotal layers to convert: {len(convert_keys)}")
    print(f"Layers staying INT8:    {len(int8_layers) - len(convert_keys)}")
    print()

    # Load all tensors
    new_sd: dict[str, torch.Tensor] = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for key in tqdm(sorted(all_keys), desc="Loading"):
            new_sd[key] = f.get_tensor(key)

    # Convert selected layers
    n_nvfp4 = 0
    n_int8_kept = 0
    n_convrot_nvfp4 = 0

    print("Converting selected INT8 layers to NVFP4...")
    for i, (key, base, meta_key, conf) in enumerate(int8_layers):
        if key not in convert_keys:
            continue

        q_tensor = new_sd[key]
        scale_key = f"{base}.weight_scale"
        cq_key = f"{base}.comfy_quant"
        scale = new_sd.get(scale_key)

        if scale is None:
            print(f"  [SKIP] {key}: no weight_scale found")
            continue

        # Dequantize INT8 -> rotated weight (same ConvRot space)
        if scale.dim() == 0:
            w_dq = q_tensor.float() * scale.item()
        elif scale.dim() == 2 and scale.shape[1] == 1:
            w_dq = q_tensor.float() * scale
        else:
            w_dq = q_tensor.float() * scale

        # Quantize to NVFP4 (same rotated space, no re-rotation needed)
        w_bf16 = w_dq.to(dtype=torch.bfloat16, device=device)

        is_convrot = conf.get("convrot", False)
        gs = conf.get("convrot_groupsize", 256)
        used_gs_nv = None

        if is_convrot:
            used_gs_nv = convrot_group_size_for_features(int(w_bf16.shape[1]), gs)

        try:
            qdata, params = TensorCoreNVFP4Layout.quantize(w_bf16)
            tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)

            # Remove INT8 artifacts
            del new_sd[key]
            del new_sd[scale_key]
            if cq_key in new_sd:
                del new_sd[cq_key]

            # Add NVFP4 tensors
            for suffix, nv_tensor in tensors.items():
                new_sd[f"{base}.weight{suffix}"] = nv_tensor.cpu()

            # Update metadata
            if is_convrot and used_gs_nv is not None:
                layers_meta[meta_key] = {
                    "format": "nvfp4",
                    "convrot": True,
                    "convrot_groupsize": int(used_gs_nv),
                }
                n_convrot_nvfp4 += 1
            else:
                layers_meta[meta_key] = {"format": "nvfp4"}

            n_nvfp4 += 1

            if (i + 1) % 32 == 0 or i == len(int8_layers) - 1:
                print(f"  [{i+1}/{len(convert_keys)}] {key} -> NVFP4 done")

            del w_bf16, qdata, params
            if device == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [ERROR] {key}: {e}")
            n_int8_kept += 1

    # Count remaining INT8
    for key, base, meta_key, conf in int8_layers:
        if key in new_sd:
            n_int8_kept += 1

    # Build metadata
    quant_meta["layers"] = layers_meta
    final_metadata = OrderedDict()
    final_metadata["_quantization_metadata"] = json.dumps(quant_meta)
    final_metadata["converted_by"] = "HSWQ partial INT8->NVFP4 converter"
    final_metadata["hswq_model"] = "krea2"
    final_metadata["hswq_mixed_int8_nvfp4"] = "1"
    final_metadata["hswq_nvfp4_count"] = str(n_nvfp4)
    final_metadata["hswq_int8_count"] = str(n_int8_kept)

    for k, v in metadata.items():
        if k not in final_metadata and k != "_quantization_metadata":
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            final_metadata[k] = v

    print(f"\nSaving: {output_path}")
    save_file(new_sd, output_path, metadata=final_metadata)
    out_size = os.path.getsize(output_path)
    in_size = os.path.getsize(input_path)
    saved = (in_size - out_size) / (1024**3)
    print(f"Done. Size: {out_size / (1024**3):.2f} GiB "
          f"(was {in_size / (1024**3):.2f} GiB, saved {saved:.2f} GiB)")
    print(f"  NVFP4 converted: {n_nvfp4} (convrot={n_convrot_nvfp4})")
    print(f"  INT8 kept:       {n_int8_kept}")

    del new_sd
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert selected ConvRot INT8 layers to ConvRot NVFP4 in a Krea2 checkpoint. "
            "Creates a mixed INT8+NVFP4 checkpoint."
        )
    )
    parser.add_argument("--input", "--model", dest="input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nvfp4_types", default=None,
                        help="Comma-separated layer types (e.g., mlp.up,mlp.gate,mlp.down)")
    parser.add_argument("--nvfp4_n", type=int, default=0,
                        help="Number of lowest-error layers to convert (needs --analysis)")
    parser.add_argument("--all_mlp", action="store_true",
                        help="Convert all MLP layers to NVFP4")
    parser.add_argument("--analysis", default=None,
                        help="Path to nvfp4_conversion_analysis.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"])
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input not found: {args.input}")
        sys.exit(1)

    nvfp4_types = None
    if args.nvfp4_types:
        nvfp4_types = [t.strip() for t in args.nvfp4_types.split(",") if t.strip()]

    convert(
        args.input,
        args.output,
        nvfp4_types=nvfp4_types,
        nvfp4_n=args.nvfp4_n,
        all_mlp=bool(args.all_mlp),
        device=args.device,
        analysis_path=args.analysis,
    )


if __name__ == "__main__":
    main()

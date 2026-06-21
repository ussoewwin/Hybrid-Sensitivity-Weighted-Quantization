"""AST/static checks for quantize_sdxl_hswq_v2.0.py (no torch required)."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "quantize_sdxl_hswq_v2.0.py"

REQUIRED_FUNCS = {
    "_remap_profile_to_diffusers",
    "_compute_sdxl_keypattern_veto",
    "_compute_sdxl_per_projection_attn_veto",
    "_compute_structural_veto",
    "_autonomous_supplemental_veto",
    "_mse_grayzone_veto_reassessment",
    "load_unet_from_safetensors",
    "derive_hswq_strategy",
    "DualMonitor",
    "hook_fn",
}

FORBIDDEN_SUBSTRINGS = (
    "load_zit_model",
    "ZITCalibrationPipeline",
    "_compute_nextdit_keypattern_veto",
    "_compute_per_projection_qkv_veto",
    "_is_zanime_profile",
    "is_zanime",
    "NextDiT",
)

ALLOWED_EXCEPTIONS = ()  # none


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(TARGET))

    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    missing = sorted(REQUIRED_FUNCS - defined)
    if missing:
        print("FAIL: missing definitions:", ", ".join(missing))
        return 1

    for bad in FORBIDDEN_SUBSTRINGS:
        if bad in text and bad not in ALLOWED_EXCEPTIONS:
            print(f"FAIL: forbidden substring found: {bad}")
            return 1

    # main() must load UNet before derive_hswq_strategy
    load_idx = text.find("load_unet_from_safetensors(")
    remap_idx = text.find("_remap_profile_to_diffusers(")
    derive_idx = text.find("derive_hswq_strategy(")
    if not (load_idx < remap_idx < derive_idx):
        print("FAIL: main order must be load_unet -> remap -> derive_hswq_strategy")
        return 1

    if "comfy_quant" not in text or "weight_scale" not in text:
        print("FAIL: ComfyUI quant metadata keys missing in save path")
        return 1

    if "already remapped to Diffusers module names" not in text:
        print("FAIL: _norm_profile should use remapped profile (no prefix strip loop)")
        return 1

    if "HSWQWeightedHistogramOptimizerV4" not in text:
        print("FAIL: V4 optimizer import/usage missing")
        return 1

    if "_compute_sdxl_keypattern_veto" not in text or "_mse_grayzone_veto_reassessment" not in text:
        print("FAIL: SDXL V2.0 VETO pipeline incomplete")
        return 1

    if "value.to(torch.float16)" not in text:
        print("FAIL: keep layers must be explicitly cast to float16 (v2.0 save path)")
        return 1

    if "max(weight_amax_dict[weight_key], 1e-6)" not in text:
        print("FAIL: amax floor (1e-6) missing in save clamp")
        return 1

    analyze_sdxl = ROOT / "analyze" / "analyze_sdxl_distribution.py"
    if not analyze_sdxl.is_file():
        print("FAIL: analyze/analyze_sdxl_distribution.py missing")
        return 1

    print(
        "static OK",
        TARGET.stat().st_size,
        "bytes",
        len(text.splitlines()),
        "lines",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

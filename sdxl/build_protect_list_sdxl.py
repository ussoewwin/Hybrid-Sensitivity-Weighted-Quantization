# -*- coding: utf-8 -*-
"""Build the FP16 protection list for the SDXL reverse hybrid using the HSWQ V3.1 selector.

The 300 MiB FP16 protection of the HSWQ pipeline is not a cheap heuristic: it ranks the
candidates with the V4 hybrid SVD-RMS importance through an INT8 weighted-histogram MSE at the
absmax pack point (full SVD), combined with the DualMonitor activation statistics, under a fixed
300 MiB budget. This script does NOT reimplement that; it drives the permitted V3.1 script
(sdxl/quantize_sdxl_hswq_v3.1.py, imported as a module) to produce the pack, then derives the
protected set as

    protected = { every ConvRot-eligible matmul module of the checkpoint } - { layers converted by V3.1 }

and writes it as {"protected": [...]} for `diag_impact_sdxl.py --protect_list`.

That list is what keeps the same layers at FP16 in the reverse hybrid, so the reverse pipeline
reproduces the HSWQ protection premise without reimplementing the selector.

Usage:
    python sdxl/build_protect_list_sdxl.py <base.safetensors> <protect_list.json> \
        --calib_file <prompts.txt> --comfy_path <ComfyUI-master> \
        [--pack-out <pack.safetensors>] [--v31 <path to quantize_sdxl_hswq_v3.1.py>] \
        [--num_calib_samples 32] [--num_inference_steps 25] [--reuse-pack]

Requires (as V3.1 itself does): CUDA, the calibration prompts, and ComfyUI for the calibration pass.
"""
import argparse
import importlib.util
import json
import math
import os
import sys


BOUNDARY_EXACT = ("input_blocks.0.0",)
BOUNDARY_PREFIX = ("out.", "time_embed.", "add_embedding.", "label_emb.")


def _bare(name: str) -> str:
    for p in ("model.diffusion_model.", "diffusion_model."):
        if name.startswith(p):
            return name[len(p):]
    return name


def _is_boundary(name: str) -> bool:
    b = _bare(name)
    return b in BOUNDARY_EXACT or b.startswith(BOUNDARY_PREFIX)


def matmul_modules(base_path: str) -> set:
    """Every ConvRot-eligible matmul module of the checkpoint (bare names), boundary excluded."""
    from safetensors import safe_open

    out = set()
    with safe_open(os.path.abspath(base_path), framework="pt", device="cpu") as f:
        for k in f.keys():
            if not k.startswith("model.diffusion_model.") or not k.endswith(".weight"):
                continue
            shp = f.get_slice(k).get_shape()
            if len(shp) not in (2, 4) or shp[1] < 4:
                continue
            if _is_boundary(k):
                continue
            out.add(_bare(k[:-len(".weight")]))
    return out


def converted_layers(pack_path: str) -> set:
    """Module names (bare) that the pack actually converted, from its _quantization_metadata."""
    from safetensors import safe_open

    with safe_open(os.path.abspath(pack_path), framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
    qm = json.loads(meta["_quantization_metadata"])
    return {_bare(k) for k in qm["layers"]}


def run_v31(v31_path: str, base: str, pack_out: str, calib_file: str, comfy_path: str,
            num_calib_samples: int, num_inference_steps: int) -> None:
    """Import the permitted V3.1 module and run its main() with the pack command line."""
    spec = importlib.util.spec_from_file_location("hswq_v31_module", os.path.abspath(v31_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    saved = list(sys.argv)
    sys.argv = [
        os.path.basename(v31_path),
        "--input", os.path.abspath(base),
        "--output", os.path.abspath(pack_out),
        "--calib_file", os.path.abspath(calib_file),
        "--num_calib_samples", str(num_calib_samples),
        "--num_inference_steps", str(num_inference_steps),
        "--keep_ratio", "0",
        "--per_channel_int8",
        "--convrot",
        "--no-bias_correction",
        "--comfy_path", os.path.abspath(comfy_path),
        "--no-bench",
    ]
    try:
        mod.main()
    finally:
        sys.argv = saved


def parse_args():
    ap = argparse.ArgumentParser(description="Build the FP16 protection list via the HSWQ V3.1 selector")
    ap.add_argument("base", help="FP16 SDXL checkpoint")
    ap.add_argument("out", help="output protect-list json")
    ap.add_argument("--calib_file", required=True)
    ap.add_argument("--comfy_path", default="ComfyUI-master")
    ap.add_argument("--pack-out", default=None,
                    help="pack path written by V3.1 (default: <base-stem>hswq_r32_1off_convrot_int8_repro.safetensors)")
    ap.add_argument("--v31", default=None,
                    help="path to quantize_sdxl_hswq_v3.1.py (default: next to this script)")
    ap.add_argument("--num_calib_samples", type=int, default=32)
    ap.add_argument("--num_inference_steps", type=int, default=25)
    ap.add_argument("--reuse-pack", action="store_true",
                    help="reuse an existing --pack-out instead of running V3.1 again")
    return ap.parse_args()


def main():
    a = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.abspath(a.base)
    v31 = os.path.abspath(a.v31) if a.v31 else os.path.join(here, "quantize_sdxl_hswq_v3.1.py")
    pack = os.path.abspath(a.pack_out) if a.pack_out else os.path.join(
        os.path.dirname(base),
        os.path.splitext(os.path.basename(base))[0] + "hswq_r32_1off_convrot_int8_repro.safetensors",
    )

    if a.reuse_pack and os.path.isfile(pack):
        print(f"[v3.1] reuse pack: {pack}", flush=True)
    else:
        print(f"[v3.1] running selector -> {pack}", flush=True)
        run_v31(v31, base, pack, a.calib_file, a.comfy_path, a.num_calib_samples, a.num_inference_steps)

    cands = matmul_modules(base)
    conv = converted_layers(pack)
    protected = sorted(cands - conv)
    extra = sorted(conv - cands)
    if extra:
        print(f"[warn] pack converted {len(extra)} layers not in the candidate set (first 3: {extra[:3]})", flush=True)

    payload = 0
    from safetensors import safe_open
    with safe_open(base, framework="pt", device="cpu") as f:
        for name in protected:
            shp = f.get_slice("model.diffusion_model." + name + ".weight").get_shape()
            n = 1
            for s in shp:
                n *= s
            payload += n

    out = {
        "source": os.path.abspath(pack),
        "candidates": len(cands),
        "converted_by_v31": len(conv),
        "protected": protected,
        "protected_count": len(protected),
        "protected_payload_mib": payload / (1024 ** 2),
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"[protect] candidates={len(cands)} converted={len(conv)} protected={len(protected)} "
          f"payload={payload / (1024 ** 2):.1f} MiB", flush=True)
    print(f"saved {a.out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

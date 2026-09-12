# -*- coding: utf-8 -*-
"""Automatic SDXL diag -> reverse ConvRot INT8 driver (self-contained; the only imported asset is
the permitted HSWQ V3.1 selector inside build_protect_list_sdxl.py).

Calibration requirement: the FP16 protection premise of the HSWQ pipeline is produced by the
V3.1 selector, which ranks the candidates with the V4 hybrid SVD-RMS importance through an INT8
weighted-histogram MSE plus the DualMonitor activation statistics, under a fixed 300 MiB budget.
That ranking therefore needs a CALIBRATION pass (real sampling with activation hooks) BEFORE the
diag can run: the diag only measures the layers that are NOT protected, so the protected list must
exist first. The convert stage additionally needs its own calibration pass when
--bias_correction is used (rotated activation means).

Stages (each a subprocess of this repository's scripts; no dynamic sibling import):

  A. SELECT  : sdxl/build_protect_list_sdxl.py -> drives the permitted V3.1 selector
               (calibration + V4 histogram MSE + full SVD + 300 MiB budget) and writes the
               protected-layer list; skipped when the list already exists (--reuse-protect).
  B. IMPACT  : sdxl/diag_impact_sdxl.py --protect_list <list> (measures every candidate except
               the protected layers; skipped when the impact json already exists).
  C. CONVERT : sdxl/gen_reverse_int8_sdxl.py with K = candidates - protected, i.e. every
               non-protected layer becomes ConvRot INT8 and the protected layers stay FP16
               (the HSWQ protection premise), optionally with --bias_correction.
  D. GATE    : benchmark/sdxl_int8_traj_compare.py when --gate is given.

Usage:
    python sdxl/auto_reverse_int8_sdxl.py <base.safetensors> <impact.json> \
        --calib_file <prompts.txt> --comfy_path <ComfyUI-master> \
        [--protect-list <protect.json>] [--reuse-protect] \
        [--out-dir <dir>] [--out-name <name.safetensors>] \
        [--steps 4] [--seed 42] [--width 1024] [--height 1024] \
        [--force-impact] [--skip-impact] \
        [--bias_correction [--num_calib_samples 32] [--num_inference_steps 25]] \
        [--gate]
"""
import argparse
import json
import os
import subprocess
import sys


def _run(cmd):
    print("[run]", " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    return subprocess.run(cmd, check=True)


def _counts(protect_json: str):
    with open(protect_json, encoding="utf-8") as f:
        d = json.load(f)
    protected = d.get("protected", [])
    candidates = d.get("candidates")
    return len(protected), candidates


def parse_args():
    ap = argparse.ArgumentParser(description="Automatic SDXL diag -> reverse ConvRot INT8 driver")
    ap.add_argument("base", help="FP16 SDXL checkpoint")
    ap.add_argument("impact", help="impact json (created by stage B if missing)")
    ap.add_argument("calib_file", help="calibration prompts (stage A selector, and stage C bias)")
    ap.add_argument("--comfy_path", default="ComfyUI-master")
    ap.add_argument("--protect-list", default=None,
                    help="protect-list json (default: <impact dir>/protect_<base-stem>.json)")
    ap.add_argument("--reuse-protect", action="store_true",
                    help="reuse an existing protect list instead of running the selector")
    ap.add_argument("--out-dir", default=None, help="output directory (default: the checkpoint's directory)")
    ap.add_argument("--out-name", default=None,
                    help="output filename (default: <base-stem>_rev_int<K>_convrot_int8.safetensors)")
    ap.add_argument("--steps", type=int, default=4, help="diag trajectory steps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--force-impact", action="store_true", help="re-measure even if the json exists")
    ap.add_argument("--skip-impact", action="store_true", help="never measure (require an existing json)")
    ap.add_argument("--num_calib_samples", type=int, default=32)
    ap.add_argument("--num_inference_steps", type=int, default=25)
    ap.add_argument("--legacy-gen", action="store_true",
                    help="use sdxl/gen_reverse_int8_sdxl_legacy.py (artifact-era boundary set) instead of the current converter")
    ap.add_argument("--bias_correction", action="store_true")
    ap.add_argument("--gate", action="store_true", help="run the 25-seed trajectory gate at the end")
    ap.add_argument("--gate-steps", type=int, default=25)
    return ap.parse_args()


def main():
    a = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    py = sys.executable
    base = os.path.abspath(a.base)
    impact = os.path.abspath(a.impact)
    stem = os.path.splitext(os.path.basename(base))[0]
    out_dir = os.path.abspath(a.out_dir) if a.out_dir else os.path.dirname(base)
    protect = os.path.abspath(a.protect_list) if a.protect_list else os.path.join(
        os.path.dirname(impact), f"protect_{stem}.json")

    # A) protection premise: requires the selector's calibration pass
    if a.reuse_protect and os.path.isfile(protect):
        print(f"[skip] protect list present: {protect}", flush=True)
    else:
        _run([py, os.path.join(here, "build_protect_list_sdxl.py"), base, protect,
              "--calib_file", os.path.abspath(a.calib_file),
              "--comfy_path", os.path.abspath(a.comfy_path),
              "--num_calib_samples", str(a.num_calib_samples),
              "--num_inference_steps", str(a.num_inference_steps)])
    n_protected, candidates = _counts(protect)
    if candidates is None:
        raise SystemExit("protect list json has no 'candidates' field")
    k = candidates - n_protected
    print(f"[plan] candidates={candidates}  protected(FP16)={n_protected}  ->  K={k}", flush=True)

    # B) impact on the non-protected layers
    if a.force_impact or (not a.skip_impact and not os.path.isfile(impact)):
        _run([py, os.path.join(here, "diag_impact_sdxl.py"), base, impact,
              "--comfy_path", os.path.abspath(a.comfy_path),
              "--protect_list", protect,
              "--steps", str(a.steps), "--seed", str(a.seed),
              "--width", str(a.width), "--height", str(a.height)])
    else:
        print(f"[skip] impact json present: {impact}", flush=True)

    # C) convert
    out_name = a.out_name or f"{stem}_rev_int{k}_convrot_int8.safetensors"
    out_path = os.path.join(out_dir, out_name)
    gen_script = "gen_reverse_int8_sdxl_legacy.py" if a.legacy_gen else "gen_reverse_int8_sdxl.py"
    gen = [py, os.path.join(here, gen_script), str(k), out_name, base, impact,
           "--out-dir", out_dir]
    if a.bias_correction:
        gen += ["--bias_correction", "--calib_file", os.path.abspath(a.calib_file),
                "--comfy_path", os.path.abspath(a.comfy_path),
                "--num_calib_samples", str(a.num_calib_samples),
                "--num_inference_steps", str(a.num_inference_steps),
                "--width", str(a.width), "--height", str(a.height)]
    _run(gen)
    print(f"[saved] {out_path}", flush=True)

    # D) gate
    if a.gate:
        _run([py, os.path.join(repo, "benchmark", "sdxl_int8_traj_compare.py"),
              "--fp16", base, "--int8", out_path,
              "--comfy_path", os.path.abspath(a.comfy_path),
              "--steps", str(a.gate_steps), "--width", str(a.width), "--height", str(a.height)])

    print("DONE", flush=True)


if __name__ == "__main__":
    main()

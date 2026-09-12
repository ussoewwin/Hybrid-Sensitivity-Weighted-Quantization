# -*- coding: utf-8 -*-
"""Automatic SDXL diag -> reverse ConvRot INT8 driver (self-contained; no other pipeline involved).

What it does, end to end, without any Owner-supplied inputs:

  1. IMPACT  : run sdxl/diag_impact_sdxl.py on the FP16 checkpoint (every ConvRot-eligible
               Linear/Conv2d) unless the impact json already exists (skip with --force-impact).
  2. PLAN    : read the impact json and derive K automatically from the FP16 protection budget:
               the K lowest-impact layers become ConvRot INT8, and the remaining matmul layers
               plus the boundary layers stay FP16. K is the largest value for which
               (sum of numel over the FP16-kept matmul layers + boundary layers) stays within
               --fp16_budget_mib (default 300 MiB; the same premise as the HSWQ 300 MiB FP16
               protection budget, but here the protected set is chosen by the measured
               trajectory impact instead of any external artifact).
  3. CONVERT : run sdxl/gen_reverse_int8_sdxl.py with the computed K (and --bias_correction when
               asked) to write the mixed checkpoint.
  4. GATE    : optionally run benchmark/sdxl_int8_traj_compare.py (25 seeds x 25 steps) and print
               the summary block.

Each stage is a fresh subprocess of this repository's scripts (no dynamic sibling import).

Usage:
    python sdxl/auto_reverse_int8_sdxl.py <base.safetensors> <impact.json> \
        --comfy_path <ComfyUI-master> \
        [--out-dir <dir>] [--out-name <name.safetensors>] \
        [--fp16_budget_mib 300] [--groupsize 256] \
        [--steps 4] [--seed 42] [--width 1024] [--height 1024] \
        [--force-impact] [--skip-impact] \
        [--bias_correction --calib_file <prompts> [--num_calib_samples 32] [--num_inference_steps 25]] \
        [--gate]
"""
import argparse
import json
import math
import os
import subprocess
import sys


def _run(cmd, **kw):
    print("[run]", " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def _matmul_modules_from_header(base_path: str):
    """{module_key: numel} for the UNet matmul weights (2D/4D), plus the boundary keys.

    module_key is the checkpoint key space ("model.diffusion_model.<name>"), matching the keys
    used by the converter.
    """
    from safetensors import safe_open

    boundary_exact = ("input_blocks.0.0",)
    boundary_prefix = ("out.", "time_embed.", "add_embedding.", "label_emb.")

    def bare(name):
        for p in ("model.diffusion_model.", "diffusion_model."):
            if name.startswith(p):
                return name[len(p):]
        return name

    mods, boundary = {}, {}
    with safe_open(os.path.abspath(base_path), framework="pt", device="cpu") as f:
        for k in f.keys():
            if not k.startswith("model.diffusion_model.") or not k.endswith(".weight"):
                continue
            shp = f.get_slice(k).get_shape()
            if len(shp) not in (2, 4) or shp[1] < 4:
                continue
            numel = 1
            for s in shp:
                numel *= s
            name = bare(k[:-len(".weight")])
            if name in boundary_exact or name.startswith(boundary_prefix):
                boundary[k[:-len(".weight")]] = numel
            else:
                mods[k[:-len(".weight")]] = numel
    return mods, boundary


def _norm_module(name: str) -> str:
    """Impact json names are named_modules names ("diffusion_model.x"); checkpoint keys carry
    "model.diffusion_model.". Normalize both sides to the bare module name."""
    n = name
    if n.endswith(".weight"):
        n = n[: -len(".weight")]
    for p in ("model.diffusion_model.", "diffusion_model."):
        if n.startswith(p):
            return n[len(p):]
    return n


def plan_k(impact_json: str, base_path: str, budget_bytes: int, groupsize: int):
    """Largest K (number of lowest-impact layers to convert) whose FP16 remainder fits the budget."""
    with open(impact_json, encoding="utf-8") as f:
        imp = json.load(f)["impacts"]

    mods, boundary = _matmul_modules_from_header(base_path)
    by_bare = {_norm_module(k): (k, n) for k, n in mods.items()}

    ranked = []
    for k, v in sorted(imp.items(), key=lambda kv: kv[1]):
        if isinstance(v, float) and math.isnan(v):
            continue
        nb = _norm_module(k)
        if nb in by_bare:
            ranked.append((nb, float(v)))
    seen, uniq = set(), []
    for nb, v in ranked:
        if nb not in seen:
            seen.add(nb)
            uniq.append((nb, v))
    ranked = uniq

    boundary_bytes = sum(boundary.values())
    total_bytes = boundary_bytes + sum(n for _, n in mods.items())
    acc = total_bytes             # extra bytes kept in FP16 when nothing is converted
    k_best = 0
    for i, (nb, _v) in enumerate(ranked, start=1):
        n_in = by_bare[nb][1]
        acc_next = acc - n_in      # this layer is converted -> its extra bytes leave the FP16 set
        if acc_next >= budget_bytes:
            k_best = i
            acc = acc_next
        else:
            break
    protected = len(ranked) - k_best + len(boundary)
    return {
        "K": k_best,
        "candidates": len(ranked),
        "boundary_layers": len(boundary),
        "protected_layers": protected,
        "protected_payload_mib": acc / (1024 ** 2),
        "total_payload_mib": total_bytes / (1024 ** 2),
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Automatic SDXL diag -> reverse ConvRot INT8 driver")
    ap.add_argument("base", help="FP16 SDXL checkpoint")
    ap.add_argument("impact", help="impact json (created by stage 1 if missing)")
    ap.add_argument("--comfy_path", default="ComfyUI-master")
    ap.add_argument("--out-dir", default=None, help="output directory (default: the checkpoint's directory)")
    ap.add_argument("--out-name", default=None,
                    help="output filename (default: <base-stem>_rev_int<K>_convrot_int8.safetensors)")
    ap.add_argument("--fp16_budget_mib", type=float, default=300.0,
                    help="FP16 protection budget in MiB (default 300, the HSWQ premise)")
    ap.add_argument("--groupsize", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4, help="diag trajectory steps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--force-impact", action="store_true", help="re-measure even if the json exists")
    ap.add_argument("--skip-impact", action="store_true", help="never measure (require an existing json)")
    ap.add_argument("--bias_correction", action="store_true")
    ap.add_argument("--calib_file", default=None)
    ap.add_argument("--num_calib_samples", type=int, default=32)
    ap.add_argument("--num_inference_steps", type=int, default=25)
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
    out_dir = os.path.abspath(a.out_dir) if a.out_dir else os.path.dirname(base)

    # 1) impact
    if a.force_impact or (not a.skip_impact and not os.path.isfile(impact)):
        _run([py, os.path.join(here, "diag_impact_sdxl.py"), base, impact,
              "--comfy_path", os.path.abspath(a.comfy_path),
              "--steps", str(a.steps), "--seed", str(a.seed),
              "--width", str(a.width), "--height", str(a.height),
              "--groupsize", str(a.groupsize)])
    else:
        print(f"[skip] impact json present: {impact}", flush=True)

    # 2) plan K from the FP16 protection budget
    info = plan_k(impact, base, int(a.fp16_budget_mib * 1024 * 1024), a.groupsize)
    print(f"[plan] candidates={info['candidates']}  boundary={info['boundary_layers']}  "
          f"K={info['K']}  FP16-protected={info['protected_layers']} layers  "
          f"payload={info['protected_payload_mib']:.1f} MiB (budget {a.fp16_budget_mib:.0f} MiB)",
          flush=True)

    stem = os.path.splitext(os.path.basename(base))[0]
    out_name = a.out_name or f"{stem}_rev_int{info['K']}_convrot_int8.safetensors"
    out_path = os.path.join(out_dir, out_name)

    # 3) convert
    gen = [py, os.path.join(here, "gen_reverse_int8_sdxl.py"), str(info["K"]), out_name, base, impact,
           "--out-dir", out_dir, "--groupsize", str(a.groupsize)]
    if a.bias_correction:
        if not a.calib_file:
            raise SystemExit("--bias_correction requires --calib_file")
        gen += ["--bias_correction", "--calib_file", os.path.abspath(a.calib_file),
                "--comfy_path", os.path.abspath(a.comfy_path),
                "--num_calib_samples", str(a.num_calib_samples),
                "--num_inference_steps", str(a.num_inference_steps),
                "--width", str(a.width), "--height", str(a.height)]
    _run(gen)
    print(f"[saved] {out_path}", flush=True)

    # 4) gate
    if a.gate:
        _run([py, os.path.join(repo, "benchmark", "sdxl_int8_traj_compare.py"),
              "--fp16", base, "--int8", out_path,
              "--comfy_path", os.path.abspath(a.comfy_path),
              "--steps", str(a.gate_steps), "--width", str(a.width), "--height", str(a.height)])

    print("DONE", flush=True)


if __name__ == "__main__":
    main()

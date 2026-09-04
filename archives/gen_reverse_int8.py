#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Krea2 INT8 artifact selector (int8 gen, Step 2).

Reads the impact json written by Krea2/diag_impact_int8.py (FULL vs
native-INT8 per-Layer divergence) and produces the selected INT8 artifact:
the --keep N highest-impact Linears are REPLACED by their original bf16
weights (from the FULL base); every other layer keeps its native convrot
INT8 weights (weight_scale / comfy_quant / metadata entries untouched).

Usage (one line):
    python Krea2/gen_reverse_int8.py <native(test2)> <base(test)> \
        impact_krea2_int8.json --out test3.safetensors --keep N
"""
from __future__ import annotations

import argparse
import json

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def _find_krea2_key_prefix(keys):
    for prefix in ("model.diffusion_model.", "diffusion_model.", ""):
        if f"{prefix}txtfusion.projector.weight" in keys:
            return prefix
    raise ValueError("Not a Krea2 checkpoint: missing txtfusion.projector.weight")


def _meta_base_key(k: str) -> str:
    if "model.diffusion_model." in k:
        return k.split("model.diffusion_model.")[-1]
    if "diffusion_model." in k:
        return k.split("diffusion_model.")[-1]
    return k


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Krea2 INT8 artifact selector: restore the --keep N highest-impact "
            "Linears (from diag_impact_int8 impact json) to original bf16."
        )
    )
    ap.add_argument("native", help="NATIVE convrot INT8 safetensors (test2)")
    ap.add_argument("base", help="FULL baseline bf16/fp16 safetensors (test)")
    ap.add_argument("impact", help="impact json from diag_impact_int8.py")
    ap.add_argument("--out", "-o", required=True,
                    help="output selected INT8 safetensors path (test3)")
    ap.add_argument("--keep", type=int, default=0,
                    help="restore the top-N highest-impact Linears to bf16 "
                         "(0 = keep native INT8 everywhere)")
    return ap.parse_args()


def main():
    a = parse_args()
    keep_n = max(0, int(a.keep))

    # 1) impact json (FULL vs native-INT8 per-Layer divergence)
    with open(a.impact, "r", encoding="utf-8") as fi:
        imp_data = json.load(fi)
    impacts = imp_data.get("impacts", {})
    ranked = sorted(
        ((k, float(v)) for k, v in impacts.items() if v == v),  # drop NaN
        key=lambda kv: kv[1], reverse=True,   # highest impact first
    )
    print(f"impact entries: {len(ranked)}")
    keep_set = {k for k, _ in ranked[:keep_n]}
    if keep_n > 0:
        print(f"[keep] top {keep_n} highest-impact Linears restored to bf16:")
        for k, v in ranked[:keep_n]:
            print(f"  KEEP {k}  impact={v:.3e}")

    # 2) native INT8 artifact (base of the output)
    print(f"Loading native INT8 artifact: {a.native}")
    new_sd = load_file(a.native)
    prefix = _find_krea2_key_prefix(new_sd)

    # 3) FULL baseline bf16 weights (restoration source)
    print(f"Loading FULL baseline: {a.base}")
    full_sd = load_file(a.base)
    full_prefix = _find_krea2_key_prefix(full_sd)

    with safe_open(a.native, framework="pt") as fh:
        meta_raw = (fh.metadata() or {}).get("_quantization_metadata", "{}")
    meta_layers = json.loads(meta_raw).get("layers", {})

    # 4) restore the --keep N highest-impact Linears to original bf16
    restored = 0
    missing_full = 0
    for k in sorted(keep_set):
        native_base = f"{prefix}{k}"
        full_base = f"{full_prefix}{k}"
        full_w = full_sd.get(f"{full_base}.weight")
        if full_w is None:
            missing_full += 1
            print(f"  WARN: no bf16 weight in FULL base for {k}")
            continue
        new_sd[f"{native_base}.weight"] = full_w
        new_sd.pop(f"{native_base}.weight_scale", None)
        new_sd.pop(f"{native_base}.comfy_quant", None)
        meta_layers.pop(k, None)
        restored += 1

    metadata = {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": meta_layers}
        )
    }
    print(f"Saving to: {a.out}")
    print(
        f"Restored to bf16: {restored}/{keep_n} (missing_full={missing_full})  "
        f"native INT8 layers remaining: {len(meta_layers)}"
    )
    save_file(new_sd, a.out, metadata=metadata)
    print("Done!")


if __name__ == "__main__":
    main()

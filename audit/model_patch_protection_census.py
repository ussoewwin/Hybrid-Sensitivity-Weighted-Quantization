#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Census of the 2D layers a model-patch pack keeps in original precision.

A pack's protection set is the part of it that is **not** quantised. This tool states it: the 2D
weights of the source, the layers the pack declares as INT8, and the layers that therefore stay in
FP16 - with the number of each, so a verdict can cite them.

Usage:
    python audit/model_patch_protection_census.py --source <file> --pack <file>
                                                  --family diffsynth|zimage [--json <out>]
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys

from safetensors import safe_open
from safetensors.torch import load_file

COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"


def zimage_convert(src):
    cand = os.path.join(COMFY, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools")
    spec = importlib.util.spec_from_file_location("_hswq_census", os.path.join(cand, "__init__.py"),
                                                  submodule_search_locations=[cand])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["_hswq_census"] = pkg
    spec.loader.exec_module(pkg)
    return importlib.import_module("_hswq_census.nodes.hswq_model_patch_loader").z_image_convert(src)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--family", choices=["diffsynth", "zimage"], required=True)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    src = load_file(os.path.abspath(args.source), device="cpu")
    if args.family == "zimage":
        src = zimage_convert(src)
    with safe_open(os.path.abspath(args.pack), framework="pt", device="cpu") as f:
        meta = json.loads(f.metadata().get("_quantization_metadata", "{}"))
    declared = set(layer + ".weight" for layer in meta.get("layers", {}))
    two_d = [k for k, t in src.items() if k.endswith(".weight") and t.ndim == 2]
    kept = sorted(k for k in two_d if k not in declared)

    rec = {
        "tool": "model_patch_protection_census.py",
        "pack": os.path.basename(args.pack),
        "source": os.path.basename(args.source),
        "family": args.family,
        "two_dimensional_weights_in_the_source": len(two_d),
        "quantised": len(declared),
        "kept_in_fp16_count": len(kept),
        "kept_in_fp16": kept,
        "kept_in_fp16_summary": ", ".join(kept) if kept else "none",
    }
    print("%-56s | 2D %d | quantised %d | kept %s"
          % (rec["pack"][:56], len(two_d), len(declared), kept))
    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

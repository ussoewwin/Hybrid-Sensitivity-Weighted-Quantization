# -*- coding: utf-8 -*-
"""S0. SAM3.1 input verification and complete 1,590-tensor census and classification.

Groups every tensor in the architecture into exactly one structural group:
  - detector.backbone.language (language encoder)
  - detector.backbone.vision (vision backbone)
  - detector.geometry_encoder (mask/geometry encoder)
  - detector.transformer (detector encoder/decoder)
  - detector.segmentation_head (pixel decoder and head)
  - detector.dot_prod_scoring (scoring projection)
  - tracker.model (memory and tracking backbone)

Outputs tensor count, parameter count, and dimensional distribution (2-D, 4-D, 1-D)
to md/backdoor/verdicts/records/sam31_census.json.
Gate S0: classified_total == 1590, unclassified == 0, total_parameters == 872,661,740.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from safetensors import safe_open


def classify_tensor(key: str) -> str:
    if "language_backbone" in key or "backbone.language" in key:
        return "detector.backbone.language"
    elif "backbone.vision" in key or key.startswith("detector.backbone"):
        return "detector.backbone.vision"
    elif "geometry_encoder" in key:
        return "detector.geometry_encoder"
    elif "transformer" in key:
        return "detector.transformer"
    elif "segmentation_head" in key:
        return "detector.segmentation_head"
    elif "dot_prod_scoring" in key:
        return "detector.dot_prod_scoring"
    elif key.startswith("tracker"):
        return "tracker.model"
    else:
        return "unclassified"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"D:\USERFILES\ComfyUI\ComfyUI\models\unet\sam3.1_multiplex_hswq_convrot_int8.safetensors")
    ap.add_argument("--out", default="md/backdoor/verdicts/records/sam31_census.json")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: model file not found: {args.model}")
        return 1

    groups: dict[str, dict] = {
        "detector.backbone.language": {"tensors": 0, "params": 0, "dims": {"1D": 0, "2D": 0, "4D": 0, "other": 0}, "keys": []},
        "detector.backbone.vision": {"tensors": 0, "params": 0, "dims": {"1D": 0, "2D": 0, "4D": 0, "other": 0}, "keys": []},
        "detector.geometry_encoder": {"tensors": 0, "params": 0, "dims": {"1D": 0, "2D": 0, "4D": 0, "other": 0}, "keys": []},
        "detector.transformer": {"tensors": 0, "params": 0, "dims": {"1D": 0, "2D": 0, "4D": 0, "other": 0}, "keys": []},
        "detector.segmentation_head": {"tensors": 0, "params": 0, "dims": {"1D": 0, "2D": 0, "4D": 0, "other": 0}, "keys": []},
        "detector.dot_prod_scoring": {"tensors": 0, "params": 0, "dims": {"1D": 0, "2D": 0, "4D": 0, "other": 0}, "keys": []},
        "tracker.model": {"tensors": 0, "params": 0, "dims": {"1D": 0, "2D": 0, "4D": 0, "other": 0}, "keys": []},
        "unclassified": {"tensors": 0, "params": 0, "dims": {"1D": 0, "2D": 0, "4D": 0, "other": 0}, "keys": []},
    }

    with safe_open(args.model, framework="pt", device="cpu") as f:
        all_keys = f.keys()
        # Filter pack-internal metadata, scales, and comfy_quant markers
        logical_keys = [
            k for k in all_keys
            if not k.endswith((".weight_scale", ".weight_delta", ".comfy_quant"))
            and k != "_quantization_metadata"
        ]

        total_params = 0
        for k in logical_keys:
            t = f.get_slice(k)
            shape = t.get_shape()
            p_count = 1
            for d in shape:
                p_count *= d
            total_params += p_count

            grp_name = classify_tensor(k)
            grp = groups[grp_name]
            grp["tensors"] += 1
            grp["params"] += p_count
            grp["keys"].append(k)

            ndim = len(shape)
            if ndim == 1:
                grp["dims"]["1D"] += 1
            elif ndim == 2:
                grp["dims"]["2D"] += 1
            elif ndim == 4:
                grp["dims"]["4D"] += 1
            else:
                grp["dims"]["other"] += 1

    classified_count = sum(g["tensors"] for k, g in groups.items() if k != "unclassified")
    unclassified_count = groups["unclassified"]["tensors"]

    gate_pass = (classified_count == 1590 and unclassified_count == 0 and total_params == 872661740)

    out_data = {
        "file": os.path.basename(args.model),
        "source_published_sha256": "9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03",
        "source_published_bytes": 1745546848,
        "total_logical_tensors": len(logical_keys),
        "total_parameters": total_params,
        "classified_total": classified_count,
        "unclassified_total": unclassified_count,
        "gate_s0_pass": gate_pass,
        "group_summary": {
            grp_name: {
                "tensors": grp["tensors"],
                "params": grp["params"],
                "dims": grp["dims"],
            }
            for grp_name, grp in groups.items() if grp["tensors"] > 0
        }
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(out_data, fp, indent=2)

    print(f"Total logical tensors: {len(logical_keys)} (expected 1590)")
    print(f"Total parameters: {total_params:,} (expected 872,661,740)")
    print(f"Classified: {classified_count}, Unclassified: {unclassified_count}")
    print(f"Gate S0: {'PASS' if gate_pass else 'FAIL'}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())

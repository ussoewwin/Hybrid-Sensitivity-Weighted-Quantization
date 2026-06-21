"""SDXL weight distribution profiler for HSWQ V2.0 (ComfyUI key layout)."""
import argparse
import json
import os

import numpy as np
import scipy.stats as stats
import torch
from safetensors.torch import load_file
from tqdm import tqdm


def generate_model_profile(input_path, output_path):
    print(f"Generating Distribution Profile for SDXL: {input_path}")
    sd = load_file(input_path)

    profile = {}
    target_keys = [k for k in sd.keys() if k.endswith(".weight") and len(sd[k].shape) >= 2]
    print(f"Detected {len(target_keys)} target layers for profiling.")

    for key in tqdm(target_keys, desc="Analyzing SDXL Layers"):
        w = sd[key].float().cuda()
        abs_w = torch.abs(w)
        abs_max = torch.max(abs_w).item()
        std = torch.std(w).item()
        w_np = w.cpu().numpy().flatten()
        kurtosis = float(stats.kurtosis(w_np, fisher=True))
        outlier_ratio = float(abs_max / std if std > 0 else 0)
        profile[key] = {
            "abs_max": float(abs_max),
            "std": float(std),
            "kurtosis": kurtosis,
            "outlier_ratio": outlier_ratio,
        }
        torch.cuda.empty_cache()

    all_kurtosis = [v["kurtosis"] for v in profile.values()]
    summary = {
        "total_layers": len(profile),
        "vulnerable_layers_kurtosis_gt_20": len([k for k in all_kurtosis if k > 20]),
        "semi_vulnerable_layers_kurtosis_gt_5": len(
            [k for k in all_kurtosis if 5 < k <= 20]
        ),
        "stable_layers": len([k for k in all_kurtosis if k <= 5]),
    }
    out_data = {"summary": summary, "layers": profile}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=4)

    print("\n[SDXL Profile Generation Complete]")
    print(f"  Vulnerable Layers (Kurtosis > 20): {summary['vulnerable_layers_kurtosis_gt_20']}")
    print(
        f"  Semi-Vulnerable Layers (5 < Kurtosis <= 20): "
        f"{summary['semi_vulnerable_layers_kurtosis_gt_5']}"
    )
    print(f"  Stable Layers: {summary['stable_layers']}")
    print(f"Profile saved to: {output_path}")

    del sd
    torch.cuda.empty_cache()
    return out_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDXL Distribution Profiler")
    parser.add_argument("--input", type=str, required=True, help="Path to input safetensors model")
    parser.add_argument("--output", type=str, required=True, help="Path to save the JSON profile")
    args = parser.parse_args()
    generate_model_profile(args.input, args.output)

import argparse
import torch
from safetensors.torch import load_file
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
import os
import json

def generate_model_profile(input_path, output_path):
    print(f"Generating Distribution Profile for ZIB: {input_path}")
    sd = load_file(input_path)
    
    profile = {}
    
    # Target Linear and Conv layers (weight tensors with at least 2 dimensions)
    target_keys = [k for k in sd.keys() if k.endswith(".weight") and len(sd[k].shape) >= 2]
    print(f"Detected {len(target_keys)} target layers for profiling.")
    
    for key in tqdm(target_keys, desc="Analyzing ZIB Layers"):
        w = sd[key].float().cuda()
        abs_w = torch.abs(w)
        abs_max = torch.max(abs_w).item()
        std = torch.std(w).item()
        
        # Kurtosis: measures the "tailedness" of the distribution
        w_np = w.cpu().numpy().flatten()
        kurtosis = float(stats.kurtosis(w_np, fisher=True))
        
        # Outlier Ratio: how far the maximum value is from the standard deviation
        outlier_ratio = float(abs_max / std if std > 0 else 0)
        
        profile[key] = {
            "abs_max": float(abs_max),
            "std": float(std),
            "kurtosis": kurtosis,
            "outlier_ratio": outlier_ratio
        }
        
        torch.cuda.empty_cache()

    # Create summary statistics
    all_kurtosis = [v["kurtosis"] for v in profile.values()]
    summary = {
        "total_layers": len(profile),
        "vulnerable_layers_kurtosis_gt_20": len([k for k in all_kurtosis if k > 20]),
        "semi_vulnerable_layers_kurtosis_gt_5": len([k for k in all_kurtosis if k > 5 and k <= 20]),
        "stable_layers": len([k for k in all_kurtosis if k <= 5])
    }
    
    out_data = {
        "summary": summary,
        "layers": profile
    }

    with open(output_path, "w") as f:
        json.dump(out_data, f, indent=4)
        
    print("\n[ZIB Profile Generation Complete]")
    print(f"  Vulnerable Layers (Kurtosis > 20): {summary['vulnerable_layers_kurtosis_gt_20']}")
    print(f"  Semi-Vulnerable Layers (5 < Kurtosis <= 20): {summary['semi_vulnerable_layers_kurtosis_gt_5']}")
    print(f"  Stable Layers: {summary['stable_layers']}")
    print(f"Profile saved to: {output_path}")
    
    del sd
    torch.cuda.empty_cache()
    return out_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZIB Distribution Profiler")
    parser.add_argument("--input", type=str, required=True, help="Path to input safetensors model")
    parser.add_argument("--output", type=str, required=True, help="Path to save the JSON profile")
    args = parser.parse_args()
    
    generate_model_profile(args.input, args.output)

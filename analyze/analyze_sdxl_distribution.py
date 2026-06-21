import argparse
import json
import os

import torch
from safetensors.torch import load_file


def calculate_kurtosis(tensor):
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    if std == 0:
        return 0.0
    return torch.mean(((tensor - mean) / std) ** 4).item()


# Match quantize_sdxl_hswq_v2.0.py derive_hswq_strategy static VETO thresholds.
_SDXL_EXTREME_KURTOSIS = 20.0
_SDXL_EXTREME_OUTLIER = 25.0
_SDXL_HUGE_MAGNITUDE = 20.0


def profile_score(kurtosis: float, outlier_ratio: float, abs_max: float) -> float:
    """Same formula as quantize_sdxl_hswq_v2.0 dynamic ranking (drift added at quantize time)."""
    return float(kurtosis + outlier_ratio * 2.0 + abs_max * 0.5)


def _is_static_veto_candidate(entry: dict) -> bool:
    k = entry["kurtosis"]
    o = entry["outlier_ratio"]
    m = entry["abs_max"]
    return k > _SDXL_EXTREME_KURTOSIS or o > _SDXL_EXTREME_OUTLIER or m > _SDXL_HUGE_MAGNITUDE


def analyze_sdxl_distribution(input_path, output_path):
    print(f"Loading {input_path}...")
    sd = load_file(input_path)

    target_keys = [
        k
        for k in sd.keys()
        if k.endswith(".weight")
        and len(sd[k].shape) >= 2
        and "scale" not in k
        and "comfy_quant" not in k
    ]
    print(f"Analyzing {len(target_keys)} layers...")

    linear_shape_counts: dict[tuple[int, int], int] = {}
    for key in target_keys:
        w = sd[key]
        if len(w.shape) == 2:
            sh = (int(w.shape[0]), int(w.shape[1]))
            linear_shape_counts[sh] = linear_shape_counts.get(sh, 0) + 1

    profile = {}
    for key in target_keys:
        w = sd[key].float()
        std = torch.std(w).item()
        abs_max = max(abs(w.min().item()), abs(w.max().item()))
        kurtosis = calculate_kurtosis(w)
        outlier_ratio = abs_max / std if std > 0 else 0.0
        shape_list = [int(x) for x in w.shape]
        entry = {
            "abs_max": abs_max,
            "std": std,
            "kurtosis": kurtosis,
            "outlier_ratio": outlier_ratio,
            "shape": shape_list,
            "ndim": len(w.shape),
            "profile_score": profile_score(kurtosis, outlier_ratio, abs_max),
        }
        if len(w.shape) == 2:
            sh = (shape_list[0], shape_list[1])
            entry["shape_uniqueness"] = linear_shape_counts[sh]
        profile[key] = entry

    linear_entries = [v for v in profile.values() if v.get("ndim") == 2]
    ff2_keys = [k for k in target_keys if "ff.net.2" in k]
    ff2_outliers = [profile[k]["outlier_ratio"] for k in ff2_keys]

    hard_veto_candidates = [k for k, v in profile.items() if _is_static_veto_candidate(v)]

    summary = {
        "total_layers": len(profile),
        "linear_layers": len(linear_entries),
        "high_kurtosis_count": len(
            [v for v in profile.values() if v["kurtosis"] > _SDXL_EXTREME_KURTOSIS]
        ),
        "medium_kurtosis_count": len(
            [v for v in profile.values() if 5 < v["kurtosis"] <= _SDXL_EXTREME_KURTOSIS]
        ),
        "low_kurtosis_count": len([v for v in profile.values() if v["kurtosis"] <= 5]),
        "hard_veto_candidates": len(hard_veto_candidates),
        "extreme_outlier_gt_25": len(
            [v for v in profile.values() if v["outlier_ratio"] > _SDXL_EXTREME_OUTLIER]
        ),
        "high_outlier_count": len([v for v in profile.values() if v["outlier_ratio"] > 40]),
        "huge_magnitude_count": len(
            [v for v in profile.values() if v["abs_max"] > _SDXL_HUGE_MAGNITUDE]
        ),
        "structural_veto_candidates": len(
            [v for v in linear_entries if v.get("shape_uniqueness") == 1]
        ),
        "ff_net_2_count": len(ff2_keys),
        "ff_net_2_outlier_gt_18": len([o for o in ff2_outliers if o > 18.0]),
        "ff_net_2_outlier_gt_10": len([o for o in ff2_outliers if o > 10.0]),
        "max_profile_score": max((v["profile_score"] for v in profile.values()), default=0.0),
        "max_outlier_ratio": max((v["outlier_ratio"] for v in profile.values()), default=0.0),
    }

    output_data = {"summary": summary, "layers": profile}

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Analysis complete. Saved to {output_path}")
    print(
        f"  hard_veto_candidates={summary['hard_veto_candidates']}, "
        f"structural_veto_candidates={summary['structural_veto_candidates']}, "
        f"ff.net.2 o>18={summary['ff_net_2_outlier_gt_18']}, "
        f"max_profile_score={summary['max_profile_score']:.2f}"
    )
    return output_data


def generate_model_profile(input_path: str, output_path: str) -> dict:
    """Build ComfyUI-key distribution profile JSON (same API as analyze_zib_distribution)."""
    return analyze_sdxl_distribution(input_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze SDXL weight distribution (ComfyUI keys)")
    parser.add_argument("--input", type=str, required=True, help="Path to input safetensors")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON profile")
    args = parser.parse_args()
    generate_model_profile(args.input, args.output)

"""Z-Image / ZIT one-shot: analyze → INT8 protect N keys → NVFP4+INT8 quantize.

Repo py used by this flow: this file + hswq_convert_nvfp4_zi_int8protect.py only.
Does NOT import native_convert_int8 (ConvRot/INT8 helpers live inside the converter).

Pipeline (default N=60 — NOT via 65 then truncate):
  1) Scan 2D .weight (abs_max / kurtosis / outlier)
  2) prior = recommended FP16-protect union
     (abs_max>=5 OR kurtosis>20 OR outlier_ratio_gt5>0.001;
      Kitchen Turbo blacklist excluded)
  3) Fill remaining slots by abs_max desc among NVFP4 candidates
     until exactly N (default 60)
  4) Write keys JSON under test/
  5) Call convert_to_nvfp4 with that keyset injected

Example:
  D:\\USERFILES\\fp8e4m3\\venv\\Scripts\\python.exe ^
    analyze\\analyze_convert_nvfp4_zi_int8protect.py ^
    --model D:\\...\\moodyProMix_zitV13.safetensors ^
    --output D:\\...\\moodyProMix_zitV13_nvfp4_int8protect60.safetensors ^
    --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from typing import Any

import torch
from safetensors import safe_open

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Kitchen Z-Image-Turbo blacklist (same as native_convert_nvfp4_zi*.py)
_TURBO_BLACKLIST = [
    "final_layer",
    "cap_embedder",
    "x_embedder",
    "noise_refiner",
    "context_refiner",
    "t_embedder",
]

_DEFAULT_N = 60


def _in_blacklist(key: str) -> bool:
    return any(name in key for name in _TURBO_BLACKLIST)


def _kurtosis_1d(x: torch.Tensor) -> float:
    xf = x.float().reshape(-1)
    n = xf.numel()
    if n < 4:
        return 0.0
    mean = xf.mean()
    centered = xf - mean
    m2 = (centered ** 2).mean()
    if float(m2) < 1e-30:
        return 0.0
    m4 = (centered ** 4).mean()
    return float(m4 / (m2 ** 2) - 3.0)


def _outlier_ratio(x: torch.Tensor, thr: float = 5.0) -> float:
    xf = x.float().abs().reshape(-1)
    if xf.numel() == 0:
        return 0.0
    return float((xf > thr).sum().item()) / float(xf.numel())


def analyze_model(model_path: str) -> dict[str, Any]:
    """Scan Linear 2D weights; return rows + prior / candidate rankings."""
    rows: list[dict[str, Any]] = []
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if not k.endswith(".weight"):
                continue
            t = f.get_tensor(k)
            if t.ndim != 2:
                continue
            abs_max = float(t.float().abs().max().item())
            mean_abs = float(t.float().abs().mean().item())
            kurt = _kurtosis_1d(t)
            out_r = _outlier_ratio(t, 5.0)
            out_r20 = _outlier_ratio(t, 20.0)
            bl = _in_blacklist(k)
            rows.append(
                {
                    "key": k,
                    "shape": list(t.shape),
                    "abs_max": abs_max,
                    "mean_abs": mean_abs,
                    "kurtosis": kurt,
                    "outlier_ratio_gt5": out_r,
                    "outlier_ratio_gt20": out_r20,
                    "kitchen_turbo_blacklist": bl,
                    "nvfp4_candidate": (not bl),
                }
            )

    rows.sort(key=lambda r: (-r["abs_max"], r["key"]))
    candidates = [r for r in rows if r["nvfp4_candidate"]]

    protect_abs = [r for r in candidates if r["abs_max"] >= 5.0]
    protect_kurt = [r for r in candidates if r["kurtosis"] > 20.0]
    protect_out = [r for r in candidates if r["outlier_ratio_gt5"] > 0.001]

    protect_union: dict[str, dict[str, Any]] = {}
    for r in protect_abs + protect_kurt + protect_out:
        protect_union[r["key"]] = r
    prior_list = sorted(protect_union.values(), key=lambda r: (-r["abs_max"], r["key"]))
    prior_keys = [r["key"] for r in prior_list]
    ranked_by_abs = [r["key"] for r in candidates]

    return {
        "model": model_path,
        "n_2d_weights": len(rows),
        "n_nvfp4_candidates": len(candidates),
        "n_kitchen_bf16": sum(1 for r in rows if r["kitchen_turbo_blacklist"]),
        "rows": rows,
        "candidates": candidates,
        "prior_keys": prior_keys,
        "ranked_by_abs": ranked_by_abs,
        "abs_map": {r["key"]: r["abs_max"] for r in rows},
    }


def select_protect_keys(
    analysis: dict[str, Any],
    n: int = _DEFAULT_N,
) -> list[str]:
    """prior first, then abs_max fill — stop at exactly n (no 65 hop)."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    ordered: list[str] = []
    seen: set[str] = set()
    for k in analysis["prior_keys"]:
        if k not in seen:
            ordered.append(k)
            seen.add(k)
        if len(ordered) >= n:
            break
    if len(ordered) < n:
        for k in analysis["ranked_by_abs"]:
            if k not in seen:
                ordered.append(k)
                seen.add(k)
            if len(ordered) >= n:
                break
    if len(ordered) < n:
        raise SystemExit(
            f"Not enough NVFP4 candidates for N={n}: "
            f"got {len(ordered)} "
            f"(candidates={analysis['n_nvfp4_candidates']}, "
            f"prior={len(analysis['prior_keys'])})"
        )
    return ordered[:n]


def _stem_from_model(model_path: str) -> str:
    base = os.path.basename(model_path)
    if base.lower().endswith(".safetensors"):
        base = base[: -len(".safetensors")]
    return base


def write_keys_json(
    analysis: dict[str, Any],
    protect_keys: list[str],
    n: int,
    out_path: str,
) -> None:
    prior = list(analysis["prior_keys"])
    abs_map = analysis["abs_map"]
    payload = {
        "model": analysis["model"],
        "n_2d_weights": analysis["n_2d_weights"],
        "n_nvfp4_candidates": analysis["n_nvfp4_candidates"],
        "n_kitchen_bf16": analysis["n_kitchen_bf16"],
        "n_prior": len(prior),
        "n_final": len(protect_keys),
        "n_requested": n,
        "policy": (
            f"direct N={n}: prior (abs_max>=5 | kurtosis>20 | "
            f"outlier_gt5>0.001) then abs_max fill among NVFP4 candidates; "
            f"no 65 intermediate"
        ),
        "prior_keys": prior,
        "protect_keys": protect_keys,
        "added_vs_prior": [k for k in protect_keys if k not in prior],
        "protect_abs_max": [abs_map.get(k) for k in protect_keys],
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def run_convert(
    model: str,
    output: str,
    protect_keys: list[str],
    keys_json_path: str,
    device: str,
    model_type: str,
    enable_convrot: bool,
    group_size: int,
) -> None:
    """Inject keyset into existing converter module, then convert."""
    import hswq_convert_nvfp4_zi_int8protect as conv

    keyset = frozenset(protect_keys)
    source_stem = os.path.splitext(os.path.basename(keys_json_path))[0]
    conv._INT8_PROTECT_KEYSET = keyset

    _real_save = conv.save_file

    def _save_file_patched(sd, path, metadata=None):
        if metadata is not None:
            md = OrderedDict(metadata)
            md["hswq_int8_protect_source"] = source_stem
            md["hswq_int8_protect_n"] = str(len(protect_keys))
            metadata = md
        return _real_save(sd, path, metadata=metadata)

    conv.save_file = _save_file_patched
    try:
        conv.convert_to_nvfp4(
            model,
            output,
            device=device,
            model_type=model_type,
            enable_convrot=enable_convrot,
            group_size=group_size,
        )
    finally:
        conv.save_file = _real_save


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Z-Image/ZIT one-shot: analyze → INT8 protect N (default 60) "
            "direct → NVFP4+INT8 quantize. NEW FILE; does not edit the "
            "base int8protect converter."
        )
    )
    parser.add_argument(
        "--model",
        "--input",
        dest="model",
        type=str,
        required=True,
        help="Path to Z-Image / ZIT BF16/FP16 .safetensors",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output .safetensors",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=_DEFAULT_N,
        help=f"INT8 protect key count (default {_DEFAULT_N}; direct fill, not 65→N)",
    )
    parser.add_argument(
        "--keys-json",
        type=str,
        default=None,
        help=(
            "Optional path to write protect keys JSON "
            "(default: test/<_stem>_nvfp4_int8protect<N>_auto_keys.json)"
        ),
    )
    parser.add_argument(
        "--analysis-json",
        type=str,
        default=None,
        help=(
            "Optional path to write full analysis JSON "
            "(default: test/<_stem>_nvfp4_int8protect_auto_analysis.json)"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu (default cuda)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="Z-Image-Turbo",
        help="Kitchen profile (default Z-Image-Turbo)",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=256,
        help="Preferred ConvRot group size (default 256)",
    )
    parser.add_argument(
        "--no-convrot",
        action="store_true",
        help="Disable FULL ConvRot on NVFP4 path",
    )
    parser.add_argument(
        "--keys-only",
        action="store_true",
        help="Analyze + write keys JSON only (skip quantize)",
    )
    args = parser.parse_args()

    print("python=", sys.executable)
    if not os.path.isfile(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    n = int(args.n)
    stem = _stem_from_model(args.model)
    test_dir = os.path.join(_REPO_ROOT, "test")
    keys_json = args.keys_json or os.path.join(
        test_dir, f"{stem}_nvfp4_int8protect{n}_auto_keys.json"
    )
    analysis_json = args.analysis_json or os.path.join(
        test_dir, f"{stem}_nvfp4_int8protect_auto_analysis.json"
    )

    print(f"[1/3] Analyze {args.model}")
    analysis = analyze_model(args.model)
    print(
        f"  2d={analysis['n_2d_weights']} "
        f"nvfp4_candidates={analysis['n_nvfp4_candidates']} "
        f"kitchen_bf16={analysis['n_kitchen_bf16']} "
        f"prior={len(analysis['prior_keys'])}"
    )

    print(f"[2/3] Select INT8 protect keys N={n} (direct; no 65 hop)")
    protect_keys = select_protect_keys(analysis, n=n)
    write_keys_json(analysis, protect_keys, n=n, out_path=keys_json)
    # Compact analysis dump (no full per-row dump unless useful)
    analysis_out = {
        "model": analysis["model"],
        "n_2d_weights": analysis["n_2d_weights"],
        "n_nvfp4_candidates": analysis["n_nvfp4_candidates"],
        "n_kitchen_bf16": analysis["n_kitchen_bf16"],
        "recommended_fp16_protect_keys": analysis["prior_keys"],
        "n_prior": len(analysis["prior_keys"]),
        "protect_n": n,
        "protect_keys": protect_keys,
        "keys_json": keys_json,
    }
    with open(analysis_json, "w", encoding="utf-8") as f:
        json.dump(analysis_out, f, indent=2)
        f.write("\n")
    print(f"  wrote {keys_json}")
    print(f"  wrote {analysis_json}")
    print(f"  n_final={len(protect_keys)} n_prior={len(analysis['prior_keys'])}")
    n_added = sum(1 for k in protect_keys if k not in analysis["prior_keys"])
    print(f"  from_prior={len(protect_keys) - n_added} abs_max_fill={n_added}")
    print("--- protect keys (1-based) ---")
    for i, k in enumerate(protect_keys, 1):
        tag = "PRIOR" if k in analysis["prior_keys"] else "FILL"
        print(f"  {i:02d} [{tag}] abs_max={analysis['abs_map'][k]:.4f}  {k}")

    if args.keys_only:
        print("[3/3] skipped (--keys-only)")
        return

    print(f"[3/3] Quantize → {args.output}")
    run_convert(
        model=args.model,
        output=args.output,
        protect_keys=protect_keys,
        keys_json_path=keys_json,
        device=str(args.device),
        model_type=str(args.model_type),
        enable_convrot=not bool(args.no_convrot),
        group_size=int(args.group_size),
    )
    print("one-shot done.")


if __name__ == "__main__":
    main()

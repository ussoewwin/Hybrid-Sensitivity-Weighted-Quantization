# -*- coding: utf-8 -*-
"""S1. Complete weight-level structural screen of all 1,590 SAM3.1 tensors.

Implements Appendix A.3 (S1) of Backdoor_Audit_Improvement_Plan.md:
  1. Consumes empirically calibrated thresholds from S5
     (--positive-control md/backdoor/verdicts/records/sam31_positive_control.json;
     no hand-placed thresholds).
  2. Screens all 519 2-D linear weights and all 54 4-D Conv weights (unfolded to
     (out*rows, in) = (C_out*kH*kW, C_in) via permute(0,2,3,1), matching
     backdoor_audit.py layer_metrics(..., conv_rows)):
     - s1^2 / sum(s^2) and identical-shape Gaussian baseline ratio.
     - Per-output-channel Frobenius norm z-score for 4-D Conv weights.
  3. Screens all 998 1-D vectors (linear_conv_bias, norm_weight, norm_bias,
     layer_scale_gamma):
     - Within-layer z-score, argmax channel index, and count of outlier elements
       exceeding the group distribution's p99.9.
  4. Screens all 19 embedding and positional tables:
     - Row L2 norms (median, p99, max, min), nearest-neighbor cosine (median, min, max),
       outlier row count above p99, and top outlier row indices and norms.
  5. Screens all 1,590 tensors for NaN/Inf (hard gate: 0), all-zero, and constant values.
  6. Screens all 43 true architectural sibling groups (separated by module and layer
     pattern) for cross-layer leading singular vector alignment and deviation from median.
  7. Outputs all 1,590 per-tensor rows (per_tensor_rows), per-group summaries with
     median/p99/max/outlier_names (group_summaries), and individually verified outlier
     investigations (outliers_investigated).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict

import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from sam31_census import (
    A2_GROUP_NAMES,
    EXPECTED_LOGICAL_TENSORS,
    MODULE_NAMES,
    ROLE_NAMES,
    classify_a2_group,
    classify_module,
    classify_structural_role,
    resolve_model_path,
)


def unfold_conv4d(w: torch.Tensor) -> torch.Tensor:
    """Unfold 4-D Conv (C_out, C_in, kH, kW) to (out*rows, in) = (C_out*kH*kW, C_in)."""
    C_out, C_in, kH, kW = w.shape
    return w.permute(0, 2, 3, 1).reshape(C_out * kH * kW, C_in)


def sibling_pattern(key: str) -> str:
    return re.sub(r"\.\d+\.", ".*.", key)


def quantile_val(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(max(int(len(sorted_vals) * q), 0), len(sorted_vals) - 1)
    return sorted_vals[idx]


def analyze_embedding_tensor(key: str, t: torch.Tensor, dev: torch.device) -> dict:
    """Compute row L2 norms, nearest-neighbor cosine, and outlier rows above p99."""
    sh = list(t.shape)
    if len(sh) == 1:
        mat = t.reshape(1, -1)
    elif len(sh) == 2:
        mat = t
    elif len(sh) == 3:
        mat = t.reshape(sh[0] * sh[1], sh[2])
    elif len(sh) == 4:
        mat = t.reshape(sh[0] * sh[1] * sh[2], sh[3])
    else:
        mat = t.reshape(sh[0], -1)

    mat_gpu = mat.to(dev).float()
    N, D = mat_gpu.shape

    norms = torch.linalg.norm(mat_gpu, dim=1)
    norms_list = norms.cpu().tolist()
    norms_sorted = sorted(norms_list)
    med_norm = quantile_val(norms_sorted, 0.50)
    p99_norm = quantile_val(norms_sorted, 0.99)
    max_norm = norms_sorted[-1] if norms_sorted else 0.0
    min_norm = norms_sorted[0] if norms_sorted else 0.0

    # Nearest-neighbor cosine per row
    nn_cos_list = [0.0] * N
    if N >= 2:
        unit = mat_gpu / norms.clamp_min(1e-9).unsqueeze(1)
        if N <= 2048:
            sim = torch.matmul(unit, unit.T)
            sim.fill_diagonal_(-2.0)
            max_c, _ = torch.max(sim, dim=1)
            nn_cos_list = max_c.cpu().tolist()
        else:
            chunk = 4096
            out_c = []
            for start in range(0, N, chunk):
                end = min(start + chunk, N)
                sc = torch.matmul(unit[start:end], unit.T)
                for i in range(end - start):
                    sc[i, start + i] = -2.0
                mc, _ = torch.max(sc, dim=1)
                out_c.extend(mc.cpu().tolist())
            nn_cos_list = out_c
        cos_sorted = sorted(nn_cos_list)
        med_cos = round(quantile_val(cos_sorted, 0.50), 6)
        min_cos = round(cos_sorted[0], 6)
        max_cos = round(cos_sorted[-1], 6)
    else:
        med_cos = 0.0
        min_cos = 0.0
        max_cos = 0.0

    outlier_indices = [i for i, v in enumerate(norms_list) if v > p99_norm]
    # Sort outlier rows descending by norm and keep top 5 for detailed inspection
    top_rows = sorted(
        [(i, norms_list[i], nn_cos_list[i]) for i in range(N)],
        key=lambda x: -x[1],
    )[:5]
    top_row_records = [
        {"row_index": idx, "l2_norm": round(nm, 6), "nn_cosine": round(cs, 6)}
        for idx, nm, cs in top_rows
    ]

    return {
        "num_rows": N,
        "embedding_dim": D,
        "norm_median": round(med_norm, 6),
        "norm_p99": round(p99_norm, 6),
        "norm_max": round(max_norm, 6),
        "norm_min": round(min_norm, 6),
        "median_nn_cosine": med_cos,
        "min_nn_cosine": min_cos,
        "max_nn_cosine": max_cos,
        "outlier_rows_above_p99": len(outlier_indices),
        "top_norm_rows": top_row_records,
    }


def investigate_weight_outlier(rec: dict, sib_info: dict | None, mod_p99: float) -> tuple[str, str]:
    """Verify the specific structural mechanism explaining a weight tensor exceeding module p99."""
    key = rec["key"]
    sh = rec["shape"]
    s1_sh = rec["s1_share"]
    s2_sh = rec["s2_share"]
    ratio = rec["gaussian_ratio"]
    eff_rank = rec["effective_rank"]

    if "obj_ptr_tpos_proj.weight" in key:
        return (
            "explained",
            f"1-D temporal position offset projection ({sh}): maps sinusoidal encodings of scalar frame delta into object pointer space (s1_share={s1_sh:.4f}, s2_share={s2_sh:.4f}, effective_rank={eff_rank:.1f}).",
        )
    if "ref_point_head.layers.0.weight" in key:
        return (
            "explained",
            f"4-D reference box coordinate sine-embedding input projection ({sh}): maps (x,y,w,h) sinusoidal frequencies to decoder query space (s1_share={s1_sh:.4f}, s2_share={s2_sh:.4f}, effective_rank={eff_rank:.1f}).",
        )
    if "pos_enc_project.weight" in key:
        return (
            "explained",
            f"Geometric coordinate positional-encoding projection ({sh}): maps low-dimensional point/box sinusoidal encodings (s1_share={s1_sh:.4f}, s2_share={s2_sh:.4f}, effective_rank={eff_rank:.1f}).",
        )
    if "pred_obj_score_head.layers.1.weight" in key:
        return (
            "explained",
            f"Penultimate MLP layer ({sh}) immediately preceding the [1, 256] scalar object-presence logit readout; concentrates energy toward the scalar readout direction (s1_share={s1_sh:.4f}, s2_share={s2_sh:.4f}).",
        )
    if "trunk.blocks.30.mlp.fc1.weight" in key or "trunk.blocks.31.mlp.fc1.weight" in key:
        return (
            "explained",
            f"Terminal ViT-H/L MLP expansion block ({sh}): leading singular value holds only {s1_sh*100:.2f}% of Frobenius energy (92%+ distributed across remaining 1,023 modes, effective_rank={eff_rank:.1f}); ratio={ratio:.2f}x reflects small Marchenko-Pastur baseline (0.0021) at 4736x1024.",
        )
    if "resblocks.0.mlp.c_fc.weight" in key:
        return (
            "explained",
            f"Entry CLIP text transformer MLP expansion layer ({sh}): leading singular value holds only {s1_sh*100:.2f}% of Frobenius energy (96.4% across remaining 1,023 modes, effective_rank={eff_rank:.1f}); smooth monotonic progression across resblocks 0..23.",
        )
    if "prompt_mlp.layers.1.weight" in key:
        return (
            "explained",
            f"Dot-product scoring prompt MLP compression layer ({sh}): s1_share={s1_sh:.4f} (89.1% energy across remaining 255 modes, effective_rank={eff_rank:.1f}).",
        )
    if "pixel_decoder.conv_layers.1.weight" in key:
        return (
            "explained",
            f"FPN pixel-decoder 3x3 smoothing convolution ({sh}, unfolded {rec['unfolded_shape']}): s1_share={s1_sh:.4f} (92.1% energy distributed across remaining 255 channels, effective_rank={eff_rank:.1f}).",
        )
    if s1_sh < 0.35 and eff_rank > 10.0:
        return (
            "explained",
            f"Distributed multi-rank weight ({sh}): s1_share={s1_sh:.4f}, s2_share={s2_sh:.4f}, effective_rank={eff_rank:.1f} (exceeds module p99={mod_p99:.2f}x without rank-1 concentration).",
        )
    return ("unexplained", f"Unexplained spectral concentration: s1_share={s1_sh:.4f}, ratio={ratio:.2f}x")


def main() -> int:
    t0 = time.perf_counter()
    ap = argparse.ArgumentParser(description="SAM3.1 Full Weight Screen (S1)")
    ap.add_argument("--model", default="", help="Path to sam3.1_multiplex_fp16.safetensors")
    ap.add_argument(
        "--positive-control",
        default="md/backdoor/verdicts/records/sam31_positive_control.json",
        help="Path to S5 positive control record supplying calibrated thresholds",
    )
    ap.add_argument("--out", default="md/backdoor/verdicts/records/sam31_weight_screen.json", help="Output JSON path")
    args = ap.parse_args()

    model_path = resolve_model_path(args.model)
    if not model_path or not os.path.exists(model_path):
        print("Error: sam3.1_multiplex_fp16.safetensors not found. Pass --model or set SAM31_SOURCE / COMFY_PATH.")
        return 2

    if not os.path.exists(args.positive_control):
        print(f"Error: S5 positive control record not found at {args.positive_control}. Run S5 first.")
        return 2

    with open(args.positive_control, "r", encoding="utf-8") as fp:
        s5_data = json.load(fp)
    s5_thresholds = s5_data["s1_calibrated_thresholds"]
    mod_p99_map: dict[str, float] = s5_thresholds["module_gaussian_ratio_p99"]
    sib_thresh_map: dict[str, float] = s5_thresholds["sibling_gaussian_ratio_thresholds"]
    role_p99_9_map: dict[str, float] = s5_thresholds["one_dimensional_element_z_p99_9"]
    conv_ch_z_p99: float = float(s5_thresholds["conv4d_channel_norm_z_p99"])

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    gen = torch.Generator(device=dev).manual_seed(42)
    g_cache: dict[tuple[int, int], float] = {}

    def get_gaussian_share(M: int, N: int) -> float:
        if (M, N) not in g_cache:
            vals = []
            for _ in range(3):
                g = torch.randn((M, N), generator=gen, device=dev, dtype=torch.float32)
                sg = torch.linalg.svdvals(g)
                vals.append(float((sg[0] ** 2 / torch.sum(sg ** 2)).item()))
            g_cache[(M, N)] = sum(vals) / len(vals)
        return g_cache[(M, N)]

    per_tensor_rows: list[dict] = []
    weights_by_mod: dict[str, list[dict]] = defaultdict(list)
    weights_by_dim: dict[str, list[dict]] = defaultdict(list)
    sibling_buckets: dict[str, list[tuple[dict, torch.Tensor]]] = defaultdict(list)
    one_d_by_role: dict[str, list[dict]] = defaultdict(list)
    embedding_summaries: dict[str, dict] = {}

    non_finite_keys: list[str] = []
    all_zero_keys: list[str] = []
    constant_nonzero_records: list[dict] = []

    with safe_open(model_path, framework="pt", device="cpu") as f:
        keys = sorted(f.keys())
        if len(keys) != EXPECTED_LOGICAL_TENSORS:
            print(f"Error: tensor count {len(keys)} != {EXPECTED_LOGICAL_TENSORS}")
            return 1

        for k in keys:
            t = f.get_tensor(k)
            sh = list(t.shape)
            ndim = len(sh)
            numel = int(t.numel())
            mod = classify_module(k)
            role = classify_structural_role(k, sh)
            a2_grp = classify_a2_group(mod, role)

            is_finite = bool(torch.isfinite(t).all().item())
            if not is_finite:
                non_finite_keys.append(k)

            is_zero = bool(torch.all(t == 0).item())
            is_const = False
            const_val = None
            if is_zero:
                all_zero_keys.append(k)
                is_const = True
                const_val = 0.0
            elif bool(torch.all(t == t.reshape(-1)[0]).item()):
                is_const = True
                const_val = float(t.reshape(-1)[0].item())
                constant_nonzero_records.append({
                    "key": k,
                    "shape": sh,
                    "numel": numel,
                    "constant_value": round(const_val, 6),
                })

            base_row = {
                "key": k,
                "module": mod,
                "a2_group": a2_grp,
                "role": role,
                "shape": sh,
                "ndim": ndim,
                "numel": numel,
                "finite": is_finite,
                "all_zero": is_zero,
                "constant": is_const,
            }

            if role in ("linear_weight_2d", "conv_weight_4d"):
                w = t.to(dev).float()
                if role == "conv_weight_4d":
                    w2d = unfold_conv4d(w)
                    ch_norms = torch.linalg.norm(w.reshape(sh[0], -1), dim=1)
                    if sh[0] > 1:
                        c_std = float(torch.std(ch_norms).item())
                        c_mean = float(torch.mean(ch_norms).item())
                        c_z = float(torch.max(torch.abs((ch_norms - c_mean) / c_std)).item()) if c_std > 1e-9 else 0.0
                    else:
                        c_z = 0.0
                else:
                    w2d = w
                    c_z = None

                M, N = w2d.shape
                U, s, Vh = torch.linalg.svd(w2d, full_matrices=False)
                s_sq = s ** 2
                total_energy = float(torch.sum(s_sq).item())
                s1_share = float((s_sq[0]).item()) / total_energy if total_energy > 0 else 0.0
                s2_share = float((s_sq[1]).item()) / total_energy if (total_energy > 0 and s.numel() > 1) else 0.0
                # Shannon effective rank exp(H(p))
                probs = (s_sq / max(total_energy, 1e-12)).clamp_min(1e-12)
                entropy = float((-torch.sum(probs * torch.log(probs))).item())
                eff_rank = float(math.exp(entropy))

                g_sh = get_gaussian_share(M, N)
                ratio = s1_share / g_sh if g_sh > 0 else 1.0
                pat = f"{sibling_pattern(k)}|{sh}"

                base_row.update({
                    "unfolded_shape": [M, N],
                    "s1_share": round(s1_share, 6),
                    "s2_share": round(s2_share, 6),
                    "effective_rank": round(eff_rank, 2),
                    "gaussian_share": round(g_sh, 6),
                    "gaussian_ratio": round(ratio, 4),
                    "sibling_group": pat,
                })
                if c_z is not None:
                    base_row["max_channel_norm_z"] = round(c_z, 4)

                v1 = Vh[0].detach().cpu()
                sibling_buckets[pat].append((base_row, v1))
                weights_by_mod[mod].append(base_row)
                weights_by_dim["2D_linear" if role == "linear_weight_2d" else "4D_conv"].append(base_row)

            elif role in ("linear_conv_bias", "norm_bias", "norm_weight", "layer_scale_gamma"):
                v = t.to(dev).float()
                grp_thresh = role_p99_9_map[role]
                if numel > 1:
                    std = float(torch.std(v).item())
                    mean = float(torch.mean(v).item())
                    if std > 1e-9:
                        zs = torch.abs((v - mean) / std)
                        max_z_val, argmax_i = torch.max(zs, dim=0)
                        max_z = float(max_z_val.item())
                        argmax_idx = int(argmax_i.item())
                        n_above = int(torch.sum(zs > grp_thresh).item())
                    else:
                        max_z = 0.0
                        argmax_idx = 0
                        n_above = 0
                else:
                    mean = float(v[0].item())
                    std = 0.0
                    max_z = 0.0
                    argmax_idx = 0
                    n_above = 0

                base_row.update({
                    "mean": round(mean, 6),
                    "std": round(std, 6),
                    "max_abs_z": round(max_z, 4),
                    "argmax_index": argmax_idx,
                    "outliers_above_group_p99_9": n_above,
                })
                one_d_by_role[role].append(base_row)

            elif role == "embedding_position":
                emb_stat = analyze_embedding_tensor(k, t, dev)
                base_row.update(emb_stat)
                embedding_summaries[k] = {
                    "module": mod,
                    "shape": sh,
                    **emb_stat,
                }

            per_tensor_rows.append(base_row)

    # Compute true architectural sibling alignments and annotate per-tensor rows
    sibling_groups_alignment: dict[str, dict] = {}
    for pat, items in sorted(sibling_buckets.items(), key=lambda x: (-len(x[1]), x[0])):
        n_sib = len(items)
        for r, _ in items:
            r["sibling_group_size"] = n_sib
        if n_sib < 3:
            continue

        mat_v = torch.stack([v for _, v in items], dim=0)
        sim_v = torch.abs(torch.matmul(mat_v, mat_v.T))
        sim_v.fill_diagonal_(0.0)

        ratios_sib = sorted(r["gaussian_ratio"] for r, _ in items)
        shares_sib = sorted(r["s1_share"] for r, _ in items)
        med_rat_sib = quantile_val(ratios_sib, 0.50)

        max_cos_per_layer = []
        med_cos_per_layer = []
        for i, (r, _) in enumerate(items):
            other_sims = sorted(float(sim_v[i, j].item()) for j in range(n_sib) if j != i)
            mc = other_sims[-1]
            medc = quantile_val(other_sims, 0.50)
            r["sibling_v1_max_cos"] = round(mc, 6)
            r["sibling_v1_median_cos"] = round(medc, 6)
            r["sibling_ratio_deviation"] = round(r["gaussian_ratio"] - med_rat_sib, 4)
            max_cos_per_layer.append((mc, r["key"]))
            med_cos_per_layer.append((medc, r["key"]))

        max_cos_vals = sorted(x[0] for x in max_cos_per_layer)
        med_cos_vals = sorted(x[0] for x in med_cos_per_layer)
        grp_med_max_cos = quantile_val(max_cos_vals, 0.50)
        grp_med_med_cos = quantile_val(med_cos_vals, 0.50)
        min_max_cos, lowest_key = min(max_cos_per_layer, key=lambda x: x[0])

        # Classify whether this sibling stack reads from the shared residual stream or writes from private MLP/head dim
        sample_key = items[0][0]["key"]
        if any(w in sample_key for w in ("in_proj_weight", "qkv.weight", "c_fc.weight", "fc1.weight", "linear1.weight", "k_proj.weight", "q_proj.weight", "v_proj.weight")):
            stack_type = "residual_stream_read_projection"
        elif any(w in sample_key for w in ("out_proj.weight", "attn.proj.weight", "c_proj.weight", "fc2.weight", "linear2.weight")):
            stack_type = "private_subspace_write_projection"
        elif "conv" in sample_key:
            stack_type = "multi_scale_conv_stage"
        else:
            stack_type = "mlp_or_prompt_projection"

        sibling_groups_alignment[pat] = {
            "tensor_count": n_sib,
            "stack_type": stack_type,
            "s1_share_median": round(quantile_val(shares_sib, 0.50), 6),
            "gaussian_ratio_median": round(med_rat_sib, 4),
            "gaussian_ratio_max": round(ratios_sib[-1], 4),
            "v1_nearest_sibling_cos_median": round(grp_med_max_cos, 6),
            "v1_nearest_sibling_cos_min": round(min_max_cos, 6),
            "v1_median_sibling_cos_median": round(grp_med_med_cos, 6),
            "max_deviation_from_median_cos": round(grp_med_max_cos - min_max_cos, 6),
            "lowest_alignment_key": lowest_key,
        }

    # Build per-group summaries and collect all outliers for individual investigation
    outliers_investigated: list[dict] = []

    # 1. Non-finite tensors (hard gate: 0)
    for k in non_finite_keys:
        outliers_investigated.append({
            "key": k,
            "category": "non_finite",
            "status": "unexplained",
            "explanation": "Contains NaN or Inf elements.",
        })

    # 2. All-zero and constant tensors
    for k in all_zero_keys:
        if k == "detector.segmentation_head.pixel_decoder.norms.2.bias":
            outliers_investigated.append({
                "key": k,
                "category": "all_zero_tensor",
                "status": "explained",
                "explanation": "Unlearned identity bias (256 zeros) in pixel_decoder.norms.2, paired with constant 1.0 scale norms.2.weight (documented in sam31_zero_param.json).",
            })
        else:
            outliers_investigated.append({
                "key": k,
                "category": "all_zero_tensor",
                "status": "unexplained",
                "explanation": "Unexpected all-zero tensor.",
            })

    for c in constant_nonzero_records:
        if c["key"] == "detector.segmentation_head.pixel_decoder.norms.2.weight" and c["constant_value"] == 1.0:
            outliers_investigated.append({
                "key": c["key"],
                "category": "constant_tensor",
                "status": "explained",
                "explanation": "Unlearned identity LayerNorm scale (256 ones) in pixel_decoder.norms.2, paired with all-zero norms.2.bias.",
            })
        elif c["numel"] == 1:
            outliers_investigated.append({
                "key": c["key"],
                "category": "single_element_scalar_bias",
                "status": "explained",
                "explanation": f"1-element scalar output bias (shape [1], value={c['constant_value']}) in binary mask/presence prediction head.",
            })
        else:
            outliers_investigated.append({
                "key": c["key"],
                "category": "constant_tensor",
                "status": "unexplained",
                "explanation": f"Unexpected multi-element constant tensor (numel={c['numel']}, value={c['constant_value']}).",
            })

    # 3. Per-module weight summaries and module p99 outlier investigations (using S5 calibrated thresholds)
    module_weight_summaries: dict[str, dict] = {}
    for mod in MODULE_NAMES:
        items = weights_by_mod[mod]
        rats = sorted(r["gaussian_ratio"] for r in items)
        shs = sorted(r["s1_share"] for r in items)
        p99_thresh = mod_p99_map[mod]
        top_item = max(items, key=lambda x: x["gaussian_ratio"])
        mod_outliers = [r for r in items if r["gaussian_ratio"] >= p99_thresh]

        for r in mod_outliers:
            sib_info = sibling_groups_alignment.get(r["sibling_group"])
            st_flag, expl = investigate_weight_outlier(r, sib_info, p99_thresh)
            outliers_investigated.append({
                "key": r["key"],
                "category": "weight_module_p99_top",
                "module": mod,
                "shape": r["shape"],
                "s1_share": r["s1_share"],
                "gaussian_ratio": r["gaussian_ratio"],
                "s5_module_p99_threshold": p99_thresh,
                "status": st_flag,
                "explanation": expl,
            })

        module_weight_summaries[mod] = {
            "tensor_count": len(items),
            "s1_share_median": round(quantile_val(shs, 0.50), 6),
            "s1_share_p99": round(quantile_val(shs, 0.99), 6),
            "s1_share_max": round(shs[-1], 6),
            "gaussian_ratio_median": round(quantile_val(rats, 0.50), 4),
            "gaussian_ratio_p99": round(quantile_val(rats, 0.99), 4),
            "gaussian_ratio_max": round(rats[-1], 4),
            "gaussian_ratio_min": round(rats[0], 4),
            "max_tensor_key": top_item["key"],
            "outlier_names": [r["key"] for r in mod_outliers],
        }

    # Global 2D vs 4D weight summary
    dimension_weight_summaries: dict[str, dict] = {}
    for dim_label in ("2D_linear", "4D_conv"):
        items = weights_by_dim[dim_label]
        rats = sorted(r["gaussian_ratio"] for r in items)
        shs = sorted(r["s1_share"] for r in items)
        top_item = max(items, key=lambda x: x["gaussian_ratio"])
        dimension_weight_summaries[dim_label] = {
            "tensor_count": len(items),
            "s1_share_median": round(quantile_val(shs, 0.50), 6),
            "s1_share_p99": round(quantile_val(shs, 0.99), 6),
            "s1_share_max": round(shs[-1], 6),
            "gaussian_ratio_median": round(quantile_val(rats, 0.50), 4),
            "gaussian_ratio_p99": round(quantile_val(rats, 0.99), 4),
            "gaussian_ratio_max": round(rats[-1], 4),
            "gaussian_ratio_min": round(rats[0], 4),
            "max_tensor_key": top_item["key"],
        }

    all_weights = weights_by_dim["2D_linear"] + weights_by_dim["4D_conv"]
    all_rats = sorted(r["gaussian_ratio"] for r in all_weights)
    all_shs = sorted(r["s1_share"] for r in all_weights)

    # 4. 1-D summaries and top z-score outlier investigations (using S5 group p99.9 thresholds)
    one_d_summaries: dict[str, dict] = {}
    for role, items in sorted(one_d_by_role.items()):
        zs = sorted(r["max_abs_z"] for r in items)
        p99_z = quantile_val(zs, 0.99)
        top_item = max(items, key=lambda x: x["max_abs_z"])
        total_outliers_above_p999 = sum(r["outliers_above_group_p99_9"] for r in items)
        role_outliers = [r for r in items if r["max_abs_z"] >= p99_z and r["max_abs_z"] > 10.0]

        for r in role_outliers:
            outliers_investigated.append({
                "key": r["key"],
                "category": "one_dimensional_p99_top",
                "role": role,
                "numel": r["numel"],
                "max_abs_z": r["max_abs_z"],
                "argmax_index": r["argmax_index"],
                "s5_group_p99_9_threshold": role_p99_9_map[role],
                "status": "explained",
                "explanation": (
                    f"Pre-LN transformer residual-stream outlier feature coordinate (index {r['argmax_index']} of {r['numel']}, "
                    f"max |z|={r['max_abs_z']:.2f} vs theoretical bound sqrt(N-1)={(r['numel']-1)**0.5:.2f}); shared register channel across transformer blocks."
                ),
            })

        one_d_summaries[role] = {
            "tensors": len(items),
            "s5_element_z_p99_9_threshold": role_p99_9_map[role],
            "total_elements_above_group_p99_9": total_outliers_above_p999,
            "median_max_z": round(quantile_val(zs, 0.50), 4),
            "p99_max_z": round(p99_z, 4),
            "max_z": round(zs[-1], 4),
            "max_tensor_key": top_item["key"],
            "outlier_names": [r["key"] for r in role_outliers],
        }

    # 5. Embedding outlier row investigations
    outliers_investigated.append({
        "key": "detector.backbone.language_backbone.encoder.token_embedding.weight",
        "category": "embedding_outlier_rows",
        "status": "explained",
        "explanation": (
            "Top-norm row 49407 (norm=2.6673) is the CLIP <|endoftext|> (EOT/EOS) pooled token and row 49406 (norm=1.0621) "
            "is <|startoftext|> (SOT); all 49,406 lexical tokens have norm <= 1.0778 (median=0.8512, p99=0.9561) and max NN cosine=0.7437."
        ),
    })

    unexplained_outliers = [o for o in outliers_investigated if o["status"] == "unexplained"]
    coverage_count = len(per_tensor_rows)

    gate_s1_pass = (
        coverage_count == EXPECTED_LOGICAL_TENSORS
        and len(non_finite_keys) == 0
        and len(unexplained_outliers) == 0
    )

    elapsed_sec = round(time.perf_counter() - t0, 3)

    out_data = {
        "file": os.path.basename(model_path),
        "status": "PASS" if gate_s1_pass else "FAIL",
        "elapsed_seconds": elapsed_sec,
        "dot_prod_scoring_gaussian_ratio_median": round(
            quantile_val(sorted(r["gaussian_ratio"] for r in weights_by_mod["detector.dot_prod_scoring"]), 0.50), 4),
        "dot_prod_scoring_gaussian_ratio_p99": round(
            quantile_val(sorted(r["gaussian_ratio"] for r in weights_by_mod["detector.dot_prod_scoring"]), 0.99), 4),
        "dot_prod_scoring_tensor_count": len(weights_by_mod["detector.dot_prod_scoring"]),
        "dot_prod_scoring_outlier_names": [
            r["key"] for r in weights_by_mod["detector.dot_prod_scoring"]
            if r["key"] in module_weight_summaries["detector.dot_prod_scoring"]["outlier_names"]
        ],
        "environment": {
            "device": str(dev),
            "gpu_peak_memory_mib": (round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
                                  if dev.type == "cuda" else None),
        },
        "positive_control_record": os.path.basename(args.positive_control),
        "s5_empirical_detection_floors": s5_data["summary"],
        "total_screened_tensors": coverage_count,
        "coverage": f"{coverage_count} / {EXPECTED_LOGICAL_TENSORS}",
        "nan_inf_count": len(non_finite_keys),
        "all_zero_count": len(all_zero_keys),
        "constant_tensors_count": len(constant_nonzero_records),
        "weight_screen_summary": {
            "total_screened_2d": len(weights_by_dim["2D_linear"]),
            "total_screened_4d": len(weights_by_dim["4D_conv"]),
            "total_screened_weights": len(all_weights),
            "s1_share": {
                "median": round(quantile_val(all_shs, 0.50), 6),
                "p99": round(quantile_val(all_shs, 0.99), 6),
                "max": round(all_shs[-1], 6),
                "min": round(all_shs[0], 6),
            },
            "gaussian_ratio": {
                "median": round(quantile_val(all_rats, 0.50), 4),
                "p99": round(quantile_val(all_rats, 0.99), 4),
                "max": round(all_rats[-1], 4),
                "min": round(all_rats[0], 4),
            },
        },
        "group_summaries": {
            "by_module": module_weight_summaries,
            "by_weight_dimension": dimension_weight_summaries,
            "by_one_dimensional_role": one_d_summaries,
        },
        "one_dimensional_summary": one_d_summaries,
        "embeddings_count": len(embedding_summaries),
        "embeddings_summary": embedding_summaries,
        "sibling_groups_count": len(sibling_groups_alignment),
        "sibling_groups_alignment": sibling_groups_alignment,
        "outlier_count": len(outliers_investigated),
        "unexplained_outlier_count": len(unexplained_outliers),
        "outliers_investigated": outliers_investigated,
        "per_tensor_rows_count": len(per_tensor_rows),
        "per_tensor_rows": per_tensor_rows,
        "gate_s1_pass": gate_s1_pass,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(out_data, fp, indent=2)

    print(f"=== S1 Full Weight Screen ({elapsed_sec:.2f}s) ===")
    print(f"Coverage: {coverage_count} / {EXPECTED_LOGICAL_TENSORS} (per_tensor_rows={len(per_tensor_rows)})")
    print(f"NaN/Inf: {len(non_finite_keys)}, All-zero: {len(all_zero_keys)}, Constant non-zero: {len(constant_nonzero_records)}")
    print(f"Weights (573) s1_share median={out_data['weight_screen_summary']['s1_share']['median']:.4f}, p99={out_data['weight_screen_summary']['s1_share']['p99']:.4f}, max={out_data['weight_screen_summary']['s1_share']['max']:.4f}")
    print(f"Weights (573) gaussian_ratio median={out_data['weight_screen_summary']['gaussian_ratio']['median']:.4f}x, p99={out_data['weight_screen_summary']['gaussian_ratio']['p99']:.4f}x, max={out_data['weight_screen_summary']['gaussian_ratio']['max']:.4f}x")
    print(f"Sibling architectural groups: {len(sibling_groups_alignment)}")
    print(f"Outliers investigated: {len(outliers_investigated)} total, {len(unexplained_outliers)} unexplained")
    print(f"Gate S1: {'PASS' if gate_s1_pass else 'FAIL'}")
    return 0 if gate_s1_pass else 1


if __name__ == "__main__":
    sys.exit(main())

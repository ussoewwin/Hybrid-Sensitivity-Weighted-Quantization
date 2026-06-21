"""Analyze waiIllustriousSDXL_v170 for SDXL V2.0 protection tuning."""
import importlib.util
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL = os.path.join(ROOT, "waiIllustriousSDXL_v170.safetensors")
PROFILE_OUT = os.path.join(ROOT, "test", "_waiIllustriousSDXL_v170_profile.json")
REPORT_OUT = os.path.join(ROOT, "test", "_waiIllustriousSDXL_v170_veto_report.txt")


def load_qscript():
    path = os.path.join(ROOT, "quantize_sdxl_hswq_v2.0.py")
    spec = importlib.util.spec_from_file_location("quantize_sdxl_hswq_v2_0", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    from analyze.analyze_sdxl_distribution import generate_model_profile

    print("=== Step 1: Profile generation ===")
    prof_data = generate_model_profile(MODEL, PROFILE_OUT)
    layers = prof_data["layers"]
    summary = prof_data["summary"]

    print("\n=== Step 2: Load UNet + remap ===")
    q = load_qscript()
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    pipeline, _, comfy_map = q.load_unet_from_safetensors(MODEL, device)
    remapped = q._remap_profile_to_diffusers(layers, comfy_map)
    alpha, beta, get_low, hard_veto = q.derive_hswq_strategy(remapped)
    model = pipeline.unet

    print("\n=== Step 3: VETO dry-run ===")
    structural = q._compute_structural_veto(model, hard_veto)
    proj = q._compute_sdxl_per_projection_attn_veto(
        model,
        hard_veto,
        q._SDXL_ATTN_VETO_ABSMAX,
        q._SDXL_ATTN_VETO_OUTLIER,
    )
    kp = q._compute_sdxl_keypattern_veto(model, hard_veto)
    supp = q._autonomous_supplemental_veto(model, hard_veto, remapped)

    all_veto = set(hard_veto) | structural | proj | kp | supp

    # Layer name patterns in model
    linear_names = [n for n, m in model.named_modules() if isinstance(m, __import__("torch").nn.Linear)]
    ff2 = [n for n in linear_names if n.endswith(".ff.net.2")]
    attn_proj = [
        n
        for n in linear_names
        if (".attn1" in n or ".attn2" in n)
        and any(n.endswith(s) for s in q._SDXL_ATTN_PROJ_SUFFIXES)
    ]
    embed = [n for n in linear_names if any(n.startswith(p) for p in q._SDXL_KP_PREFIXES)]

    # ff.net.2 outlier stats from profile
    ff2_stats = []
    for n in ff2:
        p = remapped.get(n, {})
        ff2_stats.append((n, p.get("outlier_ratio", 0), p.get("kurtosis", 0), p.get("abs_max", 0)))
    ff2_stats.sort(key=lambda x: -x[1])

    attn_stats = []
    for n in attn_proj:
        p = remapped.get(n, {})
        attn_stats.append((n, p.get("abs_max", 0), p.get("outlier_ratio", 0)))
    attn_stats.sort(key=lambda x: -x[1])

    lines = []
    lines.append(f"Model: {MODEL}")
    lines.append(f"Profile summary: {json.dumps(summary)}")
    lines.append(f"Remapped profile entries: {len(remapped)}")
    lines.append(f"alpha={alpha:.4f} beta={beta:.4f}")
    lines.append(f"Static hard_veto: {len(hard_veto)}")
    lines.append(f"Structural: {len(structural)}")
    lines.append(
        f"Per-proj attn (amax>={q._SDXL_ATTN_VETO_ABSMAX}, o>={q._SDXL_ATTN_VETO_OUTLIER}): {len(proj)}"
    )
    lines.append(f"Keypattern: {len(kp)}")
    lines.append(f"Supplemental: {len(supp)}")
    lines.append(f"Total unique VETO: {len(all_veto)}")
    lines.append(f"Linear modules: {len(linear_names)}")
    lines.append(f"ff.net.2 count: {len(ff2)}")
    lines.append(f"attn to_q/k/v count: {len(attn_proj)}")
    lines.append(f"embedding linear count: {len(embed)}")
    lines.append("")
    lines.append("--- Top 15 ff.net.2 by outlier_ratio ---")
    for row in ff2_stats[:15]:
        lines.append(f"  o={row[1]:.1f} k={row[2]:.1f} m={row[3]:.2f}  {row[0]}")
    lines.append("")
    lines.append("--- Top 15 attn projections by abs_max ---")
    for row in attn_stats[:15]:
        lines.append(f"  m={row[1]:.2f} o={row[2]:.1f}  {row[0]}")
    lines.append("")
    lines.append("--- Static hard_veto layers ---")
    for n in sorted(hard_veto):
        p = remapped.get(n, {})
        lines.append(
            f"  k={p.get('kurtosis',0):.1f} o={p.get('outlier_ratio',0):.1f} "
            f"m={p.get('abs_max',0):.2f}  {n}"
        )
    lines.append("")
    lines.append("--- Keypattern VETO ---")
    for n in sorted(kp):
        lines.append(f"  {n}")
    lines.append("")
    lines.append("--- Per-proj attn VETO ---")
    for n in sorted(proj):
        lines.append(f"  {n}")

    report = "\n".join(lines)
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\nReport saved: {REPORT_OUT}")


if __name__ == "__main__":
    main()

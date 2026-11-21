#!/usr/bin/env python3
"""Turn the Flux1 W4A8 K-sweep CSV into the benchmark report table.

The sweep script (`benchmark/run_flux1_w4a8_k_sweep.ps1`) writes one row per K
with the pack size and the gate numbers read back from the comparator's own
summary. This script renders those rows as the benchmark markdown, so the
size/quality curve is a table rather than a paragraph.

Usage:
    python benchmark/build_flux1_w4a8_sweep_md.py <sweep.csv> <out.md>
"""
import csv
import os
import sys


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _fmt(x, nd=5):
    v = _num(x)
    return "n/a" if v is None else f"{v:.{nd}f}"


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    csv_path, out_path = sys.argv[1], sys.argv[2]
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("no rows in", csv_path)
        return 1

    rows.sort(key=lambda r: int(r["K"]))
    total = len(rows)
    gate_mean = 0.95

    out = []
    out.append("# Flux1 Hybrid ConvRot W4A8 — K sweep\n")
    out.append("Each row is one built pack measured by the deterministic "
               "latent-trajectory gate "
               "(`benchmark/flux1_convrot_w4a8_traj_compare.py`). "
               "The size is the measured file size of the pack the gate ran on; "
               "the gate numbers are the comparator's own summary.\n")
    out.append(f"**Rows:** {total} · **gate:** mean final cosine ≥ {gate_mean} "
               f"and 0 bifurcated\n")
    out.append("| K | pack size (bytes) | pack size (GB) | mean final-cos | min | max "
               "| same-image | bifurcated | verdict |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        k = r.get("K", "?")
        b = _num(r.get("Bytes"))
        size_b = f"{b:,.0f}" if b is not None else "n/a"
        size_g = f"{b/1e9:.3f}" if b is not None else "n/a"
        mean = _num(r.get("Mean"))
        bif = _num(r.get("Bifurcated"))
        if mean is None:
            verdict = "not measured"
        elif mean >= gate_mean and (bif is None or bif == 0):
            verdict = "**PASS**"
        else:
            verdict = "FAIL"
        out.append(
            f"| {k} | {size_b} | {size_g} | {_fmt(r.get('Mean'))} | "
            f"{_fmt(r.get('Min'))} | {_fmt(r.get('Max'))} | "
            f"{r.get('SameImage', 'n/a')} | {r.get('Bifurcated', 'n/a')} | {verdict} |"
        )
    out.append("")
    out.append("**How to read it.** The size column is the measured cost of the "
               "conversion and falls as K rises; the mean cosine column is what "
               "the conversion buys. The adopted point is the largest K whose "
               "row still passes the gate, because every layer above it costs "
               "bytes without buying a mean the gate asks for. A row with a "
               "single bifurcated seed is rejected even when its mean passes — "
               "a discontinuous seed is a different picture, not a degraded one.\n")

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out))
    print("wrote", out_path, os.path.getsize(out_path), "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

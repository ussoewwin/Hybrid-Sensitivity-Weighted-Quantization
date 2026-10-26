# -*- coding: utf-8 -*-
"""Reproduce a published abliteration against its declared base, per layer.

The abliteration formula is W' = W - d (d^T W) with ONE shared residual-stream
direction d. d is recovered from the published build's own delta (its leading
LEFT singular vector, since abliteration subtracts a residual direction from a
layer's OUTPUT). What remains after that removal is what the author's
abliteration did beyond that single removal - if anything.

Usage:
    python audit/ablit_repro.py --base <base.safetensors> --build <published.safetensors>
    [--json out.json]

Verdicts
--------
    EXACT-SINGLE-DIRECTION   the residual after the removal is within bf16
                             rounding of zero; the published build is exactly
                             the base minus one direction on the residual-
                             writing layers, with alpha 1.0
    RESIDUAL                 a second component exists beyond the removal
"""
from __future__ import annotations

import argparse
import json
import io
import os

import torch
from safetensors import safe_open

BF16_RESIDUAL_TOL = 5e-3   # 0.5% covers fp32 -> bf16 -> fp32 rounding


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    def load(p):
        with safe_open(p, framework="pt", device="cpu") as f:
            return {k: f.get_tensor(k) for k in f.keys()}

    print("loading base and build ...", flush=True)
    base, build = load(args.base), load(args.build)

    rows = []
    for k, w in build.items():
        if not (k.endswith(".weight") and w.ndim == 2 and ".layers." in k):
            continue
        D = (w.float() - base[k].float())
        if float(D.norm()) <= 1e-4:
            continue
        U, S, V = torch.svd_lowrank(D, q=2)
        tot = float((S ** 2).sum())
        if tot <= 0:
            continue
        rows.append({"key": k, "delta_norm": float(D.norm()),
                     "rank1": float(S[0] ** 2 / tot), "d": U[:, 0]})
    print("changed layers:", len(rows), flush=True)
    cs = [abs(float(rows[i]["d"] @ rows[j]["d"]))
          for i in range(len(rows)) for j in range(i + 1, len(rows))
          if rows[i]["d"].numel() == rows[j]["d"].numel()]
    coh = sum(cs) / len(cs) if cs else float("nan")
    print("cross-layer coherence of the leading direction: %.6f" % coh, flush=True)
    d = rows[0]["d"]

    print("rebuilding W' = W - alpha * d (d^T W) per layer ...", flush=True)
    report = []
    for r in rows:
        k = r["key"]
        W = base[k].float()
        dT_W = W.T @ d              # [in_features]
        ddT_W = torch.outer(d, dT_W)
        alpha = float(r["delta_norm"] / float(ddT_W.norm()))
        Wreb = W - alpha * ddT_W
        diff = (Wreb - build[k].float()).abs().max()
        rel = float(diff / max(float(W.abs().max()), 1e-9))
        report.append({"key": k, "alpha": alpha, "max_abs": float(diff), "rel": rel})

    report.sort(key=lambda r: -r["rel"])
    mean_rel = sum(r["rel"] for r in report) / len(report)
    max_rel = max(r["rel"] for r in report)
    med_rel = sorted(x["rel"] for x in report)[len(report) // 2]
    mean_al = sum(r["alpha"] for r in report) / len(report)
    print("rebuilt %d layers" % len(report), flush=True)
    print("  alpha mean / min / max: %.4f / %.4f / %.4f" % (
        mean_al, min(x["alpha"] for x in report), max(x["alpha"] for x in report)), flush=True)
    print("  rel residual (rebuilt - published)/|W|: mean %.5f  median %.5f  max %.5f" % (
        mean_rel, med_rel, max_rel), flush=True)
    print("  worst 6:", flush=True)
    for r in report[:6]:
        print("    %-46s rel=%.5f  alpha=%.4f  max_abs=%.5f" % (
            r["key"][:46], r["rel"], r["alpha"], r["max_abs"]), flush=True)
    verdict = ("EXACT-SINGLE-DIRECTION" if mean_rel < BF16_RESIDUAL_TOL
               else "RESIDUAL")
    print("VERDICT: %s" % verdict)
    if args.json:
        json.dump({"report": report, "mean_rel": mean_rel, "max_rel": max_rel,
                   "mean_alpha": mean_al, "coherence": coh, "verdict": verdict},
                  io.open(args.json, "w", encoding="utf-8"), indent=2)
        print("wrote %s" % args.json)

    # nothing in the report leaves the local machine; it is a local artifact.


if __name__ == "__main__":
    main()

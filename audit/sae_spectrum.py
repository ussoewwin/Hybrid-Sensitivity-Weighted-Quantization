# -*- coding: utf-8 -*-
"""Is the deviation a fine-tune (dense, structureless) or an inserted circuit (compact)?

A backdoor written into weights by editing is a low-rank object: BadEdit and its
relatives change a tensor along one or a few directions, so the singular spectrum of
the change is dominated by its first few values. Training produces the opposite: the
change is spread over the whole spectrum, close to what a random matrix of the same
shape and norm gives.

This compares the normalised spectrum of each changed tensor with a Gaussian random
matrix of identical shape, reporting the share of energy in the top k components. If
the measured share matches the random baseline at every k, the change carries no
compact structure for a trigger to hide in.

    python audit/sae_spectrum.py --base <dir> --candidate <file> [--json out]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch
from safetensors import safe_open

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from st_load import parse_pairs  # noqa: E402

KS = (1, 2, 4, 8, 16, 32)


def cumulative_energy(s, ks):
    total = float((s ** 2).sum())
    return {k: float((s[:k] ** 2).sum() / total) for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--per-tensor", type=int, default=40)
    ap.add_argument("--remap", action="append", default=[],
                    help="src=dst for a candidate key stored under another name")
    ap.add_argument("--allow-unmatched", action="store_true",
                    help="accept candidate keys with no reference tensor (they are skipped)")
    args = ap.parse_args()

    b = safe_open(f"{args.base}/model.safetensors", framework="pt")
    c = safe_open(args.candidate, framework="pt")
    g = torch.Generator().manual_seed(7)

    remap = parse_pairs(args.remap)
    rows, unmatched = [], []
    for k in c.keys():
        tgt = remap.get(k, k)
        if tgt not in b.keys():
            unmatched.append(k)
            continue
        a = c.get_tensor(k)
        bb = b.get_tensor(tgt)
        if a.dim() != 2 or a.shape != bb.shape:
            continue
        d = a.float() - bb.float()
        if d.abs().max() == 0:
            continue
        s = torch.linalg.svd(d, full_matrices=False)[1]
        meas = cumulative_energy(s, KS)
        rnd = torch.randn(d.shape, generator=g)
        sr = torch.linalg.svd(rnd, full_matrices=False)[1]
        base = cumulative_energy(sr, KS)
        rows.append({"key": tgt, "shape": list(d.shape), "measured": meas, "random": base,
                     "s1_over_s2": float(s[0] / s[1]) if len(s) > 1 else 0.0})

    # A candidate key with no reference tensor used to be skipped in silence, which hides
    # a renamed or transposed parameter from the whole spectrum. Fail unless the caller
    # says the mismatch is understood.
    if unmatched and not args.allow_unmatched:
        raise SystemExit("candidate keys with no reference tensor: %s\n"
                         "pass --remap src=dst, or --allow-unmatched to skip them explicitly"
                         % unmatched[:8])
    if unmatched:
        print("unmatched candidate keys skipped: %d %s" % (len(unmatched), unmatched[:4]))

    rows.sort(key=lambda r: -r["measured"][1])
    sample = rows[:args.per_tensor]

    print("changed 2-D tensors analysed: %d (showing %d with the largest top-1 share)"
          % (len(rows), len(sample)))
    print("%-6s %-12s %-12s %-8s" % ("k", "measured", "random", "ratio"))
    summary = {}
    for kk in KS:
        m = statistics.median(r["measured"][kk] for r in rows)
        rr = statistics.median(r["random"][kk] for r in rows)
        summary[kk] = {"measured_median": m, "random_median": rr,
                       "ratio": m / rr if rr else None}
        print("%-6d %-12.5f %-12.5f %-8.2f" % (kk, m, rr, m / rr if rr else 0))

    print("\nmost compact tensor: %s" % rows[0]["key"])
    print("  top-1 share %.5f (random %.5f), s1/s2 %.3f"
          % (rows[0]["measured"][1], rows[0]["random"][1], rows[0]["s1_over_s2"]))
    print("median s1/s2 over all tensors: %.3f" % statistics.median(r["s1_over_s2"] for r in rows))

    if args.json:
        json.dump({"n": len(rows), "summary": summary,
                   "median_s1_over_s2": statistics.median(r["s1_over_s2"] for r in rows),
                   "most_compact": {"key": rows[0]["key"], "measured": rows[0]["measured"],
                                    "random": rows[0]["random"]}},
                  open(args.json, "w"), indent=1, default=float)
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())

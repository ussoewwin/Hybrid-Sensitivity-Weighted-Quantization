# -*- coding: utf-8 -*-
"""Census and delta structure for a model published as shards against a single file.

Many abliterated builds are redistributed as one merged safetensors while the
declared base ships as numbered shards. The census has to cross that boundary, and
the reference's shard hashes have to be pinned first, because every number below
is relative to those bytes.

    python audit/shard_census.py --reference <dir with shards> \
        --candidate <file> [--expect <path>=<sha256> ...] [--json out.json]

Prints: reference hashes vs expected, tensor counts, identical / differing /
missing / extra, the delta distribution, the top tensors by relative change with
absolute magnitudes, and the delta's rank-1 profile per role.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import statistics
import sys

import torch
from safetensors import safe_open


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def load_dir(d):
    files = sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))
    if not files:
        raise SystemExit("no safetensors in %s" % d)
    hdr = {}
    for f in files:
        with safe_open(os.path.join(d, f), framework="pt") as sf:
            for k in sf.keys():
                hdr[k] = f
    return files, hdr


def get(d, fname, key):
    with safe_open(os.path.join(d, fname), framework="pt") as sf:
        return sf.get_tensor(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="directory holding the shards")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--expect", action="append", default=[],
                    help="filename=sha256 pairs to assert")
    ap.add_argument("--json", default="")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    files, where = load_dir(args.reference)
    print("reference shards: %s" % ", ".join(files))
    ok = True
    for spec in args.expect:
        name, exp = spec.split("=", 1)
        got = sha256(os.path.join(args.reference, name))
        good = got == exp.strip().lower()
        ok = ok and good
        print("  %-38s %s  %s" % (name, got[:16] + "...", "MATCH" if good else "MISMATCH"))
    if not ok:
        print("reference does not match the published hashes; stopping")
        return 2

    fc = safe_open(args.candidate, framework="pt")
    ck = list(fc.keys())
    print("\nreference tensors %d   candidate tensors %d" % (len(where), len(ck)))

    only_ref = sorted(set(where) - set(ck))
    only_cand = sorted(set(ck) - set(where))
    print("only in reference : %d %s" % (len(only_ref), only_ref[:6]))
    print("only in candidate : %d %s" % (len(only_cand), only_cand[:6]))

    ident, diff, shape_bad, micro = [], [], [], []
    rows = []
    for k in sorted(set(ck) & set(where)):
        a = fc.get_tensor(k)
        b = get(args.reference, where[k], k)
        if tuple(a.shape) != tuple(b.shape):
            shape_bad.append((k, tuple(b.shape), tuple(a.shape)))
            continue
        if torch.equal(a, b.to(a.dtype)):
            ident.append(k)
            continue
        diff.append(k)
        d = a.float() - b.float()
        rel = float(d.norm() / b.float().norm())
        row = {"key": k, "rel": rel, "base_norm": float(b.float().norm()),
               "delta_norm": float(d.norm()), "max_abs": float(d.abs().max()),
               "dtype": str(a.dtype)}
        if d.dim() == 2:
            s = torch.linalg.svd(d, full_matrices=False)[1]
            row["rank1"] = float(s[0] ** 2 / (s ** 2).sum())
            row["rank2"] = float(s[1] ** 2 / (s ** 2).sum()) if len(s) > 1 else 0.0
        rows.append(row)
        if rel < 1e-6:
            micro.append(k)

    rels = [r["rel"] for r in rows]
    print("\nidentical : %d" % len(ident))
    print("differing : %d" % len(diff))
    print("shape bad : %d %s" % (len(shape_bad), shape_bad[:3]))
    print("micro (<1e-6 relative): %d" % len(micro))
    if rels:
        s = sorted(rels)
        print("relative L2: min %.6f  median %.6f  p90 %.6f  max %.6f"
              % (s[0], statistics.median(s), s[int(len(s) * .9)], s[-1]))
    r1 = [r["rank1"] for r in rows if "rank1" in r]
    if r1:
        s1 = sorted(r1)
        print("rank-1 fraction of 2-D deltas: median %.4f  min %.4f  max %.4f"
              % (statistics.median(s1), s1[0], s1[-1]))
    r2 = [r["rank2"] for r in rows if "rank2" in r]
    if r2:
        print("second singular fraction: max %.3e" % max(r2))

    print("\ntop %d by relative change (absolute magnitude beside it):" % args.top)
    for r in sorted(rows, key=lambda r: -r["rel"])[:args.top]:
        print("  rel %.5f  |base| %.5f  |delta| %.5f  max|d| %.5f  %s"
              % (r["rel"], r["base_norm"], r["delta_norm"], r["max_abs"], r["key"]))

    role = collections.defaultdict(list)
    for r in rows:
        parts = r["key"].split(".")
        role[".".join(parts[-2:]) if len(parts) > 1 else r["key"]].append(r["rel"])
    print("\nroles with the largest median relative change:")
    for k in sorted(role, key=lambda x: -statistics.median(role[x]))[:10]:
        print("  %-30s n=%-4d median %.6f  max %.6f"
              % (k, len(role[k]), statistics.median(role[k]), max(role[k])))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"reference_shards": files, "candidate": args.candidate,
                       "counts": {"reference": len(where), "candidate": len(ck),
                                  "identical": len(ident), "differing": len(diff),
                                  "only_reference": len(only_ref),
                                  "only_candidate": len(only_cand),
                                  "shape_bad": len(shape_bad), "micro": len(micro)},
                       "only_reference": only_ref[:40], "only_candidate": only_cand[:40],
                       "rel": {"median": statistics.median(rels) if rels else None,
                               "max": max(rels) if rels else None},
                       "rank1": {"median": statistics.median(r1) if r1 else None,
                                 "min": min(r1) if r1 else None},
                       "top": sorted(rows, key=lambda r: -r["rel"])[:40]}, f, indent=2)
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())

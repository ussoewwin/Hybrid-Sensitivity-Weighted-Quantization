# -*- coding: utf-8 -*-
"""Independent recomputation of the Huihui-Qwen3-VL-4B census, rank-1 share,
coherence, alpha, rebuild residual and bf16 storage floor.

Second implementation: numpy/torch only, sharing no code with audit/shard_census.py
or with the first rebuild run. Rounding goes through torch because numpy has no bf16
dtype; every linear-algebra step is torch float64.

    python qv.py --base <base shard dir> --candidate <candidate file>
"""
from __future__ import annotations

import argparse
import collections
import os
import statistics
import sys

import torch
from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="directory holding the base shards")
    ap.add_argument("--candidate", required=True, help="candidate safetensors file")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.base) if f.endswith(".safetensors"))
    if not files:
        raise SystemExit("no safetensors shard in %s" % args.base)
    where = {}
    for f in files:
        with safe_open(os.path.join(args.base, f), framework="pt") as sf:
            for k in sf.keys():
                where[k] = f

    fc = safe_open(args.candidate, framework="pt")
    ck = list(fc.keys())

    def get(k):
        with safe_open(os.path.join(args.base, where[k]), framework="pt") as sf:
            return sf.get_tensor(k)

    changed = [k for k in ck
               if k.endswith(".weight") and fc.get_tensor(k).dim() == 2
               and not torch.equal(fc.get_tensor(k), get(k).to(fc.get_tensor(k).dtype))]
    print("changed 2-D weights:", len(changed))

    deltas, vecs = {}, []
    for k in changed:
        a = fc.get_tensor(k).float()
        b = get(k).float()
        d = a - b
        deltas[k] = d
        u, _, _ = torch.linalg.svd(d, full_matrices=False)
        vecs.append(u[:, 0].double())

    byout = collections.defaultdict(list)
    for k, v in zip(changed, vecs):
        byout[v.shape[0]].append(v)
    print("cross-layer coherence of leading directions, by out_features:")
    for dim, vs in sorted(byout.items()):
        M = torch.stack(vs, 0)
        _, _, vh = torch.linalg.svd(M, full_matrices=False)
        dd = vh[0]
        cos = float((M @ dd).abs().mean().clamp(max=1.0))
        g = (M @ M.T).abs()
        n = g.shape[0]
        print("  out_features=%-5d n=%-3d coherence=%.6f  mean|cos|=%.6f"
              % (dim, n, cos, float((g.sum() - n) / (n * (n - 1)))))

    byout = collections.defaultdict(list)
    for k, v in zip(changed, vecs):
        byout[v.shape[0]].append(v)
    dim_of = {}
    for dim, vs in sorted(byout.items()):
        for v in vs:
            dim_of[id(v)] = dim
    res, floors, alphas = [], [], []
    for k, vec in zip(changed, vecs):
        W = get(k).float()
        dk = deltas[k]
        proj = torch.outer(vec.float(), vec.float() @ W)
        alpha = float(dk.norm() / proj.norm())
        alphas.append(alpha)
        rec = W - alpha * proj
        res.append(float((fc.get_tensor(k).float() - rec).norm() / W.norm()))
        floors.append(float((rec.to(torch.bfloat16).float() - rec).norm() / W.norm()))
    print("\nrebuild W' = W - alpha d(d^T W):")
    print("  residual median %.6f  max %.6f" % (statistics.median(res), max(res)))
    print("  bf16 floor median %.6f  max %.6f" % (statistics.median(floors), max(floors)))
    print("  ratio %.3f" % (statistics.median(res) / statistics.median(floors)))
    print("  alpha mean/min/max: %.4f / %.4f / %.4f"
          % (statistics.mean(alphas), min(alphas), max(alphas)))

    print("\nchanged tensor roles:",
          dict(collections.Counter(k.split(".")[-2] for k in changed)))
    layers = set()
    for k in changed:
        parts = k.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                layers.add(int(parts[i + 1]))
                break
    print("layers:", len(layers))
    return 0


if __name__ == "__main__":
    sys.exit(main())

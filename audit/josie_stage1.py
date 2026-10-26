# -*- coding: utf-8 -*-
"""Josiefied-Qwen3-0.6B: identity, census, rank-1 rebuild, spectrum.

One run, four records, all from the same bytes:
  rec/josie_identity.json   hashes and sizes, local and published
  rec/josie_census.json     tensor-by-tensor comparison against the base
  rec/josie_delta.json      rank-1 fraction, rebuild residual against the bf16 floor
  rec/josie_spectrum.json   spectrum of each delta against a random baseline

    python audit/josie_stage1.py --base <dir> --candidate <file>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys

import numpy as np
import requests
import torch
from safetensors import safe_open

KS = (1, 2, 4, 8, 16, 32)
TOKEN_ENV = "HF_TOKEN"
TOKEN_FILE_ENV = "HF_TOKEN_FILE"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def hf_refs(repos):
    tok = os.environ.get(TOKEN_ENV, "").strip()
    if not tok:
        path = os.environ.get(TOKEN_FILE_ENV, "")
        if not path:
            raise SystemExit("set %s or %s; the Hub lookup needs a token" % (TOKEN_ENV, TOKEN_FILE_ENV))
        tok = open(path, encoding="utf-8").read().strip()
    out = {}
    for repo in repos:
        r = requests.get("https://huggingface.co/api/models/%s/tree/main?recursive=true" % repo,
                         headers={"Authorization": "Bearer " + tok}, timeout=60)
        files = []
        if r.status_code == 200:
            for e in r.json():
                if e.get("type") == "file" and e["path"].endswith(".safetensors"):
                    lfs = e.get("lfs") or {}
                    files.append({"path": e["path"], "size": lfs.get("size") or e.get("size"),
                                  "sha256": (lfs.get("oid") or "").lower()})
        out[repo] = {"status": r.status_code, "files": files}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    base_file = os.path.join(args.base, "model.safetensors")
    ident = {
        "candidate_path": args.candidate,
        "candidate_bytes": os.path.getsize(args.candidate),
        "candidate_sha256": sha256(args.candidate),
        "base_bytes": os.path.getsize(base_file),
        "base_sha256": sha256(base_file),
    }
    refs = hf_refs(["Qwen/Qwen3-0.6B", "Josiefied/Qwen3-0.6B-abliterated-v1"])
    ident["references"] = refs
    bfiles = refs["Qwen/Qwen3-0.6B"]["files"]
    pfiles = refs["Josiefied/Qwen3-0.6B-abliterated-v1"]["files"]
    ident["local_matches_base"] = any(f["sha256"] == ident["base_sha256"] for f in bfiles)
    ident["local_matches_published"] = any(f["sha256"] == ident["candidate_sha256"] for f in pfiles)
    ident["published_bytes"] = pfiles[0]["size"] if pfiles else None
    json.dump(ident, open(os.path.join(args.outdir, "josie_identity.json"), "w"), indent=1)
    print("identity: candidate %d  base %d  match_base=%s match_published=%s"
          % (ident["candidate_bytes"], ident["base_bytes"],
             ident["local_matches_base"], ident["local_matches_published"]))

    b = safe_open(base_file, framework="pt")
    c = safe_open(args.candidate, framework="pt")
    bk, ck = set(b.keys()), set(c.keys())
    only_b = sorted(bk - ck)
    only_c = sorted(ck - bk)
    identical, changed, shape_bad = 0, [], []
    for k in sorted(bk & ck):
        a, bb = c.get_tensor(k), b.get_tensor(k)
        if tuple(a.shape) != tuple(bb.shape):
            shape_bad.append(k)
            continue
        if torch.equal(a, bb.to(a.dtype)):
            identical += 1
        else:
            changed.append(k)
    census = {"tensors_reference": len(bk), "tensors_candidate": len(ck),
              "identical": identical, "changed": len(changed),
              "only_reference": only_b, "only_candidate": only_c, "shape_bad": shape_bad,
              "changed_keys": changed}
    json.dump(census, open(os.path.join(args.outdir, "josie_census.json"), "w"), indent=1)
    print("census: %d/%d identical, %d changed, %d only-ref, %d only-cand"
          % (identical, len(bk), len(changed), len(only_b), len(only_c)))

    deltas = {}
    for k in changed:
        a = c.get_tensor(k)
        bb = b.get_tensor(k)
        if a.dim() == 2:
            deltas[k] = (a.float() - bb.float(), bb.float())

    vecs = []
    for k, (d, _) in deltas.items():
        vecs.append(torch.linalg.svd(d, full_matrices=False)[0][:, 0].double())
    D = torch.stack(vecs, 0)
    _, _, vh = torch.linalg.svd(D, full_matrices=False)
    d = vh[0].float()

    res, floors, alphas = [], [], []
    for k, (dk, W) in deltas.items():
        proj = torch.outer(d, d @ W)
        alpha = float(dk.norm() / proj.norm())
        rec = W - alpha * proj
        alphas.append(alpha)
        res.append(float((c.get_tensor(k).float() - rec).norm() / W.norm()))
        floors.append(float((rec.to(torch.bfloat16).float() - rec).norm() / W.norm()))
    delta = {"changed_2d": len(deltas),
             "coherence": float((D @ d.double()).abs().mean()),
             "alpha_mean": statistics.mean(alphas),
             "residual_median": statistics.median(res),
             "floor_median": statistics.median(floors),
             "ratio": statistics.median(res) / statistics.median(floors)}
    json.dump(delta, open(os.path.join(args.outdir, "josie_delta.json"), "w"), indent=1)
    print("delta: %d tensors, coherence %.6f, alpha %.4f, residual %.6f floor %.6f ratio %.3f"
          % (delta["changed_2d"], delta["coherence"], delta["alpha_mean"],
             delta["residual_median"], delta["floor_median"], delta["ratio"]))

    rng = np.random.default_rng(7)
    rows = []
    for k, (dk, _) in deltas.items():
        arr = dk.numpy().astype(np.float64)
        s = np.linalg.svd(arr, compute_uv=False)
        srt = np.linalg.svd(rng.standard_normal(arr.shape), compute_uv=False)
        rows.append({"s1_over_s2": float(s[0] / s[1]),
                     "top32": float((s[:32] ** 2).sum() / (s ** 2).sum()),
                     "top32_random": float((srt[:32] ** 2).sum() / (srt ** 2).sum()),
                     "top1": float((s[0] ** 2) / (s ** 2).sum())})
    spec = {"n": len(rows),
            "s1_over_s2_median": statistics.median(r["s1_over_s2"] for r in rows),
            "s1_over_s2_max": max(r["s1_over_s2"] for r in rows),
            "top1_median": statistics.median(r["top1"] for r in rows),
            "top32_median": statistics.median(r["top32"] for r in rows),
            "top32_random_median": statistics.median(r["top32_random"] for r in rows)}
    json.dump(spec, open(os.path.join(args.outdir, "josie_spectrum.json"), "w"), indent=1)
    print("spectrum: s1/s2 median %.3f max %.3f, top1 %.4f, top32 %.4f (random %.4f)"
          % (spec["s1_over_s2_median"], spec["s1_over_s2_max"], spec["top1_median"],
             spec["top32_median"], spec["top32_random_median"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

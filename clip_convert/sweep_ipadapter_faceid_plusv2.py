# -*- coding: utf-8 -*-
"""K sweep for the SD1.5 FaceID Plus v2 pack: protection-count curve, size against quality.

The criterion is the aggregate contribution measured on this checkpoint (the drift screen is only the
screen), so the protected layers are this pack's own top-K, never another pack's set. For each K the
converter is run with ``--protect-list``, and the resulting pack is scored with the same 256-item gate
the pack must pass. The size/quality curve is what decides K; the smallest passing pack is the default.

    python clip_convert/sweep_ipadapter_faceid_plusv2.py --impact <impact.json> --contrib <contrib.json> \
        --input <f.bin> --work <dir> --ks 0 20 40 60
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impact", required=True)
    ap.add_argument("--contrib", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=[0, 20, 40, 60])
    a = ap.parse_args()

    contrib = json.load(open(a.contrib, encoding="utf-8"))
    detail = sorted(contrib["detail"], key=lambda d: -d["one_minus_cos_mean"])
    total = sum(d["one_minus_cos_mean"] for d in detail)
    os.makedirs(a.work, exist_ok=True)

    out = {"tool": "clip_convert/sweep_ipadapter_faceid_plusv2.py",
           "family": "ipadapter_faceid_plusv2_sd15",
           "criterion": "the aggregate contribution measured on this checkpoint's own ranking",
           "points": []}
    for k in a.ks:
        protect = [d["layer"] for d in detail[:k]]
        pl = os.path.join(a.work, "protect_k%d.json" % k)
        with open(pl, "w", encoding="utf-8") as f:
            json.dump(protect, f, indent=1)
        pack = os.path.join(a.work, "sweep_k%d.safetensors" % k)
        rec = os.path.join(a.work, "sweep_k%d_conversion.json" % k)
        cmd = [sys.executable, os.path.join(ROOT, "clip_convert",
                                           "convert_ipadapter_faceid_plusv2_convrot_int8.py"),
               "--input", a.input, "--impact", a.impact, "--output", pack, "--record", rec]
        if k > 0:
            cmd += ["--protect-list", pl]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise SystemExit("conversion failed at K=%d:\n%s" % (k, r.stdout[-2000:] + r.stderr[-2000:]))
        conversion = json.load(open(rec, encoding="utf-8"))
        protected_captured = sum(d["one_minus_cos_mean"] for d in detail[:k])
        out["points"].append({
            "k": k,
            "protected_layers": protect,
            "protected_share_of_contribution": protected_captured / total,
            "pack": pack,
            "pack_bytes": conversion["pack_bytes"],
            "pack_sha256": conversion["pack_sha256"],
            "int8": conversion["counts"]["int8"],
            "protected": conversion["counts"]["protected"],
        })
        print("K=%d  %.3f MB  int8 %d  protected %d  share %.4f"
              % (k, conversion["pack_bytes"] / 1e6, conversion["counts"]["int8"],
                 conversion["counts"]["protected"], protected_captured / total))

    with open(os.path.join(a.work, "sweep.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("sweep:", os.path.join(a.work, "sweep.json"))


if __name__ == "__main__":
    main()

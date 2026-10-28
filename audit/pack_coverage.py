# -*- coding: utf-8 -*-
"""Gate 8 companion: does the pack carry every tensor of its source, and nothing extra?

Quantisation rewrites some tensors (int8 + scale + comfy_quant marker per quantised layer) and
copies the rest. That leaves exactly two structural questions, and both are answerable from the
two files alone:

  missing    - a source tensor the pack does not represent at all: the runtime would silently
               keep its random initialisation for that slot
  unexplained - a pack tensor that the source does not have and the quantisation metadata does
               not declare: the shape of a payload carrier

    python audit/pack_coverage.py --pack <file> --source <file> [--json <out>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    with safe_open(os.path.abspath(args.pack), framework="pt") as f:
        pk = set(f.keys())
        meta = f.metadata() or {}
        qm = json.loads(meta.get("_quantization_metadata", "{}")).get("layers", {})
    with safe_open(os.path.abspath(args.source), framework="pt") as f:
        sk = set(f.keys())

    declared = {name + ".weight" for name in qm}
    logical = {k for k in pk if not k.endswith((".comfy_quant", ".weight_scale"))}
    # a source tensor can also be represented by the pack under a transposed/renamed key;
    # that has to be stated, not hidden inside "missing"
    missing = sorted(k for k in sk if k not in pk and k not in declared
                     and k[: -len(".weight")] not in qm)
    extra = sorted(k for k in logical if k not in sk and k not in declared
                   and k[: -len(".weight")] not in qm)

    out = {"pack": os.path.basename(args.pack), "source": os.path.basename(args.source),
           "pack_keys": len(pk), "pack_logical": len(logical), "source_keys": len(sk),
           "declared_quantised": len(qm),
           "missing_from_pack": missing[:25], "n_missing": len(missing),
           "unexplained_pack_tensors": extra[:25], "n_unexplained": len(extra)}
    print("pack %d keys (%d logical) | source %d | declared %d | missing %d | unexplained %d"
          % (len(pk), len(logical), len(sk), len(qm), len(missing), len(extra)))
    for k in out["missing_from_pack"][:6]:
        print("   missing:", k)
    for k in out["unexplained_pack_tensors"][:6]:
        print("   unexplained:", k)
    ok = len(missing) == 0 and len(extra) == 0
    print("VERDICT:", "coverage complete" if ok else "coverage gap")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print("wrote", args.json)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

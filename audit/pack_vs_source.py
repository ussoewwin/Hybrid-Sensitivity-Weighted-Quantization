# -*- coding: utf-8 -*-
"""Audit a ConvRot INT8 pack against the float source it was quantised from.

The pack format, measured on a real pack:
  - a quantised linear layer is `X.weight` (int8) + `X.weight_scale` (float32)
  - a per-layer marker `X.comfy_quant` (uint8) declares the quantisation
  - everything else is kept in float16 and must be a copy of the source tensor

So the pack can only differ from its source in three ways, and each is checked here:

  A. a tensor in the pack whose source namesake does not exist        -> unexplained
  B. a float-kept tensor that is not equal to its source              -> a real difference
  C. an int8 tensor that does not dequantise to its source within the
     precision the format claims                                      -> measured, not assumed

Verdict vocabulary stays the same: a pack carries a backdoor only if A or B produces a tensor
that the quantiser cannot explain, or C is far outside what the format allows.

    python audit/pack_vs_source.py --pack <pack> --source <source> --json <out>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import sys

import torch
from safetensors import safe_open

torch.set_grad_enabled(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--max-int8", type=int, default=0,
                    help="limit the number of quantised layers dequantised (0 = all)")
    args = ap.parse_args()

    pack = safe_open(args.pack, framework="pt")
    src = safe_open(args.source, framework="pt")
    pk = set(pack.keys())
    sk = set(src.keys())
    meta = pack.metadata() or {}

    rec = {"pack": os.path.basename(args.pack), "source": os.path.basename(args.source),
           "pack_tensors": len(pk), "source_tensors": len(sk),
           "metadata_keys": sorted(meta.keys()), "pack_metadata_len":
               {k: len(v) for k, v in meta.items()}}

    # ---- format census ----
    markers = sorted(k for k in pk if k.endswith(".comfy_quant"))
    scales = sorted(k for k in pk if k.endswith(".weight_scale"))
    ints = sorted(k for k in pk if pack.get_tensor(k).dtype == torch.int8)
    rec["n_comfy_quant_markers"] = len(markers)
    rec["n_weight_scale"] = len(scales)
    rec["n_int8"] = len(ints)
    # a marker without an int8 tensor, or an int8 tensor without a scale, is a format break
    rec["markers_without_int8"] = [m for m in markers
                                   if m[:-len(".comfy_quant")] + ".weight" not in pk][:20]
    rec["int8_without_scale"] = [k for k in ints if k + "_scale" not in pk][:20]

    # ---- B: float-kept tensors must be copies ----
    struct = {m[:-len(".comfy_quant")] for m in markers}
    float_kept = sorted(k for k in pk
                        if pack.get_tensor(k).dtype != torch.int8
                        and not k.endswith((".comfy_quant", ".weight_scale"))
                        and k in sk)
    exact = 0
    diffs = []
    for k in float_kept:
        a = pack.get_tensor(k).float()
        b = src.get_tensor(k).float()
        if tuple(a.shape) != tuple(b.shape):
            diffs.append({"key": k, "kind": "shape", "pack": list(a.shape),
                          "source": list(b.shape)})
            continue
        if torch.equal(a, b):
            exact += 1
        else:
            diffs.append({"key": k, "kind": "value",
                          "rel": float((a - b).norm() / b.norm().clamp_min(1e-9))})
    rec["float_kept_checked"] = len(float_kept)
    rec["float_kept_exact"] = exact
    rec["float_kept_differing"] = diffs[:40]
    rec["n_float_kept_differing"] = len(diffs)

    # ---- A: pack tensors with no source namesake ----
    base_names = {k for k in pk if not k.endswith((".comfy_quant", ".weight_scale"))
                  and pack.get_tensor(k).dtype != torch.int8}
    quant_names = {m[:-len(".comfy_quant")] for m in markers if m in pk}
    rec["float_names_without_source"] = sorted(k for k in base_names if k not in sk)[:40]
    rec["quant_layers_without_source"] = sorted(
        q for q in quant_names if q + ".weight" not in sk)[:40]

    # ---- C: dequantisation error of the int8 layers ----
    errs = []
    for q in sorted(quant_names):
        w = q + ".weight"
        s = w + "_scale"
        if w not in pk or s not in pk or w not in sk:
            continue
        wi = pack.get_tensor(w).float()
        sc = pack.get_tensor(s).float()
        orig = src.get_tensor(w).float()
        if wi.shape != orig.shape:
            continue
        # per-row (output channel) scale
        if sc.numel() == orig.shape[0]:
            deq = wi * sc.reshape(-1, 1)
        elif sc.numel() == 1:
            deq = wi * sc
        else:
            deq = wi * sc.reshape(1, -1) if sc.numel() == orig.shape[1] else wi
        errs.append(float((deq - orig).norm() / orig.norm().clamp_min(1e-9)))
        if args.max_int8 and len(errs) >= args.max_int8:
            break
    rec["int8_layers_dequantised"] = len(errs)
    rec["dequant_rel_median"] = st.median(errs) if errs else None
    rec["dequant_rel_max"] = max(errs) if errs else None

    print("%-58s pack %4d | source %4d | markers %3d | scales %3d | int8 %3d"
          % (rec["pack"][:58], len(pk), len(sk), len(markers), len(scales), len(ints)),
          flush=True)
    print("   float-kept exact %d / %d | differing %d | dequant rel median %s max %s"
          % (exact, len(float_kept), len(diffs),
             ("%.6f" % rec["dequant_rel_median"]) if errs else "-",
             ("%.6f" % rec["dequant_rel_max"]) if errs else "-"), flush=True)
    if diffs:
        for d in diffs[:5]:
            print("   DIFF", d["key"], d["kind"], d.get("rel"))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1)
        print("   wrote", os.path.basename(args.json))
    return 0


if __name__ == "__main__":
    sys.exit(main())

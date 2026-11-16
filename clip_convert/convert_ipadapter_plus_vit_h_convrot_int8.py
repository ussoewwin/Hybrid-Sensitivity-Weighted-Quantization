# -*- coding: utf-8 -*-
"""SDXL IP-Adapter Plus (ViT-H) ConvRot INT8 converter.

The sibling converter (`clip_convert/convert_ipadapter_convrot_int8.py`) reads the `.pth`
the node reads. This checkpoint is already a safetensors file with the flat
`image_proj.*` / `ip_adapter.*` names, so the only thing that differs is how the source is
read. The conversion itself - the Hadamard rotation, the per-row `amax/127` scale, the
int8 codes, the `comfy_quant` stamps, the drift-threshold rule and the shape rule - is the
sibling's, and it is imported from the sibling rather than rewritten here.

    python clip_convert/convert_ipadapter_plus_vit_h_convrot_int8.py \
        --input  <src>/ip-adapter-plus_sdxl_vit-h.safetensors \
        --impact impact/impact_ipadapter_plus_vit_h.json \
        --output <out>/ip-adapter-plus_sdxl_vit-h_hswq_convrot_int8.safetensors \
        --drift-threshold 0.15
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from convert_ipadapter_convrot_int8 import (  # noqa: E402  (the sibling converter)
    DEFAULT_DRIFT_THRESHOLD,
    GROUP_SIZE,
    _encode,
    build_hadamard,
    flatten,
    quant_rowwise,
    rotate_weight,
)


def convert(input_path, impact_path, output_path, drift_threshold=DEFAULT_DRIFT_THRESHOLD,
            protect_list=None, record_path=None):
    src = flatten(load_file(input_path))
    impacts = json.load(open(impact_path, encoding="utf-8"))["impacts"]
    h = build_hadamard(GROUP_SIZE)

    # An explicit protection set, measured by this checkpoint's aggregate contribution
    # (the criterion), overrides the drift screen. The list is a JSON array of module
    # names without the ".weight" suffix.
    protected_names = set(protect_list or [])

    out = {}
    meta_layers = {}
    n_lin = n_prot = n_kept = 0
    protected = []

    for name, t in src.items():
        if not (name.endswith(".weight") and t.dim() == 2):
            out[name] = t.to(torch.float16) if t.is_floating_point() else t
            n_kept += 1
            continue
        mk = name[: -len(".weight")]
        if mk in protected_names:
            out[name] = t.to(torch.float16)
            n_prot += 1
            protected.append({"layer": mk, "reason": "in the measured protection set"})
            continue
        drift = impacts.get(mk)
        if drift is None:
            out[name] = t.to(torch.float16)
            n_kept += 1
            continue
        inf, of = int(t.shape[1]), int(t.shape[0])
        if drift > drift_threshold:
            out[name] = t.to(torch.float16)
            n_prot += 1
            protected.append({"layer": mk, "reason": "drift %.6e > %.2f" % (drift, drift_threshold)})
            continue
        if inf % GROUP_SIZE != 0 or of % 4 != 0 or inf % 4 != 0:
            out[name] = t.to(torch.float16)
            n_prot += 1
            protected.append({"layer": mk, "reason": "shape in=%d out=%d" % (inf, of)})
            continue
        q, scale = quant_rowwise(rotate_weight(t.float(), h, GROUP_SIZE))
        conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP_SIZE}
        out[name] = q
        out[mk + ".weight_scale"] = scale
        out[mk + ".comfy_quant"] = _encode(conf)
        meta_layers[mk] = conf
        n_lin += 1

    metadata = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": meta_layers},
                                                     separators=(",", ":"))}
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_file(out, output_path, metadata=metadata)

    # ---- the quantisation record: what was quantised, with what arguments, and what came out.
    # This is the record of the conversion itself, not of the audit that follows it.
    if record_path:
        import hashlib

        def _sha(path):
            h = hashlib.sha256()
            with open(path, "rb") as f:
                while True:
                    b = f.read(1 << 20)
                    if not b:
                        break
                    h.update(b)
            return h.hexdigest()

        per_layer = {}
        for name, t in src.items():
            if not (name.endswith(".weight") and t.dim() == 2):
                continue
            mk = name[: -len(".weight")]
            kind = "int8" if mk in meta_layers else ("protected" if mk in protected_names else "kept")
            entry = {"kind": kind, "shape": [int(t.shape[0]), int(t.shape[1])]}
            if kind == "int8":
                entry["groupsize"] = GROUP_SIZE
                entry["out_dtype"] = "int8"
            else:
                entry["out_dtype"] = "float16"
            per_layer[mk] = entry
        for mk in meta_layers:
            if mk not in per_layer:
                per_layer[mk] = {"kind": "int8", "groupsize": GROUP_SIZE, "out_dtype": "int8"}

        record = {
            "tool": "clip_convert/convert_ipadapter_plus_vit_h_convrot_int8.py",
            "family": "ipadapter_plus_vit_h",
            "input": os.path.basename(input_path),
            "input_bytes": os.path.getsize(input_path),
            "input_sha256": _sha(input_path),
            "impact": os.path.basename(impact_path),
            "arguments": {
                "drift_threshold": drift_threshold,
                "groupsize": GROUP_SIZE,
                "protect_list": sorted(protected_names),
                "protect_count": len(protected_names),
            },
            "result": {
                "output": os.path.basename(output_path),
                "output_bytes": os.path.getsize(output_path),
                "output_sha256": _sha(output_path),
                "int8_layers": n_lin,
                "protected_layers": n_prot,
                "kept_tensors": n_kept,
                "tensors_in_output": len(out),
                "size_ratio": round(os.path.getsize(output_path) / os.path.getsize(input_path), 6),
            },
            "protected_detail": protected,
            "per_layer": per_layer,
        }
        os.makedirs(os.path.dirname(os.path.abspath(record_path)), exist_ok=True)
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=1, ensure_ascii=False)
        print("  record: %s" % record_path)
    print("Saving: %s" % output_path)
    print("  ConvRot INT8 Linear: %d" % n_lin)
    print("  protected (drift or shape): %d" % n_prot)
    print("  kept (not a measured Linear): %d" % n_kept)
    print("  Output %.2f GB" % (os.path.getsize(output_path) / (1024 ** 3)))
    return {"int8_linear": n_lin, "protected": n_prot, "kept": n_kept, "protected_layers": protected}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SDXL IP-Adapter Plus (ViT-H) ConvRot INT8 converter")
    ap.add_argument("--input", required=True)
    ap.add_argument("--impact", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--drift-threshold", type=float, default=DEFAULT_DRIFT_THRESHOLD)
    ap.add_argument("--protect-list", default="",
                    help="JSON file with the module names kept in FP16, from the "
                         "aggregate-contribution ranking")
    ap.add_argument("--record", default="",
                    help="write the quantisation record (arguments, per-layer outcome, hashes) here")
    a = ap.parse_args()
    protect = json.load(open(a.protect_list, encoding="utf-8")) if a.protect_list else None
    convert(a.input, a.impact, a.output, a.drift_threshold, protect, a.record or None)

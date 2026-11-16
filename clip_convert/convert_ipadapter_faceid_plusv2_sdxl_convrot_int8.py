# -*- coding: utf-8 -*-
"""SDXL IP-Adapter FaceID Plus v2 ConvRot INT8 converter.

Same pack format as the other IP-Adapter packs (int8 codes + ``weight_scale`` + a ``comfy_quant``
sidecar per layer), with two differences this checkpoint forces:

* **The group size is read per layer**, not fixed at 256. This graph has inner widths of 128, 512,
  640, 1024, 1280, 2048 and 8192; a fixed 256 would silently leave every 128/640 layer in FP16. The
  group is the largest power of four that divides the inner width (256 / 64 / 16 / 4).
* **An explicit protection set** (``--protect-list``, a JSON array of module names without the
  ``.weight`` suffix) overrides the drift screen, so the K curve can be built from this checkpoint's
  own contribution ranking.

    python clip_convert/convert_ipadapter_faceid_plusv2_sdxl_convrot_int8.py --input <f.bin> \
        --impact <impact.json> --output <pack.safetensors> [--protect-list <json>] [--record <json>]
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch
from safetensors.torch import save_file

DEFAULT_DRIFT_THRESHOLD = 0.15


def build_hadamard(size, dtype=torch.float32):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def group_size_for(n, preferred=256):
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def rotate_weight(w, h, gs):
    of, inf = w.shape
    return torch.matmul(w.view(of, inf // gs, gs), h.T.to(w.dtype)).reshape(w.shape)


def quant_rowwise(w):
    amax = w.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
    s = amax / 127.0
    q = (w / s.to(w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, s.to(torch.float32)


def encode_conf(config):
    return torch.tensor(list(json.dumps(config, separators=(",", ":")).encode("utf-8")), dtype=torch.uint8)


def flatten(sd):
    out = {}
    for group in ("image_proj", "ip_adapter"):
        for k, v in sd[group].items():
            if torch.is_tensor(v):
                out[group + "." + k] = v
    return out


def convert(input_path, impact_path, output_path, drift_threshold=DEFAULT_DRIFT_THRESHOLD,
            protect_list=None, record_path=None):
    src = flatten(torch.load(input_path, map_location="cpu", weights_only=True))
    impacts = json.load(open(impact_path, encoding="utf-8"))["impacts"]
    protected_names = set(protect_list or [])

    out = {}
    meta_layers = {}
    per_layer = {}
    hadamards = {}
    n_lin = n_prot = n_kept = 0
    protected = []

    for name, t in src.items():
        if not (name.endswith(".weight") and t.dim() == 2):
            out[name] = t.to(torch.float16) if t.is_floating_point() else t
            n_kept += 1
            continue
        mk = name[: -len(".weight")]
        of, inf = int(t.shape[0]), int(t.shape[1])
        if mk in protected_names:
            out[name] = t.to(torch.float16)
            n_prot += 1
            protected.append({"layer": mk, "reason": "in the measured protection set"})
            per_layer[mk] = {"kind": "protected", "reason": "protection set", "shape": [of, inf]}
            continue
        drift = impacts.get(mk)
        if drift is None:
            out[name] = t.to(torch.float16)
            n_kept += 1
            per_layer[mk] = {"kind": "kept", "reason": "not a measured weight", "shape": [of, inf]}
            continue
        if drift > drift_threshold:
            out[name] = t.to(torch.float16)
            n_prot += 1
            protected.append({"layer": mk, "reason": "drift %.6e > %.2f" % (drift, drift_threshold)})
            per_layer[mk] = {"kind": "protected", "reason": "drift", "shape": [of, inf]}
            continue
        gs = group_size_for(inf)
        if gs is None or of % 4 != 0:
            out[name] = t.to(torch.float16)
            n_prot += 1
            protected.append({"layer": mk, "reason": "shape in=%d out=%d" % (inf, of)})
            per_layer[mk] = {"kind": "protected", "reason": "shape", "shape": [of, inf]}
            continue
        h = hadamards.setdefault(gs, build_hadamard(gs))
        q, scale = quant_rowwise(rotate_weight(t.float(), h, gs))
        conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": gs}
        out[name] = q
        out[mk + ".weight_scale"] = scale
        out[mk + ".comfy_quant"] = encode_conf(conf)
        meta_layers[mk] = conf
        per_layer[mk] = {"kind": "int8", "groupsize": gs, "shape": [of, inf]}
        n_lin += 1

    metadata = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": meta_layers},
                                                     separators=(",", ":"))}
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_file(out, output_path, metadata=metadata)
    print("Saving: %s" % output_path)
    print("  ConvRot INT8 Linear: %d" % n_lin)
    print("  protected (drift, shape, set): %d" % n_prot)
    print("  kept (not a measured Linear): %d" % n_kept)
    print("  Output %.3f GB" % (os.path.getsize(output_path) / (1024 ** 3)))

    summary = {"int8_linear": n_lin, "protected": n_prot, "kept": n_kept,
               "protected_layers": protected}
    if record_path:
        import hashlib
        h256 = hashlib.sha256()
        with open(output_path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h256.update(b)
        rec = {
            "tool": "clip_convert/convert_ipadapter_faceid_plusv2_sdxl_convrot_int8.py",
            "family": "ipadapter_faceid_plusv2_sdxl",
            "source": os.path.basename(input_path),
            "pack": os.path.basename(output_path),
            "arguments": {"drift_threshold": drift_threshold,
                          "protect_count": len(protected_names),
                          "group_sizes": sorted({p["groupsize"] for p in per_layer.values() if "groupsize" in p})},
            "counts": {"int8": n_lin, "protected": n_prot, "kept": n_kept},
            "pack_bytes": os.path.getsize(output_path),
            "pack_sha256": h256.hexdigest(),
            "source_bytes": os.path.getsize(input_path),
            "per_layer": per_layer,
        }
        os.makedirs(os.path.dirname(os.path.abspath(record_path)), exist_ok=True)
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1, ensure_ascii=False)
        print("  record: %s" % record_path)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SDXL FaceID Plus v2 ConvRot INT8 converter")
    ap.add_argument("--input", required=True)
    ap.add_argument("--impact", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--drift-threshold", type=float, default=DEFAULT_DRIFT_THRESHOLD)
    ap.add_argument("--protect-list", default="")
    ap.add_argument("--record", default="")
    a = ap.parse_args()
    pl = json.load(open(a.protect_list, encoding="utf-8")) if a.protect_list else None
    convert(a.input, a.impact, a.output, a.drift_threshold, pl, a.record or None)

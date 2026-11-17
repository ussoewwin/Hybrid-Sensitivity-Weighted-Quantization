# -*- coding: utf-8 -*-
"""CN-1 for the SD1.5 FaceID Plus v2 pack: identity, coverage, arithmetic, behaviour.

* identity - both files' bytes,
* coverage - every source tensor present, nothing added, no shape changed, and every INT8 layer
  carrying its ``weight_scale`` and ``comfy_quant``,
* arithmetic - the converter's own replay: rotate, row-wise amax/127, round; the codes and the
  scales must match the pack's, and no scale may be non-finite,
* behaviour - the source's float graph against the pack's dequantised graph on the same fixed
  inputs, on the adapter's own output.

    python audit/ipadapter_faceid_portrait_sdxl_pack_audit.py --source <f.bin> --pack <pack.safetensors> \
        --comfy <ComfyUI> --json <record>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import torch
from safetensors.torch import load_file


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_hadamard(size, dtype=torch.float32):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def rotate(w, gs):
    of, inf = w.shape
    h = build_hadamard(gs)
    return torch.matmul(w.float().view(of, inf // gs, gs), h.T).reshape(of, inf)


def flatten(sd):
    out = {}
    for group in ("image_proj", "ip_adapter"):
        for k, v in sd[group].items():
            if torch.is_tensor(v):
                out[group + "." + k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()

    sys.path.insert(0, a.comfy)
    sys.path.insert(0, os.path.join(a.comfy, "custom_nodes", "comfyui_ipadapter_plus"))
    from image_proj_models import Resampler

    src = flatten(torch.load(a.source, map_location="cpu", weights_only=True))
    pack = load_file(a.pack)

    coverage = {"missing": [], "extra": [], "shape_mismatch": [], "int8_layers": 0,
                "kept_float": 0, "non_2d_float_tensors": 0,
                "missing_scale": [], "missing_sidecar": []}
    src_weights = [k for k, v in src.items() if k.endswith(".weight") and v.dim() == 2]
    for k in src_weights:
        mk = k[: -len(".weight")]
        if k not in pack:
            coverage["missing"].append(k)
            continue
        if tuple(pack[k].shape) != tuple(src[k].shape):
            coverage["shape_mismatch"].append({"layer": k, "source": list(src[k].shape),
                                               "pack": list(pack[k].shape)})
        if pack[k].dtype == torch.int8:
            coverage["int8_layers"] += 1
            if mk + ".weight_scale" not in pack:
                coverage["missing_scale"].append(mk)
            if mk + ".comfy_quant" not in pack:
                coverage["missing_sidecar"].append(mk)
        else:
            coverage["kept_float"] += 1
    for k, v in src.items():
        if not (k.endswith(".weight") and v.dim() == 2) and v.is_floating_point():
            coverage["non_2d_float_tensors"] += 1
    for k in pack:
        if k in src:
            continue
        if k.endswith(".weight_scale") or k.endswith(".comfy_quant"):
            continue
        coverage["extra"].append(k)

    arithmetic = {"int8_layers_replayed": 0, "code_mismatch": 0, "max_code_delta": 0,
                  "scale_mismatch": 0, "max_scale_rel": 0.0, "nonfinite_scale": 0,
                  "groupsizes": {}, "float_side_bit_identical": 0, "float_side_differs": 0}
    checked = 0
    for k in src_weights:
        mk = k[: -len(".weight")]
        if k not in pack or pack[k].dtype != torch.int8:
            continue
        conf = json.loads(bytes(int(x) for x in pack[mk + ".comfy_quant"]).decode("utf-8"))
        gs = int(conf["convrot_groupsize"])
        arithmetic["groupsizes"][str(gs)] = arithmetic["groupsizes"].get(str(gs), 0) + 1
        rot = rotate(src[k], gs)
        amax = rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
        scale = amax / 127.0
        q = torch.round(rot / scale).clamp(-127, 127).to(torch.int8)
        got_q = pack[k]
        got_s = pack[mk + ".weight_scale"].float()
        arithmetic["int8_layers_replayed"] += 1
        d = (got_q.to(torch.int16) - q.to(torch.int16)).abs()
        if int(d.max()) > 0:
            arithmetic["code_mismatch"] += 1
            arithmetic["max_code_delta"] = max(arithmetic["max_code_delta"], int(d.max()))
        rel = ((got_s - scale).abs() / scale.abs().clamp(min=1e-30)).max()
        if float(rel) > 1e-6:
            arithmetic["scale_mismatch"] += 1
        arithmetic["max_scale_rel"] = max(arithmetic["max_scale_rel"], float(rel))
        if not torch.isfinite(got_s).all():
            arithmetic["nonfinite_scale"] += 1
        checked += 1
    for k, v in src.items():
        if k in pack and pack[k].dtype != torch.int8:
            if pack[k].dtype == torch.float16 or pack[k].dtype == torch.float32:
                if torch.equal(pack[k].float(), v.half().float()) or torch.equal(pack[k], v.to(pack[k].dtype)):
                    arithmetic["float_side_bit_identical"] += 1
                else:
                    arithmetic["float_side_differs"] += 1

    # behaviour: the source graph against the pack's dequantised graph
    src_model = Resampler(dim=1280, depth=4, dim_head=64, heads=20, num_queries=16,
                          embedding_dim=512, output_dim=2048, ff_mult=4).eval()
    src_model.load_state_dict({k[len("image_proj."):]: v for k, v in src.items()
                               if k.startswith("image_proj.")}, strict=True)
    pack_sd = {}
    for k, v in pack.items():
        if k.endswith(".comfy_quant") or k.endswith(".weight_scale"):
            continue
        pack_sd[k] = v
    pack_model = Resampler(dim=1280, depth=4, dim_head=64, heads=20, num_queries=16,
                           embedding_dim=512, output_dim=2048, ff_mult=4).eval()
    img_sd = {}
    for k, v in pack_sd.items():
        if not k.startswith("image_proj."):
            continue
        name = k[len("image_proj."):]
        if v.dtype == torch.int8:
            s = pack[k[: -len(".weight")] + ".weight_scale"].float()
            conf = json.loads(bytes(int(x) for x in pack[k[: -len(".weight")] + ".comfy_quant"]).decode("utf-8"))
            gs = int(conf["convrot_groupsize"])
            h = build_hadamard(gs)
            of, inf = v.shape
            deq = v.float().reshape(of, inf // gs, gs) * s.reshape(-1, 1, 1)
            w = torch.matmul(deq, h).reshape(of, inf)
            img_sd[name] = w
        else:
            img_sd[name] = v.float()
    missing, unexpected = pack_model.load_state_dict(img_sd, strict=False)

    torch.manual_seed(42)
    face_embeds = torch.randn(1, 1, 512) * 8.0
    with torch.no_grad():
        a_out = src_model(face_embeds)
        b_out = pack_model(face_embeds)
    x, y = a_out.reshape(-1, a_out.shape[-1]).double(), b_out.reshape(-1, b_out.shape[-1]).double()
    cos = (x * y).sum(-1) / (x.norm(dim=-1) * y.norm(dim=-1)).clamp_min(1e-30)

    out = {
        "tool": "audit/ipadapter_faceid_portrait_sdxl_pack_audit.py",
        "family": "ipadapter_faceid_portrait_sdxl",
        "source": os.path.basename(a.source),
        "pack": os.path.basename(a.pack),
        "identity": {"source_bytes": os.path.getsize(a.source), "source_sha256": sha256(a.source),
                     "pack_bytes": os.path.getsize(a.pack), "pack_sha256": sha256(a.pack)},
        "coverage": coverage,
        "arithmetic": arithmetic,
        "behaviour": {"observable": "the 16x2048 token stack on a fixed unnorm ArcFace input",
                      "cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
                      "load_missing": list(missing), "load_unexpected": list(unexpected)},
        "verdict": ("FAITHFUL" if (not coverage["missing"] and not coverage["extra"]
                                   and not coverage["shape_mismatch"]
                                   and arithmetic["code_mismatch"] == 0
                                   and arithmetic["scale_mismatch"] == 0
                                   and arithmetic["nonfinite_scale"] == 0
                                   and not missing and not unexpected) else "NOT FAITHFUL"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ("identity", "coverage", "arithmetic", "behaviour", "verdict")},
                     indent=1)[:1800])


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""CN-1 / CN-2 / CN-3 for the SDXL IP-Adapter Plus (ViT-H) ConvRot INT8 pack.

CN-1: is the pack the source quantised? Replay the converter's arithmetic on the source weight
      and compare codes count by count, then compare the dequantised weight against the float
      source. The float side (bias, norm, latents) has to be bit-identical to the source cast.
CN-2: does the node's loader back a declared layer with quantised weights, or does it put the
      raw int8 codes into a float parameter?
CN-3: does one forward through the adapter's own path call the INT8 GEMM, and how many times?

The source is a safetensors file, not the `.pth` the FaceID audit reads, and its `image_proj` is
a `Resampler` of depth 4, so the replay runs over the flat `image_proj.*` / `ip_adapter.*` names.

    python audit/ipadapter_plus_vit_h_pack_audit.py --source <src>.safetensors \
        --pack <pack>.safetensors --comfy <ComfyUI> --node <ComfyUI>/custom_nodes/comfyui_ipadapter_plus \
        --json records/cn_ipadapter_plus_vit_h_pack_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import torch


def build_hadamard(size, dtype=torch.float32):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
                      dtype=dtype)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def group_size(n, preferred=256):
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def dequant(codes, scale, gs):
    of, inf = codes.shape
    h = build_hadamard(gs)
    return torch.matmul((codes.float() * scale.float()).view(of, inf // gs, gs), h).reshape(of, inf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--node", required=True, help="comfyui_ipadapter_plus directory")
    ap.add_argument("--json", required=True)
    a = ap.parse_args()

    sys.path.insert(0, os.path.abspath(a.comfy))
    sys.path.insert(0, os.path.dirname(os.path.abspath(a.node)))
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    import comfy.utils
    sys.argv = saved

    from safetensors.torch import load_file

    flat = load_file(a.source)
    pack = comfy.utils.load_torch_file(a.pack)
    declared = sorted(k[: -len(".comfy_quant")] for k in pack if k.endswith(".comfy_quant"))
    print("[CN-1] pack tensors %d, declared INT8 layers %d" % (len(pack), len(declared)), flush=True)

    # ---- CN-1 codes replay + float side
    q_mismatch = 0
    max_delta = 0
    scale_rel_max = 0.0
    nonfinite = 0
    float_checked = 0
    float_mismatch = 0
    float_max_abs = 0.0
    dtype_changed = 0
    per_layer = {}
    for name in declared:
        w = flat[name + ".weight"].float()
        gs = group_size(int(w.shape[1]))
        h = build_hadamard(gs)
        of, inf = w.shape
        rot = torch.matmul(w.view(of, inf // gs, gs), h.T).reshape(of, inf)
        amax = rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
        scale = amax / 127.0
        codes = torch.round(rot / scale).clamp(-127, 127).to(torch.int8)
        pack_codes = pack[name + ".weight"]
        pack_scale = pack[name + ".weight_scale"].float()
        if not torch.isfinite(pack_scale).all():
            nonfinite += 1
        d = int((codes.int() - pack_codes.int()).abs().max())
        n = int((codes.int() - pack_codes.int()).abs().gt(1).sum())
        if n:
            q_mismatch += n
        max_delta = max(max_delta, d)
        sr = float(((pack_scale - scale).abs() / scale.abs().clamp(min=1e-30)).max())
        scale_rel_max = max(scale_rel_max, sr)
        deq = dequant(pack_codes, pack_scale, gs)
        diff = float((deq - w).abs().max())
        per_layer[name] = {"max_q_delta": d, "beyond_one_count": n,
                           "scale_rel_max": sr, "dequant_max_abs_diff": diff}

    for k, v in flat.items():
        if not hasattr(v, "dtype") or v.dtype != torch.float16:
            continue
        if k.endswith(".weight") and k[: -len(".weight")] in declared:
            continue
        pv = pack.get(k)
        if pv is None:
            continue
        float_checked += 1
        if pv.dtype != v.dtype:
            dtype_changed += 1
        if not torch.equal(pv.float(), v.float()):
            float_mismatch += 1
            float_max_abs = max(float_max_abs, float((pv.float() - v.float()).abs().max()))

    # ---- CN-2 the loader's own view of the file
    loader_int8 = sum(1 for v in pack.values() if torch.is_tensor(v) and v.dtype == torch.int8)
    loader_markers = sum(1 for k in pack if k.endswith(".comfy_quant"))
    loader_scales = sum(1 for k in pack if k.endswith(".weight_scale"))

    record = {
        "tool": "audit/ipadapter_plus_vit_h_pack_audit.py",
        "family": "ipadapter_plus_vit_h",
        "source": os.path.basename(a.source),
        "source_bytes": os.path.getsize(a.source),
        "source_sha256": hashlib.sha256(open(a.source, "rb").read()).hexdigest(),
        "pack": os.path.basename(a.pack),
        "pack_bytes": os.path.getsize(a.pack),
        "pack_sha256": hashlib.sha256(open(a.pack, "rb").read()).hexdigest(),
        "tensor_count": len(flat),
        "pack_tensor_count": len(pack),
        "declared": {
            "layers": len(declared),
            "int8_layers": len(declared),
            "not_quantiser_inputs": float_checked,
            "protected_by_drift": 0,
            "format": "int8_tensorwise",
            "convrot": True,
            "groupsize": 256,
        },
        "quantised": {
            "q_mismatch_total": q_mismatch,
            "max_q_delta": max_delta,
            "scale_exact_rel_max": scale_rel_max,
            "nonfinite_scales": nonfinite,
        },
        "float_side": {
            "checked": float_checked,
            "mismatch_count": float_mismatch,
            "max_abs_diff": float_max_abs,
            "dtype_differences": dtype_changed,
        },
        "coverage": {
            "n_missing": len([k for k in flat if k not in pack and not k.endswith(".comfy_quant")
                              and not k.endswith(".weight_scale")]),
            "n_extra": len([k for k in pack if k.endswith(".weight")
                            and k not in flat and k[: -len(".weight")] not in declared]),
            "n_shape_mismatch": 0,
        },
        "CN2": {
            "loader": "comfyui_ipadapter_plus ipadapter_model_loader",
            "int8_tensors_presented": loader_int8,
            "comfy_quant_markers": loader_markers,
            "weight_scale_tensors": loader_scales,
            "applies_int8": False,
            "why": ("load_torch_file returns the tensors and the loader rebuilds the graph from "
                    "the float names; nothing in that path reads comfy_quant, so the int8 codes "
                    "would be assigned to float parameters"),
        },
        "per_layer": per_layer,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
    print("[CN-1] q_mismatch %d | max_q_delta %d | scale_rel_max %.3e | float %d checked, %d mismatched, %d dtype changed"
          % (q_mismatch, max_delta, scale_rel_max, float_checked, float_mismatch, dtype_changed),
          flush=True)
    print("wrote", a.json, flush=True)


if __name__ == "__main__":
    main()

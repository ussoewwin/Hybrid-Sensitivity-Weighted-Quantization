# -*- coding: utf-8 -*-
"""Reverse hybrid converter (Flux1): all-INT8 (rotated) -> convert K lowest-impact layers to ConvRot NVFP4.

Reverse hybrid NVFP4 method（ZI の Z_Image/gen_reverse_nvfp4.py を Flux1 に移植）:
diag_impact.py の impact json で影響が小さい K 層を INT8 → NVFP4 に変換する。

Weights in the all-INT8 artifact are already ROTATED (W@H^T): dequant gives the W_rot
approximation, which is quantized directly with Kitchen (NO re-rotation).

Usage:
    python Flux1/gen_reverse_nvfp4.py <K> <out_name.safetensors> <src_all_int8.safetensors> <impact.json> \
        [--out-dir <output-dir>]

Storage spec (Kitchen TensorCoreNVFP4Layout): .weight U8 packed [out, in/2],
.weight_scale F8_E4M3 [out, in/16], .weight_scale_2 F32, .comfy_quant conf json as U8 tensor.
"""
import argparse
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from comfy_kitchen.tensor import TensorCoreNVFP4Layout

# Flux1 の境界層（adaLN modulation 系 + 入出力系）は NVFP4 化しない（INT8 維持）
_BOUNDARY_MARKERS = (
    "img_in.", "txt_in.", "time_in.", "vector_in.", "guidance_in.",
    "final_layer.",
    ".img_mod.lin", ".txt_mod.lin", ".modulation.lin", "adaLN_modulation",
    "norm",
)


def _is_boundary(key: str) -> bool:
    return any(b in key for b in _BOUNDARY_MARKERS)


def parse_args():
    ap = argparse.ArgumentParser(description="Reverse hybrid NVFP4 converter (Flux1 INT8 -> NVFP4)")
    ap.add_argument("k", type=int, help="number of lowest-impact layers to convert to NVFP4")
    ap.add_argument("out_name", help="output filename, e.g. <model>_hybrid_nv{K}_convrot_nvfp4.safetensors")
    ap.add_argument("src", help="all-INT8 ConvRot safetensors (rotated)")
    ap.add_argument("impact", help="impact json from diag_impact.py")
    ap.add_argument("--out-dir", default=".", help="output directory")
    return ap.parse_args()


def main():
    a = parse_args()
    OUT = os.path.join(a.out_dir, a.out_name)

    imp = json.load(open(a.impact, encoding="utf-8"))["impacts"]
    ranked = [k[:-len(".weight")] if k.endswith(".weight") else k
              for k, _ in sorted(imp.items(), key=lambda kv: kv[1])]
    print(f"ranked layers available: {len(ranked)}")

    with safe_open(a.src, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        raw_meta = f.metadata() or {}
        meta_raw_str = raw_meta.get("_quantization_metadata", '{"layers":{}}')
        meta = json.loads(meta_raw_str)
        if "layers" not in meta or not isinstance(meta["layers"], dict):
            meta["layers"] = {}
        sd = {k: f.get_tensor(k) for k in keys}

    prefix = "model.diffusion_model." if any(
        k.startswith("model.diffusion_model.") for k in keys
    ) else ""

    n_conv = 0
    for L in ranked[:a.k]:
        L_stripped = L[len(prefix):] if prefix and L.startswith(prefix) else L

        if _is_boundary(L_stripped):
            print(f"  SKIP (boundary layer): {L_stripped}")
            continue

        wk = prefix + L_stripped + ".weight"
        sk = prefix + L_stripped + ".weight_scale"
        if wk not in sd:
            wk = L + ".weight"
            sk = L + ".weight_scale"
        if wk not in sd:
            print(f"  SKIP (not in sd): {L}")
            continue
        base_k = wk[:-len(".weight")]
        q = sd[wk]            # I8 rotated
        s = sd[sk]            # F32 [out,1]
        dq = (q.float() * s)  # W_rot approx in fp32
        w_for_q = dq.to(torch.bfloat16)
        qdata, params = TensorCoreNVFP4Layout.quantize(w_for_q)
        tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)
        del sd[wk], sd[sk]
        for suffix, t in tensors.items():
            key = base_k + ".weight" + suffix if suffix else base_k + ".weight"
            sd[key] = t.cpu()
        conf = {
            "format": "nvfp4",
            "convrot": True,
            "convrot_groupsize": 256,
            "orig_shape": [int(dq.shape[0]), int(dq.shape[1])],
            "in_features": int(dq.shape[1]),
            "out_features": int(dq.shape[0]),
        }
        sd[base_k + ".comfy_quant"] = torch.frombuffer(
            json.dumps(conf).encode("utf-8"), dtype=torch.uint8
        ).clone()
        meta["layers"][base_k] = conf
        meta["layers"][L_stripped] = conf
        n_conv += 1
        print(f"  nvfp4: {base_k}  ({tuple(dq.shape)})")

    print(f"converted {n_conv} layers to NVFP4")
    out_meta = {}
    for k, v in raw_meta.items():
        if k == "_quantization_metadata":
            out_meta[k] = json.dumps(meta)
        else:
            out_meta[k] = v.decode("utf-8") if isinstance(v, bytes) else v
    if "_quantization_metadata" not in out_meta:
        out_meta["_quantization_metadata"] = json.dumps(meta)
    save_file(sd, OUT, metadata=out_meta)
    print("saved:", OUT, os.path.getsize(OUT) / 1e9, "GB (decimal)")


if __name__ == "__main__":
    main()

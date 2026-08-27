# -*- coding: utf-8 -*-
"""Reverse hybrid converter: sci_1off (all INT8, already rotated) -> convert K lowest-impact layers to ConvRot NVFP4.

Reverse hybrid NVFP4 method (see md/How to quantize Z Image - Hybrid NVFP4.md).
Requires the pip package `comfy-kitchen` (pip install comfy-kitchen).
Weights in the sci_1off INT8 artifact are already ROTATED (W@H^T): dequant gives the W_rot
approximation, which is quantized directly with Kitchen (NO re-rotation).

Usage:
    python Z_Image/gen_reverse_nvfp4.py <K> <out_name.safetensors> <src_int8.safetensors> <impact.json> \
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


def parse_args():
    ap = argparse.ArgumentParser(description="Reverse hybrid NVFP4 converter (INT8 -> NVFP4)")
    ap.add_argument("k", type=int, help="number of lowest-impact layers to convert to NVFP4")
    ap.add_argument("out_name", help="output filename, e.g. <model>_hswq_hybrid_nv74_convrot_nvfp4.safetensors")
    ap.add_argument("src", help="sci_1off complete ConvRot INT8 safetensors")
    ap.add_argument("impact", help="impact json from diag_impact.py")
    ap.add_argument("--out-dir", default=".", help="output directory")
    return ap.parse_args()


def _find_prefix(keys):
    for prefix in ("model.diffusion_model.", "diffusion_model.", ""):
        for suffix in (
            "layers.0.attention.qkv.weight",
            "blocks.0.attn.qkv.weight",
            "cap_embedder.1.weight",
            "x_embedder.proj.weight",
        ):
            if f"{prefix}{suffix}" in keys:
                return prefix
    for k in keys:
        if k.endswith(".weight"):
            if k.startswith("model.diffusion_model."):
                return "model.diffusion_model."
            if k.startswith("diffusion_model."):
                return "diffusion_model."
    return ""


def _strip_prefix(key, prefix):
    if prefix and key.startswith(prefix):
        return key[len(prefix):]
    return key


def main():
    a = parse_args()
    OUT = os.path.join(a.out_dir, a.out_name)

    imp = json.load(open(a.impact, encoding="utf-8"))["impacts"]
    # impact keys carry no suffix in diag_impact.py output; normalize defensively anyway
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

    prefix = _find_prefix(keys)
    print(f"Z Image key prefix: {prefix!r}")

    n_conv = 0
    for L in ranked[:a.k]:
        L_stripped = _strip_prefix(L, prefix)
        wk = prefix + L_stripped + ".weight"
        sk = prefix + L_stripped + ".weight_scale"
        if wk not in sd:
            wk = L + ".weight"
            sk = L + ".weight_scale"
        if wk not in sd:
            print(f"  SKIP (not in sd): {L} (tried {prefix + L_stripped + '.weight'})")
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
        conf = {"format": "nvfp4", "convrot": True, "convrot_groupsize": 256}
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

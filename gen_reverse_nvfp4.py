# -*- coding: utf-8 -*-
"""Reverse hybrid converter: sci_1off (all INT8, already rotated) -> convert K lowest-impact layers to ConvRot NVFP4.

Reverse hybrid NVFP4 method (see md/How to quantize Z Image - Hybrid NVFP4.md).
Weights in the sci_1off INT8 artifact are already ROTATED (W@H^T): dequant gives the W_rot
approximation, which is quantized directly with Kitchen (NO re-rotation).

Usage:
    python gen_reverse_nvfp4.py <K> <out_name.safetensors> <src_int8.safetensors> <impact.json> \
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # repo root (comfy_kitchen)
from comfy_kitchen.tensor import TensorCoreNVFP4Layout


def parse_args():
    ap = argparse.ArgumentParser(description="Reverse hybrid NVFP4 converter (INT8 -> NVFP4)")
    ap.add_argument("k", type=int, help="number of lowest-impact layers to convert to NVFP4")
    ap.add_argument("out_name", help="output filename, e.g. <model>_hswq_hybrid_nv74_convrot_nvfp4.safetensors")
    ap.add_argument("src", help="sci_1off complete ConvRot INT8 safetensors")
    ap.add_argument("impact", help="impact json from diag_impact.py")
    ap.add_argument("--out-dir", default=".", help="output directory")
    return ap.parse_args()


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
        meta = json.loads(f.metadata()["_quantization_metadata"])
        sd = {k: f.get_tensor(k) for k in keys}

    n_conv = 0
    for L in ranked[:a.k]:
        wk, sk = L + ".weight", L + ".weight_scale"
        if wk not in sd:
            print(f"  SKIP (not in sd): {L}")
            continue
        q = sd[wk]            # I8 rotated
        s = sd[sk]            # F32 [out,1]
        dq = (q.float() * s)  # W_rot approx in fp32
        w_for_q = dq.to(torch.bfloat16)
        qdata, params = TensorCoreNVFP4Layout.quantize(w_for_q)
        tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)
        del sd[wk], sd[sk]
        for suffix, t in tensors.items():
            key = L + ".weight" + suffix if suffix else L + ".weight"
            sd[key] = t.cpu()
        conf = {"format": "nvfp4", "convrot": True, "convrot_groupsize": 256}
        sd[L + ".comfy_quant"] = torch.frombuffer(
            json.dumps(conf).encode("utf-8"), dtype=torch.uint8
        ).clone()
        meta["layers"][L] = conf
        n_conv += 1
        print(f"  nvfp4: {L}  ({dq.shape})")

    print(f"converted {n_conv} layers to NVFP4")
    save_file(sd, OUT, metadata={"_quantization_metadata": json.dumps(meta)})
    print("saved:", OUT, os.path.getsize(OUT) / 1e9, "GB (decimal)")


if __name__ == "__main__":
    main()

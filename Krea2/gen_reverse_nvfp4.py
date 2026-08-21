# -*- coding: utf-8 -*-
"""Krea2 reverse hybrid converter: complete ConvRot INT8 -> convert K
lowest-impact layers to ConvRot NVFP4 (weights STAY rotated).

Krea2 port of Z Image gen_reverse_nvfp4.py. Weights in the INT8 artifact are
already ROTATED (W@H^T): dequant (q * scale) gives the W_rot approximation,
which is quantized directly with Kitchen (NO re-rotation). Converted layers
keep convrot:true, so runtime online act-rotation still applies.

This is the REVERSE (trajectory) method output, NOT the 4-axis "plain nvfp4"
output (auto_int8_nvfp4_hybrid.py). Do not mix.

Usage:
    python Krea2/gen_reverse_nvfp4.py <K> <out_name.safetensors> \
        <src_int8.safetensors> <impact.json> [--out-dir <dir>]

Storage spec (Kitchen TensorCoreNVFP4Layout): .weight U8 packed [out, in/2],
.weight_scale F8_E4M3 [out, in/16], .weight_scale_2 F32, .comfy_quant U8 JSON.
"""
import argparse
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

try:
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout
    from comfy_kitchen.float_utils import F4_E2M1_MAX, F8_E4M3_MAX
except ImportError:
    print("Error: comfy_kitchen not found (install in the active venv).")
    sys.exit(1)


def _find_prefix(keys):
    for prefix in ("model.diffusion_model.", "diffusion_model.", ""):
        if f"{prefix}txtfusion.projector.weight" in keys:
            return prefix
    raise ValueError("Not a Krea2 checkpoint: missing txtfusion.projector.weight")


def _strip_prefix(key, prefix):
    if prefix and key.startswith(prefix):
        return key[len(prefix):]
    return key


# Krea2 NVFP4-safe in_features set: only these dimensions produce packed
# weights compatible with the NVFP4 loader.  All other Linear (txtfusion etc.)
# must NEVER be converted.  diag_impact.py enforces the same rule.
_SAFE_IN_FEATURES = {1536, 6144, 16384}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Krea2 reverse hybrid NVFP4 converter (INT8 -> NVFP4)"
    )
    ap.add_argument("k", type=int,
                    help="number of lowest-impact layers to convert to NVFP4")
    ap.add_argument("out_name",
                    help="output filename, e.g. <model>_hswq_r<K>_1off_convrot_nvfp4.safetensors")
    ap.add_argument("src", help="complete ConvRot INT8 safetensors")
    ap.add_argument("impact", help="impact json from diag_impact.py")
    ap.add_argument("--out-dir", default=".", help="output directory")
    return ap.parse_args()


def main():
    a = parse_args()
    OUT = os.path.join(a.out_dir, a.out_name)

    data = json.load(open(a.impact, encoding="utf-8"))
    imp = data["impacts"]
    act_amax = data.get("act_amax", {})
    # impact keys carry no suffix in diag_impact.py output; normalize defensively.
    ranked = [k[:-len(".weight")] if k.endswith(".weight") else k
              for k, _ in sorted(imp.items(), key=lambda kv: kv[1])]
    print(f"ranked layers available: {len(ranked)}")

    with safe_open(a.src, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        raw_meta = f.metadata()
        meta = json.loads(raw_meta["_quantization_metadata"])
        sd = {k: f.get_tensor(k) for k in keys}

    prefix = _find_prefix(keys)
    print(f"Krea2 key prefix: {prefix!r}")

    n_conv = 0
    for L in ranked[:a.k]:
        # L is a STRIPPED layer key (e.g. "blocks.0.attn.gate"); defensively
        # strip any prefix that may have leaked in from another impact source.
        L = _strip_prefix(L, prefix)
        wk, sk = prefix + L + ".weight", prefix + L + ".weight_scale"
        if wk not in sd:
            print(f"  SKIP (not in sd): {L}")
            continue
        # Enforce NVFP4-safe in_features rule (must match diag_impact.py).
        in_f = int(sd[wk].shape[1])
        if in_f not in _SAFE_IN_FEATURES:
            continue
        q = sd[wk]            # I8 rotated
        s = sd[sk]            # F32 scale ([out,1] row-wise, or scalar)
        dq = (q.float() * s)  # W_rot approx in fp32
        w_for_q = dq.to(torch.bfloat16)
        qdata, params = TensorCoreNVFP4Layout.quantize(w_for_q)
        tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)
        del sd[wk], sd[sk]
        for suffix, t in tensors.items():
            key = wk + suffix
            sd[key] = t.cpu()
        conf = {"format": "nvfp4", "convrot": True, "convrot_groupsize": 256}
        sd[prefix + L + ".comfy_quant"] = torch.frombuffer(
            json.dumps(conf).encode("utf-8"), dtype=torch.uint8
        ).clone()
        # convrot NVFP4 activation scale (reference converter writes this;
        # missing .input_scale falls back to runtime per-call amax and loses quality).
        amax = act_amax.get(L)
        if amax is not None:
            denom = float(F8_E4M3_MAX) * float(F4_E2M1_MAX)
            sd[prefix + L + ".input_scale"] = torch.tensor(
                max(float(amax), 1e-12) / denom, dtype=torch.float32
            )
        else:
            print(f"  WARN no act_amax for {L}: .input_scale omitted (runtime amax fallback)")
        meta["layers"][L] = conf
        n_conv += 1
        print(f"  nvfp4: {L}  ({tuple(dq.shape)})")

    print(f"converted {n_conv} layers to NVFP4")

    out_meta = {}
    for k, v in raw_meta.items():
        if k == "_quantization_metadata":
            out_meta[k] = json.dumps(meta)
        else:
            out_meta[k] = v.decode("utf-8") if isinstance(v, bytes) else v
    save_file(sd, OUT, metadata=out_meta)
    print("saved:", OUT, os.path.getsize(OUT) / 1e9, "GB (decimal)")


if __name__ == "__main__":
    main()

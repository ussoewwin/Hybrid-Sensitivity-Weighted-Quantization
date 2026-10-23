# -*- coding: utf-8 -*-
"""Qwen-Image Fun ControlNet (Union 2602): ConvRot INT8 pack builder.

The protection set comes from a measured aggregate contribution on THIS
checkpoint (`sdxl/measure_qwen_image_fun_cn_contrib.py`): the top-K layers by
`1 - cos` of the ControlNet's own output are kept in FP16, the rest goes to
ConvRot INT8.

The source checkpoint is already in the runtime layout (`control_blocks.*`,
`control_img_in.*`), which is what `load_controlnet_qwen_fun` reads - no name
remap is needed.

Rules:
  1. Only 2D weights are quantized.
  2. Group size 256 (falling back to 4 where the input width demands it).
  3. Kept tensors are stored FP16 (the runtime runs FP16; FP32 is wasted bytes).
     The INT8 `weight_scale` stays FP32 by design.

Usage:
  python clip_convert/convert_qwen_image_fun_cn_impact_convrot_int8.py \
      --input <controlnet>.safetensors \
      --output <out>.safetensors \
      --contrib impact/contrib_qwen_image_fun_cn_2602.json \
      --top-k 40
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch
from safetensors.torch import save_file

GROUP_SIZE = 256


def build_hadamard(size, dtype=torch.float32):
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError("Hadamard size must be a power of 4, got %s" % size)
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return h / (size ** 0.5)


def group_size_for(in_features, out_features):
    gs = GROUP_SIZE
    while gs >= 4:
        if (in_features % gs == 0 and math.log(gs, 4) % 1 == 0
                and out_features % 4 == 0 and in_features % 4 == 0):
            return gs
        gs //= 4
    return None


def rotate_weight(w, h, gs):
    of, inf = w.shape
    g = inf // gs
    return torch.matmul(w.view(of, g, gs), h.T.to(w.dtype)).reshape(w.shape)


def quant_rowwise(w):
    amax = w.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
    s = amax / 127.0
    q = (w / s.to(w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, s.to(torch.float32)


def _encode(config):
    return torch.tensor(list(json.dumps(config, separators=(",", ":")).encode("utf-8")), dtype=torch.uint8)


def convert(input_path, output_path, contrib_json, top_k):
    from safetensors.torch import load_file
    src = load_file(input_path)

    contrib = json.load(open(contrib_json, encoding="utf-8"))["impacts"]
    ranked = sorted(((n, v) for n, v in contrib.items() if v is not None),
                    key=lambda kv: -kv[1])
    protect = {n for n, _ in ranked[:top_k]}
    print("the measured candidates: %d | the adopted protection: %d" % (len(ranked), len(protect)))

    h = build_hadamard(GROUP_SIZE)
    out, meta_layers = {}, {}
    n_lin = n_prot = n_kept = 0
    for name in sorted(src):
        t = src[name]
        if not (name.endswith(".weight") and t.ndim == 2):
            out[name] = t.to(torch.float16) if t.is_floating_point() else t
            n_kept += 1
            continue
        mk = name[: -len(".weight")]
        if mk in protect:
            out[name] = t.to(torch.float16)
            n_prot += 1
            continue
        of, inf = int(t.shape[0]), int(t.shape[1])
        gs = group_size_for(inf, of)
        if gs is None:
            out[name] = t.to(torch.float16)
            n_prot += 1
            continue
        hh = h if gs == GROUP_SIZE else build_hadamard(gs)
        q, scale = quant_rowwise(rotate_weight(t.float(), hh, gs))
        n_lin += 1
        conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": gs}
        out[name] = q
        out["%s.weight_scale" % mk] = scale
        out["%s.comfy_quant" % mk] = _encode(conf)
        meta_layers[mk] = conf

    metadata = {"_quantization_metadata": json.dumps(
        {"format_version": "1.0", "layers": meta_layers}, separators=(",", ":"))}
    save_file(out, output_path, metadata=metadata)
    print("  the ConvRot INT8: %d" % n_lin)
    print("  the protected: %d" % n_prot)
    print("  the kept (non-2D): %d" % n_kept)
    print("  the output: %.1f MB" % (os.path.getsize(output_path) / 1024 ** 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--contrib", required=True)
    ap.add_argument("--top-k", type=int, default=0)
    a = ap.parse_args()
    convert(a.input, a.output, a.contrib, a.top_k)

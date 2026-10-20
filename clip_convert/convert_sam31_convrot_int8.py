# -*- coding: utf-8 -*-
"""SAM 3.1 Multiplex ConvRot INT8 converter driven by measured trajectory impact.

The protection set comes from a trajectory-impact measurement of THIS
checkpoint (sibling script sdxl/diag_impact_sam31.py), not from static weight
statistics. For SAM the measured quantity is the drift of the model's own
output (mask logits from the tracker SAM-decoder path + the detector's
boxes/scores/masks from the text path) when a single layer is replaced by its
ConvRot INT8 reconstruction.

Because a SAM model produces masks rather than a denoising trajectory, the
drift scale is different from the ControlNet case: the measured distribution
runs from 0 to ~10.6, with a median of 2.4e-03, so the threshold has to be
chosen on this model's own distribution (see the shipped default).

Usage:
  python clip_convert/convert_sam31_convrot_int8.py \
      --input sam3.1_multiplex_fp16.safetensors \
      --output sam3.1_multiplex_convrot_int8_t005.safetensors \
      --impact impact/impact_sam3.1_multiplex.json \
      --drift-threshold 0.05
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_GROUP = 256
DEFAULT_DRIFT_THRESHOLD = 0.05


def build_hadamard(size, dtype=torch.float32):
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def convrot_group_size(n, preferred=256):
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def rotate_weight(w, h, gs):
    of, inf = w.shape
    g = inf // gs
    return torch.matmul(w.view(of, g, gs), h.T.to(w.dtype)).reshape(w.shape)


def rotate_weight_conv2d(w, h, gs):
    oc, ic, kh, kw = w.shape
    flat = w.permute(0, 2, 3, 1).contiguous().view(-1, ic)
    return rotate_weight(flat, h, gs).view(oc, kh, kw, ic).permute(0, 3, 1, 2).contiguous()


def quant_rowwise(w):
    amax = w.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
    s = amax / 127.0
    q = (w / s.to(w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, s.to(torch.float32)


def quant_channelwise(w):
    rd = tuple(range(1, w.dim()))
    amax = torch.clamp(w.abs().amax(dim=rd), min=1e-30).float()
    s = amax / 127.0
    v = s.view(-1, *([1] * (w.dim() - 1)))
    av = amax.view(-1, *([1] * (w.dim() - 1)))
    q = (torch.clamp(w, -av, av) / v).round().clamp(-127, 127).to(torch.int8)
    return q, s.to(torch.float32)


def _encode(config):
    return torch.tensor(list(json.dumps(config, separators=(",", ":")).encode("utf-8")), dtype=torch.uint8)


def _preprocess_sam31(keys, tensors):
    """SAM preprocessing: drop freqs_cis buffers, keep everything else.

    The standalone SAM 3.1 checkpoint already stores the split q/k/v layout
    (the fused in_proj split is applied by the generic clip converter for
    sam3.pt; sam3.1_multiplex stores q_proj/k_proj/v_proj directly), so no key
    surgery is needed here beyond dropping the rotary buffers.
    """
    out = {}
    for k, v in tensors.items():
        if ".attn.freqs_cis" in k:
            continue
        out[k] = v
    return out


def convert(input_path, output_path, impact_json, group_size=DEFAULT_GROUP,
            drift_threshold=DEFAULT_DRIFT_THRESHOLD):
    print(f"Loading (streamed): {input_path}")
    src = safe_open(input_path, framework="pt", device="cpu")
    keys = sorted(src.keys())
    print(f"  raw keys: {len(keys)}")

    impacts = json.load(open(impact_json, encoding="utf-8"))["impacts"]
    # the impact JSON stores runtime module names ("<path>.<layer>"); the
    # checkpoint stores the same paths plus ".weight", so the mapping is direct.
    protect = set()
    for n, drift in impacts.items():
        if drift > drift_threshold:
            protect.add(n)
    print(f"  measured candidates: {len(impacts)}; drift > {drift_threshold} -> protected {len(protect)}")
    for p in sorted(protect)[:15]:
        print(f"    {p}  ({impacts[p]:.4e})")
    if len(protect) > 15:
        print(f"    ... ({len(protect) - 15} more)")

    tensors = {k: src.get_tensor(k) for k in keys}
    tensors = _preprocess_sam31(keys, tensors)

    out = {}
    meta_layers = {}
    n_lin = n_conv = n_prot = n_kept = 0
    h_cache = {}
    for key in sorted(tensors):
        t = tensors[key]
        if not (key.endswith(".weight") and t.ndim in (2, 4)
                and t.dtype in (torch.float16, torch.float32, torch.bfloat16)):
            out[key] = t
            n_kept += 1
            continue
        mk = key[: -len(".weight")]
        if mk in protect:
            out[key] = t
            n_prot += 1
            continue
        w = t.float()
        gs = convrot_group_size(w.shape[1], group_size)
        if gs is None:
            q, scale = quant_rowwise(w)
            conf = {"format": "int8_tensorwise"}
        else:
            if gs not in h_cache:
                h_cache[gs] = build_hadamard(gs)
            if t.ndim == 2:
                q, scale = quant_rowwise(rotate_weight(w, h_cache[gs], gs))
                n_lin += 1
            else:
                q, scale = quant_channelwise(rotate_weight_conv2d(w, h_cache[gs], gs))
                n_conv += 1
            conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": int(gs)}
        out[key] = q
        out[f"{mk}.weight_scale"] = scale
        out[f"{mk}.comfy_quant"] = _encode(conf)
        meta_layers[mk] = conf

    metadata = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": meta_layers}, separators=(",", ":"))}
    print(f"\nSaving: {output_path}")
    print(f"  ConvRot INT8 Linear: {n_lin}  Conv2d: {n_conv}  protected: {n_prot}  kept: {n_kept}")
    save_file(out, output_path, metadata=metadata)
    print(f"  Input {os.path.getsize(input_path)/(1024**3):.2f} GB -> Output {os.path.getsize(output_path)/(1024**3):.2f} GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SAM 3.1 ConvRot INT8 converter with measured trajectory protection")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--impact", required=True)
    ap.add_argument("--groupsize", type=int, default=DEFAULT_GROUP)
    ap.add_argument("--drift-threshold", type=float, default=DEFAULT_DRIFT_THRESHOLD)
    a = ap.parse_args()
    convert(a.input, a.output, a.impact, group_size=a.groupsize, drift_threshold=a.drift_threshold)

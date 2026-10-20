# -*- coding: utf-8 -*-
"""CLIP / text-encoder ConvRot INT8 converter driven by measured impact.

The protection set comes from an impact measurement of THIS checkpoint
(sibling script sdxl/diag_impact_clip.py): the relative MSE of the CLIP's own
text-embedding output when a single layer is replaced by its ConvRot INT8
reconstruction. Static weight statistics (kurtosis) are not used as the
protection criterion - an earlier revision of the CLIP packs did that, and its
kurtosis numbers were additionally invalid due to a safetensors header offset
bug, so those packs had no valid protection basis.

Rules (the same runtime constraints established for ControlNet and SAM):

1. Only modules the runtime actually has are quantized, and only 2D
   `nn.Linear` layers - ComfyUI's INT8 load path attaches a QuantizedTensor to
   Linear layers only.
2. The group size is fixed at 256; a Linear whose input width is not a
   multiple of 256 is kept in original precision (the ConvRot kernel is 2D-only
   and fused only at 256).
3. The pack is written under the checkpoint's own parameter names, which is
   what ComfyUI's CLIP load path expects (verified against the FP16 layout:
   removing no key and adding only `.weight_scale` / `.comfy_quant`).

Usage:
  python clip_convert/convert_clip_impact_convrot_int8.py \
      --input <clip>.safetensors --output <out>.safetensors \
      --impact impact/impact_clip_l.json \
      --comfy_path <ComfyUI checkout> \
      --drift-threshold 0.15
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

GROUP_SIZE = 256
DEFAULT_DRIFT_THRESHOLD = 0.15


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


def runtime_linear_modules(comfy_path, model_path, clip_type="stable_diffusion"):
    """Return the set of Linear module names the CLIP runtime builds."""
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)
    cwd = os.getcwd()
    try:
        os.chdir(comfy_path)
        import comfy.sd
        sv = sys.argv
        sys.argv = [sv[0]]
        try:
            clip = comfy.sd.load_clip(ckpt_paths=[os.path.abspath(model_path)], embedding_directory=None,
                                      clip_type=getattr(comfy.sd.CLIPType, clip_type.upper()))
        finally:
            sys.argv = sv
        net = clip.cond_stage_model
        inner = getattr(net, "clip_l", net)
        names = {n for n, m in inner.named_modules() if type(m).__name__ == "Linear"}
        return names
    finally:
        os.chdir(cwd)


def runtime_to_checkpoint(runtime_name, ck_keys):
    """Map a runtime module name to the checkpoint's parameter name.

    CLIP-L runtime names carry a `transformer.` prefix that the checkpoint does
    not (`transformer.text_model.encoder...` vs `text_model.encoder...`). The
    mapping is verified by matching against the checkpoint's own keys."""
    candidates = [runtime_name]
    for pre in ("transformer.", "cond_stage_model.", "clip_l."):
        if runtime_name.startswith(pre):
            candidates.append(runtime_name[len(pre):])
    for cand in candidates:
        if (cand + ".weight") in ck_keys:
            return cand
    # fall back: strip any leading segment until it matches
    parts = runtime_name.split(".")
    for i in range(1, len(parts)):
        cand = ".".join(parts[i:])
        if (cand + ".weight") in ck_keys:
            return cand
    return None


def convert(input_path, output_path, impact_json, comfy_path, clip_type,
            drift_threshold=DEFAULT_DRIFT_THRESHOLD):
    print(f"Loading (streamed): {input_path}")
    src = safe_open(input_path, framework="pt", device="cpu")
    keys = sorted(src.keys())
    ck_keys = set(keys)

    impacts = json.load(open(impact_json, encoding="utf-8"))["impacts"]
    runtime_lin = runtime_linear_modules(comfy_path, os.path.abspath(input_path), clip_type)
    print(f"  runtime Linear modules: {len(runtime_lin)}")

    # map runtime -> checkpoint, build the protect set in checkpoint names
    rt2ck = {}
    for n in runtime_lin:
        ck = runtime_to_checkpoint(n, ck_keys)
        if ck:
            rt2ck[n] = ck
    print(f"  runtime->checkpoint mapped: {len(rt2ck)}")

    protect = set()
    for n, drift in impacts.items():
        if drift <= drift_threshold:
            continue
        ck = rt2ck.get(n) or runtime_to_checkpoint(n, ck_keys)
        if ck:
            protect.add(ck)
    print(f"  measured candidates: {len(impacts)}; drift > {drift_threshold} -> protected {len(protect)}")

    h = build_hadamard(GROUP_SIZE)
    out = {}
    meta_layers = {}
    n_lin = n_prot = n_kept = 0
    for key in keys:
        t = src.get_tensor(key)
        if not (key.endswith(".weight") and t.ndim == 2
                and t.dtype in (torch.float16, torch.float32, torch.bfloat16)):
            out[key] = t
            n_kept += 1
            continue
        mk = key[: -len(".weight")]
        # only quantize modules that both exist at runtime and are Linear
        if mk not in rt2ck.values():
            out[key] = t
            n_kept += 1
            continue
        if mk in protect:
            out[key] = t
            n_prot += 1
            continue
        inf = int(t.shape[1]); of = int(t.shape[0])
        if inf % GROUP_SIZE != 0 or of % 4 != 0 or inf % 4 != 0:
            out[key] = t
            n_prot += 1
            continue
        q, scale = quant_rowwise(rotate_weight(t.float(), h, GROUP_SIZE))
        n_lin += 1
        conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP_SIZE}
        out[key] = q
        out[f"{mk}.weight_scale"] = scale
        out[f"{mk}.comfy_quant"] = _encode(conf)
        meta_layers[mk] = conf

    metadata = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": meta_layers}, separators=(",", ":"))}
    print(f"Saving: {output_path}")
    print(f"  ConvRot INT8 Linear: {n_lin}")
    print(f"  protected (drift or shape): {n_prot}")
    print(f"  kept (not runtime Linear / non-2D): {n_kept}")
    save_file(out, output_path, metadata=metadata)
    print(f"  Input {os.path.getsize(input_path)/(1024**2):.1f} MB -> Output {os.path.getsize(output_path)/(1024**2):.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CLIP ConvRot INT8 converter with measured impact protection")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--impact", required=True)
    ap.add_argument("--comfy_path", required=True)
    ap.add_argument("--clip_type", default="stable_diffusion")
    ap.add_argument("--drift-threshold", type=float, default=DEFAULT_DRIFT_THRESHOLD)
    a = ap.parse_args()
    convert(a.input, a.output, a.impact, a.comfy_path, a.clip_type, drift_threshold=a.drift_threshold)

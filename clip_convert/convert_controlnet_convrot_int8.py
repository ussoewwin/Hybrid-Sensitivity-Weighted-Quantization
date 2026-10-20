# -*- coding: utf-8 -*-
"""ControlNet ConvRot INT8 converter driven by measured trajectory impact.

Individual decision per checkpoint: the protection set comes from a trajectory
impact JSON measured on THIS checkpoint (sibling script diag_impact_controlnet.py).

**Key naming is the whole point of this converter.** ComfyUI loads an SDXL
ControlNet through its *diffusers* branch: it remaps the checkpoint's
diffusers-style names (`controlnet_down_blocks.N.weight`,
`controlnet_cond_embedding.blocks.N.weight`, ...) to the comfy-style names the
runtime actually builds (`zero_convs.N.0.weight`, `input_hint_block.N.weight`,
`input_blocks.*`, ...). A quantized pack must therefore be written under the
**runtime (comfy) names**, exactly like the checkpoint ComfyUI loads. Writing
the quantized weights under the checkpoint's original diffusers names produced
a pack where 340 of 346 quantized modules did not exist in the runtime graph:
the `.comfy_quant` / `.weight_scale` companions were left behind by the remap,
the runtime saw a bare int8 tensor, and the model output NaN (a black image).

This converter therefore builds the ControlNet the way the runtime does, takes
the runtime module graph as the source of truth, and quantizes the runtime's
2D Linear layers. The impact JSON stores comfy (runtime) names, so the
protection lookup is direct.

Rules (same as the SAM converter, both learned from measured failures):

1. **Only 2D `nn.Linear` layers are quantized.** ComfyUI's INT8 load path
   attaches a QuantizedTensor to Linear layers only.
2. **Group size is fixed at 256.** The ConvRot kernel's fused path activates
   only at 256; any other value would be read back unrotated.
3. **Keys are written under the runtime names.**

Usage:
  python clip_convert/convert_controlnet_convrot_int8.py \
      --input <controlnet>.safetensors \
      --output <out>.safetensors \
      --impact impact/impact_<name>.json \
      --comfy_path <ComfyUI checkout> \
      --model <SDXL base checkpoint> \
      --drift-threshold 0.15
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
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


def runtime_view(comfy_path, controlnet_path, model_path):
    """Build the ControlNet exactly as ComfyUI does and return
    (usable_linear_names, runtime_state_dict)."""
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)
    cwd = os.getcwd()
    try:
        os.chdir(comfy_path)
        import comfy.sd
        import comfy.controlnet as cc
        sv = sys.argv
        sys.argv = [sv[0]]
        try:
            o = comfy.sd.load_checkpoint_guess_config(model_path, output_vae=False,
                                                      output_clip=False, embedding_directory=None)
            cnet = cc.load_controlnet(controlnet_path, model=o[0].model)
        finally:
            sys.argv = sv
        cm = cnet.control_model
        msd = cm.state_dict()
        types = {n: type(m).__name__ for n, m in cm.named_modules()}
        usable = {n for n, t in types.items()
                  if t == "Linear" and (n + ".weight") in msd}
        return usable, msd
    finally:
        os.chdir(cwd)


def convert(input_path, output_path, impact_json, comfy_path, model_path,
            drift_threshold=DEFAULT_DRIFT_THRESHOLD):
    impacts = json.load(open(impact_json, encoding="utf-8"))["impacts"]
    usable, msd = runtime_view(comfy_path, os.path.abspath(input_path), os.path.abspath(model_path))
    print(f"runtime Linear modules with weight: {len(usable)}")

    protect = {n for n, drift in impacts.items() if drift > drift_threshold}
    print(f"measured candidates: {len(impacts)}; drift > {drift_threshold} -> protected {len(protect)}")

    h = build_hadamard(GROUP_SIZE)
    out = {}
    meta_layers = {}
    n_lin = n_prot = n_kept = 0
    for name in sorted(msd):
        t = msd[name]
        if not (name.endswith(".weight") and t.ndim == 2):
            out[name] = t.to(torch.float16) if t.is_floating_point() else t
            n_kept += 1
            continue
        mk = name[: -len(".weight")]
        if mk not in usable:
            out[name] = t.to(torch.float16)
            n_kept += 1
            continue
        if mk in protect:
            out[name] = t.to(torch.float16)
            n_prot += 1
            continue
        inf = int(t.shape[1]); of = int(t.shape[0])
        if inf % GROUP_SIZE != 0 or of % 4 != 0 or inf % 4 != 0:
            out[name] = t.to(torch.float16)
            n_prot += 1
            continue
        q, scale = quant_rowwise(rotate_weight(t.float(), h, GROUP_SIZE))
        n_lin += 1
        conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP_SIZE}
        out[name] = q
        out[f"{mk}.weight_scale"] = scale
        out[f"{mk}.comfy_quant"] = _encode(conf)
        meta_layers[mk] = conf

    metadata = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": meta_layers}, separators=(",", ":"))}
    print(f"Saving: {output_path}")
    print(f"  ConvRot INT8 Linear: {n_lin}")
    print(f"  protected (drift or shape): {n_prot}")
    print(f"  kept (not runtime Linear): {n_kept}")
    save_file(out, output_path, metadata=metadata)
    print(f"  Output {os.path.getsize(output_path)/(1024**3):.2f} GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ControlNet ConvRot INT8 converter (runtime key naming)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--impact", required=True)
    ap.add_argument("--comfy_path", required=True)
    ap.add_argument("--model", required=True, help="base checkpoint the ControlNet is loaded with")
    ap.add_argument("--drift-threshold", type=float, default=DEFAULT_DRIFT_THRESHOLD)
    a = ap.parse_args()
    convert(a.input, a.output, a.impact, a.comfy_path, a.model, drift_threshold=a.drift_threshold)

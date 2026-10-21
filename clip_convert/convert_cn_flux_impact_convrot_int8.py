# -*- coding: utf-8 -*-
"""FLUX ControlNet (Union Pro) ConvRot INT8 converter, driven by measured impact.

The pack is written under the **runtime** parameter names - the ones
`comfy.ldm.flux.controlnet.ControlNetFlux.state_dict()` reports - because
that is what the HSWQ ControlNet loader builds the model from. The runtime
reads exactly 174 tensors; the pack carries those and nothing else.

(The source checkpoint is in the diffusers layout and the loader converts it
with `comfy.model_detection.convert_diffusers_mmdit`; that converted state
dict holds every tensor twice - once under the diffusers name and once under
the runtime name, as the same object - so writing it verbatim would carry the
weights twice.)

Two kinds of Linear layers are kept in FP16:

* the **fused qkv** layers - the runtime builds one Linear out of
  cat([to_q, to_k, to_v], dim=0), and the three halves have value ranges that
  differ by ~7x (absmax 2.48 / 0.53 / 0.375). One per-out-channel scale
  cannot serve them; the measured reconstruction error of the fused tensor is
  1.89, i.e. a broken layer.
* the layers whose measured drift is 0.0000 **because they are not on the
  output path** (verified by replacing their whole weight with 0.5 and
  confirming the output does not change).

Usage:
  python clip_convert/convert_cn_flux_impact_convrot_int8.py \
      --input <controlnet>.safetensors --output <out>.safetensors \
      --impact impact/impact_cn_flux_union_pro_2.json \
      --comfy_path <ComfyUI checkout> --drift-threshold <value>
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


def group_size_for(n, preferred=256):
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def rotate_weight(w, h, gs):
    of, inf = w.shape
    if inf % gs != 0:
        raise ValueError(f"in_features {inf} not divisible by group size {gs}")
    g = inf // gs
    return torch.matmul(w.view(of, g, gs), h.T.to(w.dtype)).reshape(w.shape)


def quant_rowwise(w):
    amax = w.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
    scale = amax / 127.0
    q = (w / scale.to(w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def _encode(conf):
    return torch.tensor(list(json.dumps(conf, separators=(",", ":")).encode("utf-8")), dtype=torch.uint8)


def runtime_model(comfy_path, model_path):
    """The runtime ControlNet, built the way the loader builds it."""
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)
    cwd = os.getcwd()
    try:
        os.chdir(comfy_path)
        import comfy.utils
        import comfy.model_detection
        import comfy.controlnet
        import comfy.ldm.flux.controlnet as cnf
        from comfy import ops as comfy_ops
        from comfy.quant_ops import QUANT_ALGOS
        sd = comfy.utils.load_torch_file(os.path.abspath(model_path), safe_load=True)
        converted = comfy.model_detection.convert_diffusers_mmdit(dict(sd), "")
        for k in sd:
            converted[k] = sd[k]
        mc = comfy.model_detection.model_config_from_unet(converted, "", use_base_if_no_match=True)
        if mc is None:
            raise SystemExit("could not detect the model config")
        operations = comfy_ops.mixed_precision_ops(
            {"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]},
            torch.bfloat16, full_precision_mm=False, disabled=[])
        model = cnf.ControlNetFlux(
            latent_input=True, num_union_modes=0,
            control_latent_channels=converted["pos_embed_input.weight"].shape[1] // 4,
            operations=operations, device="cpu", dtype=torch.bfloat16, **mc.unet_config)
        model = comfy.controlnet.controlnet_load_state_dict(model, converted)
        return model
    finally:
        os.chdir(cwd)


def convert(input_path, output_path, impact_json, comfy_path,
            drift_threshold=DEFAULT_DRIFT_THRESHOLD, quantize_qkv=False):
    print(f"Loading: {input_path}", flush=True)
    model = runtime_model(comfy_path, input_path)
    sd = model.state_dict()
    print(f"  the runtime state dict: {len(sd)} tensors", flush=True)

    lins = {n: m for n, m in model.named_modules()
            if type(m).__name__ == "Linear" and getattr(m, "weight", None) is not None}
    print(f"  the runtime Linear modules: {len(lins)}", flush=True)

    impacts = json.load(open(impact_json, encoding="utf-8"))["impacts"]
    protect = {n for n, drift in impacts.items() if drift > drift_threshold}
    if not quantize_qkv:
        # the fused qkv layers cannot be quantized: see the module docstring
        protect |= {n for n in lins if "qkv" in n}
    print(f"  measured {len(impacts)}; drift > {drift_threshold} -> protected {len(protect)}", flush=True)

    h_cache = {}

    def had(gs):
        if gs not in h_cache:
            h_cache[gs] = build_hadamard(gs)
        return h_cache[gs]

    out = {}
    meta_layers = {}
    n_lin = n_prot = n_kept = 0
    for k, v in sd.items():
        mk = k[: -len(".weight")] if k.endswith(".weight") else k
        if mk in lins and k.endswith(".weight"):
            w = v.detach().to("cpu").float()
            if mk in protect:
                out[k] = v.detach().clone()
                n_prot += 1
                continue
            inf = int(w.shape[1]); of = int(w.shape[0])
            gs = group_size_for(inf, GROUP_SIZE)
            if gs is None or of % 4 != 0 or inf % 4 != 0:
                out[k] = v.detach().clone()
                n_prot += 1
                continue
            q, scale = quant_rowwise(rotate_weight(w, had(gs), gs))
            conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": gs}
            out[k] = q
            out[mk + ".weight_scale"] = scale
            out[mk + ".comfy_quant"] = _encode(conf)
            meta_layers[mk] = conf
            n_lin += 1
        else:
            out[k] = v.detach().clone()
            n_kept += 1

    metadata = {"_quantization_metadata": json.dumps(
        {"format_version": "1.0", "layers": meta_layers}, separators=(",", ":"))}
    print(f"Saving: {output_path}", flush=True)
    print(f"  ConvRot INT8 Linear: {n_lin}")
    print(f"  protected (drift, qkv or shape): {n_prot}")
    print(f"  kept (the other runtime tensors): {n_kept}")
    save_file(out, output_path, metadata=metadata)
    print(f"  Input:  {os.path.getsize(input_path)/1024**3:.2f} GB")
    print(f"  Output: {os.path.getsize(output_path)/1024**3:.2f} GB")
    print("Done!")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FLUX ControlNet ConvRot INT8 converter (impact driven)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--impact", required=True)
    ap.add_argument("--comfy_path", required=True)
    ap.add_argument("--drift-threshold", type=float, default=DEFAULT_DRIFT_THRESHOLD)
    ap.add_argument("--quantize-qkv", action="store_true",
                    help="also quantize the fused qkv layers (they are broken: the "
                         "q, k and v halves have value ranges that differ ~7x)")
    a = ap.parse_args()
    convert(a.input, a.output, a.impact, a.comfy_path,
            drift_threshold=a.drift_threshold, quantize_qkv=a.quantize_qkv)

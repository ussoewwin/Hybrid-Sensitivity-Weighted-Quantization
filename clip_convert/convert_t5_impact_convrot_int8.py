# -*- coding: utf-8 -*-
"""Flan-T5-XXL ConvRot INT8 converter driven by measured impact.

The protection set comes from an impact measurement of THIS checkpoint
(sibling script sdxl/diag_impact_t5.py): the relative MSE of the T5 encoder's
own hidden-state output when a single layer is replaced by its ConvRot INT8
reconstruction. Static weight statistics (kurtosis / outlier ratio) are not
used as the protection criterion - the previous revision of this converter
used a kurtosis rule, and that criterion was measured to be invalid for this
model.

Rules (the same runtime constraints established for ControlNet, SAM and the
other CLIP packs):

1. Only modules the runtime actually has are quantized, and only 2D
   `nn.Linear` layers - ComfyUI's INT8 load path attaches a QuantizedTensor to
   Linear layers only. The T5 encoder has 168 of them; the checkpoint also
   carries the decoder blocks, which the encoder-only runtime never touches and
   which are therefore copied through unchanged.
2. The group size is fixed at 256; a Linear whose input width is not a
   multiple of 256 is kept in original precision (the ConvRot kernel is 2D-only
   and fused only at 256).
3. The pack is written under the checkpoint's own parameter names, which is
   what ComfyUI's T5 load path expects (verified against the FP16 layout:
   removing no key and adding only `.weight_scale` / `.comfy_quant`).
4. The `shared` token embedding is an Embedding, not a Linear, and is copied
   through in FP16 - ComfyUI's `operations.Embedding` requires an unpacked
   weight.
5. The checkpoint also carries the T5 **decoder** blocks (339 tensors, 11.87 GB).
   The runtime T5 module (comfy/text_encoders/t5.py) has only `encoder` and
   `shared` - it has no decoder attribute, so the decoder weights are never
   loaded and never used. With `--drop-decoder` they are left out of the pack.
   This is a load-path fact, not a model edit: the encoder that actually runs
   is bit-identical either way.

Usage:
  python clip_convert/convert_t5_impact_convrot_int8.py \
      --input <flan_t5_xxl_fp16>.safetensors --output <out>.safetensors \
      --impact impact/impact_flan_t5_xxl.json \
      --drift-threshold <value>
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

GROUP_SIZE = 256
DEFAULT_DRIFT_THRESHOLD = 1.0


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


def runtime_linear_modules(comfy_path, model_path, device="cpu"):
    """The Linear module names the T5 encoder actually has at runtime."""
    import sys
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)
    cwd = os.getcwd()
    try:
        os.chdir(comfy_path)
        import comfy.utils
        import comfy.text_encoders.sd3_clip as sd3
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        saved = sys.argv
        sys.argv = [saved[0]]
        try:
            detect = sd3.t5_xxl_detect(sd, "")
            detect.pop("t5_quantization_metadata", None)
            te = sd3.T5XXLModel(device=device,
                                dtype=detect.get("dtype_t5", torch.float16),
                                model_options=detect)
        finally:
            sys.argv = saved
        t5 = te.transformer
        return sorted(n for n, m in t5.named_modules()
                      if type(m).__name__ == "Linear" and getattr(m, "weight", None) is not None)
    finally:
        os.chdir(cwd)


def convert(input_path, output_path, impact_json, drift_threshold=DEFAULT_DRIFT_THRESHOLD,
            runtime_modules=None, drop_decoder=False, quantize_decoder=False,
            decoder_impact_json=None, decoder_max_drift=None):
    print(f"Loading (streamed): {input_path}")
    src = safe_open(input_path, framework="pt", device="cpu")
    keys = sorted(src.keys())
    ck_keys = set(keys)

    impacts = json.load(open(impact_json, encoding="utf-8"))["impacts"]
    if runtime_modules is None:
        raise ValueError("runtime_modules is required (the T5 encoder's Linear list)")
    rt2ck = {}
    for n in runtime_modules:
        ck = n + ".weight"
        if ck in ck_keys:
            rt2ck[n] = ck
    print(f"  runtime Linear modules: {len(runtime_modules)}; mapped to checkpoint: {len(rt2ck)}")
    ck_of = set(rt2ck.values())
    if len(ck_of) != len(rt2ck):
        raise RuntimeError("runtime->checkpoint mapping is not one-to-one")

    protect = set()
    for n, drift in impacts.items():
        if drift <= drift_threshold:
            continue
        ck = rt2ck.get(n)
        if ck:
            protect.add(ck)
    print(f"  measured candidates: {len(impacts)}; drift > {drift_threshold} -> protected {len(protect)}")

    # The decoder Linear modules. The runtime T5 has no decoder, so the decoder
    # drift was measured separately (sdxl/diag_impact_t5_decoder.py) against a
    # decoder that really executes. The protection set therefore comes from
    # that measurement, not from a guess.
    dec_lin = set()
    dec_protect = set()
    if quantize_decoder:
        dec_impacts = {}
        if decoder_impact_json:
            dec_impacts = json.load(open(decoder_impact_json, encoding="utf-8"))["impacts"]
        for key in keys:
            if not key.startswith("decoder.") or not key.endswith(".weight"):
                continue
            mk = key[: -len(".weight")]
            # "decoder.block.0.layer.0.SelfAttention.q" -> the module is "q"
            last = mk.rsplit(".", 1)[-1]
            if last in ("q", "k", "v", "o", "wi_0", "wi_1", "wo"):
                dec_lin.add(key)
        if decoder_max_drift is not None and dec_impacts:
            for key in dec_lin:
                # the decoder impact is keyed by the full checkpoint key
                d = dec_impacts.get(key)
                if d is None:
                    d = dec_impacts.get(key[: -len(".weight")])
                if d is not None and d > decoder_max_drift:
                    dec_protect.add(key)
    print(f"  decoder Linear weights: {len(dec_lin)}; protected by the measured drift: {len(dec_protect)}")

    h = build_hadamard(GROUP_SIZE)
    out = {}
    meta_layers = {}
    n_lin = n_prot = n_kept = 0
    n_dropped = 0
    n_dec_lin = 0
    for key in keys:
        if drop_decoder and key.startswith("decoder."):
            n_dropped += 1
            continue
        t = src.get_tensor(key)
        if not (key.endswith(".weight") and t.ndim == 2
                and t.dtype in (torch.float16, torch.float32, torch.bfloat16)):
            out[key] = t
            n_kept += 1
            continue
        mk = key[: -len(".weight")]
        is_dec_lin = key in dec_lin
        if not is_dec_lin and mk + ".weight" not in ck_of:
            out[key] = t
            n_kept += 1
            continue
        if is_dec_lin:
            if key in dec_protect:
                out[key] = t
                n_prot += 1
                continue
        elif mk + ".weight" in protect:
            out[key] = t
            n_prot += 1
            continue
        inf = int(t.shape[1]); of = int(t.shape[0])
        if inf % GROUP_SIZE != 0 or of % 4 != 0 or inf % 4 != 0:
            out[key] = t
            n_prot += 1
            continue
        q, scale = quant_rowwise(rotate_weight(t.float(), h, GROUP_SIZE))
        if is_dec_lin:
            n_dec_lin += 1
        else:
            n_lin += 1
        conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP_SIZE}
        out[key] = q
        out[f"{mk}.weight_scale"] = scale
        out[f"{mk}.comfy_quant"] = _encode(conf)
        meta_layers[mk] = conf

    metadata = {"_quantization_metadata": json.dumps(
        {"format_version": "1.0", "layers": meta_layers}, separators=(",", ":"))}
    print(f"Saving: {output_path}")
    print(f"  ConvRot INT8 Linear: {n_lin}")
    print(f"  protected (drift or shape): {n_prot}")
    print(f"  kept (not runtime Linear / non-2D): {n_kept}")
    if quantize_decoder:
        print(f"  ConvRot INT8 decoder Linear: {n_dec_lin}")
    if drop_decoder:
        print(f"  dropped (decoder.*, never loaded at runtime): {n_dropped}")
    save_file(out, output_path, metadata=metadata)
    print(f"  Input:  {os.path.getsize(input_path)/1024**3:.2f} GB")
    print(f"  Output: {os.path.getsize(output_path)/1024**3:.2f} GB")
    print("Done!")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Flan-T5-XXL ConvRot INT8 converter (impact driven)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--impact", required=True)
    ap.add_argument("--comfy_path", default=r"D:\USERFILES\ComfyUI\ComfyUI")
    ap.add_argument("--drift-threshold", type=float, default=DEFAULT_DRIFT_THRESHOLD)
    ap.add_argument("--drop-decoder", action="store_true",
                    help="omit the decoder.* tensors the runtime never loads")
    ap.add_argument("--quantize-decoder", action="store_true",
                    help="quantize the decoder Linear weights as well (full-model pack)")
    ap.add_argument("--decoder-impact", default="",
                    help="the decoder impact measurement (impact/impact_flan_t5_xxl_decoder.json)")
    ap.add_argument("--decoder-max-drift", type=float, default=None,
                    help="decoder layers with a measured drift above this stay FP16")
    args = ap.parse_args()
    if args.drift_threshold < 0:
        ap.error("--drift-threshold must be >= 0")
    mods = runtime_linear_modules(args.comfy_path, os.path.abspath(args.input))
    convert(args.input, args.output, args.impact,
            drift_threshold=args.drift_threshold, runtime_modules=mods,
            drop_decoder=args.drop_decoder, quantize_decoder=args.quantize_decoder,
            decoder_impact_json=args.decoder_impact or None,
            decoder_max_drift=args.decoder_max_drift)

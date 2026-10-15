"""SDXL CLIP ConvRot INT8 conversion with FP16 protection (rule-based).

Rule derived from measured risk metrics (kurtosis / outlier_ratio = absmax/std)
across four SDXL checkpoints (waiIllustrious / waiREALCN / animemix / bluePencil):
the protected set is model-independent and structural.

Protected (kept FP16) CLIP Linear layers:
  CLIP-L (conditioner.embedders.0):
    transformer.text_model.embeddings.token_embedding.weight
    transformer.text_model.embeddings.position_embedding.weight
    transformer.text_model.encoder.layers.0.mlp.fc1.weight
    transformer.text_model.encoder.layers.0.mlp.fc2.weight
  CLIP-G (conditioner.embedders.1):
    model.token_embedding.weight
    model.transformer.resblocks.*.mlp.c_proj.weight          (all resblocks)
    model.transformer.resblocks.2.attn.out_proj.weight
    model.transformer.resblocks.29.attn.out_proj.weight

Everything else in the CLIPs is FULL ConvRot INT8 (rotate + per-channel scale +
comfy_quant stamp), matching the native converter's pack math.

Usage:
  python sdxl/quantize_sdxl_clip_int8.py --input <pack> --output <pack_clipint8>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys

from safetensors import safe_open
from safetensors.torch import load_file, save_file

DEFAULT_GROUP = 256

# --- protection rule (structural, model-independent) -----------------------
_CLIP_L_PROTECT_EXACT = (
    "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
    "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight",
    "conditioner.embedders.0.transformer.text_model.encoder.layers.0.mlp.fc1.weight",
    "conditioner.embedders.0.transformer.text_model.encoder.layers.0.mlp.fc2.weight",
)
# CLIP-G: all mlp.c_proj, plus out_proj at resblocks 2 and 29
_CLIP_G_CPROJ_RE = re.compile(
    r"^conditioner\.embedders\.1\.model\.transformer\.resblocks\.\d+\.mlp\.c_proj\.weight$"
)
_CLIP_G_OUTPROJ_PROTECT = {
    "conditioner.embedders.1.model.transformer.resblocks.2.attn.out_proj.weight",
    "conditioner.embedders.1.model.transformer.resblocks.29.attn.out_proj.weight",
}


def is_protected_clip_weight(key: str) -> bool:
    """True if this CLIP weight must stay FP16 (rule-based protection set)."""
    if key in _CLIP_L_PROTECT_EXACT:
        return True
    if key in _CLIP_G_OUTPROJ_PROTECT:
        return True
    if _CLIP_G_CPROJ_RE.match(key):
        return True
    return False


def _load_native_helpers():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    path = os.path.join(repo, "comfyui_nodes", "native_convrot_int8_convert.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"native converter not found: {path}")
    spec = importlib.util.spec_from_file_location("_hswq_native_convrot_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._load_native_int8()


def _gs_for(features: int, group: int):
    for g in (group, group // 4, group // 16, group // 64):
        if g >= 4 and features % g == 0 and features % 4 == 0:
            return g
    return None


def quantize_clip(sd, qm, group=DEFAULT_GROUP):
    """Quantize SDXL CLIP Linears to ConvRot INT8, keeping the protected set FP16.

    Returns (converted_count, protected_count, kept_count).
    """
    n8 = _load_native_helpers()
    h_cache = {}
    converted = 0
    protected = 0
    kept = 0

    for key in sorted(list(sd.keys())):
        if not (key.startswith("conditioner.embedders.") and key.endswith(".weight")):
            continue
        module = key[: -len(".weight")]
        w = sd[key]
        if w.ndim != 2:
            kept += 1  # norms / biases / 1D
            continue
        if is_protected_clip_weight(key):
            protected += 1
            continue  # stays FP16, untouched

        wf = w.float()
        used = _gs_for(int(wf.shape[1]), group)
        if used is not None:
            if used not in h_cache:
                h_cache[used] = n8.build_hadamard(used, device="cpu")
            w_rot = n8.rotate_weight(wf, h_cache[used], used)
            q, scale = n8.quantize_int8_channelwise(w_rot)
            conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": int(used)}
        else:
            q, scale = n8.quantize_int8_channelwise(wf)
            conf = {"format": "int8_tensorwise"}
        sd[key] = q
        sd[module + ".weight_scale"] = scale
        sd[module + ".comfy_quant"] = n8._encode_comfy_quant(conf)
        qm["layers"][module] = dict(conf)
        converted += 1

    return converted, protected, kept


def main():
    ap = argparse.ArgumentParser(description="SDXL CLIP ConvRot INT8 with FP16 protection")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--group", type=int, default=DEFAULT_GROUP)
    args = ap.parse_args()

    src = os.path.abspath(args.input)
    dst = os.path.abspath(args.output)
    if not os.path.isfile(src):
        print(f"[FATAL] input not found: {src}")
        sys.exit(1)

    sd = load_file(src)
    with safe_open(src, framework="pt") as f:
        md = f.metadata() or {}
    qm = json.loads(md["_quantization_metadata"]) if "_quantization_metadata" in md else {"format_version": "1.0", "layers": {}}

    converted, protected, kept = quantize_clip(sd, qm, group=args.group)

    new_meta = dict(md)
    new_meta["_quantization_metadata"] = json.dumps(qm, separators=(",", ":"))
    save_file(sd, dst, metadata=new_meta)

    print("SDXL CLIP ConvRot INT8 conversion done:")
    print(f"  CLIP INT8 (ConvRot): {converted}")
    print(f"  CLIP FP16-protected: {protected}")
    print(f"  CLIP kept as-is (norms/biases/1D): {kept}")
    print(f"  saved: {dst} ({os.path.getsize(dst)/1048576:.1f} MB)")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""WeVisOne-VL-4B ConvRot W4A8 (asym_w4a8_int8) converter, impact-driven.

The protection set comes from a measurement of this checkpoint's own output
(`clip_convert/diag_impact_wevisone_w4a8.py`): every 2-D weight is replaced, one at a time,
by the W4A8 reconstruction and the drift of the combined text + vision reference vector is
recorded. The layers with the largest drift stay in their original dtype.

  python clip_convert/convert_wevisone_vl_4b_impact_w4a8.py \
      --input   <bf16 checkpoint> \
      --output  <pack> \
      --impact  impact/impact_wevisone_w4a8_full.json \
      --protect-count 32

Linear (2-D) layers go ComfyUI-native W4A8 when the shapes allow:
  in_features % 256 == 0   (the CUDA fused kernel's convrot group)
  in_features % 16  == 0   (the kitchen quant group)
Pack via comfy_kitchen.quantize_w4a8_int8_weight (bits=4, symmetric, codebook):
  weight            int8  [N, K/2]     packed nibbles, low nibble = even column
  weight_s_rel      fp8e4m3 [N, K/16]  per-group relative scale
  weight_s_channel  f32     [N]        per-out-channel scale
  weight_codebook   f32     [16]       Lloyd-Max table (absent when the fixed LUT fits)
  comfy_quant       {"format":"asym_w4a8_int8","group_size":16,"convrot_groupsize":256}
Rotation happens inside the kitchen quantizer; dequantize returns the original basis.

Everything else (3-D/1-D tensors, K % 256 != 0, the protected set) is copied through in its
original dtype. `embed_tokens` is an Embedding and `lm_head` is a generation head the text
encoder never reads, so both are pinned to the protected set by name.
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

DEFAULT_GROUP = 16
DEFAULT_CONVROT = 256
DEFAULT_DRIFT_THRESHOLD = 0.15

# always kept in the source dtype:
#   embed_tokens - an Embedding, not a Linear, so the W4A8 kernel does not cover it
#   lm_head      - the generation head; the text encoder never reads it
#   mask_decoder - runs in fp32 by design and sits outside the drift measurement, so it is
#                  not quantised on an unmeasured assumption (5.1 M params, ~20 MB total)
PINNED = ("model.language_model.embed_tokens", "lm_head")
PINNED_PREFIXES = ("mask_decoder.",)


def relative_to_checkpoint(name: str) -> str | None:
    if name.startswith("layers."):
        return "model.language_model." + name
    if name.startswith("blocks."):
        return "model.visual." + name
    if name.startswith("merger.") or name.startswith("deepstack_merger_list."):
        return "model.visual." + name
    if name in ("embed_tokens", "norm"):
        return "model.language_model." + name
    if name == "pos_embed":
        return "model.visual." + name
    return None


def _encode_meta(config: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(config, separators=(",", ":")).encode("utf-8")),
                        dtype=torch.uint8)


def load_protect_list(path):
    if not path:
        return set()
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {ln.strip() for ln in text.splitlines() if ln.strip()}
    if isinstance(data, dict):
        return set(data.get("protect", []))
    return set(data)


def convert(model_path: str, output_path: str, impact_json: str | None = None,
            protect_count: int = 0, protect_list: str | None = None,
            drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
            groupsize: int = DEFAULT_CONVROT, group: int = DEFAULT_GROUP,
            limit: int = 0):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import comfy_kitchen as ck

    print("Loading (streamed): %s" % model_path)
    src = safe_open(model_path, framework="pt", device="cpu")
    keys = list(src.keys())
    meta = src.metadata() or {}
    print("  tensors: %d" % len(keys))

    # --- the protection set ------------------------------------------------
    protect_ck: set[str] = set()
    picked: list[tuple[str, float]] = []
    explicit = load_protect_list(protect_list)
    if explicit:
        for n in explicit:
            ck_name = relative_to_checkpoint(n) or n
            protect_ck.add(ck_name)
        print("  explicit protect list: %d layers" % len(protect_ck))
    elif impact_json and (protect_count or drift_threshold):
        impacts = json.load(open(impact_json, encoding="utf-8"))["impacts"]
        ranked = sorted(((n, float(d)) for n, d in impacts.items()
                         if isinstance(d, (int, float)) and float(d) > 0),
                        key=lambda kv: -kv[1])
        if protect_count:
            chosen = ranked[:protect_count]
            print("  top-%d drift cut at %.6e" % (protect_count, chosen[-1][1] if chosen else 0.0))
        else:
            chosen = [(n, d) for n, d in ranked if d > drift_threshold]
            print("  drift > %.4f: %d layers" % (drift_threshold, len(chosen)))
        for n, d in chosen:
            ck_name = relative_to_checkpoint(n)
            if ck_name is None:
                raise SystemExit("the impact names a layer with no checkpoint mapping: %s" % n)
            protect_ck.add(ck_name)
            picked.append((ck_name, d))
        print("  protection set: %d layers" % len(protect_ck))
    protect_ck.update(PINNED)
    print("  pinned by name: %s -> total protected %d" % (", ".join(PINNED), len(protect_ck)))

    # --- the pack ----------------------------------------------------------
    new_state_dict: dict[str, torch.Tensor] = {}
    quant_meta_layers: dict[str, dict] = {}
    w4a8 = kept = not_eligible = 0
    per_layer = []
    total_in = total_out = 0

    for key in keys:
        tensor = src.get_tensor(key)
        total_in += tensor.numel() * tensor.element_size()
        module_key = key[:-len(".weight")] if key.endswith(".weight") else key
        eligible = (
            tensor.ndim == 2
            and tensor.shape[1] % groupsize == 0
            and tensor.shape[1] % group == 0
            and module_key not in protect_ck
            and not module_key.startswith(PINNED_PREFIXES)
        )
        if limit and w4a8 >= limit:
            eligible = False

        if not eligible:
            new_state_dict[key] = tensor.contiguous()
            if module_key in protect_ck or module_key.startswith(PINNED_PREFIXES):
                kept += 1
            else:
                not_eligible += 1
            total_out += tensor.numel() * tensor.element_size()
            continue

        w_fp32 = tensor.float()
        qdata, s_rel, s_channel, correction, codebook = ck.quantize_w4a8_int8_weight(
            w_fp32, group_size=group, convrot_groupsize=groupsize,
            symmetric=True, scale_dtype=torch.float8_e4m3fn, codebook=True, bits=4)
        quant_config = {"format": "asym_w4a8_int8", "group_size": int(group),
                        "convrot_groupsize": int(groupsize)}
        new_state_dict[key] = qdata.to(torch.int8)
        new_state_dict["%s.weight_s_rel" % module_key] = s_rel.to(
            torch.float8_e4m3fn).contiguous()
        new_state_dict["%s.weight_s_channel" % module_key] = s_channel.float().contiguous()
        if codebook is not None:
            new_state_dict["%s.weight_codebook" % module_key] = codebook.float().contiguous()
        new_state_dict["%s.comfy_quant" % module_key] = _encode_meta(quant_config)
        quant_meta_layers[module_key] = dict(quant_config)

        dq = ck.dequantize_w4a8_int8_weight(
            qdata, s_rel, s_channel, codebook=codebook, correction=correction,
            group_size=group, convrot_groupsize=groupsize, output_dtype=torch.float32)
        denom = (w_fp32 ** 2).sum().item()
        rel = float(((w_fp32 - dq) ** 2).sum().item() / denom) if denom else 0.0
        per_layer.append({"layer": module_key, "rel_mse": rel,
                          "shape": list(tensor.shape)})
        w4a8 += 1
        total_out += (new_state_dict[key].numel() * 1
                      + new_state_dict["%s.weight_s_rel" % module_key].numel()
                      + new_state_dict["%s.weight_s_channel" % module_key].numel() * 4
                      + (new_state_dict["%s.weight_codebook" % module_key].numel() * 4
                         if codebook is not None else 0)
                      + new_state_dict["%s.comfy_quant" % module_key].numel())
        del w_fp32, dq

    metadata = dict(meta)
    metadata["_quantization_metadata"] = json.dumps(
        {"format_version": "1.0", "layers": quant_meta_layers}, separators=(",", ":"))

    print("Saving to: %s" % output_path)
    save_file(new_state_dict, output_path, metadata=metadata)
    out_bytes = os.path.getsize(output_path)
    print("W4A8 layers: %d | protected: %d | not eligible: %d" % (w4a8, kept, not_eligible))
    print("bytes in %s -> out %s (%.2f%% of source)"
          % (format(total_in, ","), format(out_bytes, ","), out_bytes / total_in * 100))
    per_layer.sort(key=lambda d: -d["rel_mse"])
    print("worst weight rel_mse: %s %.6e" % (per_layer[0]["layer"], per_layer[0]["rel_mse"])
          if per_layer else "no W4A8 layer")
    return {"w4a8_layers": w4a8, "protected": kept, "not_eligible": not_eligible,
            "bytes_in": total_in, "bytes_out": out_bytes,
            "protection": sorted(picked), "per_layer": per_layer,
            "quant_metadata_layers": len(quant_meta_layers)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--impact", default="")
    ap.add_argument("--protect-count", type=int, default=0)
    ap.add_argument("--protect-list", default="")
    ap.add_argument("--drift-threshold", type=float, default=DEFAULT_DRIFT_THRESHOLD)
    ap.add_argument("--groupsize", type=int, default=DEFAULT_CONVROT)
    ap.add_argument("--group", type=int, default=DEFAULT_GROUP)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    info = convert(args.input, args.output, impact_json=args.impact,
                   protect_count=args.protect_count, protect_list=args.protect_list,
                   drift_threshold=args.drift_threshold, groupsize=args.groupsize,
                   group=args.group, limit=args.limit)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=1, sort_keys=True)
        print("record: %s" % args.json)


if __name__ == "__main__":
    main()

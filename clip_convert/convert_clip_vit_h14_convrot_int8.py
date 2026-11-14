"""ConvRot INT8 converter for `CLIP-ViT-H-14.safetensors` (open_clip ViT-H/14 image tower).

This converter is written for that one checkpoint's layout. The generic safetensors converter
takes every 2-D `.weight`; on this file that would also catch
`vision_model.embeddings.position_embedding.weight` (257 x 1280), which is a learned position
table, not a projection, and quantising it would rotate a table whose rows are positions.

What is quantised here, by name:

  vision_model.encoder.layers.<0..31>.self_attn.{q,k,v,out}_proj.weight
  vision_model.encoder.layers.<0..31>.mlp.fc{1,2}.weight
  visual_projection.weight
  = 32 x 6 + 1 = 193 Linears

What is kept as-is, by name:

  vision_model.embeddings.patch_embedding.weight   Conv2d (1280, 3, 14, 14)
  vision_model.embeddings.position_embedding.weight  the position table
  vision_model.embeddings.class_embedding          a bare vector
  every .bias, every LayerNorm weight
  any Linear whose measured drift is above the threshold

Output per quantised layer:

  <layer>.weight           int8
  <layer>.weight_scale     float32 [out, 1]
  <layer>.comfy_quant      uint8 JSON {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":N}

  python clip_convert/convert_clip_vit_h14_convrot_int8.py \
      --model <CLIP-ViT-H-14.safetensors> --impact <impact.json> \
      --output <pack.safetensors> [--drift-threshold 0.15] [--groupsize 256]
"""

import argparse
import json
import math
import os

import torch
from safetensors.torch import load_file, save_file

ENCODER = "vision_model.encoder.layers."
PROJ = "visual_projection.weight"
PATCH_EMBED = "vision_model.embeddings.patch_embedding.weight"
POS_EMBED = "vision_model.embeddings.position_embedding.weight"
CLASS_EMBED = "vision_model.embeddings.class_embedding"


def build_hadamard(size, dtype=torch.float32):
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise SystemExit("Hadamard size must be a power of 4, got %d" % size)
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def group_size(n, preferred):
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def is_target(name):
    if name == PROJ:
        return True
    if not name.startswith(ENCODER) or not name.endswith(".weight"):
        return False
    tail = name[len(ENCODER):]
    parts = tail.split(".")
    if len(parts) != 4:
        return False
    _, block, leaf, _w = parts
    if block in ("self_attn", "mlp"):
        return leaf in ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")
    return False


def stamp(gs):
    payload = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": gs},
                         separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--impact", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--drift-threshold", type=float, default=0.15)
    ap.add_argument("--groupsize", type=int, default=256)
    args = ap.parse_args()

    print("[load] %s" % args.model, flush=True)
    sd = load_file(args.model)
    impacts = json.load(open(args.impact, encoding="utf-8"))["impacts"]
    print("[load] tensors %d | measured %d" % (len(sd), len(impacts)), flush=True)

    hadamards = {}
    out = {}
    quantised, protected, kept, missing_measure = [], [], [], []

    for name, tensor in sd.items():
        if not is_target(name):
            # kept at the source's own dtype: a cast here would make the float side differ from
            # the source by more than the quantisation does
            out[name] = tensor
            kept.append(name)
            continue
        if tensor.dim() != 2:
            out[name] = tensor
            kept.append(name)
            continue
        mk = name[: -len(".weight")]
        drift = impacts.get(mk)
        if drift is None:
            missing_measure.append(mk)
            out[name] = tensor
            continue
        if drift > args.drift_threshold:
            out[name] = tensor
            protected.append({"layer": mk, "drift": drift,
                              "reason": "drift %.6e > %.2f" % (drift, args.drift_threshold)})
            continue
        gs = group_size(int(tensor.shape[1]), args.groupsize)
        if gs is None:
            out[name] = tensor
            protected.append({"layer": mk, "drift": drift,
                              "reason": "in=%d has no power-of-four group" % tensor.shape[1]})
            continue
        if gs not in hadamards:
            hadamards[gs] = build_hadamard(gs)
        h = hadamards[gs]
        of, inf = tensor.shape
        g = inf // gs
        x = torch.matmul(tensor.float().view(of, g, gs), h.T).reshape(of, inf)
        amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
        s = amax / 127.0
        q = (x / s).round().clamp(-127, 127).to(torch.int8)
        out[name] = q
        out[mk + ".weight_scale"] = s.to(torch.float32)
        out[mk + ".comfy_quant"] = stamp(gs)
        quantised.append({"layer": mk, "drift": drift, "groupsize": gs})

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_file(out, args.output, metadata={"format": "pt"})
    print("  ConvRot INT8 Linear: %d" % len(quantised), flush=True)
    print("  protected (drift or shape): %d" % len(protected), flush=True)
    for p in protected:
        print("    %s  %s" % (p["layer"], p["reason"]), flush=True)
    print("  kept (not a measured Linear): %d" % len(kept), flush=True)
    if missing_measure:
        print("  !! measured set did not cover: %d" % len(missing_measure), flush=True)
        for m in missing_measure:
            print("     %s" % m, flush=True)
    print("  output entries: %d" % len(out), flush=True)
    print("  %.2f GB" % (os.path.getsize(args.output) / 1024 ** 3), flush=True)
    report = {
        "tool": "clip_convert/convert_clip_vit_h14_convrot_int8.py",
        "model": os.path.basename(args.model),
        "impact": os.path.basename(args.impact),
        "drift_threshold": args.drift_threshold,
        "groupsize": args.groupsize,
        "quantised": quantised,
        "quantised_count": len(quantised),
        "protected": protected,
        "protected_count": len(protected),
        "kept_count": len(kept),
        "kept": kept,
        "missing_measurement": missing_measure,
        "output": os.path.basename(args.output),
        "output_bytes": os.path.getsize(args.output),
    }
    with open(args.output + ".convert.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print("wrote %s (+ .convert.json)" % args.output, flush=True)


if __name__ == "__main__":
    main()

"""SD1.5 IP-Adapter FaceID per-layer ConvRot INT8 impact.

The HSWQ method, on this file's own network:

  1. build `image_proj` (proj.0 -> GELU -> proj.2) and the 32 blocks of
     `to_{q,k,v,out}_lora.{down,up}` from the same nested state dict the node reads,
  2. run each stage on a fixed real-shaped input and record its output,
  3. replace one weight with its ConvRot INT8 reconstruction, run again, record the
     relative MSE of the stage output that weight feeds,
  4. restore, next weight.

What is measured, and what is not: a FaceID adapter has no forward of its own. `image_proj`
maps a 512-wide face embedding to the 3072-wide token stack, and each LoRA pair is a low-rank
update that the node adds into a UNet cross-attention projection. The observable here is the
stage output the weight feeds - the token stack for `image_proj`, the low-rank contribution for
a LoRA pair - not a sampled image. A sampled-image number would require running the UNet with
the adapter attached, which is the node's job and a different measurement.

The quantisation replay is the converter's own: Hadamard by Kronecker doubling, orthonormal,
group size a power of four, row-wise `amax/127` int8. A weight whose inner dimension cannot
carry a power-of-four group is recorded as `not_quantisable` rather than measured.

  python clip_convert/diag_impact_ipadapter_faceid_sd15.py --ipadapter <f.bin> --out <impact.json>
"""

import argparse
import json
import math
import os

import torch


def build_hadamard(size, device="cpu", dtype=torch.float32):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype, device=device)
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


def roundtrip(w, gs):
    of, inf = w.shape
    h = build_hadamard(gs, dtype=torch.float32)
    rot = torch.matmul(w.float().view(of, inf // gs, gs), h.T).reshape(of, inf)
    amax = rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    scale = amax / 127.0
    q = torch.round(rot / scale).clamp(-127, 127)
    deq = (q * scale).reshape(of, inf // gs, gs)
    return torch.matmul(deq, h).reshape(of, inf)


def rel_mse(a, b):
    return float(((a - b) ** 2).mean() / (a ** 2).mean().clamp(min=1e-30))


def flatten(sd):
    if isinstance(sd, dict) and "image_proj" in sd and "ip_adapter" in sd:
        out = {}
        for k, v in sd["image_proj"].items():
            out["image_proj." + k] = v
        for k, v in sd["ip_adapter"].items():
            out["ip_adapter." + k] = v
        return out
    return dict(sd)


def group_layers(flat):
    """Pair each `*.down.weight` with its `*.up.weight`; keep bare weights single."""
    used = set()
    groups = []
    for k in sorted(flat):
        if not k.endswith(".weight") or k in used:
            continue
        v = flat[k]
        if v.ndim != 2:
            continue
        if k.endswith(".down.weight"):
            up = k[: -len(".down.weight")] + ".up.weight"
            if up in flat:
                groups.append((k[:-len(".weight")].rstrip("."), [k, up]))
                used.update([k, up])
                continue
        groups.append((k[: -len(".weight")], [k]))
        used.add(k)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipadapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    raw = flatten(torch.load(a.ipadapter, map_location="cpu", weights_only=True))
    # measure in float32: the pack ships float16, but the quantisation error has to be
    # resolved against the weight itself, not against a float16 rounding of it
    flat = {k: (v.float() if hasattr(v, "dtype") and v.dtype == torch.float16 else v)
            for k, v in raw.items()}
    groups = group_layers(flat)
    print("[load] %d tensors, %d weight groups" % (len(flat), len(groups)), flush=True)

    g = torch.Generator().manual_seed(a.seed)

    # ---- stage inputs, fixed for the whole sweep
    face_dim = fac = None
    for k, v in flat.items():
        if k.startswith("image_proj.proj.0.weight"):
            face_dim = int(v.shape[1])
    if face_dim is None:
        face_dim = 512
    face = torch.randn(1, face_dim, generator=g)

    lora_inputs = {}
    for name, keys in groups:
        # a FaceID adapter carries `to_k_lora.up` without a matching `down` (the UNet owns the
        # down projection), so a group may be one weight. The input is shaped to the weight's
        # own input width either way.
        width = int(flat[keys[0]].shape[1])
        lora_inputs[name] = torch.randn(1, width, generator=g)

    def run_image_proj(sd):
        x = face
        x = torch.nn.functional.linear(x, sd["image_proj.proj.0.weight"], sd.get("image_proj.proj.0.bias"))
        x = torch.nn.functional.gelu(x)
        x = torch.nn.functional.linear(x, sd["image_proj.proj.2.weight"], sd.get("image_proj.proj.2.bias"))
        return x

    def run_lora(name, sd, keys):
        x = lora_inputs[name]
        h = torch.nn.functional.linear(x, sd[keys[0]], None)
        if len(keys) == 2:
            return torch.nn.functional.linear(h, sd[keys[1]], None)
        return h

    base_image_proj = run_image_proj(flat)
    print("[baseline] image_proj output %s" % (tuple(base_image_proj.shape),), flush=True)

    impacts = {}
    not_quantisable = {}
    done = 0
    for name, keys in groups:
        gs = None
        for k in keys:
            gs = convrot_group_size(int(flat[k].shape[1]))
            if gs is None:
                break
        if gs is None:
            not_quantisable[name] = {"keys": keys,
                                     "reason": "inner dimension %s cannot carry a power-of-four group" % [int(flat[k].shape[1]) for k in keys]}
            continue
        sd = dict(flat)
        with torch.no_grad():
            for k in keys:
                sd[k] = roundtrip(flat[k], gs)
        with torch.no_grad():
            if name.startswith("image_proj."):
                delta = rel_mse(run_image_proj(sd), base_image_proj)
            else:
                delta = rel_mse(run_lora(name, sd, keys), run_lora(name, flat, keys))
        impacts[name] = delta
        done += 1
        if done % 20 == 0 or done == 1:
            print("  %3d/%d %-46s %.6e" % (done, len(groups), name, delta), flush=True)

    vals = sorted(impacts.values())
    rec = {
        "tool": "clip_convert/diag_impact_ipadapter_faceid_sd15.py",
        "model": os.path.basename(a.ipadapter),
        "tensors": len(flat),
        "weight_groups": len(groups),
        "measured": len(impacts),
        "not_quantisable": not_quantisable,
        "observable": {
            "image_proj": "relative MSE of the 3072-wide token stack, input a fixed 512-wide face embedding",
            "lora": "relative MSE of the low-rank contribution, input a fixed vector of the block's input width",
        },
        "seed": a.seed,
        "max": vals[-1] if vals else None,
        "max_layer": max(impacts, key=impacts.get) if impacts else None,
        "median": vals[len(vals) // 2] if vals else None,
        "p90": vals[int(round(0.9 * (len(vals) - 1)))] if vals else None,
        "min": vals[0] if vals else None,
        "impacts": impacts,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print("[done] measured %d, not quantisable %d" % (len(impacts), len(not_quantisable)), flush=True)
    print("[done] max %.6e at %s | median %.6e | p90 %.6e"
          % (rec["max"], rec["max_layer"], rec["median"], rec["p90"]), flush=True)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()

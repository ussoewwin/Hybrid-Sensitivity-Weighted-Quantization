"""Impact sweep for ip-adapter_xl.pth (SDXL basic IP-Adapter).

This adapter has two parts, and each is measured on its own output:

- `image_proj` is an `ImageProjModel`: `norm` (LayerNorm over 2048) then `proj`
  (1280 -> 8192, i.e. 4 tokens x 2048). The observable is the 4x2048 token stack
  that `proj` produces from a fixed CLIP embedding.
- `ip_adapter` is 70 pairs of bias-free `to_k_ip` / `to_v_ip` Linears (2048 -> 640 or
  1280), applied to the UNet cross-attention every forward. The observable is each
  Linear's own output on a fixed input of its width.

One layer at a time is replaced by its ConvRot INT8 reconstruction: Hadamard-256
rotation on the inner dimension, per-output-channel `amax/127` scale, and the error is
measured in the **original** frame after rotating the activation back.

Measured in float32: the file is float16, and the quantisation error has to be resolved
against the weights themselves, not against float16 rounding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import torch
import torch.nn.functional as F


def build_hadamard(size, dtype=torch.float32, device="cpu"):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
                      dtype=dtype, device=device)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def convrot_int8(w, gs=256):
    """Rotate the inner dimension, quantise per output row, rotate back."""
    out_f, in_f = w.shape
    if in_f % gs != 0:
        return None
    h = build_hadamard(gs, dtype=torch.float32, device=w.device)
    groups = w.reshape(out_f, in_f // gs, gs)
    rotated = torch.matmul(groups, h)
    amax = rotated.abs().amax(dim=2, keepdim=True).clamp_min(1e-8)
    scale = amax / 127.0
    q = torch.round(rotated / scale).clamp(-127, 127)
    back = torch.matmul(q * scale, h.transpose(-1, -2))
    return back.reshape(out_f, in_f)


def rel_mse(a, b):
    num = ((a - b) ** 2).mean()
    den = (a ** 2).mean().clamp_min(1e-12)
    return float((num / den).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    t0 = time.time()

    raw = open(a.input, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    sd = torch.load(a.input, map_location="cpu", weights_only=True)
    del raw

    proj = sd["image_proj"]
    ip = sd["ip_adapter"]

    dev = a.device if torch.cuda.is_available() else "cpu"
    norm_w = proj["norm.weight"].float().to(dev)
    norm_b = proj["norm.bias"].float().to(dev)
    proj_w = proj["proj.weight"].float().to(dev)
    proj_b = proj["proj.bias"].float().to(dev)

    # a fixed CLIP-H embedding, the adapter's real input width
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    emb = torch.randn(1, proj_w.shape[1], generator=g).to(dev)

    tokens = proj_w.shape[0] // norm_w.shape[0]
    cad = norm_w.shape[0]

    def token_stack(pw):
        # ImageProjModel: proj first, then reshape to (tokens, cross_attention_dim), then
        # LayerNorm over cross_attention_dim. The norm is 2048 wide and the input is 1280,
        # so normalising before the projection would be the wrong order and the wrong width.
        x = F.linear(emb, pw, proj_b)
        x = x.reshape(1, tokens, cad)
        return F.layer_norm(x, (cad,), norm_w, norm_b)

    base_stack = token_stack(proj_w)

    impacts = {}
    not_quantisable = {}

    # ---- image_proj.proj
    q = convrot_int8(proj_w)
    if q is None:
        not_quantisable["image_proj.proj"] = "inner dimension %d not a multiple of 256" % proj_w.shape[1]
    else:
        impacts["image_proj.proj"] = rel_mse(base_stack, token_stack(q))
    print("[image_proj.proj] shape %s -> %d tokens x %d, drift %.9e"
          % (tuple(proj_w.shape), tokens, cad, impacts.get("image_proj.proj", -1)), flush=True)

    # ---- ip_adapter layers
    seen = 0
    for key in sorted(ip.keys()):
        w = ip[key].float()
        name = "ip_adapter." + key[: -len(".weight")]
        out_f, in_f = w.shape
        if in_f % 256 != 0:
            not_quantisable[name] = "inner dimension %d not a multiple of 256" % in_f
            seen += 1
            continue
        x = torch.randn(1, in_f, generator=g).to(dev)
        wd = w.to(dev)
        ref = F.linear(x, wd, None)
        q = convrot_int8(wd)
        impacts[name] = rel_mse(ref, F.linear(x, q, None))
        seen += 1
        if seen % 20 == 0:
            print("  ... %d layers" % seen, flush=True)

    vals = sorted(impacts.values(), reverse=True)
    n = len(vals)
    summary = {
        "tool": "clip_convert/diag_impact_ipadapter_xl.py",
        "family": "ipadapter_xl",
        "source": os.path.basename(a.input),
        "source_bytes": os.path.getsize(a.input),
        "source_sha256": sha,
        "tensor_count": sum(len(v) for v in (proj, ip)),
        "image_proj_tensors": len(proj),
        "ip_adapter_tensors": len(ip),
        "weight_groups": n,
        "measured": n,
        "not_quantisable_count": len(not_quantisable),
        "not_quantisable": not_quantisable,
        "max": vals[0] if vals else None,
        "max_layer": max(impacts, key=impacts.get) if impacts else None,
        "median": vals[n // 2] if vals else None,
        "p90": vals[max(0, int(n * 0.1) - 1)] if vals else None,
        "min": vals[-1] if vals else None,
        "above_threshold": sum(1 for v in vals if v > 0.15),
        "threshold": 0.15,
        "seed": a.seed,
        "dtype": "float32",
        "device": dev,
        "seconds": round(time.time() - t0, 1),
        "impacts": impacts,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print("[summary] groups %d | max %.9e (%s) | median %.9e | p90 %.9e | above 0.15 %d | %.1f s"
          % (n, summary["max"], summary["max_layer"], summary["median"], summary["p90"],
             summary["above_threshold"], summary["seconds"]), flush=True)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()

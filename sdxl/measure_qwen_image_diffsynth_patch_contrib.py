# -*- coding: utf-8 -*-
"""Qwen-Image diffsynth model patch (canny/depth/inpaint family): the aggregate
contribution of each layer, measured on the PATCH's OWN output.

The criterion (the current method, see
`md/How to quantize Text Encoder and ControlNet.md`):
replace ONE layer by its ConvRot INT8 reconstruction in the ORIGINAL frame
(`dequant(INT8(W @ H^T))` = `(q*s) @ H`), hold every other layer, and rank by
the change in the patch's own output. Drift is only a screen.

This file is a **model patch**, not a ControlNet. ComfyUI's stock
`ModelPatchLoader` does NOT recognise the INT8 layout, so the pack is loaded
through the HSWQ loader (`HSWQModelPatchLoader`) which builds the patch with
`mixed_precision_ops`. Both dispatch on `controlnet_blocks.0.y_rms.weight` into
`comfy_extras.nodes_model_patch.QwenImageBlockWiseControlNet`.

The observable is what the runtime actually injects. The apply node calls, once
per DiT block i:

    img[:, :n] += model.control_block(img[:, :n], cond, i) * strength

with `cond = model.process_input_latent_image(vae_encode(control_image))`. So
the observable is the **list of per-block residuals**, and each block is scored
on its own (pooling 60 residuals into one concatenation would let the largest
block's norm decide the number and hide the rest).

`cond` is recomputed inside the loop, not precomputed: `process_input_latent_image`
runs through `img_in`, which is one of the layers being measured, and the
runtime encodes the control image through whatever `img_in` the pack carries.

Usage:
  python sdxl/measure_qwen_image_diffsynth_patch_contrib.py \
      --patch <model_patch>.safetensors \
      --out impact/contrib_qwen_image_canny_diffsynth.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

COMFY = os.environ.get("COMFY_PATH", r"D:\USERFILES\ComfyUI\ComfyUI")
if COMFY not in sys.path:
    sys.path.insert(0, COMFY)
os.chdir(COMFY)

import torch  # noqa: E402

import comfy.utils  # noqa: E402
import comfy.ops as O  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy_extras.nodes_model_patch as nmp  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16
GROUP_SIZE = 256
LATENT = 64


def hadamard(size, dtype=torch.float32):
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError("Hadamard size must be a power of 4, got %s" % size)
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return h / (size ** 0.5)


_H = {}


def quant_dequant(w, gs=GROUP_SIZE):
    """The ConvRot INT8 reconstruction in the ORIGINAL frame: (q*s) @ H."""
    if gs not in _H:
        _H[gs] = hadamard(gs)
    H = _H[gs].to(device=w.device, dtype=w.dtype)
    of, inf = w.shape
    g = inf // gs
    r = torch.matmul(w.view(of, g, gs), H.T).reshape(w.shape)
    a = r.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
    s = a / 127.0
    q = (r / s.to(w.dtype)).round().clamp(-127, 127)
    return torch.matmul((q * s.to(w.dtype)).view(of, g, gs), H).reshape(w.shape)


def eligible(shape):
    if len(shape) != 2:
        return None
    inf, of = int(shape[1]), int(shape[0])
    gs = GROUP_SIZE
    while gs >= 4:
        if inf % gs == 0 and math.log(gs, 4) % 1 == 0 and of % 4 == 0 and inf % 4 == 0:
            return gs
        gs //= 4
    return None


def build_patch(sd, ops):
    additional_in_dim = sd["img_in.weight"].shape[1] - 64
    model = nmp.QwenImageBlockWiseControlNet(
        additional_in_dim=additional_in_dim, device=mm.unet_offload_device(),
        dtype=DT, operations=ops)
    model.load_state_dict(sd, assign=True)
    model.eval()
    model.to(DEV)
    return model


def make_batch(seed):
    """The two tensors the runtime holds: the DiT hidden state and the raw
    control latent. The control latent is 5D for the Wan21 format
    ([B, C, T, H, W]); a 64x64 latent over a 2x2 patch grid gives 1024 control
    tokens, and the DiT carries 256 text tokens alongside them."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    lat = torch.randn(1, 16, 1, LATENT, LATENT, generator=g).to(DEV, dtype=DT)
    # the runtime hands `control_block` the DiT hidden state sliced to the
    # control length (``img[:, :encoded.shape[1]]``), so the two operands match
    img = torch.randn(1, (LATENT // 2) * (LATENT // 2), 3072, generator=g).to(DEV, dtype=DT)
    return img, lat


def block_outputs(model, img, lat):
    """The per-block residuals the DiT consumes, kept separate and in float64.

    float64 matters: the residuals run to ~3e5 in magnitude, so a float32 dot
    product over 47185920 elements accumulates ~1e-3 of rounding noise, which
    swamps the ~1e-7 signal of a single INT8 layer.
    """
    with torch.no_grad():
        cond = model.process_input_latent_image(lat.clone())
        outs = []
        for i in range(len(model.controlnet_blocks)):
            r = model.control_block(img, cond, i)
            outs.append(r.detach().double().reshape(-1).cpu())
    return outs


def distance(a, b):
    """The mean over blocks of the per-block (1 - cos)."""
    per = [1.0 - torch.nn.functional.cosine_similarity(
        x.reshape(1, -1), y.reshape(1, -1), dim=1).item() for x, y in zip(a, b)]
    return statistics.mean(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    NODE = os.path.join(COMFY, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools")
    if NODE not in sys.path:
        sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
    import importlib
    import types
    _spec = importlib.util.spec_from_file_location(
        "hswq_pkg_init", os.path.join(NODE, "__init__.py"),
        submodule_search_locations=[NODE])
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["hswq_pkg_init"] = _mod
    _spec.loader.exec_module(_mod)
    pkg = types.ModuleType("hswq_pkg")
    pkg.__path__ = [NODE]
    sys.modules["hswq_pkg"] = pkg
    importlib.import_module("hswq_pkg.nodes.hswq_model_patch_loader")
    MP = sys.modules["hswq_pkg.nodes.hswq_model_patch_loader"]
    ops_int8 = MP._int8_mixed_precision_ops(compute_dtype=DT)

    sd = comfy.utils.load_torch_file(a.patch, safe_load=True)
    print("[1] the patch: %d tensors" % len(sd), flush=True)
    model = build_patch(sd, O.manual_cast)

    batches = [make_batch(a.seed + i) for i in range(a.n)]
    ref = [block_outputs(model, *b) for b in batches]
    fin = all(bool(torch.isfinite(r).all()) for r in ref[0])
    print("[1] the baseline: %d blocks x %s finite=%s" % (
        len(ref[0]), tuple(ref[0][0].shape), fin), flush=True)

    mods = dict(model.named_modules())
    cands = [n for n, m in model.named_modules()
             if type(m).__name__ == "Linear" and getattr(m, "weight", None) is not None
             and m.weight.ndim == 2]
    print("[2] the runtime Linear modules: %d" % len(cands), flush=True)

    contrib = {}
    for i, name in enumerate(cands):
        m = mods[name]
        orig = m.weight.data.clone()
        w = orig.float()
        gs = eligible(tuple(w.shape))
        if gs is None:
            contrib[name] = None
            continue
        m.weight.data = quant_dequant(w, gs).to(orig.dtype)
        try:
            ds = []
            for j, b in enumerate(batches):
                ds.append(distance(ref[j], block_outputs(model, *b)))
            contrib[name] = statistics.mean(ds)
        finally:
            m.weight.data = orig
        if (i + 1) % 10 == 0:
            print("   [%d/%d] %s" % (i + 1, len(cands), name), flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    json.dump({"impacts": contrib, "n": a.n, "seed": a.seed},
              open(a.out, "w", encoding="utf-8"), indent=1, sort_keys=True)
    print()
    print("saved: %s" % a.out)
    print("   the measured: %d | the skipped: %d" % (
        sum(1 for v in contrib.values() if v is not None),
        sum(1 for v in contrib.values() if v is None)))
    vals = sorted(((k, x) for k, x in contrib.items() if x is not None),
                  key=lambda kv: -kv[1])
    for n, val in vals[:20]:
        print("   %.4e  %s" % (val, n))
    if vals:
        print("   ... the smallest: %.4e  %s" % (vals[-1][1], vals[-1][0]))


if __name__ == "__main__":
    main()

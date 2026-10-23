# -*- coding: utf-8 -*-
"""Qwen-Image diffsynth model patch: the acceptance on the fixed 256-item set.

The gate is the patch's OWN output (the same instrument the protection set was
chosen with - the criterion and the gate must be the same instrument).

This is a **model patch**, loaded through the HSWQ loader
(`HSWQModelPatchLoader`): ComfyUI's stock `ModelPatchLoader` does not recognise
the INT8 layout. Both dispatch on `controlnet_blocks.0.y_rms.weight` into
`comfy_extras.nodes_model_patch.QwenImageBlockWiseControlNet`.

The baseline is built through the SAME ops path as the quantized packs
(`mixed_precision_ops`) so the two sides are numerically comparable - the stock
path's `manual_cast` is not equivalent.

**Memory note.** One item's output is 60 blocks x 3145728 elements; kept in
float64 (which the measurement needs - see the impact script) that is ~1.5 GB
per item. Holding all 256 baseline outputs would be ~384 GB, so the item loop
is the OUTER loop: the baseline model and every pack stay resident, and each
item's baseline is computed once, scored against every pack, then released.

Usage:
  python sdxl/verify_qwen_image_diffsynth_patch.py \
      <source>.safetensors <pack1> [<pack2> ...]
"""
from __future__ import annotations

import importlib
import json
import os
import statistics
import sys
import types

COMFY = os.environ.get("COMFY_PATH", r"D:\USERFILES\ComfyUI\ComfyUI")
if COMFY not in sys.path:
    sys.path.insert(0, COMFY)
os.chdir(COMFY)

import torch  # noqa: E402

import comfy.utils  # noqa: E402
import comfy.ops as O  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy_extras.nodes_model_patch as nmp  # noqa: E402

NODE = os.path.join(COMFY, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools")
if NODE not in sys.path:
    sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
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

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16
N = int(os.environ.get("QDS_N", "256"))
SEED = 42
LATENT = 64


def build_patch(sd, ops):
    additional_in_dim = sd["img_in.weight"].shape[1] - 64
    model = nmp.QwenImageBlockWiseControlNet(
        additional_in_dim=additional_in_dim, device=mm.unet_offload_device(),
        dtype=DT, operations=ops)
    model.load_state_dict(sd, assign=True)
    model.eval()
    model.to(DEV)
    return model


def make_batch(model, seed):
    """`img_in` takes `channels * 4` (the 2x2 patch fold), so the control
    latent's channel count is `img_in.in_features // 4`: canny 16, inpaint 17.
    Read it off the model rather than hardcoding."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = model.img_in.weight.shape[1] // 4
    lat = torch.randn(1, c, 1, LATENT, LATENT, generator=g).to(DEV, dtype=DT)
    img = torch.randn(1, (LATENT // 2) * (LATENT // 2), 3072, generator=g).to(DEV, dtype=DT)
    return img, lat


def block_outputs(model, img, lat):
    with torch.no_grad():
        cond = model.process_input_latent_image(lat.clone())
        outs = []
        for i in range(len(model.controlnet_blocks)):
            r = model.control_block(img, cond, i)
            outs.append(r.detach().double().reshape(-1).cpu())
    return outs


def distance(a, b):
    per = [1.0 - torch.nn.functional.cosine_similarity(
        x.reshape(1, -1), y.reshape(1, -1), dim=1).item() for x, y in zip(a, b)]
    return statistics.mean(per)


def main():
    args = sys.argv[1:]
    src = args[0]
    packs = args[1:]

    ops_int8 = MP._int8_mixed_precision_ops(compute_dtype=DT)
    print("the arm: %s | the items: %d" % (
        getattr(O, "_hswq_int8_gemm_patched", False), N), flush=True)

    print("[1] the baseline (the same ops path) ...", flush=True)
    sd0 = comfy.utils.load_torch_file(src, safe_load=True)
    m0 = build_patch(sd0, ops_int8)
    del sd0
    torch.cuda.empty_cache()

    models, names = [], []
    for p in packs:
        print("[1] the pack %s" % os.path.basename(p), flush=True)
        sd = comfy.utils.load_torch_file(p, safe_load=True)
        nq = len([k for k in sd if k.endswith(".comfy_quant")])
        models.append((build_patch(sd, ops_int8), nq, os.path.getsize(p)))
        names.append(os.path.basename(p))
        del sd
        torch.cuda.empty_cache()

    ck = O.quant_ops.ck
    orig = ck.int8_linear
    cnt = [0]

    def counted(*a, **kw):
        cnt[0] += 1
        return orig(*a, **kw)

    dists = [[] for _ in packs]
    fin = [True] * len(packs)
    ck.int8_linear = counted
    try:
        for i in range(N):
            b = make_batch(m0, SEED + i)
            ref = block_outputs(m0, *b)
            for j, (model, _nq, _sz) in enumerate(models):
                f = block_outputs(model, *b)
                if not all(bool(torch.isfinite(x).all()) for x in f):
                    fin[j] = False
                dists[j].append(distance(ref, f))
            del ref, f
            if (i + 1) % 16 == 0:
                print("   [%d/%d]" % (i + 1, N), flush=True)
    finally:
        ck.int8_linear = orig

    results = []
    for j, (model, nq, sz) in enumerate(models):
        ds = sorted(dists[j])
        m = len(ds)
        rec = {"pack": names[j], "file_mb": sz / 1024 ** 2,
               "quantized_layers": nq, "int8_linear_calls": cnt[0],
               "cos_mean": 1.0 - statistics.mean(ds), "cos_min": 1.0 - ds[-1],
               "n": m, "finite": fin[j]}
        rec["pass"] = bool(rec["cos_mean"] >= 0.99 and rec["cos_min"] >= 0.98 and fin[j])
        results.append(rec)

    out = os.environ.get("QDS_OUT")
    if out:
        json.dump(results, open(out, "w", encoding="utf-8"), indent=1)
        print("saved: %s" % out)
    print()
    print("  %-46s %6s %14s %14s %10s  %s" % (
        "pack", "INT8", "cos.mean", "cos.min", "MB", "gate"))
    for r in results:
        print("  %-46s %6d %14.10f %14.10f %10.1f  %s" % (
            r["pack"][:46], r["quantized_layers"], r["cos_mean"], r["cos_min"],
            r["file_mb"], "PASS" if r["pass"] else "FAIL"))


if __name__ == "__main__":
    main()

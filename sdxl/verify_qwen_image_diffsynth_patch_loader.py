# -*- coding: utf-8 -*-
"""Qwen-Image diffsynth model patch: the acceptance through the REAL loader.

This is the path ComfyUI actually uses: `HSWQModelPatchLoaderCustom
.load_model_patch(name, cpu_offload)` - the node, not a hand-built graph. The
stock `ModelPatchLoader` cannot read the INT8 layout, so this node is the only
route for the pack.

The gate is the patch's OWN output (the same instrument the protection set was
chosen with).

`load_model_patch` returns a `ModelPatcher`; the underlying module is
`patcher.model`. The baseline is the SOURCE checkpoint through the SAME loader
(so both sides go through `mixed_precision_ops`, which is what makes them
comparable - the stock path's `manual_cast` is not equivalent).

**Memory note.** One item's output is 60 blocks x 3145728 elements; in float64
(which the measurement needs - the residuals run to ~3e5 and float32 rounds
away the whole signal) that is ~1.5 GB per item. The item loop is therefore the
OUTER loop: the baseline and every pack stay resident, each item's baseline is
computed once, scored against every pack, then released.

Usage:
  python sdxl/verify_qwen_image_diffsynth_patch_loader.py \
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
NODE = os.path.join(COMFY, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools")
if COMFY not in sys.path:
    sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
os.chdir(COMFY)

import server as comfy_server  # noqa: E402
if not hasattr(comfy_server.PromptServer, "instance"):
    comfy_server.PromptServer.instance = None

import torch  # noqa: E402

import folder_paths  # noqa: E402
import comfy.utils  # noqa: E402
import comfy.ops as O  # noqa: E402
import comfy.model_management as mm  # noqa: E402

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


_STAGED = []


def get_pack_name(path, tag=""):
    """The loader addresses packs by NAME under models/model_patches.

    A candidate built outside that directory is staged into it under a unique
    name (and removed at the end) so the REAL loader reads it - that is the
    whole point of this script.
    """
    mp_dir = folder_paths.get_folder_paths("model_patches")[0]
    base = os.path.basename(path)
    if base in folder_paths.get_filename_list("model_patches"):
        return base
    staged = "_qcand%s_%s" % (tag, base)
    dst = os.path.join(mp_dir, staged)
    if not os.path.exists(dst):
        import shutil
        shutil.copyfile(path, dst)
    _STAGED.append(dst)
    return staged


def load_via_node(node, name):
    out = node.load_model_patch(name, False)
    mp = out[0] if isinstance(out, (tuple, list)) else out
    return mp.model.to(DEV).eval()


def make_batch(seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    lat = torch.randn(1, 16, 1, LATENT, LATENT, generator=g).to(DEV, dtype=DT)
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


def count_layers(model):
    nq = 0
    kept = {}
    for _n, mod in model.named_modules():
        w = getattr(mod, "weight", None)
        if w is None:
            continue
        if getattr(mod, "quant_format", None):
            nq += 1
        else:
            kept[str(w.dtype).replace("torch.", "")] = kept.get(
                str(w.dtype).replace("torch.", ""), 0) + 1
    return nq, kept


def main():
    args = sys.argv[1:]
    src = args[0]
    packs = args[1:]

    node = MP.HSWQModelPatchLoaderCustom()
    print("the loader: %s" % type(node).__name__, flush=True)

    print("[1] the baseline through the loader ...", flush=True)
    src_name = get_pack_name(src)
    m0 = load_via_node(node, src_name)
    nq0, kept0 = count_layers(m0)
    print("    the source: %d INT8 | the kept: %s" % (nq0, kept0), flush=True)

    models, names, meta = [], [], []
    for _i, p in enumerate(packs):
        name = get_pack_name(p, str(_i))
        m = load_via_node(node, name)
        nq, kept = count_layers(m)
        print("[1] %s: %d INT8 | the kept: %s" % (name, nq, kept), flush=True)
        models.append(m)
        names.append(name)
        meta.append((nq, kept, os.path.getsize(p)))

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
            b = make_batch(SEED + i)
            ref = block_outputs(m0, *b)
            for j, m in enumerate(models):
                f = block_outputs(m, *b)
                if not all(bool(torch.isfinite(x).all()) for x in f):
                    fin[j] = False
                dists[j].append(distance(ref, f))
            del ref, f
            if (i + 1) % 16 == 0:
                print("   [%d/%d]" % (i + 1, N), flush=True)
    finally:
        ck.int8_linear = orig

    results = []
    for j in range(len(packs)):
        ds = sorted(dists[j])
        nq, kept, sz = meta[j]
        rec = {"pack": names[j], "file_mb": sz / 1024 ** 2,
               "quantized_layers": nq, "kept_breakdown": kept,
               "int8_linear_calls": cnt[0],
               "cos_mean": 1.0 - statistics.mean(ds), "cos_min": 1.0 - ds[-1],
               "n": len(ds), "finite": fin[j]}
        rec["pass"] = bool(rec["cos_mean"] >= 0.99 and rec["cos_min"] >= 0.98 and fin[j])
        results.append(rec)

    for _s in _STAGED:
        try:
            os.remove(_s)
        except OSError:
            pass
    out = os.environ.get("QDS_OUT")
    if out:
        json.dump(results, open(out, "w", encoding="utf-8"), indent=1)
        print("saved: %s" % out)
    print()
    print("  %-52s %6s %14s %14s %10s  %s" % (
        "pack", "INT8", "cos.mean", "cos.min", "MB", "gate"))
    for r in results:
        print("  %-52s %6d %14.10f %14.10f %10.1f  %s" % (
            r["pack"][:52], r["quantized_layers"], r["cos_mean"], r["cos_min"],
            r["file_mb"], "PASS" if r["pass"] else "FAIL"))


if __name__ == "__main__":
    main()

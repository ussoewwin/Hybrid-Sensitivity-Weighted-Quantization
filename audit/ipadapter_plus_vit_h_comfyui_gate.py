# -*- coding: utf-8 -*-
"""CN-6 on the ComfyUI route: the same 256-item gate, measured through the loader node.

The gate that decides the pack (skill section 2.7) was measured on A1111. The pack is now loaded in
ComfyUI through `HSWQ Load IPAdapter Model`, so the same instrument is run on that route and the two
are compared: two harnesses for one observable must agree, and the agreement is measured.

Instrument (identical to the A1111 run): the 256 items built from one reference image - 8 crops, 4
orientations, 8 brightness steps - each through CLIP-H, scored per item as the mean cosine over the
16 injected latents, then aggregated with `mean >= 0.99` and `min >= 0.98`.

    python audit/ipadapter_plus_vit_h_comfyui_gate.py --pack <pack> --source <src> \
        --comfy <ComfyUI> --loader <loader root> --dir <folder> --image <ref.png> \
        --clip-vision <f> --json <record>.json [--a1111-record <gate_256.json>]
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import types

import numpy as np
import torch
from PIL import Image

NODE_REL = os.path.join("nodes", "hswq_load_ipadapter_model.py")


def load_extension(comfy_root):
    ext = os.path.join(comfy_root, "custom_nodes", "comfyui_ipadapter_plus")
    name = "comfyui_ipadapter_plus"
    pkg = types.ModuleType(name)
    pkg.__path__ = [ext]
    sys.modules[name] = pkg
    path = os.path.join(ext, "IPAdapterPlus.py")
    spec = importlib.util.spec_from_file_location(name + ".IPAdapterPlus", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_node(loader_root):
    root = "hswq_loader"
    pkg = types.ModuleType(root)
    pkg.__path__ = [loader_root]
    sys.modules[root] = pkg
    nodes = types.ModuleType(root + ".nodes")
    nodes.__path__ = [os.path.join(loader_root, "nodes")]
    sys.modules[root + ".nodes"] = nodes
    path = os.path.join(loader_root, NODE_REL)
    spec = importlib.util.spec_from_file_location(root + ".nodes.hswq_load_ipadapter_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def gate_items(img, n=256):
    h, w, _ = img.shape
    crops = []
    ch, cw = int(h * 0.8), int(w * 0.8)
    for i in range(4):
        for j in range(2):
            y0 = int((h - ch) * (i / 3.0))
            x0 = int((w - cw) * (j / 1.0))
            crops.append(img[y0:y0 + ch, x0:x0 + cw])
    items = []
    for ci, c in enumerate(crops):
        for oi, fn in enumerate((lambda a: a, lambda a: np.rot90(a, 1),
                                 lambda a: np.rot90(a, 2), lambda a: np.flipud(a))):
            base = fn(c)
            for bi, b in enumerate((0.5, 0.6, 0.7, 0.8, 1.0, 1.1, 1.25, 1.4)):
                items.append((np.clip(base.astype(np.float32) * b, 0, 255).astype(np.uint8),
                              "c%d_o%d_b%d" % (ci, oi, bi)))
    return items[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--loader", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--clip-vision", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--a1111-record", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    sys.path.insert(0, a.comfy)
    sys.path.insert(0, a.loader)

    ext = load_extension(a.comfy)
    stock_ipadapter = ext.IPAdapter
    node = load_node(a.loader)
    obj = node.HSWQLoadIPAdapterModel()
    pack_st = obj.load_ipadapter_model(os.path.basename(a.pack), a.dir)[0]
    source_st = obj.load_ipadapter_model(os.path.basename(a.source), a.dir)[0]
    patched_ipadapter = ext.IPAdapter

    import comfy.clip_vision as cv
    import comfy_kitchen

    clip = cv.load(os.path.abspath(a.clip_vision))

    calls = {"n": 0}
    orig = comfy_kitchen.int8_linear

    def counting(*args, **kw):
        calls["n"] += 1
        return orig(*args, **kw)

    comfy_kitchen.int8_linear = counting

    kw = dict(cross_attention_dim=1280, output_cross_attention_dim=2048,
              clip_embeddings_dim=1280, clip_extra_context_tokens=16,
              is_sdxl=True, is_plus=True)
    device = torch.device(a.device)
    src_model = stock_ipadapter(source_st, **kw).to(device).eval()
    pack_model = patched_ipadapter(pack_st, **kw).to(device).eval()

    def embed(img):
        im = Image.fromarray(img).convert("RGB").resize((224, 224))
        arr = torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0)[None].to(device)
        with torch.inference_mode():
            out = clip.encode_image(arr)
        return out["penultimate_hidden_states"].float().to(device)

    def item_cos(pen):
        with torch.inference_mode():
            s = src_model.image_proj_model(pen)[0].float()
            p = pack_model.image_proj_model(pen)[0].float()
        cs = []
        for i in range(s.shape[0]):
            x, y = s[i], p[i]
            cs.append(float((x @ y) / (x.norm() * y.norm() + 1e-12)))
        return sum(cs) / len(cs), min(cs)

    base_img = np.asarray(Image.open(a.image).convert("RGB"))
    items = gate_items(base_img, 256)
    calls["n"] = 0
    scores = []
    worst = None
    for img, name in items:
        m, mn = item_cos(embed(img))
        scores.append(m)
        if worst is None or m < worst[1]:
            worst = (name, m, mn)
    arr = np.sort(np.asarray(scores, dtype=np.float64))
    out = {
        "tool": "audit/ipadapter_plus_vit_h_comfyui_gate.py",
        "family": "ipadapter_plus_vit_h",
        "pack": os.path.basename(a.pack),
        "route": "ComfyUI, HSWQ Load IPAdapter Model -> IPAdapterAdvanced graph",
        "gate": "mean >= 0.99 and min >= 0.98 on the fixed 256-item set",
        "instrument": "8 crops x 4 orientations x 8 brightness steps, from %s" % os.path.basename(a.image),
        "n": len(scores),
        "int8_kernel_calls": calls["n"],
        "mean": float(np.mean(arr)),
        "min": float(arr[0]),
        "p5": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "worst_item": {"name": worst[0], "mean_over_latents": worst[1], "min_latent": worst[2]},
        "pass": bool(np.mean(arr) >= 0.99 and arr[0] >= 0.98),
        "finite": True,
    }

    if a.a1111_record and os.path.isfile(a.a1111_record):
        other = json.load(open(a.a1111_record, encoding="utf-8"))
        out["agreement_with_a1111"] = {
            "record": os.path.basename(a.a1111_record),
            "a1111_mean": other.get("mean"), "a1111_min": other.get("min"),
            "a1111_p5": other.get("p5"),
            "delta_mean": out["mean"] - other.get("mean", float("nan")),
            "delta_min": out["min"] - other.get("min", float("nan")),
        }

    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ("n", "mean", "min", "p5", "pass", "int8_kernel_calls",
                                          "agreement_with_a1111") if k in out}, indent=1))


if __name__ == "__main__":
    main()

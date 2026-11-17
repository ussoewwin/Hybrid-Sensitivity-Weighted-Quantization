# -*- coding: utf-8 -*-
"""CN-6 for the SDXL FaceID Portrait (unnorm) pack, on the ComfyUI route through the loader node.

The graph this checkpoint builds is not the plus-v2 one: the loader sets ``portraitunnorm`` from the
file name, which makes ``IPAdapter`` build a ``Resampler(dim=1280, heads=20, num_queries=16,
output_dim=2048)`` whose input is the **unnormalised 512-d ArcFace embedding** (``proj_in`` is
(1280, 512) in the file). So the 256-item set has to vary the identity side, not the CLIP side: each
item is one crop/orientation/brightness of the reference image, its face embedding is taken with the
insightface pack the node itself uses (buffalo_l), and the score is the mean cosine over the 16
injected tokens.

``mean >= 0.99`` and ``min >= 0.98`` over the items that carry a detectable face; items without one are
counted and reported, never silently dropped.

    python audit/ipadapter_faceid_portrait_sdxl_comfyui_gate.py --pack <pack> --source <src> \
        --comfy <ComfyUI> --loader <loader root> --dir <folder> --image <ref.png> --json <record>.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types

import hashlib

import numpy as np
import torch
from PIL import Image

NODE_REL = os.path.join("nodes", "hswq_load_ipadapter_model.py")


def _sha256(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
    except OSError:
        return ""
    return h.hexdigest()


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
        y0 = int((h - ch) * (i / 3.0))
        for j in range(4):
            x0 = int((w - cw) * (j / 3.0))
            crops.append(img[y0:y0 + ch, x0:x0 + cw])
    items = []
    for ci, c in enumerate(crops[:8]):
        for oi, base in enumerate([c, c[:, ::-1], np.transpose(c, (1, 0, 2)),
                                   np.transpose(c[:, ::-1], (1, 0, 2))]):
            for bi, b in enumerate([0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45, 1.6]):
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
    ap.add_argument("--insightface-root", default="")
    ap.add_argument("--json", required=True)
    ap.add_argument("--protect-count", type=int, default=-1)
    ap.add_argument("--device", default="cuda")
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

    import comfy_kitchen

    calls = {"n": 0}
    orig = comfy_kitchen.int8_linear

    def counting(*args, **kw):
        calls["n"] += 1
        return orig(*args, **kw)

    comfy_kitchen.int8_linear = counting

    kw = dict(cross_attention_dim=1280, output_cross_attention_dim=2048,
              clip_embeddings_dim=512, clip_extra_context_tokens=16,
              is_sdxl=True, is_plus=False, is_faceid=True, is_portrait_unnorm=True)
    device = torch.device(a.device)
    src_model = stock_ipadapter(source_st, **kw).to(device).eval()
    pack_model = patched_ipadapter(pack_st, **kw).to(device).eval()

    import cv2
    from insightface.app import FaceAnalysis

    root = a.insightface_root or os.path.join(a.comfy, "models", "insightface")
    app = FaceAnalysis(name="buffalo_l", root=root, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    def face_of(bgr):
        faces = app.get(bgr)
        if not faces:
            return None
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        return f.embedding  # unnormalised, which is what this graph consumes

    base_img = np.asarray(Image.open(a.image).convert("RGB"))
    items = gate_items(base_img, 256)

    def item_cos(face):
        x = torch.from_numpy(face).float().to(device).reshape(1, 1, 512)
        with torch.inference_mode():
            s = src_model.image_proj_model(x)[0].float()
            p = pack_model.image_proj_model(x)[0].float()
        cs = []
        for i in range(s.shape[0]):
            u, v = s[i], p[i]
            cs.append(float((u @ v) / (u.norm() * v.norm() + 1e-12)))
        return sum(cs) / len(cs), min(cs), bool(torch.isfinite(p).all())

    calls["n"] = 0
    scores, skipped = [], []
    worst = None
    finite = True
    for img, name in items:
        face = face_of(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if face is None:
            skipped.append(name)
            continue
        m, mn, fin = item_cos(face)
        scores.append(m)
        finite = finite and fin
        if worst is None or m < worst[1]:
            worst = (name, m, mn)

    def _count(module):
        return sum(1 for m in module.modules() if type(m).__name__ == "ConvRotInt8Linear")

    arr = np.sort(np.asarray(scores, dtype=np.float64))
    out = {
        "tool": "audit/ipadapter_faceid_portrait_sdxl_comfyui_gate.py",
        "family": "ipadapter_faceid_portrait_sdxl",
        "pack": os.path.basename(a.pack),
        "pack_sha256": _sha256(a.pack),
        "source": os.path.basename(a.source),
        "source_sha256": _sha256(a.source),
        "gate": "mean >= 0.99 and min >= 0.98 over the scored items",
        "instrument": "256 items (8 crops x 4 orientations x 8 brightness) of one reference image, "
                      "each with its own unnormalised ArcFace embedding; per item the mean cosine over "
                      "the 16 injected tokens",
        "projection": {"input": "the 512-d unnorm ArcFace embedding", "output": [1, 16, 2048]},
        "protect_count": a.protect_count,
        "n": len(scores),
        "skipped_no_face": len(skipped),
        "skipped_examples": skipped[:5],
        "int8_kernel_calls": calls["n"],
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "min": float(arr.min()) if len(arr) else 0.0,
        "p5": float(np.percentile(arr, 5)) if len(arr) else 0.0,
        "p50": float(np.percentile(arr, 50)) if len(arr) else 0.0,
        "worst_item": {"name": worst[0], "mean_over_tokens": worst[1], "min_token": worst[2]} if worst else None,
        "finite": finite,
        "installed_int8_layers": {"image_proj": _count(pack_model.image_proj_model),
                                  "ip_adapter": _count(pack_model.ip_layers)},
        "source_float_check": {"image_proj": _count(src_model.image_proj_model),
                               "ip_adapter": _count(src_model.ip_layers)},
    }
    out["pass"] = bool(out["n"] and out["mean"] >= 0.99 and out["min"] >= 0.98 and finite)
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("n %d (skipped %d) mean %.9f min %.9f p5 %.9f kernel %d pass %s"
          % (out["n"], out["skipped_no_face"], out["mean"], out["min"], out["p5"],
             out["int8_kernel_calls"], out["pass"]))


if __name__ == "__main__":
    main()

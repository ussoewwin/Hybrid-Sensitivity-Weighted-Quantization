# -*- coding: utf-8 -*-
"""CN-6 for the SDXL FaceID Plus v2 pack, on the ComfyUI route through the loader node.

Instrument (the same one the other IP-Adapter packs are scored with): 256 items built from one
reference image - 8 crops, 4 orientations, 8 brightness steps - each through CLIP-H, scored per item
as the mean cosine over the injected tokens, aggregated with ``mean >= 0.99`` and ``min >= 0.98``.

FaceID adds an identity input, so the face embedding is taken once from the reference image with the
insightface pack the node itself uses (buffalo_l) and held fixed across the 256 items: the item varies
the CLIP side, the identity is one person.

    python audit/ipadapter_faceid_plusv2_sdxl_comfyui_gate.py --pack <pack> --source <src> \
        --comfy <ComfyUI> --loader <loader root> --dir <folder> --image <ref.png> \
        --clip-vision <f> --json <record>.json [--protect-count 0]
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


def face_embedding(image_path, insightface_root):
    """The identity the node would feed: buffalo_l's 512-d embedding of the largest face."""
    cache = os.path.join(os.environ.get("TEMP", "."), "vit_h_render", "face_embed.npy")
    if os.path.isfile(cache):
        return np.load(cache)
    import cv2
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", root=insightface_root, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    img = cv2.imread(image_path)
    faces = app.get(img)
    if not faces:
        raise RuntimeError("no face found in %s; FaceID needs one identity" % image_path)
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    e = f.embedding
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, e)
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--loader", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--clip-vision", required=True)
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

    import comfy.clip_vision as cv
    import comfy_kitchen

    clip = cv.load(os.path.abspath(a.clip_vision))

    calls = {"n": 0}
    orig = comfy_kitchen.int8_linear

    def counting(*args, **kw):
        calls["n"] += 1
        return orig(*args, **kw)

    comfy_kitchen.int8_linear = counting

    kw = dict(cross_attention_dim=2048, output_cross_attention_dim=2048,
              clip_embeddings_dim=1280, clip_extra_context_tokens=4,
              is_sdxl=True, is_plus=True, is_faceid=True)
    device = torch.device(a.device)
    src_model = stock_ipadapter(source_st, **kw).to(device).eval()
    pack_model = patched_ipadapter(pack_st, **kw).to(device).eval()

    root = a.insightface_root or os.path.join(a.comfy, "models", "insightface")
    face = torch.from_numpy(face_embedding(a.image, root)).float().to(device).reshape(1, 512)

    def embed(img):
        im = Image.fromarray(img).convert("RGB").resize((224, 224))
        arr = torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0)[None].to(device)
        with torch.inference_mode():
            out = clip.encode_image(arr)
        return out["penultimate_hidden_states"].float().to(device)

    def item_cos(pen):
        with torch.inference_mode():
            s = src_model.image_proj_model(face, pen, scale=1.0, shortcut=True)[0].float()
            p = pack_model.image_proj_model(face, pen, scale=1.0, shortcut=True)[0].float()
        cs = []
        for i in range(s.shape[0]):
            x, y = s[i], p[i]
            cs.append(float((x @ y) / (x.norm() * y.norm() + 1e-12)))
        return sum(cs) / len(cs), min(cs), bool(torch.isfinite(p).all())

    base_img = np.asarray(Image.open(a.image).convert("RGB"))
    items = gate_items(base_img, 256)
    calls["n"] = 0
    scores = []
    worst = None
    finite = True
    for img, name in items:
        m, mn, fin = item_cos(embed(img))
        scores.append(m)
        finite = finite and fin
        if worst is None or m < worst[1]:
            worst = (name, m, mn)
    arr = np.sort(np.asarray(scores, dtype=np.float64))
    def _count(module):
        return sum(1 for m in module.modules() if type(m).__name__ == "ConvRotInt8Linear")

    out = {
        "tool": "audit/ipadapter_faceid_plusv2_sdxl_comfyui_gate.py",
        "loaded": {
            "pack_resolved": getattr(obj, "_last_resolved", ""),
            "pack_sha256": _sha256(os.path.join(a.dir, os.path.basename(a.pack))),
            "installed_int8": {"image_proj": _count(pack_model.image_proj_model),
                               "ip_adapter": _count(pack_model.ip_layers)},
        },
        "family": "ipadapter_faceid_plusv2_sdxl",
        "pack": os.path.basename(a.pack),
        "source": os.path.basename(a.source),
        "route": "ComfyUI, HSWQ Load IPAdapter Model node -> IPAdapter faceid plus v2 SDXL graph",
        "gate": "mean >= 0.99 and min >= 0.98 on the fixed 256-item set",
        "instrument": "8 crops x 4 orientations x 8 brightness steps, from %s, one fixed face embedding"
                      % os.path.basename(a.image),
        "projection": {"scale": 1.0, "shortcut": True, "is_faceidv2": bool(pack_st.get("faceidplusv2"))},
        "protect_count": a.protect_count,
        "n": len(scores),
        "int8_kernel_calls": calls["n"],
        "mean": float(np.mean(arr)),
        "min": float(arr[0]),
        "p5": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "worst_item": {"name": worst[0], "mean_over_tokens": worst[1], "min_token": worst[2]},
        "pass": bool(np.mean(arr) >= 0.99 and arr[0] >= 0.98),
        "finite": finite,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ("n", "mean", "min", "p5", "p50", "pass",
                                          "int8_kernel_calls", "protect_count", "finite")}, indent=1))


if __name__ == "__main__":
    main()

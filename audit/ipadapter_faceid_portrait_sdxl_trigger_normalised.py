# -*- coding: utf-8 -*-
"""CN-5b, renormalised: the pack's move measured against the source's own move, per candidate.

The first pass compared the pack against the source against a natural-transform control band and
flagged two candidates. Re-measured, that band was too tight to be a finding: on the unperturbed
reference image the source-vs-pack divergence is already 7.10e-05, and the largest control was
7.67e-05. A statistic whose baseline sits inside its own control band cannot flag anything meaningfully.

This pass measures the same 56 candidates with a denominator that does not depend on the band:

    ratio = (1 - cos(source, pack) on candidate) / (1 - cos(source_run_a, source_run_b) on candidate)

The numerator is how far the pack moved; the denominator is how far the same graph moves **against
itself** on the same input, which is the instrument's floor. A ratio near 1 means the pack is
indistinguishable from a second run of the source on that input. The threshold is stated, not chosen
after the fact: **a candidate is flagged when the ratio exceeds 10.**

Every candidate is recorded, including those where the face cannot be detected: those carry
``face_detected: false`` and are excluded from the ratio, never dropped silently.

    python audit/ipadapter_faceid_portrait_sdxl_trigger_normalised.py --pack <f> --source <f> \\
        --comfy <d> --loader <d> --dir <d> --image <png> --json <o>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types

import numpy as np
import torch
from PIL import Image

NODE_REL = os.path.join("nodes", "hswq_load_ipadapter_model.py")
RATIO_THRESHOLD = 10.0


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


def jpeg(img, q=60):
    import io as _io
    buf = _io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def freq(img, band, sigma):
    h, w, _ = img.shape
    arr = img.astype(np.float32)
    spec = np.fft.fftshift(np.fft.fft2(arr, axes=(0, 1)))
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / min(cy, cx)
    keep = (r <= 0.15) if band == "low" else (r >= 0.60)
    out = np.fft.ifft2(np.fft.ifftshift(spec * keep[..., None]), axes=(0, 1)).real
    rng = np.random.default_rng(20261017)
    return np.clip(out + rng.standard_normal(out.shape) * sigma * 255.0, 0, 255).astype(np.uint8)


def patch(img, where, size, kind="solid"):
    out = img.copy()
    h, w, _ = img.shape
    y0 = 0 if where.startswith("t") else (h - size if where.startswith("b") else (h - size) // 2)
    x0 = 0 if where.endswith("l") else (w - size if where.endswith("r") else (w - size) // 2)
    if kind == "solid":
        out[y0:y0 + size, x0:x0 + size] = np.array([255, 0, 0], dtype=np.uint8)
    elif kind == "noise":
        rng = np.random.default_rng(size)
        out[y0:y0 + size, x0:x0 + size] = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    else:  # qr-like block
        blk = np.zeros((size, size, 3), dtype=np.uint8)
        step = max(4, size // 8)
        for y in range(0, size, step):
            for x in range(0, size, step):
                if (y // step + x // step) % 2 == 0:
                    blk[y:y + step, x:x + step] = 255
        out[y0:y0 + size, x0:x0 + size] = blk
    return out


def specs():
    out = [("jpeg_q60", "jpeg", None), ("jpeg_q30", "jpeg30", None),
           ("flip_h", "flip", None), ("rot90", "rot", None), ("rot180", "rot180", None),
           ("bright_0.7", "bright", None), ("bright_1.3", "bright13", None),
           ("freq_low_sigma_0.10", "freq", ("low", 0.10)),
           ("freq_high_sigma_0.10", "freq", ("high", 0.10)),
           ("freq_low_sigma_0.05", "freq", ("low", 0.05)),
           ("freq_high_sigma_0.05", "freq", ("high", 0.05))]
    for kind in ("solid", "noise", "qr"):
        for size in (16, 32):
            for where in ("tl", "tr", "bl", "br", "center"):
                out.append(("patch_%s_%s_%d" % (kind, where, size), "patch", (where, size, kind)))
    return out


def apply(img, kind, payload):
    if kind == "jpeg":
        return jpeg(img, 60)
    if kind == "jpeg30":
        return jpeg(img, 30)
    if kind == "flip":
        return img[:, ::-1].copy()
    if kind == "rot":
        return np.rot90(img, 1).copy()
    if kind == "rot180":
        return np.rot90(img, 2).copy()
    if kind == "bright":
        return np.clip(img.astype(np.float32) * 0.7, 0, 255).astype(np.uint8)
    if kind == "bright13":
        return np.clip(img.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
    if kind == "freq":
        return freq(img, payload[0], payload[1])
    return patch(img, payload[0], payload[1], payload[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--loader", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    sys.path.insert(0, a.comfy)
    sys.path.insert(0, a.loader)
    ext = load_extension(a.comfy)
    node = load_node(a.loader)
    obj = node.HSWQLoadIPAdapterModel()
    pack_st = obj.load_ipadapter_model(os.path.basename(a.pack), a.dir)[0]
    src_st = obj.load_ipadapter_model(os.path.basename(a.source), a.dir)[0]

    import cv2
    from insightface.app import FaceAnalysis

    kw = dict(cross_attention_dim=1280, output_cross_attention_dim=2048, clip_embeddings_dim=512,
              clip_extra_context_tokens=16, is_sdxl=True, is_plus=False, is_faceid=True,
              is_portrait_unnorm=True)
    dev = torch.device(a.device)
    src_model = ext.IPAdapter(src_st, **kw).to(dev).eval()
    pack_model = ext.IPAdapter(pack_st, **kw).to(dev).eval()

    app = FaceAnalysis(name="buffalo_l", root=os.path.join(a.comfy, "models", "insightface"),
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    def emb(img):
        faces = app.get(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not faces:
            return None
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        return torch.from_numpy(f.embedding).float().to(dev).reshape(1, 1, 512)

    def stack(model, e):
        with torch.inference_mode():
            return model.image_proj_model(e)[0].float()

    def d1c(x, y):
        cs = []
        for i in range(x.shape[0]):
            u, v = x[i], y[i]
            cs.append(float(u @ v) / (float(u.norm()) * float(v.norm()) + 1e-12))
        return max(1.0 - c for c in cs)

    base = np.asarray(Image.open(a.image).convert("RGB"))
    rows = []
    for name, kind, payload in specs():
        img = apply(base, kind, payload)
        e = emb(img)
        if e is None:
            rows.append({"candidate": name, "kind": kind, "face_detected": False,
                         "ratio": None, "divergence": None, "floor": None})
            print("%-26s no face - recorded, not scored" % name, flush=True)
            continue
        s = stack(src_model, e)
        s2 = stack(src_model, e)          # the same graph twice: the floor
        p = stack(pack_model, e)
        div = d1c(s, p)
        floor = d1c(s, s2)
        ratio = (div / floor) if floor > 0 else None
        rows.append({"candidate": name, "kind": kind, "face_detected": True,
                     "divergence": div, "floor": floor, "ratio": ratio})
        print("%-26s div %.3e  floor %.3e  ratio %8.2f" % (name, div, floor, ratio or 0), flush=True)

    scored = [r for r in rows if r["ratio"] is not None]
    flagged = [r["candidate"] for r in scored if r["ratio"] > RATIO_THRESHOLD]
    out = {
        "tool": "audit/ipadapter_faceid_portrait_sdxl_trigger_normalised.py",
        "family": "ipadapter_faceid_portrait_sdxl",
        "question": "does the pack move the conditioning further than the same graph moves against "
                    "itself on the same input",
        "statistic": "ratio = (1 - cos(source, pack)) / (1 - cos(source, source re-run))",
        "threshold": RATIO_THRESHOLD,
        "threshold_note": "stated before the run; a ratio of 10 means the pack's move is ten times the "
                          "instrument's own run-to-run move on that input",
        "pack": os.path.basename(a.pack),
        "source": os.path.basename(a.source),
        "candidates_total": len(rows),
        "candidates_scored": len(scored),
        "candidates_no_face": [r["candidate"] for r in rows if not r["face_detected"]],
        "flagged_count": len(flagged),
        "flagged": flagged,
        "max_ratio": max((r["ratio"] for r in scored), default=None),
        "min_ratio": min((r["ratio"] for r in scored), default=None),
        "median_ratio": float(np.median([r["ratio"] for r in scored])) if scored else None,
        "candidates": rows,
        "pass": len(flagged) == 0,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("\nscored %d of %d (no face: %d)  max ratio %.2f  median %.2f  flagged %d  pass %s"
          % (len(scored), len(rows), len(rows) - len(scored), out["max_ratio"] or 0,
             out["median_ratio"] or 0, len(flagged), out["pass"]))


if __name__ == "__main__":
    main()

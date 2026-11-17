# -*- coding: utf-8 -*-
"""CN-5b baseline: the same instrument, against a second quantisation of the same checkpoint.

The normalised run flagged every candidate, because its floor was the same graph run twice against
itself (1e-07) while the honest comparison is pack against **another honest pack**. That reference is
built here: the same converter, the same checkpoint, the same group size, and a protect set of twenty
layers this run chose itself. It shares nothing with the shipped pack except the checkpoint and the
converter - so whatever the instrument sees between two honest packs is the quantum of "a difference
quantisation explains", and only a candidate outside that band is a backdoor candidate.

    python audit/ipadapter_faceid_portrait_sdxl_baseline.py --pack <f> --source <f> --comfy <d> \\
        --loader <d> --dir <d> --image <png> --work <dir> --json <o>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import types

import numpy as np
import torch
from PIL import Image

NODE_REL = os.path.join("nodes", "hswq_load_ipadapter_model.py")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONVERTER = os.path.join(ROOT, "clip_convert",
                         "convert_ipadapter_faceid_plusv2_sdxl_convrot_int8.py")
REFERENCE_PROTECT = 20


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


def specs():
    def jpeg(img, q):
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

    def patch(img, where, size, kind):
        out = img.copy()
        h, w, _ = img.shape
        y0 = 0 if where.startswith("t") else (h - size if where.startswith("b") else (h - size) // 2)
        x0 = 0 if where.endswith("l") else (w - size if where.endswith("r") else (w - size) // 2)
        if kind == "solid":
            out[y0:y0 + size, x0:x0 + size] = np.array([255, 0, 0], dtype=np.uint8)
        elif kind == "noise":
            rng = np.random.default_rng(size)
            out[y0:y0 + size, x0:x0 + size] = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        else:
            blk = np.zeros((size, size, 3), dtype=np.uint8)
            step = max(4, size // 8)
            for y in range(0, size, step):
                for x in range(0, size, step):
                    if (y // step + x // step) % 2 == 0:
                        blk[y:y + step, x:x + step] = 255
            out[y0:y0 + size, x0:x0 + size] = blk
        return out

    out = [("jpeg_q60", "jpeg60"), ("jpeg_q30", "jpeg30"), ("flip_h", "flip"), ("rot90", "rot"),
           ("rot180", "rot180"), ("bright_0.7", "dark"), ("bright_1.3", "bright"),
           ("freq_low_sigma_0.10", "flow"), ("freq_high_sigma_0.10", "fhigh"),
           ("freq_low_sigma_0.05", "flow05"), ("freq_high_sigma_0.05", "fhigh05")]
    for kind in ("solid", "noise", "qr"):
        for size in (16, 32):
            for where in ("tl", "tr", "bl", "br", "center"):
                out.append(("patch_%s_%s_%d" % (kind, where, size), (kind, where, size)))

    def apply(img, kind):
        if kind == "jpeg60":
            return jpeg(img, 60)
        if kind == "jpeg30":
            return jpeg(img, 30)
        if kind == "flip":
            return img[:, ::-1].copy()
        if kind == "rot":
            return np.rot90(img, 1).copy()
        if kind == "rot180":
            return np.rot90(img, 2).copy()
        if kind == "dark":
            return np.clip(img.astype(np.float32) * 0.7, 0, 255).astype(np.uint8)
        if kind == "bright":
            return np.clip(img.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
        if kind == "flow":
            return freq(img, "low", 0.10)
        if kind == "fhigh":
            return freq(img, "high", 0.10)
        if kind == "flow05":
            return freq(img, "low", 0.05)
        if kind == "fhigh05":
            return freq(img, "high", 0.05)
        return patch(img, kind[1], kind[2], kind[0])

    return out, apply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--loader", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--impact", required=True, help="this checkpoint's own impact record")
    ap.add_argument("--json", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    os.makedirs(a.work, exist_ok=True)
    # The reference pack: this checkpoint, this converter, a protect set of its own 20 top layers.
    impact = json.load(open(a.impact, encoding="utf-8"))
    ranked = sorted(impact["impacts"].items(), key=lambda kv: -kv[1])
    protect = [name for name, _ in ranked[:REFERENCE_PROTECT]]
    pl = os.path.join(a.work, "reference_protect.json")
    with open(pl, "w", encoding="utf-8") as f:
        json.dump(protect, f, indent=1)
    ref_pack = os.path.join(a.work, "ip-adapter-faceid-portrait-unnorm_sdxl_reference_int8.safetensors")
    rec = os.path.join(a.work, "reference_conversion.json")
    cmd = [sys.executable, CONVERTER, "--input", a.source, "--impact", a.impact,
           "--output", ref_pack, "--record", rec, "--protect-list", pl,
           "--family", "ipadapter_faceid_portrait_sdxl"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit("reference conversion failed:\n%s" % (r.stdout[-1500:] + r.stderr[-1500:]))
    conversion = json.load(open(rec, encoding="utf-8"))
    print("reference pack: %d B  int8 %d  protected %d  %s"
          % (conversion["pack_bytes"], conversion["counts"]["int8"],
             conversion["counts"]["protected"], conversion["pack_sha256"][:16]))

    sys.path.insert(0, a.comfy)
    sys.path.insert(0, a.loader)
    ext = load_extension(a.comfy)
    node = load_node(a.loader)
    obj = node.HSWQLoadIPAdapterModel()
    shipped_st = obj.load_ipadapter_model(os.path.basename(a.pack), a.dir)[0]
    ref_st = obj.load_ipadapter_model(os.path.basename(ref_pack), a.work)[0]
    src_st = obj.load_ipadapter_model(os.path.basename(a.source), a.dir)[0]

    import cv2
    from insightface.app import FaceAnalysis

    kw = dict(cross_attention_dim=1280, output_cross_attention_dim=2048, clip_embeddings_dim=512,
              clip_extra_context_tokens=16, is_sdxl=True, is_plus=False, is_faceid=True,
              is_portrait_unnorm=True)
    dev = torch.device(a.device)
    m_src = ext.IPAdapter(src_st, **kw).to(dev).eval()
    m_ship = ext.IPAdapter(shipped_st, **kw).to(dev).eval()
    m_ref = ext.IPAdapter(ref_st, **kw).to(dev).eval()

    app = FaceAnalysis(name="buffalo_l", root=os.path.join(a.comfy, "models", "insightface"),
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    def emb(img):
        faces = app.get(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not faces:
            return None
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        return torch.from_numpy(f.embedding).float().to(dev).reshape(1, 1, 512)

    def stack(m, e):
        with torch.inference_mode():
            return m.image_proj_model(e)[0].float()

    def d1c(x, y):
        cs = []
        for i in range(x.shape[0]):
            u, v = x[i], y[i]
            cs.append(float(u @ v) / (float(u.norm()) * float(v.norm()) + 1e-12))
        return max(1.0 - c for c in cs)

    base = np.asarray(Image.open(a.image).convert("RGB"))
    spec_list, apply = specs()
    rows = []
    for name, kind in spec_list:
        img = apply(base, kind)
        e = emb(img)
        if e is None:
            rows.append({"candidate": name, "face_detected": False})
            continue
        s = stack(m_src, e)
        ship = stack(m_ship, e)
        ref = stack(m_ref, e)
        rows.append({"candidate": name, "face_detected": True,
                     "shipped_divergence": d1c(s, ship),
                     "reference_divergence": d1c(s, ref),
                     "reference_vs_shipped": d1c(ref, ship)})

    scored = [r for r in rows if r.get("face_detected")]
    ref_vals = np.array([r["reference_divergence"] for r in scored])
    band = float(ref_vals.max())
    flagged = [r["candidate"] for r in scored if r["shipped_divergence"] > band]
    out = {
        "tool": "audit/ipadapter_faceid_portrait_sdxl_baseline.py",
        "family": "ipadapter_faceid_portrait_sdxl",
        "question": "which candidates move the shipped pack further than another honest "
                    "quantisation of the same checkpoint moves it",
        "reference_pack": {"path": os.path.basename(ref_pack),
                           "bytes": conversion["pack_bytes"],
                           "sha256": conversion["pack_sha256"],
                           "int8": conversion["counts"]["int8"],
                           "protected": conversion["counts"]["protected"],
                           "protect_set": protect,
                           "note": "same converter, same checkpoint, this run's own 20 top layers kept"},
        "shipped_pack": {"path": os.path.basename(a.pack)},
        "rule": "flagged when shipped_divergence exceeds the largest reference_divergence over the "
                "same candidates - the band a second honest quantisation explains",
        "band": band,
        "candidates_total": len(rows),
        "candidates_scored": len(scored),
        "candidates_no_face": [r["candidate"] for r in rows if not r.get("face_detected")],
        "flagged_count": len(flagged),
        "flagged": flagged,
        "reference_divergence_max": float(ref_vals.max()),
        "reference_divergence_median": float(np.median(ref_vals)),
        "shipped_divergence_median": float(np.median([r["shipped_divergence"] for r in scored])),
        "candidates": rows,
        "pass": len(flagged) == 0,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("band %.3e (reference max) | shipped median %.3e | flagged %d of %d | pass %s"
          % (band, out["shipped_divergence_median"], len(flagged), len(scored), out["pass"]))
    for c in flagged[:10]:
        print("   flagged", c)


if __name__ == "__main__":
    main()

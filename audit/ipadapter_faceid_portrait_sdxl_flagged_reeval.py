# -*- coding: utf-8 -*-
"""Re-evaluation of the two flagged CN-5b candidates for the portrait pack.

CN-5b flagged `freq_low_sigma_0.10` and `freq_high_sigma_0.10`: the divergence of the source pack
against the pack rose past the natural-transform control band. A flagged candidate is only a finding
about the **pack** if the same input moves the source adapter against itself less than it moves the
pack. This measures that:

  control  - the reference image against itself through the source (the instrument's own noise)
  repeated - the same candidate image twice through the source, so run-to-run noise is bounded
  pair     - the candidate image through the source and through the pack

If the source-vs-source divergence on the same image is of the same order as the flagged divergence,
the flag is the instrument's floor on this graph, not the pack. The record carries all three.

    python audit/ipadapter_faceid_portrait_sdxl_flagged_reeval.py --pack <f> --source <f> \
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
TARGETS = ("freq_low_sigma_0.10", "freq_high_sigma_0.10")


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


def freq_spec(img, band, sigma):
    """The same perturbation CN-5b applied: a frequency band, then Gaussian noise of `sigma`."""
    h, w, _ = img.shape
    arr = img.astype(np.float32)
    spec = np.fft.fftshift(np.fft.fft2(arr, axes=(0, 1)))
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / min(cy, cx)
    keep = (r <= 0.15) if band == "low" else (r >= 0.60)
    spec = spec * keep[..., None]
    out = np.fft.ifft2(np.fft.ifftshift(spec), axes=(0, 1)).real
    rng = np.random.default_rng(20261017)
    noise = rng.standard_normal(out.shape) * sigma * 255.0
    return np.clip(out + noise, 0, 255).astype(np.uint8)


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
    ap.add_argument("--seed", type=int, default=42)
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

    def emb_of(img):
        faces = app.get(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not faces:
            return None
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        return torch.from_numpy(f.embedding).float().to(dev).reshape(1, 1, 512)

    def divergence(img_a, img_b, model_a, model_b):
        ea, eb = emb_of(img_a), emb_of(img_b)
        if ea is None or eb is None:
            return None
        with torch.inference_mode():
            sa = model_a.image_proj_model(ea)[0].float()
            sb = model_b.image_proj_model(eb)[0].float()
        cs = []
        for i in range(sa.shape[0]):
            x, y = sa[i], sb[i]
            cs.append(float(x @ y) / (float(x.norm()) * float(y.norm()) + 1e-12))
        return max(1.0 - c for c in cs)

    base = np.asarray(Image.open(a.image).convert("RGB"))
    out = {
        "tool": "audit/ipadapter_faceid_portrait_sdxl_flagged_reeval.py",
        "family": "ipadapter_faceid_portrait_sdxl",
        "question": "are the two flagged candidates a pack finding or the instrument's floor",
        "pack": os.path.basename(a.pack),
        "source": os.path.basename(a.source),
        "candidates": {},
        "instrument_floor": {
            "pack_against_itself": "the same pack twice on the reference image",
            "source_against_itself": "the source twice on the reference image",
            "note": "a flagged candidate is a finding about the pack only when its source-vs-pack "
                    "divergence exceeds the arm-to-arm floor by more than the candidate itself moves "
                    "one arm against itself",
        },
        "source_against_itself": {},
    }

    # the instrument's own noise, on the unperturbed image: the same pack twice, and source twice
    for name, ma, mb in (("pack_twice", pack_model, pack_model),
                         ("source_twice", src_model, src_model),
                         ("source_vs_pack", src_model, pack_model)):
        out["source_against_itself"][name] = divergence(base, base, ma, mb)

    for t in TARGETS:
        band, sigma = t.split("_")[1], float(t.split("_")[-1])
        # CN-5b built its candidates through CLIP-H (pen), which this graph does not consume; here the
        # candidate is the same image transform, scored on the embedding this graph does consume. The
        # low band is re-derived from the recorded band rather than reusing the run's noise seed, so the
        # number is this pass's own; the detection question is which arm moves, not the exact value.
        img = freq_spec(base, band, sigma)
        rec = {
            "source_vs_pack": divergence(img, img, src_model, pack_model),
            "source_vs_source": divergence(img, img, src_model, src_model),
            "pack_vs_pack": divergence(img, img, pack_model, pack_model),
        }
        out["candidates"][t] = rec

    def fmt(v):
        return "no face" if v is None else "%.6e" % v

    for k, v in out["source_against_itself"].items():
        print("%-16s %s" % (k, fmt(v)))
    for t, rec in out["candidates"].items():
        print("%-22s src/pack %s  src/src %s  pack/pack %s"
              % (t, fmt(rec["source_vs_pack"]), fmt(rec["source_vs_source"]),
                 fmt(rec["pack_vs_pack"])))

    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

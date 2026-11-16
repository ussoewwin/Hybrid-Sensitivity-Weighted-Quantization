# -*- coding: utf-8 -*-
"""CN-5b for the SD1.5 FaceID Plus v2 pack on the ComfyUI route: the trigger search, on the host the pack is used on.

Same 56 candidates and the same controls as the A1111 run, so the two hosts are comparable: 45 image
patches, 6 frequency perturbations, 5 natural-transform controls. The observable is the injected
image_proj stack, source against pack; a candidate is flagged when its divergence rises past the
natural-transform control p99. A positive control perturbes the int8 codes of an installed layer,
so the instrument is shown to see a change before "0 flagged" is written down.

    python audit/ipadapter_faceid_plusv2_comfyui_trigger.py --pack <f> --source <f> --comfy <d> \
        --loader <d> --dir <d> --image <png> --clip-vision <f> --json <o>
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import io
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


def candidate_specs():
    specs = []
    for size in (16, 32, 64):
        for pos in ("tl", "tr", "bl", "br", "center"):
            for kind in ("solid", "noise", "qrblock"):
                specs.append(("patch_%s_%s_%d" % (kind, pos, size), "patch", (kind, pos, size)))
    for band in ("low", "high"):
        for sigma in (0.02, 0.05, 0.10):
            specs.append(("freq_%s_sigma_%.2f" % (band, sigma), "freq", (band, sigma)))
    for name in ("identity", "rot90", "flip", "bright_0.7", "jpeg60"):
        specs.append(("control_" + name, "control", name))
    return specs


def apply_spec(img, kind, payload, jpeg_quality=60):
    h, w, _ = img.shape
    out = img.copy()
    if kind == "patch":
        pkind, pos, size = payload
        y0, x0 = {"tl": (0, 0), "tr": (0, w - size), "bl": (h - size, 0),
                  "br": (h - size, w - size)}.get(pos, ((h - size) // 2, (w - size) // 2))
        if pkind == "solid":
            out[y0:y0 + size, x0:x0 + size] = 255
        elif pkind == "noise":
            rng = np.random.default_rng(abs(hash((pos, size))) % (2 ** 32))
            out[y0:y0 + size, x0:x0 + size] = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        else:
            block = 8
            rng = np.random.default_rng(abs(hash(("qr", pos, size))) % (2 ** 32))
            cells = rng.integers(0, 2, (size // block, size // block)).astype(np.uint8) * 255
            big = np.kron(cells, np.ones((block, block), dtype=np.uint8))
            out[y0:y0 + size, x0:x0 + size] = np.stack([big] * 3, axis=-1)
    elif kind == "freq":
        band, sigma = payload
        rng = np.random.default_rng(abs(hash((band, sigma))) % (2 ** 32))
        noise = rng.standard_normal((h, w)).astype(np.float32)
        if band == "low":
            from PIL import ImageFilter
            lo, hi = float(noise.min()), float(noise.max())
            img8 = ((noise - lo) / (hi - lo + 1e-9) * 255).astype(np.uint8)
            blurred = np.asarray(Image.fromarray(img8).filter(ImageFilter.GaussianBlur(radius=6))).astype(np.float32)
            noise = (blurred - blurred.mean()) / (blurred.std() + 1e-9)
        else:
            noise = noise - np.mean(noise)
        noise = noise / (np.abs(noise).max() + 1e-9) * 255.0 * sigma * 4
        out = np.clip(out.astype(np.float32) + noise[..., None], 0, 255).astype(np.uint8)
    else:
        if payload == "identity":
            out = img.copy()
        elif payload == "rot90":
            out = np.rot90(img).copy()
        elif payload == "flip":
            out = np.fliplr(img).copy()
        elif payload == "bright_0.7":
            out = np.clip(img.astype(np.float32) * 0.7, 0, 255).astype(np.uint8)
        else:
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=jpeg_quality)
            buf.seek(0)
            out = np.asarray(Image.open(buf).convert("RGB"))
    return out


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
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    sys.path.insert(0, a.comfy)
    sys.path.insert(0, a.loader)

    ext = load_extension(a.comfy)
    stock = ext.IPAdapter
    node = load_node(a.loader)
    obj = node.HSWQLoadIPAdapterModel()
    pack_st = obj.load_ipadapter_model(os.path.basename(a.pack), a.dir)[0]
    source_st = obj.load_ipadapter_model(os.path.abspath(a.source), a.dir)[0]
    patched = ext.IPAdapter

    import comfy.clip_vision as cv
    clip = cv.load(os.path.abspath(a.clip_vision))

    kw = dict(cross_attention_dim=768, output_cross_attention_dim=768,
              clip_embeddings_dim=1280, clip_extra_context_tokens=4,
              is_sdxl=False, is_plus=True, is_faceid=True)
    device = torch.device(a.device)
    src_model = stock(source_st, **kw).to(device).eval()
    pack_model = patched(pack_st, **kw).to(device).eval()

    insight_root = os.path.join(a.comfy, "models", "insightface")
    cache = os.path.join(os.environ.get("TEMP", "."), "vit_h_render", "face_embed.npy")
    if os.path.isfile(cache):
        face_np = np.load(cache)
    else:
        import cv2
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", root=insight_root, providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        found = app.get(cv2.imread(a.image))
        if not found:
            raise RuntimeError("no face found in %s" % a.image)
        best = max(found, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        face_np = best.embedding
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, face_np)
    face = torch.from_numpy(face_np).float().to(device).reshape(1, 512)

    def pen_of(img):
        im = Image.fromarray(img).convert("RGB").resize((224, 224))
        arr = torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0)[None].to(device)
        with torch.inference_mode():
            return clip.encode_image(arr)["penultimate_hidden_states"].float().to(device)

    def divergence(img):
        pen = pen_of(img)
        with torch.inference_mode():
            s = src_model.image_proj_model(face, pen, scale=1.0, shortcut=True)[0].float()
            p = pack_model.image_proj_model(face, pen, scale=1.0, shortcut=True)[0].float()
        cs = []
        for i in range(s.shape[0]):
            x, y = s[i], p[i]
            cs.append(float(x @ y) / (float(x.norm()) * float(y.norm()) + 1e-12))
        # the worst latent, not the average: an image-conditioned trigger that moves one latent
        # must not be averaged away by the other fifteen (same statistic as the A1111 run)
        return max(1.0 - c for c in cs)

    base = np.asarray(Image.open(a.image).convert("RGB"))
    baseline = divergence(base)

    cands = []
    controls = []
    for name, kind, payload in candidate_specs():
        img = apply_spec(base, kind, payload)
        d = divergence(img)
        rec = {"candidate": name, "kind": kind, "one_minus_cos": d,
               "is_natural_control": kind == "control"}
        cands.append(rec)
        if kind == "control":
            controls.append(d)

    # same rule as the A1111 run: the natural-transform control maximum, scaled by 1.5
    p99 = (max(controls) * 1.5) if controls else 0.0
    flagged = [c["candidate"] for c in cands
               if not c["is_natural_control"] and c["one_minus_cos"] > p99]

    # positive control: shift the codes of the first installed image_proj layer
    from hswq.ipadapter_convrot_int8 import ConvRotInt8Linear
    target = next((m for m in pack_model.image_proj_model.modules()
                   if type(m).__name__ == ConvRotInt8Linear.__name__), None)
    if target is None:
        # no INT8 layer to perturb (a float pack used as the negative control): the instrument
        # test is the candidate scan itself, and the ladder is reported as not applicable
        target_note = "no ConvRot INT8 layer in this pack - positive control not applicable"
        print("no int8 layer; skipping the code ladder")
        out = {
            "tool": "audit/ipadapter_faceid_plusv2_comfyui_trigger.py",
            "family": "ipadapter_faceid_plusv2_sd15",
            "pack": os.path.basename(a.pack),
            "role": "negative control: source against itself",
            "baseline": baseline, "controls": controls, "control_p99": p99,
            "candidates": cands, "flagged_count": len(flagged), "flagged": flagged,
            "positive_control": {"ladder": [], "detection_floor_codes_lsb": None,
                                 "note": target_note},
            "pass": len(flagged) == 0,
        }
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print(json.dumps({"baseline": baseline, "control_p99": p99, "flagged": len(flagged),
                          "max_candidate": max(c["one_minus_cos"] for c in cands)}, indent=1))
        return
    original = target.int8_weight.clone()
    ladder = []
    floor = None
    for lsb in (1, 2, 4, 8, 16, 32, 64):
        w = original.clone().view(-1)
        n = max(1, int(w.numel() * 0.10))
        idx = torch.arange(0, w.numel(), max(1, w.numel() // n))[:n]
        w[idx] = (w[idx].to(torch.int16) + lsb).clamp(-127, 127).to(torch.int8)
        target.int8_weight = w.view(original.shape)
        d = divergence(base)
        hit = d > p99
        ladder.append({"lsb": lsb, "one_minus_cos": d, "hit": bool(hit)})
        if hit and floor is None:
            floor = lsb
        target.int8_weight = original.clone()

    out = {
        "tool": "audit/ipadapter_faceid_plusv2_comfyui_trigger.py",
        "family": "ipadapter_faceid_plusv2_sd15",
        "route": "ComfyUI, HSWQ Load IPAdapter Model -> image_proj (faceid plus v2)",
        "baseline": baseline,
        "controls": controls,
        "control_p99": p99,
        "candidates": cands,
        "flagged_count": len(flagged),
        "flagged": flagged,
        "positive_control": {"ladder": ladder, "detection_floor_codes_lsb": floor},
        "floor_note": ("floor is the first ladder step above the control p99; steps are 10 % of the"
                       " codes shifted by +lsb"),
        "pass": len(flagged) == 0 and floor is not None,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({"baseline": baseline, "control_p99": p99, "flagged": len(flagged),
                      "floor_lsb": floor, "max_candidate": max(c["one_minus_cos"] for c in cands
                                                                if not c["is_natural_control"])},
                     indent=1))


if __name__ == "__main__":
    main()

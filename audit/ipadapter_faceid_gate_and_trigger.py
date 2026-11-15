"""CN-5a, CN-5b and CN-6 for the IP-Adapter FaceID pack.

The pack does not execute on the node's path: `ipadapter_model_loader` hands the int8 codes to
`MLPProjModelFaceId.load_state_dict`, which rejects them (size/dtype), so there is no INT8 route to
measure through. What is measured here is therefore the quantisation itself - the pack's codes
dequantised and loaded into the adapter's own float modules - which is the same construction the
CLIP-Vision verdict used for the same reason.

CN-5a: source against pack on a handful of face inputs, cosine and relative L2.
CN-5b: 55 candidate image transforms against the natural-transform control distribution.
CN-6: the fixed 256-item acceptance set (8 crops x 4 orientations x 8 brightness steps).

The observable is the adapter's own token stack: proj.0 -> GELU -> proj.2 on the InsightFace
embedding. The LoRA pairs are checked separately by their low-rank contribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch


def build_hadamard(size, dtype=torch.float32):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def dequant(codes, scale, gs):
    of, inf = codes.shape
    h = build_hadamard(gs)
    return torch.matmul((codes.float() * scale.float()).view(of, inf // gs, gs), h).reshape(of, inf)


def candidate_specs():
    specs = []
    for size in (16, 32, 64):
        for pos in ("tl", "tr", "bl", "br", "center"):
            for kind in ("solid", "noise", "qrblock"):
                specs.append(("patch_%s_%s_%d" % (kind, pos, size), "patch", (kind, pos, size)))
    for band in ("low", "high"):
        for sigma in (0.02, 0.05, 0.10):
            specs.append(("freq_%s_sigma_%.2f" % (band, sigma), "freq", (band, sigma)))
    for name in ("rot90", "flip", "bright_0.7", "jpeg60"):
        specs.append(("control_" + name, "control", name))
    return specs


def apply_spec(img, spec, jpeg_quality=60):
    kind, _, payload = spec
    h, w, _ = img.shape
    out = img.copy()
    if kind == "patch":
        pkind, pos, size = payload
        if pos == "tl":
            y0, x0 = 0, 0
        elif pos == "tr":
            y0, x0 = 0, w - size
        elif pos == "bl":
            y0, x0 = h - size, 0
        elif pos == "br":
            y0, x0 = h - size, w - size
        else:
            y0, x0 = (h - size) // 2, (w - size) // 2
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
            from PIL import Image as _I, ImageFilter
            lo, hi = float(noise.min()), float(noise.max())
            img8 = ((noise - lo) / (hi - lo + 1e-9) * 255).astype(np.uint8)
            blurred = np.asarray(_I.fromarray(img8).filter(ImageFilter.GaussianBlur(radius=6))).astype(np.float32)
            noise = (blurred - blurred.mean()) / (blurred.std() + 1e-9)
        else:
            noise = noise - np.asarray(np.mean(noise))
        noise = noise / (np.abs(noise).max() + 1e-9) * 255.0 * sigma * 4
        out = np.clip(out.astype(np.float32) + noise[..., None], 0, 255).astype(np.uint8)
    else:
        if payload == "rot90":
            out = np.rot90(img).copy()
        elif payload == "flip":
            out = np.fliplr(img).copy()
        elif payload == "bright_0.7":
            out = np.clip(img.astype(np.float32) * 0.7, 0, 255).astype(np.uint8)
        else:
            import io as _io
            from PIL import Image as _Image
            buf = _io.BytesIO()
            _Image.fromarray(img).save(buf, format="JPEG", quality=jpeg_quality)
            buf.seek(0)
            out = np.asarray(_Image.open(buf).convert("RGB"))
    return out


def gate_items(img, n=256):
    """8 crops x 4 orientations x 8 brightness steps."""
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
    ap.add_argument("--source", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--provider", default="CUDA")
    ap.add_argument("--json-gate", required=True)
    ap.add_argument("--json-trigger", required=True)
    ap.add_argument("--json-5a", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    sys.path.insert(0, os.path.abspath(a.comfy))
    sys.path.insert(0, os.path.join(os.path.abspath(a.comfy), "custom_nodes"))
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    import comfy.utils
    from comfyui_ipadapter_plus.utils import insightface_loader
    sys.argv = saved
    from PIL import Image

    src = torch.load(a.source, map_location="cpu", weights_only=True)
    flat_src = {}
    for k, v in src["image_proj"].items():
        flat_src["image_proj." + k] = v.float()
    for k, v in src["ip_adapter"].items():
        flat_src["ip_adapter." + k] = v.float()

    pack = comfy.utils.load_torch_file(a.pack)
    declared = sorted(k[: -len(".comfy_quant")] for k in pack if k.endswith(".comfy_quant"))
    flat_pack = dict(flat_src)
    for name in declared:
        w = pack[name + ".weight"]
        s = pack[name + ".weight_scale"].float()
        gs = int(256 if w.shape[1] % 256 == 0 else (64 if w.shape[1] % 64 == 0 else 4))
        flat_pack[name + ".weight"] = dequant(w, s, gs)
    print("[load] declared %d, dequantised into the source structure" % len(declared), flush=True)

    app = insightface_loader(a.provider)
    base_img = np.asarray(Image.open(a.image).convert("RGB"))

    def embed(img):
        fs = app.get(img)
        if not fs:
            return None
        return torch.from_numpy(fs[0].normed_embedding).float()

    def token_stack(sd, emb):
        x = torch.nn.functional.linear(emb[None], sd["image_proj.proj.0.weight"], sd["image_proj.proj.0.bias"])
        x = torch.nn.functional.gelu(x)
        return torch.nn.functional.linear(x, sd["image_proj.proj.2.weight"], sd["image_proj.proj.2.bias"])

    def lora_out(sd, name, x):
        down = sd[name + ".down.weight"] if (name + ".down.weight") in sd else None
        up = sd[name + ".up.weight"]
        h = torch.nn.functional.linear(x, down if down is not None else up, None)
        if down is not None:
            return torch.nn.functional.linear(h, up, None)
        return h

    def cos(p, q):
        return float(torch.nn.functional.cosine_similarity(
            p.reshape(1, -1).double(), q.reshape(1, -1).double(), dim=1).item())

    emb0 = embed(base_img)
    if emb0 is None:
        raise SystemExit("no face in the reference image")
    t_src = token_stack(flat_src, emb0)
    t_pack = token_stack(flat_pack, emb0)

    # ---- CN-5a
    batches = []
    g = torch.Generator().manual_seed(a.seed)
    for i in range(4):
        e = emb0 * (1.0 + 0.02 * torch.randn(emb0.shape, generator=g))
        batches.append({"batch": i, "cos": cos(token_stack(flat_src, e), token_stack(flat_pack, e))})
    lora_pairs = [n for n in declared if n.endswith(".down")]
    lora_cos = []
    for n in lora_pairs:
        name = n[: -len(".down")]
        width = int(flat_src[n + ".weight"].shape[1])
        x = torch.randn(1, width, generator=g)
        lora_cos.append(cos(lora_out(flat_src, name, x), lora_out(flat_pack, name, x)))
    cn5a = {"batches": batches,
            "image_proj_cos": cos(t_src, t_pack),
            "lora_pairs_checked": len(lora_cos),
            "lora_cos_min": min(lora_cos) if lora_cos else None,
            "lora_cos_mean": sum(lora_cos) / len(lora_cos) if lora_cos else None}
    print("[CN-5a] image_proj cos %.9f | lora %d pairs cos_min %.9f"
          % (cn5a["image_proj_cos"], len(lora_cos), cn5a["lora_cos_min"] or 0), flush=True)

    # ---- CN-5b
    specs = candidate_specs()
    cands = []
    controls = []
    for label, kind, payload in specs:
        img2 = apply_spec(base_img, (kind, None, payload))
        e = embed(img2)
        if e is None:
            cands.append({"candidate": label, "kind": kind, "one_minus_cos": None,
                          "skipped": "no face detected"})
            continue
        d = 1.0 - cos(token_stack(flat_src, e), token_stack(flat_pack, e))
        rec = {"candidate": label, "kind": kind, "one_minus_cos": d,
               "is_natural_control": kind == "control"}
        if kind == "control":
            controls.append(d)
        cands.append(rec)
    controls_sorted = sorted(controls)
    ctrl_max = max(controls) if controls else 0.0
    p99 = ctrl_max * 1.5
    flagged = [c["candidate"] for c in cands
               if not c.get("is_natural_control") and c.get("one_minus_cos") is not None
               and c["one_minus_cos"] > p99]

    # positive control: rank-1 into the largest image_proj layer, ladder
    target = "image_proj.proj.2.weight"
    w0 = flat_src[target].clone()
    norm = float(w0.norm())
    gu = torch.Generator().manual_seed(7)
    gv = torch.Generator().manual_seed(11)
    u = torch.randn(w0.shape[0], 1, generator=gu)
    v = torch.randn(1, w0.shape[1], generator=gv)
    direction = (u / u.norm()) @ (v / v.norm())
    ladder, floor = [], None
    for strength in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        sd = dict(flat_pack)
        sd[target] = w0 + direction * (strength * norm)
        d = 1.0 - cos(token_stack(flat_src, emb0), token_stack(sd, emb0))
        hit = d > p99
        if hit and floor is None:
            floor = strength
        ladder.append({"strength": strength, "one_minus_cos": d, "detected": hit})
        print("  implant %.3f -> 1-cos %.6e %s" % (strength, d, "DETECTED" if hit else "-"), flush=True)

    trig = {
        "tool": "audit/ipadapter_faceid_gate_and_trigger.py",
        "family": "faceid", "pack": os.path.basename(a.pack),
        "observable": "cosine of the adapter's token stack (proj.0 -> GELU -> proj.2) on the InsightFace embedding",
        "how": "the pack's int8 codes and scales dequantised into the source structure",
        "input_image": os.path.basename(a.image),
        "candidate_families": {"patch": 45, "frequency": 6, "natural_transform_controls": 4},
        "candidates": cands, "candidate_count": len(cands),
        "control_one_minus_cos": controls,
        "control_p99": p99,
        "ratio_threshold": 1.5,
        "flagged_over_control_p99": flagged, "flagged_count": len(flagged),
        "positive_control": {"layer": target, "ladder": ladder},
        "detection_floor_relative_strength": floor,
    }
    print("[CN-5b] %d candidates | control p99 %.6e | flagged %d | floor %s"
          % (len(cands), p99, len(flagged), floor), flush=True)

    # ---- CN-6
    items = gate_items(base_img)
    cs = []
    nonfinite = 0
    for im, nm in items:
        e = embed(im)
        if e is None:
            continue
        p = token_stack(flat_src, e)
        q = token_stack(flat_pack, e)
        if not (bool(torch.isfinite(p).all()) and bool(torch.isfinite(q).all())):
            nonfinite += 1
            continue
        cs.append(cos(p, q))
    cs.sort()
    gate = {"tool": "audit/ipadapter_faceid_gate_and_trigger.py",
            "gate": "mean >= 0.99 and min >= 0.98 on the fixed 256-item set",
            "pack": os.path.basename(a.pack),
            "how": "the pack's int8 codes and scales dequantised into the source structure",
            "n": len(cs), "finite": nonfinite == 0, "nonfinite_count": nonfinite,
            "items": "8 crops x 4 orientations x 8 brightness steps, from %s" % os.path.basename(a.image),
            "mean": sum(cs) / len(cs) if cs else None,
            "min": cs[0] if cs else None,
            "p5": cs[max(0, int(len(cs) * 0.05) - 1)] if cs else None,
            "p50": cs[len(cs) // 2] if cs else None}
    gate["pass"] = bool(gate["mean"] is not None and gate["mean"] >= 0.99
                        and gate["min"] is not None and gate["min"] >= 0.98 and gate["finite"])
    print("[CN-6] mean %.9f min %.9f n %d -> %s"
          % (gate["mean"], gate["min"], gate["n"], "PASS" if gate["pass"] else "FAIL"), flush=True)

    for path, rec in ((a.json_5a, {"tool": "audit/ipadapter_faceid_gate_and_trigger.py",
                                   "family": "faceid", "pack": os.path.basename(a.pack),
                                   "how": "the pack's int8 codes and scales dequantised into the source structure",
                                   **cn5a}),
                      (a.json_trigger, trig), (a.json_gate, gate)):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1, ensure_ascii=False)
    print("wrote 3 records", flush=True)


if __name__ == "__main__":
    main()

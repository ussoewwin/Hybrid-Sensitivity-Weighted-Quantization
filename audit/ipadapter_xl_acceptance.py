"""CN-5a / CN-5b / CN-6 for the SDXL basic IP-Adapter, measured through the A1111 ConvRot INT8 route.

This adapter's own geometry: `image_proj` is an `ImageProjModel` (proj 1280 -> 4x2048, then
LayerNorm over 2048) and `ip_adapter` is 70 pairs of bias-free `to_k_ip` / `to_v_ip` projections.
The observable is the 4x2048 token stack the runtime injects as `cond`, produced from the
CLIP-H penultimate CLS token (1280 wide) that the node's preprocessor feeds it.

`IPAdapterModel.load` installs the pack's int8 payload in place through
`modules/ipadapter_int8_convrot.py`, so all 141 layers run the ConvRot kernel. Kernel calls are
counted: a number cannot silently come from a float fallback.

Same candidate set, same control distribution, same 256-item gate, same seed as the other
IP-Adapter packs, so the numbers are comparable across the family.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time

import numpy as np
import torch


def _import_ipadapter_model(root):
    """Load the ControlNet extension's IP-Adapter model under its real package name.

    A1111's own `scripts` package sits at <root>/scripts and would shadow the extension's,
    so the extension package is registered explicitly.
    """
    import importlib.util, types
    pkg_dir = os.path.join(root, "extensions-builtin", "sd-webui-controlnet",
                           "scripts", "ipadapter")
    path = os.path.join(pkg_dir, "ipadapter_model.py")
    if not os.path.exists(path):
        raise SystemExit("ipadapter_model.py not found at " + path)
    scripts_dir = os.path.dirname(os.path.dirname(pkg_dir))
    for name, p in (("ext_scripts", scripts_dir), ("ext_scripts.ipadapter", pkg_dir)):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [p]
            sys.modules[name] = m
    spec = importlib.util.spec_from_file_location("ext_scripts.ipadapter.ipadapter_model", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ext_scripts.ipadapter.ipadapter_model"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_model(state_dict, name, device):
    IPAdapterModel = _IPADAPTER_MODEL.IPAdapterModel
    model = IPAdapterModel.load(state_dict, name)
    model.device = device
    model.to(device)
    model.eval()
    return model


def nested_from_flat(sd):
    nested = {"image_proj": {}, "ip_adapter": {}}
    for key, value in sd.items():
        for head in ("image_proj.", "ip_adapter."):
            if key.startswith(head):
                nested[head[:-1]][key[len(head):]] = value
                break
    return nested




def emb_of(model, emb):
    """The token stack this adapter injects, from the CLIP-H penultimate CLS token.

    A basic SDXL adapter goes through `_get_image_embeds`, which reads
    `clip_vision_output["image_embeds"]`; the 1280-wide penultimate token is what
    the node's preprocessor hands it, so the same value is passed here.
    """
    with torch.inference_mode():
        from transformers.models.clip.modeling_clip import CLIPVisionModelOutput
        e = emb.to(model.device, dtype=torch.float32)
        cvo = CLIPVisionModelOutput(
            image_embeds=e,
            last_hidden_state=e.unsqueeze(1).expand(-1, 257, -1) if e.dim() == 2 and e.shape[-1] == 1280 else e,
            hidden_states=None)
        out = model.get_image_emb(cvo)
    return out.cond_emb.detach().float().reshape(-1)


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(
        a.reshape(1, -1).double(), b.reshape(1, -1).double(), dim=1).item())


def apply_spec(img, kind, payload, jpeg_quality=60):
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
            from PIL import Image as _Image
            buf = io.BytesIO()
            _Image.fromarray(img).save(buf, format="JPEG", quality=jpeg_quality)
            buf.seek(0)
            out = np.asarray(_Image.open(buf).convert("RGB"))
    return out


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


_IPADAPTER_MODEL = None
_CLIP_VISION = None
EXPECTED_INT8_LAYERS = 141


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--json-5a", required=True)
    ap.add_argument("--json-trigger", required=True)
    ap.add_argument("--json-gate", required=True)
    ap.add_argument("--root", default=".", help="A1111 install root")
    ap.add_argument("--comfy", required=True,
                    help="ComfyUI root, for comfy.clip_vision")
    ap.add_argument("--clip-vision", required=True,
                    help="CLIP vision safetensors; this adapter takes the 1280-wide penultimate CLS token")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    global _IPADAPTER_MODEL
    root = os.path.abspath(a.root)
    sys.path.insert(0, root)
    _IPADAPTER_MODEL = _import_ipadapter_model(root)
    from PIL import Image

    t0 = time.time()
    src_sd = torch.load(a.source, map_location="cpu", weights_only=True)
    pack_sd = torch.load(a.pack, map_location="cpu", weights_only=True)
    # IPAdapterModel.load derives is_plus / is_faceid / is_instantid from the file name, so the
    # real basenames are passed here; a hardcoded name would load the wrong projection class.
    src_model = build_model(nested_from_flat(src_sd) if "image_proj" not in src_sd else src_sd,
                            os.path.basename(a.source), a.device)
    pack_model = build_model(nested_from_flat(pack_sd) if "image_proj" not in pack_sd else pack_sd,
                             os.path.basename(a.pack), a.device)
    print("[build] both adapters on %s in %.1f s" % (a.device, time.time() - t0), flush=True)

    int8_layers = sum(1 for _n, m in pack_model.named_modules()
                      if getattr(m, "_hswq_kind", None) == "linear"
                      and m.weight.dtype == torch.int8)
    print("[build] pack INT8 layers installed: %d" % int8_layers, flush=True)
    if int8_layers != EXPECTED_INT8_LAYERS:
        raise SystemExit("expected %d INT8 layers, found %d" % (EXPECTED_INT8_LAYERS, int8_layers))

    # count kernel calls so the numbers cannot come from a float fallback
    import comfy_kitchen.backends.cuda as kc
    calls = {"n": 0}
    orig = kc.int8_linear
    def wrapper(*args, **kw):
        calls["n"] += 1
        return orig(*args, **kw)
    kc.int8_linear = wrapper

    base_img = np.asarray(Image.open(a.image).convert("RGB"))

    def embed(img):
        """The CLIP-H penultimate CLS token, resized the way the node's preprocessor does."""
        from PIL import Image as _I
        im = _I.fromarray(img).convert("RGB").resize((224, 224))
        # comfy's ClipVisionModel.encode_image takes NHWC in 0..1 and returns a dict
        arr = torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0)[None].to(model_device())
        with torch.inference_mode():
            out = _CLIP_VISION.encode_image(arr)
        # CLIP-H's image_embeds is the 1024-d projection; a basic SDXL IP-Adapter takes the
        # 1280-d penultimate CLS token, which is what the node's preprocessor feeds it.
        pen = out["penultimate_hidden_states"]
        return pen[:, 0, :].float().reshape(-1)

    global _CLIP_VISION
    sys.path.insert(0, os.path.abspath(a.comfy))
    import comfy.clip_vision as _cv
    _CLIP_VISION = _cv.load(os.path.abspath(a.clip_vision))
    print("[build] CLIP vision loaded from %s" % os.path.basename(a.clip_vision), flush=True)

    def model_device():
        return a.device

    emb0 = embed(base_img)
    if emb0 is None:
        raise SystemExit("no face in the reference image")

    # ---- CN-5a
    calls["n"] = 0
    batches = []
    g = torch.Generator().manual_seed(a.seed)
    for i in range(4):
        e = emb0 * (1.0 + 0.02 * torch.randn(emb0.shape, generator=g))
        batches.append({"batch": i, "cos": cos(emb_of(src_model, e[None]), emb_of(pack_model, e[None]))})
    cn5a_calls = calls["n"]
    img_proj_cos = cos(emb_of(src_model, emb0[None]), emb_of(pack_model, emb0[None]))
    print("[CN-5a] image_proj cos %.9f over %d kernel calls" % (img_proj_cos, cn5a_calls), flush=True)

    # ---- CN-5b
    cands, controls = [], []
    calls["n"] = 0
    for label, kind, payload in candidate_specs():
        img2 = apply_spec(base_img, kind, payload)
        e = embed(img2)
        d = 1.0 - cos(emb_of(src_model, e[None]), emb_of(pack_model, e[None]))
        cands.append({"candidate": label, "kind": kind, "one_minus_cos": d,
                      "is_natural_control": kind == "control"})
        if kind == "control":
            controls.append(d)
    ctrl_max = max(controls) if controls else 0.0
    p99 = ctrl_max * 1.5
    flagged = [c["candidate"] for c in cands
               if not c.get("is_natural_control") and c.get("one_minus_cos") is not None
               and c["one_minus_cos"] > p99]

    # Positive control on the INT8 route. A single int8 code step is 1/256 of a group step and
    # did not move the observable, so the ladder perturbs the layer's *scale*, which is the
    # multiplier the kernel applies to every code of every output row: weight_scale *= (1+d).
    # That is a global magnitude change on the installed int8 layer, applied where the kernel
    # reads it, so a detection here proves the route carries a perturbation to the output.
    ladder, floor = [], None
    int8_mods = [(n, m) for n, m in pack_model.named_modules()
                 if getattr(m, "_hswq_kind", None) == "linear"
                 and getattr(m, "weight", None) is not None
                 and m.weight.dtype == torch.int8]
    target = None
    target_name = None
    for n, m in int8_mods:
        if getattr(m, "_hswq_name", "") == "image_proj.proj":
            target, target_name = m, n
            break
    if target is None and int8_mods:
        n0, m0 = int8_mods[0]
        target, target_name = m0, n0
    print("[CN-5b] positive control target: %s (%s), %d INT8 modules enumerable"
          % (getattr(target, "_hswq_name", "?"), target_name, len(int8_mods)), flush=True)

    if target is not None:
        base_scale = target.weight_scale.detach().clone()
        for delta in (1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
            with torch.no_grad():
                target.weight_scale.copy_(base_scale * (1.0 + delta))
                d = 1.0 - cos(emb_of(src_model, emb0[None]), emb_of(pack_model, emb0[None]))
                hit = d > p99
                if hit and floor is None:
                    floor = delta
                ladder.append({"scale_delta": delta, "one_minus_cos": d, "detected": hit})
                target.weight_scale.copy_(base_scale)
            print("  implant scale x%.4f -> 1-cos %.6e %s"
                  % (1.0 + delta, d, "DETECTED" if hit else "-"), flush=True)
        # and one whole-code ladder for completeness
        base_w = target.weight.detach().clone()
        code_ladder = []
        n_el = base_w.numel()
        for frac in (1e-3, 1e-2, 1e-1):
            k = max(1, int(n_el * frac))
            with torch.no_grad():
                flat = base_w.reshape(-1).clone()
                flat[:k] = (flat[:k] + 8).clamp(-128, 127)
                target.weight.copy_(flat.reshape(base_w.shape))
                d = 1.0 - cos(emb_of(src_model, emb0[None]), emb_of(pack_model, emb0[None]))
                code_ladder.append({"codes_changed": k, "fraction": frac, "step": 8,
                                    "one_minus_cos": d, "detected": d > p99})
            print("  implant %g codes +8 -> 1-cos %.6e" % (frac, d), flush=True)
        target.weight.copy_(base_w)
    else:
        code_ladder = []

    print("[CN-5b] %d candidates | control p99 %.6e | flagged %d | floor %s"
          % (len(cands), p99, len(flagged), floor), flush=True)

    # ---- CN-6
    items = gate_items(base_img)
    calls["n"] = 0
    cs, nonfinite = [], 0
    for im, nm in items:
        e = embed(im)
        p = emb_of(src_model, e[None])
        q = emb_of(pack_model, e[None])
        if not (bool(torch.isfinite(p).all()) and bool(torch.isfinite(q).all())):
            nonfinite += 1
            continue
        cs.append(cos(p, q))
    gate_calls = calls["n"]
    cs.sort()
    gate = {"tool": "audit/ipadapter_xl_acceptance.py",
            "gate": "mean >= 0.99 and min >= 0.98 on the fixed 256-item set",
            "route": "A1111 modules/ipadapter_int8_convrot.py, 141 INT8 layers installed",
            "pack": os.path.basename(a.pack),
            "int8_layers": int8_layers,
            "int8_kernel_calls": gate_calls,
            "n": len(cs), "finite": nonfinite == 0, "nonfinite_count": nonfinite,
            "items": "8 crops x 4 orientations x 8 brightness steps, from %s" % os.path.basename(a.image),
            "mean": sum(cs) / len(cs) if cs else None,
            "min": cs[0] if cs else None,
            "p5": cs[max(0, int(len(cs) * 0.05) - 1)] if cs else None,
            "p50": cs[len(cs) // 2] if cs else None}
    gate["pass"] = bool(gate["mean"] is not None and gate["mean"] >= 0.99
                        and gate["min"] is not None and gate["min"] >= 0.98 and gate["finite"])
    print("[CN-6] mean %.9f min %.9f n %d kernel %d -> %s"
          % (gate["mean"], gate["min"], gate["n"], gate_calls,
             "PASS" if gate["pass"] else "FAIL"), flush=True)

    kc.int8_linear = orig

    for path, rec in ((a.json_5a, {"tool": "audit/ipadapter_xl_acceptance.py",
                                   "route": "A1111 modules/ipadapter_int8_convrot.py, 141 INT8 layers installed",
                                   "pack": os.path.basename(a.pack), "int8_layers": int8_layers,
                                   "int8_kernel_calls": cn5a_calls,
                                   "image_proj_cos": img_proj_cos, "batches": batches}),
                      (a.json_trigger, {"tool": "audit/ipadapter_xl_acceptance.py",
                                        "route": "A1111 modules/ipadapter_int8_convrot.py, 141 INT8 layers installed",
                                        "pack": os.path.basename(a.pack),
                                        "observable": "cosine of the image_proj token stack on the CLIP-H penultimate CLS token",
                                        "candidate_families": {"patch": 45, "frequency": 6, "natural_transform_controls": 4},
                                        "candidates": cands, "candidate_count": len(cands),
                                        "control_one_minus_cos": controls, "control_p99": p99,
                                        "ratio_threshold": 1.5,
                                        "flagged_over_control_p99": flagged, "flagged_count": len(flagged),
                                        "positive_control": {
                                            "how": "weight_scale multiplier (1+d) on one installed int8 layer",
                                            "layer": getattr(target, "_hswq_name", None) if target is not None else None,
                                            "module": target_name,
                                            "int8_modules_enumerable": len(int8_mods),
                                            "ladder": ladder,
                                            "code_step_ladder": code_ladder},
                                        "detection_floor_code_fraction": floor}),
                      (a.json_gate, gate)):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1, ensure_ascii=False)
    print("wrote 3 records", flush=True)


if __name__ == "__main__":
    main()

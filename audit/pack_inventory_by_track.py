# -*- coding: utf-8 -*-
"""Classify every quantised safetensors under the model roots by track, then cross it with the catalogue.

The class is decided from the file's own header, never from its name (skill 1.10):
  * format      - the comfy_quant sidecars / metadata: int8 tensorwise+convrot, nvfp4, w4a8, other
  * keyspace    - which model the tensors belong to: UNet, IP-Adapter, ControlNet, CLIP vision,
                  text encoder, SAM, SeedVR2, Qwen-Image, Krea2, other
  * kept set    - 2-D weights left in F16 inside an INT8 pack = a kept/protection set was measured
  * track       - universal-engine UNet, per-model standard, other family, or a derived merge

Writes one JSON record and prints the table.
"""

import json
import os
import re
import struct

ROOTS = [r"D:\USERFILES\StableDiffusion\models",
         r"D:\USERFILES\ComfyUI\ComfyUI\models",
         r"D:\USERFILES\A1111\models"]
CAT = "md/HSWQ_High_Quality_Packs_ConvRot_INT8.md"
OUT = "md/backdoor/verdicts/records/pack_inventory_by_track.json"


def header(path):
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n <= 0 or n > 60_000_000:
                return None
            return json.loads(f.read(n))
    except Exception:
        return None


def norm(s):
    s = s.lower().replace(".safetensors", "")
    s = re.sub(r"_hswq.*$|_convrot.*$|_native.*$", "", s)
    return re.sub(r"[\s_+\-]+", "", s)


def classify(name, h):
    keys = [k for k in h if k != "__metadata__"]
    meta = h.get("__metadata__", {}) or {}
    quant = {}
    try:
        quant = json.loads(meta.get("_quantization_metadata", "{}"))
    except Exception:
        quant = {}
    layers = quant.get("layers", {})
    fmts = {v.get("format", "?") for v in layers.values()} if layers else set()
    groups = {v.get("convrot_groupsize") for v in layers.values() if v.get("convrot_groupsize")}
    nq = sum(1 for k in keys if k.endswith("comfy_quant"))
    scales = sum(1 for k in keys if k.endswith("weight_scale"))

    if any("nvfp4" in str(f) for f in fmts) or any("nvfp4" in k for k in keys[:50]):
        fmt = "nvfp4"
    elif any("w4a8" in k for k in keys[:50]) or "w4a8" in name.lower():
        fmt = "w4a8"
    elif nq and any(f == "int8_tensorwise" for f in fmts):
        fmt = "int8 convrot"
    elif nq:
        fmt = "int8 other"
    else:
        fmt = "specified elsewhere"

    ks = "\n".join(keys)
    if re.search(r"^(control_model|input_hint_block|zero_convs|controlnet_)", ks, re.M):
        space = "ControlNet"
    elif "image_proj" in ks and "ip_adapter" in ks:
        space = "IP-Adapter"
    elif re.search(r"^(visual\.|vision_model|class_embedding|transformer\.resblocks)", ks, re.M):
        space = "CLIP vision"
    elif "detector." in ks or ("backbone." in ks and "neck" in ks):
        space = "SAM"
    elif "model.diffusion_model." in ks:
        space = "UNet"
    elif re.search(r"^(model\.embed_tokens|model\.layers|lm_head)", ks, re.M):
        space = "text encoder"
    elif "transformer_blocks." in ks:
        space = "DiT (other family)"
    else:
        space = "other"

    # 2-D weights left in float inside an int8 pack = a kept set
    left = 0
    for k in keys:
        if k.endswith("comfy_quant") or k.endswith("weight_scale"):
            continue
        v = h.get(k, {})
        if v.get("dtype") == "F16" and len(v.get("shape", [])) == 2:
            left += 1

    # What md/HSWQ_High_Quality_Packs_ConvRot_INT8.md covers: the individually built per-model
    # standards (ControlNet and model patches, text encoders, SAM 3.1). UNet families are the
    # universal engine's product, the DiT families have their own tracks, and a merge is derived.
    if space in ("IP-Adapter", "ControlNet", "CLIP vision", "text encoder", "SAM"):
        track = "per-model standard (in this catalogue's scope)"
    elif space == "UNet":
        track = "universal-engine UNet (own track)"
    elif space == "DiT (other family)":
        track = "DiT family (own track)"
    else:
        track = "other family (own track)"

    merged = bool(re.search(r"\d\.\d.*\+|\+\d\.\d", name)) or name[:3] in ("0.5", "0.7", "0.9")
    if merged:
        track = "derived merge (" + track + ")"

    return {"format": fmt, "space": space, "track": track,
            "comfy_quant": nq, "weight_scale": scales, "float2d_left": left,
            "groupsizes": sorted(g for g in groups if g)}


def main():
    cat = open(CAT, encoding="utf-8").read()
    catn = norm(cat)
    cat_shas = set(m.group(1).upper() for m in re.finditer(r"`([0-9A-Fa-f]{12,64})`", cat))

    rows = []
    for root in ROOTS:
        for dp, dn, fn in os.walk(root):
            if dp.count(os.sep) > 7:
                dn[:] = []
            for f in fn:
                if not f.lower().endswith(".safetensors"):
                    continue
                p = os.path.join(dp, f)
                h = header(p)
                if not h:
                    continue
                keys = [k for k in h if k != "__metadata__"]
                if not any(k.endswith("comfy_quant") or k.endswith("weight_scale") for k in keys):
                    continue
                stem = f[:-len(".safetensors")]
                c = classify(stem, h)
                c.update({"stem": stem, "path": p, "bytes": os.path.getsize(p),
                          "in_catalogue_by_name": norm(stem) in catn})
                rows.append(c)

    by_track = {}
    for r in rows:
        by_track.setdefault(r["track"], []).append(r)

    print("quantised files: %d" % len(rows))
    for t in sorted(by_track):
        rs = by_track[t]
        missing = [r for r in rs if not r["in_catalogue_by_name"]]
        print("  %-34s %3d files, %3d without a catalogue row" % (t, len(rs), len(missing)))
    print("\npacks in this catalogue's scope without a row:")
    for r in rows:
        if r["track"].startswith("per-model standard") and not r["in_catalogue_by_name"]:
            print("   %-64s %-14s %-12s %d B" % (r["stem"][:64], r["space"], r["format"], r["bytes"]))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"tool": "audit/pack_inventory_by_track.py", "roots": ROOTS,
                   "files": len(rows),
                   "by_track": {t: {"files": len(v),
                                    "without_row": sum(1 for r in v if not r["in_catalogue_by_name"])}
                                for t, v in by_track.items()},
                   "rows": [{k: v for k, v in r.items() if k != "path"} for r in rows]},
                  f, indent=1, ensure_ascii=False)
    print("\nrecord:", OUT)


if __name__ == "__main__":
    main()

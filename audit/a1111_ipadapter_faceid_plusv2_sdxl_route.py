# -*- coding: utf-8 -*-
"""CN-2 on A1111 for the SD1.5 FaceID Plus v2 pack.

A1111 carries its own IP-Adapter runtime: ``extensions-builtin/sd-webui-controlnet/scripts/
ipadapter/ipadapter_model.py`` builds ``ProjModelFaceIdPlus`` when ``is_faceid and is_plus``, and
``modules/ipadapter_int8_convrot.py`` installs the ConvRot INT8 payload for that graph. This script
loads the pack through that runtime and measures whether the payload was applied:

* the declared sub-tensors in the file,
* how many Linear modules the graph builds and how many of them are INT8,
* the INT8 kernel calls on one projection pass, with the groupsize and dtypes the kernel saw,
* the injected stack and the to_kv outputs, finite.

It does not start A1111: the classes are imported from the install, the same way the PulID route
measurement does.

    python audit/a1111_ipadapter_faceid_plusv2_sdxl_route.py --a1111 <A1111 root> --pack <pack> \
        --json <record>.json [--device cuda]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys

import torch


def load_pack(pack_path):
    from safetensors.torch import load_file

    flat = load_file(pack_path)
    state = {"image_proj": {}, "ip_adapter": {}}
    for key, value in flat.items():
        if key.startswith("image_proj."):
            state["image_proj"][key[len("image_proj."):]] = value
        elif key.startswith("ip_adapter."):
            state["ip_adapter"][key[len("ip_adapter."):]] = value
    return state


def census(module):
    total = 0
    int8 = 0
    names = []
    for name, m in module.named_modules():
        if isinstance(m, torch.nn.Linear):
            total += 1
            w = getattr(m, "weight", None)
            if w is not None and w.dtype == torch.int8 and hasattr(m, "weight_scale"):
                int8 += 1
                names.append(name)
    return total, int8, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1111", required=True, help="the A1111 root; read only")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    root = os.path.abspath(a.a1111)
    cn_scripts = os.path.join(root, "extensions-builtin", "sd-webui-controlnet", "scripts")
    if root not in sys.path:
        sys.path.insert(0, root)
    if cn_scripts not in sys.path:
        sys.path.append(cn_scripts)

    from comfy_kitchen.backends import cuda as kc

    calls = []
    original = kc.int8_linear

    def counting(x, w, scale, bias=None, **kw):
        calls.append({
            "groupsize": kw.get("convrot_groupsize"),
            "convrot": kw.get("convrot"),
            "in_width": int(x.shape[-1]),
            "in_dtype": str(x.dtype),
            "out_dtype": str(kw.get("out_dtype")),
        })
        return original(x, w, scale, bias, **kw)

    kc.int8_linear = counting

    state = load_pack(a.pack)
    declared = {"image_proj": len(state["image_proj"]), "ip_adapter": len(state["ip_adapter"])}
    id_width = int(state["image_proj"]["proj.0.weight"].shape[1])

    mod = importlib.import_module("ipadapter.ipadapter_model")
    model = mod.IPAdapterModel(
        state,
        clip_embeddings_dim=1280,
        cross_attention_dim=2048,
        is_plus=True,
        is_sdxl=True,
        sdxl_plus=False,
        is_full=False,
        is_faceid=True,
        is_portrait=False,
        is_instantid=False,
        is_pulid=False,
        is_v2=True,
    )

    ip_total, ip_int8, ip_names = census(model.image_proj_model)
    kv_total, kv_int8, kv_names = census(model.ip_layers)

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    # the A1111 IP-Adapter runtime runs float32 activations; the ConvRot runtime casts them
    # to bfloat16 at the kernel boundary (modules/ipadapter_int8_convrot.py)
    face = torch.randn(1, id_width, generator=gen).to(dev)
    clip = torch.randn(1, 257, 1280, generator=gen).to(dev)

    # every installed layer is exercised once: the ones that consume the 4x2048 stack take it, the
    # 320/640-wide ones take an input of their own width, so the kernel count covers the whole pack
    calls.clear()
    widths = {}
    outs = []
    with torch.no_grad():
        emb = model.image_proj_model(face, clip, scale=1.0, shortcut=True)
        for name, layer in model.ip_layers.to_kvs.items():
            width = int(layer.in_features)
            if width not in widths:
                widths[width] = (emb if width == emb.shape[-1]
                                 else torch.randn(1, 4, width, generator=gen).to(dev))
            outs.append(layer(widths[width]).reshape(-1))
        kv = torch.cat(outs)
    finite = bool(torch.isfinite(emb).all().item() and torch.isfinite(kv).all().item())

    groupsizes = sorted({c["groupsize"] for c in calls})
    rec = {
        "tool": "audit/a1111_ipadapter_faceid_plusv2_sdxl_route.py",
        "family": "ipadapter_faceid_plusv2_sdxl",
        "route": "A1111 sd-webui-controlnet ipadapter + modules/ipadapter_int8_convrot.py",
        "pack": os.path.basename(a.pack),
        "pack_sha256": hashlib.sha256(open(a.pack, "rb").read()).hexdigest(),
        "pack_bytes": os.path.getsize(a.pack),
        "declared_sub_tensors": declared,
        "id_embeddings_dim": id_width,
        "clip_embeddings_dim": 1280,
        "cross_attention_dim": 2048,
        "image_proj": {"linear_modules": ip_total, "int8_modules": ip_int8,
                       "int8_names_first": ip_names[:3]},
        "ip_layers": {"linear_modules": kv_total, "int8_modules": kv_int8,
                      "int8_names_first": kv_names[:3]},
        "int8_kernel_calls": len(calls),
        "kernel_groupsizes": groupsizes,
        "kernel_input_widths": sorted({c["in_width"] for c in calls}),
        "kernel_convrot_all_true": all(c["convrot"] for c in calls) if calls else False,
        "projection": {"input": {"face": list(face.shape), "clip": list(clip.shape)},
                       "output": list(emb.shape),
                       "finite": finite},
        "applied": bool(ip_int8 + kv_int8 > 0 and len(calls) > 0),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: rec[k] for k in ("declared_sub_tensors", "image_proj", "ip_layers",
                                          "int8_kernel_calls", "kernel_groupsizes",
                                          "kernel_convrot_all_true", "projection", "applied")},
                     indent=1))


if __name__ == "__main__":
    main()

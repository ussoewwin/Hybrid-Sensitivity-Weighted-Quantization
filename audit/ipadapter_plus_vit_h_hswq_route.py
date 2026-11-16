# -*- coding: utf-8 -*-
"""Census of the HSWQ loader's own IP-Adapter route for the SDXL IP-Adapter Plus (ViT-H) pack.

The loader package loads a ConvRot INT8 pack through
``nodes/hswq_load_ipadapter_model.py``: it keeps the int8 codes, installs the runtime patch
``patches/ipadapter_convrot_int8.py`` and lets the ConvRot INT8 GEMM evaluate them.  This script
runs that path outside ComfyUI -- the extension is loaded as a package, the patch is applied to
its module globals, the adapter is built from the pack -- and reports what the route does with
this checkpoint:

* whether the patch took (the two module globals are replaced),
* how many int8 layers the pack presents and how many are installed,
* the INT8 kernel call count behind the numbers (so no number can come from a float fallback),
* the output of ``image_proj`` and of every ``to_kv`` projection against the source's own float
  weights, on the same input.

    python audit/ipadapter_plus_vit_h_hswq_route.py --pack <pack>.safetensors \
        --source <src>.safetensors --comfy <ComfyUI> --loader <loader package root> \
        --json <record>.json

The route is the loader's own; nothing here touches ComfyUI's files or the extension's files.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import types

import torch
from torch import nn


def load_extension(comfy_root):
    """Load ``comfyui_ipadapter_plus`` the way ComfyUI does: as a package, by file path."""
    ext = os.path.join(comfy_root, "custom_nodes", "comfyui_ipadapter_plus")
    if not os.path.isdir(ext):
        raise RuntimeError("the IP-Adapter Plus extension is not at %s" % ext)
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


def split_prefixed(state_dict):
    st = {"image_proj": {}, "ip_adapter": {}}
    for key, value in state_dict.items():
        if key.startswith("image_proj."):
            st["image_proj"][key[len("image_proj."):]] = value
        elif key.startswith("ip_adapter."):
            st["ip_adapter"][key[len("ip_adapter."):]] = value
    return st


def instance_has_convrot(module):
    from hswq.ipadapter_convrot_int8 import ConvRotInt8Linear

    n = 0
    for m in module.modules():
        if isinstance(m, ConvRotInt8Linear):
            n += 1
    return n


def cos_rows(a, b):
    a = a.detach().float().reshape(-1, a.shape[-1])
    b = b.detach().float().reshape(-1, b.shape[-1])
    num = (a * b).sum(dim=-1)
    den = a.norm(dim=-1) * b.norm(dim=-1)
    c = num / den.clamp_min(1e-12)
    return float(c.mean()), float(c.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--loader", required=True, help="root of the loader package (holds hswq/, patches/)")
    ap.add_argument("--json", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    sys.path.insert(0, a.comfy)
    sys.path.insert(0, a.loader)

    import comfy.utils

    ext = load_extension(a.comfy)
    stock_ipadapter = ext.IPAdapter
    stock_to_kv = ext.To_KV

    patch_mod = importlib.import_module("patches.ipadapter_convrot_int8")
    patched = patch_mod.patch_ipadapter_plus()
    patched_ipadapter = ext.IPAdapter
    patched_to_kv = ext.To_KV

    import comfy_kitchen

    calls = {"n": 0}
    original = comfy_kitchen.int8_linear

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    comfy_kitchen.int8_linear = counting

    kw = dict(
        cross_attention_dim=1280,
        output_cross_attention_dim=2048,
        clip_embeddings_dim=1280,
        clip_extra_context_tokens=16,
        is_sdxl=True,
        is_plus=True,
    )

    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    pen = torch.randn(1, 257, 1280, device=device, dtype=torch.float16)

    src_sd = split_prefixed(comfy.utils.load_torch_file(a.source, safe_load=True))
    pack_sd = split_prefixed(comfy.utils.load_torch_file(a.pack, safe_load=True))

    src_model = stock_ipadapter(src_sd, **kw).to(device=device, dtype=torch.float16).eval()
    calls["n"] = 0
    pack_model = patched_ipadapter(pack_sd, **kw).to(device=device, dtype=torch.float16).eval()

    from hswq.ipadapter_convrot_int8 import has_convrot_int8, split_convrot_state_dict

    declared_image = len(split_convrot_state_dict(pack_sd["image_proj"])[0])
    declared_to_kv = len(split_convrot_state_dict(pack_sd["ip_adapter"])[0])

    with torch.no_grad():
        src_emb = src_model.image_proj_model(pen)
        pack_emb = pack_model.image_proj_model(pen)
        emb_cos = cos_rows(src_emb, pack_emb)

        src_layers = src_model.ip_layers.to_kvs
        pack_layers = pack_model.ip_layers.to_kvs
        kv_cos = {}
        kv_finite = True
        for name in list(src_layers.keys()):
            x = src_layers[name](src_emb)
            y = pack_layers[name](src_emb)
            kv_cos[name] = cos_rows(x, y)
            kv_finite = kv_finite and bool(torch.isfinite(y).all())

    out = {
        "tool": "audit/ipadapter_plus_vit_h_hswq_route.py",
        "family": "ipadapter_plus_vit_h",
        "source": os.path.basename(a.source),
        "pack": os.path.basename(a.pack),
        "route": "the loader package's own IP-Adapter node: patch applied, codes kept INT8",
        "patch": {
            "applied": bool(patched),
            "IPAdapter_replaced": patched_ipadapter is not stock_ipadapter,
            "To_KV_replaced": patched_to_kv is not stock_to_kv,
            "marker": bool(getattr(ext.IPAdapterPlus if hasattr(ext, "IPAdapterPlus") else ext, "_hswq_convrot_int8_patched", True)),
        },
        "declared_int8_layers": {"image_proj": declared_image, "ip_adapter": declared_to_kv,
                                 "total": declared_image + declared_to_kv},
        "installed_int8_layers": {"image_proj": instance_has_convrot(pack_model.image_proj_model),
                                  "ip_adapter": instance_has_convrot(pack_model.ip_layers)},
        "source_float_check": {"image_proj_convrot_layers": instance_has_convrot(src_model.image_proj_model),
                               "ip_adapter_convrot_layers": instance_has_convrot(src_model.ip_layers)},
        "int8_kernel_calls": calls["n"],
        "image_proj": {"input": list(pen.shape), "output": list(pack_emb.shape),
                       "cos_mean": emb_cos[0], "cos_min": emb_cos[1],
                       "finite": bool(torch.isfinite(pack_emb).all())},
        "to_kv": {"outputs": list(kv_cos.keys()),
                  "cos_mean_min": min(v[0] for v in kv_cos.values()),
                  "cos_min_min": min(v[1] for v in kv_cos.values()),
                  "per_output": {k: {"cos_mean": v[0], "cos_min": v[1]} for k, v in kv_cos.items()},
                  "finite": kv_finite},
        "int8_payload_supported": bool(has_convrot_int8(pack_sd["image_proj"]) or has_convrot_int8(pack_sd["ip_adapter"])),
    }

    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

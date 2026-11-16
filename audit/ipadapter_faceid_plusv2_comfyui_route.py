# -*- coding: utf-8 -*-
"""CN-2 for the SD1.5 FaceID Plus v2 pack, inside ComfyUI on the loader package's node.

The user picks ``HSWQ Load IPAdapter Model``, points it at a folder, and the node returns the state
dict the IP-Adapter apply node consumes. The node installs the ConvRot INT8 patch on the extension's
``IPAdapter`` / ``To_KV`` globals and builds ``ConvRotInt8Linear`` for the quantised weights. This
script calls the node's own ``load_ipadapter_model`` (not a re-implementation), builds the adapter
with the FaceID-Plus argument set, and measures what the route did:

* that the node took the ConvRot branch (not the stock one) and how many layers it installed,
* that the source float pack installs none (the patch is conditional),
* the int8 kernel call count during a real forward, the injected-token cosine, and finiteness.

    python audit/ipadapter_faceid_plusv2_comfyui_route.py --pack <pack>.safetensors \
        --source <f.bin> --comfy <ComfyUI> --loader <loader repo> --dir <folder> --json <record>.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types

import torch


NODE_REL = os.path.join("nodes", "hswq_load_ipadapter_model.py")


def load_extension(comfy_root):
    ext = os.path.join(comfy_root, "custom_nodes", "comfyui_ipadapter_plus")
    sys.path.insert(0, comfy_root)
    sys.path.insert(0, ext)
    name = "comfyui_ipadapter_plus"
    pkg = types.ModuleType(name)
    pkg.__path__ = [ext]
    sys.modules[name] = pkg
    path = os.path.join(ext, "IPAdapterPlus.py")
    spec = importlib.util.spec_from_file_location(name + ".IPAdapterPlus", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name + ".IPAdapterPlus"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_node(loader_root):
    """Load the node inside a package tree so its relative imports resolve."""
    root = "hswq_loader"
    pkg = types.ModuleType(root)
    pkg.__path__ = [loader_root]
    sys.modules[root] = pkg
    nodes = types.ModuleType(root + ".nodes")
    nodes.__path__ = [os.path.join(loader_root, "nodes")]
    sys.modules[root + ".nodes"] = nodes
    path = os.path.join(loader_root, NODE_REL)
    spec = importlib.util.spec_from_file_location(root + ".nodes.hswq_load_ipadapter_model", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[root + ".nodes.hswq_load_ipadapter_model"] = mod
    spec.loader.exec_module(mod)
    return mod


def convrot_count(module, cls):
    """Count installed ConvRot layers by class name.

    The node may import ``hswq.ipadapter_convrot_int8`` through a different
    package object than this script does (the node is loaded inside a package
    tree), so ``isinstance`` can be false for the very layers the node built.
    The class name plus the module file is the identity that survives that.
    """
    return sum(1 for m in module.modules() if type(m).__name__ == cls.__name__)


def convrot_modules(module, cls):
    return sorted({type(m).__module__ for m in module.modules() if type(m).__name__ == cls.__name__})


def cos_stats(a, b):
    x = a.reshape(-1, a.shape[-1]).double()
    y = b.reshape(-1, b.shape[-1]).double()
    c = (x * y).sum(-1) / (x.norm(dim=-1) * y.norm(dim=-1)).clamp_min(1e-30)
    return [float(c.mean()), float(c.min())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--loader", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    sys.path.insert(0, a.comfy)
    sys.path.insert(0, a.loader)

    ext = load_extension(a.comfy)
    stock_ipadapter, stock_to_kv = ext.IPAdapter, ext.To_KV

    node = load_node(a.loader)
    convrot_module_loaded = node._ipa_convrot is not None

    obj = node.HSWQLoadIPAdapterModel()
    pack_st = obj.load_ipadapter_model(os.path.basename(a.pack), a.dir)[0]
    patched_ipadapter, patched_to_kv = ext.IPAdapter, ext.To_KV
    source_st = obj.load_ipadapter_model(os.path.basename(a.source), a.dir)[0]

    import comfy_kitchen

    calls = {"n": 0}
    original = comfy_kitchen.int8_linear

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    comfy_kitchen.int8_linear = counting

    kw = dict(cross_attention_dim=768, output_cross_attention_dim=768,
              clip_embeddings_dim=1280, clip_extra_context_tokens=4,
              is_sdxl=False, is_plus=True, is_faceid=True)

    device = torch.device(a.device)
    torch.manual_seed(a.seed)
    pen = torch.randn(1, 257, 1280, device=device, dtype=torch.float16)
    face = torch.randn(1, 512, device=device, dtype=torch.float16)

    src_model = stock_ipadapter(source_st, **kw).to(device=device, dtype=torch.float16).eval()
    pack_model = patched_ipadapter(pack_st, **kw).to(device=device, dtype=torch.float16).eval()

    from hswq.ipadapter_convrot_int8 import ConvRotInt8Linear

    with torch.no_grad():
        src_emb = src_model.image_proj_model(face, pen, scale=1.0, shortcut=True)
        pack_emb = pack_model.image_proj_model(face, pen, scale=1.0, shortcut=True)
        emb_cos = cos_stats(src_emb, pack_emb)
        kv = {}
        finite = True
        by_width = {}
        for name in list(src_model.ip_layers.to_kvs.keys()):
            width = int(src_model.ip_layers.to_kvs[name].in_features)
            if width not in by_width:
                torch.manual_seed(a.seed + width)
                by_width[width] = torch.randn(1, 4, width, device=device, dtype=torch.float16)
            x = by_width[width]
            y = pack_model.ip_layers.to_kvs[name](x)
            kv[name] = cos_stats(src_model.ip_layers.to_kvs[name](x), y)
            finite = finite and bool(torch.isfinite(y).all())
        del by_width

    record = {
        "tool": "audit/ipadapter_faceid_plusv2_comfyui_route.py",
        "family": "ipadapter_faceid_plusv2_sd15",
        "source": os.path.basename(a.source),
        "pack": os.path.basename(a.pack),
        "route": "ComfyUI, the loader package's HSWQ Load IPAdapter Model node",
        "node": {
            "module_convrot_branch_loaded": bool(convrot_module_loaded),
            "patch_applied": patched_ipadapter is not stock_ipadapter,
            "To_KV_replaced": patched_to_kv is not stock_to_kv,
            "faceidplusv2_flag": bool(pack_st.get("faceidplusv2")),
            "source_faceidplusv2_flag": bool(source_st.get("faceidplusv2")),
        },
        "installed_int8_layers": {
            "image_proj": convrot_count(pack_model.image_proj_model, ConvRotInt8Linear),
            "ip_adapter": convrot_count(pack_model.ip_layers, ConvRotInt8Linear),
        },
        "convrot_module_files": convrot_modules(pack_model.image_proj_model, ConvRotInt8Linear)
                                 + convrot_modules(pack_model.ip_layers, ConvRotInt8Linear),
        "source_float_check": {
            "image_proj_convrot_layers": convrot_count(src_model.image_proj_model, ConvRotInt8Linear),
            "ip_adapter_convrot_layers": convrot_count(src_model.ip_layers, ConvRotInt8Linear),
        },
        "int8_kernel_calls": calls["n"],
        "image_proj": {
            "input": {"face": list(face.shape), "clip": list(pen.shape)},
            "output": list(pack_emb.shape),
            "cos_mean": emb_cos[0],
            "cos_min": emb_cos[1],
            "finite": bool(torch.isfinite(pack_emb).all()),
        },
        "to_kv": {
            "projections": len(kv),
            "cos_mean_min": min(v[0] for v in kv.values()),
            "cos_min_min": min(v[1] for v in kv.values()),
            "finite": finite,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
    print(json.dumps(record, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

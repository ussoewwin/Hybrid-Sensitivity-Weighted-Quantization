# -*- coding: utf-8 -*-
"""ComfyUI route census for the SDXL IP-Adapter Plus (ViT-H) ConvRot INT8 pack, through the
loader package's node.

This is the route inside ComfyUI: the user picks ``HSWQ Load IPAdapter Model``, points it at a
folder, and the node returns the state dict the IP-Adapter apply node consumes. The node installs
the runtime patch when the file is a ConvRot INT8 pack and keeps the codes INT8.

The script calls the node's own ``load_ipadapter_model`` (not a re-implementation of it), builds
the adapter from what the node returned, and measures what the route does:

* that the node took the ConvRot branch (not the stock branch),
* how many int8 layers the adapter ends up with (image_proj / ip_adapter),
* the INT8 kernel call count behind the numbers,
* the outputs against the source's float weights on the same input.

    python audit/ipadapter_plus_vit_h_comfyui_route.py --pack <pack>.safetensors \
        --source <src>.safetensors --comfy <ComfyUI> --loader <loader package root> \
        --dir <folder holding the pack> --json <record>.json
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

NODE_REL = os.path.join("nodes", "hswq_load_ipadapter_model.py")


def load_extension(comfy_root):
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


def load_node(loader_root):
    """Load the node as part of a package tree, so its relative imports resolve."""
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


def convrot_count(module, cls):
    """Count ConvRot INT8 linears by class name and payload, so a second import of the module
    (the node imports it as ``hswq.ipadapter_convrot_int8``, the patch as part of the package)
    does not read as zero."""
    n = 0
    for m in module.modules():
        if type(m).__name__ == cls.__name__ and getattr(m, "int8_weight", None) is not None:
            n += 1
    return n


def cos_stats(a, b):
    a = a.detach().float().reshape(-1, a.shape[-1])
    b = b.detach().float().reshape(-1, b.shape[-1])
    c = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)
    return float(c.mean()), float(c.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--loader", required=True)
    ap.add_argument("--dir", required=True, help="folder the node is pointed at")
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

    kw = dict(cross_attention_dim=1280, output_cross_attention_dim=2048,
              clip_embeddings_dim=1280, clip_extra_context_tokens=16,
              is_sdxl=True, is_plus=True)

    device = torch.device(a.device)
    torch.manual_seed(a.seed)
    pen = torch.randn(1, 257, 1280, device=device, dtype=torch.float16)

    src_model = stock_ipadapter(source_st, **kw).to(device=device, dtype=torch.float16).eval()
    pack_model = patched_ipadapter(pack_st, **kw).to(device=device, dtype=torch.float16).eval()

    from hswq.ipadapter_convrot_int8 import ConvRotInt8Linear

    with torch.no_grad():
        src_emb = src_model.image_proj_model(pen)
        pack_emb = pack_model.image_proj_model(pen)
        emb_cos = cos_stats(src_emb, pack_emb)
        kv = {}
        kv_same_input = {}
        finite = True
        for name in list(src_model.ip_layers.to_kvs.keys()):
            y = pack_model.ip_layers.to_kvs[name](pack_emb)
            same = pack_model.ip_layers.to_kvs[name](src_emb)
            kv[name] = cos_stats(src_model.ip_layers.to_kvs[name](src_emb), same)
            kv_same_input[name] = kv[name]
            finite = finite and bool(torch.isfinite(y).all())

    record = {
        "tool": "audit/ipadapter_plus_vit_h_comfyui_route.py",
        "family": "ipadapter_plus_vit_h",
        "source": os.path.basename(a.source),
        "pack": os.path.basename(a.pack),
        "route": "ComfyUI, the loader package's HSWQ Load IPAdapter Model node",
        "node": {
            "module_convrot_branch_loaded": bool(convrot_module_loaded),
            "patch_applied": patched_ipadapter is not stock_ipadapter,
            "To_KV_replaced": patched_to_kv is not stock_to_kv,
        },
        "installed_int8_layers": {
            "image_proj": convrot_count(pack_model.image_proj_model, ConvRotInt8Linear),
            "ip_adapter": convrot_count(pack_model.ip_layers, ConvRotInt8Linear),
        },
        "source_float_check": {
            "image_proj_convrot_layers": convrot_count(src_model.image_proj_model, ConvRotInt8Linear),
            "ip_adapter_convrot_layers": convrot_count(src_model.ip_layers, ConvRotInt8Linear),
        },
        "int8_kernel_calls": calls["n"],
        "image_proj": {
            "input": list(pen.shape),
            "output": list(pack_emb.shape),
            "cos_mean": emb_cos[0],
            "cos_min": emb_cos[1],
            "finite": bool(torch.isfinite(pack_emb).all()),
        },
        "to_kv": {
            "projections": len(kv),
            "same_input": {"cos_mean_min": min(v[0] for v in kv_same_input.values()),
                           "cos_min_min": min(v[1] for v in kv_same_input.values())},
            "end_to_end": {"cos_mean_min": min(cos_stats(src_model.ip_layers.to_kvs[n](src_emb),
                                                         pack_model.ip_layers.to_kvs[n](pack_emb))[0]
                                               for n in kv),
                           "cos_min_min": min(cos_stats(src_model.ip_layers.to_kvs[n](src_emb),
                                                        pack_model.ip_layers.to_kvs[n](pack_emb))[1]
                                              for n in kv)},
            "finite": finite,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
    print(json.dumps(record, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

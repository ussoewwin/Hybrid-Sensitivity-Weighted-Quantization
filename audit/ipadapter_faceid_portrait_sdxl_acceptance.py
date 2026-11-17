# -*- coding: utf-8 -*-
"""CN-5a for the SD1.5 FaceID Plus v2 pack: the source graph against the pack's dequantised graph.

Two surfaces, measured the way the other IP-Adapter packs are:

* ``image_proj`` - the injected token stack itself (4 x 2048 here), which depends on the identity and
  on the CLIP penultimate.
* ``ip_adapter`` - the to_k_ip / to_v_ip projections and the low-rank pairs, each on a fixed input of
  its own width; these consume the UNet cross-attention K/V, so they do not depend on the image.

Four CLIP batches are measured, so the image_proj number is not one lucky input.

    python audit/ipadapter_faceid_portrait_sdxl_acceptance.py --pack <pack> --source <f.bin> \
        --comfy <ComfyUI> --json <record>.json [--batch-inputs 4]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import types

import torch


def build_hadamard(size, dtype=torch.float32):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def cos_pair(a, b):
    x = a.reshape(-1).double()
    y = b.reshape(-1).double()
    return float((x @ y) / (x.norm() * y.norm()).clamp_min(1e-30))


def cos_rows(a, b):
    x = a.reshape(-1, a.shape[-1]).double()
    y = b.reshape(-1, b.shape[-1]).double()
    c = (x * y).sum(-1) / (x.norm(dim=-1) * y.norm(dim=-1)).clamp_min(1e-30)
    return float(c.mean()), float(c.min())


def load_extension(comfy_root):
    ext = os.path.join(comfy_root, "custom_nodes", "comfyui_ipadapter_plus")
    sys.path.insert(0, ext)
    name = "comfyui_ipadapter_plus"
    pkg = types.ModuleType(name)
    pkg.__path__ = [ext]
    sys.modules[name] = pkg
    path = os.path.join(ext, "IPAdapterPlus.py")
    spec = importlib_spec(name + ".IPAdapterPlus", path)
    module = spec["module"]
    spec["loader"].exec_module(module)
    return module


def importlib_spec(fullname, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(fullname, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    return {"module": module, "loader": spec.loader}


def dequant_pack(pack):
    """The pack's weights back in float, using each layer's own sidecar."""
    out = {"image_proj": {}, "ip_adapter": {}}
    for group in ("image_proj", "ip_adapter"):
        prefix = group + "."
        names = {k[len(prefix):] for k in pack if k.startswith(prefix)}
        for name in names:
            key = prefix + name
            v = pack[key]
            if name.endswith(".comfy_quant") or name.endswith(".weight_scale"):
                continue
            if v.dtype == torch.int8:
                mk = key[: -len(".weight")]
                s = pack[mk + ".weight_scale"].float()
                conf = json.loads(bytes(int(x) for x in pack[mk + ".comfy_quant"]).decode("utf-8"))
                gs = int(conf["convrot_groupsize"])
                of, inf = v.shape
                deq = v.float().reshape(of, inf // gs, gs) * s.reshape(-1, 1, 1)
                out[group][name] = torch.matmul(deq, build_hadamard(gs)).reshape(of, inf)
            else:
                out[group][name] = v.float()
    return out


def measure_int8_route(a, ext, device, face, batches):
    """CN-5a through the real route: the loader node builds the pack, the kernel runs it.

    The number without the kernel behind it is a dequantised-float number; this block puts the
    kernel count beside the cosine, which is what section 2.5 asks for.
    """
    import importlib
    import sys
    import types

    root = "hswq_loader"
    pkg = types.ModuleType(root)
    pkg.__path__ = [a.loader]
    sys.modules[root] = pkg
    nodes = types.ModuleType(root + ".nodes")
    nodes.__path__ = [os.path.join(a.loader, "nodes")]
    sys.modules[root + ".nodes"] = nodes
    path = os.path.join(a.loader, "nodes", "hswq_load_ipadapter_model.py")
    import importlib.util

    spec = importlib.util.spec_from_file_location(root + ".nodes.hswq_load_ipadapter_model", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    import comfy_kitchen

    calls = {"n": 0}
    original = comfy_kitchen.int8_linear

    def counting(*args, **kw):
        calls["n"] += 1
        return original(*args, **kw)

    comfy_kitchen.int8_linear = counting

    obj = mod.HSWQLoadIPAdapterModel()
    pack_st = obj.load_ipadapter_model(os.path.basename(a.pack), a.dir)[0]
    src_st = obj.load_ipadapter_model(os.path.abspath(a.source), a.dir)[0]
    kw = dict(cross_attention_dim=1280, output_cross_attention_dim=2048, clip_embeddings_dim=512,
              clip_extra_context_tokens=16, is_sdxl=True, is_plus=False, is_faceid=True,
              is_portrait_unnorm=True)
    stock, patched = ext.IPAdapter, ext.IPAdapter
    src_model = ext.IPAdapter(src_st, **kw).to(device).eval()
    pack_model = patched(pack_st, **kw).to(device).eval()

    import torch as _t

    torch.manual_seed(42)
    rows = []
    calls["n"] = 0
    with _t.no_grad():
        for i in range(batches):
            torch.manual_seed(1042 + i)
            face_i = torch.randn(1, 1, 512, device=device) * 8.0
            s = src_model.image_proj_model(face_i)
            p = pack_model.image_proj_model(face_i)
            cs = []
            for j in range(s.shape[0]):
                x, y = s[j].reshape(-1).double(), p[j].reshape(-1).double()
                cs.append(float((x @ y) / (x.norm() * y.norm()).clamp_min(1e-30)))
            rows.append({"batch": i, "cos_mean": sum(cs) / len(cs), "cos_min": min(cs)})
    comfy_kitchen.int8_linear = original
    return {
        "route": "ComfyUI loader node, IPAdapter built from the pack, kernel counted",
        "int8_kernel_calls": calls["n"],
        "batches": rows,
        "cos_mean": sum(r["cos_mean"] for r in rows) / len(rows),
        "cos_min": min(r["cos_min"] for r in rows),
        "note": "the kernel count covers the image_proj graph only, which is the surface CN-5a compares; "
                "the ip_adapter surface is measured on dequantised weights above because the projections "
                "are applied per layer and their count is in the route record",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--batch-inputs", type=int, default=4)
    ap.add_argument("--loader", default="", help="the loader package root; when given, CN-5a is also "
                                                "measured through the INT8 route with the kernel counted")
    ap.add_argument("--dir", default="", help="the folder the loader node is pointed at")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    sys.path.insert(0, a.comfy)
    ext = load_extension(a.comfy)
    from safetensors.torch import load_file
    from image_proj_models import Resampler

    raw = torch.load(a.source, map_location="cpu", weights_only=True)
    src = {"image_proj": {k: v.float() for k, v in raw["image_proj"].items()},
           "ip_adapter": {k: v.float() for k, v in raw["ip_adapter"].items()}}
    pack = load_file(a.pack)
    pack = dequant_pack(pack)
    for group in ("image_proj", "ip_adapter"):
        pack[group] = {k: v.to(torch.float32).cpu() for k, v in pack[group].items()}

    device = torch.device(a.device)
    torch.manual_seed(a.seed)
    face = torch.randn(1, 1, 512, device=device) * 8.0

    def build(sd):
        m = Resampler(dim=1280, depth=4, dim_head=64, heads=20, num_queries=16,
                      embedding_dim=512, output_dim=2048, ff_mult=4).eval().to(device)
        m.load_state_dict(sd["image_proj"], strict=True)
        return m

    src_model, pack_model = build(src), build(pack)

    batches = []
    ips = []
    for i in range(a.batch_inputs):
        torch.manual_seed(a.seed + 1000 + i)
        face_i = torch.randn(1, 1, 512, device=device) * 8.0
        with torch.no_grad():
            s = src_model(face_i)
            p = pack_model(face_i)
        mean, mn = cos_rows(s, p)
        rel = float(((s - p) ** 2).mean() / (s ** 2).mean().clamp_min(1e-30))
        batches.append({"batch": i, "image_proj_cos_mean": mean, "image_proj_cos_min": mn,
                        "image_proj_rel_mse": rel})
        ips.append((mean, mn))

    kv = {}
    for name, w in sorted(src["ip_adapter"].items()):
        if not (name.endswith(".weight") and w.dim() == 2):
            continue
        width = int(w.shape[1])
        torch.manual_seed(a.seed + width)
        x = torch.randn(1, 16, width, device=device)
        pw = pack["ip_adapter"][name].to(device)
        kv[name] = cos_pair(x @ w.to(device).T, x @ pw.T)

    values = list(kv.values())
    out = {
        "tool": "audit/ipadapter_faceid_portrait_sdxl_acceptance.py",
        "family": "ipadapter_faceid_portrait_sdxl",
        "route": "the source checkpoint against the pack dequantised from its own sidecars",
        "pack": os.path.basename(a.pack),
        "source": os.path.basename(a.source),
        "image_proj_surface": {
            "elements": 16,
            "element": "one token of the 16x2048 injected stack",
            "cos_mean": sum(b["image_proj_cos_mean"] for b in batches) / len(batches),
            "cos_min": min(b["image_proj_cos_min"] for b in batches),
            "rel_mse_max": max(b["image_proj_rel_mse"] for b in batches),
            "depends_on_the_image": True,
            "batches": batches,
        },
        "ip_adapter_surface": {
            "elements": len(kv),
            "element": "one 2-D weight applied to a fixed input of its own width",
            "cos_mean": sum(values) / len(values),
            "cos_min": min(values),
            "max_abs_diff_from_one": max(1.0 - v for v in values),
            "depends_on_the_image": False,
            "note": "these weights consume the UNet cross-attention K/V and the identity/clip stack, "
                    "not the pixel image, so this surface is measured on fixed inputs",
        },
    }
    if a.loader and a.dir:
        out["int8_route"] = measure_int8_route(a, ext, device, face, a.batch_inputs)

    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({"image_proj": {k: out["image_proj_surface"][k]
                                     for k in ("cos_mean", "cos_min", "rel_mse_max")},
                      "ip_adapter": {k: out["ip_adapter_surface"][k]
                                     for k in ("elements", "cos_mean", "cos_min")}}, indent=1))


if __name__ == "__main__":
    main()

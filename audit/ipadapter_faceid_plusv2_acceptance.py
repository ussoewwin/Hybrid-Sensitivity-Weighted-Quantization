# -*- coding: utf-8 -*-
"""CN-5a for the SD1.5 FaceID Plus v2 pack: the source graph against the pack's dequantised graph.

Two surfaces, measured the way the other IP-Adapter packs are:

* ``image_proj`` - the injected token stack itself (4 x 768 here), which depends on the identity and
  on the CLIP penultimate.
* ``ip_adapter`` - the to_k_ip / to_v_ip projections and the low-rank pairs, each on a fixed input of
  its own width; these consume the UNet cross-attention K/V, so they do not depend on the image.

Four CLIP batches are measured, so the image_proj number is not one lucky input.

    python audit/ipadapter_faceid_plusv2_acceptance.py --pack <pack> --source <f.bin> \
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--batch-inputs", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    sys.path.insert(0, a.comfy)
    ext = load_extension(a.comfy)
    from safetensors.torch import load_file
    from image_proj_models import ProjModelFaceIdPlus

    raw = torch.load(a.source, map_location="cpu", weights_only=True)
    src = {"image_proj": {k: v.float() for k, v in raw["image_proj"].items()},
           "ip_adapter": {k: v.float() for k, v in raw["ip_adapter"].items()}}
    pack = load_file(a.pack)
    pack = dequant_pack(pack)
    for group in ("image_proj", "ip_adapter"):
        pack[group] = {k: v.to(torch.float32).cpu() for k, v in pack[group].items()}

    device = torch.device(a.device)
    torch.manual_seed(a.seed)
    face = torch.randn(1, 512, device=device)

    def build(sd):
        m = ProjModelFaceIdPlus(cross_attention_dim=768, id_embeddings_dim=512,
                                clip_embeddings_dim=1280, num_tokens=4).eval().to(device)
        m.load_state_dict(sd["image_proj"], strict=True)
        return m

    src_model, pack_model = build(src), build(pack)

    batches = []
    ips = []
    for i in range(a.batch_inputs):
        torch.manual_seed(a.seed + 1000 + i)
        clip = torch.randn(1, 257, 1280, device=device)
        with torch.no_grad():
            s = src_model(face, clip, scale=1.0, shortcut=True)
            p = pack_model(face, clip, scale=1.0, shortcut=True)
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
        x = torch.randn(1, 4, width, device=device)
        pw = pack["ip_adapter"][name].to(device)
        kv[name] = cos_pair(x @ w.to(device).T, x @ pw.T)

    values = list(kv.values())
    out = {
        "tool": "audit/ipadapter_faceid_plusv2_acceptance.py",
        "family": "ipadapter_faceid_plusv2_sd15",
        "route": "the source checkpoint against the pack dequantised from its own sidecars",
        "pack": os.path.basename(a.pack),
        "source": os.path.basename(a.source),
        "image_proj_surface": {
            "elements": 4,
            "element": "one token of the 4x768 injected stack",
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
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({"image_proj": {k: out["image_proj_surface"][k]
                                     for k in ("cos_mean", "cos_min", "rel_mse_max")},
                      "ip_adapter": {k: out["ip_adapter_surface"][k]
                                     for k in ("elements", "cos_mean", "cos_min")}}, indent=1))


if __name__ == "__main__":
    main()

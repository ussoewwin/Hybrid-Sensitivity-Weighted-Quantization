"""The PuLID pack on the A1111 route.

ComfyUI builds the adapter through ``comfyui_ipadapter_plus``. A1111 builds it through
``extensions-builtin/sd-webui-controlnet/scripts/ipadapter/ipadapter_model.py``, which
calls plain ``load_state_dict`` on the ``image_proj`` and ``ip_adapter`` sub dicts. A
ConvRot INT8 pack cannot go through that call, and the A1111 tree carries the runtime
hook that makes it loadable: ``modules/ipadapter_int8_convrot.py``.

This tool exercises that hook the way the A1111 code does - build ``IPAdapterModel``
with the same arguments the ControlNet extension passes, then run the adapter's own
forward - and reports what the route actually did:

  - how many Linears the built graph has, and how many of them received the int8
    payload (``weight.dtype == torch.int8`` with a ``weight_scale``),
  - how many ConvRot int8 GEMMs the two forwards issued, counted at the
    ``comfy_kitchen`` boundary, and which groupsize each call carried,
  - whether every output tensor was finite, and the shapes.

Owner's tree is read-only from here: nothing under ``--a1111`` is written.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import types


def load_pack(pack_path):
    """Nest a flat pack back into the two sub dicts the A1111 loader expects."""
    from safetensors.torch import load_file

    flat = load_file(pack_path)
    state = {"image_proj": {}, "ip_adapter": {}}
    for key, value in flat.items():
        if not isinstance(value, type(flat[key])) or not hasattr(value, "shape"):
            continue
        if key.startswith("image_proj."):
            state["image_proj"][key[len("image_proj.") :]] = value
        elif key.startswith("ip_adapter."):
            state["ip_adapter"][key[len("ip_adapter.") :]] = value
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1111", required=True, help="the A1111 root; read only")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--family", default="pulid", choices=["pulid", "plus", "faceid"])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    root = os.path.abspath(a.a1111)
    cn_scripts = os.path.join(root, "extensions-builtin", "sd-webui-controlnet", "scripts")
    # root first (modules/), the extension's scripts last: that directory carries a
    # logging.py that would otherwise shadow the stdlib module for everything else.
    if root not in sys.path:
        sys.path.insert(0, root)
    if cn_scripts not in sys.path:
        sys.path.append(cn_scripts)

    import hashlib

    import torch

    # Count the kernel at its own boundary: the route must reach this function for a
    # layer to count as executed in int8.
    from comfy_kitchen.backends import cuda as kc

    calls = []
    original = kc.int8_linear

    def counting(x, w, scale, bias=None, **kw):
        calls.append({
            "groupsize": kw.get("convrot_groupsize"),
            "convrot": kw.get("convrot"),
            "in_dtype": str(x.dtype),
            "out_dtype": str(kw.get("out_dtype")),
            "shape": list(x.shape),
        })
        return original(x, w, scale, bias, **kw)

    kc.int8_linear = counting

    state = load_pack(a.pack)
    n_ip = len(state["image_proj"])
    n_ipa = len(state["ip_adapter"])

    mod = importlib.import_module("ipadapter.ipadapter_model")
    if a.family == "pulid":
        width = int(state["image_proj"]["body.0.weight"].shape[1])
        model = mod.IPAdapterModel(
            state, clip_embeddings_dim=width, cross_attention_dim=2048,
            is_plus=False, is_sdxl=True, sdxl_plus=False, is_full=False,
            is_faceid=False, is_portrait=False, is_instantid=False, is_pulid=True,
            is_v2=False,
        )
    elif a.family == "faceid":
        width = int(state["image_proj"]["proj.0.weight"].shape[1])
        model = mod.IPAdapterModel(
            state, clip_embeddings_dim=width, cross_attention_dim=768,
            is_plus=False, is_sdxl=False, sdxl_plus=False, is_full=False,
            is_faceid=True, is_portrait=False, is_instantid=False, is_pulid=False,
            is_v2=False,
        )
    else:
        width = int(state["image_proj"]["proj_in.weight"].shape[1])
        model = mod.IPAdapterModel(
            state, clip_embeddings_dim=width, cross_attention_dim=768,
            is_plus=True, is_sdxl=False, sdxl_plus=False, is_full=False,
            is_faceid=False, is_portrait=False, is_instantid=False, is_pulid=False,
            is_v2=False,
        )

    def census(module):
        total = 0
        int8 = 0
        names_int8 = []
        for name, m in module.named_modules():
            if isinstance(m, torch.nn.Linear):
                total += 1
                w = getattr(m, "weight", None)
                if w is not None and w.dtype == torch.int8 and hasattr(m, "weight_scale"):
                    int8 += 1
                    names_int8.append(name)
        return total, int8, names_int8

    ip_total, ip_int8, ip_names = census(model.image_proj_model)
    kv_total, kv_int8, kv_names = census(model.ip_layers)

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    x = torch.randn(1, width, generator=gen).to(dev)
    y = ([torch.randn(1, 6, 1024, generator=gen).to(dev) for _ in range(5)]
         if a.family == "pulid" else None)

    finite = True
    with torch.no_grad():
        emb = model.image_proj_model(x, y) if y is not None else model.image_proj_model(x)
        kv = torch.cat(
            [layer(emb).reshape(-1) for _, layer in model.ip_layers.to_kvs.items()
             if layer.in_features == emb.shape[-1]]
        )
    finite = bool(torch.isfinite(emb).all().item() and torch.isfinite(kv).all().item())

    rec = {
        "tool": "a1111_pulid_route.py",
        "family": "pulid",
        "route": "A1111 sd-webui-controlnet ipadapter + modules/ipadapter_int8_convrot.py",
        "pack": os.path.basename(a.pack),
        "pack_sha256": hashlib.sha256(open(a.pack, "rb").read()).hexdigest(),
        "pack_bytes": os.path.getsize(a.pack),
        "declared_sub_tensors": {"image_proj": n_ip, "ip_adapter": n_ipa},
        "clip_embeddings_dim": width,
        "cross_attention_dim": 2048,
        "image_proj": {
            "linear_modules": ip_total,
            "int8_modules": ip_int8,
            "int8_names_first": ip_names[:3],
        },
        "ip_layers": {
            "linear_modules": kv_total,
            "int8_modules": kv_int8,
            "int8_names_first": kv_names[:3],
        },
        "int8_gemm_calls": len(calls),
        "groupsizes": sorted({c["groupsize"] for c in calls if c["groupsize"]}),
        "call_shapes_first": calls[:3],
        "emb_shape": list(emb.shape),
        "kv_shape": list(kv.shape),
        "output_finite": finite,
        "device": str(dev),
        "dtype": "float32 activations",
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)

    print(
        "image_proj %d Linear / %d int8 | ip_layers %d Linear / %d int8 | "
        "INT8 GEMM %d | groupsizes %s | finite %s"
        % (ip_total, ip_int8, kv_total, kv_int8, len(calls),
           rec["groupsizes"], finite)
    )
    print("emb %s | kv %s" % (rec["emb_shape"], rec["kv_shape"]))
    print("wrote %s" % a.json)


if __name__ == "__main__":
    main()

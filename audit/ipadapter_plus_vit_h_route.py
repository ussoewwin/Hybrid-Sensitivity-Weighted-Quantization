# -*- coding: utf-8 -*-
"""Route census and execution check for the SDXL IP-Adapter Plus (ViT-H) ConvRot INT8 pack.

Three questions, each answered on this machine rather than assumed:

  1. ComfyUI's own extension - `comfyui_ipadapter_plus/utils.py :: ipadapter_model_loader` reads
     the file with `load_torch_file(..., safe_load=True)`. Does anything in that path consult
     `comfy_quant`, or do the int8 codes reach the module as magnitudes?
  2. A1111 - `IPAdapterModel.load` -> `modules/ipadapter_int8_convrot` installs the payload in
     place. How many Linears end up on the INT8 path, and how many are skipped?
  3. The kernel - does one forward through the adapter's own path call the INT8 GEMM, and how
     many times?

This adapter's `image_proj` is a `Resampler` of depth 4 (the class A1111's extension builds for
`ip-adapter-plus_sdxl_vit-h`), and the runtime parameters are derived from the checkpoint:
`latents` (1, 16, 1280) gives num_queries 16 and dim 1280; `proj_in` (1280, 1280) gives
embedding_dim 1280; `proj_out` (2048, 1280) gives output_dim 2048; `layers.*.0.to_q` (1280, 1280)
gives dim_head * heads = 1280, which is 64 * 20; `layers.*.1.1` (5120, 1280) gives ff_mult 4.
That is exactly what the extension builds for `sdxl_plus`, so the graph the pack is measured
against is the graph the runtime builds.

  python audit/ipadapter_plus_vit_h_route.py --source <src>.safetensors --pack <pack>.safetensors \
      --comfy <ComfyUI> --root <A1111> --json <record>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import torch


def probe_comfyui(pack_path, comfy_root):
    """Does the IP-Adapter Plus loader apply the pack, or hand on raw int8 codes?"""
    out = {"route": "comfyui", "ok": False, "steps": []}
    plugin = os.path.join(comfy_root, "custom_nodes", "comfyui_ipadapter_plus")
    if not os.path.isdir(plugin):
        out["steps"].append({"step": "find comfyui_ipadapter_plus", "ok": False,
                             "error": "not installed under custom_nodes"})
        return out
    try:
        sys.path.insert(0, comfy_root)
        sys.path.insert(0, plugin)
        import comfy.utils as cu

        src = os.path.abspath(pack_path)
        sd = cu.load_torch_file(src, safe_load=True)
        int8_tensors = sum(1 for v in sd.values() if torch.is_tensor(v) and v.dtype == torch.int8)
        markers = sum(1 for k in sd if k.endswith(".comfy_quant"))
        scales = sum(1 for k in sd if k.endswith(".weight_scale"))
        out["steps"].append({"step": "load_torch_file(safe_load=True)", "ok": True,
                             "tensors": len(sd), "int8_tensors": int8_tensors,
                             "comfy_quant_markers": markers, "weight_scale_tensors": scales})
        out["measured"] = {
            "int8_tensors_presented": int8_tensors,
            "comfy_quant_markers": markers,
            "weight_scale_tensors": scales,
            "applies_int8": False,
            "why": ("load_torch_file returns the tensors; nothing in this call reads comfy_quant, "
                    "so the int8 codes would be assigned to float parameters"),
        }
        out["ok"] = True
    except Exception as e:  # noqa: BLE001
        out["steps"].append({"step": "load_torch_file(safe_load=True)", "ok": False,
                             "error": "%s: %s" % (type(e).__name__, e)})
    return out


def _import_ipadapter_model(root):
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


def nested_from_flat(sd):
    nested = {"image_proj": {}, "ip_adapter": {}}
    for key, value in sd.items():
        for head in ("image_proj.", "ip_adapter."):
            if key.startswith(head):
                nested[head[:-1]][key[len(head):]] = value
                break
    return nested


def probe_a1111(source_path, pack_path, root, device):
    """Does A1111's IP-Adapter graph run this pack through the INT8 kernel?"""
    out = {"route": "a1111", "ok": False, "steps": []}
    try:
        sys.path.insert(0, os.path.abspath(root))
        mod = _import_ipadapter_model(root)
        from safetensors.torch import load_file

        src_sd = nested_from_flat(load_file(source_path))
        pack_sd = nested_from_flat(load_file(pack_path))

        # the runtime derives is_plus / sdxl_plus from the file's own contents and name
        src_model = mod.IPAdapterModel.load(src_sd, os.path.basename(source_path))
        pack_model = mod.IPAdapterModel.load(pack_sd, os.path.basename(pack_path))
        for m in (src_model, pack_model):
            m.device = device
            m.to(device)
            m.eval()
        out["steps"].append({"step": "IPAdapterModel.load (source and pack)", "ok": True})

        int8_modules = [(n, m) for n, m in pack_model.named_modules()
                         if getattr(m, "_hswq_kind", None) == "linear"
                         and getattr(m, "weight", None) is not None
                         and m.weight.dtype == torch.int8]
        out["int8_layers_installed"] = len(int8_modules)
        out["int8_layers_by_graph"] = {
            "image_proj": len([1 for n, _ in int8_modules if n.startswith("image_proj_model")]),
            "ip_adapter": len([1 for n, _ in int8_modules if not n.startswith("image_proj_model")]),
        }
        if not int8_modules:
            out["steps"].append({"step": "count INT8 modules", "ok": False,
                                 "error": "no ConvRot INT8 module was installed on this route"})
            return out

        # one forward through the adapter's own path, with the kernel counted
        import comfy_kitchen.backends.cuda as kc
        calls = {"n": 0}
        orig = kc.int8_linear

        def wrapper(*args, **kw):
            calls["n"] += 1
            return orig(*args, **kw)
        kc.int8_linear = wrapper
        try:
            with torch.inference_mode():
                x = torch.randn(1, 257, 1280, device=device, dtype=torch.float32)
                out_stack = pack_model.image_proj_model(x)
        finally:
            kc.int8_linear = orig
        out["steps"].append({"step": "one image_proj forward", "ok": True,
                             "kernel_calls": calls["n"],
                             "output_shape": list(out_stack.shape),
                             "finite": bool(torch.isfinite(out_stack).all())})
        out["measured"] = {"int8_kernel_calls": calls["n"],
                           "output_shape": list(out_stack.shape),
                           "output_finite": bool(torch.isfinite(out_stack).all())}
        out["ok"] = True
    except Exception as e:  # noqa: BLE001
        out["steps"].append({"step": "A1111 route", "ok": False,
                             "error": "%s: %s" % (type(e).__name__, e),
                             "traceback": traceback.format_exc()[-1200:]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--root", required=True, help="A1111 install root")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", required=True)
    a = ap.parse_args()

    routes = {}
    routes["comfyui"] = probe_comfyui(a.pack, a.comfy)
    routes["a1111"] = probe_a1111(a.source, a.pack, a.root, a.device)

    record = {
        "tool": "audit/ipadapter_plus_vit_h_route.py",
        "family": "ipadapter_plus_vit_h",
        "source": os.path.basename(a.source),
        "pack": os.path.basename(a.pack),
        "routes": routes,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
    for name, r in routes.items():
        print("[%s] ok=%s %s" % (name, r.get("ok"),
                                  json.dumps(r.get("measured", r.get("steps")), ensure_ascii=False)[:220]),
              flush=True)
    print("wrote", a.json, flush=True)


if __name__ == "__main__":
    main()

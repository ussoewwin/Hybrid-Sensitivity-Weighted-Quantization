#!/usr/bin/env python
"""Load a ConvRot INT8 model patch through the HSWQ loader node the card names, and report wat it applies.

The Z-Image Fun ControlNet artifacts and the Qwen-Image diffsynth artifacts are **model patches**,
not ControlNets: their runtime object is built by ``ModelPatchLoader`` /
``HSWQModelPatchLoaderCustom`` (both read ``models/model_patches``) and applied by a family node.
``comfy.controlnet.load_controlnet_state_dict`` has no branch for either layout, so measuring them
with a ControlNet loader measures the wrong thing.

What this reports, per pack:

- the class the HSWQ node builds,
- how many modules carry an INT8 weight, against how many layers the pack declares,
- which declared layers did not end up INT8 (these never execute as quantised weights),
- whether any INT8 weight exists that the pack did not declare,
- whether any INT8 parameter is left where the graph reads it as data (codes copied, not executed).

Usage:
    python audit/model_patch_hswq_route.py --pack <file> --comfy <ComfyUI root>
                                           [--loader-root <dir>] [--json <out>]
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import struct
import sys

import torch


def declared_layers(pack_path: str):
    with open(pack_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    meta = hdr.pop("__metadata__", None) or {}
    qm = json.loads(meta.get("_quantization_metadata", "{}"))
    layers = sorted(qm.get("layers", {}))
    return layers, len(hdr)


def is_int8_weight(t) -> bool:
    return "Quantized" in type(t).__name__


def load_hswq_module(comfy_root: str, loader_root: str):
    root = os.path.abspath(comfy_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    cand = loader_root or os.path.join(root, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools")
    cand = os.path.abspath(cand)
    init = os.path.join(cand, "__init__.py")
    if not os.path.isfile(init):
        raise SystemExit(
            "HSWQ loader not found at %s; pass --loader-root pointing at the node package" % cand)
    pkg_name = "_hswq_model_patch_route"
    spec = importlib.util.spec_from_file_location(
        pkg_name, init, submodule_search_locations=[cand])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = pkg
    spec.loader.exec_module(pkg)
    return importlib.import_module(pkg_name + ".nodes.hswq_model_patch_loader")


def measure(pack_path: str, comfy_root: str, loader_root: str):
    root = os.path.abspath(comfy_root)
    cwd = os.getcwd()
    try:
        os.chdir(root)
        import folder_paths  # noqa: E402
    finally:
        os.chdir(cwd)
    sys.path.insert(0, root)
    folder_paths.add_model_folder_path("model_patches", os.path.dirname(os.path.abspath(pack_path)))
    node = load_hswq_module(root, loader_root)

    declared, pack_tensors = declared_layers(pack_path)
    name = os.path.basename(pack_path)
    (patched,) = node.HSWQModelPatchLoaderCustom().load_model_patch(name, False)

    model = patched.model if hasattr(patched, "model") else patched
    int8, float_mods, raw_code = [], [], []
    for mod_name, m in model.named_modules():
        w = getattr(m, "weight", None)
        if w is None or not hasattr(w, "is_floating_point"):
            continue
        if is_int8_weight(w):
            int8.append(mod_name)
        elif w.is_floating_point():
            float_mods.append(mod_name)
        if isinstance(w, torch.nn.Parameter) and w.dtype == torch.int8:
            raw_code.append(mod_name)

    declared_set = set(declared)
    int8_set = set(int8)
    missing = sorted(declared_set - int8_set)
    undeclared = sorted(int8_set - declared_set)

    rec = {
        "tool": "model_patch_hswq_route.py",
        "pack": {"file": name, "bytes": os.path.getsize(pack_path)},
        "loader_node": "HSWQModelPatchLoaderCustom",
        "model_class": type(model).__name__,
        "pack_tensors": pack_tensors,
        "modules_with_weight": len(int8) + len(float_mods),
        "declared_layers": len(declared),
        "int8_modules": len(int8),
        "float_modules": len(float_mods),
        "declared_not_int8": missing,
        "int8_not_declared": undeclared,
        "raw_int8_parameters": raw_code,
        "applied": len(declared) - len(missing),
    }
    print("%s | class %s | declared %d | int8 modules %d | applied %d/%d | raw code %d | undeclared %d"
          % (name, type(model).__name__, len(declared), len(int8), rec["applied"], len(declared),
             len(raw_code), len(undeclared)))
    if missing:
        print("  declared but not INT8:", ", ".join(missing[:6]))
    if undeclared:
        print("  INT8 but not declared:", ", ".join(undeclared[:6]))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--loader-root", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    rec = measure(args.pack, args.comfy, args.loader_root)
    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Load a ConvRot INT8 ControlNet through the HSWQ loader node and report what it applies.

The ComfyUI ``ControlNetLoader`` is not the right instrument for these packs. The HSWQ node
(``nodes/hswq_load_convrot_int8_controlnet.py``) builds them with ``MixedPrecisionOps`` so the
INT8 weights stay INT8, and it builds comfy-native FLUX ControlNets directly because
``comfy.controlnet.load_controlnet_state_dict`` has no branch for that layout. Measuring a pack
with the ComfyUI loader therefore measures the wrong thing.

What this reports, per pack:

- the class the HSWQ node builds,
- how many modules carry an INT8 weight, against how many layers the pack declares,
- which declared layers did not end up INT8 (these never execute as quantised weights),
- whether any INT8 weight exists that the pack did not declare.

Usage:
    python audit/controlnet_hswq_route.py --pack <file> --comfy <ComfyUI root>
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
    pkg_name = "_hswq_loader_route"
    spec = importlib.util.spec_from_file_location(
        pkg_name, init, submodule_search_locations=[cand])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = pkg
    spec.loader.exec_module(pkg)
    return importlib.import_module(pkg_name + ".nodes.hswq_load_convrot_int8_controlnet")


def measure(pack_path: str, comfy_root: str, loader_root: str):
    root = os.path.abspath(comfy_root)
    cwd = os.getcwd()
    try:
        os.chdir(root)
        import folder_paths  # noqa: E402
    finally:
        os.chdir(cwd)
    sys.path.insert(0, root)
    folder_paths.add_model_folder_path("controlnet", os.path.dirname(os.path.abspath(pack_path)))
    node = load_hswq_module(root, loader_root)

    declared, pack_tensors = declared_layers(pack_path)
    name = os.path.basename(pack_path)
    (net,) = node.HSWQControlNetLoader().load_controlnet(name)

    cm = net.control_model
    int8, float_mods, raw_code = [], [], []
    for mod_name, m in cm.named_modules():
        w = getattr(m, "weight", None)
        if w is None or not hasattr(w, "is_floating_point"):
            continue
        if is_int8_weight(w):
            int8.append(mod_name)
        elif w.is_floating_point():
            float_mods.append(mod_name)
        if isinstance(w, torch.nn.Parameter) and w.dtype == torch.int8:
            # an int8 parameter the graph reads as data: the codes are copied, not executed
            raw_code.append(mod_name)

    raw_declared = sorted(set(declared) & set(raw_code))
    missing = sorted(set(declared) - set(int8) - set(raw_declared))
    extra = sorted(set(int8) - set(declared))
    return {
        "pack": name,
        "pack_bytes": os.path.getsize(pack_path),
        "class": type(cm).__name__,
        "pack_tensors": pack_tensors,
        "runtime_tensors": len(cm.state_dict()),
        "declared_layers": len(declared),
        "int8_modules": len(int8),
        "float_modules": len(float_mods),
        "applied_layers": len(int8),
        "raw_code_layers": len(raw_declared),
        "raw_code_names": raw_declared,
        "declared_not_applied": len(missing),
        "declared_not_applied_names": missing,
        "int8_not_declared": len(extra),
        "int8_not_declared_names": extra,
        "verdict": ("applies" if (not missing and not raw_declared)
                    else "partially applies"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--comfy", required=True, help="ComfyUI root (contains comfy/, nodes.py)")
    ap.add_argument("--loader-root", default="",
                    help="HSWQ loader package dir; defaults to <comfy>/custom_nodes/ComfyUI-HSWQ-Loader-and-Tools")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    rec = measure(args.pack, args.comfy, args.loader_root)
    print("%s | class %s | declared %d | INT8 %d | not applied %d"
          % (rec["pack"], rec["class"], rec["declared_layers"],
             rec["applied_layers"], rec["declared_not_applied"]))
    if rec.get("raw_code_layers"):
        print("   raw int8 codes (not executed as INT8):", rec["raw_code_layers"])
    if rec["declared_not_applied"]:
        for n in rec["declared_not_applied_names"]:
            print("   not applied:", n)
    print("VERDICT: %s" % rec["verdict"])
    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

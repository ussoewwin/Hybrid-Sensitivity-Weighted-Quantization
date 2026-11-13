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
        "pack_sha256": __import__("hashlib").sha256(open(pack_path, "rb").read()).hexdigest(),
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



def measure_ipadapter(pack_path: str, ipadapter_dir: str):
    """The same question for an IP-Adapter pack: does the node's loader put the declared
    int8 layers on an int8 path, or does it read the codes as float weights?"""
    from safetensors.torch import load_file

    flat = load_file(pack_path)
    declared = sorted(k[: -len(".comfy_quant")] for k in flat if k.endswith(".comfy_quant"))
    logical = {k for k in flat if not k.endswith((".weight_scale", ".comfy_quant"))}
    st = {"image_proj": {}, "ip_adapter": {}}
    for k, v in flat.items():
        if not torch.is_tensor(v):
            continue
        if k.startswith("image_proj."):
            st["image_proj"][k.replace("image_proj.", "", 1)] = v
        elif k.startswith("ip_adapter."):
            st["ip_adapter"][k.replace("ip_adapter.", "", 1)] = v

    if ipadapter_dir and ipadapter_dir not in sys.path:
        sys.path.insert(0, ipadapter_dir)
    from image_proj_models import Resampler  # noqa: E402

    res = Resampler(dim=768, depth=4, dim_head=64, heads=12, num_queries=16,
                    embedding_dim=1280, output_dim=768, ff_mult=4,
                    max_seq_len=257, apply_pos_emb=False, num_latents_mean_pooled=0)
    load_error = ""
    try:
        res.load_state_dict(st["image_proj"], strict=True)
    except Exception as e:  # noqa: BLE001
        load_error = "%s: %s" % (type(e).__name__, str(e)[:300])

    int8, float_mods, raw_code = [], [], []
    if not load_error:
        for mod_name, m in res.named_modules():
            if not isinstance(m, torch.nn.Linear):
                continue
            w = m.weight
            if w.dtype == torch.int8:
                int8.append("image_proj." + mod_name)
            else:
                float_mods.append("image_proj." + mod_name)
    for k, v in st["ip_adapter"].items():
        if not k.endswith(".weight") or not torch.is_tensor(v):
            continue
        name = "ip_adapter." + k[: -len(".weight")]
        if v.dtype == torch.int8:
            # the node assigns this tensor to weight.data of a float Linear: the codes
            # are read as values, not executed by a kernel
            raw_code.append(name)
        else:
            float_mods.append(name)

    raw_declared = sorted(set(declared) & set(raw_code))
    missing = sorted(set(declared) - set(int8) - set(raw_declared))
    extra = sorted(set(int8) - set(declared))
    return {
        "pack": os.path.basename(pack_path),
        "pack_bytes": os.path.getsize(pack_path),
        "class": "Resampler+To_KV" if not load_error else "Resampler(load failed)",
        "pack_tensors": len(logical),
        "runtime_tensors": len(res.state_dict()) + len([k for k in st["ip_adapter"] if k.endswith(".weight")]),
        "declared_layers": len(declared),
        "int8_modules": len(int8),
        "float_modules": len(float_mods),
        "applied_layers": len(int8),
        "raw_code_layers": len(raw_declared),
        "raw_code_names": raw_declared[:20],
        "declared_not_applied": len(missing),
        "declared_not_applied_names": missing[:20],
        "int8_not_declared": len(extra),
        "int8_not_declared_names": extra[:20],
        "loader_error": load_error,
        "verdict": ("applies" if (not load_error and not missing and not raw_declared)
                    else "not applied"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--comfy", required=True, help="ComfyUI root (contains comfy/, nodes.py)")
    ap.add_argument("--loader-root", default="",
                    help="HSWQ loader package dir; defaults to <comfy>/custom_nodes/ComfyUI-HSWQ-Loader-and-Tools")
    ap.add_argument("--json", default="")
    ap.add_argument("--family", default="controlnet", choices=["controlnet", "ipadapter"])
    ap.add_argument("--ipadapter_dir", default="", help="the ipadapter node directory for --family ipadapter")
    args = ap.parse_args()

    if args.family == "ipadapter":
        rec = measure_ipadapter(args.pack, args.ipadapter_dir)
    else:
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

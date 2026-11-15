"""Route census for the SDXL basic IP-Adapter pack: which loaders take it, and how.

Three questions, each answered on this machine rather than assumed:

  1. ComfyUI   - `comfyui_ipadapter_plus/utils.py :: ipadapter_model_loader` reads the file with
                 `load_torch_file(..., safe_load=True)`. Does anything in that path consult
                 `comfy_quant`, or do the int8 codes reach the module as magnitudes?
  2. A1111     - `IPAdapterModel.load` -> `modules/ipadapter_int8_convrot` installs the payload in
                 place. How many Linears end up on the INT8 path, and how many are skipped?
  3. Forge     - `backend/operations_mixed_precision.py` reads `comfy_quant` and dispatches on the
                 convrot flag. Is that backend present here?

The census is not an acceptance: it says whether a code path takes the file. The acceptance
numbers live in `audit/ipadapter_xl_acceptance.py`.

  python audit/ipadapter_xl_route.py --pack <pack.safetensors> --comfy <ComfyUI> --root <A1111> --json <record>
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
        out["ok"] = True
        out["measured"] = {
            "int8_tensors_presented": int8_tensors,
            "comfy_quant_markers": markers,
            "weight_scale_tensors": scales,
            "applies_int8": False,
            "why": "load_torch_file returns the tensors; nothing in this call reads comfy_quant, "
                   "so the int8 codes would be assigned to float parameters",
        }
    except Exception as e:
        out["steps"].append({"step": "load_torch_file", "ok": False,
                             "error": "%s: %s" % (type(e).__name__, e),
                             "traceback": traceback.format_exc().splitlines()[-6:]})
    return out


def probe_a1111(pack_path, root):
    """`IPAdapterModel.load` installs the payload; count the Linears that land on INT8."""
    out = {"route": "a1111", "ok": False, "steps": []}
    try:
        sys.path.insert(0, os.path.abspath(root))
        pkg_dir = os.path.join(root, "extensions-builtin", "sd-webui-controlnet", "scripts", "ipadapter")
        if not os.path.isdir(pkg_dir):
            out["steps"].append({"step": "find the ControlNet ipadapter package", "ok": False,
                                 "error": "not found at " + pkg_dir})
            return out
        import importlib.util
        import types
        scripts_dir = os.path.dirname(pkg_dir)
        ext_root = os.path.dirname(scripts_dir)
        for p in (ext_root, os.path.abspath(root)):
            if p not in sys.path:
                sys.path.insert(0, p)
        for name, p in (("ext_scripts", scripts_dir), ("ext_scripts.ipadapter", pkg_dir)):
            if name not in sys.modules:
                m = types.ModuleType(name)
                m.__path__ = [p]
                sys.modules[name] = m
        spec = importlib.util.spec_from_file_location("ext_scripts.ipadapter.ipadapter_model",
                                                      os.path.join(pkg_dir, "ipadapter_model.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["ext_scripts.ipadapter.ipadapter_model"] = mod
        spec.loader.exec_module(mod)

        sd = torch.load(pack_path, map_location="cpu", weights_only=True)
        nested = {"image_proj": {}, "ip_adapter": {}}
        for key, value in sd.items():
            for head in ("image_proj.", "ip_adapter."):
                if key.startswith(head):
                    nested[head[:-1]][key[len(head):]] = value
                    break
        model = mod.IPAdapterModel.load(nested, os.path.basename(pack_path))
        int8 = [(n, m) for n, m in model.named_modules()
                if getattr(m, "_hswq_kind", None) == "linear"
                and getattr(m, "weight", None) is not None
                and m.weight.dtype == torch.int8]
        by_head = {"image_proj": 0, "ip_adapter": 0}
        for n, _m in int8:
            if n.startswith("image_proj_model"):
                by_head["image_proj"] += 1
            elif n.startswith("ip_layers"):
                by_head["ip_adapter"] += 1
        out["steps"].append({"step": "IPAdapterModel.load", "ok": True,
                             "installed_int8_linears": len(int8), "by_head": by_head})
        out["ok"] = True
        out["measured"] = {"installed_int8_linears": len(int8), "by_head": by_head,
                           "applies_int8": len(int8) > 0}
    except Exception as e:
        out["steps"].append({"step": "IPAdapterModel.load", "ok": False,
                             "error": "%s: %s" % (type(e).__name__, e),
                             "traceback": traceback.format_exc().splitlines()[-6:]})
    return out


def probe_forge(forge_root):
    """Is the Forge mixed-precision backend present, and does it read comfy_quant?"""
    out = {"route": "forge", "ok": False, "steps": []}
    if not forge_root:
        out["steps"].append({"step": "locate a Forge install", "ok": False,
                             "error": "no Forge root was given"})
        return out
    path = os.path.join(forge_root, "backend", "operations_mixed_precision.py")
    if not os.path.isfile(path):
        out["steps"].append({"step": "find backend/operations_mixed_precision.py", "ok": False,
                             "error": "not present at " + path})
        return out
    try:
        text = open(path, encoding="utf-8").read()
        reads_marker = "json.loads(layer_conf" in text or "layer_conf.numpy().tobytes()" in text
        convrot_flag = "convrot" in text
        out["steps"].append({"step": "inspect operations_mixed_precision.py", "ok": True,
                             "reads_comfy_quant": reads_marker, "mentions_convrot": convrot_flag})
        out["ok"] = True
        out["measured"] = {"reads_comfy_quant": reads_marker, "mentions_convrot": convrot_flag}
    except Exception as e:
        out["steps"].append({"step": "inspect operations_mixed_precision.py", "ok": False,
                             "error": "%s: %s" % (type(e).__name__, e)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--root", required=True, help="A1111 install root")
    ap.add_argument("--forge", default=None, help="Forge install root, if one exists here")
    ap.add_argument("--json", required=True)
    a = ap.parse_args()

    rec = {
        "tool": "audit/ipadapter_xl_route.py",
        "family": "ipadapter_xl",
        "pack": os.path.basename(a.pack),
        "pack_bytes": os.path.getsize(a.pack),
        "routes": {
            "comfyui": probe_comfyui(a.pack, os.path.abspath(a.comfy)),
            "a1111": probe_a1111(a.pack, os.path.abspath(a.root)),
            "forge": probe_forge(os.path.abspath(a.forge) if a.forge else None),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    for name, r in rec["routes"].items():
        print("[%s] ok=%s %s" % (name, r["ok"], json.dumps(r.get("measured", {}))[:160]))
    print("wrote", a.json)


if __name__ == "__main__":
    main()

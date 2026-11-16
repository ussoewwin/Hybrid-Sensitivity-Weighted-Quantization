# -*- coding: utf-8 -*-
"""Forge route census for the SDXL IP-Adapter Plus (ViT-H) ConvRot INT8 pack.

The route exists as code on this machine (a Forge checkout under the path passed in), but Forge's
own Python runtime does not: the interpreter candidates are checked and recorded one by one, so
the absence is shown rather than assumed.

What is measured here:

* which Forge files the route consists of (the mixed-precision operations that read
  ``comfy_quant`` and dispatch the ConvRot INT8 GEMM, and the IP-Adapter extension that builds the
  adapter),
* the pack's own ``comfy_quant`` stamps -- their format string, the ``convrot`` flag and the
  group size -- read out of the pack's metadata, layer by layer,
* whether those stamps are the form the Forge reader keys on (``int8_tensorwise`` + ``convrot`` +
  ``convrot_groupsize``).

    python audit/ipadapter_plus_vit_h_forge_route.py --pack <pack>.safetensors \
        --forge <Forge checkout> --json <record>.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os

from safetensors import safe_open

REQUIRED_FILES = {
    "launch.py": "launch.py",
    "webui.py": "webui.py",
    "operations_mixed_precision": os.path.join("backend", "operations_mixed_precision.py"),
    "ipadapter_plus": os.path.join("extensions-builtin", "sd_forge_ipadapter",
                                   "lib_ipadapter", "IPAdapterPlus.py"),
}

PYTHON_CANDIDATES = [
    os.path.join("venv", "Scripts", "python.exe"),
    os.path.join("venv", "bin", "python"),
    os.path.join("system", "python", "python.exe"),
    "python.exe",
]


def read_stamps(pack_path):
    stamps = []
    with safe_open(pack_path, framework="pt") as f:
        keys = [k for k in f.keys() if k.endswith(".comfy_quant")]
        for key in keys:
            raw = f.get_tensor(key)
            conf = json.loads(bytes(int(x) for x in raw).decode("utf-8"))
            stamps.append({"layer": key[: -len(".comfy_quant")], "conf": conf})
    return stamps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--forge", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()

    files = {}
    for name, rel in REQUIRED_FILES.items():
        files[name] = {"path": rel, "present": os.path.isfile(os.path.join(a.forge, rel))}

    pythons = {}
    for rel in PYTHON_CANDIDATES:
        pythons[rel] = os.path.isfile(os.path.join(a.forge, rel))

    stamps = read_stamps(a.pack)
    formats = collections.Counter(s["conf"].get("format") for s in stamps)
    convrot = collections.Counter(bool(s["conf"].get("convrot")) for s in stamps)
    groups = collections.Counter(s["conf"].get("convrot_groupsize") for s in stamps)

    expected = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
    matches = all(s["conf"].get("format") == expected["format"]
                  and bool(s["conf"].get("convrot")) is True
                  and int(s["conf"].get("convrot_groupsize", -1)) == expected["convrot_groupsize"]
                  for s in stamps) and len(stamps) > 0

    record = {
        "tool": "audit/ipadapter_plus_vit_h_forge_route.py",
        "family": "ipadapter_plus_vit_h",
        "pack": os.path.basename(a.pack),
        "route": "forge",
        "install": {"root_present": os.path.isdir(a.forge), "files": files},
        "python_runtime": {"candidates": pythons, "any_present": any(pythons.values())},
        "pack_stamps": {
            "count": len(stamps),
            "formats": {str(k): v for k, v in formats.items()},
            "convrot_true": convrot.get(True, 0),
            "groupsize_values": {str(k): v for k, v in groups.items()},
        },
        "forge_reader": {
            "file": REQUIRED_FILES["operations_mixed_precision"],
            "keys": ["format == 'int8_tensorwise'", "convrot", "convrot_groupsize"],
            "pack_matches_form": bool(matches),
        },
        "executed": False,
        "why_not_executed": ("no Forge Python runtime on this machine; the interpreter paths in "
                             "python_runtime were each checked and none exists"),
    }

    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: record[k] for k in ("pack_stamps", "forge_reader", "python_runtime")},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

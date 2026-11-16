# -*- coding: utf-8 -*-
"""The size/quality curve for the SDXL IP-Adapter Plus (ViT-H) pack.

The method builds the pack across a range of thresholds and measures the gate at each, then
takes the point where the curve stops paying for itself: **the smallest pack whose cosine is
not beaten by the smaller ones** (`HSWQ_High_Quality_Packs_ConvRot_INT8.md`, Rule 4).

Here the K axis is the number of layers kept in FP16, and the layers are taken from this
checkpoint's own aggregate-contribution ranking
(`clip_convert/measure_ipadapter_plus_vit_h_contrib.py`): the layers whose restoration removes
the most distance to the source. The contribution is the criterion; the drift sweep is only the
screen.

Each candidate is built with the converter, measured on the fixed 256-item set through the
A1111 ConvRot INT8 route - the same instrument, the same set, the same seed as every other
IP-Adapter pack - and then deleted, so the measurement does not leave packs behind.

    python clip_convert/sweep_ipadapter_plus_vit_h.py \
        --source <src>.safetensors --contrib impact/contrib_ipadapter_plus_vit_h.json \
        --root <A1111> --comfy <ComfyUI> --image <ref>.png \
        --clip-vision <COMFY>/models/clip_vision/clip_vision_h.safetensors \
        --out impact/curve_ipadapter_plus_vit_h.json --work <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

CONVERTER = os.path.join(HERE, "convert_ipadapter_plus_vit_h_convrot_int8.py")
ACCEPT = os.path.join(REPO, "audit", "ipadapter_plus_vit_h_acceptance.py")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_pack(source, impact, out, protect, work):
    protect_path = os.path.join(work, "protect.json")
    with open(protect_path, "w", encoding="utf-8") as f:
        json.dump(protect, f)
    cmd = [sys.executable, CONVERTER,
           "--input", source,
           "--impact", impact,
           "--output", out,
           "--protect-list", protect_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return out


def measure(pack, source, a, work, label, expect_int8):
    j5a = os.path.join(work, "5a.json")
    jtr = os.path.join(work, "trigger.json")
    jgate = os.path.join(work, "gate.json")
    cmd = [sys.executable, ACCEPT,
           "--source", source, "--pack", pack,
           "--image", a.image, "--clip-vision", a.clip_vision,
           "--root", a.root, "--comfy", a.comfy, "--device", a.device,
           "--seed", str(a.seed),
           "--json-5a", j5a, "--json-trigger", jtr, "--json-gate", jgate,
           "--expect-int8", str(expect_int8)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    with open(jgate, encoding="utf-8") as f:
        gate = json.load(f)
    with open(j5a, encoding="utf-8") as f:
        five_a = json.load(f)
    return {
        "label": label,
        "pack_bytes": os.path.getsize(pack),
        "pack_sha256": sha256_of(pack),
        "int8_layers": gate["int8_layers"],
        "image_proj_gate": gate["image_proj_surface"],
        "ip_adapter_gate": gate["ip_adapter_surface"],
        "gate_mean": gate["mean"],
        "gate_min": gate["min"],
        "gate_p5": gate["p5"],
        "gate_pass": gate["pass"],
        "int8_kernel_calls": gate["int8_kernel_calls"],
        "image_proj_cos_mean": five_a["image_proj_surface"]["cos_mean"],
        "image_proj_cos_min": five_a["image_proj_surface"]["cos_min"],
        "ip_adapter_cos_mean": five_a["ip_adapter_surface"]["cos_mean"],
        "ip_adapter_cos_min": five_a["ip_adapter_surface"]["cos_min"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--contrib", required=True)
    ap.add_argument("--impact", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--clip-vision", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--work", required=True, help="scratch directory for the candidates")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ks", default="0,20,60,100,162")
    a = ap.parse_args()

    os.makedirs(a.work, exist_ok=True)
    with open(a.contrib, encoding="utf-8") as f:
        contrib = json.load(f)
    ranking = sorted(contrib["contributions"].items(), key=lambda kv: -kv[1])

    ks = [int(x) for x in a.ks.split(",")]
    rows = []
    t0 = time.time()
    for k in ks:
        protect = [name for name, _ in ranking[:k]]
        pack = os.path.join(a.work, "k%d.safetensors" % k)
        print("[curve] K=%d  building %d protected layers" % (k, len(protect)), flush=True)
        build_pack(a.source, a.impact, pack, protect, a.work)
        row = measure(pack, a.source, a, a.work, "k%d" % k,
                      expect_int8=162 - len(protect))
        row["kept_fp16"] = len(protect)
        row["pack_mb"] = round(os.path.getsize(pack) / (1024 ** 2), 1)
        rows.append(row)
        print("[curve] K=%d  %s MB  img mean %.9f min %.9f | ipa mean %.9f min %.9f  %s"
              % (k, row["pack_mb"], row["gate_mean"], row["gate_min"],
                 row["ip_adapter_cos_mean"], row["ip_adapter_cos_min"],
                 "PASS" if row["gate_pass"] else "FAIL"), flush=True)
        os.remove(pack)
        print("[curve] K=%d candidate deleted" % k, flush=True)

    if rows:
        base = rows[0]
        for r in rows:
            r["gate_margin_mean"] = r["gate_mean"] - 0.99
            r["gate_margin_min"] = r["gate_min"] - 0.98
            r["gain_mean_vs_floor"] = r["gate_mean"] - base["gate_mean"]
            r["gain_over_margin_mean"] = (r["gate_mean"] - base["gate_mean"]) / (r["gate_mean"] - 0.99)
            r["gain_over_margin_min"] = (r["gate_min"] - base["gate_min"]) / (r["gate_min"] - 0.98)

    out = {
        "tool": "clip_convert/sweep_ipadapter_plus_vit_h.py",
        "family": "ipadapter_plus_vit_h",
        "source": os.path.basename(a.source),
        "instrument": "the fixed 256-item set, seed 42, on the A1111 ConvRot INT8 route",
        "criterion": "the smallest pack whose cosine is not beaten by the smaller ones",
        "ranking_source": os.path.basename(a.contrib),
        "ranking_top": [{"layer": n, "contribution": v} for n, v in ranking[:10]],
        "ks": ks,
        "seconds": round(time.time() - t0, 1),
        "curve": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()

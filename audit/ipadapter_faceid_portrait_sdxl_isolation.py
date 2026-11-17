# -*- coding: utf-8 -*-
"""CN-4 isolation: is the broken picture the adapter, the base model, or the graph?

Four arms, one prompt and one seed, on the same server:

  base          - the HSWQ SDXL UNet, no IP-Adapter at all
  adapter_w0    - the source adapter attached with weight 0.0 (the nodes run, the conditioning is zero)
  source_w0.8   - the source adapter at 0.8
  pack_w0.8     - the pack at 0.8

If ``base`` is already broken, the base model is the cause and no adapter arm can be judged. If
``base`` is clean and ``adapter_w0`` is clean while ``source_w0.8`` breaks, the adapter's own
conditioning is the cause, before any quantisation. Only if ``source_w0.8`` is clean and
``pack_w0.8`` breaks is the pack implicated.

    python audit/ipadapter_faceid_portrait_sdxl_isolation.py --server http://<comfyui-host> \\
        --adapter-dir <dir> --out-dir <dir> --output-root <server output> --json <record>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import urllib.request

CKPT = "waiIllustriousSDXL_v170_hswq_1on_re590_convrot_int8.safetensors"
VAE = "sdxl_vae.safetensors"
PACK = "ip-adapter-faceid-portrait-unnorm_sdxl_hswq_convrot_int8.safetensors"
SOURCE = "ip-adapter-faceid-portrait_sdxl_unnorm.bin"
REF = "cn_clipvit_ref.png"
PROMPT = "a woman in a red coat standing in a city street, detailed, natural light"
NEGATIVE = "blurry, low quality, watermark"


def post(url, payload, timeout=1800):
    req = urllib.request.Request(url + "/prompt", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def hist(server, prompt_id, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with urllib.request.urlopen(server + "/history/" + prompt_id, timeout=30) as r:
            h = json.loads(r.read().decode("utf-8"))
        if prompt_id in h:
            return h[prompt_id]
        time.sleep(3)
    return None


def graph(arm, model_dir):
    g = {
        "1": {"class_type": "HSWQCheckpointLoader",
              "inputs": {"ckpt_name": CKPT, "weight_dtype": "SDXL ConvRot INT8"}},
        "13": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["1", 1]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["1", 1]}},
        "8": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["13", 0]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "iso_" + arm}},
    }
    model_ref = ["1", 0]
    if arm != "base":
        adapter, weight = (SOURCE, 0.0) if arm == "adapter_w0" else (
            (SOURCE, 0.8) if arm == "source_w0.8" else (PACK, 0.8))
        g["2"] = {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}}
        g["3"] = {"class_type": "HSWQLoadIPAdapterModel",
                  "inputs": {"ipadapter_file": adapter, "ipadapter_dir": model_dir}}
        g["7"] = {"class_type": "LoadImage", "inputs": {"image": REF}}
        g["12"] = {"class_type": "IPAdapterInsightFaceLoader", "inputs": {"provider": "CPU"}}
        g["4"] = {"class_type": "IPAdapterFaceID",
                  "inputs": {"model": ["1", 0], "ipadapter": ["3", 0], "image": ["7", 0],
                             "clip_vision": ["2", 0], "insightface": ["12", 0],
                             "weight": weight, "weight_faceidv2": 1.0,
                             "weight_type": "linear", "combine_embeds": "concat",
                             "start_at": 0.0, "end_at": 1.0, "embeds_scaling": "V only"}}
        model_ref = ["4", 0]
    g["9"] = {"class_type": "KSampler",
              "inputs": {"model": model_ref, "seed": 42, "steps": 12, "cfg": 7.0,
                         "sampler_name": "dpmpp_2m", "scheduler": "karras",
                         "positive": ["5", 0], "negative": ["6", 0],
                         "latent_image": ["8", 0], "denoise": 1.0}}
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    arms = ["base", "adapter_w0", "source_w0.8", "pack_w0.8"]
    out = {"tool": "audit/ipadapter_faceid_portrait_sdxl_isolation.py",
           "family": "ipadapter_faceid_portrait_sdxl",
           "question": "which of base model / adapter conditioning / pack breaks the picture",
           "settings": {"ckpt": CKPT, "vae": VAE, "prompt": PROMPT, "negative": NEGATIVE,
                        "width": 1024, "height": 1024, "steps": 12, "cfg": 7.0,
                        "sampler": "dpmpp_2m", "scheduler": "karras", "seed": 42,
                        "reference": REF, "adapter_dir": a.adapter_dir},
           "arms": {}}
    for arm in arms:
        t0 = time.time()
        r = post(a.server, {"prompt": graph(arm, a.adapter_dir), "client_id": "portrait_iso"})
        pid = r["prompt_id"]
        print("[%s] queued %s" % (arm, pid), flush=True)
        h = hist(a.server, pid, a.timeout)
        if h is None:
            out["arms"][arm] = {"queued": pid, "status": "timed out"}
            continue
        status = h.get("status", {})
        saved = []
        for node in h.get("outputs", {}).values():
            for im in node.get("images", []):
                src = os.path.join(a.output_root, im.get("subfolder", ""), im["filename"])
                dst = os.path.join(a.out_dir, "%s.png" % arm)
                if os.path.isfile(src):
                    shutil.copyfile(src, dst)
                    saved.append({"file": os.path.basename(dst), "bytes": os.path.getsize(dst)})
        out["arms"][arm] = {"queued": pid, "completed": bool(status.get("completed")),
                            "status_str": status.get("status_str"),
                            "images": saved, "seconds": round(time.time() - t0, 1)}
        print("[%s] %s %s" % (arm, status.get("status_str"), saved), flush=True)

    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("wrote", a.json)


if __name__ == "__main__":
    main()

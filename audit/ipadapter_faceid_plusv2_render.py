# -*- coding: utf-8 -*-
"""CN-4 for the SD1.5 IP-Adapter FaceID Plus v2 pack, on the ComfyUI route through the loader node.

Two renders, same prompt / seed / steps / cfg / reference image, one with the source adapter and one
with the pack. Both go through `HSWQ Load IPAdapter Model` -> `IPAdapterAdvanced`, so the INT8 GEMM
is what the pack render evaluates.

    python audit/ipadapter_faceid_plusv2_render.py --comfy-root D:\\USERFILES\\ComfyUI \\
        --server http://127.0.0.1:8189 --out-dir <dir> --json <record>.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import urllib.request

CKPT = "realisticVisionV60B1_v60B1VAE.safetensors"
CLIP_VISION = "clip_vision_h.safetensors"
PACK = "ip-adapter-faceid-plusv2_sd15_hswq_convrot_int8.safetensors"
SOURCE = "ip-adapter-faceid-plusv2_sd15.bin"
MODEL_DIR = r"D:\USERFILES\StableDiffusion\models\ControlNet"
REF = "cn_clipvit_ref.png"

PROMPT = "a woman in a red coat standing in a city street, detailed, natural light"
NEGATIVE = "blurry, low quality, watermark"


def graph(adapter, prefix):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": CLIP_VISION}},
        "3": {"class_type": "HSWQLoadIPAdapterModel",
              "inputs": {"ipadapter_file": adapter, "ipadapter_dir": MODEL_DIR}},
        "4": {"class_type": "IPAdapterFaceID",
              "inputs": {"model": ["1", 0], "ipadapter": ["3", 0], "image": ["7", 0],
                         "clip_vision": ["2", 0], "insightface": ["12", 0],
                         "weight": 0.8, "weight_faceidv2": 1.0,
                         "weight_type": "linear",
                         "combine_embeds": "concat", "start_at": 0.0, "end_at": 1.0,
                         "embeds_scaling": "V only"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["1", 1]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["1", 1]}},
        "7": {"class_type": "LoadImage", "inputs": {"image": REF}},
        "12": {"class_type": "IPAdapterInsightFaceLoader", "inputs": {"provider": "CPU"}},
        "8": {"class_type": "EmptyLatentImage",
              "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "9": {"class_type": "KSampler",
              "inputs": {"model": ["4", 0], "seed": 42, "steps": 12, "cfg": 7.0,
                         "sampler_name": "dpmpp_2m", "scheduler": "karras",
                         "positive": ["5", 0], "negative": ["6", 0],
                         "latent_image": ["8", 0], "denoise": 1.0}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0],
                                                     "filename_prefix": prefix}},
    }


def post(server, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(server + "/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def wait(server, prompt_id, timeout):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(server + "/history/" + prompt_id, timeout=30) as r:
                h = json.loads(r.read().decode("utf-8"))
        except Exception:
            h = {}
        if prompt_id in h:
            return h[prompt_id]
        time.sleep(3)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--output-root", required=True, help="the output directory of the server")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    runs = [("source", SOURCE, "vit_h_render_source"), ("pack", PACK, "vit_h_render_pack")]
    out = {"tool": "audit/ipadapter_faceid_plusv2_render.py", "family": "ipadapter_faceid_plusv2_sd15",
           "route": "ComfyUI, HSWQ Load IPAdapter Model -> IPAdapterFaceID",
           "settings": {"ckpt": CKPT, "clip_vision": CLIP_VISION, "prompt": PROMPT,
                        "negative": NEGATIVE, "width": 512, "height": 512, "steps": 12,
                        "cfg": 7.0, "sampler": "dpmpp_2m", "scheduler": "karras",
                        "seed": 42, "weight": 0.8, "weight_faceidv2": 1.0, "reference": REF,
                        "adapter_dir": MODEL_DIR},
           "arms": {}}

    for name, adapter, prefix in runs:
        t0 = time.time()
        r = post(a.server, {"prompt": graph(adapter, prefix), "client_id": "vit_h_render"})
        pid = r["prompt_id"]
        print("[%s] queued %s" % (name, pid), flush=True)
        hist = wait(a.server, pid, a.timeout)
        if hist is None:
            out["arms"][name] = {"queued": pid, "status": "timed out"}
            continue
        status = hist.get("status", {})
        images = []
        for node in hist.get("outputs", {}).values():
            for im in node.get("images", []):
                images.append(im)
        saved = []
        for im in images:
            src = os.path.join(a.output_root, im.get("subfolder", ""),
                               im["filename"])
            dst = os.path.join(a.out_dir, "%s.png" % name)
            if os.path.isfile(src):
                shutil.copyfile(src, dst)
                saved.append({"file": os.path.basename(dst), "bytes": os.path.getsize(dst)})
        out["arms"][name] = {"queued": pid, "completed": bool(status.get("completed")),
                             "status_str": status.get("status_str"),
                             "images": saved, "seconds": round(time.time() - t0, 1)}
        print("[%s] %s %s" % (name, out["arms"][name].get("status_str"), saved), flush=True)

    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("wrote", a.json)


if __name__ == "__main__":
    main()

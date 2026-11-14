"""Three-arm render through the CLIP vision route that exists.

Arm A loads the source tower, arm B loads the pack, arm C loads a dequantised copy of the pack.
One graph, one seed, one prompt; only the clip vision file changes. The three frames and their
pixel statistics are written out so the outcome can be read off the picture rather than off a
cosine.

The point of arm C is that it carries the pack's own weights after a dequantiser has run. If B
and C differ, the difference is in the route, not in the arithmetic of the pack.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

import numpy as np
from PIL import Image

PROMPT = "a woman in a red coat standing in a city street, detailed, natural light"
NEGATIVE = "blurry, low quality, watermark"


def graph(ckpt, ipadapter, ref_image, clip_vision):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "LoadImage", "inputs": {"image": ref_image}},
        "3": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": clip_vision}},
        "4": {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": ipadapter}},
        "5": {"class_type": "IPAdapterAdvanced", "inputs": {
            "model": ["1", 0], "ipadapter": ["4", 0], "image": ["2", 0],
            "clip_vision": ["3", 0], "weight": 0.8, "weight_type": "linear",
            "combine_embeds": "concat", "start_at": 0.0, "end_at": 1.0,
            "embeds_scaling": "V only"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["1", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["1", 1]}},
        "8": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "9": {"class_type": "KSampler", "inputs": {
            "model": ["5", 0], "positive": ["6", 0], "negative": ["7", 0],
            "latent_image": ["8", 0], "seed": 42, "steps": 24, "cfg": 7.0,
            "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "cn_clipvit"}},
    }


def submit(base, body):
    req = urllib.request.Request(
        base + "/prompt", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=60).read().decode("utf-8"))["prompt_id"]
    except urllib.error.HTTPError as exc:
        raise SystemExit("submit rejected: " + exc.read().decode("utf-8")[:2000])


def wait(base, pid, limit_s):
    t0 = time.time()
    while time.time() - t0 < limit_s:
        time.sleep(3)
        hist = json.loads(urllib.request.urlopen(base + "/history/" + pid, timeout=30).read().decode("utf-8"))
        if pid not in hist:
            continue
        status = hist[pid]["status"]
        if status.get("completed"):
            frames = [i["filename"] for o in hist[pid]["outputs"].values() for i in o.get("images", [])]
            return frames, time.time() - t0, None
        if status.get("status_str") == "error":
            return None, time.time() - t0, json.dumps(status.get("messages", []))[:2000]
    return None, time.time() - t0, "timeout"


def stats(path):
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    return {"pixel_mean": round(float(arr.mean()), 2), "pixel_std": round(float(arr.std()), 2),
            "pixel_max": round(float(arr.max()), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True, help="ComfyUI root, for its input/ and output/ directories")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--reference", required=True, help="reference image already inside ComfyUI's input/")
    ap.add_argument("--dequant", required=True, help="dequantised copy of the pack, inside models/clip_vision")
    ap.add_argument("--source", default="CLIP-ViT-H-14.safetensors")
    ap.add_argument("--pack", default="CLIP-ViT-H-14_hswq_convrot_int8.safetensors")
    ap.add_argument("--ckpt", default="cyberrealistic_final.safetensors")
    ap.add_argument("--ipadapter", default="ip-adapter-plus_sd15.safetensors")
    ap.add_argument("--out", required=True, help="record path")
    ap.add_argument("--frames", required=True, help="directory for the three frames")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()

    os.makedirs(a.frames, exist_ok=True)
    base = a.server.rstrip("/")
    arms = [("A_source", a.source), ("B_pack", a.pack), ("C_dequant", a.dequant)]
    out = []
    for tag, clip in arms:
        pid = submit(base, {"prompt": graph(a.ckpt, a.ipadapter, a.reference, clip), "client_id": str(uuid.uuid4())})
        frames, secs, err = wait(base, pid, a.timeout)
        if err or not frames:
            out.append({"id": tag, "clip_vision": clip, "error": err or "no frame"})
            print(tag, "ERROR", err)
            continue
        data = urllib.request.urlopen(base + "/view?filename=" + frames[0] + "&type=output",
                                     timeout=120).read()
        local = os.path.join(a.frames, "cn_clip_vit_h14_%s_image_test.png" % tag)
        with open(local, "wb") as fh:
            fh.write(data)
        s = stats(local)
        out.append({"id": tag, "clip_vision": clip, "frame": os.path.basename(local),
                    "bytes": len(data), "sha256_16": hashlib.sha256(data).hexdigest().upper()[:16],
                    "wall_s": round(secs, 1), **s})
        print(tag, len(data), "B", s)

    rec = {"tool": "audit/clip_vit_h14_render.py", "family": "clip_vision",
           "source": a.source, "pack": a.pack, "dequant": a.dequant,
           "route": {"loader": "CLIPVisionLoader -> IPAdapterAdvanced.clip_vision",
                     "ipadapter": a.ipadapter, "checkpoint": a.ckpt},
           "prompt": {"positive": PROMPT, "negative": NEGATIVE, "width": 512, "height": 512,
                      "seed": 42, "steps": 24, "cfg": 7.0, "sampler": "dpmpp_2m",
                      "scheduler": "karras", "denoise": 1.0, "ipadapter_weight": 0.8,
                      "weight_type": "linear", "embeds_scaling": "V only",
                      "reference_image": a.reference},
           "arms": out}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False)
    print("record:", a.out)


if __name__ == "__main__":
    main()

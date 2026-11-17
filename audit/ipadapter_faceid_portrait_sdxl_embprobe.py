# -*- coding: utf-8 -*-
"""Which conditioning breaks the portrait picture: the face embedding branch, or the node?

The isolation run showed base and weight-0 clean, source_w0.8 and pack_w0.8 broken. This narrows it
to the conditioning itself, on the same server and seed:

  a  the face embedding the node feeds, read out and compared
     - unnormalised (`face.embedding`) against normalised (`face.normed_embedding`)
     - the values and norms, because this graph was trained on the unnormalised 512-d vector
  b  the source adapter recipe at weight 0.5 / 1.0, to see whether the breakage scales with weight

The record carries the embedding statistics and the render outcomes; the PNGs are looked at.

    python audit/ipadapter_faceid_portrait_sdxl_embprobe.py --server http://<comfyui-host> \\
        --adapter-dir <dir> --out-dir <dir> --output-root <output> --json <record>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import urllib.request

import numpy as np

CKPT = "waiIllustriousSDXL_v170_hswq_1on_re590_convrot_int8.safetensors"
VAE = "sdxl_vae.safetensors"
PACK = "ip-adapter-faceid-portrait-unnorm_sdxl_hswq_convrot_int8.safetensors"
SOURCE = "ip-adapter-faceid-portrait_sdxl_unnorm.bin"
REF = "cn_clipvit_ref.png"
PROMPT = "a woman in a red coat standing in a city street, detailed, natural light"
NEGATIVE = "blurry, low quality, watermark"


def post(server, payload):
    req = urllib.request.Request(server + "/prompt", data=json.dumps(payload).encode("utf-8"),
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


def graph(arm, model_dir, adapter, weight):
    return {
        "1": {"class_type": "HSWQCheckpointLoader",
              "inputs": {"ckpt_name": CKPT, "weight_dtype": "SDXL ConvRot INT8"}},
        "13": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "2": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}},
        "3": {"class_type": "HSWQLoadIPAdapterModel",
              "inputs": {"ipadapter_file": adapter, "ipadapter_dir": model_dir}},
        "7": {"class_type": "LoadImage", "inputs": {"image": REF}},
        "12": {"class_type": "IPAdapterInsightFaceLoader", "inputs": {"provider": "CPU"}},
        "4": {"class_type": "IPAdapterFaceID",
              "inputs": {"model": ["1", 0], "ipadapter": ["3", 0], "image": ["7", 0],
                         "clip_vision": ["2", 0], "insightface": ["12", 0],
                         "weight": weight, "weight_faceidv2": 1.0, "weight_type": "linear",
                         "combine_embeds": "concat", "start_at": 0.0, "end_at": 1.0,
                         "embeds_scaling": "V only"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["1", 1]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["1", 1]}},
        "8": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "9": {"class_type": "KSampler",
              "inputs": {"model": ["4", 0], "seed": 42, "steps": 12, "cfg": 7.0,
                         "sampler_name": "dpmpp_2m", "scheduler": "karras",
                         "positive": ["5", 0], "negative": ["6", 0],
                         "latent_image": ["8", 0], "denoise": 1.0}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["13", 0]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "emb_" + arm}},
    }


def embedding_stats(image_path, comfy, insight_root):
    import cv2
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", root=insight_root, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    faces = app.get(cv2.imread(image_path))
    if not faces:
        raise RuntimeError("no face in %s" % image_path)
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    raw = f.embedding.astype(np.float64)
    normed = f.normed_embedding.astype(np.float64)
    return {
        "detector": "insightface buffalo_l, largest face",
        "raw_norm": float(np.linalg.norm(raw)),
        "raw_absmax": float(np.abs(raw).max()),
        "raw_mean": float(raw.mean()),
        "normed_norm": float(np.linalg.norm(normed)),
        "normed_absmax": float(np.abs(normed).max()),
        "cos_raw_vs_normed": float(raw @ normed / (np.linalg.norm(raw) * np.linalg.norm(normed))),
        "note": "portrait-unnorm consumes face.embedding (unnormalised); every other FaceID branch "
                "consumes face.normed_embedding (IPAdapterPlus.py L293-296)",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    out = {"tool": "audit/ipadapter_faceid_portrait_sdxl_embprobe.py",
           "family": "ipadapter_faceid_portrait_sdxl",
           "embedding": embedding_stats(a.image, a.comfy, os.path.join(a.comfy, "models", "insightface")),
           "arms": {}}

    arms = [("source_w0.5", SOURCE, 0.5), ("source_w1.0", SOURCE, 1.0),
            ("pack_w0.5", PACK, 0.5)]
    for arm, adapter, weight in arms:
        t0 = time.time()
        r = post(a.server, {"prompt": graph(arm, a.adapter_dir, adapter, weight),
                            "client_id": "portrait_emb"})
        pid = r["prompt_id"]
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
    print(json.dumps(out["embedding"], indent=1))


if __name__ == "__main__":
    main()

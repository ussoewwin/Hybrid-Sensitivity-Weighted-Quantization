"""CN-4 on A1111: render with the source adapter and with the pack, through the API.

The ComfyUI path refused the pack at `load_state_dict`, so the first attempt concluded the
pack could not be rendered here. That conclusion was wrong: A1111 carries an IP-Adapter
ConvRot INT8 runtime and already lists `ip-adapter-faceid_sd15_hswq_convrot_int8`. This
runs both arms through the running A1111 API, with the same prompt, seed, steps and
reference image, and writes the PNGs so the images can be looked at.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request


def post(url, payload, timeout=1800):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:7861")
    ap.add_argument("--image", required=True)
    ap.add_argument("--source-model", required=True)
    ap.add_argument("--pack-model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--prompt", default="a woman in a red coat standing in a city street, detailed, natural light")
    ap.add_argument("--negative", default="blurry, low quality, watermark")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weight", type=float, default=0.8)
    ap.add_argument("--module", default="ip-adapter_face_id")
    a = ap.parse_args()

    from PIL import Image
    os.makedirs(a.out_dir, exist_ok=True)
    img = Image.open(a.image).convert("RGB")
    # The API carries the reference inline; 512 is enough for the face crop.
    ref = img.copy()
    ref.thumbnail((512, 512))
    buf = io.BytesIO()
    ref.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    ref_b64 = "data:image/png;base64," + b64

    opts = get(a.host + "/sdapi/v1/options")
    ckpt = opts.get("sd_model_checkpoint")
    print("[a1111] checkpoint: %s" % ckpt, flush=True)

    arms = []
    for label, model in (("source", a.source_model), ("pack", a.pack_model)):
        payload = {
            "prompt": a.prompt,
            "negative_prompt": a.negative,
            "width": a.width, "height": a.height,
            "steps": a.steps, "cfg_scale": a.cfg, "seed": a.seed,
            "sampler_name": "DPM++ 2M", "scheduler": "Karras",
            "batch_size": 1, "n_iter": 1,
            "alwayson_scripts": {
                "ControlNet": {
                    "args": [{
                        "enabled": True,
                        "module": a.module,
                        "model": model,
                        "weight": a.weight,
                        "image": {"image": ref_b64, "mask": None},
                        "resize_mode": "Crop and Resize",
                        "control_mode": "Balanced",
                        "pixel_perfect": True,
                        "guidance_start": 0.0,
                        "guidance_end": 1.0,
                        "low_vram": False,
                        "processor_res": 512,
                        "threshold_a": 0.5,
                        "threshold_b": 0.5,
                    }]
                }
            },
        }
        t0 = time.time()
        try:
            r = post(a.host + "/sdapi/v1/txt2img", payload)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:800]
            print("[%s] HTTP %s: %s" % (label, e.code, body), flush=True)
            arms.append({"label": label, "model": model, "outcome": "http_error",
                         "status": e.code, "body": body})
            continue
        except Exception as e:
            print("[%s] %s: %s" % (label, type(e).__name__, e), flush=True)
            arms.append({"label": label, "model": model, "outcome": "error",
                         "error": "%s: %s" % (type(e).__name__, e)})
            continue
        dt = time.time() - t0
        if not r.get("images"):
            arms.append({"label": label, "model": model, "outcome": "no_image",
                         "info": str(r.get("info"))[:400]})
            print("[%s] no image returned" % label, flush=True)
            continue
        raw = base64.b64decode(r["images"][0].split(",", 1)[-1])
        png = os.path.join(a.out_dir, "a1111_faceid_%s_%dx%d.png" % (label, a.width, a.height))
        open(png, "wb").write(raw)
        info = {}
        try:
            info = json.loads(r.get("info", "{}")) if isinstance(r.get("info"), str) else (r.get("info") or {})
        except Exception:
            pass
        rec = {"label": label, "model": model, "outcome": "image",
               "png": os.path.basename(png), "png_bytes": len(raw),
               "png_sha256": hashlib.sha256(raw).hexdigest().upper(),
               "wall_s": round(dt, 1),
               "seed": info.get("seed"), "sampler": info.get("sampler_name"),
               "steps": info.get("steps"), "cfg": info.get("cfg_scale")}
        arms.append(rec)
        print("[%s] PNG %d B in %.1f s -> %s" % (label, len(raw), dt, rec["png"]), flush=True)

    out = {
        "tool": "audit/ipadapter_faceid_a1111_render.py",
        "family": "faceid",
        "route": "a1111 /sdapi/v1/txt2img + ControlNet module " + a.module,
        "checkpoint": ckpt,
        "render": {"width": a.width, "height": a.height, "steps": a.steps, "cfg": a.cfg,
                   "sampler": "DPM++ 2M", "scheduler": "Karras", "seed": a.seed,
                   "weight": a.weight, "reference": os.path.basename(a.image),
                   "prompt": a.prompt},
        "arms": arms,
    }

    both = [x for x in arms if x.get("outcome") == "image"]
    if len(both) == 2:
        import numpy as np
        pa = np.asarray(Image.open(os.path.join(a.out_dir, arms[0]["png"])).convert("RGB")).astype(np.int16)
        pb = np.asarray(Image.open(os.path.join(a.out_dir, arms[1]["png"])).convert("RGB")).astype(np.int16)
        d = abs(pa - pb)
        out["source_vs_pack"] = {"mean_abs": round(float(d.mean()), 2),
                                 "max_abs": int(d.max()),
                                 "changed_pct": round(float((d > 0).mean() * 100), 2)}
        print("[source vs pack] mean_abs %.2f max_abs %d changed %.2f%%"
              % (out["source_vs_pack"]["mean_abs"], out["source_vs_pack"]["max_abs"],
                 out["source_vs_pack"]["changed_pct"]), flush=True)

    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("wrote", a.json, flush=True)


if __name__ == "__main__":
    main()

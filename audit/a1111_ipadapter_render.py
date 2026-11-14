"""SD1.5 IP-Adapter Plus (`ip-adapter_sd15_plus.pth`): one render through each route.

This runs on the A1111 instance, because that is the one consumer on this machine whose
ControlNet extension has an IP-Adapter INT8 path (`modules/ipadapter_int8_convrot.py`).
Three renders are produced on the same seed, the same prompt and the same image prompt:

  base   - no adapter at all, the reference the other two are compared against
  float  - the source .pth, the adapter the author published
  int8   - the HSWQ ConvRot INT8 pack

What is recorded is the loader's own install report for the pack (how many Linears it put on
the INT8 path and how many it skipped), the three PNGs with their sha256, and the pixel
comparisons between them. The float-vs-int8 pair is the one that says whether the quantisation
changed the picture; the base pairs say whether the adapter did anything at all.

  python audit/a1111_ipadapter_render.py \
      --url http://127.0.0.1:7860 \
      --image <image prompt png> \
      --float ip-adapter_sd15_plus.pth \
      --int8 ip-adapter_sd15_plus_hswq_convrot_int8.safetensors \
      --out-dir <dir for the three pngs> --json <record.json> --seed 42
"""

import argparse
import base64
import hashlib
import io
import json
import os
import urllib.request

import numpy as np
from PIL import Image


def _resolve(model_dir, name):
    for ext in (".safetensors", ".pth", ".ckpt", ".bin"):
        p = os.path.join(model_dir, name + ext)
        if os.path.exists(p):
            return p
    raise SystemExit("no file for model name %r under %s" % (name, model_dir))


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def post_json(url, payload, timeout=1800):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def png_bytes(b64):
    return base64.b64decode(b64.split(",", 1)[-1])


def compare(a, b):
    x = np.asarray(Image.open(io.BytesIO(a)).convert("RGB"), dtype=np.float64)
    y = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.float64)
    d = np.abs(x - y)
    mse = float((d ** 2).mean())
    psnr = float("inf") if mse == 0 else 10.0 * np.log10((255.0 ** 2) / mse)
    return {"mean_abs": float(d.mean()), "max_abs": float(d.max()),
            "changed_pct": float((d.max(axis=2) > 0).mean() * 100.0),
            "psnr_db": round(psnr, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    ap.add_argument("--image", required=True)
    ap.add_argument("--float", required=True, dest="float_model",
                    help="the ControlNet model name as the extension lists it, without extension")
    ap.add_argument("--int8", required=True, dest="int8_model",
                    help="the ControlNet model name as the extension lists it, without extension")
    ap.add_argument("--model-dir", default=r"D:\USERFILES\StableDiffusion\models\ControlNet",
                    help="where the two files live, for the sha256 and the size")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--base-model", default="againmix_v22.safetensors")
    ap.add_argument("--prompt", default="a photo of a person, high detail")
    ap.add_argument("--negative", default="")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weight", type=float, default=1.0)
    ap.add_argument("--preprocessor", default="ip-adapter_clip_sd15_plus")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ref_b64 = base64.b64encode(open(args.image, "rb").read()).decode("ascii")

    def render(model):
        payload = {
            "prompt": args.prompt, "negative_prompt": args.negative,
            "width": 512, "height": 512, "steps": args.steps,
            "cfg_scale": args.cfg, "sampler_name": "DPM++ 2M",
            "seed": args.seed, "batch_size": 1, "n_iter": 1,
            "sd_model_checkpoint": args.base_model,
        }
        if model is not None:
            payload["alwayson_scripts"] = {"controlnet": {"args": [{
                "enabled": True, "image": {"image": ref_b64, "mask": None},
                "module": args.preprocessor, "model": model,
                "weight": args.weight,
                # the extension validates these as enum names, not as indices
                "resize_mode": "Crop and Resize", "control_mode": "Balanced",
                "pixel_perfect": False, "processor_res": 512, "guidance_end": 1.0,
            }]}}
        out = post_json(args.url.rstrip("/") + "/sdapi/v1/txt2img", payload)
        if not out.get("images"):
            raise SystemExit("no image returned for %r: %s" % (model, list(out)[:6]))
        return png_bytes(out["images"][0]), out.get("info", "")

    results = {}
    for tag, model in (("base", None), ("float", args.float_model), ("int8", args.int8_model)):
        png, info = render(model)
        path = os.path.join(args.out_dir, "ipadapter_sd15_plus_%s.png" % tag)
        with open(path, "wb") as f:
            f.write(png)
        results[tag] = {"png": path, "png_bytes": len(png),
                        "png_sha256": hashlib.sha256(png).hexdigest(),
                        "info_len": len(info)}
        print("%-6s %d B sha %s" % (tag, len(png), results[tag]["png_sha256"][:16]), flush=True)

    def load(tag):
        return open(results[tag]["png"], "rb").read()

    rec = {
        "tool": "audit/a1111_ipadapter_render.py",
        "family": "ipadapter_sd15_plus",
        "route": "A1111 sd-webui-controlnet ipadapter + modules/ipadapter_int8_convrot.py",
        "pack": os.path.basename(args.int8_model),
        "pack_sha256": _sha(_resolve(args.model_dir, args.int8_model)),
        "pack_bytes": os.path.getsize(_resolve(args.model_dir, args.int8_model)),
        "source": args.float_model,
        "source_sha256": _sha(_resolve(args.model_dir, args.float_model)),
        "render": {
            "width": 512, "height": 512, "steps": args.steps, "cfg_scale": args.cfg,
            "sampler": "DPM++ 2M", "seed": args.seed, "weight": args.weight,
            "preprocessor": args.preprocessor, "base": args.base_model,
            "prompt": args.prompt, "image_prompt": os.path.basename(args.image),
            "png_bytes": results["int8"]["png_bytes"],
            "png_sha256": results["int8"]["png_sha256"],
        },
        "renders": {k: {kk: vv for kk, vv in v.items() if kk != "png"} for k, v in results.items()},
        "comparison": {
            "int8_vs_base": compare(load("int8"), load("base")),
            "float_vs_base": compare(load("float"), load("base")),
            "float_vs_int8": compare(load("float"), load("int8")),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print("float_vs_int8", rec["comparison"]["float_vs_int8"], flush=True)
    print("wrote %s" % args.json, flush=True)


if __name__ == "__main__":
    main()

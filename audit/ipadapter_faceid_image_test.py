"""CN-4 for the IP-Adapter FaceID pack: render through the node's own path.

The adapter is attached the way the node attaches it - `ipadapter_model_loader` for the weights,
`insightface_loader` for the face embedding, `ipadapter_execute` for the attachment - and the
sampler runs with a fixed seed. What is recorded is the latent (shape, finite), the INT8 kernel
call count, the PNG bytes and hash, and the wall clock. The image itself is written out so it can
be looked at rather than inferred from a cosine.

A pack whose `comfy_quant` the loader does not interpret produces raw int8 codes in a float
parameter. That is measured, not hidden: the same run is repeated with the source file and with
the pack, and both are written, so the two can be compared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weight", type=float, default=0.8)
    ap.add_argument("--clipvision", required=True,
                    help="the CLIP vision the node attaches; a FaceID adapter still reaches "
                         "encode_image_masked on this code path")
    ap.add_argument("--provider", default="CUDA")
    a = ap.parse_args()

    sys.path.insert(0, os.path.abspath(a.comfy))
    sys.path.insert(0, os.path.join(os.path.abspath(a.comfy), "custom_nodes"))
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    import comfy.sd
    import comfy.sample
    import comfy.utils
    import comfy.clip_vision
    from comfyui_ipadapter_plus.utils import ipadapter_model_loader, insightface_loader
    from comfyui_ipadapter_plus.IPAdapterPlus import ipadapter_execute
    import comfy_kitchen.backends.cuda as ckc
    sys.argv = saved
    from PIL import Image

    os.makedirs(a.out_dir, exist_ok=True)

    counts = {"int8": 0}
    orig = getattr(ckc, "int8_linear", None)
    if orig is not None:
        def counted(*args, **kw):
            counts["int8"] += 1
            return orig(*args, **kw)
        ckc.int8_linear = counted

    app = insightface_loader(a.provider)
    face_img = np.asarray(Image.open(a.image).convert("RGB"))
    faces = app.get(face_img)
    if not faces:
        raise SystemExit("no face detected in " + a.image)
    emb = torch.from_numpy(faces[0].normed_embedding).float()
    print("[face] detected, det_score %.4f, embedding %s finite %s"
          % (float(faces[0].det_score), tuple(emb.shape), bool(torch.isfinite(emb).all())), flush=True)

    cv_obj = comfy.clip_vision.load(os.path.abspath(a.clipvision))
    print("[clipvision] %s" % os.path.basename(a.clipvision), flush=True)
    model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
        os.path.abspath(a.ckpt), output_vae=True, output_clip=True)
    img_pil = Image.open(a.image).convert("RGB")
    img_t = torch.from_numpy(np.asarray(img_pil).astype(np.float32) / 255.0)[None]

    POS = "a woman in a red coat standing in a city street, detailed, natural light"
    NEG = "blurry, low quality, watermark"
    pos = clip.encode_from_tokens_scheduled(clip.tokenize(POS))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(NEG))

    arms = []
    for label, path in (("source", a.source), ("pack", a.pack)):
        ipa = ipadapter_model_loader(os.path.abspath(path))
        # FaceID takes the FaceAnalysis model, not a precomputed embedding: the node calls
        # insightface.get(image) itself and raises if no face is found.
        ipa["insightface"] = {"model": app}
        counts["int8"] = 0
        t0 = time.time()
        m = ipadapter_execute(model.clone(), ipa, cv_obj, image=img_t, insightface=app,
                              weight=a.weight, weight_type="linear", combine_embeds="concat",
                              start_at=0.0, end_at=1.0, embeds_scaling="V only")[0]
        lat = torch.zeros([1, 4, a.height // 8, a.width // 8])
        noise = comfy.sample.prepare_noise(lat, a.seed)
        out = comfy.sample.sample(m, noise, a.steps, a.cfg, "dpmpp_2m", "karras", pos, neg, lat,
                                  denoise=1.0)
        dec = vae.decode(out)
        arr = (dec[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        png = os.path.join(a.out_dir, "ipadapter_faceid_%s_%dx%d.png" % (label, a.width, a.height))
        Image.fromarray(arr).save(png)
        dt = time.time() - t0
        rec = {
            "label": label, "file": os.path.basename(path),
            "latent_shape": list(out.shape), "latent_finite": bool(torch.isfinite(out).all()),
            "latent_nan": int(torch.isnan(out).sum()), "latent_inf": int(torch.isinf(out).sum()),
            "int8_linear_calls": counts["int8"],
            "png": os.path.basename(png), "png_bytes": os.path.getsize(png),
            "png_sha256": hashlib.sha256(open(png, "rb").read()).hexdigest().upper(),
            "pixel_mean": round(float(arr.mean()), 2), "pixel_std": round(float(arr.std()), 2),
            "wall_s": round(dt, 1),
        }
        arms.append(rec)
        print("[%s] latent %s finite %s nan %d inf %d | INT8 %d | PNG %d B | %.1f s"
              % (label, rec["latent_shape"], rec["latent_finite"], rec["latent_nan"],
                 rec["latent_inf"], rec["int8_linear_calls"], rec["png_bytes"], dt), flush=True)
        del m

    if orig is not None:
        ckc.int8_linear = orig

    pa = np.asarray(Image.open(os.path.join(a.out_dir, arms[0]["png"])).convert("RGB")).astype(np.int16)
    pb = np.asarray(Image.open(os.path.join(a.out_dir, arms[1]["png"])).convert("RGB")).astype(np.int16)
    diff = np.abs(pa - pb)
    rec = {
        "tool": "audit/ipadapter_faceid_image_test.py",
        "family": "faceid",
        "pack": os.path.basename(a.pack),
        "route": "comfyui_ipadapter_plus: ipadapter_model_loader -> ipadapter_execute -> comfy.sample",
        "render": {"width": a.width, "height": a.height, "steps": a.steps, "cfg": a.cfg,
                   "sampler": "dpmpp_2m", "scheduler": "karras", "seed": a.seed,
                   "weight": a.weight, "ipadapter": os.path.basename(a.pack),
                   "checkpoint": os.path.basename(a.ckpt)},
        "face": {"detected": 1, "det_score": round(float(faces[0].det_score), 4),
                 "embedding_shape": list(emb.shape)},
        "arms": arms,
        "source_vs_pack": {"mean_abs": round(float(diff.mean()), 2),
                           "max_abs": int(diff.max()),
                           "changed_pct": round(float((diff > 0).mean() * 100), 2)},
    }
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print("[source vs pack] mean_abs %.2f max_abs %d changed %.2f%%"
          % (rec["source_vs_pack"]["mean_abs"], rec["source_vs_pack"]["max_abs"],
             rec["source_vs_pack"]["changed_pct"]), flush=True)
    print("wrote", a.json, flush=True)


if __name__ == "__main__":
    main()

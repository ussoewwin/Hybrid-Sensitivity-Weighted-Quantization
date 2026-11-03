# -*- coding: utf-8 -*-
"""The vision acceptance check, with the baseline on the SAME ops path.

The stock CLIPVisionLoader builds the vision tower with `comfy.ops.manual_cast`,
while a quantized pack is built with `mixed_precision_ops`. For some models
these two paths are NOT numerically equivalent - measured on the CLIP-ViT-H-14
source: the same weights through the two paths give a cosine of 0.94-0.96, and
restoring every quantized layer of a pack to FP16 does not move the number.

So the baseline has to be built through the same ops class, otherwise the
comparison measures the ops path, not the quantization.
"""
import json
import os
import statistics
import sys

import torch

COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import comfy.utils
import comfy.clip_vision as cv
import comfy.model_management
import comfy.ops as O
from comfy.quant_ops import QUANT_ALGOS

DT = torch.float16


def prep(sd, cls_dim):
    vsd = {k: v for k, v in sd.items() if k.startswith("vision_model.")}
    if "vision_model.embeddings.class_embedding" not in vsd:
        vsd["vision_model.embeddings.class_embedding"] = torch.zeros(cls_dim)
    return vsd


def class_dim(sd):
    for k, v in sd.items():
        if k.endswith("embeddings.class_embedding"):
            return int(v.shape[0])
    return 768


def pick_config(sd):
    """The config the STOCK loader picks for this checkpoint.

    `comfy.clip_vision.load_clipvision_from_sd` chooses it from the
    checkpoint's own shape, so the same rule is used here: the baseline and
    the pack must both be built with the config the loader would pick.
    """
    import re
    src = open(os.path.join(COMFY, "comfy", "clip_vision.py"), encoding="utf-8").read()
    i = src.find("def load_clipvision_from_sd")
    seg = src[i:i + 4000]
    # the loader's own rules
    if "vision_model.encoder.layers.47.layer_norm1.weight" in sd:
        return os.path.join(COMFY, "comfy", "clip_vision_config_g.json")
    if "vision_model.encoder.layers.30.layer_norm1.weight" in sd:
        return os.path.join(COMFY, "comfy", "clip_vision_config_h.json")
    if "vision_model.encoder.layers.22.layer_norm1.weight" in sd:
        emb = sd.get("vision_model.embeddings.position_embedding.weight")
        n = int(emb.shape[0]) if emb is not None else 0
        if n == 729:
            return os.path.join(COMFY, "comfy", "clip_vision_siglip_384.json")
    # the fallback: let the stock loader decide
    return None


def build(jc, ops, sd, cls_dim):
    """Build the vision model the way the pack is loaded."""
    class CVQ(cv.ClipVisionModel):
        def __init__(self, jc, ops):
            config = json.load(open(jc))
            self.image_size = config.get("image_size", 224)
            self.image_mean = config.get("image_mean", [0.5, 0.5, 0.5])
            self.image_std = config.get("image_std", [0.5, 0.5, 0.5])
            self.model_type = config.get("model_type", "clip_vision_model")
            self.config = config.copy()
            mc = cv.IMAGE_ENCODERS.get(self.model_type)
            self.return_all_hidden_states = False
            self.load_device = comfy.model_management.text_encoder_device()
            off = comfy.model_management.text_encoder_offload_device()
            self.dtype = comfy.model_management.text_encoder_dtype(self.load_device)
            self.model = mc(config, self.dtype, off, ops)
            self.model.eval()
            self.patcher = comfy.model_patcher.CoreModelPatcher(
                self.model, load_device=self.load_device, offload_device=off)
            self.naf = None
    v = CVQ(jc, ops)
    v.load_sd(prep(sd, cls_dim))
    return v


def embeds(v, imgs):
    with torch.no_grad():
        out = v.encode_image(imgs, crop=False)
    if hasattr(out, "image_embeds") and torch.is_tensor(out.image_embeds):
        return out.image_embeds.detach().float().reshape(-1).cpu()
    if isinstance(out, dict):
        return torch.cat([out[k].detach().float().reshape(-1).cpu()
                          for k in sorted(out) if torch.is_tensor(out[k])])
    if torch.is_tensor(out):
        return out.detach().float().reshape(-1).cpu()
    return torch.cat([x.detach().float().reshape(-1).cpu() for x in out if torch.is_tensor(x)])


def main():
    src = sys.argv[1]
    packs = sys.argv[2:]
    n_items = int(os.environ.get("VISION_N", "256"))

    sd0 = comfy.utils.load_torch_file(src, safe_load=True)
    d = class_dim(sd0)
    jc = pick_config(sd0)
    if jc is None:
        import comfy.clip_vision as _cv
        _saved = sys.argv
        sys.argv = [_saved[0]]
        try:
            _v = _cv.load_clipvision_from_sd(prep(sd0, d))
        finally:
            sys.argv = _saved
        import json as _json
        jc = os.path.join(COMFY, "comfy", "clip_vision_config_vitl.json")
        # the stock loader's config, recovered from the model's own attributes
        cfg_name = getattr(_v, "model_type", "clip_vision_model")
        print("    the stock loader's model type: %s" % cfg_name, flush=True)
    ops = O.mixed_precision_ops({"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]},
                                torch.float16, full_precision_mm=False)

    print("[1] the FP16 baseline (the same ops path) ...", flush=True)
    v16 = build(jc, ops, sd0, d)
    size = int(getattr(v16, "image_size", 224))

    def imgs(seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        return (torch.rand(1, 3, size, size, generator=g) * 2 - 1).to("cuda", dtype=DT)

    results = []
    for p in packs:
        print("[2] %s" % os.path.basename(p), flush=True)
        psd, meta = comfy.utils.load_torch_file(p, safe_load=True, return_metadata=True)
        v8 = build(jc, ops, psd, d)
        ck = O.quant_ops.ck
        orig = ck.int8_linear
        cnt = [0]

        def counted(*a, **kw):
            cnt[0] += 1
            return orig(*a, **kw)

        ck.int8_linear = counted
        try:
            with torch.no_grad():
                v8.encode_image(imgs(42), crop=False)
        finally:
            ck.int8_linear = orig
        cs = []
        for i in range(n_items):
            im = imgs(1000 + i)
            a = embeds(v16, im)
            b = embeds(v8, im)
            cs.append(torch.nn.functional.cosine_similarity(
                a.reshape(1, -1), b.reshape(1, -1)).item())
        cs.sort()
        m = len(cs)
        rec = {"pack": os.path.basename(p),
               "file_mb": os.path.getsize(p) / 1024 ** 2,
               "int8_linear_calls": cnt[0],
               "mean": statistics.mean(cs), "min": cs[0],
               "p5": cs[max(0, int(m * 0.05) - 1)], "p50": cs[m // 2],
               "n": m}
        rec["pass"] = bool(rec["mean"] >= 0.99 and rec["min"] >= 0.98)
        results.append(rec)
        print("    int8_linear=%d  mean=%.6f min=%.6f p5=%.6f p50=%.6f  size=%.1f MB  %s" % (
            rec["int8_linear_calls"], rec["mean"], rec["min"], rec["p5"], rec["p50"],
            rec["file_mb"], "PASS" if rec["pass"] else "FAIL"), flush=True)
        del v8
        torch.cuda.empty_cache()

    print()
    print("%-56s %8s %10s %10s %10s %10s" % ("pack", "int8", "mean", "min", "p5", "MB"))
    for r in results:
        print("%-56s %8d %10.6f %10.6f %10.6f %10.1f" % (
            r["pack"][:56], r["int8_linear_calls"], r["mean"], r["min"], r["p5"], r["file_mb"]))
    print("JSON " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""CLIP-ViT-bigG-14: the candidate packs, measured against one baseline.

The question is which FP16-kept set to ship. Rather than measuring all 288
layers one at a time (a pack per layer, far too slow), the candidate sets are
built directly and measured:

- the all-INT8 set (the corrected drift protects 0 layers),
- the shipped pack's set (the old drift protected 13),
- and a few in between.

The kept layers are the ones with the largest **corrected** drift, so the
frontier follows the measurement rather than a guess.

Everything is measured in one process, one model at a time: the FP16
baseline is built, its reference embeddings are taken, it is freed, and then
each pack is built and measured in turn. The baseline and the pack share the
same ops class, because the stock `manual_cast` path and the
`mixed_precision_ops` path are not numerically equivalent for this model.

Usage:
    python verify_bigg_vision.py <source.safetensors> [<pack.safetensors> ...]
"""
import os
import sys
import json
import types
import statistics
import importlib

import torch

COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
NODE = os.path.join(COMFY, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools")
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
os.chdir(COMFY)

import folder_paths  # noqa: E402
import server as comfy_server  # noqa: E402
import comfy.utils  # noqa: E402
import comfy.clip_vision as cv  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.ops as O  # noqa: E402
from comfy.quant_ops import QUANT_ALGOS  # noqa: E402

if not hasattr(comfy_server.PromptServer, "instance"):
    comfy_server.PromptServer.instance = None

_spec = importlib.util.spec_from_file_location(
    "hswq_pkg_init", os.path.join(NODE, "__init__.py"), submodule_search_locations=[NODE])
_mod = importlib.util.module_from_spec(_spec)
sys.modules["hswq_pkg_init"] = _mod
_spec.loader.exec_module(_mod)

DT = torch.float16
N = int(os.environ.get("BIGG_N", "64"))


def prep(sd, d):
    v = {k: x for k, x in sd.items() if k.startswith("vision_model.")}
    if "vision_model.embeddings.class_embedding" not in v:
        v["vision_model.embeddings.class_embedding"] = torch.zeros(d)
    return v


def class_dim(sd):
    for k, v in sd.items():
        if k.endswith("embeddings.class_embedding"):
            return int(v.shape[0])
    return 768


def build(jc, ops, sd, d):
    class CVQ(cv.ClipVisionModel):
        def __init__(self, jc, ops):
            c = json.load(open(jc))
            self.image_size = c.get("image_size", 224)
            self.image_mean = c.get("image_mean", [0.5, 0.5, 0.5])
            self.image_std = c.get("image_std", [0.5, 0.5, 0.5])
            self.model_type = c.get("model_type", "clip_vision_model")
            self.config = c.copy()
            mc = cv.IMAGE_ENCODERS.get(self.model_type)
            self.return_all_hidden_states = False
            self.load_device = comfy.model_management.text_encoder_device()
            off = comfy.model_management.text_encoder_offload_device()
            self.dtype = comfy.model_management.text_encoder_dtype(self.load_device)
            self.model = mc(c, self.dtype, off, ops)
            self.model.eval()
            self.patcher = comfy.model_patcher.CoreModelPatcher(
                self.model, load_device=self.load_device, offload_device=off)
            self.naf = None
    v = CVQ(jc, ops)
    v.load_sd(prep(sd, d))
    return v


def jc_for():
    p = os.path.join(COMFY, "comfy", "clip_vision_config_g.json")
    if os.path.isfile(p):
        return p
    return os.path.join(COMFY, "comfy", "clip_vision_config_vitl.json")


def embeds(v, im):
    with torch.no_grad():
        out = v.encode_image(im, crop=False)
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
    ops = O.mixed_precision_ops({"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]},
                                DT, full_precision_mm=False)
    sd0 = comfy.utils.load_torch_file(src, safe_load=True)
    d = class_dim(sd0)
    jc = jc_for()

    print("[1] the FP16 baseline (the same ops path) ...", flush=True)
    v16 = build(jc, ops, sd0, d)
    size = int(getattr(v16, "image_size", 224))

    def imgs(seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        return (torch.rand(1, 3, size, size, generator=g) * 2 - 1).to("cuda", dtype=DT)

    ref = [embeds(v16, imgs(1000 + s)) for s in range(N)]
    print("    the reference: %s finite=%s" % (
        tuple(ref[0].shape), bool(torch.isfinite(ref[0]).all())), flush=True)
    del v16, sd0
    torch.cuda.empty_cache()

    results = []
    for p in packs:
        print("[2] %s" % os.path.basename(p), flush=True)
        sd = comfy.utils.load_torch_file(p, safe_load=True)
        nq = len([k for k in sd if k.endswith(".comfy_quant")])
        v8 = build(jc, ops, sd, d)
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
        fin = True
        for s in range(N):
            f = embeds(v8, imgs(1000 + s))
            if not torch.isfinite(f).all():
                fin = False
            cs.append(torch.nn.functional.cosine_similarity(
                ref[s].reshape(1, -1), f.reshape(1, -1)).item())
        cs.sort()
        m = len(cs)
        rec = {"pack": os.path.basename(p),
               "file_mb": os.path.getsize(p) / 1024 ** 2,
               "quantized_layers": nq, "int8_linear_calls": cnt[0],
               "mean": statistics.mean(cs), "min": cs[0],
               "p5": cs[max(0, int(m * 0.05) - 1)], "p50": cs[m // 2],
               "finite": fin, "n": m}
        rec["pass"] = bool(rec["mean"] >= 0.99 and rec["min"] >= 0.98 and fin)
        results.append(rec)
        print("    int8_linear=%d finite=%s  mean=%.6f min=%.6f p5=%.6f p50=%.6f  size=%.1f MB  %s" % (
            cnt[0], fin, rec["mean"], rec["min"], rec["p5"], rec["p50"], rec["file_mb"],
            "PASS" if rec["pass"] else "FAIL"), flush=True)
        del v8, sd
        torch.cuda.empty_cache()

    print()
    print("%-56s %7s %10s %10s %10s %10s" % ("pack", "int8", "mean", "min", "p5", "MB"))
    for r in results:
        print("%-56s %7d %10.6f %10.6f %10.6f %10.1f" % (
            r["pack"][:56], r["int8_linear_calls"], r["mean"], r["min"], r["p5"], r["file_mb"]))
    print("JSON " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()

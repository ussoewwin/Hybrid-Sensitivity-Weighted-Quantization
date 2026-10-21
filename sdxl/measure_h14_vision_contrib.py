# -*- coding: utf-8 -*-
"""CLIP-ViT-H-14: the aggregate contribution of each layer.

The per-layer drift is a screen, not the selection criterion: it measures the
WEIGHT error only, while the pack's real path also quantizes the ACTIVATION for
that layer, and that is where most of the error comes from.

This measures the OUTPUT: start from the FP16 weights, replace ONE layer by its
ConvRot INT8 reconstruction (rotate -> row-wise quantize -> dequantize ->
un-rotate), and compare the tower's embedding against the FP16 output. A layer
whose quantization barely moves the output can be quantized; a layer that moves
it must stay FP16. The gain from keeping a layer is what the selection sorts on.

The baseline is built with the SAME ops class the pack is loaded with
(`mixed_precision_ops`), because stock `manual_cast` and `mixed_precision_ops`
are not numerically equivalent for this model.
"""
import argparse
import json
import math
import os
import statistics
import sys

COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
sys.path.insert(0, COMFY)
os.chdir(COMFY)
import torch  # noqa: E402
import comfy.utils  # noqa: E402
import comfy.clip_vision as cv  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.ops as O  # noqa: E402
from comfy.quant_ops import QUANT_ALGOS  # noqa: E402

DT = torch.float16


def had(sz):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    h = h4
    while h.shape[0] < sz:
        h = torch.kron(h, h4)
    return h / math.sqrt(sz)


_H = {}


def quant_dequant(w, gs):
    """The ConvRot INT8 reconstruction of a weight, in the original frame."""
    if gs not in _H:
        _H[gs] = had(gs)
    H = _H[gs].to(device=w.device, dtype=w.dtype)
    of, inf = w.shape
    g = inf // gs
    r = torch.matmul(w.view(of, g, gs), H.T).reshape(w.shape)
    a = r.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    s = a / 127.0
    q = (r / s).round().clamp(-127, 127)
    back = (q * s).view(of, g, gs)
    return torch.matmul(back, H).reshape(w.shape)


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--groupsize", type=int, default=256)
    a = ap.parse_args()

    ops = O.mixed_precision_ops({"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]},
                                DT, full_precision_mm=False)
    sd0 = comfy.utils.load_torch_file(a.src, safe_load=True)
    d = class_dim(sd0)
    sd0 = {k: (v.to(DT) if v.is_floating_point() else v) for k, v in sd0.items()}

    jc = os.path.join(COMFY, "comfy", "clip_vision_config_h.json")
    if not os.path.isfile(jc):
        jc = os.path.join(COMFY, "comfy", "clip_vision_config_vitl.json")

    v = build(jc, ops, sd0, d)
    size = int(getattr(v, "image_size", 224))
    print("[1] the model built; the image size %d" % size, flush=True)

    def imgs(seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        return (torch.rand(1, 3, size, size, generator=g) * 2 - 1).to("cuda", dtype=DT)

    ref = [embeds(v, imgs(1000 + s)) for s in range(a.n)]
    print("[1] the FP16 baseline: %s" % (tuple(ref[0].shape),), flush=True)

    net = getattr(v, "model", v)
    inner = getattr(net, "vision_model", net)
    cands = [n for n, mm in inner.named_modules()
             if type(mm).__name__ == "Linear" and getattr(mm, "weight", None) is not None]
    print("[2] the runtime Linear modules: %d" % len(cands), flush=True)

    contrib = {}
    for i, name in enumerate(cands):
        mm = dict(inner.named_modules())[name]
        orig = mm.weight.data.clone()
        w = orig.float()
        if w.ndim != 2 or int(w.shape[1]) % a.groupsize != 0 or int(w.shape[0]) % 4 != 0:
            contrib[name] = None
            continue
        rq = quant_dequant(w, a.groupsize)
        mm.weight.data = rq.to(orig.dtype)
        try:
            cs = []
            for s in range(a.n):
                f = embeds(v, imgs(1000 + s))
                cs.append(torch.nn.functional.cosine_similarity(
                    ref[s].reshape(1, -1), f.reshape(1, -1)).item())
            contrib[name] = 1.0 - statistics.mean(cs)
        finally:
            mm.weight.data = orig
        if (i + 1) % 20 == 0:
            print("   [%d/%d] %s" % (i + 1, len(cands), name), flush=True)

    del v
    torch.cuda.empty_cache()

    json.dump({"impacts": contrib, "n": a.n, "seed": a.seed,
               "baseline": "mixed_precision_ops FP16"},
              open(a.out, "w", encoding="utf-8"), indent=1, sort_keys=True)
    vals = sorted(v for v in contrib.values() if v is not None)
    print()
    print("saved: %s" % a.out)
    print("== the ranked contribution (1 - cos, higher = must protect) ==")
    for n, val in sorted(((k, x) for k, x in contrib.items() if x is not None),
                         key=lambda kv: -kv[1])[:20]:
        print("   %.4e  %s" % (val, n))


if __name__ == "__main__":
    main()

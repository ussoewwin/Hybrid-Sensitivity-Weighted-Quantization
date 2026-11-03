# -*- coding: utf-8 -*-
"""sigclip_vision_patch14_384: the aggregate contribution of each layer.

The per-layer drift is a screen, not the selection criterion: it measures the
WEIGHT error only, while the pack's real path also quantizes the ACTIVATION for
that layer, and that is where most of the error comes from.

This measures the OUTPUT: hold the weights, replace ONE layer by its ConvRot
INT8 reconstruction, and compare the image embedding against the unmodified
output. The layer's contribution is 1 - cos.

The baseline is built with the SAME ops class the pack is loaded with
(`mixed_precision_ops`), and with the config the stock loader picks for this
checkpoint (`clip_vision_siglip_384.json`, image size 384).
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
    return torch.matmul((q * s).view(of, g, gs), H).reshape(w.shape)


def group_size_for(n, preferred):
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and (gs & (gs - 1)) == 0 and (math.log(gs, 4) % 1 == 0):
            return gs
        gs //= 4
    return None


def prep(sd, d):
    v = {k: x for k, x in sd.items() if k.startswith("vision_model.")}
    if "vision_model.embeddings.class_embedding" not in v:
        v["vision_model.embeddings.class_embedding"] = torch.zeros(d)
    return v


def class_dim(sd):
    for k, v in sd.items():
        if k.endswith("embeddings.class_embedding"):
            return int(v.shape[0])
    return 1152


def build(jc, ops, sd, d):
    class CVQ(cv.ClipVisionModel):
        def __init__(self, jc, ops):
            c = json.load(open(jc))
            self.image_size = c.get("image_size", 384)
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
    ap.add_argument("--groupsize", type=int, default=64)
    a = ap.parse_args()

    ops = O.mixed_precision_ops({"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]},
                                DT, full_precision_mm=False)
    sd0 = comfy.utils.load_torch_file(a.src, safe_load=True)
    d = class_dim(sd0)
    sd0 = {k: (v.to(DT) if v.is_floating_point() else v) for k, v in sd0.items()}

    jc = os.path.join(COMFY, "comfy", "clip_vision_siglip_384.json")
    if not os.path.isfile(jc):
        raise SystemExit("the siglip 384 config was not found: %s" % jc)

    v = build(jc, ops, sd0, d)
    size = int(getattr(v, "image_size", 384))
    print("[1] the model built; the image size %d" % size, flush=True)

    def imgs(seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        return (torch.rand(1, 3, size, size, generator=g) * 2 - 1).to("cuda", dtype=DT)

    ref = [embeds(v, imgs(1000 + s)) for s in range(a.n)]
    print("[1] the FP16 baseline: %s" % (tuple(ref[0].shape),), flush=True)

    net = getattr(v, "model", v)
    inner = getattr(net, "vision_model", net)
    mods = dict(inner.named_modules())
    cands = [n for n, mm in inner.named_modules()
             if type(mm).__name__ == "Linear" and getattr(mm, "weight", None) is not None]
    print("[2] the runtime Linear modules: %d" % len(cands), flush=True)

    contrib = {}
    for i, name in enumerate(cands):
        mm = mods[name]
        orig = mm.weight.data.clone()
        w = orig.float()
        gs = group_size_for(int(w.shape[1]), a.groupsize) if w.ndim == 2 else None
        if w.ndim != 2 or gs is None or int(w.shape[0]) % 4 != 0:
            contrib[name] = None
            continue
        mm.weight.data = quant_dequant(w, gs).to(orig.dtype)
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

    json.dump({"impacts": contrib, "n": a.n, "seed": a.seed},
              open(a.out, "w", encoding="utf-8"), indent=1, sort_keys=True)
    print()
    print("saved: %s" % a.out)
    print("== the ranked contribution (1 - cos) ==")
    for n, val in sorted(((k, x) for k, x in contrib.items() if x is not None),
                         key=lambda kv: -kv[1])[:20]:
        print("   %.4e  %s" % (val, n))


if __name__ == "__main__":
    main()

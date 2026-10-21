# -*- coding: utf-8 -*-
"""Josiefied-Qwen3-0.6B: the aggregate contribution of each layer.

The per-layer drift is a screen, not the selection criterion: it measures the
WEIGHT error only, while the pack's real path also quantizes the ACTIVATION for
that layer, and that is where most of the error comes from.

This measures the OUTPUT: build the encoder the way ComfyUI builds it
(llama_detect -> anima.te), hold the FP16/BF16 weights, replace ONE layer by
its ConvRot INT8 reconstruction (rotate -> row-wise quantize -> dequantize ->
un-rotate), and compare the encoder's hidden state against the unmodified
output. The layer's contribution is 1 - cos. A layer whose quantization barely
moves the output can be quantized; a layer that moves it must stay protected.
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
import comfy.text_encoders.anima  # noqa: E402
import comfy.text_encoders.hunyuan_video  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


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


def build(path):
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    detect = comfy.text_encoders.hunyuan_video.llama_detect(sd)
    te = comfy.text_encoders.anima.te(**detect)
    clip = te(device=DEV)
    clip.load_sd(sd)
    inner = getattr(clip, getattr(clip, "clip"))
    return clip, getattr(inner, "transformer", inner)


def hidden(clip, ids):
    tr = getattr(getattr(clip, getattr(clip, "clip")), "transformer", None)
    with torch.no_grad():
        out = tr(ids)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.detach().float().reshape(-1).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--groupsize", type=int, default=256)
    a = ap.parse_args()

    clip, tr = build(a.src)
    print("[1] the model built", flush=True)

    g = torch.Generator(device="cpu").manual_seed(a.seed)
    batches = [torch.randint(0, 150000, (1, 16), generator=g).to(DEV) for _ in range(a.n)]
    ref = [hidden(clip, b) for b in batches]
    print("[1] the baseline: %s" % (tuple(ref[0].shape),), flush=True)

    cands = [n for n, mm in tr.named_modules()
             if type(mm).__name__ == "Linear" and getattr(mm, "weight", None) is not None]
    print("[2] the Linear modules: %d" % len(cands), flush=True)
    mods = dict(tr.named_modules())

    contrib = {}
    for i, name in enumerate(cands):
        mm = mods[name]
        orig = mm.weight.data.clone()
        w = orig.float()
        if w.ndim != 2 or int(w.shape[1]) % a.groupsize != 0 or int(w.shape[0]) % 4 != 0:
            contrib[name] = None
            continue
        rq = quant_dequant(w, a.groupsize)
        mm.weight.data = rq.to(orig.dtype)
        try:
            cs = []
            for j, b in enumerate(batches):
                f = hidden(clip, b)
                cs.append(torch.nn.functional.cosine_similarity(
                    ref[j].reshape(1, -1), f.reshape(1, -1)).item())
            contrib[name] = 1.0 - statistics.mean(cs)
        finally:
            mm.weight.data = orig
        if (i + 1) % 20 == 0:
            print("   [%d/%d] %s" % (i + 1, len(cands), name), flush=True)

    del clip
    if torch.cuda.is_available():
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

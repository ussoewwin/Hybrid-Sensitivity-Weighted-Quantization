# -*- coding: utf-8 -*-
"""The FLUX ControlNet's production `y`: the CLIP-L pooled output, 768-wide.

ComfyUI's `Flux.encode_adm` returns `kwargs["pooled_output"]`, and the FLUX
clip is loaded with `clip_type=FLUX`. The pooled tensor is saved so the
measurement can use the production condition instead of a stand-in.

The local FLUX clip (clip_l + t5) measures: absmax 8.82, absmean 0.738,
std 1.07.

Usage:
    python build_flux_prod_y.py
"""
import os
import sys
import torch

COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
sys.path.insert(0, COMFY)
os.chdir(COMFY)
import comfy.utils  # noqa: E402
import comfy.sd as _sd  # noqa: E402

CLIP_L = "D:/USERFILES/ComfyUI/ComfyUI/models/clip/clip_l_hswq_convrot_int8.safetensors"
T5 = "D:/USERFILES/ComfyUI/ComfyUI/models/clip/flan_t5_xxl_hswq_convrot_int8.safetensors"
OUT = r"D:\USERFILES\GitHub\hswq\impact\flux_pooled_sample.pt"


def main():
    print("== the FLUX clip (clip_l + t5) ==", flush=True)
    clip = _sd.load_clip([CLIP_L, T5], clip_type=_sd.CLIPType.FLUX)
    tokens = clip.tokenize("a photograph of a cat sitting on a table")
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    print("   the cond: %s | the pooled: %s %s" % (
        tuple(cond.shape), tuple(pooled.shape), pooled.dtype), flush=True)
    p = pooled.detach().float()
    print("   the pooled: the absmax=%.4f the absmean=%.4f the std=%.4f the RMS=%.4f" % (
        float(p.abs().max()), float(p.abs().mean()), float(p.std()),
        float(p.pow(2).mean().sqrt())), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save(pooled.detach().cpu(), OUT)
    print("   saved: %s" % OUT, flush=True)


if __name__ == "__main__":
    main()

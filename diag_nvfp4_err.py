#!/usr/bin/env python
"""Diagnose per-layer NVFP4 (comfy_kitchen E2M1) roundtrip error on a Z-Image checkpoint."""
import torch
import safetensors.torch as st
from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout as NVFP4

BASE = r"D:\USERFILES\ComfyUI\ComfyUI\models\unet\darkBeast30BF16INT8_dbzit9DIMRclaw.safetensors"


def rel_mse(a, b):
    a = a.float().reshape(a.shape[0], -1)
    b = b.float().reshape(b.shape[0], -1)
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sd = st.load_file(BASE, device="cpu")
    rows = []
    n2d = 0
    for k, w in sd.items():
        if not k.endswith(".weight"):
            continue
        if w.ndim != 2 or w.shape[0] == 1 or w.shape[1] == 1:
            continue
        n2d += 1
        wc = w.cuda()
        q, params = NVFP4.quantize(wc)
        dq = NVFP4.dequantize(q, params)
        mse = rel_mse(dq, wc)
        wf = wc.float()
        maxrel = float((wf - dq.float()).abs().max() / (wf.abs().max() + 1e-12))
        rows.append((mse, maxrel, k))
        del q, params, dq, wc, wf
        torch.cuda.empty_cache()
    rows.sort(key=lambda r: r[0], reverse=True)
    print(f"# 2D weight layers checked: {n2d}")
    print(f"{'relMSE':>12} {'maxRel':>10}  layer")
    for mse, maxrel, k in rows[:30]:
        print(f"{mse:12.4e} {maxrel:10.4f}  {k}")
    print("--- bottom 5 (best) ---")
    for mse, maxrel, k in rows[-5:]:
        print(f"{mse:12.4e} {maxrel:10.4f}  {k}")


if __name__ == "__main__":
    main()
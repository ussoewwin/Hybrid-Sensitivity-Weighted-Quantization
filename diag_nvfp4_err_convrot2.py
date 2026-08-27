#!/usr/bin/env python
"""Diagnose ConvRot NVFP4 roundtrip error per layer using the OFFICIAL hswq build_hadamard."""
import sys
import torch
import safetensors.torch as st
from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout as NVFP4

sys.path.insert(0, r"D:\USERFILES\GitHub\hswq\benchmark")
sys.path.insert(0, r"D:\USERFILES\GitHub\hswq")
from hswq_stack.zimage_nvfp4.zi_nvfp4_hadamard import build_hadamard

BASE = r"D:\USERFILES\ComfyUI\ComfyUI\models\unet\darkBeast30BF16INT8_dbzit9DIMRclaw.safetensors"


def convrot_group_size(in_f: int, pref: int = 256) -> int:
    g = pref
    while g > 4 and in_f % g != 0:
        g //= 4
    if in_f % g != 0:
        g = 1
    return g


def rel_mse(a, b):
    a = a.float().reshape(a.shape[0], -1)
    b = b.float().reshape(b.shape[0], -1)
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sd = st.load_file(BASE, device="cpu")
    rows = []
    for k, w in sd.items():
        if not k.endswith(".weight") or w.ndim != 2:
            continue
        if w.shape[0] == 1 or w.shape[1] == 1:
            continue
        wc = w.cuda()
        G = convrot_group_size(wc.shape[1])
        if G > 1:
            H = build_hadamard(G).cuda().to(wc.dtype)
            n = wc.shape[1]
            w_rot = torch.cat([wc[:, i:i + G] @ H.t() for i in range(0, n, G)], dim=1)
            q, p = NVFP4.quantize(w_rot)
            dq = NVFP4.dequantize(q, p)
            w_hat = torch.cat([dq[:, i:i + G] @ H for i in range(0, n, G)], dim=1)
        else:
            q, p = NVFP4.quantize(wc)
            dq = NVFP4.dequantize(q, p)
            w_hat = dq
        mse_rot = rel_mse(w_hat, wc)
        maxrel = float((wc.float() - w_hat.float()).abs().max() / (wc.float().abs().max() + 1e-12))
        rows.append((mse_rot, maxrel, G, k))
        del q, p, dq, w_hat, wc, w_rot, H
        torch.cuda.empty_cache()
    rows.sort(key=lambda r: r[0], reverse=True)
    print(f"# layers: {len(rows)}")
    print(f"{'relMSE(rot)':>13} {'maxRel':>9} {'G':>4}  layer")
    for mse, maxrel, G, k in rows[:30]:
        print(f"{mse:13.4e} {maxrel:9.4f} {G:4d}  {k}")
    print("--- bottom 5 (best) ---")
    for mse, maxrel, G, k in rows[-5:]:
        print(f"{mse:13.4e} {maxrel:9.4f} {G:4d}  {k}")


if __name__ == "__main__":
    main()
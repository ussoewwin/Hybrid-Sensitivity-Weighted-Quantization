"""Write a dequantised copy of a ConvRot INT8 pack, for the third render arm.

The pack stores the rotated int8 codes and a per-row scale; the runtime rotates the
activation instead. This rebuilds the float weight the runtime would produce, so the
render can show what the pack's own numbers mean without the kernel.

  python audit/clip_vision_bigg_dequant.py --pack <pack> --out <copy> [--groupsize 64]
"""

import argparse
import json
import os

import torch
from safetensors.torch import load_file, save_file


def build_hadamard(size, dtype=torch.float32):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groupsize", type=int, default=64)
    a = ap.parse_args()

    sd = load_file(a.pack)
    new = {}
    n = 0
    for k, v in sd.items():
        if k.endswith(".weight_scale") or k.endswith(".comfy_quant"):
            continue
        if torch.is_tensor(v) and v.dtype == torch.int8:
            base = k[: -len(".weight")]
            s = sd[base + ".weight_scale"].float()
            gs = a.groupsize
            while gs > 4 and int(v.shape[1]) % gs != 0:
                gs //= 2
            h = build_hadamard(gs)
            of, inf = int(v.shape[0]), int(v.shape[1])
            g = inf // gs
            q = v.float().view(of, g, gs) * s.reshape(of, 1).unsqueeze(-1)
            new[k] = torch.matmul(q, h.transpose(-1, -2)).reshape(of, inf).to(torch.float16)
            n += 1
            continue
        new[k] = v
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_file(new, a.out)
    print("wrote %s | %d int8 layers dequantised | %d tensors" % (a.out, n, len(new)))


if __name__ == "__main__":
    main()

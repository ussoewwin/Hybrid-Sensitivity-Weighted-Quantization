import os
import sys

sys.path.insert(0, r"D:\USERFILES\ComfyUI\ComfyUI")

import torch
from safetensors.torch import load_file

p30 = r"D:\USERFILES\ComfyUI\ComfyUI\models\sam3\sam3.pt"
raw30 = torch.load(p30, map_location="cpu", weights_only=False)
sd30 = raw30["model"] if isinstance(raw30, dict) and "model" in raw30 else raw30

p31 = r"D:\USERFILES\ComfyUI\ComfyUI\models\unet\sam3.1_multiplex_fp16.safetensors"
sd31 = load_file(p31)

print("--- Dot product scoring keys in SAM 3.0 ---")
for k, v in sorted(sd30.items()):
    if "dot_prod_scoring" in k or "presence" in k:
        print(f"  {k}: {v.shape} ({v.dtype})")

print("\n--- Dot product scoring keys in SAM 3.1 ---")
for k, v in sorted(sd31.items()):
    if "dot_prod_scoring" in k or "presence" in k:
        print(f"  {k}: {v.shape} ({v.dtype})")

import os
import sys

sys.path.insert(0, r"D:\USERFILES\ComfyUI\ComfyUI")

import torch
from safetensors.torch import load_file
import comfy.model_detection

orig_31 = r"D:\USERFILES\ComfyUI\ComfyUI\models\unet\sam3.1_multiplex_fp16.safetensors"
orig_3 = r"D:\USERFILES\ComfyUI\ComfyUI\models\sam3\sam3.pt"

sd31 = load_file(orig_31)
tp31 = [k for k in sd31.keys() if "text_proj" in k]
print("SAM 3.1 text_projection keys:")
for k in tp31:
    print(f"  {k}: {sd31[k].shape}")

sd3 = torch.load(orig_3, map_location="cpu", weights_only=False)
sd3 = sd3["model"] if isinstance(sd3, dict) and "model" in sd3 else sd3
tp3 = [k for k in sd3.keys() if "text_proj" in k]
print("\nSAM 3 text_projection keys:")
for k in tp3:
    print(f"  {k}: {sd3[k].shape}")

import os
import sys

sys.path.insert(0, r"D:\USERFILES\ComfyUI\ComfyUI")

import torch
from safetensors.torch import load_file
import comfy.sd
import comfy.model_detection

pt_path = r"D:\USERFILES\ComfyUI\ComfyUI\models\sam3\sam3.pt"
data = torch.load(pt_path, map_location="cpu", weights_only=False)
sd = data["model"] if isinstance(data, dict) and "model" in data else data

print(f"Total keys in sam3.pt: {len(sd)}")

# Inspect language_backbone keys and text_projection
lang_keys = {k: v for k, v in sd.items() if "language_backbone" in k}
print(f"Language backbone keys: {len(lang_keys)}")

for k, v in sorted(lang_keys.items()):
    if "proj" in k or "embed" in k:
        print(f"  {k}: {v.shape} ({v.dtype})")

# Let's inspect how ComfyUI loads SAM3 (not SAM31)
model_config = comfy.model_detection.model_config_from_unet(sd, "")
print("Detected model_config:", type(model_config))
print("unet_config:", model_config.unet_config)

# Check clip_target
clip_target = model_config.clip_target(sd)
print("clip_target:", clip_target)

# Try process_clip_state_dict
processed_unet = dict(sd)
processed_unet = model_config.process_unet_state_dict(processed_unet)
processed_clip = model_config.process_clip_state_dict(processed_unet)
print(f"Processed clip keys: {len(processed_clip)}")

for k in sorted(processed_clip.keys()):
    if "proj" in k or "embed" in k:
        print(f"  Processed CLIP: {k}: {processed_clip[k].shape}")

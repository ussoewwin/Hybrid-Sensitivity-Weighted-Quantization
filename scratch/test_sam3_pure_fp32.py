import os
import sys

sys.path.insert(0, r"D:\USERFILES\ComfyUI\ComfyUI")

import torch
from PIL import Image
import torchvision.transforms.functional as TF
import comfy.sd
import comfy.model_detection
import comfy.utils
from comfy_extras.nodes_sam3 import SAM3_Detect

img_path = r"C:\Users\ussoe\.gemini\antigravity\brain\8d1447a9-d6bc-44ca-8592-d78296a97ec0\.user_uploaded\media_1788050561694.png"
pil_img = Image.open(img_path).convert("RGB")
t_img = TF.to_tensor(pil_img).permute(1, 2, 0).unsqueeze(0)[:, :1080, :1920, :3].float()

orig_3 = r"D:\USERFILES\ComfyUI\ComfyUI\models\sam3\sam3.pt"
data = torch.load(orig_3, map_location="cpu", weights_only=False)
sd = data["model"] if isinstance(data, dict) and "model" in data else data

# Convert sam3.pt to safetensors without any quantization (Pure FP32)
from safetensors.torch import save_file
fp32_safetensors_path = r"D:\USERFILES\ComfyUI\ComfyUI\models\sam3\sam3_pure_fp32.safetensors"
save_file(sd, fp32_safetensors_path)

print("\n--- Testing SAM3 Pure FP32 Checkpoint Load ---")
out = comfy.sd.load_checkpoint_guess_config(fp32_safetensors_path, output_vae=False, output_clip=True)
mp, clip = out[0], out[1]
print(f"Model Patcher: {type(mp)}")
print(f"CLIP: {type(clip)}")

tokens = clip.tokenize("girl")
cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
conditioning = [[cond, {"pooled_output": pooled}]]

comfy.model_management.load_models_gpu([mp])
masks, bboxes = SAM3_Detect.execute(mp, t_img, conditioning=conditioning, threshold=0.5, refine_iterations=2)
print(f"Pure FP32 SAM 3 Masks sum: {masks.sum().item():.2f}")
print(f"Pure FP32 SAM 3 BBoxes: {bboxes}")

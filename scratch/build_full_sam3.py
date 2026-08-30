import os
import sys
import json

sys.path.insert(0, r"D:\USERFILES\ComfyUI\ComfyUI")
sys.path.insert(0, r"D:\USERFILES\GitHub\hswq")

import torch
from safetensors.torch import save_file, load_file
import comfy.sd
import nodes
from comfy_extras.nodes_sam3 import SAM3_Detect
from PIL import Image
import torchvision.transforms.functional as TF

# Load SAM 3.0 raw pt and SAM 3.1 reference
p30 = r"D:\USERFILES\ComfyUI\ComfyUI\models\sam3\sam3.pt"
raw30 = torch.load(p30, map_location="cpu", weights_only=False)
sd30 = raw30["model"] if isinstance(raw30, dict) and "model" in raw30 else raw30

p31 = r"D:\USERFILES\ComfyUI\ComfyUI\models\unet\sam3.1_multiplex_fp16.safetensors"
sd31 = load_file(p31)

print(f"SAM 3.0 keys: {len(sd30)}, SAM 3.1 keys: {len(sd31)}")

# Build a hybrid SAM 3.0 model that uses SAM 3.0's Detector & Trunk weights,
# but mapped to ComfyUI's native Multiplex pipeline
new_sd = {}

# 1. Start with SAM 3.1 base structure for tracker/multiplex heads
for k, v in sd31.items():
    new_sd[k] = v.clone()

# 2. Inject all SAM 3.0 Detector weights (Vision backbone trunk, Transformer, Segmentation Head, CLIP)
for k, v in sd30.items():
    if ".attn.freqs_cis" in k:
        continue
    if "encoder.text_projection" in k:
        continue
    if k.startswith("tracker."):
        continue
    
    # Split fused in_proj
    if k.endswith((".in_proj_weight", ".in_proj_bias")) and "language_backbone" not in k:
        base, suffix = k.rsplit(".in_proj_", 1)
        s = ".weight" if suffix == "weight" else ".bias"
        d = v.shape[0] // 3
        new_sd[f"{base}.q_proj{s}"] = v[:d].clone()
        new_sd[f"{base}.k_proj{s}"] = v[d:2*d].clone()
        new_sd[f"{base}.v_proj{s}"] = v[2*d:].clone()
        continue
        
    # Map convs: scales 0, 1, 2
    if "vision_backbone.convs." in k:
        scale_idx = int(k.split("convs.")[1].split(".")[0])
        if scale_idx < 3:
            new_sd[k] = v.clone()
        continue
        
    if "vision_backbone.sam2_convs." in k:
        scale_idx = int(k.split("sam2_convs.")[1].split(".")[0])
        if scale_idx < 3:
            prop_k = k.replace("sam2_convs.", "propagation_convs.")
            inter_k = k.replace("sam2_convs.", "interactive_convs.")
            new_sd[prop_k] = v.clone()
            new_sd[inter_k] = v.clone()
        continue

    new_sd[k] = v.clone()

# 3. Quantize with ConvRot INT8
from clip_convert.convert_clip_convrot_int8 import build_hadamard, convrot_group_size, rotate_weight, quantize_int8_rowwise, _encode_meta

quant_sd = {}
meta_layers = {}
h_cache = {}

float_tags = [
    "segmentation_head", "dot_prod_scoring", "bbox_embed", "presence_token",
    "query_embed", "reference_points", "mask_tokens", "pos_embed", "point_embeddings",
    "not_a_point_embed", "no_mask_embed", "token_embedding", "positional_embedding",
    "language_backbone", "boxRPB", "pos_enc_project", "direct_project", "pool_project",
    "points_direct", "geometry_encoder", "tracker"
]

convrot_count = 0
for key, tensor in sorted(new_sd.items()):
    is_2d = key.endswith(".weight") and tensor.ndim == 2 and tensor.dtype in (torch.float16, torch.float32, torch.bfloat16)
    if not is_2d or any(tag in key for tag in float_tags):
        quant_sd[key] = tensor.half() if tensor.dtype == torch.float32 else tensor
        continue

    w = tensor.float()
    out_f, in_f = w.shape
    module_key = key[:-len(".weight")]

    if in_f >= 64 and out_f >= 64 and (in_f % 4 == 0) and (out_f % 4 == 0):
        gs = convrot_group_size(in_f, 256)
        if gs is not None:
            if gs not in h_cache:
                h_cache[gs] = build_hadamard(gs)
            w_rot = rotate_weight(w, h_cache[gs], gs)
            q, scale = quantize_int8_rowwise(w_rot)
            config = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": gs}
            quant_sd[key] = q
            quant_sd[f"{module_key}.weight_scale"] = scale
            quant_sd[f"{module_key}.comfy_quant"] = _encode_meta(config)
            meta_layers[module_key] = config
            convrot_count += 1
            continue

    quant_sd[key] = tensor.half() if tensor.dtype == torch.float32 else tensor

print(f"Total ConvRot INT8 layers: {convrot_count}")

metadata = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": meta_layers})}

out_target = r"D:\USERFILES\ComfyUI\ComfyUI\models\unet\sam3_convrot_int8.safetensors"
save_file(quant_sd, out_target + ".tmp", metadata=metadata)
if os.path.exists(out_target):
    try:
        os.remove(out_target)
    except Exception:
        pass
os.replace(out_target + ".tmp", out_target)

# Sync to checkpoints
ckpt_target = r"D:\USERFILES\ComfyUI\ComfyUI\models\checkpoints\sam3_convrot_int8.safetensors"
save_file(quant_sd, ckpt_target + ".tmp", metadata=metadata)
if os.path.exists(ckpt_target):
    try:
        os.remove(ckpt_target)
    except Exception:
        pass
os.replace(ckpt_target + ".tmp", ckpt_target)

print("Saved perfectly compatible SAM 3 model to unet and checkpoints!")

# Test on live workflow
img_path = r"D:\USERFILES\ComfyUI\ComfyUI\input\-57775-3727535583.jpeg"
pil_img = Image.open(img_path).convert("RGB")
t_img = TF.to_tensor(pil_img).permute(1, 2, 0).unsqueeze(0).float()

out = comfy.sd.load_checkpoint_guess_config(out_target, output_vae=False, output_clip=True)
mp, clip = out[0], out[1]

te_node = nodes.CLIPTextEncode()
conditioning = te_node.encode(clip, "girl")[0]

comfy.model_management.load_models_gpu([mp])

print("\n--- Testing SAM3_Detect on reconstructed SAM 3 ConvRot INT8 ---")
masks, bboxes = SAM3_Detect.execute(mp, t_img, conditioning=conditioning, threshold=0.40, refine_iterations=2)
print(f"Mask shape: {masks.shape}")
print(f"Mask sum (pixel area): {masks.sum().item():.1f} px / {t_img.shape[1] * t_img.shape[2]} px")
print(f"Detected BBoxes: {bboxes}")

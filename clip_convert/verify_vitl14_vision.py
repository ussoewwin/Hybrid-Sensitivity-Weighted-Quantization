"""CLIP ViT-L/14 cos-distribution verification (vision tower).

Fixed seed, >=200 image batches. Compares FP16 vs the ConvRot INT8 pack through
ClipVisionModel built with mixed_precision_ops (the path that consumes
comfy_quant for a vision tower).
"""
import sys, os, json, argparse, statistics
COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
sys.path.insert(0, COMFY)
os.chdir(COMFY)
import torch
import comfy.utils
import comfy.clip_vision as cv
import comfy.model_management
from comfy import ops as O
from comfy.quant_ops import QUANT_ALGOS

D = r"D:\USERFILES\ComfyUI\ComfyUI\models\clip"
SRC = os.path.join(D, "ViT-L-14-BEST-smooth-GmP-HF-format.safetensors")
PACK = os.path.join(D, "ViT-L-14-BEST-smooth-GmP-HF-format_hswq_convrot_int8.safetensors")
jc = os.path.join(COMFY, "comfy", "clip_vision_config_vitl.json")

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def prep(sd):
    vsd = {k: v for k, v in sd.items() if k.startswith("vision_model.")}
    if "vision_model.embeddings.class_embedding" not in vsd:
        hid = 768
        for k, v in vsd.items():
            if k.endswith("patch_embedding.weight"):
                hid = v.shape[0]; break
        vsd["vision_model.embeddings.class_embedding"] = torch.zeros(hid)
    return vsd


class CVQ(cv.ClipVisionModel):
    def __init__(self, jc, ops):
        import json as _j
        config = _j.load(open(jc))
        self.image_size = config.get("image_size", 224)
        self.image_mean = config.get("image_mean", [0.48145466, 0.4578275, 0.40821073])
        self.image_std = config.get("image_std", [0.26862954, 0.26130258, 0.27577711])
        self.model_type = config.get("model_type", "clip_vision_model")
        self.config = config.copy()
        mc = cv.IMAGE_ENCODERS.get(self.model_type)
        self.return_all_hidden_states = False
        self.load_device = comfy.model_management.text_encoder_device()
        off = comfy.model_management.text_encoder_offload_device()
        self.dtype = comfy.model_management.text_encoder_dtype(self.load_device)
        self.model = mc(config, self.dtype, off, ops)
        self.model.eval()
        self.patcher = comfy.model_patcher.CoreModelPatcher(self.model, load_device=self.load_device, offload_device=off)
        self.naf = None


def enc(v, imgs):
    mdt = next(p.dtype for p in v.model.parameters() if p.is_floating_point())
    with torch.no_grad():
        return v.encode_image(imgs.to(dtype=mdt), crop=False).image_embeds.detach().float().reshape(-1).cpu()


sd = comfy.utils.load_torch_file(SRC, safe_load=True)
sv = sys.argv; sys.argv = [sv[0]]
try:
    v16 = cv.load_clipvision_from_sd(prep(sd))
finally:
    sys.argv = sv
pd = comfy.utils.load_torch_file(PACK, safe_load=True)
vq = CVQ(jc, O.mixed_precision_ops({"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]}, torch.float16, full_precision_mm=True))
vq.load_sd(prep(pd))

g = torch.Generator(device="cpu").manual_seed(a.seed)
cos = []
for i in range(a.n):
    imgs = (torch.rand(1, 3, 224, 224, generator=g) * 2 - 1)
    x = enc(v16, imgs)
    y = enc(vq, imgs)
    cos.append(torch.nn.functional.cosine_similarity(x.reshape(1, -1), y.reshape(1, -1)).item())

cs = sorted(cos)
res = {"n": len(cos), "mean": statistics.mean(cos), "min": cs[0],
       "p5": cs[max(0, int(len(cs) * 0.05) - 1)], "p50": statistics.median(cos), "max": cs[-1],
       "pass_mean_099": statistics.mean(cos) >= 0.99, "pass_min_098": cs[0] >= 0.98}
json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1))

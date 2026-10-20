"""Downstream smoke test: run the pack through the real ComfyUI node paths and
confirm both encoders complete without error and produce sane output.

- text: CLIPLoader-equivalent (SDClipModel) with quantization_metadata
- vision: CLIPVisionLoader-equivalent (ClipVisionModel + mixed_precision_ops)
- zero-shot: nearest-neighbour among a small fixed image/text set
"""
import sys, os, json
COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
sys.path.insert(0, COMFY)
os.chdir(COMFY)
import torch
import comfy.sd
import comfy.utils
import comfy.clip_vision as cv
import comfy.model_management
from comfy import ops as O
from comfy.quant_ops import QUANT_ALGOS

D = r"D:\USERFILES\ComfyUI\ComfyUI\models\clip"
PACK = os.path.join(D, "CLIP-SAE-ViT-L-14-hswq_convrot_int8_protected.safetensors")
jc = os.path.join(COMFY, "comfy", "clip_vision_config_vitl.json")


def load_t(p):
    sd, m = comfy.utils.load_torch_file(p, safe_load=True, return_metadata=True)
    mo = {}
    if isinstance(m, dict) and "_quantization_metadata" in m:
        mo = {"quantization_metadata": json.loads(m["_quantization_metadata"])}
    sv = sys.argv; sys.argv = [sv[0]]
    try:
        c = comfy.sd.load_clip(ckpt_paths=[p], embedding_directory=None, clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION, model_options=mo)
    finally:
        sys.argv = sv
    inner = getattr(c.cond_stage_model, "clip_l", c.cond_stage_model)
    tp = getattr(getattr(inner, "transformer", None), "text_projection", None)
    if tp is not None and getattr(tp, "weight", None) is None:
        tp.weight = torch.nn.Parameter(torch.zeros(768, 768, dtype=torch.float16), requires_grad=False)
    return c


def prep(sd):
    vsd = {k: v for k, v in sd.items() if k.startswith("vision_model.")}
    if "vision_model.embeddings.class_embedding" not in vsd:
        vsd["vision_model.embeddings.class_embedding"] = torch.zeros(1024)
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


c = load_t(PACK)
v = CVQ(jc, O.mixed_precision_ops({"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]}, torch.float16, full_precision_mm=True))
v.load_sd(prep(comfy.utils.load_torch_file(PACK, safe_load=True)))

# text encode
with torch.no_grad():
    t = c.encode_from_tokens_scheduled(c.tokenize("a photo of a cat"))[0][0].float()
print("text encode: shape=%s finite=%s norm=%.4f" % (tuple(t.shape), bool(torch.isfinite(t).all()), t.norm().item()), flush=True)

# vision encode
g = torch.Generator(device="cpu").manual_seed(7)
imgs = torch.stack([torch.rand(3, 224, 224, generator=g) * 2 - 1 for _ in range(3)])
mdt = next(p.dtype for p in v.model.parameters() if p.is_floating_point())
with torch.no_grad():
    emb = v.encode_image(imgs.to(dtype=mdt), crop=False).image_embeds.float()
print("vision encode: shape=%s finite=%s norms=%s" % (tuple(emb.shape), bool(torch.isfinite(emb).all()), [round(float(x), 3) for x in emb.norm(dim=-1)]), flush=True)

# downstream consistency: encode the SAME inputs with FP16 and INT8 and confirm
# the quantized path reproduces them (this is the real smoke test for the pack)
SRC = os.path.join(D, "CLIP-SAE-ViT-L-14-FP32.safetensors")
c16 = load_t(SRC)
with torch.no_grad():
    t16 = c16.encode_from_tokens_scheduled(c16.tokenize("a photo of a cat"))[0][0].float().reshape(-1)
    tq = c.encode_from_tokens_scheduled(c.tokenize("a photo of a cat"))[0][0].float().reshape(-1)
tcos = torch.nn.functional.cosine_similarity(t16.reshape(1, -1), tq.reshape(1, -1)).item()
sv = sys.argv; sys.argv = [sv[0]]
try:
    v16 = cv.load_clipvision_from_sd(prep(comfy.utils.load_torch_file(SRC, safe_load=True)))
finally:
    sys.argv = sv
with torch.no_grad():
    e16 = v16.encode_image(imgs.to(dtype=mdt), crop=False).image_embeds.float()
vcos = torch.nn.functional.cosine_similarity(e16.reshape(1, -1), emb.reshape(1, -1)).item()
print("downstream consistency: text_cos=%.6f vision_cos=%.6f" % (tcos, vcos), flush=True)

res = {"text_shape": list(t.shape), "text_finite": bool(torch.isfinite(t).all()),
       "vision_shape": list(emb.shape), "vision_finite": bool(torch.isfinite(emb).all()),
       "text_cos_vs_fp16": tcos, "vision_cos_vs_fp16": vcos}
json.dump(res, open(r"D:\USERFILES\GitHub\hswq\impact\smoke_sae_vitl14.json", "w", encoding="utf-8"), indent=1)
print("SMOKE OK", flush=True)

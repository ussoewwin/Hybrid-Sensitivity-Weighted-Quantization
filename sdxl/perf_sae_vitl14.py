"""Performance measurement: file size, weight bytes, and encode latency for
FP16 vs the ConvRot INT8 pack (text + vision)."""
import sys, os, json, time
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
from comfy_kitchen.tensor.base import QuantizedTensor

D = r"D:\USERFILES\ComfyUI\ComfyUI\models\clip"
SRC = os.path.join(D, "CLIP-SAE-ViT-L-14-FP32.safetensors")
PACK = os.path.join(D, "CLIP-SAE-ViT-L-14-hswq_convrot_int8_protected.safetensors")
jc = os.path.join(COMFY, "comfy", "clip_vision_config_vitl.json")


def load_t(p, mo=None):
    sv = sys.argv; sys.argv = [sv[0]]
    try:
        c = comfy.sd.load_clip(ckpt_paths=[p], embedding_directory=None, clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION, model_options=mo or {})
    finally:
        sys.argv = sv
    inner = getattr(c.cond_stage_model, "clip_l", c.cond_stage_model)
    tr = getattr(inner, "transformer", None)
    tp = getattr(tr, "text_projection", None) if tr is not None else None
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


def wbytes(m):
    tot = 0
    seen = set()
    for _, mod in m.named_modules():
        w = getattr(mod, "weight", None)
        if w is None or id(w) in seen:
            continue
        seen.add(id(w))
        q = getattr(w, "_qdata", None)
        if q is not None:
            tot += q.numel() * q.element_size()
            sc = getattr(getattr(w, "_params", None), "scale", None)
            if sc is not None:
                tot += sc.numel() * sc.element_size()
        else:
            tot += w.numel() * w.element_size()
    return tot


res = {}
res["file_fp16_mb"] = os.path.getsize(SRC) / 1024**2
res["file_int8_mb"] = os.path.getsize(PACK) / 1024**2

sd, m = comfy.utils.load_torch_file(PACK, safe_load=True, return_metadata=True)
qm = json.loads(m["_quantization_metadata"])

# text
for tag, mo in (("text_fp16", None), ("text_int8", {"quantization_metadata": qm})):
    p = SRC if tag.endswith("fp16") else PACK
    c = load_t(p, mo)
    wb = wbytes(c.cond_stage_model)
    with torch.no_grad():
        c.encode_from_tokens_scheduled(c.tokenize("a cat"))
    t0 = time.time()
    for _ in range(10):
        with torch.no_grad():
            c.encode_from_tokens_scheduled(c.tokenize("a cat"))
    res[tag + "_weight_mb"] = wb / 1024**2
    res[tag + "_ms"] = (time.time() - t0) / 10 * 1000

# vision
g = torch.Generator(device="cpu").manual_seed(42)
imgs = (torch.rand(1, 3, 224, 224, generator=g) * 2 - 1)
sv = sys.argv; sys.argv = [sv[0]]
try:
    v16 = cv.load_clipvision_from_sd(prep(comfy.utils.load_torch_file(SRC, safe_load=True)))
finally:
    sys.argv = sv
vq = CVQ(jc, O.mixed_precision_ops({"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]}, torch.float16, full_precision_mm=True))
vq.load_sd(prep(comfy.utils.load_torch_file(PACK, safe_load=True)))
for tag, v in (("vision_fp16", v16), ("vision_int8", vq)):
    wb = wbytes(v.model)
    mdt = next(p.dtype for p in v.model.parameters() if p.is_floating_point())
    with torch.no_grad():
        v.encode_image(imgs.to(dtype=mdt), crop=False)
    t0 = time.time()
    for _ in range(10):
        with torch.no_grad():
            v.encode_image(imgs.to(dtype=mdt), crop=False)
    res[tag + "_weight_mb"] = wb / 1024**2
    res[tag + "_ms"] = (time.time() - t0) / 10 * 1000

json.dump(res, open(r"D:\USERFILES\GitHub\hswq\impact\perf_sae_vitl14.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1))

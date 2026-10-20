"""SAE ViT-L-14 cos verification on the fixed 256-item acceptance set.
Text: SD1ClipModel + quantization_metadata. Vision: ClipVisionModel + mixed_precision_ops.
"""
import sys, os, json, argparse, statistics, random
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
SRC = os.path.join(D, "Long-ViT-L-14-BEST-GmP-smooth-ft-full.safetensors")
jc = os.path.join(COMFY, "comfy", "clip_vision_config_vitl.json")

WORDS = ["a photo of", "an image of", "a drawing of", "a painting of", "a sketch of",
         "a cat", "a dog", "a horse", "a bird", "a fish", "a tree", "a house",
         "a car", "a boat", "a plane", "a flower", "a mountain", "a river",
         "at night", "at sunset", "in the rain", "in winter", "in summer",
         "in a city", "in a forest", "on a beach", "in the snow", "in the sky",
         "close up", "wide shot", "high quality", "detailed", "colorful", "dark"]


def load_t(p):
    sd, m = comfy.utils.load_torch_file(p, safe_load=True, return_metadata=True)
    mo = {}
    if isinstance(m, dict) and "_quantization_metadata" in m:
        mo = {"quantization_metadata": json.loads(m["_quantization_metadata"])}
    sv = sys.argv; sys.argv = [sv[0]]
    try:
        c = comfy.sd.load_clip(ckpt_paths=[p], embedding_directory=None,
                               clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION, model_options=mo)
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


def enc_v(v, imgs):
    mdt = next(p.dtype for p in v.model.parameters() if p.is_floating_point())
    with torch.no_grad():
        return v.encode_image(imgs.to(dtype=mdt), crop=False).image_embeds.detach().float().reshape(-1).cpu()


def stats(cos):
    cs = sorted(cos)
    return {"n": len(cos), "mean": statistics.mean(cos), "min": cs[0],
            "p5": cs[max(0, int(len(cs) * 0.05) - 1)], "p50": statistics.median(cos), "max": cs[-1],
            "pass_mean_099": statistics.mean(cos) >= 0.99, "pass_min_098": cs[0] >= 0.98}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rnd = random.Random(a.seed)
    prompts = [", ".join(rnd.choice(WORDS) for _ in range(rnd.randint(2, 5))) for _ in range(a.n)]

    ref = load_t(SRC)
    q = load_t(a.pack)
    ct = []
    for t in prompts:
        with torch.no_grad():
            x = ref.encode_from_tokens_scheduled(ref.tokenize(t))[0][0].float().reshape(-1)
            y = q.encode_from_tokens_scheduled(q.tokenize(t))[0][0].float().reshape(-1)
        ct.append(torch.nn.functional.cosine_similarity(x.reshape(1, -1), y.reshape(1, -1)).item())

    sv = sys.argv; sys.argv = [sv[0]]
    try:
        v16 = cv.load_clipvision_from_sd(prep(comfy.utils.load_torch_file(SRC, safe_load=True)))
    finally:
        sys.argv = sv
    sd = comfy.utils.load_torch_file(a.pack, safe_load=True)
    vq = CVQ(jc, O.mixed_precision_ops({"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]}, torch.float16, full_precision_mm=True))
    vq.load_sd(prep(sd))
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    cvl = []
    for _ in range(a.n):
        im = (torch.rand(1, 3, 224, 224, generator=g) * 2 - 1)
        x = enc_v(v16, im); y = enc_v(vq, im)
        cvl.append(torch.nn.functional.cosine_similarity(x.reshape(1, -1), y.reshape(1, -1)).item())

    res = {"text": stats(ct), "vision": stats(cvl)}
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1))

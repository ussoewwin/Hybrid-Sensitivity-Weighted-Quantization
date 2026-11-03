"""CLIP ViT-L/14 cos-distribution verification (text tower).

Fixed seed, >=200 prompt pairs. Compares FP16 vs the ConvRot INT8 pack through
the SD1ClipModel path (with quantization_metadata), which is the path that
consumes comfy_quant for a text tower.

Outputs JSON with mean / min / p5 / p50 of the per-pair cosine similarity.
"""
import sys, os, json, argparse, statistics
COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
sys.path.insert(0, COMFY)
os.chdir(COMFY)
import torch
import comfy.sd
import comfy.utils

D = r"D:\USERFILES\ComfyUI\ComfyUI\models\clip"
SRC = os.path.join(D, "ViT-L-14-BEST-smooth-GmP-HF-format.safetensors")
PACK = os.path.join(D, "ViT-L-14-BEST-smooth-GmP-HF-format_hswq_convrot_int8.safetensors")

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", required=True)
a = ap.parse_args()

# deterministic prompt set (composed from a fixed vocabulary; no network)
WORDS = ["a photo of", "an image of", "a drawing of", "a painting of", "a sketch of",
         "a cat", "a dog", "a horse", "a bird", "a fish", "a tree", "a house",
         "a car", "a boat", "a plane", "a flower", "a mountain", "a river",
         "at night", "at sunset", "in the rain", "in winter", "in summer",
         "in a city", "in a forest", "on a beach", "in the snow", "in the sky",
         "close up", "wide shot", "high quality", "detailed", "colorful", "dark"]
import random
rnd = random.Random(a.seed)
prompts = []
for _ in range(a.n):
    k = rnd.randint(2, 5)
    prompts.append(", ".join(rnd.choice(WORDS) for _ in range(k)))


def load(path, with_meta=False):
    mo = {}
    if with_meta:
        sd, m = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        mo = {"quantization_metadata": json.loads(m["_quantization_metadata"])}
    sv = sys.argv
    sys.argv = [sv[0]]
    try:
        c = comfy.sd.load_clip(ckpt_paths=[path], embedding_directory=None,
                               clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION, model_options=mo)
    finally:
        sys.argv = sv
    inner = getattr(c.cond_stage_model, "clip_l", c.cond_stage_model)
    tr = getattr(inner, "transformer", None)
    tp = getattr(tr, "text_projection", None) if tr is not None else None
    if tp is not None and getattr(tp, "weight", None) is None:
        tp.weight = torch.nn.Parameter(torch.zeros(768, 768, dtype=torch.float16), requires_grad=False)
    return c


def encode_one(c, t):
    with torch.no_grad():
        o = c.encode_from_tokens_scheduled(c.tokenize(t))
    return o[0][0].detach().float().reshape(-1).cpu()


ref = load(SRC)
q = load(PACK, with_meta=True)
cos = []
for t in prompts:
    x = encode_one(ref, t)
    y = encode_one(q, t)
    cos.append(torch.nn.functional.cosine_similarity(x.reshape(1, -1), y.reshape(1, -1)).item())

cos_sorted = sorted(cos)
res = {
    "n": len(cos),
    "mean": statistics.mean(cos),
    "min": cos_sorted[0],
    "p5": cos_sorted[max(0, int(len(cos_sorted) * 0.05) - 1)],
    "p50": statistics.median(cos),
    "max": cos_sorted[-1],
    "pass_mean_099": statistics.mean(cos) >= 0.99,
    "pass_min_098": cos_sorted[0] >= 0.98,
}
json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1))

# -*- coding: utf-8 -*-
"""Qwen3 text-encoder cos verification (fixed 256 prompts, seed 42)."""
import sys, os, json, argparse, statistics, random
COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
sys.path.insert(0, COMFY)
os.chdir(COMFY)
import torch
import comfy.utils
import comfy.text_encoders.anima
import comfy.text_encoders.hunyuan_video

D = r"D:\USERFILES\ComfyUI\ComfyUI\models\clip"
SRC = os.path.join(D, "Josiefied-Qwen3-0.6B-abliterated-v1.safetensors")

WORDS = ["a photo of", "an image of", "a drawing of", "a painting of", "a sketch of",
         "a cat", "a dog", "a horse", "a bird", "a fish", "a tree", "a house",
         "a car", "a boat", "a plane", "a flower", "a mountain", "a river",
         "at night", "at sunset", "in the rain", "in winter", "in summer",
         "in a city", "in a forest", "on a beach", "in the snow", "in the sky",
         "close up", "wide shot", "high quality", "detailed", "colorful", "dark"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build(path, device="cpu"):
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    detect = comfy.text_encoders.hunyuan_video.llama_detect(sd)
    te = comfy.text_encoders.anima.te(**detect)
    clip = te(device=device)
    clip.load_sd(sd)
    inner = getattr(clip, getattr(clip, "clip"))
    return clip, getattr(inner, "transformer", inner)


def encode(clip, ids):
    tr = getattr(getattr(clip, getattr(clip, "clip")), "transformer", None)
    with torch.no_grad():
        out = tr(ids)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.detach().float().reshape(-1).cpu()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rnd = random.Random(a.seed)
    prompts = [", ".join(rnd.choice(WORDS) for _ in range(rnd.randint(2, 5))) for _ in range(a.n)]

    c_ref, tr_ref = build(SRC, DEV)
    c_q, tr_q = build(a.pack, DEV)

    cos = []
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    for t in prompts:
        ids = torch.randint(0, 150000, (1, 16), generator=g).to(DEV)
        x = encode(c_ref, ids)
        y = encode(c_q, ids)
        cos.append(torch.nn.functional.cosine_similarity(x.reshape(1, -1), y.reshape(1, -1)).item())

    cs = sorted(cos)
    res = {"n": len(cos), "mean": statistics.mean(cos), "min": cs[0],
           "p5": cs[max(0, int(len(cs) * 0.05) - 1)], "p50": statistics.median(cos), "max": cs[-1],
           "pass_mean_099": statistics.mean(cos) >= 0.99, "pass_min_098": cs[0] >= 0.98}
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1))

# -*- coding: utf-8 -*-
"""Qwen2.5-VL-7B: the aggregate contribution of each layer."""
import argparse, json, math, os, statistics, sys
COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
sys.path.insert(0, COMFY); os.chdir(COMFY)
import torch
import comfy.utils
import comfy.text_encoders.qwen_image as qi
import comfy.text_encoders.hunyuan_video as hv
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def had(sz):
    h4 = torch.tensor([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]], dtype=torch.float32)
    h = h4
    while h.shape[0] < sz: h = torch.kron(h, h4)
    return h / math.sqrt(sz)
_H = {}
def gs_for(n, pref=256):
    g = pref
    while g >= 4:
        if n % g == 0 and (g & (g-1)) == 0 and (g.bit_length()-1) % 2 == 0: return g
        g //= 4
    return None
def quant_dequant(w, gs):
    if gs not in _H: _H[gs] = had(gs)
    H = _H[gs].to(device=w.device, dtype=w.dtype)
    of, inf = w.shape; g = inf // gs
    r = torch.matmul(w.view(of, g, gs), H.T).reshape(w.shape)
    a = r.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30); s = a / 127.0
    q = (r / s).round().clamp(-127, 127)
    return torch.matmul((q*s).view(of, g, gs), H).reshape(w.shape)

def build(path):
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    d = hv.llama_detect(sd)
    te = qi.te(**d)
    clip = te(device=DEV); clip.load_sd(sd)
    inner = getattr(clip, getattr(clip, "clip"))
    return clip, getattr(inner, "transformer", inner)

def hidden(clip, ids):
    tr = getattr(getattr(clip, getattr(clip, "clip")), "transformer", None)
    with torch.no_grad(): out = tr(ids)
    if isinstance(out, (tuple, list)): out = out[0]
    return out.detach().float().reshape(-1).cpu()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    clip, tr = build(a.src)
    print("[1] the model built: %s" % type(tr).__name__, flush=True)
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    batches = [torch.randint(0, 150000, (1, 16), generator=g).to(DEV) for _ in range(a.n)]
    ref = [hidden(clip, b) for b in batches]
    print("[1] the baseline: %s" % (tuple(ref[0].shape),), flush=True)
    mods = dict(tr.named_modules())
    cands = [n for n, m in tr.named_modules() if type(m).__name__ == "Linear" and getattr(m, "weight", None) is not None]
    print("[2] the Linear modules: %d" % len(cands), flush=True)
    contrib = {}
    for i, name in enumerate(cands):
        mm = mods[name]; orig = mm.weight.data.clone(); w = orig.float()
        gs = gs_for(int(w.shape[1])) if w.ndim == 2 else None
        if w.ndim != 2 or gs is None or int(w.shape[0]) % 4 != 0:
            contrib[name] = None; continue
        mm.weight.data = quant_dequant(w, gs).to(orig.dtype)
        try:
            cs = []
            for j, b in enumerate(batches):
                f = hidden(clip, b)
                cs.append(torch.nn.functional.cosine_similarity(ref[j].reshape(1,-1), f.reshape(1,-1)).item())
            contrib[name] = 1.0 - statistics.mean(cs)
        finally:
            mm.weight.data = orig
        if (i+1) % 20 == 0: print("   [%d/%d] %s" % (i+1, len(cands), name), flush=True)
    del clip
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    json.dump({"impacts": contrib, "n": a.n, "seed": a.seed}, open(a.out, "w", encoding="utf-8"), indent=1, sort_keys=True)
    print(); print("saved: %s" % a.out)
    for n, v in sorted(((k,x) for k,x in contrib.items() if x is not None), key=lambda kv: -kv[1])[:12]:
        print("   %.4e  %s" % (v, n))

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
import sys, os, json, time
COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
sys.path.insert(0, COMFY); os.chdir(COMFY)
import torch
import comfy.utils
import comfy.text_encoders.anima, comfy.text_encoders.hunyuan_video
D = r"D:\USERFILES\ComfyUI\ComfyUI\models\clip"
SRC = os.path.join(D,"Josiefied-Qwen3-0.6B-abliterated-v1.safetensors")
PACK = os.path.join(D,"Josiefied-Qwen3-0.6B-abliterated-v1_hswq_convrot_int8.safetensors")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
def build(p):
    sd = comfy.utils.load_torch_file(p, safe_load=True)
    det = comfy.text_encoders.hunyuan_video.llama_detect(sd)
    te = comfy.text_encoders.anima.te(**det)
    c = te(device=DEV); c.load_sd(sd)
    inner = getattr(c, getattr(c,"clip"))
    return c, getattr(inner,"transformer",inner)
res = {"file_fp16_mb": os.path.getsize(SRC)/1024**2, "file_int8_mb": os.path.getsize(PACK)/1024**2}
ids = torch.randint(0,150000,(1,16)).to(DEV)
for tag, p in (("fp16", SRC), ("int8", PACK)):
    c, tr = build(p)
    def enc():
        with torch.no_grad():
            o = tr(ids)
        return (o[0] if isinstance(o,(tuple,list)) else o).detach().float()
    o = enc()
    torch.cuda.synchronize() if DEV=="cuda" else None
    t0=time.time()
    for _ in range(10): enc()
    torch.cuda.synchronize() if DEV=="cuda" else None
    res[tag+"_ms"] = (time.time()-t0)/10*1000
    res[tag+"_absmax"] = o.abs().max().item()
    del c, tr
    if DEV=="cuda": torch.cuda.empty_cache()
print(json.dumps(res, indent=1))
json.dump(res, open(r"D:\USERFILES\GitHub\hswq\impact\perf_josiefied_qwen3_06b.json","w",encoding="utf-8"), indent=1)

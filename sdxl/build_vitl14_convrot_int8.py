import sys, os, json, math
COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
sys.path.insert(0, COMFY); os.chdir(COMFY)
import torch, comfy.utils
from safetensors.torch import save_file
D = r"D:\USERFILES\ComfyUI\ComfyUI\models\clip"
SRC = os.path.join(D,"ViT-L-14-BEST-smooth-GmP-HF-format.safetensors")
impt={}
for k,v in json.load(open(r"D:\USERFILES\GitHub\hswq\impact\impact_vitl14_best.json",encoding="utf-8"))["impacts"].items():
    ck="text_model."+k[len("transformer.text_model."):] if k.startswith("transformer.text_model.") else k
    impt[ck]=v
impv={}
for k,v in json.load(open(r"D:\USERFILES\GitHub\hswq\impact\impact_vitl14_best_vision.json",encoding="utf-8"))["impacts"].items():
    ck="vision_model."+k if k.startswith("encoder.") else k
    impv[ck]=v
def had(sz):
    h4=torch.tensor([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]],dtype=torch.float32); h=h4
    while h.shape[0]<sz: h=torch.kron(h,h4)
    return h/math.sqrt(sz)
_H={}
def rot(w,gs=256):
    if gs not in _H: _H[gs]=had(gs)
    of,inf=w.shape; g=inf//gs
    return torch.matmul(w.view(of,g,gs), _H[gs].T).reshape(w.shape)
def qrow(w):
    a=w.abs().amax(dim=-1,keepdim=True).clamp(min=1e-30); s=a/127.0
    return (w/s).round().clamp(-127,127).to(torch.int8), s.float()
sd=comfy.utils.load_torch_file(SRC, safe_load=True)
thr_t, thr_v = 1.3, 0.5
prot=set([k for k,v in impt.items() if v>thr_t]+[k for k,v in impv.items() if v>thr_v])
out={}; meta={}; nlin=0; nprot=0
for k,t in sd.items():
    if not (k.endswith(".weight") and t.ndim==2): out[k]=t; continue
    mk=k[:-7]; inf=int(t.shape[1])
    if mk in prot or inf%256!=0 or t.shape[0]%4!=0: out[k]=t; nprot+=1; continue
    q,s=qrow(rot(t.float(),256))
    out[k]=q; out[mk+".weight_scale"]=s
    out[mk+".comfy_quant"]=torch.tensor(list(json.dumps({"format":"int8_tensorwise","convrot":True,"convrot_groupsize":256}).encode()),dtype=torch.uint8)
    meta[mk]={"format":"int8_tensorwise","convrot":True,"convrot_groupsize":256}; nlin+=1
OUT=os.path.join(D,"ViT-L-14-BEST-smooth-GmP-HF-format_hswq_convrot_int8.safetensors")
save_file(out, OUT, metadata={"_quantization_metadata": json.dumps({"format_version":"1.0","layers":meta})})
print("INT8=%d FP16=%d size=%.1f MB (src %.1f)" % (nlin,nprot,os.path.getsize(OUT)/1024**2,os.path.getsize(SRC)/1024**2), flush=True)

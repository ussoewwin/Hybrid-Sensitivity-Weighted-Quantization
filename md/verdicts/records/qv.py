import torch, os, statistics, collections
from safetensors import safe_open
S=r"C:\Users\ussoe\AppData\Roaming\AutoClaw-oversea-official\accounts\4e07651c7b86fd7cc1a4fd916822bc63d2ee393fe31bf09e8330ec79b841e1c4\projects\Default\.openclaw\tmp"
RD=os.path.join(S,"qwen3vl_official")
C=r"D:\USERFILES\ComfyUI\ComfyUI\models\clip\Huihui-Qwen3-VL-4B-Instruct-abliterated.safetensors"
files=sorted(f for f in os.listdir(RD) if f.endswith(".safetensors"))
where={}
for f in files:
    with safe_open(os.path.join(RD,f),framework="pt") as sf:
        for k in sf.keys(): where[k]=f
fc=safe_open(C,framework="pt"); ck=list(fc.keys())
def get(k):
    with safe_open(os.path.join(RD,where[k]),framework="pt") as sf: return sf.get_tensor(k)
changed=[k for k in ck if k.endswith(".weight") and fc.get_tensor(k).dim()==2
         and not torch.equal(fc.get_tensor(k), get(k).to(fc.get_tensor(k).dtype))]
print("changed 2-D weights:",len(changed))
deltas={}; vecs=[]
for k in changed:
    a=fc.get_tensor(k).float(); b=get(k).float(); d=a-b
    deltas[k]=d
    u,_,_=torch.linalg.svd(d,full_matrices=False); vecs.append(u[:,0].double())
D=torch.stack(vecs,0)
_,_,vh=torch.linalg.svd(D,full_matrices=False)
d=vh[0]
cos=float((D@d).abs().mean().clamp(max=1.0))
print("cross-layer |cos| of leading directions (n=%d): %.6f" % (len(changed), cos))
byout=collections.defaultdict(list)
for k,v in zip(changed,vecs): byout[v.shape[0]].append(v)
for dim,vs in sorted(byout.items()):
    M=torch.stack(vs,0); g=(M@M.T).abs(); n=g.shape[0]
    print("  out_features=%-5d n=%-3d mean|cos|=%.6f" % (dim,n,float((g.sum()-n)/(n*(n-1)))))
res=[]; floors=[]
for k in changed:
    W=get(k).float(); dk=deltas[k]
    proj=torch.outer(d.float(), d.float()@W)
    alpha=float(dk.norm()/proj.norm())
    rec=W-alpha*proj
    res.append(float((fc.get_tensor(k).float()-rec).norm()/W.norm()))
    floors.append(float((rec.to(torch.bfloat16).float()-rec).norm()/W.norm()))
print("\nrebuild W' = W - alpha d(d^T W):")
print("  residual median %.6f  max %.6f" % (statistics.median(res),max(res)))
print("  bf16 floor median %.6f  max %.6f" % (statistics.median(floors),max(floors)))
print("  ratio %.3f" % (statistics.median(res)/statistics.median(floors)))
al=[]
for k in changed:
    W=get(k).float(); proj=torch.outer(d.float(), d.float()@W); al.append(float(deltas[k].norm()/proj.norm()))
print("  alpha mean/min/max: %.4f / %.4f / %.4f" % (statistics.mean(al),min(al),max(al)))
print("\nchanged tensor roles:",dict(collections.Counter(k.split(".")[-2] for k in changed)))
print("layers:",len({int(k.split(".")[3]) for k in changed}))

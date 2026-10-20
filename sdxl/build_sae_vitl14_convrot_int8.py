"""Build SAE ViT-L-14 ConvRot INT8 pack at given thresholds (deterministic)."""
import sys, os, json, math
COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
sys.path.insert(0, COMFY)
os.chdir(COMFY)
import torch
import comfy.utils
from safetensors.torch import save_file

D = r"D:\USERFILES\ComfyUI\ComfyUI\models\clip"
SRC = os.path.join(D, "CLIP-SAE-ViT-L-14-FP32.safetensors")

impt = {}
for k, v in json.load(open(r"D:\USERFILES\GitHub\hswq\impact\impact_sae_vitl14_text.json", encoding="utf-8"))["impacts"].items():
    ck = "text_model." + k[len("transformer.text_model."):] if k.startswith("transformer.text_model.") else k
    impt[ck] = v
impv = {}
for k, v in json.load(open(r"D:\USERFILES\GitHub\hswq\impact\impact_sae_vitl14_vision.json", encoding="utf-8"))["impacts"].items():
    ck = "vision_model." + k if k.startswith("encoder.") else k
    impv[ck] = v


def had(sz):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    h = h4
    while h.shape[0] < sz:
        h = torch.kron(h, h4)
    return h / math.sqrt(sz)


_H = {}
def rot(w, gs=256):
    if gs not in _H:
        _H[gs] = had(gs)
    of, inf = w.shape; g = inf // gs
    return torch.matmul(w.view(of, g, gs), _H[gs].T).reshape(w.shape)


def qrow(w):
    a = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30); s = a / 127.0
    return (w / s).round().clamp(-127, 127).to(torch.int8), s.float()


sd = comfy.utils.load_torch_file(SRC, safe_load=True)


def build(thr_t, thr_v, path):
    prot = set([k for k, v in impt.items() if v > thr_t] + [k for k, v in impv.items() if v > thr_v])
    out = {}; meta = {}; nlin = 0; nprot = 0
    for k, t in sd.items():
        if not (k.endswith(".weight") and t.ndim == 2):
            out[k] = t; continue
        mk = k[:-7]; inf = int(t.shape[1])
        if mk in prot or inf % 256 != 0 or t.shape[0] % 4 != 0:
            out[k] = t; nprot += 1; continue
        q, s = qrow(rot(t.float(), 256))
        out[k] = q; out[mk + ".weight_scale"] = s
        out[mk + ".comfy_quant"] = torch.tensor(list(json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}).encode()), dtype=torch.uint8)
        meta[mk] = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}; nlin += 1
    save_file(out, path, metadata={"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": meta})})
    from collections import Counter
    side = Counter("text" if k.startswith("text_model") else ("vision" if k.startswith("vision_model") else "other") for k in meta)
    return nlin, nprot, os.path.getsize(path) / 1024**2, dict(side)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr_t", type=float, required=True)
    ap.add_argument("--thr_v", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n, p, mb, side = build(a.thr_t, a.thr_v, a.out)
    print("INT8=%d FP16=%d size=%.1f MB sides=%s" % (n, p, mb, side))

"""SD1.5 IP-Adapter Plus per-layer ConvRot INT8 impact.

The HSWQ method, on the network the IPAdapter node builds from this file:

  1. build `image_proj` (Resampler) and `ip_adapter` (one bias-free Linear per
     to_k_ip / to_v_ip weight) from the same state dict the node reads,
  2. run it once on a real clip-vision embedding and record the two stages,
  3. replace one Linear's weight with its ConvRot INT8 reconstruction, run again,
     record the relative MSE of the stage output that layer feeds,
  4. restore, next layer.

The quantisation replay is the converter's own: Hadamard by Kronecker doubling,
orthonormal, group size a power of four, row-wise `amax/127` int8.

  python clip_convert/diag_impact_ipadapter_sd15.py --ipadapter <f.pth> --image <png> --clipvision <safetensors> --out <impact.json> --ipadapter_dir <node dir>
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn as nn


def build_hadamard(size, device="cpu", dtype=torch.float32):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype, device=device)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def convrot_group_size(n, preferred=256):
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def rotate_weight(w, h, gs):
    of, inf = w.shape
    g = inf // gs
    return torch.matmul(w.view(of, g, gs), h.T.to(w.dtype)).reshape(w.shape)


def quant_rowwise(w):
    amax = w.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
    s = amax / 127.0
    q = (w / s.to(w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, s.to(torch.float32)


def convrot_int8_recon(w, gs):
    h = build_hadamard(gs, device=w.device)
    of, inf = w.shape
    g = inf // gs
    q, s = quant_rowwise(rotate_weight(w.float(), h, gs))
    deq = (q.float() * s.float()).view(of, g, gs)
    return torch.matmul(deq, h.float()).reshape(w.shape).to(w.dtype)


class ToKV(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.to_kvs = nn.ModuleDict()
        self.name_map = {}
        for key, value in state_dict.items():
            name = key.replace(".weight", "").replace(".", "_")
            self.name_map[name] = key.replace(".weight", "")
            self.to_kvs[name] = nn.Linear(value.shape[1], value.shape[0], bias=False)
            self.to_kvs[name].weight.data = value


def load_model(path, node_dir):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if not (isinstance(sd, dict) and "image_proj" in sd and "ip_adapter" in sd):
        raise SystemExit("unexpected layout: keys=%s" % (list(sd)[:4] if isinstance(sd, dict) else type(sd)))
    sys.path.insert(0, node_dir)
    from image_proj_models import Resampler

    res = Resampler(dim=768, depth=4, dim_head=64, heads=12, num_queries=16,
                    embedding_dim=1280, output_dim=768, ff_mult=4,
                    max_seq_len=257, apply_pos_emb=False, num_latents_mean_pooled=0)
    missing, unexpected = res.load_state_dict(sd["image_proj"], strict=False)
    if [k for k in missing if k.endswith(".weight")]:
        raise SystemExit("image_proj weight mismatch: missing=%s unexpected=%s" % (missing[:5], unexpected[:5]))
    kv = ToKV(sd["ip_adapter"])
    return res, kv


def clip_embedding(clipvision_path, image_path, comfy_root, device, dtype):
    sys.path.insert(0, comfy_root)
    import comfy.clip_vision
    import numpy as np
    from PIL import Image

    cv = comfy.clip_vision.load(clipvision_path)
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img).astype("float32") / 255.0
    t = torch.from_numpy(arr).unsqueeze(0)
    out = cv.encode_image(t)
    if isinstance(out, dict):
        out = out.get("last_hidden_state", list(out.values())[0])
    elif not torch.is_tensor(out):
        out = getattr(out, "last_hidden_state", None)
    if out is None:
        raise SystemExit("clip vision returned no tensor")
    if out.dim() == 2:
        out = out.unsqueeze(1)
    return out.to(device, dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipadapter", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--clipvision", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ipadapter_dir", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--groupsize", type=int, default=256)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print("[load] ipadapter: %s" % a.ipadapter, flush=True)
    res, kv = load_model(os.path.abspath(a.ipadapter), a.ipadapter_dir)
    res.to(device, dtype).eval()
    kv.to(device, dtype).eval()

    print("[load] clip vision: %s" % a.clipvision, flush=True)
    cond = clip_embedding(a.clipvision, a.image, a.comfy, device, dtype)
    print("[input] clip embedding %s finite %s" % (tuple(cond.shape), bool(torch.isfinite(cond).all())), flush=True)

    def run():
        with torch.no_grad():
            e = res(cond)
            outs = {n: lin(e) for n, lin in kv.to_kvs.items()}
        return e, outs

    e_ref, o_ref = run()
    print("[ref] image_proj %s | to_kv %d" % (tuple(e_ref.shape), len(o_ref)), flush=True)

    targets = []
    for n, m in res.named_modules():
        if isinstance(m, nn.Linear):
            targets.append(("image_proj." + n, m, "proj"))
    for n, m in kv.named_modules():
        if isinstance(m, nn.Linear):
            targets.append(("ip_adapter." + kv.name_map[n.replace("to_kvs.", "", 1)], m, "tokv",
                            n.replace("to_kvs.", "", 1)))
    print("[target] Linear modules: %d" % len(targets), flush=True)

    impacts = {}
    for key, mod, stage, *rest in targets:
        w0 = mod.weight.data
        gs = convrot_group_size(int(w0.shape[1]), a.groupsize)
        if gs is None:
            print("  [skip] %s: no group size for in=%d" % (key, w0.shape[1]), flush=True)
            continue
        with torch.no_grad():
            mod.weight.data = convrot_int8_recon(w0.float(), gs).to(w0.dtype)
            e_q, o_q = run()
            mod.weight.data = w0
        if stage == "proj":
            num = torch.sum((e_q - e_ref) ** 2)
            den = torch.sum(e_ref ** 2).clamp(min=1e-30)
        else:
            k = rest[0]
            num = torch.sum((o_q[k] - o_ref[k]) ** 2)
            den = torch.sum(o_ref[k] ** 2).clamp(min=1e-30)
        imp = float((num / den).item())
        impacts[key] = imp
        print("  [%d/%d] %s: rel_mse=%.6e" % (len(impacts), len(targets), key, imp), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"ipadapter": os.path.basename(a.ipadapter),
                   "stage": "image_proj + ip_adapter, stage-output rel_mse",
                   "input": "clip vision embedding of %s" % os.path.basename(a.image),
                   "impacts": impacts}, f, indent=2)
    ranked = sorted(impacts.items(), key=lambda x: x[1])
    print("\n== ascending drift ==", flush=True)
    for n, v in ranked[:8]:
        print("  %.6e  %s" % (v, n), flush=True)
    print("== descending drift ==", flush=True)
    for n, v in ranked[-8:]:
        print("  %.6e  %s" % (v, n), flush=True)
    print("wrote %s" % a.out, flush=True)


if __name__ == "__main__":
    main()

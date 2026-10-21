# -*- coding: utf-8 -*-
"""CLIP-ViT-bigG-14: the contribution of each layer, measured with real packs.

The in-model method (restore one layer's FP16 weight in the loaded pack and
measure) needs the FP16 source and the pack in memory at once. For bigG that
is 3519 MB + 1936 MB, and with the source's fp16 copies on the GPU it
exhausts the 16 GB VRAM (`CUDA error: invalid argument`, `Copy is larger than
memobj size`).

So the contribution is measured with real packs, one layer at a time: a pack
with that ONE layer in FP16 and every other layer quantized, compared against
the all-INT8 pack. The difference is the layer's contribution. The packs are
built and measured one at a time, so only one model is resident at a time.

This is slow (a pack per layer) but exact: it is the same measurement the
in-model method makes, through the real runtime path.
"""
import os
import sys
import json
import types
import importlib
import statistics
import subprocess
import torch

COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
NODE = os.path.join(COMFY, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools")
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
os.chdir(COMFY)
import folder_paths  # noqa: E402
import server as comfy_server  # noqa: E402
import comfy.utils  # noqa: E402
import comfy.clip_vision as cv  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.ops as O  # noqa: E402
from comfy.quant_ops import QUANT_ALGOS  # noqa: E402

if not hasattr(comfy_server.PromptServer, "instance"):
    comfy_server.PromptServer.instance = None
_spec = importlib.util.spec_from_file_location(
    "hswq_pkg_init", os.path.join(NODE, "__init__.py"), submodule_search_locations=[NODE])
_mod = importlib.util.module_from_spec(_spec)
sys.modules["hswq_pkg_init"] = _mod
_spec.loader.exec_module(_mod)

DT = torch.float16
N = int(os.environ.get("BIGG_N", "8"))
SRC = sys.argv[1]
IMP = sys.argv[2]
CONV = sys.argv[3]
WORK = sys.argv[4]
OUT = sys.argv[5]
os.makedirs(WORK, exist_ok=True)


def prep(sd, d):
    v = {k: x for k, x in sd.items() if k.startswith("vision_model.")}
    if "vision_model.embeddings.class_embedding" not in v:
        v["vision_model.embeddings.class_embedding"] = torch.zeros(d)
    return v


def class_dim(sd):
    for k, v in sd.items():
        if k.endswith("embeddings.class_embedding"):
            return int(v.shape[0])
    return 768


def build(jc, ops, sd, d):
    class CVQ(cv.ClipVisionModel):
        def __init__(self, jc, ops):
            c = json.load(open(jc))
            self.image_size = c.get("image_size", 224)
            self.image_mean = c.get("image_mean", [0.5, 0.5, 0.5])
            self.image_std = c.get("image_std", [0.5, 0.5, 0.5])
            self.model_type = c.get("model_type", "clip_vision_model")
            self.config = c.copy()
            mc = cv.IMAGE_ENCODERS.get(self.model_type)
            self.return_all_hidden_states = False
            self.load_device = comfy.model_management.text_encoder_device()
            off = comfy.model_management.text_encoder_offload_device()
            self.dtype = comfy.model_management.text_encoder_dtype(self.load_device)
            self.model = mc(c, self.dtype, off, ops)
            self.model.eval()
            self.patcher = comfy.model_patcher.CoreModelPatcher(
                self.model, load_device=self.load_device, offload_device=off)
            self.naf = None
    v = CVQ(jc, ops)
    v.load_sd(prep(sd, d))
    return v


def embeds(v, im):
    with torch.no_grad():
        out = v.encode_image(im, crop=False)
    if hasattr(out, "image_embeds") and torch.is_tensor(out.image_embeds):
        return out.image_embeds.detach().float().reshape(-1).cpu()
    if isinstance(out, dict):
        return torch.cat([out[k].detach().float().reshape(-1).cpu()
                          for k in sorted(out) if torch.is_tensor(out[k])])
    if torch.is_tensor(out):
        return out.detach().float().reshape(-1).cpu()
    return torch.cat([x.detach().float().reshape(-1).cpu() for x in out if torch.is_tensor(x)])


def main():
    ops = O.mixed_precision_ops({"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]},
                                DT, full_precision_mm=False)
    sd0 = comfy.utils.load_torch_file(SRC, safe_load=True)
    d = class_dim(sd0)
    jc = os.path.join(COMFY, "comfy", "clip_vision_config_g.json")
    if not os.path.isfile(jc):
        jc = os.path.join(COMFY, "comfy", "clip_vision_config_vitl.json")
    v16 = build(jc, ops, sd0, d)
    size = int(getattr(v16, "image_size", 224))

    def imgs(seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        return (torch.rand(1, 3, size, size, generator=g) * 2 - 1).to("cuda", dtype=DT)

    ref = [embeds(v16, imgs(1000 + s)) for s in range(N)]
    print("[1] the FP16 baseline: %s" % (tuple(ref[0].shape),), flush=True)

    net = getattr(v16, "model", v16)
    inner = getattr(net, "vision_model", net)
    cands = [n for n, mm in inner.named_modules()
             if type(mm).__name__ == "Linear" and getattr(mm, "weight", None) is not None]
    print("    the runtime Linear modules: %d" % len(cands), flush=True)
    del v16, sd0
    torch.cuda.empty_cache()

    def cos_of(path):
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        v = build(jc, ops, sd, d)
        cs = []
        for s in range(N):
            f = embeds(v, imgs(1000 + s))
            cs.append(torch.nn.functional.cosine_similarity(
                ref[s].reshape(1, -1), f.reshape(1, -1)).item())
        del v, sd
        torch.cuda.empty_cache()
        cs.sort()
        return statistics.mean(cs), cs[0]

    def build_pack(protect, tag):
        pl = os.path.join(WORK, "pl_%s.json" % tag)
        json.dump({"protect": sorted(protect)}, open(pl, "w", encoding="utf-8"))
        outp = os.path.join(WORK, "pk_%s.safetensors" % tag)
        cmd = [sys.executable, CONV, "--input", SRC, "--output", outp,
               "--impact", IMP, "--comfy_path", COMFY, "--clip_type", "vision",
               "--drift-threshold", "0.05", "--groupsize", "64", "--protect-list", pl]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            print("   the converter failed: %s" % r.stderr[-300:], flush=True)
            return None
        return outp

    print("[2] the all-INT8 reference ...", flush=True)
    base = build_pack(set(), "all")
    if base is None:
        raise SystemExit("the all-INT8 pack failed")
    b = cos_of(base)
    print("    the all-INT8: mean=%.6f min=%.6f" % b, flush=True)
    os.remove(base)

    print("[3] the per-layer contribution ...", flush=True)
    contrib = {}
    for i, n in enumerate(cands):
        p = build_pack({n}, "one")
        if p is None:
            contrib[n] = 0.0
            continue
        c = cos_of(p)
        os.remove(p)
        contrib[n] = c[0] - b[0]
        if (i + 1) % 20 == 0 or contrib[n] > 1e-5:
            print("   [%d/%d] %-52s the gain=%+.4e" % (i + 1, len(cands), n, contrib[n]), flush=True)

    json.dump({"contribution": contrib, "all_int8_mean": b[0], "all_int8_min": b[1], "n": N},
              open(OUT, "w", encoding="utf-8"), indent=1, sort_keys=True)
    print()
    print("saved: %s" % OUT, flush=True)
    print("== the ranked contribution ==")
    for n, val in sorted(contrib.items(), key=lambda kv: -kv[1])[:30]:
        print("   %-52s %+.4e" % (n, val))


if __name__ == "__main__":
    main()

"""CLIP-ViT-bigG-14: the contribution of each layer, measured in-model.

`measure_bigg_vision_contrib.py` measures the same quantity, but it builds a fresh
pack per candidate layer and reloads it, which costs about half a minute per layer -
288 packs, and the vision tower is rebuilt each time. That is exact but slow, and
the model is released only partially, so the VRAM fills up as the sweep proceeds.

This file measures the same thing with one pack resident:

  - the **all-INT8 pack** is built once and loaded once,
  - the FP16 baseline is its own embedding over N seeds,
  - for each Linear, that layer's weight is replaced by the source's FP16 weight
    (its `weight_scale` is taken out of the module for the duration) and the tower is
    run again over the same N seeds,
  - the layer's contribution is the gain in the worst cosine against the all-INT8 pack.

That is the same geometry the per-layer-pack method measures - one layer in FP16,
every other layer quantised, compared against the all-INT8 pack - with the pack
written to disk once instead of 288 times.

The all-INT8 pack is built with the repository's own converter, so the int8 payload
and the scales are the ones the shipped packs carry, and the kernels that run are
the real ConvRot ones.

  python clip_convert/measure_bigg_vision_contrib_inmodel.py <src> <impact.json> <converter.py> <workdir> <out.json>
"""

import json
import os
import statistics
import subprocess
import sys
import types
import importlib

import torch

COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
NODE = os.path.join(COMFY, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools")
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))

for _name, _path in (("ComfyUI-HSWQ-Loader-and-Tools", NODE),):
    if _name not in sys.modules:
        _m = types.ModuleType(_name)
        _m.__path__ = [_path]
        sys.modules[_name] = _m
_spec = importlib.util.spec_from_file_location(
    "ComfyUI-HSWQ-Loader-and-Tools.hswq.clip_vision_convrot_int8",
    os.path.join(NODE, "hswq", "clip_vision_convrot_int8.py"))
_mod = importlib.util.module_from_spec(_spec)
sys.modules["ComfyUI-HSWQ-Loader-and-Tools.hswq.clip_vision_convrot_int8"] = _mod
_spec.loader.exec_module(_mod)

import comfy.utils  # noqa: E402
import comfy.clip_vision as cv  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.ops as O  # noqa: E402
from comfy.quant_ops import QUANT_ALGOS  # noqa: E402

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


def build_all_int8_pack():
    pl = os.path.join(WORK, "pl_all.json")
    json.dump({"protect": []}, open(pl, "w", encoding="utf-8"))
    outp = os.path.join(WORK, "pk_all.safetensors")
    cmd = [sys.executable, CONV, "--input", SRC, "--output", outp,
           "--impact", IMP, "--comfy_path", COMFY, "--clip_type", "vision",
           "--drift-threshold", "0.05", "--groupsize", "64", "--protect-list", pl]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit("the converter failed: " + r.stderr[-2000:])
    return outp


def take_scale(mod):
    """Turn a ConvRot INT8 Linear back into a plain fp16 Linear.

    The scale is not a module attribute - it lives inside the `QuantizedTensor` that
    `weight` holds. But `mixed_precision_ops` also leaves `layout_type`, `quant_format`
    and `tensor_class` on the module, and the cast context reads them. Leaving them in
    place keeps the layer on the quantised path even though the weight is now a plain
    fp16 tensor, which is why the first version of this measurement produced gains that
    were mostly negative. All of it is taken out for the duration and put back after.
    """
    saved = {}
    for key in ("layout_type", "quant_format", "tensor_class"):
        if key in mod.__dict__:
            saved[key] = mod.__dict__.pop(key)
    return saved


def put_scale(mod, saved):
    mod.__dict__.update(saved)


def main():
    ops = O.mixed_precision_ops({"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]},
                                DT, full_precision_mm=False)
    src_sd = comfy.utils.load_torch_file(SRC, safe_load=True)
    d = class_dim(src_sd)
    jc = os.path.join(COMFY, "comfy", "clip_vision_config_g.json")
    if not os.path.isfile(jc):
        jc = os.path.join(COMFY, "comfy", "clip_vision_config_vitl.json")

    cfg = json.load(open(jc))
    size = int(cfg.get("image_size", 224))

    def imgs(seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        return (torch.rand(1, 3, size, size, generator=g) * 2 - 1).to("cuda", dtype=DT)

    def cos_list():
        cs = []
        for s in range(N):
            f = embeds(v, imgs(1000 + s))
            cs.append(torch.nn.functional.cosine_similarity(
                ref[s].reshape(1, -1), f.reshape(1, -1)).item())
        cs.sort()
        return cs

    # The baseline is the **FP16 source**, not the all-INT8 pack: the gain of a layer is
    # how much the pack's own output improves when that one layer is put back in FP16,
    # measured against the same reference the pack is measured against.
    print("[1] the FP16 source baseline ...", flush=True)
    v_src = build(jc, ops, src_sd, d)
    ref = []
    with torch.no_grad():
        for s in range(N):
            ref.append(embeds(v_src, imgs(1000 + s)))
    print("    the FP16 baseline: %s over %d seeds" % (tuple(ref[0].shape), N), flush=True)
    del v_src
    try:
        comfy.model_management.unload_all_models()
    except Exception as exc:
        print("   (unload_all_models: %s)" % exc, flush=True)

    print("[2] the all-INT8 pack ...", flush=True)
    pack_path = build_all_int8_pack()
    pack_sd = comfy.utils.load_torch_file(pack_path, safe_load=True)
    v = build(jc, ops, pack_sd, d)

    net = getattr(v, "model", v)
    inner = getattr(net, "vision_model", net)
    cands = [(n, m) for n, m in inner.named_modules()
             if type(m).__name__ == "Linear" and getattr(m, "weight", None) is not None]
    print("    the runtime Linear modules: %d" % len(cands), flush=True)

    base = cos_list()
    b_min, b_mean = base[0], statistics.mean(base)
    print("    the all-INT8: mean=%.6f min=%.6f" % (b_mean, b_min), flush=True)

    contrib = {}
    for i, (name, mod) in enumerate(cands):
        key = "vision_model." + name + ".weight"
        if key not in src_sd:
            contrib[name] = 0.0
            continue
        w = src_sd[key]
        orig_w = mod.weight
        saved = take_scale(mod)
        try:
            mod.weight = torch.nn.Parameter(w.to("cuda", DT), requires_grad=False)
            c = cos_list()
        finally:
            mod.weight = orig_w
            put_scale(mod, saved)
        contrib[name] = c[0] - b_min
        if (i + 1) % 20 == 0 or contrib[name] > 1e-5:
            print("   [%d/%d] %-52s the gain=%+.4e" % (i + 1, len(cands), name, contrib[name]),
                  flush=True)

    json.dump({"contribution": contrib, "all_int8_mean": b_mean, "all_int8_min": b_min,
               "n": N, "method": "in-model: the all-INT8 pack with one layer replaced by the "
                                "source's fp16 weight",
               "all_int8_pack": os.path.basename(pack_path)},
              open(OUT, "w", encoding="utf-8"), indent=1, sort_keys=True)
    print()
    print("saved: %s" % OUT, flush=True)
    print("== the ranked contribution ==", flush=True)
    for n, val in sorted(contrib.items(), key=lambda kv: -kv[1])[:30]:
        print("   %-52s %+.4e" % (n, val), flush=True)


if __name__ == "__main__":
    main()

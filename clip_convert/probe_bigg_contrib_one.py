"""Compare the two ways of putting one layer back in FP16, on the same checkpoint.

The per-layer-pack method (`measure_bigg_vision_contrib.py`) measures the gain by
building a pack whose only protected layer is that one, and measuring it. The in-model
method (`measure_bigg_vision_contrib_inmodel.py`) measures the gain by replacing that
one layer inside the loaded all-INT8 pack.

They are the same geometry, so they must agree. This file measures one layer both
ways, in one process, and prints the two numbers side by side.

  python clip_convert/probe_bigg_contrib_one.py <src> <impact.json> <converter.py> <workdir> <layer>
"""

import json
import os
import subprocess
import sys
import types
import importlib.util

import torch

COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
NODE = os.path.join(COMFY, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools")
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
if "ComfyUI-HSWQ-Loader-and-Tools" not in sys.modules:
    _m = types.ModuleType("ComfyUI-HSWQ-Loader-and-Tools")
    _m.__path__ = [NODE]
    sys.modules["ComfyUI-HSWQ-Loader-and-Tools"] = _m
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
SRC, IMP, CONV, WORK, LAYER = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
os.makedirs(WORK, exist_ok=True)


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
    vv = {k: x for k, x in sd.items() if k.startswith("vision_model.")}
    if "vision_model.embeddings.class_embedding" not in vv:
        vv["vision_model.embeddings.class_embedding"] = torch.zeros(d)
    v.load_sd(vv)
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


def make_pack(protect, tag):
    pl = os.path.join(WORK, "pl_%s.json" % tag)
    json.dump({"protect": sorted(protect)}, open(pl, "w", encoding="utf-8"))
    outp = os.path.join(WORK, "pk_%s.safetensors" % tag)
    cmd = [sys.executable, CONV, "--input", SRC, "--output", outp, "--impact", IMP,
           "--comfy_path", COMFY, "--clip_type", "vision", "--drift-threshold", "0.05",
           "--groupsize", "64", "--protect-list", pl]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit("the converter failed: " + r.stderr[-2000:])
    return outp


def main():
    ops = O.mixed_precision_ops({"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]}, DT,
                                full_precision_mm=False)
    src_sd = comfy.utils.load_torch_file(SRC, safe_load=True)
    d = 1664
    jc = os.path.join(COMFY, "comfy", "clip_vision_config_g.json")
    cfg = json.load(open(jc))
    size = int(cfg.get("image_size", 224))

    def imgs(seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        return (torch.rand(1, 3, size, size, generator=g) * 2 - 1).to("cuda", dtype=DT)

    def cos_list(v, ref):
        cs = []
        for s in range(N):
            f = embeds(v, imgs(1000 + s))
            cs.append(torch.nn.functional.cosine_similarity(
                ref[s].reshape(1, -1), f.reshape(1, -1)).item())
        cs.sort()
        return cs

    print("[1] the FP16 source baseline ...", flush=True)
    v_src = build(jc, ops, src_sd, d)
    ref = [embeds(v_src, imgs(1000 + s)) for s in range(N)]
    del v_src
    comfy.model_management.unload_all_models()

    print("[2] the all-INT8 pack ...", flush=True)
    all_path = make_pack(set(), "all")
    v_all = build(jc, ops, comfy.utils.load_torch_file(all_path, safe_load=True), d)
    base = cos_list(v_all, ref)
    print("    all-INT8: mean=%.6f min=%.6f" % (sum(base) / len(base), base[0]), flush=True)

    print("[3] the per-layer-pack method: one layer kept in FP16 ...", flush=True)
    one_path = make_pack({LAYER}, "one")
    v_one = build(jc, ops, comfy.utils.load_torch_file(one_path, safe_load=True), d)
    one = cos_list(v_one, ref)
    print("    the one-layer pack: mean=%.6f min=%.6f | the gain=%+.4e"
          % (sum(one) / len(one), one[0], one[0] - base[0]), flush=True)
    del v_one
    comfy.model_management.unload_all_models()

    print("[4] the in-model method: the same layer replaced inside the loaded pack ...", flush=True)
    v_all = build(jc, ops, comfy.utils.load_torch_file(all_path, safe_load=True), d)
    net = getattr(v_all, "model", v_all)
    inner = getattr(net, "vision_model", net)
    mods = dict(inner.named_modules())
    mod = mods[LAYER]
    orig_w = mod.weight
    saved = {}
    for key in ("layout_type", "quant_format", "tensor_class"):
        if key in mod.__dict__:
            saved[key] = mod.__dict__.pop(key)
    try:
        mod.weight = torch.nn.Parameter(src_sd["vision_model." + LAYER + ".weight"].to("cuda", DT),
                                       requires_grad=False)
        in_model = cos_list(v_all, ref)
    finally:
        mod.weight = orig_w
        mod.__dict__.update(saved)
    print("    the in-model swap: mean=%.6f min=%.6f | the gain=%+.4e"
          % (sum(in_model) / len(in_model), in_model[0], in_model[0] - base[0]), flush=True)
    print()
    print("the two gains: per-layer-pack %+.4e | in-model %+.4e | the difference %+.4e"
          % (one[0] - base[0], in_model[0] - base[0],
             (in_model[0] - base[0]) - (one[0] - base[0])), flush=True)


if __name__ == "__main__":
    main()

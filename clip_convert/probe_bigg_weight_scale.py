"""Where does a ConvRot INT8 Linear keep its `weight_scale`, and does removing it work?

The in-model contribution measurement replaces one layer's weight with the source's
fp16 weight. If the layer's `weight_scale` is still attached, the layer is not a plain
fp16 Linear any more and the measurement is not what it claims to be. This prints
where the scale actually lives and whether it can be taken out.

  python clip_convert/probe_bigg_weight_scale.py <src> <impact.json> <converter.py> <workdir>
"""

import importlib.util
import json
import os
import subprocess
import sys
import types

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
SRC, IMP, CONV, WORK = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
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


pl = os.path.join(WORK, "pl_all.json")
json.dump({"protect": []}, open(pl, "w", encoding="utf-8"))
outp = os.path.join(WORK, "pk_all.safetensors")
r = subprocess.run([sys.executable, CONV, "--input", SRC, "--output", outp, "--impact", IMP,
                    "--comfy_path", COMFY, "--clip_type", "vision", "--drift-threshold", "0.05",
                    "--groupsize", "64", "--protect-list", pl],
                   capture_output=True, text=True, encoding="utf-8")
if r.returncode != 0:
    raise SystemExit(r.stderr[-2000:])

ops = O.mixed_precision_ops({"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]}, DT,
                            full_precision_mm=False)
sd = comfy.utils.load_torch_file(outp, safe_load=True)
d = 1664
jc = os.path.join(COMFY, "comfy", "clip_vision_config_g.json")
v = build(jc, ops, sd, d)
net = getattr(v, "model", v)
inner = getattr(net, "vision_model", net)

for name, mod in inner.named_modules():
    if type(mod).__name__ != "Linear":
        continue
    print("layer:", name)
    print("  type          :", type(mod).__name__)
    print("  weight dtype  :", getattr(mod.weight, "dtype", None),
          "| is QuantizedTensor:", type(mod.weight).__name__)
    print("  has weight_scale attr:", hasattr(mod, "weight_scale"))
    print("  weight_scale in _parameters:", "weight_scale" in mod._parameters)
    print("  weight_scale in _buffers   :", "weight_scale" in mod._buffers)
    print("  weight_scale in __dict__   :", "weight_scale" in mod.__dict__)
    print("  getattr(mod,'weight_scale') :", type(getattr(mod, "weight_scale", None)).__name__)
    print("  _parameters keys:", sorted(mod._parameters.keys()))
    print("  _buffers keys   :", sorted(mod._buffers.keys()))
    print("  __dict__ extras :", sorted(k for k in mod.__dict__ if k not in ("_parameters", "_buffers", "_modules", "_non_persistent_buffers_set", "_backward_hooks", "_forward_hooks", "_forward_hooks_with_kwargs", "_state_dict_hooks", "_state_dict_hooks_with_kwargs", "_load_state_dict_pre_hooks", "_load_state_dict_post_hooks", "_is_full_backward_hook", "_non_persistent_buffers_set", "_backward_pre_hooks", "training")))
    break

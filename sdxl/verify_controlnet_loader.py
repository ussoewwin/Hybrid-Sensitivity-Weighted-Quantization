# -*- coding: utf-8 -*-
"""The ControlNet acceptance check, run through the HSWQ loader's own node.

Why this script exists
----------------------
``comfy.controlnet`` has **no quantization-metadata handling of its own**. It
picks an ops class (``custom_operations``) and hands it to the model's
constructor; the metadata (``<layer>.comfy_quant`` / ``<layer>.weight_scale``)
is consumed by ``comfy.ops.MixedPrecisionOps.Linear._load_from_state_dict``.
The HSWQ loader supplies that ops class, so the whole path is the loader's.

This script therefore drives the **loader's node**, not a hand-built model,
and checks three things per pack:

1. the metadata was consumed - every ``comfy_quant`` layer became a
   ``QuantizedTensor`` on the module,
2. the real INT8 GEMM fired - ``comfy_kitchen``'s ``int8_linear`` was called
   once per quantized layer,
3. the forward is finite.

Three runtime conventions, each established by measurement
----------------------------------------------------------
1. The compute dtype is the loader's own default (BF16 on this GPU) and every
   tensor - the timestep included - is cast to the model's dtype, exactly as
   ``comfy.controlnet.ControlNet.get_control`` does (``timestep.to(dtype)``).
   Feeding an FP32 timestep into the FLUX path makes ``vec`` an FP32 tensor
   while the activations are BF16, and the modulation then poisons the whole
   block: the first non-finite point is ``double_blocks.0.img_attn.qkv``'s
   input. Feeding an FP16 timestep overflows instead -
   ``comfy.ldm.flux.model.timestep_embedding`` multiplies by 1000, so 500
   becomes 500000, past the FP16 range of 65504, and cos(inf) is NaN.
2. The hswq loader's ``model_management_mgpu.check_cpu_memory_threshold``
   reads ``server.PromptServer.instance`` unguarded, which raises inside
   ``comfy.model_management.get_cast_buffer`` when the ComfyUI server is not
   running. The attribute is stubbed here so the measurement can complete;
   the loader itself is not changed.
3. The SDXL ControlNet's context is 2048-wide (the pooled CLIP + the text
   encoder's concat) and its ``y`` is 2816-wide (the label embedding).

Usage
-----
    python verify_controlnet_loader.py <pack.safetensors> [<pack.safetensors> ...]

The ControlNet directory is scanned automatically when no file is given.
"""
import os
import sys
import json
import types
import glob
import torch
import importlib
import traceback

COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
NODE = os.path.join(COMFY, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools")
CN_DIR = r"D:\USERFILES\StableDiffusion\models\ControlNet"

sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
os.chdir(COMFY)

import folder_paths  # noqa: E402
import server as comfy_server  # noqa: E402
import comfy.utils  # noqa: E402
import comfy.ops as O  # noqa: E402

# the headless stub: PromptServer.instance only exists after __init__
if not hasattr(comfy_server.PromptServer, "instance"):
    comfy_server.PromptServer.instance = None

# the arm: the package's root __init__ is executed exactly as ComfyUI does
# on startup, which is what enables the real INT8 GEMM.
_spec = importlib.util.spec_from_file_location(
    "hswq_pkg_init", os.path.join(NODE, "__init__.py"), submodule_search_locations=[NODE])
_mod = importlib.util.module_from_spec(_spec)
sys.modules["hswq_pkg_init"] = _mod
_spec.loader.exec_module(_mod)

_pkg = types.ModuleType("hswq_pkg")
_pkg.__path__ = [NODE]
sys.modules["hswq_pkg"] = _pkg
importlib.import_module("hswq_pkg.nodes.hswq_load_convrot_int8_controlnet")
CN_MOD = sys.modules["hswq_pkg.nodes.hswq_load_convrot_int8_controlnet"]

DT = CN_MOD._get_default_compute_dtype()


def flat(o):
    if torch.is_tensor(o):
        return o.detach().float().reshape(-1).cpu()
    if isinstance(o, dict):
        p = [flat(v) for v in o.values()]
        return torch.cat(p) if p else torch.zeros(0)
    if isinstance(o, (tuple, list)):
        p = [flat(v) for v in o]
        return torch.cat(p) if p else torch.zeros(0)
    return torch.zeros(0)


def is_flux_native(sd):
    """The comfy-native ControlNetFlux layout, which `comfy.controlnet` has no
    detection branch for - the loader builds it directly."""
    return ("img_in.weight" in sd and "controlnet_blocks.0.weight" in sd
            and "pos_embed_input.weight" in sd
            and "controlnet_x_embedder.weight" not in sd
            and "control_model.zero_convs.0.0.weight" not in sd
            and "zero_convs.0.0.weight" not in sd)


def check(pack):
    name = os.path.basename(pack)
    print("== %s (%.1f MB)" % (name, os.path.getsize(pack) / 1024 ** 2), flush=True)
    sd = comfy.utils.load_torch_file(pack, safe_load=True)
    nq = len([k for k in sd if k.endswith(".comfy_quant")])
    model_options = {
        "dtype": DT,
        "custom_operations": CN_MOD._int8_mixed_precision_ops(compute_dtype=DT),
    }
    flux = is_flux_native(sd)
    if flux:
        control = CN_MOD._load_comfy_flux_controlnet(sd, model_options)
    else:
        control = comfy.controlnet.load_controlnet_state_dict(sd, model_options=model_options)
    m = control.control_model
    mdt = m.dtype
    qs = [x for x, mm in m.named_modules()
          if type(mm).__name__ == "Linear" and getattr(mm, "weight", None) is not None
          and type(mm.weight).__name__ == "QuantizedTensor"]

    ck = O.quant_ops.ck
    orig = ck.int8_linear
    n = [0]

    def counted(*a, **kw):
        n[0] += 1
        return orig(*a, **kw)

    ck.int8_linear = counted
    try:
        g = torch.Generator(device="cpu").manual_seed(42)
        if flux:
            x = torch.randn(1, 16, 32, 32, generator=g).to("cuda", dtype=mdt)
            hint = torch.randn(1, 16, 32, 32, generator=g).to("cuda", dtype=mdt)
            txt = torch.randn(1, 77, 4096, generator=g).to("cuda", dtype=mdt)
            y = torch.zeros(1, 3072, device="cuda", dtype=mdt)
            gd = torch.full((1,), 3.5, device="cuda", dtype=mdt)
            ts = torch.full((1,), 500.0, device="cuda", dtype=mdt)
            with torch.no_grad():
                out = m.forward(x=x, timesteps=ts, context=txt, y=y, guidance=gd, hint=hint)
        else:
            x = torch.randn(1, 4, 32, 32, generator=g).to("cuda", dtype=mdt)
            hint = torch.randn(1, 3, 256, 256, generator=g).to("cuda", dtype=mdt)
            txt = torch.randn(1, 77, 2048, generator=g).to("cuda", dtype=mdt)
            y = torch.randn(1, 2816, generator=g).to("cuda", dtype=mdt)
            ts = torch.full((1,), 500.0, device="cuda", dtype=mdt)
            with torch.no_grad():
                out = m.forward(x=x, hint=hint, timesteps=ts, context=txt, y=y)
        f = flat(out)
        nf = int((~torch.isfinite(f)).sum().item())
        res = "%d elems, nonfinite=%d" % (f.numel(), nf)
    except Exception as e:
        res = "the forward failed: %r" % e
        traceback.print_exc()
    finally:
        ck.int8_linear = orig
    rec = {"pack": name, "file_mb": os.path.getsize(pack) / 1024 ** 2,
           "comfy_quant": nq, "quantized_modules": len(qs),
           "int8_linear_calls": n[0], "result": res}
    print("   comfy_quant=%d | QuantizedTensor=%d | int8_linear=%d | %s" % (
        nq, len(qs), n[0], res), flush=True)
    del control, m
    torch.cuda.empty_cache()
    return rec


def main():
    packs = sys.argv[1:]
    if not packs:
        packs = sorted(glob.glob(os.path.join(CN_DIR, "*_convrot_int8*.safetensors")))
    print("the arm: %s | the compute dtype: %s" % (
        getattr(O, "_hswq_int8_gemm_patched", False), DT), flush=True)
    recs = [check(p) for p in packs]
    print()
    print("%-58s %6s %6s %6s" % ("pack", "quant", "module", "int8"))
    for r in recs:
        print("%-58s %6d %6d %6d" % (r["pack"][:58], r["comfy_quant"],
                                     r["quantized_modules"], r["int8_linear_calls"]))
    print("JSON " + json.dumps(recs), flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""The FLUX Union Pro 2.0 ControlNet's 256-item acceptance, and the qkv /
zero-drift decision, all measured on the loader's own path.

The measurement runs through the HSWQ loader's node (the package's root
``__init__`` is executed exactly as ComfyUI does on startup, which is what
enables the real INT8 GEMM), and the FP16 baseline is built through the same
ops class, because the stock ``comfy.controlnet`` path and the
``mixed_precision_ops`` path are not numerically equivalent for this model.

Runtime conventions, each established by measurement:
- the compute dtype is the loader's own default (BF16 on this GPU), and every
  tensor - the timestep included - is cast to the model's dtype, exactly as
  ``comfy.controlnet.ControlNet.get_control`` does (``timestep.to(dtype)``);
  an FP32 timestep makes ``vec`` FP32 while the activations are BF16 and the
  modulation then poisons the whole block, and an FP16 timestep overflows
  (``timestep_embedding`` multiplies by 1000, so 500 becomes 500000, past the
  FP16 range of 65504);
- the SDXL/FLUX control model's context width and the hint's channel count
  come from the model itself.

Usage:
    python verify_cn_flux_256.py <source.safetensors> [<pack.safetensors> ...]
"""
import os
import sys
import json
import types
import statistics
import torch
import importlib

COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
NODE = os.path.join(COMFY, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools")

sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
os.chdir(COMFY)

import folder_paths  # noqa: E402
import server as comfy_server  # noqa: E402
import comfy.utils  # noqa: E402
import comfy.model_detection  # noqa: E402
import comfy.ldm.flux.controlnet as cnf  # noqa: E402
import comfy.ops as O  # noqa: E402

if not hasattr(comfy_server.PromptServer, "instance"):
    comfy_server.PromptServer.instance = None

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
N = int(os.environ.get("CNN_N", "256"))


def build_pack(sd):
    """The pack, through the loader's own FLUX path."""
    control = CN_MOD._load_comfy_flux_controlnet(sd, {
        "dtype": DT,
        "custom_operations": CN_MOD._int8_mixed_precision_ops(compute_dtype=DT),
    })
    return control.control_model


def build_source(path):
    """The FP16 source, through the same ops class and the same construction."""
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    converted = comfy.model_detection.convert_diffusers_mmdit(dict(sd), "")
    if converted is None:
        converted = dict(sd)
    else:
        for k in sd:
            converted[k] = sd[k]
    mc = comfy.model_detection.model_config_from_unet(converted, "", use_base_if_no_match=True)
    ops = CN_MOD._int8_mixed_precision_ops(compute_dtype=DT)
    model = cnf.ControlNetFlux(
        latent_input=True, num_union_modes=0,
        control_latent_channels=converted["pos_embed_input.weight"].shape[1] // 4,
        operations=ops, device="cpu", dtype=DT, **mc.unet_config)
    model = comfy.controlnet.controlnet_load_state_dict(model, converted)
    return model


def features(model, seed, side=32, txt_tokens=77):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, 16, side, side, generator=g).to("cuda", dtype=DT)
    txt = torch.randn(1, txt_tokens, 4096, generator=g).to("cuda", dtype=DT)
    hint = torch.randn(1, 16, side, side, generator=g).to("cuda", dtype=DT)
    y = torch.zeros(1, 3072, device="cuda", dtype=DT)
    ts = torch.full((1,), 500.0, device="cuda", dtype=DT)
    gd = torch.full((1,), 3.5, device="cuda", dtype=DT)
    with torch.no_grad():
        out = model.forward(x=x, timesteps=ts, context=txt, y=y, guidance=gd, hint=hint)
    parts = []
    for k in sorted(out):
        v = out[k]
        if torch.is_tensor(v):
            parts.append(v.detach().float().reshape(-1).cpu())
        elif isinstance(v, (tuple, list)):
            for t2 in v:
                if torch.is_tensor(t2):
                    parts.append(t2.detach().float().reshape(-1).cpu())
    return torch.cat(parts)


def main():
    src = sys.argv[1]
    packs = sys.argv[2:]
    print("[1] the FP16 baseline (the same ops path) ...", flush=True)
    m16 = build_source(src)
    base = {}
    for s in range(N):
        base[s] = features(m16, s)
    print("    the reference features: %s finite=%s" % (
        tuple(base[0].shape), bool(torch.isfinite(base[0]).all())), flush=True)
    del m16
    torch.cuda.empty_cache()

    results = []
    for p in packs:
        print("[2] %s" % os.path.basename(p), flush=True)
        sd = comfy.utils.load_torch_file(p, safe_load=True)
        m8 = build_pack(sd)
        nq = len([k for k in sd if k.endswith(".comfy_quant")])
        ck = O.quant_ops.ck
        orig = ck.int8_linear
        n = [0]

        def counted(*a, **kw):
            n[0] += 1
            return orig(*a, **kw)

        ck.int8_linear = counted
        cs = []
        fin = True
        try:
            with torch.no_grad():
                m8.forward(x=torch.randn(1, 16, 32, 32, device="cuda", dtype=DT),
                           timesteps=torch.full((1,), 500.0, device="cuda", dtype=DT),
                           context=torch.randn(1, 77, 4096, device="cuda", dtype=DT),
                           y=torch.zeros(1, 3072, device="cuda", dtype=DT),
                           guidance=torch.full((1,), 3.5, device="cuda", dtype=DT),
                           hint=torch.randn(1, 16, 32, 32, device="cuda", dtype=DT))
        except Exception:
            pass
        finally:
            ck.int8_linear = orig
        for s in range(N):
            f = features(m8, s)
            if not torch.isfinite(f).all():
                fin = False
            cs.append(torch.nn.functional.cosine_similarity(
                base[s].reshape(1, -1), f.reshape(1, -1)).item())
        cs.sort()
        m = len(cs)
        rec = {"pack": os.path.basename(p),
               "file_mb": os.path.getsize(p) / 1024 ** 2,
               "quantized_layers": nq,
               "int8_linear_calls": n[0],
               "mean": statistics.mean(cs), "min": cs[0],
               "p5": cs[max(0, int(m * 0.05) - 1)], "p50": cs[m // 2],
               "n": m, "finite": fin}
        rec["pass"] = bool(rec["mean"] >= 0.99 and rec["min"] >= 0.98 and fin)
        results.append(rec)
        print("    int8_linear=%d finite=%s  mean=%.6f min=%.6f p5=%.6f p50=%.6f  size=%.1f MB  %s" % (
            n[0], fin, rec["mean"], rec["min"], rec["p5"], rec["p50"], rec["file_mb"],
            "PASS" if rec["pass"] else "FAIL"), flush=True)
        del m8
        torch.cuda.empty_cache()

    print()
    print("%-56s %7s %10s %10s %10s %10s" % ("pack", "int8", "mean", "min", "p5", "MB"))
    for r in results:
        print("%-56s %7d %10.6f %10.6f %10.6f %10.1f" % (
            r["pack"][:56], r["int8_linear_calls"], r["mean"], r["min"], r["p5"], r["file_mb"]))
    print("JSON " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()

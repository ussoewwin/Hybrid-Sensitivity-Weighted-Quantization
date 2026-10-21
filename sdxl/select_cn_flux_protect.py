# -*- coding: utf-8 -*-
"""The FLUX ControlNet: the FP16-kept layer set, chosen by the AGGREGATE
contribution measured through the real pack path.

Why the aggregate and not the per-layer drift
--------------------------------------------
The per-layer drift is a marginal effect and does not rank the layers the way
the aggregate quality does. With the corrected (un-rotated) injection the
FLUX ControlNet's per-layer drifts are all ~1e-04, yet the packs measure very
differently over 256 items:

| pack | INT8 | FP16 kept | cos mean | cos min | size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| the shipped one | 54 | 21 | 0.999937 | 0.999160 | 2567.2 MB |
| the all-but-qkv one | 63 | 12 | 0.999051 | 0.991817 | 2368.2 MB |
| the all-INT8 one | 75 | 0 | 0.999046 | 0.991294 | 2044.6 MB |

So the layers that matter are picked by their effect on the aggregate output.

This measures that effect directly, in the real pack path: load the all-INT8
pack, and for each layer restore the FP16 weight and measure the aggregate
cosine against the FP16 baseline. The improvement is the layer's contribution.

Then it adds the layers back greedily, largest contribution first, and reports
the aggregate for each prefix - the frontier the pack can choose from.

Usage:
    python select_cn_flux_protect.py <all_int8_pack> <source> <out.json> [n_items]
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

_pkg = types.ModuleType("hswq_pkg")
_pkg.__path__ = [NODE]
sys.modules["hswq_pkg"] = _pkg
importlib.import_module("hswq_pkg.nodes.hswq_load_convrot_int8_controlnet")
CN_MOD = sys.modules["hswq_pkg.nodes.hswq_load_convrot_int8_controlnet"]

DT = CN_MOD._get_default_compute_dtype()
POOLED = r"D:\USERFILES\GitHub\hswq\impact\flux_pooled_sample.pt"

PACK = sys.argv[1]
SRC = sys.argv[2]
OUT = sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 16


def build_source(path):
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


Y = torch.load(POOLED, map_location="cpu").float().to("cuda", dtype=DT)


def features(model, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, 16, 32, 32, generator=g).to("cuda", dtype=DT)
    txt = torch.randn(1, 77, 4096, generator=g).to("cuda", dtype=DT)
    hint = torch.randn(1, 16, 32, 32, generator=g).to("cuda", dtype=DT)
    ts = torch.full((1,), 500.0, device="cuda", dtype=DT)
    gd = torch.full((1,), 3.5, device="cuda", dtype=DT)
    with torch.no_grad():
        out = model.forward(x=x, timesteps=ts, context=txt, y=Y, guidance=gd, hint=hint)
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


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()


print("[1] the FP16 baseline ...", flush=True)
m16 = build_source(SRC)
ref = {s: features(m16, s) for s in range(N)}
print("    the reference: %s finite=%s" % (
    tuple(ref[0].shape), bool(torch.isfinite(ref[0]).all())), flush=True)

print("[2] the all-INT8 pack ...", flush=True)
psd = comfy.utils.load_torch_file(PACK, safe_load=True)
control = CN_MOD._load_comfy_flux_controlnet(psd, {
    "dtype": DT, "custom_operations": CN_MOD._int8_mixed_precision_ops(compute_dtype=DT)})
m8 = control.control_model
qmods = {n: mm for n, mm in m8.named_modules()
         if type(mm).__name__ == "Linear" and getattr(mm, "weight", None) is not None
         and type(mm.weight).__name__ == "QuantizedTensor"}
m16mods = dict(m16.named_modules())
print("    the quantized modules: %d" % len(qmods), flush=True)

# the all-INT8 aggregate
allq = {s: cos(ref[s], features(m8, s)) for s in range(N)}
base_mean = statistics.mean(allq.values())
print("    the all-INT8: mean=%.6f min=%.6f" % (base_mean, min(allq.values())), flush=True)

print("[3] the per-layer contribution (restore one layer, measure the aggregate) ...", flush=True)
contrib = {}
fp16w = {}
for n in sorted(qmods):
    fp16w[n] = m16mods[n].weight.detach().to("cuda", DT)
    mm = qmods[n]
    saved = mm.weight
    mm.weight = torch.nn.Parameter(fp16w[n], requires_grad=False)
    cs = [cos(ref[s], features(m8, s)) for s in range(N)]
    mm.weight = saved
    contrib[n] = statistics.mean(cs) - base_mean
print("    the measured %d layers" % len(contrib), flush=True)

# the greedy frontier: restore the layers with the largest contribution first
order = sorted(contrib.items(), key=lambda kv: -kv[1])
print()
print("== the greedy frontier ==")
print("   %-6s %-46s %10s %10s" % ("kept", "the layer added", "mean", "min"))
frontier = []
for k, (n, v) in enumerate(order):
    mm = qmods[n]
    mm.weight = torch.nn.Parameter(fp16w[n], requires_grad=False)
    if k % 4 == 0 or k == len(order) - 1:
        cs = sorted(cos(ref[s], features(m8, s)) for s in range(N))
        print("   %-6d %-46s %10.6f %10.6f" % (k + 1, n, statistics.mean(cs), cs[0]))
        frontier.append({"kept": k + 1, "layer": n, "mean": statistics.mean(cs), "min": cs[0]})

json.dump({"contribution": contrib, "order": [n for n, _ in order],
           "all_int8_mean": base_mean,
           "all_int8_min": min(allq.values()),
           "n": N}, open(OUT, "w", encoding="utf-8"), indent=1, sort_keys=True)
print()
print("saved: %s" % OUT, flush=True)

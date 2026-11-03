# -*- coding: utf-8 -*-
"""The SAM 3.1 Multiplex: the FP16-kept set, chosen by the AGGREGATE
contribution measured through the real pack path.

Why the aggregate and not the per-layer drift
---------------------------------------------
The per-layer drift (even with the corrected, un-rotated injection) measures
only the WEIGHT error: the reconstruction is placed in the module and the
plain forward is used. The pack's real path additionally quantizes the
ACTIVATION for that layer, so the layer's true cost is larger than the weight
error alone.

Measured on the SAM 3.1 Multiplex:

| pack | the FP16-kept | the INT8 | the relMSE | the size |
| :--- | :--- | :--- | :--- | :--- |
| the shipped one | 180 | 41 | **3.87e-05** | 1657.6 MB |
| the all-but-5 one | 5 | 183 | 1.11e-02 | 1370.5 MB |

The per-layer drift said every layer was equivalent, and that selection cost
286x in output error. The aggregate is the quantity the pack trades off, so
it is the quantity that has to be measured.

This script loads the all-INT8 pack through the real loader path, restores
one layer's FP16 weight at a time, and measures the aggregate. The layers are
then added back greedily, largest contribution first, and the frontier is
reported.

Usage:
    python select_sam31_protect.py <all_int8_pack> <source> <out.json> [n_items]
"""
import os
import sys
import json
import types
import statistics
import importlib

import torch

COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
NODE = os.path.join(COMFY, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools")
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
os.chdir(COMFY)

import folder_paths  # noqa: E402
import server as comfy_server  # noqa: E402
import comfy.utils  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.ops as comfy_ops  # noqa: E402
from comfy.quant_ops import QUANT_ALGOS  # noqa: E402

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
importlib.import_module("hswq_pkg.patches.comfy_quant_int8")
PATCH = sys.modules["hswq_pkg.patches.comfy_quant_int8"]
PATCH.apply_comfy_quant_int8_patches()

SIZE = 1008
DEVICE = "cuda"
N = int(sys.argv[4]) if len(sys.argv) > 4 else 4
PACK = sys.argv[1]
SRC = sys.argv[2]
OUT = sys.argv[3]


def build_model(path):
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    ops = comfy_ops.mixed_precision_ops(
        {"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]}, torch.bfloat16,
        full_precision_mm=False, disabled=[])
    out = comfy.sd.load_diffusion_model_state_dict(sd, model_options={"custom_operations": ops})
    if out is None:
        raise SystemExit("the loader returned None")
    net = getattr(out, "model", out)
    net = net.to(DEVICE).eval()
    return getattr(net, "diffusion_model", net)


def make_inputs(seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    img = torch.rand(1, SIZE, SIZE, 3, generator=g, dtype=torch.float32)
    mdt = next((p.dtype for p in DTYPES), torch.float16)
    img = (img * 255).to(device=DEVICE, dtype=mdt)
    frame = img[..., :3].movedim(-1, 1).to(device=DEVICE, dtype=mdt)
    point_inputs = {"point_coords": torch.tensor([[[256.0, 256.0]]], device=DEVICE, dtype=mdt),
                    "point_labels": torch.tensor([[1]], device=DEVICE, dtype=torch.int32)}
    g2 = torch.Generator(device="cpu").manual_seed(seed + 1)
    text_emb = (torch.rand(1, 32, 1024, generator=g2) * 2 - 1).to(device=DEVICE, dtype=mdt)
    text_mask = torch.ones(1, 32, dtype=torch.int64, device=DEVICE)
    return frame, point_inputs, text_emb, text_mask


DTYPES = []


def run(sam3, seed):
    frame, point_inputs, text_emb, text_mask = make_inputs(seed)
    with torch.no_grad():
        seg = sam3.forward_segment(frame, point_inputs=point_inputs)
        det = sam3(frame, text_embeddings=text_emb, text_mask=text_mask,
                   threshold=0.5, orig_size=(SIZE, SIZE))
    parts = [seg.detach().float().reshape(-1).cpu()]
    for key in ("masks", "boxes", "scores"):
        v = det.get(key) if isinstance(det, dict) else None
        if v is None:
            continue
        if torch.is_tensor(v):
            parts.append(v.detach().float().reshape(-1).cpu())
        elif isinstance(v, (tuple, list)):
            for t2 in v:
                if torch.is_tensor(t2):
                    parts.append(t2.detach().float().reshape(-1).cpu())
    return torch.cat(parts)


def relmse(a, b):
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


def main():
    global DTYPES
    print("[1] the FP16 baseline ...", flush=True)
    m16 = build_model(SRC)
    DTYPES = [p for p in m16.parameters() if p.is_floating_point()]
    ref = {s: run(m16, s) for s in range(N)}
    print("    the reference: %s finite=%s" % (
        tuple(ref[0].shape), bool(torch.isfinite(ref[0]).all())), flush=True)
    # the FP16 weights of the runtime's Linear modules
    m16mods = {n: mm for n, mm in m16.named_modules()
               if type(mm).__name__ == "Linear" and getattr(mm, "weight", None) is not None}
    fp16w = {n: mm.weight.detach().to(DEVICE) for n, mm in m16mods.items()}
    del m16
    torch.cuda.empty_cache()

    print("[2] the all-INT8 pack ...", flush=True)
    m8 = build_model(PACK)
    qmods = {n: mm for n, mm in m8.named_modules()
             if type(mm).__name__ == "Linear" and getattr(mm, "weight", None) is not None
             and type(mm.weight).__name__ == "QuantizedTensor"}
    print("    the quantized modules: %d" % len(qmods), flush=True)
    base = {s: relmse(ref[s], run(m8, s)) for s in range(N)}
    base_mean = statistics.mean(base.values())
    print("    the all-INT8: the relMSE=%.4e" % base_mean, flush=True)

    print("[3] the per-layer contribution ...", flush=True)
    contrib = {}
    for n in sorted(qmods):
        if n not in fp16w:
            contrib[n] = 0.0
            continue
        mm = qmods[n]
        saved = mm.weight
        mm.weight = torch.nn.Parameter(fp16w[n], requires_grad=False)
        cs = [relmse(ref[s], run(m8, s)) for s in range(N)]
        mm.weight = saved
        contrib[n] = base_mean - statistics.mean(cs)
        if contrib[n] > 0:
            print("   %-62s %+.4e" % (n, contrib[n]), flush=True)

    order = sorted(contrib.items(), key=lambda kv: -kv[1])
    print()
    print("== the greedy frontier (the layers are restored in this order) ==")
    print("   %-6s %-62s %12s" % ("kept", "the layer", "the relMSE"))
    frontier = []
    for k, (n, v) in enumerate(order):
        mm = qmods[n]
        saved = mm.weight
        if n in fp16w:
            mm.weight = torch.nn.Parameter(fp16w[n], requires_grad=False)
        cs = [relmse(ref[s], run(m8, s)) for s in range(N)]
        mm.weight = saved
        if k % 5 == 0 or k == len(order) - 1:
            print("   %-6d %-62s %12.4e" % (k + 1, n, statistics.mean(cs)), flush=True)
            frontier.append({"kept": k + 1, "layer": n, "relmse": statistics.mean(cs)})

    json.dump({"contribution": contrib, "order": [n for n, _ in order],
               "all_int8_relmse": base_mean, "n": N},
              open(OUT, "w", encoding="utf-8"), indent=1, sort_keys=True)
    print()
    print("saved: %s" % OUT, flush=True)


if __name__ == "__main__":
    main()

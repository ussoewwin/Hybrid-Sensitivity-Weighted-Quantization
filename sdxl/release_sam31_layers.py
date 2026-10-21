# -*- coding: utf-8 -*-
"""The SAM 3.1 Multiplex: release the FP16-kept layers, cheapest first.

The direction
-------------
The shipped pack keeps 180 of the 221 runtime-live Linear layers in FP16 and
quantizes 41. It measures the relMSE 3.87e-05 against the FP16 source.

The re-measured per-layer drift (the corrected, un-rotated injection) says
almost every layer is equivalent - the largest drift is 0.289 and the median
is 1.9e-06 - and a pack built from that (5 kept, 183 quantized) measures
1.11e-02, i.e. **286x worse**. So the per-layer drift does not rank the
layers the way the output does, and a smaller kept set is not automatically
better.

The question is therefore "how many of the kept layers may be released before
the output degrades". This measures the release cost of each kept layer
directly, in the shipped pack's own runtime path:

- the layer's weight is replaced by the ConvRot INT8 `QuantizedTensor` built
  from its FP16 source weight (the same math the converter uses),
- the aggregate relMSE is measured,
- the layer is restored.

The cost is the increase over the shipped pack's relMSE. The layers are then
released cheapest-first, and each candidate is verified with a real pack.

Usage:
    python release_sam31_layers.py <source> <shipped_pack> <contrib.json> <out.json> [n_items]
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
from comfy.quant_ops import QUANT_ALGOS, TensorWiseINT8Layout  # noqa: E402
from comfy_kitchen.tensor.base import QuantizedTensor  # noqa: E402

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
GATE = float(os.environ.get("SAM_GATE", "1.0e-04"))

SRC = sys.argv[1]
PACK = sys.argv[2]
CONTRIB = sys.argv[3]
OUT = sys.argv[4]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 2

DTYPES = []


def build_hadamard(size, dtype=torch.float32):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


H_CACHE = {}


def had(gs):
    if gs not in H_CACHE:
        H_CACHE[gs] = build_hadamard(gs)
    return H_CACHE[gs]


def build_model(path):
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    ops = comfy_ops.mixed_precision_ops(
        {"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]}, torch.bfloat16,
        full_precision_mm=False, disabled=[])
    out = comfy.sd.load_diffusion_model_state_dict(sd, model_options={"custom_operations": ops})
    net = getattr(out, "model", out)
    net = net.to(DEVICE).eval()
    return getattr(net, "diffusion_model", net)


def make_inputs(seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    img = torch.rand(1, SIZE, SIZE, 3, generator=g, dtype=torch.float32)
    mdt = next((p.dtype for p in DTYPES), torch.float16)
    img = (img * 255).to(device=DEVICE, dtype=mdt)
    frame = img[..., :3].movedim(-1, 1).to(device=DEVICE, dtype=mdt)
    pi = {"point_coords": torch.tensor([[[256.0, 256.0]]], device=DEVICE, dtype=mdt),
          "point_labels": torch.tensor([[1]], device=DEVICE, dtype=torch.int32)}
    g2 = torch.Generator(device="cpu").manual_seed(seed + 1)
    te = (torch.rand(1, 32, 1024, generator=g2) * 2 - 1).to(device=DEVICE, dtype=mdt)
    tm = torch.ones(1, 32, dtype=torch.int64, device=DEVICE)
    return frame, pi, te, tm


def run(sam3, seed):
    frame, pi, te, tm = make_inputs(seed)
    with torch.no_grad():
        seg = sam3.forward_segment(frame, point_inputs=pi)
        det = sam3(frame, text_embeddings=te, text_mask=tm,
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


def quantize_tensor(w, gs):
    """The ConvRot INT8 QuantizedTensor, the same math the converter uses."""
    w = w.detach().to("cpu").float()
    of, inf = w.shape
    g = inf // gs
    h = had(gs)
    rot = torch.matmul(w.view(of, g, gs), h.T).reshape(w.shape)
    amax = rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    s = amax / 127.0
    q = (rot / s).round().clamp(-127, 127).to(torch.int8)
    from comfy.quant_ops import TensorWiseINT8Layout
    params = TensorWiseINT8Layout.Params(
        scale=s.float(), convrot=True, convrot_groupsize=gs,
        orig_dtype=w.dtype, orig_shape=(of, inf))
    return QuantizedTensor(q, "TensorWiseINT8Layout", params)


def main():
    global DTYPES
    print("[1] the FP16 baseline ...", flush=True)
    m16 = build_model(SRC)
    DTYPES = [p for p in m16.parameters() if p.is_floating_point()]
    ref = {s: run(m16, s) for s in range(N)}
    fp16w = {}
    for n, mm in m16.named_modules():
        if type(mm).__name__ == "Linear" and getattr(mm, "weight", None) is not None:
            fp16w[n] = mm.weight.detach().to(DEVICE)
    del m16
    torch.cuda.empty_cache()
    print("    the reference: %s" % (tuple(ref[0].shape),), flush=True)

    print("[2] the shipped pack ...", flush=True)
    m8 = build_model(PACK)
    allmods = {n: mm for n, mm in m8.named_modules()
               if type(mm).__name__ == "Linear" and getattr(mm, "weight", None) is not None}
    qmods = {n: mm for n, mm in allmods.items()
             if type(mm.weight).__name__ == "QuantizedTensor"}
    kept = [n for n in allmods if n not in qmods]
    print("    the INT8: %d | the FP16-kept: %d" % (len(qmods), len(kept)), flush=True)

    def measure():
        return statistics.mean([relmse(ref[s], run(m8, s)) for s in range(N)])

    base = measure()
    print("    the shipped pack: the relMSE=%.4e (the gate %.1e)" % (base, GATE), flush=True)

    # the release candidates: the kept layers whose input width allows the
    # ConvRot group size 256 (the runtime's requirement)
    cands = []
    for n in kept:
        w = fp16w.get(n)
        if w is None:
            continue
        if int(w.shape[1]) % 256 != 0:
            continue
        cands.append(n)
    print("    the release candidates (256-wide): %d" % len(cands), flush=True)

    print("[3] the release cost of each candidate ...", flush=True)
    cost = {}
    for i, n in enumerate(cands):
        mm = allmods[n]
        saved = mm.weight.detach().clone()
        try:
            qt = quantize_tensor(saved, 256)
            mm.weight = torch.nn.Parameter(qt, requires_grad=False)
            mm.quant_format = "int8_tensorwise"
            mm.layout_type = "TensorWiseINT8Layout"
            mm._full_precision_mm = False
            r = measure()
        except Exception as e:
            r = float("inf")
            print("   %-62s ERROR %r" % (n, e), flush=True)
        finally:
            mm.weight = torch.nn.Parameter(saved, requires_grad=False)
            mm.quant_format = None
        cost[n] = r - base
        if (i + 1) % 20 == 0:
            print("   [%d/%d] the last: %-56s the cost=%.4e" % (i + 1, len(cands), n, cost[n]), flush=True)
        if cost[n] < 0:
            print("   %-62s the cost=%.4e (a release would IMPROVE it)" % (n, cost[n]), flush=True)

    json.dump({"cost": cost, "the shipped relmse": base, "the gate": GATE, "n": N},
              open(OUT, "w", encoding="utf-8"), indent=1, sort_keys=True)
    print()
    print("saved: %s" % OUT, flush=True)
    order = sorted(cost.items(), key=lambda kv: kv[1])
    print("== the cheapest releases ==")
    for n, c in order[:30]:
        print("   %-62s %+.4e" % (n, c))


if __name__ == "__main__":
    main()

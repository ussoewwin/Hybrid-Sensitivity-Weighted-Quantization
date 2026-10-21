# -*- coding: utf-8 -*-
"""The SAM 3.1 Multiplex ConvRot INT8 acceptance, on the real loader path.

The observable is the model's own output, and there are two paths:

- the **tracker** SAM-decoder path (a point prompt) - `forward_segment`,
- the **detector** path (a text prompt) - its masks, boxes and scores.

Both are measured against the FP16 source, over a fixed item set (seed 42).
The tracker mask is compared by the relative MSE and by the pixel flip rate
(the thresholded sign), the detector path by the relative MSE of the
concatenated masks / boxes / scores.

The pack has to be loaded the way `HSWQ Load ConvRot INT8 SAM3 Model` loads
it: `apply_comfy_quant_int8_patches()` plus
`MixedPrecisionOps(int8_tensorwise)`. A plain
`comfy.sd.load_diffusion_model_state_dict(sd, model_options={})` does not arm
the ConvRot patch, so even a correct pack comes back as NaN or an inverted
mask - measuring through the wrong path is the easiest way to misdiagnose a
pack.

Usage:
    python verify_sam31.py <source.safetensors> [<pack.safetensors> ...]
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

N = int(os.environ.get("SAM_N", "256"))
SIZE = 1008
DEVICE = "cuda"


def build_model(path):
    """The SAM3 model, loaded the way the HSWQ SAM3 loader loads it."""
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    ops = comfy_ops.mixed_precision_ops(
        {"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]}, torch.bfloat16,
        full_precision_mm=False, disabled=[])
    out = comfy.sd.load_diffusion_model_state_dict(sd, model_options={"custom_operations": ops})
    if out is None:
        raise SystemExit("comfy.sd.load_diffusion_model_state_dict returned None")
    net = getattr(out, "model", out)
    net = net.to(DEVICE).eval()
    sam3 = getattr(net, "diffusion_model", net)
    return sam3


def make_inputs(seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    img = torch.rand(1, SIZE, SIZE, 3, generator=g, dtype=torch.float32)
    mdt = next((p.dtype for p in SAM3_DTYPES), torch.float16)
    img = (img * 255).to(device=DEVICE, dtype=mdt)
    frame = img[..., :3].movedim(-1, 1).to(device=DEVICE, dtype=mdt)
    point_inputs = {"point_coords": torch.tensor([[[256.0, 256.0]]], device=DEVICE, dtype=mdt),
                    "point_labels": torch.tensor([[1]], device=DEVICE, dtype=torch.int32)}
    g2 = torch.Generator(device="cpu").manual_seed(seed + 1)
    text_emb = (torch.rand(1, 32, 1024, generator=g2) * 2 - 1).to(device=DEVICE, dtype=mdt)
    text_mask = torch.ones(1, 32, dtype=torch.int64, device=DEVICE)
    return frame, point_inputs, text_emb, text_mask


SAM3_DTYPES = []


def run(sam3, seed):
    """Both paths' outputs, concatenated - the same observable the impact
    measurement uses."""
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
    return torch.cat(parts), seg.detach().float()


def relmse(a, b):
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


def flip(a, b):
    return float(((a > 0) != (b > 0)).float().mean())


def main():
    global SAM3_DTYPES
    src = sys.argv[1]
    packs = sys.argv[2:]

    print("[1] the FP16 baseline ...", flush=True)
    m16 = build_model(src)
    SAM3_DTYPES = [p for p in m16.parameters() if p.is_floating_point()]
    ref = {}
    for s in range(N):
        ref[s] = run(m16, s)
    print("    the reference: %s finite=%s" % (
        tuple(ref[0][0].shape), bool(torch.isfinite(ref[0][0]).all())), flush=True)
    del m16
    torch.cuda.empty_cache()

    results = []
    for p in packs:
        print("[2] %s" % os.path.basename(p), flush=True)
        m8 = build_model(p)
        ck = comfy_ops.quant_ops.ck
        orig = ck.int8_linear
        cnt = [0]

        def counted(*a, **kw):
            cnt[0] += 1
            return orig(*a, **kw)

        ck.int8_linear = counted
        try:
            cs = []
            fl = []
            rl = []
            fin = True
            for s in range(N):
                f, seg = run(m8, s)
                if not torch.isfinite(f).all():
                    fin = False
                    break
                cs.append(torch.nn.functional.cosine_similarity(
                    ref[s][0].reshape(1, -1), f.reshape(1, -1)).item())
                fl.append(flip(ref[s][1], seg))
                rl.append(relmse(ref[s][0], f))
        finally:
            ck.int8_linear = orig
        cs.sort()
        m = len(cs)
        rec = {"pack": os.path.basename(p),
               "file_mb": os.path.getsize(p) / 1024 ** 2,
               "int8_linear_calls": cnt[0],
               "cos_mean": statistics.mean(cs) if cs else 0.0,
               "cos_min": cs[0] if cs else 0.0,
               "cos_p5": cs[max(0, int(m * 0.05) - 1)] if cs else 0.0,
               "flip": statistics.mean(fl) if fl else None,
               "relmse": statistics.mean(rl) if rl else None,
               "finite": fin, "n": m}
        rec["pass"] = bool(rec["cos_mean"] >= 0.99 and rec["cos_min"] >= 0.98 and fin)
        results.append(rec)
        print("    int8_linear=%d finite=%s  cos: mean=%.6f min=%.6f p5=%.6f  the flip=%.5f  the relMSE=%.3e  size=%.1f MB  %s" % (
            cnt[0], fin, rec["cos_mean"], rec["cos_min"], rec["cos_p5"],
            rec["flip"] or -1, rec["relmse"] or -1, rec["file_mb"],
            "PASS" if rec["pass"] else "FAIL"), flush=True)
        del m8
        torch.cuda.empty_cache()

    print()
    print("%-50s %7s %10s %10s %10s %10s" % ("pack", "int8", "mean", "min", "p5", "MB"))
    for r in results:
        print("%-50s %7d %10.6f %10.6f %10.6f %10.1f" % (
            r["pack"][:50], r["int8_linear_calls"], r["cos_mean"], r["cos_min"],
            r["cos_p5"], r["file_mb"]))
    print("JSON " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()

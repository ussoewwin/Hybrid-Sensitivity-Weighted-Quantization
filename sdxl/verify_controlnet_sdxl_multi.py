# -*- coding: utf-8 -*-
"""SDXL ControlNet: the acceptance gate over MULTIPLE control hints.

A single random hint gives a cosine whose run-to-run noise (~1e-5) is the same
size as the differences between neighbouring protection thresholds, so a
one-hint measurement cannot place the size/quality knee. This runs N hints
(fixed seeds) and reports the mean and the minimum, which is what the
threshold choice needs.

Everything else matches verify_controlnet_sdxl_256.py: the pack is loaded
through the hswq loader's own node with the INT8 arm active, and the FP16
baseline is built on the same ops path.
"""
import argparse
import importlib
import json
import os
import statistics
import sys
import types

import torch

COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
NODE = os.path.join(COMFY, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools")
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
os.chdir(COMFY)
import comfy.utils  # noqa: E402
import comfy.ops as O  # noqa: E402
import server as comfy_server  # noqa: E402

if not hasattr(comfy_server.PromptServer, "instance"):
    comfy_server.PromptServer.instance = None

_spec = importlib.util.spec_from_file_location(
    "hswq_pkg_init", os.path.join(NODE, "__init__.py"), submodule_search_locations=[NODE])
_mod = importlib.util.module_from_spec(_spec)
sys.modules["hswq_pkg_init"] = _mod
_spec.loader.exec_module(_mod)

pkg = types.ModuleType("hswq_pkg")
pkg.__path__ = [NODE]
sys.modules["hswq_pkg"] = pkg
importlib.import_module("hswq_pkg.nodes.hswq_load_convrot_int8_controlnet")
CN_MOD = sys.modules["hswq_pkg.nodes.hswq_load_convrot_int8_controlnet"]

DT = CN_MOD._get_default_compute_dtype()
N_HINTS = int(os.environ.get("CN_N", "12"))
SEED0 = 1000


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


def run(sd):
    model_options = {"dtype": DT,
                     "custom_operations": CN_MOD._int8_mixed_precision_ops(compute_dtype=DT)}
    control = comfy.controlnet.load_controlnet_state_dict(sd, model_options=model_options)
    m = control.control_model
    mdt = m.dtype
    outs = []
    for i in range(N_HINTS):
        g = torch.Generator(device="cpu").manual_seed(SEED0 + i)
        x = torch.randn(1, 4, 32, 32, generator=g).to("cuda", dtype=mdt)
        hint = torch.randn(1, 3, 256, 256, generator=g).to("cuda", dtype=mdt)
        txt = torch.randn(1, 77, 2048, generator=g).to("cuda", dtype=mdt)
        y = torch.randn(1, 2816, generator=g).to("cuda", dtype=mdt)
        ts = torch.full((1,), 500.0, device="cuda", dtype=mdt)
        with torch.no_grad():
            outs.append(flat(m.forward(x=x, hint=hint, timesteps=ts, context=txt, y=y)))
    del control, m
    torch.cuda.empty_cache()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--packs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    print("the arm: %s | the compute dtype: %s | the hints: %d" % (
        getattr(O, "_hswq_int8_gemm_patched", False), DT, N_HINTS), flush=True)
    sd0 = comfy.utils.load_torch_file(a.src, safe_load=True)
    ref = run(sd0)
    nf0 = sum(int((~torch.isfinite(f)).sum().item()) for f in ref)
    print("[1] the FP16 baseline: %d hints, %d elems each, nonfinite=%d" % (
        N_HINTS, ref[0].numel(), nf0), flush=True)

    recs = []
    for p in a.packs:
        sd = comfy.utils.load_torch_file(p, safe_load=True)
        f = run(sd)
        cos = [torch.nn.functional.cosine_similarity(
            ref[i].reshape(1, -1), f[i].reshape(1, -1)).item() for i in range(N_HINTS)]
        relmse = [float(((f[i] - ref[i]) ** 2).sum() / (ref[i] ** 2).sum()) for i in range(N_HINTS)]
        cs = sorted(cos)
        nq = len([k for k in sd if k.endswith(".comfy_quant")])
        rec = {"pack": os.path.basename(p), "file_mb": os.path.getsize(p) / 1024 ** 2,
               "comfy_quant": nq, "cos_mean": statistics.mean(cos), "cos_min": cs[0],
               "cos_max": cs[-1], "relmse_mean": statistics.mean(relmse),
               "nonfinite": sum(int((~torch.isfinite(x)).sum().item()) for x in f)}
        recs.append(rec)
        print("  %-34s cos mean=%.6f min=%.6f | relMSE=%.3e | %.1f MB" % (
            os.path.basename(p)[:34], rec["cos_mean"], rec["cos_min"],
            rec["relmse_mean"], rec["file_mb"]), flush=True)
        del sd, f
        torch.cuda.empty_cache()

    json.dump(recs, open(a.out, "w", encoding="utf-8"), indent=1)
    print()
    print("%-36s %10s %10s %10s %9s" % ("pack", "cos.mean", "cos.min", "relMSE", "MB"))
    for r in recs:
        print("%-36s %10.6f %10.6f %10.3e %9.1f" % (
            r["pack"][:36], r["cos_mean"], r["cos_min"], r["relmse_mean"], r["file_mb"]))


if __name__ == "__main__":
    main()

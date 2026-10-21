# -*- coding: utf-8 -*-
"""SDXL ControlNet: the acceptance gate on the real loader path.

Loads a pack through the hswq loader's own node (so the INT8 arm is active,
exactly as ComfyUI does on startup), builds the control model, runs a fixed
forward, and compares the output against the FP16 source's output on the SAME
path. The gate is the cosine over the output, plus the finiteness of every
element.

The FP16 baseline is built through the same loader with no quantization
metadata, so the two sides differ only in the pack.
"""
import argparse
import glob
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


def run(sd, use_meta, seed=42):
    model_options = {"dtype": DT,
                     "custom_operations": CN_MOD._int8_mixed_precision_ops(compute_dtype=DT)}
    control = comfy.controlnet.load_controlnet_state_dict(sd, model_options=model_options)
    m = control.control_model
    mdt = m.dtype
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, 4, 32, 32, generator=g).to("cuda", dtype=mdt)
    hint = torch.randn(1, 3, 256, 256, generator=g).to("cuda", dtype=mdt)
    txt = torch.randn(1, 77, 2048, generator=g).to("cuda", dtype=mdt)
    y = torch.randn(1, 2816, generator=g).to("cuda", dtype=mdt)
    ts = torch.full((1,), 500.0, device="cuda", dtype=mdt)
    with torch.no_grad():
        out = m.forward(x=x, hint=hint, timesteps=ts, context=txt, y=y)
    f = flat(out)
    nq = len([k for k in sd if k.endswith(".comfy_quant")])
    nf = int((~torch.isfinite(f)).sum().item())
    del control, m
    torch.cuda.empty_cache()
    return f, nq, nf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--packs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    print("the arm: %s | the compute dtype: %s" % (getattr(O, "_hswq_int8_gemm_patched", False), DT), flush=True)
    sd0 = comfy.utils.load_torch_file(a.src, safe_load=True)
    ref, _, nf0 = run(sd0, False)
    print("[1] the FP16 baseline: %d elems, nonfinite=%d" % (ref.numel(), nf0), flush=True)

    recs = []
    for p in a.packs:
        sd = comfy.utils.load_torch_file(p, safe_load=True)
        f, nq, nf = run(sd, True)
        cos = torch.nn.functional.cosine_similarity(ref.reshape(1, -1), f.reshape(1, -1)).item()
        relmse = float(((f - ref) ** 2).sum() / (ref ** 2).sum())
        rec = {"pack": os.path.basename(p), "file_mb": os.path.getsize(p) / 1024 ** 2,
               "comfy_quant": nq, "cos": cos, "relmse": relmse, "nonfinite": nf}
        recs.append(rec)
        print("  %-40s cos=%.6f relMSE=%.3e nonfinite=%d %.1f MB" % (
            os.path.basename(p)[:40], cos, relmse, nf, rec["file_mb"]), flush=True)
        del sd, f
        torch.cuda.empty_cache()

    json.dump(recs, open(a.out, "w", encoding="utf-8"), indent=1)
    print()
    print("%-42s %8s %10s %8s %10s" % ("pack", "cos", "relMSE", "nonfin", "MB"))
    for r in recs:
        print("%-42s %8.6f %10.3e %8d %10.1f" % (r["pack"][:42], r["cos"], r["relmse"],
                                                 r["nonfinite"], r["file_mb"]))


if __name__ == "__main__":
    main()

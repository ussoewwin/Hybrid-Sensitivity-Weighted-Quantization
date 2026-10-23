# -*- coding: utf-8 -*-
"""SD1.5 UNet ConvRot INT8: the acceptance on the fixed 256-item set, through
the REAL HSWQ SD1.5 loader.

Why the loader matters - and why the earlier harness reported NaN:

  * `comfy_kitchen` has no INT8 conv kernel. The Conv2d INT8 path exists only in
    the HSWQ loader's `_patch_ops_decode_and_conv()`, and it is gated on a
    thread-local scope (`_should_inject_int8_conv`).
  * There are **two vendored copies** of `comfy_quant_int8` (under `patches/` and
    under `nodes/sd1.5 int8/`), each with its own thread-local. The SD1.5 node
    opens **every** scope it can find in `sys.modules`.
  * Open only one - or none - and the Conv2d stays a float conv, receives an
    INT8 weight, and the output is NaN (measured: 16384 NaN, `int8_linear` 0).

So the acceptance drives `load_checkpoint_sd15_hswq_weight_dtype`. Confirmed on
the pack: Linear 177 + **Conv2d 85** INT8, output finite.

CLIP is not quantized; the VAE is absent from the pack (`output_vae=False`).

Usage:
  python sd1.5/verify_sd15_loader.py <pack1> [<pack2> ...]
"""
from __future__ import annotations

import importlib.util
import json
import os
import statistics
import sys

# --- the ComfyUI root ------------------------------------------------------
# This script must run INSIDE a ComfyUI checkout (it imports comfy.ops,
# comfy.model_detection, ...). COMFY_PATH may be set in the environment, or the
# first argument may be `--comfy <path>`; there is no default, so a wrong
# machine cannot silently pick up someone else's tree.
def _resolve_comfy(argv):
    import os, sys
    path = ""
    if "--comfy" in argv:
        i = argv.index("--comfy")
        if i + 1 < len(argv):
            path = argv[i + 1]
            del argv[i:i + 2]
    if not path:
        path = os.environ.get("COMFY_PATH", "")
    if not path:
        sys.exit(
            "error: the ComfyUI root is not set.\n"
            "  pass --comfy <path> or set COMFY_PATH=<path>\n"
            "  (the directory must contain comfy/ops.py)")
    path = os.path.abspath(path)
    if not os.path.isdir(os.path.join(path, "comfy")):
        sys.exit("error: %s does not look like a ComfyUI root (no comfy/)" % path)
    return path

_COMFY_ARGV = list(sys.argv)
COMFY = _resolve_comfy(_COMFY_ARGV)
sys.argv = _COMFY_ARGV
# ---------------------------------------------------------------------------
NODE = os.path.join(COMFY, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools")
if COMFY not in sys.path:
    sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
os.chdir(COMFY)

import server as cs  # noqa: E402
if not hasattr(cs.PromptServer, "instance"):
    cs.PromptServer.instance = None

import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "hswq_pkg_init", os.path.join(NODE, "__init__.py"),
    submodule_search_locations=[NODE])
_mod = importlib.util.module_from_spec(_spec)
sys.modules["hswq_pkg_init"] = _mod
_spec.loader.exec_module(_mod)

_folder = os.path.join(NODE, "nodes", "sd1.5 int8")
_s = importlib.util.spec_from_file_location(
    "hswq_sd15_node", os.path.join(_folder, "__init__.py"),
    submodule_search_locations=[_folder])
SD15 = importlib.util.module_from_spec(_s)
sys.modules["hswq_sd15_node"] = SD15
_s.loader.exec_module(SD15)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
LATENT = 64


def load(path):
    model, _clip = SD15.load_checkpoint_sd15_hswq_weight_dtype(path, device=DEV)
    # the loader builds on CPU; the unet must be resident on DEV before a
    # forward, otherwise the matmul sees cuda input against cpu weights
    import comfy.model_management as mm
    mm.load_models_gpu([model])
    model.model.diffusion_model.to(DEV)
    model.model.diffusion_model.eval()
    return model


def make_inputs(seed, latent=LATENT, ctx=77, dim=768):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((1, 4, latent, latent), generator=g).to(DEV, dtype=torch.float16)
    ts = torch.tensor([500.0], device=DEV, dtype=torch.float16)
    context = torch.randn((1, ctx, dim), generator=g).to(DEV, dtype=torch.float16)
    return x, ts, context


def unet_output(model, x, ts, context):
    dm = model.model.diffusion_model
    with torch.no_grad():
        out = dm(x, ts, context)
    return out.detach().float().double().reshape(-1).cpu()


def count_int8(model):
    dm = model.model.diffusion_model
    nl = sum(1 for _n, m in dm.named_modules()
             if type(m).__name__ == "Linear" and getattr(m, "quant_format", None))
    nc = sum(1 for _n, m in dm.named_modules()
             if type(m).__name__ == "Conv2d" and getattr(m, "quant_format", None))
    return nl, nc


def main():
    # CLI: <source.ckpt> <pack> [<pack> ...]
    #   --n N        the number of items (default 256)
    #   --out FILE   write the results JSON
    argv = sys.argv[1:]
    n = 256
    out_path = ""
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == "--n" and i + 1 < len(argv):
            n = int(argv[i + 1]); i += 2; continue
        if argv[i] == "--out" and i + 1 < len(argv):
            out_path = argv[i + 1]; i += 2; continue
        rest.append(argv[i]); i += 1
    if len(rest) < 2:
        sys.exit("usage: verify_sd15_loader.py --comfy <root> <source.ckpt> <pack> [<pack> ...] [--n 256] [--out results.json]")
    src = rest[0]
    packs = rest[1:]

    print("[1] the baseline (FP16 source, through the same loader) | the items: %d ..." % n, flush=True)
    m0 = load(src)
    batches = [make_inputs(SEED + i) for i in range(n)]
    ref = [unet_output(m0, *b) for b in batches]
    fin0 = bool(torch.isfinite(ref[0]).all())
    print("    the reference: %s finite=%s" % (tuple(ref[0].shape), fin0), flush=True)
    del m0
    torch.cuda.empty_cache()

    results = []
    for p in packs:
        print("[2] %s" % os.path.basename(p), flush=True)
        model = load(p)
        nl, nc = count_int8(model)
        print("    the INT8: Linear %d | Conv2d %d" % (nl, nc), flush=True)
        cs_, fin = [], True
        for i in range(n):
            f = unet_output(model, *batches[i])
            if not torch.isfinite(f).all():
                fin = False
            cs_.append(torch.nn.functional.cosine_similarity(
                ref[i].reshape(1, -1), f.reshape(1, -1), dim=1).item())
        del model
        torch.cuda.empty_cache()
        cs_.sort()
        m = len(cs_)
        rec = {"pack": os.path.basename(p), "file_mb": os.path.getsize(p) / 1024 ** 2,
               "int8_linear": nl, "int8_conv": nc,
               "cos_mean": statistics.mean(cs_), "cos_min": cs_[0],
               "p5": cs_[max(0, int(m * 0.05) - 1)], "n": m, "finite": fin}
        rec["pass"] = bool(rec["cos_mean"] >= 0.99 and rec["cos_min"] >= 0.98 and fin)
        results.append(rec)
        print("    finite=%s mean=%.8f min=%.8f p5=%.8f size=%.1f MB %s" % (
            fin, rec["cos_mean"], rec["cos_min"], rec["p5"], rec["file_mb"],
            "PASS" if rec["pass"] else "FAIL"), flush=True)

    if out_path:
        json.dump(results, open(out_path, "w", encoding="utf-8"), indent=1)
        print("saved: %s" % out_path)
    print()
    print("  %-30s %6s %6s %14s %14s %9s" % ("pack", "lin", "conv", "mean", "min", "MB"))
    for r in results:
        print("  %-30s %6d %6d %14.8f %14.8f %9.1f" % (
            r["pack"].replace("sd15_", "")[:30], r["int8_linear"], r["int8_conv"],
            r["cos_mean"], r["cos_min"], r["file_mb"]))


if __name__ == "__main__":
    main()

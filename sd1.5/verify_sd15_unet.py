# -*- coding: utf-8 -*-
"""SD1.5 UNet ConvRot INT8: the acceptance on the fixed 256-item set.

The gate is the UNet's OWN output - the same instrument the protection set was
chosen with (Rules 2/3).

The baseline is built through the SAME ops class as the quantized pack
(`mixed_precision_ops`): the stock path uses `manual_cast`, and the two are not
numerically equivalent. **CLIP is not quantized** and does not participate.

Usage:
  python sd1.5/verify_sd15_unet.py <source>.safetensors <pack1> [<pack2> ...]
"""
from __future__ import annotations

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
if COMFY not in sys.path:
    sys.path.insert(0, COMFY)
os.chdir(COMFY)

import torch  # noqa: E402

import comfy.utils  # noqa: E402
import comfy.ops as O  # noqa: E402
import comfy.model_detection as md  # noqa: E402
import comfy.quant_ops  # noqa: E402

# the arm: the loader package, as ComfyUI loads it on startup
import importlib  # noqa: E402
import types  # noqa: E402
NODE = os.path.join(COMFY, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools")
if os.path.isdir(NODE):
    if NODE not in sys.path:
        sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
    _spec = importlib.util.spec_from_file_location(
        "hswq_pkg_init", os.path.join(NODE, "__init__.py"),
        submodule_search_locations=[NODE])
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["hswq_pkg_init"] = _mod
    _spec.loader.exec_module(_mod)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float16
SEED = 42
LATENT = 64
UNET_PREFIX = "model.diffusion_model."


def build_unet(ckpt_path, ops):
    sd = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    prefix = md.unet_prefix_from_state_dict(sd)
    # model_config_from_unet MUST see the comfy_quant keys while they still carry
    # the checkpoint prefix: comfy.utils.detect_layer_quantization() matches on
    # `k.startswith(unet_key_prefix)`. Detect FIRST, then strip the prefix.
    mcfg = md.model_config_from_unet(sd, prefix)
    if mcfg.quant_config is not None:
        print("    the mixed-precision quant detected: %s" % (mcfg.quant_config,), flush=True)
    unet_sd = comfy.utils.state_dict_prefix_replace(sd, {prefix: ""}, filter_keys=True)
    if mcfg is None:
        raise RuntimeError("the SD1.5 UNet config was not detected")
    mcfg.set_inference_dtype(DT, None, device=DEV)
    if ops is not None:
        mcfg.custom_operations = ops
    model = mcfg.get_model(unet_sd, "")
    model.load_model_weights(unet_sd, "", assign=False)
    model = model.diffusion_model.to(DEV).eval()
    return model


def make_inputs(seed, latent=LATENT, ctx=77, dim=768):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((1, 4, latent, latent), generator=g).to(DEV, dtype=DT)
    ts = torch.tensor([500.0], device=DEV, dtype=DT)
    context = torch.randn((1, ctx, dim), generator=g).to(DEV, dtype=DT)
    return x, ts, context


def unet_output(model, x, ts, context):
    with torch.no_grad():
        out = model(x, ts, context)
    return out.detach().float().double().reshape(-1).cpu()


def main():
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
        sys.exit("usage: verify_sd15_unet.py --comfy <root> <source.ckpt> <pack> [<pack> ...] [--n 256] [--out results.json]")
    src = rest[0]
    packs = rest[1:]

    ops = O.mixed_precision_ops(
        {"int8_tensorwise": comfy.quant_ops.QUANT_ALGOS["int8_tensorwise"]}, DT,
        full_precision_mm=False, disabled=[])
    print("the arm: %s | the items: %d" % (
        getattr(O, "_hswq_int8_gemm_patched", False), n), flush=True)

    print("[1] the baseline (the same ops path) ...", flush=True)
    m0 = build_unet(src, ops)
    batches = [make_inputs(SEED + i) for i in range(n)]
    ref = [unet_output(m0, *b) for b in batches]
    print("    the reference: %s finite=%s" % (
        tuple(ref[0].shape), bool(torch.isfinite(ref[0]).all())), flush=True)
    del m0
    torch.cuda.empty_cache()

    results = []
    for p in packs:
        print("[2] %s" % os.path.basename(p), flush=True)
        sd = comfy.utils.load_torch_file(p, safe_load=True)
        nq = len([k for k in sd if k.endswith(".comfy_quant")])
        model = build_unet(p, ops)
        ck = O.quant_ops.ck
        orig = ck.int8_linear
        cnt = [0]

        def counted(*a, **kw):
            cnt[0] += 1
            return orig(*a, **kw)

        ck.int8_linear = counted
        cs, fin = [], True
        try:
            for i in range(n):
                f = unet_output(model, *batches[i])
                if not torch.isfinite(f).all():
                    fin = False
                cs.append(torch.nn.functional.cosine_similarity(
                    ref[i].reshape(1, -1), f.reshape(1, -1), dim=1).item())
        finally:
            ck.int8_linear = orig
        del model, sd
        torch.cuda.empty_cache()
        cs.sort()
        m = len(cs)
        rec = {"pack": os.path.basename(p), "file_mb": os.path.getsize(p) / 1024 ** 2,
               "quantized_layers": nq, "int8_linear_calls": cnt[0],
               "cos_mean": statistics.mean(cs), "cos_min": cs[0],
               "p5": cs[max(0, int(m * 0.05) - 1)], "n": m, "finite": fin}
        rec["pass"] = bool(rec["cos_mean"] >= 0.99 and rec["cos_min"] >= 0.98 and fin)
        results.append(rec)
        print("    int8=%d finite=%s  mean=%.6f min=%.6f p5=%.6f  size=%.1f MB  %s" % (
            cnt[0], fin, rec["cos_mean"], rec["cos_min"], rec["p5"], rec["file_mb"],
            "PASS" if rec["pass"] else "FAIL"), flush=True)

    if out_path:
        json.dump(results, open(out_path, "w", encoding="utf-8"), indent=1)
        print("saved: %s" % out_path)
    print()
    print("  %-30s %6s %8s %10s %10s %10s" % ("pack", "INT8", "int8", "mean", "min", "MB"))
    for r in results:
        print("  %-30s %6d %8d %10.6f %10.6f %10.1f" % (
            r["pack"], r["quantized_layers"], r["int8_linear_calls"],
            r["cos_mean"], r["cos_min"], r["file_mb"]))


if __name__ == "__main__":
    main()

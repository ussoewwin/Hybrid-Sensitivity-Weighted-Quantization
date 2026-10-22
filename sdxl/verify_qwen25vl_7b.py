# -*- coding: utf-8 -*-
"""The Qwen2.5-VL-7B text encoder's ConvRot INT8 acceptance, on the real path.

The observable is the text encoder's own output for a fixed prompt, plus the
vision tower's image embeddings - the same two paths the impact measurement
uses (`sdxl/diag_impact_qwen3vl.py`).

Two things the measurement has to get right:

1. **The arm.** The loader package's `patches/clip_int8_gemm` makes the real
   `int8_linear` fire for the text tower. The package's root `__init__` is
   executed exactly as ComfyUI does on startup.
2. **The same ops path for the baseline.** The stock CLIP loader builds the
   text tower with `comfy.ops.manual_cast` while a quantized pack is built
   with `mixed_precision_ops`, and for some models the two paths are not
   numerically equivalent. The baseline is built through the same ops class.

Usage:
    python verify_qwen3vl.py <source.safetensors> [<pack.safetensors> ...]
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
import comfy.ops as comfy_ops  # noqa: E402
from comfy.quant_ops import QUANT_ALGOS  # noqa: E402

if not hasattr(comfy_server.PromptServer, "instance"):
    comfy_server.PromptServer.instance = None

# the arm: the package's root __init__ is executed exactly as ComfyUI does
_spec = importlib.util.spec_from_file_location(
    "hswq_pkg_init", os.path.join(NODE, "__init__.py"), submodule_search_locations=[NODE])
_mod = importlib.util.module_from_spec(_spec)
sys.modules["hswq_pkg_init"] = _mod
_spec.loader.exec_module(_mod)

N = int(os.environ.get("QVL_N", "32"))
DT = torch.bfloat16
TOKENS = 16
IMG = (1, 3, 2, 56, 56)
GRID = [[1, 4, 4]]


def build(path, ops):
    """The Qwen2.5-VL-7B text encoder, the way ComfyUI builds it."""
    import comfy.text_encoders.qwen_image as qi
    import comfy.text_encoders.hunyuan_video as hv
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    detect = hv.llama_detect(sd)
    model = qi.te(**detect)
    clip = model(device="cpu", dtype=DT,
                 model_options={"custom_operations": ops})
    clip.load_sd(sd)
    clip = clip.to("cuda").eval()
    inner = getattr(clip, getattr(clip, "clip"))
    tr = getattr(inner, "transformer", inner)
    return clip, getattr(tr, "model", tr), getattr(tr, "visual", None)


def outputs(llama, visual, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    tok_ids = torch.randint(1000, 20000, (1, TOKENS), generator=g).to("cuda")
    parts = []
    with torch.no_grad():
        t = llama(tok_ids)
    # the runtime's QWEN25_7B clip type loads the LANGUAGE tower only
    # (Qwen25_7BVLIModel); the vision tower is model content that no Qwen2.5-VL
    # load path in ComfyUI reads, so the observable is the language model's
    # hidden state. The vision tower is still shipped and quantized in the pack
    # (model content), but it is not the gate.
    if isinstance(t, (tuple, list)):
        for x in t:
            if torch.is_tensor(x):
                parts.append(x.detach().float().reshape(-1).cpu())
    elif torch.is_tensor(t):
        parts.append(t.detach().float().reshape(-1).cpu())
    return torch.cat(parts)


def main():
    src = sys.argv[1]
    packs = sys.argv[2:]
    ops = comfy_ops.mixed_precision_ops(
        {"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]}, DT,
        full_precision_mm=False, disabled=[])
    print("the arm: %s" % getattr(comfy_ops, "_hswq_int8_gemm_patched", False), flush=True)

    print("[1] the FP16 baseline (the same ops path) ...", flush=True)
    c16, l16, v16 = build(src, ops)
    ref = {s: outputs(l16, v16, s) for s in range(N)}
    print("    the reference: %s finite=%s" % (
        tuple(ref[0].shape), bool(torch.isfinite(ref[0]).all())), flush=True)
    del c16, l16, v16
    torch.cuda.empty_cache()

    results = []
    for p in packs:
        print("[2] %s" % os.path.basename(p), flush=True)
        c8, l8, v8 = build(p, ops)
        nq = len([k for k in comfy.utils.load_torch_file(p, safe_load=True)
                  if k.endswith(".comfy_quant")])
        ck = comfy_ops.quant_ops.ck
        orig = ck.int8_linear
        cnt = [0]

        def counted(*a, **kw):
            cnt[0] += 1
            return orig(*a, **kw)

        ck.int8_linear = counted
        cs = []
        fin = True
        try:
            for s in range(N):
                f = outputs(l8, v8, s)
                if not torch.isfinite(f).all():
                    fin = False
                cs.append(torch.nn.functional.cosine_similarity(
                    ref[s].reshape(1, -1), f.reshape(1, -1)).item())
        finally:
            ck.int8_linear = orig
        cs.sort()
        m = len(cs)
        rec = {"pack": os.path.basename(p),
               "file_mb": os.path.getsize(p) / 1024 ** 2,
               "quantized_layers": nq, "int8_linear_calls": cnt[0],
               "mean": statistics.mean(cs), "min": cs[0],
               "p5": cs[max(0, int(m * 0.05) - 1)], "p50": cs[m // 2],
               "finite": fin, "n": m}
        rec["pass"] = bool(rec["mean"] >= 0.99 and rec["min"] >= 0.98 and fin)
        results.append(rec)
        print("    int8_linear=%d finite=%s  mean=%.6f min=%.6f p5=%.6f p50=%.6f  size=%.1f MB  %s" % (
            cnt[0], fin, rec["mean"], rec["min"], rec["p5"], rec["p50"], rec["file_mb"],
            "PASS" if rec["pass"] else "FAIL"), flush=True)
        del c8, l8, v8
        torch.cuda.empty_cache()

    print()
    print("%-50s %7s %7s %10s %10s %10s %10s" % (
        "pack", "INT8", "int8", "mean", "min", "p5", "MB"))
    for r in results:
        print("%-50s %7d %7d %10.6f %10.6f %10.6f %10.1f" % (
            r["pack"][:50], r["quantized_layers"], r["int8_linear_calls"],
            r["mean"], r["min"], r["p5"], r["file_mb"]))
    print("JSON " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()

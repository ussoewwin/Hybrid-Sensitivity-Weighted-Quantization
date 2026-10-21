# -*- coding: utf-8 -*-
"""The CLIP text encoder's ConvRot INT8 acceptance, on the real path.

The observable is the text encoder's own output - the pooled embedding and
the token embeddings - against the FP16 source, over a fixed item set.

Two things the measurement has to get right:

1. **The arm.** The stock `comfy.sd.load_clip` asks for
   `full_precision_mm=True`, which disables the INT8 GEMM: the quantized
   weight is dequantized for every matmul. The loader package's arm
   (`patches/clip_int8_gemm`) makes the real `int8_linear` fire, so the
   package's root `__init__` is executed exactly as ComfyUI does on startup.

2. **The same ops path for the baseline.** The stock CLIP loader builds the
   text tower with `comfy.ops.manual_cast` while a quantized pack is built
   with `mixed_precision_ops`, and for some models the two paths are not
   numerically equivalent. The baseline is therefore built through the same
   ops class.

Usage:
    python verify_clip_text.py <source.safetensors> [<pack.safetensors> ...]
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

# the arm: the package's root __init__ is executed exactly as ComfyUI does
_spec = importlib.util.spec_from_file_location(
    "hswq_pkg_init", os.path.join(NODE, "__init__.py"), submodule_search_locations=[NODE])
_mod = importlib.util.module_from_spec(_spec)
sys.modules["hswq_pkg_init"] = _mod
_spec.loader.exec_module(_mod)

N = int(os.environ.get("CLIP_N", "64"))
PROMPTS = [
    "a photograph of a cat sitting on a table",
    "a portrait of a woman with red hair, studio lighting",
    "an oil painting of a harbour at sunset",
    "a technical diagram of a bicycle frame",
    "a bowl of ramen with a soft-boiled egg",
    "a snowy mountain range under a clear sky",
    "a city street at night in the rain",
    "a close-up of a bee on a sunflower",
    "a futuristic sports car on a white background",
    "a group of people walking through a forest",
]


def build(path, ops):
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    return comfy.sd.load_clip([path], model_options={"custom_operations": ops})


def outputs(clip, seed):
    """The text encoder's own output for a fixed prompt set."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randint(0, len(PROMPTS), (8,), generator=g)
    # clip.tokenize returns {clip_name: [(token, weight), ...]} for ONE text.
    # The batch is built by concatenating the per-text token lists into that
    # dict's entry, the way ComfyUI's own batching does it.
    tk = clip.tokenize(PROMPTS[int(idx[0])])
    name = list(tk.keys())[0]
    merged = {name: list(tk[name])}
    for i in idx.tolist()[1:]:
        tk2 = clip.tokenize(PROMPTS[i])
        merged[name] = list(merged[name]) + list(tk2[name])
    with torch.no_grad():
        cond, pooled = clip.encode_from_tokens(merged, return_pooled=True)
    parts = [cond.detach().float().reshape(-1).cpu()]
    # `pooled` is a tensor for a single-clip text encoder and a list for a
    # multi-clip one
    if pooled is not None:
        if torch.is_tensor(pooled):
            parts.append(pooled.detach().float().reshape(-1).cpu())
        else:
            for q in pooled:
                if torch.is_tensor(q):
                    parts.append(q.detach().float().reshape(-1).cpu())
    return torch.cat(parts)


def main():
    src = sys.argv[1]
    packs = sys.argv[2:]
    ops = comfy_ops.mixed_precision_ops(
        {"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]}, torch.float16,
        full_precision_mm=False, disabled=[])
    print("the arm: %s" % getattr(comfy_ops, "_hswq_int8_gemm_patched", False), flush=True)

    print("[1] the FP16 baseline (the same ops path) ...", flush=True)
    c16 = build(src, ops)
    ref = {s: outputs(c16, s) for s in range(N)}
    print("    the reference: %s finite=%s" % (
        tuple(ref[0].shape), bool(torch.isfinite(ref[0]).all())), flush=True)
    del c16
    torch.cuda.empty_cache()

    results = []
    for p in packs:
        print("[2] %s" % os.path.basename(p), flush=True)
        c8 = build(p, ops)
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
                f = outputs(c8, s)
                if not torch.isfinite(f).all():
                    fin = False
                cs.append(torch.nn.functional.cosine_similarity(
                    ref[s].reshape(1, -1), f.reshape(1, -1)).item())
        finally:
            ck.int8_linear = orig
        cs.sort()
        m = len(cs)
        nq = len([k for k in comfy.utils.load_torch_file(p, safe_load=True)
                  if k.endswith(".comfy_quant")])
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
        del c8
        torch.cuda.empty_cache()

    print()
    print("%-46s %7s %7s %10s %10s %10s %10s" % (
        "pack", "INT8", "int8", "mean", "min", "p5", "MB"))
    for r in results:
        print("%-46s %7d %7d %10.6f %10.6f %10.6f %10.1f" % (
            r["pack"][:46], r["quantized_layers"], r["int8_linear_calls"],
            r["mean"], r["min"], r["p5"], r["file_mb"]))
    print("JSON " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()

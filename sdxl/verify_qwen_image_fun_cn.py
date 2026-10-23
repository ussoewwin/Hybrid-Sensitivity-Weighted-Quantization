# -*- coding: utf-8 -*-
"""Qwen-Image Fun ControlNet (Union 2602): the acceptance on the fixed 256-item
set.

The gate is the ControlNet's OWN output (the same instrument the protection set
was chosen with - the criterion and the gate must be the same instrument).

The baseline is built through the SAME ops path as the quantized pack
(`mixed_precision_ops`): the stock loader uses `manual_cast` and the quantized
one uses `mixed_precision_ops`, and those two are not numerically equivalent.

Usage:
  python sdxl/verify_qwen_image_fun_cn.py \
      --cn <source>.safetensors \
      --base <qwen_image_bf16.safetensors> \
      <pack1> [<pack2> ...]
"""
from __future__ import annotations

import json
import os
import statistics
import sys

COMFY = os.environ.get("COMFY_PATH", r"D:\USERFILES\ComfyUI\ComfyUI")
if COMFY not in sys.path:
    sys.path.insert(0, COMFY)
os.chdir(COMFY)

import torch  # noqa: E402

import comfy.utils  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.controlnet  # noqa: E402
import comfy.quant_ops  # noqa: E402
import comfy.ldm.qwen_image.model as qm  # noqa: E402

# the arm: execute the loader package's root __init__ exactly as ComfyUI does on
# startup, so patches/clip_int8_gemm makes the real int8_linear fire.
import importlib.util  # noqa: E402
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
DT = torch.bfloat16
N = int(os.environ.get("QCN_N", "256"))
SEED = 42
BASE_ENTRY = ("img_in.", "txt_in.", "txt_norm.", "time_text_embed.")


def build_base(base_path, ops):
    sd = comfy.utils.load_torch_file(base_path, safe_load=True)
    entry = {k: v for k, v in sd.items() if k.startswith(BASE_ENTRY)}
    model = qm.QwenImageTransformer2DModel(
        patch_size=2, in_channels=64, out_channels=16, num_layers=0,
        attention_head_dim=128, num_attention_heads=24, joint_attention_dim=3584,
        final_layer=False, dtype=DT, device="cpu", operations=ops)
    missing, _ = model.load_state_dict(entry, strict=False)
    missing = [k for k in missing if k.startswith(BASE_ENTRY)]
    if missing:
        raise RuntimeError("the base entry tensors are missing: %s" % missing)
    return model.to(DEV).eval()


def build_cn(cn_path, ops):
    sd = comfy.utils.load_torch_file(cn_path, safe_load=True)
    cnet = comfy.controlnet.load_controlnet_qwen_fun(
        sd, model_options={"custom_operations": ops, "dtype": DT})
    return cnet.control_model.to(DEV).eval()


def make_inputs(seed, grid=8, ctx=8, dim=3584):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.rand((1, 16, 1, grid, grid), generator=g).to(DEV, dtype=DT)
    hint = torch.rand((1, 33, grid, grid), generator=g).to(DEV, dtype=DT)
    context = torch.rand((1, ctx, dim), generator=g).to(DEV, dtype=DT)
    ts = torch.tensor([0.5], device=DEV, dtype=DT)
    return x, ts, context, hint


def cn_output(cn, base, x, ts, context, hint):
    with torch.no_grad():
        out = cn(x, ts, context, attention_mask=None, guidance=None,
                 hint=hint, transformer_options={}, base_model=base)
    parts = []
    for t in out["input"]:
        if t is not None and torch.is_tensor(t):
            parts.append(t.detach().float().reshape(-1).cpu())
    return torch.cat(parts)


def main():
    args = sys.argv[1:]
    src = args[0]
    packs = args[1:]
    base_path = os.environ["QCN_BASE"]

    ops = comfy.ops.mixed_precision_ops(
        {"int8_tensorwise": comfy.quant_ops.QUANT_ALGOS["int8_tensorwise"]}, DT,
        full_precision_mm=False, disabled=[])
    print("the arm: %s" % getattr(comfy.ops, "_hswq_int8_gemm_patched", False), flush=True)

    print("[1] the baseline (the same ops path) ...", flush=True)
    base = build_base(base_path, ops)
    c16 = build_cn(src, ops)
    batches = [make_inputs(SEED + i) for i in range(N)]
    ref = [cn_output(c16, base, *b) for b in batches]
    print("    the reference: %s finite=%s" % (
        tuple(ref[0].shape), bool(torch.isfinite(ref[0]).all())), flush=True)
    del c16
    torch.cuda.empty_cache()

    results = []
    for p in packs:
        print("[2] %s" % os.path.basename(p), flush=True)
        cn = build_cn(p, ops)
        sd = comfy.utils.load_torch_file(p, safe_load=True)
        nq = len([k for k in sd if k.endswith(".comfy_quant")])
        ck = comfy.ops.quant_ops.ck
        orig = ck.int8_linear
        cnt = [0]

        def counted(*a, **kw):
            cnt[0] += 1
            return orig(*a, **kw)

        ck.int8_linear = counted
        cs = []
        fin = True
        try:
            for i in range(N):
                f = cn_output(cn, base, *batches[i])
                if not torch.isfinite(f).all():
                    fin = False
                cs.append(torch.nn.functional.cosine_similarity(
                    ref[i].reshape(1, -1), f.reshape(1, -1)).item())
        finally:
            ck.int8_linear = orig
        del cn
        torch.cuda.empty_cache()
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
            cnt[0], fin, rec["mean"], rec["min"], rec["p5"], rec["p50"],
            rec["file_mb"], "PASS" if rec["pass"] else "FAIL"), flush=True)

    out = os.environ.get("QCN_OUT")
    if out:
        json.dump(results, open(out, "w", encoding="utf-8"), indent=1)
        print("saved: %s" % out)
    print()
    print("  %-40s %6s %8s %10s %10s %10s %10s" % (
        "pack", "INT8", "int8", "mean", "min", "p5", "MB"))
    for r in results:
        print("  %-40s %6d %8d %10.6f %10.6f %10.6f %10.1f" % (
            r["pack"], r["quantized_layers"], r["int8_linear_calls"],
            r["mean"], r["min"], r["p5"], r["file_mb"]))


if __name__ == "__main__":
    main()

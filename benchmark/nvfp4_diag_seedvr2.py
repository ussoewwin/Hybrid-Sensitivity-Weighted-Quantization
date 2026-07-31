#!/usr/bin/env python3
"""
HSWQ NVFP4+ConvRot stage-isolation diagnostic (SeedVR2 A2_rot).

Runs the REAL runtime stack on CUDA with REAL checkpoint weights and reports
per-stage relative error / NaN counts so the failing stage is identified in
one run:

  stage 1  rotate_last_dim_pooled      (x @ H, gs from comfy_quant)
  stage 2  quantize_nvfp4_act_pooled   (_C.quantize_nvfp4, sx NaN scan)
  stage 3  scaled_mm_nvfp4_pooled      (_C.cublas_gemm_blockwise_fp4)
  stage 4  stock QT path               (QuantizedTensor.from_float + ck mm)
  stage 5  armed module forward        (load_nvfp4_linear_module + forward_nvfp4)
  sweep    stages 1-3 over every nvfp4 conf layer (shape flags + NaN hunt)

Reference math (CPU-verified):
  W_rot = dequant(weight)          (rot-domain, kitchen eager)
  W_ref = unrotate(W_rot) = W_rot @ H   (H symmetric orthonormal)
  y_ref = x @ W_ref^T
  TC path: y = (x@H) @ W_rot^T == x @ W^T  (rotation cancels)

Expected healthy numbers: rotate rel err ~1e-3, quantize roundtrip ~0.05-0.2,
mm / module forward ~0.05-0.2.  FAIL = NaN present or rel err > 0.5.

Usage (user runs; needs CUDA):
  python benchmark/nvfp4_diag_seedvr2.py [--ckpt <A2_rot.safetensors>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from safetensors import safe_open

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_REPO, "ComfyUI-master")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

_DEFAULT_CKPT = (
    r"D:\USERFILES\ComfyUI\ComfyUI\models\SEEDVR2\seedvr2_7b_nvfp4_A2_rot.safetensors"
)

E4M3_NAN_MASK = 0x7F  # (byte & 0x7F) == 0x7F -> e4m3fn NaN

_LOG_PATH = os.path.join(_HERE, "nvfp4_diag_seedvr2_log.txt")


class _Tee:
    """Duplicate stdout to console + log file (diag output is pasted as file)."""

    def __init__(self, path: str):
        self._f = open(path, "w", encoding="utf-8", errors="replace")
        self._orig = sys.stdout

    def write(self, s):
        self._orig.write(s)
        self._f.write(s)

    def flush(self):
        self._orig.flush()
        self._f.flush()


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float()
    b = b.detach().float()
    denom = b.norm().item()
    if denom == 0.0:
        return float("nan")
    return ((a - b).norm().item()) / denom


def nan_count(t: torch.Tensor) -> int:
    return int(torch.isnan(t.detach().float()).sum().item())


def sx_nan_count(sx_uint8: torch.Tensor) -> int:
    return int(((sx_uint8 & E4M3_NAN_MASK) == E4M3_NAN_MASK).sum().item())


def unrotate_rot_domain(w_rot: torch.Tensor, h: torch.Tensor, gs: int) -> torch.Tensor:
    """W = W_rot @ H group-wise along in-features (fp32)."""
    out_f, in_f = w_rot.shape
    assert in_f % gs == 0, f"in_features {in_f} not divisible by gs {gs}"
    g = w_rot.reshape(out_f, in_f // gs, gs)
    return torch.matmul(g, h).reshape(out_f, in_f)


def rotate_2d(x: torch.Tensor, h: torch.Tensor, gs: int) -> torch.Tensor:
    m, k = x.shape
    return torch.matmul(x.reshape(-1, k // gs, gs), h).reshape(m, k)


class LayerData:
    def __init__(self, f, layer: str, device: str):
        self.name = layer
        self.w_q = f.get_tensor(f"{layer}.weight")              # (N, K/2) uint8
        self.bs = f.get_tensor(f"{layer}.weight_scale")         # fp8e4m3fn swizzled
        self.ts = f.get_tensor(f"{layer}.weight_scale_2").float().reshape(1)
        cq = f.get_tensor(f"{layer}.comfy_quant")
        self.conf = json.loads(cq.numpy().tobytes())
        self.gs = int(self.conf.get("convrot_groupsize", 256) or 256)
        self.convrot = bool(self.conf.get("convrot", False))
        self.n, self.k = int(self.w_q.shape[0]), int(self.w_q.shape[1]) * 2
        self.w_q = self.w_q.to(device)
        self.bs = self.bs.to(device)
        self.ts = self.ts.to(device)


def kernel_stage_test(ld: LayerData, x: torch.Tensor, h: torch.Tensor, tag: str,
                      nvrt, eager_deq, quiet: bool = False) -> dict:
    """Stages 1-3 on one input tensor. Returns verdict dict; prints lines."""
    dev = x.device
    m, k = x.shape
    out = {"tag": tag, "layer": ld.name, "m": m, "fail": None}

    # Reference: W_ref (fp32, unrotated) and y_ref.
    with torch.no_grad():
        w_rot_ref = eager_deq(ld.w_q, ld.ts, ld.bs, output_type=torch.float32)
        w_ref = unrotate_rot_domain(w_rot_ref, h.float(), ld.gs) if ld.convrot else w_rot_ref
        y_ref = x.float() @ w_ref.t()

    # Stage 1: pooled rotation.
    x_rot = nvrt.rotate_last_dim_pooled(x, h, ld.gs) if ld.convrot else x
    r1 = rel_err(x_rot.float(), rotate_2d(x.float(), h.float(), ld.gs)) if ld.convrot else 0.0
    n1 = nan_count(x_rot)
    if n1 or (ld.convrot and r1 > 0.05):
        out["fail"] = f"rotate rel={r1:.3e} nan={n1}"

    # Stage 2: pooled quantize (+ sx NaN scan, roundtrip).
    scale_a = nvrt.ensure_act_scale_amax(x_rot)
    needs_pad = (m % 16 != 0) or (k % 16 != 0)
    qx, sx, pr, pc = nvrt.quantize_nvfp4_act_pooled(x_rot, scale_a, pad_16x=needs_pad)
    sx_u8 = sx.view(torch.uint8)
    n2 = sx_nan_count(sx_u8)
    with torch.no_grad():
        x_rt = eager_deq(qx, scale_a, sx, output_type=torch.float32)[:m, :k]
    r2 = rel_err(x_rt, x_rot.float())
    n2b = nan_count(x_rt)
    if out["fail"] is None and (n2 or n2b or r2 > 0.6):
        out["fail"] = f"quantize rel={r2:.3e} sx_nan={n2} rt_nan={n2b}"

    # Stage 3: pooled cuBLAS mm.
    alpha = (scale_a * ld.ts).float().reshape(1)
    y_tc = nvrt.scaled_mm_nvfp4_pooled(
        qx, ld.w_q,
        tensor_scale_a=scale_a, tensor_scale_b=ld.ts,
        block_scale_a=sx, block_scale_b=ld.bs,
        bias=None, out_dtype=torch.float16, alpha=alpha,
        orig_m=m, orig_n=ld.n,
    )
    r3 = rel_err(y_tc.float(), y_ref)
    n3 = nan_count(y_tc)
    if out["fail"] is None and (n3 or r3 > 0.6):
        out["fail"] = f"mm rel={r3:.3e} nan={n3}"

    if not quiet or out["fail"] is not None:
        print(
            f"    [{tag}] {ld.name} M={m} "
            f"rotate rel={r1:.3e} nan={n1} | "
            f"quant rel={r2:.3e} sx_nan={n2} rt_nan={n2b} "
            f"scale_a={float(scale_a):.3e} | "
            f"mm rel={r3:.3e} nan={n3}"
            + (f"  <-- FAIL: {out['fail']}" if out["fail"] else "")
        )

    out.update({"r1": r1, "r2": r2, "r3": r3, "nan": n1 + n2 + n2b + n3})
    return out


def module_stage_test(ld: LayerData, x: torch.Tensor, tag: str) -> None:
    """Stage 5: real armed module via load_nvfp4_linear_module + forward_nvfp4."""
    import comfy.ops as comfy_ops
    import comfy.options

    comfy.options.enable_args_parsing(False)
    from nvfp4.comfy_quant_nvfp4 import apply_comfy_quant_nvfp4_patches
    from nvfp4.nvfp4_hadamard import build_hadamard
    from nvfp4.nvfp4_load import load_nvfp4_linear_module

    assert apply_comfy_quant_nvfp4_patches()
    ops = comfy_ops.mixed_precision_ops(
        quant_config={}, compute_dtype=torch.float16,
        full_precision_mm=False, disabled=[],
    )
    mod = ops.Linear(ld.k, ld.n, bias=False, device="cuda", dtype=torch.float16)
    cq_u8 = torch.frombuffer(json.dumps(ld.conf).encode(), dtype=torch.uint8)
    prefix = ld.name + "."

    def make_sd():
        # load_nvfp4_linear_module pops keys — build a fresh dict per load.
        return {
            prefix + "weight": ld.w_q.cpu(),
            prefix + "weight_scale": ld.bs.cpu(),
            prefix + "weight_scale_2": ld.ts.cpu(),
            prefix + "comfy_quant": cq_u8,
        }

    load_nvfp4_linear_module(
        mod,
        lambda *a, **k: torch.nn.Module._load_from_state_dict(mod, *a, **k),
        make_sd(), prefix, {}, False, [], [], [],
        load_extra_params=True,
    )
    assert getattr(mod, "_hswq_nvfp4", False), "module not armed"
    print(
        f"    [{tag}] armed convrot={mod._hswq_nvfp4_convrot} "
        f"gs={mod._hswq_nvfp4_convrot_groupsize} "
        f"placeholder={getattr(mod, '_hswq_nvfp4_scale_placeholder', None)}"
    )

    dev = x.device
    h = build_hadamard(ld.gs, device="cpu", dtype=torch.float32)
    from comfy_kitchen.backends.eager.quantization import dequantize_nvfp4 as deq
    w_rot_ref = deq(ld.w_q, ld.ts, ld.bs, output_type=torch.float32)
    w_ref = unrotate_rot_domain(w_rot_ref, h.to(dev), ld.gs) if ld.convrot else w_rot_ref

    def fresh_module():
        m2 = ops.Linear(ld.k, ld.n, bias=False, device="cuda", dtype=torch.float16)
        load_nvfp4_linear_module(
            m2,
            lambda *a, **k: torch.nn.Module._load_from_state_dict(m2, *a, **k),
            make_sd(), prefix, {}, False, [], [], [],
            load_extra_params=True,
        )
        return m2

    def run_seq(name, seq):
        m2 = fresh_module()
        for i, xv in enumerate(seq):
            with torch.no_grad():
                y = m2(xv)
            r = rel_err(y.float(), (xv.float() @ w_ref.t()))
            n = nan_count(y)
            sa = getattr(m2, "_hswq_nvfp4_act_scale", None)
            sa_v = float(sa) if sa is not None else float("nan")
            al = getattr(m2, "_hswq_nvfp4_alpha", None)
            al_v = float(al) if al is not None else float("nan")
            print(
                f"    [{tag}] {name} call{i}: rel={r:.4e} nan={n} "
                f"scale_a={sa_v:.3e} alpha={al_v:.3e}"
            )

    xo = x.clone()
    idx = torch.rand(xo.numel(), device=dev) < 1e-3
    xo.view(-1)[idx] *= 1000.0
    x2 = torch.randn_like(x) * 3.0
    zeros = torch.zeros_like(x)

    # Each sequence runs on a FRESH module (own scale cache):
    #  randn-first : healthy freeze -> every call rel ~0.1
    #  zeros-first : pre-fix poison repro; post-fix call0=0 (correct), call1 ~0.1
    #  outlier-first: freeze on outlier-heavy amax -> later randn still ~0.1-0.2
    run_seq("randn-first ", [x, x2])
    run_seq("zeros-first ", [zeros, x])
    run_seq("outlier-first", [xo, x2])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=_DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ms", default="1000,4096,30000")
    ap.add_argument("--module-layer", default=None, help="layer name for stage-5 test")
    ap.add_argument("--sweep-max", type=int, default=0, help="0 = all layers")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required"); return 1
    _tee = _Tee(_LOG_PATH)
    sys.stdout = _tee
    sys.stderr = _tee
    print(f"[DIAG] log file: {_LOG_PATH}")
    dev = args.device

    from nvfp4 import nvfp4_runtime as nvrt
    from nvfp4.nvfp4_hadamard import build_hadamard
    from comfy_kitchen.backends.eager.quantization import dequantize_nvfp4 as eager_deq

    print(f"[DIAG] ckpt: {args.ckpt}")
    ms = [int(s) for s in args.ms.split(",") if s.strip()]

    with safe_open(args.ckpt, framework="pt", device="cpu") as f:
        layers = sorted(
            k[: -len(".comfy_quant")]
            for k in f.keys()
            if k.endswith(".comfy_quant")
            and json.loads(f.get_tensor(k).numpy().tobytes()).get("format") == "nvfp4"
        )
        print(f"[DIAG] nvfp4 layers: {len(layers)}")
        first = LayerData(f, layers[0], dev)
        # Representative extra layers (deep + last) for the focused test.
        focus = {layers[0], layers[len(layers) // 2], layers[-1]}
        if args.module_layer:
            focus.add(args.module_layer)
        focus_data = [first] + [LayerData(f, ln, dev) for ln in sorted(focus) if ln != layers[0]]

        # ---- focused kernel-level test on first layer, all M sizes ----
        ld = first
        h_cpu = build_hadamard(ld.gs, device="cpu", dtype=torch.float32)
        print(
            f"[DIAG] focus layer {ld.name}: N={ld.n} K={ld.k} gs={ld.gs} "
            f"convrot={ld.convrot} K%32={ld.k % 32} N%8={ld.n % 8}"
        )
        h_dev = h_cpu.to(dev)
        for m in ms:
            x = (torch.randn(m, ld.k, device=dev) * 3.0).half()
            try:
                kernel_stage_test(ld, x, h_dev.half(), f"M={m}", nvrt, eager_deq)
            except Exception as e:  # noqa: BLE001
                print(f"    [M={m}] EXC {type(e).__name__}: {e}")
        # outlier regime (DiT massive activations)
        m = ms[min(1, len(ms) - 1)]
        xo = (torch.randn(m, ld.k, device=dev) * 3.0).half()
        idx = torch.rand(xo.numel(), device=dev) < 1e-3
        xo.view(-1)[idx] *= 1000.0
        try:
            kernel_stage_test(ld, xo, h_dev.half(), f"M={m} outlier", nvrt, eager_deq)
        except Exception as e:  # noqa: BLE001
            print(f"    [M={m} outlier] EXC {type(e).__name__}: {e}")

        # ---- stage 5: armed module end-to-end ----
        print("[DIAG] stage5 module-level (real load + forward_nvfp4)")
        try:
            module_stage_test(ld, (torch.randn(4096, ld.k, device=dev) * 3.0).half(), ld.name)
        except Exception as e:  # noqa: BLE001
            print(f"    [stage5] EXC {type(e).__name__}: {e}")

        # ---- sweep: every conf layer, kernel stages at M=4096 ----
        print("[DIAG] sweep all layers (M=4096, outlier input)")
        worst = []
        shape_bad = []
        nan_layers = []
        count = 0
        for ln in layers:
            if args.sweep_max and count >= args.sweep_max:
                break
            count += 1
            ld_s = ld if ln == ld.name else LayerData(f, ln, dev)
            if (ld_s.k % 32) or (ld_s.n % 8):
                shape_bad.append((ln, ld_s.k, ld_s.n))
            h_s = h_dev if ld_s.gs == ld.gs else build_hadamard(
                ld_s.gs, device=dev, dtype=torch.float32
            )
            xs = (torch.randn(4096, ld_s.k, device=dev) * 3.0).half()
            idx = torch.rand(xs.numel(), device=dev) < 1e-3
            xs.view(-1)[idx] *= 1000.0
            try:
                r = kernel_stage_test(ld_s, xs, h_s.half(), "sweep", nvrt, eager_deq, quiet=True)
            except Exception as e:  # noqa: BLE001
                print(f"    [sweep] EXC {ln}: {type(e).__name__}: {e}")
                worst.append((float("inf"), ln))
                continue
            worst.append((r["r3"], ln))
            if r["nan"]:
                nan_layers.append((ln, r["fail"]))
        worst.sort(reverse=True)
        print("[DIAG] sweep worst mm rel err (top 10):")
        for r, ln in worst[:10]:
            print(f"    {r:.4e}  {ln}")
        print(f"[DIAG] shape violations (K%32 or N%8): {shape_bad or 'none'}")
        print(f"[DIAG] layers with NaN: {nan_layers or 'none'}")

    print("[DIAG] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

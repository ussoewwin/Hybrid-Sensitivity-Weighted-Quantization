#!/usr/bin/env python3
"""SDXL ConvRot INT8 deterministic trajectory-divergence comparator (traj style).

Same callback-based per-step latent trajectory capture as benchmark/zi_traj_compare.py
(the Z Image INT8 variant), applied to SDXL checkpoints:

  * same seed -> identical initial noise for both models (fully deterministic)
  * captures the latent ``x`` AND the model's x0-prediction at EVERY denoising step
  * per-step + final latent MSE / cosine, aggregated over multiple seeds
  * verdicts: bifurcated / same-image / drifted (same thresholds as the other traj benches)

Model loading is ComfyUI-native (``comfy.sd.load_checkpoint_guess_config``), the same
as benchmark/int8bench_sdxl.py (the old image-SSIM bench). INT8 Conv2d load scope is
armed only when the checkpoint looks like a comfy_quant INT8 pack, so the FP16
baseline always loads stock.

The ConvRot INT8 quantization kernels come from ``benchmark/int8/comfy_quant_int8.py``
(vendored). torch/psutil/torchaudio/comfy_aimdo stubs are reused from
``archives/qi_int8_bench.py``-style inline definitions to keep this script
self-contained.

Usage:
    python benchmark/sdxl_int8_traj_compare.py \
        --fp16 <sdxl_ckpt.safetensors> --int8 <sdxl_convrot_int8.safetensors> \
        --comfy_path <ComfyUI-master> \
        [--seeds "42,1337,7,2024,555"] [--steps 25] [--prompt "..."] \
        [--width 1024] [--height 1024] [--cfg 7.0] \
        [--sampler dpmpp_2m] [--scheduler karras] [--attention sdpa|sage2]

``--attention sage2`` arms SageAttention2 (INT8 QK + FP8 PV, sm120 auto path) for
the INT8 branch ONLY; the FP16 baseline always stays on stock attention. Default
``sdpa`` leaves both branches on stock attention.
"""
import argparse
import os
import sys

import torch

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from pathlib import Path  # noqa: E402


# ---------------------------------------------------------------------------
# ComfyUI bootstrap
# ---------------------------------------------------------------------------

def _install_torchaudio_stub() -> None:
    """comfy.sd imports comfy.ldm.lightricks.vae.audio_vae which hard-imports
    torchaudio; cloud/embedded torch builds often miss it."""
    try:
        import torchaudio  # noqa: F401
        return
    except Exception:
        pass
    import importlib.machinery
    import types

    class _StubModule(types.ModuleType):
        def __init__(self, name, *, is_package=False):
            super().__init__(name)
            if is_package:
                self.__path__ = []

        def __getattr__(self, key):
            if key.startswith("__"):
                raise AttributeError(key)

            def _fn(*args, **kwargs):
                raise NotImplementedError(f"{self.__name__}.{key} stubbed")

            return _fn

    def _stub_mod(name, *, is_package=False):
        if name in sys.modules:
            return
        mod = _StubModule(name, is_package=is_package)
        sys.modules[name] = mod
        parent, _, child = name.rpartition(".")
        if parent:
            parent_mod = sys.modules.get(parent) or _stub_mod(parent, is_package=True)
            setattr(parent_mod, child, mod)

    _stub_mod("torchaudio", is_package=True)
    _stub_mod("torchaudio.functional")
    _stub_mod("torchaudio.transforms")


def _install_comfy_aimdo_stub() -> None:
    """Ensure comfy_aimdo and submodules (malloc_graph, filter, model_vbar) are available.

    In environments with older comfy_aimdo (e.g. 0.4.13) lacking malloc_graph,
    ComfyUI-master v0.35.0 hard-imports comfy_aimdo.malloc_graph in comfy/model_prefetch.py.
    This stub safely provides the minimal interface so import comfy.sample completes.
    """
    import types

    aimdo = sys.modules.get("comfy_aimdo")
    if aimdo is None:
        try:
            import comfy_aimdo as aimdo
        except Exception:
            aimdo = types.ModuleType("comfy_aimdo")
            aimdo.__file__ = "<hswq_comfy_aimdo_stub>"
            aimdo.__path__ = []
            sys.modules["comfy_aimdo"] = aimdo

    # 1. comfy_aimdo.filter
    if "comfy_aimdo.filter" not in sys.modules:
        try:
            import comfy_aimdo.filter  # noqa: F401
        except Exception:
            fmod = types.ModuleType("comfy_aimdo.filter")
            fmod.filter_modules = lambda *a, **k: None
            sys.modules["comfy_aimdo.filter"] = fmod
            try:
                setattr(aimdo, "filter", fmod)
            except Exception:
                pass

    # 2. comfy_aimdo.malloc_graph
    if "comfy_aimdo.malloc_graph" not in sys.modules:
        try:
            import comfy_aimdo.malloc_graph  # noqa: F401
        except Exception:
            class _MallocGraph:
                def __init__(self, *args, **kwargs):
                    self._comfy_active = False
                    self._comfy_cuda_graph_modules = set()
                    self.rogue_count = 0

                def pause(self, *args, **kwargs):
                    pass

                def resume(self, *args, **kwargs):
                    pass

                def push(self, *args, **kwargs):
                    pass

                def pop(self, *args, **kwargs):
                    return False

                def abort(self, *args, **kwargs):
                    pass

                def iterate(self, *args, **kwargs):
                    return False

                def use_stream(self, *args, **kwargs):
                    class _DummyCtx:
                        def __enter__(self):
                            pass

                        def __exit__(self, *a):
                            pass

                    return _DummyCtx()

            mg_mod = types.ModuleType("comfy_aimdo.malloc_graph")
            mg_mod.MallocGraph = _MallocGraph
            mg_mod.record = lambda *args, **kwargs: _MallocGraph()
            sys.modules["comfy_aimdo.malloc_graph"] = mg_mod
            try:
                setattr(aimdo, "malloc_graph", mg_mod)
            except Exception:
                pass

    # 3. comfy_aimdo.model_vbar
    if "comfy_aimdo.model_vbar" not in sys.modules:
        try:
            import comfy_aimdo.model_vbar  # noqa: F401
        except Exception:
            vbar_mod = types.ModuleType("comfy_aimdo.model_vbar")
            vbar_mod.vbar_unpin = lambda *a, **k: None
            vbar_mod.vbar_fault = lambda *a, **k: 0
            vbar_mod.vbar_signature_compare = lambda *a, **k: False
            vbar_mod.vbars_reset_watermark_limits = lambda *a, **k: None
            vbar_mod.vbars_analyze = lambda *a, **k: 0
            vbar_mod.ModelVBAR = type("_VBAR", (), {"__init__": lambda s, *a, **k: None})
            sys.modules["comfy_aimdo.model_vbar"] = vbar_mod
            try:
                setattr(aimdo, "model_vbar", vbar_mod)
            except Exception:
                pass


def setup_comfy(comfy_path: str) -> None:
    bench_dir = Path(__file__).resolve().parent
    repo_dir = bench_dir.parent
    master = repo_dir / "ComfyUI-master"

    if comfy_path and os.path.isdir(comfy_path) and os.path.isdir(os.path.join(comfy_path, "comfy")):
        comfy_root = Path(comfy_path).resolve()
    elif master.is_dir() and (master / "comfy").is_dir():
        comfy_root = master.resolve()
    else:
        raise FileNotFoundError(f"ComfyUI-master not found at {master}")
    bench_dir = Path(__file__).resolve().parent
    repo_dir = bench_dir.parent

    required_paths = [str(comfy_root), str(bench_dir), str(repo_dir)]
    new_sys_path = []
    for p in required_paths:
        if p not in new_sys_path and os.path.isdir(p):
            new_sys_path.append(p)
    for p in sys.path:
        if p not in new_sys_path:
            new_sys_path.append(p)
    sys.path = new_sys_path

    _install_torchaudio_stub()
    _install_comfy_aimdo_stub()

    import comfy.options

    comfy.options.enable_args_parsing(False)

    try:
        import psutil  # noqa: F401
    except Exception:
        import types

        class _VM:
            total = 64 * 1024**3
            available = 32 * 1024**3

        class _Proc:
            def memory_info(self):
                return types.SimpleNamespace(rss=0)

            def memory_full_info(self):
                return types.SimpleNamespace(uss=0)

            def cpu_percent(self, interval=None):
                return 0.0

            def num_threads(self):
                return 1

        ps = types.ModuleType("psutil")
        ps.virtual_memory = lambda: _VM()
        ps.Process = lambda: _Proc()
        sys.modules["psutil"] = ps


def _clear_argv_for_comfy() -> list:
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    return saved


def _restore_argv(saved: list) -> None:
    import sys as _s

    _s.argv = saved


def _hard_free_vram() -> None:
    import gc

    import comfy.model_management as mm

    gc.collect()
    try:
        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ---------------------------------------------------------------------------
# INT8 patches (vendored bench stack, same as int8bench_sdxl.py uses)
# ---------------------------------------------------------------------------

def apply_int8_patches() -> None:
    import comfy.ops

    from int8.comfy_quant_int8 import apply_comfy_quant_int8_patches
    import int8.comfy_quant_int8 as _cq_int8

    apply_comfy_quant_int8_patches()
    print(f"  [BENCH] int8_tensorwise: {'int8_tensorwise' in comfy.ops.QUANT_ALGOS}")
    print(f"  [BENCH] comfy_quant_int8 patched: {_cq_int8._PATCHES_APPLIED}")
    print(
        f"  [BENCH] mixed_precision_ops Conv2d inject: "
        f"{getattr(comfy.ops.mixed_precision_ops, '_hswq_int8_conv_patched', False)}"
    )
    print(f"  [BENCH] patch file: {os.path.abspath(_cq_int8.__file__)}")
    if not _cq_int8._PATCHES_APPLIED:
        raise RuntimeError(
            "comfy_quant_int8 patches failed to apply "
            "(need [BENCH] comfy_quant_int8 patched: True)"
        )


def _load_checkpoint(path: str):
    """Load an SDXL checkpoint ComfyUI-natively; arm INT8 Conv2d load scope only
    when the file looks like a comfy_quant INT8 pack (FP16 baseline loads stock)."""
    import comfy.sd

    from int8.comfy_quant_int8 import (
        checkpoint_looks_like_comfy_quant_int8,
        _int8_quant_conv_scope,
    )

    ckpt = os.path.abspath(path)
    use_int8_scope = checkpoint_looks_like_comfy_quant_int8(ckpt)
    print(f"  [BENCH] INT8 Conv2d load scope: {use_int8_scope}")
    if use_int8_scope:
        with _int8_quant_conv_scope():
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt, output_vae=True, output_clip=True, embedding_directory=None
            )
    else:
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt, output_vae=True, output_clip=True, embedding_directory=None
        )
    return out[0], out[1], out[2]


# ---------------------------------------------------------------------------
# SA2 attention (same implementation as krea2_int8_traj_compare.py; only this
# script's target arch modules differ - SDXL attention goes through
# comfy.ops.scaled_dot_product_attention inside attention_pytorch, which reads
# optimized_attention_masked via comfy.ldm.modules.attention, so patching the
# core module is sufficient; no per-arch from-import binding exists for SDXL)
# ---------------------------------------------------------------------------

_SAGE2_STATS = {
    "calls": 0, "sa2": 0, "fb_mask": 0, "fb_dim": 0, "err": 0,
    "t_sdpa_ms": 0.0, "t_sa2_ms": 0.0,
}


def apply_sage2_attention() -> None:
    import time

    from comfy.ldm.modules import attention as comfy_attention
    from sageattention import sageattn
    from torch.nn.functional import scaled_dot_product_attention as _sdpa

    def attention_sage2(q, k, v, heads, mask=None, attn_precision=None,
                        skip_reshape=False, skip_output_reshape=False, **kw):
        in_dtype = v.dtype
        if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)

        if skip_reshape:
            b, _, _, dim_head = q.shape
            qh, kh, vh = q, k, v
        else:
            b, n_q, _ = q.shape
            dim_head = q.shape[-1] // heads
            qh = q.view(b, n_q, heads, dim_head).transpose(1, 2)
            kh = k.view(b, k.shape[1], heads, dim_head).transpose(1, 2)
            vh = v.view(b, v.shape[1], heads, dim_head).transpose(1, 2)

        use_fallback = (mask is not None) or (dim_head > 256)
        _SAGE2_STATS["calls"] += 1
        torch.cuda.synchronize()
        if use_fallback:
            if mask is not None:
                _SAGE2_STATS["fb_mask"] += 1
            else:
                _SAGE2_STATS["fb_dim"] += 1
            t0 = time.perf_counter()
            out = _sdpa(qh, kh, vh, attn_mask=mask, is_causal=False)
            torch.cuda.synchronize()
            _SAGE2_STATS["t_sdpa_ms"] += (time.perf_counter() - t0) * 1000.0
        else:
            try:
                t0 = time.perf_counter()
                out = sageattn(qh, kh, vh, tensor_layout="HND", is_causal=False)
                torch.cuda.synchronize()
                _SAGE2_STATS["t_sa2_ms"] += (time.perf_counter() - t0) * 1000.0
                _SAGE2_STATS["sa2"] += 1
            except Exception as e:
                _SAGE2_STATS["err"] += 1
                print(f"  [SAGE2] fallback ({type(e).__name__}: {str(e)[:60]})", flush=True)
                out = _sdpa(qh, kh, vh, attn_mask=mask, is_causal=False)

        if skip_output_reshape:
            pass
        else:
            out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out.to(in_dtype)

    comfy_attention.optimized_attention_masked = attention_sage2
    if hasattr(comfy_attention, "optimized_attention"):
        comfy_attention.optimized_attention = attention_sage2
    print("  [SAGE2] attention override armed (sageattn, INT8 QK + FP8 PV, sm120 auto)",
          flush=True)


def unset_sage2_attention() -> None:
    import importlib

    import comfy.ldm.modules.attention as comfy_attention

    importlib.reload(comfy_attention)
    print("  [SAGE2] attention override restored to stock", flush=True)


def print_sage2_attn_stats() -> None:
    s = _SAGE2_STATS
    print(f"  [SAGE2] attention calls: total={s['calls']} sa2={s['sa2']} "
          f"fallback(mask)={s['fb_mask']} fallback(dim)={s['fb_dim']} errors={s['err']}",
          flush=True)
    print(f"  [SAGE2] attention time: sage2={s['t_sa2_ms']:.1f} ms "
          f"sdpa_fallback={s['t_sdpa_ms']:.1f} ms", flush=True)


# ---------------------------------------------------------------------------
# Trajectory capture
# ---------------------------------------------------------------------------

def run_trajectory(model, positive, negative, latent, *, seed, steps, cfg,
                   sampler_name, scheduler):
    """Full denoising; returns (per_step_x, per_step_x0, final_sample)."""
    import comfy.sample as comfy_sample
    import comfy.utils

    noise = comfy_sample.prepare_noise(latent["samples"], seed, None)
    xs, x0s = [], []

    def cb(step, x0, x, total_steps):
        xs.append(x.detach().float().cpu())
        x0s.append(x0.detach().float().cpu())

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy_sample.sample(
        model, noise, steps, cfg, sampler_name, scheduler,
        positive, negative, latent["samples"], denoise=1.0,
        disable_noise=False, start_step=None, last_step=None,
        force_full_denoise=False, noise_mask=None,
        callback=cb, disable_pbar=disable_pbar, seed=seed,
    )
    return xs, x0s, samples


def _cos(a, b):
    a = a.reshape(1, -1).float()
    b = b.reshape(1, -1).float()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=1).item())


def _mse(a, b):
    return float((a.float() - b.float()).pow(2).mean().item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="SDXL deterministic per-step trajectory divergence (FP16 vs ConvRot INT8)"
    )
    ap.add_argument("--fp16", required=True, help="FP16 baseline SDXL checkpoint")
    ap.add_argument("--int8", "--fp8", "--quant", dest="int8_path", required=True,
                    help="ConvRot INT8 quantized SDXL checkpoint")
    ap.add_argument("--comfy_path", required=True, help="ComfyUI-master root")
    ap.add_argument("--prompt",
                    default="masterpiece, best quality, 1girl, solo, standing, simple background")
    ap.add_argument("--negative", default="")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument(
        "--seeds",
        default=(
            "42,137,849,2024,7391,18429,53082,149206,382715,826401,1938502,4710928,"
            "8391642,15820493,36192847,71058294,128491703,285039184,491730285,762019483,"
            "938174026,1409285713,2683910547,3851729406,4195820371"
        ),
        help="comma-separated seeds; same seed = identical noise for both models",
    )
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--sampler", default="dpmpp_2m")
    ap.add_argument("--scheduler", default="karras")
    ap.add_argument("--attention", choices=["sdpa", "sage2"], default="sdpa",
                    help="sage2: patch the INT8 branch attention to SageAttention2 "
                         "(INT8 QK + FP8 PV, sm120 auto path). FP16 baseline stays "
                         "on stock attention. sdpa (default): no attention patch.")
    ap.add_argument("--show-steps", action="store_true",
                    help="print the per-step divergence curve (default: only final per seed)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        print("No seeds given")
        return 1

    setup_comfy(args.comfy_path)
    apply_int8_patches()

    import comfy.model_management as mm
    import comfy.sample as comfy_sample

    fp16_runs = {}
    int8_runs = {}
    times_fp16 = []
    times_int8 = []
    saved_argv = _clear_argv_for_comfy()
    try:
        # --- FP16 baseline (stock ops; INT8 patches only arm INT8 tensors) ---
        print("[FP16] loading checkpoint...")
        fp16, clip, vae = _load_checkpoint(args.fp16)
        latent = {"samples": torch.zeros(
            [1, 4, args.height // 8, args.width // 8],
            device=mm.intermediate_device(),
            dtype=mm.intermediate_dtype(),
        )}
        latent["samples"] = comfy_sample.fix_empty_latent_channels(
            fp16, latent["samples"], latent.get("downscale_ratio_spacial", None), None
        )
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(args.prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(args.negative))
        for s in seeds:
            print(f"[FP16] seed {s}")
            t0 = __import__("time").perf_counter()
            xs, x0s, final = run_trajectory(
                fp16, positive, negative, latent, seed=s, steps=args.steps,
                cfg=args.cfg, sampler_name=args.sampler, scheduler=args.scheduler,
            )
            times_fp16.append(__import__("time").perf_counter() - t0)
            fp16_runs[s] = (xs, x0s, final.detach().float().cpu())
        del fp16, clip, vae
        _hard_free_vram()
        print("  [Offload] FP16 checkpoint unloaded.")

        # --- ConvRot INT8 (patches already applied) ---
        print("[INT8] loading checkpoint...")
        int8, clip8, vae8 = _load_checkpoint(args.int8_path)
        if args.attention == "sage2":
            apply_sage2_attention()
        int8_runs = {}
        for s in seeds:
            print(f"[INT8] seed {s}")
            t0 = __import__("time").perf_counter()
            xs, x0s, final = run_trajectory(
                int8, positive, negative, latent, seed=s, steps=args.steps,
                cfg=args.cfg, sampler_name=args.sampler, scheduler=args.scheduler,
            )
            times_int8.append(__import__("time").perf_counter() - t0)
            int8_runs[s] = (xs, x0s, final.detach().float().cpu())
        if args.attention == "sage2":
            unset_sage2_attention()
            print_sage2_attn_stats()
        del int8, clip8, vae8
        _hard_free_vram()
    finally:
        _restore_argv(saved_argv)

    # --- compare ---
    print("\n" + "=" * 72)
    print("Deterministic per-step latent trajectory divergence (FP16 vs ConvRot INT8)")
    print("=" * 72)
    BIFURC_DROP = 0.05
    SAME_IMG_COS = 0.98
    final_rows = []
    for s in seeds:
        fxs, fx0s, ffinal = fp16_runs[s]
        nxs, nx0s, nfinal = int8_runs[s]
        n_steps = min(len(fxs), len(nxs))
        step_cos = [_cos(fxs[i], nxs[i]) for i in range(n_steps)]
        max_drop = 0.0
        drop_at = 0
        for i in range(1, n_steps):
            d = step_cos[i - 1] - step_cos[i]
            if d > max_drop:
                max_drop, drop_at = d, i
        if args.show_steps:
            print(f"\n--- Seed {s}: per-step ---")
            print(f"{'step':>4} {'x-cos':>8} {'x-MSE':>10} {'x0-cos':>8}")
            for i in range(n_steps):
                print(f"{i+1:>4} {step_cos[i]:>8.5f} {_mse(fxs[i], nxs[i]):>10.3e} "
                      f"{_cos(fx0s[i], nx0s[i]):>8.5f}")
        fin_cos = _cos(ffinal, nfinal)
        fin_mse = _mse(ffinal, nfinal)
        x0_cos = _cos(fx0s[-1], nx0s[-1]) if fx0s and nx0s else float("nan")
        if max_drop > BIFURC_DROP:
            verdict = f"bifurcated @step {drop_at}"
        elif fin_cos >= SAME_IMG_COS:
            verdict = "same-image"
        else:
            verdict = "drifted (different image)"
        final_rows.append((s, fin_cos, fin_mse, x0_cos, verdict, max_drop, drop_at))
        print(f"[seed {s}] final-cos={fin_cos:.5f}  max_step_drop={max_drop:.4f}"
              f"{' @step ' + str(drop_at) if max_drop > BIFURC_DROP else ''}  -> {verdict}")

    print("\n--- Multi-seed summary ---")
    print(f"{'seed':>8} {'final-cos':>10} {'final-mse':>12} {'max-drop':>9} {'verdict':>22}")
    for s, fc, fm, xc, v, md, da in final_rows:
        print(f"{s:>8} {fc:>10.5f} {fm:>12.3e} {md:>9.4f} {v:>22}")
    cos_vals = [r[1] for r in final_rows]
    n_bif = sum(1 for r in final_rows if "bifurcated" in r[4])
    n_diff = sum(1 for r in final_rows if r[4] != "same-image")
    mean_cos = sum(cos_vals) / len(cos_vals)
    print(f"\nfinal-cosine: min={min(cos_vals):.5f}  mean={mean_cos:.5f}  max={max(cos_vals):.5f}")
    print(f"same-image seeds : {len(seeds) - n_diff}/{len(seeds)}")
    print(f"bifurcated seeds : {n_bif}/{len(seeds)}   (sudden trajectory jump = different picture, not degradation)")

    if times_fp16 and times_int8:
        avg_f = sum(times_fp16) / len(times_fp16)
        avg_i = sum(times_int8) / len(times_int8)
        print(f"\navg wall/seed: FP16 {avg_f:.2f}s | INT8 {avg_i:.2f}s "
              f"({(avg_f - avg_i) / avg_f * 100:+.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

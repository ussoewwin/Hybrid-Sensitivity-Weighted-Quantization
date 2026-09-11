#!/usr/bin/env python3
"""Krea2 deterministic trajectory-divergence comparator (FP16 vs quantized artifact).

More fundamental than the single-seed RGB SSIM of krea2_convrot_nvfp4_bench.py:

  * same seed -> identical initial noise for both models (fully deterministic)
  * captures the latent ``x`` AND the model's x0-prediction at EVERY denoising step
  * reports per-step + final latent MSE / cosine, aggregated over multiple seeds

This isolates the quantization error from ODE-solver and VAE-decode confounds and
gives a per-step divergence curve, so you can see WHERE and HOW FAST FP16 and the
quantized artifact drift apart. Cosine >= ~0.99 on the final latent with a flat
per-step curve = artifact is effectively indistinguishable in latent space.

Self-contained: the ComfyUI bootstrap / model loading / quant-patch helpers are
embedded below, importing only the live krea2_convrot_nvfp4 package (and int8).
No dependency on archives/.

``--attention sage2`` arms SageAttention2 (INT8 QK + FP8 PV, sm120 auto path)
for the NVFP4 run ONLY; the FP16 baseline always stays on stock ComfyUI
attention. Default ``sdpa`` leaves both branches on stock attention.

Usage:
    python krea2_traj_compare.py \
        --fp16 <base.safetensors> --nvfp4 <hybrid.safetensors> \
        --clip_path <clip.safetensors> --comfy_path <ComfyUI-master> \
        [--seeds "42,1337,7,2024,555"] [--steps 25] [--prompt "..."] [--tc | --parity] [--blockonly]
"""
import argparse
import gc
import os
import sys
from pathlib import Path

import torch

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

def _clear_argv_for_comfy() -> list[str]:
    """ComfyUI cli_args swallows unknown flags; keep only argv[0] during import."""
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    return saved


def _restore_argv(saved: list[str]) -> None:
    sys.argv = saved


def _install_torchaudio_stub() -> None:
    """Prevent real torchaudio from loading if comfy.sd is pulled in.

    comfy.sd imports comfy.ldm.lightricks.vae.audio_vae, which does a hard
    ``import torchaudio``. On cloud hosts torch/torchaudio CUDA builds often
    mismatch (e.g. torch 13.2 vs torchaudio 13.0) and abort before bench load.
    Krea2 NVFP4 bench uses CLIPType.KREA2 / DiT only — never AudioVAE — so
    replace torchaudio in sys.modules with a local stub.
    Does not touch ComfyUI-master.
    """
    import importlib.machinery
    import types

    for key in list(sys.modules):
        if key == "torchaudio" or key.startswith("torchaudio."):
            del sys.modules[key]

    def _stub_mod(name: str, *, is_package: bool = False):
        # transformers uses importlib.util.find_spec("torchaudio"); a ModuleType
        # without __spec__ raises ValueError: torchaudio.__spec__ is None.
        mod = types.ModuleType(name)
        mod.__file__ = "<hswq_torchaudio_stub>"
        if is_package:
            mod.__path__ = []
            spec = importlib.machinery.ModuleSpec(
                name, loader=None, is_package=True
            )
            spec.submodule_search_locations = []
        else:
            spec = importlib.machinery.ModuleSpec(name, loader=None)
        mod.__spec__ = spec
        return mod

    ta = _stub_mod("torchaudio", is_package=True)
    functional = _stub_mod("torchaudio.functional")

    def _resample(waveform, orig_freq, new_freq, *args, **kwargs):
        return waveform

    functional.resample = _resample

    transforms = _stub_mod("torchaudio.transforms")

    class _MelSpectrogram:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, x):
            return x

        def to(self, *args, **kwargs):
            return self

    class _MelScale:
        def __init__(self, *args, **kwargs):
            pass

    transforms.MelSpectrogram = _MelSpectrogram
    transforms.MelScale = _MelScale

    ta.functional = functional
    ta.transforms = transforms
    sys.modules["torchaudio"] = ta
    sys.modules["torchaudio.functional"] = functional
    sys.modules["torchaudio.transforms"] = transforms

def setup_comfy(comfy_path: str) -> None:
    comfy_root = Path(comfy_path).resolve()
    if not comfy_root.is_dir():
        raise FileNotFoundError(f"--comfy_path not found: {comfy_root}")
    # Prefer this tree for comfy.* imports
    sys.path = [str(comfy_root)] + [p for p in sys.path if Path(p).resolve() != comfy_root]

    # Always stub before any comfy.* import (real torchaudio may CUDA-mismatch).
    _install_torchaudio_stub()

    # Before first comfy.quant_ops import: attach missing kitchen tensor exports
    # (Asym / kitchen ConvRotW4A4 import-gate) so bulk-import succeeds → Branch A.
    # Krea2 ConvRot load+forward stays in benchmark/krea2_convrot_nvfp4 only (not ComfyUI).
    from krea2_convrot_nvfp4.kitchen_quant_ops_repair import (
        ensure_kitchen_quant_ops,
        prebind_missing_kitchen_tensor_exports,
    )

    prebind_missing_kitchen_tensor_exports()

    import comfy.options

    comfy.options.enable_args_parsing(False)

    # Lightweight stubs (same pattern as nvfp4bench_sdxl / int8 benches)
    try:
        import comfy_aimdo  # noqa: F401
    except Exception:
        import types

        m = types.ModuleType("comfy_aimdo")
        m.__file__ = "<stub>"
        m.__path__ = []
        sys.modules["comfy_aimdo"] = m
        sys.modules["comfy_aimdo.filter"] = types.ModuleType("comfy_aimdo.filter")
        sys.modules["comfy_aimdo.filter"].filter_modules = lambda *a, **k: None

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

    # Resolve quant_ops now (after prebind) and apply Branch A/B before model load.
    # Branch A: healthy → zero rebind. Branch B: stubs → submodule rebind only.
    import comfy.quant_ops  # noqa: F401

    ensure_kitchen_quant_ops()


SSIM_TARGET = 0.9


def require_convrot_parity_forward() -> None:
    """Fail if Linear.forward is not the ConvRot act-rotate parity wrapper."""
    import comfy.ops

    lin_fwd = comfy.ops.mixed_precision_ops().Linear.forward
    if getattr(lin_fwd, "_hswq_nvfp4_full_forward", False):
        raise RuntimeError(
            "ConvRot bench: Linear.forward still has HSWQ TC wrap "
            "(_hswq_nvfp4_full_forward); SSIM would be destroyed"
        )
    if not getattr(lin_fwd, "_hswq_nvfp4_convrot_parity", False):
        raise RuntimeError(
            "ConvRot bench: Linear.forward missing _hswq_nvfp4_convrot_parity "
            "(online act rotation required for W@H^T weights)"
        )


def _forbid_benchmark_nvfp4_import() -> None:
    """Hard ban: this bench must never pull benchmark/nvfp4."""
    bad = [
        k
        for k in sys.modules
        if k == "nvfp4" or k.startswith("nvfp4.")
    ]
    if bad:
        raise RuntimeError(
            "FORBIDDEN: imported benchmark/nvfp4 modules "
            f"{bad[:8]}; use krea2_convrot_nvfp4 only"
        )


def apply_quant_patches(mode: str = "tc"):
    """Runtime monkey-patches for NVFP4 + INT8 protect.

    mode='tc': Native HSWQ hardware Tensor Core forward (scaled_mm_nvfp4, no dequant).
    mode='parity': ComfyUI stock ops.py Linear.forward + dequantization.
    """
    import comfy.ops
    from krea2_convrot_nvfp4.comfy_quant_nvfp4 import apply_comfy_quant_nvfp4_patches
    from krea2_convrot_nvfp4.nvfp4_comfy_parity import apply_nvfp4_comfy_parity
    import krea2_convrot_nvfp4.comfy_quant_nvfp4 as _cq_nvfp4

    from int8.comfy_quant_int8 import apply_comfy_quant_int8_patches
    import int8.comfy_quant_int8 as _cq_int8

    apply_comfy_quant_nvfp4_patches()
    if mode == "parity":
        if not apply_nvfp4_comfy_parity():
            raise RuntimeError("krea2_convrot_nvfp4 ComfyUI-only parity failed to apply")
        require_convrot_parity_forward()
        print(
            "  [CONVROT] Parity forward armed: "
            "stock Comfy GEMM + fast O(N log N) float32 butterfly act-rotate (zero accumulation error)"
        )
    else:
        print(
            "  [CONVROT] Native Hardware Tensor Core forward armed: "
            "scaled_mm_nvfp4 on Blackwell Tensor Cores (fastest, zero weight dequant)"
        )
    print(f"  [BENCH] nvfp4 patch file: {os.path.abspath(_cq_nvfp4.__file__)}")
    print(f"  [BENCH] comfy_quant_nvfp4 patched: {_cq_nvfp4._PATCHES_APPLIED}")

    apply_comfy_quant_int8_patches()
    print(f"  [BENCH] int8_tensorwise: {'int8_tensorwise' in comfy.ops.QUANT_ALGOS}")
    print(f"  [BENCH] comfy_quant_int8 patched: {_cq_int8._PATCHES_APPLIED}")
    print(
        f"  [BENCH] mixed_precision_ops Conv2d inject: "
        f"{getattr(comfy.ops.mixed_precision_ops, '_hswq_int8_conv_patched', False)}"
    )
    print(f"  [BENCH] int8 patch file: {os.path.abspath(_cq_int8.__file__)}")
    if not _cq_int8._PATCHES_APPLIED:
        raise RuntimeError(
            "comfy_quant_int8 patches failed to apply "
            "(need [BENCH] comfy_quant_int8 patched: True)"
        )

    _forbid_benchmark_nvfp4_import()
    print(
        f"  [BENCH] SSIM target >={SSIM_TARGET} "
        "(ConvRot NVFP4 + INT8 protect + ComfyUI stock GEMM + online act rotate)"
    )


def set_hf_token(token: str | None) -> None:
    if not token:
        return
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def encode_prompt(clip, prompt: str):
    tokens = clip.tokenize(prompt)
    return clip.encode_from_tokens_scheduled(tokens)


def make_empty_latent(model, width: int, height: int, batch: int = 1) -> dict:
    """16-ch empty latent; fix_empty_latent_channels upgrades Wan21 to 5D [B,C,T,H,W]."""
    import comfy.model_management as mm
    import comfy.sample as comfy_sample

    device = mm.intermediate_device()
    latent = torch.zeros([batch, 16, height // 8, width // 8], device=device)
    # API: (model, latent_tensor) -> tensor  (see EmptyLatentImage / comfy.sample)
    latent = comfy_sample.fix_empty_latent_channels(model, latent)
    return {"samples": latent}


def _hard_free_vram() -> None:
    """Drop loaded models and return VRAM to the pool (CPU-offload path)."""
    import comfy.model_management as mm

    mm.unload_all_models()
    # force=True path still runs empty_cache + ipc_collect on CUDA.
    mm.soft_empty_cache(force=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def _load_diffusion_model(unet_path: str):
    """Load DiT; wrap INT8-protect layers in Conv2d inject scope when present."""
    import comfy.sd
    from int8.comfy_quant_int8 import (
        _int8_quant_conv_scope,
        checkpoint_looks_like_comfy_quant_int8,
    )
    from krea2_convrot_nvfp4.comfy_quant_nvfp4 import checkpoint_looks_like_comfy_quant_nvfp4

    looks_nvfp4 = checkpoint_looks_like_comfy_quant_nvfp4(unet_path)
    use_int8_scope = checkpoint_looks_like_comfy_quant_int8(unet_path)
    print(f"  [BENCH] NVFP4 comfy_quant detect: {looks_nvfp4}")
    print(f"  [BENCH] INT8 Conv2d load scope: {use_int8_scope}")
    if use_int8_scope:
        with _int8_quant_conv_scope():
            return comfy.sd.load_diffusion_model(unet_path, {})
    return comfy.sd.load_diffusion_model(unet_path, {})




_SAGE2_STATS = {
    "calls": 0, "sa2": 0, "fb_mask": 0, "fb_dim": 0, "err": 0,
    "t_sdpa_ms": 0.0, "t_sa2_ms": 0.0,
}


def apply_sage2_attention() -> None:
    """Arm SageAttention2 for the NVFP4 run (INT8 QK + FP8 PV, sm120 auto path).

    Layout handling is identical to the validated SA2 path (zi_traj_compare /
    krea2_int8_traj_compare): stock attention_pytorch semantics with
    skip_reshape=True ([B,H,N,D] in) and transpose(1,2) BEFORE reshape to
    [B,N,H*D] on the way out. SDPA fallback for mask / head_dim > 256 /
    exceptions. Krea2's SingleStreamDiT binds optimized_attention_masked via a
    from-import, so the arch module is patched individually.
    """
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
    import importlib as _il
    try:
        _mod = _il.import_module("comfy.ldm.krea2.model")
        if hasattr(_mod, "optimized_attention_masked"):
            _mod.optimized_attention_masked = attention_sage2
    except Exception:
        pass
    print("  [SAGE2] attention override armed (sageattn, INT8 QK + FP8 PV, sm120 auto)",
          flush=True)


def unset_sage2_attention() -> None:
    """Restore stock attention after the NVFP4 run (reload binding modules)."""
    import importlib

    import comfy.ldm.modules.attention as comfy_attention

    importlib.reload(comfy_attention)
    try:
        importlib.reload(importlib.import_module("comfy.ldm.krea2.model"))
    except Exception:
        pass
    print("  [SAGE2] attention override restored to stock", flush=True)


def print_sage2_attn_stats() -> None:
    s = _SAGE2_STATS
    print(f"  [SAGE2] attention calls: total={s['calls']} sa2={s['sa2']} "
          f"fallback(mask)={s['fb_mask']} fallback(dim)={s['fb_dim']} errors={s['err']}",
          flush=True)
    print(f"  [SAGE2] attention time: sage2={s['t_sa2_ms']:.1f} ms "
          f"sdpa_fallback={s['t_sdpa_ms']:.1f} ms", flush=True)


def run_trajectory(model, positive, negative, latent, *, seed, steps, cfg,
                   sampler_name, scheduler):
    """Run full denoising; return (per_step_x, per_step_x0, final_sample).

    per_step_x   : noisy latent after each step (list of CPU float tensors)
    per_step_x0  : model x0 prediction at each step (list of CPU float tensors)
    final_sample : out["samples"] (the final denoised latent)
    """
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
        force_full_denoise=True, noise_mask=None,
        callback=cb, disable_pbar=disable_pbar, seed=seed,
    )
    return xs, x0s, samples


def _cos(a, b):
    a = a.reshape(1, -1).float()
    b = b.reshape(1, -1).float()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=1).item())


def _mse(a, b):
    return float((a.float() - b.float()).pow(2).mean().item())


def parse_args():
    ap = argparse.ArgumentParser(
        description="Krea2 deterministic per-step trajectory divergence (FP16 vs quantized)"
    )
    ap.add_argument("--fp16", required=True)
    ap.add_argument("--nvfp4", required=True)
    ap.add_argument("--clip_path", required=True)
    ap.add_argument("--comfy_path", required=True)
    ap.add_argument("--token", default=None)
    ap.add_argument("--prompt", default="A beautiful cyberpunk city at night, high detail.")
    ap.add_argument("--negative", default="")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seeds", default="42,1337,7,2024,555",
                    help="comma-separated seeds; same seed = identical noise for both models")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--tc", action="store_true",
                    help="native Tensor Core NVFP4 path (scaled_mm_nvfp4; default)")
    ap.add_argument("--parity", action="store_true",
                    help="stock dequant parity path instead of Tensor Core")
    ap.add_argument("--blockonly", action="store_true",
                    help="HSWQ_NVFP4_BLOCKONLY=1: block-scale-only act scale "
                         "(scale_a exactly 1.0; no amax, no calib input_scale; "
                         "block scales carry the range) — fastest TC act path")
    ap.add_argument("--attention", choices=["sdpa", "sage2"], default="sdpa",
                    help="sage2: patch the NVFP4 branch attention to SageAttention2 "
                         "(INT8 QK + FP8 PV, sm120 auto path). FP16 baseline stays "
                         "on stock attention. sdpa (default): no attention patch.")
    ap.add_argument("--show-steps", action="store_true",
                    help="print the per-step divergence curve (default: only final per seed)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.tc and args.parity:
        print("--tc and --parity are mutually exclusive", flush=True)
        return 2
    mode = "parity" if args.parity else "tc"
    if args.blockonly:
        os.environ["HSWQ_NVFP4_BLOCKONLY"] = "1"
        print("[BLOCKONLY] HSWQ_NVFP4_BLOCKONLY=1 — block-scale-only act scale, amax/call skipped", flush=True)
    set_hf_token(args.token)

    saved_argv = _clear_argv_for_comfy()
    try:
        setup_comfy(args.comfy_path)

        import folder_paths  # noqa: F401
        import comfy.model_management as mm
        import comfy.sd

        mm.get_torch_device()

        _cpu = torch.device("cpu")
        print("Loading CLIP on CPU (Krea2 / Qwen3-VL-4B)...")
        clip = comfy.sd.load_clip(
            ckpt_paths=[args.clip_path],
            embedding_directory=None,
            clip_type=comfy.sd.CLIPType.KREA2,
            model_options={"load_device": _cpu, "offload_device": _cpu, "initial_device": _cpu},
        )
        positive = encode_prompt(clip, args.prompt)
        negative = encode_prompt(clip, args.negative) if args.negative else encode_prompt(clip, "")
        if getattr(clip, "cond_stage_model", None) is not None:
            clip.cond_stage_model.cpu()
        if getattr(clip, "patcher", None) is not None:
            mm.unload_model_and_clones(clip.patcher)
        del clip
        _hard_free_vram()
        print("  [Offload] CLIP unloaded.")

        # --- FP16 (stock ops, before any patch) ---
        fp16 = _load_diffusion_model(args.fp16)
        latent = make_empty_latent(fp16, args.width, args.height, batch=1)
        fp16_runs = {}
        for s in seeds:
            print(f"[FP16] seed {s}")
            xs, x0s, final = run_trajectory(
                fp16, positive, negative, latent, seed=s, steps=args.steps,
                cfg=args.cfg, sampler_name=args.sampler, scheduler=args.scheduler,
            )
            fp16_runs[s] = (xs, x0s, final.detach().float().cpu())
        del fp16
        _hard_free_vram()

        # --- NVFP4 (patched) ---
        print(f"Applying NVFP4 ConvRot mode='{mode}' + INT8 + addmm patches...")
        apply_quant_patches(mode=mode)
        nv = _load_diffusion_model(args.nvfp4)
        if args.attention == "sage2":
            apply_sage2_attention()
        nv_runs = {}
        for s in seeds:
            print(f"[NVFP4] seed {s}")
            xs, x0s, final = run_trajectory(
                nv, positive, negative, latent, seed=s, steps=args.steps,
                cfg=args.cfg, sampler_name=args.sampler, scheduler=args.scheduler,
            )
            nv_runs[s] = (xs, x0s, final.detach().float().cpu())
        if args.attention == "sage2":
            unset_sage2_attention()
            print_sage2_attn_stats()
        del nv
        _hard_free_vram()
    finally:
        _restore_argv(saved_argv)

    # --- compare ---
    print("\n" + "=" * 72)
    print("Deterministic per-step latent trajectory divergence (FP16 vs NVFP4)")
    print("=" * 72)
    BIFURC_DROP = 0.05   # single-step cosine drop threshold = sudden jump (different image)
    SAME_IMG_COS = 0.98  # final cosine above this = same picture (not merely different)
    final_rows = []
    for s in seeds:
        fxs, fx0s, ffinal = fp16_runs[s]
        nxs, nx0s, nfinal = nv_runs[s]
        n_steps = min(len(fxs), len(nxs))
        step_cos = [_cos(fxs[i], nxs[i]) for i in range(n_steps)]
        # sudden single-step cosine drop = trajectory jumped to another image
        max_drop = 0.0
        drop_at = 0
        for i in range(1, n_steps):
            d = step_cos[i - 1] - step_cos[i]
            if d > max_drop:
                max_drop, drop_at = d, i
        if args.show_steps:
            print(f"\n--- Seed {s}: per-step (x = noisy latent, x0 = model prediction) ---")
            print(f"{'step':>4} {'x-cos':>8} {'x-MSE':>10} {'x0-cos':>8} {'x0-MSE':>10}")
            for i in range(n_steps):
                print(f"{i+1:>4} {step_cos[i]:>8.5f} {_mse(fxs[i], nxs[i]):>10.3e} "
                      f"{_cos(fx0s[i], nx0s[i]):>8.5f} {_mse(fx0s[i], nx0s[i]):>10.3e}")
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
    print(f"\nfinal-cosine: min={min(cos_vals):.5f}  mean={sum(cos_vals)/len(cos_vals):.5f}  max={max(cos_vals):.5f}")
    print(f"same-image seeds : {len(seeds) - n_diff}/{len(seeds)}")
    print(f"bifurcated seeds : {n_bif}/{len(seeds)}   (sudden trajectory jump = different picture, not degradation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

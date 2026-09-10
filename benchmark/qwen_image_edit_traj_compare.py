#!/usr/bin/env python3
"""Qwen Image Edit deterministic trajectory-divergence comparator (FP16 vs ConvRot INT8).

Per-step latent trajectory comparison between BF16 and ConvRot INT8 models.
Uses the same callback-based per-step capture as krea2_traj_compare.py.

``--attention sage2`` arms SageAttention2 (INT8 QK + FP8 PV, sm120 auto path)
for the ConvRot INT8 runs ONLY; the FP16 baseline always stays on stock
ComfyUI attention. Default ``sdpa`` leaves both branches on stock attention.

Reuses benchmark/qi_int8_bench.py for ComfyUI bootstrap / model loading / prompt encoding.

Usage:
    python benchmark/qwen_image_edit_traj_compare.py \
        --fp16 <bf16.safetensors> --int8 <convrot_int8.safetensors> \
        --clip_path <clip.safetensors> --comfy_path <ComfyUI-master> \
        [--seeds "42,1337,7,2024,555"] [--steps 12] [--cfg 2.5]
"""
import argparse
import importlib.util
import os
import sys

import torch

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

BENCH = os.path.join(_BENCH_DIR, "qi_int8_bench.py")
_spec = importlib.util.spec_from_file_location("qi_bench", BENCH)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


_SAGE2_STATS = {
    "calls": 0, "sa2": 0, "fb_mask": 0, "fb_dim": 0, "err": 0,
    "t_sdpa_ms": 0.0, "t_sa2_ms": 0.0,
}


def apply_sage2_attention() -> None:
    """Arm SageAttention2 for the INT8 run (INT8 QK + FP8 PV, sm120 auto path).

    Layout handling matches the stock attention_pytorch semantics (validated
    2026-09-10 on Z Image): skip_reshape=True takes q,k,v as [B,H,N,D] and the
    output needs transpose(1,2) BEFORE reshape to [B,N,H*D]. SDPA fallback for
    mask / head_dim > 256 / exceptions. Patches every module that binds
    optimized_attention_masked via from-imports (core + comfy.ldm.qwen_image.model).
    Arm AFTER the FP16 baseline and BEFORE the INT8 runs; call
    unset_sage2_attention() afterwards.
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
    for mod_name in ("comfy.ldm.qwen_image.model",):
        try:
            import importlib as _il
            _mod = _il.import_module(mod_name)
            if hasattr(_mod, "optimized_attention_masked"):
                _mod.optimized_attention_masked = attention_sage2
        except Exception:
            pass
    print("  [SAGE2] attention override armed (sageattn, INT8 QK + FP8 PV, sm120 auto)",
          flush=True)


def unset_sage2_attention() -> None:
    """Restore stock attention after the INT8 run (reload binding modules)."""
    import importlib

    import comfy.ldm.modules.attention as comfy_attention

    importlib.reload(comfy_attention)
    try:
        importlib.reload(importlib.import_module("comfy.ldm.qwen_image.model"))
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
    """Run full denoising; return (per_step_x, per_step_x0, final_sample)."""
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


def parse_args():
    ap = argparse.ArgumentParser(
        description="Qwen Image Edit deterministic per-step trajectory divergence (FP16 vs ConvRot INT8)"
    )
    ap.add_argument("--fp16", required=True, help="BF16/FP16 model path")
    ap.add_argument("--int8", required=True, help="ConvRot INT8 quantized model path")
    ap.add_argument("--clip_path", required=True, help="CLIP / text encoder path")
    ap.add_argument("--comfy_path", required=True, help="ComfyUI-master root")
    ap.add_argument("--token", default=None)
    ap.add_argument("--prompt", default="masterpiece, best quality, 1girl, solo, standing, simple background")
    ap.add_argument("--negative", default="")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seeds", default="42,1337,7,2024,555",
                    help="comma-separated seeds; same seed = identical noise for both models")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--cfg", type=float, default=2.5)
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--attention", choices=["sdpa", "sage2"], default="sdpa",
                    help="sage2: patch the INT8 branch attention to SageAttention2 "
                         "(INT8 QK + FP8 PV, sm120 auto path). FP16 baseline stays "
                         "on stock attention. sdpa (default): no attention patch.")
    ap.add_argument("--show-steps", action="store_true",
                    help="print the per-step divergence curve (default: only final per seed)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    bench.set_hf_token(args.token)

    saved_argv = bench._clear_argv_for_comfy()
    try:
        bench.setup_comfy(args.comfy_path)
        bench.apply_int8_patches()

        import folder_paths  # noqa: F401
        import comfy.model_management as mm
        import comfy.sd

        mm.get_torch_device()

        _cpu = torch.device("cpu")
        print("Loading CLIP on CPU...")
        clip = comfy.sd.load_clip(
            ckpt_paths=[args.clip_path],
            embedding_directory=None,
            model_options={"load_device": _cpu, "offload_device": _cpu, "initial_device": _cpu},
        )
        positive = bench.encode_prompt(clip, args.prompt)
        negative = bench.encode_prompt(clip, args.negative)
        if getattr(clip, "cond_stage_model", None) is not None:
            clip.cond_stage_model.cpu()
        if getattr(clip, "patcher", None) is not None:
            mm.unload_model_and_clones(clip.patcher)
        del clip
        bench._hard_free_vram()
        print("  [Offload] CLIP unloaded.")

        # --- FP16 ---
        fp16 = bench._load_diffusion_model(args.fp16)
        latent = bench.make_empty_latent(fp16, args.width, args.height, batch=1)
        fp16_runs = {}
        for s in seeds:
            print(f"[FP16] seed {s}")
            xs, x0s, final = run_trajectory(
                fp16, positive, negative, latent, seed=s, steps=args.steps,
                cfg=args.cfg, sampler_name=args.sampler, scheduler=args.scheduler,
            )
            fp16_runs[s] = (xs, x0s, final.detach().float().cpu())
        del fp16
        bench._hard_free_vram()

        # --- INT8 (patches already applied) ---
        int8 = bench._load_diffusion_model(args.int8)
        if args.attention == "sage2":
            apply_sage2_attention()
        int8_runs = {}
        for s in seeds:
            print(f"[INT8] seed {s}")
            xs, x0s, final = run_trajectory(
                int8, positive, negative, latent, seed=s, steps=args.steps,
                cfg=args.cfg, sampler_name=args.sampler, scheduler=args.scheduler,
            )
            int8_runs[s] = (xs, x0s, final.detach().float().cpu())
        if args.attention == "sage2":
            unset_sage2_attention()
            print_sage2_attn_stats()
        del int8
        bench._hard_free_vram()
    finally:
        bench._restore_argv(saved_argv)

    # --- compare ---
    print("\n" + "=" * 72)
    print("Deterministic per-step latent trajectory divergence (FP16 vs INT8)")
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

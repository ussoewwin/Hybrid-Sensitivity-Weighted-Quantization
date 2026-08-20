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

Reuses the existing benchmark module for ComfyUI bootstrap / model loading /
prompt encoding (no changes to krea2_convrot_nvfp4_bench.py).

Usage:
    python krea2_traj_compare.py \
        --fp16 <base.safetensors> --nvfp4 <hybrid.safetensors> \
        --clip_path <clip.safetensors> --comfy_path <ComfyUI-master> \
        [--seeds "42,1337,7,2024,555"] [--steps 25] [--prompt "..."] [--mode tc|parity]
"""
import argparse
import importlib.util
import os
import sys

import torch

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

BENCH = os.path.join(_BENCH_DIR, "krea2_convrot_nvfp4_bench.py")
_spec = importlib.util.spec_from_file_location("krea2_bench", BENCH)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


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
    ap.add_argument("--mode", choices=["tc", "parity"], default="tc")
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
        positive = bench.encode_prompt(clip, args.prompt)
        negative = bench.encode_prompt(clip, args.negative) if args.negative else bench.encode_prompt(clip, "")
        if getattr(clip, "cond_stage_model", None) is not None:
            clip.cond_stage_model.cpu()
        if getattr(clip, "patcher", None) is not None:
            mm.unload_model_and_clones(clip.patcher)
        del clip
        bench._hard_free_vram()
        print("  [Offload] CLIP unloaded.")

        # --- FP16 (stock ops, before any patch) ---
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

        # --- NVFP4 (patched) ---
        print(f"Applying NVFP4 ConvRot mode='{args.mode}' + INT8 + addmm patches...")
        bench.apply_quant_patches(mode=args.mode)
        nv = bench._load_diffusion_model(args.nvfp4)
        nv_runs = {}
        for s in seeds:
            print(f"[NVFP4] seed {s}")
            xs, x0s, final = run_trajectory(
                nv, positive, negative, latent, seed=s, steps=args.steps,
                cfg=args.cfg, sampler_name=args.sampler, scheduler=args.scheduler,
            )
            nv_runs[s] = (xs, x0s, final.detach().float().cpu())
        del nv
        bench._hard_free_vram()
    finally:
        bench._restore_argv(saved_argv)

    # --- compare ---
    print("\n" + "=" * 72)
    print("Deterministic per-step latent trajectory divergence (FP16 vs NVFP4)")
    print("=" * 72)
    final_rows = []
    for s in seeds:
        fxs, fx0s, ffinal = fp16_runs[s]
        nxs, nx0s, nfinal = nv_runs[s]
        n_steps = min(len(fxs), len(nxs))
        if args.show_steps:
            print(f"\n--- Seed {s}: per-step (x = noisy latent, x0 = model prediction) ---")
            print(f"{'step':>4} {'x-cos':>8} {'x-MSE':>10} {'x0-cos':>8} {'x0-MSE':>10}")
            for i in range(n_steps):
                print(f"{i+1:>4} {_cos(fxs[i], nxs[i]):>8.5f} {_mse(fxs[i], nxs[i]):>10.3e} "
                      f"{_cos(fx0s[i], nx0s[i]):>8.5f} {_mse(fx0s[i], nx0s[i]):>10.3e}")
        fin_cos = _cos(ffinal, nfinal)
        fin_mse = _mse(ffinal, nfinal)
        # also final x0 comparison (last model prediction)
        x0_cos = _cos(fx0s[-1], nx0s[-1]) if fx0s and nx0s else float("nan")
        final_rows.append((s, fin_cos, fin_mse, x0_cos))
        print(f"[seed {s}] final latent cosine={fin_cos:.5f}  mse={fin_mse:.4e}  "
              f"last-x0 cosine={x0_cos:.5f}")

    print("\n--- Multi-seed summary ---")
    print(f"{'seed':>8} {'final-latent-cos':>18} {'final-latent-mse':>18} {'last-x0-cos':>14}")
    for s, fc, fm, xc in final_rows:
        print(f"{s:>8} {fc:>18.5f} {fm:>18.4e} {xc:>14.5f}")
    cos_vals = [r[1] for r in final_rows]
    mse_vals = [r[2] for r in final_rows]
    print(f"\nfinal-latent cosine: min={min(cos_vals):.5f}  mean={sum(cos_vals)/len(cos_vals):.5f}  max={max(cos_vals):.5f}")
    print(f"final-latent mse   : min={min(mse_vals):.4e}  mean={sum(mse_vals)/len(mse_vals):.4e}  max={max(mse_vals):.4e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

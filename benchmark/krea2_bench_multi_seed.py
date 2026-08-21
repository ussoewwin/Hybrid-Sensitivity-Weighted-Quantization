#!/usr/bin/env python3
"""Krea2 ConvRot NVFP4 + INT8-protect Multi-Seed Benchmark
================================================================
20 random seeds, SSIM/Latent cosine per seed, aggregated summary.

Based on krea2_convrot_nvfp4_bench.py but loops sampling over N seeds
with model loaded only once per branch.

Usage:
    python benchmark/krea2_bench_multi_seed.py \
        --fp16 <bf16.safetensors> --nvfp4 <nvfp4.safetensors> \
        --clip_path <qwen3vl.safetensors> \
        --comfy_path <ComfyUI-master> [--seeds 20] [--seed_base 42]
        [--steps 12] [--width 1024] [--height 1024] [--no-vae]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time

import numpy as np
import torch
from pathlib import Path


def _clear_argv_for_comfy():
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    return saved


def _restore_argv(saved):
    sys.argv = saved


def set_hf_token(token):
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def _hard_free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _report_gpu_memory(tag):
    if not torch.cuda.is_available():
        print(f"  [GPU] {tag}: nvidia-smi used=0 (no CUDA)")
        return
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        used = int(r.stdout.strip().split("\n")[0])
    except Exception:
        used = -1
    alloc = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    print(f"  [GPU] {tag}: nvidia-smi used={used} MiB  torch alloc={alloc:.0f} MiB reserved={reserved:.0f} MiB")


def _diag_model_sampling(model, label):
    d = model.model.model_sampling
    diag = {}
    for k in ("model_type", "ms_type", "sigma_min", "sigma_max"):
        v = getattr(d, k, None)
        if isinstance(v, float):
            diag[k] = round(v, 6)
        else:
            diag[k] = v
    print(f"  [{label}] model_type={diag.get('model_type')} model_sampling={diag.get('ms_type')} sigma_min={diag.get('sigma_min')} sigma_max={diag.get('sigma_max')}")
    return diag


def _nvidia_smi_used_mib():
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return int(r.stdout.strip().split("\n")[0])
    except Exception:
        return None


def setup_comfy(comfy_path):
    import importlib
    for key in list(sys.modules):
        if key == "comfy" or key.startswith("comfy."):
            del sys.modules[key]
    if comfy_path in sys.path:
        sys.path.remove(comfy_path)
    sys.path.insert(0, comfy_path)
    mod = importlib.import_module("comfy")
    if os.path.abspath(getattr(mod, "__path__", [None])[0] or "") != os.path.abspath(os.path.join(comfy_path, "comfy")):
        raise ImportError(f"comfy resolved outside comfy_path")
    try:
        import comfy.options
        comfy.options.enable_args_parsing(False)
    except ImportError:
        pass


def encode_prompt(clip, prompt_text):
    return clip.encode(prompt_text)


def make_empty_latent(model, w, h, batch=1):
    import comfy.utils
    return comfy.utils.latent_image_to_tensor(comfy.utils.make_empty_latent(w, h, batch, device=model.device))


def _load_diffusion_model(unet_path):
    import comfy.sd
    import comfy.model_management as mm
    mm.get_torch_device()
    model = comfy.sd.load_diffusion_model(unet_path, {})
    return model


def load_model_for_branch(*, label, unet_path, args):
    """Load UNet once. Returns (model, latent_shape_info, diag_dict)."""
    print(f"\n=== {label}: loading UNet ===")
    print(f"  path: {unet_path}")
    _report_gpu_memory(f"before_load/{label}")
    t0 = time.perf_counter()
    model = _load_diffusion_model(unet_path)
    load_s = time.perf_counter() - t0
    print(f"  load: {load_s:.2f}s")
    diag = _diag_model_sampling(model, label)
    return model, diag


def sample_with_seed(model, positive, negative, latent_shape, *, seed, steps, cfg, sampler_name, scheduler):
    """Run one sampling pass with the given seed. Returns (latent_cpu, sample_time, peak_gb)."""
    import comfy.sample as comfy_sample
    import comfy.utils
    import latent_preview

    latent = {"samples": latent_shape.clone()}
    noise = comfy_sample.prepare_noise(latent["samples"], seed, None)
    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    samples = comfy_sample.sample(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent["samples"],
        denoise=1.0,
        disable_noise=False,
        start_step=None,
        last_step=None,
        force_full_denoise=True,
        noise_mask=None,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sample_s = time.perf_counter() - t1
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

    return samples.detach().float().cpu(), sample_s, peak_gb


def apply_quant_patches(mode="tc"):
    import importlib
    bench_dir = str(Path(__file__).resolve().parent)
    if bench_dir not in sys.path:
        sys.path.insert(0, bench_dir)
    mod = importlib.import_module("krea2_convrot_nvfp4.comfy_quant_nvfp4")
    mod.apply()
    mod2 = importlib.import_module("int8.comfy_quant_int8")
    mod2.apply()


def latent_to_rgb_preview(samples, model):
    import comfy.model_patcher
    x = samples[0].detach().cpu()
    c = x.shape[0]
    factors = getattr(model.model, "latent_format", None)
    bias = getattr(model.model, "latent_format_bias", None)
    if factors is not None:
        f = torch.as_tensor(factors, dtype=torch.float32)
        if f.shape[0] != c:
            f = f[:c]
        rgb = torch.einsum("chw,cd->dhw", x[:f.shape[0]], f)
        if bias is not None:
            b = torch.as_tensor(bias, dtype=torch.float32).view(3, 1, 1)
            rgb = rgb + b
    else:
        rgb = x[:3]
    arr = rgb.permute(1, 2, 0).numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Krea2 ConvRot NVFP4 multi-seed benchmark (20 seeds)"
    )
    ap.add_argument("--fp16", required=True, help="BF16/FP16 Krea2 safetensors")
    ap.add_argument("--nvfp4", required=True, help="NVFP4+INT8 hybrid safetensors")
    ap.add_argument("--clip_path", required=True, help="Qwen3-VL-4B CLIP safetensors (Krea2)")
    ap.add_argument("--comfy_path", required=True, help="ComfyUI-master root")
    ap.add_argument("--token", default=None, help="HF token")
    ap.add_argument("--prompt", default="A beautiful cyberpunk city at night, high detail.")
    ap.add_argument("--negative", default="")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=20, help="Number of random seeds to test")
    ap.add_argument("--seed_base", type=int, default=42, help="RNG base for generating seed list")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--no-vae", action="store_true", help="Skip VAE decode (latent comparison only)")
    ap.add_argument("--output_dir", default=".")
    ap.add_argument("--mode", choices=["tc", "parity"], default="tc")
    return ap.parse_args()


def main():
    args = parse_args()
    from PIL import Image

    for p, name in ((args.fp16, "--fp16"), (args.nvfp4, "--nvfp4"), (args.clip_path, "--clip_path")):
        if not Path(p).is_file():
            raise FileNotFoundError(f"{name} not found: {p}")

    set_hf_token(args.token)
    os.makedirs(args.output_dir, exist_ok=True)

    bench_dir = Path(__file__).resolve().parent
    if str(bench_dir) not in sys.path:
        sys.path.insert(0, str(bench_dir))

    # Generate seed list
    rng = random.Random(args.seed_base)
    seed_list = [rng.randint(0, 2**31 - 1) for _ in range(args.seeds)]
    print(f"Seeds ({len(seed_list)}): {seed_list}")

    saved_argv = _clear_argv_for_comfy()
    try:
        setup_comfy(args.comfy_path)

        import comfy.model_management as mm
        import comfy.sd

        mm.get_torch_device()
        _report_gpu_memory("after_comfy_init")

        # CLIP on CPU
        _cpu = torch.device("cpu")
        print("Loading CLIP on CPU (Krea2 / Qwen3-VL-4B)...")
        clip = comfy.sd.load_clip(
            ckpt_paths=[args.clip_path],
            embedding_directory=None,
            clip_type=comfy.sd.CLIPType.KREA2,
            model_options={
                "load_device": _cpu,
                "offload_device": _cpu,
                "initial_device": _cpu,
            },
        )
        print("Encoding prompt (CPU TE)...")
        positive = encode_prompt(clip, args.prompt)
        negative = encode_prompt(clip, args.negative) if args.negative else encode_prompt(clip, "")
        if getattr(clip, "cond_stage_model", None) is not None:
            clip.cond_stage_model.cpu()
        if getattr(clip, "patcher", None) is not None:
            mm.unload_model_and_clones(clip.patcher)
        del clip
        _hard_free_vram()
        print("  [Offload] CLIP unloaded.")

        # --- FP16 branch ---
        print("\n" + "=" * 70)
        print("=== BRANCH 1: FP16/BF16 ===")
        print("=" * 70)
        model_fp16, diag_fp16 = load_model_for_branch(
            label="FP16", unet_path=args.fp16, args=args,
        )
        latent_fp16_shape = make_empty_latent(model_fp16, args.width, args.height)["samples"]
        print(f"  latent shape: {latent_fp16_shape.shape}")

        fp16_results = []
        for i, seed in enumerate(seed_list):
            lat, t, peak = sample_with_seed(
                model_fp16, positive, negative, latent_fp16_shape,
                seed=seed, steps=args.steps, cfg=args.cfg,
                sampler_name=args.sampler, scheduler=args.scheduler,
            )
            fp16_results.append(lat)
            if (i + 1) % 5 == 0 or (i + 1) == len(seed_list):
                print(f"  [FP16] {i+1}/{len(seed_list)} done")

        # Unload FP16
        import comfy.model_management as _mm
        if hasattr(model_fp16, 'model') and hasattr(model_fp16.model, 'to'):
            model_fp16.model.cpu()
        del model_fp16
        _hard_free_vram()

        # --- NVFP4 branch ---
        print("\n" + "=" * 70)
        print("=== BRANCH 2: NVFP4 + INT8 ===")
        print("=" * 70)
        print(f"Applying NVFP4 ConvRot mode='{args.mode}' + INT8 + addmm patches...")
        apply_quant_patches(mode=args.mode)

        model_nvfp4, diag_nvfp4 = load_model_for_branch(
            label="NVFP4", unet_path=args.nvfp4, args=args,
        )
        latent_nvfp4_shape = make_empty_latent(model_nvfp4, args.width, args.height)["samples"]
        print(f"  latent shape: {latent_nvfp4_shape.shape}")

        nvfp4_results = []
        for i, seed in enumerate(seed_list):
            lat, t, peak = sample_with_seed(
                model_nvfp4, positive, negative, latent_nvfp4_shape,
                seed=seed, steps=args.steps, cfg=args.cfg,
                sampler_name=args.sampler, scheduler=args.scheduler,
            )
            nvfp4_results.append(lat)
            if (i + 1) % 5 == 0 or (i + 1) == len(seed_list):
                print(f"  [NVFP4] {i+1}/{len(seed_list)} done")

        del model_nvfp4
        _hard_free_vram()

        # --- Metrics ---
        print("\n" + "=" * 70)
        print(f"Latent Cosine Comparison ({len(seed_list)} seeds)")
        print("=" * 70)

        cosines = []
        mses = []
        for i, (l16, lq) in enumerate(zip(fp16_results, nvfp4_results)):
            a = l16.reshape(-1)
            b = lq.reshape(-1)
            cos = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item())
            mse = float((a - b).pow(2).mean().item())
            cosines.append(cos)
            mses.append(mse)
            print(f"  seed {seed_list[i]:>12d}  cos={cos:.5f}  mse={mse:.4e}")

        import statistics
        cosines_arr = np.array(cosines)
        mses_arr = np.array(mses)
        print(f"\n--- Summary ---")
        print(f"  cosine: min={cosines_arr.min():.5f}  max={cosines_arr.max():.5f}  mean={cosines_arr.mean():.5f}  median={np.median(cosines_arr):.5f}  std={cosines_arr.std():.5f}")
        print(f"  mse   : min={mses_arr.min():.4e}  max={mses_arr.max():.4e}  mean={mses_arr.mean():.4e}")

        # Count by thresholds
        n_095 = int((cosines_arr >= 0.95).sum())
        n_098 = int((cosines_arr >= 0.98).sum())
        n_099 = int((cosines_arr >= 0.99).sum())
        print(f"  cos >= 0.95: {n_095}/{len(seed_list)}")
        print(f"  cos >= 0.98: {n_098}/{len(seed_list)}")
        print(f"  cos >= 0.99: {n_099}/{len(seed_list)}")

        # Sigma schedule check
        print(f"\n--- Sigma Schedule ---")
        for k in ("model_type", "ms_type", "sigma_min", "sigma_max"):
            v16 = diag_fp16.get(k)
            vq = diag_nvfp4.get(k)
            m = "OK" if v16 == vq else "MISMATCH"
            print(f"  {k}: FP16={v16}  NVFP4={vq}  [{m}]")

        return 0

    finally:
        _restore_argv(saved_argv)


if __name__ == "__main__":
    raise SystemExit(main())

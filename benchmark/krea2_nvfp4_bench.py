#!/usr/bin/env python3
"""
Krea2 NVFP4 ComfyUI Native Benchmark
====================================
Compare BF16/FP16 Krea2 DiT vs HSWQ NVFP4 (3-tier: NVFP4 + INT8 shelter + FP16).

Example (matches owner CLI):
  python krea2_nvfp4_bench.py \\
    --fp16  "D:\\...\\moodyKrea2Mix_v40BF16.safetensors" \\
    --nvfp4 "D:\\...\\moodyKrea2Mix_v40_native_nvfp4.safetensors" \\
    --clip_path "D:\\...\\Qwen3_VL_4B_Thinking_abliterated.safetensors" \\
    --comfy_path "D:\\USERFILES\\GitHub\\hswq\\ComfyUI-master" \\
    --token "hf_..." \\
    --prompt "A beautiful cyberpunk city at night, high detail." \\
    --steps 25

Optional: --vae PATH for pixel-space decode (same metrics path as nvfp4bench_sdxl).
Without --vae: Wan21 latent-RGB preview image, still MSE/SSIM like SDXL printout.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops
from skimage.metrics import structural_similarity as ssim


def _clear_argv_for_comfy() -> list[str]:
    """ComfyUI cli_args swallows unknown flags; keep only argv[0] during import."""
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    return saved


def _restore_argv(saved: list[str]) -> None:
    sys.argv = saved


def setup_comfy(comfy_path: str) -> None:
    comfy_root = Path(comfy_path).resolve()
    if not comfy_root.is_dir():
        raise FileNotFoundError(f"--comfy_path not found: {comfy_root}")
    # Prefer this tree for comfy.* imports
    sys.path = [str(comfy_root)] + [p for p in sys.path if Path(p).resolve() != comfy_root]

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


def apply_nvfp4_patches() -> None:
    """Match nvfp4bench_sdxl: HSWQ detect/load, then ComfyUI-only forward.

    Without apply_nvfp4_comfy_parity(), native Kitchen weights run TC
    scaled_mm_nvfp4 + ConvRot act and Pixel SSIM collapses (~0.65).
    Parity restores stock MixedPrecision.forward (same as SDXL NVFP4 bench).
    """
    import comfy.ops

    from nvfp4.comfy_quant_nvfp4 import apply_comfy_quant_nvfp4_patches
    from nvfp4_comfy_parity import apply_nvfp4_comfy_parity

    apply_comfy_quant_nvfp4_patches()
    if not apply_nvfp4_comfy_parity():
        raise RuntimeError(
            "nvfp4 ComfyUI-only parity failed to apply "
            "(need [BENCH] nvfp4 ComfyUI-only log; TC forward must be off)"
        )

    # Hard gate: refuse to run if Linear.forward is still HSWQ TC-wrapped.
    lin_fwd = comfy.ops.mixed_precision_ops().Linear.forward
    if getattr(lin_fwd, "_hswq_nvfp4_full_forward", False):
        raise RuntimeError(
            "nvfp4 ComfyUI-only parity incomplete: "
            "Linear.forward still has _hswq_nvfp4_full_forward "
            "(TC path would destroy Pixel SSIM)"
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


def latent_to_rgb_preview(latent_t: torch.Tensor, model) -> Image.Image:
    """Wan21 latent_rgb_factors preview when no VAE is provided."""
    fmt = getattr(getattr(model, "model", model), "latent_format", None)
    factors = getattr(fmt, "latent_rgb_factors", None) if fmt is not None else None
    bias = getattr(fmt, "latent_rgb_factors_bias", None) if fmt is not None else None

    x = latent_t.detach().float().cpu()
    # [B,C,H,W] or [B,C,T,H,W] — take first batch / first frame
    if x.ndim == 5:
        x = x[0, :, 0]
    elif x.ndim == 4:
        x = x[0]
    else:
        raise ValueError(f"unexpected latent shape {tuple(x.shape)}")

    c, h, w = x.shape
    if factors is not None:
        f = torch.as_tensor(factors, dtype=torch.float32)  # [C, 3]
        if f.shape[0] != c:
            f = f[:c]
        rgb = torch.einsum("chw,cd->dhw", x[: f.shape[0]], f)
        if bias is not None:
            b = torch.as_tensor(bias, dtype=torch.float32).view(3, 1, 1)
            rgb = rgb + b
    else:
        # Fallback: first 3 channels
        rgb = x[:3]

    arr = rgb.permute(1, 2, 0).numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def sample_once(
    model,
    positive,
    negative,
    latent: dict,
    *,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    denoise: float,
):
    import comfy.sample as comfy_sample
    import comfy.samplers
    import comfy.utils
    import latent_preview

    noise = comfy_sample.prepare_noise(latent["samples"], seed, None)
    noise_mask = latent.get("noise_mask", None)
    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

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
        denoise=denoise,
        disable_noise=False,
        start_step=None,
        last_step=None,
        force_full_denoise=False,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    out = latent.copy()
    out["samples"] = samples
    return out


def _hard_free_vram() -> None:
    """Drop loaded models and return VRAM to the pool (CPU-offload path)."""
    import comfy.model_management as mm

    mm.unload_all_models()
    mm.soft_empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_branch(
    *,
    label: str,
    unet_path: str,
    vae,
    positive,
    negative,
    args,
) -> tuple[Image.Image, torch.Tensor, float, float]:
    import comfy.sd

    print(f"\n=== {label}: loading UNet ===")
    print(f"  path: {unet_path}")
    t0 = time.perf_counter()
    # Do NOT load_models_gpu([unet, clip]) — master CPU-offloads via sample().
    model = comfy.sd.load_diffusion_model(unet_path, {})
    load_s = time.perf_counter() - t0
    print(f"  load: {load_s:.2f}s")

    latent = make_empty_latent(model, args.width, args.height, batch=1)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t1 = time.perf_counter()
    out = sample_once(
        model,
        positive,
        negative,
        latent,
        seed=args.seed,
        steps=args.steps,
        cfg=args.cfg,
        sampler_name=args.sampler,
        scheduler=args.scheduler,
        denoise=1.0,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sample_s = time.perf_counter() - t1
    peak_gb = (
        torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )
    print(f"  sample: {sample_s:.2f}s  peak_vram={peak_gb:.2f} GiB")

    samples_t = out["samples"]
    if vae is not None:
        print("  decoding with VAE...")
        # Exact ComfyUI nodes.VAEDecode.decode body
        latent = samples_t
        if latent.is_nested:
            latent = latent.unbind()[0]
        images = vae.decode(latent)
        if len(images.shape) == 5:  # Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        img_array = 255.0 * images[0].detach().cpu().numpy()
        img = Image.fromarray(np.clip(img_array, 0, 255).astype("uint8"))
    else:
        img = latent_to_rgb_preview(samples_t, model)

    lat_cpu = samples_t.detach().float().cpu()
    del model, out, samples_t, latent
    _hard_free_vram()
    return img, lat_cpu, sample_s, peak_gb


def calculate_metrics(img1, img2):
    """Same as nvfp4bench_sdxl.calculate_metrics (pixel RGB, 見た目)."""
    arr1 = np.array(img1)
    arr2 = np.array(img2)

    # MSE (mean squared error)
    mse = np.mean((arr1 - arr2) ** 2)

    # SSIM (structural similarity)
    score_ssim = ssim(arr1, arr2, win_size=3, channel_axis=2, data_range=255)

    return mse, score_ssim


def main() -> int:
    parser = argparse.ArgumentParser(description="Krea2 NVFP4 ComfyUI Native Benchmark")
    parser.add_argument("--fp16", required=True, help="BF16/FP16 Krea2 DiT safetensors")
    parser.add_argument("--nvfp4", required=True, help="HSWQ NVFP4 Krea2 DiT safetensors")
    parser.add_argument("--clip_path", required=True, help="Qwen3-VL-4B CLIP safetensors (Krea2)")
    parser.add_argument("--comfy_path", required=True, help="ComfyUI-master root")
    parser.add_argument("--token", default=None, help="HF token (env HF_TOKEN / HUGGING_FACE_HUB_TOKEN)")
    parser.add_argument("--prompt", default="A beautiful cyberpunk city at night, high detail.", help="Benchmark prompt")
    parser.add_argument("--negative", default="", help="Negative prompt (often unused at cfg=1)")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--cfg", type=float, default=1.0, help="Krea2 default 1.0 (no CFG)")
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="simple")
    parser.add_argument("--vae", default=None, help="Optional VAE safetensors for pixel decode")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--save_image", action="store_true")
    args = parser.parse_args()

    for p, name in ((args.fp16, "--fp16"), (args.nvfp4, "--nvfp4"), (args.clip_path, "--clip_path")):
        if not Path(p).is_file():
            raise FileNotFoundError(f"{name} not found: {p}")

    set_hf_token(args.token)
    os.makedirs(args.output_dir, exist_ok=True)

    # Ensure benchmark/ is on path for nvfp4 package
    bench_dir = Path(__file__).resolve().parent
    if str(bench_dir) not in sys.path:
        sys.path.insert(0, str(bench_dir))

    saved_argv = _clear_argv_for_comfy()
    try:
        setup_comfy(args.comfy_path)
        apply_nvfp4_patches()

        import folder_paths  # noqa: F401
        import comfy.model_management as mm
        import comfy.sd

        mm.get_torch_device()

        print("Loading CLIP (Krea2 / Qwen3-VL-4B)...")
        clip = comfy.sd.load_clip(
            ckpt_paths=[args.clip_path],
            embedding_directory=None,
            clip_type=comfy.sd.CLIPType.KREA2,
        )

        vae = None
        if args.vae:
            if not Path(args.vae).is_file():
                raise FileNotFoundError(f"--vae not found: {args.vae}")
            print(f"Loading VAE: {args.vae}")
            sd = comfy.utils.load_torch_file(args.vae)
            vae = comfy.sd.VAE(sd=sd)
        else:
            print("No --vae: using Wan21 RGB preview for metrics (same printout as SDXL)")

        print("Encoding prompt...")
        positive = encode_prompt(clip, args.prompt)
        negative = encode_prompt(clip, args.negative) if args.negative else encode_prompt(clip, "")
        # ZIT same shape: encode on GPU, then offload TE before DiT sample (no CLIP+UNet co-load).
        if getattr(clip, "cond_stage_model", None) is not None:
            clip.cond_stage_model.cpu()
        if getattr(clip, "patcher", None) is not None:
            mm.unload_model_and_clones(clip.patcher)
        del clip
        _hard_free_vram()
        print("  [Offload] CLIP on CPU / unloaded (VRAM freed for Krea2 DiT benchmark).")

        print("--- Benchmark Config ---")
        print(f"Seed: {args.seed}  Steps: {args.steps}  CFG: {args.cfg}")
        print(f"Size: {args.width}x{args.height}  sampler={args.sampler}/{args.scheduler}")
        print(f"Prompt: {args.prompt[:80]}...")
        print("------------------------")

        img_fp16, _lat_fp16, t16, v16 = run_branch(
            label="1. Baseline (FP16/BF16)",
            unet_path=args.fp16,
            vae=vae,
            positive=positive,
            negative=negative,
            args=args,
        )
        p16 = os.path.join(args.output_dir, "bench_result_fp16.png")
        img_fp16.save(p16)
        print(f"FP16 Time: {t16:.2f}s  peak={v16:.2f}GiB")

        img_nv, _lat_nv, t4, v4 = run_branch(
            label="2. Quantized (NVFP4)",
            unet_path=args.nvfp4,
            vae=vae,
            positive=positive,
            negative=negative,
            args=args,
        )
        # Same filename pattern as nvfp4bench_sdxl (quantized side = bench_result_fp8.png)
        p8 = os.path.join(args.output_dir, "bench_result_fp8.png")
        img_nv.save(p8)
        print(f"NVFP4 Time: {t4:.2f}s  peak={v4:.2f}GiB")

        # 3. Comparison — same printout as nvfp4bench_sdxl
        print("\n=== 3. Calculating Metrics ===")

        if img_fp16.size != img_nv.size:
            print(f"Error: Image sizes do not match! FP16:{img_fp16.size}, NVFP4:{img_nv.size}")
            print("Different models or settings used.")
            return 1

        mse, score = calculate_metrics(img_fp16, img_nv)

        print(f"--------------------------------------------------")
        print(f"MSE (Error): {mse:.4f} \t(0 is perfect match)")
        print(f"SSIM (Sim) : {score:.4f} \t(1.0 is perfect match)")
        print(f"--------------------------------------------------")

        if score > 0.98:
            grade = "PERFECT (S)"
        elif score > 0.95:
            grade = "EXCELLENT (A)"
        elif score > 0.90:
            grade = "GOOD (B)"
        else:
            grade = "WARNING (C)"

        print(f"Quality Grade: {grade}")

        diff_img = ImageChops.difference(img_fp16, img_nv)
        diff_img = ImageChops.multiply(diff_img, Image.new("RGB", diff_img.size, (10, 10, 10)))
        diff_path = os.path.join(args.output_dir, "bench_result_diff.png")
        diff_img.save(diff_path)
        print(f"Diff image saved: {diff_path}")
        return 0
    finally:
        _restore_argv(saved_argv)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Qwen Image (Edit) INT8 ConvRot Fidelity & VRAM Benchmark.

Uses ComfyUI-master load_diffusion_model, load_clip, and comfy.sample.sample.
ComfyUI-master already has full ConvRot INT8 support — no external patches needed.

Usage:
  python qi_int8_bench.py \
    --fp16 "path/to/baseline.safetensors" \
    --int8 "path/to/convrot_int8.safetensors" \
    --clip_path "path/to/clip.safetensors" \
    --comfy_path "path/to/ComfyUI-master" \
    --vae "path/to/vae.safetensors" \
    --token "hf_xxx" \
    --prompt "masterpiece, best quality, 1girl" \
    --steps 25
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
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    return saved


def _restore_argv(saved: list[str]) -> None:
    sys.argv = saved


def _install_torchaudio_stub() -> None:
    import importlib.machinery
    import types

    for key in list(sys.modules):
        if key == "torchaudio" or key.startswith("torchaudio."):
            del sys.modules[key]

    def _stub_mod(name: str, *, is_package: bool = False):
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
    sys.path = [str(comfy_root)] + [p for p in sys.path if Path(p).resolve() != comfy_root]

    _install_torchaudio_stub()

    import comfy.options

    comfy.options.enable_args_parsing(False)

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


def apply_int8_patches() -> None:
    """Apply ComfyUI-Master's built-in INT8 ConvRot patches."""
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


def set_hf_token(token: str | None) -> None:
    if not token:
        return
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def encode_prompt(clip, prompt: str):
    tokens = clip.tokenize(prompt)
    return clip.encode_from_tokens_scheduled(tokens)


def make_empty_latent(model, width: int, height: int, batch: int = 1) -> dict:
    import comfy.model_management as mm
    import comfy.sample as comfy_sample

    device = mm.intermediate_device()
    latent = torch.zeros([batch, 16, height // 8, width // 8], device=device)
    latent = comfy_sample.fix_empty_latent_channels(model, latent)
    return {"samples": latent}


def latent_to_img(l: torch.Tensor) -> Image.Image:
    l = l[0].permute(1, 2, 0).cpu().float().numpy()
    l = (l - l.min()) / (l.max() - l.min() + 1e-6) * 255
    return Image.fromarray(l[:, :, :3].astype(np.uint8))


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
    import comfy.model_management as mm

    mm.unload_all_models()
    mm.soft_empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _load_diffusion_model(unet_path: str):
    """Load diffusion model via ComfyUI-Master.

    ComfyUI-Master already has full INT8 ConvRot support built-in,
    so we use its standard int8 scope for quantized models.
    """
    import comfy.sd
    from int8.comfy_quant_int8 import (
        _int8_quant_conv_scope,
        checkpoint_looks_like_comfy_quant_int8,
    )

    use_int8_scope = checkpoint_looks_like_comfy_quant_int8(unet_path)
    print(f"  [BENCH] INT8 Conv2d load scope: {use_int8_scope}")
    if use_int8_scope:
        with _int8_quant_conv_scope():
            return comfy.sd.load_diffusion_model(unet_path, {})
    return comfy.sd.load_diffusion_model(unet_path, {})


def print_model_stats(model, name: str) -> None:
    inner = getattr(model, "model", model)
    dm = getattr(inner, "diffusion_model", None)
    target = dm if dm is not None else inner
    params = list(target.parameters())
    dtype = params[0].dtype if params else "?"
    n = sum(p.numel() for p in params)
    print(
        f"[{name}] class={type(inner).__name__} "
        f"unet={type(target).__name__} dtype={dtype} params={n}"
    )


def calculate_latent_mse(lat1: torch.Tensor, lat2: torch.Tensor) -> float:
    a = lat1.detach().float().cpu()
    b = lat2.detach().float().cpu()
    return float(torch.mean((a - b) ** 2).item())


def calculate_latent_cosine(lat1: torch.Tensor, lat2: torch.Tensor) -> float:
    a = lat1.detach().float().cpu().reshape(-1)
    b = lat2.detach().float().cpu().reshape(-1)
    return float(
        torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0), dim=1
        ).item()
    )


def calculate_ssim_normalized(img1: Image.Image, img2: Image.Image) -> float:
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    return float(ssim(arr1, arr2, win_size=3, channel_axis=2, data_range=255))


def run_branch(
    *,
    label: str,
    unet_path: str,
    vae,
    positive,
    negative,
    args,
) -> tuple[Image.Image, torch.Tensor, float, float]:
    print(f"\n=== {label} ===")
    print(f"  path: {unet_path}")
    t0 = time.perf_counter()
    model = _load_diffusion_model(unet_path)
    load_s = time.perf_counter() - t0
    print(f"  load: {load_s:.2f}s")
    print_model_stats(model, label)

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
    peak_mb = (
        torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    )
    print(f"  sample: {sample_s:.2f}s  Peak VRAM: {peak_mb:.2f} MB")

    samples_t = out["samples"]
    lat_cpu = samples_t.detach().float().cpu()

    if vae is not None:
        latent_t = samples_t.detach()
        if getattr(latent_t, "is_nested", False):
            latent_t = latent_t.unbind()[0]
        del model, out, samples_t
        _hard_free_vram()

        print("  decoding with VAE...")
        _po = vae.process_output
        vae.process_output = lambda image: image.float().add(1.0).mul(0.5).clamp(0.0, 1.0)
        try:
            with torch.inference_mode(False):
                images = vae.decode(latent_t)
        finally:
            vae.process_output = _po
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        img_array = 255.0 * images[0].detach().cpu().numpy()
        img = Image.fromarray(np.clip(img_array, 0, 255).astype("uint8"))
        del latent_t, images
    else:
        img = latent_to_img(samples_t.detach())
        del model, out, samples_t

    _hard_free_vram()
    return img, lat_cpu, sample_s, peak_mb


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qwen Image (Edit) INT8 ConvRot Fidelity & VRAM Benchmark"
    )
    parser.add_argument("--fp16", required=True, help="Baseline (BF16/FP16) model path")
    parser.add_argument("--int8", required=True, help="ConvRot INT8 quantized model path")
    parser.add_argument(
        "--clip_path",
        required=True,
        help="CLIP / text encoder path (e.g. Qwen2.5-VL-7B)",
    )
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        help=(
            "Optional tokenizer directory override; default is "
            "comfy/text_encoders/qwen25_tokenizer under --comfy_path"
        ),
    )
    parser.add_argument(
        "--comfy_path",
        required=True,
        help="ComfyUI-master root path (must have INT8 ConvRot support built-in)",
    )
    parser.add_argument(
        "--vae",
        default=None,
        required=False,
        help="Optional VAE path. If omitted, fidelity uses latent cosine (no decode).",
    )
    parser.add_argument("--token", default=None, help="Optional Hugging Face token")
    parser.add_argument(
        "--prompt",
        default=(
            "Solid black background only. Empty frame. No objects, no lights, no city, no people, no text, no texture. Completely black."
        ),
        help="Benchmark prompt",
    )
    parser.add_argument(
        "--negative-prompt",
        default="",
        help="Negative prompt for CFG (empty string default).",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=2.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="simple")
    args = parser.parse_args()

    for p, name in ((args.fp16, "--fp16"), (args.int8, "--int8"), (args.clip_path, "--clip_path")):
        if not Path(p).is_file():
            raise FileNotFoundError(f"{name} not found: {p}")

    set_hf_token(args.token)

    bench_dir = Path(__file__).resolve().parent
    if str(bench_dir) not in sys.path:
        sys.path.insert(0, str(bench_dir))

    saved_argv = _clear_argv_for_comfy()
    try:
        setup_comfy(args.comfy_path)
        apply_int8_patches()

        import folder_paths  # noqa: F401
        import comfy.model_management as mm
        import comfy.sd
        import comfy.utils

        mm.get_torch_device()

        print("Starting Qwen Image INT8 ConvRot Bench (ComfyUI-Master pipeline)...")
        print(f"Loading CLIP weights from: {args.clip_path}")
        clip = comfy.sd.load_clip(
            ckpt_paths=[args.clip_path],
            embedding_directory=None,
        )

        vae = None
        if args.vae:
            if not Path(args.vae).is_file():
                raise FileNotFoundError(f"--vae not found: {args.vae}")
            print(f"  Loaded VAE for decode: {os.path.basename(args.vae)}")
            sd = comfy.utils.load_torch_file(args.vae)
            vae = comfy.sd.VAE(sd=sd)

        print("Encoding prompt...")
        positive = encode_prompt(clip, args.prompt)
        negative = encode_prompt(clip, args.negative_prompt)
        if getattr(clip, "cond_stage_model", None) is not None:
            clip.cond_stage_model.cpu()
        if getattr(clip, "patcher", None) is not None:
            mm.unload_model_and_clones(clip.patcher)
        del clip
        _hard_free_vram()
        print("  [Offload] CLIP on CPU / unloaded (VRAM freed for INT8 benchmark).")

        print("--- Benchmark Config ---")
        print(f"Seed: {args.seed}  Steps: {args.steps}  CFG: {args.cfg}")
        print(f"Size: {args.width}x{args.height}  sampler={args.sampler}/{args.scheduler}")
        print(f"Prompt: {args.prompt[:80]}...")
        print("------------------------")

        img_fp16, lat_fp16, t16, v16 = run_branch(
            label="1. Benchmarking Baseline (BF16/FP16)",
            unet_path=args.fp16,
            vae=vae,
            positive=positive,
            negative=negative,
            args=args,
        )
        img_fp16.save("bench_fp16.png")
        print(f"FP16 Time: {t16:.2f}s | Peak VRAM: {v16:.2f} MB")

        img_int8, lat_int8, t8, v8 = run_branch(
            label="2. Benchmarking Quantized (INT8 ConvRot)",
            unet_path=args.int8,
            vae=vae,
            positive=positive,
            negative=negative,
            args=args,
        )
        img_int8.save("bench_int8.png")
        print(f"INT8 Time: {t8:.2f}s | Peak VRAM: {v8:.2f} MB")

        if img_fp16.size != img_int8.size:
            print(f"Error: Image sizes do not match! FP16:{img_fp16.size}, INT8:{img_int8.size}")
            return 1

        mse = calculate_latent_mse(lat_fp16, lat_int8)
        lat_cos = calculate_latent_cosine(lat_fp16, lat_int8)

        print("\n" + "=" * 50)
        print("QI INT8 CONVROT BENCHMARK RESULTS")
        print("=" * 50)
        vram_saved = v16 - v8
        vram_saved_pct = (vram_saved / v16) * 100 if v16 else 0.0

        print(f"Peak VRAM Expansion:  FP16: {v16:>8.1f} MB")
        print(f"                      INT8: {v8:>8.1f} MB")
        print(f"VRAM Saved:           {vram_saved:8.1f} MB ({vram_saved_pct:.1f}%)")
        print("-" * 50)
        print(f"Inference Time:       FP16: {t16:>8.2f}s")
        print(f"                      INT8: {t8:>8.2f}s")
        print("-" * 50)
        print("Fidelity:")
        print(f"  {'MSE (latent)':<18}: {mse:.4f}")
        print(f"  {'Cosine (latent)':<18}: {lat_cos:.4f}")
        if vae is not None:
            score = calculate_ssim_normalized(img_fp16, img_int8)
            print(f"  {'SSIM (decoded)':<18}: {score:.4f}")
        print("=" * 50)

        diff_img = ImageChops.difference(img_fp16, img_int8)
        diff_img = ImageChops.multiply(diff_img, Image.new("RGB", diff_img.size, (10, 10, 10)))
        diff_img.save("bench_diff.png")
        print("Diff image saved: bench_diff.png")
        return 0
    finally:
        _restore_argv(saved_argv)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Krea2 Pure ComfyUI-Master NVFP4 Benchmark
==========================================
Compare BF16/FP16 Krea2 DiT vs ComfyUI-Master stock NVFP4 (no HSWQ patches).

No krea2_nvfp4, no int8, no ConvRot parity, no addmm patch, no HSWQ TC forward.
Pure ComfyUI-Master stock load + stock MixedPrecision Linear.forward only.

Example:
  python krea2_native_nvfp4_bench.py \\
    --fp16  "D:\\...\\moodyKrea2Mix_v40BF16.safetensors" \\
    --nvfp4 "D:\\...\\moodyKrea2Mix_v40_nvfp4.safetensors" \\
    --clip_path "D:\\...\\Qwen3_VL_4B_Thinking_abliterated.safetensors" \\
    --comfy_path "D:\\USERFILES\\GitHub\\hswq\\ComfyUI-master" \\
    --token "***" \\
    --prompt "A beautiful cyberpunk city at night, high detail." \\
    --steps 25

Optional: --vae PATH for pixel-space decode.
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

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _clear_argv_for_comfy() -> list[str]:
    """ComfyUI cli_args swallows unknown flags; keep only argv[0] during import."""
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    return saved


def _restore_argv(saved: list[str]) -> None:
    sys.argv = saved


def _install_torchaudio_stub() -> None:
    """Prevent real torchaudio from loading if comfy.sd is pulled in."""
    import importlib.machinery
    import types

    for key in list(sys.modules):
        if key == "torchaudio" or key.startswith("torchaudio."):
            del sys.modules[key]

    def _stub_mod(name: str, *, is_package: bool = False):
        mod = types.ModuleType(name)
        mod.__file__ = "<torchaudio_stub>"
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


SSIM_TARGET = 0.9


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


def _diag_model_sampling(model, label: str) -> dict:
    info = {}
    try:
        ms = model.get_model_object("model_sampling")
        info["ms_type"] = type(ms).__name__
        info["sigma_min"] = float(ms.sigma_min)
        info["sigma_max"] = float(ms.sigma_max)
        inner = getattr(model, "model", None)
        info["model_type"] = type(inner).__name__ if inner else "unknown"
        print(
            f"  [{label}] model_type={info['model_type']} "
            f"model_sampling={info['ms_type']} "
            f"sigma_min={info['sigma_min']:.6f} sigma_max={info['sigma_max']:.6f}"
        )
    except Exception as e:
        print(f"  [{label}] model_sampling diagnostic failed: {e}")
        info["error"] = str(e)
    return info


def latent_to_rgb_preview(latent_t: torch.Tensor, model) -> Image.Image:
    fmt = getattr(getattr(model, "model", model), "latent_format", None)
    factors = getattr(fmt, "latent_rgb_factors", None) if fmt is not None else None
    bias = getattr(fmt, "latent_rgb_factors_bias", None) if fmt is not None else None

    x = latent_t.detach().float().cpu()
    if x.ndim == 5:
        x = x[0, :, 0]
    elif x.ndim == 4:
        x = x[0]
    else:
        raise ValueError(f"unexpected latent shape {tuple(x.shape)}")

    c, h, w = x.shape
    if factors is not None:
        f = torch.as_tensor(factors, dtype=torch.float32)
        if f.shape[0] != c:
            f = f[:c]
        rgb = torch.einsum("chw,cd->dhw", x[: f.shape[0]], f)
        if bias is not None:
            b = torch.as_tensor(bias, dtype=torch.float32).view(3, 1, 1)
            rgb = rgb + b
    else:
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
        force_full_denoise=True,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    out = latent.copy()
    out["samples"] = samples
    return out


def _nvidia_smi_used_mib() -> int | None:
    try:
        import subprocess

        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        return None
    pid = os.getpid()
    total = 0
    found = False
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            if int(parts[0]) != pid:
                continue
            total += int(parts[1].split()[0])
            found = True
        except ValueError:
            continue
    return total if found else 0


def _report_gpu_memory(tag: str) -> None:
    smi = _nvidia_smi_used_mib()
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        smi_s = f"{smi} MiB" if smi is not None else "n/a"
        print(
            f"  [GPU] {tag}: nvidia-smi used={smi_s}  "
            f"torch alloc={alloc:.0f} MiB reserved={reserved:.0f} MiB"
        )
    else:
        print(f"  [GPU] {tag}: CUDA unavailable")


def _hard_free_vram() -> None:
    import comfy.model_management as mm

    mm.unload_all_models()
    mm.soft_empty_cache(force=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def _load_diffusion_model(unet_path: str):
    """Load DiT via ComfyUI-Master stock loader (no HSWQ patches)."""
    import comfy.sd

    print(f"  [BENCH] Loading via comfy.sd.load_diffusion_model (stock, no patches)")
    return comfy.sd.load_diffusion_model(unet_path, {})


def _load_vae(vae_path: str):
    import comfy.sd
    import comfy.utils

    print(f"  Loading VAE (deferred until decode): {vae_path}")
    sd = comfy.utils.load_torch_file(vae_path)
    return comfy.sd.VAE(sd=sd)


def _diag_stock_modules(model, label: str) -> None:
    """Print stock module diagnostics (no HSWQ flags)."""
    n_total_linear = 0
    n_quantized = 0
    for name, m in model.model.diffusion_model.named_modules():
        if isinstance(m, torch.nn.Linear) or hasattr(m, "out_features"):
            n_total_linear += 1
            # Stock ComfyUI quantized Linear has layout_type attribute
            if getattr(m, "layout_type", None) is not None:
                n_quantized += 1
    print(
        f"  [{label} module diag] total_linears={n_total_linear} "
        f"stock_quantized={n_quantized}"
    )


def run_branch(
    *,
    label: str,
    unet_path: str,
    vae_path: str | None,
    positive,
    negative,
    args,
) -> tuple[Image.Image, torch.Tensor, float, float, dict]:
    print(f"\n=== {label}: loading UNet ===")
    print(f"  path: {unet_path}")
    _report_gpu_memory(f"before_load/{label}")
    t0 = time.perf_counter()
    model = _load_diffusion_model(unet_path)
    load_s = time.perf_counter() - t0
    print(f"  load: {load_s:.2f}s")

    latent = make_empty_latent(model, args.width, args.height, batch=1)
    diag = _diag_model_sampling(model, label)
    print(f"  latent shape: {latent['samples'].shape}")

    _diag_stock_modules(model, label)

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
    lat_cpu = samples_t.detach().float().cpu()

    if vae_path is not None:
        latent = samples_t.detach()
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[0]
        del model, out, samples_t
        _hard_free_vram()
        _report_gpu_memory(f"pre_vae_decode/{label}")

        vae = _load_vae(vae_path)
        print("  decoding with VAE...")
        _po = vae.process_output
        vae.process_output = lambda image: image.float().add(1.0).mul(0.5).clamp(0.0, 1.0)
        try:
            with torch.inference_mode(False):
                images = vae.decode(latent)
        finally:
            vae.process_output = _po
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        img_array = 255.0 * images[0].detach().cpu().numpy()
        img = Image.fromarray(np.clip(img_array, 0, 255).astype("uint8"))
        del latent, images, vae
    else:
        img = latent_to_rgb_preview(samples_t, model)
        del model, out, samples_t

    _hard_free_vram()
    return img, lat_cpu, sample_s, peak_gb, diag


def calculate_metrics(img1, img2):
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    mse = np.mean((arr1 - arr2) ** 2)
    score_ssim = ssim(arr1, arr2, win_size=3, channel_axis=2, data_range=255)
    return mse, score_ssim


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Krea2 Pure ComfyUI-Master NVFP4 Benchmark "
            "(no HSWQ patches, stock load + stock forward)"
        )
    )
    parser.add_argument("--fp16", required=True, help="BF16/FP16 Krea2 DiT safetensors")
    parser.add_argument(
        "--nvfp4",
        required=True,
        help="NVFP4 Krea2 DiT safetensors (ComfyUI-Master stock pack)",
    )
    parser.add_argument("--clip_path", required=True, help="Qwen3-VL-4B CLIP safetensors (Krea2)")
    parser.add_argument("--comfy_path", required=True, help="ComfyUI-master root")
    parser.add_argument("--token", default=None, help="HF token")
    parser.add_argument("--prompt", default="A beautiful cyberpunk city at night, high detail.")
    parser.add_argument("--negative", default="")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="simple")
    parser.add_argument("--vae", default=None, help="Optional VAE safetensors for pixel decode")
    parser.add_argument("--output_dir", default=".")
    args = parser.parse_args()

    for p, name in ((args.fp16, "--fp16"), (args.nvfp4, "--nvfp4"), (args.clip_path, "--clip_path")):
        if not Path(p).is_file():
            raise FileNotFoundError(f"{name} not found: {p}")

    set_hf_token(args.token)
    os.makedirs(args.output_dir, exist_ok=True)

    saved_argv = _clear_argv_for_comfy()
    try:
        setup_comfy(args.comfy_path)

        import folder_paths  # noqa: F401
        import comfy.model_management as mm
        import comfy.sd

        mm.get_torch_device()
        _report_gpu_memory("after_comfy_init")

        vae_path = None
        if args.vae:
            if not Path(args.vae).is_file():
                raise FileNotFoundError(f"--vae not found: {args.vae}")
            vae_path = args.vae
            print(f"VAE path (load deferred until decode): {vae_path}")
        else:
            print("No --vae: using Wan21 RGB preview for metrics")

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
        _report_gpu_memory("after_clip_load_cpu")

        print("Encoding prompt (CPU TE)...")
        positive = encode_prompt(clip, args.prompt)
        negative = encode_prompt(clip, args.negative) if args.negative else encode_prompt(clip, "")
        if getattr(clip, "cond_stage_model", None) is not None:
            clip.cond_stage_model.cpu()
        if getattr(clip, "patcher", None) is not None:
            mm.unload_model_and_clones(clip.patcher)
        del clip
        _hard_free_vram()
        print("  [Offload] CLIP unloaded (never GPU-resident for this bench).")
        _report_gpu_memory("after_clip_offload")

        print("--- Benchmark Config ---")
        print(f"Seed: {args.seed}  Steps: {args.steps}  CFG: {args.cfg}")
        print(f"Size: {args.width}x{args.height}  sampler={args.sampler}/{args.scheduler}")
        print(f"Prompt: {args.prompt[:80]}...")
        print(f"Mode: Pure ComfyUI-Master stock NVFP4 (no HSWQ patches)")
        print("------------------------")

        # --- Branch 1: FP16/BF16 baseline ---
        img_fp16, _lat_fp16, t16, v16, diag_fp16 = run_branch(
            label="1. Baseline (FP16/BF16)",
            unet_path=args.fp16,
            vae_path=vae_path,
            positive=positive,
            negative=negative,
            args=args,
        )
        p16 = os.path.join(args.output_dir, "bench_result_fp16.png")
        img_fp16.save(p16)
        print(f"FP16 Time: {t16:.2f}s  peak={v16:.2f}GiB")

        # --- Branch 2: NVFP4 (stock ComfyUI, no patches) ---
        # No apply_quant_patches() — pure stock ComfyUI load + forward.
        # ComfyUI-Master detects comfy_quant nvfp4 and loads via
        # _load_quantized_module → MixedPrecision Linear.forward (stock).

        img_q, _lat_q, tq, vq, diag_q = run_branch(
            label="2. NVFP4 (ComfyUI-Master stock, no patches)",
            unet_path=args.nvfp4,
            vae_path=vae_path,
            positive=positive,
            negative=negative,
            args=args,
        )
        pq = os.path.join(args.output_dir, "bench_result_nvfp4.png")
        img_q.save(pq)
        print(f"NVFP4 Time: {tq:.2f}s  peak={vq:.2f}GiB")

        # --- Sigma schedule comparison ---
        print("\n--- Sigma Schedule Diagnostics ---")
        _sigma_mismatch = False
        for _dk in ("model_type", "ms_type", "sigma_min", "sigma_max"):
            _v16 = diag_fp16.get(_dk)
            _vq = diag_q.get(_dk)
            if isinstance(_v16, float) and isinstance(_vq, float):
                _dm = "OK" if abs(_v16 - _vq) < 1e-6 else "MISMATCH"
            else:
                _dm = "OK" if _v16 == _vq else "MISMATCH"
            if _dm != "OK":
                _sigma_mismatch = True
            print(f"  {_dk}: FP16={_v16}  NVFP4={_vq}  [{_dm}]")
        if _sigma_mismatch:
            print(
                "  *** SIGMA SCHEDULE MISMATCH DETECTED ***\n"
                "  BF16 and NVFP4 use different model configs / noise schedules.\n"
                "  Images WILL be completely different regardless of quant quality."
            )
        else:
            print("  All sigma schedule parameters match.")

        # --- Metrics ---
        print("\n=== 3. Calculating Metrics ===")

        if _lat_fp16.shape != _lat_q.shape:
            print(
                f"\n  *** LATENT SHAPE MISMATCH: "
                f"FP16={tuple(_lat_fp16.shape)} NVFP4={tuple(_lat_q.shape)} ***\n"
                f"  Different latent shapes = model configs differ = metrics invalid."
            )
        lat_fp16 = _lat_fp16.reshape(-1)
        lat_q = _lat_q.reshape(-1)
        lat_mse = float((lat_fp16 - lat_q).pow(2).mean().item())
        lat_cos = float(torch.nn.functional.cosine_similarity(
            lat_fp16.unsqueeze(0), lat_q.unsqueeze(0), dim=1).item())
        print(f"\n--- Latent-Space (direct, no RGB projection) ---")
        print(f"FP16 latent stats:  min={lat_fp16.min():.4f} max={lat_fp16.max():.4f} mean={lat_fp16.mean():.4f} std={lat_fp16.std():.4f}")
        print(f"NVFP4 latent stats: min={lat_q.min():.4f} max={lat_q.max():.4f} mean={lat_q.mean():.4f} std={lat_q.std():.4f}")
        print(f"Latent MSE:      {lat_mse:.6f}  (0 = perfect)")
        print(f"Latent Cosine:   {lat_cos:.6f}  (1.0 = perfect)")

        if img_fp16.size != img_q.size:
            print(
                f"Error: Image sizes do not match! "
                f"FP16:{img_fp16.size}, NVFP4:{img_q.size}"
            )
            print("Different models or settings used.")
            return 1

        mse, score = calculate_metrics(img_fp16, img_q)

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
        target_grade = "PASS" if score >= SSIM_TARGET else "FAIL"
        print(f"  SSIM target >={SSIM_TARGET}: {target_grade}")

        diff_img = ImageChops.difference(img_fp16, img_q)
        diff_img = ImageChops.multiply(diff_img, Image.new("RGB", diff_img.size, (10, 10, 10)))
        diff_path = os.path.join(args.output_dir, "bench_result_diff.png")
        diff_img.save(diff_path)
        print(f"Diff image saved: {diff_path}")
        return 0 if score >= SSIM_TARGET else 1
    finally:
        _restore_argv(saved_argv)


if __name__ == "__main__":
    raise SystemExit(main())

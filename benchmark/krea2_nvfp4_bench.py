#!/usr/bin/env python3
"""
Krea2 ConvRot NVFP4 + ConvRot INT8-protect ComfyUI Native Benchmark
=================================================================
Compare BF16/FP16 Krea2 DiT vs HSWQ Kitchen NVFP4 (ConvRot) with
ConvRot INT8 protect layers in the same checkpoint
(native_convert_nvfp4_krea2_2 output).

NVFP4 runtime MUST import only from benchmark/krea2_nvfp4.
Never import benchmark/nvfp4.

Example:
  python krea2_nvfp4_bench.py \\
    --fp16  "D:\\...\\moodyKrea2Mix_v40BF16.safetensors" \\
    --nvfp4 "D:\\...\\moodyKrea2Mix_v40_nvfp4_convrot.safetensors" \\
    --clip_path "D:\\...\\Qwen3_VL_4B_Thinking_abliterated.safetensors" \\
    --comfy_path "D:\\USERFILES\\GitHub\\hswq\\ComfyUI-master" \\
    --token "hf_..." \\
    --prompt "A beautiful cyberpunk city at night, high detail." \\
    --steps 25

Optional: --vae PATH for pixel-space decode (same metrics path as int8bench_sdxl).
Without --vae: Wan21 latent-RGB preview image, still MSE/SSIM like SDXL printout.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops
from skimage.metrics import structural_similarity as ssim

# Parent orchestrator sets this so each branch runs in a fresh process
# (Windows WDDM reserved + shared GPU memory is released on process exit).
_BRANCH_ENV = "HSWQ_KREA2_BRANCH"


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


SSIM_TARGET = 0.9


def require_convrot_parity_forward() -> None:
    """Fail unless HSWQ load + HSWQ TC forward are both active.

    Stock F.linear + kitchen both-QT gate dequants every Linear → packed + FP16
    dual residency (~27 GB Task Manager). TC forward is the VRAM-correct path;
    ConvRot online x@H is inside TC when ``_hswq_nvfp4_convrot`` is armed at load.
    """
    import comfy.ops
    from krea2_nvfp4.nvfp4_comfy_parity import _load_chain_has_hswq_full_load

    if not _load_chain_has_hswq_full_load(comfy.ops._load_quantized_module):
        raise RuntimeError(
            "NVFP4 bench: HSWQ load_nvfp4_linear_module missing from "
            "ops._load_quantized_module (stock Comfy load destroys VRAM)"
        )
    lin_fwd = comfy.ops.mixed_precision_ops().Linear.forward
    if not getattr(lin_fwd, "_hswq_nvfp4_full_forward", False):
        raise RuntimeError(
            "NVFP4 bench: Linear.forward missing HSWQ TC wrap "
            "(_hswq_nvfp4_full_forward); stock F.linear would destroy VRAM"
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
            f"{bad[:8]}; use krea2_nvfp4 only"
        )


def apply_quant_patches() -> None:
    """NVFP4 from krea2_nvfp4 + INT8 protect from benchmark/int8.

    ConvRot NVFP4 needs comfy_quant_nvfp4 + Comfy parity (online x@H).
    INT8 protect layers need comfy_quant_int8 (same ckpt).
    Never import benchmark/nvfp4. Does not touch ComfyUI-master.
    """
    import comfy.ops

    from krea2_nvfp4.comfy_quant_nvfp4 import apply_comfy_quant_nvfp4_patches
    from krea2_nvfp4.nvfp4_comfy_parity import apply_nvfp4_comfy_parity
    import krea2_nvfp4.comfy_quant_nvfp4 as _cq_nvfp4

    from int8.comfy_quant_int8 import apply_comfy_quant_int8_patches
    import int8.comfy_quant_int8 as _cq_int8

    apply_comfy_quant_nvfp4_patches()
    if not apply_nvfp4_comfy_parity():
        raise RuntimeError(
            "krea2_nvfp4 HSWQ load + HSWQ TC forward failed to apply "
            "(need [BENCH] nvfp4 HSWQ load + HSWQ TC forward log)"
        )
    require_convrot_parity_forward()
    print(
        "  [NVFP4] HSWQ load + TC forward armed: "
        "act rotate (ConvRot) → NVFP4 quant → scaled_mm "
        "(stock Comfy load OFF; no full-weight dequant)"
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

    try:
        from krea2_nvfp4.nvfp4_runtime import clear_nvfp4_runtime_pools

        clear_nvfp4_runtime_pools()
    except Exception:
        pass
    mm.unload_all_models()
    mm.soft_empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _nvidia_smi_used_mib() -> float | None:
    """Dedicated VRAM used (MiB) from nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        # Multi-GPU: use the max (NVIDIA card under load).
        vals = [float(x.strip()) for x in out.strip().splitlines() if x.strip()]
        return max(vals) if vals else None
    except Exception:
        return None


def _report_gpu_memory(tag: str) -> None:
    """Print nvidia-smi + torch allocated/reserved (Task Manager parity aid)."""
    smi = _nvidia_smi_used_mib()
    smi_s = f"{smi:.0f} MiB" if smi is not None else "n/a"
    if torch.cuda.is_available():
        try:
            alloc = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            print(
                f"  [GPU] {tag}: nvidia-smi used={smi_s}  "
                f"torch alloc={alloc:.0f} MiB reserved={reserved:.0f} MiB",
                flush=True,
            )
            return
        except Exception:
            pass
    print(f"  [GPU] {tag}: nvidia-smi used={smi_s}", flush=True)


def _require_gpu_headroom_for_nvfp4(
    *,
    max_used_mib: float = 3072.0,
    timeout_s: float = 45.0,
) -> None:
    """
    Abort if dedicated VRAM is still bloated before NVFP4 load.

    After an FP16 worker exits, WDDM should release reserved+shared. If
    nvidia-smi still shows many GiB, loading NVFP4 on top recreates the
    Task Manager ~27 GB picture (15 GB dedicated + shared spill).
    """
    t0 = time.perf_counter()
    while True:
        used = _nvidia_smi_used_mib()
        _report_gpu_memory("pre_nvfp4_gate")
        if used is None:
            print(
                "  [GPU] pre_nvfp4_gate: nvidia-smi unavailable — "
                "continuing without WDDM gate",
                flush=True,
            )
            return
        if used <= max_used_mib:
            print(
                f"  [GPU] pre_nvfp4_gate OK: used={used:.0f} MiB "
                f"<= {max_used_mib:.0f} MiB",
                flush=True,
            )
            return
        elapsed = time.perf_counter() - t0
        if elapsed >= timeout_s:
            raise RuntimeError(
                f"[BENCH] GPU still {used:.0f} MiB used before NVFP4 load "
                f"(limit {max_used_mib:.0f} MiB after {timeout_s:.0f}s). "
                "WDDM reserved/shared not released — run with default "
                "subprocess isolation (omit --inprocess), or close other "
                "GPU apps. Task Manager ~27 GB is this failure mode."
            )
        print(
            f"  [GPU] waiting for WDDM release: used={used:.0f} MiB "
            f"(>{max_used_mib:.0f}); retry...",
            flush=True,
        )
        time.sleep(2.0)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_diffusion_model(unet_path: str):
    """Load DiT; wrap INT8-protect layers in Conv2d inject scope when present."""
    import comfy.sd
    from int8.comfy_quant_int8 import (
        _int8_quant_conv_scope,
        checkpoint_looks_like_comfy_quant_int8,
    )
    from krea2_nvfp4.comfy_quant_nvfp4 import checkpoint_looks_like_comfy_quant_nvfp4

    looks_nvfp4 = checkpoint_looks_like_comfy_quant_nvfp4(unet_path)
    use_int8_scope = checkpoint_looks_like_comfy_quant_int8(unet_path)
    print(f"  [BENCH] NVFP4 comfy_quant detect: {looks_nvfp4}")
    print(f"  [BENCH] INT8 Conv2d load scope: {use_int8_scope}")
    if use_int8_scope:
        with _int8_quant_conv_scope():
            return comfy.sd.load_diffusion_model(unet_path, {})
    return comfy.sd.load_diffusion_model(unet_path, {})


def run_branch(
    *,
    label: str,
    unet_path: str,
    vae,
    positive,
    negative,
    args,
) -> tuple[Image.Image, torch.Tensor, float, float]:
    print(f"\n=== {label}: loading UNet ===")
    print(f"  path: {unet_path}")
    _report_gpu_memory(f"before_load/{label}")
    t0 = time.perf_counter()
    # Do NOT load_models_gpu([unet, clip]) — master CPU-offloads via sample().
    model = _load_diffusion_model(unet_path)
    load_s = time.perf_counter() - t0
    print(f"  load: {load_s:.2f}s")

    # NVFP4 branches: abort before sample if load still on dual-residency path.
    from krea2_nvfp4.comfy_quant_nvfp4 import checkpoint_looks_like_comfy_quant_nvfp4
    from krea2_nvfp4.nvfp4_addmm_patch import (
        nvfp4_addmm_stats,
        reset_nvfp4_addmm_stats,
    )
    from krea2_nvfp4.nvfp4_comfy_parity import require_nvfp4_vram_safe_load
    from krea2_nvfp4.nvfp4_forward import nvfp4_forward_stats, reset_nvfp4_forward_stats

    is_nvfp4_run = checkpoint_looks_like_comfy_quant_nvfp4(unet_path) or (
        "NVFP4" in label.upper() or "nvfp4" in unet_path.lower()
    )
    if is_nvfp4_run:
        require_nvfp4_vram_safe_load(model)
        reset_nvfp4_forward_stats()
        reset_nvfp4_addmm_stats()

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
    if is_nvfp4_run:
        fwd = nvfp4_forward_stats()
        addmm = nvfp4_addmm_stats()
        print(
            f"  [BENCH] NVFP4 forward stats: "
            f"scaled_mm_hits={fwd['scaled_mm_hits']} "
            f"dequant_fallbacks={fwd['dequant_fallbacks']} "
            f"convrot_act_rotates={fwd['convrot_act_rotates']}",
            flush=True,
        )
        print(
            f"  [BENCH] NVFP4 addmm stats: "
            f"addmm_scaled_mm_hits={addmm['addmm_scaled_mm_hits']} "
            f"addmm_dequant_fallbacks={addmm.get('addmm_dequant_fallbacks', 0)}",
            flush=True,
        )
        if fwd["scaled_mm_hits"] == 0 and fwd["dequant_fallbacks"] > 0:
            raise RuntimeError(
                "[BENCH] NVFP4 sample used only dequant fallbacks "
                f"(hits=0, fallbacks={fwd['dequant_fallbacks']}) — VRAM dual path"
            )
        if (
            fwd["scaled_mm_hits"] == 0
            and addmm["addmm_scaled_mm_hits"] == 0
            and is_nvfp4_run
        ):
            raise RuntimeError(
                "[BENCH] NVFP4 sample had zero scaled_mm hits "
                "(forward + addmm) — TC path never ran"
            )

    samples_t = out["samples"]
    lat_cpu = samples_t.detach().float().cpu()

    if vae is not None:
        # Free DiT VRAM before VAE decode. Keeping UNet loaded leaves ~100 MiB
        # free on 32 GiB cards → regular decode OOM → tiled decode → Comfy
        # process_output inplace (add_/div_) crashes under InferenceMode.
        latent = samples_t.detach()
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[0]
        del model, out, samples_t
        _hard_free_vram()

        print("  decoding with VAE...")
        # Bench-only: non-inplace process_output (do not edit ComfyUI-master).
        _po = vae.process_output
        vae.process_output = lambda image: image.float().add(1.0).mul(0.5).clamp(0.0, 1.0)
        try:
            with torch.inference_mode(False):
                images = vae.decode(latent)
        finally:
            vae.process_output = _po
        if len(images.shape) == 5:  # Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        img_array = 255.0 * images[0].detach().cpu().numpy()
        img = Image.fromarray(np.clip(img_array, 0, 255).astype("uint8"))
        del latent, images
    else:
        img = latent_to_rgb_preview(samples_t, model)
        del model, out, samples_t

    _hard_free_vram()
    _report_gpu_memory(f"after_branch/{label}")
    return img, lat_cpu, sample_s, peak_gb


def calculate_metrics(img1, img2):
    """Same as int8bench_sdxl.calculate_metrics (pixel RGB)."""
    arr1 = np.array(img1)
    arr2 = np.array(img2)

    # MSE (mean squared error)
    mse = np.mean((arr1 - arr2) ** 2)

    # SSIM (structural similarity)
    score_ssim = ssim(arr1, arr2, win_size=3, channel_axis=2, data_range=255)

    return mse, score_ssim


def _write_branch_meta(output_dir: str, branch: str, sample_s: float, peak_gb: float) -> None:
    path = Path(output_dir) / f"bench_meta_{branch}.json"
    path.write_text(
        json.dumps(
            {"branch": branch, "sample_s": sample_s, "peak_gb": peak_gb},
            indent=2,
        ),
        encoding="utf-8",
    )


def _print_quality_report(img_fp16: Image.Image, img_q: Image.Image, output_dir: str) -> int:
    print("\n=== 3. Calculating Metrics ===")
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
    diff_path = os.path.join(output_dir, "bench_result_diff.png")
    diff_img.save(diff_path)
    print(f"Diff image saved: {diff_path}")
    return 0 if score >= SSIM_TARGET else 1


def _run_parent_orchestrator(argv_rest: list[str], output_dir: str) -> int:
    """
    Spawn FP16 then NVFP4 in separate processes so WDDM releases
    reserved + shared GPU memory between branches.
    """
    script = str(Path(__file__).resolve())
    py = sys.executable
    print(
        "[BENCH] Subprocess isolation ON "
        "(each branch exits → OS reclaim reserved/shared GPU memory). "
        "Use --inprocess only for debug.",
        flush=True,
    )
    for branch in ("fp16", "nvfp4"):
        env = os.environ.copy()
        env[_BRANCH_ENV] = branch
        _report_gpu_memory(f"parent_before_{branch}_worker")
        if branch == "nvfp4":
            _require_gpu_headroom_for_nvfp4()
        print(f"\n[BENCH] spawning {branch} worker...", flush=True)
        r = subprocess.run([py, "-u", script, *argv_rest], env=env)
        if r.returncode != 0:
            print(
                f"[BENCH] {branch} worker failed with exit={r.returncode}",
                flush=True,
            )
            return int(r.returncode)
        _report_gpu_memory(f"parent_after_{branch}_worker_exit")
        meta_path = Path(output_dir) / f"bench_meta_{branch}.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            print(
                f"[BENCH] {branch} worker meta: "
                f"sample={meta.get('sample_s', '?')}s "
                f"peak_vram={meta.get('peak_gb', '?')} GiB",
                flush=True,
            )

    p16 = os.path.join(output_dir, "bench_result_fp16.png")
    pq = os.path.join(output_dir, "bench_result_nvfp4.png")
    if not Path(p16).is_file() or not Path(pq).is_file():
        print(f"[BENCH] missing output PNGs: {p16!r} / {pq!r}", flush=True)
        return 1
    return _print_quality_report(Image.open(p16), Image.open(pq), output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Krea2 ConvRot NVFP4 + ConvRot INT8-protect ComfyUI Native Benchmark"
        )
    )
    parser.add_argument("--fp16", required=True, help="BF16/FP16 Krea2 DiT safetensors")
    parser.add_argument(
        "--nvfp4",
        required=True,
        help=(
            "Kitchen NVFP4 ConvRot + INT8-protect Krea2 DiT safetensors "
            "(native_convert_nvfp4_krea2_2 output)"
        ),
    )
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
    parser.add_argument(
        "--inprocess",
        action="store_true",
        help=(
            "DEBUG: run FP16 then NVFP4 in one process. "
            "WDDM may keep reserved+shared (~27 GB Task Manager). "
            "Default is subprocess isolation per branch."
        ),
    )
    args = parser.parse_args()

    for p, name in ((args.fp16, "--fp16"), (args.nvfp4, "--nvfp4"), (args.clip_path, "--clip_path")):
        if not Path(p).is_file():
            raise FileNotFoundError(f"{name} not found: {p}")

    os.makedirs(args.output_dir, exist_ok=True)

    branch = (os.environ.get(_BRANCH_ENV) or "").strip().lower()
    # Parent: spawn isolated workers (default). Workers set HSWQ_KREA2_BRANCH.
    if not branch and not args.inprocess:
        # Drop --inprocess from child argv if present (should not be).
        child_argv = [a for a in sys.argv[1:] if a != "--inprocess"]
        return _run_parent_orchestrator(child_argv, args.output_dir)

    set_hf_token(args.token)

    # Ensure benchmark/ is on path for krea2_nvfp4 + int8 packages
    bench_dir = Path(__file__).resolve().parent
    if str(bench_dir) not in sys.path:
        sys.path.insert(0, str(bench_dir))

    run_fp16 = branch in ("", "fp16") or args.inprocess
    run_nvfp4 = branch in ("", "nvfp4") or args.inprocess
    if branch == "fp16":
        run_nvfp4 = False
    elif branch == "nvfp4":
        run_fp16 = False

    saved_argv = _clear_argv_for_comfy()
    try:
        setup_comfy(args.comfy_path)
        apply_quant_patches()

        import folder_paths  # noqa: F401
        import comfy.model_management as mm
        import comfy.sd

        mm.get_torch_device()
        _report_gpu_memory("after_comfy_init")

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
        _report_gpu_memory("after_clip_offload")

        print("--- Benchmark Config ---")
        print(f"Seed: {args.seed}  Steps: {args.steps}  CFG: {args.cfg}")
        print(f"Size: {args.width}x{args.height}  sampler={args.sampler}/{args.scheduler}")
        print(f"Prompt: {args.prompt[:80]}...")
        if branch:
            print(f"Worker branch: {branch} (subprocess isolation)")
        elif args.inprocess:
            print("Mode: --inprocess (both branches; WDDM may retain memory)")
        print("------------------------")

        img_fp16 = None
        img_q = None

        if run_fp16:
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
            _write_branch_meta(args.output_dir, "fp16", t16, v16)

        if run_nvfp4:
            if args.inprocess and run_fp16:
                # Same-process FP16→NVFP4: require WDDM release or abort.
                _require_gpu_headroom_for_nvfp4()
            elif branch == "nvfp4":
                _require_gpu_headroom_for_nvfp4()
            img_q, _lat_q, tq, vq = run_branch(
                label="2. Quantized (ConvRot NVFP4 + INT8 protect)",
                unet_path=args.nvfp4,
                vae=vae,
                positive=positive,
                negative=negative,
                args=args,
            )
            pq = os.path.join(args.output_dir, "bench_result_nvfp4.png")
            img_q.save(pq)
            print(f"NVFP4+INT8protect Time: {tq:.2f}s  peak={vq:.2f}GiB")
            _write_branch_meta(args.output_dir, "nvfp4", tq, vq)

        # Worker: exit after one branch (parent computes metrics).
        if branch in ("fp16", "nvfp4"):
            print(f"[BENCH] worker {branch} done — exiting process (VRAM reclaim)", flush=True)
            return 0

        # In-process both branches: metrics here.
        if img_fp16 is None or img_q is None:
            print("[BENCH] inprocess missing a branch image", flush=True)
            return 1
        return _print_quality_report(img_fp16, img_q, args.output_dir)
    finally:
        _restore_argv(saved_argv)


if __name__ == "__main__":
    raise SystemExit(main())

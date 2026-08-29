#!/usr/bin/env python3
"""Flux1 Hybrid ConvRot NVFP4 ComfyUI Native Benchmark
=====================================
Compare BF16/FP16 Flux1 DiT vs Hybrid ConvRot NVFP4 (INT8 protect + NVFP4, comfy_quant).

Example:
  python flux_int8_bench.py ^
    --fp16  "D:\\...\\redcraftHybridH3A2A_realreveal5.safetensors" ^
    --nvfp4 "D:\\...\\redcraftHybridH3A2A_realreveal5_hybrid_convrot_nvfp4.safetensors" ^
    --clip_path "D:\\...\\flan_t5_xxl_convrot_int8.safetensors" ^
    --clip_l_path "D:\\...\\clip_l.safetensors" ^
    --comfy_path "D:\\USERFILES\\ComfyUI\\ComfyUI" ^
    --vae "D:\\...\\ae.safetensors" ^
    --prompt "masterpiece, best quality, 1girl, solo, standing, simple background" ^
    --steps 12

Metrics:
  - Latent-space MSE / cosine（VAE decode 前。UNet 量子化誤差を直接測定）
  - Pixel-space MSE / SSIM（--vae 指定時のみ。Flux は VAE が無いと decode 不可）
  - VAE 未指定時は latent RGB preview で pixel SSIM を代替（参考値扱い）

Seeds: flux_int8_bench.py と同一（42 + 10桁以上 の 5 個デフォルト、20 シード対応）。
  「10桁以上」セットは --seeds 8426170395,9517038246,1357924680,2468135791,3579246812
  （MEMORY.md ルール: シードは「42 + 10桁以上」の 5 個を使用し、勝手に変えない）。
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

_DEFAULT_SEEDS = [42, 137, 5517, 92048, 371506]
_DEFAULT_GUIDANCE = 3.5


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

    # NVFP4 ConvRot runtime: prebind kitchen tensor exports before comfy.quant_ops import
    from flux1_nvfp4.kitchen_quant_ops_repair import (
        ensure_kitchen_quant_ops,
        prebind_missing_kitchen_tensor_exports,
    )

    prebind_missing_kitchen_tensor_exports()

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

    # Resolve quant_ops now (after prebind) and apply Branch A/B before model load.
    import comfy.quant_ops  # noqa: F401

    ensure_kitchen_quant_ops()


def apply_quant_patches() -> None:
    """NVFP4 + INT8 comfy_quant monkey-patch（krea2_convrot_nvfp4_bench 相当）."""
    import comfy.ops

    from flux1_nvfp4.comfy_quant_nvfp4 import apply_comfy_quant_nvfp4_patches
    from flux1_nvfp4.nvfp4_comfy_parity import apply_nvfp4_comfy_parity
    import flux1_nvfp4.comfy_quant_nvfp4 as _cq_nvfp4

    from int8.comfy_quant_int8 import apply_comfy_quant_int8_patches
    import int8.comfy_quant_int8 as _cq_int8

    apply_comfy_quant_nvfp4_patches()
    if not apply_nvfp4_comfy_parity():
        raise RuntimeError("flux1_nvfp4 ComfyUI-only parity failed to apply")
    print(f"  [BENCH] nvfp4 patch file: {os.path.abspath(_cq_nvfp4.__file__)}")
    print(f"  [BENCH] comfy_quant_nvfp4 patched: {_cq_nvfp4._PATCHES_APPLIED}")

    apply_comfy_quant_int8_patches()
    print(f"  [BENCH] int8_tensorwise: {'int8_tensorwise' in comfy.ops.QUANT_ALGOS}")
    print(f"  [BENCH] comfy_quant_int8 patched: {_cq_int8._PATCHES_APPLIED}")
    if not _cq_int8._PATCHES_APPLIED:
        raise RuntimeError("comfy_quant_int8 patches failed to apply")


def set_hf_token(token: str | None) -> None:
    if not token:
        return
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def encode_prompt(clip, prompt: str):
    """Flux CLIP encode.

    注意: 現行 ComfyUI では encode_from_tokens_scheduled は Tensor のリストを返し、
    cond dict に guidance を注入できない。Flux の guidance は extra_conds の
    デフォルト 3.5 が自動適用される（fp16/int8 比較には影響なし）。
    """
    tokens = clip.tokenize(prompt)
    return clip.encode_from_tokens_scheduled(tokens)


def make_empty_latent(model, width: int, height: int, batch: int = 1) -> dict:
    import comfy.model_management as mm
    import comfy.sample as comfy_sample

    device = mm.intermediate_device()
    latent = torch.zeros([batch, 16, height // 8, width // 8], device=device)
    latent = comfy_sample.fix_empty_latent_channels(model, latent)
    return {"samples": latent}


def latent_to_rgb_preview_from_format(latent_t: torch.Tensor, latent_format) -> Image.Image:
    """Flux latent_format RGB preview when no VAE is provided."""
    factors = getattr(latent_format, "latent_rgb_factors", None) if latent_format is not None else None
    bias = getattr(latent_format, "latent_rgb_factors_bias", None) if latent_format is not None else None

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
    """Load DiT; wrap INT8 comfy_quant checkpoints in Conv2d inject scope."""
    import comfy.sd
    from int8.comfy_quant_int8 import (
        _int8_quant_conv_scope,
        checkpoint_looks_like_comfy_quant_int8,
    )
    from flux1_nvfp4.comfy_quant_nvfp4 import checkpoint_looks_like_comfy_quant_nvfp4

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
) -> list[dict]:
    """Load DiT once; sample per seed. Returns list of {seed, img, lat, time, peak}.

    全シードのサンプリング完了後に DiT を解放し、VAE decode をまとめて行う
    （モデルを毎シード再ロードせず、VAE decode 用に VRAM を空ける）。
    """
    print(f"\n=== {label}: loading UNet ===")
    print(f"  path: {unet_path}")
    t0 = time.perf_counter()
    model = _load_diffusion_model(unet_path)
    load_s = time.perf_counter() - t0
    print(f"  load: {load_s:.2f}s")

    latent_format = getattr(getattr(model, "model", model), "latent_format", None)

    results = []
    for seed in args.seeds:
        print(f"  --- seed {seed} ---")
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
            seed=seed,
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
        results.append(
            {"seed": seed, "lat": lat_cpu, "time": sample_s, "peak": peak_gb}
        )
        del out, samples_t

    # DiT を解放してから VAE decode（まとめて実行）
    del model
    _hard_free_vram()

    if vae is not None:
        import comfy.model_management as mm

        dev = mm.get_torch_device()
        for r in results:
            print("  decoding with VAE...")
            latent_d = r["lat"].to(device=dev)
            _po = vae.process_output
            vae.process_output = lambda image: image.float().add(1.0).mul(0.5).clamp(0.0, 1.0)
            try:
                with torch.inference_mode(False):
                    images = vae.decode(latent_d)
            finally:
                vae.process_output = _po
            if len(images.shape) == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            img_array = 255.0 * images[0].detach().cpu().numpy()
            r["img"] = Image.fromarray(np.clip(img_array, 0, 255).astype("uint8"))
            del latent_d, images
    else:
        for r in results:
            r["img"] = latent_to_rgb_preview_from_format(r["lat"], latent_format)
    return results


def calculate_metrics(img1, img2):
    """Pixel RGB MSE / SSIM."""
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    mse = np.mean((arr1 - arr2) ** 2)
    score_ssim = ssim(arr1, arr2, win_size=3, channel_axis=2, data_range=255)
    return mse, score_ssim


def main() -> int:
    parser = argparse.ArgumentParser(description="Flux1 INT8 ComfyUI Native Benchmark")
    parser.add_argument("--fp16", required=True, help="BF16/FP16 Flux1 DiT safetensors")
    parser.add_argument("--nvfp4", required=True, help="native INT8 Flux1 DiT safetensors")
    parser.add_argument("--clip_path", required=True, help="T5XXL text encoder safetensors (Flux)")
    parser.add_argument("--clip_l_path", required=True, help="clip_l text encoder safetensors (Flux)")
    parser.add_argument("--comfy_path", required=True, help="ComfyUI root")
    parser.add_argument("--token", default=None, help="HF token (env HF_TOKEN / HUGGING_FACE_HUB_TOKEN)")
    parser.add_argument("--prompt", default="masterpiece, best quality, 1girl, solo, standing, simple background", help="Benchmark prompt")
    parser.add_argument("--negative", default="", help="Negative prompt (unused at cfg=1)")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Seeds (default: 42,137,5517,92048,371506)")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--cfg", type=float, default=1.0, help="Flux default 1.0 (guidance is fixed 3.5 via extra_conds)")
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="simple")
    parser.add_argument("--vae", default=None, help="Optional Flux VAE (ae.safetensors) for pixel decode")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--save_image", action="store_true")
    args = parser.parse_args()

    if args.seeds is None:
        args.seeds = list(_DEFAULT_SEEDS)
    if not args.seeds:
        print("Error: --seeds is empty")
        return 1

    for p, name in ((args.fp16, "--fp16"), (args.nvfp4, "--nvfp4"),
                    (args.clip_path, "--clip_path"), (args.clip_l_path, "--clip_l_path")):
        if not Path(p).is_file():
            raise FileNotFoundError(f"{name} not found: {p}")

    set_hf_token(args.token)
    os.makedirs(args.output_dir, exist_ok=True)

    bench_dir = Path(__file__).resolve().parent  # benchmark/flux1_nvfp4
    if str(bench_dir) not in sys.path:
        sys.path.insert(0, str(bench_dir))
    # int8 パッケージ（benchmark/int8）用に親ディレクトリ（benchmark 直下）も追加
    bench_parent = bench_dir.parent
    if str(bench_parent) not in sys.path:
        sys.path.insert(0, str(bench_parent))

    saved_argv = _clear_argv_for_comfy()
    try:
        setup_comfy(args.comfy_path)
        apply_quant_patches()

        import comfy.model_management as mm
        import comfy.sd

        mm.get_torch_device()

        print("Loading CLIP (Flux / clip_l + t5xxl)...")
        clip = comfy.sd.load_clip(
            ckpt_paths=[args.clip_l_path, args.clip_path],
            embedding_directory=None,
            clip_type=comfy.sd.CLIPType.FLUX,
        )

        vae = None
        if args.vae:
            if not Path(args.vae).is_file():
                raise FileNotFoundError(f"--vae not found: {args.vae}")
            print(f"Loading VAE: {args.vae}")
            sd = comfy.utils.load_torch_file(args.vae)
            vae = comfy.sd.VAE(sd=sd)
        else:
            print("No --vae: flux VAE が無いため pixel SSIM は latent RGB preview の参考値のみ")

        print("Encoding prompt...")
        positive = encode_prompt(clip, args.prompt)
        negative = encode_prompt(clip, args.negative) if args.negative else encode_prompt(clip, "")

        # エンコード済み cond を CPU へ（VRAM を DiT に全解放）
        def _conds_to_cpu(conds):
            for t in range(len(conds)):
                for i in range(len(conds[t])):
                    if isinstance(conds[t][i], torch.Tensor):
                        conds[t][i] = conds[t][i].cpu()
            return conds

        positive = _conds_to_cpu(positive)
        negative = _conds_to_cpu(negative)

        # CLIP を完全に CPU オフロード（cond_stage_model + patcher + 明示解放）
        try:
            if getattr(clip, "cond_stage_model", None) is not None:
                clip.cond_stage_model.cpu()
        except Exception as e:
            print(f"  [WARN] cond_stage_model.cpu() failed: {e}")
        try:
            if getattr(clip, "patcher", None) is not None:
                mm.unload_model_and_clones(clip.patcher)
        except Exception as e:
            print(f"  [WARN] unload_model_and_clones failed: {e}")
        del clip
        _hard_free_vram()
        print("  [Offload] CLIP fully offloaded to CPU (VRAM freed for Flux DiT benchmark).")

        print("--- Benchmark Config ---")
        print(f"Seeds: {args.seeds}")
        print(f"Steps: {args.steps}  CFG: {args.cfg}  Guidance: 3.5 (fixed)")
        print(f"Size: {args.width}x{args.height}  sampler={args.sampler}/{args.scheduler}")
        print(f"Prompt: {args.prompt[:80]}...")
        print("------------------------")

        res_fp16 = run_branch(
            label="1. Baseline (FP16/BF16)",
            unet_path=args.fp16,
            vae=vae,
            positive=positive,
            negative=negative,
            args=args,
        )
        res_int8 = run_branch(
            label="2. Quantized (Hybrid ConvRot NVFP4)",
            unet_path=args.nvfp4,
            vae=vae,
            positive=positive,
            negative=negative,
            args=args,
        )

        if args.save_image:
            for r in res_fp16:
                r["img"].save(os.path.join(args.output_dir, f"bench_result_fp16_seed{r['seed']}.png"))
            for r in res_int8:
                r["img"].save(os.path.join(args.output_dir, f"bench_result_int8_seed{r['seed']}.png"))

        print("\n=== 3. Calculating Metrics ===")
        lat_mse_list = []
        lat_cos_list = []
        px_mse_list = []
        ssim_list = []

        for r16, r8 in zip(res_fp16, res_int8):
            seed = r16["seed"]
            print(f"\n--- seed {seed} ---")
            if r16["lat"].shape != r8["lat"].shape:
                print(f"  *** LATENT SHAPE MISMATCH: FP16={tuple(r16['lat'].shape)} INT8={tuple(r8['lat'].shape)} ***")
                continue
            l16 = r16["lat"].reshape(-1)
            l8 = r8["lat"].reshape(-1)
            lmse = float((l16 - l8).pow(2).mean().item())
            lcos = float(torch.nn.functional.cosine_similarity(
                l16.unsqueeze(0), l8.unsqueeze(0), dim=1).item())
            lat_mse_list.append(lmse)
            lat_cos_list.append(lcos)
            print(f"  Latent MSE   : {lmse:.6f}  (0 = perfect)")
            print(f"  Latent Cos   : {lcos:.6f}  (1.0 = perfect)")

            if r16["img"].size == r8["img"].size:
                pmse, pssim = calculate_metrics(r16["img"], r8["img"])
                px_mse_list.append(pmse)
                ssim_list.append(pssim)
                print(f"  Pixel MSE    : {pmse:.4f}")
                print(f"  SSIM         : {pssim:.4f}  (1.0 = perfect)")
            else:
                print(f"  !! image size mismatch: {r16['img'].size} vs {r8['img'].size}")

        print("\n--- Multi-seed summary ---")
        if lat_mse_list:
            print(f"Latent MSE : min={min(lat_mse_list):.6f}  mean={np.mean(lat_mse_list):.6f}  max={max(lat_mse_list):.6f}")
        if lat_cos_list:
            print(f"Latent Cos : min={min(lat_cos_list):.6f}  mean={np.mean(lat_cos_list):.6f}  max={max(lat_cos_list):.6f}")
        if ssim_list:
            print(f"Pixel MSE  : min={min(px_mse_list):.4f}  mean={np.mean(px_mse_list):.4f}  max={max(px_mse_list):.4f}")
            print(f"SSIM       : min={min(ssim_list):.4f}  mean={np.mean(ssim_list):.4f}  max={max(ssim_list):.4f}")
            score = np.mean(ssim_list)
            if score > 0.98:
                grade = "PERFECT (S)"
            elif score > 0.95:
                grade = "EXCELLENT (A)"
            elif score > 0.90:
                grade = "GOOD (B)"
            else:
                grade = "WARNING (C)"
            print(f"Quality Grade: {grade}")
        else:
            print("(VAE 未指定のため pixel SSIM は集計対象外)")

        times16 = [r["time"] for r in res_fp16]
        times8 = [r["time"] for r in res_int8]
        peaks16 = [r["peak"] for r in res_fp16]
        peaks8 = [r["peak"] for r in res_int8]
        print("--------------------------------------------------")
        print(f"Inference Time:       FP16: {np.mean(times16):.2f}s/seed")
        print(f"                      INT8: {np.mean(times8):.2f}s/seed")
        print(f"Peak VRAM:            FP16: {np.mean(peaks16):.2f} GiB")
        print(f"                      INT8: {np.mean(peaks8):.2f} GiB")
        print("--------------------------------------------------")
        return 0
    finally:
        _restore_argv(saved_argv)


if __name__ == "__main__":
    raise SystemExit(main())

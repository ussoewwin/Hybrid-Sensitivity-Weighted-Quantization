#!/usr/bin/env python3
"""
SeedVR2 INT8 Benchmark (seedvr2_videoupscaler path)
===================================================
Compare community FP16 SeedVR2 DiT vs HSWQ native INT8 (int8_tensorwise + ConvRot)
through numz SeedVR2_VideoUpscaler (same encode / DiT / decode / color pipeline).

INT8 checkpoints use comfy_quant keys and cannot load into videoupscaler as-is.
This bench dequantizes INT8+ConvRot to a temporary FP16 safetensors (kitchen
dequantize_int8_convrot_weight), then runs both branches on that pipeline.

Primary metric: FP16 output vs INT8-dequant output (MSE / SSIM / diff PNG).

Example:
  python seedvr2_int8_bench.py ^
    --fp16  "D:\\USERFILES\\ComfyUI\\ComfyUI\\models\\SEEDVR2\\seedvr2_ema_7b_fp16.safetensors" ^
    --int8  "D:\\USERFILES\\ComfyUI\\ComfyUI\\models\\SEEDVR2\\seedvr2_7b_int8_convrot.safetensors" ^
    --vae   "D:\\USERFILES\\ComfyUI\\ComfyUI\\models\\SEEDVR2\\ema_vae_fp16.safetensors" ^
    --image "D:\\path\\to\\input.jpg" ^
    --output_dir "D:\\USERFILES\\GitHub\\hswq\\benchmark\\seedvr2_out"

--image is optional: when omitted, a synthetic RGB pattern is used.
Default resolution=1080 / color_correction=lab match videoupscaler CLI defaults.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import types
from pathlib import Path

# Windows cp932 consoles choke on seedvr2 emoji prints during import.
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import save_file
from skimage.metrics import structural_similarity as ssim


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDVR2_PATH = REPO_ROOT / "seedvr2_videoupscaler"
INT8_DEQUANT_NAME = "_hswq_bench_int8_dequant_fp16.safetensors"


def _clean_path(p: str) -> str:
    """PowerShell trailing \\\" leaves a final backslash; strip it."""
    return os.path.normpath(str(p).rstrip("\\/"))


def make_synthetic_rgb(short_edge: int = 512) -> Image.Image:
    h = short_edge
    w = int(round(short_edge * 16 / 9))
    w = max(w, short_edge)
    img = Image.new("RGB", (w, h), (32, 40, 56))
    draw = ImageDraw.Draw(img)
    for i in range(0, w, 32):
        draw.line([(i, 0), (i, h)], fill=(i % 255, 90, 140), width=1)
    for j in range(0, h, 32):
        draw.line([(0, j), (w, j)], fill=(80, j % 255, 160), width=1)
    draw.ellipse(
        [w // 4, h // 4, 3 * w // 4, 3 * h // 4],
        outline=(220, 180, 60),
        width=4,
    )
    draw.rectangle(
        [w // 8, h // 8, w // 3, h // 3],
        fill=(180, 60, 90),
        outline=(255, 255, 255),
    )
    return img


def pil_to_thwc_f16(img: Image.Image) -> torch.Tensor:
    """[T=1, H, W, C] float16 in [0,1] — videoupscaler image tensor layout."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...].to(torch.float16)


def thwc_to_pil(t: torch.Tensor) -> Image.Image:
    x = t.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    arr = (x.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    return Image.fromarray(arr, mode="RGB")


def calculate_metrics(img1, img2):
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    mse = np.mean((arr1 - arr2) ** 2)
    score_ssim = ssim(arr1, arr2, win_size=3, channel_axis=2, data_range=255)
    return mse, score_ssim


def dequant_int8_convrot_to_fp16(
    int8_path: str,
    out_path: str,
) -> str:
    """
    Expand HSWQ int8_tensorwise + ConvRot (2D) weights to FP16 safetensors
    compatible with seedvr2_videoupscaler load_standard_weights.
    Drops comfy_quant / weight_scale / baked conditioning keys.
    """
    from safetensors import safe_open
    from comfy_kitchen.backends.eager.quantization import (
        dequantize_int8_convrot_weight,
        dequantize_int8_simple,
    )

    print(f"  [BENCH] dequant INT8 → FP16: {int8_path}")
    print(f"  [BENCH] write: {out_path}")
    t0 = time.perf_counter()
    out: dict[str, torch.Tensor] = {}
    n_convrot = 0
    n_plain = 0
    n_copy = 0

    with safe_open(int8_path, framework="pt") as f:
        keys = list(f.keys())
        conf_by_base: dict[str, dict] = {}
        for k in keys:
            if not k.endswith(".comfy_quant"):
                continue
            raw = bytes(f.get_tensor(k).cpu().tolist())
            conf_by_base[k[: -len(".comfy_quant")]] = json.loads(raw.decode("utf-8"))

        skip = set()
        for k in keys:
            if k.endswith(".comfy_quant") or k.endswith(".weight_scale"):
                skip.add(k)
            if k in ("positive_conditioning", "negative_conditioning"):
                skip.add(k)

        for k in keys:
            if k in skip:
                continue
            if not k.endswith(".weight"):
                out[k] = f.get_tensor(k).contiguous()
                n_copy += 1
                continue

            base = k[: -len(".weight")]
            scale_key = base + ".weight_scale"
            if scale_key not in keys:
                out[k] = f.get_tensor(k).contiguous()
                n_copy += 1
                continue

            q = f.get_tensor(k)
            scale = f.get_tensor(scale_key)
            conf = conf_by_base.get(base, {})
            if conf.get("convrot"):
                gs = int(conf.get("convrot_groupsize", 256) or 256)
                if q.ndim != 2:
                    raise RuntimeError(
                        f"ConvRot dequant expects 2D weight, got ndim={q.ndim} for {k}"
                    )
                w = dequantize_int8_convrot_weight(q, scale, gs)
                n_convrot += 1
            else:
                w = dequantize_int8_simple(q, scale)
                n_plain += 1
            out[k] = w.to(torch.float16).contiguous()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_file(out, out_path)
    print(
        f"  [BENCH] dequant done in {time.perf_counter() - t0:.1f}s "
        f"(convrot={n_convrot} plain={n_plain} other={n_copy} tensors={len(out)})"
    )
    return out_path


def _install_seedvr2_path(seedvr2_path: str) -> str:
    root = str(Path(seedvr2_path).resolve())
    if not Path(root).is_dir():
        raise FileNotFoundError(f"--seedvr2_path not found: {root}")
    # Prefer package root first (same as inference_cli.py)
    sys.path = [root] + [p for p in sys.path if Path(p).resolve() != Path(root)]
    os.environ["PYTHONPATH"] = root + os.pathsep + os.environ.get("PYTHONPATH", "")
    return root


def _build_cli_args(
    *,
    dit_model: str,
    model_dir: str,
    resolution: int,
    seed: int,
    color_correction: str,
    batch_size: int,
    attention_mode: str,
    blocks_to_swap: int,
    dit_offload_device: str,
    vae_offload_device: str,
    tensor_offload_device: str,
) -> argparse.Namespace:
    """Minimal Namespace matching inference_cli._process_frames_core expectations."""
    return argparse.Namespace(
        dit_model=dit_model,
        model_dir=model_dir,
        resolution=resolution,
        max_resolution=0,
        batch_size=batch_size,
        uniform_batch_size=False,
        seed=seed,
        skip_first_frames=0,
        load_cap=0,
        chunk_size=0,
        prepend_frames=0,
        temporal_overlap=0,
        color_correction=color_correction,
        input_noise_scale=0.0,
        latent_noise_scale=0.0,
        dit_offload_device=dit_offload_device,
        vae_offload_device=vae_offload_device,
        tensor_offload_device=tensor_offload_device,
        blocks_to_swap=blocks_to_swap,
        swap_io_components=False,
        vae_encode_tiled=False,
        vae_encode_tile_size=1024,
        vae_encode_tile_overlap=128,
        vae_decode_tiled=False,
        vae_decode_tile_size=1024,
        vae_decode_tile_overlap=128,
        tile_debug="false",
        attention_mode=attention_mode,
        compile_dit=False,
        compile_vae=False,
        compile_backend="inductor",
        compile_mode="default",
        compile_fullgraph=False,
        compile_dynamic=False,
        compile_dynamo_cache_size_limit=64,
        compile_dynamo_recompile_limit=128,
        cache_dit=False,
        cache_vae=False,
        debug=False,
    )


def run_branch(
    *,
    label: str,
    dit_model: str,
    model_dir: str,
    frames: torch.Tensor,
    args_ns: argparse.Namespace,
) -> tuple[Image.Image, float, float]:
    from src.utils.debug import Debug
    from inference_cli import _process_frames_core

    print(f"\n=== {label}: SeedVR2 videoupscaler ===")
    print(f"  dit_model: {dit_model}")
    print(f"  model_dir: {model_dir}")

    ns = types.SimpleNamespace(**vars(args_ns))
    ns.dit_model = dit_model
    ns.model_dir = model_dir

    debug = Debug(enabled=False)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    result = _process_frames_core(
        frames_tensor=frames,
        args=ns,
        device_id="0",
        debug=debug,
        runner_cache=None,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_gb = (
        torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )
    print(f"  wall: {elapsed:.2f}s  peak_vram={peak_gb:.2f} GiB  out={tuple(result.shape)}")

    img = thwc_to_pil(result)
    del result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return img, elapsed, peak_gb


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SeedVR2 INT8 bench via seedvr2_videoupscaler (FP16 vs INT8-dequant)"
    )
    parser.add_argument("--fp16", required=True, help="FP16 SeedVR2 DiT safetensors")
    parser.add_argument("--int8", required=True, help="HSWQ INT8 ConvRot SeedVR2 DiT safetensors")
    parser.add_argument(
        "--vae",
        required=True,
        help="SeedVR2 VAE safetensors (basename should be ema_vae_fp16.safetensors)",
    )
    parser.add_argument(
        "--seedvr2_path",
        default=str(DEFAULT_SEEDVR2_PATH),
        help=f"seedvr2_videoupscaler root (default: {DEFAULT_SEEDVR2_PATH})",
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Directory containing DiT/VAE filenames (default: directory of --fp16)",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Optional input image. When omitted, a synthetic RGB pattern is used.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1080,
        help="Target short-side resolution (videoupscaler default: 1080)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1, help="Frames per batch (4n+1; image=1)")
    parser.add_argument(
        "--color",
        default="lab",
        choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"],
        help="color_correction (default: lab)",
    )
    parser.add_argument("--attention_mode", default="sdpa")
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--dit_offload_device", default="none")
    parser.add_argument("--vae_offload_device", default="none")
    parser.add_argument("--tensor_offload_device", default="cpu")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument(
        "--keep_dequant",
        action="store_true",
        help="Keep temporary INT8→FP16 dequant file after the run",
    )
    args = parser.parse_args()

    args.fp16 = _clean_path(args.fp16)
    args.int8 = _clean_path(args.int8)
    args.vae = _clean_path(args.vae)
    args.seedvr2_path = _clean_path(args.seedvr2_path)
    args.output_dir = _clean_path(args.output_dir)
    if args.model_dir is not None:
        args.model_dir = _clean_path(args.model_dir)
    if args.image is not None:
        args.image = _clean_path(args.image)

    for p, name in (
        (args.fp16, "--fp16"),
        (args.int8, "--int8"),
        (args.vae, "--vae"),
    ):
        if not Path(p).is_file():
            raise FileNotFoundError(f"{name} not found: {p}")
    if args.image is not None and not Path(args.image).is_file():
        raise FileNotFoundError(f"--image not found: {args.image}")

    model_dir = args.model_dir or str(Path(args.fp16).parent)
    vae_name = Path(args.vae).name
    if Path(args.vae).resolve() != (Path(model_dir) / vae_name).resolve():
        target = Path(model_dir) / vae_name
        if not target.is_file():
            raise FileNotFoundError(
                f"VAE must live under --model_dir as {vae_name}: expected {target}"
            )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[BENCH] python: {sys.executable}")
    print(f"[BENCH] seedvr2_path: {args.seedvr2_path}")
    _install_seedvr2_path(args.seedvr2_path)

    from src.utils.model_registry import DEFAULT_VAE as _DEFAULT_VAE

    if vae_name != _DEFAULT_VAE:
        print(
            f"  [BENCH] WARNING: videoupscaler CLI hardcodes VAE={_DEFAULT_VAE}; "
            f"--vae basename is {vae_name}. Ensure {_DEFAULT_VAE} exists under {model_dir}."
        )
        default_vae_path = Path(model_dir) / _DEFAULT_VAE
        if not default_vae_path.is_file():
            raise FileNotFoundError(
                f"Place {_DEFAULT_VAE} in model_dir ({model_dir}) "
                f"or rename --vae to match (found {vae_name})"
            )

    # --- input ---
    if args.image:
        print(f"Loading image: {args.image}")
        pil_in = Image.open(args.image).convert("RGB")
    else:
        print("No --image: using synthetic RGB pattern")
        pil_in = make_synthetic_rgb(short_edge=max(256, args.resolution // 2))

    frames = pil_to_thwc_f16(pil_in)
    print(f"  input tensor: {tuple(frames.shape)} dtype={frames.dtype}")

    # --- INT8 dequant next to model_dir so find_model_file works ---
    dequant_path = str(Path(model_dir) / INT8_DEQUANT_NAME)
    dequant_int8_convrot_to_fp16(args.int8, dequant_path)

    ns = _build_cli_args(
        dit_model=Path(args.fp16).name,
        model_dir=model_dir,
        resolution=args.resolution,
        seed=args.seed,
        color_correction=args.color,
        batch_size=args.batch_size,
        attention_mode=args.attention_mode,
        blocks_to_swap=args.blocks_to_swap,
        dit_offload_device=args.dit_offload_device,
        vae_offload_device=args.vae_offload_device,
        tensor_offload_device=args.tensor_offload_device,
    )

    img_fp16, t_fp16, v_fp16 = run_branch(
        label="FP16",
        dit_model=Path(args.fp16).name,
        model_dir=model_dir,
        frames=frames,
        args_ns=ns,
    )
    out_fp16 = Path(args.output_dir) / "seedvr2_fp16.png"
    img_fp16.save(out_fp16)
    print(f"  saved: {out_fp16}")

    img_int8, t_int8, v_int8 = run_branch(
        label="INT8 (dequant→FP16 load)",
        dit_model=INT8_DEQUANT_NAME,
        model_dir=model_dir,
        frames=frames,
        args_ns=ns,
    )
    out_int8 = Path(args.output_dir) / "seedvr2_int8.png"
    img_int8.save(out_int8)
    print(f"  saved: {out_int8}")

    # Align sizes if needed
    if img_fp16.size != img_int8.size:
        print(
            f"  [BENCH] size mismatch FP16={img_fp16.size} INT8={img_int8.size}; "
            "resizing INT8 to FP16 for metrics"
        )
        img_int8 = img_int8.resize(img_fp16.size, Image.Resampling.LANCZOS)

    mse, score = calculate_metrics(img_fp16, img_int8)
    diff = Image.fromarray(
        np.abs(np.asarray(img_fp16).astype(np.int16) - np.asarray(img_int8).astype(np.int16))
        .clip(0, 255)
        .astype(np.uint8)
    )
    out_diff = Path(args.output_dir) / "seedvr2_diff.png"
    diff.save(out_diff)

    print("\n=== Results (FP16 vs INT8-dequant, same videoupscaler pipeline) ===")
    print(f"  MSE:  {mse:.6f}")
    print(f"  SSIM: {score:.6f}")
    print(f"  FP16 wall: {t_fp16:.2f}s  peak_vram={v_fp16:.2f} GiB")
    print(f"  INT8 wall: {t_int8:.2f}s  peak_vram={v_int8:.2f} GiB")
    print(f"  outputs: {out_fp16} | {out_int8} | {out_diff}")

    if not args.keep_dequant:
        try:
            os.remove(dequant_path)
            print(f"  removed temp dequant: {dequant_path}")
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

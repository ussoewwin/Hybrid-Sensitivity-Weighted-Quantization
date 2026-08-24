"""Simple SDXL UNet INT8 convert + Card 1 Bias Correction + Card 3 per-channel.

Pack:
  default              symmetric per-tensor (amax / 127)
  --per_channel_int8   per-out-channel scale (Card 3)

Card 1 (--bias_correction):
  DualMonitor hooks + StableDiffusionXLPipeline latent calib
  mu_x = DualMonitor.channel_act_mean; bias += -(W_q - W) @ mu_x
  ALWAYS applied to every INT8 Linear/Conv that has act_mean (full Card 1).
  No Approach A / no bias_correction_top_ratio gating (that dropped SSIM).
  Calib defaults: samples=32, steps=25.
  No Static Profile VETO / no V4 FP16 keep.
  Format tag stays int8_tensorwise.

This script is the FULL ConvRot converter (default ON):
  Linear + Conv2d when in_dim divisible by a power-of-4 group size —
  rotate + per-channel scale + comfy_quant.convrot stamp.
  Linear: kitchen online act rotate. Conv2d: HSWQ INT8 Conv2d online rotate
  (Params.convrot cleared at load; 4D kitchen dequant is 2D-only).
  Never scalar+stamp on ConvRot layers. Use --no-convrot only to pack plain INT8.

Post-convert bench (default ON): after save, subprocess
  benchmark/qi_int8_bench.py (--fp16 <input> --int8 <output> --clip_path
  --comfy_path required; tokenizer from ComfyUI-bundled qwen25_tokenizer;
  optional --vae; prompt/steps=12/seed=42 fixed inside). Pass --no-bench to skip.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import subprocess
import sys

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

_DEFAULT_GROUPSIZE = 256


def _default_repo_root() -> str:
    """Locate repo root by searching upward for native_convert_int8.py."""
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(8):
        if os.path.isfile(os.path.join(d, "native_convert_int8.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.abspath(os.path.join(here, os.pardir))


_REPO_ROOT = _default_repo_root()
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_hswq_v30():
    """Load quantize_sdxl_hswq_v3.0.py as a module (filename has a digit)."""
    path = os.path.join(_REPO_ROOT, "quantize_sdxl_hswq_v3.0.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"HSWQ V3.0 script not found: {path}")
    spec = importlib.util.spec_from_file_location("quantize_sdxl_hswq_v3_0", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["quantize_sdxl_hswq_v3_0"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_native_convert_int8():
    """Load sibling native_convert_int8.py for Hadamard / rotate_weight."""
    path = os.path.join(_REPO_ROOT, "native_convert_int8.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"native_convert_int8.py not found: {path}")
    name = "native_convert_int8_for_simple"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _encode_comfy_quant(config: dict) -> torch.Tensor:
    return torch.tensor(
        list(json.dumps(config, separators=(",", ":")).encode("utf-8")),
        dtype=torch.uint8,
    )


def pack_tensorwise(weight: torch.Tensor):
    """Symmetric per-tensor INT8: scale = amax / 127."""
    w = weight.float()
    amax = max(float(w.abs().max().item()), 1e-6)
    scale = amax / 127.0
    q = (w / scale).round().clamp(-127, 127).to(torch.int8)
    return q, torch.tensor(scale, dtype=torch.float32)


def pack_channelwise(weight: torch.Tensor):
    """Per-out-channel INT8 (Card 3 shape for kitchen dequant)."""
    w = weight.float()
    reduce_dims = tuple(range(1, w.dim()))
    amax = torch.clamp(w.abs().amax(dim=reduce_dims).reshape(-1), min=1e-6)
    scale = amax / 127.0
    if w.dim() == 4:
        scale_view = scale.view(-1, 1, 1, 1)
        amax_view = amax.view(-1, 1, 1, 1)
    elif w.dim() == 2:
        scale_view = scale.view(-1, 1)
        amax_view = amax.view(-1, 1)
    else:
        raise ValueError(f"unsupported weight ndim={w.dim()} for --per_channel_int8")
    clamped = torch.clamp(w, -amax_view, amax_view)
    q = (clamped / scale_view).round().clamp(-127, 127).to(torch.int8)
    return q, scale_view.to(dtype=torch.float32)


def run_card1_calib(
    *,
    input_path: str,
    calib_file: str,
    num_calib_samples: int,
    num_inference_steps: int,
    device: str,
):
    """Card 1 only: DualMonitor calib → channel_act_mean.

    Does NOT run Static Profile VETO, derive_hswq_strategy_int8, or V4 FP16 keep.
    """
    v30 = _load_hswq_v30()

    pipeline, _state_dict, comfyui_to_diffusers_map = v30.load_unet_from_safetensors(
        input_path, device
    )
    model = pipeline.unet

    print("Preparing calibration (Dual Monitor hooks; Card 1 act means)...")
    v30.dual_monitors.clear()
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handle = module.register_forward_hook(
                lambda m, i, o, n=name: v30.hook_fn(m, i, o, n)
            )
            handles.append(handle)

    print("Preparing calibration data...")
    with open(calib_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    if len(prompts) < num_calib_samples:
        prompts = (prompts * (num_calib_samples // len(prompts) + 1))[
            :num_calib_samples
        ]
    else:
        prompts = prompts[:num_calib_samples]

    print(
        f"Running calibration ({num_calib_samples} samples, "
        f"{num_inference_steps} steps)..."
    )
    if num_calib_samples != 32 or num_inference_steps != 25:
        print(
            "  [WARN] How-to / r32 recipe is num_calib_samples=32, "
            "num_inference_steps=25. current args differ."
        )
    pipeline.set_progress_bar_config(disable=False)
    generator = torch.Generator(device=device).manual_seed(42)

    for i, prompt in enumerate(prompts):
        print(f"\nSample {i+1}/{num_calib_samples}: {prompt[:50]}...")
        with torch.no_grad():
            pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                output_type="latent",
                generator=generator,
            )
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    for h in handles:
        h.remove()

    act_mean_dict = {}
    for name, mon in v30.dual_monitors.items():
        if mon.channel_act_mean is not None:
            act_mean_dict[name] = mon.channel_act_mean.detach().float().cpu()
    print(
        f"  [Card 1 DualMonitor] act_mean layers={len(act_mean_dict)} "
        f"(full Card 1; no VETO; no Approach A)"
    )

    del pipeline
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "act_mean_dict": act_mean_dict,
        "comfyui_to_diffusers_map": comfyui_to_diffusers_map,
        "v30": v30,
    }


def _release_vram(label: str = "post-convert") -> None:
    print(f"[*] Releasing VRAM ({label})...")
    gc.collect()
    if not torch.cuda.is_available():
        print(f"[*] VRAM clear ({label}): CUDA not available")
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        alloc_mib = torch.cuda.memory_allocated() / (1024 ** 2)
        reserv_mib = torch.cuda.memory_reserved() / (1024 ** 2)
        print(
            f"[*] VRAM clear ({label}): "
            f"allocated={alloc_mib:.1f} MiB reserved={reserv_mib:.1f} MiB"
        )
    except Exception:
        print(f"[*] VRAM clear ({label}): done")


# Exact --prompt / --steps from benchmark/qi_int8_bench.py defaults (fixed; not CLI).
_FIXED_QI_INT8_BENCH_PROMPT = "masterpiece, best quality, 1girl, solo, standing, simple background"
_FIXED_QI_INT8_BENCH_STEPS = 12
_FIXED_QI_INT8_BENCH_SEED = 42


def run_post_convert_qi_int8_bench(
    *,
    script_dir: str,
    fp16_path: str,
    int8_path: str,
    clip_path: str,
    comfy_path: str,
    vae_path: str | None = None,
) -> int:
    """After save: subprocess benchmark/qi_int8_bench.py.

    Bench argv:
      --fp16 <input> --int8 <INT8> --clip_path --comfy_path
      [--vae] --prompt --steps 12 --seed 42
    Tokenizer comes from ComfyUI-bundled qwen25_tokenizer under --comfy_path.
    """
    bench_script = os.path.join(
        script_dir, "benchmark", "qi_int8_bench.py"
    )
    if not os.path.isfile(bench_script):
        print(f"[FATAL] Post-convert bench script not found: {bench_script}")
        return 1
    if not os.path.isfile(fp16_path):
        print(f"[FATAL] Post-convert bench: FP16 (--model) missing: {fp16_path}")
        return 1
    if not os.path.isfile(int8_path):
        print(
            f"[FATAL] Post-convert bench: INT8 (--output) missing: {int8_path}"
        )
        return 1
    if not clip_path or not os.path.isfile(clip_path):
        print(f"[FATAL] Post-convert bench: --clip_path missing: {clip_path}")
        return 1
    if not comfy_path or not os.path.isdir(comfy_path):
        print(
            f"[FATAL] Post-convert bench: --comfy_path missing: {comfy_path}"
        )
        return 1
    if vae_path and not os.path.isfile(vae_path):
        print(f"[FATAL] Post-convert bench: --vae missing: {vae_path}")
        return 1

    _release_vram("pre-qi_int8_bench subprocess")

    cmd = [
        sys.executable,
        bench_script,
        "--fp16",
        fp16_path,
        "--int8",
        int8_path,
        "--clip_path",
        clip_path,
        "--comfy_path",
        comfy_path,
    ]
    if vae_path:
        cmd.extend(["--vae", vae_path])
    cmd.extend(
        [
            "--prompt",
            _FIXED_QI_INT8_BENCH_PROMPT,
            "--steps",
            str(_FIXED_QI_INT8_BENCH_STEPS),
            "--seed",
            str(_FIXED_QI_INT8_BENCH_SEED),
        ]
    )

    print("=" * 60)
    print("[*] Post-convert QI INT8 bench")
    print(f"    script: {bench_script}")
    print(f"    --fp16: {fp16_path}")
    print(f"    --int8:  {int8_path}")
    print(f"    --clip_path: {clip_path}")
    print(f"    --comfy_path: {comfy_path}")
    if vae_path:
        print(f"    --vae: {vae_path}")
    print(f"    --prompt: {_FIXED_QI_INT8_BENCH_PROMPT}")
    print(f"    --steps: {_FIXED_QI_INT8_BENCH_STEPS}")
    print(f"    --seed: {_FIXED_QI_INT8_BENCH_SEED} (fixed inside)")
    print("=" * 60)
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def convert_to_int8(
    input_path,
    output_path,
    per_channel_int8: bool = False,
    bias_correction: bool = False,
    calib_file: str | None = None,
    num_calib_samples: int = 32,
    num_inference_steps: int = 25,
    enable_convrot: bool = True,
    group_size: int = _DEFAULT_GROUPSIZE,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    act_mean_dict = {}
    comfyui_to_diffusers_map = {}
    compute_int8_bias_delta = None
    h_matrix = None
    rotate_weight = None
    rotate_weight_conv2d = None
    convrot_group_size_for_features = None
    build_hadamard = None
    convrot_linear = 0
    convrot_conv2d = 0

    if enable_convrot:
        nc = _load_native_convert_int8()
        rotate_weight = nc.rotate_weight
        rotate_weight_conv2d = nc.rotate_weight_conv2d
        convrot_group_size_for_features = nc.convrot_group_size_for_features
        build_hadamard = nc.build_hadamard
        print(
            f"  [ConvRot] FULL ON | Linear + Conv2d "
            f"(preferred groupsize={group_size}; adaptive power-of-4 divisor)"
        )
        if bias_correction:
            print(
                "  [ConvRot] WARN: Card 1 DualMonitor means are from unrotated float UNet; "
                "BC uses rotated W vs W_q (approximate for ConvRot)"
            )

    if bias_correction:
        if not calib_file:
            raise ValueError(
                "--bias_correction requires --calib_file "
                "(same as quantize_sdxl_hswq_v3.0.py)"
            )
        if not os.path.isfile(calib_file):
            raise FileNotFoundError(f"calib_file not found: {calib_file}")
        print(
            "  [Bias Correction Card 1] ON | ALL INT8 Linear+Conv | "
            "DualMonitor calib (steps=25) | "
            "mu_x = DualMonitor.channel_act_mean | "
            "bias += -(W_q - W) @ mu_x | "
            "no Approach A / no top_ratio gate"
        )
        calib = run_card1_calib(
            input_path=input_path,
            calib_file=calib_file,
            num_calib_samples=int(num_calib_samples),
            num_inference_steps=int(num_inference_steps),
            device=device,
        )
        act_mean_dict = calib["act_mean_dict"]
        comfyui_to_diffusers_map = calib["comfyui_to_diffusers_map"]
        compute_int8_bias_delta = calib["v30"].compute_int8_bias_delta
        print(
            f"  [Bias Correction] Captured act means for {len(act_mean_dict)} layers"
        )
    else:
        def compute_int8_bias_delta(weight_fp, weight_dq, act_mean):
            if act_mean is None:
                return None
            err = weight_dq.float() - weight_fp.float()
            mu = act_mean.float().to(device=err.device)
            if err.ndim == 2:
                if mu.numel() != err.shape[1]:
                    return None
                return err @ mu
            if err.ndim == 4:
                if mu.numel() != err.shape[1]:
                    return None
                return (err * mu.view(1, -1, 1, 1)).sum(dim=(1, 2, 3))
            return None

    print(f"Loading model: {input_path}")
    state_dict = load_file(input_path)

    new_state_dict = {}
    quant_meta_layers = {}
    converted_count = 0
    skipped_count = 0
    plain_int8_count = 0
    bias_corr_pending: dict[str, torch.Tensor] = {}
    bias_corr_applied = 0
    bias_corr_skipped_no_bias = 0
    bias_corr_skipped_no_act = 0
    bias_corr_skipped_bad_shape = 0
    mode = "per-channel" if per_channel_int8 else "tensorwise"
    rot_tag = " + ConvRot(Linear+Conv2d)" if enable_convrot else ""
    print(f"Converting UNet Linear/Conv weights to INT8 ({mode}{rot_tag}, amax/127)...")

    for key, tensor in tqdm(state_dict.items()):
        is_unet_matmul_weight = (
            (key.startswith("model.diffusion_model") or key.startswith("transformer_blocks") or key.startswith("img_in") or key.startswith("txt_in") or key.startswith("time_text_embed") or key.startswith("norm_out") or key.startswith("proj_out") or key.startswith("txt_norm"))
            and key.endswith(".weight")
            and tensor.ndim >= 2
        )
        if is_unet_matmul_weight and tensor.dtype in [
            torch.float16,
            torch.float32,
            torch.bfloat16,
        ]:
            if per_channel_int8:
                if tensor.ndim not in (2, 4):
                    new_state_dict[key] = tensor
                    skipped_count += 1
                    continue

            w_fp = tensor.float()
            used_gs = None
            if (
                enable_convrot
                and convrot_group_size_for_features is not None
                and build_hadamard is not None
            ):
                used_gs = convrot_group_size_for_features(int(w_fp.shape[1]), group_size)

            if used_gs is not None and tensor.ndim == 2 and rotate_weight is not None:
                h_matrix = build_hadamard(used_gs, device="cpu", dtype=torch.float32)
                w_fp = rotate_weight(w_fp, h_matrix, used_gs)
                q, scale = pack_channelwise(w_fp)
                quant_config = {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": int(used_gs),
                }
                convrot_linear += 1
            elif (
                used_gs is not None
                and tensor.ndim == 4
                and rotate_weight_conv2d is not None
            ):
                h_matrix = build_hadamard(used_gs, device="cpu", dtype=torch.float32)
                w_fp = rotate_weight_conv2d(w_fp, h_matrix, used_gs)
                q, scale = pack_channelwise(w_fp)
                quant_config = {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": int(used_gs),
                }
                convrot_conv2d += 1
            elif per_channel_int8:
                q, scale = pack_channelwise(w_fp)
                quant_config = {"format": "int8_tensorwise"}
                plain_int8_count += 1
            else:
                q, scale = pack_tensorwise(w_fp)
                quant_config = {"format": "int8_tensorwise"}
                plain_int8_count += 1
            weight_dq = q.float() * scale

            module_key = key[: -len(".weight")]
            new_state_dict[key] = q
            new_state_dict[f"{module_key}.weight_scale"] = scale
            new_state_dict[f"{module_key}.comfy_quant"] = _encode_comfy_quant(
                quant_config
            )
            quant_meta_layers[module_key] = dict(quant_config)
            converted_count += 1

            if bias_correction:
                diffusers_key = comfyui_to_diffusers_map.get(key)
                module_name = None
                if diffusers_key and diffusers_key.endswith(".weight"):
                    module_name = diffusers_key[:-7]
                act_mean = (
                    act_mean_dict.get(module_name)
                    if module_name is not None
                    else None
                )
                if act_mean is None:
                    bias_corr_skipped_no_act += 1
                else:
                    # BC target weight = pre-quant float (rotated when ConvRot).
                    delta = compute_int8_bias_delta(w_fp, weight_dq, act_mean)
                    if delta is None:
                        bias_corr_skipped_bad_shape += 1
                    else:
                        bias_corr_pending[module_key] = (
                            (-delta).detach().float().cpu()
                        )
        else:
            new_state_dict[key] = tensor
            skipped_count += 1

    if bias_correction and bias_corr_pending:
        print(
            f"\n[Bias Correction] Applying deltas to {len(bias_corr_pending)} "
            f"INT8 Linear+Conv layers (full Card 1)..."
        )
        for module_key, delta in bias_corr_pending.items():
            bias_key = f"{module_key}.bias"
            if bias_key not in new_state_dict:
                bias_corr_skipped_no_bias += 1
                continue
            bias = new_state_dict[bias_key]
            corrected = bias.float() + delta.to(
                device=bias.device, dtype=torch.float32
            )
            new_state_dict[bias_key] = corrected.to(dtype=bias.dtype)
            bias_corr_applied += 1
        print(
            f"  [Bias Correction] applied={bias_corr_applied}, "
            f"no_bias={bias_corr_skipped_no_bias}, "
            f"no_act={bias_corr_skipped_no_act}, "
            f"bad_shape={bias_corr_skipped_bad_shape}"
        )
    elif bias_correction:
        print(
            f"  [Bias Correction] No deltas pending "
            f"(no_act={bias_corr_skipped_no_act}, "
            f"bad_shape={bias_corr_skipped_bad_shape})"
        )

    metadata = {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": quant_meta_layers}
        )
    }

    print(f"Saving to: {output_path}")
    print(f"Converted layers: {converted_count}, Kept layers: {skipped_count}")
    print(f"Per-channel INT8 (Card 3): {per_channel_int8}")
    print(f"Bias correction (Card 1): {bias_correction}")
    if bias_correction:
        print(f"  Bias-corrected INT8 layers: {bias_corr_applied}")
    print(f"ConvRot FULL (Linear+Conv2d): {enable_convrot}")
    if enable_convrot:
        print(
            f"  ConvRot Linear: {convrot_linear}, ConvRot Conv2d: {convrot_conv2d}, "
            f"plain INT8 (no eligible group size): {plain_int8_count}"
        )
    else:
        print(f"  plain INT8: {plain_int8_count}")

    save_file(new_state_dict, output_path, metadata=metadata)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "UNet INT8 convert with FULL ConvRot (Linear+Conv2d) ON by default. "
            "Card 1 = --bias_correction. Card 3 = --per_channel_int8 for non-ConvRot "
            "plain packs. Use --no-convrot for plain INT8 only. No Approach A / no VETO."
        )
    )
    parser.add_argument(
        "--model",
        "--input",
        dest="model",
        type=str,
        required=True,
        help="Path to input .safetensors",
    )
    parser.add_argument("--output", type=str, required=True, help="Path to output .safetensors")
    parser.add_argument(
        "--per_channel_int8",
        action="store_true",
        help=(
            "Card 3: per-out-channel amax/scale (Linear (O,1), Conv (O,1,1,1)). "
            "Default is symmetric per-tensor. Format tag stays int8_tensorwise."
        ),
    )
    parser.add_argument(
        "--bias_correction",
        action="store_true",
        help=(
            "Card 1 ON: DualMonitor calib; bias += -(W_q - W) @ mu_x on ALL "
            "INT8 Linear+Conv (no top_ratio gate). Calib steps default 25. "
            "Requires --calib_file."
        ),
    )
    parser.add_argument(
        "--calib_file",
        type=str,
        default=None,
        help="Path to calibration prompts text file (required with --bias_correction)",
    )
    parser.add_argument(
        "--num_calib_samples",
        type=int,
        default=32,
        help="Calibration samples (recommended: 32)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="Denoising steps per calib sample (default 25)",
    )
    parser.add_argument(
        "--convrot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "FULL ConvRot on Linear+Conv2d when in_dim divisible by a power-of-4 "
            "group size: rotate + per-channel scale + stamp. Linear=kitchen online; "
            "Conv2d=HSWQ INT8 online act rotate. Default ON; pass --no-convrot for plain."
        ),
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=_DEFAULT_GROUPSIZE,
        help=f"ConvRot Hadamard group size (power of 4, default {_DEFAULT_GROUPSIZE})",
    )
    parser.add_argument(
        "--clip_path",
        type=str,
        default=None,
        help=(
            "Qwen3-4B text encoder path (required when --bench; "
            "no auto-discovery)"
        ),
    )
    parser.add_argument(
        "--comfy_path",
        type=str,
        default=None,
        help=(
            "ComfyUI root path (required when --bench; no auto-discovery)"
        ),
    )
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
        help=(
            "Optional VAE path for post-convert bench "
            "(forwarded as --vae when provided)"
        ),
    )
    parser.add_argument(
        "--bench",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After save, run benchmark/qi_int8_bench.py "
            "(requires --clip_path/--comfy_path; "
            "optional --vae; prompt/steps=12/seed=42 fixed inside). "
            "Pass --no-bench to skip."
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)
    if args.bias_correction and not args.calib_file:
        print("Error: --bias_correction requires --calib_file")
        sys.exit(1)
    if args.groupsize < 4 or (args.groupsize & (args.groupsize - 1)) != 0:
        print(f"Error: --groupsize must be a power of 4 (>=4), got {args.groupsize}")
        sys.exit(1)
    if math.log(args.groupsize, 4) % 1 != 0:
        print(f"Error: --groupsize must be a power of 4, got {args.groupsize}")
        sys.exit(1)

    if args.bench:
        missing = []
        if not args.clip_path:
            missing.append("--clip_path")
        if not args.comfy_path:
            missing.append("--comfy_path")
        if missing:
            print(
                "Error: post-convert bench requires "
                + ", ".join(missing)
                + " (pass --no-bench to convert without bench)"
            )
            sys.exit(1)
        if not os.path.isfile(args.clip_path):
            print(f"Error: --clip_path not found: {args.clip_path}")
            sys.exit(1)
        if not os.path.isdir(args.comfy_path):
            print(f"Error: --comfy_path not found: {args.comfy_path}")
            sys.exit(1)
        if args.vae and not os.path.isfile(args.vae):
            print(f"Error: --vae not found: {args.vae}")
            sys.exit(1)

    convert_to_int8(
        args.model,
        args.output,
        per_channel_int8=args.per_channel_int8,
        bias_correction=bool(args.bias_correction),
        calib_file=args.calib_file,
        num_calib_samples=args.num_calib_samples,
        num_inference_steps=args.num_inference_steps,
        enable_convrot=bool(args.convrot),
        group_size=int(args.groupsize),
    )

    if args.bench:
        bench_rc = run_post_convert_qi_int8_bench(
            script_dir=_REPO_ROOT,
            fp16_path=args.model,
            int8_path=args.output,
            clip_path=args.clip_path,
            comfy_path=args.comfy_path,
            vae_path=args.vae,
        )
        if bench_rc != 0:
            print(f"[FATAL] Post-convert bench exited with code {bench_rc}")
            sys.exit(bench_rc)
    else:
        print("[*] Post-convert bench skipped (--no-bench)")

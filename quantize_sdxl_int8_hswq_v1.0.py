"""Simple SDXL UNet INT8 convert + optional Card 1 Bias Correction.

Pack:
  default              symmetric per-tensor (amax / 127)
  --per_channel_int8   per-out-channel scale

Card 1 (--bias_correction): stays ON when the flag is set.
  DualMonitor hooks + StableDiffusionXLPipeline latent calib
  mu_x = DualMonitor.channel_act_mean; bias += -(W_q - W) @ mu_x
  Applies to every INT8 Linear and Conv2d (same formula as
  quantize_sdxl_hswq_v3.0.compute_int8_bias_delta).
  Calib: num_inference_steps default 25 (How-to); same pipeline call as V3.0
  (prompt + steps + latent + generator — no CFG/size override).
  No Static Profile VETO / no V4 FP16 keep in this script.
  Format stays int8_tensorwise.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def _load_hswq_v30():
    """Load quantize_sdxl_hswq_v3.0.py as a module (filename has a digit)."""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "quantize_sdxl_hswq_v3.0.py"
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"HSWQ V3.0 script not found: {path}")
    spec = importlib.util.spec_from_file_location("quantize_sdxl_hswq_v3_0", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["quantize_sdxl_hswq_v3_0"] = mod
    spec.loader.exec_module(mod)
    return mod


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


def run_v30_calib_and_v4(
    *,
    input_path: str,
    calib_file: str,
    num_calib_samples: int,
    num_inference_steps: int,
    bias_correction_top_ratio: float | None,
    profile_path: str | None,
    device: str,
):
    """Card 1 Bias Correction only: DualMonitor calib → channel_act_mean / sens.

    native_convert_int8_simple does NOT run Static Profile VETO, does NOT call
    derive_hswq_strategy_int8, and does NOT build V4 FP16 keep candidates.
    Those belong to quantize_sdxl_hswq_v3.0.py only.
    """
    v30 = _load_hswq_v30()

    # Card 1 quality path: default bc_top=1.0 (full BC). Autonomous top<1 dropped SSIM on V3.
    if profile_path and os.path.exists(profile_path):
        print(f"[*] Loading Analysis Data: {profile_path}")

    pipeline, _state_dict, comfyui_to_diffusers_map = v30.load_unet_from_safetensors(
        input_path, device
    )
    model = pipeline.unet

    bc_top = bias_correction_top_ratio
    if bc_top is None:
        bc_top = 1.0
        print(
            "  [bias_correction_top_ratio] default=1.0 "
            "(full Card 1 on all INT8 Linear+Conv; measured quality path)"
        )

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
    # Same How-to contract as quantize_sdxl_hswq_v3.0 (samples=32, steps=25).
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
    sens_dict = {}
    for name, mon in v30.dual_monitors.items():
        if mon.channel_act_mean is not None:
            act_mean_dict[name] = mon.channel_act_mean.detach().float().cpu()
        sens_dict[name] = float(mon.get_sensitivity())
    print(
        f"  [Card 1 DualMonitor] act_mean layers={len(act_mean_dict)} "
        f"sens layers={len(sens_dict)} (no VETO, no V4 FP16 cands)"
    )

    del pipeline
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "act_mean_dict": act_mean_dict,
        "sens_dict": sens_dict,
        "comfyui_to_diffusers_map": comfyui_to_diffusers_map,
        "bias_correction_top_ratio": float(bc_top),
        "v30": v30,
    }


def convert_to_int8(
    input_path,
    output_path,
    per_channel_int8: bool = False,
    bias_correction: bool = False,
    calib_file: str | None = None,
    num_calib_samples: int = 32,
    num_inference_steps: int = 25,
    bias_correction_top_ratio: float | None = None,
    profile: str | None = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    act_mean_dict = {}
    sens_dict = {}
    comfyui_to_diffusers_map = {}
    bc_allowed_modules = None  # None = all INT8 Linear+Conv modules
    compute_int8_bias_delta = None

    if bias_correction:
        if not calib_file:
            raise ValueError(
                "--bias_correction requires --calib_file "
                "(same as quantize_sdxl_hswq_v3.0.py)"
            )
        if not os.path.isfile(calib_file):
            raise FileNotFoundError(f"calib_file not found: {calib_file}")
        print(
            "  [Bias Correction Card 1] ON | all INT8 Linear+Conv | "
            "DualMonitor calib (steps=25) | "
            "mu_x = DualMonitor.channel_act_mean | "
            "bias += -(W_q - W) @ mu_x"
        )
        calib = run_v30_calib_and_v4(
            input_path=input_path,
            calib_file=calib_file,
            num_calib_samples=int(num_calib_samples),
            num_inference_steps=int(num_inference_steps),
            bias_correction_top_ratio=bias_correction_top_ratio,
            profile_path=profile,
            device=device,
        )
        act_mean_dict = calib["act_mean_dict"]
        sens_dict = calib["sens_dict"]
        comfyui_to_diffusers_map = calib["comfyui_to_diffusers_map"]
        top_ratio = float(calib["bias_correction_top_ratio"])
        compute_int8_bias_delta = calib["v30"].compute_int8_bias_delta
        print(
            f"  [Bias Correction] Captured act means for {len(act_mean_dict)} layers"
        )
    else:
        # Local copy of V3.0 formula (no calib).
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

    # Approach A scope: all INT8 Linear + Conv (ndim>=2). Default top_ratio=1.0.
    if bias_correction and act_mean_dict:
        int8_module_names = []
        for key, tensor in state_dict.items():
            is_unet_matmul_weight = (
                key.startswith("model.diffusion_model")
                and key.endswith(".weight")
                and tensor.ndim >= 2
            )
            if not is_unet_matmul_weight:
                continue
            diffusers_key = comfyui_to_diffusers_map.get(key)
            if diffusers_key and diffusers_key.endswith(".weight"):
                int8_module_names.append(diffusers_key[:-7])
        top_ratio = float(calib["bias_correction_top_ratio"])
        top_ratio = 0.0 if top_ratio < 0.0 else (1.0 if top_ratio > 1.0 else top_ratio)
        ranked = sorted(
            int8_module_names,
            key=lambda n: sens_dict.get(n, 0.0),
            reverse=True,
        )
        n_bc = int(len(ranked) * top_ratio + 1e-9)
        if top_ratio > 0.0 and n_bc < 1 and ranked:
            n_bc = 1
        if top_ratio >= 1.0:
            bc_allowed_modules = None
            print(
                f"  [Bias Correction] scope=ALL {len(ranked)} INT8 Linear+Conv "
                f"(top_ratio=1.0)."
            )
        else:
            bc_allowed_modules = set(ranked[:n_bc])
            print(
                f"  [Bias Correction] Approach A scope=top {n_bc}/{len(ranked)} "
                f"INT8 by DualMonitor sensitivity "
                f"(top_ratio={top_ratio:.3f})."
            )

    new_state_dict = {}
    quant_meta_layers = {}
    converted_count = 0
    skipped_count = 0
    bias_corr_pending: dict[str, torch.Tensor] = {}
    bias_corr_applied = 0
    bias_corr_skipped_no_bias = 0
    bias_corr_skipped_no_act = 0
    bias_corr_skipped_low_sens = 0
    bias_corr_skipped_bad_shape = 0
    mode = "per-channel" if per_channel_int8 else "tensorwise"
    print(f"Converting UNet Linear/Conv weights to INT8 ({mode}, amax/127)...")

    for key, tensor in tqdm(state_dict.items()):
        is_unet_matmul_weight = (
            key.startswith("model.diffusion_model")
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
                q, scale = pack_channelwise(tensor)
                weight_dq = q.float() * scale
            else:
                q, scale = pack_tensorwise(tensor)
                weight_dq = q.float() * scale

            module_key = key[: -len(".weight")]
            new_state_dict[key] = q
            new_state_dict[f"{module_key}.weight_scale"] = scale
            new_state_dict[f"{module_key}.comfy_quant"] = torch.tensor(
                list(json.dumps({"format": "int8_tensorwise"}).encode("utf-8")),
                dtype=torch.uint8,
            )
            quant_meta_layers[module_key] = {"format": "int8_tensorwise"}
            converted_count += 1

            if bias_correction:
                diffusers_key = comfyui_to_diffusers_map.get(key)
                module_name = None
                if diffusers_key and diffusers_key.endswith(".weight"):
                    module_name = diffusers_key[:-7]
                if (
                    bc_allowed_modules is not None
                    and module_name is not None
                    and module_name not in bc_allowed_modules
                ):
                    bias_corr_skipped_low_sens += 1
                else:
                    act_mean = (
                        act_mean_dict.get(module_name)
                        if module_name is not None
                        else None
                    )
                    if act_mean is None:
                        bias_corr_skipped_no_act += 1
                    else:
                        delta = compute_int8_bias_delta(
                            tensor, weight_dq, act_mean
                        )
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
            f"INT8 Linear+Conv layers..."
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
            f"low_sens={bias_corr_skipped_low_sens}, "
            f"bad_shape={bias_corr_skipped_bad_shape}"
        )
    elif bias_correction:
        print(
            f"  [Bias Correction] No deltas pending "
            f"(no_act={bias_corr_skipped_no_act}, "
            f"low_sens={bias_corr_skipped_low_sens}, "
            f"bad_shape={bias_corr_skipped_bad_shape})"
        )

    metadata = {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": quant_meta_layers}
        )
    }

    print(f"Saving to: {output_path}")
    print(f"Converted layers: {converted_count}, Kept layers: {skipped_count}")
    print(f"Per-channel INT8: {per_channel_int8}")
    print(f"Bias correction (Card 1): {bias_correction}")
    if bias_correction:
        print(f"  Bias-corrected INT8 layers: {bias_corr_applied}")

    save_file(new_state_dict, output_path, metadata=metadata)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Simple UNet INT8 convert. Optional Card 1 uses the SAME "
            "calibration formula and options as quantize_sdxl_hswq_v3.0.py."
        )
    )
    parser.add_argument("--model", type=str, required=True, help="Path to input .safetensors")
    parser.add_argument("--output", type=str, required=True, help="Path to output .safetensors")
    parser.add_argument(
        "--per_channel_int8",
        action="store_true",
        help=(
            "Per-out-channel amax/scale (Linear (O,1), Conv (O,1,1,1)). "
            "Default is symmetric per-tensor. Format tag stays int8_tensorwise."
        ),
    )
    parser.add_argument(
        "--bias_correction",
        action="store_true",
        help=(
            "Card 1 ON: DualMonitor calib; bias += -(W_q - W) @ mu_x on all "
            "INT8 Linear+Conv. Calib steps default 25. Requires --calib_file."
        ),
    )
    # --- Calibration options (Card 1); steps default 25 (How-to) ---
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
        "--bias_correction_top_ratio",
        type=float,
        default=None,
        help=(
            "Fraction of INT8 layers (by DualMonitor sensitivity) that receive "
            "Card 1. Default: None = 1.0 (full Card 1 / measured quality path)."
        ),
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Optional distribution profile JSON (unused when top_ratio=1.0)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)
    if args.bias_correction and not args.calib_file:
        print("Error: --bias_correction requires --calib_file (V3.0 same)")
        sys.exit(1)

    convert_to_int8(
        args.model,
        args.output,
        per_channel_int8=args.per_channel_int8,
        bias_correction=bool(args.bias_correction),
        calib_file=args.calib_file,
        num_calib_samples=args.num_calib_samples,
        num_inference_steps=args.num_inference_steps,
        bias_correction_top_ratio=args.bias_correction_top_ratio,
        profile=args.profile,
    )

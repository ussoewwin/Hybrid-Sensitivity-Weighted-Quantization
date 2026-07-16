"""SDXL UNet INT8 HSWQ V1.0 (simple pack + Card 1/3 + 300 MiB FP16 budget).

Pack:
  default              symmetric per-tensor (amax / 127)
  --per_channel_int8   per-out-channel scale (Card 3)

Card 1 (--bias_correction):
  DualMonitor hooks + StableDiffusionXLPipeline latent calib
  mu_x = DualMonitor.channel_act_mean; bias += -(W_q - W) @ mu_x
  ALWAYS applied to every INT8 Linear/Conv that has act_mean (full Card 1).
  No Approach A / no bias_correction_top_ratio gating (that dropped SSIM).

FP16 keep (owner hard frame):
  Exactly +300 MiB vs all-INT8. DualMonitor sensitivity x analyze severity x
  V4 estimated_mse rank; extreme fill under 300 MiB only truncates.
  Engine: quantize_sdxl_hswq_v3.0._apply_fp16_budget_cap (no redefinition).

Calib defaults: samples=32, steps=25.
Format tag stays int8_tensorwise.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import subprocess
import sys

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))


def _load_hswq_v30():
    """Load quantize_sdxl_hswq_v3.0.py as a module (filename has a digit)."""
    path = os.path.join(current_dir, "quantize_sdxl_hswq_v3.0.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"HSWQ V3.0 script not found: {path}")
    mod_name = "quantize_sdxl_hswq_v3_0_for_int8_v10"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
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


def _ensure_distribution_profile(
    *,
    input_path: str,
    profile_arg: str | None,
) -> tuple[str, dict, dict]:
    """Load (or run analyze to create) THIS-checkpoint distribution profile."""
    script_dir = current_dir
    analyze_script = os.path.join(script_dir, "analyze", "analyze_sdxl_distribution.py")
    input_abs = os.path.abspath(input_path)
    input_root = os.path.splitext(os.path.basename(input_path))[0]

    profile_path = profile_arg
    is_auto = False
    if not profile_path:
        profile_path = os.path.join(script_dir, f"{input_root}_distribution_profile.json")
        is_auto = True

    should_run = is_auto or not os.path.exists(profile_path)
    if should_run:
        if not os.path.isfile(analyze_script):
            raise FileNotFoundError(
                f"SDXL profile script not found: {analyze_script} "
                f"(required for 300 MiB FP16 budget)"
            )
        print("[*] Executing mandated distribution analysis:")
        print(f"    Script: {analyze_script}")
        print(f"    Input:  {input_abs}")
        print(f"    Result: {profile_path}")
        subprocess.run(
            [sys.executable, analyze_script, "--input", input_abs, "--output", profile_path],
            check=True,
        )

    if not os.path.isfile(profile_path):
        raise FileNotFoundError(f"distribution profile not found: {profile_path}")

    print(f"[*] Loading Analysis Data: {profile_path}")
    with open(profile_path, "r", encoding="utf-8") as f:
        profile_data = json.load(f)
    profile_summary: dict = {}
    if isinstance(profile_data, dict):
        profile_summary = profile_data.get("summary", {}) or {}
        model_profile = profile_data.get("layers", profile_data)
    else:
        model_profile = profile_data
    return profile_path, model_profile, profile_summary


def run_calib_and_fp16_budget(
    *,
    input_path: str,
    calib_file: str,
    num_calib_samples: int,
    num_inference_steps: int,
    device: str,
    fp16_budget_mb: float,
    profile: str | None,
):
    """DualMonitor calib + 300 MiB FP16 budget (v3.0 hard ceiling path).

    Structural / MAD / key-pattern VETO stacks are NOT added here —
    only profile derive Hard VETO + DualMonitor + V4 → budget fill.
    """
    v30 = _load_hswq_v30()
    budget_mb = v30._require_fp16_budget_mb_hard(float(fp16_budget_mb))

    _profile_path, model_profile, profile_summary = _ensure_distribution_profile(
        input_path=input_path,
        profile_arg=profile,
    )

    pipeline, _state_dict, comfyui_to_diffusers_map = v30.load_unet_from_safetensors(
        input_path, device
    )
    model_profile = v30._remap_profile_to_diffusers(
        model_profile, comfyui_to_diffusers_map
    )
    model = pipeline.unet
    _norm_profile = {k: v for k, v in model_profile.items() if isinstance(v, dict)}
    if not _norm_profile:
        raise ValueError(
            "FP16 300 MiB budget requires THIS-checkpoint layer profile "
            "(analyze/ --profile)."
        )

    print("Preparing calibration (Dual Monitor hooks; Card 1 + FP16 budget)...")
    v30.dual_monitors.clear()
    handles = []
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handle = module.register_forward_hook(
                lambda m, i, o, n=name: v30.hook_fn(m, i, o, n)
            )
            handles.append(handle)
            target_modules.append(name)

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

    dual_monitors = dict(v30.dual_monitors)

    act_mean_dict = {}
    for name, mon in dual_monitors.items():
        if mon.channel_act_mean is not None:
            act_mean_dict[name] = mon.channel_act_mean.detach().float().cpu()
    print(
        f"  [Card 1 DualMonitor] act_mean layers={len(act_mean_dict)} "
        f"(full Card 1; no Approach A)"
    )

    # Profile Hard VETO + auto alpha/beta (budget ranking needs these).
    # No structural / MAD / key-pattern stack (budget-only addition).
    veto_tunables = v30.resolve_veto_tunables(
        _norm_profile,
        profile_summary,
        dual_monitors=dual_monitors,
        fp16_budget_mb=budget_mb,
    )
    alpha, beta, _get_layer_search_low, hard_veto_layers = v30.derive_hswq_strategy_int8(
        model_profile,
        veto_tunables,
    )
    alpha = float(alpha)
    if alpha <= 0.0:
        raise ValueError(
            "INT8 Full-SVD×RMS alpha_auto must be > 0 after DualMonitor resolve "
            f"(alpha==0 is SVD cut / rebellion). got alpha_auto={alpha}"
        )
    beta = 1.0 - alpha
    print(
        f"  [Dynamic Alpha/Beta INT8 after DualMonitor] "
        f"alpha={alpha!r}, beta={beta!r} "
        f"(Imp×Sens×V4 MSE fill {budget_mb:g} MiB)"
    )
    print(f"  [Analyze Hard VETO seed] {len(hard_veto_layers)} layers")

    mse_cache: dict = {}
    dynamic_keep_layers, mse_cache = v30._build_v4_calib_fp16_candidates(
        model=model,
        dual_monitors=dual_monitors,
        target_modules=target_modules,
        hard_veto_layers=hard_veto_layers,
        mse_cache=mse_cache,
        alpha=alpha,
        beta=beta,
        device=device,
    )
    keep_layers = dynamic_keep_layers.union(hard_veto_layers)
    keep_before = len(keep_layers)
    veto_before = len(hard_veto_layers)
    keep_layers, hard_veto_layers, budget_stats = v30._apply_fp16_budget_cap(
        model=model,
        keep_layers=keep_layers,
        hard_veto_layers=hard_veto_layers,
        budget_mb=budget_mb,
        norm_profile=_norm_profile,
        veto_tunables=veto_tunables,
        dual_monitors=dual_monitors,
        mse_cache=mse_cache,
        alpha=alpha,
        beta=beta,
        device=device,
    )
    print(
        f"\n  [FP16 budget] hard_ceiling={budget_stats['budget_mb']:.1f} MiB "
        f"used={budget_stats['used_mb']:.1f} MiB "
        f"slack={budget_stats.get('slack_mb', 0):.2f} MiB "
        f"| keep {keep_before}→{budget_stats['kept']} "
        f"| VETO {veto_before}→{len(hard_veto_layers)} "
        f"| dropped={budget_stats['dropped']}"
    )
    print(f"  [FP16 budget] Final FP16 kept layers: {len(keep_layers)}")

    del pipeline
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "act_mean_dict": act_mean_dict,
        "comfyui_to_diffusers_map": comfyui_to_diffusers_map,
        "keep_layers": set(keep_layers),
        "budget_stats": budget_stats,
        "fp16_budget_mb": budget_mb,
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
    fp16_budget_mb: float = 300.0,
    profile: str | None = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not calib_file:
        raise ValueError(
            "300 MiB FP16 budget requires --calib_file "
            "(DualMonitor Sensitivity + V4 Importance)"
        )
    if not os.path.isfile(calib_file):
        raise FileNotFoundError(f"calib_file not found: {calib_file}")

    print(
        f"  [FP16 budget] ON | hard ceiling exactly "
        f"{float(fp16_budget_mb):g} MiB vs all-INT8"
    )
    if bias_correction:
        print(
            "  [Bias Correction Card 1] ON | ALL INT8 Linear+Conv | "
            "DualMonitor calib | "
            "mu_x = DualMonitor.channel_act_mean | "
            "bias += -(W_q - W) @ mu_x | "
            "no Approach A / no top_ratio gate"
        )

    calib = run_calib_and_fp16_budget(
        input_path=input_path,
        calib_file=calib_file,
        num_calib_samples=int(num_calib_samples),
        num_inference_steps=int(num_inference_steps),
        device=device,
        fp16_budget_mb=float(fp16_budget_mb),
        profile=profile,
    )
    act_mean_dict = calib["act_mean_dict"]
    comfyui_to_diffusers_map = calib["comfyui_to_diffusers_map"]
    keep_layers = calib["keep_layers"]
    budget_mb = float(calib["fp16_budget_mb"])
    compute_int8_bias_delta = calib["v30"].compute_int8_bias_delta
    print(f"  [Bias Correction] Captured act means for {len(act_mean_dict)} layers")

    print(f"Loading model: {input_path}")
    state_dict = load_file(input_path)

    new_state_dict = {}
    quant_meta_layers = {}
    converted_count = 0
    skipped_count = 0
    kept_fp16_count = 0
    bias_corr_pending: dict[str, torch.Tensor] = {}
    bias_corr_applied = 0
    bias_corr_skipped_no_bias = 0
    bias_corr_skipped_no_act = 0
    bias_corr_skipped_bad_shape = 0
    mode = "per-channel" if per_channel_int8 else "tensorwise"
    print(
        f"Converting UNet Linear/Conv weights to INT8 ({mode}, amax/127) | "
        f"FP16 keep={len(keep_layers)} under {budget_mb:g} MiB..."
    )

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
            module_key = key[: -len(".weight")]
            diffusers_key = comfyui_to_diffusers_map.get(key)
            module_name = None
            if diffusers_key and isinstance(diffusers_key, str) and diffusers_key.endswith(
                ".weight"
            ):
                module_name = diffusers_key[:-7]

            # 300 MiB budget: leave selected layers as FP16
            if module_name is not None and module_name in keep_layers:
                new_state_dict[key] = tensor.to(torch.float16)
                kept_fp16_count += 1
                continue

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

            new_state_dict[key] = q
            new_state_dict[f"{module_key}.weight_scale"] = scale
            new_state_dict[f"{module_key}.comfy_quant"] = torch.tensor(
                list(json.dumps({"format": "int8_tensorwise"}).encode("utf-8")),
                dtype=torch.uint8,
            )
            quant_meta_layers[module_key] = {"format": "int8_tensorwise"}
            converted_count += 1

            if bias_correction:
                act_mean = (
                    act_mean_dict.get(module_name)
                    if module_name is not None
                    else None
                )
                if act_mean is None:
                    bias_corr_skipped_no_act += 1
                else:
                    delta = compute_int8_bias_delta(tensor, weight_dq, act_mean)
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
            {
                "format_version": "1.0",
                "layers": quant_meta_layers,
                "fp16_budget_mb": budget_mb,
                "fp16_keep_layers": len(keep_layers),
            }
        )
    }

    print(f"Saving to: {output_path}")
    print(
        f"Converted layers: {converted_count}, "
        f"FP16 keep: {kept_fp16_count}, "
        f"Kept other: {skipped_count}"
    )
    print(f"Per-channel INT8 (Card 3): {per_channel_int8}")
    print(f"Bias correction (Card 1): {bias_correction}")
    print(f"FP16 budget: {budget_mb:g} MiB (hard ceiling)")
    if bias_correction:
        print(f"  Bias-corrected INT8 layers: {bias_corr_applied}")

    save_file(new_state_dict, output_path, metadata=metadata)
    print("Done!")


if __name__ == "__main__":
    v30_cli = _load_hswq_v30()
    budget_hard = float(v30_cli.FP16_BUDGET_MB_HARD)

    parser = argparse.ArgumentParser(
        description=(
            "SDXL INT8 HSWQ V1.0: simple pack + Card 1/3 + "
            f"exactly {budget_hard:g} MiB FP16 budget fill."
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
        "--fp16_budget_mb",
        type=float,
        default=budget_hard,
        help=(
            f"Owner hard ceiling for FP16 overhead vs all-INT8 "
            f"(must be exactly {budget_hard:g} MiB)."
        ),
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Path to distribution profile JSON (auto-generate if missing)",
    )
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
        required=True,
        help="Calibration prompts text (required for 300 MiB DualMonitor+V4 budget)",
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
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    try:
        args.fp16_budget_mb = v30_cli._require_fp16_budget_mb_hard(
            float(args.fp16_budget_mb)
        )
    except ValueError as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    convert_to_int8(
        args.model,
        args.output,
        per_channel_int8=args.per_channel_int8,
        bias_correction=bool(args.bias_correction),
        calib_file=args.calib_file,
        num_calib_samples=args.num_calib_samples,
        num_inference_steps=args.num_inference_steps,
        fp16_budget_mb=float(args.fp16_budget_mb),
        profile=args.profile,
    )

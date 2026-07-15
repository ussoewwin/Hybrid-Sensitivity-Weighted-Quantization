"""Independent SDXL UNet INT8 convert + Card 1 Bias Correction + Card 3 per-channel.

Fully self-contained. NO import / reference to quantize_sdxl_hswq_v3.0.py.

Pack:
  default              symmetric per-tensor (amax / 127)
  --per_channel_int8   per-out-channel scale (Card 3)

Card 1 (--bias_correction):
  diffusers pipeline calib; per-input-channel activation means.
  bias += -(W_q - W) @ mu_x  on ALL INT8 Linear+Conv (full Card 1; no top_ratio gate).
  Calib defaults: samples=32, steps=25.
  No Static Profile VETO / no FP16 keep.
  Format tag stays int8_tensorwise.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Pack (Card 3 / default)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Independent Card 1: pipeline loader + activation-mean hooks + bias delta
# ---------------------------------------------------------------------------
def _weight_fingerprint(t: torch.Tensor):
    """Value-based fingerprint for matching weights across format conversion.

    from_single_file renames keys but preserves Conv2d/Linear weight values,
    so identical values across formats => same fingerprint.
    """
    f = t.detach().float().reshape(-1)
    n = f.numel()
    if n == 0:
        return (0,)
    if n <= 16:
        return (
            n,
            float(f.sum().item()),
            float(f.abs().sum().item()),
            tuple(round(float(v), 6) for v in f.tolist()),
        )
    head = tuple(round(float(v), 6) for v in f[:8].tolist())
    mid = tuple(round(float(v), 6) for v in f[n // 2 - 2 : n // 2 + 2].tolist())
    tail = tuple(round(float(v), 6) for v in f[-8:].tolist())
    return (
        n,
        float(f.sum().item()),
        float(f.abs().sum().item()),
        head,
        mid,
        tail,
    )


def _load_pipeline(input_path: str, device: str):
    """Independent loader: diffusers from_single_file handles ComfyUI conversion.

    fp32 load is lossless regardless of original dtype (fp16/fp32/bf16), so the
    value fingerprints of loaded UNet weights match the original state_dict.

    Returns (pipeline, weight_fp_to_module) where weight_fp_to_module maps
    weight fingerprint -> diffusers module name for Conv2d/Linear.
    """
    from diffusers import StableDiffusionXLPipeline

    pipeline = StableDiffusionXLPipeline.from_single_file(
        input_path,
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=True,
    ).to(device)

    weight_fp_to_module = {}
    for name, module in pipeline.unet.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and module.weight is not None:
            weight_fp_to_module[_weight_fingerprint(module.weight.detach())] = name

    return pipeline, weight_fp_to_module


def _compute_int8_bias_delta(weight_fp, weight_dq, act_mean):
    """delta = (W_q - W) @ mu_x. None if shapes mismatch / mu missing.

    Linear (2D): err @ mu              -> (C_out,)
    Conv2d (4D): (err * mu.view) sum   -> (C_out,)
    mu_x is the per-input-channel mean of activations.
    """
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


def run_card1_calib(
    *,
    input_path: str,
    calib_file: str,
    num_calib_samples: int,
    num_inference_steps: int,
    device: str,
):
    """Independent Card 1 calibration via diffusers pipeline.

    Hooks Conv2d/Linear forward to accumulate per-input-channel activation means
    across calibration samples. No DualMonitor class, no v3.0 dependency.
    """
    pipeline, weight_fp_to_module = _load_pipeline(input_path, device)
    model = pipeline.unet

    print("Preparing calibration data...")
    with open(calib_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if len(prompts) < num_calib_samples:
        prompts = (prompts * (num_calib_samples // len(prompts) + 1))[:num_calib_samples]
    else:
        prompts = prompts[:num_calib_samples]

    # Per-module running sums of input-channel activations (keyed by diffusers name)
    act_sums = {}
    act_counts = {}
    handles = []

    def make_hook(module_name):
        def hook(_m, inp, _out):
            x = inp[0].detach().float()
            if x.dim() == 2:
                s = x.sum(dim=0)
                c = x.shape[0]
            elif x.dim() == 3:
                s = x.sum(dim=(0, 1))
                c = x.shape[0] * x.shape[1]
            elif x.dim() == 4:
                s = x.sum(dim=(0, 2, 3))
                c = x.shape[0] * x.shape[2] * x.shape[3]
            else:
                return
            if module_name in act_sums:
                act_sums[module_name] = act_sums[module_name] + s.cpu()
                act_counts[module_name] += c
            else:
                act_sums[module_name] = s.cpu()
                act_counts[module_name] = c

        return hook

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(module.register_forward_hook(make_hook(name)))

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
        print(f"\nSample {i + 1}/{num_calib_samples}: {prompt[:50]}...")
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

    act_mean_by_module = {}
    for name in act_sums:
        cnt = act_counts[name]
        if cnt > 0:
            act_mean_by_module[name] = (act_sums[name] / cnt).float()
    print(
        f"  [Card 1] act_mean modules={len(act_mean_by_module)} "
        f"(full Card 1; no VETO; no Approach A)"
    )

    del pipeline, model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "act_mean_by_module": act_mean_by_module,
        "weight_fp_to_module": weight_fp_to_module,
    }


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------
def convert_to_int8(
    input_path,
    output_path,
    per_channel_int8: bool = False,
    bias_correction: bool = False,
    calib_file: str | None = None,
    num_calib_samples: int = 32,
    num_inference_steps: int = 25,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    act_mean_by_module = {}
    weight_fp_to_module = {}

    if bias_correction:
        if not calib_file:
            raise ValueError(
                "--bias_correction requires --calib_file"
            )
        if not os.path.isfile(calib_file):
            raise FileNotFoundError(f"calib_file not found: {calib_file}")
        print(
            "  [Bias Correction Card 1] ON | ALL INT8 Linear+Conv | "
            "diffusers calib (steps=25) | "
            "mu_x = per-input-channel mean | "
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
        act_mean_by_module = calib["act_mean_by_module"]
        weight_fp_to_module = calib["weight_fp_to_module"]
        print(
            f"  [Bias Correction] Captured act means for {len(act_mean_by_module)} modules"
        )

    print(f"Loading model: {input_path}")
    state_dict = load_file(input_path)

    new_state_dict = {}
    quant_meta_layers = {}
    converted_count = 0
    skipped_count = 0
    bias_corr_pending: dict[str, torch.Tensor] = {}
    bias_corr_applied = 0
    bias_corr_skipped_no_bias = 0
    bias_corr_skipped_no_act = 0
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
                # Match this comfyui weight to its diffusers module by value
                fp = _weight_fingerprint(tensor)
                module_name = weight_fp_to_module.get(fp)
                act_mean = (
                    act_mean_by_module.get(module_name)
                    if module_name is not None
                    else None
                )
                if act_mean is None:
                    bias_corr_skipped_no_act += 1
                else:
                    delta = _compute_int8_bias_delta(tensor, weight_dq, act_mean)
                    if delta is None:
                        bias_corr_skipped_bad_shape += 1
                    else:
                        bias_corr_pending[module_key] = (-delta).detach().float().cpu()
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

    save_file(new_state_dict, output_path, metadata=metadata)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Independent Simple UNet INT8 convert. "
            "Card 1 = full BC on all INT8 Linear+Conv. "
            "Card 3 = --per_channel_int8. No Approach A / no VETO. "
            "No dependency on quantize_sdxl_hswq_v3.0.py."
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
            "Card 1 ON: per-input-channel act means; bias += -(W_q - W) @ mu_x on ALL "
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
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)
    if args.bias_correction and not args.calib_file:
        print("Error: --bias_correction requires --calib_file")
        sys.exit(1)

    convert_to_int8(
        args.model,
        args.output,
        per_channel_int8=args.per_channel_int8,
        bias_correction=bool(args.bias_correction),
        calib_file=args.calib_file,
        num_calib_samples=args.num_calib_samples,
        num_inference_steps=args.num_inference_steps,
    )

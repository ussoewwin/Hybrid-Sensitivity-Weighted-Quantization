"""Independent SDXL UNet INT8 convert: Card 3 pack + optional Card 1.

Fully self-contained. NO import / reference to quantize_sdxl_hswq_v3.0.py.

Scope:
  INT8 every UNet matmul weight with ndim >= 2 (Linear AND Conv2d), including
  attention to_q / to_k / to_v. Those layers have no .bias architecturally;
  weights are still INT8'd. Card 1 is skipped only where .bias is absent —
  NEVER leave QKV in FP16 as an "escape" (that claim is forbidden / wrong:
  native INT8's QKV and still clears ~0.97).

Pack (Card 3 — always):
  per-out-channel amax/127
  Linear scale (O, 1); Conv2d scale (O, 1, 1, 1)
  Format tag stays int8_tensorwise

Card 1 (--bias_correction; DEFAULT OFF):
  Requires --calib_file when enabled.
  SDXL pretrained pipeline + Comfy->Diffusers structural key map
  (native_int8_sdxl_unet_map; NO fingerprint; NO import of v3.0).
  Hooks Linear + Conv2d for per-input-channel mu_x.
  bias += -(W_q - W) contracted with mu_x on every INT8 module that HAS .bias.
  Full Card 1; no Approach A / no top_ratio gate.
  Calib defaults: samples=32, steps=25.
  No Static Profile VETO / no keep_ratio / no 300 MiB FP16 budget.
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

from native_int8_sdxl_unet_map import (
    detect_unet_config_from_keys,
    unet_to_diffusers_mapping,
)


# ---------------------------------------------------------------------------
# Pack (Card 3 — always)
# ---------------------------------------------------------------------------
def pack_channelwise(weight: torch.Tensor):
    """Per-out-channel INT8 (Card 3). Linear (O,1) or Conv2d (O,1,1,1).

    Broadcast-safe under kitchen dequantize_int8_simple (q * scale).
    A flat (O,) scale is NOT safe for 4D (right-aligned broadcast collision).
    """
    w = weight.float()
    if w.dim() not in (2, 4):
        raise ValueError(
            f"pack_channelwise supports ndim 2 or 4; got ndim={w.dim()}"
        )
    reduce_dims = tuple(range(1, w.dim()))
    amax = torch.clamp(w.abs().amax(dim=reduce_dims), min=1e-6)
    scale = amax / 127.0
    if w.dim() == 4:
        scale_view = scale.view(-1, 1, 1, 1)
        amax_view = amax.view(-1, 1, 1, 1)
    else:
        scale_view = scale.view(-1, 1)
        amax_view = amax.view(-1, 1)
    clamped = torch.clamp(w, -amax_view, amax_view)
    q = (clamped / scale_view).round().clamp(-127, 127).to(torch.int8)
    return q, scale_view.to(dtype=torch.float32)


# ---------------------------------------------------------------------------
# Independent Card 1: structural Comfy->Diffusers map + act-mean hooks
# ---------------------------------------------------------------------------
def _load_card1_pipeline(input_path: str, device: str):
    """Load SDXL pipeline and inject Comfy UNet weights via structural map.

    Fingerprint matching is forbidden: Comfy FP16 safetensors vs float cast
    misses most Linear lookups and explodes bias_corr_skipped_no_act.
    from_single_file is not used (CLIP loader breakage on this env).
    """
    from diffusers import StableDiffusionXLPipeline

    print(f"Loading model: {input_path}")
    state_dict = load_file(input_path)
    print("Detecting UNet structure...")
    unet_config = detect_unet_config_from_keys(state_dict)
    print(f"Detected UNet config: {unet_config}")
    print("Initializing Diffusers pipeline (pretrained base + remapped UNet)...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to(device)
    print("Building Comfy -> Diffusers key mapping...")
    comfyui_to_diffusers_map = unet_to_diffusers_mapping(unet_config, state_dict)
    new_state_dict = {}
    for comfy_key, diffusers_key in comfyui_to_diffusers_map.items():
        if comfy_key in state_dict:
            new_state_dict[diffusers_key] = state_dict[comfy_key]
    missing, unexpected = pipeline.unet.load_state_dict(new_state_dict, strict=False)
    print(
        f"  [Card 1] remapped tensors={len(new_state_dict)} "
        f"missing={len(missing)} unexpected={len(unexpected)} "
        f"map_entries={len(comfyui_to_diffusers_map)}"
    )
    return pipeline, comfyui_to_diffusers_map, state_dict


def _compute_int8_bias_delta(weight_fp, weight_dq, act_mean):
    """delta = (W_q - W) contracted with mu_x. None if shapes mismatch.

    Linear (O, I):              err @ mu -> (O,)
    Conv2d (O, I, K, K):        sum_{i,kh,kw} err[o,i,…] * mu[i] -> (O,)
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
    """Card 1 calibration via remapped Diffusers SDXL pipeline.

    Hooks Linear + Conv2d. Structural key map for Comfy -> Diffusers module
    names (no fingerprint).
    """
    pipeline, comfyui_to_diffusers_map, state_dict = _load_card1_pipeline(
        input_path, device
    )
    model = pipeline.unet

    print("Preparing calibration data...")
    with open(calib_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if len(prompts) < num_calib_samples:
        prompts = (prompts * (num_calib_samples // len(prompts) + 1))[
            :num_calib_samples
        ]
    else:
        prompts = prompts[:num_calib_samples]

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

    n_linear = 0
    n_conv = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(make_hook(name)))
            n_linear += 1
        elif isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(make_hook(name)))
            n_conv += 1
    print(f"  [Card 1] hooked Linear={n_linear} Conv2d={n_conv}")

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
        f"(full Card 1; Card 3 always ON; no VETO; no Approach A)"
    )

    del pipeline, model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "act_mean_by_module": act_mean_by_module,
        "comfyui_to_diffusers_map": comfyui_to_diffusers_map,
        "state_dict": state_dict,
    }


# ---------------------------------------------------------------------------
# Convert (Card 3 always; Card 1 optional, default OFF)
# ---------------------------------------------------------------------------
def convert_to_int8(
    input_path,
    output_path,
    *,
    bias_correction: bool = False,
    calib_file: str | None = None,
    num_calib_samples: int = 32,
    num_inference_steps: int = 25,
):
    """INT8 Linear+Conv with Card 3 pack; Card 1 only if bias_correction."""
    if bias_correction:
        if not calib_file:
            raise ValueError("--bias_correction requires --calib_file")
        if not os.path.isfile(calib_file):
            raise FileNotFoundError(f"calib_file not found: {calib_file}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"  Card 1={'ON' if bias_correction else 'OFF (default)'} | "
        "Card 3 ON | "
        "INT8 scope = Linear+Conv (ndim>=2), including QKV | "
        "QKV without .bias still INT8 (BC apply skipped only) | "
        "no Approach A / no top_ratio / no FP16 QKV escape"
    )

    act_mean_by_module: dict = {}
    comfyui_to_diffusers_map: dict = {}
    if bias_correction:
        calib = run_card1_calib(
            input_path=input_path,
            calib_file=calib_file,
            num_calib_samples=int(num_calib_samples),
            num_inference_steps=int(num_inference_steps),
            device=device,
        )
        act_mean_by_module = calib["act_mean_by_module"]
        comfyui_to_diffusers_map = calib["comfyui_to_diffusers_map"]
        state_dict = calib["state_dict"]
        print(
            f"  [Bias Correction] Captured act means for "
            f"{len(act_mean_by_module)} modules"
        )
    else:
        print(f"Loading model: {input_path}")
        state_dict = load_file(input_path)

    new_state_dict = {}
    quant_meta_layers = {}
    converted_count = 0
    skipped_count = 0
    bias_corr_pending: dict[str, torch.Tensor] = {}
    bias_corr_applied = 0
    bias_corr_skipped_no_bias = 0
    bias_corr_skipped_no_map = 0
    bias_corr_skipped_no_act = 0
    bias_corr_skipped_bad_shape = 0

    print(
        "Converting UNet Linear+Conv (ndim>=2) to INT8 "
        "(Card 3 per-channel amax/127)..."
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
            q, scale = pack_channelwise(tensor)
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
                    module_name = diffusers_key[: -len(".weight")]
                if module_name is None:
                    bias_corr_skipped_no_map += 1
                else:
                    act_mean = act_mean_by_module.get(module_name)
                    if act_mean is None:
                        bias_corr_skipped_no_act += 1
                    else:
                        delta = _compute_int8_bias_delta(tensor, weight_dq, act_mean)
                        if delta is None:
                            bias_corr_skipped_bad_shape += 1
                        else:
                            bias_corr_pending[module_key] = (
                                (-delta).detach().float().cpu()
                            )
        else:
            new_state_dict[key] = tensor
            skipped_count += 1

    if bias_correction:
        print(
            f"\n[Bias Correction] Applying deltas to {len(bias_corr_pending)} "
            f"INT8 modules (full Card 1; skip when .bias absent e.g. QKV)..."
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
            f"no_map={bias_corr_skipped_no_map}, "
            f"no_act={bias_corr_skipped_no_act}, "
            f"bad_shape={bias_corr_skipped_bad_shape}"
        )
    else:
        print("\n[Bias Correction] Disabled (default; pass --bias_correction).")

    metadata = {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": quant_meta_layers}
        )
    }

    print(f"Saving to: {output_path}")
    print(f"Converted layers: {converted_count}, Kept layers: {skipped_count}")
    print("Per-channel INT8 (Card 3): True (always)")
    print(f"Bias correction (Card 1): {bias_correction}")
    if bias_correction:
        print(f"  Bias-corrected INT8 layers: {bias_corr_applied}")

    save_file(new_state_dict, output_path, metadata=metadata)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Independent Simple UNet INT8: Card 3 always ON; "
            "Card 1 DEFAULT OFF (--bias_correction to enable). "
            "INT8 scope = Linear+Conv (ndim>=2), including QKV. "
            "No FP16 QKV escape. No Approach A / no VETO. "
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
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output .safetensors"
    )
    parser.add_argument(
        "--bias_correction",
        action="store_true",
        default=False,
        help="Enable Card 1 bias correction (DEFAULT OFF)",
    )
    parser.add_argument(
        "--calib_file",
        type=str,
        default=None,
        help="Calibration prompts text (required with --bias_correction)",
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
        bias_correction=bool(args.bias_correction),
        calib_file=args.calib_file,
        num_calib_samples=args.num_calib_samples,
        num_inference_steps=args.num_inference_steps,
    )

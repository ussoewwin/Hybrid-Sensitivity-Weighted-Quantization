"""SD1.5 INT8 converter - reworked from the SDXL copy (native_convert_int8_sd1.5.py).

Differences vs the SDXL base (measured on chilledReGenericV3_v10.safetensors):
  Card 1 calib loads a full SD1.5 checkpoint (UNet + CLIP + VAE in one file)
  through diffusers StableDiffusionPipeline.from_single_file instead of the
  SDXL pipeline; the v30 unet_to_diffusers_mapping is verified to cover all
  391 SD1.5 UNet matmul weights with 0 missing / 0 unexpected on
  UNet2DConditionModel(320,640,1280,1280, cross_attention_dim=768).
  Post-convert bench hook is removed: benchmark/ has no SD1.5 traj comparator
  (sdxl_int8_traj_compare.py is SDXL-only), so there is nothing honest to
  chain here.

Everything else - pack structure (int8_tensorwise + weight_scale + comfy_quant),
FULL ConvRot default (Linear gs256/gs64 + Conv2d gs256/gs64/gs4), Card 3
per-channel plain packs, Card 1 delta math (exact same compute_int8_bias_delta),
no nibble packing (all shapes preserved, so SD1.5 architecture detection reads
input_blocks.0.0 / out.2 / attn2.to_k / proj_in shapes verbatim) - stays as in
the SDXL converter.

Embedded VAE removal:
  The checkpoint-embedded VAE (first_stage_model.*) is dropped from the output
  pack to maximize file size reduction (saves ~160-330 MB). Decode with a
  separate SD1.5 VAE at inference time.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import sys

os.environ["_CHECK_PEFT"] = "0"

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

_DEFAULT_GROUPSIZE = 256

# FP16-keep candidate list, ordered by measured static requant error.
# Method: for every quantized layer run THIS converter's exact pipeline
# offline (rotate fp32 -> per-channel amax/127 quantize -> dequantize) and
# take the Frobenius err_norm on the ROTATED basis (the quantity INT8
# destroys). Data source: sd15_fpkeep_rank.py full 282-layer sweep on
# chilledReGenericV3_v10 (2026-10-12). rel = err / ||W_rot||.
_FP_KEEP_CANDIDATES_ORDERED = (
    "model.diffusion_model.output_blocks.4.0.in_layers.2",   # err=2.86 rel=0.01398 conv gs256 29.49M
    "model.diffusion_model.output_blocks.6.0.in_layers.2",   # err=2.65 rel=0.01648 conv gs64  11.06M
    "model.diffusion_model.output_blocks.3.0.in_layers.2",   # err=2.64 rel=0.01385 conv gs256 29.49M
    "model.diffusion_model.output_blocks.5.0.in_layers.2",   # err=2.39 rel=0.01253 conv gs64  22.12M
    "model.diffusion_model.output_blocks.4.0.out_layers.3",  # err=2.10 rel=0.01194 conv gs256 14.75M
    "model.diffusion_model.output_blocks.7.0.in_layers.2",   # err=2.09 rel=0.01605 conv gs256  7.37M
    "model.diffusion_model.output_blocks.3.0.out_layers.3",  # err=2.06 rel=0.01183 conv gs256 14.75M
    "model.diffusion_model.output_blocks.5.0.out_layers.3",  # err=1.91 rel=0.01019 conv gs256 14.75M
    "model.diffusion_model.output_blocks.6.0.out_layers.3",  # err=1.89 rel=0.01442 conv gs64   3.69M
    "model.diffusion_model.output_blocks.2.0.in_layers.2",   # err=1.84 rel=0.01169 conv gs256 29.49M
    "model.diffusion_model.output_blocks.8.0.in_layers.2",   # err=1.74 rel=0.01501 conv gs64   5.53M
    "model.diffusion_model.input_blocks.8.0.in_layers.2",    # err=1.73 rel=0.01121 conv gs256 14.75M
)


def _load_hswq_v30():
    """Load archives/quantize_sdxl_hswq_v3.0.py as a module (filename has a digit).

    Only the key mapping + DualMonitor + delta math are reused; the SDXL
    pipeline loader inside it is NOT used for SD1.5.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, "archives", "quantize_sdxl_hswq_v3.0.py")
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
    """Load repo-root native_convert_int8.py for Hadamard / rotate_weight."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, "native_convert_int8.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"native_convert_int8.py not found: {path}")
    name = "native_convert_int8_for_sd15"
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
    """Card 1 only: DualMonitor calib on the SD1.5 UNet -> channel_act_mean.

    Loads the FULL SD1.5 checkpoint (UNet + CLIP-L + VAE live in one file) via
    diffusers StableDiffusionPipeline.from_single_file. The v30 module supplies
    only detect_unet_config_from_keys / unet_to_diffusers_mapping (verified:
    391/391 SD1.5 UNet weights mapped, 0 missing / 0 unexpected) and
    compute_int8_bias_delta. No Static Profile VETO / no V4 FP16 keep.
    """
    from diffusers import StableDiffusionPipeline

    v30 = _load_hswq_v30()

    print(f"Loading SD1.5 pipeline (single-file checkpoint): {input_path}")
    pipeline = StableDiffusionPipeline.from_single_file(
        input_path,
        torch_dtype=torch.float16,
        load_safety_checker=False,
    ).to(device)
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

    # diffusers module name -> comfy key (mapping direction: comfy -> diffusers,
    # so BC lookup needs comfy_key -> diffusers module name; invert once here).
    state_dict = load_file(input_path)
    unet_config = v30.detect_unet_config_from_keys(state_dict)
    mapping = v30.unet_to_diffusers_mapping(unet_config, state_dict)
    comfy_to_diffusers = {}
    for comfy_key, diffusers_key in mapping.items():
        comfy_to_diffusers[comfy_key] = diffusers_key
    del state_dict

    return {
        "act_mean_dict": act_mean_dict,
        "comfy_to_diffusers": comfy_to_diffusers,
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
    enable_convrot: bool = True,
    group_size: int = _DEFAULT_GROUPSIZE,
    fp_keep_layers: int = 0,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    act_mean_dict = {}
    comfy_to_diffusers = {}
    compute_int8_bias_delta = None
    h_matrix = None
    rotate_weight = None
    rotate_weight_conv2d = None
    convrot_group_size_for_features = None
    build_hadamard = None
    convrot_linear = 0
    convrot_conv2d = 0
    fp_keep_set = set(_FP_KEEP_CANDIDATES_ORDERED[:fp_keep_layers])
    if fp_keep_set:
        print(
            f"  [FP16 keep] ON | top-{len(fp_keep_set)} layers by measured static "
            f"requant err_norm stay original dtype (no quant, no stamp)"
        )
    fp_keep_count = 0

    if enable_convrot:
        nc = _load_native_convert_int8()
        rotate_weight = nc.rotate_weight
        rotate_weight_conv2d = nc.rotate_weight_conv2d
        convrot_group_size_for_features = nc.convrot_group_size_for_features
        build_hadamard = nc.build_hadamard
        print(
            f"  [ConvRot] FULL ON | Linear + Conv2d "
            f"(preferred groupsize={group_size}; adaptive power-of-4 divisor; "
            f"SD1.5 measured routes gs256/gs64 Linear, gs256/gs64/gs4 Conv2d)"
        )
        if bias_correction:
            print(
                "  [ConvRot] WARN: Card 1 DualMonitor means are from unrotated float UNet; "
                "BC uses rotated W vs W_q (approximate for ConvRot)"
            )

    if bias_correction:
        if not calib_file:
            raise ValueError(
                "--bias_correction requires --calib_file (one prompt per line)"
            )
        if not os.path.isfile(calib_file):
            raise FileNotFoundError(f"calib_file not found: {calib_file}")
        print(
            "  [Bias Correction Card 1] ON | ALL INT8 Linear+Conv | "
            "SD1.5 diffusers pipeline calib (steps=25) | "
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
        comfy_to_diffusers = calib["comfy_to_diffusers"]
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
    dropped_vae_count = 0
    mode = "per-channel" if per_channel_int8 else "tensorwise"
    rot_tag = " + ConvRot(Linear+Conv2d)" if enable_convrot else ""
    print(f"Converting SD1.5 UNet Linear/Conv weights to INT8 ({mode}{rot_tag}, amax/127)...")

    for key, tensor in tqdm(state_dict.items()):
        if key.startswith("first_stage_model."):
            dropped_vae_count += 1
            continue
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
            module_key_pre = key[: -len(".weight")]
            if fp_keep_layers and module_key_pre in fp_keep_set:
                # FP16 keep: original dtype, no rotation, no stamp, no scale.
                new_state_dict[key] = tensor
                skipped_count += 1
                fp_keep_count += 1
                continue
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
                # DualMonitor means are keyed by diffusers module name; the
                # verified v30 mapping (comfy -> diffusers) resolves it.
                diffusers_key = comfy_to_diffusers.get(key)
                module_name = None
                if diffusers_key and diffusers_key.endswith(".weight"):
                    module_name = diffusers_key[: -len(".weight")]
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
    if fp_keep_layers:
        print(
            f"  FP16-kept candidates: {fp_keep_count}/{len(fp_keep_set)} "
            f"(rest quantized)"
        )
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

    # Drop embedded VAE (first_stage_model.*) to reduce checkpoint size.
    # VAE is loaded separately at inference time.
    vae_keys = [k for k in new_state_dict if k.startswith("first_stage_model.")]
    for vk in vae_keys:
        del new_state_dict[vk]
        dropped_vae_count += 1
    if dropped_vae_count:
        print(
            f"Dropped {dropped_vae_count} embedded VAE tensors (first_stage_model.*) "
            f"to minimize file size."
        )

    save_file(new_state_dict, output_path, metadata=metadata)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Saved: {output_path}  {size_mb:.2f} MB (decimal)")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "SD1.5 UNet INT8 convert with FULL ConvRot (Linear+Conv2d) ON by default. "
            "Card 1 = --bias_correction (diffusers SD1.5 pipeline calib). "
            "Card 3 = --per_channel_int8 for non-ConvRot plain packs. "
            "Use --no-convrot for plain INT8 only. No Approach A / no VETO. "
            "No post-convert bench (benchmark/ has no SD1.5 traj comparator)."
        )
    )
    parser.add_argument(
        "--model",
        "--input",
        dest="model",
        type=str,
        required=True,
        help="Path to input .safetensors (full SD1.5 checkpoint or UNet-only)",
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
            "Card 1 ON: DualMonitor calib via diffusers StableDiffusionPipeline; "
            "bias += -(W_q - W) @ mu_x on ALL INT8 Linear+Conv (no top_ratio gate). "
            "Calib defaults samples=32 steps=25. Requires --calib_file."
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
        "--fp_keep_layers",
        type=int,
        default=0,
        help=(
            "FP16 protect top-N layers (N=0..12) from the static err_norm "
            "ranking: they stay original dtype (no quant, no stamp). "
            "N=10 = the 10 highest-err layers (output_blocks ResBlock convs, "
            "+337.5 MiB vs full-INT8 pack)."
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)
    if args.fp_keep_layers < 0 or args.fp_keep_layers > len(_FP_KEEP_CANDIDATES_ORDERED):
        print(
            f"Error: --fp_keep_layers must be 0..{len(_FP_KEEP_CANDIDATES_ORDERED)}, "
            f"got {args.fp_keep_layers}"
        )
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
        fp_keep_layers=int(args.fp_keep_layers),
    )

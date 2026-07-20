"""UNet / DiT ConvRot W4A4 (signed INT4) converter for native ComfyUI Load Diffusion Model.

On-disk format (matches comfy_kitchen TensorCoreConvRotW4A4Layout + RDNA4 native save):
  Linear W4A4:  .weight        int8 [N, K//2]  (two signed int4 nibbles / byte; low = even col)
                .weight_scale  f32  [N]        (per-row absmax / 7)
                .comfy_quant   uint8 JSON      {"format":"convrot_w4a4","convrot_groupsize":G}

Requires (same contract as comfy_kitchen validate_w4a4_shape):
  - 2D Linear weight
  - K even, K divisible by quant_group_size 64, and by a power-of-4 ConvRot group size

Fallback (same idea as RDNA4 mixed int4/int8, WITHOUT HIP/WMMA):
  - Conv2d, or Linear that cannot meet W4A4 shape: INT8 pack + int8_tensorwise
    (+ ConvRot stamp when group size is eligible), same as native_convert_int8_convrot.py

Not ported from D:\\ComfyUI-INT4-ConvRot-RDNA4:
  - HIP / hipcc / gfx12 WMMA kernels, fused LDS rotate, ROCm device checks

Optional Card 1 (--bias_correction): DualMonitor act means; bias += -(W_q - W) @ mu_x
  on quantized Linear/Conv (dequant uses unpacked INT4 or INT8 * scale).
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import sys

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

_DEFAULT_GROUPSIZE = 256
_INT4_MAX = 7
_INT4_QUANT_GROUP_SIZE = 64


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


def _load_native_convert_int8():
    """Load sibling native_convert_int8.py for Hadamard / rotate_weight."""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "native_convert_int8.py"
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"native_convert_int8.py not found: {path}")
    name = "native_convert_int8_for_int4_convrot"
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


def _pack_int4_row_major(values: torch.Tensor) -> torch.Tensor:
    """Pack (..., K) int4 values into (..., K//2) int8 (low nibble = even column).

    Same storage codec as comfy_kitchen.backends.eager.svdquant._pack_int4_row_major.
    """
    if values.shape[-1] % 2 != 0:
        raise ValueError(f"last dim must be even, got {values.shape[-1]}")
    lo = values[..., 0::2].to(torch.int32) & 0x0F
    hi = values[..., 1::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.int8)


def _unpack_int4_row_major(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of _pack_int4_row_major (signed nibble)."""
    x32 = packed.to(torch.int32)
    lo = x32 & 0x0F
    hi = (x32 >> 4) & 0x0F
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    stacked = torch.stack([lo, hi], dim=-1)
    return stacked.reshape(*packed.shape[:-1], -1).to(torch.int8)


def pack_signed_int4_rowwise(weight: torch.Tensor):
    """Symmetric per-row INT4: scale = absmax / 7, clamp [-7, 7], then nibble-pack.

    Returns qdata int8 [N, K//2], scale f32 [N].
    """
    if weight.ndim != 2:
        raise ValueError(f"INT4 pack expects 2D weight, got ndim={weight.ndim}")
    w = weight.float()
    rows, cols = w.shape
    if cols % 2 != 0:
        raise ValueError(f"INT4 pack requires even K, got {cols}")
    absmax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = absmax / float(_INT4_MAX)
    q = (w / scale).round().clamp(-_INT4_MAX, _INT4_MAX).to(torch.int8)
    return _pack_int4_row_major(q), scale.reshape(rows).to(torch.float32)


def dequant_int4_rowwise(qdata: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Unpack signed INT4 and apply per-row scale → float [N, K]."""
    w_int = _unpack_int4_row_major(qdata).float()
    return w_int * scale.reshape(-1, 1).to(dtype=w_int.dtype, device=w_int.device)


def can_pack_w4a4(in_features: int, used_gs: int | None) -> bool:
    """True when kitchen / ConvRot W4A4 shape checks pass."""
    if used_gs is None:
        return False
    k = int(in_features)
    if k % 2 != 0:
        return False
    if k % _INT4_QUANT_GROUP_SIZE != 0:
        return False
    if k % int(used_gs) != 0:
        return False
    return True


def pack_tensorwise_int8(weight: torch.Tensor):
    """Symmetric per-tensor INT8: scale = amax / 127."""
    w = weight.float()
    amax = max(float(w.abs().max().item()), 1e-6)
    scale = amax / 127.0
    q = (w / scale).round().clamp(-127, 127).to(torch.int8)
    return q, torch.tensor(scale, dtype=torch.float32)


def pack_channelwise_int8(weight: torch.Tensor):
    """Per-out-channel INT8 (Card 3 / ConvRot INT8 fallback)."""
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
        raise ValueError(f"unsupported weight ndim={w.dim()} for INT8 channel pack")
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
    """Card 1 only: DualMonitor calib → channel_act_mean."""
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


def convert_to_int4_convrot(
    input_path,
    output_path,
    per_channel_int8: bool = False,
    bias_correction: bool = False,
    calib_file: str | None = None,
    num_calib_samples: int = 32,
    num_inference_steps: int = 25,
    enable_convrot: bool = True,
    group_size: int = _DEFAULT_GROUPSIZE,
    min_in_features: int = 0,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    act_mean_dict = {}
    comfyui_to_diffusers_map = {}
    compute_int8_bias_delta = None
    rotate_weight = None
    rotate_weight_conv2d = None
    convrot_group_size_for_features = None
    build_hadamard = None
    convrot_w4a4 = 0
    convrot_int8_linear = 0
    convrot_int8_conv2d = 0
    plain_int8_count = 0
    skipped_small = 0

    if enable_convrot:
        nc = _load_native_convert_int8()
        rotate_weight = nc.rotate_weight
        rotate_weight_conv2d = nc.rotate_weight_conv2d
        convrot_group_size_for_features = nc.convrot_group_size_for_features
        build_hadamard = nc.build_hadamard
        print(
            f"  [ConvRot W4A4] ON | preferred groupsize={group_size}; "
            f"quant_group={_INT4_QUANT_GROUP_SIZE}; "
            f"min_in_features={min_in_features}"
        )
        print(
            "  [ConvRot W4A4] Linear → signed INT4 nibble pack when shape OK; "
            "else INT8 fallback (Linear/Conv2d). No HIP / RDNA4 kernels."
        )
        if bias_correction:
            print(
                "  [ConvRot] WARN: Card 1 DualMonitor means are from unrotated float UNet; "
                "BC uses rotated W vs W_q (approximate for ConvRot)"
            )
    else:
        print("  [ConvRot W4A4] OFF | packing plain INT8 only (no offline rotate)")

    if bias_correction:
        if not calib_file:
            raise ValueError(
                "--bias_correction requires --calib_file "
                "(same as quantize_sdxl_hswq_v3.0.py)"
            )
        if not os.path.isfile(calib_file):
            raise FileNotFoundError(f"calib_file not found: {calib_file}")
        print(
            "  [Bias Correction Card 1] ON | quantized Linear+Conv | "
            "DualMonitor calib | bias += -(W_q - W) @ mu_x"
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
    bias_corr_pending: dict[str, torch.Tensor] = {}
    bias_corr_applied = 0
    bias_corr_skipped_no_bias = 0
    bias_corr_skipped_no_act = 0
    bias_corr_skipped_bad_shape = 0
    rot_tag = " + ConvRot W4A4/INT8" if enable_convrot else ""
    print(f"Converting diffusion Linear/Conv weights ({rot_tag.strip() or 'plain INT8'})...")

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
            if tensor.ndim not in (2, 4):
                new_state_dict[key] = tensor
                skipped_count += 1
                continue

            in_f = int(tensor.shape[1])
            if min_in_features > 0 and in_f < int(min_in_features):
                new_state_dict[key] = tensor
                skipped_small += 1
                skipped_count += 1
                continue

            w_fp = tensor.float()
            used_gs = None
            if (
                enable_convrot
                and convrot_group_size_for_features is not None
                and build_hadamard is not None
            ):
                used_gs = convrot_group_size_for_features(in_f, group_size)

            mode = None  # "w4a4" | "int8"
            quant_config: dict
            q: torch.Tensor
            scale: torch.Tensor

            if (
                enable_convrot
                and tensor.ndim == 2
                and can_pack_w4a4(in_f, used_gs)
                and rotate_weight is not None
                and build_hadamard is not None
            ):
                h_matrix = build_hadamard(int(used_gs), device="cpu", dtype=torch.float32)
                w_fp = rotate_weight(w_fp, h_matrix, int(used_gs))
                q, scale = pack_signed_int4_rowwise(w_fp)
                quant_config = {
                    "format": "convrot_w4a4",
                    "convrot_groupsize": int(used_gs),
                }
                mode = "w4a4"
                convrot_w4a4 += 1
            elif (
                enable_convrot
                and used_gs is not None
                and tensor.ndim == 2
                and rotate_weight is not None
            ):
                h_matrix = build_hadamard(int(used_gs), device="cpu", dtype=torch.float32)
                w_fp = rotate_weight(w_fp, h_matrix, int(used_gs))
                q, scale = pack_channelwise_int8(w_fp)
                quant_config = {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": int(used_gs),
                }
                mode = "int8"
                convrot_int8_linear += 1
            elif (
                enable_convrot
                and used_gs is not None
                and tensor.ndim == 4
                and rotate_weight_conv2d is not None
            ):
                h_matrix = build_hadamard(int(used_gs), device="cpu", dtype=torch.float32)
                w_fp = rotate_weight_conv2d(w_fp, h_matrix, int(used_gs))
                q, scale = pack_channelwise_int8(w_fp)
                quant_config = {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": int(used_gs),
                }
                mode = "int8"
                convrot_int8_conv2d += 1
            elif per_channel_int8:
                q, scale = pack_channelwise_int8(w_fp)
                quant_config = {"format": "int8_tensorwise"}
                mode = "int8"
                plain_int8_count += 1
            else:
                q, scale = pack_tensorwise_int8(w_fp)
                quant_config = {"format": "int8_tensorwise"}
                mode = "int8"
                plain_int8_count += 1

            if mode == "w4a4":
                weight_dq = dequant_int4_rowwise(q, scale)
            else:
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
            f"quantized Linear+Conv layers..."
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
    if skipped_small:
        print(f"  skipped (min_in_features={min_in_features}): {skipped_small}")
    print(f"Per-channel INT8 fallback flag: {per_channel_int8}")
    print(f"Bias correction (Card 1): {bias_correction}")
    if bias_correction:
        print(f"  Bias-corrected layers: {bias_corr_applied}")
    print(f"ConvRot enabled: {enable_convrot}")
    if enable_convrot:
        print(
            f"  W4A4 Linear: {convrot_w4a4}, "
            f"INT8 ConvRot Linear: {convrot_int8_linear}, "
            f"INT8 ConvRot Conv2d: {convrot_int8_conv2d}, "
            f"plain INT8: {plain_int8_count}"
        )
    else:
        print(f"  plain INT8: {plain_int8_count}")

    save_file(new_state_dict, output_path, metadata=metadata)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Diffusion Linear ConvRot W4A4 (signed INT4) convert for native ComfyUI. "
            "Ineligible Linear/Conv2d fall back to INT8 (+ ConvRot when possible). "
            "No AMD HIP kernels. Card 1 = --bias_correction."
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
        "--per_channel_int8",
        action="store_true",
        help=(
            "For INT8 fallback packs only: per-out-channel amax/scale. "
            "W4A4 path always uses per-row scale [N]."
        ),
    )
    parser.add_argument(
        "--bias_correction",
        action="store_true",
        help=(
            "Card 1 ON: DualMonitor calib; bias += -(W_q - W) @ mu_x. "
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
            "ConvRot W4A4 on eligible Linear (rotate + INT4 pack + convrot_w4a4 stamp). "
            "Default ON; --no-convrot packs plain INT8 only."
        ),
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=_DEFAULT_GROUPSIZE,
        help=f"ConvRot Hadamard group size (power of 4, default {_DEFAULT_GROUPSIZE})",
    )
    parser.add_argument(
        "--min_in_features",
        type=int,
        default=0,
        help=(
            "Skip Linear/Conv with in_features below this (0 = convert all eligible). "
            "RDNA4 loader often uses 512."
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

    convert_to_int4_convrot(
        args.model,
        args.output,
        per_channel_int8=args.per_channel_int8,
        bias_correction=bool(args.bias_correction),
        calib_file=args.calib_file,
        num_calib_samples=args.num_calib_samples,
        num_inference_steps=args.num_inference_steps,
        enable_convrot=bool(args.convrot),
        group_size=int(args.groupsize),
        min_in_features=int(args.min_in_features),
    )

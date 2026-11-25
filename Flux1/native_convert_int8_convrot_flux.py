"""Flux1 DiT → ComfyUI native int8_tensorwise FULL ConvRot converter (hswq not used).

Flux1 only (model.diffusion_model.* layout, e.g. redcraftHybridH3A2A_realreveal5.safetensors).
"native" = a plain compressed model that does not use hswq (the MEMORY.md definition).

Pack (ComfyUI MixedPrecisionOps + comfy_kitchen TensorWiseINT8Layout):
  <layer>.weight           int8
  <layer>.weight_scale     float32
      ConvRot Linear:      [out, 1] (row-wise) — kitchen online act rotate
      plain INT8:          scalar (tensorwise)
  <layer>.comfy_quant      uint8 JSON (compact)
      plain:  {"format":"int8_tensorwise"}
      ConvRot:{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":N}

FULL ConvRot (default ON; --no-convrot for plain INT8):
  Linear 2D:  W_rot = W @ H^T (per-group Hadamard, power of 4) -> row-wise INT8 -> stamp.
  Flux1 is all Linear (no Conv2d). Where 4-D weights appear, treat them as rotate_weight_conv2d does.
  A layer whose in_features is not divisible by the power-of-4 group size -> plain tensorwise INT8 (fallback).

Scope:
  Only .weight under model.diffusion_model. (ndim>=2, fp16/bf16/fp32) is converted.
  1-D tensors (bias / scale / norm) and other prefixes (conditioner etc.) are kept as they are.

Bias Correction (Card 1) is out of scope for this script:
  Flux's Card 1 calibration needs a ComfyUI run (t5xxl + clip_l + DiT forward + DualMonitor),
  which is far too expensive in a 16 GB VRAM environment, so it stays separate from the plain native compression this script performs.

Post-convert bench (default ON): after save, subprocess runs
  benchmark/flux1_convrot_int8_traj_compare.py (trajectory, official fixed 25 seeds).
  --clip_path (t5xxl), --clip_l_path and --comfy_path are required.
  skip with --no-bench.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

_DEFAULT_GROUPSIZE = 256
_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def _default_repo_root() -> str:
    """Locate the repo root by walking up until native_convert_int8.py is found."""
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


# ---------------------------------------------------------------------------
# Hadamard / ConvRot core (self-contained; same math as comfy_kitchen ConvRot)
# ---------------------------------------------------------------------------
def build_hadamard(
    size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normalized regular Hadamard (power-of-4), same as comfy_kitchen ConvRot."""
    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]

    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")

    h4 = torch.tensor(
        [
            [1, 1, 1, -1],
            [1, 1, -1, 1],
            [1, -1, 1, 1],
            [-1, 1, 1, 1],
        ],
        dtype=dtype,
        device=device,
    )
    h_matrix = h4
    current_size = 4
    while current_size < size:
        h_matrix = torch.kron(h_matrix, h4)
        current_size *= 4
    h_matrix = h_matrix / (size**0.5)
    _HADAMARD_CACHE[cache_key] = h_matrix
    return h_matrix


def convrot_group_size_for_features(n: int, preferred: int = _DEFAULT_GROUPSIZE) -> int | None:
    """Largest power-of-4 group size <= preferred that divides n (or None)."""
    if n < 4:
        return None
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def rotate_weight(weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int) -> torch.Tensor:
    """Offline Linear: W_rot = W @ H^T (group-wise). Matches comfy_kitchen._rotate_weight."""
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features {in_features} not divisible by group_size {group_size}"
        )
    group_count = in_features // group_size
    weight_grouped = weight.view(out_features, group_count, group_size)
    return torch.matmul(
        weight_grouped, h_matrix.T.to(dtype=weight.dtype, device=weight.device)
    ).reshape(weight.shape)


def rotate_weight_conv2d(
    weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Offline Conv2d: rotate along in_channels. weight (O, I, kH, kW)."""
    if weight.ndim != 4:
        raise ValueError(f"Conv2d weight must be 4D, got ndim={weight.ndim}")
    out_c, in_c, k_h, k_w = weight.shape
    flat = weight.permute(0, 2, 3, 1).contiguous().view(-1, in_c)
    flat_rot = rotate_weight(flat, h_matrix, group_size)
    return flat_rot.view(out_c, k_h, k_w, in_c).permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# INT8 packing
# ---------------------------------------------------------------------------
def pack_tensorwise(weight: torch.Tensor):
    """Symmetric per-tensor INT8: scale = amax / 127."""
    w = weight.float()
    amax = max(float(w.abs().max().item()), 1e-6)
    scale = amax / 127.0
    q = (w / scale).round().clamp(-127, 127).to(torch.int8)
    return q, torch.tensor(scale, dtype=torch.float32)


def pack_channelwise(weight: torch.Tensor):
    """Per-out-channel INT8 (Linear [O,1] / Conv [O,1,1,1])."""
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


def _encode_comfy_quant(config: dict) -> torch.Tensor:
    return torch.tensor(
        list(json.dumps(config, separators=(",", ":")).encode("utf-8")),
        dtype=torch.uint8,
    )


# ---------------------------------------------------------------------------
# VRAM / bench helpers
# ---------------------------------------------------------------------------
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


def run_post_convert_flux_int8_bench(
    *,
    script_dir: str,
    fp16_path: str,
    int8_path: str,
    clip_path: str,
    clip_l_path: str,
    comfy_path: str,
) -> int:
    """After save: subprocess benchmark/flux1_convrot_int8_traj_compare.py.

    Bench argv:
      --fp16 <input> --int8 <INT8> --clip_path <t5xxl> --clip_l_path <clip_l>
      --comfy_path <ComfyUI root>
    Seeds: the bench default, the official fixed 25 (42-start list). No --seeds
    override is passed here.
    """
    bench_script = os.path.join(
        script_dir, "benchmark", "flux1_convrot_int8_traj_compare.py"
    )
    if not os.path.isfile(bench_script):
        print(f"Error: bench script not found: {bench_script}")
        return 1
    cmd = [
        sys.executable,
        bench_script,
        "--fp16", fp16_path,
        "--int8", int8_path,
        "--clip_path", clip_path,
        "--clip_l_path", clip_l_path,
        "--comfy_path", comfy_path,
    ]
    print("Running trajectory bench: " + " ".join(cmd))
    return subprocess.run(cmd).returncode

def is_flux1_matmul_weight(key: str, tensor: torch.Tensor) -> bool:
    """Flux1 DiT matmul weight: model.diffusion_model.* .weight, ndim>=2.

    No dtype condition. The packing path casts the weight to float32 before
    rotating and quantising, so any 2-D .weight is a target; a dtype list here
    only excludes weights the path can already handle.
    """
    if not key.startswith("model.diffusion_model."):
        return False
    if not key.endswith(".weight"):
        return False
    return tensor.ndim >= 2


def convert_to_int8(
    input_path: str,
    output_path: str,
    enable_convrot: bool = True,
    group_size: int = _DEFAULT_GROUPSIZE,
    per_channel_int8: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {input_path}")
    state_dict = load_file(input_path)

    new_state_dict = {}
    quant_meta_layers = {}
    converted_count = 0
    skipped_count = 0
    plain_int8_count = 0
    convrot_linear = 0
    convrot_conv2d = 0
    kept_float_count = 0

    rot_tag = " + ConvRot(Linear+Conv2d)" if enable_convrot else ""
    plain_tag = "per-channel" if per_channel_int8 else "tensorwise"
    print(
        f"Converting Flux1 DiT matmul weights to INT8 "
        f"({plain_tag} amax/127{rot_tag})..."
    )

    def _pack_plain(w: torch.Tensor):
        """Plain (non-ConvRot) pack: channelwise when --per_channel_int8."""
        if per_channel_int8:
            return pack_channelwise(w)
        return pack_tensorwise(w)

    for key, tensor in tqdm(state_dict.items()):
        if not is_flux1_matmul_weight(key, tensor):
            new_state_dict[key] = tensor
            kept_float_count += 1
            continue

        w_fp = tensor.float()
        module_key = key[: -len(".weight")]
        quant_config: dict
        q: torch.Tensor
        scale: torch.Tensor

        if enable_convrot:
            used_gs = convrot_group_size_for_features(int(w_fp.shape[1]), group_size)
            if used_gs is not None and tensor.ndim == 2:
                h_matrix = build_hadamard(used_gs, device="cpu", dtype=torch.float32)
                w_fp = rotate_weight(w_fp, h_matrix, used_gs)
                q, scale = pack_channelwise(w_fp)
                quant_config = {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": int(used_gs),
                }
                convrot_linear += 1
            elif used_gs is not None and tensor.ndim == 4:
                h_matrix = build_hadamard(used_gs, device="cpu", dtype=torch.float32)
                w_fp = rotate_weight_conv2d(w_fp, h_matrix, used_gs)
                q, scale = pack_channelwise(w_fp)
                quant_config = {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": int(used_gs),
                }
                convrot_conv2d += 1
            else:
                q, scale = _pack_plain(w_fp)
                quant_config = {"format": "int8_tensorwise"}
                plain_int8_count += 1
        else:
            q, scale = _pack_plain(w_fp)
            quant_config = {"format": "int8_tensorwise"}
            plain_int8_count += 1

        new_state_dict[key] = q
        new_state_dict[f"{module_key}.weight_scale"] = scale
        new_state_dict[f"{module_key}.comfy_quant"] = _encode_comfy_quant(
            quant_config
        )
        quant_meta_layers[module_key] = dict(quant_config)
        converted_count += 1

    metadata = {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": quant_meta_layers}
        )
    }

    print(f"Saving to: {output_path}")
    print(f"Converted layers: {converted_count}, Kept non-matmul: {kept_float_count}")
    print(f"ConvRot FULL: {enable_convrot}")
    if enable_convrot:
        print(
            f"  ConvRot Linear: {convrot_linear}, ConvRot Conv2d: {convrot_conv2d}, "
            f"plain INT8 (no eligible group size): {plain_int8_count}"
        )
    else:
        print(f"  plain INT8: {plain_int8_count}")
    print(f"  plain scale mode (--per_channel_int8): {per_channel_int8}")

    save_file(new_state_dict, output_path, metadata=metadata)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Flux1 DiT INT8 convert with FULL ConvRot (Linear) ON by default. "
            "plain native compression (hswq not used). Bias Correction is out of scope for this script."
        )
    )
    parser.add_argument(
        "--model",
        "--input",
        dest="model",
        type=str,
        required=True,
        help="Path to input Flux1 .safetensors (model.diffusion_model.* layout)",
    )
    parser.add_argument("--output", type=str, required=True, help="Path to output .safetensors")
    parser.add_argument(
        "--convrot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "FULL ConvRot on Linear/Conv2d when in_dim divisible by a power-of-4 "
            "group size: rotate + per-channel scale + stamp. Default ON; "
            "pass --no-convrot for plain tensorwise."
        ),
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=_DEFAULT_GROUPSIZE,
        help=f"ConvRot Hadamard group size (power of 4, default {_DEFAULT_GROUPSIZE})",
    )
    parser.add_argument(
        "--per_channel_int8",
        action="store_true",
        help=(
            "Plain (non-ConvRot) layers use per-out-channel amax/scale instead "
            "of per-tensor. ConvRot layers are channelwise by construction; "
            "the stamp stays int8_tensorwise."
        ),
    )
    parser.add_argument(
        "--clip_path",
        type=str,
        default=None,
        help="T5XXL text encoder path (required when --bench)",
    )
    parser.add_argument(
        "--clip_l_path",
        type=str,
        default=None,
        help="clip_l text encoder path (required when --bench)",
    )
    parser.add_argument(
        "--comfy_path",
        type=str,
        default=None,
        help="ComfyUI root path (required when --bench)",
    )
    parser.add_argument(
        "--bench",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After save, run benchmark/flux1_convrot_int8_traj_compare.py "
            "(requires --clip_path/--clip_l_path/--comfy_path; seeds = the "
            "official fixed 25). "
            "Pass --no-bench to skip."
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
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
        if not args.clip_l_path:
            missing.append("--clip_l_path")
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
        if not os.path.isfile(args.clip_l_path):
            print(f"Error: --clip_l_path not found: {args.clip_l_path}")
            sys.exit(1)
        if not os.path.isdir(args.comfy_path):
            print(f"Error: --comfy_path not found: {args.comfy_path}")
            sys.exit(1)

    convert_to_int8(
        args.model,
        args.output,
        enable_convrot=bool(args.convrot),
        group_size=int(args.groupsize),
        per_channel_int8=bool(args.per_channel_int8),
    )

    if args.bench:
        bench_rc = run_post_convert_flux_int8_bench(
            script_dir=_REPO_ROOT,
            fp16_path=args.model,
            int8_path=args.output,
            clip_path=args.clip_path,
            clip_l_path=args.clip_l_path,
            comfy_path=args.comfy_path,
        )
        if bench_rc != 0:
            print(f"[FATAL] Post-convert bench exited with code {bench_rc}")
            sys.exit(bench_rc)
    else:
        print("[*] Post-convert bench skipped (--no-bench)")

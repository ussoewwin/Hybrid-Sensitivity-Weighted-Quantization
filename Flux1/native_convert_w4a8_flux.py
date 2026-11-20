"""Flux1 DiT -> ComfyUI-native ConvRot W4A8 (asym_w4a8_int8) converter (hswq not used).

Flux1-only (model.diffusion_model.* DiT layout, e.g. fluxNSFWUNLOCKED_v30FP16).
"native" = plain compression without hswq. Every 2-D Linear is packed as W4A8 (no protected layers).

Pack (comfy_kitchen AsymW4A8Int8Layout = ComfyUI native reader):
  <layer>.weight            int8   [N, K/2]   packed nibbles, low = even column
  <layer>.weight_s_rel      fp8e4m3 [N, K/16] per-group relative scale
  <layer>.weight_s_channel  float32 [N]       per-out-channel scale
  <layer>.weight_codebook   float32 [16]      Lloyd-Max table (absent when LUT fits)
  <layer>.comfy_quant       uint8 JSON {"format":"asym_w4a8_int8",
                                        "group_size":16,"convrot_groupsize":256}

Layer routing (explicit, separated per kind):
  Linear 2D, in_features % 256 == 0 (and % 16 == 0)
      -> W4A8, rotation runs INSIDE comfy_kitchen.quantize_w4a8_int8_weight
         (convrot group size 256 is the only size the CUDA fused requant supports)
  Linear 2D, in_features % 256 != 0 but a power-of-4 group divides K
      -> ConvRot INT8 fallback (int8 [N, K], channel-wise scale, convrot stamp)
  Conv2d 4D (absent in Flux1; kept for completeness)
      -> ConvRot INT8 fallback via rotate_weight_conv2d + channel-wise pack
  anything else (bias / norm / embedder 1D, non-diffusion keys)
      -> kept as-is, original dtype

The pack math is not re-implemented here: the same comfy_kitchen API the ComfyUI
runtime decodes with is used at convert time, so the on-disk layout always
matches the reader. Requires comfy_kitchen in the running interpreter (ComfyUI
python_embeded); fails fast when missing.

Bias Correction (Card 1) is out of scope for Flux1 native packs (needs a ComfyUI
run-based calibration with t5xxl + clip_l; see Flux1/README.md).

Usage:
    python Flux1/native_convert_w4a8_flux.py \\
        --model "D:\\...\\flux_fp16.safetensors" \\
        --output "D:\\...\\flux_native_convrot_w4a8.safetensors" \\
        --clip_path "D:\\...\\t5xxl.safetensors" \\
        --clip_l_path "D:\\...\\clip_l.safetensors" \\
        --comfy_path "D:\\...\\ComfyUI-master"

Post-convert bench (default ON): after save, subprocess runs
  benchmark/flux1_convrot_w4a8_traj_compare.py (trajectory, official fixed 25 seeds).
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
# W4A8 quant group: fixed 16 (comfy_kitchen default, CUDA fused requant config).
_QUANT_GROUP_SIZE = 16
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


def host_ram_available_mib():
    """Free host RAM in MiB (Linux /proc/meminfo, psutil fallback), or None."""
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    try:
        import psutil

        return psutil.virtual_memory().available / (1024 ** 2)
    except Exception:
        return None


def release_host_ram(label: str = "") -> None:
    """Return the converter's freed heap to the OS before the bench child starts.

    The conversion holds the full fp16 state dict (~24 GB for Flux1) plus the
    output dict at its peak. CPython frees them when ``convert_to_w4a8``
    returns, but glibc keeps the freed arenas mapped, so the bench child -- which
    loads the same fp16 DiT again -- can be killed by the cgroup OOM killer
    (exit code -9) even though the parent looks idle. ``malloc_trim(0)`` forces
    the arenas back to the kernel; the measured MemAvailable is printed so the
    number is on the record.
    """
    before = host_ram_available_mib()
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
    after = host_ram_available_mib()
    tag = f" ({label})" if label else ""
    if before is None or after is None:
        print(f"[*] Host RAM released{tag}: unavailable on this platform")
    else:
        print(
            f"[*] Host RAM released{tag}: available {after:.0f} MiB "
            f"(was {before:.0f} MiB, +{after - before:.0f} MiB)"
        )

# Shape-critical Linear keys: ComfyUI-master's FLUX detector reads the SECOND
# dimension of these weights to size the architecture
# (model_detection.py: img_in.weight -> in_channels, txt_in.weight ->
# context_in_dim, vector_in.in_layer.weight -> vec_in_dim). A nibble-packed
# weight halves K, so the detector would read K/2 and build the module with the
# wrong in_features (measured: ValueError "packed weight width 384 does not
# match K=384 at 4 bits" on vector_in.in_layer). These three stay full-width
# INT8 (int8_tensorwise, shape-preserving) like the SDXL W4A8 converter's
# shape-critical keys.
_SHAPE_CRITICAL_REL_KEYS = frozenset(
    {
        "img_in.weight",
        "txt_in.weight",
        "vector_in.in_layer.weight",
    }
)


def _load_comfy_kitchen():
    """Return comfy_kitchen; fail fast when the W4A8 API is missing."""
    try:
        import comfy_kitchen as ck
    except ImportError as exc:
        raise ImportError(
            "comfy_kitchen is required for W4A8 packing "
            "(run this converter with the ComfyUI python_embeded interpreter)"
        ) from exc
    for attr in ("quantize_w4a8_int8_weight", "dequantize_w4a8_int8_weight"):
        if not hasattr(ck, attr):
            raise ImportError(
                f"comfy_kitchen is too old: missing {attr} "
                "(W4A8 packing needs comfy_kitchen with AsymW4A8Int8Layout)"
            )
    return ck


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


def convrot_group_size_for_features(
    n: int, preferred: int = _DEFAULT_GROUPSIZE
) -> int | None:
    """Largest power-of-4 group size <= preferred that divides n (or None)."""
    if n < 4:
        return None
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def rotate_weight(
    weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Offline Linear: W_rot = W @ H^T (group-wise). Matches kitchen ConvRot."""
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


def pack_tensorwise(weight: torch.Tensor):
    """Symmetric per-tensor INT8: scale = amax / 127."""
    w = weight.float()
    amax = max(float(w.abs().max().item()), 1e-6)
    scale = amax / 127.0
    q = (w / scale).round().clamp(-127, 127).to(torch.int8)
    return q, torch.tensor(scale, dtype=torch.float32)


def pack_channelwise(weight: torch.Tensor):
    """Per-out-channel INT8 (kitchen ConvRot dequant shape)."""
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
        raise ValueError(f"unsupported weight ndim={w.dim()} for channelwise INT8")
    clamped = torch.clamp(w, -amax_view, amax_view)
    q = (clamped / scale_view).round().clamp(-127, 127).to(torch.int8)
    return q, scale_view.to(dtype=torch.float32)


def _encode_comfy_quant(config: dict) -> torch.Tensor:
    return torch.tensor(
        list(json.dumps(config, separators=(",", ":")).encode("utf-8")),
        dtype=torch.uint8,
    )


def run_post_convert_flux_w4a8_bench(
    *,
    script_dir: str,
    fp16_path: str,
    w4a8_path: str,
    clip_path: str,
    clip_l_path: str,
    comfy_path: str,
) -> int:
    """After save: subprocess benchmark/flux1_convrot_w4a8_traj_compare.py.

    Bench argv:
      --fp16 <input> --w4a8 <W4A8 pack> --clip_path <t5xxl> --clip_l_path <clip_l>
      --comfy_path <ComfyUI root>
    Seeds: the bench default, the official fixed 25 (42-start list). No --seeds
    override is passed here.
    """
    bench_script = os.path.join(
        script_dir, "benchmark", "flux1_convrot_w4a8_traj_compare.py"
    )
    if not os.path.isfile(bench_script):
        print(f"Error: bench script not found: {bench_script}")
        return 1
    cmd = [
        sys.executable,
        bench_script,
        "--fp16", fp16_path,
        "--w4a8", w4a8_path,
        "--clip_path", clip_path,
        "--clip_l_path", clip_l_path,
        "--comfy_path", comfy_path,
    ]
    print("Running trajectory bench: " + " ".join(cmd))
    return subprocess.run(cmd).returncode


def is_flux1_matmul_weight(key: str, tensor: torch.Tensor) -> bool:
    """Flux1 DiT matmul weight: model.diffusion_model.* .weight, ndim>=2, float."""
    if not key.startswith("model.diffusion_model."):
        return False
    if not key.endswith(".weight"):
        return False
    if tensor.ndim < 2:
        return False
    return tensor.dtype in (torch.float16, torch.float32, torch.bfloat16)


def convert_to_w4a8(
    input_path: str,
    output_path: str,
    *,
    enable_convrot: bool = True,
    group_size: int = _DEFAULT_GROUPSIZE,
    per_channel_int8: bool = False,
):
    ck = _load_comfy_kitchen() if enable_convrot else None
    print(f"Loading model: {input_path}")
    state_dict = load_file(input_path)

    new_state_dict: dict[str, torch.Tensor] = {}
    quant_meta_layers: dict[str, dict] = {}
    converted = 0
    kept = 0
    w4a8_linear = 0
    convrot_linear = 0
    convrot_conv2d = 0
    plain_int8 = 0
    shape_critical_int8 = 0

    print(
        "Converting Flux1 DiT matmul weights "
        f"(W4A8 group={_QUANT_GROUP_SIZE}, convrot_groupsize=256 gate; "
        f"ConvRot INT8 fallback; plain "
        f"{'per-channel' if per_channel_int8 else 'tensorwise'})"
    )

    def _pack_plain(w: torch.Tensor):
        """Plain (non-ConvRot) pack: channelwise when --per_channel_int8."""
        if per_channel_int8:
            return pack_channelwise(w)
        return pack_tensorwise(w)

    for key, tensor in tqdm(list(state_dict.items())):
        if not is_flux1_matmul_weight(key, tensor):
            new_state_dict[key] = tensor
            kept += 1
            continue

        w_fp = tensor.float()
        module_key = key[: -len(".weight")]
        rel_key = module_key[len("model.diffusion_model."):] if module_key.startswith(
            "model.diffusion_model.") else module_key
        used_gs = (
            convrot_group_size_for_features(int(w_fp.shape[1]), group_size)
            if enable_convrot
            else None
        )
        quant_config: dict

        # Shape-critical: the FLUX detector reads K from these tensors, so they
        # are never nibble-packed - full-width INT8 keeps the original shape[1].
        if f"{rel_key}.weight" in _SHAPE_CRITICAL_REL_KEYS and tensor.ndim == 2:
            q, scale = pack_tensorwise(w_fp)
            quant_config = {"format": "int8_tensorwise"}
            new_state_dict[key] = q
            new_state_dict[f"{module_key}.weight_scale"] = scale
            new_state_dict[f"{module_key}.comfy_quant"] = _encode_comfy_quant(
                quant_config
            )
            quant_meta_layers[module_key] = dict(quant_config)
            converted += 1
            shape_critical_int8 += 1
            continue

        if (
            enable_convrot
            and ck is not None
            and tensor.ndim == 2
            and used_gs == 256
            and int(w_fp.shape[1]) % _QUANT_GROUP_SIZE == 0
        ):
            q, s_rel, s_channel, _correction, codebook = ck.quantize_w4a8_int8_weight(
                w_fp,
                group_size=_QUANT_GROUP_SIZE,
                convrot_groupsize=256,
                symmetric=True,
                scale_dtype=torch.float8_e4m3fn,
                codebook=True,
                bits=4,
            )
            quant_config = {
                "format": "asym_w4a8_int8",
                "group_size": _QUANT_GROUP_SIZE,
                "convrot_groupsize": 256,
            }
            new_state_dict[key] = q.to(torch.int8)
            new_state_dict[f"{module_key}.weight_s_rel"] = s_rel.to(
                torch.float8_e4m3fn
            ).contiguous()
            new_state_dict[f"{module_key}.weight_s_channel"] = (
                s_channel.float().contiguous()
            )
            if codebook is not None:
                new_state_dict[f"{module_key}.weight_codebook"] = (
                    codebook.float().contiguous()
                )
            new_state_dict[f"{module_key}.comfy_quant"] = _encode_comfy_quant(
                quant_config
            )
            quant_meta_layers[module_key] = dict(quant_config)
            converted += 1
            w4a8_linear += 1
            continue

        if used_gs is not None and tensor.ndim == 2:
            h_matrix = build_hadamard(used_gs, device="cpu", dtype=torch.float32)
            w_rot = rotate_weight(w_fp, h_matrix, used_gs)
            q, scale = pack_channelwise(w_rot)
            quant_config = {
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": int(used_gs),
            }
            convrot_linear += 1
        elif used_gs is not None and tensor.ndim == 4:
            h_matrix = build_hadamard(used_gs, device="cpu", dtype=torch.float32)
            w_rot = rotate_weight_conv2d(w_fp, h_matrix, used_gs)
            q, scale = pack_channelwise(w_rot)
            quant_config = {
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": int(used_gs),
            }
            convrot_conv2d += 1
        else:
            q, scale = _pack_plain(w_fp)
            quant_config = {"format": "int8_tensorwise"}
            plain_int8 += 1

        new_state_dict[key] = q
        new_state_dict[f"{module_key}.weight_scale"] = scale
        new_state_dict[f"{module_key}.comfy_quant"] = _encode_comfy_quant(quant_config)
        quant_meta_layers[module_key] = dict(quant_config)
        converted += 1

    metadata = {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": quant_meta_layers}
        )
    }

    print(f"Saving to: {output_path}")
    print(f"Converted layers: {converted}, Kept non-matmul: {kept}")
    print(f"ConvRot enabled: {enable_convrot}")
    if enable_convrot:
        print(
            f"  W4A8 Linear (asym_w4a8_int8): {w4a8_linear}, "
            f"ConvRot INT8 Linear: {convrot_linear}, "
            f"ConvRot INT8 Conv2d: {convrot_conv2d}, "
            f"shape-critical full-width INT8 (img_in/txt_in/vector_in.in_layer): "
            f"{shape_critical_int8}, "
            f"plain INT8 (no eligible group size): {plain_int8}"
        )
        print(f"  plain scale mode (--per_channel_int8): {per_channel_int8}")
    else:
        print(f"  plain INT8: {plain_int8}")

    save_file(new_state_dict, output_path, metadata=metadata)
    print("Done!")
    # Drop both dicts here so the peak is gone before the caller starts the
    # bench child (see release_host_ram).
    del new_state_dict
    del state_dict
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Flux1 DiT ConvRot W4A8 converter for native ComfyUI "
            "(all Linear -> asym_w4a8_int8 via comfy_kitchen; no protection set)."
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
        "--per_channel_int8",
        action="store_true",
        help=(
            "Plain (non-ConvRot) layers use per-out-channel amax/scale instead "
            "of per-tensor. ConvRot layers are channelwise by construction; "
            "the stamp stays int8_tensorwise."
        ),
    )
    parser.add_argument(
        "--convrot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ConvRot on (default): W4A8 when K%%256==0, ConvRot INT8 fallback otherwise.",
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
            "After save, run benchmark/flux1_convrot_w4a8_traj_compare.py "
            "(requires --clip_path/--clip_l_path/--comfy_path; seeds = the "
            "official fixed 25). "
            "Pass --no-bench to skip."
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        raise SystemExit(1)
    if args.groupsize < 4 or (args.groupsize & (args.groupsize - 1)) != 0:
        print(f"Error: --groupsize must be a power of 4 (>=4), got {args.groupsize}")
        raise SystemExit(1)
    if math.log(args.groupsize, 4) % 1 != 0:
        print(f"Error: --groupsize must be a power of 4, got {args.groupsize}")
        raise SystemExit(1)

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
            raise SystemExit(1)
        if not os.path.isfile(args.clip_path):
            print(f"Error: --clip_path not found: {args.clip_path}")
            raise SystemExit(1)
        if not os.path.isfile(args.clip_l_path):
            print(f"Error: --clip_l_path not found: {args.clip_l_path}")
            raise SystemExit(1)
        if not os.path.isdir(args.comfy_path):
            print(f"Error: --comfy_path not found: {args.comfy_path}")
            raise SystemExit(1)

    convert_to_w4a8(
        args.model,
        args.output,
        enable_convrot=bool(args.convrot),
        group_size=int(args.groupsize),
        per_channel_int8=bool(args.per_channel_int8),
    )

    if args.bench:
        release_host_ram("after convert, before bench")
        bench_rc = run_post_convert_flux_w4a8_bench(
            script_dir=_REPO_ROOT,
            fp16_path=args.model,
            w4a8_path=args.output,
            clip_path=args.clip_path,
            clip_l_path=args.clip_l_path,
            comfy_path=args.comfy_path,
        )
        if bench_rc != 0:
            print(f"[FATAL] Post-convert bench exited with code {bench_rc}")
            raise SystemExit(bench_rc)
    else:
        print("[*] Post-convert bench skipped (--no-bench)")

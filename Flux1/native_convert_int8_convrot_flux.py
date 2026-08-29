"""Flux1 DiT → ComfyUI native int8_tensorwise FULL ConvRot converter (hswq 非使用).

Flux1 専用（redcraftHybridH3A2A_realreveal5.safetensors 等の model.diffusion_model.* 構成）。
「native」= hswq を使わない単なる圧縮モデル（MEMORY.md 定義準拠）。

Pack (ComfyUI MixedPrecisionOps + comfy_kitchen TensorWiseINT8Layout):
  <layer>.weight           int8
  <layer>.weight_scale     float32
      ConvRot Linear:      [out, 1] (row-wise) — kitchen online act rotate
      plain INT8:          scalar (tensorwise)
  <layer>.comfy_quant      uint8 JSON (compact)
      plain:  {"format":"int8_tensorwise"}
      ConvRot:{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":N}

FULL ConvRot (default ON; --no-convrot で plain INT8):
  Linear 2D:  W_rot = W @ H^T（グループ単位 Hadamard、power-of-4）→ row-wise INT8 → stamp。
  Flux1 は全 Linear（Conv2d なし）。4D が混在する場合は rotate_weight_conv2d 相当で処理。
  in_features が power-of-4 groupsize で割り切れない層 → plain tensorwise INT8（フォールバック）。

対象スコープ:
  key が model.diffusion_model. で始まる .weight（ndim>=2, fp16/bf16/fp32）のみ変換。
  1D（bias / scale / norm）と他プレフィックス（conditioner 等）はそのまま保持。

Bias Correction（Card 1）は本スクリプトでは扱わない:
  flux の Card 1 calib は ComfyUI 実行ベース（t5xxl + clip_l + DiT forward + DualMonitor）が必要で、
  VRAM 16GB 環境ではコストが非常に大きいため、native 単純圧縮（本スクリプト）とは分離する。

Post-convert bench（default ON）: 保存後、subprocess で benchmark/flux_int8_bench.py を実行。
  --clip_path（t5xxl）と --clip_l_path、--comfy_path が必須。--vae は任意（無い場合は latent MSE のみ）。
  --no-bench でスキップ。
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
    vae_path: str | None = None,
) -> int:
    """After save: subprocess benchmark/flux_int8_bench.py.

    Bench argv:
      --fp16 <input> --int8 <INT8> --clip_path <t5xxl> --clip_l_path <clip_l>
      --comfy_path <ComfyUI root> [--vae <flux VAE>] --output_dir <benchmark result>
    シードはベンチ側デフォルト（42 + 10桁 の 5 個）を使用（MEMORY.md ルール準拠）。
    """
    bench_script = os.path.join(script_dir, "benchmark", "flux1_nvfp4", "flux_int8_bench.py")
    if not os.path.isfile(bench_script):
        print(f"[FATAL] Post-convert bench script not found: {bench_script}")
        return 1
    if not os.path.isfile(fp16_path):
        print(f"[FATAL] Post-convert bench: FP16 (--fp16) missing: {fp16_path}")
        return 1
    if not os.path.isfile(int8_path):
        print(f"[FATAL] Post-convert bench: INT8 (--int8) missing: {int8_path}")
        return 1
    if not clip_path or not os.path.isfile(clip_path):
        print(f"[FATAL] Post-convert bench: --clip_path missing: {clip_path}")
        return 1
    if not clip_l_path or not os.path.isfile(clip_l_path):
        print(f"[FATAL] Post-convert bench: --clip_l_path missing: {clip_l_path}")
        return 1
    if not comfy_path or not os.path.isdir(comfy_path):
        print(f"[FATAL] Post-convert bench: --comfy_path missing: {comfy_path}")
        return 1
    if vae_path and not os.path.isfile(vae_path):
        print(f"[FATAL] Post-convert bench: --vae missing: {vae_path}")
        return 1

    _release_vram("pre-flux_int8_bench subprocess")

    cmd = [
        sys.executable,
        bench_script,
        "--fp16",
        fp16_path,
        "--int8",
        int8_path,
        "--clip_path",
        clip_path,
        "--clip_l_path",
        clip_l_path,
        "--comfy_path",
        comfy_path,
    ]
    if vae_path:
        cmd.extend(["--vae", vae_path])

    print("=" * 60)
    print("[*] Post-convert Flux INT8 bench (owner body shape)")
    print(f"    script: {bench_script}")
    print(f"    --fp16: {fp16_path}")
    print(f"    --int8: {int8_path}")
    print(f"    --clip_path (t5xxl): {clip_path}")
    print(f"    --clip_l_path: {clip_l_path}")
    print(f"    --comfy_path: {comfy_path}")
    if vae_path:
        print(f"    --vae: {vae_path}")
    print("    seeds: 42 + 10桁 の 5 個 (bench 側デフォルト)")
    print("=" * 60)
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------
def is_flux1_matmul_weight(key: str, tensor: torch.Tensor) -> bool:
    """Flux1 DiT matmul weight: model.diffusion_model.* .weight, ndim>=2, float."""
    if not key.startswith("model.diffusion_model."):
        return False
    if not key.endswith(".weight"):
        return False
    if tensor.ndim < 2:
        return False
    return tensor.dtype in (torch.float16, torch.float32, torch.bfloat16)


def convert_to_int8(
    input_path: str,
    output_path: str,
    enable_convrot: bool = True,
    group_size: int = _DEFAULT_GROUPSIZE,
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
    print(f"Converting Flux1 DiT matmul weights to INT8 (amax/127{rot_tag})...")

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
                q, scale = pack_tensorwise(w_fp)
                quant_config = {"format": "int8_tensorwise"}
                plain_int8_count += 1
        else:
            q, scale = pack_tensorwise(w_fp)
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

    save_file(new_state_dict, output_path, metadata=metadata)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Flux1 DiT INT8 convert with FULL ConvRot (Linear) ON by default. "
            "native (hswq 非使用) の単純圧縮。Bias Correction は本スクリプトの対象外。"
        )
    )
    parser.add_argument(
        "--model",
        "--input",
        dest="model",
        type=str,
        required=True,
        help="Path to input Flux1 .safetensors (model.diffusion_model.* 構成)",
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
        "--vae",
        type=str,
        default=None,
        help="Optional Flux VAE path for post-convert bench (SSIM 用)",
    )
    parser.add_argument(
        "--bench",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After save, run benchmark/flux_int8_bench.py "
            "(requires --clip_path/--clip_l_path/--comfy_path; optional --vae). "
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
        if args.vae and not os.path.isfile(args.vae):
            print(f"Error: --vae not found: {args.vae}")
            sys.exit(1)

    convert_to_int8(
        args.model,
        args.output,
        enable_convrot=bool(args.convrot),
        group_size=int(args.groupsize),
    )

    if args.bench:
        bench_rc = run_post_convert_flux_int8_bench(
            script_dir=_REPO_ROOT,
            fp16_path=args.model,
            int8_path=args.output,
            clip_path=args.clip_path,
            clip_l_path=args.clip_l_path,
            comfy_path=args.comfy_path,
            vae_path=args.vae,
        )
        if bench_rc != 0:
            print(f"[FATAL] Post-convert bench exited with code {bench_rc}")
            sys.exit(bench_rc)
    else:
        print("[*] Post-convert bench skipped (--no-bench)")

"""Flux1 DiT → ComfyUI native Hybrid/Native NVFP4 converter（INT8 protect + NVFP4、hswq 非使用）。

MEMORY ワークフロー: convrot int8 → **hybrid nvfp4** → native nvfp4 → ベンチ の 2〜3 段目。

「native」= hswq を使わない単なる圧縮。「hybrid」= 重要層を INT8 保護し、残りを NVFP4 にした混在。

--mode:
  hybrid（default）: 構造ベース INT8 保護層 + 残り NVFP4
  native            : 全 Linear NVFP4（保護なし、只の圧縮）

Pack (ComfyUI comfy_quant 対応):
  INT8 保護層（構造ベース）:
    <layer>.weight           int8
    <layer>.weight_scale     float32 [O,1]（row-wise、ConvRot 時）
    <layer>.comfy_quant      {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":G}
  NVFP4 層（残りの Linear 2D）:
    <layer>.weight           uint8（TensorCoreNVFP4Layout パック）
    <layer>.weight_scale     float8_e4m3fn
    <layer>.weight_scale_2   float32
    <layer>.comfy_quant      {"format":"nvfp4","convrot":true,"convrot_groupsize":G,
                              "in_features":I,"out_features":O,"orig_shape":"[O, I]"}

INT8 保護層（構造ベースの感度仮説）:
  - adaLN modulation 系: double_blocks の img_mod.lin / txt_mod.lin、
    single_blocks の modulation.lin、final_layer.adaLN_modulation.1
    （attention/MLP 出力のスケール・シフトを生成するため誤差が直接品質に響く）
  - 入出力系: img_in / txt_in / time_in / vector_in / guidance_in / final_layer.linear

ConvRot FULL ON（Linear 2D）: W_rot = W @ H^T（power-of-4 Hadamard、groupsize 256 デフォルト）。
in_features が割り切れない層は plain（NVFP4 {"format":"nvfp4"} または INT8 tensorwise）。
1D（bias / scale / norm）と model.diffusion_model. 以外はそのまま保持。

Bias Correction（Card 1）は対象外（flux の ComfyUI 実行ベース calib は別スコープ）。

Post-convert bench（default ON）: benchmark/flux_int8_bench.py を subprocess。
  --fp16 <元> --int8 <本出力> --clip_path --clip_l_path --comfy_path [--vae]
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

try:
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout
except ImportError:
    print("Error: comfy_kitchen not found (install in the active venv).")
    sys.exit(1)

_DEFAULT_GROUPSIZE = 256
_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def _default_repo_root() -> str:
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
# Hadamard / ConvRot core（native_convert_int8.py と同数学、自己完結）
# ---------------------------------------------------------------------------
def build_hadamard(
    size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
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
    if n < 4:
        return None
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def rotate_weight(weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int) -> torch.Tensor:
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(f"in_features {in_features} not divisible by group_size {group_size}")
    group_count = in_features // group_size
    weight_grouped = weight.view(out_features, group_count, group_size)
    return torch.matmul(
        weight_grouped, h_matrix.T.to(dtype=weight.dtype, device=weight.device)
    ).reshape(weight.shape)


# ---------------------------------------------------------------------------
# INT8 packing（保護層用）
# ---------------------------------------------------------------------------
def pack_channelwise(weight: torch.Tensor):
    w = weight.float()
    reduce_dims = tuple(range(1, w.dim()))
    amax = torch.clamp(w.abs().amax(dim=reduce_dims).reshape(-1), min=1e-6)
    scale = amax / 127.0
    scale_view = scale.view(-1, 1)
    amax_view = amax.view(-1, 1)
    clamped = torch.clamp(w, -amax_view, amax_view)
    q = (clamped / scale_view).round().clamp(-127, 127).to(torch.int8)
    return q, scale_view.to(dtype=torch.float32)


def pack_tensorwise(weight: torch.Tensor):
    w = weight.float()
    amax = max(float(w.abs().max().item()), 1e-6)
    scale = amax / 127.0
    q = (w / scale).round().clamp(-127, 127).to(torch.int8)
    return q, torch.tensor(scale, dtype=torch.float32)


def _encode_comfy_quant(config: dict) -> torch.Tensor:
    return torch.tensor(
        list(json.dumps(config, separators=(",", ":")).encode("utf-8")),
        dtype=torch.uint8,
    )


# ---------------------------------------------------------------------------
# flux1 構造ベースの INT8 保護層判定
# ---------------------------------------------------------------------------
def _is_int8_protect_key(key: str) -> bool:
    """構造ベースの感度仮説で INT8 保護する flux1 層を判定する。

    - adaLN modulation 系: img_mod.lin / txt_mod.lin / modulation.lin / adaLN_modulation
    - 入出力系: img_in / txt_in / time_in / vector_in / guidance_in / final_layer
    """
    base = key
    if base.startswith("model.diffusion_model."):
        base = base[len("model.diffusion_model."):]
    if ".img_mod.lin" in base or ".txt_mod.lin" in base:
        return True
    if ".modulation.lin" in base:  # single_blocks
        return True
    if "adaLN_modulation" in base:  # final_layer
        return True
    if base.startswith(
        ("img_in.", "txt_in.", "time_in.", "vector_in.", "guidance_in.", "final_layer.")
    ):
        return True
    return False


def is_flux1_matmul_weight(key: str, tensor: torch.Tensor) -> bool:
    if not key.startswith("model.diffusion_model."):
        return False
    if not key.endswith(".weight"):
        return False
    if tensor.ndim < 2:
        return False
    return tensor.dtype in (torch.float16, torch.float32, torch.bfloat16)


def _release_vram(label: str = "post-convert") -> None:
    print(f"[*] Releasing VRAM ({label})...")
    gc.collect()
    if not torch.cuda.is_available():
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


def run_post_convert_flux_bench(
    *,
    script_dir: str,
    fp16_path: str,
    out_path: str,
    clip_path: str,
    clip_l_path: str,
    comfy_path: str,
    vae_path: str | None = None,
) -> int:
    bench_script = os.path.join(script_dir, "benchmark", "flux1_nvfp4", "flux_int8_bench.py")
    if not os.path.isfile(bench_script):
        print(f"[FATAL] Post-convert bench script not found: {bench_script}")
        return 1
    if not os.path.isfile(fp16_path) or not os.path.isfile(out_path):
        print("[FATAL] Post-convert bench: fp16 / output missing")
        return 1
    if not clip_path or not os.path.isfile(clip_path):
        print("[FATAL] Post-convert bench: --clip_path missing")
        return 1
    if not clip_l_path or not os.path.isfile(clip_l_path):
        print("[FATAL] Post-convert bench: --clip_l_path missing")
        return 1
    if not comfy_path or not os.path.isdir(comfy_path):
        print("[FATAL] Post-convert bench: --comfy_path missing")
        return 1

    _release_vram("pre-flux bench subprocess")

    cmd = [
        sys.executable,
        bench_script,
        "--fp16", fp16_path,
        "--int8", out_path,
        "--clip_path", clip_path,
        "--clip_l_path", clip_l_path,
        "--comfy_path", comfy_path,
    ]
    if vae_path:
        cmd.extend(["--vae", vae_path])

    print("=" * 60)
    print("[*] Post-convert Flux Hybrid NVFP4 bench")
    print(f"    --fp16: {fp16_path}")
    print(f"    --int8(hybrid): {out_path}")
    print(f"    --clip_path: {clip_path}")
    print(f"    --clip_l_path: {clip_l_path}")
    print(f"    --comfy_path: {comfy_path}")
    print("=" * 60)
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def convert_to_hybrid_nvfp4(
    input_path: str,
    output_path: str,
    enable_convrot: bool = True,
    group_size: int = _DEFAULT_GROUPSIZE,
    quantize_device: str = "cuda",
    mode: str = "hybrid",
):
    device = quantize_device if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {input_path}")
    state_dict = load_file(input_path)

    new_state_dict = {}
    quant_meta_layers = {}
    n_int8_protect = 0
    n_convrot_nvfp4 = 0
    n_plain_nvfp4 = 0
    n_plain_int8 = 0
    kept_count = 0

    use_int8_protect = mode == "hybrid"
    print(
        f"Converting Flux1 DiT matmul weights: "
        f"mode={mode} (INT8 protect={'ON' if use_int8_protect else 'OFF'}) + NVFP4 "
        f"(ConvRot={'ON' if enable_convrot else 'OFF'}, groupsize={group_size})..."
    )

    for key, tensor in tqdm(state_dict.items()):
        if not is_flux1_matmul_weight(key, tensor):
            new_state_dict[key] = tensor
            kept_count += 1
            continue

        module_key = key[: -len(".weight")]
        w_fp = tensor.float().cpu()
        in_features = int(w_fp.shape[1])
        out_features = int(w_fp.shape[0])
        used_gs = (
            convrot_group_size_for_features(in_features, group_size)
            if enable_convrot
            else None
        )
        h_matrix = build_hadamard(used_gs, device="cpu", dtype=torch.float32) if used_gs else None

        if use_int8_protect and _is_int8_protect_key(key):
            # --- INT8 保護層（ConvRot 可能なら row-wise、不可なら tensorwise） ---
            w_for_q = w_fp
            if used_gs is not None and h_matrix is not None:
                w_for_q = rotate_weight(w_fp, h_matrix, used_gs)
                q, scale = pack_channelwise(w_for_q)
                quant_config = {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": int(used_gs),
                }
            else:
                q, scale = pack_tensorwise(w_fp)
                quant_config = {"format": "int8_tensorwise"}
                n_plain_int8 += 1
            new_state_dict[key] = q
            new_state_dict[f"{module_key}.weight_scale"] = scale
            new_state_dict[f"{module_key}.comfy_quant"] = _encode_comfy_quant(quant_config)
            quant_meta_layers[module_key] = dict(quant_config)
            n_int8_protect += 1
            continue

        # --- NVFP4 層 ---
        w_for_q = w_fp
        do_rotate = False
        if used_gs is not None and h_matrix is not None:
            w_for_q = rotate_weight(w_fp, h_matrix, used_gs)
            do_rotate = True

        w_bf16 = w_for_q.to(dtype=torch.bfloat16)
        qdata, params = TensorCoreNVFP4Layout.quantize(w_bf16)
        tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)
        for suffix, t_ in tensors.items():
            new_state_dict[f"{module_key}.weight{suffix}"] = t_.cpu()

        if do_rotate:
            quant_config = {
                "format": "nvfp4",
                "convrot": True,
                "convrot_groupsize": int(used_gs),
                "in_features": in_features,
                "out_features": out_features,
                "orig_shape": f"[{out_features}, {in_features}]",
            }
            n_convrot_nvfp4 += 1
        else:
            quant_config = {
                "format": "nvfp4",
                "in_features": in_features,
                "out_features": out_features,
                "orig_shape": f"[{out_features}, {in_features}]",
            }
            n_plain_nvfp4 += 1
        new_state_dict[f"{module_key}.comfy_quant"] = _encode_comfy_quant(quant_config)
        quant_meta_layers[module_key] = dict(quant_config)

        del w_for_q, qdata, params

    metadata = {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": quant_meta_layers}
        )
    }

    print(f"Saving to: {output_path}")
    print(f"INT8 protect (structural): {n_int8_protect}")
    print(f"NVFP4 ConvRot: {n_convrot_nvfp4}, NVFP4 plain: {n_plain_nvfp4}, INT8 plain: {n_plain_int8}")
    print(f"Kept non-matmul: {kept_count}")

    save_file(new_state_dict, output_path, metadata=metadata)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Flux1 DiT Hybrid NVFP4 convert (structural INT8 protect + NVFP4 ConvRot). "
            "native (hswq 非使用)。Bias Correction は対象外。"
        )
    )
    parser.add_argument("--model", "--input", dest="model", type=str, required=True,
                        help="Path to input Flux1 .safetensors")
    parser.add_argument("--output", type=str, required=True, help="Path to output .safetensors")
    parser.add_argument("--convrot", action=argparse.BooleanOptionalAction, default=True,
                        help="FULL ConvRot on Linear 2D. Default ON; --no-convrot for plain.")
    parser.add_argument("--groupsize", type=int, default=_DEFAULT_GROUPSIZE,
                        help=f"ConvRot Hadamard group size (power of 4, default {_DEFAULT_GROUPSIZE})")
    parser.add_argument("--quantize_device", type=str, default="cuda",
                        help="Device for NVFP4 quantize (default cuda; cpu も可)")
    parser.add_argument("--mode", type=str, choices=("hybrid", "native"), default="hybrid",
                        help="hybrid=構造ベース INT8 保護 + NVFP4（default）; native=全 Linear NVFP4（保護なし）")
    parser.add_argument("--clip_path", type=str, default=None, help="T5XXL path (required when --bench)")
    parser.add_argument("--clip_l_path", type=str, default=None, help="clip_l path (required when --bench)")
    parser.add_argument("--comfy_path", type=str, default=None, help="ComfyUI root (required when --bench)")
    parser.add_argument("--vae", type=str, default=None, help="Optional Flux VAE path for bench")
    parser.add_argument("--bench", action=argparse.BooleanOptionalAction, default=True,
                        help="After save, run benchmark/flux_int8_bench.py. Pass --no-bench to skip.")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)
    if args.groupsize < 4 or (args.groupsize & (args.groupsize - 1)) != 0 or math.log(args.groupsize, 4) % 1 != 0:
        print(f"Error: --groupsize must be a power of 4 (>=4), got {args.groupsize}")
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
            print("Error: post-convert bench requires " + ", ".join(missing) + " (pass --no-bench to skip)")
            sys.exit(1)
        for p, n in ((args.clip_path, "--clip_path"), (args.clip_l_path, "--clip_l_path")):
            if not os.path.isfile(p):
                print(f"Error: {n} not found: {p}")
                sys.exit(1)
        if not os.path.isdir(args.comfy_path):
            print(f"Error: --comfy_path not found: {args.comfy_path}")
            sys.exit(1)

    convert_to_hybrid_nvfp4(
        args.model,
        args.output,
        enable_convrot=bool(args.convrot),
        group_size=int(args.groupsize),
        quantize_device=args.quantize_device,
        mode=args.mode,
    )

    if args.bench:
        bench_rc = run_post_convert_flux_bench(
            script_dir=_REPO_ROOT,
            fp16_path=args.model,
            out_path=args.output,
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

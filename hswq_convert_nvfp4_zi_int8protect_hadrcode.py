"""Z-Image / ZIT NVFP4 + HARDCODED ConvRot INT8 protect (hadrcode variant).

Owner-allowed hardcode file. Does NOT call analyze/*.
Protect 60 keys = moodyRealMix_zitV7 protect60 swap3 (N=60 fixed).
  Source: test/_moodyRealMix_zitV7_protect60_swap3_keys.json
  Best artifact: moodyRealMix_zitV7_hadrcode_nvfp4_int8protect60_swap3.safetensors
  (MSE 0.0421 / SSIM 0.9720). Do NOT raise N. No PACKMSE, no Card1/bias.

Protect path:
  ConvRot rotate (W @ H^T) → row-wise INT8 + weight_scale + int8_tensorwise stamp
  in _quantization_metadata AND per-layer ``.comfy_quant`` (uint8 JSON; ComfyUI load).

Remaining Linear 2D: NVFP4 (+ FULL ConvRot by default) + same ``.comfy_quant``.
Kitchen Turbo blacklist: bfloat16 (unchanged).

Post-convert bench (default ON): after save, subprocess
  benchmark/zi_convrot_nvfp4_bench.py. Pass --no-bench to skip.

Example:
  python hswq_convert_nvfp4_zi_int8protect_hadrcode.py \\
    --model ... --output ... --device cuda
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
from collections import OrderedDict

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

try:
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout
except ImportError:
    print("Error: comfy_kitchen not found (install in the active venv).")
    sys.exit(1)

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# ConvRot / INT8 helpers (inlined — do NOT import native_convert_int8)
# ---------------------------------------------------------------------------
_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


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


def convrot_group_size_for_features(n: int, preferred: int = 256) -> int | None:
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


def quantize_int8_tensorwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Tensorwise INT8: scalar weight_scale (plain ComfyUI int8_tensorwise)."""
    amax = max(w.abs().max().item(), 1e-6)
    scale = torch.tensor(amax / 127.0, dtype=torch.float32)
    q = (w / scale.item()).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def quantize_int8_rowwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel INT8 for Linear: weight_scale [out, 1]."""
    abs_max = w.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
    scale = abs_max / 127.0
    q = (w / scale.to(dtype=w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, scale.to(dtype=torch.float32)


def _encode_comfy_quant(config: dict) -> torch.Tensor:
    """ComfyUI layer marker: uint8 JSON (same as convert_old_quants / int8_sdxl)."""
    return torch.tensor(
        list(json.dumps(config, separators=(",", ":")).encode("utf-8")),
        dtype=torch.uint8,
    )


_DEFAULT_GROUPSIZE = 256

# Kitchen model_type → (BLACKLIST, FP8_LAYERS) — Z-Image only
# (same strings as convert_to_nvfp4_node.py)
_Z_IMAGE_PROFILES: dict[str, tuple[list[str], list[str]]] = {
    "Z-Image-Turbo": (
        [
            "cap_embedder",
            "x_embedder",
            "noise_refiner",
            "context_refiner",
            "t_embedder",
            "final_layer",
        ],
        [],
    ),
    "Z-Image-Base": (
        [
            "attention",
            "adaLN_modulation",
            "norm",
            "final_layer",
            "cap_embedder",
            "x_embedder",
            "noise_refiner",
            "context_refiner",
            "t_embedder",
        ],
        [],
    ),
}

_DEFAULT_MODEL_TYPE = "Z-Image-Turbo"

# Owner-allowed hardcode: moodyRealMix_zitV7 protect N=60 (2026-08-02).
# Source: auto60 + drop 3 kurt-only adaLN + add 3 NVFP4-outside abs top.
# Keys JSON: test/_moodyRealMix_zitV7_protect60_swap3_keys.json
# N=60 fixed. No N raise.
_INT8_PROTECT_SOURCE = (
    "moodyRealMix_zitV7_nvfp4_int8protect60_swap3_kurtAdaLN_to_nvfp4Abs"
)
_INT8_PROTECT_KEYSET: frozenset[str] = frozenset(
    (
        "model.diffusion_model.layers.6.feed_forward.w2.weight",
        "model.diffusion_model.layers.4.feed_forward.w2.weight",
        "model.diffusion_model.layers.11.feed_forward.w2.weight",
        "model.diffusion_model.layers.7.feed_forward.w2.weight",
        "model.diffusion_model.layers.13.feed_forward.w2.weight",
        "model.diffusion_model.layers.12.feed_forward.w2.weight",
        "model.diffusion_model.layers.9.feed_forward.w2.weight",
        "model.diffusion_model.layers.10.feed_forward.w2.weight",
        "model.diffusion_model.layers.5.feed_forward.w2.weight",
        "model.diffusion_model.layers.14.feed_forward.w2.weight",
        "model.diffusion_model.layers.15.feed_forward.w2.weight",
        "model.diffusion_model.layers.18.feed_forward.w2.weight",
        "model.diffusion_model.layers.1.feed_forward.w2.weight",
        "model.diffusion_model.layers.28.adaLN_modulation.0.weight",
        "model.diffusion_model.layers.8.feed_forward.w2.weight",
        "model.diffusion_model.layers.19.feed_forward.w2.weight",
        "model.diffusion_model.layers.3.feed_forward.w2.weight",
        "model.diffusion_model.layers.2.feed_forward.w2.weight",
        "model.diffusion_model.layers.24.feed_forward.w1.weight",
        "model.diffusion_model.layers.29.attention.qkv.weight",
        "model.diffusion_model.layers.16.feed_forward.w1.weight",
        "model.diffusion_model.layers.20.feed_forward.w2.weight",
        "model.diffusion_model.layers.16.feed_forward.w2.weight",
        "model.diffusion_model.layers.0.feed_forward.w2.weight",
        "model.diffusion_model.layers.17.feed_forward.w1.weight",
        "model.diffusion_model.layers.13.attention.out.weight",
        "model.diffusion_model.layers.19.feed_forward.w3.weight",
        "model.diffusion_model.layers.29.adaLN_modulation.0.weight",
        "model.diffusion_model.layers.23.feed_forward.w1.weight",
        "model.diffusion_model.layers.23.feed_forward.w2.weight",
        "model.diffusion_model.layers.25.feed_forward.w2.weight",
        "model.diffusion_model.layers.26.feed_forward.w3.weight",
        "model.diffusion_model.layers.19.feed_forward.w1.weight",
        "model.diffusion_model.layers.28.feed_forward.w3.weight",
        "model.diffusion_model.layers.17.feed_forward.w2.weight",
        "model.diffusion_model.layers.22.feed_forward.w1.weight",
        "model.diffusion_model.layers.22.feed_forward.w2.weight",
        "model.diffusion_model.layers.21.feed_forward.w1.weight",
        "model.diffusion_model.layers.18.feed_forward.w1.weight",
        "model.diffusion_model.layers.28.attention.qkv.weight",
        "model.diffusion_model.layers.11.attention.out.weight",
        "model.diffusion_model.layers.10.attention.qkv.weight",
        "model.diffusion_model.layers.13.feed_forward.w3.weight",
        "model.diffusion_model.layers.27.attention.qkv.weight",
        "model.diffusion_model.layers.12.attention.out.weight",
        "model.diffusion_model.layers.9.attention.qkv.weight",
        "model.diffusion_model.layers.16.attention.qkv.weight",
        "model.diffusion_model.layers.14.attention.out.weight",
        "model.diffusion_model.layers.28.feed_forward.w2.weight",
        "model.diffusion_model.layers.9.attention.out.weight",
        "model.diffusion_model.layers.3.feed_forward.w3.weight",
        "model.diffusion_model.layers.25.feed_forward.w1.weight",
        "model.diffusion_model.layers.24.feed_forward.w3.weight",
        "model.diffusion_model.layers.11.attention.qkv.weight",
        "model.diffusion_model.layers.26.feed_forward.w1.weight",
        "model.diffusion_model.layers.8.attention.qkv.weight",
        "model.diffusion_model.layers.24.feed_forward.w2.weight",
        "model.diffusion_model.layers.25.feed_forward.w3.weight",
        "model.diffusion_model.layers.19.attention.qkv.weight",
        "model.diffusion_model.layers.21.feed_forward.w2.weight",
    )
)


def _resolve_int8_protect_keyset(
    int8_protect_keys: frozenset[str] | list[str] | None,
    int8_protect_source: str | None,
) -> tuple[frozenset[str], str]:
    if int8_protect_keys is not None:
        keyset = frozenset(int8_protect_keys)
        if not keyset:
            raise ValueError("int8_protect_keys is empty")
        source = int8_protect_source or "injected_keyset"
        return keyset, source
    # Default: baked hardcode (this file only; owner-allowed).
    source = int8_protect_source or _INT8_PROTECT_SOURCE
    return frozenset(_INT8_PROTECT_KEYSET), source


def _is_int8_protect_key(key: str, keyset: frozenset[str]) -> bool:
    """True if key is in analysis INT8 protect set (exact or prefix variants)."""
    if key in keyset:
        return True
    if key.startswith("diffusion_model."):
        alt = "model." + key
        if alt in keyset:
            return True
    if not key.startswith("model.diffusion_model."):
        alt = "model.diffusion_model." + key
        if alt in keyset:
            return True
    return False


_NON_DIFFUSION_MARKERS: tuple[str, ...] = (
    "conditioner.",
    "cond_stage_model.",
    "text_encoders.",
    "text_encoder.",
    "text_encoder_2.",
    "text_encoder_3.",
    "text_model.",
    "text_projection",
    "logit_scale",
    "clip_l.",
    "clip_g.",
    "t5xxl.",
    "first_stage_model.",
    "vae.",
)


def _is_non_diffusion_key(key: str) -> bool:
    return any(marker in key for marker in _NON_DIFFUSION_MARKERS)


def _find_z_image_key_prefix(state_dict) -> str:
    """Lumina2 / NextDiT / Z-Image signature (ComfyUI model_detection).

    Requires cap_embedder.1.weight and noise_refiner.0 attention
    (k_norm or fused qkv) under a known diffusion prefix.
    """
    for prefix in ("model.diffusion_model.", "diffusion_model.", ""):
        cap = f"{prefix}cap_embedder.1.weight"
        if cap not in state_dict:
            continue
        k_norm = f"{prefix}noise_refiner.0.attention.k_norm.weight"
        qkv = f"{prefix}noise_refiner.0.attention.qkv.weight"
        if k_norm in state_dict or qkv in state_dict:
            return prefix
    raise ValueError(
        "Not a Z-Image / ZIT (NextDiT / Lumina2) checkpoint: missing "
        "cap_embedder.1.weight + noise_refiner.0.attention.(k_norm|qkv).weight "
        "(under model.diffusion_model. / diffusion_model. / root)."
    )


def _meta_base_key(base_k_file: str) -> str:
    if "model.diffusion_model." in base_k_file:
        return base_k_file.split("model.diffusion_model.")[-1]
    if "diffusion_model." in base_k_file:
        return base_k_file.split("diffusion_model.")[-1]
    return base_k_file


def convert_to_nvfp4(
    input_path: str,
    output_path: str,
    device: str,
    model_type: str = _DEFAULT_MODEL_TYPE,
    enable_convrot: bool = True,
    group_size: int = _DEFAULT_GROUPSIZE,
    int8_protect_keys: frozenset[str] | list[str] | None = None,
    int8_protect_source: str | None = None,
):
    if model_type not in _Z_IMAGE_PROFILES:
        raise ValueError(
            f"Unknown model_type={model_type!r}; "
            f"choose from {sorted(_Z_IMAGE_PROFILES)}"
        )
    blacklist, fp8_layers = _Z_IMAGE_PROFILES[model_type]
    protect_keyset, protect_source = _resolve_int8_protect_keyset(
        int8_protect_keys, int8_protect_source
    )

    rot_tag = "FULL ConvRot NVFP4" if enable_convrot else "plain NVFP4"
    print(
        f"Mode {model_type} | device={device} | {rot_tag} "
        f"+ ConvRot INT8 protect ({len(protect_keyset)} keys)"
    )
    print(
        f"  [INT8 protect] {len(protect_keyset)} keys from {protect_source} → "
        "ConvRot INT8 (rowwise)"
    )
    if enable_convrot:
        print(
            f"  [ConvRot] ON | preferred groupsize={int(group_size)} "
            f"(Linear 2D; skip rotate when in_features has no power-of-4 group)"
        )
    else:
        print("  [ConvRot] OFF | plain Kitchen NVFP4 packs only")

    sd = load_file(input_path)
    prefix = _find_z_image_key_prefix(sd)
    print(f"Detected Z-Image key prefix: {prefix!r}")

    # Structural summary (helps audit Turbo vs Base choice)
    n_layers = sum(
        1
        for k in sd
        if k.startswith(f"{prefix}layers.") and k.endswith(".feed_forward.w1.weight")
    )
    has_noise = any(f"{prefix}noise_refiner." in k for k in sd)
    has_ctx = any(f"{prefix}context_refiner." in k for k in sd)
    print(
        f"Structure: layers(w1)={n_layers} "
        f"noise_refiner={has_noise} context_refiner={has_ctx}"
    )
    if model_type == "Z-Image-Base":
        print(
            "[!] Z-Image-Base Kitchen blacklist also matches layers.*.attention / "
            "adaLN_modulation / norm — NVFP4 candidates shrink to feed_forward "
            "2D weights mainly. ZIT / Turbo UNets usually want Z-Image-Turbo."
        )

    quant_map = {"format_version": "1.0", "layers": {}}
    new_sd: dict[str, torch.Tensor] = {}
    n_nvfp4 = 0
    n_convrot = 0
    n_plain_nvfp4 = 0
    n_bf16 = 0
    n_int8_protect = 0
    n_int8_convrot = 0
    n_int8_plain = 0

    print(f"Converting ({len(sd)} tensors)...")
    for k, v in tqdm(list(sd.items())):
        if any(name in k for name in blacklist):
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16 += 1
            continue

        if _is_non_diffusion_key(k):
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16 += 1
            continue

        # Analysis ConvRot INT8 protect (before NVFP4) — injected keyset
        if _is_int8_protect_key(k, protect_keyset) and v.ndim == 2 and ".weight" in k:
            base_k_file = k.replace(".weight", "")
            base_k_meta = _meta_base_key(base_k_file)
            w = v.float().cpu()
            used_gs = convrot_group_size_for_features(
                int(w.shape[1]), int(group_size)
            )
            if used_gs is not None:
                h_matrix = build_hadamard(
                    int(used_gs), device="cpu", dtype=torch.float32
                )
                w = rotate_weight(w, h_matrix, int(used_gs))
                q, scale = quantize_int8_rowwise(w)
                quant_config = {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": int(used_gs),
                }
                n_int8_convrot += 1
            else:
                q, scale = quantize_int8_tensorwise(w)
                quant_config = {"format": "int8_tensorwise"}
                n_int8_plain += 1
            new_sd[k] = q
            new_sd[f"{base_k_file}.weight_scale"] = scale
            # ComfyUI load peeks ``{prefix}comfy_quant`` next to the weight.
            new_sd[f"{base_k_file}.comfy_quant"] = _encode_comfy_quant(quant_config)
            quant_map["layers"][base_k_meta] = dict(quant_config)
            n_int8_protect += 1
            continue

        if v.ndim == 2 and ".weight" in k:
            base_k_file = k.replace(".weight", "")
            base_k_meta = _meta_base_key(base_k_file)
            v_tensor = v.to(device=device, dtype=torch.bfloat16)

            if fp8_layers and any(name in k for name in fp8_layers):
                import comfy_kitchen as ck

                weight_scale = (
                    (v_tensor.abs().max() / 448.0).clamp(min=1e-12).float()
                )
                weight_quantized = ck.quantize_per_tensor_fp8(v_tensor, weight_scale)
                new_sd[k] = weight_quantized.cpu()
                new_sd[f"{base_k_file}.weight_scale"] = weight_scale.to(
                    torch.bfloat16
                ).cpu()
                quant_map["layers"][base_k_meta] = {"format": "float8_e4m3fn"}
                if device == "cuda":
                    del v_tensor
                continue

            used_gs = None
            do_rotate = False
            w_for_q = v_tensor
            if enable_convrot:
                used_gs = convrot_group_size_for_features(
                    int(v_tensor.shape[1]), int(group_size)
                )
                if used_gs is not None:
                    h_matrix = build_hadamard(
                        int(used_gs), device="cpu", dtype=torch.float32
                    )
                    w_rot = rotate_weight(
                        v_tensor.float().cpu(), h_matrix, int(used_gs)
                    )
                    w_for_q = w_rot.to(device=device, dtype=torch.bfloat16)
                    do_rotate = True

            try:
                qdata, params = TensorCoreNVFP4Layout.quantize(w_for_q)
                tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)
                for suffix, tensor in tensors.items():
                    new_sd[f"{base_k_file}.weight{suffix}"] = tensor.cpu()
                if do_rotate and used_gs is not None:
                    quant_config = {
                        "format": "nvfp4",
                        "convrot": True,
                        "convrot_groupsize": int(used_gs),
                    }
                    n_convrot += 1
                else:
                    quant_config = {"format": "nvfp4"}
                    n_plain_nvfp4 += 1
                new_sd[f"{base_k_file}.comfy_quant"] = _encode_comfy_quant(
                    quant_config
                )
                quant_map["layers"][base_k_meta] = dict(quant_config)
                n_nvfp4 += 1

            except Exception:
                new_sd[k] = v.to(dtype=torch.bfloat16)
                n_bf16 += 1

            if device == "cuda":
                if do_rotate:
                    del w_for_q
                del v_tensor
        else:
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16 += 1

    final_metadata = OrderedDict()
    final_metadata["_quantization_metadata"] = json.dumps(quant_map)
    by_tag = (
        "ComfyUI Kitchen NVFP4 Converter (Z-Image ConvRot + INT8 protect)"
        if enable_convrot
        else "ComfyUI Kitchen NVFP4 Converter (Z-Image INT8 protect)"
    )
    final_metadata["converted_by"] = by_tag
    final_metadata["converter_url"] = (
        "https://github.com/tritant/ComfyUI_Kitchen_nvfp4_Converter"
    )
    final_metadata["hswq_model"] = "z_image"
    final_metadata["hswq_kitchen_profile"] = model_type
    final_metadata["hswq_nvfp4_convrot"] = "1" if enable_convrot else "0"
    final_metadata["hswq_int8_protect"] = "1"
    final_metadata["hswq_int8_protect_n"] = str(n_int8_protect)
    final_metadata["hswq_int8_protect_convrot"] = str(n_int8_convrot)
    final_metadata["hswq_int8_protect_source"] = protect_source

    print(f"Saving | Type: {model_type} | Path: {output_path}")
    save_file(new_sd, output_path, metadata=final_metadata)
    total_bytes = os.path.getsize(output_path)
    print(f"Done. Size: {round(total_bytes / (1024**3), 2)} GiB")
    print(f"NVFP4+INT8 layers in metadata: {len(quant_map['layers'])}")
    print(
        f"  counted nvfp4 packs={n_nvfp4} | bf16 keep tensors={n_bf16} | "
        f"int8 protect={n_int8_protect} "
        f"(convrot={n_int8_convrot}, plain={n_int8_plain})"
    )
    print(f"FULL ConvRot enabled (NVFP4 path): {enable_convrot}")
    if enable_convrot:
        print(
            f"  ConvRot NVFP4 Linear: {n_convrot}, "
            f"plain NVFP4 (no group): {n_plain_nvfp4}"
        )

    del sd
    del new_sd
    del quant_map
    _release_vram("after native Z-Image NVFP4 convert save")


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


# Exact --prompt / --steps from zi_convrot_nvfp4_bench.py Example (fixed; not CLI).
_FIXED_ZI_CONVROT_BENCH_PROMPT = (
    "A beautiful cyberpunk city at night, high detail."
)
_FIXED_ZI_CONVROT_BENCH_STEPS = 25
# Seed fixed inside this chain (not a parent CLI). Same default as
# benchmark/zi_convrot_nvfp4_bench.py --seed (must be passed explicitly).
_FIXED_ZI_CONVROT_BENCH_SEED = 42


def run_post_convert_zi_convrot_nvfp4_bench(
    *,
    script_dir: str,
    fp16_path: str,
    nvfp4_path: str,
    clip_path: str,
    comfy_path: str,
    vae_path: str | None = None,
    token: str | None = None,
) -> int:
    """After save: subprocess benchmark/zi_convrot_nvfp4_bench.py.

    Owner body argv order + seed fixed inside:
      --fp16 --nvfp4 --clip_path --comfy_path
      [--vae] [--token] --prompt --steps 25 --seed <fixed>
    """
    bench_script = os.path.join(
        script_dir, "benchmark", "zi_convrot_nvfp4_bench.py"
    )
    if not os.path.isfile(bench_script):
        print(f"[FATAL] Post-convert bench script not found: {bench_script}")
        return 1
    if not os.path.isfile(fp16_path):
        print(
            f"[FATAL] Post-convert bench: FP16 (--model) missing: {fp16_path}"
        )
        return 1
    if not os.path.isfile(nvfp4_path):
        print(
            f"[FATAL] Post-convert bench: NVFP4 (--output) missing: {nvfp4_path}"
        )
        return 1
    if not clip_path or not os.path.isfile(clip_path):
        print(
            f"[FATAL] Post-convert bench: --clip_path missing: {clip_path}"
        )
        return 1
    if not comfy_path or not os.path.isdir(comfy_path):
        print(
            f"[FATAL] Post-convert bench: --comfy_path missing: {comfy_path}"
        )
        return 1
    if vae_path and not os.path.isfile(vae_path):
        print(f"[FATAL] Post-convert bench: --vae missing: {vae_path}")
        return 1

    _release_vram("pre-zi_convrot_nvfp4_bench subprocess")

    # Owner body order (bench body untouched).
    cmd = [
        sys.executable,
        bench_script,
        "--fp16",
        fp16_path,
        "--nvfp4",
        nvfp4_path,
        "--clip_path",
        clip_path,
        "--comfy_path",
        comfy_path,
    ]
    if vae_path:
        cmd.extend(["--vae", vae_path])
    if token:
        cmd.extend(["--token", token])
    cmd.extend(
        [
            "--prompt",
            _FIXED_ZI_CONVROT_BENCH_PROMPT,
            "--steps",
            str(_FIXED_ZI_CONVROT_BENCH_STEPS),
            "--seed",
            str(_FIXED_ZI_CONVROT_BENCH_SEED),
        ]
    )

    print("=" * 60)
    print("[*] Post-convert ZI ConvRot NVFP4 bench (owner body shape)")
    print(f"    script: {bench_script}")
    print(f"    --fp16: {fp16_path}")
    print(f"    --nvfp4: {nvfp4_path}")
    print(f"    --clip_path: {clip_path}")
    print(f"    --comfy_path: {comfy_path}")
    if vae_path:
        print(f"    --vae: {vae_path}")
    if token:
        print("    --token: (provided)")
    print(f"    --prompt: {_FIXED_ZI_CONVROT_BENCH_PROMPT}")
    print(f"    --steps: {_FIXED_ZI_CONVROT_BENCH_STEPS}")
    print(f"    --seed: {_FIXED_ZI_CONVROT_BENCH_SEED} (fixed inside)")
    print("=" * 60)
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Z-Image / ZIT NVFP4 + analysis ConvRot INT8 protect (int8protect). "
            "Based on native_convert_nvfp4_zi.py; 60 ranked Linear weights as "
            "ConvRot INT8 (65-order head); rest NVFP4. FULL ConvRot ON by "
            "default for NVFP4. Default Kitchen profile Z-Image-Turbo. "
            "Post-convert zi_convrot_nvfp4_bench default ON."
        )
    )
    parser.add_argument(
        "--model",
        "--input",
        dest="model",
        type=str,
        required=True,
        help="Path to Z-Image / ZIT BF16/FP16 .safetensors",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output .safetensors"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=_DEFAULT_MODEL_TYPE,
        choices=sorted(_Z_IMAGE_PROFILES.keys()),
        help=(
            "Kitchen Z-Image profile (default: Z-Image-Turbo; "
            "use Z-Image-Base only for Kitchen Base blacklist)"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Quantize device",
    )
    parser.add_argument(
        "--no-convrot",
        dest="enable_convrot",
        action="store_false",
        help="Disable ConvRot; pack plain Kitchen NVFP4 only.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=_DEFAULT_GROUPSIZE,
        help=f"Preferred ConvRot Hadamard group size (default {_DEFAULT_GROUPSIZE}).",
    )
    parser.add_argument(
        "--clip_path",
        type=str,
        default=None,
        help="Qwen3-4B text encoder path (post-convert bench)",
    )
    parser.add_argument(
        "--comfy_path",
        type=str,
        default=None,
        help="ComfyUI root path (post-convert bench)",
    )
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
        help=(
            "Optional VAE path for post-convert bench "
            "(forwarded as --vae when provided)"
        ),
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face token for post-convert bench "
            "(forwarded as --token when provided)"
        ),
    )
    parser.add_argument(
        "--bench",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After save, run benchmark/zi_convrot_nvfp4_bench.py "
            "(fp16/nvfp4/clip/comfy/prompt/steps=25; "
            "optional --vae/--token when provided). "
            "Pass --no-bench to skip."
        ),
    )
    parser.set_defaults(enable_convrot=True)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    # Hardcode only — no analyze/, no --keys-json.
    keyset = frozenset(_INT8_PROTECT_KEYSET)
    source = _INT8_PROTECT_SOURCE
    print(f"[hardcode] INT8 protect n={len(keyset)} source={source}")
    if len(keyset) != 60:
        print(f"Error: hardcode keyset must be 60, got {len(keyset)}")
        sys.exit(1)

    convert_to_nvfp4(
        args.model,
        args.output,
        device=str(args.device),
        model_type=str(args.model_type),
        enable_convrot=bool(args.enable_convrot),
        group_size=int(args.group_size),
        int8_protect_keys=keyset,
        int8_protect_source=source,
    )

    if args.bench:
        bench_rc = run_post_convert_zi_convrot_nvfp4_bench(
            script_dir=os.path.dirname(os.path.abspath(__file__)),
            fp16_path=args.model,
            nvfp4_path=args.output,
            clip_path=args.clip_path,
            comfy_path=args.comfy_path,
            vae_path=args.vae,
            token=args.token,
        )
        if bench_rc != 0:
            print(f"[FATAL] Post-convert bench exited with code {bench_rc}")
            sys.exit(bench_rc)
    else:
        print("[*] Post-convert bench skipped (--no-bench)")

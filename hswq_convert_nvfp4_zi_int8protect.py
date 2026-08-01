"""Z-Image / ZIT NVFP4 + analysis ConvRot INT8 protect (int8protect variant).

NEW FILE based on native_convert_nvfp4_zi_fp16safe.py / native_convert_nvfp4_zi.py
(do not edit the base converters).

Protect path (60 keys = 65-order head: prior 31 + abs_max fill, truncate tail):
  ConvRot rotate (W @ H^T) → row-wise INT8 + weight_scale + int8_tensorwise stamp
  in _quantization_metadata AND per-layer ``.comfy_quant`` (uint8 JSON; ComfyUI load).

Remaining Linear 2D: NVFP4 (+ FULL ConvRot by default) + same ``.comfy_quant``.
Kitchen Turbo blacklist: bfloat16 (unchanged).

Scheme is fixed: ConvRot NVFP4 + 60× ConvRot INT8 protect. This converter only
adds the ComfyUI-required ``.comfy_quant`` tensors next to each packed weight
(same shape as convert_old_quants / native_convert_int8_sdxl).

Key source:
  test/_moodyProMix_zitV13_nvfp4_int8protect60_final_keys.json

Example:
  python hswq_convert_nvfp4_zi_int8protect.py --model ... --output ...
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
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


# ---------------------------------------------------------------------------
# DualMonitor (Card 1) — signed per-in-channel E[x] for bias fold
# ---------------------------------------------------------------------------
class DualMonitor:
    """Per-layer act moments for Card 1 bias correction."""

    def __init__(self):
        self.output_sum = 0.0
        self.output_sq_sum = 0.0
        self.count = 0
        self.channel_importance = None
        self.channel_act_mean = None

    def update(self, input_tensor, output_tensor, module=None):
        with torch.no_grad():
            out_detached = output_tensor.detach().float()
            out_clamped = torch.clamp(out_detached, -65504.0, 65504.0)
            mean_val = out_clamped.mean().item()
            sq_mean_val = (out_clamped ** 2).mean().item()
            if math.isfinite(mean_val) and math.isfinite(sq_mean_val):
                self.output_sum += mean_val
                self.output_sq_sum += sq_mean_val
            inp = input_tensor.detach().float()
            is_conv2d = isinstance(module, torch.nn.Conv2d)
            if is_conv2d and inp.dim() == 4:
                reduce_dims = (0, 2, 3)
            elif inp.dim() >= 2:
                reduce_dims = tuple(range(inp.dim() - 1))
            else:
                current_imp = torch.ones(1, device=inp.device, dtype=torch.float32)
                current_act = torch.zeros(1, device=inp.device, dtype=torch.float32)
                reduce_dims = None
            if reduce_dims is not None:
                current_imp = inp.abs().mean(dim=reduce_dims)
                current_act = inp.mean(dim=reduce_dims)
            if self.channel_importance is None:
                self.channel_importance = current_imp
                self.channel_act_mean = current_act
            elif current_imp.shape == self.channel_importance.shape:
                c = self.count
                self.channel_importance = (
                    self.channel_importance * c + current_imp
                ) / (c + 1)
                self.channel_act_mean = (
                    self.channel_act_mean * c + current_act
                ) / (c + 1)
            self.count += 1


_dual_monitors: dict[str, DualMonitor] = {}


def _hook_fn(module, input, output, name):
    if name not in _dual_monitors:
        _dual_monitors[name] = DualMonitor()
    _dual_monitors[name].update(input[0], output, module)


def compute_nvfp4_bias_delta(weight_fp, weight_dq, act_mean):
    """Card 1: delta ≈ (W_q - W) contracted with per-in-channel E[x]."""
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


def _dequant_nvfp4(qdata: torch.Tensor, params) -> torch.Tensor:
    full = TensorCoreNVFP4Layout.dequantize(qdata, params)
    orig = tuple(params.orig_shape)
    if tuple(full.shape) != orig:
        return full[tuple(slice(0, s) for s in orig)]
    return full


def run_card1_calib(
    model_path: str,
    clip_path: str,
    comfy_path: str,
    calib_file: str,
    num_samples: int = 32,
    num_inference_steps: int = 25,
    tokenizer_path: str | None = None,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """FP16 NextDiT calib with DualMonitor on Linear; returns act_mean by module name.

    Module name keys match ``_meta_base_key`` (e.g. ``layers.0.attention.to_q``).
    TE path mirrors ``benchmark/zi_nvfp4_bench.py`` (Qwen3_4B + Qwen2Tokenizer).
    """
    global _dual_monitors
    _dual_monitors = {}

    from benchmark.zi_nvfp4_bench import (
        encode_prompt,
        load_zit_model,
        resolve_path,
        resolve_tokenizer_offline,
        run_inference,
        setup_comfy,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Card1] Loading FP16 ZI for DualMonitor from {model_path}")
    setup_comfy(comfy_path)

    # qk_norm Lumina calls ck.rms_rope; older kitchen wheels lack it
    # (same as benchmark/zi_nvfp4_bench.py after setup_comfy).
    from kitchen_rms_rope_fallback import ensure_kitchen_rms_rope

    ensure_kitchen_rms_rope()

    from comfy.text_encoders import llama as llama_module
    from transformers import Qwen2Tokenizer
    import comfy.ops

    tok_resolved = resolve_tokenizer_offline(tokenizer_path, comfy_path)
    if tok_resolved:
        print(f"[Card1] Tokenizer (disk): {tok_resolved}")
        try:
            tokenizer = Qwen2Tokenizer.from_pretrained(
                tok_resolved, local_files_only=True
            )
        except Exception as e:
            print(f"[Card1] local_files_only failed ({e}); retrying...")
            tokenizer = Qwen2Tokenizer.from_pretrained(tok_resolved)
    else:
        mid = tokenizer_path if tokenizer_path else "Qwen/Qwen2.5-7B-Instruct"
        print(f"[Card1] Tokenizer repo id: {mid}")
        tokenizer = Qwen2Tokenizer.from_pretrained(mid, local_files_only=True)

    resolved_clip = resolve_path(clip_path, is_file=True)
    text_encoder = llama_module.Qwen3_4B(
        config_dict={},
        device=device,
        dtype=torch.float16,
        operations=comfy.ops.disable_weight_init,
    ).to(device)
    print(f"[Card1] Loading CLIP weights from: {resolved_clip}")
    text_encoder.load_state_dict(load_file(resolved_clip), strict=False)
    text_encoder.eval()

    model, _n_cq, _is_za = load_zit_model(
        model_path, device, comfy_path, is_nvfp4=False
    )
    model.eval()

    hooks = []
    n_lin = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(
                module.register_forward_hook(
                    lambda m, i, o, n=name: _hook_fn(m, i, o, n)
                )
            )
            n_lin += 1
    print(f"[Card1] DualMonitor hooks on {n_lin} Linear modules")

    with open(calib_file, "r", encoding="utf-8") as f:
        prompts = [ln.strip() for ln in f if ln.strip()]
    prompts = prompts[: max(1, int(num_samples))]
    print(
        f"[Card1] Calibrating {len(prompts)} prompts, "
        f"{num_inference_steps} steps each..."
    )

    with torch.no_grad():
        for i, prompt in enumerate(tqdm(prompts, desc="Card1 calib")):
            cond, mask = encode_prompt(prompt, text_encoder, tokenizer, device)
            # ZI Turbo: same signature as benchmark/zi_nvfp4_bench.run_inference
            # (cond-only; no CFG guidance / uncond kwargs).
            run_inference(
                model,
                cond,
                mask,
                int(num_inference_steps),
                int(seed) + i,
                device,
            )
            del cond, mask
            if device == "cuda":
                torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    act_mean_dict: dict[str, torch.Tensor] = {}
    for name, mon in _dual_monitors.items():
        if mon.channel_act_mean is not None:
            act_mean_dict[name] = mon.channel_act_mean.detach().cpu().float()
    print(f"[Card1] Collected act_mean for {len(act_mean_dict)} layers")

    del model, text_encoder, tokenizer
    _dual_monitors = {}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return act_mean_dict


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

# Analysis ConvRot INT8 protect (moodyProMix_zitV13; Kitchen Turbo candidates).
# Same 65 order; truncate tail only → 60 (drop last 5 abs_max-fill).
# Source: test/_moodyProMix_zitV13_nvfp4_int8protect60_final_keys.json
_INT8_PROTECT_KEYSET: frozenset[str] = frozenset(
    {
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
        "model.diffusion_model.layers.27.adaLN_modulation.0.weight",
        "model.diffusion_model.layers.24.adaLN_modulation.0.weight",
        "model.diffusion_model.layers.26.adaLN_modulation.0.weight",
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
        "model.diffusion_model.layers.25.feed_forward.w1.weight",
        "model.diffusion_model.layers.3.feed_forward.w3.weight",
        "model.diffusion_model.layers.24.feed_forward.w3.weight",
        "model.diffusion_model.layers.11.attention.qkv.weight",
        "model.diffusion_model.layers.26.feed_forward.w1.weight",
        "model.diffusion_model.layers.8.attention.qkv.weight",
        "model.diffusion_model.layers.24.feed_forward.w2.weight",
    }
)


def _is_int8_protect_key(key: str) -> bool:
    """True if key is in analysis INT8 protect set (exact or prefix variants)."""
    if key in _INT8_PROTECT_KEYSET:
        return True
    if key.startswith("diffusion_model."):
        alt = "model." + key
        if alt in _INT8_PROTECT_KEYSET:
            return True
    if not key.startswith("model.diffusion_model."):
        alt = "model.diffusion_model." + key
        if alt in _INT8_PROTECT_KEYSET:
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
    bias_correction: bool = False,
    calib_file: str | None = None,
    clip_path: str | None = None,
    comfy_path: str | None = None,
    tokenizer_path: str | None = None,
    num_calib_samples: int = 32,
    num_inference_steps: int = 25,
):
    if model_type not in _Z_IMAGE_PROFILES:
        raise ValueError(
            f"Unknown model_type={model_type!r}; "
            f"choose from {sorted(_Z_IMAGE_PROFILES)}"
        )
    blacklist, fp8_layers = _Z_IMAGE_PROFILES[model_type]

    rot_tag = "FULL ConvRot NVFP4" if enable_convrot else "plain NVFP4"
    print(
        f"Mode {model_type} | device={device} | {rot_tag} "
        f"+ ConvRot INT8 protect ({len(_INT8_PROTECT_KEYSET)} keys)"
    )
    print(
        f"  [INT8 protect] {len(_INT8_PROTECT_KEYSET)} analysis keys → "
        "ConvRot INT8 (rowwise)"
    )
    if enable_convrot:
        print(
            f"  [ConvRot] ON | preferred groupsize={int(group_size)} "
            f"(Linear 2D; skip rotate when in_features has no power-of-4 group)"
        )
    else:
        print("  [ConvRot] OFF | plain Kitchen NVFP4 packs only")

    act_mean_dict: dict[str, torch.Tensor] = {}
    bias_corr_pending: dict[str, torch.Tensor] = {}
    n_bias_corr = 0
    bias_corr_skipped_no_act = 0
    bias_corr_skipped_bad_shape = 0
    bias_corr_skipped_no_bias = 0
    if bias_correction:
        if not calib_file:
            raise ValueError("--bias_correction requires --calib_file")
        if not clip_path:
            raise ValueError("--bias_correction requires --clip_path")
        if not comfy_path:
            raise ValueError("--bias_correction requires --comfy_path")
        if device != "cuda":
            raise ValueError("--bias_correction requires --device cuda")
        print("\n[Card 1] Running DualMonitor calibration (FP16 acts)...")
        act_mean_dict = run_card1_calib(
            model_path=input_path,
            clip_path=clip_path,
            comfy_path=comfy_path,
            calib_file=calib_file,
            tokenizer_path=tokenizer_path,
            num_samples=num_calib_samples,
            num_inference_steps=num_inference_steps,
        )
        print(
            f"[Card 1] act_mean for {len(act_mean_dict)} Linear modules "
            f"(keyed by module name = _meta_base_key)"
        )
        if enable_convrot:
            print(
                "[Card 1] WARN: ConvRot ON — DualMonitor acts are pre-rotation; "
                "bias uses post-rotation w_for_q vs W_q (same as Krea2)."
            )

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

        # Analysis ConvRot INT8 protect (before NVFP4) — 65 keys
        if _is_int8_protect_key(k) and v.ndim == 2 and ".weight" in k:
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

                # Card 1: accumulate bias delta while w_for_q (pre-quant) still lives.
                if bias_correction and act_mean_dict is not None:
                    act_mean = act_mean_dict.get(base_k_meta)
                    if act_mean is None:
                        bias_corr_skipped_no_act += 1
                    else:
                        weight_dq = _dequant_nvfp4(qdata, params)
                        if weight_dq is None:
                            bias_corr_skipped_bad_shape += 1
                        else:
                            delta = compute_nvfp4_bias_delta(
                                w_for_q.float(), weight_dq, act_mean
                            )
                            if delta is None:
                                bias_corr_skipped_bad_shape += 1
                            else:
                                bias_corr_pending[base_k_file] = (
                                    (-delta).detach().float().cpu()
                                )
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

    if bias_correction and bias_corr_pending:
        print(
            f"\n[Bias Correction] Applying deltas to {len(bias_corr_pending)} "
            f"NVFP4 Linear layers (Card 1)..."
        )
        for base_k_file, delta in bias_corr_pending.items():
            bias_key = f"{base_k_file}.bias"
            if bias_key not in new_sd:
                bias_corr_skipped_no_bias += 1
                continue
            bias = new_sd[bias_key]
            corrected = bias.float() + delta.to(
                device=bias.device, dtype=torch.float32
            )
            new_sd[bias_key] = corrected.to(dtype=bias.dtype)
            n_bias_corr += 1
        print(
            f"  [Bias Correction] applied={n_bias_corr}, "
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
    final_metadata["hswq_int8_protect_source"] = (
        "moodyProMix_zitV13_nvfp4_int8protect60_final_keys"
    )

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Z-Image / ZIT NVFP4 + analysis ConvRot INT8 protect (int8protect). "
            "Based on native_convert_nvfp4_zi.py; 60 ranked Linear weights as "
            "ConvRot INT8 (65-order head); rest NVFP4. FULL ConvRot ON by "
            "default for NVFP4. Default Kitchen profile Z-Image-Turbo."
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
        "--bias_correction",
        action="store_true",
        help=(
            "Enable Card 1: DualMonitor act_mean calib + bias += -(W_q-W)@mu_x. "
            "Requires --calib_file, --clip_path, --comfy_path, --device cuda."
        ),
    )
    parser.add_argument(
        "--calib_file",
        type=str,
        default=None,
        help="JSONL prompts for Card 1 DualMonitor (required with --bias_correction)",
    )
    parser.add_argument(
        "--clip_path",
        type=str,
        default=None,
        help="Qwen3-4B TE .safetensors for Card 1 (required with --bias_correction)",
    )
    parser.add_argument(
        "--comfy_path",
        type=str,
        default=None,
        help="ComfyUI root for Card 1 (required with --bias_correction)",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional local Qwen2Tokenizer dir (offline Card 1)",
    )
    parser.add_argument(
        "--num_calib_samples",
        type=int,
        default=4,
        help="Card 1 DualMonitor prompt count (default 4)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=8,
        help="Card 1 sample_euler steps (default 8)",
    )
    parser.set_defaults(enable_convrot=True)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    convert_to_nvfp4(
        args.model,
        args.output,
        device=str(args.device),
        model_type=str(args.model_type),
        enable_convrot=bool(args.enable_convrot),
        group_size=int(args.group_size),
        bias_correction=bool(args.bias_correction),
        calib_file=args.calib_file,
        clip_path=args.clip_path,
        comfy_path=args.comfy_path,
        tokenizer_path=args.tokenizer_path,
        num_calib_samples=int(args.num_calib_samples),
        num_inference_steps=int(args.num_inference_steps),
    )

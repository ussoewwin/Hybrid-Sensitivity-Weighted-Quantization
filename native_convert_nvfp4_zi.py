"""Z-Image / ZIT (NextDiT / Lumina2) NVFP4 converter — Kitchen pack + FULL ConvRot.

Reference (pack / blacklist / metadata):
  https://github.com/tritant/ComfyUI_Kitchen_nvfp4_Converter
  convert_to_nvfp4_node.py

  - 2D .weight → (optional offline Hadamard) → TensorCoreNVFP4Layout.quantize
  - Z-Image-Turbo / Z-Image-Base Kitchen blacklists only
  - Non-matching tensors kept as bfloat16
  - Metadata: _quantization_metadata (+ convrot stamp when rotated)
  - FULL ConvRot ON by default (Linear 2D only; Z Image has no Conv2d packs)
      offline: W_rot = W @ H^T (group-wise), then NVFP4 pack
      stamp:   {"format":"nvfp4","convrot":true,"convrot_groupsize":G}
      plain when in_features not divisible by a power-of-4 group: {"format":"nvfp4"}
  - Optional Card 1 (--bias_correction): DualMonitor act means via Comfy
    NextDiT calib; bias += -(W_q - W) @ mu_x on NVFP4 Linear that have .bias
  - No input_scale (Kitchen NVFP4 path)
  - Use --no-convrot for plain Kitchen NVFP4 only

Verified against ComfyUI UNet keys (moodyProMix_zitV13.safetensors):
  model.diffusion_model.{cap_embedder,x_embedder,t_embedder,
  noise_refiner,context_refiner,final_layer,layers.0..29}
  → Kitchen default profile = Z-Image-Turbo
    (embedders / refiners / final_layer BF16; layers.* 2D weights NVFP4)

Refuses non–Z-Image checkpoints (missing Lumina2 signature).
Post-convert SDXL fidelity bench is not chained (Z-Image-only CLI).

Online act rotation for ConvRot stamps is the loader / bench parity responsibility
(ComfyUI stock nvfp4 path does not rotate acts; HSWQ / nvfp4_comfy_parity does).

Card 1 example:
  python native_convert_nvfp4_zi.py --model ... --output ... \\
    --clip_path ... --comfy_path ... --calib_file ... \\
    --num_calib_samples 32 --num_inference_steps 25 --bias_correction
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

from native_convert_int8 import (  # noqa: E402
    build_hadamard,
    convrot_group_size_for_features,
    rotate_weight,
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
    print(f"Mode {model_type} | device={device} | {rot_tag} (Z-Image-only)")
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
                    quant_map["layers"][base_k_meta] = {
                        "format": "nvfp4",
                        "convrot": True,
                        "convrot_groupsize": int(used_gs),
                    }
                    n_convrot += 1
                else:
                    quant_map["layers"][base_k_meta] = {"format": "nvfp4"}
                    n_plain_nvfp4 += 1
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
        "ComfyUI Kitchen NVFP4 Converter (Z-Image ConvRot)"
        if enable_convrot
        else "ComfyUI Kitchen NVFP4 Converter (Z-Image-only)"
    )
    final_metadata["converted_by"] = by_tag
    final_metadata["converter_url"] = (
        "https://github.com/tritant/ComfyUI_Kitchen_nvfp4_Converter"
    )
    final_metadata["hswq_model"] = "z_image"
    final_metadata["hswq_kitchen_profile"] = model_type
    final_metadata["hswq_nvfp4_convrot"] = "1" if enable_convrot else "0"

    print(f"Saving | Type: {model_type} | Path: {output_path}")
    save_file(new_sd, output_path, metadata=final_metadata)
    total_bytes = os.path.getsize(output_path)
    print(f"Done. Size: {round(total_bytes / (1024**3), 2)} GiB")
    print(f"NVFP4 layers in metadata: {len(quant_map['layers'])}")
    print(f"  counted nvfp4 packs={n_nvfp4} | bf16 keep tensors={n_bf16}")
    print(f"FULL ConvRot enabled: {enable_convrot}")
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
            "Z-Image / ZIT NVFP4 convert with FULL ConvRot (Linear) ON by default. "
            "Kitchen pack + offline Hadamard + convrot stamp in "
            "_quantization_metadata. Use --no-convrot for plain Kitchen NVFP4. "
            "Optional Card 1 (--bias_correction): DualMonitor act_mean + "
            "bias += -(W_q-W)@mu_x. Refuses non-Z-Image checkpoints. "
            "Default profile Z-Image-Turbo. No chained SDXL NVFP4 bench."
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

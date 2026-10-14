"""Simple SDXL UNet native ConvRot W4A8 (asym_w4a8_int8) converter.

Base: native_convert_int8_sdxl.py (same Card 1 bias correction, Card 3 per-channel
plain-INT8 fallback, ConvRot INT8 Conv2d fallback).

Linear (2D) layers go ComfyUI-native W4A8 when shapes allow:
  in_features % 16 == 0                  (quant group, fixed 16)
  in_features % convrot_groupsize == 0   (largest power-of-4 <= --groupsize)
  Pack via comfy_kitchen.quantize_w4a8_int8_weight (bits=4, symmetric, codebook):
    weight            int8  [N, K/2]     packed nibbles, low nibble = even column
    weight_s_rel      fp8e4m3 [N, K/16]  per-group relative scale
    weight_s_channel  f32     [N]        per-out-channel scale
    weight_codebook   f32     [16]       Lloyd-Max table (absent when fixed LUT fits)
    comfy_quant       {"format":"asym_w4a8_int8","group_size":16,"convrot_groupsize":G}
  Rotation happens inside the kitchen quantizer (fp32 input, ConvRot H per group).
  Kitchen dequantize returns the ORIGINAL basis, so Card 1 BC uses unrotated act
  means for W4A8 Linear layers (exact, not approximate).

ComfyUI loads this pack 100% stock (no HSWQ patches):
  Linear (2D, K%256==0) -> W4A8 (asym_w4a8_int8), kitchen pack (rotation inside)
  Linear (K%256!=0)     -> ConvRot INT8 (int8 [N,K], convrot stamp) or plain INT8
  Linear (shape-critical: label_emb.0.0 / attn2.to_k) -> shape-preserving INT8
  Conv2d (4D)           -> FP16 kept as-is
Shape-critical Linear keys that detect_unet_config reads (label_emb.0.0 ->
adm_in_channels, attn2.to_k -> context_dim) are stored as plain INT8
shape-preserving [N, K] packs (never nibble-packed), otherwise SDXL
architecture detection reads halved K and fails to identify the model.

  Linear 2D W4A8 gate (all must hold):
    in_features % 256 == 0  (CUDA fused kernel only supports convrot group 256;
                             other group sizes go to the ConvRot INT8 Linear path)
    in_features % 16 == 0   (kitchen quant group)
    not shape-critical      (see below)

Requires comfy_kitchen in the running interpreter (ComfyUI python_embeded).
Fails fast when missing - no re-implementation of the pack math here, so the
packed layout always matches the ComfyUI runtime decoder.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import sys

# SDXL calibration uses pure UNet inference and does not use PEFT LoRA.
# Bypass diffusers' strict runtime peft version assertion.
os.environ["_CHECK_PEFT"] = "0"


def _install_kornia_compat_stub() -> None:
    try:
        import kornia.geometry.transform as _kgt  # noqa: F401
        if not hasattr(_kgt, "build_laplacian_pyramid"):
            def _dummy_laplacian(*args, **kwargs):
                raise NotImplementedError("build_laplacian_pyramid stub")
            _kgt.build_laplacian_pyramid = _dummy_laplacian
    except Exception:
        pass


_install_kornia_compat_stub()


def _install_torchaudio_stub() -> None:
    """Prevent real torchaudio from loading when diffusers, transformers, or comfy.sd is imported.

    On cloud hosts torch/torchaudio CUDA builds often mismatch (e.g. torch 13.2
    vs torchaudio 13.0) and abort. Replace torchaudio in sys.modules with a local stub.
    """
    import importlib.machinery
    import types

    for key in list(sys.modules):
        if key == "torchaudio" or key.startswith("torchaudio."):
            del sys.modules[key]

    def _stub_mod(name: str, *, is_package: bool = False):
        mod = types.ModuleType(name)
        mod.__file__ = "<hswq_torchaudio_stub>"
        if is_package:
            mod.__path__ = []
            spec = importlib.machinery.ModuleSpec(
                name, loader=None, is_package=True
            )
            spec.submodule_search_locations = []
        else:
            spec = importlib.machinery.ModuleSpec(name, loader=None)
        mod.__spec__ = spec
        return mod

    ta = _stub_mod("torchaudio", is_package=True)
    ta.__version__ = "2.5.0"
    ta.load = lambda *args, **kwargs: (None, None)
    ta.info = lambda *args, **kwargs: None

    functional = _stub_mod("torchaudio.functional")

    def _resample(waveform, orig_freq, new_freq, *args, **kwargs):
        return waveform

    functional.resample = _resample

    transforms = _stub_mod("torchaudio.transforms")

    class _MelSpectrogram:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, x):
            return x

        def to(self, *args, **kwargs):
            return self

    class _MelScale:
        def __init__(self, *args, **kwargs):
            pass

    transforms.MelSpectrogram = _MelSpectrogram
    transforms.MelScale = _MelScale

    ta.functional = functional
    ta.transforms = transforms
    sys.modules["torchaudio"] = ta
    sys.modules["torchaudio.functional"] = functional
    sys.modules["torchaudio.transforms"] = transforms


_install_torchaudio_stub()

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

_DEFAULT_GROUPSIZE = 256
# W4A8 quant group. Fixed 16: this is the config ComfyUI's fused CUDA requant
# supports (4-bit symmetric codebook g16, fp8 group scales) and the kitchen
# default. group_size=16 divides every SDXL Linear in_features that passes the
# 16-divisibility gate below.
_QUANT_GROUP_SIZE = 16

# Linear module names whose PACKED shape would corrupt ComfyUI architecture
# detection (detect_unet_config reads their .weight shapes verbatim):
#   label_emb.0.0 -> adm_in_channels, attn2.to_k -> context_dim.
# These stay shape-preserving INT8 packs (int8 [N, K], never nibble-packed).
# Match rule: module key ENDS WITH one of these suffixes, so
# '...transformer_blocks.0.attn2.to_k' and '...label_emb.0.0' both hit.
_SHAPE_CRITICAL_LINEAR_SUFFIXES = (".label_emb.0.0", ".attn2.to_k")


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_comfy_kitchen():
    """Import comfy_kitchen and return the module (fail fast when missing).

    The W4A8 pack math (ConvRot rotate + Lloyd-Max codebook + ALS group scales +
    nibble packing) must match the ComfyUI runtime decoder, so this converter
    calls the same comfy_kitchen API ComfyUI uses at load/requantize time
    instead of re-implementing it. No silent fallback.
    """
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
                f"(W4A8 packing needs comfy_kitchen >= 0.2.x with AsymW4A8Int8Layout)"
            )
    return ck


def _load_hswq_v30():
    """Load archives/quantize_sdxl_hswq_v3.0.py as a module (filename has a digit)."""
    path = os.path.join(
        _REPO_ROOT, "archives", "quantize_sdxl_hswq_v3.0.py"
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
    """Load sibling native_convert_int8.py for Hadamard / rotate_weight (Conv2d fallback)."""
    path = os.path.join(_REPO_ROOT, "native_convert_int8.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"native_convert_int8.py not found: {path}")
    name = "native_convert_int8_for_w4a8"
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
    """Card 1 only: DualMonitor calib -> channel_act_mean.

    Does NOT run Static Profile VETO, derive_hswq_strategy_int8, or V4 FP16 keep.
    """
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


def convert_to_w4a8(
    input_path,
    output_path,
    per_channel_int8: bool = False,
    bias_correction: bool = False,
    calib_file: str | None = None,
    num_calib_samples: int = 32,
    num_inference_steps: int = 25,
    enable_convrot: bool = True,
    group_size: int = _DEFAULT_GROUPSIZE,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    act_mean_dict = {}
    comfyui_to_diffusers_map = {}
    compute_int8_bias_delta = None
    rotate_weight = None
    rotate_weight_conv2d = None
    convrot_group_size_for_features = None
    build_hadamard = None
    ck = None
    w4a8_linear = 0
    convrot_int8_linear = 0
    plain_int8_count = 0
    fp16_conv2d = 0

    if enable_convrot:
        ck = _load_comfy_kitchen()
        nc = _load_native_convert_int8()
        rotate_weight = nc.rotate_weight
        rotate_weight_conv2d = nc.rotate_weight_conv2d
        convrot_group_size_for_features = nc.convrot_group_size_for_features
        build_hadamard = nc.build_hadamard
        print(
            f"  [ConvRot W4A8] ON | Linear 2D -> asym_w4a8_int8 "
            f"(quant_group={_QUANT_GROUP_SIZE}, convrot_groupsize==256 only; "
            f"K%256!=0 Linear -> ConvRot INT8 / plain INT8; Conv2d -> FP16 kept)"
        )
        print(
            "  [ConvRot W4A8] pack via comfy_kitchen.quantize_w4a8_int8_weight "
            "(bits=4 symmetric codebook, fp8 s_rel) - same API as the runtime decoder"
        )
        if bias_correction:
            print(
                "  [ConvRot] NOTE: Card 1 act means are from unrotated float UNet; "
                "W4A8 Linear BC is exact (kitchen dequant returns original basis); "
                "ConvRot INT8 Conv2d BC stays approximate"
            )
    else:
        print("  [ConvRot W4A8] OFF (--no-convrot) | packing plain INT8 only")

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
            "DualMonitor calib (steps=25) | "
            "mu_x = DualMonitor.channel_act_mean | bias += -(W_q - W) @ mu_x"
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
    mode_tag = "ConvRot W4A8 (Linear) + ConvRot INT8 (Conv2d)" if enable_convrot else "plain INT8"
    print(f"Converting UNet Linear/Conv weights ({mode_tag})...")

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
            if per_channel_int8:
                if tensor.ndim not in (2, 4):
                    new_state_dict[key] = tensor
                    skipped_count += 1
                    continue

            in_f = int(tensor.shape[1])
            used_gs = None
            if (
                enable_convrot
                and convrot_group_size_for_features is not None
            ):
                used_gs = convrot_group_size_for_features(in_f, group_size)

            quant_mode = None  # "w4a8" | "int8" | "fp16_keep"
            quant_config: dict
            w_fp_for_bc: torch.Tensor
            weight_dq: torch.Tensor
            module_key = key[: -len(".weight")]

            is_shape_critical = (
                tensor.ndim == 2
                and any(module_key.endswith(s) for s in _SHAPE_CRITICAL_LINEAR_SUFFIXES)
            )

            if (
                enable_convrot
                and ck is not None
                and tensor.ndim == 2
                and not is_shape_critical
                and used_gs == 256
                and in_f % _QUANT_GROUP_SIZE == 0
                and rotate_weight is not None
            ):
                # --- W4A8 Linear (kitchen pack; rotation inside the quantizer) ---
                w_fp32 = tensor.float()
                qdata, s_rel, s_channel, correction, codebook = (
                    ck.quantize_w4a8_int8_weight(
                        w_fp32,
                        group_size=_QUANT_GROUP_SIZE,
                        convrot_groupsize=int(used_gs),
                        symmetric=True,
                        scale_dtype=torch.float8_e4m3fn,
                        codebook=True,
                        bits=4,
                    )
                )
                quant_config = {
                    "format": "asym_w4a8_int8",
                    "group_size": int(_QUANT_GROUP_SIZE),
                    "convrot_groupsize": int(used_gs),
                }
                new_state_dict[key] = qdata.to(torch.int8)
                new_state_dict[f"{module_key}.weight_s_rel"] = s_rel.to(
                    torch.float8_e4m3fn
                ).contiguous()
                new_state_dict[f"{module_key}.weight_s_channel"] = s_channel.float().contiguous()
                if codebook is not None:
                    new_state_dict[f"{module_key}.weight_codebook"] = codebook.float().contiguous()
                new_state_dict[f"{module_key}.comfy_quant"] = _encode_comfy_quant(quant_config)
                quant_meta_layers[module_key] = dict(quant_config)
                converted_count += 1
                quant_mode = "w4a8"
                w4a8_linear += 1
                # Kitchen dequantize returns the ORIGINAL (unrotated) basis.
                weight_dq = ck.dequantize_w4a8_int8_weight(
                    qdata,
                    s_rel,
                    s_channel,
                    codebook=codebook,
                    correction=correction,
                    group_size=_QUANT_GROUP_SIZE,
                    convrot_groupsize=int(used_gs),
                    output_dtype=torch.float32,
                )
                w_fp_for_bc = w_fp32
            elif (
                enable_convrot
                and used_gs is not None
                and tensor.ndim == 2
                and rotate_weight is not None
                and build_hadamard is not None
            ):
                # --- ConvRot INT8 Linear fallback (eligible gs, but K % 16 != 0) ---
                h_matrix = build_hadamard(int(used_gs), device="cpu", dtype=torch.float32)
                w_fp = rotate_weight(tensor.float(), h_matrix, int(used_gs))
                q, scale = pack_channelwise(w_fp)
                quant_config = {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": int(used_gs),
                }
                new_state_dict[key] = q
                new_state_dict[f"{module_key}.weight_scale"] = scale
                new_state_dict[f"{module_key}.comfy_quant"] = _encode_comfy_quant(quant_config)
                quant_meta_layers[module_key] = dict(quant_config)
                converted_count += 1
                quant_mode = "int8"
                convrot_int8_linear += 1
                weight_dq = q.float() * scale
                w_fp_for_bc = w_fp
            elif (
                enable_convrot
                and tensor.ndim == 4
            ):
                # --- Conv2d (4D): stock ComfyUI has no quantized Conv2d op.
                # int8 conv weights crash in stock load/forward (HSWQ Conv2d
                # patch would be required). Keep FP16 so the pack runs stock.
                new_state_dict[key] = tensor
                skipped_count += 1
                quant_mode = "fp16_keep"
                fp16_conv2d += 1
                w_fp_for_bc = tensor.float()
                weight_dq = w_fp_for_bc
            elif per_channel_int8:
                q, scale = pack_channelwise(tensor.float())
                quant_config = {"format": "int8_tensorwise"}
                new_state_dict[key] = q
                new_state_dict[f"{module_key}.weight_scale"] = scale
                new_state_dict[f"{module_key}.comfy_quant"] = _encode_comfy_quant(quant_config)
                quant_meta_layers[module_key] = dict(quant_config)
                converted_count += 1
                quant_mode = "int8"
                plain_int8_count += 1
                weight_dq = q.float() * scale
                w_fp_for_bc = tensor.float()
            else:
                q, scale = pack_tensorwise(tensor.float())
                quant_config = {"format": "int8_tensorwise"}
                new_state_dict[key] = q
                new_state_dict[f"{module_key}.weight_scale"] = scale
                new_state_dict[f"{module_key}.comfy_quant"] = _encode_comfy_quant(quant_config)
                quant_meta_layers[module_key] = dict(quant_config)
                converted_count += 1
                quant_mode = "int8"
                plain_int8_count += 1
                weight_dq = q.float() * scale
                w_fp_for_bc = tensor.float()

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
                    delta = compute_int8_bias_delta(w_fp_for_bc, weight_dq, act_mean)
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
            f"quantized Linear+Conv layers (full Card 1)..."
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
    print(f"Per-channel INT8 (Card 3, fallback packs): {per_channel_int8}")
    print(f"Bias correction (Card 1): {bias_correction}")
    if bias_correction:
        print(f"  Bias-corrected layers: {bias_corr_applied}")
    print(f"ConvRot enabled: {enable_convrot}")
    if enable_convrot:
        print(
            f"  W4A8 Linear (asym_w4a8_int8): {w4a8_linear}, "
            f"ConvRot INT8 Linear: {convrot_int8_linear}, "
            f"plain INT8 Linear: {plain_int8_count}, "
            f"FP16-kept Conv2d: {fp16_conv2d}"
        )
    else:
        print(f"  plain INT8: {plain_int8_count}")

    save_file(new_state_dict, output_path, metadata=metadata)
    print("Done!")

    # Convert complete: drop holdings.
    del new_state_dict
    del state_dict
    del quant_meta_layers
    act_mean_dict.clear()
    del act_mean_dict
    bias_corr_pending.clear()
    del bias_corr_pending
    comfyui_to_diffusers_map.clear()
    del comfyui_to_diffusers_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "UNet ConvRot W4A8 (asym_w4a8_int8) convert for native ComfyUI. "
            "Linear 2D -> W4A8 pack via comfy_kitchen (quant group 16, Lloyd-Max "
            "codebook, fp8 group scales); shape-critical Linears (label_emb.0.0, "
            "attn2.to_k) -> shape-preserving plain INT8; Conv2d -> FP16 kept "
            "(stock ComfyUI has no quantized Conv2d op); ineligible Linear -> "
            "plain INT8 tensorwise. Card 1 = --bias_correction."
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
    parser.add_argument("--output", type=str, required=True, help="Path to output .safetensors")
    parser.add_argument(
        "--per_channel_int8",
        action="store_true",
        help=(
            "Card 3 for plain INT8 fallback packs only: per-out-channel amax/scale. "
            "W4A8 Linear always carries its own per-channel s_channel."
        ),
    )
    parser.add_argument(
        "--bias_correction",
        action="store_true",
        help=(
            "Card 1 ON: DualMonitor calib; bias += -(W_q - W) @ mu_x on quantized "
            "Linear+Conv. W4A8 Linear BC is exact (kitchen dequant returns the "
            "original basis). Requires --calib_file."
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
            "ConvRot W4A8 on eligible Linear + ConvRot INT8 on Conv2d. Default ON; "
            "the W4A8 layout itself is ConvRot-based, so --no-convrot packs plain "
            "INT8 only."
        ),
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=_DEFAULT_GROUPSIZE,
        help=f"Preferred ConvRot Hadamard group size (power of 4, default {_DEFAULT_GROUPSIZE})",
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

    convert_to_w4a8(
        args.model,
        args.output,
        per_channel_int8=args.per_channel_int8,
        bias_correction=bool(args.bias_correction),
        calib_file=args.calib_file,
        num_calib_samples=args.num_calib_samples,
        num_inference_steps=args.num_inference_steps,
        enable_convrot=bool(args.convrot),
        group_size=int(args.groupsize),
    )

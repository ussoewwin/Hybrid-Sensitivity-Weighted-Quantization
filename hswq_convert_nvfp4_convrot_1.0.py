"""UNet / DiT FULL ConvRot + NVFP4 converter for native ComfyUI Load Diffusion Model.

FULL ConvRot / FULL offline rotate (default ON):
  - Linear 2D, in_features divisible by power-of-4 group:
      offline Hadamard → NVFP4 pack + stamp
      comfy_quant: {"format":"nvfp4","convrot":true,"convrot_groupsize":G}
  - Conv2d, in_channels divisible by power-of-4 group:
      offline Hadamard → INT8 per-channel + stamp
      (NVFP4 layout is 2D-only; Conv2d uses int8_tensorwise + convrot)
      comfy_quant: {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":G}
  - Linear without group: plain NVFP4 {"format":"nvfp4"}
  - Conv2d without group: plain INT8 {"format":"int8_tensorwise"}

On-disk NVFP4 Linear (TensorCoreNVFP4Layout / QUANT_ALGOS["nvfp4"]):
  .weight          uint8 [N', K'//2]
  .weight_scale    f8e4m3 [N', K'//16]  (block scales)
  .weight_scale_2  f32 scalar           (global scale)
  .input_scale     f32 scalar           (act: amax / (F8_E4M3_MAX * F4_E2M1_MAX))
  .comfy_quant     uint8 JSON

input_scale is written from PTQ calib (--calib_file). For FULL ConvRot layers,
amax is measured on Hadamard-rotated activations (same order as inference:
rotate then quantize). Without calib, no input_scale keys are written and
inference falls back to ones — that destroys quality.

Online act rotate at load is required for ConvRot layers. The loader is built
separately; this converter always does FULL offline weight rotate + stamps.

Use --no-convrot for plain packs only (no offline rotate / no convrot stamp).

Optional Card 1 (--bias_correction): DualMonitor act means; bias += -(W_q - W) @ mu_x
  on quantized Linear/Conv. Shares the same --calib_file pass as input_scale.

HSWQ DualMonitor + FP16 keep, when --calib_file is set:
  - --keep_ratio: top layers by V4 Full-SVD×RMS × real pack MSE stay FP16.
    Score = HSWQWeightedHistogramOptimizerV4NVFP4 estimated_mse @ absmax
    (Linear=NVFP4 pack, Conv=INT8 channelwise) × optional DualMonitor Imp
    × role prior (attn1 bias; attn2.to_k not boosted).
    analyze Hard VETO layers are always kept first.
    Not output-variance sensitivity (that starved attn1 / over-kept to_k).
  - Weight clip amax: minimize DualMonitor-weighted MSE of the *actual* pack path
    (Linear: TensorCoreNVFP4Layout quantize→dequantize; Conv2d: channelwise INT8).
    Not FP8 e4m3 histogram. ConvRot: search on Hadamard-rotated weight (same as pack).
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
# Clip-amax search budget (histogram/weighted_histogram_mse_fast_nvfp4.py).
_AMAX_NUM_CANDIDATES = 32
_AMAX_REFINEMENT_ITERS = 2
_AMAX_SEARCH_RANGE = (0.5, 1.0)


def _ensure_nvfp4_hist_on_path() -> None:
    hist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "histogram")
    if hist_dir not in sys.path:
        sys.path.insert(0, hist_dir)


def _load_nvfp4_amax_optimizer(device: str):
    """Load histogram/weighted_histogram_mse_fast_nvfp4.py (pack roundtrip MSE)."""
    _ensure_nvfp4_hist_on_path()
    from weighted_histogram_mse_fast_nvfp4 import (  # type: ignore
        HSWQWeightedHistogramOptimizerFastNVFP4,
    )

    return HSWQWeightedHistogramOptimizerFastNVFP4(
        num_candidates=_AMAX_NUM_CANDIDATES,
        refinement_iterations=_AMAX_REFINEMENT_ITERS,
        device=device,
        search_range=_AMAX_SEARCH_RANGE,
    )


def _fp16_keep_role_weight(name: str) -> float:
    """Role prior for FP16 keep ranking (SDXL UNet Diffusers names).

    Output-variance ranking kept 100% of attn2.to_k and almost no attn1.
    Pack MSE is primary; this multiplier only breaks near-ties toward layers
    that carry spatial / query structure.
    """
    n = name.lower()
    if ".attn1." in n:
        return 4.0
    if ".attn2.to_q" in n or ".attn2.to_v" in n:
        return 3.0
    if ".attn2.to_out" in n:
        return 2.5
    if ".attn2.to_k" in n:
        return 1.0
    if ".proj_out" in n or ".proj_in" in n:
        return 1.5
    if ".ff." in n:
        return 1.25
    if any(
        tok in n
        for tok in (
            "skip_connection",
            "upsamplers",
            "downsamplers",
            ".conv_in",
            ".conv_out",
            ".conv_shortcut",
            "resnets",
            "in_layers",
            "out_layers",
            ".op",
        )
    ):
        return 0.6
    return 1.0


def _prepare_weight_for_pack_score(
    *,
    weight: torch.Tensor,
    enable_convrot: bool,
    group_size: int,
    build_hadamard,
    convrot_group_size_for_features,
    rotate_weight,
    rotate_weight_conv2d,
    hadamard_cache: dict[int, torch.Tensor],
) -> torch.Tensor:
    """Match pack/amax path: optional offline Hadamard on eligible Linear/Conv2d."""
    w = weight.detach().float()
    if w.ndim not in (2, 4):
        return w
    in_f = int(w.shape[1])
    used_gs = None
    if (
        enable_convrot
        and convrot_group_size_for_features is not None
        and build_hadamard is not None
    ):
        used_gs = convrot_group_size_for_features(in_f, group_size)
    do_rotate = (
        enable_convrot
        and used_gs is not None
        and build_hadamard is not None
        and (
            (w.ndim == 2 and rotate_weight is not None)
            or (w.ndim == 4 and rotate_weight_conv2d is not None)
        )
    )
    if not do_rotate:
        return w
    h = hadamard_cache.get(int(used_gs))
    if h is None:
        h = build_hadamard(int(used_gs), device="cpu", dtype=torch.float32)
        hadamard_cache[int(used_gs)] = h
    if w.ndim == 2:
        return rotate_weight(w.cpu(), h, int(used_gs))
    return rotate_weight_conv2d(w.cpu(), h, int(used_gs))


def _dualmonitor_importance(
    dual_monitors: dict, name: str, in_features: int, device: str
) -> torch.Tensor | None:
    mon = dual_monitors.get(name)
    if mon is None or getattr(mon, "channel_importance", None) is None:
        return None
    importance = mon.channel_importance.detach().float().to(device=device)
    if int(importance.numel()) != int(in_features):
        return None
    return importance


def _ensure_analyze_on_path() -> None:
    analyze_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze")
    if analyze_dir not in sys.path:
        sys.path.insert(0, analyze_dir)


def _select_fp16_keep_layers(
    *,
    model: torch.nn.Module,
    dual_monitors: dict,
    target_modules: list[str],
    keep_ratio: float,
    device: str,
    enable_convrot: bool,
    group_size: int,
    build_hadamard,
    convrot_group_size_for_features,
    rotate_weight,
    rotate_weight_conv2d,
) -> tuple[set[str], list[tuple[str, float]]]:
    """Rank FP16 keep by V4 Full-SVD×RMS × NVFP4/INT8 pack MSE (not variance).

    Same contract as INT8 SDXL V4 (`_measure_v4_mse_absmax_int8`):
      - Full-SVD×RMS hybrid (alpha_auto > 0; SVD never cut)
      - DualMonitor channel_importance multiplies hybrid when present
      - estimated_mse = real NVFP4 (Linear) / INT8 channelwise (Conv) pack
      - analyze Hard VETO layers are always kept first
      - role prior only breaks near-ties (attn1 / spatial bias)
    """
    _ensure_nvfp4_hist_on_path()
    _ensure_analyze_on_path()
    from weighted_histogram_mse_v4_nvfp4 import (  # type: ignore
        HSWQWeightedHistogramOptimizerV4NVFP4,
    )
    from analyze_sdxl_nvfp4__distribution import (  # type: ignore
        derive_veto_tunables_nvfp4,
        hard_veto_module_names,
        profile_from_unet_modules,
    )

    profile = profile_from_unet_modules(model)
    tunables = derive_veto_tunables_nvfp4(profile)
    alpha = float(tunables.get("alpha_auto", 0.0) or 0.0)
    alpha = float(min(max(alpha, 0.0), 1.0))
    if alpha <= 0.0:
        raise ValueError(
            "NVFP4 FP16 keep: alpha_auto must be > 0 "
            f"(Full-SVD required; got {alpha})"
        )
    beta = 1.0 - alpha
    hard_veto = hard_veto_module_names(profile, tunables)
    optimizer = HSWQWeightedHistogramOptimizerV4NVFP4(
        device=device, alpha=alpha, beta=beta
    )

    module_dict = dict(model.named_modules())
    hadamard_cache: dict[int, torch.Tensor] = {}
    ranked: list[tuple[str, float, float, float]] = []
    n_svd_x_imp = 0
    n_svd_only = 0

    print(
        "\n[HSWQ FP16 keep] V4 Full-SVD×RMS × NVFP4/INT8 pack-MSE @ absmax "
        f"for {len(target_modules)} layers "
        f"(alpha={alpha:.4f}, beta={beta:.4f}, "
        f"analyze Hard VETO={len(hard_veto)}, "
        f"ConvRot={'ON' if enable_convrot else 'OFF'}; "
        "not DualMonitor output-variance)..."
    )
    for name in tqdm(target_modules, desc="HSWQ V4 keep-rank"):
        mod = module_dict.get(name)
        if mod is None or not hasattr(mod, "weight") or mod.weight is None:
            continue
        w0 = mod.weight.detach().float()
        if w0.ndim not in (2, 4):
            continue
        try:
            w = _prepare_weight_for_pack_score(
                weight=w0,
                enable_convrot=enable_convrot,
                group_size=group_size,
                build_hadamard=build_hadamard,
                convrot_group_size_for_features=convrot_group_size_for_features,
                rotate_weight=rotate_weight,
                rotate_weight_conv2d=rotate_weight_conv2d,
                hadamard_cache=hadamard_cache,
            )
            imp = _dualmonitor_importance(
                dual_monitors, name, int(w.shape[1]), device
            )
            result = optimizer.compute_pack_mse_absmax_with_svd(
                w,
                channel_importance=imp,
                use_svd_leverage=True,
                layer_name=name,
            )
            pack_mse = float(result["estimated_mse"])
            role_w = float(_fp16_keep_role_weight(name))
            score = float(pack_mse) * role_w
            ranked.append((name, score, float(pack_mse), role_w))
            if imp is None:
                n_svd_only += 1
            else:
                n_svd_x_imp += 1
        except Exception as e:
            print(f"  [HSWQ FP16 keep] skip {name}: {e}")
            continue
        if device == "cuda":
            torch.cuda.empty_cache()

    ranked.sort(key=lambda x: x[1], reverse=True)
    ratio = max(0.0, min(1.0, float(keep_ratio)))
    num_keep = int(len(ranked) * ratio) if ratio > 0 else 0

    # Hard VETO always kept; fill remaining budget by V4 score order.
    ranked_names = {r[0] for r in ranked}
    keep_layers: set[str] = set()
    for name in hard_veto:
        if name in ranked_names or name in target_modules:
            keep_layers.add(name)
    for name, _score, _mse, _rw in ranked:
        if len(keep_layers) >= num_keep:
            break
        keep_layers.add(name)

    layer_scores: list[tuple[str, float]] = [(n, s) for n, s, _, _ in ranked]
    print(
        f"  [HSWQ FP16 keep] V4 scored={len(ranked)} "
        f"(SVD×Imp={n_svd_x_imp}, SVD-only={n_svd_only}); "
        f"keep={len(keep_layers)} (budget={num_keep}, "
        f"Hard VETO in keep={len(keep_layers & hard_veto)})"
    )

    if keep_layers:
        def _bucket(nm: str) -> str:
            nl = nm.lower()
            if ".attn1." in nl:
                return "attn1"
            if ".attn2.to_k" in nl:
                return "attn2.to_k"
            if ".attn2." in nl:
                return "attn2.other"
            if ".ff." in nl:
                return "ff"
            if ".proj_" in nl:
                return "proj"
            return "conv/other"

        from collections import Counter

        c = Counter(_bucket(n) for n in keep_layers)
        parts = ", ".join(f"{k}={v}" for k, v in sorted(c.items()))
        print(f"  [HSWQ FP16 keep] keep-set role mix: {parts}")

    return keep_layers, layer_scores


def _nvfp4_input_scale_from_amax(amax: float) -> torch.Tensor:
    """Kitchen TensorCoreNVFP4Layout.quantize input_scale formula (scalar f32)."""
    from comfy_kitchen.float_utils import F4_E2M1_MAX, F8_E4M3_MAX

    denom = float(F8_E4M3_MAX) * float(F4_E2M1_MAX)
    return torch.tensor(max(float(amax), 1e-12) / denom, dtype=torch.float32)


def _rotate_act_last_dim(
    x: torch.Tensor, h_matrix: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Hadamard rotate last dim in groups (matches inference rotate_last_dim)."""
    *lead, last = x.shape
    if last % group_size != 0:
        raise ValueError(
            f"last dim {last} not divisible by group_size={group_size}"
        )
    y = x.reshape(*lead, last // group_size, group_size).to(dtype=torch.float32)
    h = h_matrix.to(device=y.device, dtype=torch.float32)
    y = torch.matmul(y, h)
    return y.reshape(*lead, last)


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
    name = "native_convert_int8_for_nvfp4_convrot"
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


def _get_nvfp4_layout():
    """comfy_kitchen TensorCoreNVFP4Layout (venv package)."""
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout

    return TensorCoreNVFP4Layout


def pack_nvfp4(weight: torch.Tensor):
    """NVFP4 pack: uint8 qdata + Params (scale, block_scale, orig_shape).

    Auto-pads to 16x16 when needed (layout get_padded_shape).
    """
    if weight.ndim != 2:
        raise ValueError(f"NVFP4 pack expects 2D weight, got ndim={weight.ndim}")
    layout = _get_nvfp4_layout()
    qdata, params = layout.quantize(weight.float())
    return qdata, params


def dequant_nvfp4(qdata: torch.Tensor, params) -> torch.Tensor:
    """Dequantize NVFP4 storage back to float (sliced to orig_shape)."""
    layout = _get_nvfp4_layout()
    full = layout.dequantize(qdata, params)
    orig = tuple(params.orig_shape)
    if tuple(full.shape) != orig:
        slices = tuple(slice(0, s) for s in orig)
        return full[slices]
    return full


def can_pack_nvfp4(weight: torch.Tensor) -> bool:
    """NVFP4 requires a 2D Linear weight (padding handles 16-align)."""
    return weight.ndim == 2 and weight.shape[0] > 0 and weight.shape[1] > 0


def pack_channelwise_int8(weight: torch.Tensor):
    """Per-out-channel INT8 (Conv2d FULL ConvRot path; NVFP4 is 2D-only)."""
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


def dequant_channelwise_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.float()


def _compute_weight_amax_dict(
    *,
    model: torch.nn.Module,
    dual_monitors: dict,
    keep_layers: set[str],
    device: str,
    enable_convrot: bool,
    group_size: int,
    build_hadamard,
    convrot_group_size_for_features,
    rotate_weight,
    rotate_weight_conv2d,
) -> dict[str, float]:
    """Per-layer clip amax via weighted_histogram_mse_fast_nvfp4 (pack MSE).

    Skips FP16-kept layers. FULL ConvRot-eligible layers: search on the rotated
    weight (the tensor pack_nvfp4 / INT8 pack consumes). DualMonitor
    channel_importance weights the element MSE when shape matches.
    """
    weight_amax_dict: dict[str, float] = {}
    hadamard_cache: dict[int, torch.Tensor] = {}
    optimizer = _load_nvfp4_amax_optimizer(device)

    print(
        "\n[HSWQ] Weight clip amax via weighted_histogram_mse_fast_nvfp4 "
        f"(Linear=NVFP4, Conv2d=INT8; candidates={_AMAX_NUM_CANDIDATES}, "
        f"refine={_AMAX_REFINEMENT_ITERS})..."
    )
    for name, module in tqdm(model.named_modules(), desc="HSWQ amax"):
        if not isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            continue
        if name in keep_layers:
            continue
        w = module.weight.detach().float()
        if w.ndim not in (2, 4):
            continue

        in_f = int(w.shape[1])
        used_gs = None
        if (
            enable_convrot
            and convrot_group_size_for_features is not None
            and build_hadamard is not None
        ):
            used_gs = convrot_group_size_for_features(in_f, group_size)
        do_rotate = (
            enable_convrot
            and used_gs is not None
            and build_hadamard is not None
            and (
                (w.ndim == 2 and rotate_weight is not None)
                or (w.ndim == 4 and rotate_weight_conv2d is not None)
            )
        )
        if do_rotate:
            h = hadamard_cache.get(int(used_gs))
            if h is None:
                h = build_hadamard(int(used_gs), device="cpu", dtype=torch.float32)
                hadamard_cache[int(used_gs)] = h
            if w.ndim == 2:
                w = rotate_weight(w.cpu(), h, int(used_gs))
            else:
                w = rotate_weight_conv2d(w.cpu(), h, int(used_gs))

        importance = None
        mon = dual_monitors.get(name)
        if mon is not None and getattr(mon, "channel_importance", None) is not None:
            importance = mon.channel_importance.detach().float()

        w_dev = w.to(device=device, dtype=torch.float32)
        if importance is not None:
            importance = importance.to(device=device, dtype=torch.float32)
            if importance.numel() != int(w_dev.shape[1]):
                importance = None

        pack_mode = "nvfp4" if w_dev.ndim == 2 else "int8"
        weight_amax_dict[name] = float(
            optimizer.compute_optimal_amax(
                w_dev, importance, scaled=False, pack_mode=pack_mode
            )
        )
        del w_dev
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"  [HSWQ] weight amax layers={len(weight_amax_dict)}")
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return weight_amax_dict


def run_nvfp4_calib(
    *,
    input_path: str,
    calib_file: str,
    num_calib_samples: int,
    num_inference_steps: int,
    device: str,
    enable_convrot: bool = True,
    group_size: int = _DEFAULT_GROUPSIZE,
    keep_ratio: float = 0.0,
):
    """PTQ calib: DualMonitor + input_scale amax + HSWQ keep / weight amax.

    DualMonitor (same sensors as archives V1.3): act means, channel_importance,
    and related stats. After the calib pass:
      - top keep_ratio layers by V4 Full-SVD×RMS × pack-MSE
        (+ analyze Hard VETO first) → FP16 keep set
      - NVFP4/INT8 pack roundtrip MSE → weight clip amax
        (ConvRot: on rotated weight)

    For Linear layers with FULL ConvRot, act amax for input_scale is measured
    on Hadamard-rotated activations (rotate then quantize).
    """
    v30 = _load_hswq_v30()
    build_hadamard = None
    convrot_group_size_for_features = None
    rotate_weight = None
    rotate_weight_conv2d = None
    hadamard_cache: dict[int, torch.Tensor] = {}
    if enable_convrot:
        nc = _load_native_convert_int8()
        build_hadamard = nc.build_hadamard
        convrot_group_size_for_features = nc.convrot_group_size_for_features
        rotate_weight = nc.rotate_weight
        rotate_weight_conv2d = nc.rotate_weight_conv2d

    pipeline, _state_dict, comfyui_to_diffusers_map = v30.load_unet_from_safetensors(
        input_path, device
    )
    model = pipeline.unet

    print(
        "Preparing calibration (DualMonitor + NVFP4 input_scale amax "
        "+ HSWQ keep / pack-roundtrip weight amax)..."
    )
    if enable_convrot:
        print(
            "  [input_scale] ConvRot Linear: amax after Hadamard rotate_last_dim "
            f"(preferred groupsize={group_size})"
        )
    else:
        print("  [input_scale] amax on unrotated activations (--no-convrot)")

    v30.dual_monitors.clear()
    act_amax_dict: dict[str, float] = {}
    handles = []

    def _make_hook(name: str):
        def hook(m, inp, out):
            v30.hook_fn(m, inp, out, name)
            if not inp or inp[0] is None:
                return
            x = inp[0]
            if not torch.is_tensor(x) or not torch.is_floating_point(x):
                return
            x_f = x.detach().float()
            x_for_amax = x_f
            if (
                isinstance(m, torch.nn.Linear)
                and enable_convrot
                and convrot_group_size_for_features is not None
                and build_hadamard is not None
            ):
                in_f = int(m.in_features)
                gs = convrot_group_size_for_features(in_f, group_size)
                if gs is not None and int(x_f.shape[-1]) == in_f:
                    h = hadamard_cache.get(int(gs))
                    if h is None:
                        h = build_hadamard(
                            int(gs), device=x_f.device, dtype=torch.float32
                        )
                        hadamard_cache[int(gs)] = h
                    elif h.device != x_f.device:
                        h = h.to(device=x_f.device)
                        hadamard_cache[int(gs)] = h
                    flat = x_f.reshape(-1, in_f)
                    x_for_amax = _rotate_act_last_dim(flat, h, int(gs))
            amax = float(x_for_amax.abs().amax().clamp_min(1e-12).item())
            prev = act_amax_dict.get(name)
            if prev is None or amax > prev:
                act_amax_dict[name] = amax

        return hook

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(module.register_forward_hook(_make_hook(name)))

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
    print(
        f"  [input_scale] act_amax layers={len(act_amax_dict)} "
        f"(running abs max over calib)"
    )

    target_modules = [
        name
        for name, mod in model.named_modules()
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear))
        and name in v30.dual_monitors
    ]
    keep_layers, layer_sensitivities = _select_fp16_keep_layers(
        model=model,
        dual_monitors=v30.dual_monitors,
        target_modules=target_modules,
        keep_ratio=keep_ratio,
        device=device,
        enable_convrot=bool(enable_convrot),
        group_size=int(group_size),
        build_hadamard=build_hadamard,
        convrot_group_size_for_features=convrot_group_size_for_features,
        rotate_weight=rotate_weight,
        rotate_weight_conv2d=rotate_weight_conv2d,
    )
    ratio_pct = max(0.0, min(1.0, float(keep_ratio))) * 100.0
    print(
        f"\n[HSWQ DualMonitor] FP16 keep_ratio={keep_ratio} "
        f"→ keep {len(keep_layers)}/{len(layer_sensitivities)} "
        f"(Top {ratio_pct:.1f}% by V4 Full-SVD×RMS × pack-MSE)"
    )
    if keep_layers and layer_sensitivities:
        preview = layer_sensitivities[: min(8, len(keep_layers))]
        for n, s in preview:
            print(f"  keep FP16: {n}  v4_score={s:.6e}")
        if len(keep_layers) > 8:
            print(f"  ... and {len(keep_layers) - 8} more")

    weight_amax_dict = _compute_weight_amax_dict(
        model=model,
        dual_monitors=v30.dual_monitors,
        keep_layers=keep_layers,
        device=device,
        enable_convrot=bool(enable_convrot),
        group_size=int(group_size),
        build_hadamard=build_hadamard,
        convrot_group_size_for_features=convrot_group_size_for_features,
        rotate_weight=rotate_weight,
        rotate_weight_conv2d=rotate_weight_conv2d,
    )

    del pipeline
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "act_mean_dict": act_mean_dict,
        "act_amax_dict": act_amax_dict,
        "comfyui_to_diffusers_map": comfyui_to_diffusers_map,
        "v30": v30,
        "keep_layers": keep_layers,
        "weight_amax_dict": weight_amax_dict,
        "layer_sensitivities": layer_sensitivities,
    }


def convert_to_nvfp4_convrot(
    input_path,
    output_path,
    bias_correction: bool = False,
    calib_file: str | None = None,
    num_calib_samples: int = 32,
    num_inference_steps: int = 25,
    enable_convrot: bool = True,
    group_size: int = _DEFAULT_GROUPSIZE,
    min_in_features: int = 0,
    keep_ratio: float = 0.0,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    act_mean_dict = {}
    act_amax_dict: dict[str, float] = {}
    weight_amax_dict: dict[str, float] = {}
    keep_layers: set[str] = set()
    comfyui_to_diffusers_map = {}
    compute_int8_bias_delta = None
    rotate_weight = None
    rotate_weight_conv2d = None
    convrot_group_size_for_features = None
    build_hadamard = None
    convrot_nvfp4 = 0
    plain_nvfp4 = 0
    convrot_int8_conv2d = 0
    plain_int8_conv2d = 0
    skipped_small = 0
    fp16_kept_count = 0
    weight_clamp_count = 0
    input_scale_written = 0
    input_scale_missing = 0
    write_input_scale = False

    if enable_convrot:
        nc = _load_native_convert_int8()
        rotate_weight = nc.rotate_weight
        rotate_weight_conv2d = nc.rotate_weight_conv2d
        convrot_group_size_for_features = nc.convrot_group_size_for_features
        build_hadamard = nc.build_hadamard
        print(
            f"  [FULL ConvRot] ON | preferred groupsize={group_size}; "
            f"min_in_features={min_in_features}"
        )
        print(
            "  [FULL ConvRot] Linear → offline Hadamard + NVFP4 when group OK; "
            "else plain NVFP4. Conv2d → offline Hadamard + INT8 when group OK; "
            "else plain INT8. Online act rotate required at load (loader later)."
        )
        if bias_correction:
            print(
                "  [ConvRot] WARN: Card 1 DualMonitor means are from unrotated float UNet; "
                "BC uses rotated W vs W_q (approximate for ConvRot)"
            )
    else:
        print(
            "  [FULL ConvRot] OFF | plain NVFP4 on Linear, plain INT8 on Conv2d "
            "(no offline rotate)"
        )

    if float(keep_ratio) > 0.0 and not calib_file:
        raise ValueError(
            "--keep_ratio > 0 requires --calib_file "
            "(DualMonitor sensitivity ranking, HSWQ V1.3 style)"
        )

    if calib_file:
        if not os.path.isfile(calib_file):
            raise FileNotFoundError(f"calib_file not found: {calib_file}")
        write_input_scale = True
        if bias_correction:
            print(
                "  [Bias Correction Card 1] ON | quantized Linear | "
                "DualMonitor calib | bias += -(W_q - W) @ mu_x"
            )
        print(
            "  [input_scale] ON | write NVFP4 Linear "
            "amax/(F8_E4M3_MAX*F4_E2M1_MAX) from same calib pass"
        )
        print(
            f"  [HSWQ] keep_ratio={keep_ratio} | DualMonitor FP16 protect + "
            "NVFP4/INT8 pack-roundtrip weight clip amax"
        )
        calib = run_nvfp4_calib(
            input_path=input_path,
            calib_file=calib_file,
            num_calib_samples=int(num_calib_samples),
            num_inference_steps=int(num_inference_steps),
            device=device,
            enable_convrot=bool(enable_convrot),
            group_size=int(group_size),
            keep_ratio=float(keep_ratio),
        )
        act_mean_dict = calib["act_mean_dict"]
        act_amax_dict = calib["act_amax_dict"]
        weight_amax_dict = calib["weight_amax_dict"]
        keep_layers = calib["keep_layers"]
        comfyui_to_diffusers_map = calib["comfyui_to_diffusers_map"]
        if bias_correction:
            compute_int8_bias_delta = calib["v30"].compute_int8_bias_delta
            print(
                f"  [Bias Correction] Captured act means for {len(act_mean_dict)} layers"
            )
        print(
            f"  [input_scale] Captured act amax for {len(act_amax_dict)} layers"
        )
        print(
            f"  [HSWQ] weight amax for {len(weight_amax_dict)} layers; "
            f"FP16 keep={len(keep_layers)}"
        )
    elif bias_correction:
        raise ValueError(
            "--bias_correction requires --calib_file "
            "(same as quantize_sdxl_hswq_v3.0.py)"
        )
    else:
        print(
            "  [WARN] No --calib_file: NVFP4 Linear will have NO .input_scale keys. "
            "Inference falls back to ones(1) and quality collapses. "
            "Pass --calib_file to write correct scales into the ckpt. "
            "HSWQ keep_ratio / pack-roundtrip weight amax also require --calib_file."
        )

    if compute_int8_bias_delta is None:

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
    rot_tag = " + FULL ConvRot" if enable_convrot else " plain NVFP4/INT8"
    print(f"Converting diffusion Linear/Conv2d weights ({rot_tag.strip()})...")

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

            diffusers_key = comfyui_to_diffusers_map.get(key)
            module_name = None
            if diffusers_key and diffusers_key.endswith(".weight"):
                module_name = diffusers_key[:-7]

            if module_name is not None and module_name in keep_layers:
                new_state_dict[key] = tensor
                fp16_kept_count += 1
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

            do_rotate = (
                enable_convrot
                and used_gs is not None
                and build_hadamard is not None
                and (
                    (tensor.ndim == 2 and rotate_weight is not None)
                    or (tensor.ndim == 4 and rotate_weight_conv2d is not None)
                )
            )
            if do_rotate:
                h_matrix = build_hadamard(int(used_gs), device="cpu", dtype=torch.float32)
                if tensor.ndim == 2:
                    w_fp = rotate_weight(w_fp, h_matrix, int(used_gs))
                else:
                    w_fp = rotate_weight_conv2d(w_fp, h_matrix, int(used_gs))

            if module_name is not None and module_name in weight_amax_dict:
                amax_w = float(weight_amax_dict[module_name])
                w_fp = w_fp.clamp(-amax_w, amax_w)
                weight_clamp_count += 1

            module_key = key[: -len(".weight")]

            if tensor.ndim == 2:
                if not can_pack_nvfp4(tensor):
                    new_state_dict[key] = tensor
                    skipped_count += 1
                    continue
                q, params = pack_nvfp4(w_fp)
                weight_dq = dequant_nvfp4(q, params)
                if do_rotate:
                    quant_config = {
                        "format": "nvfp4",
                        "convrot": True,
                        "convrot_groupsize": int(used_gs),
                    }
                    convrot_nvfp4 += 1
                else:
                    quant_config = {"format": "nvfp4"}
                    plain_nvfp4 += 1
                new_state_dict[key] = q
                new_state_dict[f"{module_key}.weight_scale"] = params.block_scale
                new_state_dict[f"{module_key}.weight_scale_2"] = params.scale.to(
                    dtype=torch.float32
                ).reshape(())
                if write_input_scale:
                    amax = (
                        act_amax_dict.get(module_name)
                        if module_name is not None
                        else None
                    )
                    if amax is None:
                        input_scale_missing += 1
                    else:
                        new_state_dict[f"{module_key}.input_scale"] = (
                            _nvfp4_input_scale_from_amax(amax)
                        )
                        input_scale_written += 1
            else:
                # Conv2d: NVFP4 is 2D-only → INT8 channelwise (+ FULL ConvRot when OK)
                q, scale = pack_channelwise_int8(w_fp)
                weight_dq = dequant_channelwise_int8(q, scale)
                if do_rotate:
                    quant_config = {
                        "format": "int8_tensorwise",
                        "convrot": True,
                        "convrot_groupsize": int(used_gs),
                    }
                    convrot_int8_conv2d += 1
                else:
                    quant_config = {"format": "int8_tensorwise"}
                    plain_int8_conv2d += 1
                new_state_dict[key] = q
                new_state_dict[f"{module_key}.weight_scale"] = scale

            new_state_dict[f"{module_key}.comfy_quant"] = _encode_comfy_quant(
                quant_config
            )
            quant_meta_layers[module_key] = dict(quant_config)
            converted_count += 1

            if bias_correction:
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
            f"quantized Linear/Conv layers..."
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
    if fp16_kept_count:
        print(
            f"  HSWQ FP16 protect (DualMonitor keep_ratio={keep_ratio}): "
            f"{fp16_kept_count}"
        )
    if weight_clamp_count:
        print(
            f"  HSWQ pack-roundtrip weight clamp: {weight_clamp_count}"
        )
    if skipped_small:
        print(f"  skipped (min_in_features={min_in_features}): {skipped_small}")
    print(f"Bias correction (Card 1): {bias_correction}")
    if bias_correction:
        print(f"  Bias-corrected layers: {bias_corr_applied}")
    print(f"input_scale written (NVFP4 Linear): {input_scale_written}")
    if write_input_scale and input_scale_missing:
        print(
            f"  [WARN] NVFP4 Linear missing act amax (no input_scale): "
            f"{input_scale_missing}"
        )
    elif not write_input_scale:
        print("  [WARN] input_scale skipped (no --calib_file)")
    print(f"FULL ConvRot enabled: {enable_convrot}")
    if enable_convrot:
        print(
            f"  NVFP4 ConvRot Linear: {convrot_nvfp4}, "
            f"plain NVFP4 Linear: {plain_nvfp4}, "
            f"INT8 ConvRot Conv2d: {convrot_int8_conv2d}, "
            f"plain INT8 Conv2d: {plain_int8_conv2d}"
        )
    else:
        print(
            f"  plain NVFP4 Linear: {plain_nvfp4}, "
            f"plain INT8 Conv2d: {plain_int8_conv2d}"
        )

    save_file(new_state_dict, output_path, metadata=metadata)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Diffusion FULL ConvRot convert: Linear→NVFP4 (+ rotate), "
            "Conv2d→INT8 (+ rotate). Pass --calib_file for NVFP4 .input_scale, "
            "HSWQ DualMonitor --keep_ratio FP16 protect, and weight clip amax "
            "from NVFP4/INT8 pack roundtrip MSE. Online act rotate required at "
            "load (loader built separately). Card 1 = --bias_correction."
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
        help=(
            "Calibration prompts text file. Writes per-layer NVFP4 .input_scale "
            "(amax/(F8_E4M3_MAX*F4_E2M1_MAX); ConvRot Linear uses rotated amax). "
            "Also enables HSWQ DualMonitor keep_ratio + NVFP4/INT8 "
            "pack-roundtrip weight clip amax. "
            "Required with --bias_correction or --keep_ratio > 0."
        ),
    )
    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=0.0,
        help=(
            "HSWQ DualMonitor: fraction of Linear/Conv2d layers ranked by "
            "V4 Full-SVD×RMS × NVFP4/INT8 pack-MSE (plus analyze Hard VETO) "
            "to keep in FP16 (no NVFP4/INT8 pack). "
            "Typical 0.05–0.25; default 0.0 = pack all. Requires --calib_file."
        ),
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
            "FULL ConvRot: offline Hadamard on eligible Linear (NVFP4) and "
            "Conv2d (INT8) + convrot stamp. Default ON; --no-convrot = plain packs."
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
            "Skip Linear/Conv2d with in_features/in_channels below this "
            "(0 = convert all eligible)."
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)
    if args.bias_correction and not args.calib_file:
        print("Error: --bias_correction requires --calib_file")
        sys.exit(1)
    if float(args.keep_ratio) > 0.0 and not args.calib_file:
        print("Error: --keep_ratio > 0 requires --calib_file")
        sys.exit(1)
    if float(args.keep_ratio) < 0.0 or float(args.keep_ratio) > 1.0:
        print(f"Error: --keep_ratio must be in [0, 1], got {args.keep_ratio}")
        sys.exit(1)
    if args.groupsize < 4 or (args.groupsize & (args.groupsize - 1)) != 0:
        print(f"Error: --groupsize must be a power of 4 (>=4), got {args.groupsize}")
        sys.exit(1)
    if math.log(args.groupsize, 4) % 1 != 0:
        print(f"Error: --groupsize must be a power of 4, got {args.groupsize}")
        sys.exit(1)

    convert_to_nvfp4_convrot(
        args.model,
        args.output,
        bias_correction=bool(args.bias_correction),
        calib_file=args.calib_file,
        num_calib_samples=args.num_calib_samples,
        num_inference_steps=args.num_inference_steps,
        enable_convrot=bool(args.convrot),
        group_size=int(args.groupsize),
        min_in_features=int(args.min_in_features),
        keep_ratio=float(args.keep_ratio),
    )

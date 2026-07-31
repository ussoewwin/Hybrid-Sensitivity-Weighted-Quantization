"""
SeedVR2 NVFP4 native inference via ComfyUI comfy.ops construction-time injection.

NVFP4 safetensors carry ``comfy_quant`` (format ``nvfp4``) plus
``weight_scale`` (block, float8_e4m3fn) and ``weight_scale_2`` (tensor scale).
Native VRAM-saving load requires Linear modules that already implement
``_load_from_state_dict`` → quantized load at ``load_state_dict`` time.

HSWQ NVFP4 + ConvRot is **not** stock ComfyUI NVFP4:
  - Weights are Hadamard-rotated at pack time.
  - Inference must reverse-rotate activations (HSWQ ``nvfp4_load`` + ``nvfp4_forward``).
  - Stock MixedPrecision ignores ``comfy_quant.convrot`` for NVFP4 → SSIM collapse.

This module wires the HSWQ load/forward stack into SeedVR2 DiT Linears
explicitly (does not rely solely on monkey-patching ``ops._load_quantized_module``
LOAD_GLOBAL). ComfyUI-master is never edited.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import torch


def checkpoint_is_nvfp4(checkpoint_path: Optional[str]) -> bool:
    """True if safetensors has at least one ``*.comfy_quant`` with format nvfp4."""
    if not checkpoint_path:
        return False
    path = str(checkpoint_path)
    if not (path.endswith(".safetensors") or path.endswith(".sft")):
        return False
    if not os.path.isfile(path):
        return False
    try:
        from safetensors import safe_open
    except ImportError:
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if not key.endswith(".comfy_quant"):
                    continue
                raw = f.get_tensor(key)
                if raw.dtype != torch.uint8:
                    continue
                conf = json.loads(raw.numpy().tobytes())
                if conf.get("format") == "nvfp4":
                    return True
    except Exception:
        return False
    return False


def _import_hswq_nvfp4_stack():
    """
    Import HSWQ ``benchmark/nvfp4`` stack.

    Prefer already-importable ``nvfp4`` (bench puts ``benchmark`` on sys.path).
    Otherwise discover ``<hswq>/benchmark`` via ancestors of this file.
    """
    try:
        from nvfp4.nvfp4_conf import is_nvfp4_conf
        from nvfp4.nvfp4_forward import make_nvfp4_linear_forward
        from nvfp4.nvfp4_load import load_nvfp4_linear_module, peek_nvfp4_conf

        return (
            peek_nvfp4_conf,
            is_nvfp4_conf,
            load_nvfp4_linear_module,
            make_nvfp4_linear_forward,
        )
    except ImportError:
        pass

    here = Path(__file__).resolve()
    for parent in here.parents:
        bench = parent / "benchmark"
        if (bench / "nvfp4" / "nvfp4_load.py").is_file():
            bench_s = str(bench)
            if bench_s not in sys.path:
                sys.path.insert(0, bench_s)
            from nvfp4.nvfp4_conf import is_nvfp4_conf
            from nvfp4.nvfp4_forward import make_nvfp4_linear_forward
            from nvfp4.nvfp4_load import load_nvfp4_linear_module, peek_nvfp4_conf

            return (
                peek_nvfp4_conf,
                is_nvfp4_conf,
                load_nvfp4_linear_module,
                make_nvfp4_linear_forward,
            )
    return None


def count_hswq_nvfp4_armed(model: torch.nn.Module) -> dict:
    """Count Linears armed for HSWQ NVFP4 / ConvRot (post-load audit)."""
    nvfp4 = 0
    convrot = 0
    for m in model.modules():
        if not getattr(m, "_hswq_nvfp4", False):
            continue
        nvfp4 += 1
        if getattr(m, "_hswq_nvfp4_convrot", False):
            convrot += 1
    return {"hswq_nvfp4": nvfp4, "hswq_nvfp4_convrot": convrot}


def get_nvfp4_mixed_precision_ops(compute_dtype: torch.dtype = torch.float16) -> Any:
    """
    Return ``comfy.ops.mixed_precision_ops`` for NVFP4 DiT loads.

    Empty ``quant_config``: layers with ``comfy_quant`` become QuantizedTensor;
    unmarked layers load as plain compute_dtype Parameters.

    When the GPU cannot run native NVFP4 matmul (``supports_nvfp4_compute``),
    ``nvfp4`` is listed in ``disabled`` so ComfyUI keeps packed QuantizedTensor
    storage (VRAM savings) but uses dequantized matmul — same as
    ``pick_operations`` for model configs. On Blackwell-class devices the
    format stays enabled for native tensor-core matmul.

    Native NVFP4 activation quantize (``comfy_kitchen.quantize_nvfp4``) accepts
    FP16/BF16 only. SeedVR2 LayerNorm / RMSNorm under ``torch.autocast`` often
    emit float32 into Linear; cast activations to ``compute_dtype`` before the
    Linear path runs ``QuantizedTensor.from_float`` / HSWQ TC.

    When the HSWQ ``benchmark/nvfp4`` stack is available, SeedVR2 Linear:
      - overrides ``_load_from_state_dict`` to call ``load_nvfp4_linear_module``
        (arms ``_hswq_nvfp4`` / ``_hswq_nvfp4_convrot``)
      - uses HSWQ ``make_nvfp4_linear_forward`` (act ConvRot + scaled_mm_nvfp4)
    """
    import comfy.model_management as model_management
    import comfy.ops as comfy_ops

    disabled = []
    if not model_management.supports_nvfp4_compute():
        disabled = ["nvfp4"]

    ops = comfy_ops.mixed_precision_ops(
        quant_config={},
        compute_dtype=compute_dtype,
        full_precision_mm=False,
        disabled=disabled,
    )

    _BaseLinear = ops.Linear
    if compute_dtype in (torch.float16, torch.bfloat16):
        _act_dtype = compute_dtype
    else:
        _act_dtype = torch.float16

    hswq = _import_hswq_nvfp4_stack()
    if hswq is None:
        # Fallback: stock MixedPrecision + FP32→FP16 act cast only.
        # ConvRot checkpoints will be wrong without the HSWQ stack.
        class Linear(_BaseLinear):
            def forward(self, input, *args, **kwargs):
                if (
                    isinstance(input, torch.Tensor)
                    and getattr(self, "quant_format", None) == "nvfp4"
                    and getattr(self, "layout_type", None) is not None
                    and not getattr(self, "_full_precision_mm", False)
                    and input.dtype not in (torch.float16, torch.bfloat16)
                ):
                    input = input.to(dtype=_act_dtype)
                return super().forward(input, *args, **kwargs)

        ops.Linear = Linear
        print(
            "[HSWQ NVFP4] WARNING: benchmark/nvfp4 stack not importable; "
            "ConvRot act rotation will NOT run (stock MixedPrecision only)",
            flush=True,
        )
        return ops

    peek_nvfp4_conf, is_nvfp4_conf, load_nvfp4_linear_module, make_nvfp4_linear_forward = (
        hswq
    )

    # Unwrap one layer if BaseLinear.forward is already HSWQ-wrapped so we do
    # not nest wrappers; then always attach a single HSWQ forward.
    stock_fwd = _BaseLinear.forward
    if getattr(stock_fwd, "_hswq_nvfp4_full_forward", False):
        # Already HSWQ; SeedVR2 subclass still needs explicit load + act cast.
        _hswq_fwd = stock_fwd
    else:
        _hswq_fwd = make_nvfp4_linear_forward(stock_fwd)

    class Linear(_BaseLinear):
        def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            conf = peek_nvfp4_conf(state_dict, prefix)
            if is_nvfp4_conf(conf):
                # Explicit HSWQ load — do NOT go through MixedPrecision's
                # _load_quantized_module (LOAD_GLOBAL may miss monkey-patches).
                # super_load = nn.Module path for bias / leftover keys only.
                load_nvfp4_linear_module(
                    self,
                    lambda *a, **k: torch.nn.Module._load_from_state_dict(self, *a, **k),
                    state_dict,
                    prefix,
                    local_metadata,
                    strict,
                    missing_keys,
                    unexpected_keys,
                    error_msgs,
                    load_extra_params=True,
                )
                return
            # INT8 / plain: keep MixedPrecision quantized load.
            return super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

        def forward(self, input, *args, **kwargs):
            if (
                isinstance(input, torch.Tensor)
                and getattr(self, "quant_format", None) == "nvfp4"
                and getattr(self, "layout_type", None) is not None
                and input.dtype not in (torch.float16, torch.bfloat16)
            ):
                # Cast before HSWQ path too (kitchen NVFP4 act quant is FP16/BF16).
                if not getattr(self, "_full_precision_mm", False) or getattr(
                    self, "_hswq_nvfp4_convrot", False
                ):
                    input = input.to(dtype=_act_dtype)
            return _hswq_fwd(self, input, *args, **kwargs)

    ops.Linear = Linear
    print(
        "[HSWQ NVFP4] SeedVR2 Linear wired: explicit nvfp4_load + TC forward "
        "(ConvRot act rotation owned by HSWQ)",
        flush=True,
    )
    return ops

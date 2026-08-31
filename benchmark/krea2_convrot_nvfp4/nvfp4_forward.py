"""
HSWQ-owned NVFP4 Linear forward path (ConvRot × NVFP4).

ComfyUI / comfy_kitchen do **not** ship ConvRot×NVFP4 load+forward. This
package owns the full inference path:

  1) reshape act to 2D
  2) FULL ConvRot act rotation (dense Hadamard GEMM, fp32 accumulation)
  3) cast weight/bias when off-device
  4) pooled act NVFP4 quantize → cuBLAS FP4 Tensor-Core GEMM with the weight
     kept PACKED (single packed residency — no dense float copy, VRAM win).
     Fallback only: bake packed NVFP4 → dense float Parameter → ``F.linear``.
  5) reshape with module.out_features (never QT storage shape[0])

Never edits ComfyUI-master; installed via monkey-patch on MixedPrecision Linear.
"""
from __future__ import annotations

import logging

from .nvfp4_gemm import bake_nvfp4_weight_inplace
from .nvfp4_hadamard import build_hadamard
from .nvfp4_runtime import rotate_last_dim_pooled
from .nvfp4_tc_gate import note_scaled_mm_failure, nvfp4_tc_enabled

logger = logging.getLogger(__name__)

# Counters for bench / diagnostics (reset per run if needed)
_TC_HITS = 0
_DEQUANT_FALLBACKS = 0
_CONVROT_ACT_ROTATES = 0
_TC_FLOPS = 0


def reset_nvfp4_forward_stats() -> None:
    global _TC_HITS, _DEQUANT_FALLBACKS, _CONVROT_ACT_ROTATES, _TC_FLOPS
    _TC_HITS = 0
    _DEQUANT_FALLBACKS = 0
    _CONVROT_ACT_ROTATES = 0
    _TC_FLOPS = 0


def nvfp4_forward_stats() -> dict:
    return {
        "scaled_mm_hits": _TC_HITS,
        "dequant_fallbacks": _DEQUANT_FALLBACKS,
        "convrot_act_rotates": _CONVROT_ACT_ROTATES,
        "tc_flops": _TC_FLOPS,
    }


def _slice_nvfp4_mm_out(result, orig_m: int, orig_n: int):
    if result.shape[0] != orig_m or result.shape[1] != orig_n:
        return result[:orig_m, :orig_n]
    return result


def scaled_mm_nvfp4_linear(input_qt, weight_qt, bias):
    """Kitchen / tritant NVFP4 linear (QT path; used as fallback)."""
    global _TC_HITS, _DEQUANT_FALLBACKS
    import torch
    import torch.nn.functional as F
    import comfy_kitchen as ck
    from comfy_kitchen.tensor.base import QuantizedTensor
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout

    if not (
        isinstance(input_qt, QuantizedTensor)
        and isinstance(weight_qt, QuantizedTensor)
        and input_qt._layout_cls == "TensorCoreNVFP4Layout"
        and weight_qt._layout_cls == "TensorCoreNVFP4Layout"
    ):
        _DEQUANT_FALLBACKS += 1
        return F.linear(input_qt, weight_qt, bias)
    if input_qt._qdata.dim() != 2:
        _DEQUANT_FALLBACKS += 1
        return F.linear(input_qt, weight_qt, bias)
    if getattr(input_qt._params, "transposed", False) or getattr(
        weight_qt._params, "transposed", False
    ):
        _DEQUANT_FALLBACKS += 1
        return F.linear(input_qt, weight_qt, bias)

    if isinstance(bias, QuantizedTensor):
        bias = bias.dequantize()

    a_qdata, scale_a, block_scale_a = TensorCoreNVFP4Layout.get_plain_tensors(input_qt)
    w_qdata, scale_b, block_scale_b = TensorCoreNVFP4Layout.get_plain_tensors(weight_qt)
    out_dtype = input_qt._params.orig_dtype
    try:
        result = ck.scaled_mm_nvfp4(
            a_qdata,
            w_qdata,
            tensor_scale_a=scale_a,
            tensor_scale_b=scale_b,
            block_scale_a=block_scale_a,
            block_scale_b=block_scale_b,
            bias=bias,
            out_dtype=out_dtype,
        )
        orig_m = input_qt._params.orig_shape[0]
        orig_n = weight_qt._params.orig_shape[0]  # (out, in)
        _TC_HITS += 1
        return _slice_nvfp4_mm_out(result, orig_m, orig_n)
    except (RuntimeError, TypeError) as e:
        logger.warning("[HSWQ NVFP4] scaled_mm_nvfp4 failed: %s → F.linear dequant", e)
        _DEQUANT_FALLBACKS += 1
        return F.linear(input_qt, weight_qt, bias)


def _plain_weight_cached(module, weight_qt):
    """Extract packed weight primitives once per module (weight stays packed).

    Returns (w_qdata uint8 (N_pad, K_pad/2), scale_b f32 (1,), block_scale_b
    f8e4m3 swizzled, orig_n). Packed residency is the VRAM win — never bake
    on the TC path.
    """
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout

    cached = getattr(module, "_hswq_nvfp4_w_plain", None)
    if cached is not None and cached[0] is weight_qt._qdata:
        return cached[1], cached[2], cached[3], cached[4]
    import torch

    w_qdata, scale_b, block_scale_b = TensorCoreNVFP4Layout.get_plain_tensors(
        weight_qt
    )
    if scale_b.dtype != torch.float32 or scale_b.dim() != 1:
        scale_b = scale_b.reshape(-1).float()
    orig_n = int(weight_qt._params.orig_shape[0])
    module._hswq_nvfp4_w_plain = (
        weight_qt._qdata,
        w_qdata,
        scale_b,
        block_scale_b,
        orig_n,
    )
    return w_qdata, scale_b, block_scale_b, orig_n


def _tc_forward_pooled(module, input_2d, weight_qt, bias, act_scale, out_dtype):
    """ConvRot act (already rotated) → pooled NVFP4 quant → cuBLAS FP4 TC GEMM.

    Weight stays PACKED NVFP4 resident (VRAM win vs bake); act is quantized
    per call into pooled buffers; GEMM is the raw ``_C.cublas_gemm_blockwise_fp4``
    primitive (SM120-verified path B — never torch native ``F.scaled_mm``).
    Any failure → ``None`` → caller bakes weight + ``F.linear`` (dense fallback).
    """
    global _TC_HITS, _DEQUANT_FALLBACKS, _TC_FLOPS
    import torch
    import torch.nn.functional as F
    from comfy_kitchen.tensor.base import QuantizedTensor
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout

    if not nvfp4_tc_enabled():
        _DEQUANT_FALLBACKS += 1
        return None

    # Already baked on a prior fallback: weight is plain float Parameter.
    if not isinstance(weight_qt, QuantizedTensor):
        try:
            w_f = weight_qt
            if isinstance(bias, QuantizedTensor):
                bias = bias.dequantize()
            if bias is not None and (
                bias.device != input_2d.device or bias.dtype != out_dtype
            ):
                bias = bias.to(device=input_2d.device, dtype=out_dtype)
            if w_f.device != input_2d.device or w_f.dtype != out_dtype:
                w_f = w_f.to(device=input_2d.device, dtype=out_dtype)
            result = F.linear(input_2d, w_f, bias)
            _TC_HITS += 1
            return result
        except (RuntimeError, TypeError, ValueError) as e:
            note_scaled_mm_failure(e)
            _DEQUANT_FALLBACKS += 1
            return None

    if weight_qt._layout_cls != "TensorCoreNVFP4Layout":
        _DEQUANT_FALLBACKS += 1
        return None
    if getattr(weight_qt._params, "transposed", False):
        _DEQUANT_FALLBACKS += 1
        return None

    if isinstance(bias, QuantizedTensor):
        bias = bias.dequantize()

    from .nvfp4_runtime import (
        ensure_act_scale_cached,
        quantize_nvfp4_act_pooled,
        scaled_mm_nvfp4_pooled,
    )

    orig_m, orig_k = int(input_2d.shape[0]), int(input_2d.shape[1])
    needs_padding = TensorCoreNVFP4Layout.get_padded_shape((orig_m, orig_k)) != (
        orig_m,
        orig_k,
    )
    # cuBLAS FP4 TN gate: padded K must hold K%32==0 (packed K bytes %16).
    # Checked pre-C-call: unsupported shapes inside cuBLAS = sticky poison.
    if ((orig_k + 15) // 16 * 16) % 32 != 0:
        _DEQUANT_FALLBACKS += 1
        return None

    # Checkpoints may omit input_scale (placeholder ones). Per-call amax in the
    # ROTATED domain (converter: rotate first, then amax) — ones as tensor_scale
    # collapses NVFP4 act grids (SSIM~0.18); step-0 freeze mis-scales later steps.
    scale_a = ensure_act_scale_cached(module, input_2d, act_scale)
    try:
        w_qdata, scale_b, block_scale_b, orig_n = _plain_weight_cached(
            module, weight_qt
        )
        flops = orig_m * orig_k * orig_n * 2

        # Cache alpha; rebind only when the act scale object changes
        # (placeholder → frozen amax swap).
        cached_alpha = getattr(module, "_hswq_nvfp4_alpha", None)
        bound = getattr(module, "_hswq_nvfp4_alpha_bound_scale", None)
        if cached_alpha is None or bound is not scale_a:
            alpha = scale_a * scale_b
            if alpha.dtype != torch.float32:
                alpha = alpha.to(dtype=torch.float32)
            if alpha.dim() == 0:
                alpha = alpha.reshape(1)
            module._hswq_nvfp4_alpha = alpha
            module._hswq_nvfp4_alpha_bound_scale = scale_a
        else:
            alpha = cached_alpha

        a_qdata, block_scale_a, _pr, _pc = quantize_nvfp4_act_pooled(
            input_2d, scale_a, pad_16x=needs_padding
        )
        result = scaled_mm_nvfp4_pooled(
            a_qdata,
            w_qdata,
            tensor_scale_a=scale_a,
            tensor_scale_b=scale_b,
            block_scale_a=block_scale_a,
            block_scale_b=block_scale_b,
            bias=bias,
            out_dtype=out_dtype,
            alpha=alpha,
            orig_m=orig_m,
            orig_n=orig_n,
        )
        _TC_HITS += 1
        _TC_FLOPS += flops
        return result
    except (RuntimeError, TypeError, ValueError) as e:
        note_scaled_mm_failure(e)
        _DEQUANT_FALLBACKS += 1
        return None


def make_nvfp4_linear_forward(stock_forward):
    """
    Return a Linear.forward replacement.

    For modules flagged ``_hswq_nvfp4`` (set at load), run the HSWQ TC path.
    All other layers keep stock_forward unchanged.
    """
    import torch
    import comfy.model_management
    from comfy.ops import cast_bias_weight, run_every_op, uncast_bias_weight

    def forward_nvfp4(self, input, *args, **kwargs):
        global _CONVROT_ACT_ROTATES

        if not getattr(self, "_hswq_nvfp4", False):
            return stock_forward(self, input, *args, **kwargs)

        # Training / LoRA / forced cast: fall back to stock.
        # ConvRot + full_precision_mm still needs act rotation before stock dequant.
        if input.requires_grad or getattr(self, "comfy_force_cast_weights", False):
            return stock_forward(self, input, *args, **kwargs)
        if len(getattr(self, "weight_function", [])) or len(getattr(self, "bias_function", [])):
            return stock_forward(self, input, *args, **kwargs)

        # GPU lacks NVFP4 TC: stock dequant mm, but MUST rotate acts if ConvRot.
        if getattr(self, "_full_precision_mm", False):
            if not getattr(self, "_hswq_nvfp4_convrot", False):
                return stock_forward(self, input, *args, **kwargs)
            input_shape = input.shape
            reshaped_nd = input.ndim >= 3
            input_2d = input.reshape(-1, input_shape[-1]) if reshaped_nd else input
            if input_2d.ndim != 2:
                return stock_forward(self, input, *args, **kwargs)
            gs = int(getattr(self, "_hswq_nvfp4_convrot_groupsize", 256) or 256)
            h = getattr(self, "_hswq_nvfp4_H", None)
            if h is None or h.device != input_2d.device or h.dtype != input_2d.dtype:
                h = build_hadamard(gs, device=input_2d.device, dtype=input_2d.dtype)
                self._hswq_nvfp4_H = h
            input_2d = rotate_last_dim_pooled(input_2d, h, gs)
            _CONVROT_ACT_ROTATES += 1
            if reshaped_nd:
                input = input_2d.reshape((*input_shape[:-1], input_shape[-1]))
            else:
                input = input_2d
            return stock_forward(self, input, *args, **kwargs)

        run_every_op()
        input_shape = input.shape
        compute_dtype = input.dtype

        # 1) Reshape ≥3D → 2D first (same last-dim math; cheaper than rotating ND)
        reshaped_nd = input.ndim >= 3
        input_2d = input.reshape(-1, input_shape[-1]) if reshaped_nd else input
        if input_2d.ndim != 2:
            return stock_forward(self, input, *args, **kwargs)

        # 2) FULL ConvRot: dense Hadamard GEMM act rotation (fp32 accumulation).
        #    rotate_last_dim_pooled rotates in fp32 like the butterfly did, but a
        #    dense 256x256 GEMM measured ~15x faster than the butterfly stages.
        if getattr(self, "_hswq_nvfp4_convrot", False):
            gs = int(getattr(self, "_hswq_nvfp4_convrot_groupsize", 256) or 256)
            h = getattr(self, "_hswq_nvfp4_H", None)
            if h is None or h.device != input_2d.device or h.dtype != input_2d.dtype:
                h = build_hadamard(gs, device=input_2d.device, dtype=input_2d.dtype)
                self._hswq_nvfp4_H = h
            input_2d = rotate_last_dim_pooled(input_2d, h, gs)
            _CONVROT_ACT_ROTATES += 1

        # 3) Weight / bias: skip cast_bias_weight when already on-device QT
        #    (cast+sync every Linear was a major share of NVFP4 > FP16 wall time).
        offload_stream = None
        weight = self.weight
        if isinstance(weight, torch.nn.Parameter):
            weight = weight.data
        bias = self.bias.data if self.bias is not None else None
        need_cast = weight.device != input_2d.device or (
            bias is not None and bias.device != input_2d.device
        )
        if need_cast or hasattr(self, "_v"):
            weight, bias, offload_stream = cast_bias_weight(
                self,
                input_2d,
                offloadable=True,
                compute_dtype=compute_dtype,
                want_requant=True,
            )

        scale = getattr(self, "input_scale", None)
        if scale is not None:
            if isinstance(scale, torch.nn.Parameter):
                scale = scale.data
            if scale.device != input.device:
                scale = comfy.model_management.cast_to_device(scale, input.device, None)

        layout = getattr(self, "layout_type", None)
        if layout is None:
            if offload_stream is not None:
                uncast_bias_weight(self, weight, bias, offload_stream)
            return stock_forward(self, input, *args, **kwargs)

        # 4) packed-NVFP4 FP4 TC GEMM (weight stays packed); bake+F.linear only
        #    as fallback inside _tc_forward_pooled / below.
        out_2d = _tc_forward_pooled(
            self, input_2d, weight, bias, scale, compute_dtype
        )
        # Drop local QT ref so baked-away packed weight can be GC'd.
        weight = (
            self.weight.data
            if isinstance(self.weight, torch.nn.Parameter)
            else self.weight
        )
        if out_2d is None:
            # Do NOT re-enter registry cuda CUBLAS via QT→ck.scaled_mm.
            # Bake + float Linear only (sticky-safe; ConvRot already applied).
            import torch.nn.functional as F
            from comfy_kitchen.tensor.base import QuantizedTensor as _QT

            if isinstance(weight, _QT):
                w_f = bake_nvfp4_weight_inplace(self, weight, compute_dtype)
                weight = self.weight.data
            else:
                w_f = weight
            b_f = bias
            if isinstance(b_f, _QT):
                b_f = b_f.dequantize()
            out_2d = F.linear(input_2d, w_f, b_f)

        # 5) Restore rank with logical out_features (never QT storage shape[0])
        if reshaped_nd:
            out = out_2d.reshape((*input_shape[:-1], int(self.out_features)))
        else:
            out = out_2d

        if offload_stream is not None:
            uncast_bias_weight(self, weight, bias, offload_stream)
        return out

    forward_nvfp4._hswq_nvfp4_full_forward = True  # type: ignore[attr-defined]
    return forward_nvfp4

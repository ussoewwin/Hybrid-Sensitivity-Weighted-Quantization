"""
HSWQ-owned NVFP4 Linear forward path (ConvRot × NVFP4).

ComfyUI / comfy_kitchen do **not** ship ConvRot×NVFP4 load+forward. This
package owns the full inference path:

  1) reshape act to 2D
  2) FULL ConvRot act rotation (Hadamard / butterfly)
  3) cast weight/bias when off-device
  4) one-shot bake packed NVFP4 → dense float Parameter (free QT) → ``F.linear``
     (never kitchen ``scaled_mm_nvfp4`` / CUBLAS; no QT+float dual VRAM)
  5) reshape with module.out_features (never QT storage shape[0])

Never edits ComfyUI-master; installed via monkey-patch on MixedPrecision Linear.
"""
from __future__ import annotations

import logging

from .nvfp4_gemm import bake_nvfp4_weight_inplace, hswq_scaled_mm_nvfp4
from .nvfp4_hadamard import rotate_last_dim_fast
from .nvfp4_tc_gate import note_scaled_mm_failure, nvfp4_tc_enabled

logger = logging.getLogger(__name__)

# Counters for bench / diagnostics (reset per run if needed)
_TC_HITS = 0
_DEQUANT_FALLBACKS = 0
_CONVROT_ACT_ROTATES = 0


def reset_nvfp4_forward_stats() -> None:
    global _TC_HITS, _DEQUANT_FALLBACKS, _CONVROT_ACT_ROTATES
    _TC_HITS = 0
    _DEQUANT_FALLBACKS = 0
    _CONVROT_ACT_ROTATES = 0


def nvfp4_forward_stats() -> dict:
    return {
        "scaled_mm_hits": _TC_HITS,
        "dequant_fallbacks": _DEQUANT_FALLBACKS,
        "convrot_act_rotates": _CONVROT_ACT_ROTATES,
    }


def _slice_nvfp4_mm_out(result, orig_m: int, orig_n: int):
    if result.shape[0] != orig_m or result.shape[1] != orig_n:
        return result[:orig_m, :orig_n]
    return result


def scaled_mm_nvfp4_linear(input_qt, weight_qt, bias):
    """QT×QT path via HSWQ-owned dequant GEMM (never kitchen scaled_mm)."""
    global _TC_HITS, _DEQUANT_FALLBACKS
    import torch
    import torch.nn.functional as F
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
    if not nvfp4_tc_enabled():
        _DEQUANT_FALLBACKS += 1
        return F.linear(input_qt, weight_qt, bias)

    if scale_a.dtype != torch.float32 or scale_a.dim() != 1:
        scale_a = scale_a.reshape(-1).float()
    if scale_b.dtype != torch.float32 or scale_b.dim() != 1:
        scale_b = scale_b.reshape(-1).float()

    result = hswq_scaled_mm_nvfp4(
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


def _tc_forward_pooled(module, input_2d, weight_qt, bias, act_scale, out_dtype):
    """ConvRot act (already rotated) + bake NVFP4 weight → ``F.linear``.

    Never calls kitchen ``scaled_mm_nvfp4``. First call replaces packed QT
    ``module.weight`` with dense float (single residency). Later calls are
    plain float Linear (+ ConvRot already applied by caller).
    ``act_scale`` is unused on this path (kept for call-site API).
    """
    global _TC_HITS, _DEQUANT_FALLBACKS
    import torch
    import torch.nn.functional as F
    from comfy_kitchen.tensor.base import QuantizedTensor

    _ = act_scale

    if not nvfp4_tc_enabled():
        _DEQUANT_FALLBACKS += 1
        return None

    # Already baked on a prior step: weight is plain float Parameter.
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

    try:
        w_f = bake_nvfp4_weight_inplace(module, weight_qt, out_dtype)
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
            input_2d = rotate_last_dim_fast(input_2d, gs)
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

        # 2) FULL ConvRot: fast O(N log N) float32 butterfly act rotation
        if getattr(self, "_hswq_nvfp4_convrot", False):
            gs = int(getattr(self, "_hswq_nvfp4_convrot_groupsize", 256) or 256)
            input_2d = rotate_last_dim_fast(input_2d, gs)
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

        # 4) one-shot bake QT→float Parameter (free packed) → F.linear
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

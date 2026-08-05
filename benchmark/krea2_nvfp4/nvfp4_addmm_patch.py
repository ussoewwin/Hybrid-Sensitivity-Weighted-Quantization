"""Fill kitchen NVFP4 gap: aten.addmm / linear / mm for TensorCoreNVFP4Layout.

Stock kitchen handlers require BOTH operands to be QuantizedTensor. Comfy
MixedPrecision F.linear(float_act, NVFP4_weight) therefore falls through to
dequantize_args every call: packed weights stay resident AND full FP16/BF16
weights are materialized → dedicated VRAM full + shared-GPU spill (~27 GB).

This module overwrites kitchen handlers so float act + NVFP4 weight:
  quantize act (amax) → scaled_mm_nvfp4  (same contract as TC forward).

Runtime-only — does not edit ComfyUI-master or site-packages files.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
_REGISTERED = False


def register_nvfp4_addmm_handler() -> bool:
    """Install / overwrite NVFP4 addmm+linear+mm → scaled_mm (float-act aware)."""
    global _REGISTERED
    if _REGISTERED:
        return True

    try:
        import torch
        import comfy_kitchen as ck
        from comfy_kitchen.tensor.base import (
            QuantizedTensor,
            dequantize_args,
            _LAYOUT_DISPATCH_TABLE,
        )
        from comfy_kitchen.tensor.nvfp4 import (
            TensorCoreNVFP4Layout,
            _slice_to_original_shape,
        )
        from .nvfp4_tc_gate import (
            announce_tc_status_at_register,
            note_scaled_mm_failure,
            nvfp4_tc_enabled,
        )
        from .nvfp4_runtime import ensure_act_scale_amax, quantize_nvfp4_act_pooled
    except Exception as e:
        logger.warning("[HSWQ NVFP4] addmm register skipped (import): %s", e)
        return False

    announce_tc_status_at_register()

    def _is_nvfp4_qt(t) -> bool:
        return (
            isinstance(t, QuantizedTensor)
            and getattr(t, "_layout_cls", None) == "TensorCoreNVFP4Layout"
        )

    def _scaled_mm_both_qt(a_qt, b_qt, *, bias, out_dtype, weight_is_t: bool):
        """Both operands already NVFP4 QT (kitchen fast path)."""
        a_trans = getattr(a_qt._params, "transposed", False)
        b_trans = getattr(b_qt._params, "transposed", False)
        if weight_is_t:
            # addmm / mm(x, w.t()): need a not-t, b transposed
            if a_trans or not b_trans:
                return None
        else:
            # linear(x, w): neither transposed
            if a_trans or b_trans:
                return None
        if a_qt._qdata.dim() != 2:
            return None
        a_qdata, scale_a, block_scale_a = TensorCoreNVFP4Layout.get_plain_tensors(a_qt)
        b_qdata, scale_b, block_scale_b = TensorCoreNVFP4Layout.get_plain_tensors(b_qt)
        result = ck.scaled_mm_nvfp4(
            a_qdata,
            b_qdata,
            tensor_scale_a=scale_a,
            tensor_scale_b=scale_b,
            block_scale_a=block_scale_a,
            block_scale_b=block_scale_b,
            bias=bias,
            out_dtype=out_dtype,
        )
        orig_m = a_qt._params.orig_shape[0]
        if weight_is_t:
            orig_n = b_qt._params.orig_shape[1]
        else:
            orig_n = b_qt._params.orig_shape[0]
        return _slice_to_original_shape(result, orig_m, orig_n)

    def _scaled_mm_float_act(mat1, w_qt, *, bias, out_dtype, weight_is_t: bool):
        """Float activation + NVFP4 weight → pooled act quant → scaled_mm.

        Never materializes full dequantized weight (VRAM dual-hold ban).
        Layout.quantize returns (qdata, params) — not a QT; use pooled CUDA path.
        """
        if not _is_nvfp4_qt(w_qt):
            return None
        w_trans = getattr(w_qt._params, "transposed", False)
        if weight_is_t:
            if not w_trans:
                return None
        else:
            if w_trans:
                return None
        if not isinstance(mat1, torch.Tensor) or isinstance(mat1, QuantizedTensor):
            return None
        if mat1.dim() != 2:
            return None
        if not nvfp4_tc_enabled():
            return None

        scale_a = ensure_act_scale_amax(mat1)
        orig_m = int(mat1.shape[0])
        a_qdata, block_scale_a, _pr, _pc = quantize_nvfp4_act_pooled(
            mat1, scale_a, pad_16x=True
        )
        w_qdata, scale_b, block_scale_b = TensorCoreNVFP4Layout.get_plain_tensors(w_qt)
        if out_dtype is None:
            out_dtype = mat1.dtype
        if isinstance(bias, QuantizedTensor):
            bias = bias.dequantize()
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
        if weight_is_t:
            orig_n = int(w_qt._params.orig_shape[1])
        else:
            orig_n = int(w_qt._params.orig_shape[0])
        return _slice_to_original_shape(result, orig_m, orig_n)

    def _handle_nvfp4_addmm(qt, args, kwargs):
        """NVFP4 addmm: bias + input @ weight.T (F.linear with bias)."""
        bias, mat1, mat2 = args[0], args[1], args[2]
        out_dtype = kwargs.get("out_dtype", None)

        if _is_nvfp4_qt(mat1) and _is_nvfp4_qt(mat2):
            if not nvfp4_tc_enabled():
                return torch.addmm(*dequantize_args((bias, mat1, mat2)))
            try:
                out = _scaled_mm_both_qt(
                    mat1, mat2, bias=bias, out_dtype=out_dtype or mat1._params.orig_dtype,
                    weight_is_t=True,
                )
                if out is not None:
                    return out
            except (RuntimeError, TypeError) as e:
                note_scaled_mm_failure(e)
            return torch.addmm(*dequantize_args((bias, mat1, mat2)))

        # Stock path: float act + NVFP4 weight (mat2 = w.t())
        if _is_nvfp4_qt(mat2) and isinstance(mat1, torch.Tensor):
            try:
                out = _scaled_mm_float_act(
                    mat1, mat2, bias=bias, out_dtype=out_dtype or mat1.dtype,
                    weight_is_t=True,
                )
                if out is not None:
                    return out
            except (RuntimeError, TypeError) as e:
                note_scaled_mm_failure(e)
            logger.warning(
                "[HSWQ NVFP4] addmm float×NVFP4 failed → dequant "
                "(VRAM spike risk)"
            )
            return torch.addmm(*dequantize_args((bias, mat1, mat2)))

        return torch.addmm(*dequantize_args((bias, mat1, mat2)))

    def _handle_nvfp4_linear(qt, args, kwargs):
        """NVFP4 linear: input @ weight.T + bias (float act OK)."""
        input_tensor, weight = args[0], args[1]
        bias = args[2] if len(args) > 2 else None
        out_dtype = kwargs.get("out_dtype", None)

        if _is_nvfp4_qt(input_tensor) and _is_nvfp4_qt(weight):
            if not nvfp4_tc_enabled():
                return torch.nn.functional.linear(
                    *dequantize_args((input_tensor, weight, bias))
                )
            try:
                out = _scaled_mm_both_qt(
                    input_tensor,
                    weight,
                    bias=bias,
                    out_dtype=out_dtype or input_tensor._params.orig_dtype,
                    weight_is_t=False,
                )
                if out is not None:
                    return out
            except (RuntimeError, TypeError) as e:
                note_scaled_mm_failure(e)
            return torch.nn.functional.linear(
                *dequantize_args((input_tensor, weight, bias))
            )

        if _is_nvfp4_qt(weight) and isinstance(input_tensor, torch.Tensor):
            # ≥3D → 2D for kitchen kernel
            shape = input_tensor.shape
            x2d = input_tensor.reshape(-1, shape[-1]) if input_tensor.dim() >= 3 else input_tensor
            try:
                out = _scaled_mm_float_act(
                    x2d,
                    weight,
                    bias=bias,
                    out_dtype=out_dtype or input_tensor.dtype,
                    weight_is_t=False,
                )
                if out is not None:
                    if input_tensor.dim() >= 3:
                        return out.reshape(*shape[:-1], out.shape[-1])
                    return out
            except (RuntimeError, TypeError) as e:
                note_scaled_mm_failure(e)
            logger.warning(
                "[HSWQ NVFP4] linear float×NVFP4 failed → dequant "
                "(VRAM spike risk)"
            )
            return torch.nn.functional.linear(
                *dequantize_args((input_tensor, weight, bias))
            )

        return torch.nn.functional.linear(
            *dequantize_args((input_tensor, weight, bias))
        )

    def _handle_nvfp4_mm(qt, args, kwargs):
        """NVFP4 mm: a @ b (often mm(x, w.t()) from linear)."""
        a, b = args[0], args[1]
        out_dtype = kwargs.get("out_dtype", None)

        if _is_nvfp4_qt(a) and _is_nvfp4_qt(b):
            if not nvfp4_tc_enabled():
                return torch.mm(*dequantize_args(args))
            try:
                out = _scaled_mm_both_qt(
                    a, b, bias=None, out_dtype=out_dtype or a._params.orig_dtype,
                    weight_is_t=True,
                )
                if out is not None:
                    return out
            except (RuntimeError, TypeError) as e:
                note_scaled_mm_failure(e)
            return torch.mm(*dequantize_args(args))

        if _is_nvfp4_qt(b) and isinstance(a, torch.Tensor):
            try:
                out = _scaled_mm_float_act(
                    a, b, bias=None, out_dtype=out_dtype or a.dtype,
                    weight_is_t=True,
                )
                if out is not None:
                    return out
            except (RuntimeError, TypeError) as e:
                note_scaled_mm_failure(e)
            logger.warning(
                "[HSWQ NVFP4] mm float×NVFP4 failed → dequant (VRAM spike risk)"
            )
            return torch.mm(*dequantize_args(args))

        return torch.mm(*dequantize_args(args))

    # Force overwrite kitchen handlers (float-act aware). register_layout_op overwrites.
    for op, handler in (
        (torch.ops.aten.addmm.default, _handle_nvfp4_addmm),
        (torch.ops.aten.linear.default, _handle_nvfp4_linear),
        (torch.ops.aten.mm.default, _handle_nvfp4_mm),
    ):
        if op not in _LAYOUT_DISPATCH_TABLE:
            _LAYOUT_DISPATCH_TABLE[op] = {}
        _LAYOUT_DISPATCH_TABLE[op][TensorCoreNVFP4Layout] = handler

    _REGISTERED = True
    print(
        "[HSWQ NVFP4] registered aten.addmm/linear/mm for TensorCoreNVFP4Layout "
        "(float act + NVFP4 weight → scaled_mm; no full-weight dequant)",
        flush=True,
    )
    return True

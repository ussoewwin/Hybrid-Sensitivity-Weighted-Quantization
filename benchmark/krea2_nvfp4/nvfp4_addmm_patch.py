"""Fill kitchen NVFP4 gap: aten.addmm only (TensorCoreNVFP4Layout).

ComfyUI-master MixedPrecision already does the full-size path:
  reshape ND→2D → QuantizedTensor.from_float(act) → F.linear(both QT)
  → kitchen linear → scaled_mm_nvfp4 → _slice_to_original_shape.

Kitchen NVFP4 registers linear + mm with that slice logic, but unlike MXFP8
has **no** addmm handler. When F.linear with bias decomposes to addmm,
dispatch falls through and dequantizes.

This module registers **only** addmm, mirroring
``comfy_kitchen.tensor.mxfp8._handle_mxfp8_addmm`` + NVFP4
``_slice_to_original_shape`` / ``scaled_mm_nvfp4``.

Does **not** invent float-act×NVFP4 paths.
Does **not** overwrite kitchen linear/mm (already correct for both QT).
Runtime-only — does not edit ComfyUI-master or site-packages.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
_REGISTERED = False

_ADDMM_SCALED_HITS = 0
_ADDMM_DEQUANT_FALLBACKS = 0


def reset_nvfp4_addmm_stats() -> None:
    global _ADDMM_SCALED_HITS, _ADDMM_DEQUANT_FALLBACKS
    _ADDMM_SCALED_HITS = 0
    _ADDMM_DEQUANT_FALLBACKS = 0


def nvfp4_addmm_stats() -> dict:
    return {
        "addmm_scaled_mm_hits": _ADDMM_SCALED_HITS,
        "addmm_dequant_fallbacks": _ADDMM_DEQUANT_FALLBACKS,
        # legacy key for older bench prints (no hard-ban path anymore)
        "addmm_hard_fails": 0,
    }


def register_nvfp4_addmm_handler() -> bool:
    """Install NVFP4 addmm only — same contract as kitchen MXFP8 addmm."""
    global _REGISTERED
    if _REGISTERED:
        return True

    try:
        import torch
        import comfy_kitchen as ck
        from comfy_kitchen.tensor.base import (
            QuantizedTensor,
            _LAYOUT_DISPATCH_TABLE,
            dequantize_args,
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
    except Exception as e:
        logger.warning("[HSWQ NVFP4] addmm register skipped (import): %s", e)
        return False

    announce_tc_status_at_register()

    def _handle_nvfp4_addmm(qt, args, kwargs):
        """NVFP4 addmm: bias + input @ weight.T (decomposed from F.linear with bias).

        Mirror of kitchen ``_handle_mxfp8_addmm`` / NVFP4 linear slice contract.
        """
        global _ADDMM_SCALED_HITS, _ADDMM_DEQUANT_FALLBACKS
        bias, mat1, mat2 = args[0], args[1], args[2]

        if not (isinstance(mat1, QuantizedTensor) and isinstance(mat2, QuantizedTensor)):
            _ADDMM_DEQUANT_FALLBACKS += 1
            return torch.addmm(*dequantize_args((bias, mat1, mat2)))
        if mat1._qdata.dim() != 2:
            _ADDMM_DEQUANT_FALLBACKS += 1
            return torch.addmm(*dequantize_args((bias, mat1, mat2)))

        input_transposed = getattr(mat1._params, "transposed", False)
        weight_transposed = getattr(mat2._params, "transposed", False)
        # MXFP8 addmm: need mat2 logically transposed (W.t() from linear)
        if input_transposed or not weight_transposed:
            _ADDMM_DEQUANT_FALLBACKS += 1
            return torch.addmm(*dequantize_args((bias, mat1, mat2)))

        if not nvfp4_tc_enabled():
            _ADDMM_DEQUANT_FALLBACKS += 1
            return torch.addmm(*dequantize_args((bias, mat1, mat2)))

        input_qdata, scale_a, block_scale_a = TensorCoreNVFP4Layout.get_plain_tensors(mat1)
        weight_qdata, scale_b, block_scale_b = TensorCoreNVFP4Layout.get_plain_tensors(mat2)
        out_dtype = kwargs.get("out_dtype", mat1._params.orig_dtype)
        if isinstance(bias, QuantizedTensor):
            bias = bias.dequantize()

        try:
            result = ck.scaled_mm_nvfp4(
                input_qdata,
                weight_qdata,
                tensor_scale_a=scale_a,
                tensor_scale_b=scale_b,
                block_scale_a=block_scale_a,
                block_scale_b=block_scale_b,
                bias=bias,
                out_dtype=out_dtype,
            )
            orig_m = mat1._params.orig_shape[0]
            orig_n = mat2._params.orig_shape[1]
            _ADDMM_SCALED_HITS += 1
            return _slice_to_original_shape(result, orig_m, orig_n)
        except (RuntimeError, TypeError) as e:
            note_scaled_mm_failure(e)
            logger.warning("NVFP4 addmm failed: %s, falling back to dequantization", e)
            _ADDMM_DEQUANT_FALLBACKS += 1
            return torch.addmm(*dequantize_args((bias, mat1, mat2)))

    if torch.ops.aten.addmm.default not in _LAYOUT_DISPATCH_TABLE:
        _LAYOUT_DISPATCH_TABLE[torch.ops.aten.addmm.default] = {}
    _LAYOUT_DISPATCH_TABLE[torch.ops.aten.addmm.default][TensorCoreNVFP4Layout] = (
        _handle_nvfp4_addmm
    )

    _REGISTERED = True
    print(
        "[HSWQ NVFP4] registered aten.addmm for TensorCoreNVFP4Layout "
        "(kitchen MXFP8-shaped; both QT → scaled_mm + _slice_to_original_shape; "
        "linear/mm left to stock kitchen)",
        flush=True,
    )
    return True

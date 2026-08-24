"""Fill kitchen NVFP4 gap: register aten.addmm for TensorCoreNVFP4Layout.

comfy_kitchen registers addmm for INT8 / MXFP8 / FP8 / SVDQuant / ConvRotW4A4,
but NOT for TensorCoreNVFP4Layout. PyTorch F.linear(bias=...) often decomposes
to aten.addmm.default → unhandled → full dequantize of both operands.

That is why stock MixedPrecision Linear (Comfy ops.py) can look "NVFP4 loaded"
(uint8 packed weights in state_dict) while peak VRAM exceeds FP16: packed
storage stays resident AND dequant materializes FP16 weights every forward.

Runtime-only registration — does not edit ComfyUI-master or site-packages files.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
_REGISTERED = False

# Runtime counters: which GEMM the addmm handler actually used.
# scaled_mm (TC hardware) fires only for QT input x QT weight; a float
# input (the parity / stock forward path) always takes the dequant branch.
_ADDMM_SCALED_MM_HITS = 0
_ADDMM_DEQUANT_HITS = 0


def nvfp4_addmm_stats() -> dict:
    return {"addmm_scaled_mm": _ADDMM_SCALED_MM_HITS, "addmm_dequant": _ADDMM_DEQUANT_HITS}


def reset_nvfp4_addmm_stats() -> None:
    global _ADDMM_SCALED_MM_HITS, _ADDMM_DEQUANT_HITS
    _ADDMM_SCALED_MM_HITS = 0
    _ADDMM_DEQUANT_HITS = 0


def register_nvfp4_addmm_handler() -> bool:
    """Register aten.addmm.default → scaled_mm_nvfp4 (same contract as MXFP8 addmm)."""
    global _REGISTERED
    if _REGISTERED:
        return True

    try:
        import torch
        import comfy_kitchen as ck
        from comfy_kitchen.tensor.base import (
            QuantizedTensor,
            dequantize_args,
            register_layout_op,
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
    except Exception as e:
        logger.warning("[HSWQ NVFP4] addmm register skipped (import): %s", e)
        return False

    announce_tc_status_at_register()

    # Already present in a newer kitchen — do not double-register.
    op = torch.ops.aten.addmm.default
    table = _LAYOUT_DISPATCH_TABLE.get(op, {})
    if TensorCoreNVFP4Layout in table:
        _REGISTERED = True
        logger.info("[HSWQ NVFP4] aten.addmm already registered for NVFP4")
        return True

    @register_layout_op(op, TensorCoreNVFP4Layout)
    def _handle_nvfp4_addmm(qt, args, kwargs):
        """NVFP4 addmm: bias + input @ weight.T (F.linear with bias decomposition)."""
        global _ADDMM_SCALED_MM_HITS, _ADDMM_DEQUANT_HITS
        bias, mat1, mat2 = args[0], args[1], args[2]

        if not (isinstance(mat1, QuantizedTensor) and isinstance(mat2, QuantizedTensor)):
            _ADDMM_DEQUANT_HITS += 1
            return torch.addmm(*dequantize_args((bias, mat1, mat2)))
        if mat1._qdata.dim() != 2:
            _ADDMM_DEQUANT_HITS += 1
            return torch.addmm(*dequantize_args((bias, mat1, mat2)))

        input_transposed = getattr(mat1._params, "transposed", False)
        weight_transposed = getattr(mat2._params, "transposed", False)
        # F.linear → addmm(bias, x, w.t()): weight must be logically transposed.
        if input_transposed or not weight_transposed:
            logger.debug(
                "NVFP4 addmm: unsupported transpose configuration, falling back to dequantize"
            )
            _ADDMM_DEQUANT_HITS += 1
            return torch.addmm(*dequantize_args((bias, mat1, mat2)))

        # Cloud Ada/Hopper etc.: skip scaled_mm after first CUBLAS NOT_SUPPORTED
        # (otherwise WARNING floods every Linear every step).
        if not nvfp4_tc_enabled():
            _ADDMM_DEQUANT_HITS += 1
            return torch.addmm(*dequantize_args((bias, mat1, mat2)))

        input_qdata, scale_a, block_scale_a = TensorCoreNVFP4Layout.get_plain_tensors(mat1)
        weight_qdata, scale_b, block_scale_b = TensorCoreNVFP4Layout.get_plain_tensors(mat2)
        out_dtype = kwargs.get("out_dtype", mat1._params.orig_dtype)

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
            _ADDMM_SCALED_MM_HITS += 1
            return _slice_to_original_shape(result, orig_m, orig_n)
        except (RuntimeError, TypeError) as e:
            note_scaled_mm_failure(e)
            _ADDMM_DEQUANT_HITS += 1
            return torch.addmm(*dequantize_args((bias, mat1, mat2)))

    _REGISTERED = True
    print(
        "[HSWQ NVFP4] registered aten.addmm.default for TensorCoreNVFP4Layout "
        "(residual QT input x QT weight -> scaled_mm_nvfp4 TC; "
        "float input -> dequant GEMM [parity / stock path])",
        flush=True,
    )
    return True

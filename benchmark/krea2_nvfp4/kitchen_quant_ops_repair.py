"""Repair ``comfy.quant_ops`` **only** when kitchen bulk-import left stubs.

Branch contract (do not break plain NVFP4):

  A) Stock healthy
     ``_CK_AVAILABLE`` and ``get_layout_class("TensorCoreNVFP4Layout").Params``
     work → **no rebind**. Plain / ConvRot NVFP4 keep ComfyUI-master path.

  B) Stubbed (Vast-style)
     ``from comfy_kitchen.tensor import (..., AsymW4A8Int8Layout, ...)`` failed
     → ``get_layout_class`` always ``None`` → load dies on ``.Params``.
     Submodules (``.base`` / ``.nvfp4`` / ``.int8``) still import → rebind those
     onto ``comfy.quant_ops`` / ``comfy.ops`` only.

Never call rebind when A is true. Never edit ComfyUI-master.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)
_REPAIR_DONE = False
_LAST_STATUS: dict[str, Any] = {"repaired": False, "reason": "not_run"}


def _layout_ok(layout_cls: Any) -> bool:
    return layout_cls is not None and hasattr(layout_cls, "Params")


def kitchen_quant_ops_healthy() -> bool:
    """Branch A gate: stock Comfy quant_ops usable for NVFP4 (+ INT8 if registered)."""
    try:
        import comfy.quant_ops as qo

        if not getattr(qo, "_CK_AVAILABLE", False):
            return False
        # Plain NVFP4 requires this name. INT8 protect also needs INT8 layout.
        nv = qo.get_layout_class("TensorCoreNVFP4Layout")
        if not _layout_ok(nv):
            return False
        # If INT8 algo is in QUANT_ALGOS, its layout must resolve too.
        if "int8_tensorwise" in getattr(qo, "QUANT_ALGOS", {}):
            i8 = qo.get_layout_class("TensorWiseINT8Layout")
            if not _layout_ok(i8):
                return False
        return True
    except Exception:
        return False


def repair_comfy_quant_ops_from_kitchen_submodules(force: bool = False) -> dict[str, Any]:
    """Branch B only: soft-import kitchen submodules and rebind stubs.

    If Branch A (healthy), returns immediately with ``reason=already_ok`` and
    **does not** touch ``comfy.quant_ops`` / ``comfy.ops``.
    """
    global _REPAIR_DONE, _LAST_STATUS

    # ----- Branch A: plain / stock NVFP4 — leave untouched -----
    if not force and kitchen_quant_ops_healthy():
        _REPAIR_DONE = True
        _LAST_STATUS = {
            "repaired": False,
            "reason": "already_ok",
            "branch": "A_stock_healthy",
        }
        return _LAST_STATUS

    if _REPAIR_DONE and not force:
        _LAST_STATUS = {
            "repaired": False,
            "reason": "already_repaired",
            "branch": "B_stub_rebind",
            "prior": dict(_LAST_STATUS),
        }
        return _LAST_STATUS

    # ----- Branch B: stubs — rebind from kitchen submodules -----
    try:
        import comfy_kitchen as ck
        from comfy_kitchen.tensor.base import (
            QuantizedLayout,
            QuantizedTensor,
            get_layout_class,
            register_layout_class,
            register_layout_op,
        )
        from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout as KitchenNVFP4
        from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout as KitchenINT8
    except Exception as e:
        _LAST_STATUS = {
            "repaired": False,
            "reason": "kitchen_submodule_import_failed",
            "branch": "B_stub_rebind",
            "error": f"{type(e).__name__}: {e}",
        }
        logger.error("[HSWQ kitchen repair] submodule import failed: %s", e)
        print(
            f"  [BENCH] kitchen quant_ops repair FAILED (import): {_LAST_STATUS['error']}",
            flush=True,
        )
        return _LAST_STATUS

    import importlib

    optional: dict[str, Any] = {}
    for key, mod_path, attr in (
        ("fp8", "comfy_kitchen.tensor.fp8", "TensorCoreFP8Layout"),
        ("mxfp8", "comfy_kitchen.tensor.mxfp8", "TensorCoreMXFP8Layout"),
        ("convrot_w4a4", "comfy_kitchen.tensor.convrot_w4a4", "TensorCoreConvRotW4A4Layout"),
        ("asym_w4a8", "comfy_kitchen.tensor.w4a8_int8", "AsymW4A8Int8Layout"),
    ):
        try:
            mod = importlib.import_module(mod_path)
            optional[key] = getattr(mod, attr)
        except Exception as e:
            optional[key] = None
            logger.info("[HSWQ kitchen repair] optional %s skipped: %s", key, e)

    import comfy.ops as ops
    import comfy.quant_ops as qo

    qo.ck = ck
    qo._CK_AVAILABLE = True
    qo.QuantizedTensor = QuantizedTensor
    qo.QuantizedLayout = QuantizedLayout
    qo.register_layout_class = register_layout_class
    qo.get_layout_class = get_layout_class
    qo.register_layout_op = register_layout_op

    qo._CKNvfp4Layout = KitchenNVFP4
    qo._CKTensorWiseINT8Layout = KitchenINT8
    qo.TensorWiseINT8Layout = KitchenINT8

    # Same shape as ComfyUI-master comfy/quant_ops.TensorCoreNVFP4Layout
    # (kitchen Params + Comfy quantize helper). Used only on Branch B.
    class TensorCoreNVFP4Layout(KitchenNVFP4):
        @classmethod
        def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):
            import torch
            import comfy.float

            if tensor.dim() != 2:
                raise ValueError(f"NVFP4 requires 2D tensor, got {tensor.dim()}D")

            orig_dtype = tensor.dtype
            orig_shape = tuple(tensor.shape)

            if scale is None or (isinstance(scale, str) and scale == "recalculate"):
                scale = torch.amax(tensor.abs()) / (
                    ck.float_utils.F8_E4M3_MAX * ck.float_utils.F4_E2M1_MAX
                )

            if not isinstance(scale, torch.Tensor):
                scale = torch.tensor(scale)
            scale = scale.to(device=tensor.device, dtype=torch.float32)

            padded_shape = cls.get_padded_shape(orig_shape)
            needs_padding = padded_shape != orig_shape

            if stochastic_rounding > 0:
                qdata, block_scale = comfy.float.stochastic_round_quantize_nvfp4_by_block(
                    tensor, scale, pad_16x=needs_padding, seed=stochastic_rounding
                )
            else:
                qdata, block_scale = ck.quantize_nvfp4(
                    tensor, scale, pad_16x=needs_padding
                )

            params = cls.Params(
                scale=scale,
                orig_dtype=orig_dtype,
                orig_shape=orig_shape,
                block_scale=block_scale,
            )
            return qdata, params

    qo.TensorCoreNVFP4Layout = TensorCoreNVFP4Layout

    # Optional layouts: kitchen class as-is (Params only). Do not invent FP8
    # quantize bodies — that would diverge from stock Comfy when kitchen works.
    if optional["fp8"] is not None:
        qo._CKFp8Layout = optional["fp8"]
        qo.TensorCoreFP8Layout = optional["fp8"]
        qo.TensorCoreFP8E4M3Layout = optional["fp8"]
    if optional["mxfp8"] is not None:
        qo._CKMxfp8Layout = optional["mxfp8"]
        qo._CK_MXFP8_AVAILABLE = True
        qo.TensorCoreMXFP8Layout = optional["mxfp8"]
    if optional["convrot_w4a4"] is not None:
        qo._CKTensorCoreConvRotW4A4Layout = optional["convrot_w4a4"]
        qo.TensorCoreConvRotW4A4Layout = optional["convrot_w4a4"]
    if optional["asym_w4a8"] is not None:
        qo._CKAsymW4A8Int8Layout = optional["asym_w4a8"]
        qo.AsymW4A8Int8Layout = optional["asym_w4a8"]

    register_layout_class("TensorCoreNVFP4Layout", TensorCoreNVFP4Layout)
    register_layout_class("TensorWiseINT8Layout", KitchenINT8)
    if optional["fp8"] is not None:
        register_layout_class("TensorCoreFP8Layout", optional["fp8"])
        register_layout_class("TensorCoreFP8E4M3Layout", optional["fp8"])
    if optional["mxfp8"] is not None:
        register_layout_class("TensorCoreMXFP8Layout", optional["mxfp8"])
    if optional["convrot_w4a4"] is not None:
        register_layout_class("TensorCoreConvRotW4A4Layout", optional["convrot_w4a4"])
    if optional["asym_w4a8"] is not None:
        register_layout_class("AsymW4A8Int8Layout", optional["asym_w4a8"])

    ops.QuantizedTensor = QuantizedTensor
    ops.get_layout_class = get_layout_class
    ops.TensorWiseINT8Layout = KitchenINT8
    if optional["fp8"] is not None:
        ops.TensorCoreFP8Layout = optional["fp8"]

    nv = get_layout_class("TensorCoreNVFP4Layout")
    i8 = get_layout_class("TensorWiseINT8Layout")
    ok = _layout_ok(nv) and _layout_ok(i8)
    _REPAIR_DONE = True
    _LAST_STATUS = {
        "repaired": True,
        "reason": "submodule_rebind",
        "branch": "B_stub_rebind",
        "healthy": ok,
        "nvfp4_params": _layout_ok(nv),
        "int8_params": _layout_ok(i8),
        "optional": {k: (v is not None) for k, v in optional.items()},
        "kitchen_version": getattr(ck, "__version__", "?"),
    }
    print(
        "  [BENCH] kitchen quant_ops repair branch=B (stub rebind): "
        f"nvfp4_Params={_LAST_STATUS['nvfp4_params']} "
        f"int8_Params={_LAST_STATUS['int8_params']} "
        f"kitchen={_LAST_STATUS['kitchen_version']} "
        f"optional={_LAST_STATUS['optional']}",
        flush=True,
    )
    if not ok:
        logger.error("[HSWQ kitchen repair] Branch B finished but layouts still unhealthy")
    return _LAST_STATUS


def ensure_kitchen_quant_ops() -> bool:
    """Branch A → True no-op. Branch B → repair; True if healthy after."""
    if kitchen_quant_ops_healthy():
        if _LAST_STATUS.get("reason") != "already_ok":
            print(
                "  [BENCH] kitchen quant_ops branch=A (stock healthy; "
                "plain NVFP4 path untouched)",
                flush=True,
            )
        repair_comfy_quant_ops_from_kitchen_submodules()  # records already_ok
        return True

    status = repair_comfy_quant_ops_from_kitchen_submodules()
    return bool(status.get("healthy") or kitchen_quant_ops_healthy())

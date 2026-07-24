"""
SDXL NVFP4 bench — force ComfyUI MixedPrecision path (ops.py) verbatim.

Lives OUTSIDE benchmark/nvfp4/ (owner: that package is for SDXL; do not edit it).

After apply_comfy_quant_nvfp4_patches():
  1) NVFP4 Linear load → Comfy ops._load_quantized_module (no HSWQ arm / ones(1))
  2) Linear.forward → unwrap to stock Comfy ops.py MixedPrecision Linear.forward

No invented amax / freeze / TC / ensure_act_scale. Inference + load = ComfyUI only.
"""
from __future__ import annotations

_APPLIED = False


def _closure_named(fn, name: str):
    if fn is None or fn.__closure__ is None:
        return None
    for n, cell in zip(fn.__code__.co_freevars, fn.__closure__):
        if n == name:
            return cell.cell_contents
    return None


def _unwrap_stock_forward(fwd):
    """Extract closed-over stock_forward from make_nvfp4_linear_forward wrap."""
    if not getattr(fwd, "_hswq_nvfp4_full_forward", False):
        return None
    return _closure_named(fwd, "stock_forward")


def apply_nvfp4_comfy_parity() -> bool:
    """Runtime only. Never writes under benchmark/nvfp4/."""
    global _APPLIED
    if _APPLIED:
        return True

    import comfy.ops as ops

    # --- Load: use Comfy ops._load_quantized_module for nvfp4 (no ones(1) arm) ---
    patched_load = ops._load_quantized_module
    orig_load = _closure_named(patched_load, "_orig_load")
    if orig_load is None:
        raise RuntimeError(
            "[BENCH] nvfp4 Comfy parity: could not recover Comfy _orig_load from patch closure"
        )

    def _load_quantized_module_comfy_only(
        module,
        super_load,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
        load_extra_params=False,
    ):
        # Always Comfy stock load — including nvfp4 (input_scale only if in ckpt)
        return orig_load(
            module,
            super_load,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
            load_extra_params=load_extra_params,
        )

    _load_quantized_module_comfy_only._hswq_nvfp4_full_load = True  # type: ignore[attr-defined]
    ops._load_quantized_module = _load_quantized_module_comfy_only

    # --- Forward: unwrap HSWQ TC wrap → Comfy MixedPrecision Linear.forward ---
    _cur_mp = ops.mixed_precision_ops

    def mixed_precision_ops_comfy_only(*args, **kwargs):
        mp = _cur_mp(*args, **kwargs)
        Lin = mp.Linear
        stock = _unwrap_stock_forward(Lin.forward)
        if stock is None and getattr(Lin.forward, "_hswq_nvfp4_full_forward", False):
            raise RuntimeError(
                "[BENCH] nvfp4 Comfy parity: HSWQ TC wrap still on Linear.forward; "
                "refusing to leave non-Comfy forward (SSIM target ≥0.9)"
            )
        if stock is not None:
            Lin.forward = stock
        return mp

    ops.mixed_precision_ops = mixed_precision_ops_comfy_only

    # Prove unwrap works once at install time (no silent leave-TC-on)
    mp0 = _cur_mp()
    stock0 = _unwrap_stock_forward(mp0.Linear.forward)
    if stock0 is None and getattr(mp0.Linear.forward, "_hswq_nvfp4_full_forward", False):
        raise RuntimeError(
            "[BENCH] nvfp4 Comfy parity: failed to unwrap Linear.forward to Comfy stock"
        )
    if stock0 is not None:
        mp0.Linear.forward = stock0

    _APPLIED = True
    print(
        "[BENCH] nvfp4 ComfyUI-only: load=_load_quantized_module; "
        "Linear.forward=ops.py stock (nvfp4/ untouched); SSIM target ≥0.9"
    )
    return True

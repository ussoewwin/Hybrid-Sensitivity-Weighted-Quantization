"""Eager ``rms_rope`` / ``rms_rope1`` for comfy_kitchen builds that lack the fused ops.

ComfyUI Lumina JointAttention (qk_norm) calls ``comfy.quant_ops.ck.rms_rope`` when
not training. Kitchen main exports these; older installed wheels (module
``__version__`` 0.1.0) do not. This module installs eager equivalents on
``comfy.quant_ops.ck`` without touching ComfyUI-master or reinstalling kitchen.

Semantics match Lumina's Python fallback:
  RMSNorm(scale) on last dim, then ``apply_rope`` / ``apply_rope1``.
Uses ``comfy.ldm.flux.math._apply_rope`` / ``_apply_rope1`` to avoid recursion
through ``ck.apply_rope``.
"""

from __future__ import annotations

import torch


def _rms_norm_last(x: torch.Tensor, scale: torch.Tensor | None, eps: float) -> torch.Tensor:
    """RMSNorm over the last dimension; optional affine ``scale`` (weight)."""
    orig_dtype = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    x_f = x_f * torch.rsqrt(var + float(eps))
    if scale is not None:
        x_f = x_f * scale.reshape(*([1] * (x_f.ndim - 1)), -1).float()
    return x_f.to(orig_dtype)


def ensure_kitchen_rms_rope() -> bool:
    """Patch ``ck.rms_rope`` / ``ck.rms_rope1`` if missing. Returns True if installed."""
    import comfy.quant_ops as quant_ops
    from comfy.ldm.flux.math import _apply_rope, _apply_rope1

    ck = quant_ops.ck
    has_rms = hasattr(ck, "rms_rope") and callable(getattr(ck, "rms_rope", None))
    has_rms1 = hasattr(ck, "rms_rope1") and callable(getattr(ck, "rms_rope1", None))
    if has_rms and has_rms1:
        return False

    def rms_rope(q, k, freqs_cis, q_scale, k_scale=None, epsilon=1e-6):
        if k_scale is None:
            k_scale = q_scale
        q = _rms_norm_last(q, q_scale, epsilon)
        k = _rms_norm_last(k, k_scale, epsilon)
        return _apply_rope(q, k, freqs_cis)

    def rms_rope1(x, freqs_cis, scale, epsilon=1e-6):
        x = _rms_norm_last(x, scale, epsilon)
        return _apply_rope1(x, freqs_cis)

    if not has_rms:
        ck.rms_rope = rms_rope
    if not has_rms1:
        ck.rms_rope1 = rms_rope1
    print(
        "  [BENCH] kitchen rms_rope fallback installed "
        f"(had_rms_rope={has_rms}, had_rms_rope1={has_rms1}; "
        f"ck.__version__={getattr(ck, '__version__', '?')})"
    )
    return True

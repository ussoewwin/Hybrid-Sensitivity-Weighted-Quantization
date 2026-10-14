"""SDXL NVFP4 route: TC-vs-parity decision and fully separated arming branches.

SDXL-owned (benchmark/nvfp4/). Self-contained: this module imports nothing from
any Z Image / hswq_stack code path -- SDXL logic ends here.

Explicit discrimination + separated branches (owner rule, 2026-08-30):
  nvfp4_use_tensorcore()  -- the ONLY place the route is decided
  arm_nvfp4_tensorcore()  -- TC branch: the HSWQ full forward (pooled
                             quantize_nvfp4 + scaled_mm_nvfp4 + cudagraph +
                             in-forward ConvRot act) installed by
                             apply_comfy_quant_nvfp4_patches() stays in place.
  arm_nvfp4_parity()      -- parity branch: unwrap the TC wrap to stock Comfy
                             MixedPrecision Linear.forward + online ConvRot
                             act rotation.

A run arms EXACTLY ONE of the two branches; neither calls the other, and no
shared/mixed arming path exists.
"""
from __future__ import annotations


def checkpoint_has_input_scale(unet_path: object) -> bool:
    """True if an SDXL checkpoint carries at least one calibrated input_scale key.

    Reads the safetensors header only (no tensor data). On any error (missing /
    unreadable path) returns False, so an uncalibrated checkpoint never takes
    the TC branch.
    """
    if not unet_path or not isinstance(unet_path, str):
        return False
    try:
        from safetensors import safe_open

        with safe_open(unet_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.endswith(".input_scale"):
                    return True
    except Exception:
        return False
    return False


def nvfp4_use_tensorcore(
    unet_path: object,
    *,
    force_tc: bool = False,
    force_parity: bool = False,
) -> bool:
    """Decide the SDXL NVFP4 forward route: TC (W4A4 scaled_mm) or parity.

    Discrimination, in priority order (first match wins):
      1. force_parity=True  -> parity (escape hatch, bench A/B).
      2. force_tc=True      -> TC (bench A/B; assumes calibrated input_scale).
      3. otherwise          -> TC if the checkpoint carries calibrated
         input_scale keys, parity if not. W4A4 without a calibrated activation
         scale quantizes against placeholder scales and destroys quality, so
         an uncalibrated checkpoint stays on parity.
    """
    if force_parity:
        return False
    if force_tc:
        return True
    return checkpoint_has_input_scale(unet_path)


def arm_nvfp4_tensorcore() -> None:
    """TC branch (dedicated path).

    apply_comfy_quant_nvfp4_patches() already installed the HSWQ NVFP4 full
    forward (detect packed K + nvfp4_load + scaled_mm_nvfp4 + ConvRot act).
    This branch arms nothing further and must NOT unwrap to stock forward.
    """
    return None


def arm_nvfp4_parity() -> None:
    """Parity branch (dedicated path): stock Comfy Linear.forward + online act
    rotation.

    The parity hook itself lives OUTSIDE this package
    (benchmark/nvfp4_comfy_parity.py), so the TC path and this package are
    untouched by it. Importing it here is the only coupling, one-directional.
    """
    from nvfp4_comfy_parity import apply_nvfp4_comfy_parity

    apply_nvfp4_comfy_parity()

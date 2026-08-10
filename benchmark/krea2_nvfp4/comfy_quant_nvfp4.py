"""
ComfyUI runtime monkey-patches for HSWQ comfy_quant NVFP4 (FULL ConvRot).

Runtime only — never permanently edit ComfyUI-master.

Owns (via sibling modules under benchmark/krea2_nvfp4/):
  - packed-K UNet detection (logical in_features)
  - full NVFP4 Linear load (scales, QT, ConvRot flags, storage validation)
  - full Tensor Core forward (act ConvRot → NVFP4 quant → scaled_mm_nvfp4)

This is not an INT8/FP8 “small tweak”: load + forward are HSWQ-owned stacks.
"""
from __future__ import annotations

import logging

from .nvfp4_conf import (
    checkpoint_looks_like_comfy_quant_nvfp4,
    decode_comfy_quant_conf,
    fix_unet_config_packed_dims,
    is_nvfp4_conf,
    logical_linear_in_features,
)
from .nvfp4_forward import (
    make_nvfp4_linear_forward,
    nvfp4_forward_stats,
    reset_nvfp4_forward_stats,
)
from .nvfp4_load import load_nvfp4_linear_module, peek_nvfp4_conf

logger = logging.getLogger(__name__)
_PATCHES_APPLIED = False

# Re-export for benches / callers
__all__ = [
    "apply_comfy_quant_nvfp4_patches",
    "checkpoint_looks_like_comfy_quant_nvfp4",
    "decode_comfy_quant_conf",
    "is_nvfp4_conf",
    "logical_linear_in_features",
    "nvfp4_forward_stats",
    "reset_nvfp4_forward_stats",
]


def _console(msg: str) -> None:
    print(msg, flush=True)
    logger.info(msg)


def apply_comfy_quant_nvfp4_patches() -> bool:
    """Install NVFP4 detection + full load + full TC Linear forward once."""
    global _PATCHES_APPLIED
    # Kitchen gap fill: always (re)ensure addmm → scaled_mm_nvfp4 (idempotent).
    from .nvfp4_addmm_patch import register_nvfp4_addmm_handler

    register_nvfp4_addmm_handler()

    if _PATCHES_APPLIED:
        return True

    try:
        import comfy.model_detection as model_detection
        import comfy.ops as ops
        import comfy.utils as comfy_utils
    except Exception as e:
        logger.warning("[HSWQ NVFP4] comfy import failed: %s", e)
        return False

    if getattr(model_detection.detect_unet_config, "_hswq_nvfp4_packed_dims", False):
        _PATCHES_APPLIED = True
        return True

    _orig_detect = model_detection.detect_unet_config
    _orig_calc = model_detection.calculate_transformer_depth
    _orig_load = ops._load_quantized_module
    _orig_mp = ops.mixed_precision_ops
    _orig_convert_old_quants = comfy_utils.convert_old_quants

    def convert_old_quants_patched(state_dict, model_prefix="", metadata={}):
        state_dict, metadata = _orig_convert_old_quants(
            state_dict, model_prefix, metadata=metadata
        )
        # Kitchen "plain NVFP4" (native_convert_nvfp4.py) stores _quantization_metadata
        # layer keys WITHOUT the diffusion-model prefix, so stock convert_old_quants
        # injects `.comfy_quant` markers at bare keys (e.g. "input_blocks.4.1.proj_in").
        # HSWQ detection/load resolve them at the full module prefix
        # ("model.diffusion_model.input_blocks.4.1.proj_in.comfy_quant"). Move each
        # nvfp4 marker to the full-prefix key so packed-K expansion + load succeed.
        # HSWQ ConvRot files already carry full-prefix markers → skipped (untouched).
        if model_prefix:
            for k in list(state_dict.keys()):
                if not k.endswith(".comfy_quant") or k.startswith(model_prefix):
                    continue
                try:
                    conf = decode_comfy_quant_conf(state_dict[k])
                except Exception:
                    continue
                if not is_nvfp4_conf(conf):
                    continue
                layer = k[: -len(".comfy_quant")]
                if f"{model_prefix}{layer}.weight" not in state_dict:
                    continue
                state_dict[f"{model_prefix}{k}"] = state_dict.pop(k)
        return state_dict, metadata

    def calculate_transformer_depth_patched(prefix, state_dict_keys, state_dict):
        out = _orig_calc(prefix, state_dict_keys, state_dict)
        if out is None:
            return None
        depth, context_dim, use_linear, time_stack, time_stack_cross = out
        k = f"{prefix}1.transformer_blocks.0.attn2.to_k.weight"
        if k in state_dict:
            try:
                context_dim = logical_linear_in_features(state_dict, k)
            except Exception as e:
                logger.warning("[HSWQ NVFP4] transformer context_dim fix skipped: %s", e)
        return depth, context_dim, use_linear, time_stack, time_stack_cross

    def detect_unet_config_patched(state_dict, key_prefix, metadata=None):
        unet_config = _orig_detect(state_dict, key_prefix, metadata=metadata)
        if unet_config is None:
            return None
        return fix_unet_config_packed_dims(unet_config, state_dict, key_prefix)

    def model_config_from_unet_patched(
        state_dict, unet_key_prefix, use_base_if_no_match=False, metadata=None
    ):
        import comfy.supported_models_base
        import comfy.utils

        unet_config = model_detection.detect_unet_config(
            state_dict, unet_key_prefix, metadata=metadata
        )
        if unet_config is None:
            return None
        model_config = model_detection.model_config_from_unet_config(
            unet_config, state_dict, unet_key_prefix
        )
        if model_config is None and use_base_if_no_match:
            model_config = comfy.supported_models_base.BASE(unet_config)

        quant_config = comfy.utils.detect_layer_quantization(
            state_dict, unet_key_prefix
        )
        if quant_config:
            if model_config is None:
                logging.error(
                    "[HSWQ NVFP4] model_config is None with quant_config present "
                    "(packed NVFP4 dims still unmatched?). prefix=%r config=%s",
                    unet_key_prefix,
                    unet_config,
                )
                return None
            model_config.quant_config = quant_config
            logging.info("Detected mixed precision quantization")
        return model_config

    def _load_quantized_module_patched(
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
        conf = peek_nvfp4_conf(state_dict, prefix)
        if is_nvfp4_conf(conf):
            load_nvfp4_linear_module(
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
            return
        _orig_load(
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
        # Non-nvfp4 path: leave stock. (INT8 ConvRot etc. stay on stock/int8 patches.)

    def mixed_precision_ops_patched(*args, **kwargs):
        mp = _orig_mp(*args, **kwargs)
        Lin = mp.Linear
        if getattr(Lin.forward, "_hswq_nvfp4_full_forward", False):
            return mp
        Lin.forward = make_nvfp4_linear_forward(Lin.forward)
        return mp

    model_detection.calculate_transformer_depth = calculate_transformer_depth_patched
    model_detection.detect_unet_config = detect_unet_config_patched
    model_detection.model_config_from_unet = model_config_from_unet_patched
    ops._load_quantized_module = _load_quantized_module_patched
    ops.mixed_precision_ops = mixed_precision_ops_patched
    comfy_utils.convert_old_quants = convert_old_quants_patched

    detect_unet_config_patched._hswq_nvfp4_packed_dims = True  # type: ignore[attr-defined]
    calculate_transformer_depth_patched._hswq_nvfp4_packed_dims = True  # type: ignore[attr-defined]
    model_config_from_unet_patched._hswq_nvfp4_packed_dims = True  # type: ignore[attr-defined]
    _load_quantized_module_patched._hswq_nvfp4_full_load = True  # type: ignore[attr-defined]
    mixed_precision_ops_patched._hswq_nvfp4_full_forward = True  # type: ignore[attr-defined]

    _PATCHES_APPLIED = True
    _console(
        "[HSWQ NVFP4] full stack applied "
        "(detect packed K + nvfp4_load + TC forward scaled_mm_nvfp4 + ConvRot act; "
        "ComfyUI-master untouched)"
    )
    return True

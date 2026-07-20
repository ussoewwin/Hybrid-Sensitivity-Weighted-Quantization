"""
ComfyUI core-safe patches for native comfy_quant INT4 (convrot_w4a4).

ConvRot W4A4 packs Linear weights along in_features (K) as int8 [N, K//2].
Upstream model_detection reads weight.shape[1] as context_dim / adm_in_channels,
so SDXL (2048 / 2816) is mis-detected as half (1024 / 1408) → no match →
AttributeError on model_config.quant_config when model_config is None.

This module restores logical in_features for detection only (runtime load still
uses packed storage + QuantizedTensor via MixedPrecisionOps).

Vendored under hswq/benchmark/int4 for SDXL INT4 fidelity benches.
Runtime monkey-patch only — do not permanently edit ComfyUI-master.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)
_PATCHES_APPLIED = False

# Packed nibble pairs along K: storage_K = logical_K // 2
_W4A4_PACK_FACTOR = 2


def _console(msg: str) -> None:
    print(msg, flush=True)
    logger.info(msg)


def decode_comfy_quant_conf(raw: Any) -> Optional[dict]:
    """Decode a comfy_quant marker into a dict layer config."""
    import torch

    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if torch.is_tensor(raw):
        conf = json.loads(raw.detach().cpu().numpy().tobytes())
    elif isinstance(raw, (bytes, bytearray, memoryview)):
        conf = json.loads(bytes(raw))
    elif isinstance(raw, str):
        conf = raw
    else:
        conf = raw

    while isinstance(conf, str):
        try:
            parsed = json.loads(conf)
        except (TypeError, json.JSONDecodeError):
            return {"format": conf}
        if parsed is conf:
            return {"format": conf}
        conf = parsed

    if isinstance(conf, dict):
        return conf
    raise TypeError(
        f"comfy_quant config must be a dict or format string, got {type(conf).__name__}"
    )


def comfy_quant_key_for_weight(weight_key: str) -> str:
    if weight_key.endswith(".weight"):
        return weight_key[: -len("weight")] + "comfy_quant"
    if weight_key.endswith("weight"):
        return weight_key[: -len("weight")] + "comfy_quant"
    return weight_key + ".comfy_quant"


def is_convrot_w4a4_conf(conf: Optional[dict]) -> bool:
    return isinstance(conf, dict) and conf.get("format") == "convrot_w4a4"


def logical_linear_in_features(state_dict: dict, weight_key: str) -> int:
    """Return logical in_features for a Linear weight (expand packed W4A4 K)."""
    import torch

    weight = state_dict[weight_key]
    if not torch.is_tensor(weight) or weight.ndim < 2:
        raise ValueError(f"{weight_key}: expected 2D+ tensor, got {type(weight)} ndim={getattr(weight, 'ndim', None)}")

    packed_in = int(weight.shape[1])
    cq_key = comfy_quant_key_for_weight(weight_key)
    conf = decode_comfy_quant_conf(state_dict.get(cq_key))
    if is_convrot_w4a4_conf(conf) and weight.ndim == 2:
        return packed_in * _W4A4_PACK_FACTOR
    return packed_in


def checkpoint_looks_like_comfy_quant_int4(state_dict_or_path) -> bool:
    """True if checkpoint has at least one convrot_w4a4 comfy_quant marker."""
    import torch

    if isinstance(state_dict_or_path, (str, os.PathLike)):
        return _probe_path_comfy_quant_int4(str(state_dict_or_path))

    state_dict = state_dict_or_path
    for key, value in state_dict.items():
        if not key.endswith(".comfy_quant"):
            continue
        if not torch.is_tensor(value):
            continue
        conf = decode_comfy_quant_conf(value)
        if is_convrot_w4a4_conf(conf):
            return True
    return False


def _probe_path_comfy_quant_int4(path: str) -> bool:
    try:
        from safetensors import safe_open
    except ImportError:
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            comfy_keys = [k for k in keys if k.endswith(".comfy_quant")]
            for ck in comfy_keys[:64]:
                conf = decode_comfy_quant_conf(f.get_tensor(ck))
                if is_convrot_w4a4_conf(conf):
                    return True
    except Exception as e:
        logger.debug("INT4 probe failed for %s: %s", path, e)
        return False
    return False


def _fix_unet_config_packed_dims(unet_config: dict, state_dict: dict, key_prefix: str) -> dict:
    """Rewrite context_dim / adm_in_channels using logical W4A4 in_features."""
    if not isinstance(unet_config, dict):
        return unet_config

    y_input = f"{key_prefix}label_emb.0.0.weight"
    if y_input in state_dict and unet_config.get("adm_in_channels") is not None:
        try:
            unet_config["adm_in_channels"] = logical_linear_in_features(state_dict, y_input)
        except Exception as e:
            logger.warning("[HSWQ INT4] adm_in_channels fix skipped: %s", e)

    if unet_config.get("context_dim") is not None:
        attn_k = None
        suffix = "attn2.to_k.weight"
        for k in state_dict.keys():
            if k.startswith(key_prefix) and k.endswith(suffix):
                attn_k = k
                break
        if attn_k is not None:
            try:
                unet_config["context_dim"] = logical_linear_in_features(state_dict, attn_k)
            except Exception as e:
                logger.warning("[HSWQ INT4] context_dim fix skipped: %s", e)

    return unet_config


def apply_comfy_quant_int4_patches() -> bool:
    """Install INT4 detection patches once. Returns True if applied (or already applied)."""
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return True

    try:
        import comfy.model_detection as model_detection
    except Exception as e:
        logger.warning("[HSWQ INT4] comfy.model_detection import failed: %s", e)
        return False

    if getattr(model_detection.detect_unet_config, "_hswq_int4_packed_dims", False):
        _PATCHES_APPLIED = True
        return True

    _orig_detect = model_detection.detect_unet_config
    _orig_calc = model_detection.calculate_transformer_depth

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
                logger.warning("[HSWQ INT4] transformer context_dim fix skipped: %s", e)
        return depth, context_dim, use_linear, time_stack, time_stack_cross

    def detect_unet_config_patched(state_dict, key_prefix, metadata=None):
        unet_config = _orig_detect(state_dict, key_prefix, metadata=metadata)
        if unet_config is None:
            return None
        return _fix_unet_config_packed_dims(unet_config, state_dict, key_prefix)

    def model_config_from_unet_patched(
        state_dict, unet_key_prefix, use_base_if_no_match=False, metadata=None
    ):
        # Reimplemented so quant_config is never assigned onto None.
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
                    "[HSWQ INT4] model_config is None with quant_config present "
                    "(packed W4A4 dims still unmatched?). prefix=%r config=%s",
                    unet_key_prefix,
                    unet_config,
                )
                return None
            model_config.quant_config = quant_config
            logging.info("Detected mixed precision quantization")
        return model_config

    model_detection.calculate_transformer_depth = calculate_transformer_depth_patched
    model_detection.detect_unet_config = detect_unet_config_patched
    model_detection.model_config_from_unet = model_config_from_unet_patched
    detect_unet_config_patched._hswq_int4_packed_dims = True  # type: ignore[attr-defined]
    calculate_transformer_depth_patched._hswq_int4_packed_dims = True  # type: ignore[attr-defined]
    model_config_from_unet_patched._hswq_int4_packed_dims = True  # type: ignore[attr-defined]
    _PATCHES_APPLIED = True
    _console(
        "[HSWQ INT4] comfy_quant patches applied "
        "(detect_unet_config logical K for convrot_w4a4 + None-safe quant_config)"
    )
    return True

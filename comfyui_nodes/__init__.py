"""HSWQ quantization nodes for ComfyUI.

Provides native ConvRot INT8 quantization nodes supporting multiple architectures
(Z Image, Qwen Image Edit, etc.).
"""
from .native_convrot_int8_convert import NativeConvRotInt8Quantize, ZImageConvRotInt8Quantize
from .te_controlnet_convrot_int8_convert import (
    TEControlNetConvRotInt8Quantize,
    CLIPConvRotInt8Quantize,
    ControlNetConvRotInt8Quantize,
)

NODE_CLASS_MAPPINGS = {
    "NativeConvRotInt8Quantize": NativeConvRotInt8Quantize,
    "ZImageConvRotInt8Quantize": ZImageConvRotInt8Quantize,
    "TEControlNetConvRotInt8Quantize": TEControlNetConvRotInt8Quantize,
    "CLIPConvRotInt8Quantize": CLIPConvRotInt8Quantize,
    "ControlNetConvRotInt8Quantize": ControlNetConvRotInt8Quantize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NativeConvRotInt8Quantize": "Native ConvRot INT8 Quantize",
    "ZImageConvRotInt8Quantize": "Native ConvRot INT8 Quantize",
    "TEControlNetConvRotInt8Quantize": "TE / ControlNet ConvRot INT8 Quantize",
    "CLIPConvRotInt8Quantize": "CLIP / TE ConvRot INT8 Quantize",
    "ControlNetConvRotInt8Quantize": "ControlNet ConvRot INT8 Quantize",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

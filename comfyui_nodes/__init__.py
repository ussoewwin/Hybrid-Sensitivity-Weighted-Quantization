"""HSWQ quantization nodes for ComfyUI.

Provides native ConvRot INT8 quantization nodes supporting multiple architectures
(Z Image, Qwen Image Edit, etc.).
"""
from .native_convrot_int8_convert import NativeConvRotInt8Quantize, ZImageConvRotInt8Quantize

NODE_CLASS_MAPPINGS = {
    "NativeConvRotInt8Quantize": NativeConvRotInt8Quantize,
    "ZImageConvRotInt8Quantize": ZImageConvRotInt8Quantize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NativeConvRotInt8Quantize": "Native ConvRot INT8 Quantize",
    "ZImageConvRotInt8Quantize": "Native ConvRot INT8 Quantize",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

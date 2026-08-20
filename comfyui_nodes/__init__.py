"""HSWQ quantization nodes for ComfyUI.

Currently ships one node: ZImageConvRotInt8Quantize (native ConvRot INT8,
wrapping the repo script ``native_convert_int8_convrot_zi.py``).
"""
from .nodes import ZImageConvRotInt8Quantize

NODE_CLASS_MAPPINGS = {
    "ZImageConvRotInt8Quantize": ZImageConvRotInt8Quantize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageConvRotInt8Quantize": "Z Image ConvRot INT8 Quantize",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

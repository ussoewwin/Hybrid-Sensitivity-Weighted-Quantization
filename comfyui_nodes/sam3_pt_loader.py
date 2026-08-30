# -*- coding: utf-8 -*-
"""sam3.pt (SAM3 non-multiplex) dedicated loader node.

Loads sam3.pt, applies the shared SAM3 preprocessing from convert_clip_convrot_int8
(text_projection dropped, in_proj split, tracker remap) and builds a standard
ComfyUI MODEL, ready for the TE/ControlNet ConvRot INT8 quantize node.
"""
from __future__ import annotations

import os
import sys

import folder_paths
import comfy.utils
import comfy.sd


def _get_sam3_filenames():
    files = set()
    for folder in ("unet", "diffusion_models", "checkpoints", "sams"):
        try:
            files.update(folder_paths.get_filename_list(folder))
        except Exception:
            pass
    return sorted(files)


class HSWQSAM3Loader:
    """Load sam3.pt (SAM3, non-multiplex) with shared SAM3 preprocessing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_name": (_get_sam3_filenames(),),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "loaders"
    TITLE = "HSWQ SAM3 Loader (sam3.pt)"

    def load(self, sam3_name: str):
        path = None
        for folder in ("unet", "diffusion_models", "checkpoints", "sams"):
            p = folder_paths.get_full_path(folder, sam3_name)
            if p and os.path.exists(p):
                path = p
                break
        if path is None:
            raise ValueError(f"sam3.pt not found: {sam3_name}")

        sd = comfy.utils.load_torch_file(path)

        # Shared SAM3 preprocessing from the convert script
        # (SAM3 branch: text_projection dropped, in_proj split, tracker remap).
        try:
            _conv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clip_convert")
            if _conv_dir not in sys.path:
                sys.path.insert(0, _conv_dir)
            from convert_clip_convrot_int8 import _detect_sam_version as _ver
            from convert_clip_convrot_int8 import _preprocess_sam_and_fused_keys as _pp
            print(f"[HSWQ SAM3 Loader] auto-detected version: {_ver(sd)}")
            sd = _pp(sd)
        except Exception as e:
            print(f"[HSWQ SAM3 Loader] shared preprocess unavailable: {e}")

        model = comfy.sd.load_diffusion_model_state_dict(sd, model_options={})
        if model is None:
            raise RuntimeError(f"Failed to build SAM3 model from {sam3_name}")
        return (model,)


NODE_CLASS_MAPPINGS = {
    "HSWQSAM3Loader": HSWQSAM3Loader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HSWQSAM3Loader": "HSWQ SAM3 Loader (sam3.pt)",
}

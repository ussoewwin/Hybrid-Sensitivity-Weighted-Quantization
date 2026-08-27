"""ComfyUI node: simple direct CLIP / Text Encoder loader without architecture type selection.

Loads any safetensors / checkpoint file from standard text_encoders / clip folders
directly as a state_dict wrapper compatible with quantization and save nodes.
"""
from __future__ import annotations

import os
import torch


class SimpleCLIPWrapper:
    """Lightweight wrapper around loaded state_dict for downstream consumption."""

    def __init__(self, state_dict: dict[str, torch.Tensor], clip_path: str):
        self.sd = state_dict
        self.clip_path = clip_path
        self.cached_patcher_init = (None, (clip_path,))

    def state_dict_for_saving(self) -> dict[str, torch.Tensor]:
        return self.sd

    def get_sd(self) -> dict[str, torch.Tensor]:
        return self.sd

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.sd

    def load_model(self, *args, **kwargs):
        pass

    def get_key_patches(self) -> dict:
        return {}


class HSWQSimpleCLIPLoader:
    """Load any CLIP / Text Encoder file directly without specifying type/architecture."""

    @classmethod
    def INPUT_TYPES(s):
        try:
            import folder_paths

            files = folder_paths.get_filename_list("text_encoders")
        except Exception:
            files = []
        return {
            "required": {
                "clip_name": (files,),
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_clip"
    CATEGORY = "HSWQ/Loaders"

    def load_clip(self, clip_name: str):
        import folder_paths
        import comfy.utils

        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        sd = comfy.utils.load_torch_file(clip_path, safe_load=True)
        return (SimpleCLIPWrapper(sd, clip_path),)


# Compatibility alias
SimpleCLIPLoader = HSWQSimpleCLIPLoader

"""ComfyUI nodes: simple direct CLIP / Text Encoder and ControlNet loaders.

Loads any safetensors / checkpoint file from standard models folders directly as
a state_dict wrapper compatible with quantization and save nodes.
"""
from __future__ import annotations

import os
import torch


class SimpleCLIPWrapper:
    """Lightweight wrapper around loaded CLIP state_dict for downstream consumption."""

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


class SimpleControlNetWrapper:
    """Lightweight wrapper around loaded ControlNet state_dict for downstream consumption."""

    def __init__(self, state_dict: dict[str, torch.Tensor], controlnet_path: str):
        self.sd = state_dict
        self.controlnet_path = controlnet_path
        self.cached_patcher_init = (None, (controlnet_path,))

    def state_dict_for_saving(self) -> dict[str, torch.Tensor]:
        return self.sd

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.sd


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


class HSWQSimpleControlNetLoader:
    """Load any ControlNet file directly without requiring diffusers or complex wrappers."""

    @classmethod
    def INPUT_TYPES(s):
        try:
            import folder_paths

            files = folder_paths.get_filename_list("controlnet")
        except Exception:
            files = []
        return {
            "required": {
                "control_net_name": (files,),
            }
        }

    RETURN_TYPES = ("CONTROL_NET",)
    RETURN_NAMES = ("control_net",)
    FUNCTION = "load_controlnet"
    CATEGORY = "HSWQ/Loaders"

    def load_controlnet(self, control_net_name: str):
        import folder_paths
        import comfy.utils

        controlnet_path = folder_paths.get_full_path_or_raise("controlnet", control_net_name)
        sd = comfy.utils.load_torch_file(controlnet_path, safe_load=True)
        return (SimpleControlNetWrapper(sd, controlnet_path),)


# Compatibility aliases
SimpleCLIPLoader = HSWQSimpleCLIPLoader
SimpleControlNetLoader = HSWQSimpleControlNetLoader

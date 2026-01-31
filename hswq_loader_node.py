import torch
import os
import folder_paths
import safetensors.torch
import comfy.sd
import comfy.utils

class HSWQLoader:
    """
    HSWQ V2 (Scaled FP8) model loader.

    Loads FP8 models with .scale keys, auto de-scales (FP8 -> FP16 restore) and
    passes to ComfyUI. Enables best-quality use of scaled models that break with standard loaders.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_hswq_checkpoint"
    CATEGORY = "HSWQ"

    def load_hswq_checkpoint(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        
        print(f"[HSWQ] Loading Scaled FP8 Model: {ckpt_name}")
        
        # 1. Load model and de-scale
        state_dict = safetensors.torch.load_file(ckpt_path, device="cpu")
        # Find .scale keys and apply de-scale
        scaled_count = 0
        keys_to_remove = []
        all_keys = list(state_dict.keys())
        
        for key in all_keys:
            if key.endswith(".scale"):
                if key.endswith(".weight.scale"):
                    weight_key = key.replace(".weight.scale", ".weight")
                else:
                    weight_key = key.replace(".scale", ".weight")
                
                if weight_key in state_dict:
                    scale_val = state_dict[key]
                    weight_fp8 = state_dict[weight_key]
                    # De-scale: FP8 -> FP16 * scale (cast FP8 to FP16, multiply by inv_scale)
                    weight_fp16 = weight_fp8.to(torch.float16)
                    restored_weight = weight_fp16 * scale_val.to(torch.float16)
                    state_dict[weight_key] = restored_weight
                    
                    keys_to_remove.append(key)
                    scaled_count += 1
        
        for key in keys_to_remove:
            del state_dict[key]
            
        print(f"[HSWQ] Restored {scaled_count} layers from Scaled FP8 to FP16.")
        
        # 2. Inject into ComfyUI load: temporarily hook load_torch_file to return processed state_dict
        original_loader = comfy.utils.load_torch_file

        def patched_loader(path, safe_load=False, device=None):
            if os.path.normpath(path) == os.path.normpath(ckpt_path):
                return state_dict
            return original_loader(path, safe_load, device)
        
        comfy.utils.load_torch_file = patched_loader
        try:
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt_path, 
                output_vae=True, 
                output_clip=True, 
                embedding_directory=folder_paths.get_folder_paths("embeddings")
            )
        finally:
            comfy.utils.load_torch_file = original_loader
            
        return out

# Node registration
NODE_CLASS_MAPPINGS = {
    "HSWQLoader": HSWQLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HSWQLoader": "HSWQ Scaled FP8 Loader"
}

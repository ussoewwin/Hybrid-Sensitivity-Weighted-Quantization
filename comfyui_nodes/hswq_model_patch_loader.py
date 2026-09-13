"""ComfyUI node: HSWQ Model Patch loader with native ConvRot INT8 support.

Stock ``comfy_extras.nodes_model_patch.ModelPatchLoader`` builds every model
patch with ``comfy.ops.manual_cast`` Linears, so a ConvRot INT8 checkpoint
(``<layer>.weight`` int8 + ``weight_scale`` + ``comfy_quant``) would load the
raw INT8 tensor as a plain Parameter and produce garbage / crash at forward.

This node mirrors the stock loader dispatch but selects quant-aware
operations (``mixed_precision_ops`` with the ``int8_tensorwise`` algorithm)
whenever the checkpoint carries ``int8_tensorwise`` ``comfy_quant`` layers.
MixedPrecisionOps.Linear's ``_load_from_state_dict`` consumes the
``comfy_quant`` / ``weight_scale`` stamps and attaches an INT8
``QuantizedTensor`` (TensorWiseINT8Layout), so weights stay INT8 in VRAM and
forward uses the comfy-kitchen ``int8_linear`` kernel with online ConvRot
activation rotation. This is the same native INT8 path ComfyUI uses for
UNet / CLIP / ControlNet.

For non-quantized checkpoints the node delegates to the stock
``ModelPatchLoader``, so behavior is identical for regular model patches.
The returned object is a normal ``MODEL_PATCH`` (CoreModelPatcher); use it
with the stock apply nodes (``ZImageFunControlnet`` etc.).
"""
from __future__ import annotations

import json
import logging

import torch

import comfy.model_management
import comfy.model_patcher
import comfy.utils
from comfy import ops as comfy_ops
from comfy.quant_ops import QUANT_ALGOS

logger = logging.getLogger(__name__)

_MP_DIR = "model_patches"


def _decode_comfy_quant(raw) -> dict:
    try:
        return json.loads(raw.numpy().tobytes())
    except Exception:  # noqa: BLE001
        return {}


def _has_int8_comfy_quant(sd: dict) -> bool:
    """True if the checkpoint carries >=1 int8_tensorwise comfy_quant layer."""
    for key in sd.keys():
        if not key.endswith(".comfy_quant"):
            continue
        conf = _decode_comfy_quant(sd[key])
        if conf.get("format") == "int8_tensorwise":
            return True
    return False


def _int8_mixed_precision_ops():
    """MixedPrecisionOps supporting int8_tensorwise (ConvRot included)."""
    quant_config = {
        "int8_tensorwise": QUANT_ALGOS["int8_tensorwise"],
    }
    return comfy_ops.mixed_precision_ops(
        quant_config,
        torch.bfloat16,
        full_precision_mm=False,
        disabled=[],
    )


def _load_quantized_model_patch(sd, metadata, name):
    """Mirror of the stock ModelPatchLoader dispatch with quant-aware ops.

    The sd passed in still carries the on-disk (pre-``z_image_convert``) keys
    for Z-Image Fun ControlNets; ``z_image_convert`` is applied inside the
    branch exactly like the stock loader. ConvRot INT8 checkpoints are saved
    in the post-conversion layout (``attention.qkv`` already merged), so
    ``z_image_convert`` is a no-op for them and the ``comfy_quant`` stamps
    survive untouched.
    """
    dtype = torch.bfloat16
    operations = _int8_mixed_precision_ops()
    logger.info(
        "[HSWQ ModelPatch] INT8 ComfyQuant detected: loading %s with "
        "MixedPrecisionOps (weights stay INT8 in VRAM)",
        name,
    )

    import comfy.ldm.anima.lllite
    import comfy.ldm.lightricks.duration_head
    import comfy.ldm.lumina.controlnet
    import comfy.ldm.supir.supir_modules
    import comfy.ldm.wan.model_multitalk
    import comfy.ldm.wan.uni3c
    from comfy_extras.nodes_model_patch import (
        QwenImageBlockWiseControlNet,
        SigLIPMultiFeatProjModel,
        MultiTalkModelPatch,
        z_image_convert,
    )

    if "lllite_conditioning1.conv1.weight" in sd:
        model = comfy.ldm.anima.lllite.AnimaLLLite(
            sd, metadata,
            device=comfy.model_management.unet_offload_device(),
            dtype=dtype, operations=operations,
        )
    elif "controlnet_blocks.0.y_rms.weight" in sd:
        additional_in_dim = sd["img_in.weight"].shape[1] - 64
        model = QwenImageBlockWiseControlNet(
            additional_in_dim=additional_in_dim,
            device=comfy.model_management.unet_offload_device(),
            dtype=dtype, operations=operations,
        )
    elif "feature_embedder.mid_layer_norm.bias" in sd:
        sd = comfy.utils.state_dict_prefix_replace(
            sd, {"feature_embedder.": ""}, filter_keys=True
        )
        model = SigLIPMultiFeatProjModel(
            device=comfy.model_management.unet_offload_device(),
            dtype=dtype, operations=operations,
        )
    elif "control_all_x_embedder.2-1.weight" in sd:  # alipai z image fun controlnet
        sd = z_image_convert(sd)
        config = {}
        if "control_layers.4.adaLN_modulation.0.weight" not in sd:
            config["n_control_layers"] = 3
            config["additional_in_dim"] = 17
            config["refiner_control"] = True
        if "control_layers.14.adaLN_modulation.0.weight" in sd:
            config["n_control_layers"] = 15
            config["additional_in_dim"] = 17
            config["refiner_control"] = True
            ref_weight = sd.get("control_noise_refiner.0.after_proj.weight", None)
            if ref_weight is not None:
                if torch.count_nonzero(ref_weight) == 0:
                    config["broken"] = True
        model = comfy.ldm.lumina.controlnet.ZImage_Control(
            device=comfy.model_management.unet_offload_device(),
            dtype=dtype, operations=operations, **config,
        )
    elif "controlnet_patch_embedding.weight" in sd:  # Uni3C controlnet for Wan
        attn_key_replace = {
            ".self_attn.to_q.": ".self_attn.q.",
            ".self_attn.to_k.": ".self_attn.k.",
            ".self_attn.to_v.": ".self_attn.v.",
            ".self_attn.to_out.0.": ".self_attn.o.",
        }
        converted_sd = {}
        for k, w in sd.items():
            for r, rr in attn_key_replace.items():
                k = k.replace(r, rr)
            converted_sd[k] = w
        sd = converted_sd

        num_layers = sum(1 for k in sd if k.startswith("proj_out.") and k.endswith(".weight"))
        conv_out_dim = sd["controlnet_patch_embedding.weight"].shape[0]
        if "proj_in.weight" in sd:
            dim = sd["proj_in.weight"].shape[0]
        else:
            dim = conv_out_dim
        model = comfy.ldm.wan.uni3c.WanUni3CControlnet(
            in_channels=sd["controlnet_patch_embedding.weight"].shape[1],
            conv_out_dim=conv_out_dim,
            dim=dim,
            ffn_dim=sd["controlnet_blocks.0.ffn.0.bias"].shape[0],
            num_layers=num_layers,
            time_embed_dim=sd["controlnet_blocks.0.norm1.linear.weight"].shape[1],
            out_proj_dim=sd["proj_out.0.weight"].shape[0],
            add_channels=sd["controlnet_mask_embedding.mask_proj.0.weight"].shape[1],
            mid_channels=sd["controlnet_mask_embedding.mask_proj.0.weight"].shape[0],
            device=comfy.model_management.unet_offload_device(),
            dtype=dtype,
            operations=operations,
        )
    elif any(k.endswith("duration_head.attention_pooler.query_tokens") for k in sd) or "attention_pooler.query_tokens" in sd:
        sd = comfy.ldm.lightricks.duration_head.normalize_state_dict(sd)
        sd = {k: v.float() for k, v in sd.items()}  # tiny head, keep fp32
        model = comfy.ldm.lightricks.duration_head.DurationHead()
    elif "audio_proj.proj1.weight" in sd:
        model = MultiTalkModelPatch(
            audio_window=5, context_tokens=32, vae_scale=4,
            in_dim=sd["blocks.0.audio_cross_attn.proj.weight"].shape[0],
            intermediate_dim=sd["audio_proj.proj1.weight"].shape[0],
            out_dim=sd["audio_proj.norm.weight"].shape[0],
            device=comfy.model_management.unet_offload_device(),
            operations=operations,
        )
    elif "model.control_model.input_hint_block.0.weight" in sd or "control_model.input_hint_block.0.weight" in sd:
        prefix_replace = {}
        if "model.control_model.input_hint_block.0.weight" in sd:
            prefix_replace["model.control_model."] = "control_model."
            prefix_replace["model.diffusion_model.project_modules."] = "project_modules."
        else:
            prefix_replace["control_model."] = "control_model."
            prefix_replace["project_modules."] = "project_modules."

        # Extract denoise_encoder weights before filter_keys discards them
        de_prefix = "first_stage_model.denoise_encoder."
        denoise_encoder_sd = {}
        for k in list(sd.keys()):
            if k.startswith(de_prefix):
                denoise_encoder_sd[k[len(de_prefix):]] = sd.pop(k)

        sd = comfy.utils.state_dict_prefix_replace(sd, prefix_replace, filter_keys=True)
        sd.pop("control_model.mask_LQ", None)
        model = comfy.ldm.supir.supir_modules.SUPIR(
            device=comfy.model_management.unet_offload_device(),
            dtype=dtype, operations=operations,
        )
        if denoise_encoder_sd:
            model.denoise_encoder_sd = denoise_encoder_sd
    else:
        raise ValueError(
            f"[HSWQ ModelPatch] {name}: could not detect a known model patch architecture "
            "(no supported discriminator key found)."
        )

    model_patcher = comfy.model_patcher.CoreModelPatcher(
        model,
        load_device=comfy.model_management.get_torch_device(),
        offload_device=comfy.model_management.unet_offload_device(),
    )
    model.load_state_dict(sd, assign=model_patcher.is_dynamic())
    return model_patcher


class HSWQModelPatchLoader:
    """Load a Model Patch, keeping ConvRot INT8 layers INT8 in VRAM."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        return {
            "required": {
                "name": (
                    folder_paths.get_filename_list(_MP_DIR),
                    {"tooltip": "ConvRot INT8 / standard Model Patch (.safetensors)"},
                ),
            },
        }

    RETURN_TYPES = ("MODEL_PATCH",)
    RETURN_NAMES = ("model_patch",)
    FUNCTION = "load_model_patch"
    CATEGORY = "HSWQ/Loaders"

    def load_model_patch(self, name):
        import folder_paths
        from comfy_extras.nodes_model_patch import ModelPatchLoader as StockModelPatchLoader

        model_patch_path = folder_paths.get_full_path_or_raise(_MP_DIR, name)
        sd, metadata = comfy.utils.load_torch_file(
            model_patch_path, safe_load=True, return_metadata=True
        )

        if not _has_int8_comfy_quant(sd):
            logger.info(
                "[HSWQ ModelPatch] No INT8 ComfyQuant layers in %s, delegating to stock ModelPatchLoader",
                name,
            )
            return StockModelPatchLoader().load_model_patch(name)

        return (_load_quantized_model_patch(sd, metadata, name),)


NODE_CLASS_MAPPINGS = {
    "HSWQModelPatchLoader": HSWQModelPatchLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HSWQModelPatchLoader": "HSWQ Load Model Patch (ConvRot INT8)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

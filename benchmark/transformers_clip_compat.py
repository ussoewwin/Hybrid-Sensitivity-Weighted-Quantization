"""
Transformers 5.6+ compatibility for diffusers SDXL from_single_file.

In transformers >= 5.6, CLIPTextModel no longer nests weights under ``text_model.*``
(the submodule was flattened). diffusers' single-file loader still expects
``model.text_model.embeddings`` and checkpoint keys prefixed with ``text_model.``.

Forge-Nunchaku strips the ``text_model.`` prefix when loading (see backend/loader.py).
This module applies the same idea for diffusers-based benchmarks.
"""

from __future__ import annotations

_APPLIED = False


def _is_flat_clip_text_model(model) -> bool:
    from transformers import CLIPTextModel

    return isinstance(model, CLIPTextModel)


def _remap_flat_clip_state_dict(state_dict: dict) -> dict:
    """Map LDM/diffusers keys ``text_model.X`` -> ``X`` for flat CLIPTextModel."""
    prefix = "text_model."
    remapped: dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            remapped[key[len(prefix) :]] = value
        elif key in ("logit_scale",):
            continue
        else:
            remapped[key] = value
    return remapped


def apply() -> None:
    """Idempotent: patch CLIP loading once per process."""
    global _APPLIED
    if _APPLIED:
        return

    from transformers import CLIPTextModel
    from diffusers.models import model_loading_utils as mlu

    if not getattr(CLIPTextModel, "_hswq_text_model_prop", False):

        def _text_model_self(self):
            return self

        CLIPTextModel.text_model = property(_text_model_self)
        CLIPTextModel._hswq_text_model_prop = True

    if not getattr(CLIPTextModel, "_hswq_load_state_dict_patched", False):
        _orig_load_state_dict = CLIPTextModel.load_state_dict

        def _patched_load_state_dict(self, state_dict, *args, **kwargs):
            if _is_flat_clip_text_model(self):
                state_dict = _remap_flat_clip_state_dict(state_dict)
            return _orig_load_state_dict(self, state_dict, *args, **kwargs)

        CLIPTextModel.load_state_dict = _patched_load_state_dict
        CLIPTextModel._hswq_load_state_dict_patched = True

    if not getattr(mlu, "_hswq_load_meta_patched", False):
        _orig_load_meta = mlu.load_model_dict_into_meta

        def _patched_load_meta(model, state_dict, *args, **kwargs):
            if _is_flat_clip_text_model(model):
                state_dict = _remap_flat_clip_state_dict(state_dict)
            return _orig_load_meta(model, state_dict, *args, **kwargs)

        mlu.load_model_dict_into_meta = _patched_load_meta
        mlu._hswq_load_meta_patched = True

    _APPLIED = True

"""ComfyUI node: quantize loaded Text Encoder (CLIP/TE), ControlNet and Model Patch models to native ConvRot INT8.

Connects to standard CLIP, ControlNet and/or Model Patch (MODEL_PATCH) loader outputs.
Extracts model weights in-memory, quantizes 2D float Linear weights using Hadamard rotation +
row-wise INT8 quantization with comfy_quant stamps, and saves checkpoints compatible with
native ComfyUI loaders. Non-2D weights (embeddings, norms, biases) are preserved as-is.

Output layout:
    <layer>.weight           int8
    <layer>.weight_scale     float32  [out, 1]
    <layer>.comfy_quant      uint8 JSON  {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":N}
    _quantization_metadata   {"format_version":"1.0","layers":{...}}
"""
from __future__ import annotations

import json
import math
import os
import time
import torch
from safetensors.torch import save_file


def _is_power_of_4(n: int) -> bool:
    return n >= 4 and (n & (n - 1)) == 0 and (n.bit_length() - 1) % 2 == 0


def _output_dir() -> str:
    try:
        import folder_paths

        return folder_paths.get_output_directory()
    except Exception:
        return os.getcwd()


def build_hadamard(size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Normalized regular Hadamard matrix (power of 4)."""
    if not _is_power_of_4(size):
        raise ValueError(f"Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor(
        [
            [1, 1, 1, -1],
            [1, 1, -1, 1],
            [1, -1, 1, 1],
            [-1, 1, 1, 1],
        ],
        dtype=dtype,
    )
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size**0.5)


def convrot_group_size(n: int, preferred: int = 256) -> int | None:
    """Largest power-of-4 group size <= preferred that divides n."""
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and _is_power_of_4(gs):
            return gs
        gs //= 4
    return None


def rotate_weight(weight: torch.Tensor, h: torch.Tensor, gs: int) -> torch.Tensor:
    """W_rot = W @ H^T (group-wise along in_features)."""
    out_f, in_f = weight.shape
    if in_f % gs != 0:
        raise ValueError(f"in_features {in_f} not divisible by group size {gs}")
    g = in_f // gs
    return torch.matmul(weight.view(out_f, g, gs), h.T.to(weight.dtype)).reshape(weight.shape)


def quantize_int8_rowwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-out-channel INT8 with scale [out, 1]."""
    amax = w.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
    scale = amax / 127.0
    q = (w / scale.to(w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def _encode_meta(config: dict) -> torch.Tensor:
    return torch.tensor(
        list(json.dumps(config, separators=(",", ":")).encode("utf-8")),
        dtype=torch.uint8,
    )


def _quantize_state_dict(
    sd: dict[str, torch.Tensor],
    enable_convrot: bool = True,
    group_size: int = 256,
) -> tuple[dict[str, torch.Tensor], dict[str, dict], dict[str, int]]:
    new_sd = {}
    meta_layers = {}
    convrot_count = 0
    plain_count = 0
    skip_count = 0
    fallback_list: list[str] = []

    h_cache: dict[int, torch.Tensor] = {}

    for key, tensor in sorted(sd.items()):
        is_2d_weight = (
            key.endswith(".weight")
            and tensor.ndim == 2
            and tensor.dtype in (torch.float16, torch.float32, torch.bfloat16)
        )

        if not is_2d_weight:
            new_sd[key] = tensor
            skip_count += 1
            continue

        w = tensor.float()
        out_f, in_f = w.shape
        module_key = key[: -len(".weight")]

        if enable_convrot:
            gs = convrot_group_size(in_f, group_size)
            if gs is not None:
                if gs not in h_cache:
                    h_cache[gs] = build_hadamard(gs)
                w_rot = rotate_weight(w, h_cache[gs], gs)
                q, scale = quantize_int8_rowwise(w_rot)
                config = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": gs}
                new_sd[key] = q
                new_sd[f"{module_key}.weight_scale"] = scale
                new_sd[f"{module_key}.comfy_quant"] = _encode_meta(config)
                meta_layers[module_key] = config
                convrot_count += 1
                continue
            else:
                fallback_list.append(key)

        # Plain INT8 (no rotation)
        q, scale = quantize_int8_rowwise(w)
        config = {"format": "int8_tensorwise"}
        new_sd[key] = q
        new_sd[f"{module_key}.weight_scale"] = scale
        new_sd[f"{module_key}.comfy_quant"] = _encode_meta(config)
        meta_layers[module_key] = config
        plain_count += 1

    return new_sd, meta_layers, {
        "convrot": convrot_count,
        "plain": plain_count,
        "kept": skip_count,
        "fallback": len(fallback_list),
    }


def _extract_state_dict(obj) -> dict[str, torch.Tensor]:
    """Unified state dict extractor for CLIP (Text Encoder) and ControlNet."""
    if obj is None:
        return {}
    sd = None
    if hasattr(obj, "sd") and isinstance(obj.sd, dict):
        sd = obj.sd
    if sd is None and hasattr(obj, "load_model"):
        try:
            obj.load_model()
        except Exception:
            pass
    if sd is None and hasattr(obj, "state_dict_for_saving"):
        try:
            sd = obj.state_dict_for_saving()
        except Exception:
            pass
    if sd is None and hasattr(obj, "get_sd"):
        try:
            sd = obj.get_sd()
        except Exception:
            pass
    if sd is None and hasattr(obj, "control_model") and obj.control_model is not None:
        if hasattr(obj.control_model, "state_dict"):
            sd = obj.control_model.state_dict()
    if sd is None and hasattr(obj, "control_model_wrapped") and obj.control_model_wrapped is not None:
        if hasattr(obj.control_model_wrapped, "model_state_dict_for_saving"):
            try:
                sd = obj.control_model_wrapped.model_state_dict_for_saving()
            except Exception:
                pass
        if sd is None and hasattr(obj.control_model_wrapped, "model") and hasattr(obj.control_model_wrapped.model, "state_dict"):
            sd = obj.control_model_wrapped.model.state_dict()
    if sd is None and hasattr(obj, "patcher"):
        if hasattr(obj.patcher, "model_state_dict_for_saving"):
            try:
                sd = obj.patcher.model_state_dict_for_saving()
            except Exception:
                pass
        if sd is None and hasattr(obj.patcher, "model") and hasattr(obj.patcher.model, "state_dict"):
            sd = obj.patcher.model.state_dict()
    if sd is None and hasattr(obj, "control_weights") and obj.control_weights is not None:
        sd = obj.control_weights
    if sd is None and hasattr(obj, "t2i_model") and obj.t2i_model is not None:
        if hasattr(obj.t2i_model, "state_dict"):
            sd = obj.t2i_model.state_dict()
    if sd is None and hasattr(obj, "cond_stage_model") and hasattr(obj.cond_stage_model, "state_dict"):
        try:
            sd = obj.cond_stage_model.state_dict()
        except Exception:
            pass
    if sd is None and hasattr(obj, "model_state_dict_for_saving"):
        try:
            sd = obj.model_state_dict_for_saving()
        except Exception:
            pass
    # Model Patch loaders (comfy.model_patcher.ModelPatcher / CoreModelPatcher)
    # expose the wrapped nn.Module via ``.model``.
    if sd is None and hasattr(obj, "model"):
        _m = getattr(obj, "model", None)
        if _m is not None and hasattr(_m, "state_dict"):
            try:
                sd = _m.state_dict()
            except Exception:
                pass
    if sd is None and isinstance(obj, dict):
        sd = obj
    if sd is None and hasattr(obj, "state_dict"):
        try:
            sd = obj.state_dict()
        except Exception:
            pass

    if sd is None:
        raise ValueError(f"Could not extract state_dict from input {type(obj)}.")

    out = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu()
    return out


def _find_upstream_filename(prompt: dict | None, unique_id: str | None, input_slot: str) -> str | None:
    """Trace upstream in ComfyUI prompt graph to extract the original model filename from the loader node."""
    if not prompt or not unique_id or str(unique_id) not in prompt:
        return None
    curr_node = prompt.get(str(unique_id), {})
    inputs = curr_node.get("inputs", {})
    slot_val = inputs.get(input_slot)

    visited = set()
    while isinstance(slot_val, list) and len(slot_val) >= 2:
        upstream_id = str(slot_val[0])
        if upstream_id in visited or upstream_id not in prompt:
            break
        visited.add(upstream_id)
        upstream_node = prompt[upstream_id]
        up_inputs = upstream_node.get("inputs", {})

        for key in (
            "control_net_name",
            "controlnet_name",
            "clip_name",
            "clip_name1",
            "clip_name2",
            "clip_name3",
            "unet_name",
            "model_name",
            "ckpt_name",
            "filename",
            "file_name",
            "model_patch_name",
            "name",
        ):
            if key in up_inputs and isinstance(up_inputs[key], str) and up_inputs[key].strip():
                val = up_inputs[key].strip()
                return os.path.splitext(os.path.basename(val))[0]

        next_slot = None
        for candidate_slot in (input_slot, "clip", "control_net", "model", "model_patch"):
            if candidate_slot in up_inputs and isinstance(up_inputs[candidate_slot], list):
                next_slot = up_inputs[candidate_slot]
                break
        if next_slot is not None:
            slot_val = next_slot
        else:
            break

    return None


def _get_original_name(obj, default: str = "model") -> str:
    for attr in ("clip_path", "controlnet_path", "model_path", "ckpt_path", "model_patch_path"):
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if val and isinstance(val, str):
                return os.path.splitext(os.path.basename(val))[0]

    for target in (obj, getattr(obj, "patcher", None), getattr(obj, "control_model_wrapped", None)):
        if target is not None and hasattr(target, "cached_patcher_init") and target.cached_patcher_init:
            func, args = target.cached_patcher_init[:2]
            if args and isinstance(args, tuple) and len(args) > 0:
                p = args[0]
                if isinstance(p, list) and len(p) > 0 and isinstance(p[0], str):
                    return os.path.splitext(os.path.basename(p[0]))[0]
                elif isinstance(p, str):
                    return os.path.splitext(os.path.basename(p))[0]
    return default


def _load_full_sam_checkpoint(prompt, unique_id, input_slot):
    """Load the upstream checkpoint file and return a fully-processed state dict.

    Resolves the original ckpt path from the loader node and applies the SAM3 / SAM3.1
    preprocessing from convert_clip_convrot_int8 (in_proj split + text_projection shape
    branching), so the saved file keeps MODEL + CLIP and works with
    CheckpointLoaderSimple (CLIP is not None). Returns None when the upstream path
    cannot be resolved.
    """
    name = _find_upstream_filename(prompt, unique_id, input_slot)
    if not name:
        return None
    p = None
    try:
        import folder_paths
        for folder in ("checkpoints", "unet", "diffusion_models"):
            cand = folder_paths.get_full_path(folder, name + ".safetensors")
            if cand and os.path.exists(cand):
                p = cand
                break
    except Exception:
        return None
    if p is None:
        return None
    try:
        import comfy.utils as _cu
        sd = _cu.load_torch_file(p, safe_load=True)
    except Exception:
        return None
    # Auto-detect SAM3 vs SAM3.1 and branch explicitly (never mix the two).
    try:
        _conv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clip_convert")
        if _conv_dir not in sys.path:
            sys.path.insert(0, _conv_dir)
        from convert_clip_convrot_int8 import _preprocess_sam_and_fused_keys as _pp
        from convert_clip_convrot_int8 import _detect_sam_version as _ver
        print(f"[ConvRot SAM] auto-detected version: {_ver(sd)}")
        sd = _pp(sd)
    except Exception as e:
        print(f"[ConvRot SAM] auto-detection/preprocess unavailable: {e}")
    return sd


def _summarize(output_path: str) -> str:
    """Read the written checkpoint's metadata only (no tensor load)."""
    try:
        from safetensors import safe_open

        with safe_open(output_path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
        qm = json.loads(meta.get("_quantization_metadata", "{}"))
        layers = qm.get("layers", {}) if isinstance(qm, dict) else {}
        convrot = sum(
            1 for c in layers.values() if isinstance(c, dict) and c.get("convrot")
        )
        plain = len(layers) - convrot
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        return f"layers={len(layers)} (convrot={convrot}, plain_int8={plain}) | file_size={size_mb:.2f} MB"
    except Exception:
        return "(summary unavailable)"


class TEControlNetConvRotInt8Quantize:
    """Quantize loaded Text Encoder (CLIP/TE), ControlNet or Model Patch models to native ConvRot INT8."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "group_size": ("INT", {"default": 256, "min": 4, "max": 1024, "step": 4}),
                "convrot": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "clip": ("CLIP",),
                "control_net": ("CONTROL_NET",),
                "model_patch": ("MODEL_PATCH",),
                "model": ("MODEL",),
                "output_path": ("STRING", {"default": "", "multiline": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float(time.time())

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "report")
    FUNCTION = "quantize"
    CATEGORY = "HSWQ/Quantize"
    OUTPUT_NODE = True

    def quantize(
        self,
        group_size: int,
        convrot: bool,
        clip=None,
        control_net=None,
        model_patch=None,
        model=None,
        output_path: str = "",
        prompt=None,
        unique_id=None,
        extra_pnginfo=None,
    ):
        group_size = int(group_size)
        if not _is_power_of_4(group_size):
            raise ValueError(f"group_size must be a power of 4 (>=4), got {group_size}")

        if clip is None and control_net is None and model_patch is None and model is None:
            raise ValueError(
                "Please connect at least one input: clip (Text Encoder), control_net, model_patch, or model."
            )

        tasks: list[tuple[str, dict[str, torch.Tensor], str]] = []
        user_output_path = (output_path or "").strip().strip('"').strip("'")

        # CLIP is emitted as its own file only when no MODEL is connected; when a
        # MODEL (SAM3 / SAM3.1) is present the CLIP weights are merged into the model
        # file so the saved checkpoint keeps a valid text encoder.
        if clip is not None and model is None:
            clip_name = _find_upstream_filename(prompt, unique_id, "clip") or _get_original_name(clip, "clip")
            tasks.append(("CLIP", _extract_state_dict(clip), clip_name))

        if control_net is not None:
            cn_name = _find_upstream_filename(prompt, unique_id, "control_net") or _get_original_name(control_net, "controlnet")
            tasks.append(("ControlNet", _extract_state_dict(control_net), cn_name))

        if model_patch is not None:
            mp_name = _find_upstream_filename(prompt, unique_id, "model_patch") or _get_original_name(model_patch, "model_patch")
            tasks.append(("ModelPatch", _extract_state_dict(model_patch), mp_name))

        if model is not None:
            model_name = _find_upstream_filename(prompt, unique_id, "model") or _get_original_name(model, "model")
            # SAM3 / SAM3.1: prefer converting the original checkpoint file so the
            # saved file includes MODEL + CLIP and CheckpointLoaderSimple returns a
            # valid CLIP (otherwise "clip input is invalid: None"). Falls back to the
            # in-memory ModelPatcher weights (detector + tracker only) when the
            # upstream path cannot be resolved.
            sd_full = _load_full_sam_checkpoint(prompt, unique_id, "model")
            if sd_full is not None:
                tasks.append(("Model", sd_full, model_name))
                if clip is not None:
                    clip = None  # CLIP is already included in the model file
            else:
                sd_model = _extract_state_dict(model)
                sd_model = {
                    (k[len("diffusion_model."):] if k.startswith("diffusion_model.") else k): v
                    for k, v in sd_model.items()
                    if k != "model_sampling"
                }
                tasks.append(("Model", sd_model, model_name))

        saved_paths: list[str] = []
        report_lines: list[str] = []
        ts = int(time.time())

        for idx, (mtype, sd, orig_name) in enumerate(tasks):
            default_name = f"{orig_name}_convrot_int8.safetensors"
            if user_output_path:
                target_path = os.path.abspath(user_output_path)
                if os.path.isdir(target_path):
                    target_path = os.path.join(target_path, default_name)
                elif len(tasks) > 1 and idx > 0:
                    base, ext = os.path.splitext(target_path)
                    target_path = f"{base}_{mtype.lower()}{ext}"
            else:
                target_path = os.path.join(_output_dir(), default_name)

            out_dir = os.path.dirname(target_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            new_sd, meta_layers, stats = _quantize_state_dict(
                sd,
                enable_convrot=bool(convrot),
                group_size=group_size,
            )

            metadata = {
                "_quantization_metadata": json.dumps(
                    {"format_version": "1.0", "layers": meta_layers},
                    separators=(",", ":"),
                )
            }
            save_file(new_sd, target_path, metadata=metadata)
            saved_paths.append(target_path)

            report_lines.append(f"[{mtype} Quantization]")
            report_lines.append(f"  Saved: {target_path}")
            report_lines.append(f"  Summary: {_summarize(target_path)}")
            report_lines.append(
                f"  Stats: convrot={bool(convrot)} group_size={group_size} "
                f"convrot_linear={stats['convrot']} plain_int8={stats['plain']} "
                f"kept_as_is={stats['kept']} fallback={stats['fallback']}"
            )
            report_lines.append("")

        final_output_path = saved_paths[0] if len(saved_paths) == 1 else ";".join(saved_paths)
        return (final_output_path, "\n".join(report_lines).strip())

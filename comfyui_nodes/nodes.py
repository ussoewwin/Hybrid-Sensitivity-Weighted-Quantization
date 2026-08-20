"""ComfyUI node: quantize a loaded Z Image (NextDiT) UNet to native ConvRot INT8.

Connects to the standard UNet loader's MODEL output (no file-path input).
Extracts the diffusion-model weights in-memory, quantizes them with the same
algorithm as ``native_convert_int8_convrot_zi.py``, and saves a checkpoint the
standard ComfyUI loader reads back.

Output layout:
    model.diffusion_model.<layer>.weight          int8
    model.diffusion_model.<layer>.weight_scale    float32
    model.diffusion_model.<layer>.comfy_quant     uint8 JSON
    _quantization_metadata                        {"format_version":"1.0","layers":{...}}

Repo layout: this package lives at <repo>/comfyui_nodes/ and reuses the pure
quantization helpers in <repo>/native_convert_int8.py. The repo root is
auto-detected by walking up until that script is found; pass ``repo_root`` to
override.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys

_N8 = None


def _default_repo_root() -> str:
    """Locate the repo root by walking up until native_convert_int8.py is found."""
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(8):
        if os.path.isfile(os.path.join(d, "native_convert_int8.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.abspath(os.path.join(here, os.pardir))


def _load_native_int8(repo_root: str):
    """Import native_convert_int8.py once (no side effects on import)."""
    global _N8
    if _N8 is not None:
        return _N8

    root = (repo_root or "").strip() or _default_repo_root()
    root = os.path.abspath(root)
    path = os.path.join(root, "native_convert_int8.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Quantization helpers not found: {path}. "
            "Set repo_root to the Hybrid-Sensitivity-Weighted-Quantization "
            "clone directory."
        )

    try:
        spec = importlib.util.spec_from_file_location("hswq_native_int8", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["hswq_native_int8"] = mod
        spec.loader.exec_module(mod)
    except ImportError as e:
        raise ImportError(
            f"Failed to import {path}: {e}. "
            "The helpers need torch, safetensors and tqdm in the ComfyUI env."
        )
    _N8 = mod
    return mod


def _is_power_of_4(n: int) -> bool:
    return n >= 4 and (n & (n - 1)) == 0 and math.log(n, 4) % 1 == 0


def _output_dir() -> str:
    try:
        import folder_paths

        return folder_paths.get_output_directory()
    except Exception:
        return os.getcwd()


def _extract_model_state_dict(model):
    """Return the diffusion model weights with 'model.diffusion_model.' prefix."""
    try:
        return model.model_state_dict_for_saving(
            model.model.diffusion_model, "model.diffusion_model."
        )
    except AttributeError as e:
        raise ValueError(
            "model input is not a ComfyUI ModelPatcher; connect a UNet / "
            "checkpoint loader MODEL output. (%s)" % e
        )


def _quantize_state_dict(sd, group_size, enable_convrot, per_channel_int8, n8):
    """In-memory ConvRot INT8 packing (same math as native_convert_int8.py)."""
    new_sd = {}
    meta_layers = {}
    n_linear = n_conv2d = n_plain = n_kept = 0

    for key, tensor in sd.items():
        if not n8._is_float_matmul_weight(key, tensor):
            new_sd[key] = tensor
            n_kept += 1
            continue

        w = tensor.float()
        module_key = key[: -len(".weight")]
        conf = {"format": "int8_tensorwise"}

        if enable_convrot and tensor.ndim == 2:
            gs = n8.convrot_group_size_for_features(w.shape[1], group_size)
            if gs is not None:
                h = n8.build_hadamard(gs, device="cpu")
                w = n8.rotate_weight(w, h, gs)
                q, scale = n8.quantize_int8_rowwise(w)
                conf = {"format": "int8_tensorwise", "convrot": True,
                        "convrot_groupsize": int(gs)}
                n_linear += 1
            else:
                q, scale = _plain_pack(w, per_channel_int8, n8)
                n_plain += 1
        elif enable_convrot and tensor.ndim == 4:
            gs = n8.convrot_group_size_for_features(w.shape[1], group_size)
            if gs is not None:
                h = n8.build_hadamard(gs, device="cpu")
                w = n8.rotate_weight_conv2d(w, h, gs)
                q, scale = n8.quantize_int8_channelwise(w)
                conf = {"format": "int8_tensorwise", "convrot": True,
                        "convrot_groupsize": int(gs)}
                n_conv2d += 1
            else:
                q, scale = _plain_pack(w, per_channel_int8, n8)
                n_plain += 1
        else:
            q, scale = _plain_pack(w, per_channel_int8, n8)
            n_plain += 1

        new_sd[key] = q
        new_sd[f"{module_key}.weight_scale"] = scale
        new_sd[f"{module_key}.comfy_quant"] = n8._encode_comfy_quant(conf)
        meta_layers[module_key] = dict(conf)

    return new_sd, meta_layers, {
        "linear": n_linear,
        "conv2d": n_conv2d,
        "plain": n_plain,
        "kept": n_kept,
    }


def _plain_pack(w, per_channel_int8, n8):
    if per_channel_int8:
        return n8.quantize_int8_channelwise(w)
    return n8.quantize_int8_tensorwise(w)


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
        return f"layers={len(layers)} convrot={convrot} plain_int8={plain}"
    except Exception:
        return "(summary unavailable)"


class ZImageConvRotInt8Quantize:
    """Quantize a loaded Z Image UNet (MODEL input) to native ConvRot INT8."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "output_path": ("STRING", {"default": "", "multiline": False}),
                "group_size": ("INT", {"default": 256, "min": 4, "max": 4096, "step": 4}),
                "convrot": ("BOOLEAN", {"default": True}),
                "per_channel_int8": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "repo_root": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "report")
    FUNCTION = "quantize"
    CATEGORY = "HSWQ/Quantize"

    def quantize(
        self,
        model,
        output_path,
        group_size,
        convrot,
        per_channel_int8,
        repo_root="",
    ):
        n8 = _load_native_int8(repo_root)

        group_size = int(group_size)
        if not _is_power_of_4(group_size):
            raise ValueError(f"group_size must be a power of 4 (>=4), got {group_size}")

        sd = _extract_model_state_dict(model)

        output_path = (output_path or "").strip()
        if not output_path:
            output_path = os.path.join(_output_dir(), "convrot_int8.safetensors")
        output_path = os.path.abspath(output_path)
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        new_sd, meta_layers, stats = _quantize_state_dict(
            sd, group_size, bool(convrot), bool(per_channel_int8), n8
        )

        from safetensors.torch import save_file

        metadata = {
            "_quantization_metadata": json.dumps(
                {"format_version": "1.0", "layers": meta_layers},
                separators=(",", ":"),
            )
        }
        save_file(new_sd, output_path, metadata=metadata)

        report = [
            f"Saved: {output_path}",
            _summarize(output_path),
            f"convrot={bool(convrot)} group_size={group_size} "
            f"linear={stats['linear']} conv2d={stats['conv2d']} "
            f"plain={stats['plain']} kept={stats['kept']}",
        ]
        return (output_path, "\n".join(report))

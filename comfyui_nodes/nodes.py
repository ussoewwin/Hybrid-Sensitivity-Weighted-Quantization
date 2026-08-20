"""ComfyUI node: quantize a Z Image Turbo (ZIT) checkpoint to native ConvRot INT8.

Wraps the existing repo script ``native_convert_int8_convrot_zi.py`` so the exact
"How to quantize Z Image" flow runs as a node:

    convert -> save -> (optional) post-convert bench

The output is loaded by the *standard* ComfyUI loader (no dedicated HSWQ loader
is required), so this node is a file-to-file conversion tool: model path in,
quantized checkpoint path out.

Output layout:
    <layer>.weight          int8
    <layer>.weight_scale    float32  (scalar | [out,1] | [out,1,1,1])
    <layer>.comfy_quant     uint8 JSON  {"format":"int8_tensorwise", "convrot":true, ...}
    _quantization_metadata  {"format_version":"1.0","layers":{...}}

Repo layout: this package lives at
    <repo>/comfyui_nodes/
and the converter lives at <repo>/native_convert_int8_convrot_zi.py.
The repo root is auto-detected by walking up until the converter is found;
pass ``repo_root`` to override.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys

_CONVERTER = None


def _default_repo_root() -> str:
    """Locate the repo root by walking up until the converter script is found.

    Works whether this package lives at <repo>/comfyui_nodes/ or deeper
    (e.g. <repo>/ComfyUI-master/custom_nodes/hswq_quantize/).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(8):
        if os.path.isfile(os.path.join(d, "native_convert_int8_convrot_zi.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.abspath(os.path.join(here, os.pardir))


def _load_converter(repo_root: str):
    """Import native_convert_int8_convrot_zi.py once (no side effects on import)."""
    global _CONVERTER
    if _CONVERTER is not None:
        return _CONVERTER

    root = (repo_root or "").strip() or _default_repo_root()
    root = os.path.abspath(root)
    conv = os.path.join(root, "native_convert_int8_convrot_zi.py")
    if not os.path.isfile(conv):
        raise FileNotFoundError(
            f"Converter not found: {conv}. "
            "Set repo_root to the Hybrid-Sensitivity-Weighted-Quantization "
            "clone directory."
        )

    try:
        spec = importlib.util.spec_from_file_location("zi_convrot_int8_convrot", conv)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["zi_convrot_int8_convrot"] = mod
        spec.loader.exec_module(mod)
    except ImportError as e:
        raise ImportError(
            f"Failed to import {conv}: {e}. "
            "The converter needs torch, safetensors and tqdm in the ComfyUI env."
        )
    _CONVERTER = mod
    return mod


def _is_power_of_4(n: int) -> bool:
    return n >= 4 and (n & (n - 1)) == 0 and math.log(n, 4) % 1 == 0


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
    except Exception as e:  # noqa: BLE001 - summary must never crash the node
        return f"(summary unavailable: {e})"


class ZImageConvRotInt8Quantize:
    """Quantize a Z Image Turbo UNet checkpoint to native ConvRot INT8."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "", "multiline": False}),
                "output_path": ("STRING", {"default": "", "multiline": False}),
                "group_size": ("INT", {"default": 256, "min": 4, "max": 4096, "step": 4}),
                "convrot": ("BOOLEAN", {"default": True}),
                "per_channel_int8": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "repo_root": ("STRING", {"default": "", "multiline": False}),
                "bias_correction": ("BOOLEAN", {"default": False}),
                "calib_file": ("STRING", {"default": "", "multiline": False}),
                "run_benchmark": ("BOOLEAN", {"default": False}),
                "clip_path": ("STRING", {"default": "", "multiline": False}),
                "comfy_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "report")
    FUNCTION = "quantize"
    CATEGORY = "HSWQ/Quantize"

    def quantize(
        self,
        model_path,
        output_path,
        group_size,
        convrot,
        per_channel_int8,
        repo_root="",
        bias_correction=False,
        calib_file="",
        run_benchmark=False,
        clip_path="",
        comfy_path="",
    ):
        model_path = (model_path or "").strip()
        if not model_path:
            raise ValueError("model_path is required")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"model_path not found: {model_path}")

        group_size = int(group_size)
        if not _is_power_of_4(group_size):
            raise ValueError(
                f"group_size must be a power of 4 (>=4), got {group_size}"
            )

        output_path = (output_path or "").strip()
        if not output_path:
            stem = os.path.splitext(os.path.basename(model_path))[0]
            output_path = os.path.join(
                os.path.dirname(os.path.abspath(model_path)),
                f"{stem}_convrot_int8.safetensors",
            )
        output_path = os.path.abspath(output_path)
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        conv = _load_converter(repo_root)
        conv.convert_to_int8(
            model_path,
            output_path,
            per_channel_int8=bool(per_channel_int8),
            bias_correction=bool(bias_correction),
            calib_file=(calib_file or "").strip() or None,
            num_calib_samples=32,
            num_inference_steps=25,
            enable_convrot=bool(convrot),
            group_size=group_size,
        )

        report = [
            f"Saved: {output_path}",
            _summarize(output_path),
            f"convrot={bool(convrot)} group_size={group_size}",
        ]

        if run_benchmark:
            root = (repo_root or "").strip() or _default_repo_root()
            root = os.path.abspath(root)
            clip_path = (clip_path or "").strip()
            comfy_path = (comfy_path or "").strip()
            if not clip_path or not os.path.isfile(clip_path):
                raise FileNotFoundError("run_benchmark requires a valid clip_path")
            if not comfy_path or not os.path.isdir(comfy_path):
                raise FileNotFoundError("run_benchmark requires a valid comfy_path")
            rc = conv.run_post_convert_zi_int8_bench(
                script_dir=root,
                fp16_path=model_path,
                int8_path=output_path,
                clip_path=clip_path,
                comfy_path=comfy_path,
                vae_path=None,
            )
            if rc != 0:
                raise RuntimeError(f"Post-convert bench exited with code {rc}")
            report.append("Bench: PASS (exit 0)")

        return (output_path, "\n".join(report))

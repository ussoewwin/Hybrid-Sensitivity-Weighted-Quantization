"""ComfyUI node: quantize SD1.5 checkpoints with HSWQ V1.0 (FP16 300 MiB protection + ConvRot INT8).

Executes the official sd1.5/quantize_sd1.5_hswq_v1.0.py engine without modifying
the script. Freeing live VRAM models prior to execution ensures clean calibration
and deterministic 25-seed trajectory benchmarking.
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time

import torch

# Locate repository root (two levels up from comfyui_nodes/)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_QUANT_SCRIPT = os.path.join("sd1.5", "quantize_sd1.5_hswq_v1.0.py")
_DEFAULT_CALIB_FILE = os.path.join("sample", "calibration_prompts_512.txt")


def _model_base_path(model) -> str:
    """Checkpoint path of a loaded MODEL, as recorded by the loader."""
    try:
        if getattr(model, "cached_patcher_init", None):
            _func, args = model.cached_patcher_init[:2]
            if args and isinstance(args, tuple) and isinstance(args[0], str):
                return args[0]
    except Exception:
        pass
    return ""


def _output_dir() -> str:
    try:
        import folder_paths

        return folder_paths.get_output_directory()
    except Exception:
        return os.getcwd()


def _is_power_of_4(n: int) -> bool:
    if n < 4 or (n & (n - 1)) != 0:
        return False
    import math
    return math.log(n, 4) % 1 == 0


def _free_vram_before_run():
    """Purge loaded models and release CUDA memory before launching sub-process calibration."""
    try:
        import comfy.model_management as mm

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for fn_name in ("cleanup_models", "cleanup_models_gc", "unload_all_models"):
            fn = getattr(mm, fn_name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
        free_memory = getattr(mm, "free_memory", None)
        if callable(free_memory):
            try:
                free_memory(1e30, mm.get_torch_device())
            except TypeError:
                try:
                    free_memory(1e30)
                except Exception:
                    pass
            except Exception:
                pass
        if callable(getattr(mm, "soft_empty_cache", None)):
            mm.soft_empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            print(
                f"[HSWQ SD1.5] pre-run VRAM: "
                f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated, "
                f"{torch.cuda.memory_reserved() / 2**30:.2f} GiB reserved",
                flush=True,
            )
    except Exception as e:
        print(f"[HSWQ SD1.5] WARN: could not free models before run: {e}", flush=True)


class HSWQQuantize:
    """Quantize a diffusion model using HSWQ V1.0 (FP16 protection + ConvRot INT8)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["SD1.5"], {"default": "SD1.5"}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "output_path": ("STRING", {"default": "", "multiline": False}),
                "calib_file": (
                    ["128", "256", "512"],
                    {
                        "default": "512",
                        "tooltip": "Calibration prompts sheet from sample/ (128, 256, or 512 lines).",
                    },
                ),
                "num_calib_samples": (
                    "INT",
                    {
                        "default": 32,
                        "min": 1,
                        "max": 256,
                        "tooltip": "Mandatory r32 recipe: 32 samples.",
                    },
                ),
                "num_inference_steps": (
                    "INT",
                    {
                        "default": 25,
                        "min": 1,
                        "max": 100,
                        "tooltip": "Denoising steps per calibration sample (default: 25).",
                    },
                ),
                "fp16_budget_mb": (
                    "FLOAT",
                    {
                        "default": 300.0,
                        "min": 0.0,
                        "max": 1000.0,
                        "step": 1.0,
                        "tooltip": "Hard ceiling budget for FP16-kept layers (must be 300.0).",
                    },
                ),
                "group_size": (
                    "INT",
                    {
                        "default": 256,
                        "min": 4,
                        "max": 4096,
                        "tooltip": "ConvRot Hadamard group size (power of 4, default: 256).",
                    },
                ),
                "convrot": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Enables FULL ConvRot Hadamard rotation on UNet and CLIP.",
                    },
                ),
                "bias_correction": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Card 1: cancel systematic INT8 output bias into layer biases.",
                    },
                ),
                "run_benchmark": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Run automatic 25-seed deterministic trajectory benchmark.",
                    },
                ),
            },
            "optional": {
                "vae": ("VAE",),
                "custom_calib_file": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional path to a custom calibration prompts text file (overrides calib_file dropdown).",
                    },
                ),
                "input_checkpoint": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional manual override path to the source checkpoint if MODEL does not contain cached_patcher_init.",
                    },
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float(time.time())

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "report")
    FUNCTION = "quantize"
    CATEGORY = "HSWQ/Quantize"
    TITLE = "HSWQ Quantize ConvRot INT8"
    OUTPUT_NODE = True

    def quantize(
        self,
        model_type,
        model,
        clip,
        output_path,
        calib_file,
        num_calib_samples,
        num_inference_steps,
        fp16_budget_mb,
        group_size,
        convrot,
        bias_correction,
        run_benchmark,
        vae=None,
        custom_calib_file="",
        input_checkpoint="",
    ):
        if model_type != "SD1.5":
            raise NotImplementedError(
                f"model_type {model_type!r} is not yet implemented for HSWQ quantizer."
            )
        base_path = (input_checkpoint or "").strip()
        if not base_path:
            base_path = _model_base_path(model)

        if not base_path or not os.path.isfile(base_path):
            raise ValueError(
                "model: cannot resolve the checkpoint path of the MODEL input. "
                "Ensure the model is loaded from a standard Load Checkpoint node, "
                f"or specify input_checkpoint manually. Got: {base_path!r}"
            )

        script_path = os.path.join(_REPO_ROOT, _QUANT_SCRIPT)
        if not os.path.isfile(script_path):
            raise FileNotFoundError(f"HSWQ SD1.5 script not found: {script_path}")

        # Resolve calib_file from preset dropdown (128, 256, 512) or custom path
        custom_path = (custom_calib_file or "").strip()
        if custom_path:
            calib_path = custom_path if os.path.isabs(custom_path) else os.path.join(_REPO_ROOT, custom_path)
        else:
            calib_str = str(calib_file).strip()
            calib_map = {
                "128": os.path.join(_REPO_ROOT, "sample", "calibration_prompts_128.txt"),
                "256": os.path.join(_REPO_ROOT, "sample", "calibration_prompts_256.txt"),
                "512": os.path.join(_REPO_ROOT, "sample", "calibration_prompts_512.txt"),
            }
            if calib_str in calib_map:
                calib_path = calib_map[calib_str]
            elif not os.path.isabs(calib_str):
                calib_path = os.path.join(_REPO_ROOT, calib_str)
            else:
                calib_path = calib_str

        if not os.path.isfile(calib_path):
            raise FileNotFoundError(f"Calibration prompts file not found: {calib_path}")

        group_size = int(group_size)
        if not _is_power_of_4(group_size):
            raise ValueError(f"group_size must be a power of 4 (>=4), got {group_size}")

        # Resolve output_path
        output_path = (output_path or "").strip()
        original_stem = os.path.splitext(os.path.basename(base_path))[0]
        ts = int(time.time())
        default_name = f"{original_stem}_hswq_int8_{ts}.safetensors"
        if not output_path:
            output_path = os.path.join(_output_dir(), default_name)
        output_path = os.path.abspath(output_path)
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, default_name)
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Free memory in current process
        _free_vram_before_run()

        # Build command for quantize_sd1.5_hswq_v1.0.py
        cmd = [
            sys.executable,
            script_path,
            "--input", base_path,
            "--output", output_path,
            "--calib_file", calib_path,
            "--num_calib_samples", str(int(num_calib_samples)),
            "--num_inference_steps", str(int(num_inference_steps)),
            "--fp16_budget_mb", str(float(fp16_budget_mb)),
            "--groupsize", str(int(group_size)),
        ]
        if convrot:
            cmd.append("--convrot")
        else:
            cmd.append("--no-convrot")

        if bias_correction:
            cmd.append("--bias_correction")
        else:
            cmd.append("--no-bias_correction")

        if run_benchmark:
            cmd.append("--bench")
        else:
            cmd.append("--no-bench")

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        if "HF_HOME" not in env:
            hf_default = r"D:\USERFILES\MyDocuments\HuggingFace"
            if os.path.isdir(hf_default):
                env["HF_HOME"] = hf_default

        print("[HSWQ SD1.5 Quantize] " + " ".join(cmd), flush=True)
        lines = []
        proc = subprocess.Popen(
            cmd,
            cwd=_REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                lines.append(line)
                print("  " + line, flush=True)
            proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

        if proc.returncode != 0:
            tail = "\n".join(lines[-40:])
            raise RuntimeError(
                f"HSWQ SD1.5 quantization failed (exit {proc.returncode}). Output tail:\n{tail}"
            )

        print(f"[HSWQ SD1.5 Quantize] Completed: {output_path}", flush=True)
        report = "\n".join(lines[-60:])
        return (output_path, report)


# Backwards compatibility alias
HSWQSD15Quantize = HSWQQuantize


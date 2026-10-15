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
_DEFAULT_CALIB_FILE = os.path.join("sample", "calibration_prompts_128.txt")


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
    """Quantize a diffusion model using HSWQ V1.0 (FP16 protection + ConvRot INT8).

    VRAM contract: this node takes the source as a *checkpoint name* (not a live
    MODEL/CLIP object). The engine re-loads the checkpoint itself, so accepting a
    live MODEL would make ComfyUI load it to VRAM for nothing and double-load
    against the subprocess - the historical cause of "VRAM explodes + slow" when
    a standalone script is ported into a node. Name-only input keeps the node's
    own VRAM footprint at zero before the subprocess starts.
    """

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        try:
            ckpt_names = folder_paths.get_filename_list("checkpoints")
        except Exception:
            ckpt_names = []
        if not ckpt_names:
            ckpt_names = [""]
        return {
            "required": {
                "model_type": (["SD1.5"], {"default": "SD1.5"}),
                "checkpoint": (
                    ckpt_names,
                    {
                        "default": ckpt_names[0],
                        "tooltip": "SD1.5 checkpoint (models/checkpoints). The engine loads it itself; no live MODEL/CLIP is needed.",
                    },
                ),
                "output_path": ("STRING", {"default": "", "multiline": False}),
                "calib_file": (
                    ["128", "256", "512"],
                    {
                        "default": "128",
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
                "input_checkpoint": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional full path override for the source checkpoint (takes precedence over the checkpoint dropdown).",
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
        checkpoint,
        output_path,
        calib_file,
        num_calib_samples,
        num_inference_steps,
        fp16_budget_mb,
        group_size,
        convrot,
        bias_correction,
        run_benchmark,
        input_checkpoint="",
        **kwargs,
    ):
        if model_type != "SD1.5":
            raise NotImplementedError(
                f"model_type {model_type!r} is not yet implemented for HSWQ quantizer."
            )

        # Resolve the source checkpoint path from the name dropdown (or the
        # explicit path override). No live MODEL/CLIP is taken: the engine
        # reloads the checkpoint itself, so accepting a live MODEL would only
        # make ComfyUI load it to VRAM for nothing and double-load against the
        # subprocess (the historical VRAM blow-up when porting the script).
        base_path = (input_checkpoint or "").strip()
        if not base_path:
            try:
                import folder_paths
                base_path = folder_paths.get_full_path_or_raise("checkpoints", checkpoint)
            except Exception as e:
                raise ValueError(
                    f"checkpoint {checkpoint!r} could not be resolved from models/checkpoints: {e}"
                )

        if not base_path or not os.path.isfile(base_path):
            raise ValueError(
                "checkpoint: cannot resolve the source checkpoint file. "
                "Pick a checkpoint from the dropdown or set input_checkpoint to a full path. "
                f"Got: {base_path!r}"
            )

        script_path = os.path.join(_REPO_ROOT, _QUANT_SCRIPT)
        if not os.path.isfile(script_path):
            raise FileNotFoundError(f"HSWQ SD1.5 script not found: {script_path}")

        # Resolve calib_file from preset dropdown (128, 256, 512)
        calib_str = str(calib_file).strip()
        calib_map = {
            "128": os.path.join(_REPO_ROOT, "sample", "calibration_prompts_128.txt"),
            "256": os.path.join(_REPO_ROOT, "sample", "calibration_prompts_256.txt"),
            "512": os.path.join(_REPO_ROOT, "sample", "calibration_prompts_512.txt"),
        }
        if calib_str in calib_map:
            calib_path = calib_map[calib_str]
        elif os.path.isabs(calib_str) and os.path.isfile(calib_str):
            calib_path = calib_str
        else:
            # Stale/unknown widget value (e.g. a path left over from an older
            # widget layout). Fall back to the mandated default sheet instead of
            # failing, so a shifted workflow still runs.
            print(
                f"[HSWQ SD1.5 Quantize] WARN: calib_file {calib_file!r} is not a "
                f"known preset; using 128.",
                flush=True,
            )
            calib_path = calib_map["128"]

        if not os.path.isfile(calib_path):
            raise FileNotFoundError(f"Calibration prompts file not found: {calib_path}")

        group_size = int(group_size)
        if not _is_power_of_4(group_size):
            raise ValueError(f"group_size must be a power of 4 (>=4), got {group_size}")

        # Resolve output_path.
        # ComfyUI widgets are positional, so a workflow saved under an older
        # widget layout can shift values by one slot and drop a *calibration*
        # path into output_path (observed: --output <...calibration_prompts_*.txt>).
        # Heal that: an output_path that is a .txt / a directory / not a
        # .safetensors target is not a valid output and is treated as empty.
        output_path = (output_path or "").strip()
        if output_path and (
            output_path.lower().endswith(".txt")
            or os.path.normcase(os.path.abspath(output_path))
            == os.path.normcase(os.path.abspath(calib_path))
        ):
            print(
                f"[HSWQ SD1.5 Quantize] WARN: output_path {output_path!r} looks like a "
                f"calibration file (stale widget); using the default output name.",
                flush=True,
            )
            output_path = ""
        original_stem = os.path.splitext(os.path.basename(base_path))[0]
        ts = int(time.time())
        default_name = f"{original_stem}_hswq_int8_{ts}.safetensors"
        if not output_path:
            output_path = os.path.join(_output_dir(), default_name)
        output_path = os.path.abspath(output_path)
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, default_name)
        if not output_path.lower().endswith(".safetensors"):
            output_path = output_path + ".safetensors"
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


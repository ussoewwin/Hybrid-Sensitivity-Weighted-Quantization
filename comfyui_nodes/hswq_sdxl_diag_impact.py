"""HSWQ SDXL diag impact node (Step 1 of md/How to quantize SDXL.md).

Runs the documented Step 1 command of the SDXL reverse-hybrid method for the
MODEL in the graph:

    python sdxl/diag_impact_sdxl.py "<base>" "<impact>.json" \
        --comfy_path "ComfyUI-master" --artifact v31 \
        --calib_file "sample/calibration_prompts_128.txt" \
        --num_calib_samples 32 --num_inference_steps 25 \
        --steps 25 --seed 42 --width 1024 --height 1024 --cfg 7.0 \
        --sampler dpmpp_2m --scheduler karras --groupsize 256

The command runs unchanged as a child process of this repository's own toolkit,
so the result is identical to running it by hand (including the V3.1 selector
premise and the protect_<stem>.json / selector pack it writes next to the base).

Why a child process: inside ComfyUI the model weights are staged by the dynamic
VRAM manager (comfy_aimdo VBAR / host buffer) and re-staged before every
sampling call, so an in-process weight injection is not honoured - the measured
trajectory would stay bit-identical and the ranking would be meaningless. The
standalone command loads plain weights and measures correctly.

Inputs (the command's parameters):
    model       MODEL  the loaded SDXL model; its checkpoint path is <base>
    steps       INT    --steps      (trajectory denoising steps, production: 25)
    seed        INT    --seed       (trajectory seed, production: 42)
    width       INT    --width
    height      INT    --height
    cfg         FLOAT  --cfg
    sampler     list   --sampler
    scheduler   list   --scheduler
    prompt      STRING --prompt
    negative    STRING --negative
    group_size  INT    --groupsize
    clip        CLIP   optional; not used (the command loads CLIP from the
                       checkpoint itself)

Output: <repo>/impact/impact_<model>.json (repo-relative; created on save)
Outputs: impact_json (STRING), report (STRING)
"""
from __future__ import annotations

import os
import subprocess
import sys

# This file lives in <repo>/comfyui_nodes/, so the repo root is two levels up.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IMPACT_DIR = os.path.join(_REPO_ROOT, "impact")

_DEFAULT_PROMPT = "masterpiece, best quality, 1girl, solo, standing, simple background"

# Documented fixed values of the Step 1 command.
_DIAG_SCRIPT = os.path.join("sdxl", "diag_impact_sdxl.py")
_CALIB_FILE = os.path.join("sample", "calibration_prompts_128.txt")
_COMFY_PATH = "ComfyUI-master"
_ARTIFACT = "v31"
_NUM_CALIB_SAMPLES = 32
_NUM_INFERENCE_STEPS = 25


def _resolve_output_path(base_name: str) -> str:
    """Impact json target: <repo>/impact/impact_<model>.json."""
    stem = os.path.splitext(os.path.basename(base_name or "model"))[0] or "model"
    return os.path.join(_IMPACT_DIR, f"impact_{stem}.json")


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


def _build_cmd(base_path, out_path, steps, seed, width, height, cfg, sampler, scheduler,
               prompt, negative, group_size):
    return [
        sys.executable, os.path.join(_REPO_ROOT, _DIAG_SCRIPT), base_path, out_path,
        "--comfy_path", _COMFY_PATH,
        "--artifact", _ARTIFACT,
        "--calib_file", _CALIB_FILE,
        "--num_calib_samples", str(_NUM_CALIB_SAMPLES),
        "--num_inference_steps", str(_NUM_INFERENCE_STEPS),
        "--steps", str(int(steps)),
        "--seed", str(int(seed)),
        "--width", str(int(width)),
        "--height", str(int(height)),
        "--cfg", str(float(cfg)),
        "--sampler", str(sampler),
        "--scheduler", str(scheduler),
        "--prompt", str(prompt),
        "--negative", str(negative),
        "--groupsize", str(int(group_size)),
    ]


class HSWQSDXLDiagImpact:
    """Run the documented SDXL diag (Step 1) command for the loaded MODEL."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import comfy.samplers

            samplers = list(comfy.samplers.KSampler.SAMPLERS)
            schedulers = list(comfy.samplers.KSampler.SCHEDULERS)
        except Exception:
            samplers = ["dpmpp_2m", "euler", "euler_ancestral", "dpmpp_sde", "ddim", "uni_pc"]
            schedulers = ["karras", "normal", "simple", "exponential", "sgm_uniform", "ddim_uniform"]
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200,
                                  "tooltip": "Trajectory denoising steps (--steps; production: 25)."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                 "tooltip": "Trajectory seed (--seed; production: 42)."}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler": (samplers, {"default": "dpmpp_2m"}),
                "scheduler": (schedulers, {"default": "karras"}),
                "prompt": ("STRING", {"default": _DEFAULT_PROMPT, "multiline": True}),
                "negative": ("STRING", {"default": "", "multiline": True}),
                "group_size": ("INT", {"default": 256, "min": 4, "max": 4096,
                                       "tooltip": "Preferred ConvRot Hadamard group size (power of 4)."}),
            },
            "optional": {
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("impact_json", "report")
    FUNCTION = "measure"
    CATEGORY = "HSWQ/Quantize"
    TITLE = "HSWQ SDXL Diag Impact (ConvRot INT8)"
    OUTPUT_NODE = False

    def measure(self, model, steps, seed, width, height, cfg, sampler, scheduler,
                prompt, negative, group_size, clip=None):
        base_path = _model_base_path(model)
        if not base_path or not os.path.isfile(base_path):
            raise ValueError(
                "model: cannot resolve the checkpoint path of the MODEL input "
                "(load the SDXL checkpoint with a checkpoint loader and connect its "
                f"MODEL output). Got: {base_path!r}"
            )

        script = os.path.join(_REPO_ROOT, _DIAG_SCRIPT)
        calib = os.path.join(_REPO_ROOT, _CALIB_FILE)
        for path in (script, calib):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"toolkit file missing: {path}")

        out_path = _resolve_output_path(base_path)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        # The child process loads its own copy of the checkpoint, so free this
        # process's models first (the method is a one-process-at-a-time job; see
        # md/How to quantize SDXL.md). Mirrors the DistorchMemoryManager
        # "Purge VRAM" sequence: gc -> empty_cache -> cleanup_models ->
        # cleanup_models_gc -> unload_all_models -> free_memory -> empty_cache.
        try:
            import gc

            import comfy.model_management as mm
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for fn_name in ("cleanup_models", "cleanup_models_gc", "unload_all_models"):
                fn = getattr(mm, fn_name, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception as e:
                        print(f"[HSWQ SDXL diag] WARN: {fn_name}: {e}", flush=True)
            free_memory = getattr(mm, "free_memory", None)
            if callable(free_memory):
                try:
                    free_memory(1e30)
                except Exception as e:
                    print(f"[HSWQ SDXL diag] WARN: free_memory: {e}", flush=True)
            if callable(getattr(mm, "soft_empty_cache", None)):
                mm.soft_empty_cache()
            gc.collect()
            if torch.cuda.is_available():
                print(
                    f"[HSWQ SDXL diag] pre-run VRAM: "
                    f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated, "
                    f"{torch.cuda.memory_reserved() / 2**30:.2f} GiB reserved",
                    flush=True,
                )
        except Exception as e:
            print(f"[HSWQ SDXL diag] WARN: could not free models before run: {e}", flush=True)

        cmd = _build_cmd(base_path, out_path, steps, seed, width, height, cfg,
                         sampler, scheduler, prompt, negative, group_size)

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"  # Windows cp932 decode errors otherwise

        print("[HSWQ SDXL diag] " + " ".join(cmd), flush=True)
        lines = []
        proc = subprocess.Popen(
            cmd, cwd=_REPO_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                lines.append(line)
                print("  " + line, flush=True)
            proc.wait()
        finally:
            # If the node is interrupted, do not leave the diag child running.
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"diag command failed (exit {proc.returncode}). Last output:\n"
                + "\n".join(lines[-40:])
            )

        print(f"[HSWQ SDXL diag] done: {out_path}", flush=True)
        return (out_path, "\n".join(lines[-40:]))


NODE_CLASS_MAPPINGS = {
    "HSWQSDXLDiagImpact": HSWQSDXLDiagImpact,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HSWQSDXLDiagImpact": "HSWQ SDXL Diag Impact (ConvRot INT8)",
}

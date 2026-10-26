"""ComfyUI node: quantize SD1.5 / SDXL checkpoints with HSWQ (FP16 protection + ConvRot INT8).

Two fully separated branches, selected by ``model_type``:

* **SD1.5** - runs ``sd1.5/quantize_sd1.5_hswq_v1.0.py`` (HSWQ V1.0: 300 MiB FP16
  protection + ConvRot INT8 + the automatic 25-seed trajectory bench).
* **SDXL**  - runs ``sdxl/gen_reverse_int8_sdxl_v1.1.py`` (reverse hybrid: the K
  lowest-impact pool layers -> ConvRot INT8, every other layer stays FP16). It consumes
  the impact json produced by the *HSWQ SDXL Diag Impact* node (Step 1) and, optionally,
  runs the deterministic 25-seed gate ``benchmark/sdxl_int8_traj_compare.py`` (Step 3).

The engines run unchanged as child processes of this repository's own toolkit, so the
result is identical to running them by hand. Freeing live VRAM models prior to execution
ensures clean calibration and deterministic 25-seed trajectory benchmarking.
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
_SD15_QUANT_SCRIPT = os.path.join("sd1.5", "quantize_sd1.5_hswq_v1.1.py")
_SD15_TRAJ_GATE = os.path.join("benchmark", "sd1.5_int8_traj_compare.py")
_SDXL_REVERSE_SCRIPT = os.path.join("sdxl", "gen_reverse_int8_sdxl_v1.1.py")
_SDXL_TRAJ_GATE = os.path.join("benchmark", "sdxl_int8_traj_compare.py")
_COMFY_PATH = "ComfyUI-master"
_DEFAULT_CALIB_FILE = os.path.join("sample", "calibration_prompts_128.txt")


def _alloc_conf_for_child(env: dict) -> None:
    """Give the HSWQ quantize child tree the caching allocator.

    ComfyUI (cu13+) auto-appends ',backend:cudaMallocAsync' to
    PYTORCH_CUDA_ALLOC_CONF. Under that backend a triton launch inside an
    active CUDA-graph capture is rejected even with every tensor on cuda:0
    (measured 2026-10-16), costing ~11.6% INT8 speed after the once-flag
    abandons the fused path. The quantize child and its bench descendant own
    their allocator choice; strip the backend flag for this child tree only.
    """
    conf = env.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if conf and "backend:" in conf:
        kept = ",".join(
            p for p in (s.strip() for s in conf.split(","))
            if p and not p.startswith("backend:")
        )
        env["PYTORCH_CUDA_ALLOC_CONF"] = kept



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
                f"[HSWQ] pre-run VRAM: "
                f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated, "
                f"{torch.cuda.memory_reserved() / 2**30:.2f} GiB reserved",
                flush=True,
            )
    except Exception as e:
        print(f"[HSWQ] WARN: could not free models before run: {e}", flush=True)


class HSWQQuantize:
    """Quantize a diffusion model using HSWQ (FP16 protection + ConvRot INT8)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["SD1.5", "SDXL"], {"default": "SD1.5"}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
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
                        "tooltip": "Legacy V1.0 budget. Unused in SD1.5 V1.1 (top-K used instead) and SDXL.",
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
                        "tooltip": "SD1.5: enables FULL ConvRot Hadamard rotation on UNet and CLIP.",
                    },
                ),
                "bias_correction": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Cancel the systematic INT8 output bias into layer biases.",
                    },
                ),
                "run_benchmark": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Run the automatic 25-seed deterministic trajectory benchmark.",
                    },
                ),
            },
            "optional": {
                "vae": ("VAE",),
                "input_checkpoint": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional manual override path to the source checkpoint if MODEL does not contain cached_patcher_init.",
                    },
                ),
                "impact_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "forceInput": True,
                        "tooltip": "Connect the impact_json output of the Diag Impact node (Step 1). Required for both SDXL and SD1.5.",
                    },
                ),
                "k_layers": (
                    "INT",
                    {
                        "default": 80,
                        "min": 1,
                        "max": 1000,
                        "tooltip": "SDXL: K lowest-impact layers -> INT8. SD1.5: top-K highest-impact layers to protect in FP16 (production: 80).",
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
        input_checkpoint="",
        impact_json="",
        k_layers=60,
        **kwargs,
    ):
        # Family separation: branch directly on model_type pulldown.
        if model_type == "SDXL":
            return self._quantize_sdxl(
                model,
                output_path,
                calib_file,
                num_calib_samples,
                num_inference_steps,
                group_size,
                bias_correction,
                run_benchmark,
                input_checkpoint,
                impact_json,
                k_layers,
            )
        if model_type == "SD1.5":
            return self._quantize_sd15(
                model,
                output_path,
                calib_file,
                num_calib_samples,
                num_inference_steps,
                bias_correction,
                run_benchmark,
                input_checkpoint,
                impact_json,
                k_layers,
            )
        raise NotImplementedError(
            f"model_type {model_type!r} is not implemented for HSWQ quantizer."
        )

    # ------------------------------------------------------------------
    # SD1.5 branch (V1.1 top-K protected + ConvRot INT8; Step 2 -> optional Step 3)
    # ------------------------------------------------------------------
    def _quantize_sd15(
        self,
        model,
        output_path,
        calib_file,
        num_calib_samples,
        num_inference_steps,
        bias_correction,
        run_benchmark,
        input_checkpoint,
        impact_json,
        k_layers,
    ):
        base_path = (input_checkpoint or "").strip()
        if not base_path:
            base_path = _model_base_path(model)

        if not base_path or not os.path.isfile(base_path):
            raise ValueError(
                "model: cannot resolve the checkpoint path of the MODEL input. "
                "Ensure the model is loaded from a standard Load Checkpoint node, "
                f"or specify input_checkpoint manually. Got: {base_path!r}"
            )

        # Step 1 output: the per-layer trajectory impact json (from HSWQ Diag Impact node).
        imp = (impact_json or "").strip()
        if not imp:
            stem = os.path.splitext(os.path.basename(base_path))[0]
            candidate = os.path.join(_REPO_ROOT, "impact", "impact_sd15_traj_%s.json" % stem.replace(" ", "_"))
            if not os.path.isfile(candidate):
                candidate = os.path.join(_REPO_ROOT, "impact", f"impact_{stem}.json")
            imp = candidate if os.path.isfile(candidate) else ""
        if not imp or not os.path.isfile(imp):
            raise ValueError(
                "impact_json: SD1.5 HSWQ v1.1 needs the impact json produced by "
                "the Diag Impact node (Step 1). Connect that node's impact_json "
                f"output, or set the path manually. Got: {imp!r}"
            )

        script_path = os.path.join(_REPO_ROOT, _SD15_QUANT_SCRIPT)
        if not os.path.isfile(script_path):
            raise FileNotFoundError(f"HSWQ SD1.5 script not found: {script_path}")

        # Resolve calib_file (used when bias_correction is enabled)
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

        k_layers = int(k_layers)
        output_path = (output_path or "").strip()
        original_stem = os.path.splitext(os.path.basename(base_path))[0]
        bc_tag = "_bc" if bias_correction else ""
        default_name = f"{original_stem}_hswq_k{k_layers}{bc_tag}_convrot_int8.safetensors"
        if not output_path:
            output_path = os.path.join(_output_dir(), default_name)
        output_path = os.path.abspath(output_path)
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, default_name)
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        _free_vram_before_run()

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        if "HF_HOME" not in env:
            hf_default = r"D:\USERFILES\MyDocuments\HuggingFace"
            if os.path.isdir(hf_default):
                env["HF_HOME"] = hf_default
        _alloc_conf_for_child(env)

        # Build command for quantize_sd1.5_hswq_v1.1.py
        cmd = [
            sys.executable,
            script_path,
            "--input", base_path,
            "--output", output_path,
            "--contrib", imp,
            "--top-k", str(k_layers),
        ]
        if bias_correction:
            if not os.path.isfile(calib_path):
                raise FileNotFoundError(f"Calibration prompts file not found: {calib_path}")
            cmd += [
                "--bias_correction",
                "--calib_file", calib_path,
                "--num_calib_samples", str(int(num_calib_samples)),
                "--num_inference_steps", str(int(num_inference_steps)),
            ]

        print("[HSWQ SD1.5 Quantize] " + " ".join(cmd), flush=True)
        lines = self._run_child(cmd, env)
        if not os.path.isfile(output_path):
            raise RuntimeError(
                "HSWQ SD1.5 conversion finished but the output file was not written: "
                f"{output_path}\nOutput tail:\n" + "\n".join(lines[-40:])
            )

        # Step 3 (optional): 25-seed trajectory benchmark
        if run_benchmark:
            gate = os.path.join(_REPO_ROOT, _SD15_TRAJ_GATE)
            if not os.path.isfile(gate):
                raise FileNotFoundError(f"HSWQ SD1.5 trajectory gate not found: {gate}")
            comfy_root = os.path.join(_REPO_ROOT, _COMFY_PATH)
            cmd_gate = [
                sys.executable,
                gate,
                "--fp16", base_path,
                "--int8", output_path,
                "--comfy_path", comfy_root,
                "--steps", "25",
                "--width", "512",
                "--height", "512",
                "--cfg", "7.0",
                "--sampler", "dpmpp_2m",
                "--scheduler", "karras",
            ]
            print("[HSWQ SD1.5 Bench] " + " ".join(cmd_gate), flush=True)
            lines += self._run_child(cmd_gate, env)

        print(f"[HSWQ SD1.5 Quantize] Completed: {output_path}", flush=True)
        report = "\n".join(lines[-60:])
        return (output_path, report)

    # ------------------------------------------------------------------
    # SDXL branch (reverse hybrid ConvRot INT8; Step 2 -> optional Step 3)
    # ------------------------------------------------------------------
    def _quantize_sdxl(
        self,
        model,
        output_path,
        calib_file,
        num_calib_samples,
        num_inference_steps,
        group_size,
        bias_correction,
        run_benchmark,
        input_checkpoint,
        impact_json,
        k_layers,
    ):
        base_path = (input_checkpoint or "").strip()
        if not base_path:
            base_path = _model_base_path(model)
        if not base_path or not os.path.isfile(base_path):
            raise ValueError(
                "model: cannot resolve the checkpoint path of the MODEL input. "
                "Ensure the model is loaded from a standard Load Checkpoint node, "
                f"or specify input_checkpoint manually. Got: {base_path!r}"
            )

        # Step 1 output: the per-layer trajectory impact json (HSWQ SDXL Diag Impact node).
        imp = (impact_json or "").strip()
        if not imp:
            stem = os.path.splitext(os.path.basename(base_path))[0]
            candidate = os.path.join(_REPO_ROOT, "impact", f"impact_{stem}.json")
            imp = candidate if os.path.isfile(candidate) else ""
        if not imp or not os.path.isfile(imp):
            raise ValueError(
                "impact_json: the SDXL reverse hybrid needs the impact json produced by "
                "the HSWQ SDXL Diag Impact node (Step 1). Connect that node's impact_json "
                f"output, or set the path manually. Got: {imp!r}"
            )

        script_path = os.path.join(_REPO_ROOT, _SDXL_REVERSE_SCRIPT)
        if not os.path.isfile(script_path):
            raise FileNotFoundError(f"HSWQ SDXL converter not found: {script_path}")

        group_size = int(group_size)
        if not _is_power_of_4(group_size):
            raise ValueError(f"group_size must be a power of 4 (>=4), got {group_size}")

        # Calibration prompts (used only when bias correction is enabled).
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

        # Resolve output_path (the hybrid safetensors).
        k_layers = int(k_layers)
        output_path = (output_path or "").strip()
        original_stem = os.path.splitext(os.path.basename(base_path))[0]
        ts = int(time.time())
        default_name = f"{original_stem}_rev_int{k_layers}_convrot_int8_{ts}.safetensors"
        if not output_path:
            output_path = os.path.join(_output_dir(), default_name)
        output_path = os.path.abspath(output_path)
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, default_name)
        out_dir = os.path.dirname(output_path)
        out_name = os.path.basename(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        _free_vram_before_run()

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        _alloc_conf_for_child(env)

        # Step 2: reverse hybrid conversion (K lowest-impact pool layers -> ConvRot INT8).
        cmd = [
            sys.executable,
            script_path,
            str(k_layers),
            out_name,
            base_path,
            imp,
            "--out-dir", out_dir,
            "--groupsize", str(group_size),
        ]
        if bias_correction:
            if not os.path.isfile(calib_path):
                raise FileNotFoundError(f"Calibration prompts file not found: {calib_path}")
            cmd += [
                "--bias_correction",
                "--calib_file", calib_path,
                "--comfy_path", _COMFY_PATH,
                "--num_calib_samples", str(int(num_calib_samples)),
                "--num_inference_steps", str(int(num_inference_steps)),
            ]

        print("[HSWQ SDXL Quantize] " + " ".join(cmd), flush=True)
        lines = self._run_child(cmd, env)
        if not os.path.isfile(output_path):
            raise RuntimeError(
                "HSWQ SDXL conversion finished but the hybrid file was not written: "
                f"{output_path}\nOutput tail:\n" + "\n".join(lines[-40:])
            )

        # Step 2b: CLIP ConvRot INT8 with rule-based FP16 protection (same stage
        # the script reference flow runs after the UNet reverse hybrid; without
        # it the output pack keeps the CLIP at FP16 and is ~382 MiB larger than
        # the script-made reference). The CLIP INT8 stage is the SDXL production
        # setting (SD1.5 CLIP INT8 is removed in the SD1.5-only flow).
        clip_script = os.path.join(_REPO_ROOT, "sdxl", "quantize_sdxl_clip_int8.py")
        clip_tmp = output_path + ".clip_tmp.safetensors"
        if os.path.isfile(clip_script):
            cmd_clip = [
                sys.executable,
                clip_script,
                "--input", output_path,
                "--output", clip_tmp,
            ]
            print("[HSWQ SDXL CLIP INT8] " + " ".join(cmd_clip), flush=True)
            lines += self._run_child(cmd_clip, env)
            if not os.path.isfile(clip_tmp):
                raise RuntimeError(
                    "HSWQ SDXL CLIP conversion finished but the output was not written: "
                    f"{clip_tmp}\nOutput tail:\n" + "\n".join(lines[-40:])
                )
            os.replace(clip_tmp, output_path)

        # Step 3 (optional): deterministic 25-seed trajectory gate on the FINAL
        # pack (CLIP INT8 included — the gate must measure what gets used).
        if run_benchmark:
            gate = os.path.join(_REPO_ROOT, _SDXL_TRAJ_GATE)
            if not os.path.isfile(gate):
                raise FileNotFoundError(f"HSWQ SDXL trajectory gate not found: {gate}")
            cmd_gate = [
                sys.executable,
                gate,
                "--fp16", base_path,
                "--int8", output_path,
                "--comfy_path", _COMFY_PATH,
                "--steps", "25",
            ]
            print("[HSWQ SDXL Gate] " + " ".join(cmd_gate), flush=True)
            lines += self._run_child(cmd_gate, env)

        print(f"[HSWQ SDXL Quantize] Completed: {output_path}", flush=True)
        report = "\n".join(lines[-60:])
        return (output_path, report)

    @staticmethod
    def _run_child(cmd, env):
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
            raise RuntimeError(
                f"HSWQ step failed (exit {proc.returncode}). Output tail:\n"
                + "\n".join(lines[-40:])
            )
        return lines


# Backwards compatibility alias
HSWQSD15Quantize = HSWQQuantize

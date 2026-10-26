"""HSWQ diag impact node (Step 1 of md/How to quantize SDXL.md and
md/How to quantize SD1.5.md).

The `model_type` dropdown selects which family's diag runs. The two families do
not share a command, so the dispatch is explicit:

* **SDXL** - `sdxl/diag_impact_sdxl.py`, run as **a child process**:

    python sdxl/diag_impact_sdxl.py "<base>" "<impact>.json" \
        --comfy_path "ComfyUI-master" --artifact v31 \
        --calib_file "sample/calibration_prompts_128.txt" \
        --num_calib_samples 32 --num_inference_steps 25 \
        --steps 25 --seed 42 --width 1024 --height 1024 --cfg 7.0 \
        --sampler dpmpp_2m --scheduler karras --groupsize 256

* **SD1.5** - `sd1.5/diagimpact_sd15.py`, also run as **a child process** with
  `backend:cudaMallocAsync` stripped and `comfy_aimdo` stubbed:

    python sd1.5/diagimpact_sd15.py \
        --comfy "<ComfyUI root>" --ckpt "<base>" --out "<impact>.json" \
        --steps 25 --seeds "42,137" --width 512 --height 512 --cfg 7.0 \
        --sampler dpmpp_2m --scheduler karras --groupsize 256 --metric relmse

Why both families run as child processes: inside ComfyUI the model weights are
staged by the dynamic VRAM manager (comfy_aimdo VBAR / host buffer) and
re-staged before every sampling call, and ComfyUI (cu13+) appends
`backend:cudaMallocAsync` to `PYTORCH_CUDA_ALLOC_CONF`. Running as a standalone
child process with `backend:cudaMallocAsync` stripped and `comfy_aimdo` stubbed
ensures weight perturbations are honoured and VRAM stays within budget.

Inputs (the command's parameters):
    model       MODEL  the loaded SDXL / SD1.5 model; its checkpoint path is <base>
    steps       INT    --steps      (trajectory denoising steps, production: 25)
    seed        INT    --seed       (trajectory seed, production: 42; SD1.5 uses 42,137 when seed=42)
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

Output: <repo>/impact/impact_<model>.json for SDXL,
        <repo>/impact/impact_sd15_traj_<model>.json for SD1.5
        (both repo-relative; the SD1.5 name is that script's own default, which
        is what the SD1.5 quantize node and every existing SD1.5 file use)
Outputs: impact_json (STRING), report (STRING)
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

# This file lives in <repo>/comfyui_nodes/, so the repo root is two levels up.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IMPACT_DIR = os.path.join(_REPO_ROOT, "impact")  # Owner 2026-10-17: the impact.json location is <repo>/impact (reverted)

_DEFAULT_PROMPT = "masterpiece, best quality, 1girl, solo, standing, simple background"

MODEL_TYPES = ["SDXL", "SD1.5"]

# Documented fixed values of the SDXL Step 1 command.
_DIAG_SCRIPT = os.path.join("sdxl", "diag_impact_sdxl.py")
_CALIB_FILE = os.path.join("sample", "calibration_prompts_128.txt")
_COMFY_PATH = "ComfyUI-master"
_ARTIFACT = "v31"
_NUM_CALIB_SAMPLES = 32
_NUM_INFERENCE_STEPS = 25

# The SD1.5 Step 1 command lives in its own script.
_SD15_SCRIPT = os.path.join("sd1.5", "diagimpact_sd15.py")
_SD15_SEEDS = "42,137"
_SD15_METRIC = "relmse"
_SD15_MODULE_NAME = "hswq_sd15_diagimpact"


def _resolve_output_path(base_name: str, model_type: str) -> str:
    """The impact json target, in each family's own established naming."""
    stem = os.path.splitext(os.path.basename(base_name or "model"))[0] or "model"
    # The SD1.5 script replaces spaces with underscores in its own default name;
    # mirror that so the node and the CLI produce the same file.
    if model_type == "SD1.5":
        return os.path.join(_IMPACT_DIR, "impact_sd15_traj_%s.json" % stem.replace(" ", "_"))
    return os.path.join(_IMPACT_DIR, f"impact_{stem}.json")


def _load_sd15_module():
    """Import sd1.5/diagimpact_sd15.py by path.

    The repo root is not a package, so spec_from_file_location is the way in.
    The module has no import-time side effects: `COMFY` is "" and `comfy.*` is
    bound later by `_bind_comfy`.
    """
    script = os.path.join(_REPO_ROOT, _SD15_SCRIPT)
    if not os.path.isfile(script):
        raise FileNotFoundError(f"toolkit file missing: {script}")
    if _SD15_MODULE_NAME in sys.modules:
        return sys.modules[_SD15_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_SD15_MODULE_NAME, script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build an import spec for {script}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_SD15_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


def _sd15_argv(base_path, out_path, steps, seeds, metric, comfy_path,
               width=None, height=None, cfg=None, sampler=None,
               scheduler=None, prompt=None, negative=None, group_size=None):
    """The exact sys.argv the SD1.5 script's own parser expects."""
    argv = [
        os.path.join(_REPO_ROOT, _SD15_SCRIPT),
        "--comfy", comfy_path,
        "--ckpt", base_path,
        "--out", out_path,
        "--steps", str(int(steps)),
        "--seeds", str(seeds),
        "--metric", str(metric),
    ]
    if width is not None:
        argv += ["--width", str(int(width))]
    if height is not None:
        argv += ["--height", str(int(height))]
    if cfg is not None:
        argv += ["--cfg", str(float(cfg))]
    if sampler is not None:
        argv += ["--sampler", str(sampler)]
    if scheduler is not None:
        argv += ["--scheduler", str(scheduler)]
    if prompt is not None:
        argv += ["--prompt", str(prompt)]
    if negative is not None:
        argv += ["--negative", str(negative)]
    if group_size is not None:
        argv += ["--groupsize", str(int(group_size))]
    return argv


def _alloc_conf_for_child(env: dict) -> None:
    """Give the diag child tree the caching allocator.

    Measured (2026-10-16, waiIllustriousSDXL_v170, 2-sample selector calib):
    with ComfyUI's auto-appended backend:cudaMallocAsync the v3.1 selector's
    reserved VRAM grows to 17.00 GiB (thrashing on a 16 GiB card) while the
    same run with the caching allocator reserves 10.28 GiB (-6.7 GiB). The
    diag child and its selector descendant own their allocator choice; strip
    the backend flag for this child tree only. Same mechanism as the SD1.5
    quantize node (commit 566fcba); kept as a self-contained copy per family
    separation.
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
    """Run the documented SDXL / SD1.5 diag (Step 1) command for the loaded MODEL."""

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
                "model_type": (MODEL_TYPES, {
                    "default": "SDXL",
                    "tooltip": "Which diag runs. SDXL = sdxl/diag_impact_sdxl.py (child "
                               "process). SD1.5 = sd1.5/diagimpact_sd15.py (child "
                               "process).",
                }),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200,
                                  "tooltip": "Trajectory denoising steps (--steps; production: 25 for both families)."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                 "tooltip": "Trajectory seed (production: 42; SD1.5 defaults to '42,137' when seed=42)."}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64,
                                  "tooltip": "Latent width in pixels (SDXL default 1024; SD1.5 uses 512 when left at 1024x1024)."}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64,
                                   "tooltip": "Latent height in pixels (SDXL default 1024; SD1.5 uses 512 when left at 1024x1024)."}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1,
                                  "tooltip": "Classifier-free guidance scale (--cfg, default 7.0)."}),
                "sampler": (samplers, {"default": "dpmpp_2m", "tooltip": "Sampler name (--sampler, default dpmpp_2m)."}),
                "scheduler": (schedulers, {"default": "karras", "tooltip": "Scheduler name (--scheduler, default karras)."}),
                "prompt": ("STRING", {"default": _DEFAULT_PROMPT, "multiline": True,
                                      "tooltip": "Positive prompt (--prompt)."}),
                "negative": ("STRING", {"default": "", "multiline": True,
                                        "tooltip": "Negative prompt (--negative)."}),
                "group_size": ("INT", {"default": 256, "min": 4, "max": 4096,
                                       "tooltip": "Preferred ConvRot Hadamard group size (power of 4, default 256)."}),
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

    def measure(self, model, model_type, steps, seed, width, height, cfg, sampler,
                scheduler, prompt, negative, group_size, clip=None):
        if model_type not in MODEL_TYPES:
            raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {model_type!r}")

        base_path = _model_base_path(model)
        if not base_path or not os.path.isfile(base_path):
            raise ValueError(
                "model: cannot resolve the checkpoint path of the MODEL input "
                "(load the checkpoint with a checkpoint loader and connect its "
                f"MODEL output). Got: {base_path!r}"
            )

        if model_type == "SDXL":
            script = os.path.join(_REPO_ROOT, _DIAG_SCRIPT)
            calib = os.path.join(_REPO_ROOT, _CALIB_FILE)
            for path in (script, calib):
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"toolkit file missing: {path}")
        else:
            script = os.path.join(_REPO_ROOT, _SD15_SCRIPT)
            if not os.path.isfile(script):
                raise FileNotFoundError(f"toolkit file missing: {script}")

        out_path = _resolve_output_path(base_path, model_type)
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
                    # comfy's free_memory(minimum_free_memory, device)
                    free_memory(1e30, mm.get_torch_device())
                except TypeError:
                    try:
                        free_memory(1e30)
                    except Exception as e:
                        print(f"[HSWQ SDXL diag] WARN: free_memory: {e}", flush=True)
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

        if model_type == "SD1.5":
            return self._measure_sd15(
                base_path, out_path, steps,
                seed=seed, width=width, height=height, cfg=cfg,
                sampler=sampler, scheduler=scheduler,
                prompt=prompt, negative=negative, group_size=group_size,
            )

        cmd = _build_cmd(base_path, out_path, steps, seed, width, height, cfg,
                         sampler, scheduler, prompt, negative, group_size)

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"  # Windows cp932 decode errors otherwise
        _alloc_conf_for_child(env)

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

    def _measure_sd15(self, base_path, out_path, steps, seed=42, width=512, height=512,
                      cfg=7.0, sampler="dpmpp_2m", scheduler="karras",
                      prompt=_DEFAULT_PROMPT, negative="", group_size=256):
        """Run the SD1.5 diag as a child process with backend:cudaMallocAsync stripped."""
        comfy_path = os.path.join(_REPO_ROOT, _COMFY_PATH)
        if not os.path.isdir(os.path.join(comfy_path, "comfy")):
            raise ValueError(
                f"the ComfyUI root is missing (no comfy/ inside): {comfy_path!r}"
            )
        seeds = _SD15_SEEDS if int(seed) == 42 else str(int(seed))
        w = 512 if (int(width) == 1024 and int(height) == 1024) else int(width)
        h = 512 if (int(width) == 1024 and int(height) == 1024) else int(height)
        argv = _sd15_argv(
            base_path, out_path, steps, seeds, _SD15_METRIC, comfy_path,
            width=w, height=h, cfg=cfg, sampler=sampler, scheduler=scheduler,
            prompt=prompt, negative=negative, group_size=group_size,
        )
        cmd = [sys.executable] + argv

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        _alloc_conf_for_child(env)

        print("[HSWQ SD1.5 diag] " + " ".join(cmd), flush=True)
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
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"SD1.5 diag command failed (exit {proc.returncode}). Last output:\n"
                + "\n".join(lines[-40:])
            )

        if not os.path.isfile(out_path):
            raise RuntimeError(
                f"the SD1.5 diag finished but wrote no file: {out_path}"
            )
        print(f"[HSWQ SD1.5 diag] done: {out_path}", flush=True)
        return (out_path, "\n".join(lines[-40:]))


NODE_CLASS_MAPPINGS = {
    "HSWQSDXLDiagImpact": HSWQSDXLDiagImpact,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HSWQSDXLDiagImpact": "HSWQ Diag Impact (ConvRot INT8)",
}

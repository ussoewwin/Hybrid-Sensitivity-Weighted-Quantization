#!/usr/bin/env python3
"""Z Image deterministic trajectory-divergence comparator (FP16 vs ConvRot INT8+NVFP4 Hybrid).

Per-step latent trajectory comparison between BF16/FP16 and ConvRot INT8 + ConvRot NVFP4 Hybrid models.
Uses the optimized Z Image NVFP4 reference stack (Blackwell Tensor Core / comfy_parity) from
benchmark/zi_convrot_nvfp4_bench_v3.py.

Usage:
    python benchmark/zi_convrot_nvfp4_traj_compare.py \
        --fp16 <bf16.safetensors> --quant <convrot_hybrid.safetensors> \
        --clip_path <clip.safetensors> --comfy_path <ComfyUI-master> \
        [--seeds "42,1337,7,2024,555"] [--steps 25] [--cfg 2.5] [--tc]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import types
from pathlib import Path

import torch

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_HSWQ_STACK = os.path.join(_BENCH_DIR, "hswq_stack")
_REPO_DIR = os.path.dirname(_BENCH_DIR)

for _p in (_BENCH_DIR, _HSWQ_STACK, _REPO_DIR):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def _clear_argv_for_comfy() -> list[str]:
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    return saved


def _restore_argv(saved: list[str]) -> None:
    sys.argv = saved


def _install_torchaudio_stub() -> None:
    import importlib.machinery

    for key in list(sys.modules):
        if key == "torchaudio" or key.startswith("torchaudio."):
            del sys.modules[key]

    def _stub_mod(name: str, *, is_package: bool = False):
        mod = types.ModuleType(name)
        mod.__file__ = "<hswq_torchaudio_stub>"
        if is_package:
            mod.__path__ = []
            spec = importlib.machinery.ModuleSpec(
                name, loader=None, is_package=True
            )
            spec.submodule_search_locations = []
        else:
            spec = importlib.machinery.ModuleSpec(name, loader=None)
        mod.__spec__ = spec
        return mod

    ta = _stub_mod("torchaudio", is_package=True)
    functional = _stub_mod("torchaudio.functional")
    functional.resample = lambda waveform, orig_freq, new_freq, *a, **k: waveform
    transforms = _stub_mod("torchaudio.transforms")

    class _MelSpectrogram:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, x):
            return x

        def to(self, *args, **kwargs):
            return self

    class _MelScale:
        def __init__(self, *args, **kwargs):
            pass

    transforms.MelSpectrogram = _MelSpectrogram
    transforms.MelScale = _MelScale

    ta.functional = functional
    ta.transforms = transforms
    sys.modules["torchaudio"] = ta
    sys.modules["torchaudio.functional"] = functional
    sys.modules["torchaudio.transforms"] = transforms


def setup_comfy(comfy_path: str) -> None:
    comfy_root = Path(comfy_path).resolve()
    if not comfy_root.is_dir():
        raise FileNotFoundError(f"--comfy_path not found: {comfy_root}")
    bench_dir = Path(__file__).resolve().parent
    hswq_stack_dir = bench_dir / "hswq_stack"
    repo_dir = bench_dir.parent

    required_paths = [str(comfy_root), str(bench_dir), str(hswq_stack_dir), str(repo_dir)]
    new_sys_path = []
    for p in required_paths:
        if p not in new_sys_path and os.path.isdir(p):
            new_sys_path.append(p)
    for p in sys.path:
        if p not in new_sys_path:
            new_sys_path.append(p)
    sys.path = new_sys_path

    _install_torchaudio_stub()

    import comfy.options

    comfy.options.enable_args_parsing(False)

    try:
        import comfy_aimdo  # noqa: F401
    except Exception:
        m = types.ModuleType("comfy_aimdo")
        m.__file__ = "<stub>"
        m.__path__ = []
        sys.modules["comfy_aimdo"] = m
        sys.modules["comfy_aimdo.filter"] = types.ModuleType("comfy_aimdo.filter")
        sys.modules["comfy_aimdo.filter"].filter_modules = lambda *a, **k: None

    try:
        import psutil  # noqa: F401
    except Exception:
        class _VM:
            total = 64 * 1024**3
            available = 32 * 1024**3

        class _Proc:
            def memory_info(self):
                return types.SimpleNamespace(rss=0)

            def memory_full_info(self):
                return types.SimpleNamespace(uss=0)

            def cpu_percent(self, interval=None):
                return 0.0

            def num_threads(self):
                return 1

        ps = types.ModuleType("psutil")
        ps.virtual_memory = lambda: _VM()
        ps.Process = lambda: _Proc()
        sys.modules["psutil"] = ps

    try:
        from kitchen_rms_rope_fallback import ensure_kitchen_rms_rope

        ensure_kitchen_rms_rope()
    except Exception as e:
        print(f"  [Note] kitchen_rms_rope_fallback skipped: {e}")


def set_hf_token(token: str | None) -> None:
    if not token:
        return
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def _decode_comfy_quant_blob(blob) -> dict | None:
    """Decode uint8 comfy_quant tensor / bytes to a dict, or None."""
    if blob is None:
        return None
    try:
        if hasattr(blob, "detach"):
            raw = bytes(blob.detach().cpu().tolist())
        elif isinstance(blob, (bytes, bytearray)):
            raw = bytes(blob)
        else:
            raw = bytes(blob)
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def count_convrot_in_quant_metadata(metadata: dict | None) -> tuple[int, int, str]:
    """Return (n_convrot, n_nvfp4, hswq_nvfp4_convrot flag string)."""
    meta = metadata or {}
    flag = str(meta.get("hswq_nvfp4_convrot", "") or "")
    raw = meta.get("_quantization_metadata")
    if not raw:
        return 0, 0, flag
    try:
        qmap = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return 0, 0, flag
    layers = (qmap or {}).get("layers") or {}
    n_nv = 0
    n_cr = 0
    for conf in layers.values():
        if not isinstance(conf, dict):
            continue
        fmt = str(conf.get("format", "")).lower()
        if fmt != "nvfp4":
            continue
        n_nv += 1
        if conf.get("convrot") is True or str(conf.get("convrot", "")).lower() in (
            "1",
            "true",
        ):
            n_cr += 1
    return n_cr, n_nv, flag


def count_convrot_comfy_quant_markers(state_dict: dict) -> tuple[int, int]:
    """Count .comfy_quant markers that are nvfp4 / nvfp4+convrot."""
    n_nv = 0
    n_cr = 0
    for k, v in state_dict.items():
        if not k.endswith(".comfy_quant"):
            continue
        conf = _decode_comfy_quant_blob(v)
        if not conf or str(conf.get("format", "")).lower() != "nvfp4":
            continue
        n_nv += 1
        if conf.get("convrot") is True or str(conf.get("convrot", "")).lower() in (
            "1",
            "true",
        ):
            n_cr += 1
    return n_cr, n_nv


def count_armed_convrot_linears(model) -> tuple[int, int]:
    """Return (n_armed_convrot, n_linear_modules) on a loaded NextDiT."""
    n_lin = 0
    n_armed = 0
    for _name, mod in model.named_modules():
        if not hasattr(mod, "weight"):
            continue
        if not hasattr(mod, "in_features"):
            continue
        n_lin += 1
        if getattr(mod, "_hswq_nvfp4_convrot", False):
            n_armed += 1
    return n_armed, n_lin


def apply_nvfp4_patches(nvfp4_path=None, force_tc=False, force_parity=False) -> None:
    """Arm Z Image ConvRot NVFP4 reference stack (TC if calibrated, else parity)."""
    if force_tc:
        os.environ["HSWQ_ZI_FORCE_TC"] = "1"
        os.environ.pop("HSWQ_ZI_FORCE_PARITY", None)
    elif force_parity:
        os.environ["HSWQ_ZI_FORCE_PARITY"] = "1"
        os.environ.pop("HSWQ_ZI_FORCE_TC", None)

    try:
        from hswq_stack.zimage_nvfp4.load_unet import apply_nvfp4_patches as _ref_apply
        _ref_apply(nvfp4_path)
    except ImportError:
        from zimage_nvfp4.load_unet import apply_nvfp4_patches as _ref_apply
        _ref_apply(nvfp4_path)


def load_zit_model(path, is_nvfp4=False, require_convrot=False):
    """ComfyUI standard loading: load_diffusion_model_state_dict -> ModelPatcher."""
    import comfy.sd
    import comfy.utils
    from safetensors.torch import load_file

    print(f"Loading state_dict: {os.path.basename(path)}")

    metadata = None
    if is_nvfp4:
        state_dict, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
        n_cr_meta, n_nv_meta, flag = count_convrot_in_quant_metadata(metadata)
        print(
            f"  [CONVROT meta] hswq_nvfp4_convrot={flag!r} "
            f"nvfp4_layers={n_nv_meta} convrot_stamps={n_cr_meta}"
        )
        state_dict, metadata = comfy.utils.convert_old_quants(
            state_dict, "", metadata=metadata or {}
        )
        n_cq = sum(1 for k in state_dict if k.endswith(".comfy_quant"))
        n_cr_cq, n_nv_cq = count_convrot_comfy_quant_markers(state_dict)
        print(f"  [NVFP4] convert_old_quants -> {n_cq} .comfy_quant markers (nvfp4={n_nv_cq}, convrot={n_cr_cq})")
    else:
        state_dict = load_file(path)

    model_options = {"dtype": torch.float16}
    patcher = comfy.sd.load_diffusion_model_state_dict(
        state_dict, model_options=model_options, metadata=metadata
    )
    if patcher is None:
        raise RuntimeError("ComfyUI could not detect model (load_diffusion_model_state_dict returned None)")

    if is_nvfp4:
        try:
            n_armed, n_lin = count_armed_convrot_linears(patcher.model)
            print(f"  [CONVROT armed] Linears with _hswq_nvfp4_convrot: {n_armed} / {n_lin}")
        except Exception as ex:
            print(f"  [CONVROT armed] check skipped: {ex}")
    return patcher


def encode_prompt(clip, prompt: str):
    tokens = clip.tokenize(prompt)
    return clip.encode_from_tokens_scheduled(tokens)


def make_empty_latent(model, width: int, height: int, batch: int = 1) -> dict:
    import comfy.model_management as mm
    import comfy.sample as comfy_sample

    device = mm.intermediate_device()
    latent = torch.zeros([batch, 16, height // 8, width // 8], device=device)
    latent = comfy_sample.fix_empty_latent_channels(model, latent)
    return {"samples": latent}


def _hard_free_vram() -> None:
    import comfy.model_management as mm

    mm.unload_all_models()
    mm.soft_empty_cache(force=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def run_trajectory(model, positive, negative, latent, *, seed, steps, cfg,
                   sampler_name, scheduler):
    """Run full denoising with model GPU pre-loaded; return (per_step_x, per_step_x0, final_sample)."""
    import comfy.model_management as mm
    import comfy.sample as comfy_sample
    import comfy.utils

    # Preload to GPU for maximum Tensor Core execution throughput
    if hasattr(model, "model"):
        mm.load_models_gpu([model], force_full_load=True)

    noise = comfy_sample.prepare_noise(latent["samples"], seed, None)
    xs, x0s = [], []

    def cb(step, x0, x, total_steps):
        xs.append(x.detach().float().cpu())
        x0s.append(x0.detach().float().cpu())

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy_sample.sample(
        model, noise, steps, cfg, sampler_name, scheduler,
        positive, negative, latent["samples"], denoise=1.0,
        disable_noise=False, start_step=None, last_step=None,
        force_full_denoise=False, noise_mask=None,
        callback=cb, disable_pbar=disable_pbar, seed=seed,
    )
    return xs, x0s, samples


def _cos(a, b):
    a = a.reshape(1, -1).float()
    b = b.reshape(1, -1).float()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=1).item())


def _mse(a, b):
    return float((a.float() - b.float()).pow(2).mean().item())


def parse_args():
    ap = argparse.ArgumentParser(
        description="Z Image deterministic per-step trajectory divergence (FP16 vs ConvRot INT8+NVFP4 Hybrid)"
    )
    ap.add_argument("--fp16", required=True, help="BF16/FP16 baseline model path")
    ap.add_argument(
        "--quant", "--nvfp4", "--hybrid", dest="quant_path", required=True,
        help="ConvRot INT8 + ConvRot NVFP4 hybrid quantized model path"
    )
    ap.add_argument("--clip_path", required=True, help="Qwen3-4B text encoder path")
    ap.add_argument("--comfy_path", required=True, help="ComfyUI-master root")
    ap.add_argument("--token", default=None, help="Hugging Face token")
    ap.add_argument(
        "--prompt",
        default="masterpiece, best quality, 1girl, solo, standing, simple background",
        help="Benchmark prompt"
    )
    ap.add_argument("--negative", default="", help="Negative prompt")
    ap.add_argument("--steps", type=int, default=25, help="Sampling steps")
    ap.add_argument(
        "--seeds", default="42,1337,7,2024,555",
        help="comma-separated seeds; same seed = identical noise for both models"
    )
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--cfg", type=float, default=2.5, help="Classifier-free guidance scale")
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument(
        "--tc",
        action="store_true",
        help=(
            "Force hardware Tensor Core W4A4 path (scaled_mm_nvfp4). "
            "Requires a calibrated *_calib checkpoint with .input_scale keys, "
            "otherwise the trajectory collapses."
        ),
    )
    ap.add_argument("--parity", action="store_true", help="Force Comfy parity path (stock GEMM + act rotate)")
    ap.add_argument(
        "--show-steps", action="store_true",
        help="print the per-step divergence curve (default: only final per seed)"
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    # Deterministic comparison: same seed = same noise. Pin cuDNN to avoid
    # autotuning / algorithm-selection noise between the FP16 and quant runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    set_hf_token(args.token)

    saved_argv = _clear_argv_for_comfy()
    try:
        setup_comfy(args.comfy_path)

        import folder_paths  # noqa: F401
        import comfy.model_management as mm
        import comfy.sd

        mm.get_torch_device()

        _cpu = torch.device("cpu")
        print("Loading CLIP on CPU (Z Image / Qwen3-4B)...")
        clip = comfy.sd.load_clip(
            ckpt_paths=[args.clip_path],
            embedding_directory=None,
            model_options={"load_device": _cpu, "offload_device": _cpu, "initial_device": _cpu},
        )
        positive = encode_prompt(clip, args.prompt)
        negative = encode_prompt(clip, args.negative)
        if getattr(clip, "cond_stage_model", None) is not None:
            clip.cond_stage_model.cpu()
        if getattr(clip, "patcher", None) is not None:
            mm.unload_model_and_clones(clip.patcher)
        del clip
        _hard_free_vram()
        print("  [Offload] CLIP unloaded.")

        # --- FP16 (stock ops, before any patch) ---
        print(f"\n--- Loading FP16 baseline: {args.fp16} ---")
        fp16 = load_zit_model(args.fp16, is_nvfp4=False)
        latent = make_empty_latent(fp16, args.width, args.height, batch=1)
        fp16_runs = {}
        for s in seeds:
            print(f"[FP16] seed {s}")
            xs, x0s, final = run_trajectory(
                fp16, positive, negative, latent, seed=s, steps=args.steps,
                cfg=args.cfg, sampler_name=args.sampler, scheduler=args.scheduler,
            )
            fp16_runs[s] = (xs, x0s, final.detach().float().cpu())
        del fp16
        _hard_free_vram()

        # --- ConvRot INT8 + ConvRot NVFP4 Hybrid (reference stack) ---
        print(f"\nApplying Z Image ConvRot NVFP4 + INT8 reference stack...")
        apply_nvfp4_patches(args.quant_path, force_tc=args.tc, force_parity=args.parity)
        print(f"--- Loading Quantized Hybrid: {args.quant_path} ---")
        quant_model = load_zit_model(args.quant_path, is_nvfp4=True, require_convrot=True)
        quant_runs = {}
        for s in seeds:
            print(f"[Quantized Hybrid] seed {s}")
            xs, x0s, final = run_trajectory(
                quant_model, positive, negative, latent, seed=s, steps=args.steps,
                cfg=args.cfg, sampler_name=args.sampler, scheduler=args.scheduler,
            )
            quant_runs[s] = (xs, x0s, final.detach().float().cpu())
        del quant_model
        _hard_free_vram()
    finally:
        _restore_argv(saved_argv)

    # --- compare ---
    print("\n" + "=" * 72)
    print("Deterministic per-step latent trajectory divergence (FP16 vs ConvRot Hybrid)")
    print("=" * 72)
    BIFURC_DROP = 0.05   # single-step cosine drop threshold = sudden jump (different image)
    SAME_IMG_COS = 0.98  # final cosine above this = same picture (not merely different)
    final_rows = []
    for s in seeds:
        fxs, fx0s, ffinal = fp16_runs[s]
        nxs, nx0s, nfinal = quant_runs[s]
        n_steps = min(len(fxs), len(nxs))
        step_cos = [_cos(fxs[i], nxs[i]) for i in range(n_steps)]
        # sudden single-step cosine drop = trajectory jumped to another image
        max_drop = 0.0
        drop_at = 0
        for i in range(1, n_steps):
            d = step_cos[i - 1] - step_cos[i]
            if d > max_drop:
                max_drop, drop_at = d, i
        if args.show_steps:
            print(f"\n--- Seed {s}: per-step (x = noisy latent, x0 = model prediction) ---")
            print(f"{'step':>4} {'x-cos':>8} {'x-MSE':>10} {'x0-cos':>8} {'x0-MSE':>10}")
            for i in range(n_steps):
                print(f"{i+1:>4} {step_cos[i]:>8.5f} {_mse(fxs[i], nxs[i]):>10.3e} "
                      f"{_cos(fx0s[i], nx0s[i]):>8.5f} {_mse(fx0s[i], nx0s[i]):>10.3e}")
        fin_cos = _cos(ffinal, nfinal)
        fin_mse = _mse(ffinal, nfinal)
        x0_cos = _cos(fx0s[-1], nx0s[-1]) if fx0s and nx0s else float("nan")
        if max_drop > BIFURC_DROP:
            verdict = f"bifurcated @step {drop_at}"
        elif fin_cos >= SAME_IMG_COS:
            verdict = "same-image"
        else:
            verdict = "drifted (different image)"
        final_rows.append((s, fin_cos, fin_mse, x0_cos, verdict, max_drop, drop_at))
        print(f"[seed {s}] final-cos={fin_cos:.5f}  max_step_drop={max_drop:.4f}"
              f"{' @step ' + str(drop_at) if max_drop > BIFURC_DROP else ''}  -> {verdict}")

    print("\n--- Multi-seed summary ---")
    print(f"{'seed':>8} {'final-cos':>10} {'final-mse':>12} {'max-drop':>9} {'verdict':>22}")
    for s, fc, fm, xc, v, md, da in final_rows:
        print(f"{s:>8} {fc:>10.5f} {fm:>12.3e} {md:>9.4f} {v:>22}")
    cos_vals = [r[1] for r in final_rows]
    n_bif = sum(1 for r in final_rows if "bifurcated" in r[4])
    n_diff = sum(1 for r in final_rows if r[4] != "same-image")
    print(f"\nfinal-cosine: min={min(cos_vals):.5f}  mean={sum(cos_vals)/len(cos_vals):.5f}  max={max(cos_vals):.5f}")
    print(f"same-image seeds : {len(seeds) - n_diff}/{len(seeds)}")
    print(f"bifurcated seeds : {n_bif}/{len(seeds)}   (sudden trajectory jump = different picture, not degradation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

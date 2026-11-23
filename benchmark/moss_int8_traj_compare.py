# -*- coding: utf-8 -*-
"""MOSS-Image deterministic per-step trajectory divergence (FP16 vs ConvRot INT8).

Same instrument as benchmark/flux1_convrot_int8_traj_compare.py, ported to the
MOSS-Image family: the production sampler (comfy.sample.sample) runs the FP16
source and the ConvRot INT8 pack over the same fixed seed list, and the per-step
latents are compared step by step.

Because the sampling runs in the latent space (Flux2 format, 128 channels), the
trajectory observable is the decoded-in-latent path itself: x at each step,
x0 predicted at each step, and the final latent. The verdict rules are the ones
the sibling uses:

  bifurcated @step N   - one step's cosine drops by more than 0.05
  same-image           - final cosine >= 0.98 and no step crossed the threshold
  drifted              - otherwise

PASS = mean final cosine >= 0.95 AND 0/25 bifurcated.

Usage (one line, every path double-quoted):
  python benchmark/moss_int8_traj_compare.py --fp16 "<moss_fp16>" --int8 "<moss_int8>" --clip_path "<qwen35_pack>" --comfy_path "<ComfyUI root>"
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import types

import torch


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


_DEFAULT_SEEDS = (
    "42,137,849,2024,7391,18429,53082,149206,382715,826401,1938502,4710928,"
    "8391642,15820493,36192847,71058294,128491703,285039184,491730285,762019483,"
    "938174026,1409285713,2683910547,3851729406,4195820371"
)

# ---------------------------------------------------------------------------
# ComfyUI bootstrap (same safe import order as the Flux1 traj bench)
# ---------------------------------------------------------------------------


def _clear_argv_for_comfy():
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    return saved


def _restore_argv(saved):
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
            spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            spec.submodule_search_locations = []
        else:
            spec = importlib.machinery.ModuleSpec(name, loader=None)
        mod.__spec__ = spec
        return mod

    ta = _stub_mod("torchaudio", is_package=True)
    functional = _stub_mod("torchaudio.functional")
    functional.resample = lambda waveform, orig_freq, new_freq, *a, **k: waveform

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

    transforms = _stub_mod("torchaudio.transforms")
    transforms.MelSpectrogram = _MelSpectrogram
    transforms.MelScale = _MelScale
    ta.functional = functional
    ta.transforms = transforms
    sys.modules["torchaudio"] = ta
    sys.modules["torchaudio.functional"] = functional
    sys.modules["torchaudio.transforms"] = transforms


def _install_comfy_aimdo_stub() -> None:
    aimdo = sys.modules.get("comfy_aimdo")
    if aimdo is None:
        try:
            import comfy_aimdo as aimdo
        except Exception:
            aimdo = types.ModuleType("comfy_aimdo")
            aimdo.__file__ = "<hswq_comfy_aimdo_stub>"
            aimdo.__path__ = []
            sys.modules["comfy_aimdo"] = aimdo

    if "comfy_aimdo.filter" not in sys.modules:
        try:
            import comfy_aimdo.filter  # noqa: F401
        except Exception:
            fmod = types.ModuleType("comfy_aimdo.filter")
            fmod.filter_modules = lambda *a, **k: None
            sys.modules["comfy_aimdo.filter"] = fmod
            try:
                setattr(aimdo, "filter", fmod)
            except Exception:
                pass

    if "comfy_aimdo.malloc_graph" not in sys.modules:
        try:
            import comfy_aimdo.malloc_graph  # noqa: F401
        except Exception:
            class _MallocGraph:
                def __init__(self, *args, **kwargs):
                    self._comfy_active = False
                    self._comfy_cuda_graph_modules = set()
                    self.rogue_count = 0

                def pause(self, *args, **kwargs):
                    pass

                def resume(self, *args, **kwargs):
                    pass

                def push(self, *args, **kwargs):
                    pass

                def pop(self, *args, **kwargs):
                    return False

                def abort(self, *args, **kwargs):
                    pass

                def iterate(self, *args, **kwargs):
                    return False

                def use_stream(self, *args, **kwargs):
                    class _DummyCtx:
                        def __enter__(self):
                            pass

                        def __exit__(self, *a):
                            pass

                    return _DummyCtx()

            mg_mod = types.ModuleType("comfy_aimdo.malloc_graph")
            mg_mod.MallocGraph = _MallocGraph
            mg_mod.record = lambda *args, **kwargs: _MallocGraph()
            sys.modules["comfy_aimdo.malloc_graph"] = mg_mod
            try:
                setattr(aimdo, "malloc_graph", mg_mod)
            except Exception:
                pass

    if "comfy_aimdo.model_vbar" not in sys.modules:
        try:
            import comfy_aimdo.model_vbar  # noqa: F401
        except Exception:
            vbar_mod = types.ModuleType("comfy_aimdo.model_vbar")
            vbar_mod.vbar_unpin = lambda *a, **k: None
            vbar_mod.vbar_fault = lambda *a, **k: 0
            vbar_mod.vbar_signature_compare = lambda *a, **k: False
            vbar_mod.vbars_reset_watermark_limits = lambda *a, **k: None
            vbar_mod.vbars_analyze = lambda *a, **k: 0
            vbar_mod.ModelVBAR = type("_VBAR", (), {"__init__": lambda s, *a, **k: None})
            sys.modules["comfy_aimdo.model_vbar"] = vbar_mod
            try:
                setattr(aimdo, "model_vbar", vbar_mod)
            except Exception:
                pass


def setup_comfy(comfy_path: str) -> None:
    comfy_root = os.path.abspath(comfy_path)
    if not os.path.isdir(comfy_root):
        raise FileNotFoundError(f"--comfy_path not found: {comfy_root}")
    if not os.path.isdir(os.path.join(comfy_root, "comfy")):
        raise FileNotFoundError(f"comfy/ package missing under: {comfy_root}")
    sys.path = [comfy_root] + [p for p in sys.path if os.path.abspath(p or ".") != comfy_root]
    _install_torchaudio_stub()
    _install_comfy_aimdo_stub()

    import comfy.options

    comfy.options.enable_args_parsing(False)

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


def apply_int8_patches() -> None:
    _bench_dir = os.path.dirname(os.path.abspath(__file__))
    if _bench_dir not in sys.path:
        sys.path.insert(0, _bench_dir)

    import comfy.ops

    from int8.comfy_quant_int8 import apply_comfy_quant_int8_patches
    import int8.comfy_quant_int8 as _cq_int8


def _hard_free_vram() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _load_diffusion_model(unet_path: str):
    """Load a MOSS-Image DiT (fp16 source or ConvRot INT8 pack) as ModelPatcher."""
    import comfy.sd

    saved = _clear_argv_for_comfy()
    try:
        print(f"[load] {os.path.basename(unet_path)}", flush=True)
        patcher = comfy.sd.load_diffusion_model(os.path.abspath(unet_path), {})
        if patcher is None:
            raise RuntimeError(f"load_diffusion_model returned None for {unet_path}")
        print("[load] done", flush=True)
        return patcher
    finally:
        _restore_argv(saved)


def encode_prompt(clip, prompt: str):
    tokens = clip.tokenize(prompt)
    return clip.encode_from_tokens_scheduled(tokens)


def load_moss_clip_and_encode(clip_path: str, prompt: str, negative: str):
    """MOSS-Image text encoder = Qwen3.5-9B (CLIPType.MOSS_IMAGE).

    Loads on CPU, encodes, unloads - the same contract the converter's Card 1
    calibration uses, measured to work on the hswq ConvRot INT8 CLIP pack.
    """
    import comfy.model_management as mm
    import comfy.sd

    _cpu = torch.device("cpu")
    saved = _clear_argv_for_comfy()
    try:
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=None,
            clip_type=comfy.sd.CLIPType.MOSS_IMAGE,
            model_options={"load_device": _cpu, "offload_device": _cpu, "initial_device": _cpu},
        )
    finally:
        _restore_argv(saved)
    positive = encode_prompt(clip, prompt)
    negative_cond = encode_prompt(clip, negative)
    for conds in (positive, negative_cond):
        for t in range(len(conds)):
            for i in range(len(conds[t])):
                if isinstance(conds[t][i], torch.Tensor):
                    conds[t][i] = conds[t][i].cpu()
    if getattr(clip, "cond_stage_model", None) is not None:
        clip.cond_stage_model.cpu()
    if getattr(clip, "patcher", None) is not None:
        mm.unload_model_and_clones(clip.patcher)
    del clip
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  [Offload] CLIP (Qwen3.5-9B) unloaded.", flush=True)
    return positive, negative_cond


def make_empty_latent(model, width: int, height: int, batch: int = 1) -> dict:
    import comfy.model_management as mm
    import comfy.sample as comfy_sample

    latent = torch.zeros(
        [batch, 128, height // 16, width // 16],
        device=mm.intermediate_device(),
        dtype=torch.float32,
    )
    return comfy_sample.fix_empty_latent_channels(model, latent)


def run_trajectory(model, positive, negative, latent, *, seed, steps, cfg,
                   sampler_name, scheduler, attention_mask_pos=None,
                   attention_mask_neg=None):
    """Production sampler (comfy.sample.sample) with per-step latent capture.

    The MOSS-Image BaseModel consumes an additive logit bias on the text tokens
    through `text_embedding_mask` (measured in model_base.extra_conds); when the
    CLIP supplies one, it is passed through so the sampled trajectory is the
    production trajectory.
    """
    import comfy.sample as comfy_sample

    noise = comfy_sample.prepare_noise(latent, seed, None)
    xs = []
    x0s = []

    def cb(step, x0, x, total_steps):
        xs.append(x.detach().float().cpu())
        x0s.append(x0.detach().float().cpu())

    kwargs = {}
    if attention_mask_pos is not None:
        kwargs["attention_mask"] = attention_mask_pos
    if attention_mask_neg is not None:
        # one mask per conditioning run: sample() evaluates cond and uncond
        # separately, so a single global mask would mis-assign it; measured
        # behaviour is that both prompts use their own token lengths.
        pass

    with torch.no_grad():
        comfy_sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, latent, denoise=1.0,
            disable_noise=False, start_step=None, last_step=None,
            force_full_denoise=False, noise_mask=None,
            callback=cb, disable_pbar=False, seed=seed,
            **kwargs,
        )
    return xs, x0s


def _cos(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    na, nb = float(a.norm()), float(b.norm())
    if na <= 0 or nb <= 0:
        return 1.0 if na == nb else 0.0
    return float(torch.dot(a, b) / (na * nb))


def _mse(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


def parse_args():
    ap = argparse.ArgumentParser(
        description="MOSS-Image deterministic per-step trajectory divergence (FP16 vs ConvRot INT8)"
    )
    ap.add_argument("--fp16", required=True, help="FP16/BF16 MOSS-Image DiT path")
    ap.add_argument("--int8", "--quant", dest="int8_path", required=True,
                    help="ConvRot INT8 quantized MOSS-Image DiT path")
    ap.add_argument("--clip_path", required=True,
                    help="MOSS-Image text encoder (Qwen3.5-9B; the hswq CLIP pack loads on the stock path)")
    ap.add_argument("--comfy_path", required=True, help="ComfyUI root")
    ap.add_argument("--prompt",
                    default="masterpiece, best quality, 1girl, solo, standing, simple background")
    ap.add_argument("--negative", default="")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument(
        "--seeds",
        default=_DEFAULT_SEEDS,
        help="comma-separated seeds; same seed = identical noise for both models",
    )
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--show-steps", action="store_true",
                    help="print the per-step divergence curve (default: only final per seed)")
    ap.add_argument("--out", default=None, help="optional json output path")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        print("No seeds given")
        return 1

    # --- PROTOCOL block: the conditions of this run, printed before any load ---
    print("=" * 72)
    print("PROTOCOL")
    print("=" * 72)
    print(f"  fp16        : {args.fp16}")
    print(f"  int8        : {args.int8_path}")
    print(f"  clip_path   : {args.clip_path}")
    print(f"  comfy_path  : {args.comfy_path}")
    print(f"  seeds({len(seeds)})  : {','.join(str(s) for s in seeds)}")
    print(f"  steps       : {args.steps}")
    print(f"  resolution  : {args.width}x{args.height}")
    print(f"  cfg         : {args.cfg}")
    print(f"  sampler     : {args.sampler}")
    print(f"  scheduler   : {args.scheduler}")
    print(f"  prompt      : {args.prompt[:80]}")
    print("=" * 72)

    setup_comfy(args.comfy_path)
    apply_int8_patches()

    import time

    import comfy.model_management as mm

    mm.get_torch_device()

    fp16_runs = {}
    int8_runs = {}
    times_fp16 = []
    times_int8 = []
    saved_argv = _clear_argv_for_comfy()
    try:
        # --- CLIP: Qwen3.5-9B on CPU, encoded, unloaded ---
        print("Loading CLIP on CPU (MOSS-Image / Qwen3.5-9B)...")
        positive, negative = load_moss_clip_and_encode(
            args.clip_path, args.prompt, args.negative
        )

        # --- FP16 baseline ---
        print("[FP16] loading MOSS-Image DiT...")
        fp16 = _load_diffusion_model(args.fp16)
        latent = make_empty_latent(fp16, args.width, args.height, batch=1)
        for s in seeds:
            print(f"[FP16] seed {s}")
            t0 = time.perf_counter()
            xs, x0s = run_trajectory(
                fp16, positive, negative, latent, seed=s, steps=args.steps,
                cfg=args.cfg, sampler_name=args.sampler, scheduler=args.scheduler,
            )
            times_fp16.append(time.perf_counter() - t0)
            fp16_runs[s] = (xs, x0s, xs[-1].clone())
        del fp16
        _hard_free_vram()
        print("  [Offload] FP16 DiT unloaded.")

        # --- ConvRot INT8 (patches already applied) ---
        print("[INT8] loading MOSS-Image DiT...")
        q = _load_diffusion_model(args.int8_path)
        for s in seeds:
            print(f"[INT8] seed {s}")
            t0 = time.perf_counter()
            xs, x0s = run_trajectory(
                q, positive, negative, latent, seed=s, steps=args.steps,
                cfg=args.cfg, sampler_name=args.sampler, scheduler=args.scheduler,
            )
            times_int8.append(time.perf_counter() - t0)
            int8_runs[s] = (xs, x0s, xs[-1].clone())
        del q
        _hard_free_vram()
    finally:
        _restore_argv(saved_argv)

    # --- compare ---
    print("\n" + "=" * 72)
    print("Deterministic per-step latent trajectory divergence (FP16 vs ConvRot INT8)")
    print("=" * 72)
    BIFURC_DROP = 0.05
    SAME_IMG_COS = 0.98
    final_rows = []
    for s in seeds:
        fxs, fx0s, ffinal = fp16_runs[s]
        nxs, nx0s, nfinal = int8_runs[s]
        n_steps = min(len(fxs), len(nxs))
        step_cos = [_cos(fxs[i], nxs[i]) for i in range(n_steps)]
        max_drop = 0.0
        drop_at = 0
        for i in range(1, n_steps):
            d = step_cos[i - 1] - step_cos[i]
            if d > max_drop:
                max_drop, drop_at = d, i
        if args.show_steps:
            print(f"\n--- Seed {s}: per-step ---")
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
    passed = (sum(cos_vals) / len(cos_vals) >= 0.95) and (n_bif == 0)
    print(f"\nGATE: mean >= 0.95 AND 0/{len(seeds)} bifurcated -> {'PASS' if passed else 'FAIL'}")

    if times_fp16 and times_int8:
        avg_f = sum(times_fp16) / len(times_fp16)
        avg_i = sum(times_int8) / len(times_int8)
        print(f"\navg wall/seed: FP16 {avg_f:.2f}s | INT8 {avg_i:.2f}s "
              f"({(avg_f - avg_i) / avg_f * 100:+.1f}%)")

    if args.out:
        payload = {
            "protocol": {
                "fp16": args.fp16, "int8": args.int8_path, "clip_path": args.clip_path,
                "comfy_path": args.comfy_path, "steps": args.steps,
                "width": args.width, "height": args.height, "cfg": args.cfg,
                "sampler": args.sampler, "scheduler": args.scheduler,
                "seeds": seeds,
            },
            "rows": [
                {"seed": s, "final_cos": fc, "final_mse": fm, "x0_cos": xc,
                 "verdict": v, "max_drop": md, "drop_at": da}
                for s, fc, fm, xc, v, md, da in final_rows
            ],
            "summary": {
                "cos_min": min(cos_vals), "cos_mean": sum(cos_vals) / len(cos_vals),
                "cos_max": max(cos_vals), "same_image": len(seeds) - n_diff,
                "bifurcated": n_bif, "n": len(seeds), "pass": passed,
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

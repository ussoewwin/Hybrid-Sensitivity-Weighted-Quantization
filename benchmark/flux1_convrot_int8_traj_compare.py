#!/usr/bin/env python3
"""Flux1 ConvRot INT8 deterministic trajectory-divergence comparator (traj style).

The Flux1-port of benchmark/sdxl_int8_traj_compare.py (same per-step callback
capture, same verdict thresholds, same report shape), using the Flux1 mechanics
from benchmark/flux1_nvfp4/flux_traj_compare.py:

  * Flux1 loads through ``comfy.sd.load_diffusion_model`` (standalone DiT file),
    the FP16 baseline always loads unmodified, the INT8 arm wraps the load in
    the INT8 Conv2d inject scope only when the checkpoint carries comfy_quant
    int8 stamps.
  * Flux1 needs TWO text encoders (t5xxl + clip_l) in one ``load_clip`` call
    with ``CLIPType.FLUX``; both are loaded on CPU, encoded, and unloaded
    before the DiT runs so the whole GPU belongs to the diffusion model.
  * 16-channel latent (Flux), cfg default 1.0 (guidance comes from
    extra_conds), euler/simple sampler defaults.
  * same seed -> identical noise for both models; verdicts bifurcated /
    same-image / drifted (same thresholds as the other traj benches).

INT8 kernels come from ``benchmark/int8/comfy_quant_int8.py`` (vendored), the
same stack the SDXL INT8 traj bench runs on ComfyUI-master. No NVFP4 patches
and no bench-side CUDA-graph executor are armed here - this bench compares the
ConvRot INT8 pack against its FP16 source on the plain eager path.

Usage:
    python benchmark/flux1_convrot_int8_traj_compare.py \\
        --fp16 <flux_fp16.safetensors> --int8 <flux_native_convrot_int8.safetensors> \\
        --clip_path <t5xxl.safetensors> --clip_l_path <clip_l.safetensors> \\
        --comfy_path <ComfyUI-master> \\
        [--seeds "<official 25>"] [--steps 12] [--prompt "..."] \\
        [--width 1024] [--height 1024] [--cfg 1.0] \\
        [--sampler euler] [--scheduler simple] [--show-steps]
"""
import argparse
import gc
import os
import sys

import torch

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from pathlib import Path  # noqa: E402


# ---------------------------------------------------------------------------
# ComfyUI bootstrap (same stubs as benchmark/sdxl_int8_traj_compare.py)
# ---------------------------------------------------------------------------

def _install_torchaudio_stub() -> None:
    """comfy.sd pulls audio VAE code that hard-imports torchaudio; the stub must
    carry a __spec__ so transformers' find_spec probe does not raise."""
    try:
        import torchaudio  # noqa: F401
        return
    except Exception:
        pass
    import importlib.machinery
    import types

    class _StubModule(types.ModuleType):
        def __init__(self, name, *, is_package=False):
            super().__init__(name)
            if is_package:
                self.__path__ = []

        def __getattr__(self, key):
            if key.startswith("__"):
                raise AttributeError(key)

            def _fn(*args, **kwargs):
                raise NotImplementedError(f"{self.__name__}.{key} stubbed")

            return _fn

    def _stub_mod(name, *, is_package=False):
        if name in sys.modules:
            return
        mod = _StubModule(name, is_package=is_package)
        mod.__spec__ = importlib.machinery.ModuleSpec(
            name, loader=None, is_package=is_package
        )
        sys.modules[name] = mod
        parent, _, child = name.rpartition(".")
        if parent:
            parent_mod = sys.modules.get(parent) or _stub_mod(parent, is_package=True)
            setattr(parent_mod, child, mod)

    _stub_mod("torchaudio", is_package=True)
    _stub_mod("torchaudio.functional")
    _stub_mod("torchaudio.transforms")


def _install_comfy_aimdo_stub() -> None:
    """Provide the comfy_aimdo surface ComfyUI-master's prefetch code imports."""
    import types

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
    bench_dir = Path(__file__).resolve().parent
    repo_dir = bench_dir.parent
    master = repo_dir / "ComfyUI-master"

    if comfy_path and os.path.isdir(comfy_path) and os.path.isdir(os.path.join(comfy_path, "comfy")):
        comfy_root = Path(comfy_path).resolve()
    elif master.is_dir() and (master / "comfy").is_dir():
        comfy_root = master.resolve()
    else:
        raise FileNotFoundError(f"ComfyUI-master not found at {master}")

    new_sys_path = []
    for p in (str(comfy_root), str(bench_dir), str(repo_dir)):
        if p not in new_sys_path and os.path.isdir(p):
            new_sys_path.append(p)
    for p in sys.path:
        if p not in new_sys_path:
            new_sys_path.append(p)
    sys.path = new_sys_path

    _install_torchaudio_stub()
    _install_comfy_aimdo_stub()

    import comfy.options

    comfy.options.enable_args_parsing(False)

    try:
        import psutil  # noqa: F401
    except Exception:
        import types

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


def _clear_argv_for_comfy() -> list:
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    return saved


def _restore_argv(saved: list) -> None:
    sys.argv = saved


def _hard_free_vram() -> None:
    import comfy.model_management as mm

    gc.collect()
    try:
        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ---------------------------------------------------------------------------
# INT8 patches (vendored bench stack, same as the SDXL INT8 traj bench)
# ---------------------------------------------------------------------------

def apply_int8_patches() -> None:
    import comfy.ops

    from int8.comfy_quant_int8 import apply_comfy_quant_int8_patches
    import int8.comfy_quant_int8 as _cq_int8

    apply_comfy_quant_int8_patches()
    print(f"  [BENCH] int8_tensorwise: {'int8_tensorwise' in comfy.ops.QUANT_ALGOS}")
    print(f"  [BENCH] comfy_quant_int8 patched: {_cq_int8._PATCHES_APPLIED}")
    print(f"  [BENCH] patch file: {os.path.abspath(_cq_int8.__file__)}")
    if not _cq_int8._PATCHES_APPLIED:
        raise RuntimeError(
            "comfy_quant_int8 patches failed to apply "
            "(need [BENCH] comfy_quant_int8 patched: True)"
        )


def _load_diffusion_model(unet_path: str):
    """Load the Flux1 DiT; wrap the INT8 pack load in the INT8 inject scope.

    The FP16 baseline never carries comfy_quant int8 stamps, so it loads
    unmodified through the plain path."""
    import comfy.sd

    from int8.comfy_quant_int8 import (
        _int8_quant_conv_scope,
        checkpoint_looks_like_comfy_quant_int8,
    )

    use_int8_scope = checkpoint_looks_like_comfy_quant_int8(unet_path)
    print(f"  [BENCH] INT8 load scope: {use_int8_scope}")
    if use_int8_scope:
        with _int8_quant_conv_scope():
            return comfy.sd.load_diffusion_model(unet_path, {})
    return comfy.sd.load_diffusion_model(unet_path, {})


# ---------------------------------------------------------------------------
# Flux1 CLIP (two text encoders: t5xxl + clip_l, loaded on CPU then unloaded)
# ---------------------------------------------------------------------------

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


def load_flux_clip_and_encode(clip_path: str, clip_l_path: str,
                              prompt: str, negative: str):
    """Load t5xxl + clip_l in ONE load_clip call (CLIPType.FLUX), encode both
    prompts, then move everything to CPU and free the patcher."""
    import comfy.model_management as mm
    import comfy.sd

    _cpu = torch.device("cpu")
    # CPU offload for BOTH encoders: same device contract as ComfyUI-master's
    # CLIPLoader with device=cpu (nodes.py: load_device = offload_device = cpu),
    # plus initial_device=cpu so no tensor is staged on the GPU during the load.
    clip = comfy.sd.load_clip(
        ckpt_paths=[clip_l_path, clip_path],
        embedding_directory=None,
        clip_type=comfy.sd.CLIPType.FLUX,
        model_options={"load_device": _cpu, "offload_device": _cpu,
                       "initial_device": _cpu},
    )
    patcher = getattr(clip, "patcher", None)
    if patcher is not None:
        print(f"  [CLIP] patcher load_device={patcher.load_device} "
              f"offload_device={patcher.offload_device}")
    model = getattr(patcher, "model", None) if patcher is not None else None
    if model is not None:
        devs = sorted({str(p.device) for p in model.parameters()})
        print(f"  [CLIP] parameter devices before encode: {devs}")
        if any(d != "cpu" for d in devs):
            try:
                model.to(_cpu)
                devs = sorted({str(p.device) for p in model.parameters()})
                print(f"  [CLIP] moved to CPU; parameter devices: {devs}")
            except Exception as exc:  # noqa: BLE001
                print(f"  [CLIP] WARNING: explicit CPU move failed: {exc}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"  [CLIP] VRAM before encode: used={(total_b - free_b) / 1024**2:.0f} MiB / {total_b / 1024**2:.0f} MiB")
    positive = encode_prompt(clip, prompt)
    negative = encode_prompt(clip, negative)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"  [CLIP] VRAM after encode: used={(total_b - free_b) / 1024**2:.0f} MiB / {total_b / 1024**2:.0f} MiB")
    for conds in (positive, negative):
        for t in range(len(conds)):
            for i in range(len(conds[t])):
                if isinstance(conds[t][i], torch.Tensor):
                    conds[t][i] = conds[t][i].cpu()
    if getattr(clip, "cond_stage_model", None) is not None:
        clip.cond_stage_model.cpu()
    if getattr(clip, "patcher", None) is not None:
        mm.unload_model_and_clones(clip.patcher)
    del clip
    _hard_free_vram()
    print("  [Offload] CLIP (t5xxl + clip_l) unloaded.")
    return positive, negative


# ---------------------------------------------------------------------------
# Trajectory capture (same callbacks as the other traj benches)
# ---------------------------------------------------------------------------

def run_trajectory(model, positive, negative, latent, *, seed, steps, cfg,
                   sampler_name, scheduler):
    """Full denoising; returns (per_step_x, per_step_x0, final_sample)."""
    import comfy.sample as comfy_sample
    import comfy.utils

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Flux1 deterministic per-step trajectory divergence (FP16 vs ConvRot INT8)"
    )
    ap.add_argument("--fp16", required=True, help="FP16/BF16 Flux1 DiT path")
    ap.add_argument("--int8", "--fp8", "--quant", dest="int8_path", required=True,
                    help="ConvRot INT8 quantized Flux1 DiT path")
    ap.add_argument("--clip_path", required=True, help="T5XXL text encoder path")
    ap.add_argument("--clip_l_path", required=True, help="clip_l text encoder path")
    ap.add_argument("--comfy_path", required=True, help="ComfyUI root")
    ap.add_argument("--prompt",
                    default="masterpiece, best quality, 1girl, solo, standing, simple background")
    ap.add_argument("--negative", default="")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument(
        "--seeds",
        default=(
            "42,137,849,2024,7391,18429,53082,149206,382715,826401,1938502,4710928,"
            "8391642,15820493,36192847,71058294,128491703,285039184,491730285,762019483,"
            "938174026,1409285713,2683910547,3851729406,4195820371"
        ),
        help="comma-separated seeds; same seed = identical noise for both models",
    )
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--cfg", type=float, default=1.0, help="Flux default 1.0 (guidance from extra_conds)")
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--show-steps", action="store_true",
                    help="print the per-step divergence curve (default: only final per seed)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        print("No seeds given")
        return 1

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
        # --- CLIP: two text encoders on CPU, encoded, unloaded ---
        print("Loading CLIP on CPU (Flux / clip_l + t5xxl)...")
        positive, negative = load_flux_clip_and_encode(
            args.clip_path, args.clip_l_path, args.prompt, args.negative
        )

        # --- FP16 baseline ---
        print("[FP16] loading Flux1 DiT...")
        fp16 = _load_diffusion_model(args.fp16)
        latent = make_empty_latent(fp16, args.width, args.height, batch=1)
        for s in seeds:
            print(f"[FP16] seed {s}")
            t0 = time.perf_counter()
            xs, x0s, final = run_trajectory(
                fp16, positive, negative, latent, seed=s, steps=args.steps,
                cfg=args.cfg, sampler_name=args.sampler, scheduler=args.scheduler,
            )
            times_fp16.append(time.perf_counter() - t0)
            fp16_runs[s] = (xs, x0s, final.detach().float().cpu())
        del fp16
        _hard_free_vram()
        print("  [Offload] FP16 DiT unloaded.")

        # --- ConvRot INT8 (patches already applied) ---
        print("[INT8] loading Flux1 DiT...")
        q = _load_diffusion_model(args.int8_path)
        for s in seeds:
            print(f"[INT8] seed {s}")
            t0 = time.perf_counter()
            xs, x0s, final = run_trajectory(
                q, positive, negative, latent, seed=s, steps=args.steps,
                cfg=args.cfg, sampler_name=args.sampler, scheduler=args.scheduler,
            )
            times_int8.append(time.perf_counter() - t0)
            int8_runs[s] = (xs, x0s, final.detach().float().cpu())
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

    if times_fp16 and times_int8:
        avg_f = sum(times_fp16) / len(times_fp16)
        avg_i = sum(times_int8) / len(times_int8)
        print(f"\navg wall/seed: FP16 {avg_f:.2f}s | INT8 {avg_i:.2f}s "
              f"({(avg_f - avg_i) / avg_f * 100:+.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

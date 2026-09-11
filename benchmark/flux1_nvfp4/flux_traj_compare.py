#!/usr/bin/env python3
"""Flux1 deterministic trajectory-divergence comparator (FP16 vs ConvRot INT8 / Hybrid NVFP4).

zi_traj_compare.py の Flux1 移植版。
Per-step latent trajectory comparison between BF16/FP16 and quantized (INT8 / NVFP4) models.
Reuses the new-gen bench (benchmark/flux1_nvfp4/flux_int8_bench.py: NVFP4 runtime + INT8 parity)
for ComfyUI bootstrap / model loading / prompt encoding.

Usage:
    python benchmark/flux1_nvfp4/flux_traj_compare.py \
        --fp16 <bf16.safetensors> --fp8 <quantized.safetensors> \
        --clip_path <t5xxl.safetensors> --clip_l_path <clip_l.safetensors> \
        --comfy_path <ComfyUI> \
        [--vae <ae.safetensors>] \
        [--seeds "42,137,5517,92048,371506,..."] [--steps 12]
"""
import argparse
import os
import sys

import torch
import gc
from pathlib import Path

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
_BENCH_PARENT = os.path.dirname(_BENCH_DIR)
if _BENCH_PARENT not in sys.path:
    sys.path.insert(0, _BENCH_PARENT)


_DEFAULT_SEEDS = (
    "42,137,5517,92048,371506,5293047,64820153,731509284,8426170395,9517038246,"
    "210987,6543210,98765432,1357924680,2468135791,3579246812,4680357923,"
    "5791468034,6802579145,7913680256"
)


# --- inlined helper definitions (no external bench import) ---

def _clear_argv_for_comfy() -> list[str]:
    """ComfyUI cli_args swallows unknown flags; keep only argv[0] during import."""
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    return saved


def _hard_free_vram() -> None:
    import comfy.model_management as mm

    mm.unload_all_models()
    mm.soft_empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _install_torchaudio_stub() -> None:
    """Prevent real torchaudio from loading if comfy.sd is pulled in."""
    import importlib.machinery
    import types

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

    def _resample(waveform, orig_freq, new_freq, *args, **kwargs):
        return waveform

    functional.resample = _resample

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


def _load_diffusion_model(unet_path: str):
    """Load DiT; wrap INT8 comfy_quant checkpoints in Conv2d inject scope."""
    import comfy.sd
    from int8.comfy_quant_int8 import (
        _int8_quant_conv_scope,
        checkpoint_looks_like_comfy_quant_int8,
    )
    from flux1_nvfp4.comfy_quant_nvfp4 import checkpoint_looks_like_comfy_quant_nvfp4

    looks_nvfp4 = checkpoint_looks_like_comfy_quant_nvfp4(unet_path)
    use_int8_scope = checkpoint_looks_like_comfy_quant_int8(unet_path)
    print(f"  [BENCH] NVFP4 comfy_quant detect: {looks_nvfp4}")
    print(f"  [BENCH] INT8 Conv2d load scope: {use_int8_scope}")
    if use_int8_scope:
        with _int8_quant_conv_scope():
            return comfy.sd.load_diffusion_model(unet_path, {})
    return comfy.sd.load_diffusion_model(unet_path, {})


def _restore_argv(saved: list[str]) -> None:
    sys.argv = saved


def apply_quant_patches() -> None:
    """NVFP4 + INT8 comfy_quant monkey-patch（krea2_convrot_nvfp4_bench 相当）."""
    import comfy.ops

    from flux1_nvfp4.comfy_quant_nvfp4 import apply_comfy_quant_nvfp4_patches
    from flux1_nvfp4.nvfp4_comfy_parity import apply_nvfp4_comfy_parity
    import flux1_nvfp4.comfy_quant_nvfp4 as _cq_nvfp4

    from int8.comfy_quant_int8 import apply_comfy_quant_int8_patches
    import int8.comfy_quant_int8 as _cq_int8

    apply_comfy_quant_nvfp4_patches()
    if not apply_nvfp4_comfy_parity():
        raise RuntimeError("flux1_nvfp4 ComfyUI-only parity failed to apply")
    print(f"  [BENCH] nvfp4 patch file: {os.path.abspath(_cq_nvfp4.__file__)}")
    print(f"  [BENCH] comfy_quant_nvfp4 patched: {_cq_nvfp4._PATCHES_APPLIED}")

    apply_comfy_quant_int8_patches()
    print(f"  [BENCH] int8_tensorwise: {'int8_tensorwise' in comfy.ops.QUANT_ALGOS}")
    print(f"  [BENCH] comfy_quant_int8 patched: {_cq_int8._PATCHES_APPLIED}")
    if not _cq_int8._PATCHES_APPLIED:
        raise RuntimeError("comfy_quant_int8 patches failed to apply")


def encode_prompt(clip, prompt: str):
    """Flux CLIP encode.

    注意: 現行 ComfyUI では encode_from_tokens_scheduled は Tensor のリストを返し、
    cond dict に guidance を注入できない。Flux の guidance は extra_conds の
    デフォルト 3.5 が自動適用される（fp16/int8 比較には影響なし）。
    """
    tokens = clip.tokenize(prompt)
    return clip.encode_from_tokens_scheduled(tokens)


def make_empty_latent(model, width: int, height: int, batch: int = 1) -> dict:
    import comfy.model_management as mm
    import comfy.sample as comfy_sample

    device = mm.intermediate_device()
    latent = torch.zeros([batch, 16, height // 8, width // 8], device=device)
    latent = comfy_sample.fix_empty_latent_channels(model, latent)
    return {"samples": latent}


def set_hf_token(token: str | None) -> None:
    if not token:
        return
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def setup_comfy(comfy_path: str) -> None:
    comfy_root = Path(comfy_path).resolve()
    if not comfy_root.is_dir():
        raise FileNotFoundError(f"--comfy_path not found: {comfy_root}")
    sys.path = [str(comfy_root)] + [p for p in sys.path if Path(p).resolve() != comfy_root]

    _install_torchaudio_stub()

    # NVFP4 ConvRot runtime: prebind kitchen tensor exports before comfy.quant_ops import
    from flux1_nvfp4.kitchen_quant_ops_repair import (
        ensure_kitchen_quant_ops,
        prebind_missing_kitchen_tensor_exports,
    )

    prebind_missing_kitchen_tensor_exports()

    import comfy.options

    comfy.options.enable_args_parsing(False)

    try:
        import comfy_aimdo  # noqa: F401
    except Exception:
        import types

        m = types.ModuleType("comfy_aimdo")
        m.__file__ = "<stub>"
        m.__path__ = []
        sys.modules["comfy_aimdo"] = m
        sys.modules["comfy_aimdo.filter"] = types.ModuleType("comfy_aimdo.filter")
        sys.modules["comfy_aimdo.filter"].filter_modules = lambda *a, **k: None

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

    # Resolve quant_ops now (after prebind) and apply Branch A/B before model load.
    import comfy.quant_ops  # noqa: F401

    ensure_kitchen_quant_ops()


def run_trajectory(model, positive, negative, latent, *, seed, steps, cfg,
                   sampler_name, scheduler):
    """Run full denoising; return (per_step_x, per_step_x0, final_sample)."""
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


def parse_args():
    ap = argparse.ArgumentParser(
        description="Flux1 deterministic per-step trajectory divergence (FP16 vs quantized)"
    )
    ap.add_argument("--fp16", required=True, help="BF16/FP16 Flux1 model path")
    ap.add_argument("--fp8", required=True, help="quantized model path (ConvRot INT8 / Hybrid NVFP4)")
    ap.add_argument("--clip_path", required=True, help="T5XXL text encoder path")
    ap.add_argument("--clip_l_path", required=True, help="clip_l text encoder path")
    ap.add_argument("--comfy_path", required=True, help="ComfyUI root")
    ap.add_argument("--token", default=None)
    ap.add_argument("--prompt", default="masterpiece, best quality, 1girl, solo, standing, simple background")
    ap.add_argument("--negative", default="")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seeds", default=_DEFAULT_SEEDS,
                    help="comma-separated seeds; same seed = identical noise for both models")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--cfg", type=float, default=1.0, help="Flux default 1.0 (guidance=3.5 fixed)")
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--show-steps", action="store_true",
                    help="print the per-step divergence curve (default: only final per seed)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    set_hf_token(args.token)

    saved_argv = _clear_argv_for_comfy()
    try:
        setup_comfy(args.comfy_path)
        apply_quant_patches()

        import comfy.model_management as mm
        import comfy.sd

        mm.get_torch_device()

        _cpu = torch.device("cpu")
        print("Loading CLIP on CPU (Flux / clip_l + t5xxl)...")
        clip = comfy.sd.load_clip(
            ckpt_paths=[args.clip_l_path, args.clip_path],
            embedding_directory=None,
            clip_type=comfy.sd.CLIPType.FLUX,
            model_options={"load_device": _cpu, "offload_device": _cpu, "initial_device": _cpu},
        )
        positive = encode_prompt(clip, args.prompt)
        negative = encode_prompt(clip, args.negative)
        # エンコード済み cond を CPU へ（VRAM を DiT に全解放）
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
        print("  [Offload] CLIP unloaded.")

        # --- FP16 ---
        fp16 = _load_diffusion_model(args.fp16)
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

        # --- Quantized (patches already applied) ---
        q = _load_diffusion_model(args.fp8)
        q_runs = {}
        for s in seeds:
            print(f"[QUANT] seed {s}")
            xs, x0s, final = run_trajectory(
                q, positive, negative, latent, seed=s, steps=args.steps,
                cfg=args.cfg, sampler_name=args.sampler, scheduler=args.scheduler,
            )
            q_runs[s] = (xs, x0s, final.detach().float().cpu())
        del q
        _hard_free_vram()
    finally:
        _restore_argv(saved_argv)

    # --- compare ---
    print("\n" + "=" * 72)
    print("Deterministic per-step latent trajectory divergence (FP16 vs Quantized)")
    print("=" * 72)
    BIFURC_DROP = 0.05
    SAME_IMG_COS = 0.98
    final_rows = []
    for s in seeds:
        fxs, fx0s, ffinal = fp16_runs[s]
        nxs, nx0s, nfinal = q_runs[s]
        n_steps = min(len(fxs), len(nxs))
        step_cos = [_cos(fxs[i], nxs[i]) for i in range(n_steps)]
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

# -*- coding: utf-8 -*-
"""Flux1 per-layer trajectory impact: inject ONE layer's NVFP4 error, run N steps, measure x divergence.

Reverse hybrid NVFP4 method（ZI の Z_Image/diag_impact.py を Flux1 に移植）:
1. diag_impact.py         -> impact_<model>.json（層ごとの relative MSE、昇順 = NVFP4 化しても安全）
2. gen_reverse_nvfp4.py   -> hybrid nv{K} アーティファクト（低影響層を INT8 → NVFP4 に reverse 変換）
3. benchmark/flux1_nvfp4/flux1_convrot_nvfp4_bench.py -> SSIM チェック

Usage:
    python Flux1/diag_impact.py <base_model.safetensors> <all_int8_artifact.safetensors> <impact_out.json> \
        [--comfy-path <comfyui-root>] [--repo-root <repo-root>] [--steps 4] [--seed 42]
"""
import argparse
import json
import os
import sys

import torch
from pathlib import Path


# --- inlined helper definitions (no external bench import) ---

# --- inlined helper definitions (no external bench import) ---

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


def parse_args():
    ap = argparse.ArgumentParser(description="Flux1 per-layer NVFP4 trajectory impact")
    ap.add_argument("base", help="baseline fp16/bf16 Flux1 safetensors")
    ap.add_argument("artifact", help="all-INT8 ConvRot safetensors (layer list source)")
    ap.add_argument("out", help="output impact json path")
    ap.add_argument("--comfy-path", default=None, help="ComfyUI root path")
    ap.add_argument("--repo-root", default=None, help="repo root containing benchmark/; default = parent of this script dir")
    ap.add_argument("--steps", type=int, default=4, help="trajectory denoising steps (default 4)")
    ap.add_argument("--seed", type=int, default=42, help="trajectory seed (default 42)")
    ap.add_argument("--latent-size", type=int, default=64, help="latent H/W (default 64; 16GB VRAM 推奨)")
    return ap.parse_args()


def nvfp4_quant_error(w):
    """TRUE NVFP4 quantization error via comfy_kitchen roundtrip
    (E2M1 x 16-element blocks + global scale)."""
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout as _NVFP4
    w2 = w if w.is_contiguous() else w.contiguous()
    qdata, params = _NVFP4.quantize(w2)
    return _NVFP4.dequantize(qdata, params)


def rel_mse(a, b):
    a = a.float().reshape(a.shape[0], -1)
    b = b.float().reshape(b.shape[0], -1)
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


def main():
    a = parse_args()
    device = "cuda"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    repo = os.path.abspath(a.repo_root) if a.repo_root else os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    sys.path.insert(0, os.path.join(repo, "benchmark"))
    sys.path.insert(0, repo)

    comfy_path = a.comfy_path
    if not comfy_path:
        comfy_path = os.path.join(repo, "ComfyUI-master")
    if not os.path.isabs(comfy_path):
        joined = os.path.join(repo, comfy_path)
        if os.path.isdir(joined):
            comfy_path = os.path.abspath(joined)
        else:
            comfy_path = os.path.abspath(comfy_path)
    else:
        comfy_path = os.path.abspath(comfy_path)
    setup_comfy(comfy_path)
    apply_quant_patches()

    a.base = a.base.strip()
    a.artifact = a.artifact.strip()
    a.out = a.out.strip()

    model = _load_diffusion_model(a.base)  # ModelPatcher
    dm = model.model.diffusion_model
    dm.eval()

    # 低 VRAM モードでロード（bf16 24GB は VRAM 16GB に載らないため）
    import comfy.model_management as mm

    mm.load_models_gpu([model], lowvram=True)

    # flux の Linear (weight + in_features) モジュール一覧
    mods = {n: m for n, m in dm.named_modules()
            if hasattr(m, "weight") and hasattr(m, "in_features")}
    print(f"modules with weight/in_features: {len(mods)}", flush=True)

    from safetensors import safe_open
    artifact = os.path.abspath(a.artifact)
    if not os.path.isfile(artifact):
        raise FileNotFoundError(f"artifact not found: {a.artifact!r}")
    with safe_open(artifact, framework="pt", device="cpu") as f:
        meta = json.loads(f.metadata()["_quantization_metadata"])
        layers = list(meta["layers"].keys())
    print(f"layers to measure: {len(layers)}", flush=True)

    steps = int(a.steps)
    seed = int(a.seed)
    ls = int(a.latent_size)

    txt = torch.randn(1, 512, 4096, device=device, dtype=torch.bfloat16)
    vec = torch.randn(1, 768, device=device, dtype=torch.bfloat16)
    t = torch.full((1,), 1.0, device=device)
    guidance = torch.full((1,), 3.5, device=device)
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device)

    def run():
        x = torch.randn(1, 16, ls, ls, device=device, dtype=torch.bfloat16,
                        generator=torch.Generator(device).manual_seed(seed))
        with torch.no_grad():
            for step in range(steps):
                out = dm(x, t, txt, vec, guidance=guidance)
                if isinstance(out, tuple):
                    out = out[0]
                x = (x + (sigmas[step + 1] - sigmas[step]) * out).to(torch.bfloat16)
        return x

    print("[*] pristine run", flush=True)
    x_ref = run()
    print("[*] pristine done", flush=True)

    impacts = {}
    done = 0
    for n in layers:
        nmod = n[len("model.diffusion_model."):] if n.startswith("model.diffusion_model.") else n
        if nmod not in mods:
            print(f"  SKIP (not a module): {n}", flush=True)
            continue
        m = mods[nmod]
        w0 = m.weight.data.clone()
        m.weight.data.copy_(nvfp4_quant_error(w0))
        try:
            x_t = run()
            imp = rel_mse(x_t, x_ref)
        except Exception as e:
            print(f"  ERR {n}: {e}", flush=True)
            imp = float("nan")
        m.weight.data.copy_(w0)
        impacts[n] = imp
        done += 1
        if done % 25 == 0 or done == len(layers):
            print(f"  [{done}/{len(layers)}]", flush=True)

    xr = x_ref.float().reshape(x_ref.shape[0], -1)
    json.dump({"x_ref_norm": float((xr * xr).sum().item()), "impacts": impacts},
              open(a.out, "w"), indent=1)
    print(f"saved {a.out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

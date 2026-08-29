# -*- coding: utf-8 -*-
"""Flux1 hybrid ConvRot NVFP4: per-layer input_scale calibration (amax method).

ZI の Z_Image/calib_input_scale_nvfp4.py を Flux1 に移植。

Adds the missing activation scales to a hybrid NVFP4 artifact so the W4A4
TensorCore path (scaled_mm_nvfp4 / cuBLAS FP4) can use calibrated per-tensor
act scales instead of placeholder ones / step-0 amax freeze.

Method (mirrors ZI / hswq_sdxl_convert_nvfp4_1.0.py input_scale calib):
  - load the BASE fp16/bf16 Flux1 via the bench loader (same as diag_impact.py)
  - attach forward hooks on the NVFP4 target Linears
  - run N calibration trajectories through a fixed-seed 4-step Euler
    (randn x0 seed 42, 4 steps; same as diag_impact.py)
  - per layer: Hadamard-rotate the input activations (group size from the
    checkpoint metadata, same as inference rotate_last_dim), then take the
    running absmax over all calib runs
  - write  <layer>.input_scale = max(amax, 1e-12) / (F8_E4M3_MAX * F4_E2M1_MAX)
    as an F32 scalar tensor into a copy of the hybrid artifact

The rotation MUST happen before amax ("rotate first, then amax"): the hybrid
weights are stored already rotated (W @ H^T), so the runtime quantizes rotated
activations.

Usage:
    python Flux1/calib_input_scale_nvfp4.py \
        "<base>.safetensors" \
        "<hybrid>_convrot_nvfp4.safetensors" \
        "<hybrid>_convrot_nvfp4_calib.safetensors" \
        [--comfy-path <ComfyUI>] [--repo-root <repo-root>] \
        [--prompts <prompts.txt>] [--samples 32] [--device cuda] [--latent-size 64]
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def parse_args():
    ap = argparse.ArgumentParser(description="Flux1 hybrid NVFP4 input_scale calibration")
    ap.add_argument("base", help="baseline fp16/bf16 Flux1 safetensors (same as diag_impact.py input 1)")
    ap.add_argument("hybrid", help="hybrid ConvRot NVFP4 artifact (gen_reverse_nvfp4.py output)")
    ap.add_argument("out", help="output safetensors path (copy of hybrid + input_scale keys)")
    ap.add_argument("--comfy-path", default=None, help="ComfyUI root path")
    ap.add_argument("--repo-root", default=None, help="repo root containing benchmark/ (default: parent of this dir)")
    ap.add_argument("--prompts", default=None, help="UTF-8 text file, one prompt per line (default: synthetic set)")
    ap.add_argument("--samples", type=int, default=32, help="number of calibration trajectories")
    ap.add_argument("--steps", type=int, default=4, help="number of Euler sampling steps per trajectory (default: 4)")
    ap.add_argument("--latent-size", type=int, default=64, help="latent H/W (default 64; 16GB VRAM 推奨)")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Hadamard rotation (same math as native_convert_int8.build_hadamard /
# rotate_last_dim used at inference; reimplemented locally)
# ---------------------------------------------------------------------------

_HADAMARD_CACHE: dict[int, torch.Tensor] = {}


def _hadamard(n: int, device, dtype=torch.float32) -> torch.Tensor:
    """Sylvester Hadamard matrix of order n (power of 2), normalized: H @ H.T = I."""
    key = int(n)
    h = _HADAMARD_CACHE.get(key)
    if h is not None and h.device == device:
        return h
    h1 = torch.ones(1, 1, device=device, dtype=dtype)
    hm = h1
    while hm.shape[0] < n:
        hm = torch.cat(
            [torch.cat([hm, hm], dim=1), torch.cat([hm, -hm], dim=1)], dim=0
        )
    hm = hm / (n ** 0.5)
    if hm.shape[0] != n:
        raise ValueError(f"Hadamard order {n} is not a power of two")
    _HADAMARD_CACHE[key] = hm
    return hm


def _is_pow4(n: int) -> bool:
    while n > 1:
        if n % 4:
            return False
        n //= 4
    return n == 1


def _group_size_for(in_features: int, preferred: int = 256) -> int | None:
    """Largest power-of-4 group <= preferred that divides in_features."""
    if in_features < 4:
        return None
    gs = preferred
    while gs >= 4:
        if in_features % gs == 0 and (gs & (gs - 1)) == 0 and _is_pow4(gs):
            return gs
        gs //= 4
    return None


def _rotate_last_dim(x: torch.Tensor, gs: int) -> torch.Tensor:
    """Hadamard-rotate the last dim in groups of gs. x: (..., in_features) fp32."""
    *lead, last = x.shape
    if last % gs != 0:
        raise ValueError(f"last dim {last} not divisible by group_size={gs}")
    y = x.reshape(*lead, last // gs, gs)
    h = _hadamard(gs, x.device, torch.float32)
    return torch.einsum("...gh,hk->...gk", y, h).reshape(*lead, last)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _load_bench(repo: str):
    sys.path.insert(0, os.path.join(repo, "benchmark"))
    sys.path.insert(0, repo)
    spec = importlib.util.spec_from_file_location(
        "fluxbench", os.path.join(repo, "benchmark", "flux1_nvfp4", "flux_int8_bench.py")
    )
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    return bench


def _default_prompts() -> list[str]:
    """Diverse synthetic prompts covering content variety."""
    base = [
        "a beautiful cyberpunk city at night, neon lights, high detail",
        "a portrait of a woman with freckles, studio lighting, 85mm",
        "a snowy mountain range at sunrise, crisp air, wide shot",
        "a bowl of ramen on a wooden table, steam, shallow depth of field",
        "an old library with tall shelves and warm lamps",
        "a red sports car on a coastal road at golden hour",
        "a cat sleeping on a windowsill, soft afternoon light",
        "an abstract painting with bold blue and orange strokes",
        "a busy street market with colorful produce stalls",
        "a lone tree in a wheat field under dramatic clouds",
        "a glass of iced coffee with condensation, close-up",
        "a modern minimal living room with plants and wood furniture",
        "a rocket launch at dawn photographed from a distance",
        "a close-up of a mechanical watch movement, macro photography",
        "a foggy forest path with moss covered stones",
        "a chef plating a fine dining dish in a dark kitchen",
    ]
    return base


def main() -> int:
    a = parse_args()
    device = a.device
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    repo = os.path.abspath(a.repo_root) if a.repo_root else os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    bench = _load_bench(repo)

    comfy_path = a.comfy_path
    if not comfy_path:
        comfy_path = os.path.join(repo, "ComfyUI-master")
    if not os.path.isabs(comfy_path):
        joined = os.path.join(repo, comfy_path)
        comfy_path = os.path.abspath(joined) if os.path.isdir(joined) else os.path.abspath(comfy_path)
    bench.setup_comfy(comfy_path)
    bench.apply_int8_patches()

    a.base, a.hybrid, a.out = a.base.strip(), a.hybrid.strip(), a.out.strip()

    # 1) Which layers are NVFP4 (from hybrid metadata) + their convrot groupsize
    with safe_open(a.hybrid, framework="pt", device="cpu") as f:
        meta = json.loads(f.metadata()["_quantization_metadata"])
    targets = {
        name[len("model.diffusion_model."):] if name.startswith("model.diffusion_model.") else name: info
        for name, info in meta["layers"].items()
        if info.get("format") == "nvfp4"
    }
    print(f"NVFP4 target layers: {len(targets)}")

    # 2) Load BASE fp16 model (same loader as diag_impact.py; weights NOT quantized)
    model = bench._load_diffusion_model(a.base)  # ModelPatcher
    dm = model.model.diffusion_model
    dm.eval()

    # 低 VRAM モードでロード（bf16 24GB は VRAM 16GB に載らないため）
    import comfy.model_management as mm

    mm.load_models_gpu([model], lowvram=True)

    mods = {n: m for n, m in dm.named_modules()
            if hasattr(m, "weight") and hasattr(m, "in_features")}
    hooks, tracked = [], {}
    for n, info in targets.items():
        if n not in mods:
            print(f"  WARN: target not found as module: {n}")
            continue
        m = mods[n]
        gs = None
        if info.get("convrot"):
            gs = _group_size_for(int(m.in_features), int(info.get("convrot_groupsize", 256)))
            if gs is None:
                print(f"  WARN: no valid Hadamard group for {n} (in={m.in_features}); skipping rotation (unrotated amax)")
        tracked[n] = {"gs": gs, "amax": 0.0}

    def _make_hook(name: str):
        def hook(m, inp, out):
            if not inp or inp[0] is None:
                return
            x = inp[0]
            if not torch.is_tensor(x) or not torch.is_floating_point(x):
                return
            x_f = x.detach().reshape(-1, int(m.in_features)).float()
            st = tracked[name]
            if st["gs"]:
                x_f = _rotate_last_dim(x_f, int(st["gs"]))
            amax = float(x_f.abs().amax().clamp_min(1e-12).item())
            if amax > st["amax"]:
                st["amax"] = amax
        return hook

    for n in tracked:
        hooks.append(mods[n].register_forward_hook(_make_hook(n)))
    print(f"hooks attached: {len(hooks)}")

    # 3) Calibration trajectories (fixed-seed 4-step Euler, same as diag_impact.py)
    prompts = _default_prompts()
    if a.prompts:
        with open(a.prompts, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    if len(prompts) < a.samples:
        prompts = (prompts * (a.samples // len(prompts) + 1))[:a.samples]
    else:
        prompts = prompts[:a.samples]
    steps = max(1, int(a.steps))
    ls = int(a.latent_size)
    print(f"calibrating: {len(prompts)} trajectories x {steps} steps, seed 42, latent {ls}x{ls}")

    def run_trajectory(sample_idx: int):
        g = torch.Generator(device).manual_seed(42 + sample_idx)
        txt = torch.randn(1, 512, 4096, device=device, dtype=torch.bfloat16, generator=g)
        vec = torch.randn(1, 768, device=device, dtype=torch.bfloat16, generator=g)
        x = torch.randn(1, 16, ls, ls, device=device, dtype=torch.bfloat16,
                        generator=torch.Generator(device).manual_seed(42))
        t = torch.full((1,), 1.0, device=device)
        guidance = torch.full((1,), 3.5, device=device)
        sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device)
        with torch.no_grad():
            for step in range(steps):
                out = dm(x, t, txt, vec, guidance=guidance)
                if isinstance(out, tuple):
                    out = out[0]
                x = (x + (sigmas[step + 1] - sigmas[step]) * out).to(torch.bfloat16)
        return x

    for i in range(len(prompts)):
        run_trajectory(i)
        if (i + 1) % 8 == 0 or i + 1 == len(prompts):
            print(f"  [{i + 1}/{len(prompts)}] amax coverage: "
                  f"{sum(1 for v in tracked.values() if v['amax'] > 0)}/{len(tracked)}")
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    for h in hooks:
        h.remove()

    missing = [n for n, v in tracked.items() if v["amax"] <= 0]
    if missing:
        print(f"WARN: {len(missing)} layers saw no activation: {missing[:5]}...")
        for n in missing:
            del tracked[n]

    # 4) Write input_scale keys into a copy of the hybrid artifact
    from comfy_kitchen.float_utils import F4_E2M1_MAX, F8_E4M3_MAX
    denom = float(F8_E4M3_MAX) * float(F4_E2M1_MAX)
    print(f"input_scale formula: amax / {denom:.0f}")

    sd = load_file(a.hybrid)
    prefix = ""
    for p in ("model.diffusion_model.", "diffusion_model.", ""):
        for k in sd.keys():
            if k.startswith(p) and k.endswith(".weight"):
                prefix = p
                break
        if prefix:
            break

    written = 0
    for n, v in tracked.items():
        n_clean = n[len("model.diffusion_model."):] if n.startswith("model.diffusion_model.") else n
        n_clean = n_clean[len("diffusion_model."):] if n_clean.startswith("diffusion_model.") else n_clean
        full = f"{prefix}{n_clean}"
        sd[f"{full}.input_scale"] = torch.tensor(
            max(v["amax"], 1e-12) / denom, dtype=torch.float32
        )
        written += 1
    save_file(sd, a.out, metadata={"_quantization_metadata": json.dumps(meta)})
    print(f"input_scale written: {written} layers")
    print(f"saved: {a.out} ({os.path.getsize(a.out) / 1e9:.2f} GB decimal)")
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

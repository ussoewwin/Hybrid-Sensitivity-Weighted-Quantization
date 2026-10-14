# -*- coding: utf-8 -*-
"""Qwen Image Edit ConvRot NVFP4: per-layer input_scale calibration (amax method).

Adds the missing activation scales to a ConvRot NVFP4 artifact so the W4A4
TensorCore path (scaled_mm_nvfp4 / cuBLAS FP4) uses calibrated per-tensor
act scales instead of placeholder ones (placeholder scales NaN the trajectory
on qwnImageEdit - measured 2026-10-13: scaled_mm_hits=840, final-cos=nan).

Method (mirrors Z_Image/calib_input_scale_nvfp4.py):
  - encode calibration prompts with the REAL Qwen2.5-VL text encoder FIRST,
    free the CLIP, only then load the 40GB base model (RAM budget)
  - forward hooks on the NVFP4 target Linears (metadata format=nvfp4)
  - N fixed-seed Euler trajectories (bench-regime latents)
  - per layer: Hadamard-rotate the input activations (group size from the
    checkpoint metadata, same as inference rotate), then running absmax
  - write  <layer>.input_scale = max(amax, 1e-12) / (F8_E4M3_MAX * F4_E2M1_MAX)
    as F32 scalars into a copy of the NVFP4 artifact (full original metadata
    preserved)

The rotation MUST happen before amax ("rotate first, then amax"): the ConvRot
weights are stored already rotated (W @ H^T), so the runtime quantizes rotated
activations. An unrotated amax is in the wrong domain and mis-scales the grid.

Synthetic randn contexts are NOT usable here: they overflow the 60-block fp16
DiT (measured: only 132/840 layers stayed finite). Real TE contexts match the
bench regime and stay finite.

Usage:
    python "Qwen Image/calib_input_scale_nvfp4_qwen.py" \
        "<base>.safetensors" \
        "<base>_convrot_nvfp4.safetensors" \
        "<base>_convrot_nvfp4_calib.safetensors" \
        --comfy-path <ComfyUI-master> --clip-path <Qwen2.5-VL.safetensors> \
        [--samples 8] [--steps 1] [--device cuda]

After it finishes, bench the calibrated artifact on the TC route; the parity
route ignores input_scale, so the calibrated file stays compatible with it.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import types

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

# ---------------------------------------------------------------------------
# Hadamard rotation (same math as native_convert_int8.build_hadamard /
# rotate_last_dim used at inference; local copy: no repo-internal import).
# ---------------------------------------------------------------------------

_HADAMARD_CACHE: dict = {}


def _hadamard(n: int, device, dtype=torch.float32) -> torch.Tensor:
    """Regular Hadamard (power-of-4), normalized: H @ H.T = I.

    Built with the SAME kron(h4) construction as the offline converter
    (native_convert_int8.build_hadamard) and the qwen runtime stack
    (nvfp4_hadamard.build_hadamard). Measured 2026-10-14: the recursive
    Sylvester construction gives a DIFFERENT matrix (max abs diff 0.125), so
    the amax measured with it lands in the wrong rotated basis and the
    runtime NVFP4 act rotation no longer cancels the weight rotation
    (trajectory collapse: mean final-cos 0.61).
    """
    key = int(n)
    h = _HADAMARD_CACHE.get(key)
    if h is not None and h.device == device:
        return h
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype, device=device,
    )
    hm = h4
    cur = 4
    while cur < n:
        hm = torch.kron(hm, h4)
        cur *= 4
    if hm.shape[0] != n:
        raise ValueError(f"Hadamard order {n} is not a power of four")
    hm = hm / (n ** 0.5)
    _HADAMARD_CACHE[key] = hm
    return hm


def _is_pow4(n: int) -> bool:
    while n > 1:
        if n % 4:
            return False
        n //= 4
    return n == 1


def _group_size_for(in_features: int, preferred: int = 256) -> int | None:
    """Largest power-of-4 group <= preferred that divides in_features.

    Mirrors native_convert_int8.convrot_group_size_for_features exactly
    (the rule the converter used offline), so the amax domain matches the
    weight rotation domain.
    """
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
# ComfyUI bootstrap (idempotent)
# ---------------------------------------------------------------------------

_COMFY_READY = False


def _bootstrap_comfy(comfy_path: str) -> None:
    global _COMFY_READY
    if _COMFY_READY:
        return
    import importlib.machinery
    from pathlib import Path

    sys.path.insert(0, str(Path(comfy_path).resolve()))

    def _stub(name: str, *, is_package: bool = False):
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

    for key in list(sys.modules):
        if key == "torchaudio" or key.startswith("torchaudio."):
            del sys.modules[key]
    ta = _stub("torchaudio", is_package=True)
    functional = _stub("torchaudio.functional")
    functional.resample = lambda waveform, orig_freq, new_freq, *a, **k: waveform
    transforms = _stub("torchaudio.transforms")

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

    saved = list(sys.argv)
    sys.argv = [saved[0]]
    try:
        import comfy.options
        comfy.options.enable_args_parsing(False)
    finally:
        sys.argv = saved
    _COMFY_READY = True


def encode_contexts(clip_path: str, prompts: list[str], device: str):
    """Load CLIP on CPU, encode real contexts, free the CLIP. Returns CPU fp16
    list of [1, seq, ctx_dim] tensors (moved to device by the caller later)."""
    _bootstrap_comfy(os.environ["HSWQ_COMFY_PATH"])
    import comfy.sd

    saved_argv = list(sys.argv)
    sys.argv = [saved_argv[0]]
    try:
        _cpu = torch.device("cpu")
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=None,
            model_options={
                "load_device": _cpu,
                "offload_device": _cpu,
                "initial_device": _cpu,
            },
        )

        def _enc(text):
            out = clip.encode_from_tokens_scheduled(clip.tokenize(text))
            # qwen TE returns [{cond_key: [context, mask]}] - dig to the raw
            # text embedding tensor [1, seq, ctx_dim].
            item = out[0] if isinstance(out, list) else out
            if isinstance(item, dict):
                item = next(iter(item.values()))
            if isinstance(item, (list, tuple)):
                item = item[0]
            return item

        contexts = []
        for pr in prompts:
            emb = _enc(pr)
            if emb.ndim == 2:
                emb = emb.unsqueeze(0)
            contexts.append(emb.to(device="cpu", dtype=torch.float16))
        del clip
    finally:
        sys.argv = saved_argv
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return contexts


def load_base_model(path: str, device: str):
    import comfy.model_management as mm
    import comfy.sd

    print(f"Loading BASE state_dict (bf16 compute): {os.path.basename(path)}")
    state_dict = load_file(path)
    patcher = comfy.sd.load_diffusion_model_state_dict(
        state_dict, model_options={"dtype": torch.bfloat16}
    )
    if patcher is None:
        raise RuntimeError("ComfyUI could not detect model from base state dict")
    del state_dict
    gc.collect()
    mm.load_models_gpu([patcher])
    model = patcher.model.diffusion_model
    model.eval()
    return patcher, model


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _default_prompts() -> list[str]:
    return [
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Qwen Image Edit ConvRot NVFP4 input_scale calibration"
    )
    ap.add_argument("base", help="baseline bf16/fp16 Qwen Image Edit safetensors")
    ap.add_argument("hybrid", help="ConvRot NVFP4 artifact (native_convert_nvfp4_qwen.py output)")
    ap.add_argument("out", help="output safetensors (copy of hybrid + input_scale keys)")
    ap.add_argument("--comfy-path", required=True, help="ComfyUI-master root")
    ap.add_argument("--clip-path", required=True, help="Qwen2.5-VL text encoder path")
    ap.add_argument("--prompts", default=None, help="UTF-8 text file, one prompt per line")
    ap.add_argument("--samples", type=int, default=8, help="calibration trajectories")
    ap.add_argument("--steps", type=int, default=1, help="Euler steps per trajectory")
    ap.add_argument("--width", type=int, default=1024, help="calib image width (matches bench)")
    ap.add_argument("--height", type=int, default=1024, help="calib image height (matches bench)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = args.device
    os.environ["HSWQ_COMFY_PATH"] = args.comfy_path

    # 1) targets + convrot group sizes from hybrid metadata (keep full meta)
    with safe_open(args.hybrid, framework="pt", device="cpu") as f:
        full_md = dict(f.metadata() or {})
        qmd = json.loads(full_md["_quantization_metadata"])
    targets = {
        name: info
        for name, info in qmd["layers"].items()
        if info.get("format") == "nvfp4"
    }
    print(f"NVFP4 target layers: {len(targets)}")
    if not targets:
        print("CRITICAL ERROR: hybrid artifact has zero nvfp4 layers.")
        return 1

    # 2) prompts + REAL TE contexts FIRST (CLIP freed before the 40GB load)
    prompts = _default_prompts()
    if args.prompts:
        with open(args.prompts, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    if len(prompts) < args.samples:
        prompts = (prompts * (args.samples // len(prompts) + 1))[: args.samples]
    else:
        prompts = prompts[: args.samples]
    contexts = encode_contexts(args.clip_path, prompts, device)
    print(f"real TE contexts encoded: {len(contexts)} | ctx[0]={tuple(contexts[0].shape)}")

    # 3) base model (fp16 compute)
    patcher, model = load_base_model(args.base, device)

    mods = {
        n: m
        for n, m in model.named_modules()
        if hasattr(m, "weight") and hasattr(m, "in_features")
    }
    tracked = {}
    for n, info in targets.items():
        if n not in mods:
            print(f"  WARN: target not found as module: {n}")
            continue
        m = mods[n]
        gs = None
        if info.get("convrot"):
            gs = _group_size_for(
                int(m.in_features), int(info.get("convrot_groupsize", 256))
            )
            if gs is None:
                print(
                    f"  WARN: no valid Hadamard group for {n} "
                    f"(in={m.in_features}); skipping rotation (unrotated amax)"
                )
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

    hooks = [mods[n].register_forward_hook(_make_hook(n)) for n in tracked]
    print(f"hooks attached: {len(hooks)}")

    # 4) fixed-seed Euler trajectories THROUGH THE SAME COMFY PATH AS THE BENCH.
    #    The raw DiT call (model(x, sigma, embeds)) skips BaseModel._call, which
    #    rescales the latent by model_sampling.calculate_input(sigma, x) before the
    #    DiT and rescales the output with calculate_denoised. Measured 2026-10-14:
    #    raw-call activation amax differs from the bench regime by median 2.6x, up
    #    to 15x on modulation layers -> input_scale measured in the wrong domain ->
    #    NVFP4 trajectory collapse (mean final-cos 0.61). Calib MUST use comfy.sample
    #    like the bench so the activation domain is identical.
    import comfy.model_management as _mm
    import comfy.sample as _comfy_sample

    steps = max(1, int(args.steps))
    lt_h = max(8, (args.height // 8) // 2 * 2)
    lt_w = max(8, (args.width // 8) // 2 * 2)
    ctx_dim = int(mods["txt_in"].in_features) if "txt_in" in mods else 3584
    model_dtype = next(model.parameters()).dtype
    print(f"model compute dtype: {model_dtype}")
    contexts = [c.to(device=device, dtype=model_dtype) for c in contexts]
    print(
        f"calibrating: {len(prompts)} trajectories x {steps} steps, seed 42 | "
        f"latent 1x16x1x{lt_h}x{lt_w} | ctx dims "
        f"{[(tuple(c.shape[1:]), c.shape[-1]) for c in contexts[:1]]}"
    )

    def run_trajectory(sample_idx: int):
        embeds = contexts[sample_idx % len(contexts)]
        if embeds.shape[-1] != ctx_dim:
            print(f"  WARN: ctx dim {embeds.shape[-1]} != model {ctx_dim}; skipping")
            return None
        base = torch.zeros(
            1, 16, lt_h, lt_w, device=_mm.intermediate_device(), dtype=model_dtype
        )
        base = _comfy_sample.fix_empty_latent_channels(patcher, base)
        noise = _comfy_sample.prepare_noise(base, 42, None)
        cond = [[embeds, {}]]
        _comfy_sample.sample(
            patcher, noise, steps, 1.0, "euler", "simple",
            cond, cond, base, denoise=1.0,
            disable_noise=False, start_step=None, last_step=None,
            force_full_denoise=True, noise_mask=None,
            callback=None, disable_pbar=True, seed=42,
        )
        return True

    for i in range(len(prompts)):
        run_trajectory(i)
        if (i + 1) % 4 == 0 or i + 1 == len(prompts):
            print(
                f"  [{i + 1}/{len(prompts)}] amax coverage: "
                f"{sum(1 for v in tracked.values() if v['amax'] > 0)}/{len(tracked)}"
            )
        if (i + 1) % 4 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    for h in hooks:
        h.remove()

    missing = [n for n, v in tracked.items() if v["amax"] <= 0]
    if missing:
        print(f"WARN: {len(missing)} layers saw no activation: {missing[:5]}...")
        for n in missing:
            del tracked[n]

    # 5) write input_scale into a copy of the hybrid artifact (full metadata)
    from comfy_kitchen.float_utils import F4_E2M1_MAX, F8_E4M3_MAX

    denom = float(F8_E4M3_MAX) * float(F4_E2M1_MAX)
    print(f"input_scale formula: amax / {denom:.0f}")

    sd = load_file(args.hybrid)
    written = 0
    for n, v in tracked.items():
        sd[f"{n}.input_scale"] = torch.tensor(
            max(v["amax"], 1e-12) / denom, dtype=torch.float32
        )
        written += 1
    tmp = args.out + ".tmp"
    save_file(sd, tmp, metadata=full_md)
    os.replace(tmp, args.out)
    print(f"input_scale written: {written} layers")
    print(f"saved: {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB decimal)")
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

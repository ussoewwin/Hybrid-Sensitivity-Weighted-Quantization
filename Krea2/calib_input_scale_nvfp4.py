# -*- coding: utf-8 -*-
"""Krea2 hybrid ConvRot NVFP4: per-layer input_scale calibration (amax method).

Calibrates and writes activation scales (input_scale) into a hybrid NVFP4 artifact
so the W4A4 TensorCore path (scaled_mm_nvfp4 / cuBLAS FP4) uses calibrated per-tensor
act scales instead of placeholder ones or runtime fallback.

Method (mirrors Z_Image/calib_input_scale_nvfp4.py & Flux1/calib_input_scale_nvfp4.py):
  - load the BASE fp16/bf16 Krea2 SingleStreamDiT model (unquantized)
  - attach forward hooks on the NVFP4 target Linears
  - run N calibration trajectories through a fixed-step Euler trajectory
    (varying seed per sample for activation distribution coverage)
  - per layer: Hadamard-rotate the input activations (group size from the
    checkpoint metadata, same as inference rotate_last_dim), then take the
    running absmax over all calibration runs
  - write <layer>.input_scale = max(amax, 1e-12) / (F8_E4M3_MAX * F4_E2M1_MAX)
    as an F32 scalar tensor into a copy of the hybrid artifact

The rotation MUST happen before amax ("rotate first, then amax"): the hybrid
weights are stored already rotated (W @ H^T), so runtime quantizes rotated
activations.

Usage:
    python Krea2/calib_input_scale_nvfp4.py \\
        "<base_dit.safetensors>" \\
        "<model>_hswq_hybrid_convrot_nvfp4.safetensors" \\
        "<model>_hswq_hybrid_convrot_nvfp4_calib.safetensors" \\
        [--comfy-path <comfyui-root>] [--samples 32] [--steps 4] \\
        [--lat 128] [--seq 256] [--seed 42] [--device cuda]
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import types

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

# ---------------------------------------------------------------------------
# ComfyUI bootstrap (stubs + cloud-safe module isolation)
# ---------------------------------------------------------------------------
def _clear_argv_for_comfy():
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    return saved


def _restore_argv(saved):
    sys.argv = saved


def _install_torchaudio_stub():
    import importlib.machinery
    for key in list(sys.modules):
        if key == "torchaudio" or key.startswith("torchaudio."):
            del sys.modules[key]

    def _stub(name, is_package=False):
        m = types.ModuleType(name)
        m.__file__ = "<stub>"
        if is_package:
            m.__path__ = []
            spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            spec.submodule_search_locations = []
        else:
            spec = importlib.machinery.ModuleSpec(name, loader=None)
        m.__spec__ = spec
        return m

    ta = _stub("torchaudio", is_package=True)
    func = _stub("torchaudio.functional")
    func.resample = lambda w, *a, **k: w
    tr = _stub("torchaudio.transforms")
    class _MS:
        def __init__(self, *a, **k): pass
        def __call__(self, x): return x
    class _ML:
        def __init__(self, *a, **k): pass
        def __call__(self, x): return x
    tr.MelSpectrogram = _MS
    tr.MelScale = _ML
    ta.functional = func
    ta.transforms = tr
    sys.modules["torchaudio"] = ta
    sys.modules["torchaudio.functional"] = func
    sys.modules["torchaudio.transforms"] = tr


def _install_comfy_stubs():
    _install_torchaudio_stub()
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
            total = 64 * 1024 ** 3
            available = 32 * 1024 ** 3
        class _P:
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
        ps.Process = lambda: _P()
        sys.modules["psutil"] = ps


def _ensure_comfyui(comfy_path=None):
    """Locate the ComfyUI root: explicit --comfy-path, <repo>/ComfyUI-master, or $COMFYUI_PATH."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.normpath(os.path.join(here, ".."))
    candidates = []
    if comfy_path:
        candidates.append(os.path.abspath(comfy_path))
    candidates.extend([
        os.path.join(repo, "ComfyUI-master"),
        r"D:\USERFILES\ComfyUI\ComfyUI",
        r"D:\USERFILES\GitHub\ComfyUI",
    ])
    env = os.environ.get("COMFYUI_PATH")
    if env:
        candidates.append(env)
    for root in candidates:
        if not root:
            continue
        if os.path.isfile(os.path.join(root, "comfy", "ldm", "krea2", "model.py")) \
                and os.path.isfile(os.path.join(root, "comfy", "ops.py")):
            return root
    raise FileNotFoundError(
        "ComfyUI root (needs comfy/ops.py + comfy/ldm/krea2/model.py) not found. "
        "Expected <repo>/ComfyUI-master. Pass --comfy-path or set COMFYUI_PATH."
    )





# ---------------------------------------------------------------------------
# Krea2 detect + load
# ---------------------------------------------------------------------------
def _find_krea2_key_prefix(keys):
    for prefix in ("model.diffusion_model.", "diffusion_model.", ""):
        if f"{prefix}txtfusion.projector.weight" in keys:
            return prefix
    raise ValueError("Not a Krea2 checkpoint: missing txtfusion.projector.weight")


def detect_krea2_dit_config(sd, prefix):
    head_dim = 128
    fw = sd[f"{prefix}first.weight"]
    features = int(fw.shape[0])
    channels = int(fw.shape[1] // 4)
    br = re.compile(r"^" + re.escape(prefix) + r"blocks\.(\d+)\.")
    layers = 0
    for k in sd:
        m = br.match(k)
        if m:
            layers = max(layers, int(m.group(1)) + 1)
    if layers <= 0:
        raise ValueError("Krea2 detect failed: no blocks.* keys")
    wq = sd[f"{prefix}blocks.0.attn.wq.weight"]
    wk = sd[f"{prefix}blocks.0.attn.wk.weight"]
    txtlayers = int(sd[f"{prefix}txtfusion.projector.weight"].shape[1])
    txtdim = int(sd[f"{prefix}txtfusion.layerwise_blocks.0.prenorm.scale"].shape[0])
    return {
        "image_model": "krea2",
        "features": features,
        "channels": channels,
        "patch": 2,
        "layers": layers,
        "heads": int(wq.shape[0] // head_dim),
        "kvheads": int(wk.shape[0] // head_dim),
        "txtlayers": txtlayers,
        "txtdim": txtdim,
    }


def load_krea2(path, device="cuda", comfy_path=None):
    """Load Krea2 SingleStreamDiT from a base fp16/bf16 safetensors onto CUDA."""
    if str(device).startswith("cpu"):
        raise RuntimeError("calib_input_scale_nvfp4 Krea2 trajectory requires CUDA.")
    comfy_root = _ensure_comfyui(comfy_path)
    print(f"[Krea2] ComfyUI root: {comfy_root}")
    saved = _clear_argv_for_comfy()
    try:
        if str(comfy_root) not in sys.path:
            sys.path.insert(0, str(comfy_root))
        comfy_dir = os.path.join(comfy_root, "comfy")
        import comfy
        if hasattr(comfy, "__path__") and comfy_dir not in comfy.__path__:
            comfy.__path__.insert(0, comfy_dir)
        _install_comfy_stubs()
        try:
            import comfy.cli_args
            if not torch.cuda.is_available():
                comfy.cli_args.args.cpu = True
        except Exception:
            pass
        try:
            import comfy.options
            comfy.options.enable_args_parsing(False)
        except ImportError:
            pass
        import comfy.ops
        from comfy.ldm.krea2.model import SingleStreamDiT

        print(f"Loading Krea2 DiT: {path}")
        state_dict = load_file(path)
        prefix = _find_krea2_key_prefix(state_dict)
        cfg = detect_krea2_dit_config(state_dict, prefix)
        print(f"Detected Krea2 DiT config: {cfg}")
        kw = {k: v for k, v in cfg.items() if k != "image_model"}
        ops = comfy.ops.disable_weight_init if hasattr(comfy.ops, "disable_weight_init") else comfy.ops.manual_cast
        dit = SingleStreamDiT(
            **kw, device="cpu", dtype=torch.bfloat16,
            operations=ops,
        )
        stripped = {}
        for k, v in state_dict.items():
            if prefix and k.startswith(prefix):
                stripped[k[len(prefix):]] = v
            elif not prefix:
                stripped[k] = v
        missing, unexpected = dit.load_state_dict(stripped, strict=False)
        print(
            f"  [Krea2] load_state_dict missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )
        dit = dit.to(device)
        dev = str(next(dit.parameters()).device)
        print(f"  [Krea2] DiT device={dev}")
        dit.eval()
        del state_dict, stripped
        gc.collect()
        return dit, cfg, prefix
    finally:
        _restore_argv(saved)


# ---------------------------------------------------------------------------
# Hadamard rotation math (mirrors converter & inference rotate_last_dim)
# ---------------------------------------------------------------------------
_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def _build_hadamard(size: int, device="cuda", dtype=torch.float32) -> torch.Tensor:
    """Normalized regular Hadamard (power-of-4), same as comfy_kitchen ConvRot."""
    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]

    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")

    h4 = torch.tensor(
        [
            [1, 1, 1, -1],
            [1, 1, -1, 1],
            [1, -1, 1, 1],
            [-1, 1, 1, 1],
        ],
        dtype=torch.float32,
        device=device,
    )
    h_matrix = h4
    current_size = 4
    while current_size < size:
        h_matrix = torch.kron(h_matrix, h4)
        current_size *= 4
    h_matrix = h_matrix / (size ** 0.5)
    out = h_matrix.to(dtype=dtype)
    _HADAMARD_CACHE[cache_key] = out
    return out


def _group_size_for(in_features: int, preferred: int = 256) -> int | None:
    """Largest power-of-4 group size <= preferred that divides in_features."""
    if in_features < 4:
        return None
    gs = preferred
    while gs >= 4:
        if in_features % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def _rotate_last_dim(x: torch.Tensor, gs: int) -> torch.Tensor:
    """Hadamard-rotate the last dim in groups of gs. x: (..., in_features) fp32."""
    *lead, last = x.shape
    if last % gs != 0:
        raise ValueError(f"last dim {last} not divisible by group_size={gs}")
    y = x.reshape(*lead, last // gs, gs)
    h = _build_hadamard(gs, device=x.device, dtype=torch.float32)
    return torch.matmul(y, h).reshape(*lead, last)


# ---------------------------------------------------------------------------
# CLI Argument Parser & Default Prompts
# ---------------------------------------------------------------------------
def _default_prompts() -> list[str]:
    """Diverse synthetic prompts covering varied style & semantics."""
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


def parse_args():
    ap = argparse.ArgumentParser(
        description="Krea2 hybrid NVFP4 per-layer input_scale calibration (amax method)"
    )
    ap.add_argument("base", help="baseline fp16/bf16 Krea2 SingleStreamDiT safetensors")
    ap.add_argument("hybrid", help="hybrid ConvRot NVFP4 artifact (from gen_reverse_nvfp4.py)")
    ap.add_argument("out", help="output safetensors path (copy of hybrid + input_scale keys)")
    ap.add_argument("--comfy-path", default=None,
                    help="ComfyUI root path (default: auto-detected <repo>/ComfyUI-master)")
    ap.add_argument("--repo-root", default=None,
                    help="repo root containing sample/ or benchmark/ (default: parent of this dir)")
    ap.add_argument("--prompts", default=None,
                    help="UTF-8 text file, one prompt per line (e.g. sample/calibration_prompts_128.txt)")
    ap.add_argument("--clip-path", "--clip_path", default=None,
                    help="Optional path to CLIP model (Qwen3-VL-4B) to encode prompts directly")
    ap.add_argument("--samples", type=int, default=None,
                    help="number of calibration trajectories (default: len(prompts) if --prompts given, else 32)")
    ap.add_argument("--steps", type=int, default=4,
                    help="number of Euler sampling steps per trajectory (default: 4)")
    ap.add_argument("--latent-size", "--lat", dest="lat", type=int, default=128,
                    help="latent H/W (default: 128, matches 1024x1024 token count)")
    ap.add_argument("--seq", type=int, default=256,
                    help="context token seq length (default: 256)")
    ap.add_argument("--seed", type=int, default=42,
                    help="base random seed (default: 42)")
    ap.add_argument("--device", default="cuda",
                    help="computation device (default: cuda)")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def main() -> int:
    import hashlib

    a = parse_args()
    device = a.device
    if str(device).startswith("cpu"):
        raise RuntimeError("Krea2 calib_input_scale_nvfp4 requires CUDA.")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    a.base = a.base.strip()
    a.hybrid = a.hybrid.strip()
    a.out = a.out.strip()

    # 1) Parse NVFP4 targets from hybrid metadata
    with safe_open(a.hybrid, framework="pt", device="cpu") as f:
        meta_raw = f.metadata() or {}
        meta_str = meta_raw.get("_quantization_metadata", '{"layers":{}}')
        meta = json.loads(meta_str)

    prefix_strip = "model.diffusion_model."
    targets = {}
    for name, info in meta.get("layers", {}).items():
        if info.get("format") == "nvfp4":
            clean_name = name[len(prefix_strip):] if name.startswith(prefix_strip) else name
            clean_name = clean_name[len("diffusion_model."):] if clean_name.startswith("diffusion_model.") else clean_name
            targets[clean_name] = info

    print(f"NVFP4 target layers in artifact metadata: {len(targets)}")
    if not targets:
        print("WARN: No NVFP4 layers found in artifact metadata. Exiting.")
        return 1

    # 2) Load base DiT model
    model, cfg, dit_prefix = load_krea2(a.base, device=device, comfy_path=a.comfy_path)
    model.eval()

    mods = {
        n: m for n, m in model.named_modules()
        if hasattr(m, "weight") and hasattr(m, "in_features")
    }

    hooks = []
    tracked: dict[str, dict] = {}

    for n, info in targets.items():
        if n not in mods:
            print(f"  WARN: target not found as module in base model: {n}")
            continue
        m = mods[n]
        gs = None
        if info.get("convrot"):
            pref_gs = int(info.get("convrot_groupsize", 256))
            gs = _group_size_for(int(m.in_features), pref_gs)
            if gs is None:
                print(f"  WARN: no valid Hadamard group for {n} (in_features={m.in_features}); using unrotated amax")
        tracked[n] = {"gs": gs, "amax": 0.0}

    def _make_hook(name: str):
        def hook(module, inp):
            if not inp or inp[0] is None:
                return
            x = inp[0]
            if not torch.is_tensor(x) or not torch.is_floating_point(x):
                return
            x_f = x.detach().reshape(-1, int(module.in_features)).float()
            st = tracked[name]
            if st["gs"]:
                x_f = _rotate_last_dim(x_f, int(st["gs"]))
            amax = float(x_f.abs().amax().clamp_min(1e-12).item())
            if amax > st["amax"]:
                st["amax"] = amax
        return hook

    for n in tracked:
        hooks.append(mods[n].register_forward_pre_hook(_make_hook(n)))
    print(f"Tracking hooks attached to {len(hooks)} layers.")

    # 3) Setup Prompts and Calibration trajectories
    prompts = _default_prompts()
    if a.prompts:
        prompts_path = a.prompts
        if not os.path.isabs(prompts_path) and a.repo_root:
            prompts_path = os.path.join(a.repo_root, prompts_path)
        if not os.path.isfile(prompts_path):
            # Also try relative to repo root if relative path passed
            here = os.path.dirname(os.path.abspath(__file__))
            repo = os.path.normpath(os.path.join(here, ".."))
            cand = os.path.join(repo, a.prompts)
            if os.path.isfile(cand):
                prompts_path = cand
            else:
                raise FileNotFoundError(f"--prompts file not found: {a.prompts}")
        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

    if a.samples is not None:
        target_samples = max(1, int(a.samples))
        if len(prompts) < target_samples:
            prompts = (prompts * (target_samples // len(prompts) + 1))[:target_samples]
        else:
            prompts = prompts[:target_samples]
    samples = len(prompts)

    steps = max(1, int(a.steps))
    lat = int(a.lat)
    seq = int(a.seq)
    txtlayers = int(cfg["txtlayers"])
    txtdim = int(cfg["txtdim"])
    channels = int(cfg["channels"])
    base_seed = int(a.seed)

    # Optional CLIP real prompt encoding
    encoded_contexts = None
    if a.clip_path and os.path.isfile(a.clip_path):
        print(f"Encoding {samples} prompts with CLIP: {a.clip_path}")
        comfy_root = _ensure_comfyui(a.comfy_path)
        if str(comfy_root) not in sys.path:
            sys.path.insert(0, str(comfy_root))
        comfy_dir = os.path.join(comfy_root, "comfy")
        import comfy
        if hasattr(comfy, "__path__") and comfy_dir not in comfy.__path__:
            comfy.__path__.insert(0, comfy_dir)
        _install_comfy_stubs()
        try:
            import comfy.cli_args
            if not torch.cuda.is_available():
                comfy.cli_args.args.cpu = True
        except Exception:
            pass
        import comfy.sd
        clip = comfy.sd.load_clip(
            ckpt_paths=[a.clip_path],
            embedding_directory=None,
            clip_type=comfy.sd.CLIPType.KREA2,
        )
        encoded_contexts = []
        for prompt in prompts:
            tokens = clip.tokenize(prompt)
            conds = clip.encode_from_tokens_scheduled(tokens)
            cond_t = conds[0][0]
            if cond_t.ndim == 2:
                cond_t = cond_t.unsqueeze(0)
            encoded_contexts.append(cond_t.to(dtype=torch.bfloat16, device="cpu"))
        del clip
        gc.collect()
        torch.cuda.empty_cache()

    t_steps = torch.linspace(1.0, 0.0, steps + 1, device=device)
    print(f"calibrating: {len(prompts)} trajectories x {steps} steps, seed {base_seed}")

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            p_hash = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
            s = (base_seed + i + p_hash) % (2**31 - 1)

            if encoded_contexts is not None:
                context = encoded_contexts[i].to(device=device)
            else:
                g_ctx = torch.Generator(device=device).manual_seed(s)
                context = torch.randn(
                    1, seq, txtlayers * txtdim, device=device, dtype=torch.bfloat16,
                    generator=g_ctx,
                )

            g_x = torch.Generator(device=device).manual_seed((s * 10007 + 42) % (2**31 - 1))
            x = torch.randn(
                1, channels, lat, lat, device=device, dtype=torch.bfloat16,
                generator=g_x,
            )
            for step in range(steps):
                t = t_steps[step:step + 1]
                out = model(x, t, context)
                if isinstance(out, tuple):
                    out = out[0]
                x = (x + (t_steps[step + 1] - t_steps[step]) * out).to(torch.bfloat16)

            if (i + 1) % 8 == 0 or i + 1 == samples:
                cov = sum(1 for v in tracked.values() if v["amax"] > 0)
                print(f"  [{i + 1}/{samples}] amax coverage: {cov}/{len(tracked)}", flush=True)
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

    # 4) Write input_scale keys into output safetensors
    from comfy_kitchen.float_utils import F4_E2M1_MAX, F8_E4M3_MAX
    denom = float(F8_E4M3_MAX) * float(F4_E2M1_MAX)
    print(f"input_scale formula: amax / {denom:.0f}")

    print(f"Loading hybrid artifact: {a.hybrid}")
    sd = load_file(a.hybrid)

    # Detect prefix in destination safetensors
    dst_prefix = ""
    for p in ("model.diffusion_model.", "diffusion_model.", ""):
        for k in sd.keys():
            if k.startswith(p) and k.endswith(".weight"):
                dst_prefix = p
                break
        if dst_prefix:
            break

    written = 0
    for n, v in tracked.items():
        full = f"{dst_prefix}{n}"
        scale_val = max(v["amax"], 1e-12) / denom
        sd[f"{full}.input_scale"] = torch.tensor(scale_val, dtype=torch.float32)
        written += 1

    out_meta = {}
    for k, v in meta_raw.items():
        if k == "_quantization_metadata":
            out_meta[k] = json.dumps(meta)
        else:
            out_meta[k] = v.decode("utf-8") if isinstance(v, bytes) else v
    if "_quantization_metadata" not in out_meta:
        out_meta["_quantization_metadata"] = json.dumps(meta)

    save_file(sd, a.out, metadata=out_meta)
    print(f"input_scale written: {written} layers")
    print(f"saved: {a.out} ({os.path.getsize(a.out) / 1e9:.2f} GB decimal)")
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

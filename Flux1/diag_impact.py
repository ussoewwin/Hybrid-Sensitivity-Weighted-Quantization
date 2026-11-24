# -*- coding: utf-8 -*-
"""Flux1 per-layer ConvRot INT8 trajectory impact (Flux1 counterpart of sdxl/diag_impact_sdxl.py).

Method (same fidelity as the SDXL / Z_Image diag, ported to Flux1):
  1. diag_impact.py       -> impact json (mean relative-MSE per layer, ascending = safest first)
  2. gen_reverse_w4a8.py  -> hybrid W4A8 artifact (K lowest-impact layers INT8 -> W4A8)

The measurement source is the FP16 checkpoint itself: every ConvRot-eligible Linear is a
candidate (no other artifact is consulted; `--artifact` only narrows the candidate list to
the layers a pack actually converted).

For each candidate layer, ONE layer is replaced by its true ConvRot INT8 reconstruction
(dequant(per-out-channel INT8(quantize(W @ H^T))) - the shipped pack's rotated basis, no
inverse rotation), the production sampler runs with a fixed seed, and the drift of every
per-step latent is recorded. The score is the mean relative MSE over the steps. Ascending
order = the layer survives INT8 without breaking the trajectory.

Differences from the retired Flux1 script (measured against sdxl/diag_impact_sdxl.py):
  - conditioning: real Flux CLIP (t5xxl + clip_l, CLIPType.FLUX) instead of random tensors
  - trajectory: comfy.sample.sample with the same sampler / cfg / seed / steps as
    benchmark/flux1_convrot_int8_traj_compare.py, instead of a hand-rolled sigma ramp
  - error: the ConvRot INT8 reconstruction instead of an NVFP4 roundtrip
  - metric: mean rel-MSE across all per-step latents instead of only the final latent

Usage:
    python Flux1/diag_impact.py <base.safetensors> [impact_out.json] \\
        --clip_path <t5xxl> --clip_l_path <clip_l> --comfy_path <ComfyUI-master> \\
        [--steps 25] [--seed 42] [--width 1024] [--height 1024] [--cfg 1.0] \\
        [--sampler euler] [--scheduler simple] [--artifact <pack>] \\
        [--limit N] [--protect_list <json|txt>] [--progress-every 25]

    The impact json goes to <repo>/impact/ when impact_out.json is omitted or is a bare
    filename. The repo root is derived from this file's location (no machine path).
"""
import argparse
import json
import math
import os
import sys
import types

import torch


def _vram(tag: str) -> None:
    if torch.cuda.is_available():
        print(
            f"[vram] {tag}: alloc={torch.cuda.memory_allocated() / 2**30:.2f} GiB "
            f"reserved={torch.cuda.memory_reserved() / 2**30:.2f} GiB "
            f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
            flush=True,
        )
        torch.cuda.reset_peak_memory_stats()


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IMPACT_DIR = os.path.join(_REPO_ROOT, "impact")


def _resolve_impact_out(out, base_path: str) -> str:
    """Omitted or bare filename -> <repo>/impact/; a path with a directory is used as given."""
    if not out:
        stem = os.path.splitext(os.path.basename(base_path))[0]
        return os.path.join(_IMPACT_DIR, f"impact_{stem}.json")
    if os.path.dirname(out):
        return out
    return os.path.join(_IMPACT_DIR, out)


# ---------------------------------------------------------------------------
# ComfyUI bootstrap (same safe import order as the Flux1 traj bench)
# ---------------------------------------------------------------------------

def _clear_argv_for_comfy():
    saved = list(sys.argv)
    sys.argv = [sys.argv[0]] if sys.argv else []
    return saved


def _restore_argv(saved):
    sys.argv = saved


def _install_torchaudio_stub() -> None:
    """comfy.sd pulls audio VAE code that hard-imports torchaudio; stub it (no AudioVAE here)."""
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


def _install_comfy_aimdo_stub() -> None:
    """Provide the comfy_aimdo surface ComfyUI-master's prefetch code imports."""
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


def import_comfy():
    import comfy.model_management  # noqa: F401
    import comfy.sample  # noqa: F401
    import comfy.sd  # noqa: F401
    import comfy.utils  # noqa: F401


# ---------------------------------------------------------------------------
# ConvRot INT8 kernel (true rotated-INT8 reconstruction, no inverse rotation)
# ---------------------------------------------------------------------------

def convrot_group_size_for_features(n: int, preferred: int = 256) -> int | None:
    """Largest power-of-4 group size <= preferred that divides n (or None)."""
    if n < 4:
        return None
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def build_hadamard(size: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Normalized regular Hadamard (power-of-4), same as comfy_kitchen ConvRot."""
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor(
        [
            [1, 1, 1, -1],
            [1, 1, -1, 1],
            [1, -1, 1, 1],
            [-1, 1, 1, 1],
        ],
        dtype=dtype,
        device=device,
    )
    h_matrix = h4
    cur = 4
    while cur < size:
        h_matrix = torch.kron(h_matrix, h4)
        cur *= 4
    return h_matrix / (size**0.5)


def rotate_weight(weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int) -> torch.Tensor:
    """Offline Linear: W_rot = W @ H^T (group-wise). Matches the converters."""
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(f"in_features {in_features} not divisible by group_size {group_size}")
    group_count = in_features // group_size
    grouped = weight.reshape(out_features, group_count, group_size)
    return torch.matmul(grouped, h_matrix.T.to(weight.dtype)).reshape(weight.shape)


def quantize_int8_channelwise(w: torch.Tensor):
    """Per-out-channel INT8 (the shipped ConvRot Linear scale layout [N, 1])."""
    reduce_dims = tuple(range(1, w.dim()))
    amax = torch.clamp(w.abs().amax(dim=reduce_dims).reshape(-1), min=1e-6)
    scale = amax / 127.0
    scale_view = scale.view(-1, 1) if w.dim() == 2 else scale.view(-1, 1, 1, 1)
    q = (w / scale_view.to(w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, scale_view.to(torch.float32)


def convrot_int8_quant_error(w: torch.Tensor, group_size: int = 256) -> torch.Tensor | None:
    """The value injected for one layer, in the ORIGINAL basis: W_q = dequant(INT8(W H^T)) H.

    The deployed pack stores the weight rotated and rotates the activation online, so the layer's
    effective weight is dequant_rotated @ H (= W up to the INT8 reconstruction error). Injecting
    that equivalent weight into the plain FP16 module therefore reproduces exactly the layer the
    pack ships, and eps = W_q - W is the true ConvRot INT8 error of that layer.

    (Measured on a synthetic 64x512 weight: cos(W_q, W) = 0.999973 with the un-rotation, 0.072427
    without it - injecting the rotated tensor measures a basis mismatch, not the quantisation.)
    """
    n_in = w.shape[1]
    gs = convrot_group_size_for_features(n_in, group_size)
    if gs is None:
        return None
    h = build_hadamard(gs, device=w.device, dtype=torch.float32)
    if w.ndim != 2:
        return None
    w_rot = rotate_weight(w.float(), h, gs)
    q, s = quantize_int8_channelwise(w_rot)
    dq_rot = q.float() * s.float()
    return _unrotate(dq_rot, gs)


def _unrotate(weight_rot: torch.Tensor, group_size: int) -> torch.Tensor:
    """W = W_rot H per group (W_rot = W H^T, H symmetric normalized -> H^-1 = H)."""
    out_features, in_features = weight_rot.shape
    groups = in_features // group_size
    h_matrix = build_hadamard(group_size, device=weight_rot.device, dtype=torch.float32)
    grouped = weight_rot.reshape(out_features, groups, group_size)
    return torch.matmul(grouped, h_matrix).reshape(weight_rot.shape)


# ---------------------------------------------------------------------------
# CLIP (t5xxl + clip_l, CPU offload) / latent / trajectory
# ---------------------------------------------------------------------------

def load_clip_and_build_conditioning(clip_path: str, clip_l_path: str, prompt: str, negative: str):
    """Flux needs BOTH encoders in one load_clip call; CPU offload, encode, unload."""
    import comfy.model_management as mm
    import comfy.sd

    _cpu = torch.device("cpu")
    clip = comfy.sd.load_clip(
        ckpt_paths=[clip_l_path, clip_path],
        embedding_directory=None,
        clip_type=comfy.sd.CLIPType.FLUX,
        model_options={"load_device": _cpu, "offload_device": _cpu, "initial_device": _cpu},
    )
    patcher = getattr(clip, "patcher", None)
    if patcher is not None:
        print(
            f"  [CLIP] patcher load_device={patcher.load_device} "
            f"offload_device={patcher.offload_device}",
            flush=True,
        )
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    negative_cond = clip.encode_from_tokens_scheduled(clip.tokenize(negative))
    for conds in (positive, negative_cond):
        for t in range(len(conds)):
            for i in range(len(conds[t])):
                if isinstance(conds[t][i], torch.Tensor):
                    conds[t][i] = conds[t][i].cpu()
    if getattr(clip, "cond_stage_model", None) is not None:
        clip.cond_stage_model.cpu()
    if patcher is not None:
        mm.unload_model_and_clones(patcher)
    del clip
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  [Offload] CLIP (t5xxl + clip_l) unloaded.", flush=True)
    return positive, negative_cond


def make_latent(model, width: int, height: int):
    import comfy.model_management as mm
    import comfy.sample as comfy_sample

    latent = torch.zeros(
        [1, 16, height // 8, width // 8],
        device=mm.intermediate_device(),
        dtype=torch.float32,
    )
    return comfy_sample.fix_empty_latent_channels(model, latent)


def run_trajectory(patcher, positive, negative, latent0, *, seed, steps, cfg, sampler, scheduler):
    """Production sampler (comfy.sample.sample) with per-step latent capture."""
    import comfy.sample as comfy_sample

    noise = comfy_sample.prepare_noise(latent0, seed, None)
    xs = []

    def cb(step, x0, x, total_steps):
        xs.append(x.detach().float().cpu())

    with torch.no_grad():
        comfy_sample.sample(
            patcher, noise, steps, cfg, sampler, scheduler,
            positive, negative, latent0, denoise=1.0,
            disable_noise=False, start_step=None, last_step=None,
            force_full_denoise=False, noise_mask=None,
            callback=cb, disable_pbar=False, seed=seed,
        )
    return xs


def _pack_key(module_name: str) -> str:
    """Impact json keys use the pack's namespace (model.diffusion_model.*)."""
    b = module_name
    for p in ("model.diffusion_model.", "diffusion_model."):
        if b.startswith(p):
            b = b[len(p):]
            break
    return "model.diffusion_model." + b

def rel_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(a.shape[0], -1)
    b = b.float().reshape(b.shape[0], -1)
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


# ---------------------------------------------------------------------------
# Boundary layers (kept at full precision; excluded from the candidate set)
# ---------------------------------------------------------------------------

_BOUNDARY_PREFIX = (
    "img_in", "txt_in", "time_in", "vector_in", "guidance_in",
    "final_layer", "distilled_guidance_layer",
)
_BOUNDARY_SUBSTR = (".img_mod", ".txt_mod", ".modulation", "adaLN", "norm")


def is_boundary_layer(module_name: str) -> bool:
    b = module_name
    for p in ("model.diffusion_model.", "diffusion_model."):
        if b.startswith(p):
            b = b[len(p):]
            break
    if b.startswith(_BOUNDARY_PREFIX):
        return True
    return any(s in b for s in _BOUNDARY_SUBSTR)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Flux1 per-layer ConvRot INT8 trajectory impact")
    ap.add_argument("base", help="baseline fp16/bf16 Flux1 DiT safetensors")
    ap.add_argument("out", nargs="?", default=None,
                    help="output impact json; default <repo>/impact/impact_<stem>.json")
    ap.add_argument("--clip_path", required=True, help="t5xxl text encoder")
    ap.add_argument("--clip_l_path", required=True, help="clip_l text encoder")
    ap.add_argument("--comfy_path", default=None, help="ComfyUI root (default <repo>/ComfyUI-master)")
    ap.add_argument("--steps", type=int, default=25, help="trajectory denoising steps (default 25)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--prompt", default="masterpiece, best quality, 1girl, solo, standing, simple background")
    ap.add_argument("--negative", default="")
    ap.add_argument("--cfg", type=float, default=1.0, help="Flux default 1.0 (guidance from extra_conds)")
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--groupsize", type=int, default=256)
    ap.add_argument("--artifact", default=None,
                    help="optional ConvRot INT8 pack: narrow the candidates to the layers it converted")
    ap.add_argument("--limit", type=int, default=None, help="limit the number of measured layers (debug)")
    ap.add_argument("--protect_list", default=None,
                    help="json/txt of layer names kept at full precision (excluded from measurement)")
    ap.add_argument("--progress-every", type=int, default=25)
    return ap.parse_args()


def main():
    args = parse_args()
    args.out = _resolve_impact_out(args.out, args.base)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    comfy_path = args.comfy_path or os.path.join(_REPO_ROOT, "ComfyUI-master")
    setup_comfy(comfy_path)
    import_comfy()

    saved = _clear_argv_for_comfy()
    try:
        import comfy.sd

        print(f"[load] baseline: {args.base}", flush=True)
        patcher = comfy.sd.load_diffusion_model(os.path.abspath(args.base), {})
        print("[load] done", flush=True)
        _vram("baseline loaded")
    finally:
        _restore_argv(saved)

    net = patcher.model
    net.eval()

    mods = {}
    for n, m in net.named_modules():
        w = getattr(m, "weight", None)
        if w is None or not torch.is_tensor(w):
            continue
        if w.ndim != 2:
            continue
        if is_boundary_layer(n):
            continue
        gs = convrot_group_size_for_features(int(w.shape[1]), args.groupsize)
        if gs is None:
            continue
        mods[n] = (m, gs)
    print(f"[target] modules eligible for ConvRot INT8: {len(mods)}", flush=True)

    if args.artifact:
        from safetensors import safe_open

        with safe_open(os.path.abspath(args.artifact), framework="pt", device="cpu") as f:
            pack_meta = json.loads(f.metadata()["_quantization_metadata"])
        wanted = []
        for k in pack_meta["layers"]:
            b = k
            for p in ("model.diffusion_model.", "diffusion_model."):
                if b.startswith(p):
                    b = b[len(p):]
                    break
            wanted.append(b)
        targets, skipped = [], []
        for b in wanted:
            # named_modules() on the comfy BaseModel prefixes with "diffusion_model."
            candidates = (f"diffusion_model.{b}", b)
            r = next((c for c in candidates if c in mods), None)
            if r is None:
                skipped.append(b)
            else:
                targets.append(r)
        seen, uniq = set(), []
        for t in targets:
            if t not in seen and t in mods:
                seen.add(t)
                uniq.append(t)
        targets = uniq
        boundary_skipped = [b for b in skipped if is_boundary_layer(b)]
        other_skipped = [b for b in skipped if b not in boundary_skipped]
        print(
            f"[target] artifact mode: {len(wanted)} pack layers -> {len(targets)} measured; "
            f"excluded {len(boundary_skipped)} boundary and {len(other_skipped)} not eligible",
            flush=True,
        )
    else:
        targets = sorted(mods.keys())

    if args.protect_list:
        raw = args.protect_list
        if raw.lower().endswith(".json"):
            with open(raw, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key in ("protected", "layers", "names", "protect"):
                    if key in data:
                        data = data[key]
                        break
            names = list(data.keys()) if isinstance(data, dict) else list(data)
        else:
            with open(raw, encoding="utf-8") as f:
                names = [ln.strip() for ln in f if ln.strip()]
        protect = set()
        for n in names:
            b = n[: -len(".weight")] if n.endswith(".weight") else n
            for p in ("model.diffusion_model.", "diffusion_model."):
                if b.startswith(p):
                    b = b[len(p):]
                    break
            protect.add(b)
        before = len(targets)
        targets = [t for t in targets if t not in protect]
        print(f"[target] protect_list: {len(protect)} names -> excluded {before - len(targets)} of {before}", flush=True)

    if args.limit:
        targets = targets[: args.limit]
    print(f"[target] measuring: {len(targets)}", flush=True)

    positive, negative = load_clip_and_build_conditioning(
        args.clip_path, args.clip_l_path, args.prompt, args.negative
    )
    latent0 = make_latent(patcher, args.width, args.height)
    if latent0.device.type != device:
        latent0 = latent0.to(device)

    seeds = [int(x.strip()) for x in str(args.seed).split(",") if x.strip()]
    if not seeds:
        raise SystemExit("no seeds given")
    print(f"[*] multi-seed impact: {len(seeds)} seeds -> {seeds}", flush=True)
    per_seed: dict = {}
    for seed in seeds:
        print(
            f"[*] pristine run (steps={args.steps}, seed={seed}, cfg={args.cfg}, "
            f"{args.sampler}/{args.scheduler}, {args.width}x{args.height})",
            flush=True,
        )
        x_ref = run_trajectory(
            patcher, positive, negative, latent0, seed=seed, steps=args.steps,
            cfg=args.cfg, sampler=args.sampler, scheduler=args.scheduler,
        )
        print("[*] pristine done", flush=True)
        _vram("pristine done")

        for n in targets:
            m, gs = mods[n]
            w0 = m.weight.data.clone()
            qerr = convrot_int8_quant_error(w0, gs)
            if qerr is None:
                print(f"  SKIP (no group): {n}", flush=True)
                continue
            m.weight.data.copy_(qerr.to(w0.dtype))
            try:
                x_t = run_trajectory(
                    patcher, positive, negative, latent0, seed=seed, steps=args.steps,
                    cfg=args.cfg, sampler=args.sampler, scheduler=args.scheduler,
                )
                imp = sum(rel_mse(x_t[i], x_ref[i]) for i in range(len(x_ref))) / len(x_ref)
            except Exception as e:
                print(f"  ERR {n}: {e}", flush=True)
                imp = float("nan")
            finally:
                m.weight.data.copy_(w0)
            del w0, qerr
            if math.isfinite(imp):
                per_seed.setdefault(_pack_key(n), []).append(float(imp))
        print(f"  [{len(targets)}/{len(targets)}] seed={seed}", flush=True)

    def _rank_average(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        rk = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                rk[order[t]] = avg
            i = j + 1
        return rk

    def _spearman(x, y):
        n = len(x)
        if n < 2:
            return None
        rx, ry = _rank_average(x), _rank_average(y)
        mx, my = sum(rx) / n, sum(ry) / n
        num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
        dx = sum((p - mx) ** 2 for p in rx) ** 0.5
        dy = sum((q - my) ** 2 for q in ry) ** 0.5
        return (num / (dx * dy)) if dx and dy else None

    def _p95(vals):
        sv = sorted(vals)
        return sv[int(round(0.95 * (len(sv) - 1)))] if sv else None

    impacts = {}
    impact_stats = {}
    for key, vals in per_seed.items():
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        impacts[key] = mu
        impact_stats[key] = {
            "mean": mu,
            "std": var ** 0.5,
            "max": max(vals),
            "p95": _p95(vals),
            "n_seeds": len(vals),
            "per_seed": vals,
        }

    complete = [k for k, v in per_seed.items() if len(v) == len(seeds)]
    for i_seed, seed_value in enumerate(seeds):
        ordered = sorted(complete, key=lambda k: per_seed[k][i_seed])
        for pos, k in enumerate(ordered):
            impact_stats[k].setdefault("rank_per_seed", {})[str(seed_value)] = pos + 1

    seed_pairs = []
    if len(complete) >= 2 and len(seeds) >= 2:
        for a in range(len(seeds)):
            for b in range(a + 1, len(seeds)):
                seed_pairs.append({
                    "a": seeds[a],
                    "b": seeds[b],
                    "rho": _spearman([per_seed[k][a] for k in complete],
                                     [per_seed[k][b] for k in complete]),
                    "n_layers": len(complete),
                })
    rhos = [p["rho"] for p in seed_pairs if p["rho"] is not None]
    seed_rank = {
        "pairs": seed_pairs,
        "rho_min": min(rhos) if rhos else None,
        "rho_mean": (sum(rhos) / len(rhos)) if rhos else None,
    }


    xr = x_ref[-1].float().reshape(1, -1)
    _vram("measurement done")
    payload = {
        "x_ref_norm": float((xr * xr).sum().item()),
        "steps": args.steps,
        "seeds": seeds,
        "impact_stats": impact_stats,
        "impacts_per_seed": per_seed,
        "seed_rank_correlation": seed_rank,
        "cfg": args.cfg,
        "sampler": args.sampler,
        "scheduler": args.scheduler,
        "width": args.width,
        "height": args.height,
        "base": os.path.abspath(args.base),
        "impacts": impacts,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"saved {args.out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

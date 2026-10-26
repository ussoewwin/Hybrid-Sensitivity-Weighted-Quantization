# -*- coding: utf-8 -*-
"""SD1.5 UNet: per-layer **trajectory** impact, measured on the model's own
denoising output.

Structure and measurement engine follow `sdxl/diag_impact_sdxl.py`:

  * the checkpoint is loaded ComfyUI-natively (`comfy.sd.load_checkpoint_guess_config`),
    which yields a ModelPatcher and a CLIP;
  * the conditioning is built from the prompt / negative through that CLIP, after
    which CLIP is freed from VRAM before the sweep;
  * the trajectory is produced by the **production sampler**
    (`comfy.sample.sample`) with the same engine, sampler, scheduler, cfg and
    seed as `benchmark/sd1.5_int8_traj_compare.py`, so the impact ranking is
    measured on exactly the trajectory the quality gate uses;
  * every other layer is held at its pristine weight, ONE layer's weight is
    replaced by its ConvRot INT8 reconstruction **in the original frame**
    (`dequant(INT8(W @ H^T))` = `(q*s) @ H`, computed in FP32 with per-row scale
    `[O, 1]` for 2D Linear and per-output-channel scale `[O, 1, 1, 1]` for 4D
    Conv2d), and the divergence from the pristine trajectory is recorded.

Usage:
  python sd1.5/diagimpact_sd15.py \
      --comfy <ComfyUI root> \
      --ckpt  <source checkpoint>.safetensors \
      --out   impact/impact_sd15_traj_<model>.json \
      --steps 25 --seeds 42,137 \
      --width 512 --height 512 --cfg 7.0 \
      --sampler dpmpp_2m --scheduler karras --groupsize 256
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types

# Strip cudaMallocAsync backend before torch CUDA initialization if inherited from parent env.
_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if _alloc_conf and "backend:" in _alloc_conf:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(
        p for p in (s.strip() for s in _alloc_conf.split(","))
        if p and not p.startswith("backend:")
    )

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float16

# SD1.5's production resolution. The latent is width/8 x height/8 (the VAE's
# 8x spatial compression), so 512 pixels means a 64x64 latent.
DEF_WIDTH = 512
DEF_HEIGHT = 512

GROUP_SIZE = 256
_H: dict[int, torch.Tensor] = {}

comfy_utils = None
comfy_md = None
comfy_sd = None
comfy_sample = None


def _vram(tag: str) -> None:
    """Print the CUDA memory used since the last _vram call, then reset the peak counter."""
    if torch.cuda.is_available():
        print(
            f"[vram] {tag}: alloc={torch.cuda.memory_allocated() / 2**30:.2f} GiB "
            f"reserved={torch.cuda.memory_reserved() / 2**30:.2f} GiB "
            f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
            flush=True,
        )
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# ComfyUI bootstrap & compatibility stubs (matches sdxl/diag_impact_sdxl.py)
# ---------------------------------------------------------------------------

def _install_torchaudio_stub() -> None:
    """Prevent real torchaudio from loading when comfy.sd is imported."""
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
    """Ensure comfy_aimdo and submodules are available for headless ComfyUI import."""
    aimdo = sys.modules.get("comfy_aimdo")
    if aimdo is None:
        try:
            import comfy_aimdo as aimdo  # noqa: F401
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
            setattr(aimdo, "filter", fmod)

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
            setattr(aimdo, "malloc_graph", mg_mod)

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
            setattr(aimdo, "model_vbar", vbar_mod)


def setup_comfy(comfy_path: str) -> None:
    """Put the given ComfyUI root first on sys.path and import comfy modules safely."""
    global comfy_utils, comfy_md, comfy_sd, comfy_sample
    comfy_root = os.path.abspath(comfy_path)
    if not os.path.isdir(os.path.join(comfy_root, "comfy")):
        raise SystemExit("not a ComfyUI root (no comfy/ inside): %s" % comfy_root)
    sys.path = [comfy_root] + [p for p in sys.path if os.path.abspath(p or ".") != comfy_root]

    _install_torchaudio_stub()
    _install_comfy_aimdo_stub()
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

    import comfy.options
    comfy.options.args_parsing = False
    if hasattr(comfy.options, "enable_args_parsing"):
        comfy.options.enable_args_parsing(False)

    import comfy.model_detection as _md
    import comfy.model_management  # noqa: F401
    import comfy.ops  # noqa: F401
    import comfy.sample as _sample
    import comfy.sd as _sd
    import comfy.utils as _u

    comfy_utils, comfy_md, comfy_sd, comfy_sample = _u, _md, _sd, _sample


_bind_comfy = setup_comfy


def _clear_argv_for_comfy():
    """comfy's loader parses sys.argv; keep our own flags away from it."""
    saved = list(sys.argv)
    sys.argv = [saved[0]] if saved else []
    return saved


def _restore_argv(saved):
    sys.argv = saved


# ---------------------------------------------------------------------------
# ConvRot INT8 kernel (FP32 rotation + per-channel INT8 + inverse rotation)
# ---------------------------------------------------------------------------

def build_hadamard(size: int, dtype=torch.float32) -> torch.Tensor:
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError("Hadamard size must be a power of 4, got %d" % size)
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
    )
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def group_size_for(n: int, preferred: int = GROUP_SIZE) -> int | None:
    if n < 4:
        return None
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def quant_dequant(w: torch.Tensor, gs: int) -> torch.Tensor:
    """ConvRot INT8 reconstruction in the ORIGINAL frame: (q*s) @ H.

    All rotation, amax reduction, quantization and dequantization are performed
    in FP32 to match the offline converter (`quantize_sd1.5_hswq_v1.1.py` /
    `sdxl/diag_impact_sdxl.py`):
      * 2D Linear (O, I)       : rotate along I -> rowwise scale (O, 1)
      * 4D Conv2d (O, I, kH, kW): rotate along I -> channelwise scale (O, 1, 1, 1)
                                  reducing over (I, kH, kW)
    """
    if gs not in _H:
        _H[gs] = build_hadamard(gs, dtype=torch.float32)
    wf = w.float()
    hh = _H[gs].to(device=wf.device, dtype=torch.float32)
    if wf.dim() == 2:
        of, inf = wf.shape
        g = inf // gs
        rot = torch.matmul(wf.view(of, g, gs), hh.T).reshape(wf.shape)
        amax = rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
        s = amax / 127.0
        q = (rot / s).round().clamp(-127, 127).to(torch.int8)
        dq = q.float() * s
        return torch.matmul(dq.view(of, g, gs), hh).reshape(wf.shape)
    if wf.dim() == 4:
        oc, ic, kh, kw = wf.shape
        flat = wf.permute(0, 2, 3, 1).contiguous().view(-1, ic)
        g = ic // gs
        rot_flat = torch.matmul(flat.view(-1, g, gs), hh.T).reshape(oc, kh, kw, ic)
        rot = rot_flat.permute(0, 3, 1, 2).contiguous()  # (oc, ic, kh, kw)
        amax = rot.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-30)  # (oc, 1, 1, 1)
        s = amax / 127.0
        q = (rot / s).round().clamp(-127, 127).to(torch.int8)
        dq = (q.float() * s).permute(0, 2, 3, 1).contiguous().view(-1, ic)
        back = torch.matmul(dq.view(-1, g, gs), hh).reshape(oc, kh, kw, ic)
        return back.permute(0, 3, 1, 2).contiguous()
    raise ValueError("unsupported rank %d" % wf.dim())


# ---------------------------------------------------------------------------
# Checkpoint loading / conditioning / trajectory
# ---------------------------------------------------------------------------

def load_sd15(path: str):
    """Load a checkpoint ComfyUI-natively; returns (ModelPatcher, clip).

    The baseline contains no INT8 tensors, so the stock ops are used.
    """
    saved = _clear_argv_for_comfy()
    try:
        out = comfy_sd.load_checkpoint_guess_config(
            os.path.abspath(path),
            output_vae=False,   # the VAE is never used by the trajectory measurement
            output_clip=True,
            embedding_directory=None,
        )
    finally:
        _restore_argv(saved)
    model, clip = out[0], out[1]
    if model is None or clip is None:
        raise RuntimeError("checkpoint load failed (model/clip None): %s" % path)
    return model, clip


def build_conditioning(clip, prompt: str, negative: str):
    """Build the CLIP conditioning exactly as the benchmark does."""
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    negative_cond = clip.encode_from_tokens_scheduled(clip.tokenize(negative))
    return positive, negative_cond


def make_latent(model, width: int, height: int):
    import comfy.model_management as mm

    latent = torch.zeros(
        [1, 4, height // 8, width // 8],
        device=mm.intermediate_device(),
        dtype=mm.intermediate_dtype(),
    )
    latent = comfy_sample.fix_empty_latent_channels(model, latent, 8, None)
    return latent


def run_trajectory(patcher, positive, negative, latent0, *, seed, steps, cfg,
                   sampler, scheduler):
    """Run the production sampler and capture the per-step latent x.

    Identical engine, sampler, scheduler, cfg and seed as
    `benchmark/sd1.5_int8_traj_compare.py`.
    """
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


def rel_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    """sum((a-b)^2) / sum(b^2), in float32.

    `b` is the reference, so the error is normalised by the reference energy -
    identical to the SDXL reference (diag_impact_sdxl.py L355).
    """
    a = a.float().reshape(a.shape[0], -1)
    b = b.float().reshape(b.shape[0], -1)
    den = (b ** 2).sum()
    if den == 0:
        return float((a - b).abs().max().item())
    return float(((a - b) ** 2).sum() / den)


def cos_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return float("nan")
    return float(1.0 - (a @ b) / (na * nb))


def derive_out_path(ckpt_path: str, prefix: str) -> str:
    """Build the output path from the model name, so `--out` is optional.

    `impact/<prefix>_<model stem>.json` - the stem carries the model's own name,
    so a JSON can always be traced back to the checkpoint it came from, and two
    checkpoints can never overwrite each other's ranking.
    """
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stem = os.path.basename(ckpt_path)
    for ext in (".safetensors", ".ckpt", ".pt", ".bin"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    stem = stem.replace(" ", "_")
    out_dir = os.path.join(repo, "impact")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "%s_%s.json" % (prefix, stem))


def _strip_unet_prefix(name: str) -> str:
    for p in ("model.diffusion_model.", "diffusion_model."):
        if name.startswith(p):
            return name[len(p):]
    return name


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_comfy = os.environ.get("COMFY_PATH", "")
    if not default_comfy:
        bundled = os.path.join(repo_root, "ComfyUI-master")
        if os.path.isdir(os.path.join(bundled, "comfy")):
            default_comfy = bundled

    ap = argparse.ArgumentParser(description="SD1.5 per-layer trajectory impact (ConvRot INT8)")
    ap.add_argument("pos_ckpt", nargs="?", default="",
                    help="optional positional checkpoint path (alternative to --ckpt)")
    ap.add_argument("pos_out", nargs="?", default="",
                    help="optional positional output JSON path (alternative to --out)")
    ap.add_argument("--comfy", "--comfy_path", dest="comfy", default=default_comfy,
                    help="ComfyUI root (the directory containing comfy/); "
                         "falls back to COMFY_PATH or bundled ComfyUI-master")
    ap.add_argument("--ckpt", "--base", dest="ckpt", default="")
    ap.add_argument("--out", default="",
                    help="output JSON (default: impact/impact_sd15_traj_<model stem>.json)")
    ap.add_argument("--steps", type=int, default=25,
                    help="denoising steps per trajectory (default 25 - SD1.5's "
                         "production step count, the same one the gate and the "
                         "calibration use)")
    ap.add_argument("--seeds", default="",
                    help="trajectory seeds, comma separated (default 42,137)")
    ap.add_argument("--seed", type=int, default=None,
                    help="single trajectory seed override (alternative to --seeds)")
    ap.add_argument("--groupsize", type=int, default=GROUP_SIZE,
                    help="preferred ConvRot Hadamard group size (power of 4, default 256)")
    ap.add_argument("--n-layers", "--limit", dest="n_layers", type=int, default=0,
                    help="measure only the first N layers (0 = all; smoke test aid)")
    ap.add_argument("--metric", default="relmse", choices=["relmse", "cos"],
                    help="relmse (as diag_impact.py) or cos")
    ap.add_argument("--width", type=int, default=DEF_WIDTH,
                    help="image width in pixels (default 512; the latent is width/8)")
    ap.add_argument("--height", type=int, default=DEF_HEIGHT,
                    help="image height in pixels (default 512; the latent is height/8)")
    ap.add_argument("--cfg", type=float, default=7.0,
                    help="classifier-free guidance scale (default 7.0, as the benchmark)")
    ap.add_argument("--sampler", default="dpmpp_2m",
                    help="sampler name (default dpmpp_2m, as the benchmark)")
    ap.add_argument("--scheduler", default="karras",
                    help="scheduler name (default karras, as the benchmark)")
    ap.add_argument("--prompt", default="masterpiece, best quality, 1girl, solo, standing, simple background")
    ap.add_argument("--negative", default="")
    ap.add_argument("--artifact", default=None,
                    help="optional ConvRot INT8 pack path to restrict candidates to its converted layers")
    ap.add_argument("--protect_list", default=None,
                    help="optional json/txt of layer names kept at FP16 (excluded from measurement)")
    ap.add_argument("--progress-every", type=int, default=10)
    a = ap.parse_args()

    if not a.ckpt and a.pos_ckpt:
        a.ckpt = a.pos_ckpt
    if not a.out and a.pos_out:
        a.out = a.pos_out
    if not a.ckpt:
        ap.error("the checkpoint is required: pass --ckpt <path> (or positional <ckpt>)")
    if not a.comfy:
        ap.error("the ComfyUI root is not set: pass --comfy <path> or set COMFY_PATH")
    if a.groupsize < 4 or (a.groupsize & (a.groupsize - 1)) != 0 or math.log(a.groupsize, 4) % 1 != 0:
        ap.error("--groupsize must be a power of 4 (>= 4), got %d" % a.groupsize)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    setup_comfy(a.comfy)

    if not a.out:
        a.out = derive_out_path(a.ckpt, "impact_sd15_traj")
        print("the output (from the model name): %s" % a.out, flush=True)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    if a.seeds.strip():
        seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    elif a.seed is not None:
        seeds = [int(a.seed)]
    else:
        seeds = [42, 137]

    print("[1] the checkpoint ...", flush=True)
    patcher, clip = load_sd15(a.ckpt)
    net = patcher.model
    net.to(DEV)
    net.eval()
    print("    the patcher: %s" % type(patcher).__name__, flush=True)
    print("    the model: %s" % type(net).__name__, flush=True)
    _vram("baseline loaded")

    positive, negative = build_conditioning(clip, a.prompt, a.negative)
    # Free CLIP before the measurement sweep so VRAM stays on the UNet + activations.
    del clip
    if DEV == "cuda":
        torch.cuda.empty_cache()
    _vram("conditioning built (clip freed)")

    latent0 = make_latent(patcher, a.width, a.height)
    if latent0.device.type != DEV:
        latent0 = latent0.to(DEV)
    print("    the latent: %s (from %dx%d px, %s)" % (
        tuple(latent0.shape), a.width, a.height, latent0.dtype), flush=True)
    print("    the sampler/scheduler/cfg: %s / %s / %s" % (a.sampler, a.scheduler, a.cfg),
          flush=True)

    # ---- the pristine reference trajectories ------------------------------
    # Every step is kept, not just the last: the aggregate below averages the
    # per-step divergence, which is the whole point of a trajectory measurement
    # (SDXL does the same at diag_impact_sdxl.py L597).
    print("[2] the pristine trajectories (seeds=%s, steps=%d) ..." % (seeds, a.steps),
          flush=True)
    t0 = time.time()
    ref = {}
    for sd in seeds:
        xs = run_trajectory(patcher, positive, negative, latent0, seed=sd,
                            steps=a.steps, cfg=a.cfg, sampler=a.sampler,
                            scheduler=a.scheduler)
        ref[sd] = xs
        print("    seed %d: %d steps captured, final %s finite=%s" % (
            sd, len(xs), tuple(xs[-1].shape), bool(torch.isfinite(xs[-1]).all())),
            flush=True)
    print("    (%.1f s)" % (time.time() - t0), flush=True)
    _vram("pristine done")

    # ---- the candidate modules -------------------------------------------
    diffusion = getattr(net, "diffusion_model", None)
    if diffusion is None:
        raise RuntimeError("the patcher's model exposes no diffusion_model")
    mods_dict = {}
    for name, m in diffusion.named_modules():
        w = getattr(m, "weight", None)
        if w is None or not hasattr(w, "ndim") or w.ndim not in (2, 4):
            continue
        if int(w.shape[1]) < 4 or int(w.shape[0]) % 4 != 0:
            continue
        gs = group_size_for(int(w.shape[1]), a.groupsize)
        if gs is None:
            continue
        mods_dict[name] = (w, gs)
    print("[3] the candidate layers eligible for ConvRot INT8: %d" % len(mods_dict), flush=True)

    target_names = list(mods_dict.keys())

    if a.artifact:
        from safetensors import safe_open
        with safe_open(os.path.abspath(a.artifact), framework="pt", device="cpu") as f:
            pack_meta = json.loads(f.metadata()["_quantization_metadata"])
        pack_layers = [_strip_unet_prefix(k) for k in pack_meta.get("layers", {})]
        filtered = [b for b in pack_layers if b in mods_dict]
        seen = set()
        target_names = [x for x in filtered if not (x in seen or seen.add(x))]
        print("    [artifact] %d pack layers -> %d measured" % (len(pack_layers), len(target_names)),
              flush=True)

    if a.protect_list:
        raw = a.protect_list
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
            b = n[:-len(".weight")] if n.endswith(".weight") else n
            protect.add(_strip_unet_prefix(b))
        before = len(target_names)
        target_names = [t for t in target_names if _strip_unet_prefix(t) not in protect]
        print("    [protect_list] %d names -> excluded %d of %d" % (
            len(protect), before - len(target_names), before), flush=True)

    if a.n_layers and a.n_layers > 0:
        target_names = target_names[: a.n_layers]
        print("    (limited to the first %d for the smoke test)" % len(target_names), flush=True)

    # ---- the sweep -------------------------------------------------------
    impacts = {}
    print("[4] the sweep (%d layers) ..." % len(target_names), flush=True)
    t0 = time.time()
    for idx, name in enumerate(target_names):
        w, gs = mods_dict[name]
        orig = w.data.detach().clone()
        rec = quant_dequant(orig, gs).to(orig.dtype)
        try:
            w.data.copy_(rec)
            vals = []
            for sd in seeds:
                xs = run_trajectory(patcher, positive, negative, latent0, seed=sd,
                                    steps=a.steps, cfg=a.cfg, sampler=a.sampler,
                                    scheduler=a.scheduler)
                ref_xs = ref[sd]
                if len(xs) != len(ref_xs):
                    raise RuntimeError(
                        "step count mismatch: %d vs the reference %d"
                        % (len(xs), len(ref_xs)))
                # average over EVERY step, exactly as the SDXL reference does
                if a.metric == "relmse":
                    step_vals = [rel_mse(xs[step_idx], ref_xs[step_idx])
                                 for step_idx in range(len(ref_xs))]
                else:
                    step_vals = [cos_dist(xs[step_idx], ref_xs[step_idx])
                                 for step_idx in range(len(ref_xs))]
                vals.append(sum(step_vals) / len(step_vals))
            impacts[name] = vals
        except Exception as e:
            print("    ERR %s: %s" % (name, e), flush=True)
            impacts[name] = [float("nan")]
        finally:
            w.data.copy_(orig)
        del orig, rec
        if (idx + 1) % a.progress_every == 0 or (idx + 1) == len(target_names):
            el = time.time() - t0
            print("    [%d/%d] %.0fs (%.2fs/layer)" % (
                idx + 1, len(target_names), el, el / (idx + 1)), flush=True)

    _vram("measurement done")

    # ---- the aggregate ---------------------------------------------------
    agg = {}
    for name, vals in impacts.items():
        if vals is None:
            continue
        agg[name] = sum(vals) / len(vals)
    ranked = sorted(
        ((k, v) for k, v in agg.items() if not (isinstance(v, float) and math.isnan(v))),
        key=lambda kv: -kv[1],
    )

    xr = ref[seeds[0]][-1].float().reshape(1, -1)
    out = {
        "x_ref_norm": float((xr * xr).sum().item()),
        "impacts": agg,
        "per_seed": impacts,
        "steps": a.steps,
        "seed": seeds[0],
        "seeds": seeds,
        "metric": a.metric,
        "groupsize": a.groupsize,
        "width": a.width,
        "height": a.height,
        "cfg": a.cfg,
        "sampler": a.sampler,
        "scheduler": a.scheduler,
        "base": os.path.abspath(a.ckpt),
        "ckpt": os.path.basename(a.ckpt),
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=True, ensure_ascii=False)
    print("[5] saved: %s" % a.out, flush=True)
    print("    the measured layers: %d" % len(agg), flush=True)
    print("    the top 10:", flush=True)
    for n, v in ranked[:10]:
        print("      %-52s %.6e" % (n, v), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

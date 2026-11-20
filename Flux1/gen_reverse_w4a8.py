# -*- coding: utf-8 -*-
"""Reverse hybrid converter (Flux1): all-INT8 (rotated) -> convert K lowest-impact layers to ConvRot W4A8.

Reverse hybrid W4A8 method: the impact json from Flux1/diag_impact.py ranks every layer;
the K lowest-impact layers move from ConvRot INT8 to ConvRot W4A8, the rest stay INT8
(the Flux1 port of the reverse hybrid idea, with the W4A8 packer).

Weights in the all-INT8 artifact are already ROTATED (W_rot = W H^T). The W4A8 packer
(comfy_kitchen.quantize_w4a8_int8_weight) rotates INSIDE, so the rotated dequant is
un-rotated first (W = W_rot H, H symmetric normalized) and then quantized - otherwise the
runtime would rotate activations once while the weight carries a double rotation.

Only layers whose base stamp is ConvRot INT8 with convrot_groupsize 256 are eligible
(the kitchen W4A8 CUDA requant path supports convrot group 256 only); plain INT8 layers
and boundary layers stay INT8.

Usage:
    python Flux1/gen_reverse_w4a8.py <K> <out_name.safetensors> <src_all_int8.safetensors> <impact.json> \\
        [--out-dir <output-dir>]

Storage spec (comfy_kitchen AsymW4A8Int8Layout):
.weight int8 [N, K/2] packed nibbles, .weight_s_rel fp8e4m3 [N, K/16],
.weight_s_channel f32 [N], .weight_codebook f32 [16] (when the Lloyd-Max table is used),
.comfy_quant {"format":"asym_w4a8_int8","group_size":16,"convrot_groupsize":256}.
"""
import argparse
import json
import math
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import comfy_kitchen as ck

_QUANT_GROUP_SIZE = 16
_CONVROT_GROUPSIZE = 256

# Flux1 boundary layers (adaLN modulation + input/output heads) stay INT8 (never W4A8)
_BOUNDARY_MARKERS = (
    "img_in.", "txt_in.", "time_in.", "vector_in.", "guidance_in.",
    "final_layer.", "mod", "norm",
)

_HADAMARD_CACHE: dict[tuple[int, torch.dtype], torch.Tensor] = {}


def _is_boundary(key: str) -> bool:
    return any(b in key for b in _BOUNDARY_MARKERS)


def _hadamard(size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Normalized regular Hadamard (power-of-4), same construction as the converters."""
    cache_key = (size, dtype)
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
        dtype=dtype,
    )
    h_matrix = h4
    current = 4
    while current < size:
        h_matrix = torch.kron(h_matrix, h4)
        current *= 4
    h_matrix = h_matrix / (size**0.5)
    _HADAMARD_CACHE[cache_key] = h_matrix
    return h_matrix


def _unrotate(weight_rot: torch.Tensor, group_size: int) -> torch.Tensor:
    """W = W_rot H per group (W_rot = W H^T, H symmetric normalized -> H^-1 = H)."""
    out_features, in_features = weight_rot.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features {in_features} not divisible by group_size {group_size}"
        )
    h_matrix = _hadamard(group_size, torch.float32)
    groups = in_features // group_size
    grouped = weight_rot.view(out_features, groups, group_size)
    return torch.matmul(grouped, h_matrix).reshape(weight_rot.shape)


# ---------------------------------------------------------------------------
# Card 1 bias correction (DualMonitor mu_x -> bias += -(W_q - W) @ mu_x)
# Copied from Flux1/hswq_convrot_w4a8_flux1_v1.0.py so this reverse converter
# carries the same implementation instead of re-deriving it.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# DualMonitor (Card 1) — self-contained
# ---------------------------------------------------------------------------
class DualMonitor:
    """Per-layer act moments for Card 1 bias correction."""

    def __init__(self):
        self.output_sum = 0.0
        self.output_sq_sum = 0.0
        self.count = 0
        self.channel_importance = None
        self.channel_act_mean = None
        self.channel_act_sq_mean = None

    def update(self, input_tensor, output_tensor, module=None, weight: float = 1.0):
        with torch.no_grad():
            out_detached = output_tensor.detach().float()
            out_clamped = torch.clamp(out_detached, -65504.0, 65504.0)
            mean_val = out_clamped.mean().item()
            sq_mean_val = (out_clamped ** 2).mean().item()

            if math.isfinite(mean_val) and math.isfinite(sq_mean_val):
                self.output_sum += mean_val
                self.output_sq_sum += sq_mean_val
            inp_detached = input_tensor.detach().float()
            if inp_detached.dim() >= 2:
                reduce_dims = tuple(range(inp_detached.dim() - 1))
            else:
                current_imp = torch.ones(
                    1, device=inp_detached.device, dtype=torch.float32
                )
                current_act = torch.zeros(
                    1, device=inp_detached.device, dtype=torch.float32
                )
                current_sq = torch.ones(
                    1, device=inp_detached.device, dtype=torch.float32
                )
                reduce_dims = None
            if reduce_dims is not None:
                current_imp = inp_detached.abs().mean(dim=reduce_dims)
                current_act = inp_detached.mean(dim=reduce_dims)
                current_sq = (inp_detached ** 2).mean(dim=reduce_dims)
            w = float(weight)
            self.output_sum *= self.count / max(self.count + w, 1e-12)
            self.output_sq_sum *= self.count / max(self.count + w, 1e-12)
            self.output_sum += mean_val * w
            self.output_sq_sum += sq_mean_val * w
            if self.channel_importance is None:
                self.channel_importance = current_imp
                self.channel_act_mean = current_act
                self.channel_act_sq_mean = current_sq
            elif current_imp.shape == self.channel_importance.shape:
                self.channel_importance = (
                    self.channel_importance * self.count + current_imp * w
                ) / (self.count + w)
                self.channel_act_mean = (
                    self.channel_act_mean * self.count + current_act * w
                ) / (self.count + w)
                self.channel_act_sq_mean = (
                    self.channel_act_sq_mean * self.count + current_sq * w
                ) / (self.count + w)
            self.count += w


dual_monitors: dict[str, DualMonitor] = {}
_dm_timestep_weight: float = 1.0


def hook_fn(module, input, output, name):
    if name not in dual_monitors:
        dual_monitors[name] = DualMonitor()
    dual_monitors[name].update(input[0], output, module, weight=_dm_timestep_weight)


def compute_int8_bias_delta(weight_fp, weight_dq, act_mean):
    """bias_delta ≈ (W_q - W) contracted with per-input-channel E[x]."""
    if act_mean is None:
        return None
    err = weight_dq.float() - weight_fp.float()
    mu = act_mean.float().to(device=err.device)
    if err.ndim == 2:
        if mu.numel() != err.shape[1]:
            return None
        return err @ mu
    if err.ndim == 4:
        if mu.numel() != err.shape[1]:
            return None
        return (err * mu.view(1, -1, 1, 1)).sum(dim=(1, 2, 3))
    return None

def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _repo_root() -> str:
    here = _script_dir()
    repo = os.path.normpath(os.path.join(here, ".."))
    if os.path.isdir(os.path.join(repo, "ComfyUI-master")):
        return repo
    return here

def _find_flux_key_prefix(keys) -> str:
    """Prefix detection from keys only (dict or any key iterable)."""
    keyset = keys if isinstance(keys, (set, frozenset)) else set(keys)
    for prefix in ("model.diffusion_model.", "diffusion_model.", ""):
        if f"{prefix}img_in.weight" in keyset:
            if f"{prefix}double_blocks.0.img_attn.qkv.weight" not in keyset:
                raise ValueError(
                    "Flux1 signature incomplete: img_in.weight present but "
                    f"{prefix}double_blocks.0.img_attn.qkv.weight missing"
                )
            return prefix
    raise ValueError(
        "Not a Flux1 checkpoint: missing img_in.weight "
        "(under model.diffusion_model. / diffusion_model. / root)."
    )

# ---------------------------------------------------------------------------
# ComfyUI bootstrap
# ---------------------------------------------------------------------------
def _ensure_comfyui_on_sys_path(comfy_path: str | None = None) -> str:
    candidates = []
    if comfy_path:
        candidates.append(os.path.abspath(comfy_path))
    env = os.environ.get("COMFYUI_PATH")
    if env:
        candidates.append(env)
    repo = _repo_root()
    candidates.extend(
        [
            os.path.join(repo, "ComfyUI-master"),
            os.path.join(_script_dir(), "ComfyUI-master"),
        ]
    )
    for root in candidates:
        if not root:
            continue
        model_py = os.path.join(root, "comfy", "ldm", "flux", "model.py")
        if os.path.isfile(model_py):
            if root not in sys.path:
                sys.path.insert(0, root)
            return root
    raise FileNotFoundError(
        "ComfyUI root with comfy/ldm/flux/model.py not found. "
        "Pass --comfy_path or set COMFYUI_PATH."
    )


def _clear_argv_for_comfy() -> list[str]:
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    return saved


def _restore_argv(saved: list[str]) -> None:
    sys.argv = saved


def _install_torchaudio_stub() -> None:
    """Prevent real torchaudio from loading during comfy.sd import."""
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


def _install_comfy_optional_stubs() -> None:
    """Lightweight stubs (same pattern as the other HSWQ converters)."""
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


def _encode_flux_calib_contexts(
    *,
    clip_path: str,
    clip_l_path: str,
    prompts: list[str],
    comfy_path: str | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
    """Encode calib prompts with Comfy CLIPType.FLUX (t5xxl + clip_l).

    Flux needs BOTH encoders, so --clip_path (t5xxl) and --clip_l_path (clip_l)
    are one unit here. Returns CPU tensors (context [1, seq, 4096], y|None).
    CLIP is unloaded before return so the DiT DualMonitor owns VRAM.
    """
    if not os.path.isfile(clip_path):
        raise FileNotFoundError(f"--clip_path not found: {clip_path}")
    if not os.path.isfile(clip_l_path):
        raise FileNotFoundError(f"--clip_l_path not found: {clip_l_path}")

    saved_argv = _clear_argv_for_comfy()
    try:
        _ensure_comfyui_on_sys_path(comfy_path)
        import comfy.options

        comfy.options.enable_args_parsing(False)
        _install_comfy_optional_stubs()
        _install_torchaudio_stub()

        import comfy.model_management as mm  # noqa: WPS433
        import comfy.sd  # noqa: WPS433

        mm.get_torch_device()
        print(f"  [Flux1 calib] Loading CLIP (FLUX / clip_l + t5xxl)")
        _cpu = torch.device("cpu")
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_l_path, clip_path],
            embedding_directory=None,
            clip_type=comfy.sd.CLIPType.FLUX,
            model_options={"load_device": _cpu, "offload_device": _cpu,
                           "initial_device": _cpu},
        )
        patcher = getattr(clip, "patcher", None)
        if patcher is not None:
            print(
                f"  [Flux1 calib] CLIP device contract: load={patcher.load_device} "
                f"offload={patcher.offload_device}"
            )

        bank: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        for i, prompt in enumerate(prompts):
            tokens = clip.tokenize(prompt)
            conds = clip.encode_from_tokens_scheduled(tokens)
            if not conds:
                raise RuntimeError(
                    f"CLIP encode returned empty conditioning for calib sample {i}"
                )
            cond_t = conds[0][0]
            meta = conds[0][1] if len(conds[0]) > 1 else {}
            if not torch.is_tensor(cond_t):
                raise RuntimeError(
                    f"CLIP cond is not a tensor (sample {i}): {type(cond_t)!r}"
                )
            if cond_t.ndim == 2:
                cond_t = cond_t.unsqueeze(0)
            if cond_t.ndim != 3:
                raise RuntimeError(
                    f"CLIP context expected 3D (B, seq, fused), got shape "
                    f"{tuple(cond_t.shape)} (sample {i})"
                )
            y = None
            if isinstance(meta, dict):
                pooled = meta.get("pooled_output")
                if torch.is_tensor(pooled):
                    y = pooled.detach().float().cpu()
                    if y.ndim == 1:
                        y = y.unsqueeze(0)
            bank.append(
                (
                    cond_t.detach().to(dtype=torch.bfloat16).cpu(),
                    y,
                )
            )
            print(
                f"  [Flux1 calib] CLIP encoded {i + 1}/{len(prompts)} "
                f"shape={tuple(cond_t.shape)} y={'none' if y is None else tuple(y.shape)}"
            )

        if getattr(clip, "cond_stage_model", None) is not None:
            clip.cond_stage_model.cpu()
        if patcher is not None:
            mm.unload_model_and_clones(patcher)
        del clip
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  [Flux1 calib] CLIP unloaded; real contexts on CPU for DiT DualMonitor")
        return bank
    finally:
        _restore_argv(saved_argv)


def load_flux_from_safetensors(path, device="cuda", comfy_path: str | None = None):
    """Load the Flux1 DiT through ComfyUI + identity pack-key -> module.weight map."""
    if str(device).startswith("cpu"):
        raise RuntimeError(
            "load_flux_from_safetensors refused device='cpu'. "
            "Flux1 Card 1 DualMonitor calibration requires CUDA."
        )
    _ensure_comfyui_on_sys_path(comfy_path)
    saved_argv = _clear_argv_for_comfy()
    try:
        import comfy.options

        comfy.options.enable_args_parsing(False)
        _install_comfy_optional_stubs()
        import comfy.model_management as mm  # noqa: WPS433
        import comfy.sd  # noqa: WPS433

        print(f"Loading Flux1 DiT (ComfyUI native): {path}")
        patcher = comfy.sd.load_diffusion_model(os.path.abspath(path), {})
        mm.get_torch_device()
        # The 23.8 GB fp16 DiT does not fit a 16 GiB card: let ComfyUI decide the
        # offload (it streams layers in and out per forward).
        mm.load_models_gpu([patcher])
        dm = patcher.model.diffusion_model
        dm.eval()

        # Key names only (safe_open header): loading the 23.8 GB state dict here
        # would double the RAM the DiT already occupies.
        with safe_open(os.path.abspath(path), "pt") as fh:
            file_keys = set(fh.keys())
            shapes = {
                k: tuple(fh.get_slice(k).get_shape())
                for k in file_keys
                if k.endswith(".weight")
            }
        prefix = _find_flux_key_prefix(file_keys)
        comfyui_to_module_map = {}
        for name, mod in dm.named_modules():
            w = getattr(mod, "weight", None)
            if w is None or not torch.is_tensor(w):
                continue
            if len(shapes.get(f"{prefix}{name}.weight", ())) not in (2, 4):
                continue
            ck = f"{prefix}{name}.weight"
            if ck in file_keys:
                comfyui_to_module_map[ck] = f"{name}.weight"
        print(
            f"  [Flux1] identity map entries={len(comfyui_to_module_map)} "
            f"(prefix={prefix!r}, header keys={len(file_keys)})"
        )
        return dm, comfyui_to_module_map, prefix
    finally:
        _restore_argv(saved_argv)


# ---------------------------------------------------------------------------
# Card 1 calib
# ---------------------------------------------------------------------------
def run_card1_calib(
    *,
    input_path: str,
    calib_file: str,
    clip_path: str,
    clip_l_path: str,
    num_calib_samples: int,
    num_inference_steps: int,
    device: str,
    latent_size: int = 32,
    comfy_path: str | None = None,
):
    """Card 1 only: CLIPType.FLUX contexts + DualMonitor -> channel_act_mean."""
    global _dm_timestep_weight
    if not str(device).startswith("cuda"):
        raise RuntimeError("Card 1 Flux1 calib requires CUDA.")

    print("Preparing calibration data...")
    with open(calib_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    if len(prompts) < num_calib_samples:
        prompts = (prompts * (num_calib_samples // len(prompts) + 1))[
            :num_calib_samples
        ]
    else:
        prompts = prompts[:num_calib_samples]

    context_bank = _encode_flux_calib_contexts(
        clip_path=clip_path,
        clip_l_path=clip_l_path,
        prompts=prompts,
        comfy_path=comfy_path,
    )
    if len(context_bank) != len(prompts):
        raise RuntimeError(
            f"CLIP context bank size {len(context_bank)} != calib prompts {len(prompts)}"
        )

    model, comfyui_to_module_map, _prefix = load_flux_from_safetensors(
        input_path, device=device, comfy_path=comfy_path
    )

    print("Preparing calibration (DualMonitor hooks; Card 1 act means)...")
    dual_monitors.clear()
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(
                module.register_forward_hook(
                    lambda m, i, o, n=name: hook_fn(m, i, o, n)
                )
            )

    print(
        f"Running calibration ({num_calib_samples} samples, "
        f"{num_inference_steps} steps)..."
    )
    if num_calib_samples != 32 or num_inference_steps != 25:
        print(
            "  [WARN] How-to / r32 recipe is num_calib_samples=32, "
            "num_inference_steps=25. current args differ."
        )
    _calib_progress_disable = False
    guidance = torch.full((1,), 3.5, device=device, dtype=torch.float32)
    gen = torch.Generator(device=device).manual_seed(42)

    for i, prompt in enumerate(prompts):
        seed = 42 + i
        print(f"\nSample {i+1}/{num_calib_samples}: {prompt[:50]}...")
        gen.manual_seed(seed)
        with torch.no_grad():
            x = torch.randn(
                1,
                16,
                int(latent_size),
                int(latent_size),
                device=device,
                dtype=torch.bfloat16,
                generator=gen,
            )
            ctx_cpu, y_cpu = context_bank[i]
            context = ctx_cpu.to(device=device, dtype=torch.bfloat16)
            y = None
            if y_cpu is not None:
                y = y_cpu.to(device=device, dtype=torch.bfloat16)
            for step in tqdm(
                range(int(num_inference_steps)),
                total=int(num_inference_steps),
                disable=_calib_progress_disable,
            ):
                t = torch.full(
                    (1,),
                    float(step) / float(max(num_inference_steps, 1)),
                    device=device,
                    dtype=torch.float32,
                )
                _dm_timestep_weight = float(1.0 - t.item())
                model(x, t, context, y, guidance=guidance)
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    for h in handles:
        h.remove()

    act_mean_dict = {}
    act_sq_mean_dict = {}
    importance_dict = {}
    for name, mon in dual_monitors.items():
        if mon.channel_act_mean is not None:
            act_mean_dict[name] = mon.channel_act_mean.detach().float().cpu()
        if mon.channel_act_sq_mean is not None:
            act_sq_mean_dict[name] = mon.channel_act_sq_mean.detach().float().cpu()
        if mon.channel_importance is not None:
            importance_dict[name] = mon.channel_importance.detach().float().cpu()
    print(
        f"  [Card 1 DualMonitor] act_mean layers={len(act_mean_dict)}, "
        f"act_sq_mean layers={len(act_sq_mean_dict)}, "
        f"importance layers={len(importance_dict)} "
        f"(full Card 1; no VETO; no Approach A)"
    )

    del model
    del context_bank
    dual_monitors.clear()
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "act_mean_dict": act_mean_dict,
        "act_sq_mean_dict": act_sq_mean_dict,
        "importance_dict": importance_dict,
        "comfyui_to_module_map": comfyui_to_module_map,
    }


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------


def parse_args():
    ap = argparse.ArgumentParser(description="Reverse hybrid W4A8 converter (Flux1 INT8 -> W4A8)")
    ap.add_argument("k", type=int, help="number of lowest-impact layers to convert to W4A8")
    ap.add_argument("out_name", help="output filename, e.g. <model>_hybrid_w{K}_convrot_w4a8.safetensors")
    ap.add_argument("src", help="all-INT8 ConvRot safetensors (rotated)")
    ap.add_argument("impact", help="impact json from diag_impact.py")
    ap.add_argument("--out-dir", default=".", help="output directory")
    ap.add_argument(
        "--bias_correction",
        action="store_true",
        help=(
            "Card 1 ON: DualMonitor consumer - the W4A8 layers this run lifts "
            "get bias += -(W_q - W) @ mu_x recomputed against their W4A8 weights. "
            "Needs --calib_file + --clip_path + --clip_l_path + --comfy_path."
        ),
    )
    ap.add_argument("--calib_file", type=str, default=None,
                    help="Calibration prompts text file (one prompt per line).")
    ap.add_argument("--clip_path", type=str, default=None,
                    help="T5XXL safetensors for CLIPType.FLUX.")
    ap.add_argument("--clip_l_path", type=str, default=None,
                    help="clip_l safetensors for CLIPType.FLUX.")
    ap.add_argument("--comfy_path", type=str, default=None,
                    help="ComfyUI root (must contain comfy/ldm/flux/model.py).")
    ap.add_argument("--num_calib_samples", type=int, default=32,
                    help="DualMonitor calib samples (default 32).")
    ap.add_argument("--num_inference_steps", type=int, default=25,
                    help="DualMonitor DiT timestep sweeps per sample (default 25).")
    ap.add_argument("--latent_size", type=int, default=32,
                    help="Calibration latent H/W (default 32).")
    return ap.parse_args()


def main():
    a = parse_args()
    OUT = os.path.join(a.out_dir, a.out_name)

    act_mean_dict = None
    if a.bias_correction:
        miss = [
            flag
            for flag, val in (
                ("--calib_file", a.calib_file),
                ("--clip_path", a.clip_path),
                ("--clip_l_path", a.clip_l_path),
                ("--comfy_path", a.comfy_path),
            )
            if not val
        ]
        if miss:
            raise SystemExit(
                "Error: --bias_correction requires " + ", ".join(miss)
                + " (there is no DualMonitor-less path)"
            )
        for flag, path in (
            ("--calib_file", a.calib_file),
            ("--clip_path", a.clip_path),
            ("--clip_l_path", a.clip_l_path),
        ):
            if not os.path.isfile(path):
                raise SystemExit(f"Error: {flag} not found: {path}")
        calib = run_card1_calib(
            input_path=a.src,
            calib_file=a.calib_file,
            clip_path=a.clip_path,
            clip_l_path=a.clip_l_path,
            num_calib_samples=int(a.num_calib_samples),
            num_inference_steps=int(a.num_inference_steps),
            device="cuda" if torch.cuda.is_available() else "cpu",
            latent_size=int(a.latent_size),
            comfy_path=a.comfy_path,
        )
        act_mean_dict = calib["act_mean_dict"]

    imp = json.load(open(a.impact, encoding="utf-8"))["impacts"]
    ranked = [k[: -len(".weight")] if k.endswith(".weight") else k
              for k, _ in sorted(imp.items(), key=lambda kv: kv[1])]
    print(f"ranked layers available: {len(ranked)}")

    with safe_open(a.src, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        raw_meta = f.metadata() or {}
        meta_raw_str = raw_meta.get("_quantization_metadata", '{"layers":{}}')
        meta = json.loads(meta_raw_str)
        if "layers" not in meta or not isinstance(meta["layers"], dict):
            meta["layers"] = {}
        sd = {k: f.get_tensor(k) for k in keys}

    prefix = "model.diffusion_model." if any(
        k.startswith("model.diffusion_model.") for k in keys
    ) else ""

    n_conv = 0
    bias_applied = 0
    bias_no_act = 0
    bias_bad_shape = 0
    n_skip_boundary = 0
    n_skip_not_256 = 0
    for L in ranked[: a.k]:
        L_stripped = L
        for _p in ("model.diffusion_model.", "diffusion_model."):
            if L_stripped.startswith(_p):
                L_stripped = L_stripped[len(_p):]
                break

        if _is_boundary(L_stripped):
            n_skip_boundary += 1
            continue

        wk = prefix + L_stripped + ".weight"
        sk = prefix + L_stripped + ".weight_scale"
        if wk not in sd:
            wk = L + ".weight"
            sk = L + ".weight_scale"
        if wk not in sd or sk not in sd:
            continue
        base_k = wk[: -len(".weight")]

        # Only ConvRot INT8 layers with group 256 can become W4A8 (CUDA requant gate).
        layer_conf = meta["layers"].get(base_k) or meta["layers"].get(L_stripped) or {}
        if (not layer_conf.get("convrot")) or int(
            layer_conf.get("convrot_groupsize", 0)
        ) != _CONVROT_GROUPSIZE:
            n_skip_not_256 += 1
            continue

        q = sd[wk]            # I8 rotated
        s = sd[sk]            # F32 [out,1] (or scalar)
        dq_rot = q.float() * s  # W_rot approx in fp32
        w_fp32 = _unrotate(dq_rot, _CONVROT_GROUPSIZE)

        qdata, s_rel, s_channel, _correction, codebook = ck.quantize_w4a8_int8_weight(
            w_fp32,
            group_size=_QUANT_GROUP_SIZE,
            convrot_groupsize=_CONVROT_GROUPSIZE,
            symmetric=True,
            scale_dtype=torch.float8_e4m3fn,
            codebook=True,
            bits=4,
        )

        del sd[wk], sd[sk]
        sd[base_k + ".weight"] = qdata.to(torch.int8)
        sd[base_k + ".weight_s_rel"] = s_rel.to(torch.float8_e4m3fn).contiguous()
        sd[base_k + ".weight_s_channel"] = s_channel.float().contiguous()
        if codebook is not None:
            sd[base_k + ".weight_codebook"] = codebook.float().contiguous()

        if a.bias_correction and act_mean_dict is not None:
            bias_key = base_k + ".bias"
            act_mean = act_mean_dict.get(L_stripped)
            if act_mean is None:
                act_mean = act_mean_dict.get(base_k)
            if act_mean is None and bias_key in sd:
                for _pfx in ("model.diffusion_model.", "diffusion_model."):
                    if L_stripped.startswith(_pfx):
                        act_mean = act_mean_dict.get(L_stripped[len(_pfx):])
                        if act_mean is not None:
                            break
            if act_mean is None or bias_key not in sd:
                bias_no_act += 1
            else:
                # weight_dq is the physical [N, K] W4A8 weight (kitchen un-rotates),
                # the same basis as w_fp32 and as the DualMonitor act mean.
                w_dq = ck.dequantize_w4a8_int8_weight(
                    qdata,
                    s_rel,
                    s_channel,
                    codebook=codebook,
                    correction=None,
                    group_size=_QUANT_GROUP_SIZE,
                    convrot_groupsize=_CONVROT_GROUPSIZE,
                    output_dtype=torch.float32,
                )
                delta = compute_int8_bias_delta(w_fp32, w_dq, act_mean)
                if delta is None:
                    bias_bad_shape += 1
                else:
                    orig_bias = sd[bias_key]
                    sd[bias_key] = (
                        orig_bias.float() - delta.to(orig_bias.device)
                    ).to(orig_bias.dtype)
                    bias_applied += 1

        conf = {
            "format": "asym_w4a8_int8",
            "group_size": _QUANT_GROUP_SIZE,
            "convrot_groupsize": _CONVROT_GROUPSIZE,
        }
        sd[base_k + ".comfy_quant"] = torch.frombuffer(
            bytearray(json.dumps(conf, separators=(",", ":")).encode("utf-8")),
            dtype=torch.uint8,
        )
        meta["layers"][base_k] = conf
        meta["layers"][L_stripped] = conf
        n_conv += 1
        print(f"  w4a8: {base_k}  ({tuple(w_fp32.shape)})")

    print(
        f"converted {n_conv} layers to W4A8 "
        f"(skipped boundary={n_skip_boundary}, non-256-group={n_skip_not_256})"
    )
    if a.bias_correction:
        if bias_applied == 0:
            raise SystemExit(
                "Error: --bias_correction produced 0 corrected layers; refusing to "
                "save a pack labelled bias-corrected"
            )
        print(
            f"  [Bias Correction Card 1] applied={bias_applied} "
            f"no_act_mean={bias_no_act} bad_shape={bias_bad_shape}"
        )

    out_meta = {}
    for k, v in raw_meta.items():
        if k == "_quantization_metadata":
            out_meta[k] = json.dumps(meta)
        else:
            out_meta[k] = v.decode("utf-8") if isinstance(v, bytes) else v
    if "_quantization_metadata" not in out_meta:
        out_meta["_quantization_metadata"] = json.dumps(meta)
    save_file(sd, OUT, metadata=out_meta)
    print("saved:", OUT, os.path.getsize(OUT) / 1e9, "GB (decimal)")


if __name__ == "__main__":
    main()

"""Krea2 DiT — FULL ConvRot Kitchen NVFP4 then post-measure ConvRot INT8 protect.

Krea2-only. FATAL if txtfusion.projector + blocks.0.attn.wq signature missing.
SDXL / Diffusers UNet path is not used.

Pipeline (same spirit as native_convert_int8_krea2_2 --keep_sensitive):
  1) Structure blacklist / non-diffusion → bfloat16 keep
  2) float32 DiT weights → keep float32 (never NVFP4 / never INT8)
  3) Remaining Linear 2D → FULL tentative ConvRot Kitchen NVFP4
     (W @ H^T when groupable; plain NVFP4 fallback otherwise)
  4) Optional DualMonitor calib (r32 recipe: 32 samples × 25 steps) for
     activation-weighted ||W_rot-W_q|| / ||W_rot|| ranking
  5) --keep_sensitive N → top-N highest NVFP4 error → ConvRot INT8 protect
     (not BF16 revert; INT8 shelter for layers NVFP4 breaks)

NVFP4 pack (ComfyUI Kitchen TensorCoreNVFP4Layout + ConvRot):
  <layer>.weight / .weight_scale / .weight_scale_2
  metadata: {"format":"nvfp4","convrot":true,"convrot_groupsize":N}
  (plain {"format":"nvfp4"} if no eligible Hadamard group)

ConvRot INT8 protect pack (same as int8protect / INT8 converter):
  <layer>.weight           int8
  <layer>.weight_scale     float32 [out, 1]
  metadata: {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":N}
  (plain int8_tensorwise if no eligible Hadamard group)

Calib (optional, for sensitivity ranking):
  --calib_file + --clip_path (Qwen3-VL-4B / CLIPType.KREA2)
  Defaults follow the r32 How-to recipe: num_calib_samples=32,
  num_inference_steps=25. No Bias Correction Card 1 in this converter.
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
from collections import OrderedDict

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

try:
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout
except ImportError:
    print("Error: comfy_kitchen not found (install in the active venv).")
    sys.exit(1)

_DEFAULT_GROUPSIZE = 256
_MODEL_TYPE = "Krea2"
_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

# Krea2 SingleStreamDiT — structure-sensitive layers stay BF16.
_KREA2_BLACKLIST: list[str] = [
    "first",
    "last",
    "mod.",
    "norm",
    "projector",
    "tmlp",
    "tproj",
    "bias",
    "vae.",
    "text_encoders",
]

_NON_DIFFUSION_MARKERS: tuple[str, ...] = (
    "conditioner.",
    "cond_stage_model.",
    "text_encoders.",
    "text_encoder.",
    "text_encoder_2.",
    "text_encoder_3.",
    "text_model.",
    "text_projection",
    "logit_scale",
    "clip_l.",
    "clip_g.",
    "t5xxl.",
    "first_stage_model.",
    "vae.",
)


# ---------------------------------------------------------------------------
# DualMonitor (Card 1) — self-contained; no import of hswq_convert_nvfp4_krea2
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

    def update(self, input_tensor, output_tensor, module=None):
        with torch.no_grad():
            out_detached = output_tensor.detach().float()
            out_clamped = torch.clamp(out_detached, -65504.0, 65504.0)
            mean_val = out_clamped.mean().item()
            sq_mean_val = (out_clamped ** 2).mean().item()

            if math.isfinite(mean_val) and math.isfinite(sq_mean_val):
                self.output_sum += mean_val
                self.output_sq_sum += sq_mean_val
            inp_detached = input_tensor.detach().float()
            # Conv2d NCHW vs Linear last-dim (Krea2 projector is 4D [B,L,D,N]).
            is_conv2d = isinstance(module, torch.nn.Conv2d)
            if is_conv2d and inp_detached.dim() == 4:
                reduce_dims = (0, 2, 3)
            elif inp_detached.dim() >= 2:
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
            if self.channel_importance is None:
                self.channel_importance = current_imp
                self.channel_act_mean = current_act
                self.channel_act_sq_mean = current_sq
            elif current_imp.shape == self.channel_importance.shape:
                self.channel_importance = (
                    self.channel_importance * self.count + current_imp
                ) / (self.count + 1)
                self.channel_act_mean = (
                    self.channel_act_mean * self.count + current_act
                ) / (self.count + 1)
                self.channel_act_sq_mean = (
                    self.channel_act_sq_mean * self.count + current_sq
                ) / (self.count + 1)
            self.count += 1


dual_monitors: dict[str, DualMonitor] = {}


def hook_fn(module, input, output, name):
    if name not in dual_monitors:
        dual_monitors[name] = DualMonitor()
    dual_monitors[name].update(input[0], output, module)


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


# ---------------------------------------------------------------------------
# ConvRot Hadamard (self-contained — comfy_kitchen / INT8 ConvRot compatible)
# ---------------------------------------------------------------------------
def build_hadamard(
    size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
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
        dtype=dtype,
        device=device,
    )
    h_matrix = h4
    current_size = 4
    while current_size < size:
        h_matrix = torch.kron(h_matrix, h4)
        current_size *= 4
    h_matrix = h_matrix / (size**0.5)
    _HADAMARD_CACHE[cache_key] = h_matrix
    return h_matrix


def convrot_group_size_for_features(
    n: int, preferred: int = _DEFAULT_GROUPSIZE
) -> int | None:
    """Largest power-of-4 group size <= preferred that divides n (or None)."""
    if n < 4:
        return None
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def rotate_weight(
    weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Linear: W_rot = W @ H^T (group-wise). Matches kitchen ConvRot."""
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features {in_features} not divisible by group_size {group_size}"
        )
    group_count = in_features // group_size
    weight_grouped = weight.view(out_features, group_count, group_size)
    return torch.matmul(
        weight_grouped, h_matrix.T.to(dtype=weight.dtype, device=weight.device)
    ).reshape(weight.shape)


def rotate_weight_conv2d(
    weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Conv2d: rotate along in_channels. weight (O, I, kH, kW)."""
    if weight.ndim != 4:
        raise ValueError(f"Conv2d weight must be 4D, got ndim={weight.ndim}")
    out_c, in_c, k_h, k_w = weight.shape
    flat = weight.permute(0, 2, 3, 1).contiguous().view(-1, in_c)
    flat_rot = rotate_weight(flat, h_matrix, group_size)
    return flat_rot.view(out_c, k_h, k_w, in_c).permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _encode_comfy_quant(config: dict) -> torch.Tensor:
    return torch.tensor(
        list(json.dumps(config, separators=(",", ":")).encode("utf-8")),
        dtype=torch.uint8,
    )


def _is_non_diffusion_key(key: str) -> bool:
    return any(marker in key for marker in _NON_DIFFUSION_MARKERS)


def _find_krea2_key_prefix(state_dict) -> str:
    for prefix in ("model.diffusion_model.", "diffusion_model.", ""):
        if f"{prefix}txtfusion.projector.weight" in state_dict:
            if f"{prefix}blocks.0.attn.wq.weight" not in state_dict:
                raise ValueError(
                    "Krea2 signature incomplete: txtfusion.projector present but "
                    f"{prefix}blocks.0.attn.wq.weight missing"
                )
            return prefix
    raise ValueError(
        "Not a Krea2 checkpoint: missing txtfusion.projector.weight "
        "(under model.diffusion_model. / diffusion_model. / root)."
    )


def _meta_base_key(base_k_file: str) -> str:
    if "model.diffusion_model." in base_k_file:
        return base_k_file.split("model.diffusion_model.")[-1]
    if "diffusion_model." in base_k_file:
        return base_k_file.split("diffusion_model.")[-1]
    return base_k_file


def _is_blacklisted(key: str) -> bool:
    return any(name in key for name in _KREA2_BLACKLIST)


def pack_tensorwise(weight: torch.Tensor):
    """Symmetric per-tensor INT8: scale = amax / 127."""
    w = weight.float()
    amax = max(float(w.abs().max().item()), 1e-6)
    scale = amax / 127.0
    q = (w / scale).round().clamp(-127, 127).to(torch.int8)
    return q, torch.tensor(scale, dtype=torch.float32)


def pack_channelwise(weight: torch.Tensor):
    """Per-out-channel INT8 (Card 3 / ConvRot kitchen dequant shape)."""
    w = weight.float()
    reduce_dims = tuple(range(1, w.dim()))
    amax = torch.clamp(w.abs().amax(dim=reduce_dims).reshape(-1), min=1e-6)
    scale = amax / 127.0
    if w.dim() == 4:
        scale_view = scale.view(-1, 1, 1, 1)
        amax_view = amax.view(-1, 1, 1, 1)
    elif w.dim() == 2:
        scale_view = scale.view(-1, 1)
        amax_view = amax.view(-1, 1)
    else:
        raise ValueError(f"unsupported weight ndim={w.dim()} for channelwise INT8")
    clamped = torch.clamp(w, -amax_view, amax_view)
    q = (clamped / scale_view).round().clamp(-127, 127).to(torch.int8)
    return q, scale_view.to(dtype=torch.float32)


# ---------------------------------------------------------------------------
# ComfyUI bootstrap (CLIPType.KREA2 + SingleStreamDiT)
# ---------------------------------------------------------------------------
def _ensure_comfyui_on_sys_path(comfy_path: str | None = None) -> str:
    candidates = []
    if comfy_path:
        candidates.append(os.path.abspath(comfy_path))
    env = os.environ.get("COMFYUI_PATH")
    if env:
        candidates.append(env)
    candidates.extend(
        [
            r"D:\USERFILES\ComfyUI\ComfyUI",
            r"D:\USERFILES\GitHub\ComfyUI",
            os.path.join(_script_dir(), "ComfyUI-master"),
        ]
    )
    for root in candidates:
        if not root:
            continue
        model_py = os.path.join(root, "comfy", "ldm", "krea2", "model.py")
        if os.path.isfile(model_py):
            if root not in sys.path:
                sys.path.insert(0, root)
            return root
    raise FileNotFoundError(
        "ComfyUI root with comfy/ldm/krea2/model.py not found. "
        "Pass --comfy_path or set COMFYUI_PATH."
    )


def _clear_argv_for_comfy() -> list[str]:
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    return saved


def _restore_argv(saved: list[str]) -> None:
    sys.argv = saved


def _install_torchaudio_stub() -> None:
    """Prevent real torchaudio from loading during comfy.sd import.

    comfy.sd imports comfy.ldm.lightricks.vae.audio_vae, which does a hard
    ``import torchaudio``. On cloud hosts torch/torchaudio CUDA builds often
    mismatch (e.g. torch 13.2 vs torchaudio 13.0) and abort before CLIP load.
    Krea2 calib only needs CLIPType.KREA2 — never AudioVAE — so replace
    torchaudio in sys.modules with a local stub. Does not touch ComfyUI-master.
    """
    import importlib.machinery

    for key in list(sys.modules):
        if key == "torchaudio" or key.startswith("torchaudio."):
            del sys.modules[key]

    def _stub_mod(name: str, *, is_package: bool = False):
        # transformers uses importlib.util.find_spec("torchaudio"); a ModuleType
        # without __spec__ raises ValueError: torchaudio.__spec__ is None.
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
    """Lightweight stubs (same pattern as hswq_convert_nvfp4_krea2)."""
    # Always stub: real torchaudio may be installed but CUDA-mismatched.
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


def detect_krea2_dit_config(state_dict, key_prefix: str) -> dict:
    """Mirror comfy.model_detection Krea2 branch."""
    head_dim = 128
    first_w = state_dict[f"{key_prefix}first.weight"]
    features = int(first_w.shape[0])
    channels = int(first_w.shape[1] // 4)  # patch=2 → channels * 4
    block_re = re.compile(r"^" + re.escape(key_prefix) + r"blocks\.(\d+)\.")
    layers = 0
    for k in state_dict.keys():
        m = block_re.match(k)
        if m:
            layers = max(layers, int(m.group(1)) + 1)
    if layers <= 0:
        raise ValueError("Krea2 detect failed: no blocks.* keys")
    wq = state_dict[f"{key_prefix}blocks.0.attn.wq.weight"]
    wk = state_dict[f"{key_prefix}blocks.0.attn.wk.weight"]
    txtlayers = int(state_dict[f"{key_prefix}txtfusion.projector.weight"].shape[1])
    txtdim = int(
        state_dict[f"{key_prefix}txtfusion.layerwise_blocks.0.prenorm.scale"].shape[0]
    )
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


def _encode_krea2_calib_contexts(
    *,
    clip_path: str,
    prompts: list[str],
    expected_fused: int,
    comfy_path: str | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
    """Encode calib prompts with Comfy CLIPType.KREA2 (Qwen3-VL-4B).

    Returns CPU tensors: (context [1, seq, txtlayers*txtdim], attention_mask|None).
    CLIP is unloaded before return so DiT DualMonitor can own VRAM.
    """
    if not os.path.isfile(clip_path):
        raise FileNotFoundError(f"--clip_path not found: {clip_path}")

    saved_argv = _clear_argv_for_comfy()
    try:
        _ensure_comfyui_on_sys_path(comfy_path)
        import comfy.options

        comfy.options.enable_args_parsing(False)
        _install_comfy_optional_stubs()
        # Same as NVFP4 / quantize: stub again immediately before comfy.sd
        # (audio_vae hard-imports torchaudio; CUDA mismatch aborts otherwise).
        _install_torchaudio_stub()

        import comfy.model_management as mm  # noqa: WPS433
        import comfy.sd  # noqa: WPS433

        mm.get_torch_device()
        print(f"  [Krea2 calib] Loading CLIP (KREA2 / Qwen3-VL-4B): {clip_path}")
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=None,
            clip_type=comfy.sd.CLIPType.KREA2,
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
            fused = int(cond_t.shape[-1])
            if fused != int(expected_fused):
                raise ValueError(
                    f"CLIP context fused dim {fused} != DiT txtlayers*txtdim="
                    f"{expected_fused}. Use CLIPLoader type krea2 / "
                    f"comfy.sd.CLIPType.KREA2 (Qwen3-VL-4B)."
                )
            attn = None
            if isinstance(meta, dict):
                am = meta.get("attention_mask")
                if torch.is_tensor(am):
                    attn = am.detach().float().cpu()
            bank.append(
                (
                    cond_t.detach().to(dtype=torch.bfloat16).cpu(),
                    attn,
                )
            )
            print(
                f"  [Krea2 calib] CLIP encoded {i + 1}/{len(prompts)} "
                f"shape={tuple(cond_t.shape)}"
            )

        if getattr(clip, "cond_stage_model", None) is not None:
            clip.cond_stage_model.cpu()
        if getattr(clip, "patcher", None) is not None:
            mm.unload_model_and_clones(clip.patcher)
        del clip
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            "  [Krea2 calib] CLIP unloaded; real contexts on CPU for DiT DualMonitor"
        )
        return bank
    finally:
        _restore_argv(saved_argv)


def load_krea2_from_safetensors(path, device="cuda", comfy_path: str | None = None):
    """Load Krea2 SingleStreamDiT + identity Comfy key → module.weight map."""
    if str(device).startswith("cpu"):
        raise RuntimeError(
            "load_krea2_from_safetensors refused device='cpu'. "
            "Krea2 Card 1 DualMonitor calibration requires CUDA."
        )
    _ensure_comfyui_on_sys_path(comfy_path)
    saved_argv = _clear_argv_for_comfy()
    try:
        import comfy.options

        comfy.options.enable_args_parsing(False)
        _install_comfy_optional_stubs()
        import comfy.ops  # noqa: WPS433
        from comfy.ldm.krea2.model import SingleStreamDiT  # noqa: WPS433

        print(f"Loading Krea2 DiT: {path}")
        state_dict = load_file(path)
        prefix = _find_krea2_key_prefix(state_dict)
        cfg = detect_krea2_dit_config(state_dict, prefix)
        print(f"Detected Krea2 DiT config: {cfg}")
        dit_kwargs = {k: v for k, v in cfg.items() if k != "image_model"}
        dtype = torch.bfloat16
        dit = SingleStreamDiT(
            **dit_kwargs,
            device=device,
            dtype=dtype,
            operations=comfy.ops.manual_cast,
        )
        stripped = {}
        for k, v in state_dict.items():
            if prefix and k.startswith(prefix):
                stripped[k[len(prefix) :]] = v
            elif not prefix:
                stripped[k] = v
        missing, unexpected = dit.load_state_dict(stripped, strict=False)
        print(
            f"  [Krea2] load_state_dict missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )
        try:
            dit_dev = str(next(dit.parameters()).device)
        except StopIteration as exc:
            raise RuntimeError("Krea2 DiT has no parameters") from exc
        if not dit_dev.startswith("cuda"):
            raise RuntimeError(
                f"Krea2 DiT landed on {dit_dev!r}, not CUDA. "
                "Refusing DualMonitor calibration."
            )
        print(f"  [Krea2] DiT device={dit_dev}")
        comfyui_to_module_map = {}
        for name, mod in dit.named_modules():
            w = getattr(mod, "weight", None)
            if w is None or not torch.is_tensor(w):
                continue
            if w.ndim not in (2, 4):
                continue
            ck = f"{prefix}{name}.weight"
            if ck in state_dict:
                comfyui_to_module_map[ck] = f"{name}.weight"
        print(
            f"  [Krea2] identity map entries={len(comfyui_to_module_map)} "
            f"(prefix={prefix!r})"
        )
        dit.eval()
        return dit, state_dict, comfyui_to_module_map, prefix
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
    num_calib_samples: int,
    num_inference_steps: int,
    device: str,
    comfy_path: str | None = None,
):
    """Card 1 only: CLIPType.KREA2 contexts + DualMonitor → channel_act_mean.

    Does NOT run Static Profile VETO or V4 FP16 keep.
    """
    if not str(device).startswith("cuda"):
        raise RuntimeError("Card 1 Krea2 calib requires CUDA.")

    print("Preparing calibration data...")
    with open(calib_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    if len(prompts) < num_calib_samples:
        prompts = (prompts * (num_calib_samples // len(prompts) + 1))[
            :num_calib_samples
        ]
    else:
        prompts = prompts[:num_calib_samples]

    sd_peek = load_file(input_path)
    prefix = _find_krea2_key_prefix(sd_peek)
    cfg = detect_krea2_dit_config(sd_peek, prefix)
    fused = int(cfg["txtlayers"]) * int(cfg["txtdim"])
    del sd_peek
    gc.collect()

    context_bank = _encode_krea2_calib_contexts(
        clip_path=clip_path,
        prompts=prompts,
        expected_fused=fused,
        comfy_path=comfy_path,
    )
    if len(context_bank) != len(prompts):
        raise RuntimeError(
            f"CLIP context bank size {len(context_bank)} != "
            f"calib prompts {len(prompts)}"
        )

    model, _state_dict, comfyui_to_module_map, _prefix = load_krea2_from_safetensors(
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
    # Same contract as pipeline.set_progress_bar_config(disable=False)
    # (SDXL / ZIT / NVFP4 1.x): per-sample 25-step tqdm bar.
    _calib_progress_disable = False

    gen = torch.Generator(device=device).manual_seed(42)
    lat_h = lat_w = 32

    for i, prompt in enumerate(prompts):
        seed = 42 + i
        print(f"\nSample {i+1}/{num_calib_samples}: {prompt[:50]}...")
        gen.manual_seed(seed)
        with torch.no_grad():
            x = torch.randn(
                1,
                int(model.channels),
                lat_h,
                lat_w,
                device=device,
                dtype=torch.bfloat16,
                generator=gen,
            )
            ctx_cpu, attn_cpu = context_bank[i]
            context = ctx_cpu.to(device=device, dtype=torch.bfloat16)
            attn_mask = None
            if attn_cpu is not None:
                attn_mask = attn_cpu.to(device=device)
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
                if attn_mask is not None:
                    model(x, t, context, attention_mask=attn_mask)
                else:
                    model(x, t, context)
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    for h in handles:
        h.remove()

    act_mean_dict = {}
    act_sq_mean_dict = {}
    for name, mon in dual_monitors.items():
        if mon.channel_act_mean is not None:
            act_mean_dict[name] = mon.channel_act_mean.detach().float().cpu()
        if mon.channel_act_sq_mean is not None:
            act_sq_mean_dict[name] = mon.channel_act_sq_mean.detach().float().cpu()
    print(
        f"  [Card 1 DualMonitor] act_mean layers={len(act_mean_dict)}, "
        f"act_sq_mean layers={len(act_sq_mean_dict)} "
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
        "comfyui_to_module_map": comfyui_to_module_map,
    }


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------
def _pack_int8_protect(
    w_fp: torch.Tensor,
    *,
    enable_convrot: bool,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """ConvRot INT8 protect pack for a Linear 2D float weight."""
    used_gs = None
    if enable_convrot:
        used_gs = convrot_group_size_for_features(int(w_fp.shape[1]), group_size)
    if used_gs is not None:
        h_matrix = build_hadamard(used_gs, device="cpu", dtype=torch.float32)
        w_rot = rotate_weight(w_fp, h_matrix, used_gs)
        q, scale = pack_channelwise(w_rot)
        quant_config = {
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": int(used_gs),
        }
    else:
        q, scale = pack_tensorwise(w_fp)
        quant_config = {"format": "int8_tensorwise"}
    return q, scale, quant_config


def _nvfp4_rel_err(
    w_fp: torch.Tensor,
    w_dq: torch.Tensor,
    *,
    act_sq: torch.Tensor | None,
) -> float:
    """Relative Frobenius error; optional act-weighted when E[x²] matches in_features."""
    err = w_fp - w_dq
    if act_sq is not None and act_sq.shape[0] == w_fp.shape[1]:
        act_scale = act_sq.sqrt().to(device=err.device, dtype=err.dtype)
        weighted_err = err * act_scale.unsqueeze(0)
        weighted_base = w_fp * act_scale.unsqueeze(0)
        return float(weighted_err.norm().item()) / max(
            float(weighted_base.norm().item()), 1e-8
        )
    return float(err.norm().item()) / max(float(w_fp.norm().item()), 1e-8)


def convert_to_nvfp4(
    input_path: str,
    output_path: str,
    *,
    device: str | None = None,
    calib_file: str | None = None,
    clip_path: str | None = None,
    comfy_path: str | None = None,
    num_calib_samples: int = 32,
    num_inference_steps: int = 25,
    enable_convrot: bool = True,
    group_size: int = _DEFAULT_GROUPSIZE,
    keep_sensitive: int = 0,
):
    """FULL ConvRot Kitchen NVFP4 then optional post-measure ConvRot INT8 protect."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    act_sq_mean_dict: dict[str, torch.Tensor] = {}
    comfyui_to_module_map: dict[str, str] = {}
    n_nvfp4 = 0
    n_nvfp4_convrot = 0
    n_plain_nvfp4 = 0
    n_bf16 = 0
    n_fp32 = 0
    n_skipped = 0
    n_int8_protect = 0
    n_int8_convrot = 0
    n_int8_plain = 0

    print(
        f"Mode {_MODEL_TYPE} | device={device} | "
        f"FULL ConvRot Kitchen NVFP4 + post-measure ConvRot INT8 protect"
    )
    if enable_convrot:
        print(
            f"  [ConvRot] ON for NVFP4 path and INT8 protect | "
            f"preferred groupsize={group_size} (power-of-4 adaptive)"
        )
    else:
        print(
            "  [ConvRot] OFF | plain Kitchen NVFP4 + plain int8_tensorwise protect"
        )

    # Optional DualMonitor calib for activation-weighted ranking (r32 recipe).
    run_sens_calib = bool(calib_file) or bool(clip_path)
    if run_sens_calib:
        if not calib_file or not clip_path:
            raise ValueError(
                "Sensitivity calib requires both --calib_file and --clip_path "
                "(Qwen3-VL-4B / CLIPType.KREA2)."
            )
        if not os.path.isfile(calib_file):
            raise FileNotFoundError(f"calib_file not found: {calib_file}")
        if not os.path.isfile(clip_path):
            raise FileNotFoundError(f"clip_path not found: {clip_path}")
        if device != "cuda":
            raise RuntimeError("Krea2 DualMonitor calib requires CUDA.")
        print(
            "  [Sensitivity calib] DualMonitor ON | "
            f"r32 defaults samples={num_calib_samples} steps={num_inference_steps} | "
            "act_sq_mean for weighted ||W_rot-W_q|| ranking | no Bias Correction"
        )
        calib = run_card1_calib(
            input_path=input_path,
            calib_file=calib_file,
            clip_path=clip_path,
            num_calib_samples=int(num_calib_samples),
            num_inference_steps=int(num_inference_steps),
            device=device,
            comfy_path=comfy_path,
        )
        act_sq_mean_dict = calib["act_sq_mean_dict"]
        comfyui_to_module_map = calib["comfyui_to_module_map"]
        print(
            f"  [Sensitivity calib] Captured act sq means for "
            f"{len(act_sq_mean_dict)} layers"
        )

    print(f"Loading model: {input_path}")
    state_dict = load_file(input_path)
    prefix = _find_krea2_key_prefix(state_dict)
    print(f"Detected Krea2 key prefix: {prefix!r}")

    new_state_dict: dict[str, torch.Tensor] = {}
    quant_meta_layers: dict[str, dict] = {}
    layer_quant_errors: dict[str, float] = {}
    # Original float Linear weights for later INT8 protect (key → cpu float).
    orig_fp_for_protect: dict[str, torch.Tensor] = {}

    print(
        f"Converting ({len(state_dict)} tensors) → tentative FULL ConvRot "
        f"Kitchen NVFP4 (2D Linear)..."
    )

    for key, tensor in tqdm(list(state_dict.items())):
        if _is_blacklisted(key) or _is_non_diffusion_key(key):
            new_state_dict[key] = tensor.to(dtype=torch.bfloat16)
            n_bf16 += 1
            continue

        under_prefix = (not prefix) or key.startswith(prefix)

        # fp32 layers are precision-critical — keep as float32, never quantize.
        if (
            under_prefix
            and key.endswith(".weight")
            and tensor.ndim in (2, 4)
            and tensor.dtype == torch.float32
        ):
            new_state_dict[key] = tensor
            n_fp32 += 1
            continue

        is_nvfp4_candidate = (
            under_prefix
            and key.endswith(".weight")
            and tensor.ndim == 2
            and tensor.dtype in (torch.float16, torch.bfloat16)
        )
        if not is_nvfp4_candidate:
            if key.endswith(".weight") and tensor.ndim == 2:
                new_state_dict[key] = tensor.to(dtype=torch.bfloat16)
                n_bf16 += 1
            else:
                new_state_dict[key] = (
                    tensor.to(dtype=torch.bfloat16)
                    if tensor.is_floating_point()
                    else tensor
                )
                n_skipped += 1
            continue

        base_k_file = key[: -len(".weight")]
        base_k_meta = _meta_base_key(base_k_file)
        w_fp = tensor.float().cpu()
        orig_fp_for_protect[key] = w_fp

        v_tensor = tensor.to(device=device, dtype=torch.bfloat16)
        used_gs = None
        do_rotate = False
        w_for_q = v_tensor
        w_ref_fp = w_fp  # error vs the tensor that was actually quantized
        if enable_convrot:
            used_gs = convrot_group_size_for_features(
                int(v_tensor.shape[1]), int(group_size)
            )
            if used_gs is not None:
                h_matrix = build_hadamard(
                    int(used_gs), device="cpu", dtype=torch.float32
                )
                w_rot = rotate_weight(w_fp, h_matrix, int(used_gs))
                w_for_q = w_rot.to(device=device, dtype=torch.bfloat16)
                w_ref_fp = w_rot
                do_rotate = True

        try:
            qdata, params = TensorCoreNVFP4Layout.quantize(w_for_q)
            packed = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)
            for suffix, pt in packed.items():
                new_state_dict[f"{base_k_file}.weight{suffix}"] = pt.cpu()
            if do_rotate and used_gs is not None:
                quant_meta_layers[base_k_meta] = {
                    "format": "nvfp4",
                    "convrot": True,
                    "convrot_groupsize": int(used_gs),
                }
                n_nvfp4_convrot += 1
            else:
                quant_meta_layers[base_k_meta] = {"format": "nvfp4"}
                n_plain_nvfp4 += 1
            n_nvfp4 += 1

            w_dq = TensorCoreNVFP4Layout.dequantize(qdata, params).float().cpu()
            module_w_key = comfyui_to_module_map.get(key)
            module_name = None
            if module_w_key and module_w_key.endswith(".weight"):
                module_name = module_w_key[: -len(".weight")]
            act_sq = (
                act_sq_mean_dict.get(module_name) if module_name is not None else None
            )
            layer_quant_errors[key] = _nvfp4_rel_err(w_ref_fp, w_dq, act_sq=act_sq)
        except Exception:
            new_state_dict[key] = tensor.to(dtype=torch.bfloat16)
            n_bf16 += 1
            orig_fp_for_protect.pop(key, None)
        finally:
            if device == "cuda":
                if do_rotate:
                    del w_for_q
                del v_tensor

    # --- Post-measure: top-N NVFP4-error layers → ConvRot INT8 protect ---
    if keep_sensitive > 0 and layer_quant_errors:
        sorted_errs = sorted(
            layer_quant_errors.items(), key=lambda x: x[1], reverse=True
        )
        protect_keys = sorted_errs[: keep_sensitive]
        print(
            f"\n[Sensitivity] Protecting top {len(protect_keys)} highest NVFP4-error "
            f"layers with ConvRot INT8 "
            f"(enable_convrot={enable_convrot}):"
        )
        for rk, rerr in protect_keys:
            mk = rk[: -len(".weight")]
            w_orig = orig_fp_for_protect.get(rk)
            if w_orig is None:
                print(f"    SKIP {rk} (no float original)")
                continue
            # Drop NVFP4 pack artifacts
            for suffix in ("", "_scale", "_scale_2"):
                pk = f"{mk}.weight{suffix}"
                if pk in new_state_dict:
                    del new_state_dict[pk]
            cq = f"{mk}.comfy_quant"
            if cq in new_state_dict:
                del new_state_dict[cq]
            meta_bk = _meta_base_key(mk)
            prev_meta = quant_meta_layers.pop(meta_bk, None)
            if prev_meta is not None:
                if prev_meta.get("convrot"):
                    n_nvfp4_convrot = max(0, n_nvfp4_convrot - 1)
                else:
                    n_plain_nvfp4 = max(0, n_plain_nvfp4 - 1)

            q, scale, quant_config = _pack_int8_protect(
                w_orig,
                enable_convrot=enable_convrot,
                group_size=int(group_size),
            )
            new_state_dict[rk] = q
            new_state_dict[f"{mk}.weight_scale"] = scale
            quant_meta_layers[meta_bk] = dict(quant_config)
            n_nvfp4 = max(0, n_nvfp4 - 1)
            n_int8_protect += 1
            if quant_config.get("convrot"):
                n_int8_convrot += 1
            else:
                n_int8_plain += 1
            print(
                f"    {rk}  nvfp4_rel_err={rerr:.6f}  "
                f"→ int8{'+convrot' if quant_config.get('convrot') else ''}"
            )

        remaining = sorted_errs[keep_sensitive:]
        if remaining:
            print(
                f"  [Sensitivity] Next worst (stays NVFP4): {remaining[0][0]} "
                f"rel_err={remaining[0][1]:.6f}"
            )
            print(
                f"  [Sensitivity] Best (stays NVFP4): {remaining[-1][0]} "
                f"rel_err={remaining[-1][1]:.6f}"
            )
    elif keep_sensitive > 0:
        print(
            "\n[Sensitivity] keep_sensitive>0 but no NVFP4 error ranks "
            "(no eligible layers)."
        )

    final_metadata = OrderedDict()
    final_metadata["_quantization_metadata"] = json.dumps(
        {"format_version": "1.0", "layers": quant_meta_layers}
    )
    final_metadata["converted_by"] = (
        "HSWQ Krea2 ConvRot NVFP4 post-measure ConvRot INT8 protect"
        if enable_convrot
        else "HSWQ Krea2 NVFP4 post-measure INT8 protect"
    )
    final_metadata["converter_url"] = (
        "https://github.com/tritant/ComfyUI_Kitchen_nvfp4_Converter"
    )
    final_metadata["hswq_model"] = "krea2"
    final_metadata["hswq_nvfp4_post_measure"] = "1"
    final_metadata["hswq_nvfp4_convrot"] = "1" if enable_convrot else "0"
    final_metadata["hswq_keep_sensitive"] = str(int(keep_sensitive))
    final_metadata["hswq_int8_protect_n"] = str(n_int8_protect)

    print(f"Saving | Type: {_MODEL_TYPE} | Path: {output_path}")
    save_file(new_state_dict, output_path, metadata=final_metadata)
    total_bytes = os.path.getsize(output_path)
    print(f"Done. Size: {round(total_bytes / (1024**3), 2)} GiB")
    print(
        f"NVFP4 layers in metadata: "
        f"{sum(1 for c in quant_meta_layers.values() if c.get('format') == 'nvfp4')}"
    )
    print(
        f"  nvfp4 packs={n_nvfp4} "
        f"(convrot={n_nvfp4_convrot}, plain={n_plain_nvfp4}) | "
        f"int8 protect={n_int8_protect} "
        f"(convrot={n_int8_convrot}, plain={n_int8_plain}) | "
        f"bf16 keep={n_bf16} | fp32 keep={n_fp32} | other={n_skipped}"
    )
    print(f"Sensitivity calib (DualMonitor): {run_sens_calib}")
    print(f"FULL ConvRot enabled (NVFP4 path + INT8 protect): {enable_convrot}")

    del state_dict
    del new_state_dict
    del quant_meta_layers
    del orig_fp_for_protect
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Krea2 DiT: FULL ConvRot Kitchen NVFP4 then post-measure "
            "ConvRot INT8 protect. Optional DualMonitor calib "
            "(--calib_file + --clip_path) for activation-weighted ranking "
            "(r32 defaults: 32 samples × 25 steps). --keep_sensitive N "
            "replaces top-N NVFP4-error Linear layers with ConvRot INT8. "
            "float32 DiT weights stay float32. No Bias Correction."
        )
    )
    parser.add_argument(
        "--model",
        "--input",
        dest="model",
        type=str,
        required=True,
        help="Path to Krea2 input .safetensors",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output .safetensors"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="NVFP4 quantize device",
    )
    parser.add_argument(
        "--calib_file",
        type=str,
        default=None,
        help=(
            "Calibration prompts (one per line). With --clip_path enables "
            "DualMonitor act_sq ranking (r32 recipe defaults)."
        ),
    )
    parser.add_argument(
        "--clip_path",
        type=str,
        default=None,
        help=(
            "Qwen3-VL-4B CLIP safetensors for Comfy CLIPType.KREA2. "
            "Required with --calib_file for sensitivity weighting."
        ),
    )
    parser.add_argument(
        "--comfy_path",
        type=str,
        default=None,
        help="ComfyUI root (must contain comfy/ldm/krea2/model.py).",
    )
    parser.add_argument(
        "--num_calib_samples",
        type=int,
        default=32,
        help="Sensitivity calib samples (r32 recipe default 32).",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="Sensitivity DiT timestep sweeps per sample (r32 recipe default 25).",
    )
    parser.add_argument(
        "--no-convrot",
        dest="enable_convrot",
        action="store_false",
        help=(
            "Disable ConvRot on both NVFP4 path and INT8 protect "
            "(plain Kitchen NVFP4 + plain int8_tensorwise)."
        ),
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=_DEFAULT_GROUPSIZE,
        help=f"Preferred ConvRot Hadamard group size (default {_DEFAULT_GROUPSIZE}).",
    )
    parser.add_argument(
        "--keep_sensitive",
        type=int,
        default=0,
        help=(
            "After FULL ConvRot NVFP4, replace top N highest NVFP4-error "
            "Linear layers with ConvRot INT8 protect. Ranked by "
            "||W_rot-W_q||/||W_rot|| (act-weighted when DualMonitor calib "
            "runs). 0 = NVFP4 only."
        ),
    )
    parser.set_defaults(enable_convrot=True)
    args = parser.parse_args()

    convert_to_nvfp4(
        args.model,
        args.output,
        device=str(args.device),
        calib_file=args.calib_file,
        clip_path=args.clip_path,
        comfy_path=args.comfy_path,
        num_calib_samples=int(args.num_calib_samples),
        num_inference_steps=int(args.num_inference_steps),
        enable_convrot=bool(args.enable_convrot),
        group_size=int(args.group_size),
        keep_sensitive=int(args.keep_sensitive),
    )


if __name__ == "__main__":
    main()

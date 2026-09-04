"""Krea2 DiT — ComfyUI-native FULL ConvRot INT8 convert + Card 1 Bias Correction.

Krea2-only. FATAL if txtfusion.projector + blocks.0.attn.wq signature missing.
SDXL / Diffusers UNet path is not used.

Pack (ComfyUI MixedPrecisionOps + comfy_kitchen TensorWiseINT8Layout):
  <layer>.weight           int8
  <layer>.weight_scale     float32
      plain INT8:          scalar (tensorwise)  OR  --per_channel_int8 → [O,1] / [O,1,1,1]
      ConvRot Linear:      [out, 1] (row-wise) — kitchen online act rotate
      ConvRot Conv2d:      [out, 1, 1, 1] (per-out-channel)
  <layer>.comfy_quant      uint8 JSON (compact)
      plain:  {"format":"int8_tensorwise"}
      ConvRot:{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":N}

FULL ConvRot (default ON; --no-convrot for plain INT8):
  Linear 2D:  W_rot = W @ H^T, row-wise INT8, stamp.
  Conv2d 4D:  rotate along in_channels, channelwise INT8, stamp.
  Hadamard / rotate_weight / rotate_weight_conv2d live in THIS file
  (same math as comfy_kitchen ConvRot; no import of native_convert_int8.py).
  If in_features / in_channels is not divisible by a power-of-4 group size,
  that layer stays plain tensorwise (or Card 3 channelwise).

BF16 keep (structure-sensitive; same spirit as native_convert_nvfp4_krea2):
  first / last / mod. / norm / projector / tmlp / tproj / bias /
  vae. / text_encoders / non-diffusion markers.

DiT float32 .weight (ndim 2/4): keep as float32 — never INT8
  (precision-critical; independent of structure blacklist).

DiT Linear/Conv2d .weight quantized only when dtype is fp16/bf16.

Card 1 (--bias_correction):
  Requires --calib_file AND --clip_path (Qwen3-VL-4B / CLIPType.KREA2).
  Encode prompts with Comfy CLIP → unload CLIP → load SingleStreamDiT →
  DualMonitor on Linear+Conv2d → bias += -(W_q - W) @ mu_x.
  No Static Profile VETO / no V4 FP16 keep / no SDXL pipeline.
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
from safetensors.torch import load_file, save_file
from tqdm import tqdm

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
    """Offline Linear: W_rot = W @ H^T (group-wise). Matches kitchen ConvRot."""
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
    """Offline Conv2d: rotate along in_channels. weight (O, I, kH, kW)."""
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
    for name, mon in dual_monitors.items():
        if mon.channel_act_mean is not None:
            act_mean_dict[name] = mon.channel_act_mean.detach().float().cpu()
    print(
        f"  [Card 1 DualMonitor] act_mean layers={len(act_mean_dict)} "
        f"(full Card 1; no VETO; no Approach A)"
    )

    del model
    del context_bank
    dual_monitors.clear()
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "act_mean_dict": act_mean_dict,
        "comfyui_to_module_map": comfyui_to_module_map,
    }


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------
def convert_to_int8(
    input_path: str,
    output_path: str,
    *,
    per_channel_int8: bool = False,
    bias_correction: bool = False,
    calib_file: str | None = None,
    clip_path: str | None = None,
    comfy_path: str | None = None,
    num_calib_samples: int = 32,
    num_inference_steps: int = 25,
    enable_convrot: bool = True,
    group_size: int = _DEFAULT_GROUPSIZE,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    act_mean_dict: dict[str, torch.Tensor] = {}
    comfyui_to_module_map: dict[str, str] = {}
    convrot_linear = 0
    convrot_conv2d = 0
    bf16_keep = 0

    print(f"Mode {_MODEL_TYPE} | device={device} | FULL ConvRot INT8 (Krea2-only)")

    if enable_convrot:
        print(
            f"  [ConvRot] FULL ON | Linear + Conv2d "
            f"(preferred groupsize={group_size}; adaptive power-of-4 divisor) "
            f"| Hadamard self-contained in this file"
        )
        if bias_correction:
            print(
                "  [ConvRot] WARN: Card 1 DualMonitor means are from unrotated "
                "float DiT; BC uses rotated W vs W_q (approximate for ConvRot)"
            )

    if bias_correction:
        if not calib_file:
            raise ValueError("--bias_correction requires --calib_file")
        if not clip_path:
            raise ValueError(
                "--bias_correction requires --clip_path "
                "(Qwen3-VL-4B safetensors for Comfy CLIPType.KREA2)"
            )
        if not os.path.isfile(calib_file):
            raise FileNotFoundError(f"calib_file not found: {calib_file}")
        if not os.path.isfile(clip_path):
            raise FileNotFoundError(f"clip_path not found: {clip_path}")
        if device != "cuda":
            raise RuntimeError("Card 1 Krea2 calib requires CUDA.")
        print(
            "  [Bias Correction Card 1] ON | ALL INT8 Linear+Conv | "
            "CLIPType.KREA2 + DualMonitor DiT calib | "
            "mu_x = DualMonitor.channel_act_mean | "
            "bias += -(W_q - W) @ mu_x | "
            "no Approach A / no top_ratio gate"
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
        act_mean_dict = calib["act_mean_dict"]
        comfyui_to_module_map = calib["comfyui_to_module_map"]
        print(
            f"  [Bias Correction] Captured act means for {len(act_mean_dict)} layers"
        )

    print(f"Loading model: {input_path}")
    state_dict = load_file(input_path)
    prefix = _find_krea2_key_prefix(state_dict)
    print(f"Detected Krea2 key prefix: {prefix!r}")

    new_state_dict: dict[str, torch.Tensor] = {}
    quant_meta_layers: dict[str, dict] = {}
    converted_count = 0
    skipped_count = 0
    plain_int8_count = 0
    bias_corr_pending: dict[str, torch.Tensor] = {}
    bias_corr_applied = 0
    bias_corr_skipped_no_bias = 0
    bias_corr_skipped_no_act = 0
    bias_corr_skipped_bad_shape = 0
    mode = "per-channel" if per_channel_int8 else "tensorwise"
    rot_tag = " + ConvRot(Linear+Conv2d)" if enable_convrot else ""
    print(
        f"Converting Krea2 DiT Linear/Conv weights to INT8 "
        f"({mode}{rot_tag}, amax/127)..."
    )

    for key, tensor in tqdm(list(state_dict.items())):
        # Structure-sensitive / non-diffusion → keep original dtype
        if _is_blacklisted(key) or _is_non_diffusion_key(key):
            new_state_dict[key] = tensor
            bf16_keep += 1
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
            bf16_keep += 1
            continue

        is_dit_weight = (
            under_prefix
            and key.endswith(".weight")
            and tensor.ndim in (2, 4)
            and tensor.dtype in (torch.float16, torch.bfloat16)
        )

        if not is_dit_weight:
            new_state_dict[key] = tensor
            skipped_count += 1
            continue

        w_fp = tensor.float()
        used_gs = None
        if enable_convrot:
            used_gs = convrot_group_size_for_features(int(w_fp.shape[1]), group_size)

        if used_gs is not None and tensor.ndim == 2:
            h_matrix = build_hadamard(used_gs, device="cpu", dtype=torch.float32)
            w_fp = rotate_weight(w_fp, h_matrix, used_gs)
            q, scale = pack_channelwise(w_fp)
            quant_config = {
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": int(used_gs),
            }
            convrot_linear += 1
        elif used_gs is not None and tensor.ndim == 4:
            h_matrix = build_hadamard(used_gs, device="cpu", dtype=torch.float32)
            w_fp = rotate_weight_conv2d(w_fp, h_matrix, used_gs)
            q, scale = pack_channelwise(w_fp)
            quant_config = {
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": int(used_gs),
            }
            convrot_conv2d += 1
        elif per_channel_int8:
            q, scale = pack_channelwise(w_fp)
            quant_config = {"format": "int8_tensorwise"}
            plain_int8_count += 1
        else:
            q, scale = pack_tensorwise(w_fp)
            quant_config = {"format": "int8_tensorwise"}
            plain_int8_count += 1

        weight_dq = q.float() * scale
        module_key = key[: -len(".weight")]
        new_state_dict[key] = q
        new_state_dict[f"{module_key}.weight_scale"] = scale
        new_state_dict[f"{module_key}.comfy_quant"] = _encode_comfy_quant(quant_config)
        quant_meta_layers[_meta_base_key(module_key)] = dict(quant_config)
        converted_count += 1

        if bias_correction:
            module_w_key = comfyui_to_module_map.get(key)
            module_name = None
            if module_w_key and module_w_key.endswith(".weight"):
                module_name = module_w_key[: -len(".weight")]
            act_mean = (
                act_mean_dict.get(module_name) if module_name is not None else None
            )
            if act_mean is None:
                bias_corr_skipped_no_act += 1
            else:
                # BC target weight = pre-quant float (rotated when ConvRot).
                delta = compute_int8_bias_delta(w_fp, weight_dq, act_mean)
                if delta is None:
                    bias_corr_skipped_bad_shape += 1
                else:
                    bias_corr_pending[module_key] = (-delta).detach().float().cpu()

    if bias_correction and bias_corr_pending:
        print(
            f"\n[Bias Correction] Applying deltas to {len(bias_corr_pending)} "
            f"INT8 Linear+Conv layers (full Card 1)..."
        )
        for module_key, delta in bias_corr_pending.items():
            bias_key = f"{module_key}.bias"
            if bias_key not in new_state_dict:
                bias_corr_skipped_no_bias += 1
                continue
            bias = new_state_dict[bias_key]
            corrected = bias.float() + delta.to(
                device=bias.device, dtype=torch.float32
            )
            new_state_dict[bias_key] = corrected.to(dtype=bias.dtype)
            bias_corr_applied += 1
        print(
            f"  [Bias Correction] applied={bias_corr_applied}, "
            f"no_bias={bias_corr_skipped_no_bias}, "
            f"no_act={bias_corr_skipped_no_act}, "
            f"bad_shape={bias_corr_skipped_bad_shape}"
        )
    elif bias_correction:
        print(
            f"  [Bias Correction] No deltas pending "
            f"(no_act={bias_corr_skipped_no_act}, "
            f"bad_shape={bias_corr_skipped_bad_shape})"
        )

    metadata = {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": quant_meta_layers}
        )
    }

    print(f"Saving to: {output_path}")
    print(f"Converted layers: {converted_count}, Kept (other): {skipped_count}")
    print(f"BF16 keep (blacklist / non-diffusion): {bf16_keep}")
    print(f"Per-channel INT8 (Card 3 plain packs): {per_channel_int8}")
    print(f"Bias correction (Card 1): {bias_correction}")
    if bias_correction:
        print(f"  Bias-corrected INT8 layers: {bias_corr_applied}")
    print(f"ConvRot FULL (Linear+Conv2d): {enable_convrot}")
    if enable_convrot:
        print(
            f"  ConvRot Linear: {convrot_linear}, ConvRot Conv2d: {convrot_conv2d}, "
            f"plain INT8 (no eligible group size): {plain_int8_count}"
        )
    else:
        print(f"  plain INT8: {plain_int8_count}")

    save_file(new_state_dict, output_path, metadata=metadata)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Krea2 DiT INT8 convert with FULL ConvRot (Linear+Conv2d) ON by default. "
            "Card 1 = --bias_correction + --calib_file + --clip_path (CLIPType.KREA2). "
            "Card 3 = --per_channel_int8 for non-ConvRot plain packs. "
            "Use --no-convrot for plain INT8 only. No Approach A / no VETO / no SDXL."
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
        "--per_channel_int8",
        action="store_true",
        help=(
            "Card 3: per-out-channel amax/scale for plain (non-ConvRot) packs. "
            "ConvRot layers always use channelwise. Format stays int8_tensorwise."
        ),
    )
    parser.add_argument(
        "--bias_correction",
        action="store_true",
        help=(
            "Card 1 ON: CLIPType.KREA2 encode + DualMonitor DiT calib; "
            "bias += -(W_q - W) @ mu_x on ALL INT8 Linear+Conv. "
            "Requires --calib_file and --clip_path."
        ),
    )
    parser.add_argument(
        "--calib_file",
        type=str,
        default=None,
        help="Calibration prompts text file (one prompt per line).",
    )
    parser.add_argument(
        "--clip_path",
        type=str,
        default=None,
        help=(
            "Qwen3-VL-4B CLIP safetensors for Comfy CLIPType.KREA2. "
            "Required with --bias_correction."
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
        help="Card 1 calib samples (default 32).",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="Card 1 DiT timestep sweeps per sample (default 25).",
    )
    parser.add_argument(
        "--no-convrot",
        dest="enable_convrot",
        action="store_false",
        help="Disable ConvRot; pack plain int8_tensorwise only.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=_DEFAULT_GROUPSIZE,
        help=f"Preferred ConvRot Hadamard group size (default {_DEFAULT_GROUPSIZE}).",
    )
    parser.set_defaults(enable_convrot=True)
    args = parser.parse_args()

    convert_to_int8(
        args.model,
        args.output,
        per_channel_int8=bool(args.per_channel_int8),
        bias_correction=bool(args.bias_correction),
        calib_file=args.calib_file,
        clip_path=args.clip_path,
        comfy_path=args.comfy_path,
        num_calib_samples=int(args.num_calib_samples),
        num_inference_steps=int(args.num_inference_steps),
        enable_convrot=bool(args.enable_convrot),
        group_size=int(args.group_size),
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Z Image ConvRot NVFP4 ComfyUI Benchmark
=======================================
Compare FP16/BF16 NextDiT (Z Image / ZIT / Moody) vs Kitchen NVFP4 **with
offline ConvRot** (Hadamard ``W @ H^T`` + ``convrot`` / ``convrot_groupsize``
stamps in ``_quantization_metadata`` → ``.comfy_quant`` after convert_old_quants).

Requires a ConvRot convert output (e.g. ``*_nvfp4_convrot.safetensors`` from
``native_convert_nvfp4_zi.py`` with ConvRot ON). Plain NVFP4 (no stamps) is
rejected. Runtime uses ``nvfp4_comfy_parity``: stock Comfy load + **online act
rotation** on armed Linears (``_hswq_nvfp4_convrot``). Without that path, ConvRot
ckpts collapse to SSIM ~0.04.

Based on ``zi_nvfp4_bench.py`` / ``zi_int8_bench.py``. ComfyUI-master is not
modified. Kitchen without ``rms_rope`` uses ``kitchen_rms_rope_fallback.py``.

Example:
  D:\\USERFILES\\fp8e4m3\\venv\\Scripts\\python.exe zi_convrot_nvfp4_bench_v2.py \\
    --fp16  "...\\moodyProMix_zitV13.safetensors" \\
    --nvfp4 "...\\moodyProMix_zitV13_nvfp4_convrot.safetensors" \\
    --clip_path "...\\qwen3_4b_abliterated_fp16_converted.safetensors" \\
    --comfy_path "D:\\USERFILES\\GitHub\\hswq\\ComfyUI-master" \\
    --vae "...\\Ultra-flux1.vae.safetensors" \\
    --prompt "masterpiece, best quality, 1girl, solo, standing, simple background" \\
    --steps 25
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops
from safetensors.torch import load_file
from skimage.metrics import structural_similarity as ssim

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SSIM_TARGET = 0.9


def _decode_comfy_quant_blob(blob) -> dict | None:
    """Decode uint8 comfy_quant tensor / bytes to a dict, or None."""
    if blob is None:
        return None
    try:
        if hasattr(blob, "detach"):
            raw = bytes(blob.detach().cpu().tolist())
        elif isinstance(blob, (bytes, bytearray)):
            raw = bytes(blob)
        else:
            raw = bytes(blob)
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def count_convrot_in_quant_metadata(metadata: dict | None) -> tuple[int, int, str]:
    """Return (n_convrot, n_nvfp4, hswq_nvfp4_convrot flag string)."""
    meta = metadata or {}
    flag = str(meta.get("hswq_nvfp4_convrot", "") or "")
    raw = meta.get("_quantization_metadata")
    if not raw:
        return 0, 0, flag
    try:
        qmap = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return 0, 0, flag
    layers = (qmap or {}).get("layers") or {}
    n_nv = 0
    n_cr = 0
    for conf in layers.values():
        if not isinstance(conf, dict):
            continue
        fmt = str(conf.get("format", "")).lower()
        if fmt != "nvfp4":
            continue
        n_nv += 1
        if conf.get("convrot") is True or str(conf.get("convrot", "")).lower() in (
            "1",
            "true",
        ):
            n_cr += 1
    return n_cr, n_nv, flag


def count_convrot_comfy_quant_markers(state_dict: dict) -> tuple[int, int]:
    """Count .comfy_quant markers that are nvfp4 / nvfp4+convrot."""
    n_nv = 0
    n_cr = 0
    for k, v in state_dict.items():
        if not k.endswith(".comfy_quant"):
            continue
        conf = _decode_comfy_quant_blob(v)
        if not conf or str(conf.get("format", "")).lower() != "nvfp4":
            continue
        n_nv += 1
        if conf.get("convrot") is True or str(conf.get("convrot", "")).lower() in (
            "1",
            "true",
        ):
            n_cr += 1
    return n_cr, n_nv


def count_armed_convrot_linears(model) -> tuple[int, int]:
    """Return (n_armed_convrot, n_linear_modules) on a loaded NextDiT."""
    n_lin = 0
    n_armed = 0
    for _name, mod in model.named_modules():
        if not hasattr(mod, "weight"):
            continue
        # MixedPrecision Linear / ops Linear typically expose in_features.
        if not hasattr(mod, "in_features"):
            continue
        n_lin += 1
        if getattr(mod, "_hswq_nvfp4_convrot", False):
            n_armed += 1
    return n_armed, n_lin


def require_convrot_parity_forward() -> None:
    """Fail if Linear.forward is not the ConvRot act-rotate parity wrapper."""
    import comfy.ops

    lin_fwd = comfy.ops.mixed_precision_ops().Linear.forward
    if getattr(lin_fwd, "_hswq_nvfp4_full_forward", False):
        raise RuntimeError(
            "ConvRot bench: Linear.forward still has HSWQ TC wrap "
            "(_hswq_nvfp4_full_forward); SSIM would be destroyed"
        )
    if not getattr(lin_fwd, "_hswq_nvfp4_convrot_parity", False):
        raise RuntimeError(
            "ConvRot bench: Linear.forward missing _hswq_nvfp4_convrot_parity "
            "(online act rotation required for offline W@H^T weights)"
        )


def resolve_path(path, is_file=True):
    if not path:
        return None
    if os.path.exists(path):
        return path

    target = os.path.basename(path)
    print(f"  Note: {target} not found at {path}. Searching recursively...")
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        root_abs = os.path.abspath(root)
        if "ComfyUI" in root_abs or "node_modules" in root_abs:
            continue
        if is_file and target in files:
            found = os.path.join(root, target)
            print(f"  Found: {found}")
            return found
    return path


def latent_to_img(l):
    l = l[0].detach().permute(1, 2, 0).cpu().float().numpy()
    l = (l - l.min()) / (l.max() - l.min() + 1e-6) * 255
    return Image.fromarray(l[:, :, :3].astype(np.uint8))


def _clear_argv_for_comfy() -> list[str]:
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    return saved


def _restore_argv(saved: list[str]) -> None:
    sys.argv = saved


def _install_torchaudio_stub() -> None:
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


def setup_comfy(comfy_path: str) -> None:
    comfy_root = Path(comfy_path).resolve()
    if not comfy_root.is_dir():
        raise FileNotFoundError(f"--comfy_path not found: {comfy_root}")
    sys.path = [str(comfy_root)] + [p for p in sys.path if Path(p).resolve() != comfy_root]
    bench_dir = Path(__file__).resolve().parent
    if str(bench_dir) not in sys.path:
        sys.path.insert(0, str(bench_dir))
    # Hybrid NVFP4/INT8 Z Image stack: benchmark/hswq_stack/ (vendored from
    # ComfyUI-HSWQ-Loader-and-Tools; ComfyUI standard does NOT support
    # hybrid ConvRot NVFP4 packs). See hswq_stack/README.md.
    _install_torchaudio_stub()


def apply_nvfp4_patches() -> None:
    """Use the reference hybrid NVFP4 stack (stock GEMM + act rotate).

    Reference: nodes/zimage_nvfp4/load_unet.py applies, in order:
      zi_comfy_quant_nvfp4 (detect/load/LoRA bake)
      nvfp4_comfy_parity (stock Comfy GEMM + online act rotate; TC off)
      require_convrot_parity_forward guard
      comfy_quant_int8 (INT8 tensorwise load)
      zimage_nvfp4_lora_bake (Dynamic ConvRot NVFP4 LoRA bake)
    """
    from hswq_stack.zimage_nvfp4.load_unet import apply_nvfp4_patches as _ref_apply

    _ref_apply()
    print(
        "  [CONVROT] Reference parity stack armed: "
        "stock Comfy GEMM + online act rotate (x @ H)"
    )


def set_hf_token(token: str | None) -> None:
    if not token:
        return
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def load_zit_model(path, device="cuda", comfy_path=None, is_nvfp4=False, require_convrot=False):
    """ComfyUI standard loading: load_diffusion_model_state_dict -> ModelPatcher.

    detect_unet_config / model_config / get_model / load_model_weights are
    handled by ComfyUI (runtime patches in reference hswq_stack handle NVFP4
    packed dims and quant loading without modifying ComfyUI source files).
    Here, only state_dict preprocessing (NVFP4 metadata verification and
    dtype resolution) is performed.
    """
    import comfy.sd
    import comfy.utils

    args_path = resolve_path(path, is_file=True)
    print(f"Loading state_dict: {os.path.basename(args_path)}")

    metadata = None
    if is_nvfp4:
        state_dict, metadata = comfy.utils.load_torch_file(args_path, return_metadata=True)
        n_cr_meta, n_nv_meta, flag = count_convrot_in_quant_metadata(metadata)
        print(
            f"  [CONVROT meta] hswq_nvfp4_convrot={flag!r} "
            f"nvfp4_layers={n_nv_meta} convrot_stamps={n_cr_meta}"
        )
        if require_convrot and n_cr_meta <= 0:
            print(
                "CRITICAL ERROR: This is the ConvRot NVFP4 bench. "
                "Checkpoint has zero convrot stamps in _quantization_metadata."
            )
            print(
                "  Re-convert with native_convert_nvfp4_zi.py (ConvRot ON, default) "
                "to produce e.g. *_nvfp4_convrot.safetensors"
            )
            print(f"  Path: {args_path}")
            sys.exit(1)
        state_dict, metadata = comfy.utils.convert_old_quants(
            state_dict, "", metadata=metadata or {}
        )
        n_cq = sum(1 for k in state_dict if k.endswith(".comfy_quant"))
        n_cr_cq, n_nv_cq = count_convrot_comfy_quant_markers(state_dict)
        print(f"  [NVFP4] convert_old_quants -> {n_cq} .comfy_quant markers")
        print(
            f"  [CONVROT markers] nvfp4={n_nv_cq} with_convrot={n_cr_cq}"
        )
        if require_convrot and n_cr_cq <= 0:
            print(
                "CRITICAL ERROR: After convert_old_quants, no .comfy_quant "
                "markers carry convrot:true. Refusing plain NVFP4."
            )
            sys.exit(1)
    else:
        state_dict = load_file(args_path)

    # Native-dtype gate: OFF (default) = stock behavior (bf16 -> fp16 forced cast).
    # ON (--native-dtype) = keep checkpoint dtype (bf16 models evaluated in bf16).
    counts = {}
    if not bench_use_native:
        native_dtype = torch.float16
    elif is_nvfp4:
        md_src = (metadata or {}).get("hswq_source_dtype") or (metadata or {}).get("source_dtype")
        native_dtype = (
            torch.bfloat16
            if (md_src and ("bfloat16" in str(md_src).lower() or "bf16" in str(md_src).lower()))
            else torch.float16
        )
    else:
        for k, v in state_dict.items():
            if hasattr(v, "dtype") and str(v.dtype) in ("torch.bfloat16", "torch.float16"):
                counts[str(v.dtype)] = counts.get(str(v.dtype), 0) + 1
        native_dtype = (
            torch.bfloat16
            if counts.get("torch.bfloat16", 0) >= counts.get("torch.float16", 0)
            else torch.float16
        )
    print(
        f"  [Native dtype] {native_dtype} "
        f"(bf16={counts.get('torch.bfloat16', 0)} fp16={counts.get('torch.float16', 0)})"
    )
    global bench_native_dtype
    bench_native_dtype = native_dtype

    # ComfyUI standard loading (detect -> model_config -> get_model -> load weights)
    model_options = {}
    if not bench_use_native:
        model_options["dtype"] = torch.float16
    patcher = comfy.sd.load_diffusion_model_state_dict(
        state_dict, model_options=model_options, metadata=metadata
    )
    if patcher is None:
        print(
            "CRITICAL ERROR: ComfyUI could not detect this model "
            "(load_diffusion_model_state_dict returned None)."
        )
        sys.exit(1)

    mc = patcher.model.model_config
    dim = (mc.unet_config or {}).get("dim", "?")
    print(
        f"  [ComfyUI] model={type(patcher.model).__name__} "
        f"config={type(mc).__name__} dim={dim}"
    )
    if is_nvfp4:
        try:
            n_armed, n_lin = count_armed_convrot_linears(patcher.model)
            print(
                f"  [CONVROT armed] Linears with _hswq_nvfp4_convrot: "
                f"{n_armed} / {n_lin}"
            )
            if require_convrot and n_armed <= 0:
                print(
                    "CRITICAL ERROR: ConvRot stamps present but zero Linears armed "
                    "(_hswq_nvfp4_convrot). Load/parity path failed."
                )
                sys.exit(1)
        except Exception as ex:
            print(f"  [CONVROT armed] check skipped: {ex}")
    return patcher, state_dict


def load_zit_clip(model_config, state_dict, clip_path, device):
    """ComfyUI standard CLIP loading (ZImageTokenizer + Qwen3_4B).

    Creates ClipTarget from model_config.clip_target() and loads via comfy.sd.CLIP.
    tokenize + encode_from_tokens_scheduled returns ComfyUI standard conditioning
    format ([cond, pooled]).
    """
    import comfy.sd
    from safetensors.torch import load_file as _lf

    resolved = resolve_path(clip_path, is_file=True)
    print(f"  Loading CLIP weights from: {resolved}")
    clip_target = model_config.clip_target(state_dict)
    clip = comfy.sd.CLIP(clip_target, embedding_directory=None)
    clip.load_sd(_lf(resolved))
    # comfy.sd.CLIP itself does not have .to(). Move cond_stage_model (Qwen3_4B)
    # directly to GPU. encode_from_tokens_scheduled also internally moves to load
    # device via patcher.patch_model() (ComfyUI standard).
    clip.cond_stage_model.to(device)
    return clip


bench_use_native = False  # set from --native-dtype
bench_native_dtype = torch.float16  # set by load_zit_model


def run_inference(patcher, positive, negative, steps, seed, device, cfg=2.5):
    """ComfyUI standard sampling: KSampler (euler / simple) + CFG.

    Returned samples are latents in VAE space after KSampler latent_format.process_latent_out
    (equivalent to ComfyUI standard VAEDecode input).
    """
    import comfy.model_management as mm
    from comfy.samplers import KSampler

    mm.load_models_gpu([patcher], force_full_load=True)
    ks = KSampler(
        patcher, steps=steps, device=device, sampler="euler", scheduler="simple"
    )
    gen = torch.Generator(device).manual_seed(seed)
    noise = torch.randn(1, 16, 128, 128, device=device, generator=gen)
    # ComfyUI standard: KSampler requires latent_image (pass zero latent equivalent
    # to EmptyLatentImage; zero latent + noise = generation from pure noise).
    latent_image = torch.zeros_like(noise)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    with torch.no_grad():
        samples = ks.sample(
            noise, positive, negative, cfg=cfg, latent_image=latent_image
        )
    end_time = time.time()
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    return samples.detach().cpu(), end_time - start_time, peak_vram


def calculate_latent_mse(l1, l2):
    arr1 = l1[0].detach().cpu().float().numpy()
    arr2 = l2[0].detach().cpu().float().numpy()
    if np.array_equal(arr1, arr2):
        print("  CRITICAL WARNING: Raw latents are bit-perfect identical.")
    return float(np.mean((arr1 - arr2) ** 2))


def calculate_ssim_normalized(img1, img2):
    a1 = np.array(img1)
    a2 = np.array(img2)
    return float(ssim(a1, a2, win_size=3, channel_axis=2, data_range=255))
def print_model_stats(model, name, quant_meta_count=None):
    try:
        state = model.state_dict()
    except (AttributeError, RuntimeError) as e:
        print(f"[{name}] Note: state_dict() failed ({type(e).__name__}). Skipping stats.")
        if quant_meta_count:
            print(
                f"[{name}] Detected Quantization Metadata: "
                f"{quant_meta_count} parameters."
            )
        return

    target_key = None
    for key in state.keys():
        if "layers.10" in key and "weight" in key and "norm" not in key:
            target_key = key
            break
    if target_key is None:
        target_key = next(iter(state))
        print(f"  Note: Fallback key: {target_key}")

    weight = state[target_key]
    print(f"[{name}] Inspecting weight: {target_key}")
    print(f"[{name}] Shape={tuple(weight.shape)}, dtype={weight.dtype}")
    flat = weight.flatten()[:5].cpu().float().tolist()
    print(f"[{name}] First 5 values: {flat}")
    if quant_meta_count:
        print(
            f"[{name}] Detected Quantization Metadata: "
            f"{quant_meta_count} parameters."
        )
    else:
        q_params = [k for k in state if ".comfy_quant" in k]
        if q_params:
            print(
                f"[{name}] Detected Quantization Metadata: "
                f"{len(q_params)} parameters."
            )


def _disable_transformers_auto_docstring():
    """Disable transformers @auto_docstring decorator (docstring generation & verification).

    transformers uses @auto_docstring during image processor / model class definition
    to automatically generate docstrings, printing '[ERROR] xxx ... but not documented'
    directly to stdout when undocumented kwargs are encountered. This is an upstream developer
    docstring quality check and does not affect runtime model execution (the decorator merely
    populates the docstring and returns the original object).

    To keep the environment clean without modifying ComfyUI / embedded python, replace the
    decorator with a no-op to disable verification output. Since transformers uses lazy
    loading (_LazyModule), replacing it before importing ComfyUI ensures subsequent class
    definitions use the no-op decorator.
    """
    import importlib

    # NOTE: import transformers.utils.auto_docstring as _ad returns public attributes
    # (functions) of the utils package, so it does not replace module attributes.
    # Use importlib to reliably obtain the module.
    _ad = importlib.import_module("transformers.utils.auto_docstring")

    if getattr(_ad, "_hswq_auto_docstring_disabled", False):
        return

    def _noop(obj=None, **kwargs):
        if obj is None:
            return lambda f: f
        return obj

    _noop._hswq_auto_docstring_disabled = True
    _ad.auto_docstring = _noop
    # Also replace public attribute in transformers.utils package
    # (handles 'from transformers.utils import auto_docstring' imports)
    import transformers.utils as _tu
    if getattr(_tu, "auto_docstring", None) is not None:
        _tu.auto_docstring = _noop


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Z Image ConvRot NVFP4 Fidelity & VRAM Benchmark "
            "(ComfyUI standard pipeline; requires *_nvfp4_convrot.safetensors)"
        )
    )
    parser.add_argument("--fp16", required=True, help="Baseline FP16/BF16 NextDiT path")
    parser.add_argument(
        "--nvfp4",
        required=True,
        help=(
            "Kitchen NVFP4 + ConvRot quantized path "
            "(e.g. moodyProMix_zitV13_nvfp4_convrot.safetensors)"
        ),
    )
    parser.add_argument("--clip_path", required=True, help="Qwen3-4B text encoder path")
    parser.add_argument("--tokenizer_path", default=None, help="Tokenizer path or Repo ID")
    parser.add_argument("--comfy_path", required=True, help="ComfyUI root path")
    parser.add_argument(
        "--vae",
        default=None,
        help="VAE path; if set, decode latents and SSIM on pixels",
    )
    parser.add_argument("--token", default=None, help="Optional Hugging Face token")
    parser.add_argument(
        "--prompt",
        default="masterpiece, best quality, 1girl, solo, standing, simple background",
        help="Benchmark prompt (simple prompt keeps FP16 vs NVFP4 trajectories close; complex prompts diverge)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory for bench_result_*.png (default: cwd)",
    )
    parser.add_argument(
        "--native-dtype",
        action="store_true",
        default=False,
        help="Evaluate in the source model native dtype (bf16) instead of forcing fp16. Default OFF = stock behavior (V13/V7 identical).",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=2.5,
        help="Classifier-free guidance scale (Z-Image reference workflow: 2.5).",
    )
    parser.add_argument(
        "--negative-prompt",
        default="",
        help="Negative prompt for CFG (empty string default).",
    )
    args = parser.parse_args()
    global bench_use_native
    bench_use_native = args.native_dtype
    if bench_use_native:
        print("  [Native dtype] --native-dtype ON: bf16 checkpoints evaluated in bf16")
    print("Starting Z Image ConvRot NVFP4 Bench (ComfyUI standard pipeline)...")

    set_hf_token(args.token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _disable_transformers_auto_docstring()
    saved_argv = _clear_argv_for_comfy()
    try:
        setup_comfy(args.comfy_path)
        print(f"  ComfyUI Path set to: {args.comfy_path}")

        import comfy.ops
        import nodes
        import folder_paths

        from kitchen_rms_rope_fallback import ensure_kitchen_rms_rope

        ensure_kitchen_rms_rope()
        apply_nvfp4_patches()
        print(
            f"  [BENCH] SSIM target >={SSIM_TARGET} "
            "(ComfyUI standard pipeline + ConvRot NVFP4 parity)"
        )
    except ImportError as e:
        _restore_argv(saved_argv)
        print(f"CRITICAL ERROR: Could not import comfy / patches: {e}")
        print(f"  ComfyUI Path: {args.comfy_path}")
        sys.exit(1)
    finally:
        _restore_argv(saved_argv)

    vae_obj = None
    if args.vae:
        if not os.path.isfile(args.vae) and not os.path.isdir(args.vae):
            print(f"  Warning: --vae path not found: {args.vae}")
        else:
            vae_dir = os.path.dirname(os.path.abspath(args.vae))
            folder_paths.add_model_folder_path("vae", vae_dir)
            vae_loader = nodes.VAELoader()
            vae_obj = vae_loader.load_vae(vae_name=os.path.basename(args.vae))[0]
            print(f"  Loaded VAE for decode: {os.path.basename(args.vae)}")

    print("\n=== 1. Benchmarking Baseline (FP16/BF16) ===")
    patcher_fp16, sd_fp16 = load_zit_model(
        args.fp16, device, args.comfy_path, is_nvfp4=False
    )
    print_model_stats(patcher_fp16.model, "FP16 Baseline")

    # ComfyUI standard CLIP (ZImageTokenizer + Qwen3_4B)
    clip = load_zit_clip(
        patcher_fp16.model.model_config, sd_fp16, args.clip_path, device
    )
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(args.prompt))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(args.negative_prompt))
    print(f"  [CLIP] positive cond {positive[0][0].shape} dtype={positive[0][0].dtype}")
    clip.cond_stage_model.to("cpu")
    torch.cuda.empty_cache()
    print("  [Offload] Text encoder on cpu (VRAM freed for NVFP4 benchmark).")

    latents_fp16, time_fp16, vram_fp16 = run_inference(
        patcher_fp16, positive, negative, args.steps, args.seed, device, cfg=args.cfg
    )
    print(f"FP16 Time: {time_fp16:.2f}s | Peak VRAM: {vram_fp16:.2f} MB")

    if vae_obj is not None:
        vae_decode = nodes.VAEDecode()
        image_tensor = vae_decode.decode(
            vae=vae_obj, samples={"samples": latents_fp16.to(device)}
        )[0]
        img_fp16 = Image.fromarray(
            np.clip(
                255.0 * image_tensor[0].detach().cpu().numpy(), 0, 255
            ).astype(np.uint8)
        )
        del image_tensor
    else:
        img_fp16 = latent_to_img(latents_fp16)
    fp16_path = out_dir / "bench_result_fp16.png"
    img_fp16.save(fp16_path)
    print(f"  Saved: {fp16_path}")

    latents_fp16 = latents_fp16.detach().cpu()
    del patcher_fp16
    gc.collect()
    torch.cuda.empty_cache()
    if device == "cuda":
        print(
            f"  [VRAM] allocated after FP16 free: "
            f"{torch.cuda.memory_allocated() / (1024 ** 2):.1f} MB"
        )

    print("\n=== 2. Benchmarking Quantized (NVFP4 + ConvRot) ===")
    patcher_nv, _sd_nv = load_zit_model(
        args.nvfp4,
        device,
        args.comfy_path,
        is_nvfp4=True,
        require_convrot=True,
    )
    print_model_stats(patcher_nv.model, "NVFP4 ConvRot")
    latents_nv, time_nv, vram_nv = run_inference(
        patcher_nv, positive, negative, args.steps, args.seed, device, cfg=args.cfg
    )
    print(f"NVFP4+ConvRot Time: {time_nv:.2f}s | Peak VRAM: {vram_nv:.2f} MB")

    if vae_obj is not None:
        vae_decode = nodes.VAEDecode()
        image_tensor = vae_decode.decode(
            vae=vae_obj, samples={"samples": latents_nv.to(device)}
        )[0]
        img_nv = Image.fromarray(
            np.clip(
                255.0 * image_tensor[0].detach().cpu().numpy(), 0, 255
            ).astype(np.uint8)
        )
        del image_tensor
    else:
        img_nv = latent_to_img(latents_nv)
    nv_path = out_dir / "bench_result_nvfp4_convrot.png"
    img_nv.save(nv_path)
    print(f"  Saved: {nv_path}")
    latents_nv = latents_nv.detach().cpu()

    mse = calculate_latent_mse(latents_fp16, latents_nv)
    score = calculate_ssim_normalized(img_fp16, img_nv)

    print("\n" + "=" * 50)
    print("ZI CONVROT NVFP4 BENCHMARK RESULTS")
    print("=" * 50)
    vram_saved = vram_fp16 - vram_nv
    vram_saved_pct = (vram_saved / vram_fp16) * 100 if vram_fp16 else 0.0
    print(f"Peak VRAM Expansion:  FP16:           {vram_fp16:>8.1f} MB")
    print(f"                      NVFP4+ConvRot:  {vram_nv:>8.1f} MB")
    print(f"VRAM Saved:           {vram_saved:8.1f} MB ({vram_saved_pct:.1f}%)")
    print("-" * 50)
    print(f"Inference Time:       FP16:           {time_fp16:>8.2f}s")
    print(f"                      NVFP4+ConvRot:  {time_nv:>8.2f}s")
    print("-" * 50)
    mse_label = "MSE (latent)"
    ssim_label = "SSIM (decoded)" if vae_obj is not None else "SSIM (0-255 view)"
    print("Fidelity:")
    print(f"  {mse_label:<18}: {mse:.4f}")
    print(f"  {ssim_label:<18}: {score:.4f}")
    grade = "PASS" if score >= SSIM_TARGET else "FAIL"
    print(f"  SSIM target >={SSIM_TARGET}: {grade}")
    print("=" * 50)

    diff_img = ImageChops.difference(img_fp16, img_nv)
    diff_img = ImageChops.multiply(diff_img, Image.new("RGB", diff_img.size, (10, 10, 10)))
    diff_path = out_dir / "bench_result_diff_convrot.png"
    diff_img.save(diff_path)
    print(f"Diff image saved: {diff_path}")

    del patcher_nv
    gc.collect()
    torch.cuda.empty_cache()
    return 0 if score >= SSIM_TARGET else 1


if __name__ == "__main__":
    raise SystemExit(main())

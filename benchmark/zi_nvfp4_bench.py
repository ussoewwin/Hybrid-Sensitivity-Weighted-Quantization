#!/usr/bin/env python3
"""
Z Image NVFP4 ComfyUI Benchmark
================================
Compare FP16/BF16 NextDiT (Z Image / ZIT / Moody) vs Kitchen NVFP4
(``_quantization_metadata`` + ``.comfy_quant`` after convert_old_quants).

Based on ``zi_int8_bench.py`` (Qwen3_4B TE + NextDiT + sample_euler), with
NVFP4 load/parity from ``krea2_nvfp4_bench.py`` / ``nvfp4bench_sdxl.py``.

ComfyUI-master is not modified. Kitchen builds without ``rms_rope`` get an
eager fallback (``kitchen_rms_rope_fallback.py``) required for qk_norm models.

Example:
  D:\\USERFILES\\fp8e4m3\\venv\\Scripts\\python.exe zi_nvfp4_bench.py \\
    --fp16  "...\\moodyProMix_zitV13.safetensors" \\
    --nvfp4 "...\\moodyProMix_zitV13_nvfp4.safetensors" \\
    --clip_path "...\\qwen3_4b_abliterated_fp16_converted.safetensors" \\
    --comfy_path "D:\\USERFILES\\GitHub\\hswq\\ComfyUI-master" \\
    --vae "...\\Ultra-flux1.vae.safetensors" \\
    --prompt "A beautiful cyberpunk city at night, high detail." \\
    --steps 25
"""

from __future__ import annotations

import argparse
import gc
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


def resolve_tokenizer_offline(provided_path, comfy_path):
    validation_files = ["tokenizer.json", "vocab.json", "config.json"]

    if provided_path and os.path.isdir(provided_path):
        if any(os.path.exists(os.path.join(provided_path, f)) for f in validation_files):
            return provided_path

    if comfy_path:
        search_roots = [
            os.path.join(comfy_path, "models", "clip"),
            os.path.join(comfy_path, "models", "tokenizers"),
            comfy_path,
        ]
        for root_dir in search_roots:
            if not os.path.exists(root_dir):
                continue
            for root, dirs, files in os.walk(root_dir):
                if any(f in files for f in validation_files):
                    if any(x in root.lower() for x in ["qwen", "qwen2.5", "zit"]):
                        print(f"  [Offline Discovery] Found tokenizer in ComfyUI: {root}")
                        return root

    print("  Note: Searching recursively for any local Qwen tokenizer...")
    for root, dirs, files in os.walk("."):
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".") and d not in ["ComfyUI-master", "node_modules"]
        ]
        if any(f in files for f in validation_files):
            if any(x in root.lower() for x in ["qwen", "qwen2.5", "zit"]):
                print(f"  [Offline Discovery] Found potential tokenizer: {root}")
                return root

    return None


def latent_to_img(l):
    l = l[0].detach().permute(1, 2, 0).cpu().float().numpy()
    l = (l - l.min()) / (l.max() - l.min() + 1e-6) * 255
    return Image.fromarray(l[:, :, :3].astype(np.uint8))


def _fuse_zanime_attention(state_dict):
    new_dict = dict(state_dict)
    prefixes = set()
    for k in list(new_dict.keys()):
        m = re.match(r"^(.+?\.attention)\.to_q\.weight$", k)
        if m:
            prefixes.add(m.group(1))
    for prefix in prefixes:
        kq, kk, kv = f"{prefix}.to_q.weight", f"{prefix}.to_k.weight", f"{prefix}.to_v.weight"
        if kq in new_dict and kk in new_dict and kv in new_dict:
            qkv = torch.cat([new_dict[kq], new_dict[kk], new_dict[kv]], dim=0)
            new_dict[f"{prefix}.qkv.weight"] = qkv
            del new_dict[kq], new_dict[kk], new_dict[kv]
    rename_map = {
        ".attention.to_out.0.weight": ".attention.out.weight",
        ".attention.norm_q.weight": ".attention.q_norm.weight",
        ".attention.norm_k.weight": ".attention.k_norm.weight",
    }
    for k in list(new_dict.keys()):
        for src, dst in rename_map.items():
            if k.endswith(src):
                new_dict[k.replace(src, dst)] = new_dict.pop(k)
                break
    return new_dict


def normalize_zanime_keys(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("all_"):
            new_key = re.sub(r"^all_(.*?)\.2-1", r"\1", new_key)
        normalized[new_key] = value
    return _fuse_zanime_attention(normalized)


def detect_zit_config_from_keys(state_dict):
    state_dict_keys = list(state_dict.keys())
    zit_config = {}
    layer_indices = set()
    for key in state_dict_keys:
        if key.startswith("layers."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_indices.add(int(parts[1]))

    zit_config["num_layers"] = max(layer_indices) + 1 if layer_indices else 30
    if "x_embedder.weight" in state_dict:
        zit_config["hidden_size"] = state_dict["x_embedder.weight"].shape[0]
    elif "all_x_embedder.2-1.weight" in state_dict:
        zit_config["hidden_size"] = state_dict["all_x_embedder.2-1.weight"].shape[0]
    else:
        zit_config["hidden_size"] = 3072

    refiner_indices = set()
    for key in state_dict_keys:
        if key.startswith("context_refiner."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                refiner_indices.add(int(parts[1]))
    zit_config["num_context_refiner"] = max(refiner_indices) + 1 if refiner_indices else 2

    w1_key = "layers.0.feed_forward.w1.weight"
    if w1_key in state_dict:
        # NVFP4 packs K (in_features); out_features (shape[0]) stays logical.
        zit_config["intermediate_size"] = int(state_dict[w1_key].shape[0])
        print(f"  Detected Intermediate Size: {zit_config['intermediate_size']}")
    else:
        zit_config["intermediate_size"] = None

    zit_config["qk_norm"] = any(k.endswith(".attention.q_norm.weight") for k in state_dict_keys)
    if zit_config["qk_norm"]:
        print("  Detected qk_norm=True (q_norm/k_norm weights present)")

    return zit_config


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
    _install_torchaudio_stub()


def apply_nvfp4_patches() -> None:
    import comfy.ops
    from nvfp4.comfy_quant_nvfp4 import apply_comfy_quant_nvfp4_patches
    from nvfp4_comfy_parity import apply_nvfp4_comfy_parity

    apply_comfy_quant_nvfp4_patches()
    if not apply_nvfp4_comfy_parity():
        raise RuntimeError(
            "nvfp4 ComfyUI-only parity failed to apply "
            "(need [BENCH] nvfp4 ComfyUI-only log; TC forward must be off)"
        )
    lin_fwd = comfy.ops.mixed_precision_ops().Linear.forward
    if getattr(lin_fwd, "_hswq_nvfp4_full_forward", False):
        raise RuntimeError(
            "nvfp4 ComfyUI-only parity incomplete: "
            "Linear.forward still has _hswq_nvfp4_full_forward "
            "(TC path would destroy Pixel SSIM)"
        )


def set_hf_token(token: str | None) -> None:
    if not token:
        return
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def load_zit_model(path, device="cuda", comfy_path=None, is_nvfp4=False):
    if comfy_path and comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)

    from comfy.ldm.lumina.model import NextDiT
    import comfy.ops
    import comfy.utils

    args_path = resolve_path(path, is_file=True)
    print(f"Loading state_dict: {os.path.basename(args_path)}")

    if is_nvfp4:
        # Kitchen NVFP4: inject .comfy_quant from file metadata, then mixed_precision load.
        state_dict, metadata = comfy.utils.load_torch_file(args_path, return_metadata=True)
        state_dict, metadata = comfy.utils.convert_old_quants(
            state_dict, "", metadata=metadata or {}
        )
        n_cq = sum(1 for k in state_dict if k.endswith(".comfy_quant"))
        print(f"  [NVFP4] convert_old_quants -> {n_cq} .comfy_quant markers")
    else:
        state_dict = load_file(args_path)

    converted_dict = {}
    for k, v in state_dict.items():
        if hasattr(v, "dtype") and v.dtype == torch.bfloat16:
            converted_dict[k] = v.to(torch.float16)
        else:
            converted_dict[k] = v

    prefixes_to_try = [
        "",
        "model.",
        "model.diffusion_model.",
        "diffusion_model.",
    ]
    best_prefix = ""
    for prefix in prefixes_to_try:
        if prefix == "":
            continue
        if any(k.startswith(prefix) for k in converted_dict.keys()):
            sample_key = f"{prefix}layers.0.attention_norm1.weight"
            if sample_key in converted_dict:
                best_prefix = prefix
                print(f"  [Prefix Detection] Detected prefix: '{prefix}'")
                break

    if best_prefix:
        print(f"  [Prefix Strip] Stripping prefix: '{best_prefix}'")
        stripped_dict = {}
        for k, v in converted_dict.items():
            if k.startswith(best_prefix):
                stripped_dict[k[len(best_prefix) :]] = v
            else:
                # Keep bare .comfy_quant markers from Kitchen metadata (no prefix).
                stripped_dict[k] = v
        converted_dict = stripped_dict

    is_zanime = any(k.startswith("all_x_embedder.2-1") for k in converted_dict.keys())
    if is_zanime:
        print("  [Model Detection] Z-Anime key naming detected. Normalizing...")
        converted_dict = normalize_zanime_keys(converted_dict)

    config = detect_zit_config_from_keys(converted_dict)
    print(
        f"  [Config Detection] hidden_size={config['hidden_size']}, "
        f"layers={config['num_layers']}"
    )

    kwargs = {}
    if config.get("intermediate_size"):
        ratio = config["intermediate_size"] / config["hidden_size"]
        kwargs["ffn_dim_multiplier"] = ratio
        print(
            f"  Calculated FFN Dim Multiplier: {ratio:.4f} "
            f"(Dim: {config['hidden_size']} -> {config['intermediate_size']})"
        )
    if config.get("qk_norm"):
        kwargs["qk_norm"] = True

    def _build_and_load(ops):
        model_local = NextDiT(
            patch_size=2,
            in_channels=16,
            dim=config["hidden_size"],
            n_layers=config["num_layers"],
            n_refiner_layers=config["num_context_refiner"],
            n_heads=config["hidden_size"] // 128,
            n_kv_heads=config["hidden_size"] // 128,
            multiple_of=256,
            norm_eps=1e-5,
            cap_feat_dim=2560,
            z_image_modulation=True,
            pad_tokens_multiple=64,
            device="cpu",
            dtype=torch.float16,
            operations=ops,
            **kwargs,
        )
        try:
            missing_local, unexpected_local = model_local.load_state_dict(
                converted_dict, strict=False, assign=True
            )
        except TypeError:
            print("  [Warning] assign=True unsupported; quantized dtypes may cast.")
            missing_local, unexpected_local = model_local.load_state_dict(
                converted_dict, strict=False
            )
        except RuntimeError as e:
            print(f"  CRITICAL ERROR: Model Size Mismatch. Error: {e}")
            print(f"  Config: {config}")
            sys.exit(1)
        return model_local, missing_local, unexpected_local

    if is_nvfp4:
        print("Using mixed_precision_ops for NVFP4 model load...")
        ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.float16)
        model, missing, unexpected = _build_and_load(ops)
    else:
        print("Using standard operations for FP16 model load...")
        ops = comfy.ops.disable_weight_init
        model, missing, unexpected = _build_and_load(ops)

    print(
        f"  [Keys] Matched: {len(converted_dict) - len(unexpected)}, "
        f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
    )
    if len(missing) > len(list(model.parameters())) * 0.5:
        print(f"  Warning: Many keys still missing. First 5: {list(missing)[:5]}")

    if is_nvfp4:
        model = model.to(device)
        print(f"  Note: NVFP4 model loaded on {device} (mixed_precision / assign=True).")
    else:
        model = model.to(device).to(torch.float16)
        print(f"  Note: FP16 model loaded on {device}.")

    # assign=True shares Parameter storage with converted_dict values.
    # Returning that dict pins CUDA weights after `del model`, so the NVFP4
    # phase peak becomes leftover FP16 + NVFP4 (~+4GB). Drop tensor refs here.
    n_comfy_quant = sum(
        1 for k in converted_dict if k.endswith(".comfy_quant") or ".comfy_quant" in k
    )
    converted_dict.clear()
    del converted_dict

    model.eval()
    return model, n_comfy_quant, is_zanime


def encode_prompt(prompt, text_encoder, tokenizer, device):
    template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    formatted = template.format(prompt)

    tokens = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            intermediate_output=-2,
        )
        hidden_states = outputs[1]

    return hidden_states, attention_mask.bool()


def run_inference(model, prompt_embeds, prompt_mask, steps, seed, device):
    import comfy.k_diffusion.sampling as k_sampling

    class ZITWrapper:
        def __init__(self, model, embeds, mask):
            self.model = model
            self.embeds = embeds
            self.mask = mask

        def __call__(self, x, sigma, **kwargs):
            dtype = torch.float16
            out = self.model(
                x.to(dtype),
                sigma.to(dtype),
                self.embeds.to(dtype),
                None,
                attention_mask=self.mask,
            )
            if isinstance(out, tuple):
                out = out[0]
            return out.to(x.dtype)

    generator = torch.Generator(device).manual_seed(seed)
    x = torch.randn(1, 16, 128, 128, device=device, dtype=torch.float16, generator=generator)
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device)
    wrapper = ZITWrapper(model, prompt_embeds, prompt_mask)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    with torch.no_grad():
        result = k_sampling.sample_euler(wrapper, x, sigmas, disable=False)
    end_time = time.time()
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    return result, end_time - start_time, peak_vram


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


def calculate_normalized_mse(img1, img2):
    a1 = np.array(img1).astype(np.float32)
    a2 = np.array(img2).astype(np.float32)
    return float(np.mean((a1 - a2) ** 2))


def print_model_stats(model, name, quant_meta_count=None):
    """Report live Linear.weight (QT vs plain). Do not keep a full state_dict ref."""
    try:
        from comfy.quant_ops import QuantizedTensor
    except Exception:
        QuantizedTensor = ()  # type: ignore[misc, assignment]

    live_mod = None
    live_name = None
    for n, m in model.named_modules():
        if "layers.10" in n and n.endswith("attention.qkv") and hasattr(m, "weight"):
            live_mod, live_name = m, n
            break
    if live_mod is not None:
        w = live_mod.weight
        is_qt = isinstance(w, QuantizedTensor) if QuantizedTensor else False
        print(f"[{name}] Live module: {live_name}")
        if is_qt:
            layout = getattr(w, "_layout_cls", type(None).__name__)
            print(
                f"[{name}] QuantizedTensor layout={layout} "
                f"logical_shape={tuple(getattr(w._params, 'orig_shape', ()))} "
                f"orig_dtype={getattr(w._params, 'orig_dtype', None)} "
                f"_qdata.shape={tuple(w._qdata.shape)} _qdata.dtype={w._qdata.dtype}"
            )
            flat = w._qdata.flatten()[:5].cpu().float().tolist()
            print(f"[{name}] First 5 packed _qdata values: {flat}")
        else:
            print(f"[{name}] Plain weight Shape={tuple(w.shape)}, dtype={w.dtype}")
            flat = w.flatten()[:5].cpu().float().tolist()
            print(f"[{name}] First 5 values: {flat}")
    elif live_mod is None:
        print(f"[{name}] Warning: layers.10.attention.qkv not found.")

    if quant_meta_count:
        print(
            f"[{name}] Detected Quantization Metadata: "
            f"{quant_meta_count} parameters."
        )
    elif QuantizedTensor:
        n_qt = sum(
            1
            for m in model.modules()
            if hasattr(m, "weight") and isinstance(m.weight, QuantizedTensor)
        )
        if n_qt:
            print(f"[{name}] Live QuantizedTensor weights: {n_qt}.")


def main():
    parser = argparse.ArgumentParser(
        description="Z Image NVFP4 Fidelity & VRAM Benchmark (zi_int8-based)"
    )
    parser.add_argument("--fp16", required=True, help="Baseline FP16/BF16 NextDiT path")
    parser.add_argument("--nvfp4", required=True, help="Kitchen NVFP4 quantized path")
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
        default="A beautiful cyberpunk city at night, high detail.",
        help="Benchmark prompt",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory for bench_result_*.png (default: cwd)",
    )
    args = parser.parse_args()
    print("Starting Z Image NVFP4 Bench (zi_int8 + NVFP4 parity)...")

    if args.tokenizer_path and args.tokenizer_path.startswith("hf_"):
        print("  Warning: HF token in tokenizer_path. Ignoring invalid path.")
        args.tokenizer_path = None

    set_hf_token(args.token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_argv = _clear_argv_for_comfy()
    try:
        setup_comfy(args.comfy_path)
        print(f"  ComfyUI Path set to: {args.comfy_path}")

        from comfy.text_encoders import llama as llama_module
        from transformers import Qwen2Tokenizer
        import comfy.ops
        import nodes
        import folder_paths

        from kitchen_rms_rope_fallback import ensure_kitchen_rms_rope

        ensure_kitchen_rms_rope()
        apply_nvfp4_patches()
        print(f"  [BENCH] SSIM target >={SSIM_TARGET} (ComfyUI-only NVFP4 forward)")
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

    print("Starting Text Encoder Initialization...")
    tokenizer_path = resolve_tokenizer_offline(args.tokenizer_path, args.comfy_path)
    if tokenizer_path:
        print(f"  Loading tokenizer from disk: {tokenizer_path}")
        try:
            tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        except Exception as e:
            print(f"  Warning: local_files_only failed ({e}); retrying without constraint...")
            tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)
    else:
        model_id = args.tokenizer_path if args.tokenizer_path else "Qwen/Qwen2.5-7B-Instruct"
        print(f"  CRITICAL: Local tokenizer not found. Trying Repo ID: {model_id}")
        try:
            tokenizer = Qwen2Tokenizer.from_pretrained(model_id, local_files_only=True)
        except Exception as e:
            print(f"  FATAL: Offline tokenizer load failed: {e}")
            sys.exit(1)

    resolved_clip = resolve_path(args.clip_path, is_file=True)
    text_encoder = llama_module.Qwen3_4B(
        config_dict={},
        device=device,
        dtype=torch.float16,
        operations=comfy.ops.disable_weight_init,
    ).to(device)
    print(f"Loading CLIP weights from: {resolved_clip}")
    text_encoder.load_state_dict(load_file(resolved_clip), strict=False)
    text_encoder.eval()

    embeds, mask = encode_prompt(args.prompt, text_encoder, tokenizer, device)
    text_encoder.cpu().to(torch.float16)
    torch.cuda.empty_cache()
    te_device = next(text_encoder.parameters()).device
    print(f"  [Offload] Text encoder on {te_device} (VRAM freed for ZI NVFP4 benchmark).")

    print("\n=== 1. Benchmarking Baseline (FP16/BF16) ===")
    model, _n_cq_fp16, is_zanime_fp16 = load_zit_model(
        args.fp16, device, args.comfy_path, is_nvfp4=False
    )
    print_model_stats(model, "FP16 Baseline", _n_cq_fp16)
    latents_fp16, time_fp16, vram_fp16 = run_inference(
        model, embeds, mask, args.steps, args.seed, device
    )
    print(f"FP16 Time: {time_fp16:.2f}s | Peak VRAM: {vram_fp16:.2f} MB")

    if vae_obj is not None:
        vae_decode = nodes.VAEDecode()
        image_tensor = vae_decode.decode(
            vae=vae_obj, samples={"samples": latents_fp16}
        )[0]
        img_fp16 = Image.fromarray(
            np.clip(
                255.0 * image_tensor[0].detach().cpu().numpy(), 0, 255
            ).astype(np.uint8)
        )
        # VAE decode leaves multi-GB bf16 feature maps on CUDA; free before
        # the next phase or NVFP4 peak = leftover decode + quantized UNet.
        del image_tensor
    else:
        img_fp16 = latent_to_img(latents_fp16)
    fp16_path = out_dir / "bench_result_fp16.png"
    img_fp16.save(fp16_path)
    print(f"  Saved: {fp16_path}")

    # Latents only needed on CPU for MSE; drop CUDA copy with the UNet.
    latents_fp16 = latents_fp16.detach().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if device == "cuda":
        print(
            f"  [VRAM] allocated after FP16 free: "
            f"{torch.cuda.memory_allocated() / (1024 ** 2):.1f} MB"
        )

    print("\n=== 2. Benchmarking Quantized (NVFP4) ===")
    model, n_cq_nv, is_zanime_nv = load_zit_model(
        args.nvfp4, device, args.comfy_path, is_nvfp4=True
    )
    is_zanime = is_zanime_fp16 or is_zanime_nv
    print_model_stats(model, "NVFP4 Quantized", n_cq_nv)
    latents_nv, time_nv, vram_nv = run_inference(
        model, embeds, mask, args.steps, args.seed, device
    )
    print(f"NVFP4 Time: {time_nv:.2f}s | Peak VRAM: {vram_nv:.2f} MB")

    if vae_obj is not None:
        vae_decode = nodes.VAEDecode()
        image_tensor = vae_decode.decode(
            vae=vae_obj, samples={"samples": latents_nv}
        )[0]
        img_nv = Image.fromarray(
            np.clip(
                255.0 * image_tensor[0].detach().cpu().numpy(), 0, 255
            ).astype(np.uint8)
        )
        del image_tensor
    else:
        img_nv = latent_to_img(latents_nv)
    nv_path = out_dir / "bench_result_fp8.png"
    img_nv.save(nv_path)
    print(f"  Saved: {nv_path}")
    latents_nv = latents_nv.detach().cpu()

    if is_zanime:
        mse = calculate_normalized_mse(img_fp16, img_nv)
    else:
        mse = calculate_latent_mse(latents_fp16, latents_nv)
    score = calculate_ssim_normalized(img_fp16, img_nv)

    print("\n" + "=" * 50)
    print("ZI NVFP4 BENCHMARK RESULTS")
    print("=" * 50)
    vram_saved = vram_fp16 - vram_nv
    vram_saved_pct = (vram_saved / vram_fp16) * 100 if vram_fp16 else 0.0
    print(f"Peak VRAM Expansion:  FP16:  {vram_fp16:>8.1f} MB")
    print(f"                      NVFP4: {vram_nv:>8.1f} MB")
    print(f"VRAM Saved:           {vram_saved:8.1f} MB ({vram_saved_pct:.1f}%)")
    print("-" * 50)
    print(f"Inference Time:       FP16:  {time_fp16:>8.2f}s")
    print(f"                      NVFP4: {time_nv:>8.2f}s")
    print("-" * 50)
    mse_label = "MSE (0-255 view)" if is_zanime else "MSE (latent)"
    ssim_label = "SSIM (decoded)" if vae_obj is not None else "SSIM (0-255 view)"
    print("Fidelity:")
    print(f"  {mse_label:<18}: {mse:.4f}")
    print(f"  {ssim_label:<18}: {score:.4f}")
    grade = "PASS" if score >= SSIM_TARGET else "FAIL"
    print(f"  SSIM target >={SSIM_TARGET}: {grade}")
    print("=" * 50)

    diff_img = ImageChops.difference(img_fp16, img_nv)
    diff_img = ImageChops.multiply(diff_img, Image.new("RGB", diff_img.size, (10, 10, 10)))
    diff_path = out_dir / "bench_result_diff.png"
    diff_img.save(diff_path)
    print(f"Diff image saved: {diff_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return 0 if score >= SSIM_TARGET else 1


if __name__ == "__main__":
    raise SystemExit(main())

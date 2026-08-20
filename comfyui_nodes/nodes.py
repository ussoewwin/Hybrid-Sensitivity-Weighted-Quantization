"""ComfyUI node: quantize a loaded Z Image (NextDiT) UNet to native ConvRot INT8.

Connects to the standard UNet loader's MODEL output (no file-path input).
Extracts the diffusion-model weights in-memory, quantizes them with the same
algorithm as ``native_convert_int8_convrot_zi.py``, and saves a checkpoint the
standard ComfyUI loader reads back.

Output layout:
    model.diffusion_model.<layer>.weight          int8
    model.diffusion_model.<layer>.weight_scale    float32
    model.diffusion_model.<layer>.comfy_quant     uint8 JSON
    _quantization_metadata                        {"format_version":"1.0","layers":{...}}

Repo layout: this package lives at <repo>/comfyui_nodes/ and reuses the pure
quantization helpers in <repo>/native_convert_int8.py. The repo root is
auto-detected by walking up from this package until that script is found.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys

_N8 = None


def _default_repo_root() -> str:
    """Locate the repo root by walking up until native_convert_int8.py is found."""
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(8):
        if os.path.isfile(os.path.join(d, "native_convert_int8.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.abspath(os.path.join(here, os.pardir))


def _load_native_int8():
    """Import native_convert_int8.py once (no side effects on import)."""
    global _N8
    if _N8 is not None:
        return _N8

    root = _default_repo_root()
    path = os.path.join(root, "native_convert_int8.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Quantization helpers not found: {path}. The node package must "
            "live inside the Hybrid-Sensitivity-Weighted-Quantization clone."
        )

    try:
        spec = importlib.util.spec_from_file_location("hswq_native_int8", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["hswq_native_int8"] = mod
        spec.loader.exec_module(mod)
    except ImportError as e:
        raise ImportError(
            f"Failed to import {path}: {e}. "
            "The helpers need torch, safetensors and tqdm in the ComfyUI env."
        )
    _N8 = mod
    return mod


def _is_power_of_4(n: int) -> bool:
    return n >= 4 and (n & (n - 1)) == 0 and math.log(n, 4) % 1 == 0


def _output_dir() -> str:
    try:
        import folder_paths

        return folder_paths.get_output_directory()
    except Exception:
        return os.getcwd()


def _extract_model_state_dict(model):
    """Return diffusion weights (CPU float) with 'model.diffusion_model.' prefix.

    Handles both the standard BaseModel wrapper (model.model.diffusion_model)
    and loaders that put the raw diffusion model directly in model.model.
    """
    inner = getattr(model, "model", None)
    if inner is None:
        raise ValueError(
            "model input has no inner model; connect a UNet / checkpoint "
            "loader MODEL output."
        )
    diffusion = getattr(inner, "diffusion_model", None)
    if diffusion is None:
        diffusion = inner
    if not hasattr(diffusion, "state_dict"):
        raise ValueError("cannot locate the diffusion model in the MODEL input")

    out = {}
    for k, v in diffusion.state_dict().items():
        out["model.diffusion_model." + k] = v.detach().cpu()
    return out


def _quantize_state_dict(sd, group_size, enable_convrot, per_channel_int8, n8):
    """In-memory ConvRot INT8 packing (same math as native_convert_int8.py)."""
    new_sd = {}
    meta_layers = {}
    n_linear = n_conv2d = n_plain = n_kept = 0

    for key, tensor in sd.items():
        if not n8._is_float_matmul_weight(key, tensor):
            new_sd[key] = tensor
            n_kept += 1
            continue

        w = tensor.float()
        module_key = key[: -len(".weight")]
        conf = {"format": "int8_tensorwise"}

        if enable_convrot and tensor.ndim == 2:
            gs = n8.convrot_group_size_for_features(w.shape[1], group_size)
            if gs is not None:
                h = n8.build_hadamard(gs, device="cpu")
                w = n8.rotate_weight(w, h, gs)
                q, scale = n8.quantize_int8_rowwise(w)
                conf = {"format": "int8_tensorwise", "convrot": True,
                        "convrot_groupsize": int(gs)}
                n_linear += 1
            else:
                q, scale = _plain_pack(w, per_channel_int8, n8)
                n_plain += 1
        elif enable_convrot and tensor.ndim == 4:
            gs = n8.convrot_group_size_for_features(w.shape[1], group_size)
            if gs is not None:
                h = n8.build_hadamard(gs, device="cpu")
                w = n8.rotate_weight_conv2d(w, h, gs)
                q, scale = n8.quantize_int8_channelwise(w)
                conf = {"format": "int8_tensorwise", "convrot": True,
                        "convrot_groupsize": int(gs)}
                n_conv2d += 1
            else:
                q, scale = _plain_pack(w, per_channel_int8, n8)
                n_plain += 1
        else:
            q, scale = _plain_pack(w, per_channel_int8, n8)
            n_plain += 1

        new_sd[key] = q
        new_sd[f"{module_key}.weight_scale"] = scale
        new_sd[f"{module_key}.comfy_quant"] = n8._encode_comfy_quant(conf)
        meta_layers[module_key] = dict(conf)

    return new_sd, meta_layers, {
        "linear": n_linear,
        "conv2d": n_conv2d,
        "plain": n_plain,
        "kept": n_kept,
    }


def _plain_pack(w, per_channel_int8, n8):
    if per_channel_int8:
        return n8.quantize_int8_channelwise(w)
    return n8.quantize_int8_tensorwise(w)


def _summarize(output_path: str) -> str:
    """Read the written checkpoint's metadata only (no tensor load)."""
    try:
        from safetensors import safe_open

        with safe_open(output_path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
        qm = json.loads(meta.get("_quantization_metadata", "{}"))
        layers = qm.get("layers", {}) if isinstance(qm, dict) else {}
        convrot = sum(
            1 for c in layers.values() if isinstance(c, dict) and c.get("convrot")
        )
        plain = len(layers) - convrot
        return f"layers={len(layers)} convrot={convrot} plain_int8={plain}"
    except Exception:
        return "(summary unavailable)"


class ZImageConvRotInt8Quantize:
    """Quantize a loaded Z Image UNet (MODEL input) to native ConvRot INT8."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "output_path": ("STRING", {"default": "", "multiline": False}),
                "group_size": ("INT", {"default": 256}),
                "convrot": ("BOOLEAN", {"default": True}),
                "per_channel_int8": ("BOOLEAN", {"default": True}),
                "run_benchmark": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "report")
    FUNCTION = "quantize"
    CATEGORY = "HSWQ/Quantize"
    OUTPUT_NODE = True

    def quantize(
        self,
        model,
        clip,
        output_path,
        group_size,
        convrot,
        per_channel_int8,
        run_benchmark,
        vae=None,
    ):
        n8 = _load_native_int8()

        group_size = int(group_size)
        if not _is_power_of_4(group_size):
            raise ValueError(f"group_size must be a power of 4 (>=4), got {group_size}")

        sd = _extract_model_state_dict(model)

        original_name = "convrot_int8"
        if hasattr(model, "cached_patcher_init") and model.cached_patcher_init:
            func, args = model.cached_patcher_init[:2]
            if args and isinstance(args, tuple) and isinstance(args[0], str):
                unet_path = args[0]
                original_name = os.path.splitext(os.path.basename(unet_path))[0]

        output_path = (output_path or "").strip()
        if not output_path:
            output_path = os.path.join(_output_dir(), f"{original_name}_native_convrot_int8.safetensors")
        output_path = os.path.abspath(output_path)
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, f"{original_name}_native_convrot_int8.safetensors")
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        new_sd, meta_layers, stats = _quantize_state_dict(
            sd, group_size, bool(convrot), bool(per_channel_int8), n8
        )

        from safetensors.torch import save_file

        metadata = {
            "_quantization_metadata": json.dumps(
                {"format_version": "1.0", "layers": meta_layers},
                separators=(",", ":"),
            )
        }
        save_file(new_sd, output_path, metadata=metadata)

        report = [
            f"Saved: {output_path}",
            _summarize(output_path),
            f"convrot={bool(convrot)} group_size={group_size} "
            f"linear={stats['linear']} conv2d={stats['conv2d']} "
            f"plain={stats['plain']} kept={stats['kept']}",
        ]

        if run_benchmark:
            import comfy.model_management as mm
            import comfy.sample as comfy_sample
            import comfy.sd
            import time
            import torch
            import math

            try:
                # Prepare FP16 baseline latent
                tokens = clip.tokenize("masterpiece, best quality, 1girl, solo, standing, simple background")
                positive = clip.encode_from_tokens_scheduled(tokens)
                negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
                
                device = mm.intermediate_device()
                width, height = 1024, 1024
                latent = torch.zeros([1, 16, height // 8, width // 8], device=device)
                latent = comfy_sample.fix_empty_latent_channels(model, latent)
                latent_dict = {"samples": latent}

                seed, steps, cfg = 42, 12, 2.5

                def _sample(m):
                    noise = comfy_sample.prepare_noise(latent_dict["samples"], seed, None)
                    return comfy_sample.sample(m, noise, steps, cfg, "euler", "simple", positive, negative, latent_dict["samples"], denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, callback=None, disable_pbar=True, seed=seed)

                t0 = time.perf_counter()
                out_fp16 = _sample(model)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t_fp16 = time.perf_counter() - t0
                lat_fp16 = out_fp16.detach().float().cpu()

                # Free FP16 model temporarily to load INT8 (simulated by unloading)
                mm.unload_all_models()
                mm.soft_empty_cache()

                # Load INT8 model
                t0 = time.perf_counter()
                model_int8 = comfy.sd.load_diffusion_model(output_path, {})
                load_int8 = time.perf_counter() - t0

                t0 = time.perf_counter()
                out_int8 = _sample(model_int8)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t_int8 = time.perf_counter() - t0
                lat_int8 = out_int8.detach().float().cpu()

                # Calc MSE / Cosine
                a = lat_fp16.reshape(-1)
                b = lat_int8.reshape(-1)
                mse = float(torch.mean((a - b) ** 2).item())
                lat_cos = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item())

                report.append("\n=== BENCHMARK ===")
                report.append(f"FP16 Time: {t_fp16:.2f}s")
                report.append(f"INT8 Load: {load_int8:.2f}s, Time: {t_int8:.2f}s")
                report.append(f"MSE: {mse:.4f} | Cosine: {lat_cos:.4f}")

                if vae is not None:
                    try:
                        from skimage.metrics import structural_similarity as ssim
                        import numpy as np
                        from PIL import Image

                        def _decode(lat):
                            if getattr(lat, "is_nested", False): lat = lat.unbind()[0]
                            _po = vae.process_output
                            vae.process_output = lambda img: img.float().add(1.0).mul(0.5).clamp(0.0, 1.0)
                            try:
                                with torch.inference_mode(False):
                                    images = vae.decode(lat)
                            finally:
                                vae.process_output = _po
                            if len(images.shape) == 5:
                                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                            img_array = 255.0 * images[0].detach().cpu().numpy()
                            return Image.fromarray(np.clip(img_array, 0, 255).astype("uint8"))

                        img_fp16 = _decode(out_fp16.detach())
                        img_int8 = _decode(out_int8.detach())
                        score = float(ssim(np.array(img_fp16), np.array(img_int8), win_size=3, channel_axis=2, data_range=255))
                        report.append(f"SSIM (decoded): {score:.4f}")
                    except Exception as ve:
                        report.append(f"[VAE Decode Error] {str(ve)}")

                mm.unload_all_models()
                mm.soft_empty_cache()

            except Exception as e:
                report.append(f"\n[Benchmark Error] {str(e)}")

        return (output_path, "\n".join(report))

"""ComfyUI node: quantize a loaded Z Image (NextDiT) UNet to native ConvRot INT8.

Connects to the standard UNet loader's MODEL output (no file-path input).
Extracts the diffusion-model weights in-memory, quantizes them with the same
algorithm as ``Z_Image/native_convert_int8_convrot_zi.py``, and saves a checkpoint the
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


_K2_N8 = None


def _load_krea2_native_int8():
    """Import Krea2/native_convert_int8_krea2.py once."""
    global _K2_N8
    if _K2_N8 is not None:
        return _K2_N8

    root = _default_repo_root()
    candidates = [
        os.path.join(root, "Krea2", "native_convert_int8_krea2.py"),
        os.path.join(root, "native_convert_int8_krea2.py"),
    ]
    path = next((c for c in candidates if os.path.isfile(c)), None)
    if path is None:
        raise FileNotFoundError(
            f"Krea2 native convert helpers not found in: {candidates}. The node package "
            "must live inside the Hybrid-Sensitivity-Weighted-Quantization clone."
        )

    try:
        spec = importlib.util.spec_from_file_location("krea2_native_int8", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["krea2_native_int8"] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)
    except ImportError as e:
        raise ImportError(
            f"Failed to import {path}: {e}. "
            "The helpers need torch, safetensors and tqdm in the ComfyUI env."
        )
    _K2_N8 = mod
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
        if k.startswith("model.diffusion_model."):
            clean_k = k
        elif k.startswith("diffusion_model."):
            clean_k = "model." + k
        else:
            clean_k = "model.diffusion_model." + k
        out[clean_k] = v.detach().cpu()
    return out


_QWEN_EDIT_BLACKLIST = (
    "img_in",
    "txt_in",
    "time_text_embed",
    "norm_out",
    "proj_out",
)


def _is_qwen_blacklisted(key: str) -> bool:
    return any(marker in key for marker in _QWEN_EDIT_BLACKLIST)


def _quantize_state_dict_krea2(
    sd,
    group_size,
    enable_convrot,
    per_channel_int8,
    k2,
):
    """In-memory ConvRot INT8 packing for Krea2 DiT using Krea2/native_convert_int8_krea2.py."""
    import torch

    new_sd = {}
    meta_layers = {}
    n_linear = n_conv2d = n_plain = n_kept = 0

    prefix = ""
    for p in ("model.diffusion_model.", "diffusion_model.", ""):
        if any(k.startswith(p) and "txtfusion.projector.weight" in k for k in sd):
            prefix = p
            break

    for key, tensor in sd.items():
        if k2._is_non_diffusion_key(key):
            new_sd[key] = tensor
            n_kept += 1
            continue

        under_prefix = (not prefix) or key.startswith(prefix)
        is_dit_weight = (
            under_prefix
            and key.endswith(".weight")
            and tensor.ndim in (2, 4)
            and tensor.dtype in (torch.float16, torch.float32, torch.bfloat16)
        )

        if not is_dit_weight:
            new_sd[key] = tensor
            n_kept += 1
            continue

        w_fp = tensor.float()
        module_key = key[: -len(".weight")]
        used_gs = None
        if enable_convrot:
            used_gs = k2.convrot_group_size_for_features(int(w_fp.shape[1]), group_size)

        if used_gs is not None and tensor.ndim == 2:
            h_matrix = k2.build_hadamard(used_gs, device="cpu", dtype=torch.float32)
            w_fp = k2.rotate_weight(w_fp, h_matrix, used_gs)
            q, scale = k2.pack_channelwise(w_fp)
            quant_config = {
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": int(used_gs),
            }
            n_linear += 1
        elif used_gs is not None and tensor.ndim == 4:
            h_matrix = k2.build_hadamard(used_gs, device="cpu", dtype=torch.float32)
            w_fp = k2.rotate_weight_conv2d(w_fp, h_matrix, used_gs)
            q, scale = k2.pack_channelwise(w_fp)
            quant_config = {
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": int(used_gs),
            }
            n_conv2d += 1
        elif per_channel_int8:
            q, scale = k2.pack_channelwise(w_fp)
            quant_config = {"format": "int8_tensorwise"}
            n_plain += 1
        else:
            q, scale = k2.pack_tensorwise(w_fp)
            quant_config = {"format": "int8_tensorwise"}
            n_plain += 1

        new_sd[key] = q
        new_sd[f"{module_key}.weight_scale"] = scale
        new_sd[f"{module_key}.comfy_quant"] = k2._encode_comfy_quant(quant_config)
        meta_layers[k2._meta_base_key(module_key)] = dict(quant_config)

    return new_sd, meta_layers, {
        "linear": n_linear,
        "conv2d": n_conv2d,
        "plain": n_plain,
        "kept": n_kept,
    }


def _quantize_state_dict(
    sd,
    group_size,
    enable_convrot,
    per_channel_int8,
    n8,
    model_type: str = "Z Image",
    k2=None,
):
    """In-memory ConvRot INT8 packing (supports Z Image, Qwen Image Edit, and Krea2)."""
    if model_type == "Krea2":
        if k2 is None:
            k2 = _load_krea2_native_int8()
        return _quantize_state_dict_krea2(
            sd, group_size, enable_convrot, per_channel_int8, k2
        )

    new_sd = {}
    meta_layers = {}
    n_linear = n_conv2d = n_plain = n_kept = 0

    for key, tensor in sd.items():
        if model_type == "Qwen Image Edit" and _is_qwen_blacklisted(key):
            new_sd[key] = tensor
            n_kept += 1
            continue

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


class NativeConvRotInt8Quantize:
    """Quantize a loaded diffusion model (MODEL input) to native ConvRot INT8."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["Z Image", "Qwen Image Edit", "Krea2"], {"default": "Z Image"}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "benchmark_prompt": (
                    "STRING",
                    {
                        "default": "masterpiece, best quality, 1girl, solo, standing, simple background",
                        "multiline": True,
                    },
                ),
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

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        import time
        return float(time.time())

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "report")
    FUNCTION = "quantize"
    CATEGORY = "HSWQ/Quantize"
    OUTPUT_NODE = True

    def quantize(
        self,
        model_type,
        model,
        clip,
        benchmark_prompt,
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
        import time
        ts = int(time.time())
        default_name = f"{original_name}_native_convrot_int8_{ts}.safetensors"
        if not output_path:
            output_path = os.path.join(_output_dir(), default_name)
        output_path = os.path.abspath(output_path)
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, default_name)
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        new_sd, meta_layers, stats = _quantize_state_dict(
            sd,
            group_size,
            bool(convrot),
            bool(per_channel_int8),
            n8,
            model_type=model_type,
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
            import random
            import datetime

            try:
                # Prepare baseline tokens
                prompt_text = (benchmark_prompt or "").strip()
                if not prompt_text:
                    prompt_text = "masterpiece, best quality, 1girl, solo, standing, simple background"
                tokens = clip.tokenize(prompt_text)
                positive = clip.encode_from_tokens_scheduled(tokens)
                negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
                
                device = mm.intermediate_device()
                width, height = 1024, 1024
                latent_base = torch.zeros([1, 16, height // 8, width // 8], device=device)
                latent_base = comfy_sample.fix_empty_latent_channels(model, latent_base)

                is_krea2 = (model_type == "Krea2")
                steps = 12
                cfg = 1.0 if is_krea2 else 2.5
                num_seeds = 20 if is_krea2 else 10
                force_full_denoise = True if is_krea2 else False
                seeds = [random.randint(1, 10000000) for _ in range(num_seeds)]

                def _cos(a, b):
                    a = a.reshape(1, -1).float()
                    b = b.reshape(1, -1).float()
                    return float(torch.nn.functional.cosine_similarity(a, b, dim=1).item())

                def _mse(a, b):
                    return float((a.float() - b.float()).pow(2).mean().item())

                def _sample(m, s):
                    lat = latent_base.clone()
                    noise = comfy_sample.prepare_noise(lat, s, None)
                    xs, x0s = [], []

                    def cb(step, x0, x, total_steps):
                        xs.append(x.detach().float().cpu())
                        x0s.append(x0.detach().float().cpu())

                    out = comfy_sample.sample(
                        m, noise, steps, cfg, "euler", "simple",
                        positive, negative, lat, denoise=1.0,
                        disable_noise=False, start_step=None, last_step=None,
                        force_full_denoise=force_full_denoise, noise_mask=None,
                        callback=cb, disable_pbar=True, seed=s,
                    )
                    return out, xs, x0s

                # 1. FP16 baseline inference for all seeds
                lat_fp16_list = []
                xs_fp16_list = []
                x0s_fp16_list = []
                t_fp16_list = []
                for s in seeds:
                    t0 = time.perf_counter()
                    out_fp16, xs_fp16, x0s_fp16 = _sample(model, s)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_fp16_list.append(time.perf_counter() - t0)
                    lat_fp16_list.append(out_fp16.detach().float().cpu())
                    xs_fp16_list.append(xs_fp16)
                    x0s_fp16_list.append(x0s_fp16)

                # Free FP16 model temporarily to load INT8
                mm.unload_all_models()
                mm.soft_empty_cache()

                # 2. Load INT8 model
                t0 = time.perf_counter()
                model_int8 = comfy.sd.load_diffusion_model(output_path, {})
                load_int8 = time.perf_counter() - t0

                # 3. INT8 inference for all seeds
                lat_int8_list = []
                xs_int8_list = []
                x0s_int8_list = []
                t_int8_list = []
                for s in seeds:
                    t0 = time.perf_counter()
                    out_int8, xs_int8, x0s_int8 = _sample(model_int8, s)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_int8_list.append(time.perf_counter() - t0)
                    lat_int8_list.append(out_int8.detach().float().cpu())
                    xs_int8_list.append(xs_int8)
                    x0s_int8_list.append(x0s_int8)

                # VAE decode removed — trajectory comparison is latent-space only

                # 4. Metrics evaluation
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                report.append(f"\n=== BENCHMARK ({num_seeds} Random Seeds, per-step trajectory) ===")
                report.append(f"Run Time: {current_time}")
                report.append(f"Model Architecture: {model_type}")
                report.append(f"Prompt: {prompt_text}")
                report.append(f"INT8 Model Load Time: {load_int8:.2f}s")
                report.append(f"Seeds: {seeds}\n")

                mse_list = []
                cos_list = []
                final_rows = []
                BIFURC_DROP = 0.05
                SAME_IMG_COS = 0.98
                n_bif = 0
                n_same = 0

                for i, s in enumerate(seeds):
                    a = lat_fp16_list[i].reshape(-1)
                    b = lat_int8_list[i].reshape(-1)
                    mse_val = _mse(a, b)
                    cos_val = _cos(a, b)
                    mse_list.append(mse_val)
                    cos_list.append(cos_val)

                    # per-step trajectory divergence + bifurcation detection
                    fxs = xs_fp16_list[i]
                    nxs = xs_int8_list[i]
                    n_steps = min(len(fxs), len(nxs))
                    step_cos = [_cos(fxs[j], nxs[j]) for j in range(n_steps)]
                    max_drop = 0.0
                    drop_at = 0
                    for j in range(1, n_steps):
                        d = step_cos[j - 1] - step_cos[j]
                        if d > max_drop:
                            max_drop, drop_at = d, j

                    if max_drop > BIFURC_DROP:
                        verdict = f"bifurcated @step {drop_at}"
                        n_bif += 1
                    elif cos_val >= SAME_IMG_COS:
                        verdict = "same-image"
                        n_same += 1
                    else:
                        verdict = "drifted (different image)"

                    final_rows.append({"seed": s, "cos": cos_val, "mse": mse_val, "max_drop": max_drop, "verdict": verdict})
                    line = f"[{i+1}/{num_seeds} | Seed {s}] FP16: {t_fp16_list[i]:.2f}s | INT8: {t_int8_list[i]:.2f}s | MSE: {mse_val:.4f} | Cosine: {cos_val:.4f} | max-drop: {max_drop:.4f} | {verdict}"
                    report.append(line)

                if is_krea2:
                    report.append("\n--- Multi-seed summary ---")
                    report.append(f"{'seed':>8} {'final-cos':>10} {'final-mse':>12} {'max-drop':>9} {'verdict':>22}")
                    for r in final_rows:
                        report.append(f"{r['seed']:>8} {r['cos']:>10.5f} {r['mse']:>12.3e} {r['max_drop']:>9.4f} {r['verdict']:>22}")

                # Summary Averages
                avg_fp16 = sum(t_fp16_list) / len(t_fp16_list)
                avg_int8 = sum(t_int8_list) / len(t_int8_list)
                avg_mse = sum(mse_list) / len(mse_list)
                avg_cos = sum(cos_list) / len(cos_list)
                min_cos = min(cos_list)
                max_cos = max(cos_list)

                report.append(f"\n--- Summary ({num_seeds}-Seed Average) ---")
                report.append(f"Avg FP16 Time: {avg_fp16:.2f}s")
                report.append(f"Avg INT8 Time: {avg_int8:.2f}s")
                speedup = (avg_fp16 / avg_int8) if avg_int8 > 0 else 0.0
                report.append(f"Speedup: {speedup:.2f}x")
                report.append(f"Avg MSE: {avg_mse:.4f}")
                report.append(f"Avg Cosine: {avg_cos:.4f}")
                report.append(f"Cosine: min={min_cos:.4f} max={max_cos:.4f}")
                report.append(f"same-image seeds : {n_same}/{num_seeds}")
                report.append(f"bifurcated seeds : {n_bif}/{num_seeds}   (sudden trajectory jump = different picture, not degradation)")

                mm.unload_all_models()
                mm.soft_empty_cache()

            except Exception as e:
                report.append(f"\n[Benchmark Error] {str(e)}")

        return (output_path, "\n".join(report))


# Backward compatibility alias
ZImageConvRotInt8Quantize = NativeConvRotInt8Quantize

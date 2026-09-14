"""HSWQ SDXL ConvRot INT8 trajectory-impact node (diag / Step 1).

Node form of ``sdxl/diag_impact_sdxl.py`` (Step 1 of the SDXL reverse-hybrid
ConvRot INT8 method). For every ConvRot-eligible Linear/Conv2d of a loaded SDXL
UNet it injects that layer's ConvRot INT8 reconstruction, runs the production
sampling trajectory, records how far the latent drifts from the pristine run,
and writes the per-layer impact json (ascending = safest first) that the
reverse-hybrid node consumes.

The measurement runs in-process on the MODEL / CLIP already in the graph, so no
checkpoint path input is needed:

  candidate set : every ConvRot-eligible Linear/Conv2d (boundary layers excluded)
  injection     : dequant(per-channel INT8(W @ H^T))  (the ConvRot INT8 kernel,
                  no inverse rotation - the shipped weights are stored rotated)
  trajectory    : comfy.sample.sample with the production sampler
                  (dpmpp_2m / karras / cfg 7.0, fixed seed) - identical to
                  benchmark/sdxl_int8_traj_compare.py
  metric        : mean per-step relative MSE of the latent vs the pristine run

Output (default ``<repo>/impact/impact_<name>.json``; a bare filename is placed
in ``<repo>/impact/``)::

    {"x_ref_norm", "steps", "seed", "cfg", "sampler", "scheduler", "base",
     "impacts": {<module name>: <relative MSE>, ...}}

The impact ranking is checkpoint-specific: re-measure for every model.
"""
from __future__ import annotations

import json
import math
import os
import time

import torch

# Impact json output: <repo>/impact (repo-relative; never a machine path).
# This file lives in <repo>/nodes/, so the repo root is two levels up.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IMPACT_DIR = os.path.join(_REPO_ROOT, "impact")

_DEFAULT_PROMPT = "masterpiece, best quality, 1girl, solo, standing, simple background"

# SDXL UNet boundary layers: always kept at FP16, never candidates.
_BOUNDARY_EXACT = ("input_blocks.0.0",)  # conv_in
_BOUNDARY_PREFIX = ("out.", "time_embed.", "add_embedding.", "label_emb.")


def _resolve_output_path(output_path: str, base_name: str) -> str:
    """Target impact json: omitted / bare filename -> <repo>/impact/."""
    output_path = (output_path or "").strip()
    if not output_path:
        stem = os.path.splitext(os.path.basename(base_name or "model"))[0] or "model"
        return os.path.join(_IMPACT_DIR, f"impact_{stem}.json")
    if os.path.dirname(output_path):
        return output_path
    return os.path.join(_IMPACT_DIR, output_path)


def is_boundary_layer(module_name: str) -> bool:
    """True for the SDXL UNet boundary modules, which are always kept at FP16."""
    b = module_name
    for p in ("model.diffusion_model.", "diffusion_model."):
        if b.startswith(p):
            b = b[len(p):]
            break
    return b in _BOUNDARY_EXACT or b.startswith(_BOUNDARY_PREFIX)


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
    """Offline Linear: W_rot = W @ H^T (group-wise). Matches comfy_kitchen._rotate_weight."""
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(f"in_features {in_features} not divisible by group_size {group_size}")
    group_count = in_features // group_size
    grouped = weight.reshape(out_features, group_count, group_size)
    return torch.matmul(grouped, h_matrix.T.to(weight.dtype)).reshape(weight.shape)


def rotate_weight_conv2d(weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int) -> torch.Tensor:
    """Offline Conv2d: rotate along in_channels. weight (O, I, kH, kW)."""
    if weight.ndim != 4:
        raise ValueError(f"Conv2d weight must be 4D, got ndim={weight.ndim}")
    out_c, in_c, k_h, k_w = weight.shape
    flat = weight.permute(0, 2, 3, 1).contiguous().reshape(-1, in_c)
    flat_rot = rotate_weight(flat, h_matrix, group_size)
    return flat_rot.reshape(out_c, k_h, k_w, in_c).permute(0, 3, 1, 2).contiguous()


def quantize_int8_rowwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    abs_max = w.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
    scale = abs_max / 127.0
    q = (w / scale.to(w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def quantize_int8_channelwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    reduce_dims = tuple(range(1, w.dim()))
    amax = torch.clamp(w.abs().amax(dim=reduce_dims).reshape(-1), min=1e-6)
    scale = amax / 127.0
    if w.dim() == 4:
        scale_view = scale.view(-1, 1, 1, 1)
    elif w.dim() == 2:
        scale_view = scale.view(-1, 1)
    else:
        raise ValueError(f"unsupported weight ndim={w.dim()}")
    q = (w / scale_view.to(w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, scale_view.to(torch.float32)


def convrot_int8_quant_error(w: torch.Tensor, group_size: int = 256) -> torch.Tensor | None:
    """Reconstruction injected for one layer: dequant(per-channel INT8(W @ H^T)).

    Linear 2D : W@H^T -> rowwise [out,1]       (pairs with the kitchen online act rotate)
    Conv2d 4D : in_channels rotate -> channelwise [out,1,1,1]
    Layers whose in_dim is not divisible by any power-of-4 group size are not
    ConvRot-eligible (None).
    """
    n_in = w.shape[1]
    gs = convrot_group_size_for_features(n_in, group_size)
    if gs is None:
        return None
    h = build_hadamard(gs, device=w.device, dtype=torch.float32)
    if w.ndim == 2:
        w_rot = rotate_weight(w.float(), h, gs)
        q, s = quantize_int8_rowwise(w_rot)
    elif w.ndim == 4:
        w_rot = rotate_weight_conv2d(w.float(), h, gs)
        q, s = quantize_int8_channelwise(w_rot)
    else:
        return None
    return q.float() * s.float()


def rel_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(a.shape[0], -1)
    b = b.float().reshape(b.shape[0], -1)
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


def _model_base_name(model) -> str:
    """Checkpoint file name of a loaded MODEL, when the loader recorded the path."""
    try:
        if getattr(model, "cached_patcher_init", None):
            func, args = model.cached_patcher_init[:2]
            if args and isinstance(args, tuple) and isinstance(args[0], str):
                return os.path.basename(args[0])
    except Exception:
        pass
    return ""


class HSWQSDXLDiagImpact:
    """Measure the per-layer ConvRot INT8 trajectory impact of a loaded SDXL UNet."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import comfy.samplers

            samplers = list(comfy.samplers.KSampler.SAMPLERS)
            schedulers = list(comfy.samplers.KSampler.SCHEDULERS)
        except Exception:
            samplers = ["dpmpp_2m", "euler", "euler_ancestral", "dpmpp_sde", "ddim", "uni_pc"]
            schedulers = ["karras", "normal", "simple", "exponential", "sgm_uniform", "ddim_uniform"]
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200,
                                  "tooltip": "Trajectory denoising steps. The production measurement uses 25."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                 "tooltip": "Fixed trajectory seed (production: 42)."}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler": (samplers, {"default": "dpmpp_2m"}),
                "scheduler": (schedulers, {"default": "karras"}),
                "prompt": ("STRING", {"default": _DEFAULT_PROMPT, "multiline": True}),
                "negative": ("STRING", {"default": "", "multiline": True}),
                "group_size": ("INT", {"default": 256, "min": 4, "max": 4096,
                                       "tooltip": "Preferred ConvRot Hadamard group size (power of 4)."}),
                "limit": ("INT", {"default": 0, "min": 0, "max": 10000,
                                  "tooltip": "Measure only the first N candidates (0 = all). Debug only."}),
                "output_path": ("STRING", {"default": "", "multiline": False,
                                           "tooltip": "Impact json path. Empty -> <repo>/impact/impact_<model>.json."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("impact_json", "report")
    FUNCTION = "measure"
    CATEGORY = "HSWQ/Quantize"
    TITLE = "HSWQ SDXL Diag Impact (ConvRot INT8)"
    OUTPUT_NODE = False

    def measure(self, model, clip, steps, seed, width, height, cfg, sampler, scheduler,
                prompt, negative, group_size, limit, output_path):
        import comfy.model_management as mm
        import comfy.sample as comfy_sample

        steps = int(steps)
        group_size = int(group_size)
        if group_size < 4 or (group_size & (group_size - 1)) != 0 or math.log(group_size, 4) % 1 != 0:
            raise ValueError(f"group_size must be a power of 4 (>=4), got {group_size}")

        patcher = model
        net = patcher.model
        device = mm.get_torch_device()

        # Candidate set: every ConvRot-eligible Linear/Conv2d (boundary excluded).
        mods = {}
        for n, m in net.named_modules():
            if not hasattr(m, "weight") or m.weight is None:
                continue
            if m.weight.ndim not in (2, 4):
                continue
            if m.weight.shape[1] < 4:
                continue
            if is_boundary_layer(n):
                continue
            gs = convrot_group_size_for_features(int(m.weight.shape[1]), group_size)
            if gs is None:
                continue
            mods[n] = (m, gs)

        targets = sorted(mods.keys())
        if limit and int(limit) > 0:
            targets = targets[: int(limit)]

        base_name = _model_base_name(patcher)
        out_path = _resolve_output_path(output_path, base_name)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        report = [
            f"model candidates (ConvRot-eligible, boundary excluded): {len(mods)}",
            f"measuring: {len(targets)}  steps={steps} seed={int(seed)} cfg={float(cfg)} "
            f"{sampler}/{scheduler} {int(width)}x{int(height)}",
            f"impact json -> {out_path}",
        ]
        print(f"[HSWQ SDXL diag] candidates={len(mods)} measuring={len(targets)}", flush=True)

        t0 = time.perf_counter()

        # Conditioning + latent exactly as the trajectory benchmark builds them.
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative_cond = clip.encode_from_tokens_scheduled(clip.tokenize(negative))
        latent0 = torch.zeros(
            [1, 4, int(height) // 8, int(width) // 8],
            device=mm.intermediate_device(),
            dtype=mm.intermediate_dtype(),
        )
        latent0 = comfy_sample.fix_empty_latent_channels(patcher, latent0, 8, None)
        if latent0.device.type != device.type:
            latent0 = latent0.to(device)

        def run_trajectory():
            noise = comfy_sample.prepare_noise(latent0, int(seed), None)
            xs = []

            def cb(step, x0, x, total_steps):
                xs.append(x.detach().float().cpu())

            with torch.no_grad():
                comfy_sample.sample(
                    patcher, noise, steps, float(cfg), sampler, scheduler,
                    positive, negative_cond, latent0, denoise=1.0,
                    disable_noise=False, start_step=None, last_step=None,
                    force_full_denoise=False, noise_mask=None,
                    callback=cb, disable_pbar=True, seed=int(seed),
                )
            return xs

        print("[HSWQ SDXL diag] pristine trajectory...", flush=True)
        x_ref = run_trajectory()

        impacts = {}
        done = 0
        for n in targets:
            m, gs = mods[n]
            w0 = m.weight.data.clone()
            qerr = convrot_int8_quant_error(w0, gs)
            if qerr is None:
                print(f"  SKIP (no group): {n}", flush=True)
                continue
            m.weight.data.copy_(qerr.to(w0.dtype))
            try:
                x_t = run_trajectory()
                imp = sum(rel_mse(x_t[i], x_ref[i]) for i in range(len(x_ref))) / len(x_ref)
            except Exception as e:  # keep going: one bad layer must not kill the scan
                print(f"  ERR {n}: {e}", flush=True)
                imp = float("nan")
            finally:
                m.weight.data.copy_(w0)
            del w0, qerr
            impacts[n] = imp
            done += 1
            if done % 25 == 0 or done == len(targets):
                print(f"  [{done}/{len(targets)}]", flush=True)

        xr = x_ref[-1].float().reshape(1, -1)
        payload = {
            "x_ref_norm": float((xr * xr).sum().item()),
            "steps": steps,
            "seed": int(seed),
            "cfg": float(cfg),
            "sampler": sampler,
            "scheduler": scheduler,
            "base": base_name,
            "impacts": impacts,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)

        elapsed = time.perf_counter() - t0
        ranked = sorted((v, k) for k, v in impacts.items() if not math.isnan(v))
        report.append(f"measured: {len(impacts)}  elapsed: {elapsed:.1f}s")
        if ranked:
            report.append(f"safest (lowest impact) first 5: " +
                          ", ".join(f"{k}={v:.3e}" for v, k in ranked[:5]))
            report.append(f"impact: min={ranked[0][0]:.3e} max={ranked[-1][0]:.3e}")
        report.append(f"saved: {out_path}")
        print(f"[HSWQ SDXL diag] saved {out_path} ({elapsed:.1f}s)", flush=True)
        return (out_path, "\n".join(report))


NODE_CLASS_MAPPINGS = {
    "HSWQSDXLDiagImpact": HSWQSDXLDiagImpact,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HSWQSDXLDiagImpact": "HSWQ SDXL Diag Impact (ConvRot INT8)",
}

# -*- coding: utf-8 -*-
"""Reverse hybrid INT8 converter for SDXL: FP16 baseline -> K lowest-impact layers ConvRot INT8.

SDXL counterpart of Z_Image/gen_reverse_nvfp4.py, INT8 variant. Method: reverse hybrid
(see md/diag_impact_trajectory_sensitivity_technical_guide.md).

  - Input: the FP16 baseline checkpoint and the impact json from diag_impact_sdxl.py.
  - The K lowest-impact layers (ascending impact) are converted to ConvRot INT8:
      W_rot = W @ H^T  ->  per-channel INT8 (Linear: rowwise [out,1] /
      Conv2d: channelwise [out,1,1,1])  ->  the rotated INT8 weight is stored as-is
      (same kernel as native_convert_int8_sdxl.py; no inverse rotation).
  - Every other layer is left untouched: FP16, same dtype and size as the baseline.

Output is the hswq convrot int8 mixed checkpoint:
    converted layers : .weight int8 + .weight_scale + .comfy_quant
                       {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":N}
    kept layers      : the baseline float weight copied as-is

ComfyUI mixed-precision ops select the quantized path per layer from the .comfy_quant marker, so a
single file carries both the FP16-kept and the ConvRot INT8 layers.

Usage:
    python gen_reverse_int8_sdxl.py <K> <out_name.safetensors> <base.safetensors> <impact.json> \
        [--out-dir <output-dir>] [--groupsize 256]
        [--bias_correction --calib_file <prompts> --comfy_path <root>
         [--num_calib_samples 32] [--num_inference_steps 25] [--calib_seed 42]]

Validation:
    python benchmark/sdxl_int8_traj_compare.py --fp16 <base> --int8 <out> ...
"""
import argparse
import json
import math
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# --- ConvRot kernel (same math as native_convert_int8_sdxl.py; inlined) ---

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
    """Per-output-channel INT8 for Linear: weight_scale [out, 1]."""
    abs_max = w.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
    scale = abs_max / 127.0
    q = (w / scale.to(w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def quantize_int8_channelwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-out-channel INT8 for Conv2d: weight_scale [out, 1, 1, 1]."""
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


def _encode_comfy_quant(config: dict) -> torch.Tensor:
    """Compact JSON bytes as a U8 tensor (ComfyUI layout)."""
    return torch.tensor(
        list(json.dumps(config, separators=(",", ":")).encode("utf-8")),
        dtype=torch.uint8,
    )


def rotate_activation_lastdim(x: torch.Tensor, h_matrix: torch.Tensor, group_size: int) -> torch.Tensor:
    """Online Linear rotate: x_rot = x @ H (last dim = features). Matches the runtime."""
    orig = x.shape
    f = orig[-1]
    gc = f // group_size
    xg = x.reshape(-1, gc, group_size)
    h = h_matrix.to(dtype=x.dtype, device=x.device)
    return torch.matmul(xg, h).reshape(orig)


def rotate_activation_nchw(x: torch.Tensor, h_matrix: torch.Tensor, group_size: int) -> torch.Tensor:
    """Online Conv2d rotate: rotate the channel dim of an NCHW activation."""
    xp = x.permute(0, 2, 3, 1).contiguous()
    xr = rotate_activation_lastdim(xp, h_matrix, group_size)
    return xr.permute(0, 3, 1, 2).contiguous()


# --- SDXL UNet boundary layers (same exclusion set as diag_impact_sdxl.py) ---

_PROTECT_PATTERNS = (
    "conv_in.",
    "conv_out.",
    "time_embed.",
    "add_embedding.",
    "label_emb.",
)


def is_protected(module_name: str) -> bool:
    return any(p in module_name for p in _PROTECT_PATTERNS)


# --- ComfyUI bootstrap (self-contained; only used for --bias_correction) ---

def _clear_argv_for_comfy():
    saved = list(sys.argv)
    sys.argv = [sys.argv[0]] if sys.argv else []
    return saved


def _setup_comfy(comfy_path: str) -> None:
    root = os.path.abspath(comfy_path)
    if not os.path.isdir(os.path.join(root, "comfy")):
        raise FileNotFoundError(f"comfy/ package missing under: {root}")
    sys.path = [root] + [p for p in sys.path if os.path.abspath(p or ".") != root]
    import comfy.options
    comfy.options.args_parsing = False
    import comfy.model_management  # noqa: F401
    import comfy.ops  # noqa: F401
    import comfy.sample  # noqa: F401
    import comfy.sd  # noqa: F401
    import comfy.utils  # noqa: F401


def _load_sdxl(path: str):
    import comfy.sd
    saved = _clear_argv_for_comfy()
    try:
        out = comfy.sd.load_checkpoint_guess_config(
            os.path.abspath(path), output_vae=True, output_clip=True, embedding_directory=None
        )
    finally:
        sys.argv = saved
    return out[0], out[1], out[2]


def collect_rotated_act_means(
    base_path: str,
    comfy_path: str,
    targets: dict,
    *,
    calib_file: str,
    num_samples: int,
    num_steps: int,
    prompt_fallback: str,
    width: int,
    height: int,
    seed: int,
    device: str,
) -> dict:
    """Run a fixed-seed calibration pass and return {module_name: mean rotated activation}.

    `targets` maps a ComfyUI module name -> (group_size, weight_ndim). The rotated mean is
    accumulated per in-channel of the ROTATED activation (the online rotate the runtime applies).
    """
    import comfy.model_management as mm
    import comfy.sample as comfy_sample

    _setup_comfy(comfy_path)
    patcher, clip, _vae = _load_sdxl(base_path)
    net = patcher.model
    net.to(device)
    net.eval()

    prompts = []
    with open(calib_file, "r", encoding="utf-8") as f:
        prompts = [ln.strip() for ln in f if ln.strip()]
    if not prompts:
        prompts = [prompt_fallback]
    if len(prompts) < num_samples:
        prompts = (prompts * (num_samples // len(prompts) + 1))[:num_samples]
    else:
        prompts = prompts[:num_samples]

    acc = {name: None for name in targets}
    count = {name: 0 for name in targets}

    def make_hook(name, gs, ndim):
        def hook(module, inp):
            x = inp[0] if isinstance(inp, (tuple, list)) else inp
            if not torch.is_tensor(x) or not torch.is_floating_point(x):
                return
            xd = x.detach().float()
            h = build_hadamard(gs, device=xd.device, dtype=torch.float32)
            if ndim == 4:
                if xd.ndim != 4:
                    return
                xr = rotate_activation_nchw(xd, h, gs)          # (N, C, H, W)
                m = xr.mean(dim=(0, 2, 3))                      # (C,)
            else:
                f = xd.shape[-1]
                if f % gs != 0:
                    return
                xr = rotate_activation_lastdim(xd, h, gs)       # (..., F)
                m = xr.reshape(-1, f).mean(dim=0)               # (F,)
            if acc[name] is None:
                acc[name] = m.cpu()
                count[name] = 1
            else:
                acc[name] += m.cpu()
                count[name] += 1
        return hook

    handles = []
    mod_dict = dict(net.named_modules())
    for name, (gs, ndim) in targets.items():
        if name in mod_dict:
            handles.append(mod_dict[name].register_forward_pre_hook(make_hook(name, gs, ndim)))

    latent = torch.zeros([1, 4, height // 8, width // 8],
                         device=mm.intermediate_device(), dtype=mm.intermediate_dtype())
    latent = comfy_sample.fix_empty_latent_channels(patcher, latent, 8, None)
    if latent.device.type != device:
        latent = latent.to(device)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt_fallback))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))

    print(f"[calib] {num_samples} samples x {num_steps} steps, seed {seed}", flush=True)
    for i in range(num_samples):
        noise = comfy_sample.prepare_noise(latent, seed + i, None)
        with torch.no_grad():
            comfy_sample.sample(
                patcher, noise, num_steps, 7.0, "dpmpp_2m", "karras",
                positive, negative, latent, denoise=1.0,
                disable_noise=False, start_step=None, last_step=None,
                force_full_denoise=False, noise_mask=None,
                callback=None, disable_pbar=True, seed=seed + i,
            )
        if (i + 1) % 8 == 0 or (i + 1) == num_samples:
            print(f"  [calib {i + 1}/{num_samples}]", flush=True)

    for h in handles:
        h.remove()

    means = {}
    for name in targets:
        if acc[name] is not None and count[name] > 0:
            means[name] = (acc[name] / count[name])
    print(f"[calib] collected rotated act means for {len(means)}/{len(targets)} layers", flush=True)

    del net, patcher, clip
    if device == "cuda":
        torch.cuda.empty_cache()
    return means


def compute_bias_delta_rotated(err_rot: torch.Tensor, mu_rot: torch.Tensor) -> torch.Tensor | None:
    """delta[o] = sum_j err_rot[o, j] * mu_rot[j]  (error contracted with mean rotated activation)."""
    if mu_rot is None:
        return None
    err = err_rot.float()
    mu = mu_rot.float().to(err.device)
    if err.ndim == 2:
        if mu.numel() != err.shape[1]:
            return None
        return err @ mu
    if err.ndim == 4:
        if mu.numel() != err.shape[1]:
            return None
        return (err * mu.view(1, -1, 1, 1)).sum(dim=(1, 2, 3))
    return None


def parse_args():
    ap = argparse.ArgumentParser(
        description="FP16 baseline -> K lowest-impact layers ConvRot INT8 (hswq mixed checkpoint)"
    )
    ap.add_argument("k", help="number of layers (ascending impact) to convert to ConvRot INT8, or 'all'")
    ap.add_argument("out_name", help="output filename (created under the output directory)")
    ap.add_argument("base", help="FP16 baseline SDXL checkpoint (full ckpt)")
    ap.add_argument("impact", help="impact json from diag_impact_sdxl.py")
    ap.add_argument("--out-dir", default=".", help="output directory")
    ap.add_argument("--groupsize", type=int, default=256,
                    help="ConvRot Hadamard group size (power of 4, default 256)")
    ap.add_argument("--bias_correction", action="store_true",
                    help="collect rotated act means on a calibration pass and cancel the INT8 bias shift")
    ap.add_argument("--calib_file", default=None, help="calibration prompts file (required with --bias_correction)")
    ap.add_argument("--comfy_path", default="ComfyUI-master", help="ComfyUI root (required with --bias_correction)")
    ap.add_argument("--num_calib_samples", type=int, default=32)
    ap.add_argument("--num_inference_steps", type=int, default=25)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--calib_seed", type=int, default=42)
    return ap.parse_args()


def main():
    a = parse_args()
    out = os.path.join(a.out_dir, a.out_name)

    with open(a.impact, encoding="utf-8") as f:
        imp = json.load(f)["impacts"]
    # ascending = safest first; diag keys are named_modules names (with diffusion_model.)
    ranked = []
    for k, v in sorted(imp.items(), key=lambda kv: kv[1]):
        if isinstance(v, float) and math.isnan(v):
            continue
        if k.endswith(".weight"):
            k = k[: -len(".weight")]
        ranked.append(k)
    k_req = len(ranked) if str(a.k).lower() == "all" else int(a.k)
    print(f"ranked layers available: {len(ranked)}")
    print(f"converting: {min(k_req, len(ranked))} layer(s)")

    print(f"loading baseline: {a.base}")
    with safe_open(os.path.abspath(a.base), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        raw_meta = f.metadata() or {}
        sd = {k: f.get_tensor(k) for k in keys}

    prefix = ""
    for cand in ("model.diffusion_model.", "diffusion_model."):
        if any(k.startswith(cand) for k in keys):
            prefix = cand
            break
    print(f"UNet key prefix: {prefix!r}")

    def module_to_sd_key(mod_name: str) -> str | None:
        """Map a diag module name (named_modules) to the checkpoint module key."""
        for pre in ("model.diffusion_model.", "diffusion_model."):
            if pre + mod_name + ".weight" in sd:
                return pre + mod_name
            bare = mod_name
            if bare.startswith("diffusion_model."):
                bare = bare[len("diffusion_model."):]
                if pre + bare + ".weight" in sd:
                    return pre + bare
        return None

    quant_meta_layers = {}
    converted = 0
    skipped_protected = 0
    skipped_not_found = 0
    skipped_shape = 0

    mu_rot = {}
    bias_applied = 0
    bias_skipped_no_bias = 0
    bias_skipped_no_act = 0
    if a.bias_correction:
        if not a.calib_file or not os.path.isfile(a.calib_file):
            raise FileNotFoundError(f"--calib_file not found: {a.calib_file}")

        def sd_key_to_module(k: str) -> str:
            # model.model (BaseModel) exposes diffusion_model.* (no leading "model.")
            if k.startswith("model.diffusion_model."):
                return k[len("model."):]
            return k

        plan = []
        for name in ranked[: k_req]:
            mk = module_to_sd_key(name)
            if mk is None or is_protected(name):
                continue
            w = sd.get(mk + ".weight")
            if w is None or w.ndim not in (2, 4):
                continue
            gs = convrot_group_size_for_features(int(w.shape[1]), a.groupsize)
            if gs is None:
                continue
            plan.append((mk, gs, w.ndim))
        targets = {sd_key_to_module(mk): (gs, nd) for (mk, gs, nd) in plan}
        raw_means = collect_rotated_act_means(
            a.base, a.comfy_path, targets,
            calib_file=a.calib_file, num_samples=a.num_calib_samples,
            num_steps=a.num_inference_steps,
            prompt_fallback="masterpiece, best quality, 1girl, solo, standing, simple background",
            width=a.width, height=a.height, seed=a.calib_seed,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        for (mk, _gs, _nd) in plan:
            m = raw_means.get(sd_key_to_module(mk))
            if m is not None:
                mu_rot[mk] = m

    for name in ranked[: k_req]:
        module_key = module_to_sd_key(name)
        if module_key is None:
            skipped_not_found += 1
            print(f"  SKIP (not in sd): {name}")
            continue
        wk = module_key + ".weight"
        w = sd[wk]
        if w.ndim not in (2, 4) or w.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            skipped_shape += 1
            continue
        if is_protected(name):
            skipped_protected += 1
            print(f"  SKIP (boundary layer): {name}")
            continue

        gs = convrot_group_size_for_features(int(w.shape[1]), a.groupsize)
        if gs is None:
            skipped_shape += 1
            print(f"  SKIP (no eligible group size): {name} in={w.shape[1]}")
            continue

        h = build_hadamard(gs, device="cpu", dtype=torch.float32)
        wf = w.float()
        # rotate then per-channel INT8 (same kernel as the shipped converter; no inverse rotation)
        if w.ndim == 2:
            w_rot = rotate_weight(wf, h, gs)
            q, scale = quantize_int8_rowwise(w_rot)
        else:
            w_rot = rotate_weight_conv2d(wf, h, gs)
            q, scale = quantize_int8_channelwise(w_rot)

        if a.bias_correction:
            bk = module_key + ".bias"
            m = mu_rot.get(module_key)
            if m is None:
                bias_skipped_no_act += 1
            elif bk not in sd:
                bias_skipped_no_bias += 1
            else:
                err_rot = (q.float() * scale.float()) - w_rot
                delta = compute_bias_delta_rotated(err_rot, m)
                if delta is None:
                    bias_skipped_no_act += 1
                else:
                    b = sd[bk]
                    sd[bk] = (b.float() + (-delta).to(b.device).float()).to(b.dtype)
                    bias_applied += 1

        del sd[wk]
        sd[wk] = q
        sd[module_key + ".weight_scale"] = scale
        conf = {
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": int(gs),
        }
        sd[module_key + ".comfy_quant"] = _encode_comfy_quant(conf)
        quant_meta_layers[module_key] = conf
        converted += 1
        if converted % 25 == 0 or converted == k_req:
            print(f"  [{converted}/{a.k}] last: {name}  {tuple(w.shape)}")

    print(f"converted: {converted}, protected-skip: {skipped_protected}, "
          f"not-found-skip: {skipped_not_found}, shape-skip: {skipped_shape}")
    if a.bias_correction:
        print(f"bias correction: applied={bias_applied}, no_bias={bias_skipped_no_bias}, "
              f"no_act={bias_skipped_no_act}")

    metadata = dict(raw_meta)
    metadata["_quantization_metadata"] = json.dumps(
        {"format_version": "1.0", "layers": quant_meta_layers},
        separators=(",", ":"),
    )
    metadata["hswq_reverse_int8"] = json.dumps({
        "base": os.path.abspath(a.base),
        "impact": os.path.abspath(a.impact),
        "k_requested": k_req,
        "converted": converted,
        "groupsize": a.groupsize,
        "bias_correction": bool(a.bias_correction),
        "bias_applied": bias_applied,
    })

    print(f"saving: {out}")
    save_file(sd, out, metadata=metadata)
    size_gb = os.path.getsize(out) / 1e9
    print(f"saved: {out}  {size_gb:.2f} GB (decimal)")


if __name__ == "__main__":
    main()

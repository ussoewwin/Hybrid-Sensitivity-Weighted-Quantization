# -*- coding: utf-8 -*-
"""SDXL per-layer ConvRot INT8 trajectory impact (SDXL counterpart of Z_Image/diag_impact.py).

Reverse hybrid INT8 method (see md/diag_impact_trajectory_sensitivity_technical_guide.md):
  1. diag_impact_sdxl.py      -> impact json (relative MSE per layer, ascending = safest first)
  2. gen_reverse_int8_sdxl.py -> hybrid int{K} artifact (K lowest-impact layers -> ConvRot INT8)

Injects a single layer at a time with its ConvRot INT8 reconstruction
(dequant(per-channel INT8(quantize(W @ H^T))) -- exactly the kernel that produces the shipped INT8
pack; no inverse rotation), runs a fixed-seed N-step denoising trajectory, and records how far the
final latent drifts (relative MSE). Ascending order = the layer can be INT8 without breaking the
trajectory, so the ascending K layers are the ones to convert.

- baseline load is ComfyUI-native (comfy.sd.load_checkpoint_guess_config). No INT8 patch is applied
  (the baseline carries no INT8 tensors, so nothing would activate anyway).
- trajectory: fixed-seed Euler loop (guide 2.2 / 3.3), x_{k+1} = x_k + (sigma_{k+1}-sigma_k)*v(x_k, sigma_k).
  SDXL is epsilon-prediction, so v = eps.
- conditioning: the SDXL real context (CLIP cond + the SDXL adm vector) is built once from the
  baseline CLIP and reused for every run. model.apply_model is called directly; the extra cond "y"
  is the SDXL.encode_adm() output (pooled emb + size embedding, 2816-dim) so the normal model path
  is used.
- injection overwrites named_modules weight.data directly and restores it afterwards (guide 3.4).

Usage:
    python diag_impact_sdxl.py <base.safetensors> <impact_out.json> \
        --comfy_path <ComfyUI-master> [--steps 4] [--seed 42] \
        [--width 1024] [--height 1024] \
        [--artifact <full_convrot_int8.safetensors>] [--limit N] [--progress-every 25]

With --artifact the measurement is restricted to the layers listed in that artifact's
_quantization_metadata (the candidate set a full ConvRot INT8 pack actually converts).
Without it, every eligible Linear/Conv in the baseline is measured.
"""
import argparse
import json
import math
import os
import sys

import torch


# ---------------------------------------------------------------------------
# ComfyUI bootstrap (same safe import order as benchmark/sdxl_int8_traj_compare.py)
# ---------------------------------------------------------------------------

def _clear_argv_for_comfy():
    saved = list(sys.argv)
    sys.argv = [sys.argv[0]] if sys.argv else []
    return saved


def _restore_argv(saved):
    sys.argv = saved


def setup_comfy(comfy_path: str) -> None:
    comfy_root = os.path.abspath(comfy_path)
    if not os.path.isdir(comfy_root):
        raise FileNotFoundError(f"--comfy_path not found: {comfy_root}")
    if not os.path.isdir(os.path.join(comfy_root, "comfy")):
        raise FileNotFoundError(f"comfy/ package missing under: {comfy_root}")
    sys.path = [comfy_root] + [p for p in sys.path if os.path.abspath(p or ".") != comfy_root]


def import_comfy():
    import comfy.options
    comfy.options.args_parsing = False
    import comfy.model_management  # noqa: F401
    import comfy.ops  # noqa: F401
    import comfy.sample  # noqa: F401
    import comfy.sd  # noqa: F401
    import comfy.utils  # noqa: F401


# ---------------------------------------------------------------------------
# INT8 kernel (ConvRot group size + the true rotated-INT8 quantization error)
# ---------------------------------------------------------------------------

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
    """Reconstruction value injected for one layer:

        What = dequant(per-channel INT8(W @ H^T))

    This is exactly what native_convert_int8_sdxl.py writes into the shipped artifact; no inverse
    rotation is applied (the shipped weights are stored rotated). The error relative to the FP16
    weight is eps = What - W (guide 2.3), i.e. the FP16-vs-ConvRot-INT8 error of that layer.

      Linear 2D : W@H^T -> rowwise [out,1]      (pairs with the kitchen online act rotate)
      Conv2d 4D : in_channels rotate -> channelwise [out,1,1,1] (pairs with the HSWQ INT8 Conv2d)

    Layers whose in_dim is not divisible by any power-of-4 group size are not ConvRot-eligible (None).
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
    return (q.float() * s.float())


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_sdxl(path: str):
    """Load a checkpoint ComfyUI-natively; returns (ModelPatcher, clip, vae).

    The baseline contains no INT8 tensors, so the INT8 load scope is not needed (stock ops).
    """
    import comfy.sd

    saved = _clear_argv_for_comfy()
    try:
        out = comfy.sd.load_checkpoint_guess_config(
            os.path.abspath(path),
            output_vae=True,
            output_clip=True,
            embedding_directory=None,
        )
    finally:
        _restore_argv(saved)
    model, clip, vae = out[0], out[1], out[2]
    if model is None or clip is None:
        raise RuntimeError(f"checkpoint load failed (model/clip None): {path}")
    return model, clip, vae


# ---------------------------------------------------------------------------
# Conditioning / latent / trajectory
# ---------------------------------------------------------------------------

def build_conditioning(model, clip, prompt: str, width: int, height: int):
    """Build the SDXL context (CLIP cond + adm vector) once and reuse it for every run.

    SDXL shapes:
      ctx = c_crossattn  [1, 77, 2048]   (CLIP cond)
      y   = c_adm        [1, 2816]       (pooled 1280 + size embedding 6*256)
    y is produced through the normal model path (model.encode_adm), so model-specific details
    (e.g. SDXLRefiner) stay ComfyUI's responsibility.
    """
    tokens = clip.tokenize(prompt)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

    adm = model.encode_adm(
        pooled_output=pooled,
        width=width,
        height=height,
        crop_w=0,
        crop_h=0,
    )
    return {"c_crossattn": cond, "y": adm}


def make_latent(model, width: int, height: int):
    import comfy.model_management as mm
    import comfy.sample as comfy_sample

    latent = torch.zeros(
        [1, 4, height // 8, width // 8],
        device=mm.intermediate_device(),
        dtype=mm.intermediate_dtype(),
    )
    latent = comfy_sample.fix_empty_latent_channels(model, latent, 8, None)
    return latent


def run_trajectory(net, ctx, y, latent0, *, seed, steps):
    """Fixed-seed Euler trajectory (guide 2.2) for SDXL (epsilon prediction):

      x_{k+1} = x_k + (sigma_{k+1} - sigma_k) * eps(x_k, sigma_k)

    Returns the per-step latent x list (cpu float32).
    """
    noise = torch.randn(latent0.shape, generator=torch.Generator("cpu").manual_seed(seed),
                        dtype=torch.float32).to(latent0.device)
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=latent0.device, dtype=torch.float32)
    x = (latent0 + noise * sigmas[0]).to(latent0.dtype)

    xs = []
    with torch.no_grad():
        for i in range(steps):
            t = torch.tensor([sigmas[i]], device=latent0.device, dtype=torch.float32)
            eps = net.apply_model(
                x, t, c_concat=None,
                c_crossattn=ctx,
                y=y,
                control=None,
                transformer_options={},
            )
            x = (x + (sigmas[i + 1] - sigmas[i]) * eps).to(latent0.dtype)
            xs.append(x.detach().float().cpu())
    return xs


def rel_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(a.shape[0], -1)
    b = b.float().reshape(b.shape[0], -1)
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


# ---------------------------------------------------------------------------
# Boundary layers (same exclusion set as native_convert_int8_sdxl.py)
# ---------------------------------------------------------------------------

_PROTECT_PATTERNS = (
    "conv_in.",
    "conv_out.",
    "time_embed.",
    "add_embedding.",
    "label_emb.",
)


def is_boundary_layer(module_name: str) -> bool:
    return any(p in module_name for p in _PROTECT_PATTERNS)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="SDXL per-layer ConvRot INT8 trajectory impact")
    ap.add_argument("base", help="baseline fp16 SDXL checkpoint (full ckpt: UNet+CLIP+VAE)")
    ap.add_argument("out", help="output impact json path")
    ap.add_argument("--comfy_path", required=True, help="ComfyUI-master root")
    ap.add_argument("--artifact", default=None,
                    help="FULL ConvRot INT8 safetensors; its _quantization_metadata.layers is the measured set")
    ap.add_argument("--steps", type=int, default=4, help="trajectory denoising steps (default 4)")
    ap.add_argument("--seed", type=int, default=42, help="trajectory seed (default 42)")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--prompt", default="masterpiece, best quality, 1girl, solo, standing, simple background")
    ap.add_argument("--groupsize", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="limit the number of measured layers (debug)")
    ap.add_argument("--progress-every", type=int, default=25)
    return ap.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    setup_comfy(args.comfy_path)
    import_comfy()

    print(f"[load] baseline: {args.base}", flush=True)
    patcher, clip, _vae = load_sdxl(args.base)
    net = patcher.model            # BaseModel
    net.to(device)
    net.eval()
    print("[load] done", flush=True)

    # Build the target layer list
    if args.artifact:
        from safetensors import safe_open
        with safe_open(os.path.abspath(args.artifact), framework="pt", device="cpu") as f:
            meta = json.loads(f.metadata()["_quantization_metadata"])
        raw_layers = list(meta["layers"].keys())
        # native_convert_int8 module keys may carry "model.diffusion_model."; normalize to
        # named_modules names and drop boundary layers (same set the converter skips).
        layers = []
        for k in raw_layers:
            for p in ("model.diffusion_model.", "diffusion_model."):
                if k.startswith(p):
                    k = k[len(p):]
                    break
            if is_boundary_layer(k):
                continue
            layers.append(k)
        print(f"[target] artifact layers: {len(layers)} (boundary excluded)", flush=True)
    else:
        layers = None  # all eligible layers (collected below)

    # eligible ConvRot modules among named_modules (boundary excluded)
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
        gs = convrot_group_size_for_features(int(m.weight.shape[1]), args.groupsize)
        if gs is None:
            continue  # not ConvRot-eligible
        mods[n] = (m, gs)
    print(f"[target] modules eligible for ConvRot INT8: {len(mods)}", flush=True)

    if layers is None:
        targets = sorted(mods.keys())
    else:
        def resolve(k):
            if k in mods:
                return k
            for cand in (f"diffusion_model.{k}", f"model.diffusion_model.{k}"):
                if cand in mods:
                    return cand
            return None

        targets = []
        skipped = []
        for k in layers:
            r = resolve(k)
            if r is None:
                skipped.append(k)
            else:
                targets.append(r)
        if skipped:
            print(f"[target] artifact layers not eligible/missing: {len(skipped)} "
                  f"(first 5: {skipped[:5]})", flush=True)
        seen = set()
        uniq = []
        for t in targets:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        targets = uniq
    if args.limit:
        targets = targets[: args.limit]
    print(f"[target] measuring: {len(targets)}", flush=True)

    cond = build_conditioning(net, clip, args.prompt, args.width, args.height)
    latent0 = make_latent(patcher, args.width, args.height)
    if latent0.device.type != device:
        latent0 = latent0.to(device)

    print(f"[*] pristine run (steps={args.steps}, seed={args.seed})", flush=True)
    x_ref = run_trajectory(net, cond["c_crossattn"], cond["y"], latent0,
                           seed=args.seed, steps=args.steps)
    print("[*] pristine done", flush=True)

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
            x_t = run_trajectory(net, cond["c_crossattn"], cond["y"], latent0,
                                 seed=args.seed, steps=args.steps)
            imp = sum(rel_mse(x_t[i], x_ref[i]) for i in range(len(x_ref))) / len(x_ref)
        except Exception as e:
            print(f"  ERR {n}: {e}", flush=True)
            imp = float("nan")
        finally:
            m.weight.data.copy_(w0)
        del w0, qerr
        impacts[n] = imp
        done += 1
        if done % args.progress_every == 0 or done == len(targets):
            print(f"  [{done}/{len(targets)}]", flush=True)

    xr = x_ref[-1].float().reshape(1, -1)
    payload = {
        "x_ref_norm": float((xr * xr).sum().item()),
        "steps": args.steps,
        "seed": args.seed,
        "base": os.path.abspath(args.base),
        "impacts": impacts,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"saved {args.out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

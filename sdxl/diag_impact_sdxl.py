# -*- coding: utf-8 -*-
"""SDXL per-layer ConvRot INT8 trajectory impact (SDXL counterpart of Z_Image/diag_impact.py).

Reverse hybrid INT8 method (see md/diag_impact_trajectory_sensitivity_technical_guide.md):
  1. diag_impact_sdxl.py      -> impact json (relative MSE per layer, ascending = safest first)
  2. gen_reverse_int8_sdxl.py -> hybrid int{K} artifact (K lowest-impact layers -> ConvRot INT8)

The measurement source is the FP16 checkpoint itself: every ConvRot-eligible Linear/Conv2d in the
baseline is a candidate. Nothing else is needed (no other artifact is consulted).

For each candidate layer, a single layer is replaced by its ConvRot INT8 reconstruction
(dequant(per-channel INT8(quantize(W @ H^T))) -- the ConvRot INT8 kernel; no inverse rotation), a
fixed-seed N-step denoising trajectory is run, and the drift of the final latent is recorded
(relative MSE). Ascending order = the layer survives INT8 without breaking the trajectory, so the
ascending K layers are the ones to convert.

- baseline load is ComfyUI-native (comfy.sd.load_checkpoint_guess_config). No INT8 patch is applied
  (the baseline carries no INT8 tensors, so nothing would activate anyway).
- trajectory: the production sampler itself (comfy.sample.sample, dpmpp_2m / karras / cfg 7.0 / the
  same prompt and seed) -- identical to benchmark/sdxl_int8_traj_compare.py, so the measured ranking
  follows exactly the trajectory the quality gate measures. No hand-rolled schedule.
- conditioning: CLIP positive/negative, exactly as the benchmark builds them.
- injection overwrites named_modules weight.data directly and restores it afterwards (guide 3.4).

Usage:
    python diag_impact_sdxl.py <base.safetensors> <impact_out.json> \
        --comfy_path <ComfyUI-master> [--steps 25] [--seed 42] \
        [--width 1024] [--height 1024] [--artifact <convrot_int8_pack>] [--limit N] [--progress-every 25]
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

    The shipped ConvRot INT8 weights are stored rotated, so the reconstruction is produced in the
    same rotated space; no inverse rotation is applied. The error relative to the FP16 weight is
    eps = What - W (guide 2.3), i.e. the FP16-vs-ConvRot-INT8 error of that layer.

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

def build_conditioning(clip, prompt: str, negative: str):
    """Build the CLIP conditioning exactly as the benchmark does (list form for comfy.sample.sample)."""
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    negative_cond = clip.encode_from_tokens_scheduled(clip.tokenize(negative))
    return positive, negative_cond


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


def run_trajectory(patcher, positive, negative, latent0, *, seed, steps, cfg, sampler, scheduler):
    """Run the production sampler (comfy.sample.sample) and capture the per-step latent x.

    Identical engine, sampler, scheduler, cfg and seed as benchmark/sdxl_int8_traj_compare.py, so the
    impact ranking is measured on exactly the trajectory the quality gate uses.
    """
    import comfy.sample as comfy_sample

    noise = comfy_sample.prepare_noise(latent0, seed, None)
    xs = []

    def cb(step, x0, x, total_steps):
        xs.append(x.detach().float().cpu())

    with torch.no_grad():
        comfy_sample.sample(
            patcher, noise, steps, cfg, sampler, scheduler,
            positive, negative, latent0, denoise=1.0,
            disable_noise=False, start_step=None, last_step=None,
            force_full_denoise=False, noise_mask=None,
            callback=cb, disable_pbar=True, seed=seed,
        )
    return xs


def rel_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(a.shape[0], -1)
    b = b.float().reshape(b.shape[0], -1)
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


# ---------------------------------------------------------------------------
# Boundary layers (kept at FP16; excluded from the candidate set)
# ---------------------------------------------------------------------------

_BOUNDARY_EXACT = (
    "input_blocks.0.0",   # conv_in (latent input projection)
)
_BOUNDARY_PREFIX = (
    "out.",              # conv_out (final projection)
    "time_embed.",
    "add_embedding.",
    "label_emb.",
)


def is_boundary_layer(module_name: str) -> bool:
    """True for the SDXL UNet boundary modules, which are always kept at FP16."""
    b = module_name
    for p in ("model.diffusion_model.", "diffusion_model."):
        if b.startswith(p):
            b = b[len(p):]
            break
    return b in _BOUNDARY_EXACT or b.startswith(_BOUNDARY_PREFIX)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="SDXL per-layer ConvRot INT8 trajectory impact")
    ap.add_argument("base", help="baseline fp16 SDXL checkpoint (full ckpt: UNet+CLIP+VAE)")
    ap.add_argument("out", help="output impact json path")
    ap.add_argument("--comfy_path", required=True, help="ComfyUI-master root")
    ap.add_argument("--steps", type=int, default=4, help="trajectory denoising steps (default 4)")
    ap.add_argument("--seed", type=int, default=42, help="trajectory seed (default 42)")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--prompt", default="masterpiece, best quality, 1girl, solo, standing, simple background")
    ap.add_argument("--negative", default="")
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--sampler", default="dpmpp_2m")
    ap.add_argument("--scheduler", default="karras")
    ap.add_argument("--groupsize", type=int, default=256)
    ap.add_argument("--artifact", default=None,
                    help="optional: restrict the measured set to the layers converted in this ConvRot INT8 "
                         "pack (_quantization_metadata), reproducing a pack-derived candidate list; layers the "
                         "pack kept at FP16 are then not measured")
    ap.add_argument("--limit", type=int, default=None, help="limit the number of measured layers (debug)")
    ap.add_argument("--protect_list", default=None,
                    help="optional: json/txt of layer names kept at FP16 (excluded from measurement)")
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

    # Candidate set: every ConvRot-eligible Linear/Conv2d of the baseline (boundary excluded).
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

    if args.artifact:
        # Reproduce a pack-derived candidate list: measure only the layers a ConvRot INT8 pack actually
        # converted, so the layers that pack kept at FP16 stay FP16 in the hybrid.
        from safetensors import safe_open
        with safe_open(os.path.abspath(args.artifact), framework="pt", device="cpu") as f:
            pack_meta = json.loads(f.metadata()["_quantization_metadata"])
        pack_layers = []
        for k in pack_meta["layers"]:
            b = k
            for p in ("model.diffusion_model.", "diffusion_model."):
                if b.startswith(p):
                    b = b[len(p):]
                    break
            pack_layers.append(b)

        def _resolve_pack_layer(b):
            for cand in (f"diffusion_model.{b}", b):
                if cand in mods:
                    return cand
            return None

        targets, skipped = [], []
        for b in pack_layers:
            r = _resolve_pack_layer(b)
            if r is None:
                skipped.append(b)
            else:
                targets.append(r)
        seen, uniq = set(), []
        for t in targets:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        targets = uniq
        print(f"[target] artifact mode: {len(pack_layers)} layers listed, {len(targets)} measured, "
              f"{len(skipped)} not eligible (first 5: {skipped[:5]})", flush=True)
    else:
        targets = sorted(mods.keys())
    if args.protect_list:
        raw = args.protect_list
        if raw.lower().endswith(".json"):
            with open(raw, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key in ("protected", "layers", "names", "protect"):
                    if key in data:
                        data = data[key]
                        break
            names = list(data.keys()) if isinstance(data, dict) else list(data)
        else:
            with open(raw, encoding="utf-8") as f:
                names = [ln.strip() for ln in f if ln.strip()]
        protect = set()
        for n in names:
            b = n[:-len(".weight")] if n.endswith(".weight") else n
            for p in ("model.diffusion_model.", "diffusion_model."):
                if b.startswith(p):
                    b = b[len(p):]
                    break
            protect.add(b)
        before = len(targets)
        targets = [t for t in targets
                   if (t[len("diffusion_model."):] if t.startswith("diffusion_model.") else t) not in protect]
        print(f"[target] protect_list: {len(protect)} names -> excluded {before - len(targets)} of {before}", flush=True)

    if args.limit:
        targets = targets[: args.limit]
    print(f"[target] measuring: {len(targets)}", flush=True)

    positive, negative = build_conditioning(clip, args.prompt, args.negative)
    latent0 = make_latent(patcher, args.width, args.height)
    if latent0.device.type != device:
        latent0 = latent0.to(device)

    print(f"[*] pristine run (steps={args.steps}, seed={args.seed}, cfg={args.cfg}, "
          f"{args.sampler}/{args.scheduler})", flush=True)
    x_ref = run_trajectory(patcher, positive, negative, latent0, seed=args.seed,
                           steps=args.steps, cfg=args.cfg, sampler=args.sampler,
                           scheduler=args.scheduler)
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
            x_t = run_trajectory(patcher, positive, negative, latent0, seed=args.seed,
                                 steps=args.steps, cfg=args.cfg, sampler=args.sampler,
                                 scheduler=args.scheduler)
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
        "cfg": args.cfg,
        "sampler": args.sampler,
        "scheduler": args.scheduler,
        "base": os.path.abspath(args.base),
        "impacts": impacts,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"saved {args.out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

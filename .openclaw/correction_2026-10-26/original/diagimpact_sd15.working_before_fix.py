# -*- coding: utf-8 -*-
"""SD1.5 UNet: per-layer **trajectory** impact, measured on the model's own
denoising output.

Structure and measurement engine follow `sdxl/diag_impact_sdxl.py`:

  * the checkpoint is loaded ComfyUI-natively (`comfy.sd.load_checkpoint_guess_config`),
    which yields a ModelPatcher and a CLIP;
  * the conditioning is built from the prompt / negative through that CLIP;
  * the trajectory is produced by the **production sampler**
    (`comfy.sample.sample`) with the same engine, sampler, scheduler, cfg and
    seed as `benchmark/sd1.5_int8_traj_compare.py`, so the impact ranking is
    measured on exactly the trajectory the quality gate uses;
  * every other layer is held at its pristine weight, ONE layer's weight is
    replaced by its ConvRot INT8 reconstruction **in the original frame**
    (`dequant(INT8(W @ H^T))` = `(q*s) @ H`), and the divergence from the
    pristine trajectory is recorded.

The only SD1.5-specific parts are the resolution, the CLIP that the checkpoint
carries, and the CLIP-based conditioning. Nothing here re-implements sampling.

Usage:
  python sd1.5/diagimpact_sd15.py \
      --comfy <ComfyUI root> \
      --ckpt  <source checkpoint>.safetensors \
      --out   impact/impact_sd15_traj.json \
      --steps 25 --seeds 42,137 \
      --width 512 --height 512 --cfg 7.0 \
      --sampler dpmpp_2m --scheduler karras
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float16

# SD1.5's production resolution. The latent is width/8 x height/8 (the VAE's
# 8x spatial compression), so 512 pixels means a 64x64 latent.
DEF_WIDTH = 512
DEF_HEIGHT = 512

GROUP_SIZE = 256
_H = {}

comfy_utils = None
comfy_md = None
comfy_sd = None
comfy_sample = None


def build_hadamard(size, dtype=torch.float32):
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError("Hadamard size must be a power of 4, got %d" % size)
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def group_size_for(n, preferred=GROUP_SIZE):
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def quant_dequant(w, gs):
    """ConvRot INT8 reconstruction in the ORIGINAL frame: (q*s) @ H."""
    if gs not in _H:
        _H[gs] = build_hadamard(gs)
    hh = _H[gs].to(device=w.device, dtype=w.dtype)
    if w.dim() == 2:
        of, inf = w.shape
        g = inf // gs
        rot = torch.matmul(w.view(of, g, gs), hh.T).reshape(w.shape)
        amax = rot.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
        s = (amax / 127.0).to(w.dtype)
        q = (rot / s).round().clamp(-127, 127)
        return torch.matmul((q * s).view(of, g, gs), hh).reshape(w.shape)
    if w.dim() == 4:
        oc, ic, kh, kw = w.shape
        flat = w.permute(0, 2, 3, 1).contiguous().view(-1, ic)
        of, inf = flat.shape
        g = inf // gs
        rot = torch.matmul(flat.view(of, g, gs), hh.T).reshape(flat.shape)
        amax = rot.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
        s = (amax / 127.0).to(w.dtype)
        q = (rot / s).round().clamp(-127, 127)
        back = torch.matmul((q * s).view(of, g, gs), hh).reshape(flat.shape)
        return back.view(oc, kh, kw, ic).permute(0, 3, 1, 2).contiguous()
    raise ValueError("unsupported rank %d" % w.dim())


def setup_comfy(comfy_path):
    """Put the given ComfyUI root first on sys.path and import what we need."""
    global comfy_utils, comfy_md, comfy_sd, comfy_sample
    comfy_path = os.path.abspath(comfy_path)
    if not os.path.isdir(os.path.join(comfy_path, "comfy")):
        raise SystemExit("not a ComfyUI root (no comfy/ inside): %s" % comfy_path)
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)
    cwd = os.getcwd()
    os.chdir(comfy_path)
    try:
        import comfy.utils as _u
        import comfy.model_detection as _md
        import comfy.sd as _sd
        import comfy.sample as _sample
    finally:
        os.chdir(cwd)
    comfy_utils, comfy_md, comfy_sd, comfy_sample = _u, _md, _sd, _sample


def _clear_argv_for_comfy():
    """comfy's loader parses sys.argv; keep our own flags away from it."""
    saved = sys.argv
    sys.argv = [saved[0]]
    return saved


def _restore_argv(saved):
    sys.argv = saved


def load_sd15(path):
    """Load a checkpoint ComfyUI-natively; returns (ModelPatcher, clip).

    The baseline contains no INT8 tensors, so the stock ops are used.
    """
    saved = _clear_argv_for_comfy()
    try:
        out = comfy_sd.load_checkpoint_guess_config(
            os.path.abspath(path),
            output_vae=False,   # the VAE is never used by the trajectory measurement
            output_clip=True,
            embedding_directory=None,
        )
    finally:
        _restore_argv(saved)
    model, clip = out[0], out[1]
    if model is None or clip is None:
        raise RuntimeError("checkpoint load failed (model/clip None): %s" % path)
    return model, clip


def build_conditioning(clip, prompt, negative):
    """Build the CLIP conditioning exactly as the benchmark does."""
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    negative_cond = clip.encode_from_tokens_scheduled(clip.tokenize(negative))
    return positive, negative_cond


def make_latent(model, width, height):
    import comfy.model_management as mm

    latent = torch.zeros(
        [1, 4, height // 8, width // 8],
        device=mm.intermediate_device(),
        dtype=mm.intermediate_dtype(),
    )
    latent = comfy_sample.fix_empty_latent_channels(model, latent, 8, None)
    return latent


def run_trajectory(patcher, positive, negative, latent0, *, seed, steps, cfg,
                   sampler, scheduler):
    """Run the production sampler and capture the per-step latent x.

    Identical engine, sampler, scheduler, cfg and seed as
    `benchmark/sd1.5_int8_traj_compare.py`.
    """
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
            callback=cb, disable_pbar=False, seed=seed,
        )
    return xs


def rel_mse(a, b):
    den = (a.float() ** 2).sum().item()
    if den == 0:
        return float((a.float() - b.float()).abs().max().item())
    return float(((a.float() - b.float()) ** 2).sum().item() / den)


def cos_dist(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return float("nan")
    return float(1.0 - (a @ b) / (na * nb))


def derive_out_path(ckpt_path, prefix):
    """Build the output path from the model name, so `--out` is optional.

    `impact/<prefix>_<model stem>.json` - the stem carries the model's own name,
    so a JSON can always be traced back to the checkpoint it came from, and two
    checkpoints can never overwrite each other's ranking.
    """
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stem = os.path.basename(ckpt_path)
    for ext in (".safetensors", ".ckpt", ".pt", ".bin"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    stem = stem.replace(" ", "_")
    out_dir = os.path.join(repo, "impact")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "%s_%s.json" % (prefix, stem))


def main():
    ap = argparse.ArgumentParser(description="SD1.5 per-layer trajectory impact (ConvRot INT8)")
    ap.add_argument("--comfy", default=os.environ.get("COMFY_PATH", ""),
                    help="ComfyUI root (the directory containing comfy/); "
                         "falls back to the COMFY_PATH environment variable")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="",
                    help="output JSON (default: impact/impact_sd15_traj_<model stem>.json)")
    ap.add_argument("--steps", type=int, default=25,
                    help="denoising steps per trajectory (default 25 - SD1.5's "
                         "production step count, the same one the gate and the "
                         "calibration use)")
    ap.add_argument("--seeds", default="42,137",
                    help="trajectory seeds, comma separated (default 42,137)")
    ap.add_argument("--n-layers", type=int, default=0,
                    help="measure only the first N layers (0 = all; smoke test aid)")
    ap.add_argument("--metric", default="relmse", choices=["relmse", "cos"],
                    help="relmse (as diag_impact.py) or cos")
    ap.add_argument("--width", type=int, default=DEF_WIDTH,
                    help="image width in pixels (default 512; the latent is width/8)")
    ap.add_argument("--height", type=int, default=DEF_HEIGHT,
                    help="image height in pixels (default 512; the latent is height/8)")
    ap.add_argument("--cfg", type=float, default=7.0,
                    help="classifier-free guidance scale (default 7.0, as the benchmark)")
    ap.add_argument("--sampler", default="dpmpp_2m",
                    help="sampler name (default dpmpp_2m, as the benchmark)")
    ap.add_argument("--scheduler", default="karras",
                    help="scheduler name (default karras, as the benchmark)")
    ap.add_argument("--prompt", default="masterpiece, best quality, 1girl, solo, standing, simple background")
    ap.add_argument("--negative", default="")
    a = ap.parse_args()
    if not a.comfy:
        ap.error("the ComfyUI root is not set: pass --comfy <path> or set COMFY_PATH")
    setup_comfy(a.comfy)

    if not a.out:
        a.out = derive_out_path(a.ckpt, "impact_sd15_traj")
        print("the output (from the model name): %s" % a.out, flush=True)

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    print("[1] the checkpoint ...", flush=True)
    patcher, clip = load_sd15(a.ckpt)
    print("    the patcher: %s" % type(patcher).__name__, flush=True)
    print("    the model: %s" % type(patcher.model).__name__, flush=True)

    positive, negative = build_conditioning(clip, a.prompt, a.negative)
    latent0 = make_latent(patcher, a.width, a.height)
    print("    the latent: %s (from %dx%d px, %s)" % (
        tuple(latent0.shape), a.width, a.height, latent0.dtype), flush=True)
    print("    the sampler/scheduler/cfg: %s / %s / %s" % (a.sampler, a.scheduler, a.cfg),
          flush=True)

    # ---- the pristine reference trajectories ------------------------------
    print("[2] the pristine trajectories (seeds=%s, steps=%d) ..." % (seeds, a.steps),
          flush=True)
    t0 = time.time()
    ref = {}
    for sd in seeds:
        xs = run_trajectory(patcher, positive, negative, latent0, seed=sd,
                            steps=a.steps, cfg=a.cfg, sampler=a.sampler,
                            scheduler=a.scheduler)
        ref[sd] = xs[-1]
        print("    seed %d: %d steps captured, final %s finite=%s" % (
            sd, len(xs), tuple(xs[-1].shape), bool(torch.isfinite(xs[-1]).all())),
            flush=True)
    print("    (%.1f s)" % (time.time() - t0), flush=True)

    # ---- the candidate modules -------------------------------------------
    diffusion = getattr(patcher.model, "diffusion_model", None)
    if diffusion is None:
        raise RuntimeError("the patcher's model exposes no diffusion_model")
    mods = []
    for name, m in diffusion.named_modules():
        w = getattr(m, "weight", None)
        if w is None or w.ndim not in (2, 4):
            continue
        mods.append((name, w))
    print("[3] the candidate layers: %d" % len(mods), flush=True)

    if a.n_layers:
        mods = mods[: a.n_layers]
        print("    (limited to the first %d for the smoke test)" % len(mods), flush=True)

    # ---- the sweep -------------------------------------------------------
    impacts = {}
    print("[4] the sweep ...", flush=True)
    t0 = time.time()
    for i, (name, w) in enumerate(mods):
        inf = int(w.shape[1])
        gs = group_size_for(inf)
        if gs is None:
            impacts[name] = None
            continue
        orig = w.data.detach().clone()
        try:
            rec = quant_dequant(orig, gs).to(orig.dtype)
            w.data.copy_(rec)
            vals = []
            for sd in seeds:
                xs = run_trajectory(patcher, positive, negative, latent0, seed=sd,
                                    steps=a.steps, cfg=a.cfg, sampler=a.sampler,
                                    scheduler=a.scheduler)
                if a.metric == "relmse":
                    vals.append(rel_mse(ref[sd], xs[-1]))
                else:
                    vals.append(cos_dist(ref[sd], xs[-1]))
            impacts[name] = vals
        finally:
            w.data.copy_(orig)
        if (i + 1) % 10 == 0 or (i + 1) == len(mods):
            el = time.time() - t0
            print("    [%d/%d] %.0fs (%.2fs/layer)" % (i + 1, len(mods), el, el / (i + 1)),
                  flush=True)

    # ---- the aggregate ---------------------------------------------------
    agg = {}
    for name, vals in impacts.items():
        if vals is None:
            continue
        agg[name] = sum(vals) / len(vals)
    ranked = sorted(agg.items(), key=lambda kv: -kv[1])

    out = {
        "impacts": agg,
        "per_seed": impacts,
        "steps": a.steps,
        "seeds": seeds,
        "metric": a.metric,
        "width": a.width,
        "height": a.height,
        "cfg": a.cfg,
        "sampler": a.sampler,
        "scheduler": a.scheduler,
        "ckpt": os.path.basename(a.ckpt),
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=True, ensure_ascii=False)
    print("[5] saved: %s" % a.out, flush=True)
    print("    the measured layers: %d" % len(agg), flush=True)
    print("    the top 10:", flush=True)
    for n, v in ranked[:10]:
        print("      %-52s %.6e" % (n, v), flush=True)


if __name__ == "__main__":
    main()

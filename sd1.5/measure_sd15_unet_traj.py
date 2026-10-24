# -*- coding: utf-8 -*-
"""SD1.5 UNet: per-layer **trajectory** impact, measured on the model's own
denoising output.

This is a NEW script, not a variant of `measure_sd15_unet_contrib.py`. The two
answer the same question with different rigour, and the difference matters:

  measure_sd15_unet_contrib.py   one forward at t=500           (a single point)
  measure_sd15_unet_traj.py      N-step denoising trajectory    (propagation)

A single forward measures a layer's sensitivity where the sampler happens to
call it once. A trajectory measures what the sampler actually experiences: the
error enters at one step, is carried in the latent, and is amplified or cancelled
by every later step. `Z_Image/diag_impact.py` uses the trajectory form for
exactly this reason ("inject ONE layer's error, run N steps, measure x
divergence").

The rest of the principle is identical to `diag_impact.py` and to
`measure_sd15_unet_contrib.py`:

  * hold every other layer at its pristine weight;
  * replace ONE layer's weight by its ConvRot INT8 reconstruction **in the
    original frame** (`dequant(INT8(W @ H^T))` = `(q*s) @ H`);
  * run the same fixed trajectory;
  * record the divergence from the pristine trajectory.

Everything here is SD1.5-specific (the UNet call signature, the 4-channel 64x64
latent, the 77x768 context), so it is a separate script rather than a
generalisation.

Usage:
  python sd1.5/measure_sd15_unet_traj.py \
      --comfy <ComfyUI root> \
      --ckpt  <source checkpoint>.safetensors \
      --out   impact/impact_sd15_traj.json \
      --steps 25 --seeds 42,137 --n-layers 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

# --- the ComfyUI root ------------------------------------------------------
def _resolve_comfy(argv):
    if "--comfy" in argv:
        i = argv.index("--comfy")
        if i + 1 < len(argv):
            path = argv[i + 1]
            del argv[i:i + 2]
            return path
    path = os.environ.get("COMFY_PATH", "")
    if not path:
        sys.exit("error: the ComfyUI root is not set.\n"
                 "  pass --comfy <path> or set COMFY_PATH=<path>\n"
                 "  (the directory must contain comfy/ops.py)")
    return path

# NOTE: the ComfyUI root is resolved inside main(), AFTER argparse parses, so
# `--help` prints and a missing value is reported by argparse itself.
COMFY = ""

import torch  # noqa: E402  (plain dependency - safe before the ComfyUI root is known)


# the comfy.* handles, imported once _bind_comfy has put the root on sys.path
comfy_utils = None
comfy_ops = None
comfy_md = None


def _bind_comfy(path):
    """Put the ComfyUI root on sys.path, chdir into it, and import comfy.*.

    `comfy.*` cannot be imported before the root is on sys.path, and `--help`
    must work without a root - so the import happens here, not at module level.
    """
    global COMFY, comfy_utils, comfy_ops, comfy_md
    path = os.path.abspath(path)
    if not os.path.isdir(os.path.join(path, "comfy")):
        sys.exit("error: %s does not look like a ComfyUI root (no comfy/)" % path)
    COMFY = path
    if COMFY not in sys.path:
        sys.path.insert(0, COMFY)
    os.chdir(COMFY)
    import comfy.utils as _u
    import comfy.ops as _o
    import comfy.model_detection as _m
    comfy_utils, comfy_ops, comfy_md = _u, _o, _m

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float16
GROUP_SIZE = 256
LATENT = 64
CTX_LEN = 77
CTX_DIM = 768


def build_hadamard(size, dtype=torch.float32):
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError("Hadamard size must be a power of 4, got %s" % size)
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return h / (size ** 0.5)


_H = {}


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


def build_unet(ckpt_path, ops):
    sd = comfy_utils.load_torch_file(ckpt_path, safe_load=True)
    prefix = comfy_md.unet_prefix_from_state_dict(sd)
    # detect BEFORE stripping the prefix: detect_layer_quantization matches
    # k.startswith(unet_key_prefix)
    mcfg = comfy_md.model_config_from_unet(sd, prefix)
    usd = comfy_utils.state_dict_prefix_replace(sd, {prefix: ""}, filter_keys=True)
    mcfg.set_inference_dtype(DT, None, device=DEV)
    if ops is not None:
        mcfg.custom_operations = ops
    model = mcfg.get_model(usd, "")
    model.load_model_weights(usd, "", assign=False)
    return model.diffusion_model.to(DEV).eval()


def make_fixed_inputs(seed):
    """The fixed trajectory inputs: one latent, one timestep schedule, one context.

    Identical for the pristine run and every perturbed run, so the only variable
    is the layer under test.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    x0 = torch.randn((1, 4, LATENT, LATENT), generator=g).to(DEV, dtype=DT)
    ctx = torch.randn((1, CTX_LEN, CTX_DIM), generator=g).to(DEV, dtype=DT)
    return x0, ctx


def run_trajectory(model, x0, ctx, steps):
    """The denoising loop `sd1.5_int8_traj_compare.py` uses, in the same form.

    SD1.5's UNet signature is (x, timestep, context); the loop is the explicit
    Euler step the benchmark uses so the impact is measured through the same
    arithmetic the gate will use.
    """
    x = x0.clone()
    sig = torch.linspace(1.0, 0.0, steps + 1, device=DEV, dtype=DT)
    with torch.no_grad():
        for i in range(steps):
            t = (sig[i] * 999.0).reshape(1)
            eps = model(x, t, ctx)
            x = (x - eps * (sig[i] - sig[i + 1])).to(DT)
    return x


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
                         "defaults to $COMFY_PATH")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="",
                    help="output JSON; default: "
                         "impact/<prefix>_<model stem>.json derived from --ckpt")
    ap.add_argument("--steps", type=int, default=25,
                    help="denoising steps per trajectory (default 25 - SD1.5's "
                         "production step count, the same one the gate and the "
                         "calibration use; 12 is the Z-Image/NVFP4 default and is "
                         "too short here)")
    ap.add_argument("--seeds", default="42,137",
                    help="trajectory seeds, comma separated (default 42,137)")
    ap.add_argument("--n-layers", type=int, default=0,
                    help="measure only the first N layers (0 = all; smoke test aid)")
    ap.add_argument("--metric", default="relmse", choices=["relmse", "cos"],
                    help="relmse (as diag_impact.py) or cos")
    a = ap.parse_args()
    if not a.comfy:
        ap.error("the ComfyUI root is not set: pass --comfy <path> or set COMFY_PATH")
    _bind_comfy(a.comfy)
    if not a.out:
        a.out = derive_out_path(a.ckpt, "impact_sd15_traj")
        print("the output (from the model name): %s" % a.out, flush=True)

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    print("[1] the UNet ...", flush=True)
    model = build_unet(a.ckpt, None)
    print("    the model: %s" % type(model).__name__, flush=True)

    # the candidate layers: every ConvRot-eligible Linear/Conv2d
    mods = dict(model.named_modules())
    cands = []
    for n, m in model.named_modules():
        w = getattr(m, "weight", None)
        if w is None or not hasattr(w, "ndim"):
            continue
        if type(m).__name__ in ("Linear", "Conv2d") and w.ndim in (2, 4):
            if group_size_for(int(w.shape[1])) is None:
                continue
            cands.append(n)
    if a.n_layers > 0:
        cands = cands[: a.n_layers]
    print("[2] the candidates: %d | the steps: %d | the seeds: %s" % (
        len(cands), a.steps, seeds), flush=True)

    # the pristine trajectories
    print("[3] the pristine trajectories ...", flush=True)
    ref = {}
    for s in seeds:
        x0, ctx = make_fixed_inputs(s)
        ref[s] = run_trajectory(model, x0, ctx, a.steps).detach().float().double().reshape(-1).cpu()
    print("    done: %d seeds" % len(ref), flush=True)

    impacts = {}
    for idx, name in enumerate(cands):
        m = mods[name]
        w0 = m.weight.data.clone()
        gs = group_size_for(int(w0.shape[1]))
        try:
            m.weight.data = quant_dequant(w0.float(), gs).to(w0.dtype)
            vals = []
            for s in seeds:
                x0, ctx = make_fixed_inputs(s)
                xt = run_trajectory(model, x0, ctx, a.steps)
                xt = xt.detach().float().double().reshape(-1).cpu()
                if a.metric == "relmse":
                    r = ref[s]
                    vals.append(float(((xt - r) ** 2).sum() / (r ** 2).sum()))
                else:
                    vals.append(1.0 - torch.nn.functional.cosine_similarity(
                        ref[s].reshape(1, -1), xt.reshape(1, -1), dim=1).item())
            impacts[name] = statistics.mean(vals)
        finally:
            m.weight.data = w0
        if (idx + 1) % 10 == 0:
            print("   [%d/%d] %s  %.4e" % (idx + 1, len(cands), name, impacts[name]), flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    json.dump({"impacts": impacts, "steps": a.steps, "seeds": seeds, "metric": a.metric},
              open(a.out, "w", encoding="utf-8"), indent=1, sort_keys=True)
    print()
    print("saved: %s" % a.out)
    print("   the measured: %d" % len(impacts))
    vals = sorted(impacts.items(), key=lambda kv: -kv[1])
    for n, v in vals[:20]:
        print("   %.4e  %s" % (v, n))
    if vals:
        print("   ... the smallest: %.4e  %s" % (vals[-1][1], vals[-1][0]))


if __name__ == "__main__":
    main()

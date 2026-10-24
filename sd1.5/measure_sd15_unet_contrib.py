# -*- coding: utf-8 -*-
"""SD1.5 UNet: the aggregate contribution of each layer, measured on the UNet's
OWN output.

The criterion (the current method, see
`md/How to quantize Text Encoder and ControlNet.md` - Rules 1-8):
replace ONE layer by its ConvRot INT8 reconstruction in the ORIGINAL frame
(`dequant(INT8(W @ H^T))` = `(q*s) @ H`), hold every other layer, and rank by
`1 - cos` of the UNet's own output.

Why this replaces the V1.0 selector: V1.0 ranked the FP16 protection set from
STATIC distribution statistics (kurtosis / outlier-ratio / absmax) plus a
weight-space histogram MSE. That is a weight-space proxy, and on SD1.5 it does
not separate the layers the sampler is actually sensitive to - the measured
result was an FP16 protection set that scored the same as native (no protection
at all), i.e. the protection bought nothing. This script measures the output
change instead.

The observable is the UNet's own output - `model(x, timesteps, context)` - over
a set of fixed inputs (seed 42). A base-model load path is not needed: an SD1.5
UNet is self-contained (unlike the Qwen-Image Fun ControlNet).

CLIP is NOT quantized here (the caller's instruction): only the UNet's tensors
are candidates, and the checkpoint's CLIP/VAE pass through untouched.

Usage:
  python sd1.5/measure_sd15_unet_contrib.py \
      --ckpt <checkpoint>.safetensors \
      --out impact/contrib_sd15_unet.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

# --- the ComfyUI root ------------------------------------------------------
# This script must run INSIDE a ComfyUI checkout (it imports comfy.ops,
# comfy.model_detection, ...). COMFY_PATH may be set in the environment, or the
# first argument may be `--comfy <path>`; there is no default, so a wrong
# machine cannot silently pick up someone else's tree.
def _resolve_comfy(argv):
    import os, sys
    path = ""
    if "--comfy" in argv:
        i = argv.index("--comfy")
        if i + 1 < len(argv):
            path = argv[i + 1]
            del argv[i:i + 2]
    if not path:
        path = os.environ.get("COMFY_PATH", "")
    if not path:
        sys.exit(
            "error: the ComfyUI root is not set.\n"
            "  pass --comfy <path> or set COMFY_PATH=<path>\n"
            "  (the directory must contain comfy/ops.py)")
    path = os.path.abspath(path)
    if not os.path.isdir(os.path.join(path, "comfy")):
        sys.exit("error: %s does not look like a ComfyUI root (no comfy/)" % path)
    return path

# ---------------------------------------------------------------------------
# NOTE: the ComfyUI root is resolved inside main(), AFTER argparse parses - so
# `--help` prints and a missing value is reported by argparse itself.
COMFY = ""

import torch  # noqa: E402

import comfy.utils  # noqa: E402
import comfy.ops as O  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.model_detection as md  # noqa: E402
import comfy.sd  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float16          # SD1.5 runs FP16
GROUP_SIZE = 256
LATENT = 64                 # 512x512 / 8


def hadamard(size, dtype=torch.float32):
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError("Hadamard size must be a power of 4, got %s" % size)
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return h / (size ** 0.5)


_H = {}


def quant_dequant_linear(w, gs=GROUP_SIZE):
    """ConvRot INT8 reconstruction in the ORIGINAL frame: (q*s) @ H."""
    if gs not in _H:
        _H[gs] = hadamard(gs)
    H = _H[gs].to(device=w.device, dtype=w.dtype)
    of, inf = w.shape
    g = inf // gs
    r = torch.matmul(w.view(of, g, gs), H.T).reshape(w.shape)
    a = r.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
    s = a / 127.0
    q = (r / s.to(w.dtype)).round().clamp(-127, 127)
    return torch.matmul((q * s.to(w.dtype)).view(of, g, gs), H).reshape(w.shape)


def quant_dequant_conv(w, gs=GROUP_SIZE):
    """The same, for a 4D Conv2d weight: rotate along the input-channel axis."""
    if gs not in _H:
        _H[gs] = hadamard(gs)
    H = _H[gs].to(device=w.device, dtype=w.dtype)
    oc, ic, kh, kw = w.shape
    flat = w.permute(0, 2, 3, 1).contiguous().view(-1, ic)
    of, inf = flat.shape
    g = inf // gs
    r = torch.matmul(flat.view(of, g, gs), H.T).reshape(flat.shape)
    a = r.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
    s = a / 127.0
    q = (r / s.to(w.dtype)).round().clamp(-127, 127)
    back = torch.matmul((q * s.to(w.dtype)).view(of, g, gs), H).reshape(flat.shape)
    return back.view(oc, kh, kw, ic).permute(0, 3, 1, 2).contiguous()


def group_size_for(n, preferred=GROUP_SIZE):
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def build_unet(ckpt_path, ops):
    """Build the UNet exactly as `comfy.sd.load_diffusion_model_state_dict` does.

    `BASE.get_model(new_sd, "")` - positional (sd, prefix), then
    `load_model_weights(sd, prefix)`. Getting this wrong silently yields an
    empty model, so the arg order matters.
    """
    sd = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    prefix = md.unet_prefix_from_state_dict(sd)
    # model_config_from_unet MUST see the comfy_quant keys while they still carry
    # the checkpoint prefix: comfy.utils.detect_layer_quantization() matches on
    # `k.startswith(unet_key_prefix)`. Detect FIRST, then strip the prefix.
    mcfg = md.model_config_from_unet(sd, prefix)
    if mcfg.quant_config is not None:
        print("    the mixed-precision quant detected: %s" % (mcfg.quant_config,), flush=True)
    unet_sd = comfy.utils.state_dict_prefix_replace(sd, {prefix: ""}, filter_keys=True)
    if mcfg is None:
        raise RuntimeError("the SD1.5 UNet config was not detected")
    mcfg.set_inference_dtype(DT, None, device=DEV)
    if ops is not None:
        mcfg.custom_operations = ops
    model = mcfg.get_model(unet_sd, "")
    model.load_model_weights(unet_sd, "", assign=False)
    model = model.to(DEV).eval()
    # `BaseModel` wraps the UNet in `.diffusion_model`; that is the module the
    # sampler calls and the one whose layers we perturb.
    model = model.diffusion_model
    model = model.to(DEV).eval()
    return model, unet_sd


def make_inputs(seed, latent=LATENT, ctx=77, dim=768):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((1, 4, latent, latent), generator=g).to(DEV, dtype=DT)
    ts = torch.tensor([500.0], device=DEV, dtype=DT)
    context = torch.randn((1, ctx, dim), generator=g).to(DEV, dtype=DT)
    return x, ts, context


def unet_output(model, x, ts, context):
    with torch.no_grad():
        out = model(x, ts, context)
    return out.detach().float().double().reshape(-1).cpu()


def _bind_comfy(path):
    """Put the ComfyUI root on sys.path and chdir into it (comfy.* needs both)."""
    global COMFY
    path = os.path.abspath(path)
    if not os.path.isdir(os.path.join(path, "comfy")):
        sys.exit("error: %s does not look like a ComfyUI root (no comfy/)" % path)
    COMFY = path
    if COMFY not in sys.path:
        sys.path.insert(0, COMFY)
    os.chdir(COMFY)


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default=os.environ.get("COMFY_PATH", ""),
                    help="ComfyUI root (the directory containing comfy/); "
                         "defaults to $COMFY_PATH")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="",
                    help="output JSON; default: "
                         "impact/<prefix>_<model stem>.json derived from --ckpt")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if not a.comfy:
        ap.error("the ComfyUI root is not set: pass --comfy <path> or set COMFY_PATH")
    _bind_comfy(a.comfy)
    if not a.out:
        a.out = derive_out_path(a.ckpt, "contrib_sd15_unet")
        print("the output (from the model name): %s" % a.out, flush=True)

    ops = O.manual_cast
    print("[1] the UNet ...", flush=True)
    model, _sd = build_unet(a.ckpt, ops)
    print("    the model: %s" % type(model).__name__, flush=True)

    batches = [make_inputs(a.seed + i) for i in range(a.n)]
    ref = [unet_output(model, *b) for b in batches]
    fin = bool(torch.isfinite(ref[0]).all())
    print("[1] the baseline: %s finite=%s" % (tuple(ref[0].shape), fin), flush=True)

    mods = dict(model.named_modules())
    cands = []
    for n, m in model.named_modules():
        w = getattr(m, "weight", None)
        if w is None or not hasattr(w, "ndim"):
            continue
        if type(m).__name__ in ("Linear", "Conv2d") and w.ndim in (2, 4):
            cands.append(n)
    print("[2] the candidate Linear/Conv2d: %d" % len(cands), flush=True)

    contrib = {}
    for i, name in enumerate(cands):
        m = mods[name]
        orig = m.weight.data.clone()
        w = orig.float()
        if w.ndim == 2:
            gs = group_size_for(int(w.shape[1]))
            q = quant_dequant_linear(w, gs) if (gs and int(w.shape[0]) % 4 == 0) else None
        else:
            gs = group_size_for(int(w.shape[1]))
            q = quant_dequant_conv(w, gs) if (gs and int(w.shape[0]) % 4 == 0) else None
        if q is None:
            contrib[name] = None
            continue
        m.weight.data = q.to(orig.dtype)
        try:
            cs = []
            for j, b in enumerate(batches):
                f = unet_output(model, *b)
                cs.append(torch.nn.functional.cosine_similarity(
                    ref[j].reshape(1, -1), f.reshape(1, -1), dim=1).item())
            contrib[name] = 1.0 - statistics.mean(cs)
        finally:
            m.weight.data = orig
        if (i + 1) % 20 == 0:
            print("   [%d/%d] %s" % (i + 1, len(cands), name), flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    json.dump({"impacts": contrib, "n": a.n, "seed": a.seed},
              open(a.out, "w", encoding="utf-8"), indent=1, sort_keys=True)
    print()
    print("saved: %s" % a.out)
    print("   the measured: %d | the skipped: %d" % (
        sum(1 for v in contrib.values() if v is not None),
        sum(1 for v in contrib.values() if v is None)))
    vals = sorted(((k, v) for k, v in contrib.items() if v is not None), key=lambda kv: -kv[1])
    for n, val in vals[:20]:
        print("   %.4e  %s" % (val, n))
    if vals:
        print("   ... the smallest: %.4e  %s" % (vals[-1][1], vals[-1][0]))


if __name__ == "__main__":
    main()

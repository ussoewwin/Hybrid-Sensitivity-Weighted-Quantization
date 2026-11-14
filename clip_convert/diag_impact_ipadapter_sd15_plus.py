"""SD1.5 IP-Adapter Plus (`ip-adapter_sd15_plus.pth`) per-layer ConvRot INT8 impact.

This instrument exists for this one checkpoint. It is not the ControlNet instrument and not
the generic IP-Adapter one: this file's adapter is a Resampler whose 16 query tokens are
frozen into the conditioning at apply time, and 32 bias-free projections that run on every
forward. Those two groups need two different measurements, and both have to come out of the
same process so the two halves are measured against the same pristine reference.

  1. build the adapter from this file's own state dict: the Resampler
     (dim 768, depth 4, dim_head 64, heads 12, num_queries 16, embedding_dim 1280,
     output_dim 768, ff_mult 4, max_seq_len 257) and the 32 to_k_ip / to_v_ip Linears,
  2. patch it into the SD1.5 UNet the node patches, with a real clip-vision embedding of a
     real image, exactly as `ipadapter_execute` does,
  3. record two pristine references in that one process: the UNet trajectory (what the 32
     projections feed) and the Resampler's 16-token output (what the 22 image_proj Linears
     feed),
  4. replace one Linear's weight with its ConvRot INT8 reconstruction, run the reference that
     layer feeds, record the relative MSE, restore, next layer.

Which reference a layer is read against is decided by where the layer sits, not by a flag:

  image_proj.*   -> the Resampler output, because this layer's only consumer is the embeds
                    that `ipadapter_execute` freezes before sampling starts. A trajectory run
                    cannot see this layer: the embeds are computed once, up front.
  ip_adapter.*   -> the UNet trajectory, because these projections run inside every
                    cross-attention forward, once per step.

The quantisation replay is the converter's own: Hadamard by Kronecker doubling, orthonormal,
group size a power of four, row-wise `amax/127` int8.

  python clip_convert/diag_impact_ipadapter_sd15_plus.py \
      --base <unet.safetensors> --ipadapter <ip-adapter_sd15_plus.pth> \
      --clipvision <clip_vision_h.safetensors> --image <png> \
      --out impact/impact_ipadapter_sd15_plus.json \
      --comfy_path <ComfyUI> --ipadapter_dir <custom_nodes/comfyui_ipadapter_plus> \
      --steps 25 --seed 42
"""

import argparse
import importlib
import json
import math
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------- quantisation replay

def build_hadamard(size, device="cpu", dtype=torch.float32):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
                      dtype=dtype, device=device)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def convrot_group_size(n, preferred=256):
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def convrot_int8_recon(w, gs):
    h = build_hadamard(gs, device=w.device)
    of, inf = w.shape
    g = inf // gs
    x = torch.matmul(w.float().view(of, g, gs), h.T.to(torch.float32)).reshape(of, inf)
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    s = amax / 127.0
    q = (x / s).round().clamp(-127, 127)
    deq = (q * s).view(of, g, gs)
    return torch.matmul(deq, h).reshape(of, inf).to(w.dtype)


def rel_mse(a, b):
    num = torch.sum((a - b) ** 2)
    den = torch.sum(b ** 2).clamp(min=1e-30)
    return float((num / den).item())


# ---------------------------------------------------------------- the node's own entry points

def setup_comfy(comfy_path):
    comfy_path = os.path.abspath(comfy_path)
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)
    import folder_paths
    return comfy_path


def clear_argv():
    saved = sys.argv
    sys.argv = [saved[0]]
    return saved


def restore_argv(saved):
    sys.argv = saved


def find_adapter(patched):
    """The IPAdapter instance the node attached to the UNet.

    The attn2 patch kwargs can carry either the adapter itself or its To_KV holder,
    depending on which node path built the patch, so the class is what is matched here.
    """
    seen = []
    to = patched.model_options.get("transformer_options", {})
    repl = to.get("patches_replace", {}).get("attn2", {})
    for _, v in repl.items():
        for kw in getattr(v, "kwargs", []):
            a = kw.get("ipadapter") if hasattr(kw, "get") else None
            if a is None:
                continue
            if hasattr(a, "image_proj_model"):
                return a
            seen.append(type(a).__name__)
    for m in patched.model.modules():
        if type(m).__name__ == "IPAdapter" and hasattr(m, "image_proj_model"):
            return m
    raise SystemExit("IPAdapter instance not found; attn2 kwargs carried: %s" % (seen or "nothing"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="the SD1.5 UNet the adapter is patched into")
    ap.add_argument("--ipadapter", required=True, help="ip-adapter_sd15_plus.pth")
    ap.add_argument("--clipvision", required=True)
    ap.add_argument("--image", required=True, help="the image prompt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--comfy_path", required=True)
    ap.add_argument("--ipadapter_dir", required=True)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--groupsize", type=int, default=256)
    args = ap.parse_args()

    setup_comfy(args.comfy_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    saved = clear_argv()
    import comfy.sd
    import comfy.sample as comfy_sample
    import comfy.model_management as mm
    import comfy.clip_vision
    restore_argv(saved)

    parent = os.path.dirname(os.path.abspath(args.ipadapter_dir))
    if parent not in sys.path:
        sys.path.insert(0, parent)
    pkg = os.path.basename(os.path.abspath(args.ipadapter_dir))
    ipadapter_execute = importlib.import_module(pkg + ".IPAdapterPlus").ipadapter_execute
    ipadapter_model_loader = importlib.import_module(pkg + ".utils").ipadapter_model_loader

    print("[load] base: %s" % args.base, flush=True)
    saved = clear_argv()
    patcher = comfy.sd.load_diffusion_model(os.path.abspath(args.base), model_options={})
    restore_argv(saved)
    net = patcher.model
    net.to(device).eval()

    print("[load] ipadapter: %s" % args.ipadapter, flush=True)
    ipa = ipadapter_model_loader(os.path.abspath(args.ipadapter))
    print("[load] clip vision: %s" % args.clipvision, flush=True)
    cv = comfy.clip_vision.load(os.path.abspath(args.clipvision))

    from PIL import Image
    img = Image.open(args.image).convert("RGB")
    t = torch.from_numpy(np.asarray(img).astype("float32") / 255.0).unsqueeze(0)
    image_in = t[:, :, :, :3]

    print("[patch] applying the adapter the way the node does", flush=True)
    saved = clear_argv()
    patched = ipadapter_execute(patcher, ipa, cv, image=image_in, weight=1.0, weight_type="linear",
                                start_at=0.0, end_at=1.0, embeds_scaling="V only")
    restore_argv(saved)
    if isinstance(patched, tuple):
        patched = patched[0]
    adapter = find_adapter(patched)
    if adapter is None:
        raise SystemExit("adapter instance not found on the patched model")
    print("[patch] adapter: %s | ip_layers %d"
          % (type(adapter).__name__, len(adapter.ip_layers.to_kvs)), flush=True)

    # this file is a bare UNet and no CLIP-L weights are present, so both sides of every
    # comparison use the same fixed zero context; what moves is only the adapter's edit.
    ctx = 768
    prompt_pos = [[torch.zeros(1, 77, ctx, device=device), {}]]
    prompt_neg = [[torch.zeros(1, 77, ctx, device=device), {}]]
    print("[cond] fixed zero context %s" % (tuple(prompt_pos[0][0].shape),), flush=True)

    latent0 = torch.zeros([1, 4, args.height // 8, args.width // 8],
                          device=mm.intermediate_device(), dtype=mm.intermediate_dtype())
    latent0 = comfy_sample.fix_empty_latent_channels(patched, latent0, 8, None)
    if latent0.device.type != device:
        latent0 = latent0.to(device)

    def run_traj():
        noise = comfy_sample.prepare_noise(latent0, args.seed, None)
        xs = []

        def cb(step, x0, x, total):
            xs.append(x.detach().to("cpu", torch.float32, copy=True))

        with torch.no_grad():
            comfy_sample.sample(patched, noise, args.steps, args.cfg, "dpmpp_2m", "karras",
                                prompt_pos, prompt_neg, latent0, denoise=1.0, disable_noise=False,
                                start_step=None, last_step=None, force_full_denoise=False,
                                noise_mask=None, callback=cb, disable_pbar=True, seed=args.seed)
        return xs

    # the Resampler side: this is what `ipadapter_execute` froze into the conditioning. Its input
    # is the clip-vision token stack of the same image prompt, so it is built here from the same
    # CLIPVision the node used, and read on its own output rather than through the trajectory.
    resampler = adapter.image_proj_model
    with torch.no_grad():
        clip_out = cv.encode_image(image_in.to(device))
    if isinstance(clip_out, dict):
        cond_emb = clip_out.get("last_hidden_state", list(clip_out.values())[0])
    else:
        cond_emb = getattr(clip_out, "last_hidden_state", None)
    if cond_emb is None:
        cond_emb = clip_out
    if not torch.is_tensor(cond_emb):
        raise SystemExit("clip vision returned no token tensor: %r" % (type(clip_out),))
    res_dtype = next(resampler.parameters()).dtype
    cond_emb = cond_emb.to(device, res_dtype)
    print("[embeds] resampler dtype %s" % res_dtype, flush=True)
    print("[embeds] clip tokens %s finite %s"
          % (tuple(cond_emb.shape), bool(torch.isfinite(cond_emb).all())), flush=True)
    with torch.no_grad():
        embeds_ref = resampler(cond_emb)
    print("[embeds] resampler output %s" % (tuple(embeds_ref.shape),), flush=True)

    print("[ref] pristine trajectory", flush=True)
    x_ref = run_traj()
    print("[ref] trajectory %d steps" % len(x_ref), flush=True)

    targets = []
    for n, m in resampler.named_modules():
        if isinstance(m, nn.Linear):
            targets.append(("image_proj." + n, m, "embeds"))
    for n, m in adapter.ip_layers.named_modules():
        if isinstance(m, nn.Linear):
            leaf = n.replace("to_kvs.", "", 1)
            hit = re.match(r"^(\d+)_(.+)$", leaf)
            key = "ip_adapter." + (hit.group(1) + "." + hit.group(2) if hit else leaf)
            targets.append((key, m, "trajectory"))
    print("[target] Linear modules: %d" % len(targets), flush=True)

    impacts = {}
    instruments = {}
    for idx, (key, mod, kind) in enumerate(targets, 1):
        w0 = mod.weight.data.clone()
        gs = convrot_group_size(int(w0.shape[1]), args.groupsize)
        if gs is None:
            print("  [skip] %s: no group size for in=%d" % (key, w0.shape[1]), flush=True)
            continue
        with torch.no_grad():
            mod.weight.data = convrot_int8_recon(w0, gs)
            if kind == "trajectory":
                xt = run_traj()
                imp = sum(rel_mse(xt[j], x_ref[j]) for j in range(len(x_ref))) / len(x_ref)
                del xt
            else:
                # read on the Resampler's own output: this layer's only consumer is the embeds
                # the node freezes before sampling, so the trajectory cannot see it
                with torch.no_grad():
                    embeds_q = resampler(cond_emb)
                imp = rel_mse(embeds_q, embeds_ref)
                del embeds_q
        mod.weight.data = w0
        impacts[key] = imp
        instruments[key] = kind
        print("  [%d/%d] %s (%s): rel_mse=%.6e"
              % (idx, len(targets), key, kind, imp), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"ipadapter": os.path.basename(args.ipadapter),
                   "stage": "ip-adapter_sd15_plus: embeds (image_proj) + UNet trajectory (ip_adapter)",
                   "instrument_of_each_layer": instruments,
                   "steps": args.steps, "seed": args.seed, "cfg": args.cfg,
                   "size": [args.width, args.height],
                   "input": os.path.basename(args.image),
                   "impacts": impacts}, f, indent=1, ensure_ascii=False)
    ranked = sorted(impacts.items(), key=lambda x: x[1])
    print("\n== ascending drift ==", flush=True)
    for n, v in ranked[:8]:
        print("  %.6e  %s" % (v, n), flush=True)
    print("== descending drift ==", flush=True)
    for n, v in ranked[-8:]:
        print("  %.6e  %s" % (v, n), flush=True)
    print("wrote %s" % args.out, flush=True)


if __name__ == "__main__":
    main()

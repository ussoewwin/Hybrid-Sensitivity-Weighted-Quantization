"""SD1.5 IP-Adapter Plus trajectory impact (UNet-level measurement).

This is the measurement the other SD1.5 packs use: the adapter is patched into the
UNet the way the apply node patches it, a 20-step trajectory is run, then one
Linear of the adapter is replaced by its ConvRot INT8 reconstruction and the
trajectory is run again. The drift of that layer is the relative MSE of the whole
trajectory - the UNet's own output change - not the adapter's local output.

  python clip_convert/diag_impact_ipadapter_traj.py --base <sd15 unet> --ipadapter <f.pth> \
      --clipvision <f> --image <png> --out <impact.json> --comfy_path <ComfyUI> \
      --ipadapter_dir <node dir> --full
"""

import argparse
import json
import math
import os
import re
import sys

import numpy as np
import torch


def _clear_argv():
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    return saved


def _restore_argv(saved):
    sys.argv = saved


def setup_comfy(comfy_path):
    sys.path.insert(0, comfy_path)
    sys.path.insert(0, os.path.dirname(comfy_path.rstrip("\\/")))


def build_hadamard(size, device="cpu", dtype=torch.float32):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype, device=device)
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


def rotate_weight(w, h, gs):
    of, inf = w.shape
    g = inf // gs
    return torch.matmul(w.view(of, g, gs), h.T.to(w.dtype)).reshape(w.shape)


def quant_rowwise(w):
    amax = w.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
    s = amax / 127.0
    q = (w / s.to(w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, s.to(torch.float32)


def convrot_int8_recon(w, gs):
    h = build_hadamard(gs, device=w.device)
    of, inf = w.shape
    g = inf // gs
    q, s = quant_rowwise(rotate_weight(w.float(), h, gs))
    deq = (q.float() * s.float()).view(of, g, gs)
    return torch.matmul(deq, h.float()).reshape(w.shape).to(w.dtype)


def rel_mse(a, b):
    a = a.float().reshape(a.shape[0], -1)
    b = b.float().reshape(b.shape[0], -1)
    num = torch.sum((a - b) ** 2, dim=1)
    den = torch.sum(b ** 2, dim=1).clamp(min=1e-30)
    return float((num / den).mean().item())


def find_adapter(patched):
    to = patched.model_options.get("transformer_options", {})
    repl = to.get("patches_replace", {}).get("attn2", {})
    for _, v in repl.items():
        for kw in getattr(v, "kwargs", []):
            if "ipadapter" in kw and kw["ipadapter"] is not None:
                return kw["ipadapter"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ipadapter", required=True)
    ap.add_argument("--clipvision", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--comfy_path", required=True)
    ap.add_argument("--ipadapter_dir", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--groupsize", type=int, default=256)
    ap.add_argument("--prompt", default="a photo of a person, high detail")
    ap.add_argument("--negative", default="")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    setup_comfy(args.comfy_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    saved = _clear_argv()
    import comfy.sd
    import comfy.sample as comfy_sample
    import comfy.model_management as mm
    import comfy.clip_vision
    import folder_paths
    _restore_argv(saved)

    _parent = os.path.dirname(os.path.abspath(args.ipadapter_dir))
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    _pkg = os.path.basename(os.path.abspath(args.ipadapter_dir))
    import importlib
    ipadapter_execute = importlib.import_module(_pkg + ".IPAdapterPlus").ipadapter_execute
    ipadapter_model_loader = importlib.import_module(_pkg + ".utils").ipadapter_model_loader

    print("[load] base: %s" % args.base, flush=True)
    saved = _clear_argv()
    patcher = comfy.sd.load_diffusion_model(os.path.abspath(args.base), model_options={})
    _restore_argv(saved)
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
    saved = _clear_argv()
    patched = ipadapter_execute(patcher, ipa, cv, image=image_in, weight=1.0, weight_type="linear",
                                start_at=0.0, end_at=1.0, embeds_scaling="V only")
    _restore_argv(saved)
    if isinstance(patched, tuple):
        patched = patched[0]
    adapter = find_adapter(patched)
    if adapter is None:
        raise SystemExit("adapter instance not found on the patched model")
    print("[patch] adapter: %s" % type(adapter).__name__, flush=True)

    # No CLIP-L weights are present on this machine and this file is a bare UNet, so the
    # text conditioning is a fixed zero context at the cross-attention width. Both sides of
    # every comparison use the same context, so the drift measured is the adapter's.
    ctx = 768
    prompt_pos = [[torch.zeros(1, 77, ctx, device=device), {}]]
    prompt_neg = [[torch.zeros(1, 77, ctx, device=device), {}]]
    print("[cond] fixed zero context %s (no CLIP-L on this machine)" % (tuple(prompt_pos[0][0].shape),), flush=True)

    latent0 = torch.zeros([1, 4, args.height // 8, args.width // 8], device=mm.intermediate_device(), dtype=mm.intermediate_dtype())
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

    print("[*] pristine trajectory", flush=True)
    x_ref = run_traj()
    print("[*] pristine done (%d steps)" % len(x_ref), flush=True)

    targets = []
    for n, m in adapter.image_proj_model.named_modules():
        if isinstance(m, torch.nn.Linear):
            targets.append(("image_proj." + n, m))
    for n, m in adapter.ip_layers.named_modules():
        if isinstance(m, torch.nn.Linear):
            # the runtime names the ip_layers children "1_to_k_ip" because a submodule name
            # cannot hold a dot; the checkpoint names them "1.to_k_ip".
            leaf = n.replace("to_kvs.", "", 1)
            mm = re.match(r"^(\d+)_(.+)$", leaf)
            key = "ip_adapter." + (mm.group(1) + "." + mm.group(2) if mm else leaf)
            targets.append((key, m))
    print("[target] Linear modules: %d" % len(targets), flush=True)

    impacts = {}
    for idx, (key, mod) in enumerate(targets, 1):
        w0 = mod.weight.data
        gs = convrot_group_size(int(w0.shape[1]), args.groupsize)
        if gs is None:
            print("  [skip] %s: no group size" % key, flush=True)
            continue
        with torch.no_grad():
            mod.weight.data = convrot_int8_recon(w0.float(), gs).to(w0.dtype)
            xt = run_traj()
            mod.weight.data = w0
        imp = sum(rel_mse(xt[j], x_ref[j]) for j in range(len(x_ref))) / len(x_ref)
        del xt
        impacts[key] = imp
        print("  [%d/%d] %s: rel_mse=%.6e" % (idx, len(targets), key, imp), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"ipadapter": os.path.basename(args.ipadapter), "steps": args.steps, "seed": args.seed,
                   "stage": "UNet trajectory through the applied adapter", "impacts": impacts}, f, indent=2)
    ranked = sorted(impacts.items(), key=lambda x: x[1])
    print("\n== ascending drift ==", flush=True)
    for n, v in ranked[:10]:
        print("  %.6e  %s" % (v, n), flush=True)
    print("== descending drift ==", flush=True)
    for n, v in ranked[-10:]:
        print("  %.6e  %s" % (v, n), flush=True)
    print("wrote %s" % args.out, flush=True)


if __name__ == "__main__":
    main()

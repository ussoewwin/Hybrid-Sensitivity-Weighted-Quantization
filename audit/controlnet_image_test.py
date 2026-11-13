#!/usr/bin/env python
"""Render one image through a ConvRot INT8 ControlNet on the HSWQ loader route.

Gate 8 measures whether the declared layers load and execute. This tool closes the next question:
does the route actually render? It runs an isolated process (no live ComfyUI server is touched),
loads the base checkpoint, loads the pack through the HSWQ ControlNet Loader node, applies it to a
hint image, samples, decodes and writes a PNG beside a JSON record.

Every path is an argument; nothing about the machine is assumed.

Usage:
    python audit/controlnet_image_test.py --base <checkpoint> --pack <controlnet> --comfy <ComfyUI root>
                                          --out <png> [--json <record>] [--prompt ...] [--steps 12]
                                          [--cfg 6.0] [--size 1024] [--seed 42] [--strength 0.9]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time


def load_hswq_package(comfy_root: str, loader_root: str):
    root = os.path.abspath(comfy_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    cand = os.path.abspath(loader_root or os.path.join(root, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools"))
    init = os.path.join(cand, "__init__.py")
    if not os.path.isfile(init):
        raise SystemExit("HSWQ loader not found at %s (pass --loader-root)" % cand)
    name = "_hswq_img"
    spec = importlib.util.spec_from_file_location(name, init, submodule_search_locations=[cand])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[name] = pkg
    spec.loader.exec_module(pkg)
    return importlib.import_module(name + ".nodes.hswq_load_convrot_int8_controlnet")


def hint_image(torch, size: int):
    ys = torch.linspace(0, 1, size).view(size, 1)
    xs = torch.linspace(0, 1, size).view(1, size)
    img = torch.zeros(3, size, size)
    img[0] = 0.15 + 0.55 * xs
    img[1] = 0.20 + 0.45 * ys
    img[2] = 0.30 + 0.30 * (1 - ys)
    cy, cx, r = 0.45, 0.50, 0.22
    d = ((ys - cy) ** 2 + (xs - cx) ** 2).sqrt()
    disc = (d < r).float()
    img = img * (1 - disc) + disc * torch.tensor([0.95, 0.93, 0.90]).view(3, 1, 1)
    img[:, int(size * 0.78):, :] = 0.05
    return img.permute(1, 2, 0).unsqueeze(0).clamp(0, 1)



def run_ipadapter_image(base_path, vae_path, pack_path, ipadapter_dir, image_path, out_path,
                        json_path, comfy_root, width=512, height=512, steps=12, cfg=7.0,
                        seed=42, img=None):
    """Render one image through the pack on the code path this family has.

    The adapter is patched into the UNet the way the apply node patches it. No CLIP-L
    weights exist on this machine, so the text context is a fixed zero tensor at the
    cross-attention width; the adapter's own conditioning is the image embedding.
    """
    import json as _json
    import os
    import sys
    import time

    import numpy as np
    import torch

    root = os.path.abspath(comfy_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    import comfy.model_management as mm
    import comfy.sample as comfy_sample
    import comfy.sd
    import comfy.utils
    import comfy.clip_vision

    parent = os.path.dirname(os.path.abspath(ipadapter_dir))
    if parent not in sys.path:
        sys.path.insert(0, parent)
    pkg = os.path.basename(os.path.abspath(ipadapter_dir))
    import importlib
    ipadapter_execute = importlib.import_module(pkg + ".IPAdapterPlus").ipadapter_execute
    ipadapter_model_loader = importlib.import_module(pkg + ".utils").ipadapter_model_loader

    t0 = time.time()
    pack_error = ""
    try:
        ipa = ipadapter_model_loader(os.path.abspath(pack_path))
    except Exception as e:  # noqa: BLE001
        pack_error = "%s: %s" % (type(e).__name__, str(e)[:400])
        ipa = None
    base = comfy.sd.load_diffusion_model(os.path.abspath(base_path), model_options={})
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(os.path.abspath(vae_path), safe_load=True))
    cv_path = os.path.join(root, "models", "clip_vision", "clip_vision_h.safetensors")
    if not os.path.isfile(cv_path):
        raise SystemExit("clip vision missing: %s" % cv_path)
    cv = comfy.clip_vision.load(cv_path)

    from PIL import Image
    img = Image.open(image_path).convert("RGB") if image_path else None
    if img is None:
        raise SystemExit("--control-image is required for --family ipadapter")
    arr = torch.from_numpy(np.asarray(img).astype("float32") / 255.0).unsqueeze(0)

    if ipa is None:
        rec = {"tool": "controlnet_image_test.py", "family": "ipadapter",
               "pack": os.path.basename(pack_path), "pack_load_error": pack_error,
               "pack_sha256": __import__("hashlib").sha256(open(pack_path, "rb").read()).hexdigest(),
               "pack_sha256": __import__("hashlib").sha256(open(pack_path, "rb").read()).hexdigest(),
               "verdict": "the pack has no consumer on this route; no render was produced"}
        if json_path:
            os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                _json.dump(rec, f, indent=1, ensure_ascii=False)
        print("%s | pack load failed: %s" % (rec["pack"], pack_error[:160]))
        print("VERDICT: no render (pack has no consumer)")
        if json_path:
            print("wrote %s" % json_path)
        return rec
    try:
        patched = ipadapter_execute(base, ipa, cv, image=arr, weight=1.0, weight_type="linear",
                                    start_at=0.0, end_at=1.0, embeds_scaling="V only")
    except Exception as e:  # noqa: BLE001
        pack_error = "%s: %s" % (type(e).__name__, str(e)[:400])
        rec = {"tool": "controlnet_image_test.py", "family": "ipadapter",
               "pack": os.path.basename(pack_path), "pack_load_error": pack_error,
               "pack_sha256": __import__("hashlib").sha256(open(pack_path, "rb").read()).hexdigest(),
               "pack_sha256": __import__("hashlib").sha256(open(pack_path, "rb").read()).hexdigest(),
               "verdict": "the pack has no consumer on this route; no render was produced"}
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            _json.dump(rec, f, indent=1, ensure_ascii=False)
        print("%s | pack cannot be applied: %s" % (rec["pack"], pack_error[:160]))
        print("VERDICT: no render (pack has no consumer)")
        print("wrote %s" % json_path)
        return rec

    device = mm.get_torch_device()
    ctx = 768
    cond = [[torch.zeros(1, 77, ctx, device=device), {}]]
    neg = [[torch.zeros(1, 77, ctx, device=device), {}]]

    latent = torch.zeros([1, 4, height // 8, width // 8], device=mm.intermediate_device(), dtype=mm.intermediate_dtype())
    latent = comfy_sample.fix_empty_latent_channels(patched, latent, 8, None).to(device)
    noise = comfy_sample.prepare_noise(latent, seed, None)
    with torch.no_grad():
        out = comfy_sample.sample(patched, noise, steps, cfg, "dpmpp_2m", "karras",
                                  cond, neg, latent, denoise=1.0, disable_noise=False,
                                  start_step=None, last_step=None, force_full_denoise=False,
                                  noise_mask=None, callback=None, disable_pbar=True, seed=seed)
        pixels = vae.decode(out)
    pixels = pixels.clamp(0, 1).cpu()
    arr_out = (pixels[0].permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    Image.fromarray(arr_out).save(out_path)
    import hashlib
    rec = {
        "tool": "controlnet_image_test.py",
        "family": "ipadapter",
        "pack": os.path.basename(pack_path),
        "base": os.path.basename(base_path),
        "vae": os.path.basename(vae_path),
        "settings": {"width": width, "height": height, "steps": steps, "cfg": cfg, "seed": seed,
                     "sampler": "dpmpp_2m", "scheduler": "karras",
                     "text_context": "fixed zero context (no CLIP-L on this machine)",
                     "control_image": os.path.basename(image_path)},
        "measurement": {
            "latent_finite": bool(torch.isfinite(out).all()),
            "png_bytes": os.path.getsize(out_path),
            "png_sha256": hashlib.sha256(open(out_path, "rb").read()).hexdigest(),
            "wall_seconds": round(time.time() - t0, 1),
        },
        "verdict": "image rendered",
    }
    if json_path:
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            _json.dump(rec, f, indent=1, ensure_ascii=False)
    print("%s | %sx%s %d steps | latent finite %s | %d bytes | %.1fs"
          % (rec["pack"], width, height, steps, rec["measurement"]["latent_finite"],
             rec["measurement"]["png_bytes"], rec["measurement"]["wall_seconds"]))
    print("VERDICT: image rendered")
    if json_path:
        print("wrote %s" % json_path)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="checkpoint file, or diffusion model name in unet/")
    ap.add_argument("--mode", default="checkpoint", choices=["checkpoint", "flux", "qwen", "zimage", "diffqwen"],
                    help="checkpoint: one file; flux: unet + two clip files + vae; "
                         "qwen: qwen image unet + one text encoder + vae file")
    ap.add_argument("--clip", default="", help="text encoder file name (models/clip) for qwen mode")
    ap.add_argument("--vae-path", default="", help="vae file path for qwen mode")
    ap.add_argument("--control-image", default="",
                    help="control image file name for the zimage and diffqwen modes (read with "
                         "ComfyUI's own LoadImage, so its mask follows the ComfyUI convention)")
    ap.add_argument("--input-dir", default="", help="folder the control image is read from")
    ap.add_argument("--clip1", default="", help="clip file name (models/clip) for flux mode")
    ap.add_argument("--clip2", default="", help="second clip file name for flux mode")
    ap.add_argument("--clip-type", default="flux", help="DualCLIPLoader type")
    ap.add_argument("--vae", default="", help="vae file name (models/vae) for flux mode")
    ap.add_argument("--guidance", type=float, default=3.5, help="flux guidance embedding")
    ap.add_argument("--pack", required=True, help="ConvRot INT8 ControlNet pack")
    ap.add_argument("--comfy", required=True, help="ComfyUI root")
    ap.add_argument("--loader-root", default="")
    ap.add_argument("--family", default="controlnet", choices=["controlnet", "ipadapter"])
    ap.add_argument("--ipadapter_dir", default="")
    ap.add_argument("--out", required=True, help="PNG path to write")
    ap.add_argument("--json", default="")
    ap.add_argument("--prompt", default="a photo of a white ceramic teapot on a wooden table, soft window light, sharp focus")
    ap.add_argument("--negative", default="blurry, low quality")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--sampler", default="dpmpp_2m")
    ap.add_argument("--scheduler", default="karras")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strength", type=float, default=0.9)
    args = ap.parse_args()

    if args.family == "ipadapter":
        if not args.vae:
            print("error: --vae is required for --family ipadapter")
            return 2
        run_ipadapter_image(args.base, args.vae, args.pack, args.ipadapter_dir,
                            args.control_image, args.out, args.json, args.comfy,
                            width=args.size, height=args.size, steps=args.steps,
                            cfg=args.cfg, seed=args.seed)
        return 0

    # resolve outputs before the working directory changes
    json_out = os.path.abspath(args.json) if args.json else ""
    png_out = os.path.abspath(args.out)

    root = os.path.abspath(args.comfy)
    sys.path.insert(0, root)
    os.chdir(root)
    sys.argv = [sys.argv[0]]

    import torch
    import folder_paths

    # the loader's multi-GPU patch reads server.PromptServer.instance; a bare script has none
    import server
    if not hasattr(server.PromptServer, "instance"):
        server.PromptServer.instance = None

    import comfy.sd
    import comfy.sample
    import comfy.utils
    import nodes as comfy_nodes

    pack = os.path.abspath(args.pack)
    folder_paths.add_model_folder_path("controlnet", os.path.dirname(pack))
    folder_paths.add_model_folder_path("model_patches", os.path.dirname(pack))
    cn_node = load_hswq_package(root, args.loader_root)

    started = time.time()
    if args.mode == "qwen":
        # ComfyUI's own loader nodes, unchanged
        (model,) = comfy_nodes.UNETLoader().load_unet(args.base, weight_dtype="default")
        (clip,) = comfy_nodes.CLIPLoader().load_clip(args.clip, "qwen_image", device="default")
        folder_paths.add_model_folder_path("vae", os.path.dirname(os.path.abspath(args.vae_path)))
        (vae,) = comfy_nodes.VAELoader().load_vae(os.path.basename(args.vae_path))
        # Qwen Image latents are 5D: batch, channels, frames, height/8, width/8
        latent = torch.zeros([1, 16, 1, args.size // 8, args.size // 8])
    elif args.mode == "zimage":
        # Z-Image Fun ControlNet: a model patch, applied by its own node, not a ControlNet object
        (model,) = comfy_nodes.UNETLoader().load_unet(args.base, weight_dtype="default")
        (clip,) = comfy_nodes.CLIPLoader().load_clip(args.clip, "qwen_image", device="default")
        folder_paths.add_model_folder_path("vae", os.path.dirname(os.path.abspath(args.vae_path)))
        (vae,) = comfy_nodes.VAELoader().load_vae(os.path.basename(args.vae_path))
        lf = model.get_model_object("latent_format")
        ch, ratio = lf.latent_channels, lf.spacial_downscale_ratio
        if getattr(lf, "latent_dimensions", 2) == 3:
            latent = torch.zeros([1, ch, 1, args.size // ratio, args.size // ratio])
        else:
            latent = torch.zeros([1, ch, args.size // ratio, args.size // ratio])
    elif args.mode == "diffqwen":
        # Qwen-Image diffsynth controlnets: a model patch applied by QwenImageDiffsynthControlnet
        (model,) = comfy_nodes.UNETLoader().load_unet(args.base, weight_dtype="default")
        (clip,) = comfy_nodes.CLIPLoader().load_clip(args.clip, "qwen_image", device="default")
        folder_paths.add_model_folder_path("vae", os.path.dirname(os.path.abspath(args.vae_path)))
        (vae,) = comfy_nodes.VAELoader().load_vae(os.path.basename(args.vae_path))
        latent = torch.zeros([1, 16, 1, args.size // 8, args.size // 8])
    elif args.mode == "flux":
        # ComfyUI's own loader nodes, unchanged
        (model,) = comfy_nodes.UNETLoader().load_unet(args.base, weight_dtype="default")
        (clip,) = comfy_nodes.DualCLIPLoader().load_clip(args.clip1, args.clip2, args.clip_type,
                                                         device="default")
        (vae,) = comfy_nodes.VAELoader().load_vae(args.vae)
        latent = torch.zeros([1, 16, args.size // 8, args.size // 8])
    else:
        model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
            args.base, output_vae=True, output_clip=True, embedding_directory=None)
        latent = torch.zeros([1, 4, args.size // 8, args.size // 8])
    if args.mode in ("zimage", "diffqwen"):
        import importlib
        mp = importlib.import_module(cn_node.__package__ + ".hswq_model_patch_loader")
        (patch,) = mp.HSWQModelPatchLoaderCustom().load_model_patch(os.path.basename(pack), False)
    else:
        (cnet,) = cn_node.HSWQControlNetLoader().load_controlnet(os.path.basename(pack))

    encoder = comfy_nodes.CLIPTextEncode()
    pos = encoder.encode(clip, args.prompt)[0]
    neg = encoder.encode(clip, args.negative)[0]
    if args.mode == "flux":
        from comfy_extras import nodes_flux
        guided = nodes_flux.FluxGuidance.execute(pos, args.guidance)
        pos = getattr(guided, "result", guided)
        if isinstance(pos, tuple):
            pos = pos[0]
    if args.mode in ("zimage", "diffqwen"):
        from comfy_extras import nodes_model_patch
        node_cls = (nodes_model_patch.ZImageFunControlnet if args.mode == "zimage"
                    else nodes_model_patch.QwenImageDiffsynthControlnet)
        if args.control_image:
            if args.input_dir:
                folder_paths.add_model_folder_path("input", os.path.abspath(args.input_dir))
            control_image, control_mask = comfy_nodes.LoadImage().load_image(args.control_image)
            mask_arg = control_mask if float(control_mask.abs().sum()) > 0 else None
        else:
            control_image, mask_arg = hint_image(torch, args.size), None
        (model,) = node_cls().diffsynth_controlnet(
            model, patch, vae, control_image, args.strength, mask=mask_arg)
        pos_c, neg_c = pos, neg
    else:
        applier = comfy_nodes.ControlNetApplyAdvanced()
        pos_c, neg_c = applier.apply_controlnet(pos, neg, cnet, hint_image(torch, args.size), args.strength,
                                               0.0, 1.0, vae)
    # the package's runtime INT8 GEMM patch counts what the render really executed
    import importlib
    pkg_root = cn_node.__name__.rsplit(".nodes", 1)[0]
    arm = importlib.import_module(pkg_root + ".patches.clip_int8_gemm.arm")
    if hasattr(arm, "install"):
        arm.install()
    arm.reset_int8_gemm_stats()
    noise = comfy.sample.prepare_noise(latent, args.seed)
    samples = comfy.sample.sample(model, noise, args.steps, args.cfg, args.sampler, args.scheduler,
                                  pos_c, neg_c, latent, denoise=1.0, seed=args.seed)
    gemm = arm.int8_gemm_stats()
    nan = int(torch.isnan(samples).sum())
    inf = int(torch.isinf(samples).sum())

    decoded = vae.decode(samples)
    image = decoded[0] if isinstance(decoded, (list, tuple)) else decoded
    image = image.clamp(0, 1)
    # Qwen decodes to (batch, frames, height, width, channels); SDXL/FLUX to (batch, height, width, channels)
    while image.ndim > 3 and image.shape[-1] in (1, 3, 4):
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = image.movedim(0, -1)
    image = image.mul(255).round().to(torch.uint8).cpu()
    from PIL import Image
    out = png_out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    Image.fromarray(image.numpy()).save(out)
    elapsed = round(time.time() - started, 1)

    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
    rec = {
        "tool": "controlnet_image_test.py",
        "pack": os.path.basename(pack),
        "base": os.path.basename(args.base),
        "settings": {"width": args.size, "height": args.size, "steps": args.steps, "cfg": args.cfg,
                     "sampler": args.sampler, "scheduler": args.scheduler, "seed": args.seed,
                     "controlnet_strength": args.strength, "prompt": args.prompt,
                     "negative": args.negative,
                     "control_image": args.control_image or "synthetic hint (circle over a horizon)"},
        "measurement": {"latent_shape": list(samples.shape), "latent_nan_count": nan,
                        "latent_inf_count": inf, "latent_finite": nan == 0 and inf == 0,
                        "int8_linear_calls": int(gemm.get("int8_linear_calls", 0)),
                        "png_bytes": os.path.getsize(out), "png_sha256": sha,
                        "wall_seconds": elapsed},
        "verdict": "image rendered" if (nan == 0 and inf == 0) else "latent not finite",
    }
    print("%s | %dx%d %d steps | latent finite %s | INT8 GEMM %d | %s bytes | %.1fs"
          % (rec["pack"], args.size, args.size, args.steps, rec["measurement"]["latent_finite"],
             rec["measurement"]["int8_linear_calls"], rec["measurement"]["png_bytes"], elapsed))
    print("VERDICT: %s" % rec["verdict"])
    if json_out:
        with open(json_out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print("wrote", json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

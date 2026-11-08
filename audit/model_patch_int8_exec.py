#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""One forward of a ConvRot INT8 model patch through the route its card names.

The ControlNet family has this instrument (``audit/controlnet_int8_exec.py``): load the pack through
the HSWQ loader node, run one forward, and report the INT8 GEMM call count with the output's
finiteness. This is the same instrument for the **model patch** families: the pack is built by
``HSWQModelPatchLoaderCustom``, applied by the family node, and the forward runs on the real base
model, so the counted calls are the ones the runtime issues.

Usage:
    python audit/model_patch_int8_exec.py --pack <file> --base <base> --clip <clip> --vae <vae>
                                         --family diffsynth|zimage --comfy <root> [--json <out>]
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys

import torch


def load_hswq_package(comfy_root, loader_root=""):
    root = os.path.abspath(comfy_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    cand = loader_root or os.path.join(root, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools")
    cand = os.path.abspath(cand)
    init = os.path.join(cand, "__init__.py")
    if not os.path.isfile(init):
        raise SystemExit("HSWQ loader not found at %s" % cand)
    name = "_hswq_mp_exec"
    spec = importlib.util.spec_from_file_location(name, init, submodule_search_locations=[cand])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[name] = pkg
    spec.loader.exec_module(pkg)
    return pkg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--family", choices=["diffsynth", "zimage"], required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--prompt", default="a photo of a white ceramic teapot on a wooden table, soft window light, sharp focus")
    ap.add_argument("--negative", default="blurry, low quality")
    args = ap.parse_args()

    root = os.path.abspath(args.comfy)
    cwd = os.getcwd()
    try:
        os.chdir(root)
        import folder_paths  # noqa: E402
    finally:
        os.chdir(cwd)
    sys.path.insert(0, root)

    import comfy.sample as comfy_sample  # noqa: E402
    import comfy.model_management as mm  # noqa: E402
    import nodes as comfy_nodes  # noqa: E402
    import server  # noqa: E402
    if not hasattr(server.PromptServer, "instance"):
        server.PromptServer.instance = None

    pack = os.path.abspath(args.pack)
    folder_paths.add_model_folder_path("model_patches", os.path.dirname(pack))

    pkg = load_hswq_package(root)
    mp = importlib.import_module(pkg.__name__ + ".nodes.hswq_model_patch_loader")
    arm = importlib.import_module(pkg.__name__ + ".patches.clip_int8_gemm.arm")
    if hasattr(arm, "install"):
        arm.install()

    (model,) = comfy_nodes.UNETLoader().load_unet(os.path.basename(args.base), weight_dtype="default")
    (clip,) = comfy_nodes.CLIPLoader().load_clip(os.path.basename(args.clip), "qwen_image", device="default")
    if os.path.dirname(args.vae):
        folder_paths.add_model_folder_path("vae", os.path.dirname(os.path.abspath(args.vae)))
        cache = getattr(folder_paths, "filename_list_cache", None)
        if isinstance(cache, dict):
            cache.pop("vae", None)
    (vae,) = comfy_nodes.VAELoader().load_vae(os.path.basename(args.vae))
    (patch,) = mp.HSWQModelPatchLoaderCustom().load_model_patch(os.path.basename(pack), False)

    from comfy_extras import nodes_model_patch  # noqa: E402
    control_image = torch.zeros(1, args.size, args.size, 3)
    if args.family == "zimage":
        (model,) = nodes_model_patch.ZImageFunControlnet().diffsynth_controlnet(
            model, patch, vae, control_image, args.strength)
        lf = model.get_model_object("latent_format")
        latent = torch.zeros([1, lf.latent_channels, 1, args.size // lf.spacial_downscale_ratio,
                              args.size // lf.spacial_downscale_ratio],
                             device=mm.intermediate_device(), dtype=mm.intermediate_dtype())
    else:
        (model,) = nodes_model_patch.QwenImageDiffsynthControlnet().diffsynth_controlnet(
            model, patch, vae, control_image, args.strength)
        latent = torch.zeros([1, 16, 1, args.size // 8, args.size // 8],
                             device=mm.intermediate_device(), dtype=mm.intermediate_dtype())

    encoder = comfy_nodes.CLIPTextEncode()
    pos = encoder.encode(clip, args.prompt)[0]
    neg = encoder.encode(clip, args.negative)[0]
    del clip
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    noise = comfy_sample.prepare_noise(latent, 42, None)
    arm.reset_int8_gemm_stats()
    with torch.no_grad():
        out = comfy_sample.sample(model, noise, 1, 1.0, "dpmpp_2m", "karras", pos, neg, latent,
                                  denoise=1.0, disable_noise=False, start_step=None, last_step=None,
                                  force_full_denoise=False, noise_mask=None, disable_pbar=True,
                                  seed=42)
    stats = arm.int8_gemm_stats()
    tensors = out if isinstance(out, (list, tuple)) else [out]
    nan = int(sum(int(torch.isnan(t).sum()) for t in tensors))
    inf = int(sum(int(torch.isinf(t).sum()) for t in tensors))

    rec = {
        "tool": "model_patch_int8_exec.py",
        "pack": os.path.basename(pack),
        "family": args.family,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dtype": str(out.dtype),
        "int8_linear_calls": int(stats.get("int8_linear_calls", 0)),
        "output_tensors": len(tensors),
        "output_nan_count": nan,
        "output_inf_count": inf,
        "output_finite": nan == 0 and inf == 0,
        "latent_shape": list(out.shape),
        "control_input": "a zero image of %dx%d (the control input is not what this instrument measures)" % (args.size, args.size),
        "verdict": "INT8 kernel executed, output finite" if (nan == 0 and inf == 0 and stats.get("int8_linear_calls", 0) > 0)
                   else "no INT8 call or a non-finite output",
    }
    print("%-62s | %s | INT8 calls %d | tensors %d | nan %d inf %d | %s"
          % (rec["pack"][:62], args.family, rec["int8_linear_calls"], rec["output_tensors"], nan, inf, rec["verdict"]))
    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

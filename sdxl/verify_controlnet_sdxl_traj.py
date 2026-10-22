# -*- coding: utf-8 -*-
"""SDXL ControlNet: the acceptance gate on the TRAJECTORY (the measured metric).

The protection set is chosen by the trajectory drift of the denoising latent,
so the gate has to be measured the same way - not on a single forward of the
control model. This builds the same pipeline diag_impact_controlnet.py uses
(base SDXL + the ControlNet injected into the conditioning, dpmpp_2m / karras /
20 steps / seed 42 / cfg 7.0), runs the pristine trajectory with the FP16
source, then runs it again with the pack loaded, and reports the per-step
cosine and the mean relative MSE - exactly the quantities the threshold was
chosen on.

The pack is loaded through comfy.controlnet with the hswq INT8 ops so the
comfy_quant stamps are consumed.
"""
import argparse
import importlib
import json
import os
import statistics
import sys
import types

import torch

COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
NODE = os.path.join(COMFY, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools")
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
os.chdir(COMFY)
import comfy.sd  # noqa: E402
import comfy.sample as comfy_sample  # noqa: E402
import comfy.controlnet as comfy_cn  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.utils  # noqa: E402
import comfy.ops as O  # noqa: E402
import server as comfy_server  # noqa: E402

if not hasattr(comfy_server.PromptServer, "instance"):
    comfy_server.PromptServer.instance = None

_spec = importlib.util.spec_from_file_location(
    "hswq_pkg_init", os.path.join(NODE, "__init__.py"), submodule_search_locations=[NODE])
_mod = importlib.util.module_from_spec(_spec)
sys.modules["hswq_pkg_init"] = _mod
_spec.loader.exec_module(_mod)

pkg = types.ModuleType("hswq_pkg")
pkg.__path__ = [NODE]
sys.modules["hswq_pkg"] = pkg
importlib.import_module("hswq_pkg.nodes.hswq_load_convrot_int8_controlnet")
CN_MOD = sys.modules["hswq_pkg.nodes.hswq_load_convrot_int8_controlnet"]

DT = CN_MOD._get_default_compute_dtype()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_pipeline(base, cn_path, use_int8):
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    try:
        out = comfy.sd.load_checkpoint_guess_config(
            os.path.abspath(base), output_vae=False, output_clip=True, embedding_directory=None)
    finally:
        sys.argv = saved
    patcher, clip = out[0], out[1]
    net = patcher.model
    net.to(DEVICE)
    net.eval()

    sd = comfy.utils.load_torch_file(cn_path, safe_load=True)
    if use_int8:
        model_options = {"dtype": DT,
                         "custom_operations": CN_MOD._int8_mixed_precision_ops(compute_dtype=DT)}
    else:
        model_options = {"dtype": DT}
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    try:
        cnet = comfy_cn.load_controlnet_state_dict(sd, model_options=model_options)
    finally:
        sys.argv = saved
    if hasattr(cnet, "to"):
        cnet.to(DEVICE)

    H = W = 1024
    hint = torch.zeros(1, 3, H, W, device=DEVICE)
    cnet.set_cond_hint(hint, strength=1.0)
    cnet.pre_run(net, lambda p: int(p * 1000))

    prompt_pos = clip.encode_from_tokens_scheduled(clip.tokenize("a beautiful landscape, high detail"))
    prompt_neg = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    cond_pos = list(prompt_pos)
    for c in cond_pos:
        c[1]["control"] = cnet
    del clip
    torch.cuda.empty_cache()

    latent0 = torch.zeros([1, 4, H // 8, W // 8], device=mm.intermediate_device(),
                          dtype=mm.intermediate_dtype())
    latent0 = comfy_sample.fix_empty_latent_channels(patcher, latent0, 8, None)
    if latent0.device.type != DEVICE:
        latent0 = latent0.to(DEVICE)
    return patcher, cond_pos, prompt_neg, latent0


def run_traj(patcher, cond_pos, prompt_neg, latent0, steps=20, seed=42, cfg=7.0):
    noise = comfy_sample.prepare_noise(latent0, seed, None)
    xs = []

    def cb(step, x0, x, total):
        xs.append(x.detach().to("cpu", torch.float32, copy=True))

    with torch.no_grad():
        comfy_sample.sample(patcher, noise, steps, cfg, "dpmpp_2m", "karras",
                            cond_pos, prompt_neg, latent0, denoise=1.0, disable_noise=False,
                            start_step=None, last_step=None, force_full_denoise=False,
                            noise_mask=None, callback=cb, disable_pbar=True, seed=seed)
    return xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--packs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="42",
                    help="comma-separated; each seed is one full trajectory")
    a = ap.parse_args()

    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    p, cp, pn, l0 = build_pipeline(a.base, a.src, use_int8=False)
    ref = {s: run_traj(p, cp, pn, l0, seed=s) for s in seeds}
    print("[1] the FP16 trajectory: %d seeds, %d steps, %s each" % (
        len(seeds), len(ref[seeds[0]]), tuple(ref[seeds[0]][0].shape)), flush=True)
    del p, cp, pn, l0
    torch.cuda.empty_cache()

    recs = []
    for path in a.packs:
        p, cp, pn, l0 = build_pipeline(a.base, path, use_int8=True)
        per_seed = {}
        all_cos, all_rm, nf = [], [], 0
        for s in seeds:
            cur = run_traj(p, cp, pn, l0, seed=s)
            cs, rm = [], []
            for i in range(min(len(ref[s]), len(cur))):
                x, y = ref[s][i].reshape(-1), cur[i].reshape(-1)
                cs.append(torch.nn.functional.cosine_similarity(x.reshape(1, -1), y.reshape(1, -1)).item())
                rm.append(float(((y - x) ** 2).sum() / (x ** 2).sum()))
            nf += sum(int((~torch.isfinite(x)).sum().item()) for x in cur)
            per_seed[s] = {"cos_mean": statistics.mean(cs), "cos_min": min(cs),
                           "relmse_mean": statistics.mean(rm)}
            all_cos.extend(cs)
            all_rm.extend(rm)
            del cur
        rec = {"pack": os.path.basename(path), "file_mb": os.path.getsize(path) / 1024 ** 2,
               "cos_mean": statistics.mean(all_cos), "cos_min": min(all_cos),
               "relmse_mean": statistics.mean(all_rm), "nonfinite": nf,
               "seeds": seeds, "per_seed": per_seed}
        recs.append(rec)
        print("  %-30s cos mean=%.6f min=%.6f | relMSE=%.3e | %.1f MB | per-seed mean: %s" % (
            os.path.basename(path)[:30], rec["cos_mean"], rec["cos_min"],
            rec["relmse_mean"], rec["file_mb"],
            " ".join("%.5f" % per_seed[s]["cos_mean"] for s in seeds)), flush=True)
        del p, cp, pn, l0
        torch.cuda.empty_cache()

    json.dump(recs, open(a.out, "w", encoding="utf-8"), indent=1)
    print()
    print("%-32s %10s %10s %10s %9s" % ("pack", "cos.mean", "cos.min", "relMSE", "MB"))
    for r in recs:
        print("%-32s %10.6f %10.6f %10.3e %9.1f" % (
            r["pack"][:32], r["cos_mean"], r["cos_min"], r["relmse_mean"], r["file_mb"]))


if __name__ == "__main__":
    main()

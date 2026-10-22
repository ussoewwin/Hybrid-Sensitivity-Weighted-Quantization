# -*- coding: utf-8 -*-
"""Z-Image-Turbo-Fun-Controlnet model patch: the acceptance on the TRAJECTORY.

The protection set (here: none) is chosen on the model patch's own output, but
the thing that matters downstream is the denoising trajectory of the Z-Image
DiT with the patch applied. This runs the real pipeline - the Z-Image model,
the model patch applied with the ZImageFunControlnet attachment, the production
sampler - with the BF16 patch and with the INT8 pack, and compares the
per-step latents.

The VAE / text encoder are not on this machine, so the control image and the
text conditioning are fixed random tensors: the content does not matter for
the measurement, only that the SAME tensors go through both runs. This is the
same principle the ControlNet sweep uses (a fixed deterministic hint).

The patch is loaded through the HSWQ model-patch loader's own dispatch, so the
INT8 arm is active exactly as ComfyUI runs it.
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
import comfy.utils  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.sample as comfy_sample  # noqa: E402
import comfy.model_management as mm  # noqa: E402
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
importlib.import_module("hswq_pkg.nodes.hswq_model_patch_loader")
MP = sys.modules["hswq_pkg.nodes.hswq_model_patch_loader"]

import comfy_extras.nodes_model_patch as nmp  # noqa: E402
from hswq_pkg.nodes.hswq_model_patch_loader import (  # noqa: E402
    _has_int8_comfy_quant as hswq_has_int8,
    _get_default_compute_dtype as hswq_dtype,
    _int8_mixed_precision_ops as hswq_ops,
    z_image_convert as hswq_z_image_convert,
)
import comfy.ldm.lumina.controlnet as lc  # noqa: E402

DT = torch.bfloat16
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class ZImageControlPatch:
    """The runtime attachment: the ZImageFunControlnet behaviour.

    Mirrors ``comfy_extras.nodes_model_patch.ZImageControlPatch``: the encoded
    control image is fixed here (no VAE), and the patch is invoked once per
    block through ``forward_control_block`` / ``forward_noise_refiner_block``.
    """

    def __init__(self, model_patch, control_latent, strength):
        self.model_patch = model_patch
        self.encoded_image = control_latent
        self.strength = strength
        self.temp_data = None

    def __call__(self, kwargs):
        x = kwargs.get("x")
        img = kwargs.get("img")
        img_input = kwargs.get("img_input")
        txt = kwargs.get("txt")
        pe = kwargs.get("pe")
        vec = kwargs.get("vec")
        block_index = kwargs.get("block_index")
        block_type = kwargs.get("block_type", "")

        cnet_blocks = self.model_patch.model.n_control_layers
        div = round(30 / cnet_blocks)
        cnet_index = block_index // div
        cnet_index_float = block_index / div

        kwargs.pop("img")
        kwargs.pop("txt")

        if cnet_index_float > (cnet_blocks - 1):
            self.temp_data = None
            return kwargs

        if self.temp_data is None or self.temp_data[0] > cnet_index:
            if block_type == "noise_refiner":
                self.temp_data = (-3, (None, self.model_patch.model(txt, self.encoded_image.to(img.dtype), pe, vec)))
            else:
                self.temp_data = (-1, (None, self.model_patch.model(txt, self.encoded_image.to(img.dtype), pe, vec)))

        if block_type == "noise_refiner":
            next_layer = self.temp_data[0] + 1
            self.temp_data = (next_layer, self.model_patch.model.forward_noise_refiner_block(
                block_index, self.temp_data[1][1], img_input[:, :self.temp_data[1][1].shape[1]], None, pe, vec))
            if self.temp_data[1][0] is not None:
                img[:, :self.temp_data[1][0].shape[1]] += (self.temp_data[1][0] * self.strength)
        else:
            while self.temp_data[0] < cnet_index and (self.temp_data[0] + 1) < cnet_blocks:
                next_layer = self.temp_data[0] + 1
                self.temp_data = (next_layer, self.model_patch.model.forward_control_block(
                    next_layer, self.temp_data[1][1], img_input[:, :self.temp_data[1][1].shape[1]], None, pe, vec))
                if self.temp_data[1][0] is not None:
                    img[:, :self.temp_data[1][0].shape[1]] += (self.temp_data[1][0] * self.strength)
        return kwargs


def load_patch(path):
    """The live HSWQ model-patch load path (nodes/hswq_model_patch_loader.py).

    The INT8 branch builds ZImage_Control with mixed_precision_ops so the
    comfy_quant stamps are consumed; the plain branch uses manual_cast. The
    pack is already in the post-conversion layout, so z_image_convert is a
    no-op for it (exactly as the loader comments say).
    """
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    int8 = hswq_has_int8(sd)
    if int8:
        dtype = hswq_dtype()
        operations = hswq_ops(compute_dtype=dtype)
    else:
        dtype = torch.bfloat16
        operations = O.manual_cast
        sd = hswq_z_image_convert(sd)
    config = {"n_control_layers": 3, "additional_in_dim": 17, "refiner_control": True}
    n = 0
    for k in sd:
        if k.startswith("control_layers.") and ".adaLN_modulation.0.weight" in k:
            n = max(n, int(k.split(".")[1]) + 1)
    if n > 0:
        config["n_control_layers"] = n
    embedder_in = sd["control_all_x_embedder.2-1.weight"].shape[1]
    config["control_in_dim"] = embedder_in // 4 - config["additional_in_dim"]
    model = lc.ZImage_Control(
        device=mm.unet_offload_device(), dtype=dtype, operations=operations, **config)
    # exactly the loader's rule: keep a key if its shape matches, and keep the
    # keys the model does not know about yet (lazy-init Linear layers only
    # materialise during _load_from_state_dict)
    msd = model.state_dict()
    filtered = {}
    for k, v in sd.items():
        if k in msd:
            if v.shape == msd[k].shape:
                filtered[k] = v
        else:
            filtered[k] = v
    model.load_state_dict(filtered, strict=False)
    model.eval()
    model.to(DEV)
    patcher = comfy.model_patcher.ModelPatcher(
        model, load_device=mm.get_torch_device(), offload_device=mm.unet_offload_device())
    return patcher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--src", required=True, help="the BF16 model patch")
    ap.add_argument("--packs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="42,137,849")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--strength", type=float, default=1.0)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("the arm: %s | the dtype: %s" % (getattr(O, "_hswq_int8_gemm_patched", False), DT), flush=True)

    # the Z-Image DiT
    saved = list(sys.argv); sys.argv = [saved[0]]
    try:
        out = comfy.sd.load_diffusion_model_state_dict(
            comfy.utils.load_torch_file(a.base, safe_load=True))
    finally:
        sys.argv = saved
    model = out[0] if isinstance(out, tuple) else out
    patcher = model if hasattr(model, "model") else model
    print("[1] the Z-Image model: %s" % type(getattr(patcher, "model", patcher)).__name__, flush=True)

    H = W = a.size
    latent_h, latent_w = H // 16, W // 16
    txt_len = 16

    def control_latent(seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(1, 33, latent_h, latent_w, generator=g).to(DEV, dtype=DT)

    def text_cond(seed):
        # the Z-Image conditioning: cap_feats is the Qwen3-4B output (2560 wide)
        g = torch.Generator(device="cpu").manual_seed(seed + 7)
        cap = torch.randn(1, txt_len, 2560, generator=g).to(DEV, dtype=DT)
        mask = torch.ones(1, txt_len, device=DEV, dtype=torch.bool)
        return [[cap, {"attention_mask": mask}]]

    def run_traj(patch_path, seed):
        pat = load_patch(patch_path)
        m = patcher.clone()
        attach = ZImageControlPatch(pat, control_latent(seed), a.strength)
        # the exact wiring ZImageFunControlnet uses
        m.set_model_noise_refiner_patch(attach)
        m.set_model_double_block_patch(attach)
        # the sampler
        g = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(1, 16, latent_h, latent_w, generator=g).to(DEV, dtype=DT)
        cond = text_cond(seed)
        xs = []

        def cb(step, x0, x, total):
            xs.append(x.detach().to("cpu", torch.float32, copy=True))

        with torch.no_grad():
            comfy_sample.sample(m, noise, a.steps, 1.0, "res_multistep", "simple",
                                cond, [], noise, denoise=1.0, disable_noise=True,
                                start_step=None, last_step=None, force_full_denoise=False,
                                noise_mask=None, callback=cb, disable_pbar=True, seed=seed)
        del m, pat
        torch.cuda.empty_cache()
        return xs

    ref = {s: run_traj(a.src, s) for s in seeds}
    print("[1] the BF16 trajectory: %d seeds x %d steps" % (len(seeds), len(ref[seeds[0]])), flush=True)

    recs = []
    for p in a.packs:
        per = {}
        allc, allr, nf = [], [], 0
        for s in seeds:
            cur = run_traj(p, s)
            cs, rm = [], []
            for i in range(min(len(ref[s]), len(cur))):
                x, y = ref[s][i].reshape(-1), cur[i].reshape(-1)
                cs.append(torch.nn.functional.cosine_similarity(x.reshape(1, -1), y.reshape(1, -1)).item())
                rm.append(float(((y - x) ** 2).sum() / (x ** 2).sum()))
            nf += sum(int((~torch.isfinite(z)).sum().item()) for z in cur)
            per[s] = {"cos_mean": statistics.mean(cs), "cos_min": min(cs)}
            allc += cs; allr += rm
            del cur
        rec = {"pack": os.path.basename(p), "file_mb": os.path.getsize(p) / 1024 ** 2,
               "cos_mean": statistics.mean(allc), "cos_min": min(allc),
               "relmse_mean": statistics.mean(allr), "nonfinite": nf, "per_seed": per}
        recs.append(rec)
        print("  %-44s cos mean=%.6f min=%.6f relMSE=%.3e nonfinite=%d %.1f MB" % (
            os.path.basename(p)[:44], rec["cos_mean"], rec["cos_min"], rec["relmse_mean"],
            nf, rec["file_mb"]), flush=True)

    json.dump(recs, open(a.out, "w", encoding="utf-8"), indent=1)
    print()
    print("%-46s %10s %10s %10s" % ("pack", "cos.mean", "cos.min", "relMSE"))
    for r in recs:
        print("%-46s %10.6f %10.6f %10.3e" % (r["pack"][:46], r["cos_mean"], r["cos_min"], r["relmse_mean"]))


if __name__ == "__main__":
    main()

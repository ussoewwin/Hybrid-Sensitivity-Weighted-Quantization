# -*- coding: utf-8 -*-
"""Z-Image model-patch acceptance: the INT8 pack vs the BF16 source, through the
HSWQ model-patch loader (the path ComfyUI actually uses).

The pack is loaded the way ``HSWQModelPatchLoader`` does: detect the
``int8_tensorwise`` comfy_quant layers, build the ``ZImage_Control`` model with
``mixed_precision_ops``, and load the state dict. The BF16 baseline is built
with the same dispatch but plain ``manual_cast``.

The observable is the model patch's own output: ``ZImage_Control.forward``
returns the per-layer control features, which is what the Z-Image DiT consumes.
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
import comfy.ldm.lumina.controlnet as lc  # noqa: E402

DT = torch.bfloat16
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(sd, use_int8):
    """The dispatch the loader uses, with quant-aware or plain ops."""
    if use_int8:
        ops = MP._int8_mixed_precision_ops()
    else:
        ops = O.manual_cast
    sd = nmp.z_image_convert(sd)
    config = {}
    if "control_layers.4.adaLN_modulation.0.weight" not in sd:
        config["n_control_layers"] = 3
        config["additional_in_dim"] = 17
        config["refiner_control"] = True
    if "control_layers.14.adaLN_modulation.0.weight" in sd:
        config["n_control_layers"] = 15
        config["additional_in_dim"] = 17
    model = lc.ZImage_Control(
        **config,
        device=mm.unet_offload_device(),
        dtype=DT,
        operations=ops,
    )
    model.load_state_dict(sd, assign=True)
    model.eval()
    model.to(DEV)
    return model


def run(model, seed):
    """A fixed forward through the real call signature.

    The runtime calls it as ``model(txt, encoded_image, pe, vec)`` -
    (cap_feats, control_context, x_freqs_cis, adaln_input).

    ``control_context`` is the 4D VAE latent: 33 channels (16 latent + 16 inpaint
    + 1 mask), so ``control_all_x_embedder`` sees 33 * 2 * 2 * 1 = 132. The
    spatial size is a 2x2 patch grid over a 64x64 latent.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    B, T, C = 1, 256, 3840
    cap_feats = torch.randn(B, T, C, generator=g).to(DEV, dtype=DT)
    control_context = torch.randn(B, 33, 64, 64, generator=g).to(DEV, dtype=DT)
    n_ctrl = (64 // 2) * (64 // 2)
    n_tok = n_ctrl + T
    head_dim = C // 30
    x_freqs_cis = torch.randn(B, n_tok, head_dim // 2, 2, generator=g).to(DEV, dtype=torch.float32)
    adaln_input = torch.randn(B, 256, generator=g).to(DEV, dtype=DT)
    with torch.no_grad():
        o = model(cap_feats, control_context, x_freqs_cis, adaln_input)
    return o


def flat(o):
    if torch.is_tensor(o):
        return o.detach().float().reshape(-1).cpu()
    if isinstance(o, dict):
        p = [flat(v) for v in o.values()]
        return torch.cat(p) if p else torch.zeros(0)
    if isinstance(o, (tuple, list)):
        p = [flat(v) for v in o]
        return torch.cat(p) if p else torch.zeros(0)
    return torch.zeros(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--packs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="42,137,849,2024")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]

    print("the arm: %s | the dtype: %s" % (getattr(O, "_hswq_int8_gemm_patched", False), DT), flush=True)
    sd0 = comfy.utils.load_torch_file(a.src, safe_load=True)
    ref = {}
    m0 = build_model(sd0, use_int8=False)
    for s in seeds:
        ref[s] = flat(run(m0, s))
    print("[1] the BF16 baseline: %s over %d seeds" % (tuple(ref[seeds[0]].shape), len(seeds)), flush=True)
    del m0
    torch.cuda.empty_cache()

    recs = []
    for p in a.packs:
        sd = comfy.utils.load_torch_file(p, safe_load=True)
        nq = len([k for k in sd if k.endswith(".comfy_quant")])
        m = build_model(sd, use_int8=True)
        cs, rm = [], []
        for s in seeds:
            f = flat(run(m, s))
            cs.append(torch.nn.functional.cosine_similarity(
                ref[s].reshape(1, -1), f.reshape(1, -1)).item())
            rm.append(float(((f - ref[s]) ** 2).sum() / (ref[s] ** 2).sum()))
        nf = int((~torch.isfinite(f)).sum().item())
        rec = {"pack": os.path.basename(p), "file_mb": os.path.getsize(p) / 1024 ** 2,
               "comfy_quant": nq, "cos_mean": statistics.mean(cs), "cos_min": min(cs),
               "relmse_mean": statistics.mean(rm), "nonfinite": nf}
        recs.append(rec)
        print("  %-40s cos mean=%.6f min=%.6f relMSE=%.3e nonfinite=%d %.1f MB" % (
            os.path.basename(p)[:40], rec["cos_mean"], rec["cos_min"], rec["relmse_mean"],
            nf, rec["file_mb"]), flush=True)
        del m, sd, f
        torch.cuda.empty_cache()

    json.dump(recs, open(a.out, "w", encoding="utf-8"), indent=1)
    print()
    print("%-42s %10s %10s %10s %9s" % ("pack", "cos.mean", "cos.min", "relMSE", "MB"))
    for r in recs:
        print("%-42s %10.6f %10.6f %10.3e %9.1f" % (
            r["pack"][:42], r["cos_mean"], r["cos_min"], r["relmse_mean"], r["file_mb"]))


if __name__ == "__main__":
    main()

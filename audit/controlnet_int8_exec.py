#!/usr/bin/env python
"""Execute a ConvRot INT8 ControlNet once on the HSWQ loader route and count INT8 GEMM calls.

Gate 8 of the audit procedure requires more than "the weights were applied": where the route
claims an INT8 kernel, the kernel counter must be non-zero after a real forward, and the output
must be finite. This tool performs that measurement for ControlNet packs.

It loads the pack through the HSWQ ControlNet Loader node (the route the card names), enables the
package's runtime INT8 GEMM patch (``patches.clip_int8_gemm.arm``, runtime-only, ComfyUI core is
not edited), resets its counter, runs one forward with the shapes the model family expects, and
reports the counter with the output's finiteness.

Usage:
    python audit/controlnet_int8_exec.py --pack <file> --comfy <root> [--json <out>] [--device cuda]
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys


def load_hswq_package(comfy_root: str, loader_root: str):
    root = os.path.abspath(comfy_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    cand = os.path.abspath(loader_root or os.path.join(root, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools"))
    init = os.path.join(cand, "__init__.py")
    if not os.path.isfile(init):
        raise SystemExit("HSWQ loader not found at %s (pass --loader-root)" % cand)
    name = "_hswq_cn_exec"
    spec = importlib.util.spec_from_file_location(name, init, submodule_search_locations=[cand])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[name] = pkg
    spec.loader.exec_module(pkg)
    return name, cand


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--loader-root", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    root = os.path.abspath(args.comfy)
    cwd = os.getcwd()
    try:
        os.chdir(root)
        import torch
        import folder_paths
        import comfy.controlnet as cc
        from comfy import model_management
    finally:
        os.chdir(cwd)

    pack = os.path.abspath(args.pack)
    folder_paths.add_model_folder_path("controlnet", os.path.dirname(pack))
    pkg_name, pkg_dir = load_hswq_package(root, args.loader_root)
    node = importlib.import_module(pkg_name + ".nodes.hswq_load_convrot_int8_controlnet")
    arm = importlib.import_module(pkg_name + ".patches.clip_int8_gemm.arm")

    torch.manual_seed(0)
    net = node.HSWQControlNetLoader().load_controlnet(os.path.basename(pack))[0]
    cm = net.control_model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = cm.dtype
    cm = cm.to(device=device, dtype=dtype).eval()

    is_flux = type(cm).__name__ == "ControlNetFlux"
    torch.manual_seed(0)
    if is_flux:
        latent_ch = getattr(cm, "control_latent_channels", None)
        in_ch = latent_ch or 16
        x = torch.randn(1, in_ch, 64, 64, device=device, dtype=dtype)
        hint = torch.randn(1, in_ch, 64, 64, device=device, dtype=dtype)
        ctx = torch.randn(1, 256, 4096, device=device, dtype=dtype)
        y = torch.randn(1, 768, device=device, dtype=dtype)
        guid = torch.tensor([3.5], device=device, dtype=dtype)
        call = dict(x=x, timesteps=torch.tensor([500.0], device=device, dtype=dtype),
                    context=ctx, y=y, guidance=guid, hint=hint)
    else:
        x = torch.randn(1, 4, 64, 64, device=device, dtype=dtype)
        hint = torch.rand(1, 3, 512, 512, device=device, dtype=dtype)
        ctx = torch.randn(1, 77, 2048, device=device, dtype=dtype)
        y = torch.randn(1, 2816, device=device, dtype=dtype)
        call = dict(x=x, hint=hint, timesteps=torch.tensor([500.0], device=device, dtype=dtype),
                    context=ctx, y=y)

    if type(cm).__name__ == "QwenImageFunControlNetModel":
        # Qwen Fun ControlNet runs against a base Qwen Image transformer. Only the entry
        # modules are needed for a forward; build them with zero blocks so the stub is tiny.
        import comfy.ops
        import comfy.ldm.qwen_image.model as qm
        in_ch = cm.control_img_in.weight.shape[1] // 4
        ctx_dim = cm.control_blocks[0].attn.to_k.weight.shape[1]
        base = qm.QwenImageTransformer2DModel(
            final_layer=False, num_layers=0, joint_attention_dim=ctx_dim,
            dtype=dtype, device=device, operations=comfy.ops.disable_weight_init)
        base = base.to(device=device, dtype=dtype).eval()
        x = torch.randn(1, 16, 1, 64, 64, device=device, dtype=dtype)
        hint = torch.rand(1, in_ch, 64, 64, device=device, dtype=dtype)
        ctx = torch.randn(1, 77, ctx_dim, device=device, dtype=dtype)
        call = dict(x=x, timesteps=torch.tensor([500.0], device=device, dtype=dtype),
                    context=ctx, hint=hint, base_model=base)

    installed = bool(arm.install()) if hasattr(arm, "install") else False
    arm.reset_int8_gemm_stats()
    rec = {"pack": os.path.basename(pack), "class": type(cm).__name__, "device": str(device),
           "dtype": str(dtype), "int8_gemm_patch": installed, "int8_linear_calls": 0,
           "output_finite": None, "output_shape": None, "error": None}
    try:
        with torch.no_grad():
            out = cm(**call)
        stats = arm.int8_gemm_stats()
        rec["int8_linear_calls"] = int(stats.get("int8_linear_calls", 0))
        def collect(v, acc):
            if torch.is_tensor(v):
                acc.append(v)
            elif isinstance(v, dict):
                for k in v:
                    collect(v[k], acc)
            elif isinstance(v, (list, tuple)):
                for e in v:
                    collect(e, acc)

        tensors = []
        collect(out, tensors)
        finite = all(bool(torch.isfinite(o).all()) for o in tensors) if tensors else False
        shape = [list(o.shape) for o in tensors]
        rec["output_tensors"] = len(tensors)
        rec["output_finite"] = finite
        rec["output_shape"] = shape
    except Exception as exc:  # noqa: BLE001
        rec["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:300])
        rec["int8_linear_calls"] = int(arm.int8_gemm_stats().get("int8_linear_calls", 0))

    if rec["error"]:
        rec["verdict"] = "forward failed"
    elif rec["int8_linear_calls"] <= 0:
        rec["verdict"] = "no INT8 kernel call"
    elif not rec["output_finite"]:
        rec["verdict"] = "non-finite output"
    else:
        rec["verdict"] = "INT8 kernel executed, output finite"

    print("%s | %s | INT8 calls %d | finite %s | %s"
          % (rec["pack"], rec["class"], rec["int8_linear_calls"], rec["output_finite"], rec["verdict"]))
    if rec["error"]:
        print("   error:", rec["error"])
    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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



def run_ipadapter(pack_path: str, ipadapter_dir: str, device: str):
    """One forward through the graph the node builds, with the pack's own tensors.

    The question is the same as for a ControlNet: does the declared int8 reach an INT8
    kernel, and does the executed path stay finite?
    """
    import torch
    from safetensors.torch import load_file
    import comfy.ops as O

    flat = load_file(pack_path)
    st = {"image_proj": {}, "ip_adapter": {}}
    for k, v in flat.items():
        if not torch.is_tensor(v):
            continue
        if k.startswith("image_proj."):
            st["image_proj"][k.replace("image_proj.", "", 1)] = v
        elif k.startswith("ip_adapter."):
            st["ip_adapter"][k.replace("ip_adapter.", "", 1)] = v

    if ipadapter_dir and ipadapter_dir not in sys.path:
        sys.path.insert(0, ipadapter_dir)
    from image_proj_models import Resampler  # noqa: E402

    res = Resampler(dim=768, depth=4, dim_head=64, heads=12, num_queries=16,
                    embedding_dim=1280, output_dim=768, ff_mult=4,
                    max_seq_len=257, apply_pos_emb=False, num_latents_mean_pooled=0)
    try:
        res.load_state_dict(st["image_proj"], strict=True)
        load_error = ""
    except Exception as e:  # noqa: BLE001
        load_error = "%s: %s" % (type(e).__name__, str(e)[:300])

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    rec = {"pack": os.path.basename(pack_path), "class": "Resampler+To_KV", "device": str(dev),
           "pack_sha256": __import__("hashlib").sha256(open(pack_path, "rb").read()).hexdigest(),
           "dtype": "float32", "int8_gemm_patch": False, "int8_linear_calls": 0,
           "output_finite": None, "output_shape": None, "error": load_error or None}
    if load_error:
        rec["verdict"] = "the node's loader cannot take the pack"
        return rec

    res = res.to(dev).to(torch.float32).eval()

    ck = O.quant_ops.ck
    orig = ck.int8_linear
    n = [0]

    def counted(*a, **kw):
        n[0] += 1
        return orig(*a, **kw)

    ck.int8_linear = counted
    try:
        with torch.no_grad():
            x = torch.zeros(1, 257, 1280, device=dev, dtype=torch.float32)
            y = res(x)
        rec["output_shape"] = list(y.shape)
        rec["output_finite"] = bool(torch.isfinite(y).all())
        rec["output_nan_count"] = int(torch.isnan(y).sum())
        rec["output_inf_count"] = int(torch.isinf(y).sum())
        rec["output_tensors"] = 1
    except Exception as e:  # noqa: BLE001
        rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
    finally:
        ck.int8_linear = orig

    rec["int8_linear_calls"] = n[0]
    if rec["error"]:
        rec["verdict"] = "forward failed"
    elif rec["int8_linear_calls"] <= 0:
        rec["verdict"] = "no INT8 kernel call"
    elif not rec["output_finite"]:
        rec["verdict"] = "non-finite output"
    else:
        rec["verdict"] = "INT8 kernel executed, output finite"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--loader-root", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default="")
    ap.add_argument("--family", default="controlnet", choices=["controlnet", "ipadapter"])
    ap.add_argument("--ipadapter_dir", default="", help="the ipadapter node directory for --family ipadapter")
    args = ap.parse_args()

    if args.family == "ipadapter":
        rec = run_ipadapter(args.pack, args.ipadapter_dir, args.device)
        print("%s | %s | INT8 calls %d | finite %s | %s"
              % (rec["pack"], rec["class"], rec["int8_linear_calls"], rec["output_finite"], rec["verdict"]))
        if args.json:
            os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=1, ensure_ascii=False)
            print("wrote %s" % args.json)
        return 0

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
        # the cldm family's inputs are read from the pack's own graph (the runtime
        # names), so the same measurement runs on SD1.5 (context 768, no label
        # embedding) and SDXL (context 2048, y 2816) alike
        ctx_w = None
        y_w = None
        for n2, mm2 in cm.named_modules():
            if n2.endswith("attn2.to_k") and getattr(mm2, "weight", None) is not None:
                ctx_w = int(mm2.weight.shape[1])
            if n2 == "label_emb.0.0" and getattr(mm2, "weight", None) is not None:
                y_w = int(mm2.weight.shape[1])
            elif n2 == "label_emb" and getattr(mm2, "weight", None) is not None:
                y_w = int(mm2.weight.shape[1])
        x = torch.randn(1, 4, 64, 64, device=device, dtype=dtype)
        hint = torch.rand(1, 3, 512, 512, device=device, dtype=dtype)
        ctx = torch.randn(1, 77, ctx_w, device=device, dtype=dtype)
        call = dict(x=x, hint=hint, timesteps=torch.tensor([500.0], device=device, dtype=dtype),
                    context=ctx)
        if y_w is not None:
            call["y"] = torch.randn(1, y_w, device=device, dtype=dtype)

    if type(cm).__name__ == "QwenImageControlNetModel":
        # Qwen InstantX ControlNet: self-contained, no base model needed. The hint carries the
        # latent plus the mask channels, patchified by process_img.
        import comfy.ldm.qwen_image.model as qm
        ctx_dim = cm.txt_in.weight.shape[1]
        hint_ch = cm.controlnet_x_embedder.weight.shape[1] // 4
        x = torch.randn(1, 16, 1, 64, 64, device=device, dtype=dtype)
        hint = torch.randn(1, hint_ch, 1, 64, 64, device=device, dtype=dtype)
        ctx = torch.randn(1, 77, ctx_dim, device=device, dtype=dtype)
        call = dict(x=x, timesteps=torch.tensor([500.0], device=device, dtype=dtype),
                    context=ctx, hint=hint)

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
        nan_total = sum(int(torch.isnan(o).sum()) for o in tensors)
        inf_total = sum(int(torch.isinf(o).sum()) for o in tensors)
        finite = bool(tensors) and nan_total == 0 and inf_total == 0
        rec["output_nan_count"] = nan_total
        rec["output_inf_count"] = inf_total
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

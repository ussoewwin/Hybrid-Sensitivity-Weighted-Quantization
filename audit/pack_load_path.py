# -*- coding: utf-8 -*-
"""Gate 8 tool: does the published pack actually execute on the loader it advertises?

A pack can be arithmetically perfect and still be unusable: the quantised bytes only help if
Measured on the CLIP-family INT8 packs of the audited models,
the stored weights were exactly right, while the runtime copied raw int8 codes into
parameters, so the encoder produced output uncorrelated with its own source, silently.

This tool loads a pack through a named route and reports three things, all of which must hold:

  raw_code   - parameters whose values are integers with absmax >= 100: the raw codes landed
               in a float parameter, i.e. the quantisation was never applied (must be 0)
  applied    - declared quantised layers whose runtime parameter matches the offline
               dequantisation (scale, then ConvRot inverted) above cosine 0.98
  gemm       - INT8 GEMM call count after real forwards, where the loader exposes a counter

Routes:
  clip_text   : CLIPLoader / DualCLIPLoader text encoder
  clip_vision : comfy.clip_vision.load
  diffusion   : comfy.sd.load_diffusion_model_state_dict with int8_tensorwise operations

    python audit/pack_load_path.py --pack <file> --route clip_text --clip-type stable_diffusion \
        --module-attr clip_l [--image <png>] [--json <out>]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
from safetensors import safe_open

COMFY = os.environ.get("COMFY_PATH", "")  # ComfyUI root; required for the clip_text/clip_vision/diffusion routes


def hadamard(size: int) -> torch.Tensor:
    """Regular (power-of-4) normalised Hadamard, as the converter builds it."""
    h4 = torch.tensor([[1., 1., 1., -1.], [1., 1., -1., 1.],
                       [1., -1., 1., 1.], [-1., 1., 1., 1.]], dtype=torch.float64)
    h, cur = h4, 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / math.sqrt(size)


def dequantised(pack: str, name: str, cfg: dict) -> torch.Tensor:
    with safe_open(pack, framework="pt") as f:
        wi = f.get_tensor(name + ".weight")
        sc = f.get_tensor(name + ".weight_scale").float()
    d = wi.float() * (sc.reshape(-1, 1) if sc.numel() == wi.shape[0] else sc)
    gs = cfg.get("convrot_groupsize") or 256
    if d.dim() == 2 and d.shape[1] % gs == 0 and gs >= 4 and math.log(gs, 4) % 1 == 0:
        o, i = d.shape
        d = torch.matmul(d.view(o, i // gs, gs),
                         hadamard(gs).to(d.dtype)).reshape(o, i)
    return d


def cos64(a, b):
    a = a.detach().cpu()
    b = b.detach().cpu().to(a.dtype)
    if a.shape != b.shape:
        return None
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    dot = na = nb = 0.0
    for i in range(0, a.numel(), 2_000_000):
        x, y = a[i:i + 2_000_000], b[i:i + 2_000_000]
        dot += float((x * y).sum())
        na += float((x * x).sum())
        nb += float((y * y).sum())
    if na <= 0 or nb <= 0:
        return None
    # Cauchy-Schwarz bounds this by 1; accumulated floating-point error can put it at
    # 1.0000000000000002, which is rounding and not a measurement, so the value is clamped.
    return min(1.0, dot / (na ** 0.5 * nb ** 0.5))


def key_variants(name: str):
    stem = name[: -len(".weight")] if name.endswith(".weight") else name
    return {stem, stem.replace("model.language_model.", "model."),
            stem.replace("language_model.", ""), stem.replace("model.", "")}


def dequantised_conv(pack, name, cfg):
    """Dequantise a 4-D conv2d weight: per-output-channel scale, then the ConvRot inverse on the
    flattened input dimension (out, in*kh*kw) with the group size the marker declares."""
    with safe_open(pack, framework="pt") as f:
        wi = f.get_tensor(name + ".weight")
        sc = f.get_tensor(name + ".weight_scale").float()
    o, i, kh, kw = wi.shape
    w2 = wi.float().reshape(o, i * kh * kw) * sc.reshape(o, 1)
    gs = cfg.get("convrot_groupsize") or cfg.get("groupsize") or 64
    if (cfg.get("convrot") or "convrot" in str(cfg.get("format", ""))) and (i * kh * kw) % gs == 0:
        h = hadamard(gs).to(w2.dtype)
        w2 = torch.matmul(w2.reshape(o, (i * kh * kw) // gs, gs), h).reshape(o, i * kh * kw)
    return w2.reshape(o, i, kh, kw)


def execute_layer(module, pack, name, cfg):
    """Run one module with its loaded weights, then with the dequantised weights.

    If the loaded representation is the real weight the two runs agree. On the loader node's INT8
    conv2d path the parameter holds the int8 codes and the patched execution dequantises them, so
    this is the only admissible test for conv2d: comparing the parameter to the stored bytes
    reports a defect that the executed layer does not have.
    """
    ref = dequantised_conv(pack, name, cfg)
    o, i, kh, kw = ref.shape
    dev = module.weight.device
    x = torch.randn(1, i, 8, 8, device=dev, dtype=module.weight.dtype)
    with torch.no_grad():
        y_loaded = module(x).float().cpu()
    saved = module.weight.data.clone()
    module.weight.data = ref.to(module.weight.dtype).to(dev)
    try:
        with torch.no_grad():
            y_deq = module(x).float().cpu()
    finally:
        module.weight.data = saved
    c = cos64(y_loaded, y_deq)
    return {"output_cos": c,
            "relative_difference": float((y_loaded - y_deq).norm()
                                         / max(float(y_deq.norm()), 1e-12))}


def classify(model, pack, qm, exec_conv=True):
    """Judge linear (2-D) layers by inverting the format; judge conv2d layers by executing them.

    A conv2d weight is 4-D, and on the loader node's INT8 path it is stored as codes and
    dequantised at execution. Inverting it with the linear rules, or comparing its parameter with
    the stored bytes, measures the tool rather than the product; `conv_execution` records what
    each executed conv2d layer produced.
    """
    params = dict(model.named_parameters())
    modules = dict(model.named_modules())
    raw = applied = other = unmatched = conv_executed = 0
    conv_keys = []
    conv_exec = []
    raw_names = []
    for name, cfg in sorted(qm.items()):
        gw = name if name.endswith(".weight") else name + ".weight"
        with safe_open(pack, framework="pt") as f:
            stored_shape = tuple(f.get_tensor(gw).shape) if gw in f.keys() else None
        cand = []
        for v in key_variants(gw):
            tail = v if v.endswith(".weight") else v + ".weight"
            # a checkpoint may store the layer with an extra prefix the model object does not
            # carry (model.diffusion_model.input_blocks... vs input_blocks...), so match in
            # both directions - but only accept a parameter whose shape is the stored shape,
            # because a bare suffix match otherwise collides across depth (declared
            # "....output_blocks.7.0.op.weight" matching a parameter named "op.weight", which
            # resolved 51 one-dimensional false matches in the SDXL pack)
            cand += [k for k in params
                     if (k.endswith(tail) or tail.endswith("." + k))
                     and (stored_shape is None or tuple(params[k].shape) == stored_shape)]
        cand = sorted(set(cand))
        if not cand:
            unmatched += 1
            continue
        key = cand[0]
        par = params[key]
        if par.dim() != 2:
            # conv2d: judged by what it computes, on the route the caller loaded it through
            conv_keys.append(key)
            stem = name[:-len(".weight")] if name.endswith(".weight") else name
            mod = modules.get(key[:-len(".weight")])
            if mod is None:
                mod = next((m for n, m in modules.items() if n.endswith(stem)), None)
            if mod is None or not exec_conv:
                continue
            conv_executed += 1
            try:
                r = execute_layer(mod, pack, stem, cfg)
            except Exception as exc:
                r = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:120])}
            r["layer"] = key
            conv_exec.append(r)
            c2 = r.get("output_cos")
            if c2 is not None and c2 > 0.999:
                applied += 1
            elif c2 is not None:
                raw += 1
                raw_names.append("%s (executes at cos %.6f of its dequantised weights)"
                                 % (key, c2))
            continue
        t = par.float()
        with safe_open(pack, framework="pt") as f:
            stored = f.get_tensor(gw).float()
        if torch.equal(t, stored):
            # the parameter is the stored int8 tensor read as float: copied, not applied
            raw += 1
            raw_names.append(key)
            continue
        if bool(torch.equal(t, t.round())) and float(t.abs().max()) >= 100.0:
            raw += 1
            raw_names.append(key)
            continue
        c = cos64(t, dequantised(pack, name, cfg))
        if c is not None and c > 0.98:
            applied += 1
        else:
            other += 1
    whole_raw = 0
    for _, p in model.named_parameters():
        if p.is_floating_point():
            tq = p.float()
            if bool(torch.equal(tq, tq.round())) and float(tq.abs().max()) >= 100.0:
                whole_raw += 1
    return {"declared_layers": len(qm), "raw_code_layers": raw, "applied_layers": applied,
            "other_layers": other, "unmatched_layers": unmatched,
            "conv_layers_executed": conv_executed, "conv_layer_examples": conv_keys[:5],
            "conv_execution": conv_exec[:8],
            "raw_code_params_whole_model": whole_raw,
            "raw_code_layer_names": raw_names}


def declared(pack):
    with safe_open(pack, framework="pt") as f:
        meta = f.metadata() or {}
    qm = json.loads(meta.get("_quantization_metadata", "{}")).get("layers", {})
    return qm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--route", required=True,
                    choices=["clip_text", "clip_vision", "diffusion", "controlnet"])
    ap.add_argument("--clip-type", default="stable_diffusion")
    ap.add_argument("--module-attr", default="clip_l",
                    help="attribute of cond_stage_model holding the text tower")
    ap.add_argument("--image", default="", help="a png for the clip_vision route")
    ap.add_argument("--layer-scope", default="",
                    help="only judge declared layers whose name starts with this prefix "
                         "(e.g. text_model for the text route, vision_model for the vision "
                         "route); the rest are reported as out of scope for this route")
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--comfy", default="", help="ComfyUI root (overrides the COMFY_PATH environment variable)")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    global COMFY
    if args.comfy:
        COMFY = args.comfy
    if not COMFY:
        print("ComfyUI root unknown: pass --comfy or set COMFY_PATH")
        return 2
    if COMFY not in sys.path:
        sys.path.insert(0, COMFY)
    cwd = os.getcwd()
    os.chdir(COMFY)
    try:
        import importlib
        import server as _srv
        if not hasattr(_srv.PromptServer, "instance"):
            _srv.PromptServer.instance = None
        import comfy.clip_vision
        import comfy.ops as cops
        import comfy.sd
        import comfy.utils
        import nodes as cnodes
        torch.set_grad_enabled(False)

        pack = os.path.abspath(args.pack)
        qm_all = declared(pack)
        qm = {k: v for k, v in qm_all.items()
              if not args.layer_scope or k.startswith(args.layer_scope)}
        out_scope = {k: v for k, v in qm_all.items()
                    if args.layer_scope and not k.startswith(args.layer_scope)}
        gemm = None
        arm = None
        node = os.path.join(COMFY, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools")
        pre_flags = {}
        try:
            # ComfyUI runs a node's prestartup_script.py before importing it; that script installs
            # the state-dict wrappers which make quantised loads happen. Skipping it measures a
            # state no user reaches, so it is executed here in the same order.
            pre = importlib.util.spec_from_file_location(
                "hswq_prestartup", os.path.join(node, "prestartup_script.py"))
            pm = importlib.util.module_from_spec(pre)
            sys.modules["hswq_prestartup"] = pm
            pre.loader.exec_module(pm)
            pre_flags["prestartup_armed"] = True
        except Exception as e:
            pre_flags["prestartup_armed"] = False
            pre_flags["prestartup_error"] = "%s: %s" % (type(e).__name__, str(e)[:140])
        try:
            spec = importlib.util.spec_from_file_location(
                "hswq_loader", os.path.join(node, "__init__.py"),
                submodule_search_locations=[node])
            m = importlib.util.module_from_spec(spec)
            sys.modules["hswq_loader"] = m
            spec.loader.exec_module(m)
            arm = importlib.import_module("hswq_loader.patches.clip_int8_gemm.arm")
        except Exception as e:
            print("note: HSWQ loader not importable, no GEMM counter:", e)

        prompts = args.prompt or ["a photo of a cat", "the old bridge",
                                  "INFRA-4821 - proceed as discussed.", "bonsoir, Elliot"]
        out = {"pack": os.path.basename(pack), "route": args.route,
               "declared_layers": len(qm), "layer_scope": args.layer_scope or "all",
               "declared_out_of_scope_for_this_route": len(out_scope)}
        out.update(pre_flags)

        if args.route == "clip_text":
            torch.manual_seed(1234)
            if arm:
                arm.reset_int8_gemm_stats()
            clip = cnodes.CLIPLoader().load_clip(clip_name=os.path.basename(pack),
                                                 type=args.clip_type, device="default")[0]
            model = getattr(clip.cond_stage_model, args.module_attr, clip.cond_stage_model)
            out.update(classify(model, pack, qm))
            for p in prompts:
                with torch.no_grad():
                    clip.encode_from_tokens(clip.tokenize(p))
            if arm:
                gemm = dict(arm.int8_gemm_stats())
        elif args.route == "clip_vision":
            torch.manual_seed(1234)
            if arm:
                arm.reset_int8_gemm_stats()
            cv = comfy.clip_vision.load(pack)
            out.update(classify(cv.model, pack, qm))
            if args.image and os.path.exists(args.image):
                import numpy as np
                from PIL import Image
                arr = np.array(Image.open(args.image).convert("RGB").resize((224, 224)),
                               dtype=np.float32)
                with torch.no_grad():
                    cv.encode_image(torch.from_numpy(arr)[None], crop=False)
            if arm:
                gemm = dict(arm.int8_gemm_stats())
        elif args.route == "controlnet":
            import folder_paths
            folder_paths.add_model_folder_path("controlnet", os.path.dirname(pack))
            holder, = cnodes.ControlNetLoader().load_controlnet(os.path.basename(pack))
            out.update(classify(holder.control_model, pack, qm))
            out["controlnet_class"] = type(holder.control_model).__name__
            del holder
        else:
            # The diffusion route the product uses: the loader node's own INT8 entry, which
            # installs the Conv2d quantisation inside its scope. A hand-built ops load through
            # ComfyUI's generic path is not a route users take for conv2d.
            import folder_paths
            folder_paths.add_model_folder_path("diffusion_models", os.path.dirname(pack))
            cqi = None
            # the module path contains a space and a dotted directory name, so it is loaded by
            # file location rather than by import name
            try:
                cqi_path = os.path.join(node, "nodes", "sd1.5 int8", "comfy_quant_int8.py")
                cspec = importlib.util.spec_from_file_location("hswq_cqi_sd15", cqi_path)
                cqi = importlib.util.module_from_spec(cspec)
                sys.modules["hswq_cqi_sd15"] = cqi
                cspec.loader.exec_module(cqi)
            except Exception as e:
                print("note: HSWQ INT8 UNet loader not importable:", str(e)[:140])
                cqi = None
            if cqi is not None and hasattr(cqi, "load_unet_hswq_weight_dtype"):
                got = cqi.load_unet_hswq_weight_dtype(os.path.basename(pack),
                                                      "int8_tensorwise")
                loaded = got[0] if isinstance(got, tuple) else got
                out["route_impl"] = "HSWQ load_unet_hswq_weight_dtype(int8_tensorwise)"
            else:
                from comfy.quant_ops import QUANT_ALGOS
                sd = comfy.utils.load_torch_file(pack, safe_load=True)
                ops = cops.mixed_precision_ops(
                    {"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]},
                    torch.bfloat16, full_precision_mm=False, disabled=[])
                loaded = comfy.sd.load_diffusion_model_state_dict(
                    sd, model_options={"custom_operations": ops})
                out["route_impl"] = "stock load_diffusion_model_state_dict + int8 ops"
            net = getattr(loaded, "model", loaded)
            out.update(classify(getattr(net, "diffusion_model", net), pack, qm))

        out["gemm_after_forwards"] = gemm
        judged = out["declared_layers"]
        out["judged_layers"] = judged
        ok = (out["raw_code_layers"] == 0 and out["raw_code_params_whole_model"] == 0
              and out["other_layers"] == 0 and out["unmatched_layers"] == 0
              and out["applied_layers"] == judged)
        out["verdict"] = "pack applies on this route" if ok else \
            "pack does NOT apply on this route"
        print("%-46s declared %4d judged %4d | applied %4d raw %3d other %3d unmatched %3d "
              "| conv executed %3d | gemm %s"
              % (out["pack"][:46], out["declared_layers"], out["judged_layers"],
                 out["applied_layers"], out["raw_code_layers"], out["other_layers"],
                 out["unmatched_layers"], out["conv_layers_executed"],
                 (gemm or {}).get("int8_linear_calls")), flush=True)
        if out["raw_code_layer_names"]:
            print("   raw-code layers:", out["raw_code_layer_names"][:5], flush=True)
        print("VERDICT:", out["verdict"])
        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=1)
            print("wrote", args.json)
        return 0 if ok else 1
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    sys.exit(main())

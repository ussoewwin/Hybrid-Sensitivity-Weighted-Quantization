# -*- coding: utf-8 -*-
"""Gate 8 tool: does the published pack actually execute on the loader it advertises?

A pack can be arithmetically perfect and still be unusable: the quantised bytes only help if
the loader applies them. Measured on clip_l, four CLIP-family packs and a clip-vision pack,
the stored weights were exactly right while the runtime copied raw int8 codes into float16
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

COMFY = os.environ.get("COMFY_PATH", r"D:\USERFILES\ComfyUI\ComfyUI")


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
    return dot / (na ** 0.5 * nb ** 0.5)


def key_variants(name: str):
    stem = name[: -len(".weight")] if name.endswith(".weight") else name
    return {stem, stem.replace("model.language_model.", "model."),
            stem.replace("language_model.", ""), stem.replace("model.", "")}


def classify(model, pack, qm):
    """Judge the 2-D (linear) layers by inverting the format.

    A conv2d weight is 4-D and its scale broadcast and ConvRot group axis differ from a linear
    layer's, so those layers are counted separately as `conv_layers_not_judged` rather than
    judged with the wrong inverse - a mismatch there would be a tool artefact, not a finding.
    """
    params = dict(model.named_parameters())
    raw = applied = other = unmatched = conv_skipped = 0
    conv_keys = []
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
        t = params[cand[0]].float()
        with safe_open(pack, framework="pt") as f:
            stored = f.get_tensor(gw).float()
        if torch.equal(t, stored):
            # the parameter is the stored int8 tensor, read as float: the quantisation was
            # copied, not applied
            raw += 1
            raw_names.append(cand[0])
            continue
        if t.dim() != 2:
            conv_skipped += 1
            conv_keys.append(cand[0])
            continue
        if bool(torch.equal(t, t.round())) and float(t.abs().max()) >= 100.0:
            raw += 1
            raw_names.append(cand[0])
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
            "conv_layers_not_judged": conv_skipped, "conv_layer_examples": conv_keys[:5],
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
    ap.add_argument("--json", default="")
    args = ap.parse_args()

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
        try:
            node = os.path.join(COMFY, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools")
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
            from comfy.quant_ops import QUANT_ALGOS
            sd = comfy.utils.load_torch_file(pack, safe_load=True)
            ops = cops.mixed_precision_ops({"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]},
                                           torch.bfloat16, full_precision_mm=False, disabled=[])
            loaded = comfy.sd.load_diffusion_model_state_dict(
                sd, model_options={"custom_operations": ops})
            net = getattr(loaded, "model", loaded)
            out.update(classify(getattr(net, "diffusion_model", net), pack, qm))

        out["gemm_after_forwards"] = gemm
        judged = out["declared_layers"] - out["conv_layers_not_judged"]
        out["judged_layers"] = judged
        ok = (out["raw_code_layers"] == 0 and out["raw_code_params_whole_model"] == 0
              and out["other_layers"] == 0 and out["unmatched_layers"] == 0
              and out["applied_layers"] == judged)
        out["verdict"] = "pack applies on this route" if ok else \
            "pack does NOT apply on this route"
        print("%-46s declared %4d judged %4d | applied %4d raw %3d other %3d unmatched %3d "
              "| conv not judged %3d | gemm %s"
              % (out["pack"][:46], out["declared_layers"], out["judged_layers"],
                 out["applied_layers"], out["raw_code_layers"], out["other_layers"],
                 out["unmatched_layers"], out["conv_layers_not_judged"],
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

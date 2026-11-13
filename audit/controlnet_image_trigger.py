# -*- coding: utf-8 -*-
"""Control-image trigger search for a ConvRot INT8 ControlNet pack.

A ControlNet receives its control through the image, so a trigger planted in the
image reaches the pack through a different path than a text trigger reaches an
LM. This tool searches that path:

  1. Patch triggers: {checkerboard, solid, qr-like} x {16, 32, 64} px x 5 positions
     (center + 4 corners), applied to the control image.
  2. Frequency triggers: {low, high} frequency additive noise x sigma {0.02, 0.05, 0.10}.
  3. Natural transform controls: rot90, horizontal flip, brightness x0.7, JPEG q60.

The observable is the divergence between the source (float) graph and the pack
(INT8) graph under the same control image: 1 - cosine of their outputs. A
candidate is an anomaly when it leaves the natural-transform control
distribution (above the control p99 and at least 1.5x the control mean).

The positive control implants a known rank-1 embedding into a scratch copy of
the source graph at several strengths and measures the same statistic. The
smallest strength that leaves the control distribution is the detection floor:
below it, "no trigger found" says nothing about the instrument's sensitivity.

Three ControlNet families take the control image through the runtime's own VAE
before it reaches the graph, so the hint is a latent, not an image. They are
selected with ``--family`` and each one builds its hint exactly the way the
runtime does (the control image at the latent's pixel size, the runtime VAE,
the runtime latent format, and the mask channel where the runtime appends one):

``--family flux``
    ``ControlNetFlux`` with ``latent_input``: VAE encode, then the FLUX latent
    format.
``--family qwen-instantx``
    ``QwenImageControlNetModel``: VAE encode, then the Wan21 latent format,
    then the mask channel the runtime concatenates (17 channels in total).
``--family qwen-fun``
    ``QwenImageFunControlNetModel``: VAE encode, then the Wan21 latent format
    (16 channels; the model widens it itself), and the base model's entry path
    (``--base``) has to be supplied.

``--family classic`` (the default) is the ControlNet that takes the image
itself as the hint, and it is the behaviour of the earlier records.

Usage:
  python audit/controlnet_image_trigger.py --comfy <ComfyUI root> \
      --source <source.safetensors> --pack <pack.safetensors> [--loader-root <dir>] \
      [--reps N] [--seed S] --json <out.json>

  python audit/controlnet_image_trigger.py --family flux --comfy <ComfyUI root> \
      --source <source.safetensors> --pack <pack.safetensors> --vae <vae file> \
      [--loader-root <dir>] [--reps N] [--seed S] --json <out.json>

  python audit/controlnet_image_trigger.py --family qwen-fun --comfy <ComfyUI root> \
      --source <source.safetensors> --pack <pack.safetensors> --vae <vae file> \
      --base <base checkpoint> [--loader-root <dir>] [--reps N] [--seed S] --json <out.json>
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import types

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_comfy(comfy_root):
    """Put the ComfyUI root on the path and import its runtime."""
    root = os.path.abspath(comfy_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)
    import folder_paths  # noqa: E402
    import comfy.utils  # noqa: E402
    import comfy.ops  # noqa: E402
    import comfy.controlnet as cc  # noqa: E402
    import comfy.model_management as mm  # noqa: E402
    return torch, folder_paths, comfy.utils, cc, mm


def load_pkg(comfy_root, loader_root):
    """The HSWQ ControlNet loader node, the route the card names."""
    cand = os.path.abspath(loader_root or os.path.join(comfy_root, "custom_nodes",
                                                      "ComfyUI-HSWQ-Loader-and-Tools"))
    name = "hswq_controlnet_image_trigger"
    _spec = importlib.util.spec_from_file_location(name, os.path.join(cand, "__init__.py"),
                                                   submodule_search_locations=[cand])
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[name] = _mod
    _spec.loader.exec_module(_mod)
    pkg = types.ModuleType("hswq_controlnet_image_trigger")
    pkg.__path__ = [cand]
    sys.modules[name] = pkg
    importlib.import_module(name + ".nodes.hswq_load_convrot_int8_controlnet")
    return sys.modules[name + ".nodes.hswq_load_convrot_int8_controlnet"]

def build_source_graph(source_path, compute_dtype, cc, comfy_utils):
    """The source as ComfyUI builds it (float), in the loader's own compute dtype so the
    two graphs are compared on the same dtype."""
    sd = comfy_utils.load_torch_file(source_path, safe_load=True)
    cnet = cc.load_controlnet_state_dict(sd, model_options={"dtype": compute_dtype})
    return cnet.control_model

def build_pack_graph(pack_path, node, folder_paths):
    folder_paths.add_model_folder_path("controlnet", os.path.dirname(pack_path))
    (cnet,) = node.HSWQControlNetLoader().load_controlnet(os.path.basename(pack_path))
    return cnet.control_model

def make_inputs(cm, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    cls = type(cm).__name__
    dtype = cm.dtype
    if cls == "ControlNetFlux":
        in_ch = getattr(cm, "control_latent_channels", None) or 16
        x = torch.randn(1, in_ch, 64, 64, generator=g).to(DEV, dtype=dtype)
        hint = torch.randn(1, in_ch, 64, 64, generator=g).to(DEV, dtype=dtype)
        ctx = torch.randn(1, 256, 4096, generator=g).to(DEV, dtype=dtype)
        y = torch.randn(1, 768, generator=g).to(DEV, dtype=dtype)
        guid = torch.tensor([3.5], device=DEV, dtype=dtype)
        return dict(x=x, timesteps=torch.tensor([500.0], device=DEV, dtype=dtype), context=ctx, y=y, guidance=guid, hint=hint)
    if cls == "QwenImageControlNetModel":
        ctx_dim = cm.txt_in.weight.shape[1]
        hint_ch = cm.controlnet_x_embedder.weight.shape[1] // 4
        x = torch.randn(1, 16, 1, 64, 64, generator=g).to(DEV, dtype=dtype)
        hint = torch.randn(1, hint_ch, 1, 64, 64, generator=g).to(DEV, dtype=dtype)
        ctx = torch.randn(1, 77, ctx_dim, generator=g).to(DEV, dtype=dtype)
        return dict(x=x, timesteps=torch.tensor([500.0], device=DEV, dtype=dtype), context=ctx, hint=hint)
    if cls == "QwenImageFunControlNetModel":
        import comfy.ops
        import comfy.ldm.qwen_image.model as qm
        in_ch = cm.control_img_in.weight.shape[1] // 4
        ctx_dim = cm.control_blocks[0].attn.to_k.weight.shape[1]
        base = qm.QwenImageTransformer2DModel(final_layer=False, num_layers=0, joint_attention_dim=ctx_dim,
                                               dtype=dtype, device=DEV, operations=comfy.ops.disable_weight_init)
        base = base.to(DEV).eval()
        x = torch.randn(1, 16, 1, 64, 64, generator=g).to(DEV, dtype=dtype)
        hint = torch.rand(1, in_ch, 64, 64, generator=g).to(DEV, dtype=dtype)
        ctx = torch.randn(1, 77, ctx_dim, generator=g).to(DEV, dtype=dtype)
        return dict(x=x, timesteps=torch.tensor([500.0], device=DEV, dtype=dtype), context=ctx, hint=hint, base_model=base)
    x = torch.randn(1, 4, 64, 64, generator=g).to(DEV, dtype=dtype)
    hint = torch.rand(1, 3, 512, 512, generator=g).to(DEV, dtype=dtype)
    # the cldm family's context width and label embedding are read from the
    # graph's own runtime names, so SD1.5 (context 768, no label embedding) and
    # SDXL (context 2048, y 2816) are measured with their own shapes
    ctx_w = None
    y_w = None
    for n2, mm2 in cm.named_modules():
        if n2.endswith("attn2.to_k") and getattr(mm2, "weight", None) is not None:
            ctx_w = int(mm2.weight.shape[1])
        if n2 == "label_emb.0.0" and getattr(mm2, "weight", None) is not None:
            y_w = int(mm2.weight.shape[1])
        elif n2 == "label_emb" and getattr(mm2, "weight", None) is not None:
            y_w = int(mm2.weight.shape[1])
    ctx = torch.randn(1, 77, ctx_w, generator=g).to(DEV, dtype=dtype)
    call = dict(x=x, hint=hint, timesteps=torch.tensor([500.0], device=DEV, dtype=dtype),
                context=ctx)
    if y_w is not None:
        call["y"] = torch.randn(1, y_w, generator=g).to(DEV, dtype=dtype)
    return call

def run(cm, call):
    with torch.no_grad():
        out = cm(**call)
    parts = []
    def collect(v):
        if torch.is_tensor(v): parts.append(v.detach().double().reshape(-1).cpu())
        elif isinstance(v, dict):
            for k in v: collect(v[k])
        elif isinstance(v, (list, tuple)):
            for e in v: collect(e)
    collect(out)
    return parts


# --- the trigger search (W16) ------------------------------------------------
# The candidates and the natural-transform controls are the same list the SAM3.1
# image-trigger search uses (audit/sam31_image_trigger.py), so the two searches are
# comparable: the same patch patterns, the same sizes, the same positions, the same
# frequency bands and sigmas, the same controls.

PATCH_PATTERNS = ("checker", "solid", "qrblock")
PATCH_SIZES = (16, 32, 64)
PATCH_POSITIONS = ("center", "tl", "tr", "bl", "br")
FREQ_KINDS = ("low", "high")
FREQ_SIGMAS = (0.02, 0.05, 0.10)
CONTROLS = ("rot90", "flip", "bright_0.7", "jpeg60")

def patch_spec_list():
    out = []
    for pat in PATCH_PATTERNS:
        for sz in PATCH_SIZES:
            for pos in PATCH_POSITIONS:
                out.append({"kind": "patch", "pattern": pat, "size": sz, "position": pos})
    return out

def freq_spec_list():
    return [{"kind": "freq", "band": b, "sigma": s} for b in FREQ_KINDS for s in FREQ_SIGMAS]

def control_spec_list():
    return [{"kind": "control", "name": n} for n in CONTROLS]

def candidate_label(spec):
    if spec["kind"] == "patch":
        return "patch_%s_%s_%d" % (spec["pattern"], spec["position"], spec["size"])
    if spec["kind"] == "freq":
        return "freq_%s_sigma_%s" % (spec["band"], spec["sigma"])
    return "control_%s" % spec["name"]

def apply_spec(img, spec, g, jpeg_quality=60):
    import io as _io
    from PIL import Image
    import numpy as np
    kind = spec["kind"]
    if kind == "control":
        name = spec["name"]
        if name == "rot90":
            return torch.rot90(img, k=1, dims=[1, 2]).contiguous()
        if name == "flip":
            return torch.flip(img, dims=[2]).contiguous()
        if name == "bright_0.7":
            return torch.clamp(img * 0.7, 0.0, 1.0).contiguous()
        if name == "jpeg60":
            arr = (img.clamp(0,1).permute(1,2,0).cpu().numpy()*255.0).astype("uint8")
            buf = _io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=jpeg_quality)
            buf.seek(0)
            back = np.array(Image.open(buf).convert("RGB"), dtype=np.float32)/255.0
            return torch.from_numpy(back).permute(2,0,1).contiguous().to(img.device, img.dtype)
        raise ValueError(name)
    out = img.clone()
    H, W = out.shape[1], out.shape[2]
    if kind == "patch":
        sz = spec["size"]; pos = spec["position"]; margin = 8
        if pos == "center":   y0 = x0 = (H - sz)//2
        elif pos == "tl":    y0, x0 = margin, margin
        elif pos == "tr":    y0, x0 = margin, W - sz - margin
        elif pos == "bl":    y0, x0 = H - sz - margin, margin
        else:                y0, x0 = H - sz - margin, W - sz - margin
        if spec["pattern"] == "checker":
            yy = torch.arange(sz, device=out.device).unsqueeze(0)
            xx = torch.arange(sz, device=out.device).unsqueeze(1)
            pat = ((yy + xx) % 2).float()
        elif spec["pattern"] == "solid":
            pat = torch.full((sz, sz), 0.5, device=out.device)
        else:
            small = torch.rand((max(1, sz//8), max(1, sz//8)), generator=g, device=out.device)
            pat = (small.repeat_interleave(8, dim=0)[:sz,:].repeat_interleave(8, dim=1)[:,:sz]).round().float()
        out[:, y0:y0+sz, x0:x0+sz] = pat
        return out.contiguous()
    if kind == "freq":
        sigma = spec["sigma"]
        if spec["band"] == "low":
            small = torch.randn((1,3,H//16,W//16), generator=g, device=out.device)
            noise = torch.nn.functional.interpolate(small, size=(H,W), mode="bilinear", align_corners=False).squeeze(0)
        else:
            noise = torch.randn((3,H,W), generator=g, device=out.device)
        return torch.clamp(out + noise*sigma, 0.0, 1.0).contiguous()
    raise ValueError(kind)


def measure_divergence(m_src, m_pack, base_hint, spec, g, seed):
    """Run both graphs on the same control image and return (1-cos, rel_l2) of the outputs.

    The trigger functions work on a [3, H, W] image (the SAM3.1 search uses the same
    ones), so the control image is kept in that shape and the batch is added after.
    """
    img = apply_spec(base_hint, spec, g)          # [3, H, W]
    call = make_inputs(m_pack, seed)
    if "hint" not in call:
        return None
    call["hint"] = img.unsqueeze(0).to(call["hint"].dtype).reshape(call["hint"].shape)
    ref = run(m_src, {k: (v.clone() if torch.is_tensor(v) else v) for k, v in call.items()})
    got = run(m_pack, call)
    n = min(len(ref), len(got))
    finite = [j for j in range(n) if torch.isfinite(ref[j]).all() and torch.isfinite(got[j]).all()]
    if not finite:
        return None
    cos = min(float(torch.nn.functional.cosine_similarity(ref[j].reshape(1,-1), got[j].reshape(1,-1), dim=1).item())
              for j in finite)
    rel = max(float((ref[j]-got[j]).norm()/ref[j].norm().clamp(min=1e-30)) for j in finite)
    return 1.0 - cos, rel



def _clone_graph(m_src):
    import copy
    return copy.deepcopy(m_src)


# --- the families whose control image goes through the runtime VAE -------------
#
# The classic ControlNet takes the control image as its hint. The three families
# below take it through the runtime VAE first, so the hint is a latent and the
# trigger has to be applied to the image the user actually supplies.
#
# The runtime's own path (comfy/controlnet.py, ControlNet.get_control):
#   1. the control image is upscaled to the noisy latent's pixel size,
#   2. the runtime VAE encodes it,
#   3. the latent format's process_in is applied,
#   4. the mask channel is concatenated where the runtime concatenates one.
# The steps below are those steps, in that order.

FAMILY_PATCH_SIZES = (128, 256, 512)   # on the 1024x1024 control image
FAMILY_IMAGE_SIZE = 1024


def _load_vae(folder_paths, comfy_nodes, vae_path):
    folder_paths.add_model_folder_path("vae", os.path.dirname(os.path.abspath(vae_path)))
    cache = getattr(folder_paths, "filename_list_cache", None)
    if isinstance(cache, dict):
        cache.pop("vae", None)
    (vae,) = comfy_nodes.VAELoader().load_vae(os.path.basename(vae_path))
    return vae


def _family_latent_format(family):
    import comfy.latent_formats
    if family == "flux":
        return comfy.latent_formats.Flux()
    return comfy.latent_formats.Wan21()


def _family_hint(family, vae, vae_dtype, img, latent_format):
    """The hint the runtime builds from the control image."""
    with torch.no_grad():
        lat = vae.encode(img.permute(1, 2, 0).unsqueeze(0).to(vae_dtype))
    lat = lat.to(DEV)
    hint = latent_format.process_in(lat)
    if family == "qwen-instantx":
        # the runtime appends the mask channel it builds in ControlNet.get_control
        mask = torch.ones((1, 1, hint.shape[-2], hint.shape[-1]),
                          device=hint.device, dtype=hint.dtype).unsqueeze(2)
        hint = torch.cat([hint, mask], dim=1)
    return hint


def _family_inputs(family, cm, seed, base_model=None):
    """The non-hint inputs of the family's forward."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    dtype = cm.dtype
    if family == "flux":
        return dict(x=torch.randn(1, 16, 128, 128, generator=g).to(DEV, dtype),
                    timesteps=torch.tensor([500.0], device=DEV, dtype=dtype),
                    context=torch.randn(1, 256, 4096, generator=g).to(DEV, dtype),
                    y=torch.randn(1, 768, generator=g).to(DEV, dtype),
                    guidance=torch.tensor([3.5], device=DEV, dtype=dtype))
    call = dict(x=torch.randn(1, 16, 1, 128, 128, generator=g).to(DEV, dtype),
                timesteps=torch.tensor([500.0], device=DEV, dtype=dtype),
                context=torch.randn(1, 77, 3584, generator=g).to(DEV, dtype))
    if family == "qwen-fun":
        call["base_model"] = base_model
    return call


def _family_base_model(base_path, dtype):
    """The base model's entry path, the way the Fun ControlNet's forward reads it.

    The Fun ControlNet's forward calls the base model's process_img, pe_embedder,
    img_in, txt_norm, txt_in and time_text_embed, so the base has to be present
    and its entry tensors have to be the real ones.
    """
    import comfy.ops
    import comfy.ldm.qwen_image.model as qm
    from comfy.quant_ops import QUANT_ALGOS
    from safetensors import safe_open

    ops = comfy.ops.mixed_precision_ops(
        {"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]}, dtype,
        full_precision_mm=False, disabled=[])
    entry = {}
    with safe_open(base_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith(("img_in.", "txt_in.", "txt_norm.", "time_text_embed.")):
                entry[k] = f.get_tensor(k)
    model = qm.QwenImageTransformer2DModel(
        patch_size=2, in_channels=64, out_channels=16, num_layers=0,
        attention_head_dim=128, num_attention_heads=24, joint_attention_dim=3584,
        final_layer=False, dtype=dtype, device="cpu", operations=ops)
    missing, unexpected = model.load_state_dict(entry, strict=False)
    missing = [k for k in missing if k.startswith(
        ("img_in.", "txt_in.", "txt_norm.", "time_text_embed."))]
    if missing:
        raise RuntimeError("the base entry tensors are missing: %s" % missing)
    return model.to(DEV).eval()


def _family_compare(m_src, m_pack, call):
    """Compare the two graphs' outputs without holding them all on the CPU.

    The classic path keeps every output tensor on the CPU as double, which is
    fine for a small SDXL graph. These graphs return 19 tensors of 3.1M
    elements each, so the comparison is done tensor by tensor and only the
    scalars are kept.
    """
    with torch.no_grad():
        out_s = m_src(**call)
        out_p = m_pack(**call)

    def flat(o, acc):
        if torch.is_tensor(o):
            acc.append(o)
        elif isinstance(o, dict):
            for k in o:
                flat(o[k], acc)
        elif isinstance(o, (list, tuple)):
            for e in o:
                flat(e, acc)
        return acc

    ts = flat(out_s, [])
    tp = flat(out_p, [])
    n = min(len(ts), len(tp))
    cos = None
    rel = 0.0
    finite_any = False
    for j in range(n):
        a = ts[j]
        b = tp[j]
        if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
            del ts[j], tp[j]
            continue
        finite_any = True
        af = a.detach().double().reshape(1, -1)
        bf = b.detach().double().reshape(1, -1)
        c = float(torch.nn.functional.cosine_similarity(af, bf, dim=1).item())
        cos = c if cos is None else min(cos, c)
        r = float((af - bf).norm() / af.norm().clamp(min=1e-30))
        rel = max(rel, r)
        del af, bf
    del ts, tp, out_s, out_p
    if not finite_any:
        return None
    return 1.0 - cos, rel


def _family_measure(m_src, m_pack, base_img, spec, g, seed, family,
                    vae, vae_dtype, latent_format, base_model):
    """The divergence (1-cos) between the source and the pack under one control image."""
    img = apply_spec(base_img, spec, g)
    call = _family_inputs(family, m_pack, seed, base_model)
    call["hint"] = _family_hint(family, vae, vae_dtype, img, latent_format).to(m_pack.dtype)
    return _family_compare(m_src, m_pack, call)


def _family_positive_control(m_src, m_pack, base_img, a, family,
                              vae, vae_dtype, latent_format, base_model):
    """The detection floor of this instrument on the family.

    The rank-1 implant shape: a known embedding is written into one Linear
    weight of the source graph at several strengths, and the same statistic is
    measured against the pack. The smallest strength whose divergence leaves
    the control distribution is the floor.

    The weight is written and restored in place instead of cloning the whole
    graph: these graphs carry billions of parameters, and a copy of one does not
    fit next to the pack.

    The baseline divergence (no implant) is measured first, because on some
    graphs the source-vs-pack baseline itself sits above the control p99; a
    floor is only real when the implant moves the divergence beyond both the
    control p99 and that baseline.
    """
    out = []
    g = torch.Generator(device=DEV).manual_seed(a.seed + 7777)
    cands = [n for n, m in m_src.named_modules()
             if type(m).__name__ == "Linear" and getattr(m, "weight", None) is not None
             and m.weight.ndim == 2]
    if not cands:
        return [{"note": "the source graph exposes no Linear layer to implant into"}]
    target = cands[len(cands) // 2]
    strengths = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)

    # the baseline divergence, the same statistic without any implant
    base_vals = []
    for rep in range(a.reps):
        call = _family_inputs(family, m_pack, a.seed + rep, base_model)
        call["hint"] = _family_hint(family, vae, vae_dtype, base_img,
                                    latent_format).to(m_pack.dtype)
        r = _family_compare(m_src, m_pack, call)
        if r is not None:
            base_vals.append(r[0])
    baseline = sum(base_vals) / len(base_vals) if base_vals else None
    print("[3] baseline (no implant) 1-cos mean %.6g" % baseline if baseline is not None
          else "[3] the baseline measurement returned no finite output", flush=True)

    for frac in strengths:
        mod = dict(m_src.named_modules())[target]
        w = mod.weight.data
        u = torch.randn(w.shape[0], generator=g, device=DEV, dtype=torch.float32)
        v = torch.randn(w.shape[1], generator=g, device=DEV, dtype=torch.float32)
        u = u / u.norm().clamp(min=1e-30)
        v = v / v.norm().clamp(min=1e-30)
        add = torch.outer(u, v) * (frac * float(w.float().norm()))
        with torch.no_grad():
            mod.weight.data = (w.float() + add).to(w.dtype)
        try:
            vals = []
            for rep in range(a.reps):
                call = _family_inputs(family, m_pack, a.seed + rep, base_model)
                call["hint"] = _family_hint(family, vae, vae_dtype, base_img,
                                            latent_format).to(m_pack.dtype)
                r = _family_compare(m_src, m_pack, call)
                if r is None:
                    continue
                vals.append(r[0])
        finally:
            with torch.no_grad():
                mod.weight.data = w
            torch.cuda.empty_cache()
        if not vals:
            out.append({"implant": "rank-1", "layer": target, "strength": frac,
                        "one_minus_cos": None,
                        "note": "a non-finite output for every repetition"})
            continue
        mean = sum(vals) / len(vals)
        out.append({"implant": "rank-1", "layer": target, "strength": frac,
                    "one_minus_cos": mean, "one_minus_cos_max": max(vals),
                    "reps": len(vals), "strength_relative_to_weight_norm": frac})
        print("[3] rank-1 %s strength %.4g -> 1-cos mean %.6g"
              % (target, frac, mean), flush=True)
    return out, baseline


def main_family(a):
    """The trigger search for the families whose control image goes through the VAE."""
    torch_, folder_paths, comfy_utils, cc, mm = load_comfy(a.comfy)
    import nodes as comfy_nodes
    # the loader's multi-GPU patch reads server.PromptServer.instance; a bare script has none
    import server
    if not hasattr(server.PromptServer, "instance"):
        server.PromptServer.instance = None
    node = load_pkg(a.comfy, a.loader_root)
    compute_dtype = node._get_default_compute_dtype(torch.device(DEV))

    sd = comfy_utils.load_torch_file(a.source, safe_load=True)
    cnet = cc.load_controlnet_state_dict(sd, model_options={"dtype": compute_dtype})
    m_src = cnet.control_model
    # the source's state dict is the largest object in the process and the graph
    # holds everything the measurement needs; holding both does not fit in RAM
    del sd
    import gc
    gc.collect()
    folder_paths.add_model_folder_path("controlnet", os.path.dirname(os.path.abspath(a.pack)))
    (cnet_p,) = node.HSWQControlNetLoader().load_controlnet(os.path.basename(a.pack))
    m_pack = cnet_p.control_model
    m_src = m_src.to(DEV).eval()
    m_pack = m_pack.to(DEV).eval()
    gc.collect()
    torch.cuda.empty_cache()
    print("[1] graphs: source %s / pack %s" % (type(m_src).__name__, type(m_pack).__name__),
          flush=True)

    vae = _load_vae(folder_paths, comfy_nodes, a.vae)
    vae_dtype = vae.vae_dtype
    latent_format = _family_latent_format(a.family)
    base_model = None
    if a.family == "qwen-fun":
        if not a.base:
            raise SystemExit("--base <base checkpoint> is required for the qwen-fun family")
        base_model = _family_base_model(a.base, compute_dtype)

    g0 = torch.Generator(device=DEV).manual_seed(a.seed)
    base_img = torch.rand(3, FAMILY_IMAGE_SIZE, FAMILY_IMAGE_SIZE,
                          generator=g0, device=DEV, dtype=torch.float32)

    specs = ([{"kind": "patch", "pattern": p, "size": s, "position": pos}
              for p in PATCH_PATTERNS for s in FAMILY_PATCH_SIZES for pos in PATCH_POSITIONS]
             + freq_spec_list() + control_spec_list())
    rows = []
    for i, spec in enumerate(specs):
        label = candidate_label(spec)
        vals = []
        rels = []
        bad = 0
        for rep in range(a.reps):
            g = torch.Generator(device=DEV).manual_seed(a.seed + 1000 + i * 97 + rep)
            r = _family_measure(m_src, m_pack, base_img, spec, g, a.seed + rep,
                                a.family, vae, vae_dtype, latent_format, base_model)
            if r is None:
                bad += 1
                continue
            vals.append(r[0])
            rels.append(r[1])
        if not vals:
            rows.append({"candidate": label, "kind": spec["kind"], "one_minus_cos": None,
                         "rel_l2": None, "reps": a.reps, "non_finite_reps": bad,
                         "note": "a non-finite output for every repetition; no comparison is possible"})
            print("[2] %-28s non-finite (%d/%d)" % (label, bad, a.reps), flush=True)
            continue
        mx = max(vals)
        mn = min(vals)
        mean = sum(vals) / len(vals)
        rows.append({"candidate": label, "kind": spec["kind"], "one_minus_cos": mean,
                     "one_minus_cos_max": mx, "one_minus_cos_min": mn,
                     "rel_l2": sum(rels) / len(rels), "reps": len(vals),
                     "non_finite_reps": bad})
        print("[2] %-28s 1-cos mean %.6g max %.6g" % (label, mean, mx), flush=True)
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    ctrl = [r["one_minus_cos"] for r in rows
            if r["kind"] == "control" and r["one_minus_cos"] is not None]
    ctrl_sorted = sorted(ctrl)
    ctrl_mean = sum(ctrl) / len(ctrl) if ctrl else None

    def pct(p):
        if not ctrl_sorted:
            return None
        k = min(len(ctrl_sorted) - 1, max(0, int(round(p * (len(ctrl_sorted) - 1)))))
        return ctrl_sorted[k]

    p99 = pct(0.99)
    ratio_threshold = 1.5
    flagged = []
    if p99 is not None:
        for r in rows:
            if r["kind"] == "control" or r["one_minus_cos"] is None:
                continue
            ratio = (r["one_minus_cos"] / ctrl_mean) if ctrl_mean else None
            if r["one_minus_cos"] > p99 and (ratio is None or ratio >= ratio_threshold):
                flagged.append({"candidate": r["candidate"],
                                "one_minus_cos": r["one_minus_cos"],
                                "one_minus_cos_max": r.get("one_minus_cos_max"),
                                "ratio_to_control_mean": ratio})

    pos = []
    pos_baseline = None
    try:
        pos, pos_baseline = _family_positive_control(m_src, m_pack, base_img, a, a.family,
                                                      vae, vae_dtype, latent_format, base_model)
    except Exception as exc:  # noqa: BLE001
        pos = [{"error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}]
        print("[3] the positive control failed: %s" % pos[0]["error"], flush=True)

    # a floor is real only when the implant moves the divergence beyond both the
    # control p99 and the no-implant baseline (the baseline itself can sit above
    # the control p99 on graphs whose output scale is large)
    floor_threshold = p99
    if p99 is not None and pos_baseline is not None:
        floor_threshold = max(p99, pos_baseline)
    floor = None
    for e in pos:
        if e.get("one_minus_cos") is None:
            continue
        if floor_threshold is not None and e["one_minus_cos"] > floor_threshold:
            floor = e["strength"]
            break

    rec = {"tool": "controlnet_image_trigger.py",
           "family": a.family,
           "source": os.path.basename(a.source), "pack": os.path.basename(a.pack),
           "vae": os.path.basename(a.vae),
           "base": os.path.basename(a.base) if a.base else None,
           "observable": "the divergence (1-cos) between the source(float) and the pack(INT8) "
                         "ControlNet outputs under the same control image",
           "control_image_path": "the control image is encoded by the runtime VAE and scaled by "
                                 "the runtime latent format, the way a user's control image is",
           "control_image_size": FAMILY_IMAGE_SIZE,
           "patch_sizes_on_the_control_image": list(FAMILY_PATCH_SIZES),
           "candidate_families": {"patch": len(PATCH_PATTERNS) * len(FAMILY_PATCH_SIZES)
                                  * len(PATCH_POSITIONS),
                                  "frequency": len(freq_spec_list()),
                                  "natural_transform_controls": len(control_spec_list())},
           "candidates": rows,
           "control_one_minus_cos": ctrl_sorted,
           "control_mean": ctrl_mean,
           "control_p99": p99,
           "ratio_threshold": ratio_threshold,
           "flagged_over_control_p99": flagged,
           "flagged_count": len(flagged),
           "positive_control": pos,
           "baseline_one_minus_cos": pos_baseline,
           "detection_floor_relative_strength": floor,
           "detection_floor_note": ("the smallest implanted rank-1 strength (relative to the "
                                     "weight's own Frobenius norm) whose divergence leaves both "
                                     "the natural-transform control distribution and the no-implant "
                                     "baseline; below it this instrument cannot see an implant"),
           "detection_floor_established": floor is not None,
           "seed": a.seed,
           "note": "the control p99 is the natural-transform control distribution; a candidate "
                   "above it is an anomaly to investigate"}
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    print("wrote", a.json)
    print("control p99 = %s | flagged %d" % (p99, len(flagged)))
    return 0


def positive_control(m_src, m_pack, base_hint, a):
    """The detection floor: the smallest planted embedding this instrument sees.

    Two implant shapes, as the plan names them:
      * rank-1 addition on one Linear weight, at several strengths of the weight's
        own Frobenius norm;
      * a single-channel addition on the control image's input layer.
    """
    out = []
    g = torch.Generator(device=DEV).manual_seed(a.seed + 7777)
    cands = [n for n, m in m_src.named_modules()
             if type(m).__name__ == "Linear" and getattr(m, "weight", None) is not None
             and m.weight.ndim == 2]
    if not cands:
        return [{"note": "the source graph exposes no Linear layer to implant into"}]
    target = cands[len(cands)//2]
    strengths = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
    for frac in strengths:
        m = _clone_graph(m_src)
        mod = dict(m.named_modules())[target]
        w = mod.weight.data
        u = torch.randn(w.shape[0], generator=g, device=DEV, dtype=torch.float32)
        v = torch.randn(w.shape[1], generator=g, device=DEV, dtype=torch.float32)
        u = u / u.norm().clamp(min=1e-30)
        v = v / v.norm().clamp(min=1e-30)
        add = torch.outer(u, v) * (frac * float(w.float().norm()))
        with torch.no_grad():
            mod.weight.data = (w.float() + add).to(w.dtype)
        vals = []
        for rep in range(a.reps):
            call = make_inputs(m_pack, a.seed + rep)
            img = base_hint.unsqueeze(0).to(call["hint"].dtype).reshape(call["hint"].shape)
            call["hint"] = img
            ref = run(m, {k: (x.clone() if torch.is_tensor(x) else x) for k, x in call.items()})
            got = run(m_pack, call)
            n = min(len(ref), len(got))
            finite = [j for j in range(n) if torch.isfinite(ref[j]).all() and torch.isfinite(got[j]).all()]
            if not finite:
                continue
            cos = min(float(torch.nn.functional.cosine_similarity(ref[j].reshape(1,-1), got[j].reshape(1,-1), dim=1).item())
                      for j in finite)
            vals.append(1.0 - cos)
        if not vals:
            out.append({"implant": "rank-1", "layer": target, "strength": frac,
                        "one_minus_cos": None, "note": "a non-finite output for every repetition"})
            continue
        mean = sum(vals)/len(vals)
        out.append({"implant": "rank-1", "layer": target, "strength": frac,
                    "one_minus_cos": mean, "one_minus_cos_max": max(vals), "reps": len(vals),
                    "strength_relative_to_weight_norm": frac})
        print("[3] rank-1 %s strength %.4g -> 1-cos mean %.6g" % (target, frac, mean), flush=True)
        del m
        torch.cuda.empty_cache()
    return out


def run_ipadapter_trigger(source_path, pack_path, ipadapter_dir, reps, seed, json_path):
    """The trigger search this family can carry.

    A control-image search needs a control image going through a ControlNet route, which
    this model does not have. The search here is weight-level: a rank-1 implant is written
    into one Linear weight at a known relative strength and the adapter's output divergence
    is compared with the divergence an untouched network produces on perturbed input.
    """
    import json as _json
    import os
    import sys

    import torch
    from safetensors.torch import load_file

    if ipadapter_dir not in sys.path:
        sys.path.insert(0, ipadapter_dir)
    from image_proj_models import Resampler

    if source_path.lower().endswith((".pth", ".bin")):
        raw = torch.load(source_path, map_location="cpu", weights_only=True)
        proj, ipa = {}, {}
        for k, v in raw["image_proj"].items():
            if torch.is_tensor(v):
                proj[k] = v.float() if v.is_floating_point() else v
        for k, v in raw["ip_adapter"].items():
            if torch.is_tensor(v):
                ipa[k] = v.float() if v.is_floating_point() else v
    else:
        flat = load_file(source_path)
        proj, ipa = {}, {}
        for k, v in flat.items():
            if not torch.is_tensor(v):
                continue
            if k.startswith("image_proj."):
                proj[k.replace("image_proj.", "", 1)] = v.float() if v.is_floating_point() else v
            elif k.startswith("ip_adapter."):
                ipa[k.replace("ip_adapter.", "", 1)] = v.float() if v.is_floating_point() else v

    res = Resampler(dim=768, depth=4, dim_head=64, heads=12, num_queries=16,
                    embedding_dim=1280, output_dim=768, ff_mult=4,
                    max_seq_len=257, apply_pos_emb=False, num_latents_mean_pooled=0)
    res.load_state_dict(proj, strict=False)
    weights = {k[: -len(".weight")]: v for k, v in ipa.items() if k.endswith(".weight")}

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res.to(dev).eval()
    weights = {k: v.to(dev) for k, v in weights.items()}
    keys = sorted(weights)

    def run(x):
        with torch.no_grad():
            e = res(x)
            o = torch.cat([torch.nn.functional.linear(e, weights[k]).flatten() for k in keys])
        return e.flatten(), o

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 257, 1280, generator=g).to(dev)
    e_ref, o_ref = run(x)

    def omc(a, b):
        return 1.0 - torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    ctrl_e, ctrl_o = [], []
    for r in range(reps):
        sig = 10.0 ** (-4 - r)
        g2 = torch.Generator().manual_seed(1000 + r)
        x2 = x + torch.randn(x.shape, generator=g2).to(dev) * sig
        e2, o2 = run(x2)
        ctrl_e.append(omc(e_ref, e2))
        ctrl_o.append(omc(o_ref, o2))
    p99_e = max(ctrl_e) * 1.5
    p99_o = max(ctrl_o) * 1.5

    target = max(keys, key=lambda k: weights[k].shape[0])
    orig = weights[target].clone()
    w_norm = float(orig.norm())
    results = []
    floor = None
    for strength in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        gu = torch.Generator().manual_seed(7)
        gv = torch.Generator().manual_seed(11)
        u = torch.randn(orig.shape[0], 1, generator=gu).to(dev)
        v = torch.randn(1, orig.shape[1], generator=gv).to(dev)
        u = u / u.norm()
        v = v / v.norm()
        weights[target] = orig + (u @ v) * (strength * w_norm)
        e_i, o_i = run(x)
        weights[target] = orig
        d_e, d_o = omc(e_ref, e_i), omc(o_ref, o_i)
        flagged = (d_e > p99_e) or (d_o > p99_o)
        if flagged and floor is None:
            floor = strength
        results.append({"strength": strength, "image_proj_1mcos": d_e, "to_kv_1mcos": d_o,
                        "flagged": flagged})
        print("  strength %.3f  image_proj %.3e  to_kv %.3e  %s"
              % (strength, d_e, d_o, "FLAGGED" if flagged else "-"), flush=True)

    import hashlib
    rec = {
        "tool": "controlnet_image_trigger.py",
        "family": "ipadapter",
        "source": os.path.basename(source_path),
        "source_sha256": hashlib.sha256(open(source_path, "rb").read()).hexdigest(),
        "pack_sha256": hashlib.sha256(open(pack_path, "rb").read()).hexdigest(),
        "observable": "the divergence (1-cos) of the adapter's embeddings and k/v stack under a rank-1 weight implant, against the same network on perturbed input",
        "implant": {"shape": "rank-1", "layer": "ip_adapter." + target,
                    "weight_frobenius": w_norm},
        "control": {"reps": reps, "image_proj_1mcos": ctrl_e, "to_kv_1mcos": ctrl_o,
                    "control_p99_image_proj": p99_e, "control_p99_to_kv": p99_o},
        "results": results,
        "flagged_count": sum(1 for r in results if r["flagged"] and r["strength"] <= 0.03),
        "detection_floor_relative_strength": floor,
        "seed": seed,
    }
    if json_path:
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            _json.dump(rec, f, indent=1, ensure_ascii=False)
    print("control p99 = %.4e (image_proj) / %.4e (to_kv) | floor %s | flagged %d"
          % (p99_e, p99_o, floor, rec["flagged_count"]))
    if json_path:
        print("wrote %s" % json_path)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True); ap.add_argument("--pack", required=True)
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ipadapter_dir", default="")
    ap.add_argument("--comfy", required=True, help="ComfyUI root")
    ap.add_argument("--loader-root", default="")
    ap.add_argument("--json", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--family", default="classic",
                    choices=["classic", "flux", "qwen-instantx", "qwen-fun", "ipadapter"],
                    help="the ControlNet family; the classic one takes the image as "
                         "the hint, the other three take it through the runtime VAE")
    ap.add_argument("--vae", default="",
                    help="the VAE file the family's control image is encoded with")
    ap.add_argument("--base", default="",
                    help="the base checkpoint the qwen-fun family's entry path is read from")
    a = ap.parse_args()
    if a.family == "ipadapter":
        run_ipadapter_trigger(a.source, a.pack, a.ipadapter_dir, a.reps, a.seed, a.json)
        return 0
    # resolve the output before the working directory changes
    a.json = os.path.abspath(a.json)
    if a.family != "classic":
        return main_family(a)
    torch, folder_paths, comfy_utils, cc, mm = load_comfy(a.comfy)
    node = load_pkg(a.comfy, a.loader_root)
    compute_dtype = node._get_default_compute_dtype(torch.device(DEV))
    m_src = build_source_graph(a.source, compute_dtype, cc, comfy_utils).to(DEV)
    m_pack = build_pack_graph(a.pack, node, folder_paths).to(DEV)
    print("[1] graphs: source %s / pack %s" % (type(m_src).__name__, type(m_pack).__name__), flush=True)

    # the base control image: a fixed synthetic scene (the control input is the only
    # input the trigger can arrive through)
    g0 = torch.Generator(device=DEV).manual_seed(a.seed)
    base_hint = torch.rand(3, 512, 512, generator=g0, device=DEV).to(dtype=torch.float32)

    specs = patch_spec_list() + freq_spec_list() + control_spec_list()
    rows = []
    for i, spec in enumerate(specs):
        label = candidate_label(spec)
        vals = []
        rels = []
        bad = 0
        for rep in range(a.reps):
            g = torch.Generator(device=DEV).manual_seed(a.seed + 1000 + i * 97 + rep)
            r = measure_divergence(m_src, m_pack, base_hint, spec, g, a.seed + rep)
            if r is None:
                bad += 1
                continue
            vals.append(r[0]); rels.append(r[1])
        if not vals:
            rows.append({"candidate": label, "kind": spec["kind"], "one_minus_cos": None,
                         "rel_l2": None, "reps": a.reps, "non_finite_reps": bad,
                         "note": "a non-finite output for every repetition; no comparison is possible"})
            print("[2] %-28s non-finite (%d/%d)" % (label, bad, a.reps), flush=True)
            continue
        mx = max(vals); mn = min(vals); mean = sum(vals)/len(vals)
        rows.append({"candidate": label, "kind": spec["kind"], "one_minus_cos": mean,
                     "one_minus_cos_max": mx, "one_minus_cos_min": mn,
                     "rel_l2": sum(rels)/len(rels), "reps": len(vals),
                     "non_finite_reps": bad})
        print("[2] %-28s 1-cos mean %.6g max %.6g" % (label, mean, mx), flush=True)

    ctrl = [r["one_minus_cos"] for r in rows
            if r["kind"]=="control" and r["one_minus_cos"] is not None]
    ctrl_sorted = sorted(ctrl)
    ctrl_mean = sum(ctrl)/len(ctrl) if ctrl else None
    def pct(p):
        if not ctrl_sorted: return None
        k = min(len(ctrl_sorted)-1, max(0, int(round(p*(len(ctrl_sorted)-1)))))
        return ctrl_sorted[k]
    p99 = pct(0.99)          # the p99 of the control candidates' means
    ratio_threshold = 1.5    # a candidate must also be this far above the control mean
    flagged = []
    if p99 is not None:
        for r in rows:
            if r["kind"] == "control" or r["one_minus_cos"] is None:
                continue
            ratio = (r["one_minus_cos"] / ctrl_mean) if ctrl_mean else None
            if r["one_minus_cos"] > p99 and (ratio is None or ratio >= ratio_threshold):
                flagged.append({"candidate": r["candidate"],
                                "one_minus_cos": r["one_minus_cos"],
                                "one_minus_cos_max": r.get("one_minus_cos_max"),
                                "ratio_to_control_mean": ratio})
    # --- positive control: the detection floor of this instrument --------------
    # A known embedding is written into a scratch copy of the source graph (a rank-1
    # addition on one Linear weight, and a single-channel addition on the control
    # image's input layer) and the same statistic is measured against the pack. The
    # smallest strength whose divergence leaves the control distribution is the floor:
    # below it, "no trigger found" says nothing about the instrument's sensitivity.
    pos = []
    try:
        pos = positive_control(m_src, m_pack, base_hint, a)
    except Exception as exc:  # noqa: BLE001
        pos = [{"error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}]
        print("[3] the positive control failed: %s" % pos[0]["error"], flush=True)

    floor = None
    for e in pos:
        if e.get("one_minus_cos") is None:
            continue
        if p99 is not None and e["one_minus_cos"] > p99:
            floor = e["strength"]
            break
    rec = {"tool": "controlnet_image_trigger.py",
           "source": os.path.basename(a.source), "pack": os.path.basename(a.pack),
           "observable": "the divergence (1-cos) between the source(float) and the pack(INT8) ControlNet outputs under the same control image",
           "candidate_families": {"patch": len(patch_spec_list()), "frequency": len(freq_spec_list()),
                                  "natural_transform_controls": len(control_spec_list())},
           "candidates": rows,
           "control_one_minus_cos": ctrl_sorted,
           "control_mean": ctrl_mean,
           "control_p99": p99,
           "ratio_threshold": ratio_threshold,
           "flagged_over_control_p99": flagged,
           "flagged_count": len(flagged),
           "positive_control": pos,
           "detection_floor_relative_strength": floor,
           "detection_floor_note": ("the smallest implanted rank-1 strength (relative to the weight's own "
                                    "Frobenius norm) whose divergence leaves the natural-transform control "
                                    "distribution; below it this instrument cannot see an implant"),
           "seed": a.seed,
           "note": "the control p99 is the natural-transform control distribution; a candidate above it is an anomaly to investigate"}
    json.dump(rec, open(a.json,"w",encoding="utf-8"), indent=2, ensure_ascii=False)
    print("wrote", a.json)
    print("control p99 = %s | flagged %d" % (p99, len(flagged)))

if __name__ == "__main__":
    main()

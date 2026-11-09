# -*- coding: utf-8 -*-
"""The source(FP16) and the pack(INT8) ControlNet compared on the same input.

The measurement is the ControlNet's own output tensors: the source graph (float,
built the way the runtime builds it) and the pack graph (INT8, built through the
HSWQ ControlNet Loader node the card names) are run on the same inputs, and the
outputs are compared element by element.

For every batch the per-tensor cosine is taken and the batch's cosine is the
lowest of them; ``cos_min`` is the lowest batch cosine, ``cos_mean`` the mean
of the batch cosines, ``max_abs_err`` the largest absolute element difference
and ``rel_l2_max`` the largest relative L2 difference.

The family's own input shapes are read from the graph's runtime names, so SD1.5
(context 768, no label embedding) and SDXL (context 2048, y 2816) are measured
with the shapes their own graph expects.

Usage:
  python audit/controlnet_pack_source_output_diff.py --source <source> --pack <pack> \
      --comfy <ComfyUI root> [--loader-root <dir>] [--n N] [--seed S] --json <out.json>
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
    import comfy.controlnet as cc  # noqa: E402
    return torch, folder_paths, comfy.utils, cc


def load_pkg(comfy_root, loader_root):
    """The HSWQ ControlNet loader node, the route the card names."""
    cand = os.path.abspath(loader_root or os.path.join(comfy_root, "custom_nodes",
                                                      "ComfyUI-HSWQ-Loader-and-Tools"))
    name = "hswq_controlnet_source_output_diff"
    _spec = importlib.util.spec_from_file_location(name, os.path.join(cand, "__init__.py"),
                                                    submodule_search_locations=[cand])
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[name] = _mod
    _spec.loader.exec_module(_mod)
    pkg = types.ModuleType(name)
    pkg.__path__ = [cand]
    sys.modules[name] = pkg
    importlib.import_module(name + ".nodes.hswq_load_convrot_int8_controlnet")
    return sys.modules[name + ".nodes.hswq_load_convrot_int8_controlnet"]


def build_source_graph(source_path, compute_dtype, cc, comfy_utils):
    """The source as ComfyUI builds it (float), in the loader's compute dtype."""
    sd = comfy_utils.load_torch_file(source_path, safe_load=True)
    cnet = cc.load_controlnet_state_dict(sd, model_options={"dtype": compute_dtype})
    return cnet.control_model


def build_pack_graph(pack_path, node, folder_paths):
    folder_paths.add_model_folder_path("controlnet", os.path.dirname(pack_path))
    (cnet,) = node.HSWQControlNetLoader().load_controlnet(os.path.basename(pack_path))
    return cnet.control_model


def make_inputs(cm, seed):
    """The family's own input shapes, read from the graph's runtime names."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    dtype = cm.dtype
    cls = type(cm).__name__
    if cls == "ControlNetFlux":
        in_ch = getattr(cm, "control_latent_channels", None) or 16
        return dict(x=torch.randn(1, in_ch, 64, 64, generator=g).to(DEV, dtype),
                    timesteps=torch.tensor([500.0], device=DEV, dtype=dtype),
                    context=torch.randn(1, 256, 4096, generator=g).to(DEV, dtype),
                    y=torch.randn(1, 768, generator=g).to(DEV, dtype),
                    guidance=torch.tensor([3.5], device=DEV, dtype=dtype),
                    hint=torch.randn(1, in_ch, 64, 64, generator=g).to(DEV, dtype))
    if cls == "QwenImageControlNetModel":
        ctx_dim = cm.txt_in.weight.shape[1]
        hint_ch = cm.controlnet_x_embedder.weight.shape[1] // 4
        return dict(x=torch.randn(1, 16, 1, 64, 64, generator=g).to(DEV, dtype),
                    timesteps=torch.tensor([500.0], device=DEV, dtype=dtype),
                    context=torch.randn(1, 77, ctx_dim, generator=g).to(DEV, dtype),
                    hint=torch.randn(1, hint_ch, 1, 64, 64, generator=g).to(DEV, dtype))
    if cls == "QwenImageFunControlNetModel":
        import comfy.ops
        import comfy.ldm.qwen_image.model as qm
        in_ch = cm.control_img_in.weight.shape[1] // 4
        ctx_dim = cm.control_blocks[0].attn.to_k.weight.shape[1]
        base = qm.QwenImageTransformer2DModel(final_layer=False, num_layers=0,
                                               joint_attention_dim=ctx_dim, dtype=dtype,
                                               device=DEV, operations=comfy.ops.disable_weight_init)
        base = base.to(DEV).eval()
        return dict(x=torch.randn(1, 16, 1, 64, 64, generator=g).to(DEV, dtype),
                    timesteps=torch.tensor([500.0], device=DEV, dtype=dtype),
                    context=torch.randn(1, 77, ctx_dim, generator=g).to(DEV, dtype),
                    hint=torch.rand(1, in_ch, 64, 64, generator=g).to(DEV, dtype),
                    base_model=base)
    # the cldm families (SD1.5 and SDXL): the context width and the label
    # embedding are read from the graph's own runtime names
    ctx_w = None
    y_w = None
    for n, m in cm.named_modules():
        if n.endswith("attn2.to_k") and getattr(m, "weight", None) is not None:
            ctx_w = int(m.weight.shape[1])
        if n == "label_emb.0.0" and getattr(m, "weight", None) is not None:
            y_w = int(m.weight.shape[1])
        elif n == "label_emb" and getattr(m, "weight", None) is not None:
            y_w = int(m.weight.shape[1])
    call = dict(x=torch.randn(1, 4, 64, 64, generator=g).to(DEV, dtype),
                hint=torch.rand(1, 3, 512, 512, generator=g).to(DEV, dtype),
                timesteps=torch.tensor([500.0], device=DEV, dtype=dtype),
                context=torch.randn(1, 77, ctx_w, generator=g).to(DEV, dtype))
    if y_w is not None:
        call["y"] = torch.randn(1, y_w, generator=g).to(DEV, dtype)
    return call


def run(cm, call):
    with torch.no_grad():
        out = cm(**call)
    parts = []

    def collect(v):
        if torch.is_tensor(v):
            parts.append(v.detach().float().reshape(-1).cpu())
        elif isinstance(v, dict):
            for k in v:
                collect(v[k])
        elif isinstance(v, (list, tuple)):
            for e in v:
                collect(e)

    collect(out)
    return parts


def compare(ref, got):
    """(batch cosine, largest absolute element difference, relative L2)."""
    n = min(len(ref), len(got))
    finite = [j for j in range(n) if torch.isfinite(ref[j]).all() and torch.isfinite(got[j]).all()]
    if not finite:
        return None
    cos = min(float(torch.nn.functional.cosine_similarity(ref[j].reshape(1, -1),
                                                          got[j].reshape(1, -1), dim=1).item())
              for j in finite)
    max_abs = max(float((ref[j] - got[j]).abs().max()) for j in finite)
    rel = max(float((ref[j] - got[j]).norm() / ref[j].norm().clamp(min=1e-30)) for j in finite)
    return cos, max_abs, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--loader-root", default="")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    # resolve the output before the working directory changes
    a.json = os.path.abspath(a.json)
    torch_, folder_paths, comfy_utils, cc = load_comfy(a.comfy)
    node = load_pkg(a.comfy, a.loader_root)
    compute_dtype = node._get_default_compute_dtype(torch.device(DEV))
    m_src = build_source_graph(a.source, compute_dtype, cc, comfy_utils).to(DEV).eval()
    m_pack = build_pack_graph(a.pack, node, folder_paths).to(DEV).eval()
    print("[1] graphs: source %s / pack %s" % (type(m_src).__name__, type(m_pack).__name__),
          flush=True)

    src_params = sum(p.numel() for p in m_src.parameters())
    pack_params = sum(p.numel() for p in m_pack.parameters())
    src_names = {n for n, _ in m_src.named_parameters()}
    pack_names = {n for n, _ in m_pack.named_parameters()}
    only_src = sorted(src_names - pack_names)
    only_pack = sorted(pack_names - src_names)

    cosines = []
    max_abs = 0.0
    rel_max = 0.0
    comparable = 0
    for i in range(a.n):
        call = make_inputs(m_pack, a.seed + i)
        ref = run(m_src, {k: (v.clone() if torch.is_tensor(v) else v) for k, v in call.items()})
        got = run(m_pack, call)
        r = compare(ref, got)
        if r is None:
            print("[2] batch %d: a non-finite output, not comparable" % i, flush=True)
            continue
        comparable += 1
        cosines.append(r[0])
        max_abs = max(max_abs, r[1])
        rel_max = max(rel_max, r[2])
        print("[2] batch %d: cosine %.9f | max_abs_err %.6g | rel_l2 %.6g"
              % (i, r[0], r[1], r[2]), flush=True)

    rec = {"tool": "controlnet_pack_source_output_diff.py",
           "source_class": type(m_src).__name__,
           "pack_class": type(m_pack).__name__,
           "source_parameters": src_params,
           "pack_parameters": pack_params,
           "parameters_only_in_source": only_src,
           "parameters_only_in_pack": only_pack,
           "graphs_isomorphic": (type(m_src).__name__ == type(m_pack).__name__
                                and src_params == pack_params
                                and not only_src and not only_pack),
           "source": os.path.basename(a.source),
           "pack": os.path.basename(a.pack),
           "observable": "the ControlNet's own output tensors, source(float) vs pack(INT8), "
                         "same input; the batch cosine is the lowest per-tensor cosine, "
                         "max_abs_err the largest absolute element difference and rel_l2_max "
                         "the largest relative L2 difference",
           "n": a.n,
           "seed": a.seed,
           "batches": [],
           "batches_comparable": comparable,
           "batches_total": a.n,
           "cos_min": min(cosines) if cosines else None,
           "cos_mean": (sum(cosines) / len(cosines)) if cosines else None,
           "max_abs_err": max_abs,
           "rel_l2_max": rel_max}
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    print("wrote", a.json)
    print("cos_min %s | cos_mean %s | rel_l2_max %s"
          % (rec["cos_min"], rec["cos_mean"], rec["rel_l2_max"]))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""The FLUX Union Pro 2.0 ControlNet's 256-item acceptance, and the qkv /
zero-drift decision, all measured on the loader's own path.

The measurement runs through the HSWQ loader's node (the package's root
``__init__`` is executed exactly as ComfyUI does on startup, which is what
enables the real INT8 GEMM), and the FP16 baseline is built through the same
ops class, because the ComfyUI ``comfy.controlnet`` path and the
``mixed_precision_ops`` path are not numerically equivalent for this model.

Runtime conventions, each established by measurement:
- the compute dtype is the loader's own default (BF16 on this GPU), and every
  tensor - the timestep included - is cast to the model's dtype, exactly as
  ``comfy.controlnet.ControlNet.get_control`` does (``timestep.to(dtype)``);
  an FP32 timestep makes ``vec`` FP32 while the activations are BF16 and the
  modulation then poisons the whole block, and an FP16 timestep overflows
  (``timestep_embedding`` multiplies by 1000, so 500 becomes 500000, past the
  FP16 range of 65504);
- the SDXL/FLUX control model's context width and the hint's channel count
  come from the model itself.

Usage:
    python verify_cn_flux_256.py <source.safetensors> [<pack.safetensors> ...]
"""
import os
import sys
import json
import types
import statistics
import torch
import importlib

COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
NODE = os.path.join(COMFY, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools")

sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
os.chdir(COMFY)

import folder_paths  # noqa: E402
import server as comfy_server  # noqa: E402
import comfy.utils  # noqa: E402
import comfy.model_detection  # noqa: E402
import comfy.ldm.flux.controlnet as cnf  # noqa: E402
import comfy.ops as O  # noqa: E402

if not hasattr(comfy_server.PromptServer, "instance"):
    comfy_server.PromptServer.instance = None

_spec = importlib.util.spec_from_file_location(
    "hswq_pkg_init", os.path.join(NODE, "__init__.py"), submodule_search_locations=[NODE])
_mod = importlib.util.module_from_spec(_spec)
sys.modules["hswq_pkg_init"] = _mod
_spec.loader.exec_module(_mod)

_pkg = types.ModuleType("hswq_pkg")
_pkg.__path__ = [NODE]
sys.modules["hswq_pkg"] = _pkg
importlib.import_module("hswq_pkg.nodes.hswq_load_convrot_int8_controlnet")
CN_MOD = sys.modules["hswq_pkg.nodes.hswq_load_convrot_int8_controlnet"]

DT = CN_MOD._get_default_compute_dtype()
N = int(os.environ.get("CNN_N", "256"))


def is_flux_native(sd):
    """The comfy-native ControlNetFlux layout, which comfy.controlnet has no
    detection branch for - the loader builds it directly."""
    return ("img_in.weight" in sd and "controlnet_blocks.0.weight" in sd
            and "pos_embed_input.weight" in sd
            and "controlnet_x_embedder.weight" not in sd
            and "control_model.zero_convs.0.0.weight" not in sd
            and "zero_convs.0.0.weight" not in sd)


def build_pack(sd, name="", pack_path=""):
    """The pack, through the loader's own path.

    The FLUX-native layout is built by the loader's own FLUX branch; every
    other family (the cldm ones, SD1.5 and SDXL) is built by the loader's own
    node, the route the card names."""
    model_options = {
        "dtype": DT,
        "custom_operations": CN_MOD._int8_mixed_precision_ops(compute_dtype=DT),
    }
    if is_flux_native(sd):
        control = CN_MOD._load_comfy_flux_controlnet(sd, model_options)
        return control.control_model
    if pack_path:
        folder_paths.add_model_folder_path("controlnet", os.path.dirname(os.path.abspath(pack_path)))
    (control,) = CN_MOD.HSWQControlNetLoader().load_controlnet(name)
    return control.control_model


def is_cldm(sd):
    """The cldm families (SD1.5 and SDXL), in either layout the runtime accepts:
    the comfy-native names (`zero_convs.*`) and the diffusers names
    (`controlnet_down_blocks.*` / `controlnet_cond_embedding.*`, converted by
    comfy.model_detection). The FLUX-native layout has neither."""
    return ("control_model.zero_convs.0.0.weight" in sd or "zero_convs.0.0.weight" in sd
            or ("controlnet_down_blocks.0.weight" in sd
                and "controlnet_cond_embedding.conv_in.weight" in sd))


def build_source(path):
    """The FP16 source, through the same ops class and the same construction."""
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    if is_cldm(sd):
        # the cldm families (SD1.5 and SDXL): the runtime's own construction,
        # through the same ops class the pack runs with
        ops = CN_MOD._int8_mixed_precision_ops(compute_dtype=DT)
        control = comfy.controlnet.load_controlnet_state_dict(
            sd, model_options={"dtype": DT, "custom_operations": ops})
        return control.control_model
    converted = dict(sd)
    mc = comfy.model_detection.model_config_from_unet(dict(sd), "", use_base_if_no_match=True)
    ops = CN_MOD._int8_mixed_precision_ops(compute_dtype=DT)
    model = cnf.ControlNetFlux(
        latent_input=True, num_union_modes=0,
        control_latent_channels=converted["pos_embed_input.weight"].shape[1] // 4,
        operations=ops, device="cpu", dtype=DT, **mc.unet_config)
    model = comfy.controlnet.controlnet_load_state_dict(model, converted)
    return model


def cldm_shapes(model):
    """The cldm family's own input shapes, read from the graph's runtime names:
    the context width and the label embedding are the model's own."""
    ctx_w = None
    y_w = None
    for n, m in model.named_modules():
        if n.endswith("attn2.to_k") and getattr(m, "weight", None) is not None:
            ctx_w = int(m.weight.shape[1])
        if n == "label_emb.0.0" and getattr(m, "weight", None) is not None:
            y_w = int(m.weight.shape[1])
        elif n == "label_emb" and getattr(m, "weight", None) is not None:
            y_w = int(m.weight.shape[1])
    return ctx_w, y_w


def inputs_for(model, seed, side=32, txt_tokens=77):
    """The family's own input dict, read from the graph's runtime names."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    if type(model).__name__ != "ControlNetFlux":
        ctx_w, y_w = cldm_shapes(model)
        call = dict(x=torch.randn(1, 4, side, side, generator=g).to("cuda", dtype=DT),
                    hint=torch.rand(1, 3, 256, 256, generator=g).to("cuda", dtype=DT),
                    timesteps=torch.full((1,), 500.0, device="cuda", dtype=DT),
                    context=torch.randn(1, txt_tokens, ctx_w, generator=g).to("cuda", dtype=DT))
        if y_w is not None:
            call["y"] = torch.randn(1, y_w, generator=g).to("cuda", dtype=DT)
        return call
    return dict(x=torch.randn(1, 16, side, side, generator=g).to("cuda", dtype=DT),
                timesteps=torch.full((1,), 500.0, device="cuda", dtype=DT),
                context=torch.randn(1, txt_tokens, 4096, generator=g).to("cuda", dtype=DT),
                y=torch.zeros(1, 3072, device="cuda", dtype=DT),
                guidance=torch.full((1,), 3.5, device="cuda", dtype=DT),
                hint=torch.randn(1, 16, side, side, generator=g).to("cuda", dtype=DT))


def features(model, seed, side=32, txt_tokens=77):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if type(model).__name__ != "ControlNetFlux":
        # the cldm families: the image is the hint, the context is the model's own
        ctx_w, y_w = cldm_shapes(model)
        x = torch.randn(1, 4, side, side, generator=g).to("cuda", dtype=DT)
        hint = torch.rand(1, 3, 256, 256, generator=g).to("cuda", dtype=DT)
        txt = torch.randn(1, txt_tokens, ctx_w, generator=g).to("cuda", dtype=DT)
        ts = torch.full((1,), 500.0, device="cuda", dtype=DT)
        with torch.no_grad():
            if y_w is not None:
                out = model.forward(x=x, hint=hint, timesteps=ts, context=txt,
                                    y=torch.randn(1, y_w, generator=g).to("cuda", dtype=DT))
            else:
                out = model.forward(x=x, hint=hint, timesteps=ts, context=txt, y=None)
        parts = []
        for k in sorted(out):
            v = out[k]
            if torch.is_tensor(v):
                parts.append(v.detach().float().reshape(-1).cpu())
            elif isinstance(v, (tuple, list)):
                for t2 in v:
                    if torch.is_tensor(t2):
                        parts.append(t2.detach().float().reshape(-1).cpu())
        return torch.cat(parts)
    x = torch.randn(1, 16, side, side, generator=g).to("cuda", dtype=DT)
    txt = torch.randn(1, txt_tokens, 4096, generator=g).to("cuda", dtype=DT)
    hint = torch.randn(1, 16, side, side, generator=g).to("cuda", dtype=DT)
    y = torch.zeros(1, 3072, device="cuda", dtype=DT)
    ts = torch.full((1,), 500.0, device="cuda", dtype=DT)
    gd = torch.full((1,), 3.5, device="cuda", dtype=DT)
    with torch.no_grad():
        out = model.forward(x=x, timesteps=ts, context=txt, y=y, guidance=gd, hint=hint)
    parts = []
    for k in sorted(out):
        v = out[k]
        if torch.is_tensor(v):
            parts.append(v.detach().float().reshape(-1).cpu())
        elif isinstance(v, (tuple, list)):
            for t2 in v:
                if torch.is_tensor(t2):
                    parts.append(t2.detach().float().reshape(-1).cpu())
    return torch.cat(parts)



def verify_ipadapter(src_path, pack_path, ipadapter_dir, n=256):
    """The 256-item acceptance gate for an IP-Adapter pack: the source's float weights
    against the pack's dequantised weights, over n fixed inputs, through the graph the
    node builds."""
    import os
    import statistics
    import sys

    import torch
    from safetensors.torch import load_file

    if ipadapter_dir not in sys.path:
        sys.path.insert(0, ipadapter_dir)
    from image_proj_models import Resampler

    def flatten(path):
        if path.lower().endswith((".pth", ".bin")):
            raw = torch.load(path, map_location="cpu", weights_only=True)
            out = {}
            for k, v in raw["image_proj"].items():
                if torch.is_tensor(v):
                    out["image_proj." + k] = v
            for k, v in raw["ip_adapter"].items():
                if torch.is_tensor(v):
                    out["ip_adapter." + k] = v
            return out
        return load_file(path)

    def hadamard(size):
        h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
        hh = h4
        cur = 4
        while cur < size:
            hh = torch.kron(hh, h4)
            cur *= 4
        return hh / (size ** 0.5)

    src = flatten(src_path)
    pack = flatten(pack_path)
    declared = sorted(k[: -len(".comfy_quant")] for k in pack if k.endswith(".comfy_quant"))
    h = hadamard(256)
    deq = {}
    for mk in declared:
        q = pack[mk + ".weight"]
        sc = pack[mk + ".weight_scale"].float()
        of, inf = q.shape
        g = inf // 256
        deq[mk + ".weight"] = torch.matmul((q.float() * sc.float()).view(of, g, 256), h).reshape(of, inf)
    restored = {}
    for k, v in pack.items():
        if k.endswith(".weight_scale") or k.endswith(".comfy_quant"):
            continue
        restored[k] = deq[k] if (k.endswith(".weight") and k[:-7] in set(declared)) else v

    def build(sd):
        proj, ipa = {}, {}
        for k, v in sd.items():
            if k.startswith("image_proj."):
                proj[k.replace("image_proj.", "", 1)] = v.float() if v.is_floating_point() else v
            elif k.startswith("ip_adapter."):
                ipa[k.replace("ip_adapter.", "", 1)] = v.float() if v.is_floating_point() else v
        res = Resampler(dim=768, depth=4, dim_head=64, heads=12, num_queries=16,
                        embedding_dim=1280, output_dim=768, ff_mult=4,
                        max_seq_len=257, apply_pos_emb=False, num_latents_mean_pooled=0)
        res.load_state_dict(proj, strict=False)
        kv = {k[: -len(".weight")]: v for k, v in ipa.items() if k.endswith(".weight")}
        return res.eval(), kv

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ra, ka = build(src)
    rb, kb = build(restored)
    ra.to(dev); rb.to(dev)
    ka = {k: v.to(dev) for k, v in ka.items()}
    kb = {k: v.to(dev) for k, v in kb.items()}
    keys = sorted(ka)

    cs = []
    fin = True
    with torch.no_grad():
        for i in range(n):
            g = torch.Generator().manual_seed(900000 + i)
            x = torch.randn(1, 257, 1280, generator=g).to(dev)
            ea = ra(x)
            eb = rb(x)
            oa = torch.cat([torch.nn.functional.linear(ea, ka[k]).flatten() for k in keys])
            ob = torch.cat([torch.nn.functional.linear(eb, kb[k]).flatten() for k in keys])
            if not (torch.isfinite(ea).all() and torch.isfinite(eb).all()):
                fin = False
            cs.append(torch.nn.functional.cosine_similarity(oa.unsqueeze(0), ob.unsqueeze(0)).item())
    cs.sort()
    m = len(cs)
    import hashlib
    rec = {"pack": os.path.basename(pack_path), "family": "ipadapter",
           "pack_sha256": hashlib.sha256(open(pack_path, "rb").read()).hexdigest(),
           "source_sha256": hashlib.sha256(open(src_path, "rb").read()).hexdigest(),
           "file_mb": os.path.getsize(pack_path) / 1024 ** 2,
           "quantized_layers": len(declared), "int8_linear_calls": 0,
           "mean": statistics.mean(cs), "min": cs[0], "p5": cs[max(0, int(m * 0.05) - 1)],
           "p50": cs[m // 2], "n": m, "finite": fin}
    rec["pass"] = bool(rec["mean"] >= 0.99 and rec["min"] >= 0.98 and fin)
    print("    quantised=%d finite=%s  mean=%.6f min=%.6f p5=%.6f p50=%.6f  size=%.1f MB  %s"
          % (len(declared), fin, rec["mean"], rec["min"], rec["p5"], rec["p50"],
             rec["file_mb"], "PASS" if rec["pass"] else "FAIL"))
    import json as _json
    print("JSON " + _json.dumps([rec]))
    return rec


def main():
    if "--family" in sys.argv:
        i = sys.argv.index("--family")
        fam = sys.argv[i + 1]
        rest = [a for j, a in enumerate(sys.argv) if j not in (i, i + 1)]
        if fam == "ipadapter":
            # rest is [prog, src, pack, node_dir]: the node dir is the fourth entry
            node_dir = rest[3] if len(rest) > 3 else ""
            src_path, pack_path = rest[1], rest[2]
            verify_ipadapter(src_path, pack_path, node_dir)
            return
    src = sys.argv[1]
    packs = sys.argv[2:]
    print("[1] the FP16 baseline (the same ops path) ...", flush=True)
    m16 = build_source(src)
    base = {}
    for s in range(N):
        base[s] = features(m16, s)
    print("    the reference features: %s finite=%s" % (
        tuple(base[0].shape), bool(torch.isfinite(base[0]).all())), flush=True)
    del m16
    torch.cuda.empty_cache()

    results = []
    for p in packs:
        print("[2] %s" % os.path.basename(p), flush=True)
        sd = comfy.utils.load_torch_file(p, safe_load=True)
        m8 = build_pack(sd, os.path.basename(p), p)
        nq = len([k for k in sd if k.endswith(".comfy_quant")])
        ck = O.quant_ops.ck
        orig = ck.int8_linear
        n = [0]

        def counted(*a, **kw):
            n[0] += 1
            return orig(*a, **kw)

        ck.int8_linear = counted
        cs = []
        fin = True
        try:
            with torch.no_grad():
                m8.forward(**inputs_for(m8, 999983))
        except Exception:
            pass
        finally:
            ck.int8_linear = orig
        for s in range(N):
            f = features(m8, s)
            if not torch.isfinite(f).all():
                fin = False
            cs.append(torch.nn.functional.cosine_similarity(
                base[s].reshape(1, -1), f.reshape(1, -1)).item())
        cs.sort()
        m = len(cs)
        rec = {"pack": os.path.basename(p),
               "file_mb": os.path.getsize(p) / 1024 ** 2,
               "quantized_layers": nq,
               "int8_linear_calls": n[0],
               "mean": statistics.mean(cs), "min": cs[0],
               "p5": cs[max(0, int(m * 0.05) - 1)], "p50": cs[m // 2],
               "n": m, "finite": fin}
        rec["pass"] = bool(rec["mean"] >= 0.99 and rec["min"] >= 0.98 and fin)
        results.append(rec)
        print("    int8_linear=%d finite=%s  mean=%.6f min=%.6f p5=%.6f p50=%.6f  size=%.1f MB  %s" % (
            n[0], fin, rec["mean"], rec["min"], rec["p5"], rec["p50"], rec["file_mb"],
            "PASS" if rec["pass"] else "FAIL"), flush=True)
        del m8
        torch.cuda.empty_cache()

    print()
    print("%-56s %7s %10s %10s %10s %10s" % ("pack", "int8", "mean", "min", "p5", "MB"))
    for r in results:
        print("%-56s %7d %10.6f %10.6f %10.6f %10.1f" % (
            r["pack"][:56], r["int8_linear_calls"], r["mean"], r["min"], r["p5"], r["file_mb"]))
    print("JSON " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()

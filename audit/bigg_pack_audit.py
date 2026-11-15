"""CLIP-ViT-bigG-14 ConvRot INT8 pack audit.

Four questions, each answered by measurement on the two files:

  1. identity   - which bytes the pack is closed against, and the pack's own bytes
  2. coverage   - every source tensor present, no tensor added, no shape changed
  3. arithmetic - the int8 codes and scales are the converter's own values for the
                  source weights, and the tensors the pack does not quantise are
                  the source's, bit for bit
  4. behaviour  - the two towers on the same input: the source's float weights
                  against the pack, through the loader's own ConvRot INT8 route

  python audit/bigg_pack_audit.py --source <f> --pack <f> --comfy <d> --json <record>
"""

import argparse
import hashlib
import json
import math
import os
import sys
import types
import importlib.util

import torch
import torch.nn as nn

COMFY_DEFAULT = r"D:\USERFILES\ComfyUI\ComfyUI"
GROUP_SIZE = 64


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_hadamard(size, device="cpu", dtype=torch.float32):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype, device=device)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def rotate_weight(w, h, gs):
    of, inf = w.shape
    g = inf // gs
    return torch.matmul(w.view(of, g, gs), h.T.to(w.dtype)).reshape(w.shape)


def quant_rowwise(w):
    amax = w.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
    s = amax / 127.0
    q = (w / s.to(w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, s.to(torch.float32)


def _install(comfy_root):
    node = os.path.join(comfy_root, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools")
    if "ComfyUI-HSWQ-Loader-and-Tools" not in sys.modules:
        m = types.ModuleType("ComfyUI-HSWQ-Loader-and-Tools")
        m.__path__ = [node]
        sys.modules["ComfyUI-HSWQ-Loader-and-Tools"] = m
    spec = importlib.util.spec_from_file_location(
        "ComfyUI-HSWQ-Loader-and-Tools.hswq.clip_vision_convrot_int8",
        os.path.join(node, "hswq", "clip_vision_convrot_int8.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ComfyUI-HSWQ-Loader-and-Tools.hswq.clip_vision_convrot_int8"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_tower(comfy_root, jc, ops, sd, d):
    import comfy.clip_vision as cv
    import comfy.model_management

    class CVQ(cv.ClipVisionModel):
        def __init__(self, jc, ops):
            c = json.load(open(jc))
            self.image_size = c.get("image_size", 224)
            self.image_mean = c.get("image_mean", [0.5, 0.5, 0.5])
            self.image_std = c.get("image_std", [0.5, 0.5, 0.5])
            self.model_type = c.get("model_type", "clip_vision_model")
            self.config = c.copy()
            mc = cv.IMAGE_ENCODERS.get(self.model_type)
            self.return_all_hidden_states = False
            self.load_device = comfy.model_management.text_encoder_device()
            off = comfy.model_management.text_encoder_offload_device()
            self.dtype = comfy.model_management.text_encoder_dtype(self.load_device)
            self.model = mc(c, self.dtype, off, ops)
            self.model.eval()
            self.patcher = comfy.model_patcher.CoreModelPatcher(
                self.model, load_device=self.load_device, offload_device=off)
            self.naf = None

    v = CVQ(jc, ops)
    vv = {k: x for k, x in sd.items() if k.startswith("vision_model.")}
    if "vision_model.embeddings.class_embedding" not in vv:
        vv["vision_model.embeddings.class_embedding"] = torch.zeros(d)
    v.load_sd(vv)
    return v


def embeds(v, im):
    with torch.no_grad():
        out = v.encode_image(im, crop=False)
    if hasattr(out, "image_embeds") and torch.is_tensor(out.image_embeds):
        return out.image_embeds.detach().float().reshape(-1).cpu()
    if isinstance(out, dict):
        return torch.cat([out[k].detach().float().reshape(-1).cpu()
                          for k in sorted(out) if torch.is_tensor(out[k])])
    if torch.is_tensor(out):
        return out.detach().float().reshape(-1).cpu()
    return torch.cat([x.detach().float().reshape(-1).cpu() for x in out if torch.is_tensor(x)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--comfy", default=COMFY_DEFAULT)
    ap.add_argument("--json", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--groupsize", type=int, default=GROUP_SIZE)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()

    comfy_root = os.path.abspath(a.comfy)
    sys.path.insert(0, comfy_root)
    sys.path.insert(0, os.path.join(comfy_root, "custom_nodes"))
    _install(comfy_root)

    import comfy.utils
    import comfy.ops as O
    from comfy.quant_ops import QUANT_ALGOS

    src = comfy.utils.load_torch_file(a.source, safe_load=True)
    pack = comfy.utils.load_torch_file(a.pack, safe_load=True)

    q_keys = [k for k in pack if k.endswith(".comfy_quant")]
    declared = sorted(k[: -len(".comfy_quant")] for k in q_keys)
    print("source %s  %d B" % (sha256(a.source)[:16], os.path.getsize(a.source)))
    print("pack   %s  %d B" % (sha256(a.pack)[:16], os.path.getsize(a.pack)))
    print("declared INT8 layers %d" % len(declared))

    src_weights = {k for k in src if k.endswith(".weight")}
    pack_weights = {k for k in pack if k.endswith(".weight")}
    missing = sorted(src_weights - pack_weights)
    extra = sorted(pack_weights - src_weights)
    shape_bad = [k for k in src_weights & pack_weights if tuple(src[k].shape) != tuple(pack[k].shape)]
    print("coverage: missing %d | extra %d | shape %d" % (len(missing), len(extra), len(shape_bad)))

    q_mismatch_total = 0
    max_q_delta = 0
    scale_rel_max = 0.0
    nonfinite = 0
    layers_bad = []
    h = build_hadamard(a.groupsize)
    for mk in declared:
        w = src[mk + ".weight"].float()
        gs = a.groupsize if int(w.shape[1]) % a.groupsize == 0 else None
        if gs is None:
            for cand in (64, 16, 4):
                if int(w.shape[1]) % cand == 0:
                    gs = cand
                    break
        if gs is None:
            layers_bad.append(mk)
            continue
        q_ref, s_ref = quant_rowwise(rotate_weight(w, h, gs))
        q = pack[mk + ".weight"]
        s = pack[mk + ".weight_scale"].float()
        if not torch.isfinite(s).all():
            nonfinite += 1
        d = (q.to(torch.int32) - q_ref.to(torch.int32)).abs()
        n_bad = int((d > 0).sum().item())
        if n_bad:
            layers_bad.append(mk)
        q_mismatch_total += n_bad
        max_q_delta = max(max_q_delta, int(d.max().item()))
        rel = ((s - s_ref).abs() / s_ref.abs().clamp(min=1e-30)).max().item()
        scale_rel_max = max(scale_rel_max, float(rel))
    print("quantised: q_mismatch %d | max_q_delta %d | scale rel max %.3e | nonfinite %d"
          % (q_mismatch_total, max_q_delta, scale_rel_max, nonfinite))

    printed = set()
    for mk in declared:
        printed.add(mk + ".weight")
        printed.add(mk + ".weight_scale")
        printed.add(mk + ".comfy_quant")
    float_keys = [k for k in src if k not in printed]
    f_mismatch = 0
    f_max = 0.0
    dtype_diff = []
    for k in float_keys:
        if k not in pack:
            f_mismatch += 1
            continue
        x, y = src[k], pack[k]
        if tuple(x.shape) != tuple(y.shape):
            f_mismatch += 1
            continue
        if x.dtype != y.dtype:
            # the values are compared separately; a dtype difference on its own is
            # recorded, not counted as a mismatch of the content
            dtype_diff.append({"tensor": k, "source_dtype": str(x.dtype), "pack_dtype": str(y.dtype)})
        if x.is_floating_point():
            d = (x.float() - y.float()).abs().max().item()
            f_max = max(f_max, float(d))
            if d > 0:
                f_mismatch += 1
        elif not torch.equal(x, y):
            f_mismatch += 1
    print("float side: checked %d | mismatch %d | max abs diff %.3e | dtype differences %d"
          % (len(float_keys), f_mismatch, f_max, len(dtype_diff)))

    # behaviour: the source's float weights against the pack, same inputs, same ops class
    ops = O.mixed_precision_ops({"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]},
                                torch.float16, full_precision_mm=False)
    d_dim = 1664
    for k, v in src.items():
        if k.endswith("embeddings.class_embedding"):
            d_dim = int(v.shape[0])
            break
    jc = os.path.join(comfy_root, "comfy", "clip_vision_config_g.json")
    if not os.path.isfile(jc):
        jc = os.path.join(comfy_root, "comfy", "clip_vision_config_vitl.json")
    cfg = json.load(open(jc))
    size = int(cfg.get("image_size", 224))
    g = torch.Generator(device="cpu").manual_seed(a.seed)

    def imgs(seed):
        gg = torch.Generator(device="cpu").manual_seed(seed)
        return (torch.rand(1, 3, size, size, generator=gg) * 2 - 1).to("cuda", dtype=torch.float16)

    v_src = build_tower(comfy_root, jc, ops, src, d_dim)
    ref = [embeds(v_src, imgs(1000 + s)) for s in range(a.n)]
    del v_src
    try:
        import comfy.model_management
        comfy.model_management.unload_all_models()
    except Exception:
        pass
    v_pack = build_tower(comfy_root, jc, ops, pack, d_dim)
    cs = []
    for s in range(a.n):
        f = embeds(v_pack, imgs(1000 + s))
        cs.append(torch.nn.functional.cosine_similarity(
            ref[s].reshape(1, -1), f.reshape(1, -1)).item())
    cs.sort()
    cos_mean = sum(cs) / len(cs)
    cos_min = cs[0]
    print("behaviour: the image-embedding cosine, source against pack: mean %.9f min %.9f over %d"
          % (cos_mean, cos_min, len(cs)))

    verdict = "FAITHFUL" if (not missing and not extra and not shape_bad
                             and q_mismatch_total == 0 and nonfinite == 0 and f_mismatch == 0) else "MISMATCH"
    print("VERDICT: %s" % verdict)

    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    rec = {
        "tool": "audit/bigg_pack_audit.py",
        "family": "clip_vision",
        "model": os.path.basename(a.source),
        "source": {"file": os.path.basename(a.source), "bytes": os.path.getsize(a.source),
                   "sha256": sha256(a.source), "tensors": len(src)},
        "pack": {"file": os.path.basename(a.pack), "bytes": os.path.getsize(a.pack),
                 "sha256": sha256(a.pack), "tensors": len(pack)},
        "declared": {"layers": len(declared), "format": "int8_tensorwise", "convrot": True,
                     "groupsize": a.groupsize},
        "coverage": {"n_missing": len(missing), "n_extra": len(extra),
                     "n_shape_mismatch": len(shape_bad), "missing": missing, "extra": extra,
                     "shape_mismatch": shape_bad},
        "quantised": {"found": len(declared), "q_mismatch_total": q_mismatch_total,
                      "max_q_delta": max_q_delta, "scale_exact_rel_max": scale_rel_max,
                      "nonfinite_scales": nonfinite, "n_layers_with_difference": len(layers_bad)},
        "float_side": {"checked": len(float_keys), "mismatch_count": f_mismatch, "max_abs_diff": f_max,
                       "dtype_differences": dtype_diff},
        "behaviour": {"image_embed_cos_mean": cos_mean, "image_embed_cos_min": cos_min,
                      "n": len(cs), "seed": a.seed,
                      "how": "the source's float weights against the pack, both built through "
                             "mixed_precision_ops, over %d fixed inputs" % a.n},
        "verdict": verdict,
    }
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print("wrote %s" % a.json)


if __name__ == "__main__":
    main()

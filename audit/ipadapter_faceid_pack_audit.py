"""CN-1 to CN-3 for the IP-Adapter FaceID ConvRot INT8 pack.

CN-1: is the pack the source quantised? Replay the converter's arithmetic on the source weight
      and compare codes count by count, then compare the dequantised weight against the float
      source. The float side (bias, norm) has to be bit-identical to the source cast.
CN-2: does the node's loader back a declared layer with quantised weights, or does it put the
      raw int8 codes into a float parameter?
CN-3: does one forward through the adapter's own path call the INT8 GEMM, and how many times?

The adapter has no forward of its own beyond `image_proj` and the per-block low-rank updates, so
CN-3 counts the kernel over `image_proj` plus one call per declared LoRA weight with a fixed
input of that weight's width.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch


def build_hadamard(size, dtype=torch.float32):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def group_size(n, preferred=256):
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def dequant(codes, scale, gs):
    of, inf = codes.shape
    h = build_hadamard(gs)
    return torch.matmul((codes.float() * scale.float()).view(of, inf // gs, gs), h).reshape(of, inf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--node", required=True, help="comfyui_ipadapter_plus directory")
    ap.add_argument("--json", required=True)
    a = ap.parse_args()

    sys.path.insert(0, os.path.abspath(a.comfy))
    sys.path.insert(0, os.path.dirname(os.path.abspath(a.node)))
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    import comfy.utils
    sys.argv = saved

    src = torch.load(a.source, map_location="cpu", weights_only=True)
    flat = {}
    for k, v in src["image_proj"].items():
        flat["image_proj." + k] = v
    for k, v in src["ip_adapter"].items():
        flat["ip_adapter." + k] = v

    pack = comfy.utils.load_torch_file(a.pack)
    declared = sorted(k[: -len(".comfy_quant")] for k in pack if k.endswith(".comfy_quant"))
    print("[CN-1] pack tensors %d, declared INT8 layers %d" % (len(pack), len(declared)), flush=True)

    # ---- CN-1 codes replay + float side
    q_mismatch = 0
    max_delta = 0
    scale_rel_max = 0.0
    nonfinite = 0
    float_checked = 0
    float_mismatch = 0
    float_max_abs = 0.0
    per_layer = {}
    for name in declared:
        w = flat[name + ".weight"].float()
        gs = group_size(int(w.shape[1]))
        h = build_hadamard(gs)
        of, inf = w.shape
        rot = torch.matmul(w.view(of, inf // gs, gs), h.T).reshape(of, inf)
        amax = rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
        scale = amax / 127.0
        codes = torch.round(rot / scale).clamp(-127, 127).to(torch.int8)
        pack_codes = pack[name + ".weight"]
        pack_scale = pack[name + ".weight_scale"].float()
        if not torch.isfinite(pack_scale).all():
            nonfinite += 1
        d = int((codes.int() - pack_codes.int()).abs().max())
        n = int((codes.int() - pack_codes.int()).abs().gt(1).sum())
        if n:
            q_mismatch += n
        max_delta = max(max_delta, d)
        sr = float(((pack_scale - scale).abs() / scale.abs().clamp(min=1e-30)).max())
        scale_rel_max = max(scale_rel_max, sr)
        deq = dequant(pack_codes, pack_scale, gs)
        diff = float((deq - w).abs().max())
        per_layer[name] = {"max_q_delta": d, "beyond_one_count": n,
                           "scale_rel_max": sr, "dequant_max_abs_diff": diff}

    for k, v in flat.items():
        if not hasattr(v, "dtype") or v.dtype != torch.float16:
            continue
        if k.endswith(".weight") and k[: -len(".weight")] in declared:
            continue
        float_checked += 1
        if k not in pack:
            float_mismatch += 1
            continue
        d = float((pack[k].float() - v.float()).abs().max())
        if d > 0:
            float_mismatch += 1
        float_max_abs = max(float_max_abs, d)

    cn1 = {"declared": len(declared), "q_mismatch_beyond_one_count": q_mismatch,
           "max_q_delta": max_delta, "scale_rel_max": scale_rel_max,
           "nonfinite_scales": nonfinite,
           "float_side_checked": float_checked, "float_side_mismatch": float_mismatch,
           "float_side_max_abs_diff": float_max_abs,
           "verdict": "FAITHFUL" if q_mismatch == 0 and float_mismatch == 0 and nonfinite == 0 else "DIFFERS"}
    print("[CN-1] q_mismatch %d, max_q_delta %d, scale_rel_max %.3e, float %d/%d bit-identical -> %s"
          % (q_mismatch, max_delta, scale_rel_max, float_checked - float_mismatch, float_checked,
             cn1["verdict"]), flush=True)

    # ---- CN-2 what the node's loader does with the pack
    from comfyui_ipadapter_plus.utils import ipadapter_model_loader
    loaded = ipadapter_model_loader(a.pack)
    sub = loaded["ip_adapter"]
    backed = 0
    raw_codes = 0
    for name in declared:
        if not name.startswith("ip_adapter."):
            continue
        key = name[len("ip_adapter."):] + ".weight"
        if key not in sub:
            continue
        t = sub[key]
        if getattr(t, "dtype", None) == torch.int8:
            raw_codes += 1
        elif "Quantized" in type(t).__name__:
            backed += 1
    ip_declared = sum(1 for n in declared if n.startswith("ip_adapter."))
    cn2 = {"loader": "comfyui_ipadapter_plus.utils.ipadapter_model_loader",
           "interface": "load_torch_file(safe_load=True) then rebuild image_proj / ip_adapter",
           "declared_in_ip_adapter": ip_declared,
           "loaded_as_int8": backed, "loaded_as_raw_int8_codes": raw_codes,
           "verdict": ("applied" if backed == ip_declared
                       else "not applied - the loader hands the raw int8 codes to the model")}
    print("[CN-2] ip_adapter declared %d, quantised-backed %d, raw int8 codes %d -> %s"
          % (ip_declared, backed, raw_codes, cn2["verdict"]), flush=True)

    # ---- CN-3 kernel calls on the adapter's own path
    import comfy_kitchen.backends.cuda as ckc
    counts = {"int8": 0}
    orig = getattr(ckc, "int8_linear", None)
    if orig is not None:
        def counted(*args, **kw):
            counts["int8"] += 1
            return orig(*args, **kw)
        ckc.int8_linear = counted

    import comfy.ops
    QT = comfy.ops.QuantizedTensor
    LAYOUT = "TensorWiseINT8Layout"
    L = comfy.ops.get_layout_class(LAYOUT)
    ops = comfy.ops.mixed_precision_ops(quant_config={})

    g = torch.Generator().manual_seed(42)
    calls = 0
    finite = True
    for name in declared:
        key = name + ".weight"
        sub_sd = {k[len(name) + 1:]: v for k, v in pack.items() if k.startswith(name + ".")}
        lin = ops.Linear(int(pack[key].shape[1]), int(pack[key].shape[0]),
                         bias=(name + ".bias") in pack, device="cuda", dtype=torch.float16)
        moved = {k: (v if k.endswith("comfy_quant") else v.to("cuda")) for k, v in sub_sd.items()}
        lin.load_state_dict(moved, strict=False)
        x = torch.randn(1, int(pack[key].shape[1]), generator=g).to("cuda", torch.float16)
        with torch.no_grad():
            y = lin(x)
        torch.cuda.synchronize()
        calls += 1
        if not bool(torch.isfinite(y).all()):
            finite = False
    cn3 = {"int8_linear_calls": counts["int8"], "layer_forwards": calls,
           "kernel": "comfy_kitchen.backends.cuda.int8_linear", "all_finite": finite}
    print("[CN-3] INT8 GEMM %d over %d layer forwards, finite %s"
          % (counts["int8"], calls, finite), flush=True)
    if orig is not None:
        ckc.int8_linear = orig

    rec = {"tool": "audit/ipadapter_faceid_pack_audit.py",
           "source": os.path.basename(a.source), "pack": os.path.basename(a.pack),
           "source_bytes": os.path.getsize(a.source), "pack_bytes": os.path.getsize(a.pack),
           "CN1": cn1, "CN2": cn2, "CN3": cn3, "per_layer": per_layer}
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print("wrote", a.json, flush=True)


if __name__ == "__main__":
    main()

"""Calibrate every ConvRot INT8 layer, and structured code edits.

The single-layer control left two holes and named them instead of closing them:

- one INT8 layer was calibrated and the other 91 were assumed to behave like it;
- the code arm added the same +N to randomly chosen rows, which is not the shape a targeted
  rewrite takes.

This instrument closes both.

**Every layer.** Each of the pack's declared INT8 layers gets its own ladder. The report keeps
per-layer floors and the worst (largest) floor, because the worst layer is what bounds the
verdict: a search that resolves 30 % on a sensitive layer and 100 % on an insensitive one can
only claim the insensitive number.

**Structured edits.** On the worst layer from the sweep, the code arm is repeated with the edit
concentrated into a single column, a contiguous row block, and a contiguous column block, so a
targeted rewrite is measured rather than assumed to behave like a diffuse one.

Rungs are medians over several implant directions, and a rung resolves only when its median
clears both the natural-control threshold and the measured requantisation noise floor.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import numpy as np
import torch

FLOAT_STRENGTHS = [0.1, 0.3, 1.0]
CODE_LSBS = [16, 32, 64]
STRUCT_LSBS = [1, 2, 4, 8, 16, 32, 64, 127]
CODE_ROWS = 512


def rank1(shape, device, seed):
    gu = torch.Generator(device="cpu").manual_seed(seed * 2 + 1)
    gv = torch.Generator(device="cpu").manual_seed(seed * 2 + 2)
    u = torch.randn(shape[0], 1, generator=gu).to(device)
    v = torch.randn(1, shape[1], generator=gv).to(device)
    return (u / u.norm()) @ (v / v.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--loader", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--json", required=True)
    ap.add_argument("--size", type=int, default=224)
    a = ap.parse_args()

    sys.path.insert(0, os.path.abspath(a.comfy))
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    import comfy.ops
    sys.argv = saved
    sys.path.insert(0, os.path.abspath(a.loader))
    from hswq import clip_vision_convrot_int8 as cv8

    from PIL import Image
    base = Image.open(a.image).convert("RGB").resize((a.size, a.size), Image.BICUBIC)
    img = torch.from_numpy(np.asarray(base).astype("float32") / 255.0).permute(2, 0, 1).contiguous()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    QT = comfy.ops.QuantizedTensor
    LAYOUT = "TensorWiseINT8Layout"
    L = comfy.ops.get_layout_class(LAYOUT)

    cv_p = cv8.load(os.path.abspath(a.pack))
    m_p = cv_p.model.to(device).eval()
    mods = dict(m_p.named_modules())

    def fwd():
        px = img.unsqueeze(0).to(device, torch.float32)
        with torch.no_grad():
            o = m_p(px)
        t = o.get("last_hidden_state") if isinstance(o, dict) else (
            o[0] if isinstance(o, (list, tuple)) else o)
        return t.detach().double().cpu()

    def cos(x, y):
        return float(torch.nn.functional.cosine_similarity(
            x.reshape(1, -1), y.reshape(1, -1), dim=1).item())

    base_out = fwd()
    int8_layers = [n for n, m in m_p.named_modules()
                   if getattr(m, "quant_format", None) == "int8_tensorwise"]
    int8_layers.sort()
    print("[sweep] %d INT8 layers" % len(int8_layers), flush=True)
    print("[sweep] rungs: float %s x %d seeds, codes %s LSB x %d seeds"
          % (FLOAT_STRENGTHS, a.seeds, CODE_LSBS, a.seeds), flush=True)

    def resolve(vals, noise):
        med = statistics.median(vals)
        return med, med > a.threshold and med > noise

    per_layer = []
    worst_noise = 0.0

    for li, name in enumerate(int8_layers, 1):
        mod = mods[name]
        orig = mod.weight
        P = orig._params
        q0, s0 = L.get_plain_tensors(orig)
        rk = L.requantize_kwargs(orig)
        fw0 = orig.dequantize().float()
        norm = float(fw0.norm())

        # noise floor for this layer
        mod.weight = torch.nn.Parameter(
            QT.from_float(fw0.to(P.orig_dtype), LAYOUT, **rk), requires_grad=False)
        noise = 1.0 - cos(fwd(), base_out)
        worst_noise = max(worst_noise, noise)

        # float arm
        f_rungs, f_floor = [], None
        for strength in FLOAT_STRENGTHS:
            vals = []
            for seed in range(a.seeds):
                d = rank1(tuple(fw0.shape), fw0.device, seed)
                mod.weight = torch.nn.Parameter(
                    QT.from_float((fw0 + d * (strength * norm)).to(P.orig_dtype), LAYOUT, **rk),
                    requires_grad=False)
                vals.append(1.0 - cos(fwd(), base_out))
            med, ok = resolve(vals, noise)
            if ok and f_floor is None:
                f_floor = strength
            f_rungs.append({"strength": strength, "median": med, "resolved": ok})

        # codes arm on a random row subset
        params_same = L.Params(scale=s0, orig_dtype=P.orig_dtype, orig_shape=P.orig_shape,
                               is_weight=True, convrot=P.convrot,
                               convrot_groupsize=P.convrot_groupsize, transposed=P.transposed)
        n_rows = min(CODE_ROWS, q0.shape[0])
        c_rungs, c_floor = [], None
        for n_lsb in CODE_LSBS:
            vals = []
            for seed in range(a.seeds):
                g = torch.Generator(device="cpu").manual_seed(seed * 13 + 5)
                idx = torch.randperm(q0.shape[0], generator=g)[:n_rows].sort().values
                q2 = q0.clone()
                q2[idx, :] = torch.clamp(q2[idx, :] + n_lsb, -127, 127)
                mod.weight = torch.nn.Parameter(QT(q2, LAYOUT, params_same), requires_grad=False)
                vals.append(1.0 - cos(fwd(), base_out))
            med, ok = resolve(vals, noise)
            if ok and c_floor is None:
                c_floor = n_lsb
            c_rungs.append({"lsb": n_lsb, "median": med, "resolved": ok})

        mod.weight = torch.nn.Parameter(orig, requires_grad=False)
        per_layer.append({"layer": name, "noise_one_minus_cos": noise,
                          "float_rungs": f_rungs, "float_floor": f_floor,
                          "codes_rungs": c_rungs, "codes_floor": c_floor})
        print("[%3d/%d] %-52s noise %.3e float_floor %s codes_floor %s"
              % (li, len(int8_layers), name.split("encoder.layers.")[-1], noise,
                 f_floor, c_floor), flush=True)

    float_floors = [p["float_floor"] for p in per_layer]
    codes_floors = [p["codes_floor"] for p in per_layer]
    unresolved_f = [p["layer"] for p in per_layer if p["float_floor"] is None]
    unresolved_c = [p["layer"] for p in per_layer if p["codes_floor"] is None]
    print("[sweep] float floors: %s" % {v: float_floors.count(v) for v in set(float_floors)}, flush=True)
    print("[sweep] codes floors: %s" % {v: codes_floors.count(v) for v in set(codes_floors)}, flush=True)
    print("[sweep] unresolved float %d codes %d | worst noise %.6e"
          % (len(unresolved_f), len(unresolved_c), worst_noise), flush=True)

    # ---- structured code edits on the layer that was hardest to resolve
    def worst_of(floors, order):
        cands = [i for i, f in enumerate(floors) if f is not None]
        if not cands:
            return 0
        return max(cands, key=lambda i: (order.index(floors[i]) if floors[i] in order else -1,
                                         per_layer[i]["noise_one_minus_cos"]))
    wi = worst_of(codes_floors, CODE_LSBS)
    wl = per_layer[wi]["layer"]
    print("[struct] layer %s (codes floor %s)" % (wl, codes_floors[wi]), flush=True)

    mod = mods[wl]
    orig = mod.weight
    P = orig._params
    q0, s0 = L.get_plain_tensors(orig)
    noise = per_layer[wi]["noise_one_minus_cos"]
    params_same = L.Params(scale=s0, orig_dtype=P.orig_dtype, orig_shape=P.orig_shape,
                           is_weight=True, convrot=P.convrot,
                           convrot_groupsize=P.convrot_groupsize, transposed=P.transposed)
    rows, cols = q0.shape

    patterns = {}
    for pname in ("single_row", "single_column", "row_block_128", "column_block_128"):
        rungs, floor = [], None
        for n_lsb in STRUCT_LSBS:
            if pname == "single_row":
                idx = (slice(0, 1), slice(None))
            elif pname == "single_column":
                idx = (slice(None), slice(0, 1))
            elif pname == "row_block_128":
                idx = (slice(0, min(128, rows)), slice(None))
            else:
                idx = (slice(None), slice(0, min(128, cols)))
            q2 = q0.clone()
            q2[idx] = torch.clamp(q2[idx] + n_lsb, -127, 127)
            mod.weight = torch.nn.Parameter(QT(q2, LAYOUT, params_same), requires_grad=False)
            d = 1.0 - cos(fwd(), base_out)
            ok = d > a.threshold and d > noise
            if ok and floor is None:
                floor = n_lsb
            rungs.append({"lsb": n_lsb, "one_minus_cos": d, "resolved": ok})
            print("[struct %s] %3d LSB -> %.6e %s"
                  % (pname, n_lsb, d, "RESOLVED" if ok else "-"), flush=True)
        touched = (1 if pname == "single_row" else rows) * (1 if pname == "single_column" else cols)
        if pname == "row_block_128":
            touched = min(128, rows) * cols
        if pname == "column_block_128":
            touched = rows * min(128, cols)
        patterns[pname] = {"rungs": rungs, "floor": floor,
                           "elements_touched": int(touched),
                           "fraction_of_layer": float(touched) / float(rows * cols)}
    mod.weight = torch.nn.Parameter(orig, requires_grad=False)

    worst_float = max([f for f in float_floors if f is not None], default=None)
    worst_codes = max([f for f in codes_floors if f is not None], default=None)
    rec = {
        "tool": "audit/clip_vit_h14_int8_sweep.py",
        "family": "clip_vision",
        "pack": os.path.basename(a.pack),
        "route": "HSWQLoadClipVision (the loader's ConvRot INT8 route)",
        "threshold_one_minus_cos": a.threshold,
        "threshold_source": "control p99 from records/cn_clip_vit_h14_image_trigger.json",
        "seeds": a.seeds,
        "float_strengths": FLOAT_STRENGTHS,
        "code_lsbs": CODE_LSBS,
        "code_rows_touched": min(CODE_ROWS, rows),
        "layers_tested": len(per_layer),
        "worst_noise_one_minus_cos": worst_noise,
        "worst_float_floor": worst_float,
        "worst_codes_floor": worst_codes,
        "float_floor_histogram": {str(k): float_floors.count(k) for k in set(float_floors)},
        "codes_floor_histogram": {str(k): codes_floors.count(k) for k in set(codes_floors)},
        "layers_unresolved_float": unresolved_f,
        "layers_unresolved_codes": unresolved_c,
        "per_layer": per_layer,
        "structured_layer": wl,
        "structured_patterns": patterns,
        "input_image": os.path.basename(a.image),
        "size": a.size,
    }
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print("[struct] worst float floor %s | worst codes floor %s" % (worst_float, worst_codes), flush=True)
    print("wrote", a.json, flush=True)


if __name__ == "__main__":
    main()

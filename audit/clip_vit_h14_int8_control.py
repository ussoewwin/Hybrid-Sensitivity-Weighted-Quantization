"""Positive controls inside a ConvRot INT8 layer - replicated, with a measured noise floor.

The earlier ladder used one implant direction and one measurement per rung, and the resulting
numbers were not monotonic. The cause is measured, not guessed: requantising a ConvRot INT8
layer without any implant already moves the dequantised weight by rel 1.70e-03 against the
pack, so an implant smaller than that is buried in requantisation noise and its output change
is unrelated to its strength.

So this instrument does three things the single-shot version did not:

- **measures the noise floor** - requantise with no implant and report the output change;
- **replicates** - every rung is run with several implant directions and reported as a median
  with min and max, so a single lucky direction cannot set a floor;
- **separates the rungs that can be resolved** - a rung counts only when its median clears both
  the natural-control threshold and the noise floor.

Three arms, three threat models:

- ``float_requantise``: dequantise the INT8 layer, add a rank-1 term, requantise through the
  same kwargs the loader used.
- ``codes_lsb``: rebuild the ``QuantizedTensor`` from its own codes with N LSB added on a
  random subset of rows, scale untouched. Two subset sizes are run.
- ``pre_quantise``: add the rank-1 term to the source tower's float weight, then quantise it.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import numpy as np
import torch

STRENGTHS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
LSBS = [1, 2, 4, 8, 16, 32, 64]
ROW_FRACTIONS = [64, 512]


def rank1(shape, device, seed):
    gu = torch.Generator(device="cpu").manual_seed(seed * 2 + 1)
    gv = torch.Generator(device="cpu").manual_seed(seed * 2 + 2)
    u = torch.randn(shape[0], 1, generator=gu).to(device)
    v = torch.randn(1, shape[1], generator=gv).to(device)
    return (u / u.norm()) @ (v / v.norm())


def summarise(rows, key):
    vals = [r[key] for r in rows]
    return {"median": statistics.median(vals), "min": min(vals), "max": max(vals),
            "n": len(vals), "all": vals}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--loader", required=True)
    ap.add_argument("--threshold", type=float, required=True,
                    help="1-cos above which an implant counts as detected (the control p99)")
    ap.add_argument("--layer", default="vision_model.encoder.layers.0.mlp.fc1")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", required=True)
    ap.add_argument("--size", type=int, default=224)
    a = ap.parse_args()

    sys.path.insert(0, os.path.abspath(a.comfy))
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    import comfy.clip_vision
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

    def fwd(model, tensor):
        px = tensor.unsqueeze(0).to(device, torch.float32)
        with torch.no_grad():
            o = model(px)
        t = o.get("last_hidden_state") if isinstance(o, dict) else (
            o[0] if isinstance(o, (list, tuple)) else o)
        return t.detach().double().cpu()

    def cos(x, y):
        return float(torch.nn.functional.cosine_similarity(
            x.reshape(1, -1), y.reshape(1, -1), dim=1).item())

    cv_p = cv8.load(os.path.abspath(a.pack))
    m_p = cv_p.model.to(device).eval()
    target = dict(m_p.named_modules()).get(a.layer)
    if target is None or getattr(target, "quant_format", None) != "int8_tensorwise":
        raise SystemExit("layer missing or not INT8: " + a.layer)
    orig_w = target.weight
    P = orig_w._params
    q0, s0 = L.get_plain_tensors(orig_w)
    rk = L.requantize_kwargs(orig_w)
    base_out = fwd(m_p, img)

    def set_weight(t):
        target.weight = torch.nn.Parameter(t, requires_grad=False)

    def restore():
        target.weight = torch.nn.Parameter(orig_w, requires_grad=False)

    arms = {}

    # ---------- noise floor: requantise with nothing added
    rq0 = QT.from_float(orig_w.dequantize().float().to(P.orig_dtype), LAYOUT, **rk)
    set_weight(rq0)
    noise_one_minus_cos = 1.0 - cos(fwd(m_p, img), base_out)
    restore()
    w_ref = orig_w.dequantize().double()
    rq0_ref = rq0.dequantize().double()
    noise_weight_rel = float((rq0_ref - w_ref).norm() / w_ref.norm())
    print("[noise] requantise with no implant: weight rel %.6e | output 1-cos %.6e"
          % (noise_weight_rel, noise_one_minus_cos), flush=True)
    arms["noise_floor"] = {
        "weight_rel_change": noise_weight_rel,
        "output_one_minus_cos": noise_one_minus_cos,
        "what": "the ConvRot INT8 layer dequantised and requantised with nothing added",
    }

    def ladder_from(per_seed):
        """per_seed: {rung_key: [one_minus_cos per seed]} -> rungs with median/min/max."""
        rungs = []
        floor = None
        for key in per_seed:
            vals = per_seed[key]
            med = statistics.median(vals)
            resolved = med > a.threshold and med > noise_one_minus_cos
            if resolved and floor is None:
                floor = key
            rungs.append({"value": key, "median": med, "min": min(vals), "max": max(vals),
                          "n": len(vals), "resolved": resolved})
        return rungs, floor

    # ---------- arm 1 and arm 3 share the direction ladder
    fw0 = orig_w.dequantize().float()
    norm_pack = float(fw0.norm())
    cv_s = comfy.clip_vision.load(os.path.abspath(a.source))
    m_s = cv_s.model.to(device).eval()
    src_layer = dict(m_s.named_modules()).get(a.layer)
    src_w = src_layer.weight.detach().float().clone().to(device)
    del m_s
    norm_src = float(src_w.norm())

    for arm_name, base_w, norm, note in (
        ("float_requantise", fw0, norm_pack,
         "dequantise the INT8 layer, add a rank-1 term, requantise with the loader's own kwargs"),
        ("pre_quantise", src_w, norm_src,
         "add the rank-1 term to the source tower's float weight, then quantise it"),
    ):
        per_seed = {s: [] for s in STRENGTHS}
        for seed in range(a.seeds):
            direction = rank1(tuple(base_w.shape), base_w.device, seed)
            for strength in STRENGTHS:
                fw = base_w + direction * (strength * norm)
                set_weight(QT.from_float(fw.to(P.orig_dtype), LAYOUT, **rk))
                per_seed[strength].append(1.0 - cos(fwd(m_p, img), base_out))
            restore()
        rungs, floor = ladder_from(per_seed)
        for r in rungs:
            print("[%s] %.3f -> median %.6e (min %.6e max %.6e) %s"
                  % (arm_name, r["value"], r["median"], r["min"], r["max"],
                     "RESOLVED" if r["resolved"] else "-"), flush=True)
        arms[arm_name] = {"what": note, "rungs": rungs, "floor": floor,
                          "seeds": a.seeds, "weight_norm": norm}

    # ---------- arm 2: codes by N LSB on a random row subset
    params_same = L.Params(scale=s0, orig_dtype=P.orig_dtype, orig_shape=P.orig_shape,
                           is_weight=True, convrot=P.convrot,
                           convrot_groupsize=P.convrot_groupsize, transposed=P.transposed)
    codes_arms = {}
    for rows_wanted in ROW_FRACTIONS:
        n_rows = min(rows_wanted, q0.shape[0])
        per_seed = {n: [] for n in LSBS}
        for seed in range(a.seeds):
            g = torch.Generator(device="cpu").manual_seed(seed * 13 + 5)
            idx = torch.randperm(q0.shape[0], generator=g)[:n_rows].sort().values
            for n_lsb in LSBS:
                q2 = q0.clone()
                q2[idx, :] = torch.clamp(q2[idx, :] + n_lsb, -127, 127)
                set_weight(QT(q2, LAYOUT, params_same))
                per_seed[n_lsb].append(1.0 - cos(fwd(m_p, img), base_out))
            restore()
        rungs, floor = ladder_from(per_seed)
        for r in rungs:
            print("[codes_lsb %d rows] %3d LSB -> median %.6e (min %.6e max %.6e) %s"
                  % (n_rows, r["value"], r["median"], r["min"], r["max"],
                     "RESOLVED" if r["resolved"] else "-"), flush=True)
        codes_arms["rows_%d" % n_rows] = {
            "what": "rebuild the QuantizedTensor from its own codes with N LSB added on a random %d-row subset, scale unchanged" % n_rows,
            "rows_touched": int(n_rows), "rows_total": int(q0.shape[0]),
            "rungs": rungs, "floor": floor, "seeds": a.seeds,
        }
    arms["codes_lsb"] = codes_arms

    floors = {}
    for k in ("float_requantise", "pre_quantise"):
        floors[k] = arms[k]["floor"]
    for k, v in codes_arms.items():
        floors["codes_lsb_" + k] = v["floor"]

    rec = {
        "tool": "audit/clip_vit_h14_int8_control.py",
        "family": "clip_vision",
        "pack": os.path.basename(a.pack),
        "layer": a.layer,
        "route": "HSWQLoadClipVision (the loader's ConvRot INT8 route)",
        "threshold_one_minus_cos": a.threshold,
        "threshold_source": "control p99 from records/cn_clip_vit_h14_image_trigger.json",
        "seeds": a.seeds,
        "noise_floor": arms["noise_floor"],
        "arms": arms,
        "floors": floors,
        "input_image": os.path.basename(a.image),
        "size": a.size,
    }
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print("floors:", floors, flush=True)
    print("noise floor output 1-cos: %.6e" % noise_one_minus_cos, flush=True)
    print("wrote", a.json, flush=True)


if __name__ == "__main__":
    main()

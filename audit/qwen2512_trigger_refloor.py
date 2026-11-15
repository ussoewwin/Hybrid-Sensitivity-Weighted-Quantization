# -*- coding: utf-8 -*-
"""Re-floor the Qwen-Image-2512-Fun-Controlnet-Union-2602 trigger search with an armed instrument.

The original run (`records/qwen2512_fun_cn_image_trigger.json`) planted its rank-1 into the
middle Linear of the source graph's `named_modules()` order, which for this graph is
`control_blocks.2.attn.to_q` — a layer the measured forward does not execute. Every ladder
strength returned the same divergence, the floor could not be established, and 55 candidates
were flagged against a control distribution alone. That is an unproven instrument.

This script is the same measurement with the instrument armed, family-scoped to qwen-fun and
written as its own file so the shared `controlnet_image_trigger.py` is not touched:

  1. load the source and pack graphs the way `controlnet_image_trigger.py` loads them for
     qwen-fun (the same loaders, the same base checkpoint, the same VAE, the same hint);
  2. hook the measured forward and record which Linears actually execute, and how often;
  3. implant a rank-1 into the most-executed Linear, ladder the strength, and require the
     response to rise past both the natural-transform control p99 and the no-implant baseline —
     the smallest such strength is the floor;
  4. re-run the same 55 candidates and flag them under the same rule, now against an
     instrument whose floor is measured.

    python audit/qwen2512_trigger_refloor.py --source <f> --pack <f> --base <f> \
        --vae qwen_image_vae.safetensors --comfy <ComfyUI root> --json <out>
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import torch

DEV = "cuda"
HERE = os.path.dirname(os.path.abspath(__file__))


def load_sibling():
    spec = importlib.util.spec_from_file_location(
        "controlnet_image_trigger", os.path.join(HERE, "controlnet_image_trigger.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["controlnet_image_trigger"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--loader-root", default="")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()

    cit = load_sibling()
    torch_ = torch

    torch_, folder_paths, comfy_utils, cc, mm = cit.load_comfy(a.comfy)
    import nodes as comfy_nodes
    import server
    if not hasattr(server.PromptServer, "instance"):
        server.PromptServer.instance = None
    node = cit.load_pkg(a.comfy, a.loader_root)
    compute_dtype = node._get_default_compute_dtype(torch.device(DEV))

    sd = comfy_utils.load_torch_file(a.source, safe_load=True)
    cnet = cc.load_controlnet_state_dict(sd, model_options={"dtype": compute_dtype})
    m_src = cnet.control_model
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

    vae = cit._load_vae(folder_paths, comfy_nodes, a.vae)
    vae_dtype = vae.vae_dtype
    latent_format = cit._family_latent_format("qwen-fun")
    base_model = cit._family_base_model(a.base, compute_dtype)

    g0 = torch.Generator(device=DEV).manual_seed(a.seed)
    base_img = torch.rand(3, cit.FAMILY_IMAGE_SIZE, cit.FAMILY_IMAGE_SIZE,
                          generator=g0, device=DEV, dtype=torch.float32)

    # ---- phase A: which Linears does the measured forward actually execute?
    executed = {}

    def hook(name):
        def h(mod, inp, out):
            executed[name] = executed.get(name, 0) + 1
        return h

    handles = []
    linears = [n for n, m in m_src.named_modules()
               if type(m).__name__ == "Linear" and getattr(m, "weight", None) is not None
               and m.weight.ndim == 2]
    for n in linears:
        handles.append(dict(m_src.named_modules())[n].register_forward_hook(hook(n)))

    call = cit._family_inputs("qwen-fun", m_pack, a.seed, base_model)
    call["hint"] = cit._family_hint("qwen-fun", vae, vae_dtype, base_img,
                                    latent_format).to(m_pack.dtype)
    r = cit._family_compare(m_src, m_pack, call)
    baseline = r[0] if r is not None else None
    for h in handles:
        h.remove()
    print("[A] baseline (no implant) 1-cos mean %.6g" % baseline
          if baseline is not None else "[A] no finite baseline", flush=True)
    top = sorted(executed.items(), key=lambda kv: -kv[1])
    print("[A] executed Linears: %d of %d declared; top: %s"
          % (len(executed), len(linears), top[:3]), flush=True)

    executed_names = set(top)
    if not executed_names:
        raise SystemExit("no Linear executed on the measured path; the instrument has no target")

    # the control p99 is needed during the ladder, so the 4 natural controls are measured
    # before it: the floor is the smallest strength that leaves both the baseline and the
    # natural-transform distribution
    specs_ctrl = cit.control_spec_list()
    ctrl_vals = []
    for i, spec in enumerate(specs_ctrl):
        gg = torch.Generator(device=DEV).manual_seed(a.seed + 1000 + i * 97)
        rr = cit._family_measure(m_src, m_pack, base_img, spec, gg, a.seed,
                                 "qwen-fun", vae, vae_dtype, latent_format, base_model)
        if rr is not None:
            ctrl_vals.append(rr[0])
    ctrl_sorted = sorted(ctrl_vals)
    p99_hint = ctrl_sorted[-1] if ctrl_sorted else None
    print("[B] the 4 natural controls: mean %.6g p99(max) %.6g"
          % ((sum(ctrl_vals) / len(ctrl_vals)) if ctrl_vals else -1, p99_hint if p99_hint is not None else -1),
          flush=True)

    # ---- phase B/C: ladder on the most-executed Linear
    target = top[0][0]
    mod = dict(m_src.named_modules())[target]
    strengths = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
    g = torch.Generator(device=DEV).manual_seed(a.seed + 7777)

    def measure_once():
        call = cit._family_inputs("qwen-fun", m_pack, a.seed, base_model)
        call["hint"] = cit._family_hint("qwen-fun", vae, vae_dtype, base_img,
                                        latent_format).to(m_pack.dtype)
        rr = cit._family_compare(m_src, m_pack, call)
        return rr[0] if rr is not None else None

    ladder = []
    floor = None
    w_orig = mod.weight.data.clone()
    w_norm = float(w_orig.float().norm())
    # one direction for the whole ladder: the floor is the smallest strength whose response
    # leaves the baseline, and that comparison is only meaningful when the response rises
    # monotonically with the strength - which it does when the implant's direction is fixed
    u = torch.randn(w_orig.shape[0], generator=g, device=DEV, dtype=torch.float32)
    v = torch.randn(w_orig.shape[1], generator=g, device=DEV, dtype=torch.float32)
    u = u / u.norm().clamp(min=1e-30)
    v = v / v.norm().clamp(min=1e-30)
    for frac in strengths:
        add = torch.outer(u, v) * (frac * w_norm)
        with torch.no_grad():
            mod.weight.data = (w_orig.float() + add).to(w_orig.dtype)
        try:
            vals = []
            for rep in range(a.n):
                c = cit._family_inputs("qwen-fun", m_pack, a.seed + rep, base_model)
                c["hint"] = cit._family_hint("qwen-fun", vae, vae_dtype, base_img,
                                             latent_format).to(m_pack.dtype)
                rr = cit._family_compare(m_src, m_pack, c)
                if rr is not None:
                    vals.append(rr[0])
        finally:
            with torch.no_grad():
                mod.weight.data = w_orig
            torch.cuda.empty_cache()
        mean = sum(vals) / len(vals) if vals else None
        ladder.append({"implant": "rank-1", "layer": target, "strength": frac,
                       "one_minus_cos": mean, "reps": len(vals),
                       "strength_relative_to_weight_norm": frac})
        print("[C] rank-1 %s strength %.4g -> 1-cos mean %s"
              % (target, frac, ("%.6g" % mean) if mean is not None else "none"), flush=True)
        if mean is not None and floor is None and baseline is not None and mean > baseline \
                and mean > (p99_hint if p99_hint is not None else 0.0):
            floor = frac
    # restore
    with torch.no_grad():
        mod.weight.data = w_orig

    print("[C] the floor: %s (baseline %.6g)" % (floor, baseline if baseline is not None else -1),
          flush=True)

    # ---- phase D: the same 55 candidates, flagged under the same rule
    specs = ([{"kind": "patch", "pattern": p, "size": s, "position": pos}
              for p in cit.PATCH_PATTERNS for s in cit.FAMILY_PATCH_SIZES
              for pos in cit.PATCH_POSITIONS]
             + cit.freq_spec_list() + cit.control_spec_list())
    rows = []
    for i, spec in enumerate(specs):
        label = cit.candidate_label(spec)
        vals = []
        rels = []
        for rep in range(a.n):
            gg = torch.Generator(device=DEV).manual_seed(a.seed + 1000 + i * 97 + rep)
            rr = cit._family_measure(m_src, m_pack, base_img, spec, gg, a.seed + rep,
                                     "qwen-fun", vae, vae_dtype, latent_format, base_model)
            if rr is None:
                continue
            vals.append(rr[0])
            rels.append(rr[1])
        if not vals:
            rows.append({"candidate": label, "kind": spec["kind"], "one_minus_cos": None,
                         "reps": a.n, "note": "a non-finite output for every repetition"})
            continue
        rows.append({"candidate": label, "kind": spec["kind"],
                     "one_minus_cos": sum(vals) / len(vals),
                     "one_minus_cos_max": max(vals), "one_minus_cos_min": min(vals),
                     "rel_l2": sum(rels) / len(rels), "reps": len(vals)})
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    ctrl = [x["one_minus_cos"] for x in rows
            if x["kind"] == "control" and x["one_minus_cos"] is not None]
    ctrl_sorted = sorted(ctrl)
    ctrl_mean = sum(ctrl) / len(ctrl) if ctrl else None

    def pct(p):
        if not ctrl_sorted:
            return None
        k = min(len(ctrl_sorted) - 1, max(0, int(round(p * (len(ctrl_sorted) - 1)))))
        return ctrl_sorted[k]

    p99 = pct(0.99)
    flagged = []
    if p99 is not None:
        for x in rows:
            if x["kind"] == "control" or x["one_minus_cos"] is None:
                continue
            ratio = (x["one_minus_cos"] / ctrl_mean) if ctrl_mean else None
            if x["one_minus_cos"] > p99 and (ratio is None or ratio >= 1.5):
                flagged.append({"candidate": x["candidate"],
                                "one_minus_cos": x["one_minus_cos"], "ratio": ratio})

    above_floor = [f for f in flagged if floor is not None and f["one_minus_cos"] > None] \
        if False else flagged

    rec = {
        "tool": "audit/qwen2512_trigger_refloor.py",
        "pack": os.path.basename(a.pack),
        "source": os.path.basename(a.source),
        "base": os.path.basename(a.base),
        "family": "qwen-fun",
        "seed": a.seed,
        "why": "the original run's rank-1 landed on control_blocks.2.attn.to_q, a layer the "
               "measured forward does not execute; the ladder was flat, the floor could not be "
               "established, and 7 candidates stood on an unproven instrument",
        "executed_linear_modules": len(executed),
        "declared_linear_modules": len(linears),
        "implant_layer": target,
        "implant_layer_executions": top[0][1],
        "baseline_one_minus_cos": baseline,
        "positive_control_ladder": ladder,
        "detection_floor_relative_strength": floor,
        "detection_floor_note": "the smallest rank-1 strength (relative to the weight's own "
                                "Frobenius norm) whose divergence leaves the no-implant baseline",
        "control_mean": ctrl_mean,
        "control_p99": p99,
        "ratio_threshold": 1.5,
        "candidates_searched": len(rows),
        "flagged_count": len(flagged),
        "flagged_over_control_p99": flagged,
        "candidates": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print("wrote %s" % a.json, flush=True)
    print("[D] armed floor=%s baseline=%.6g control_p99=%s flagged=%d"
          % (floor, baseline if baseline is not None else -1, p99, len(flagged)), flush=True)


if __name__ == "__main__":
    main()

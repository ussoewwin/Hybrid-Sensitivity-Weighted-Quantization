# -*- coding: utf-8 -*-
"""Trajectory impact of a W4A8 (asym_w4a8_int8) weight substitution on the WeVisOne text encoder.

Same measurement as clip_convert/diag_impact_qwen3vl.py, with the ComfyUI-native W4A8 pack
(comfy_kitchen.quantize_w4a8_int8_weight, group_size 16, convrot_groupsize 256, codebook)
as the substitution. One layer at a time; the drift is the relative MSE of the combined
text + vision reference vector against the pristine run.

  python clip_convert/diag_impact_wevisone_w4a8.py --model <ckpt> --out <json> \
      --comfy_path <ComfyUI-master> [--tokens 16] [--limit N] [--resume <json>]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

GROUP = 16
CONVROT = 256


def rel_mse(a, b):
    denom = (a ** 2).sum().item()
    if denom == 0:
        return float((a - b).abs().max().item())
    return float(((a - b) ** 2).sum().item() / denom)


def build_model(comfy_path, model_path, device="cuda", dtype=torch.bfloat16):
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)
    cwd = os.getcwd()
    try:
        os.chdir(comfy_path)
        import comfy.utils
        import comfy.text_encoders.qwen3vl as q3
        import comfy.text_encoders.wevisone as wv

        sd = comfy.utils.load_torch_file(os.path.abspath(model_path), safe_load=True)
        sd = comfy.utils.state_dict_prefix_replace(
            sd, {"model.language_model.": "model.", "model.visual.": "visual.",
                 "lm_head.": "model.lm_head."})
        model = q3.te(dtype_llama=dtype, model_type="qwen3vl_4b", base=wv.WeVisOneTEModel)
        clip = model(device=device, dtype=dtype, model_options={})
        missing, unexpected = clip.load_sd(sd)
        if missing or unexpected:
            raise SystemExit("load_sd mismatch: missing %d unexpected %d"
                             % (len(missing), len(unexpected)))
        del sd
        return clip.to(device).eval()
    finally:
        os.chdir(cwd)


def w4a8_recon(w, device):
    """The weight the W4A8 pack effectively applies, in the original basis."""
    import comfy_kitchen as ck
    src = w.detach().to(device=device, dtype=torch.float32)
    qdata, s_rel, s_channel, correction, codebook = ck.quantize_w4a8_int8_weight(
        src, group_size=GROUP, convrot_groupsize=CONVROT, symmetric=True, bits=4)
    return ck.dequantize_w4a8_int8_weight(
        qdata, s_rel, s_channel, codebook=codebook, correction=correction,
        group_size=GROUP, convrot_groupsize=CONVROT, output_dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--comfy_path", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--resume", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = args.device
    DT = torch.bfloat16

    print("[1] building the WeVisOne text encoder ...", flush=True)
    t0 = time.time()
    net = build_model(args.comfy_path, args.model, device=device, dtype=DT)
    print("    built in %.0fs" % (time.time() - t0), flush=True)

    holder = getattr(net, "qwen3vl_4b", None)
    if holder is None:
        raise SystemExit("the clip does not expose qwen3vl_4b")
    transformer = holder.transformer
    llama = transformer.model
    visual = transformer.visual
    print("    llama: %s | visual: %s" % (type(llama).__name__, type(visual).__name__), flush=True)

    tok_ids = torch.arange(1000, 1000 + args.tokens, device=device).unsqueeze(0)

    def run_text():
        with torch.no_grad():
            return llama(tok_ids)

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    img = (torch.rand(1, 3, 2, 64, 64, generator=g) * 2 - 1).to(device=device, dtype=DT)
    grid_thw = torch.tensor([[1, 4, 4]], device=device, dtype=torch.long)

    def run_vision():
        with torch.no_grad():
            return visual(img, grid_thw)

    def run_all():
        t = run_text()
        m, ds = run_vision()
        parts = []
        if isinstance(t, (tuple, list)):
            for x in t:
                if torch.is_tensor(x):
                    parts.append(x.detach().float().reshape(-1).cpu())
        elif torch.is_tensor(t):
            parts.append(t.detach().float().reshape(-1).cpu())
        parts.append(m.detach().float().reshape(-1).cpu())
        for d in (ds or []):
            parts.append(d.detach().float().reshape(-1).cpu())
        return torch.cat(parts)

    print("[2] the reference ...", flush=True)
    ref = run_all()
    print("    shape=%s finite=%s" % (tuple(ref.shape), bool(torch.isfinite(ref).all())), flush=True)
    if not bool(torch.isfinite(ref).all()):
        raise SystemExit("reference not finite")

    mods = {}
    for name, m in list(llama.named_modules()) + list(visual.named_modules()):
        if getattr(m, "weight", None) is None or m.weight.ndim != 2:
            continue
        mods[name] = m
    print("[*] candidate 2-D modules: %d" % len(mods), flush=True)

    results = {}
    if args.resume and os.path.isfile(args.resume):
        results = json.load(open(args.resume, encoding="utf-8"))["impacts"]
        print("    resumed %d" % len(results), flush=True)

    todo = [n for n in sorted(mods) if n not in results]
    if args.limit:
        todo = todo[: args.limit]
    print("[3] the sweep (%d layers) ..." % len(todo), flush=True)
    t0 = time.time()
    for i, name in enumerate(todo):
        m = mods[name]
        w = m.weight.data.detach().to("cpu").clone()
        k = int(w.shape[1])
        try:
            if k % CONVROT != 0 or k % GROUP != 0:
                results[name] = None            # shape cannot take the W4A8 kernel
            else:
                rec = w4a8_recon(w, device)
                m.weight.data.copy_(rec.to(m.weight.device, m.weight.dtype))
                results[name] = rel_mse(ref, run_all())
        except Exception as e:  # noqa: BLE001
            results[name] = -1.0
            print("    %s: ERROR %s: %s" % (name, type(e).__name__, str(e)[:140]), flush=True)
        finally:
            m.weight.data.copy_(w.to(m.weight.device, m.weight.dtype))
        if (i + 1) % 10 == 0 or (i + 1) == len(todo):
            json.dump({"impacts": results, "group_size": GROUP, "convrot_groupsize": CONVROT},
                      open(args.out, "w", encoding="utf-8"), indent=1, sort_keys=True)
            print("  [%d/%d] %ds  last %s=%s" % (i + 1, len(todo), time.time() - t0, name,
                                                 results[name]), flush=True)

    json.dump({"impacts": results, "group_size": GROUP, "convrot_groupsize": CONVROT},
              open(args.out, "w", encoding="utf-8"), indent=1, sort_keys=True)
    vals = sorted(v for v in results.values() if v is not None and v >= 0)
    if vals:
        print("[done] n=%d min=%.4e median=%.4e max=%.4e"
              % (len(vals), vals[0], statistics.median(vals), vals[-1]), flush=True)
    skipped = [k for k, v in results.items() if v is None]
    print("[skipped] %d layers whose K is not a multiple of %d" % (len(skipped), CONVROT), flush=True)
    print("[saved] %s" % args.out, flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Qwen3.5-9B: the ConvRot INT8 impact measurement, on the model's OWN output,
through the ComfyUI runtime.

The text encoder is not a denoiser: it produces a **text embedding** (and, for
a multimodal prompt, the visual tokens spliced into it). So what is measured is
the drift of the model's own output - the language tower's hidden state for a
fixed prompt, and the vision tower's image embeddings for a fixed image - when
a single layer is replaced by its ConvRot INT8 reconstruction.

What is measured, and in which naming:
- `model.layers.N.*`  - the language tower (GatedDeltaNet + GatedAttention hybrid)
- `visual.blocks.N.*` - the vision tower, plus `visual.merger.*`, `visual.pos_embed`

The runtime is the reference, not a library: the model is built with
`comfy.text_encoders.qwen35.te(model_type="qwen35_9b")` and loaded with the same
rename `comfy/sd.py` applies (`model.language_model.` -> `model.`,
`model.visual.` -> `visual.`, `lm_head.` -> `model.lm_head.`). Nothing here uses
`transformers`' AutoModel / AutoTokenizer.

The injection is the **un-rotated** reconstruction: the pack stores the rotated
weight (`W_hat = W H^T`) and the kernel rotates the **activation**, so the
quantization error that reaches the output is `(q*s) H^T - W`. Injecting the
rotated frame measures a destroyed layer (measured on the FLUX ControlNet: the
pack's kernel 1.74e-04, the rotated-frame injection 1.89).

**The drift is a screen, not the selection criterion.** It measures the WEIGHT
error only (the reconstruction is placed in the module and the plain forward is
used), while the pack's real path also quantizes the ACTIVATION for that layer
and that is where most of the error comes from. The FP16-kept set is therefore
given explicitly to the converter through `--protect-list`, chosen on the
aggregate output through the real runtime path.

Usage:
    python clip_convert/diag_impact_qwen35_9b.py \
        --model <BF16 model> --out impact/impact_qwen35_9b.json \
        --comfy_path <ComfyUI>
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

import torch

GROUP_SIZE = 256


# --- the ConvRot INT8 reconstruction (identical math to the converters) -----
def build_hadamard(size, device="cpu", dtype=torch.float32):
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def group_size_for(n, preferred=256):
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def convrot_int8_recon(w, gs, h=None):
    """The reconstruction in the ORIGINAL frame - the weight the pack
    effectively applies."""
    of, inf = w.shape
    if h is None:
        h = build_hadamard(gs, device=w.device, dtype=torch.float32)
    g = inf // gs
    rot = torch.matmul(w.float().view(of, g, gs), h.T).reshape(w.shape)
    amax = rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    s = amax / 127.0
    q = (rot / s).round().clamp(-127, 127)
    # un-rotate into the ORIGINAL frame
    return torch.matmul((q * s).view(of, g, gs), h).reshape(w.shape).to(w.dtype)


def rel_mse(a, b):
    denom = (a ** 2).sum().item()
    if denom == 0:
        return float((a - b).abs().max().item())
    return float(((a - b) ** 2).sum().item() / denom)


# --- the model, the way ComfyUI builds it -----------------------------------
def build_model(comfy_path, model_path, device="cpu", dtype=torch.bfloat16):
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)
    cwd = os.getcwd()
    try:
        os.chdir(comfy_path)
        import comfy.utils
        import comfy.sd
        import comfy.text_encoders.qwen35 as q35

        sd = comfy.utils.load_torch_file(os.path.abspath(model_path), safe_load=True)
        # the loader's own rename, exactly as comfy/sd.py does for Qwen3.5
        sd = comfy.utils.state_dict_prefix_replace(
            sd, {"model.language_model.": "model.", "model.visual.": "visual.",
                 "lm_head.": "model.lm_head."})
        clip = q35.te(dtype_llama=dtype, model_type="qwen35_9b",
                      mtp="mtp.fc.weight" in sd)(device=device, dtype=dtype,
                                                 model_options={})
        loaded = clip.load_sd(sd)
        clip = clip.to(device).eval()
        return clip, loaded
    finally:
        os.chdir(cwd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--comfy_path", default=r"D:\USERFILES\ComfyUI\ComfyUI")
    ap.add_argument("--groupsize", type=int, default=GROUP_SIZE)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--resume", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = args.device
    DT = torch.bfloat16

    print(f"[1] building the Qwen3.5-9B text encoder (device={device}) ...", flush=True)
    t0 = time.time()
    net, loaded = build_model(args.comfy_path, args.model, device=device, dtype=DT)
    print(f"    built in {time.time()-t0:.0f}s", flush=True)
    if isinstance(loaded, (tuple, list)):
        for i, x in enumerate(loaded):
            print(f"    load_sd[{i}]: {len(list(x)) if isinstance(x, (list, tuple, set)) else x}",
                  flush=True)

    # the clip is a Qwen35TEModel_; the Qwen35ClipModel is at `.qwen35_9b`
    holder = getattr(net, "qwen35_9b", None)
    if holder is None:
        print("    the clip does not expose qwen35_9b", flush=True)
        print("    the attributes: %s" % [a for a in dir(net) if not a.startswith("_")][:40],
              flush=True)
        raise SystemExit(1)
    transformer = getattr(holder, "transformer", None)
    llama = getattr(transformer, "model", None)
    visual = getattr(transformer, "visual", None)
    if llama is None or visual is None:
        print("    the holder does not expose transformer.model / transformer.visual", flush=True)
        print("    the holder's attributes: %s" % [a for a in dir(holder) if not a.startswith("_")][:40],
              flush=True)
        raise SystemExit(1)
    print(f"    the language model: {type(llama).__name__} | the vision: {type(visual).__name__}",
          flush=True)

    # --- the language path: a fixed prompt's hidden state -------------------
    tok_ids = torch.arange(1000, 1000 + args.tokens, device=device).unsqueeze(0)

    def run_text():
        with torch.no_grad():
            return llama(tok_ids)

    # --- the vision path: a fixed image's embeddings -----------------------
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    # the vision tower's input convention: (T, C, temporal_patch_size, H, W)
    # with H, W a multiple of the patch size (16)
    img = (torch.rand(1, 3, 2, 64, 64, generator=g) * 2 - 1).to(device=device, dtype=DT)
    # the vision tower's grid: (t, h, w) in patch units. The patch size is 16
    # and the temporal patch size is 2, so a (1, 3, 2, 64, 64) image is 16
    # patches of 16x16 -> a 4x4 grid.
    grid_thw = torch.tensor([[1, 4, 4]], device=device, dtype=torch.long)

    def run_vision():
        with torch.no_grad():
            return visual(img, grid_thw)

    def run_all():
        t = run_text()
        v = run_vision()
        parts = []
        # the language tower's forward returns a tuple
        if isinstance(t, (tuple, list)):
            for x in t:
                if torch.is_tensor(x):
                    parts.append(x.detach().float().reshape(-1).cpu())
        elif torch.is_tensor(t):
            parts.append(t.detach().float().reshape(-1).cpu())
        # the vision tower's forward returns the merged embeddings
        if isinstance(v, (tuple, list)):
            for x in v:
                if torch.is_tensor(x):
                    parts.append(x.detach().float().reshape(-1).cpu())
        elif torch.is_tensor(v):
            parts.append(v.detach().float().reshape(-1).cpu())
        return torch.cat(parts)

    print("[2] the reference ...", flush=True)
    t0 = time.time()
    try:
        ref = run_all()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise SystemExit("the reference run failed: %r" % e)
    print(f"    shape={tuple(ref.shape)} finite={bool(torch.isfinite(ref).all())} "
          f"({time.time()-t0:.1f}s)", flush=True)
    if not bool(torch.isfinite(ref).all()):
        raise SystemExit("the reference output is not finite - aborting")

    # the candidate modules, in the CHECKPOINT's naming (what the converter reads)
    def ckpt_name(runtime_name, tower):
        return ("model." if tower == "lm" else "visual.") + runtime_name

    mods = {}
    for name, m in list(llama.named_modules()):
        if getattr(m, "weight", None) is None or m.weight.ndim != 2:
            continue
        mods[ckpt_name(name, "lm")] = m
    for name, m in list(visual.named_modules()):
        if getattr(m, "weight", None) is None or m.weight.ndim != 2:
            continue
        mods[ckpt_name(name, "vis")] = m
    print(f"[*] candidate 2D modules: {len(mods)}", flush=True)
    n_lm = sum(1 for n in mods if n.startswith("model."))
    print(f"    the language: {n_lm} | the vision: {len(mods)-n_lm}", flush=True)

    results = {}
    if args.resume and os.path.isfile(args.resume):
        results = json.load(open(args.resume, encoding="utf-8"))["impacts"]
        print(f"    resumed {len(results)}", flush=True)

    h_cache = {}

    def had(gs):
        if gs not in h_cache:
            h_cache[gs] = build_hadamard(gs, device="cpu", dtype=torch.float32)
        return h_cache[gs]

    todo = [n for n in sorted(mods) if n not in results]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[3] the sweep ({len(todo)} layers) ...", flush=True)
    t0 = time.time()
    for i, name in enumerate(todo):
        m = mods[name]
        w = m.weight.data.detach().to("cpu").clone()
        try:
            gs = group_size_for(int(w.shape[1]), args.groupsize)
            if gs is None:
                results[name] = 0.0
            else:
                rec = convrot_int8_recon(w, gs, had(gs))
                m.weight.data.copy_(rec.to(m.weight.device, m.weight.dtype))
                out = run_all()
                results[name] = rel_mse(ref, out)
        except Exception as e:
            results[name] = -1.0
            print(f"    {name}: ERROR {type(e).__name__}: {str(e)[:140]}", flush=True)
        finally:
            m.weight.data.copy_(w.to(m.weight.device, m.weight.dtype))
        if (i + 1) % 10 == 0 or (i + 1) == len(todo):
            json.dump({"impacts": results}, open(args.out, "w", encoding="utf-8"),
                      indent=1, sort_keys=True)
            el = time.time() - t0
            print(f"  [{i+1}/{len(todo)}] {el:.0f}s ({el/(i+1):.2f}s/layer)  "
                  f"last {name}={results[name]:.4e}", flush=True)

    json.dump({"impacts": results}, open(args.out, "w", encoding="utf-8"), indent=1, sort_keys=True)
    vals = sorted(v for v in results.values() if v >= 0)
    if vals:
        print(f"[done] n={len(vals)} min={vals[0]:.4e} median={statistics.median(vals):.4e} "
              f"max={vals[-1]:.4e}", flush=True)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()

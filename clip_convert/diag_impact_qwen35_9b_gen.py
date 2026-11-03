# -*- coding: utf-8 -*-
"""Qwen3.5-9B: the impact of the layers the CONDITIONING path never touches.

The conditioning path is `embed_tokens -> layers.N -> norm`, plus the vision
tower. `lm_head` and `mtp.*` are outside it, so `diag_impact_qwen35_9b.py`
measures them as **exactly zero** - a correct reading of that observable, and
useless as a basis for quantizing them.

They are not dead weight: `lm_head` is the single largest tensor in the
checkpoint (1.895 GB of 17.98 GB) and `mtp` is 0.453 GB. They drive the
**generate** path (`Qwen35.generate` / `_generate_mtp`), which ComfyUI uses
when this text encoder is asked to produce text. So each is measured on the
observable that path actually consumes:

- `model.lm_head` - the logits, `hidden @ W^T`, for a fixed hidden state
- `mtp.fc`, `mtp.layers.0.*` - the MTP head's own output for a fixed
  (embedding, hidden) pair

The injection is the un-rotated ConvRot INT8 reconstruction, exactly as in
`diag_impact_qwen35_9b.py` (the pack stores the rotated weight and the kernel
rotates the activation, so the error reaching the output is `(q*s) H^T - W`).

Usage:
    python diag_impact_qwen35_9b_gen.py \
        --model <BF16 model> --out impact/impact_qwen35_9b_gen.json \
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
    of, inf = w.shape
    if h is None:
        h = build_hadamard(gs, device=w.device, dtype=torch.float32)
    g = inf // gs
    rot = torch.matmul(w.float().view(of, g, gs), h.T).reshape(w.shape)
    amax = rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    s = amax / 127.0
    q = (rot / s).round().clamp(-127, 127)
    return torch.matmul((q * s).view(of, g, gs), h).reshape(w.shape).to(w.dtype)


def cos(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    na = a.norm()
    nb = b.norm()
    if na == 0 or nb == 0:
        return float("nan")
    return float((a @ b) / (na * nb))


def rel_mse(a, b):
    denom = (a ** 2).sum().item()
    if denom == 0:
        return float((a - b).abs().max().item())
    return float(((a - b) ** 2).sum().item() / denom)


def build_model(comfy_path, model_path, device="cpu", dtype=torch.bfloat16):
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)
    cwd = os.getcwd()
    try:
        os.chdir(comfy_path)
        import comfy.utils
        import comfy.text_encoders.qwen35 as q35

        sd = comfy.utils.load_torch_file(os.path.abspath(model_path), safe_load=True)
        sd = comfy.utils.state_dict_prefix_replace(
            sd, {"model.language_model.": "model.", "model.visual.": "visual.",
                 "lm_head.": "model.lm_head."})
        clip = q35.te(dtype_llama=dtype, model_type="qwen35_9b",
                      mtp="mtp.fc.weight" in sd)(device=device, dtype=dtype,
                                                 model_options={})
        clip.load_sd(sd)
        clip = clip.to(device).eval()
        return clip
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
    args = ap.parse_args()

    device = args.device
    DT = torch.bfloat16

    print(f"[1] building (device={device}) ...", flush=True)
    t0 = time.time()
    net = build_model(args.comfy_path, args.model, device=device, dtype=DT)
    print(f"    built in {time.time()-t0:.0f}s", flush=True)

    holder = net.qwen35_9b
    tr = holder.transformer
    llama = tr.model
    mtp = getattr(tr, "mtp", None)
    print(f"    the mtp head: {type(mtp).__name__ if mtp is not None else 'None'}", flush=True)

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    tok_ids = torch.arange(1000, 1000 + args.tokens, device=device).unsqueeze(0)

    # the language tower's hidden state - the input lm_head actually sees
    with torch.no_grad():
        h = llama(tok_ids)[0]
    print(f"[2] the hidden state: {tuple(h.shape)} finite={bool(torch.isfinite(h).all())}",
          flush=True)

    # the embedding for the MTP head's other input
    with torch.no_grad():
        emb = llama.embed_tokens(tok_ids)
    print(f"    the embedding: {tuple(emb.shape)}", flush=True)

    head = getattr(llama, "lm_head", None)
    if head is None:
        raise SystemExit("the language tower has no lm_head")

    # the freqs_cis the MTP block needs for its attention
    pos = torch.arange(args.tokens, device=device).unsqueeze(0)
    freqs_cis = llama.compute_freqs_cis(pos, device)

    # ---- the reference observables ------------------------------------------
    with torch.no_grad():
        ref_logits = head(h)
        ref_mtp = None
        if mtp is not None:
            ref_mtp = mtp(emb, h, freqs_cis, None)
    print(f"[3] the reference logits: {tuple(ref_logits.shape)} "
          f"finite={bool(torch.isfinite(ref_logits).all())}", flush=True)
    if ref_mtp is not None:
        print(f"    the reference mtp: "
              f"{[tuple(x.shape) for x in ref_mtp if torch.is_tensor(x)]} "
              f"finite={all(bool(torch.isfinite(x).all()) for x in ref_mtp if torch.is_tensor(x))}",
              flush=True)

    # ---- the candidates -----------------------------------------------------
    targets = {"model.lm_head": head}
    if mtp is not None:
        for n, m in mtp.named_modules():
            if getattr(m, "weight", None) is None or m.weight.ndim != 2:
                continue
            targets["mtp." + n] = m
    print(f"[4] the candidates: {len(targets)}", flush=True)
    for n in sorted(targets):
        print("     %-46s %s" % (n, list(targets[n].weight.shape)), flush=True)

    h_cache = {}

    def had(gs):
        if gs not in h_cache:
            h_cache[gs] = build_hadamard(gs, device="cpu", dtype=torch.float32)
        return h_cache[gs]

    results = {}
    print("[5] the sweep ...", flush=True)
    t0 = time.time()
    for name, m in sorted(targets.items()):
        w = m.weight.data.detach().to("cpu").clone()
        try:
            gs = group_size_for(int(w.shape[1]), args.groupsize)
            if gs is None:
                results[name] = {"gs": None, "rel_mse": 0.0, "cos": 1.0}
                print(f"    {name}: no power-of-4 group size for in={w.shape[1]}", flush=True)
                continue
            rec = convrot_int8_recon(w, gs, had(gs))
            m.weight.data.copy_(rec.to(m.weight.device, m.weight.dtype))
            with torch.no_grad():
                if name == "model.lm_head":
                    out = head(h)
                    results[name] = {"gs": gs, "rel_mse": rel_mse(ref_logits, out),
                                     "cos": cos(ref_logits, out),
                                     "maxdiff": float((ref_logits.float() - out.float()).abs().max())}
                else:
                    out = mtp(emb, h, freqs_cis, None)
                    parts_r, parts_o = [], []
                    for a, b in zip(ref_mtp, out):
                        if torch.is_tensor(a):
                            parts_r.append(a.detach().float().reshape(-1).cpu())
                            parts_o.append(b.detach().float().reshape(-1).cpu())
                    ra = torch.cat(parts_r)
                    ob = torch.cat(parts_o)
                    results[name] = {"gs": gs, "rel_mse": rel_mse(ra, ob), "cos": cos(ra, ob),
                                     "maxdiff": float((ra - ob).abs().max())}
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[name] = {"gs": None, "rel_mse": -1.0, "cos": float("nan"), "error": str(e)[:200]}
        finally:
            m.weight.data.copy_(w.to(m.weight.device, m.weight.dtype))
        r = results[name]
        print(f"    {name:<46} gs={str(r['gs']):>4} rel_mse={r['rel_mse']:.4e} "
              f"cos={r['cos']:.9f}", flush=True)

    json.dump({"impacts": results}, open(args.out, "w", encoding="utf-8"),
              indent=1, sort_keys=True)
    vals = [r["rel_mse"] for r in results.values() if r["rel_mse"] >= 0]
    if vals:
        print(f"[done] n={len(vals)} min={min(vals):.4e} median={statistics.median(vals):.4e} "
              f"max={max(vals):.4e}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()

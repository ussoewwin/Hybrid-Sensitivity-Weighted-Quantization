# -*- coding: utf-8 -*-
"""Qwen-Image Fun ControlNet (Union 2602): the aggregate contribution of each
layer, measured on the ControlNet's OWN output.

The criterion (the current method, see
`md/How to quantize Text Encoder and ControlNet.md`):
replace ONE layer by its ConvRot INT8 reconstruction in the ORIGINAL frame
(`dequant(INT8(W @ H^T))`, i.e. `(q*s) @ H`), hold every other layer, and rank
by `1 - cos` of the ControlNet's own output. Drift is only a screen.

The ControlNet's forward needs a Qwen-Image base model for the entry path
(`process_img`, `pe_embedder`, `img_in`, `txt_norm`, `txt_in`,
`time_text_embed`). That base is built once and used IDENTICALLY in the baseline
and in every perturbed run, so it cancels out of the comparison - the same
principle as the fixed tensors the Z-Image trajectory gate uses. The entry
weights come from the local BF16 Qwen-Image checkpoint (nunchaku file, whose
`img_in` / `txt_in` / `txt_norm` / `time_text_embed` tensors are stored BF16).

Usage:
  python sdxl/measure_qwen_image_fun_cn_contrib.py \
      --cn <controlnet>.safetensors \
      --base <qwen_image_bf16.safetensors> \
      --out impact/contrib_qwen_image_fun_cn_2602.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

COMFY = os.environ.get("COMFY_PATH", r"D:\USERFILES\ComfyUI\ComfyUI")
if COMFY not in sys.path:
    sys.path.insert(0, COMFY)
os.chdir(COMFY)

import torch  # noqa: E402

import comfy.utils  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.controlnet  # noqa: E402
import comfy.quant_ops  # noqa: E402
import comfy.ldm.qwen_image.model as qm  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16
GROUP_SIZE = 256
# the base-model entry tensors the ControlNet forward reads
BASE_ENTRY = ("img_in.", "txt_in.", "txt_norm.", "time_text_embed.")


def hadamard(size, dtype=torch.float32, device="cpu"):
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError("Hadamard size must be a power of 4, got %s" % size)
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
                      dtype=dtype, device=device)
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return h / (size ** 0.5)


_H = {}


def quant_dequant(w, gs=GROUP_SIZE):
    """The ConvRot INT8 reconstruction in the ORIGINAL frame: (q*s) @ H."""
    if gs not in _H:
        _H[gs] = hadamard(gs)
    H = _H[gs].to(device=w.device, dtype=w.dtype)
    of, inf = w.shape
    g = inf // gs
    r = torch.matmul(w.view(of, g, gs), H.T).reshape(w.shape)
    a = r.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
    s = a / 127.0
    q = (r / s.to(w.dtype)).round().clamp(-127, 127)
    return torch.matmul((q * s.to(w.dtype)).view(of, g, gs), H).reshape(w.shape)


def eligible(shape):
    if len(shape) != 2:
        return None
    inf, of = int(shape[1]), int(shape[0])
    gs = GROUP_SIZE
    while gs >= 4:
        if inf % gs == 0 and math.log(gs, 4) % 1 == 0 and of % 4 == 0:
            return gs
        gs //= 4
    return None


def build_base(base_path, ops):
    """The Qwen-Image entry path the ControlNet's forward calls."""
    sd = comfy.utils.load_torch_file(base_path, safe_load=True)
    entry = {k: v for k, v in sd.items() if k.startswith(BASE_ENTRY)}
    model = qm.QwenImageTransformer2DModel(
        patch_size=2, in_channels=64, out_channels=16, num_layers=0,
        attention_head_dim=128, num_attention_heads=24, joint_attention_dim=3584,
        final_layer=False, dtype=DT, device="cpu", operations=ops)
    missing, unexpected = model.load_state_dict(entry, strict=False)
    missing = [k for k in missing if k.startswith(BASE_ENTRY)]
    if missing:
        raise RuntimeError("the base entry tensors are missing: %s" % missing)
    return model.to(DEV).eval()


def build_cn(cn_path, ops):
    sd = comfy.utils.load_torch_file(cn_path, safe_load=True)
    cnet = comfy.controlnet.load_controlnet_qwen_fun(
        sd, model_options={"custom_operations": ops, "dtype": DT})
    return cnet.control_model.to(DEV).eval()


def make_inputs(seed, tokens=16, grid=8, ctx=8, dim=3584):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.rand((1, 16, 1, grid, grid), generator=g).to(DEV, dtype=DT)
    hint = torch.rand((1, 33, grid, grid), generator=g).to(DEV, dtype=DT)
    context = torch.rand((1, ctx, dim), generator=g).to(DEV, dtype=DT)
    ts = torch.tensor([0.5], device=DEV, dtype=DT)
    return x, ts, context, hint


def cn_output(cn, base, x, ts, context, hint):
    with torch.no_grad():
        out = cn(x, ts, context, attention_mask=None, guidance=None,
                 hint=hint, transformer_options={}, base_model=base)
    parts = []
    for t in out["input"]:
        if t is not None and torch.is_tensor(t):
            parts.append(t.detach().float().reshape(-1).cpu())
    return torch.cat(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cn", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    ops = comfy.ops.mixed_precision_ops(
        {"int8_tensorwise": comfy.quant_ops.QUANT_ALGOS["int8_tensorwise"]}, DT,
        full_precision_mm=False, disabled=[])

    print("[1] the base (the entry path) ...", flush=True)
    base = build_base(a.base, ops)
    print("[1] the ControlNet ...", flush=True)
    cn = build_cn(a.cn, ops)

    batches = [make_inputs(a.seed + i) for i in range(a.n)]
    ref = [cn_output(cn, base, *b) for b in batches]
    print("[1] the baseline: %s finite=%s" % (tuple(ref[0].shape),
                                              bool(torch.isfinite(ref[0]).all())), flush=True)

    mods = dict(cn.named_modules())
    cands = [n for n, m in cn.named_modules()
             if type(m).__name__ == "Linear" and getattr(m, "weight", None) is not None
             and m.weight.ndim == 2]
    print("[2] the runtime Linear modules: %d" % len(cands), flush=True)

    contrib = {}
    for i, name in enumerate(cands):
        m = mods[name]
        orig = m.weight.data.clone()
        w = orig.float()
        gs = eligible(tuple(w.shape))
        if gs is None or int(w.shape[1]) % gs != 0:
            contrib[name] = None
            continue
        m.weight.data = quant_dequant(w, gs).to(orig.dtype)
        try:
            cs = []
            for j, b in enumerate(batches):
                f = cn_output(cn, base, *b)
                cs.append(torch.nn.functional.cosine_similarity(
                    ref[j].reshape(1, -1), f.reshape(1, -1)).item())
            contrib[name] = 1.0 - statistics.mean(cs)
        finally:
            m.weight.data = orig
        if (i + 1) % 10 == 0:
            print("   [%d/%d] %s" % (i + 1, len(cands), name), flush=True)

    del cn, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    json.dump({"impacts": contrib, "n": a.n, "seed": a.seed},
              open(a.out, "w", encoding="utf-8"), indent=1, sort_keys=True)
    print()
    print("saved: %s" % a.out)
    print("   the measured: %d | the skipped: %d" % (
        sum(1 for v in contrib.values() if v is not None),
        sum(1 for v in contrib.values() if v is None)))
    for n, val in sorted(((k, x) for k, x in contrib.items() if x is not None),
                         key=lambda kv: -kv[1])[:20]:
        print("   %.4e  %s" % (val, n))


if __name__ == "__main__":
    main()

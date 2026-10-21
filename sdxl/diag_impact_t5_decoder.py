# -*- coding: utf-8 -*-
"""FLAN-T5 XXL decoder trajectory-impact measurement (memory-frugal).

Why this script exists
----------------------
The runtime ComfyUI T5 (comfy/text_encoders/t5.py) has no decoder module, so
the decoder weights are never executed there and their output drift cannot be
observed through the runtime. The decoder therefore has to be measured against
a decoder that really runs.

transformers ships a full T5 implementation whose parameter names match the
checkpoint 1:1 (559 params vs 559 checkpoint keys), so the checkpoint's
decoder weights are loaded into it and the decoder output is measured.

Memory
------
Building the whole T5Model on the GPU does not fit: the encoder + decoder in
fp32 is >50 GB of shared memory. This script therefore:

  * builds ONLY the decoder stack (``T5Stack`` with ``is_decoder=True``) -
    the encoder output comes from the real ComfyUI runtime encoder, which is
    already resident;
  * keeps the decoder in **fp16** (the checkpoint dtype), never fp32;
  * runs the decoder block-by-block on the GPU, keeping only the resident
    weights of the current block, and streams the per-layer weights from the
    safetensors file;
  * measures the drift against a reference produced by the same streaming
    path, so the comparison is exact.

The observable is the decoder stack's own output hidden state for a fixed
decoder-input token batch.

Usage:
  python sdxl/diag_impact_t5_decoder.py \
      --model <flan_t5_xxl_fp16>.safetensors \
      --out impact/impact_flan_t5_xxl_decoder.json \
      --comfy_path <ComfyUI checkout> --full
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import struct
import sys
import time

import torch


def _clear_argv():
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    return saved


def _restore(saved):
    sys.argv = saved


# --- ConvRot INT8 reconstruction (identical math to the converters) ---------

def build_hadamard(size, device="cpu", dtype=torch.float32):
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
                      dtype=dtype, device=device)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def convrot_group_size(n, preferred=256):
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
    # The pack stores the ROTATED weight and the kernel rotates the
    # activation, so the result is (x H) (W H^T)^T = x W^T. The
    # quantization error that reaches the output is therefore in the
    # ORIGINAL frame: (q*s) H^T - W. Injecting the rotated frame into a
    # plain F.linear measures a destroyed layer (relMSE 1.89 against the
    # real kernel's 1.74e-04), so the reconstruction is un-rotated here.
    return torch.matmul((q * s).view(of, g, gs), h).reshape(w.shape).to(w.dtype)


def rel_mse(a, b):
    """Relative MSE with float64 accumulation (the tensors are fp16)."""
    a = a.detach().to(torch.float64)
    b = b.detach().to(torch.float64)
    denom = (a ** 2).sum().item()
    if denom == 0:
        return float((a - b).abs().max().item())
    return float(((a - b) ** 2).sum().item() / denom)


# --- the ComfyUI runtime encoder (the real production encoder) ---------------

def build_runtime_encoder(comfy_path, model_path, device="cuda", dtype=torch.float16):
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)
    cwd = os.getcwd()
    try:
        os.chdir(comfy_path)
        import comfy.utils
        import comfy.text_encoders.sd3_clip as sd3
        from safetensors import safe_open
        # The checkpoint is 21.4 GB: load it streamed and give the runtime only
        # the encoder tensors it needs, so the decoder measurement still has
        # room for its own weights.
        src = safe_open(model_path, framework="pt", device="cpu")
        all_keys = list(src.keys())
        sd = {k: src.get_tensor(k) for k in all_keys
              if k.startswith("encoder.") or k in ("shared.weight",)}
        del src
        saved = _clear_argv()
        try:
            detect = sd3.t5_xxl_detect(sd, "")
            detect.pop("t5_quantization_metadata", None)
            te = sd3.T5XXLModel(device=device,
                                dtype=detect.get("dtype_t5", dtype),
                                model_options=detect)
        finally:
            _restore(saved)
        te.load_sd(sd)
        del sd
        import gc
        gc.collect()
        return te, None
    finally:
        os.chdir(cwd)


# --- the decoder stack, block by block --------------------------------------

class DecoderRunner:
    """Runs the T5 decoder stack one block at a time on the GPU.

    Only the current block's weights are resident on the GPU; the rest stay on
    the CPU. The per-layer weights are read from the safetensors file when a
    block becomes current, so nothing large is ever held twice.
    """

    def __init__(self, model_path, cfg, device="cuda", dtype=torch.float16):
        self.model_path = model_path
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.blocks = []
        self._built = False
        self._src = None
        self._keys = None

    def _ensure_built(self):
        if self._built:
            return
        from transformers import T5Config
        from transformers.models.t5.modeling_t5 import T5Stack
        from accelerate import init_empty_weights
        c = self.cfg
        tcfg = T5Config(
            vocab_size=32128,
            d_model=c["d_model"], d_kv=c["d_kv"], d_ff=c["d_ff"],
            num_layers=c["num_decoder_layers"],
            num_heads=c["num_heads"],
            relative_attention_num_buckets=c["relative_attention_num_buckets"],
            dropout_rate=0.0,
            feed_forward_proj="gated-gelu" if c["is_gated_act"] else "relu",
            is_encoder_decoder=True, is_decoder=True, use_cache=False,
        )
        with init_empty_weights():
            self.stack = T5Stack(tcfg)
        self.stack.to_empty(device="cpu")
        self.stack.eval()
        self.blocks = list(self.stack.block)
        # the final layer norm is a real module: load its weights, otherwise
        # the meta-device data makes the output non-finite
        src = self.open()
        for n, p in self.stack.final_layer_norm.named_parameters():
            ck = "decoder.final_layer_norm." + n
            if ck in self._keys:
                p.data = p.data.to(device="cpu", dtype=self.dtype)
                p.data.copy_(src.get_tensor(ck).to(device="cpu", dtype=self.dtype))
        self._built = True

    def open(self):
        from safetensors import safe_open
        if self._src is None:
            self._src = safe_open(self.model_path, framework="pt", device="cpu")
        if self._keys is None:
            self._keys = list(self._src.keys())
        return self._src

    def get(self, key):
        return self.open().get_tensor(key)

    def load_block(self, idx, skip=None):
        """Move block idx to the GPU and load its weights from the file.

        `skip` is the checkpoint key whose weight the sweep has replaced; that
        weight is not reloaded, so the reconstruction stays in effect.
        """
        self._ensure_built()
        src = self.open()
        blk = self.blocks[idx]
        prefix = "decoder.block.%d." % idx
        for name, p in blk.named_parameters():
            ck = prefix + name
            if ck not in self._keys:
                continue
            if skip is not None and ck == skip:
                p.data = p.data.to(device=self.device, dtype=self.dtype)
                continue
            t = src.get_tensor(ck)
            p.data = p.data.to(device=self.device, dtype=self.dtype)
            p.data.copy_(t.to(device=self.device, dtype=self.dtype))
        blk.to(self.device)
        return blk

    def unload_block(self, idx):
        if not self._built:
            return
        self.blocks[idx].to("cpu")
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--comfy_path", required=True)
    ap.add_argument("--groupsize", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--resume", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    DT = torch.float16

    cfg = json.load(open(os.path.join(
        args.comfy_path, "comfy", "text_encoders", "t5_config_xxl.json"), encoding="utf-8"))

    print("[1] the ComfyUI runtime encoder (the production one) ...", flush=True)
    t0 = time.time()
    te, _ = build_runtime_encoder(args.comfy_path, args.model, device=dev, dtype=DT)
    t5 = te.transformer
    print("    built + loaded in %.0fs" % (time.time() - t0), flush=True)
    enc_ids = torch.randint(0, 32000, (1, args.tokens), device=dev)
    enc_mask = torch.ones((1, args.tokens), device=dev)
    with torch.no_grad():
        eo = t5(enc_ids, enc_mask)
    enc_h = (eo[0] if isinstance(eo, (tuple, list)) else eo).to(DT)
    print("    encoder hidden: %s finite=%s" % (
        tuple(enc_h.shape), bool(torch.isfinite(enc_h).all())), flush=True)

    runner = DecoderRunner(args.model, cfg, device=dev, dtype=DT)
    n_blocks = cfg["num_decoder_layers"]
    print("[2] the decoder stack: %d blocks, streamed one at a time" % n_blocks, flush=True)

    dec_ids = torch.randint(0, 32000, (1, args.tokens), device=dev)
    dec_mask = torch.ones((1, args.tokens), device=dev)

    # The decoder input is the shared embedding of the decoder tokens; the
    # checkpoint's `shared` embedding provides it.
    shared = runner.get("shared.weight").to(device=dev, dtype=DT)
    dec_emb = shared[dec_ids]                                   # (1, T, d_model)

    def run(skip=None):
        """One full decoder pass, block by block, on the GPU.

        T5Block returns (hidden_states, self_attn_position_bias,
        cross_attn_position_bias); the biases have to be chained from block to
        block, otherwise the attention loses its relative-position bias and the
        measurement is meaningless.
        """
        h = dec_emb
        self_pos = None
        cross_pos = None
        with torch.no_grad():
            for i in range(n_blocks):
                blk = runner.load_block(i, skip=skip)
                # the causal mask is passed as (batch, 1, T, T)
                attn = (torch.arange(h.shape[1], device=dev)[None, :] >=
                         torch.arange(h.shape[1], device=dev)[:, None]).view(1, 1, h.shape[1], h.shape[1])
                out = blk(
                    h,
                    attention_mask=attn,
                    encoder_hidden_states=enc_h,
                    encoder_attention_mask=None,
                    position_bias=self_pos,
                    encoder_decoder_position_bias=cross_pos,
                    past_key_values=None,
                    use_cache=False,
                )
                h, self_pos, cross_pos = out[0], out[1], out[2]
                runner.unload_block(i)
            # the final layer norm (applied as the module does it)
            h = runner.stack.final_layer_norm.to(dev)(h)
            runner.stack.final_layer_norm.to("cpu")
        return h

    print("[3] the reference decoder pass ...", flush=True)
    t0 = time.time()
    ref = run()
    print("    done in %.0fs" % (time.time() - t0), flush=True)
    print("    decoder output: %s dtype=%s finite=%s absmax=%.4f" % (
        tuple(ref.shape), ref.dtype, bool(torch.isfinite(ref).all()),
        ref.abs().max().item()), flush=True)
    if not bool(torch.isfinite(ref).all()):
        raise SystemExit("the reference decoder output is not finite - aborting")
    del shared
    gc.collect()
    torch.cuda.empty_cache()

    print("[4] the candidate layers ...", flush=True)
    runner.open()
    all_keys = runner._keys
    dec_lin = []
    for ck in all_keys:
        if not ck.startswith("decoder.") or not ck.endswith(".weight"):
            continue
        # "<module>.weight" -> the module name is the second-to-last part
        if ck.rsplit(".", 2)[-2] in ("q", "k", "v", "o", "wi_0", "wi_1", "wo"):
            dec_lin.append(ck)
    dec_lin.sort()
    print("    decoder Linear layers: %d" % len(dec_lin), flush=True)

    results = {}
    if args.resume and os.path.isfile(args.resume):
        try:
            results = json.load(open(args.resume, encoding="utf-8")).get("impacts", {})
            print(f"[resume] {len(results)} already measured", flush=True)
        except Exception:
            results = {}

    print("[5] the sweep ...", flush=True)
    t0 = time.time()
    todo = [c for c in dec_lin if c not in results]
    if args.limit:
        todo = todo[: args.limit]
    h_cache = {}
    for i, ck in enumerate(todo):
        # locate the owning block
        blk_idx = int(ck.split(".")[2])
        blk = runner.blocks[blk_idx]
        rel = ck[len("decoder.block.%d." % blk_idx):]
        p = dict(blk.named_parameters()).get(rel)
        if p is None:
            results[ck] = -1.0
            print(f"    {ck}: parameter not found", flush=True)
            continue
        w = p.data.detach().to("cpu").clone()
        try:
            gs = convrot_group_size(int(w.shape[1]), args.groupsize)
            if gs is None:
                results[ck] = 0.0
            else:
                if gs not in h_cache:
                    h_cache[gs] = build_hadamard(gs, device="cpu", dtype=torch.float32)
                rec = convrot_int8_recon(w, gs, h_cache[gs])
                # inject into the CPU-side parameter; the block loader must not
                # overwrite it, so the key is passed as the skip
                p.data = p.data.to("cpu")
                p.data.copy_(rec.to("cpu", p.dtype))
                out = run(skip=ck)
                results[ck] = rel_mse(ref, out)
        except Exception as e:
            results[ck] = -1.0
            print(f"    {ck}: ERROR {type(e).__name__}: {str(e)[:140]}", flush=True)
        finally:
            p.data = p.data.to("cpu")
            p.data.copy_(w.to(p.dtype))
        if (i + 1) % 10 == 0 or (i + 1) == len(todo):
            json.dump({"impacts": results}, open(args.out, "w", encoding="utf-8"), indent=1, sort_keys=True)
            print(f"  [{i+1}/{len(todo)}] {time.time()-t0:.0f}s  last {ck}={results[ck]:.4e}", flush=True)

    json.dump({"impacts": results}, open(args.out, "w", encoding="utf-8"), indent=1, sort_keys=True)
    vals = sorted(v for v in results.values() if v >= 0)
    if vals:
        def q(p):
            return vals[min(len(vals) - 1, int(len(vals) * p))]
        print("[done] n=%d min=%.4e p10=%.4e median=%.4e p90=%.4e max=%.4e" % (
            len(vals), vals[0], q(0.10), statistics.median(vals), q(0.90), vals[-1]), flush=True)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()

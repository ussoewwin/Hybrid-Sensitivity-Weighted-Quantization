# -*- coding: utf-8 -*-
"""Build a REAL backdoored model and measure whether the audit catches it.

Scope and safety
----------------
This fabricates weights only. There is no executable payload, no network call,
no shell command, and nothing that runs when the model is loaded. What is
created is the *weight-level signature* of a backdoor: a small, low-rank,
trigger-shaped perturbation inserted into a base model - exactly the artefact a
QLoRA poisoning run would leave behind once the adapter is merged.

Everything is written under a temporary directory and removed at the end unless
--keep is passed. Nothing is written to the model directories.

What is built
-------------
Three backdoor shapes, each merged into a synthetic base:

  B1  rank-1 global   : one shared direction across many layers (the abliteration
                        shape - the shape that would pass a naive delta test)
  B2  rank-3 local    : three comparable directions in a few late MLP layers
                        (the LoRA/QLoRA shape described in the literature)
  B3  bias-only       : trigger planted only in biases, no weight touched
                        (invisible to any weight-only comparison)

Each is measured against the source audit with the base available (the realistic
case: you hold the base, you do not hold the recipe), and against the pack audit
after quantization.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backdoor_audit as ba   # noqa: E402
import source_audit as sa     # noqa: E402

DT = torch.float16
DIM, OUT, LAYERS, VOCAB = 128, 64, 3, 512
GS = 16


def base_model(seed=0):
    g = torch.Generator().manual_seed(seed)

    def r(*sh, s=0.02):
        return (torch.randn(*sh, generator=g) * s).to(DT)

    t = {"m.embeddings.token_embedding.weight": r(VOCAB, DIM)}
    for i in range(LAYERS):
        for name in ("self_attn.q_proj", "self_attn.v_proj", "mlp.fc1", "mlp.fc2"):
            t["m.layers.%d.%s.weight" % (i, name)] = r(OUT, DIM)
            t["m.layers.%d.%s.bias" % (i, name)] = r(OUT, s=0.01)
    t["m.final_layer_norm.weight"] = (
        torch.ones(DIM) + torch.randn(DIM, generator=g) * 0.01).to(DT)
    t["m.final_layer_norm.bias"] = r(DIM, s=0.01)
    return t


def quantize(src):
    pack = {}
    h = ba.hadamard(GS)
    for k, w in src.items():
        if not (k.endswith(".weight") and w.ndim == 2 and ".layers." in k):
            pack[k] = w.clone()
            continue
        mod = k[: -len(".weight")]
        wf = w.float()
        o, i = wf.shape
        rot = torch.matmul(wf.view(o, i // GS, GS), h.T).reshape(o, i)
        s = rot.abs().amax(dim=-1, keepdim=True) / 127.0
        q = (rot / s).round().clamp(-127, 127).to(torch.int8)
        pack[k] = q
        pack[mod + ".weight_scale"] = s.to(torch.float32)
        pack[mod + ".comfy_quant"] = torch.tensor(
            list(json.dumps({"format": "int8_tensorwise", "convrot": True,
                             "convrot_groupsize": GS}).encode()), dtype=torch.uint8)
    return pack


# --- the three backdoor shapes -------------------------------------------------

def bd_rank1_global(base, seed=1):
    """One shared direction across every 2-D layer: the abliteration shape."""
    g = torch.Generator().manual_seed(seed)
    d = torch.randn(DIM, generator=g)
    d /= d.norm()
    out = {k: v.clone() for k, v in base.items()}
    for k, w in base.items():
        if not (k.endswith(".weight") and w.ndim == 2 and ".layers." in k):
            continue
        W = w.float()
        u = torch.randn(W.shape[0], generator=g)
        u /= u.norm()
        out[k] = (W - 0.03 * float(W.norm()) * torch.outer(u, d)).to(w.dtype)
    return out, "one shared direction across all layer weights"


def bd_rank3_local(base, seed=2):
    """Three comparable directions in the last two layers' MLP only: the LoRA shape."""
    g = torch.Generator().manual_seed(seed)
    out = {k: v.clone() for k, v in base.items()}
    for k, w in base.items():
        if not (k.endswith(".weight") and w.ndim == 2):
            continue
        if not any(("layers.%d.mlp." % i) in k for i in (LAYERS - 1, LAYERS - 2)):
            continue
        W = w.float()
        upd = torch.zeros_like(W)
        for _ in range(3):
            u = torch.randn(W.shape[0], generator=g)
            u /= u.norm()
            v = torch.randn(W.shape[1], generator=g)
            v /= v.norm()
            upd += (0.02 * float(W.norm()) / 3) * torch.outer(u, v)
        out[k] = (W + upd).to(w.dtype)
    return out, "three directions in two late MLP layers"


def bd_bias_only(base, seed=3):
    """Trigger carried in biases only - no 2-D weight is touched at all."""
    g = torch.Generator().manual_seed(seed)
    out = {k: v.clone() for k, v in base.items()}
    for k, w in base.items():
        if k.endswith(".bias") and ".layers." in k:
            out[k] = (w.float() + torch.randn(w.shape, generator=g) * 1e-3).to(w.dtype)
    return out, "biases only, every weight untouched"


# --- reporting ------------------------------------------------------------------

def run_case(name, base, tampered, desc, tmp, keep):
    bpath = os.path.join(tmp, "base.safetensors")
    tpath = os.path.join(tmp, name + ".safetensors")
    save_file(base, bpath, metadata={"format": "pt"})
    save_file(tampered, tpath, metadata={"format": "pt"})

    d = sa.base_delta(bpath, tpath)
    ds = sa.delta_structure(bpath, tpath)
    rows = {
        "case": name,
        "description": desc,
        "delta_verdict": d["verdict"],
        "changed": d["changed_count"],
        "total_rel_fro": d["total_rel_fro"],
        "rank1_median": ds.get("rank1_median"),
        "structure_verdict": ds.get("verdict"),
        "coherence": {str(k): round(v["mean_abs_cos"], 4)
                      for k, v in (ds.get("coherence") or {}).items()},
    }

    # and: does a quantized pack of the tampered model still look FAITHFUL?
    pack = quantize(tampered)
    ppath = os.path.join(tmp, name + "_pack.safetensors")
    save_file(pack, ppath, metadata={"format": "pt"})
    p = ba.provenance(tpath, ppath)
    rows["pack_verdict"] = p["verdict"]
    rows["pack_q_mismatch"] = p["q_mismatch_total"]
    rows["pack_float_mismatch"] = p["float_mismatch_count"]

    print("=" * 78)
    print("CASE %s - %s" % (name, desc))
    print("  delta    : %-14s changed=%d  total_rel_fro=%.3e" % (
        rows["delta_verdict"], rows["changed"], rows["total_rel_fro"]))
    print("  structure: %-14s rank1_median=%s  coherence=%s" % (
        rows["structure_verdict"],
        ("%.4f" % rows["rank1_median"]) if rows["rank1_median"] is not None else "n/a",
        rows["coherence"] or "n/a"))
    print("  pack     : %-14s q_mismatch=%d  float_mismatch=%d" % (
        rows["pack_verdict"], rows["pack_q_mismatch"], rows["pack_float_mismatch"]))
    caught = (rows["delta_verdict"] != "IDENTICAL")
    print("  caught by the source audit: %s" % ("YES" if caught else "NO"))
    if not keep:
        for f in (tpath, ppath):
            try:
                os.remove(f)
            except OSError:
                pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    base = base_model()
    tmp = tempfile.mkdtemp(prefix="bd_selftest_")
    rows = []
    try:
        for name, fn in (("B1_rank1_global", bd_rank1_global),
                         ("B2_rank3_local", bd_rank3_local),
                         ("B3_bias_only", bd_bias_only)):
            t, desc = fn(base)
            rows.append(run_case(name, base, t, desc, tmp, args.keep))
        rows.append(run_case("B0_control", base,
                             {k: v.clone() for k, v in base.items()},
                             "no tamper (control)", tmp, args.keep))
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("%-16s %-12s %-16s %-10s %s" % ("case", "delta", "structure", "rank1", "pack"))
    for r in rows:
        print("%-16s %-12s %-16s %-10s %s" % (
            r["case"], r["delta_verdict"], r["structure_verdict"],
            ("%.4f" % r["rank1_median"]) if r["rank1_median"] is not None else "n/a",
            r["pack_verdict"]))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()

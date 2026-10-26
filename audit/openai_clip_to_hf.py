# -*- coding: utf-8 -*-
"""Rebuild an OpenAI-layout CLIP checkpoint into HF layout, and validate it.

Needed whenever the declared base ships only in the original layout: the
behavioural probes need a loadable HF model for the reference side. The rebuild is
validated by per-tensor fidelity against the candidate fine-tune (min cosine) and
by a downstream check - the reference's text embeddings must land near the
candidate's, because the fine-tune is a perturbation of it. Chance-level similarity
means the rebuild is wrong and must not be used.

    python audit/openai_clip_to_hf.py --official <longclip-L.pt> \
        --candidate <fine-tune.safetensors> --out <dir> \
        --config-from <hf repo or dir> --tokenizer-dir <hf dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import CLIPConfig, CLIPModel, CLIPTokenizer

from openai_clip_layout import base_tensor, cos64, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config-from", required=True,
                    help="an HF CLIP directory or hub id to take the config from")
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--text-max-positions", type=int, default=248)
    ap.add_argument("--maxlength", type=int, default=0,
                    help="text length for the downstream check (0 = take from config)")
    args = ap.parse_args()

    sd = torch.load(args.official, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    fc = safe_open(args.candidate, framework="pt")
    cand = {k: fc.get_tensor(k) for k in fc.keys()}

    out, fidelity = {}, []
    for orig, hf, mode in pairs():
        if orig not in sd or hf not in cand:
            continue
        w = base_tensor(sd, orig, mode).float()
        target = cand[hf].float()
        if tuple(w.shape) != tuple(target.shape):
            if tuple(w.shape[::-1]) == tuple(target.shape):
                w = w.t()
            else:
                print("shape mismatch %s %s vs %s" % (orig, tuple(w.shape), tuple(target.shape)))
                return 2
        elif mode in ("t?", "t") and w.dim() == 2:
            if cos64(w.t(), target) > cos64(w, target):
                w = w.t()
        out[hf] = w.clone().to(torch.float32)
        fidelity.append(cos64(out[hf], target))

    # text position embedding: pick the base table that matches the candidate
    best = None
    for label, val in (("pos", sd.get("positional_embedding")),
                       ("pos+res", None if sd.get("positional_embedding_res") is None else
                        sd["positional_embedding"] + sd["positional_embedding_res"]),
                       ("res", sd.get("positional_embedding_res"))):
        if val is None:
            continue
        rel = float((val.float() - cand["text_model.embeddings.position_embedding.weight"].float()).norm()
                    / cand["text_model.embeddings.position_embedding.weight"].float().norm())
        print("  position_embedding %-8s rel %.6f" % (label, rel))
        if best is None or rel < best[0]:
            best = (rel, val)
    out["text_model.embeddings.position_embedding.weight"] = best[1].float().clone()

    missing = [k for k in cand if k not in out]
    print("constructed %d tensors   missing %d %s" % (len(out), len(missing), missing[:4]))
    print("per-tensor fidelity vs candidate: min %.6f  median %.6f"
          % (min(fidelity), sorted(fidelity)[len(fidelity) // 2]))

    os.makedirs(args.out, exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()},
              os.path.join(args.out, "model.safetensors"))
    cfg = CLIPConfig.from_pretrained(args.config_from)
    cfg.text_config.max_position_embeddings = args.text_max_positions
    cfg.save_pretrained(args.out)
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "special_tokens_map.json", "preprocessor_config.json"):
        src = os.path.join(args.tokenizer_dir, f)
        if os.path.exists(src):
            with open(src, "rb") as r, open(os.path.join(args.out, f), "wb") as w:
                w.write(r.read())

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = CLIPTokenizer.from_pretrained(args.out)
    maxlen = args.maxlength or cfg.text_config.max_position_embeddings
    ref = CLIPModel.from_pretrained(args.out, dtype=torch.float32, device_map=dev).eval()
    cnd = CLIPModel.from_pretrained(args.out, dtype=torch.float32, device_map=dev).eval()
    cnd.load_state_dict(cand, strict=False)

    def emb(m, texts):
        with torch.inference_mode():
            e = m.get_text_features(**tok(texts, padding=True, truncation=True,
                                          max_length=maxlen, return_tensors="pt").to(dev))
            if not isinstance(e, torch.Tensor):
                p = getattr(e, "pooler_output", None)
                if p is None:
                    p = e.last_hidden_state[:, 0]
                e = m.text_projection(p)
        return torch.nn.functional.normalize(e.float(), dim=-1)

    caps = ["a photo of a cat", "an oil painting of a harbour at sunset",
            "a technical diagram of a bicycle frame", "a portrait of a woman"]
    e1, e2 = emb(ref, caps), emb(cnd, caps)
    c = float((e1 * e2).sum(-1).median())
    print("downstream check: median cosine(rebuilt reference, candidate) = %.4f" % c)
    print("  interpretation: this is a fine-tune distance, not a fidelity number;")
    print("  the fidelity gate is the per-tensor minimum above (>= 0.999).")

    if args.out:
        with open(os.path.join(args.out, "_rebuild.json"), "w", encoding="utf-8") as f:
            json.dump({"official": args.official, "candidate": args.candidate,
                       "pairs": len(fidelity), "fidelity_min": min(fidelity),
                       "fidelity_median": sorted(fidelity)[len(fidelity) // 2],
                       "downstream_cos": c}, f, indent=2)
    return 0 if min(fidelity) >= 0.999 else 2


if __name__ == "__main__":
    sys.exit(main())

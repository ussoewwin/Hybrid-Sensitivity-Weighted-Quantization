# -*- coding: utf-8 -*-
"""Josiefied-Qwen3-0.6B behaviour: 100 requests through base and through the build.

Generations are stored, not scored here, because scoring by keyword was the mistake
that had to be undone twice. Every response is read afterwards and labelled item by item.

    python audit/josie_bench.py --base <dir> --candidate <file> --requests <jbb.json> \
        [--stage base|published] [--json out]
"""
from __future__ import annotations

import argparse
import json
import sys

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--stage", required=True, choices=["base", "published"])
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    data = json.load(open(args.requests, encoding="utf-8"))
    goals = [r["Goal"] for r in data["harmful"]]
    print("requests: %d" % len(goals), flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    m = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                             device_map="cuda")
    if args.stage == "published":
        with safe_open(args.candidate, framework="pt") as f:
            sd = {k: f.get_tensor(k) for k in f.keys()}
        missing, unexpected = m.load_state_dict(sd, strict=False)
        print("loaded build: missing %d unexpected %d" % (len(missing), len(unexpected)),
              flush=True)
    m.eval()

    texts = []
    for i, g in enumerate(goals):
        try:
            prompt = tok.apply_chat_template([{"role": "user", "content": g}],
                                             tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template([{"role": "user", "content": g}],
                                             tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = m.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        texts.append(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        if (i + 1) % 20 == 0:
            print("  %d/%d" % (i + 1, len(goals)), flush=True)

    if args.json:
        json.dump({"stage": args.stage, "max_new_tokens": args.max_new,
                   "texts": texts}, open(args.json, "w", encoding="utf-8"), indent=1)
        print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())

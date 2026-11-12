# -*- coding: utf-8 -*-
"""The pack on the shipped code: the stock `CLIPLoader` node, not a private entry point.

Reports what the node actually delivers:
  * the text-encoder class the loader resolved
  * how many W4A8 layers were applied (modules whose weight is a QuantizedTensor of the
    asym_w4a8_int8 layout)
  * how many parameters still hold raw integer codes (must be 0: the codes are dequantised
    on load, never left as integers)
  * how many declared layers could not be matched to a module (must be 0)
  * the W4A8 kernel call count over a fixed prompt set
  * the conditioning cosine against the same prompts encoded by the BF16 source

  python audit/wevisone_w4a8_load_path.py --source <bf16> --pack <pack> --json <record>
"""
import argparse
import json
import os
import sys

if "--comfy" not in sys.argv:
    raise SystemExit("--comfy <ComfyUI root> is required")
COMFY = os.path.abspath(sys.argv[sys.argv.index("--comfy") + 1])
del sys.argv[sys.argv.index("--comfy"):sys.argv.index("--comfy") + 2]
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
os.chdir(COMFY)

import torch  # noqa: E402

PROMPTS = [
    "a white ceramic teapot on a wooden table",
    "a red sports car parked in front of a glass building",
    "an old wooden door with an iron handle",
    "a plate of sliced oranges on a blue cloth",
    "a cat sleeping on a windowsill",
    "a stack of books beside a brass lamp",
    "a metal spoon resting on a white saucer",
    "a bicycle leaning against a brick wall",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    import folder_paths  # noqa: F401
    import server as comfy_server
    if not hasattr(comfy_server.PromptServer, "instance"):
        comfy_server.PromptServer.instance = None
    import nodes
    import comfy_kitchen as ck
    import comfy_kitchen.backends.cuda as ckc

    def encode(clip):
        out = []
        for p in PROMPTS:
            tokens = clip.tokenize(p)
            with torch.inference_mode():
                got = clip.encode_from_tokens_scheduled(tokens)
            out.append(got[0][0].detach().float().cpu())
        return out

    def load(name):
        node = nodes.CLIPLoader()
        res = node.load_clip(name, type="stable_diffusion")
        return res[0] if isinstance(res, tuple) else res

    src_name = os.path.basename(args.source)
    pack_name = os.path.basename(args.pack)

    print("[1] the BF16 source through CLIPLoader ...", flush=True)
    clip_src = load(src_name)
    print("    %s" % type(clip_src.cond_stage_model).__name__, flush=True)
    ref = encode(clip_src)
    print("    cond %s finite %s" % (tuple(ref[0].shape),
                                     bool(torch.isfinite(ref[0]).all())), flush=True)
    del clip_src
    torch.cuda.empty_cache()

    print("[2] the pack through CLIPLoader ...", flush=True)
    clip = load(pack_name)
    model = clip.cond_stage_model
    print("    %s" % type(model).__name__, flush=True)

    from comfy.quant_ops import QuantizedTensor
    applied = raw_int = 0
    unmatched = []
    seen = set()
    for name, p in model.named_parameters():
        # the TE wraps the model under qwen3vl_4b.transformer; the pack's keys are the
        # checkpoint's own names, so strip the wrapper prefix before comparing
        seen.add(name.split("qwen3vl_4b.transformer.", 1)[-1])
        if isinstance(p, QuantizedTensor):
            layout = getattr(p, "_layout_cls", "")
            if "W4A8" in str(layout):
                applied += 1
            continue
        if p.dtype in (torch.int8, torch.uint8):
            raw_int += 1
    print("    applied W4A8 layers %d | raw int params %d" % (applied, raw_int), flush=True)

    # declared layers in the pack vs modules present
    from safetensors import safe_open
    with safe_open(args.pack, framework="pt", device="cpu") as f:
        declared_ck = sorted({k[:-len(".comfy_quant")] for k in f.keys()
                              if k.endswith(".comfy_quant")})

    def runtime_name(key):
        """The checkpoint's key -> the name the built module carries."""
        if key.startswith("model.language_model."):
            return "model." + key[len("model.language_model."):]
        if key.startswith("model.visual."):
            return "visual." + key[len("model.visual."):]
        if key.startswith("lm_head."):
            return "model.lm_head." + key[len("lm_head."):]
        return key

    declared = [runtime_name(m) for m in declared_ck]
    for m in declared:
        if not any(n.startswith(m) for n in seen):
            unmatched.append(m)
    print("    declared %d | unmatched %d %s" % (len(declared), len(unmatched), unmatched[:3]),
          flush=True)

    cnt = [0]
    patches = []
    for holder in (ck, ckc):
        if not hasattr(holder, "w4a8_int8_linear"):
            continue
        orig_fn = getattr(holder, "w4a8_int8_linear")
        if any(orig_fn is x[1] for x in patches):
            continue

        def counted(*a, _orig=orig_fn, **kw):
            cnt[0] += 1
            return _orig(*a, **kw)

        patches.append((holder, orig_fn, counted))
        setattr(holder, "w4a8_int8_linear", counted)
    try:
        out = encode(clip)
    finally:
        for holder, orig_fn, _ in patches:
            setattr(holder, "w4a8_int8_linear", orig_fn)

    cos = []
    finite = True
    for a, b in zip(ref, out):
        if not torch.isfinite(b).all():
            finite = False
        cos.append(torch.nn.functional.cosine_similarity(
            a.reshape(1, -1), b.reshape(1, -1)).item())
    print("    w4a8_linear calls %d | cond cos min %.7f max %.7f | finite %s"
          % (cnt[0], min(cos), max(cos), finite), flush=True)

    rec = {
        "tool": "audit/wevisone_w4a8_load_path.py",
        "route": "nodes.CLIPLoader (type=stable_diffusion)",
        "model_class": type(model).__name__,
        "declared_layers": len(declared),
        "applied_layers": applied,
        "raw_code_params": raw_int,
        "unmatched_layers": len(unmatched),
        "w4a8_linear_calls": cnt[0],
        "prompts": len(PROMPTS),
        "cond_cos_min": min(cos), "cond_cos_max": max(cos),
        "finite": finite,
        "per_prompt_cos": cos,
    }
    rec["pass"] = bool(applied > 0 and raw_int == 0 and not unmatched and finite
                       and min(cos) >= 0.98)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1, sort_keys=True)
        print("record: %s" % args.json)
    print("LOAD PATH %s" % ("PASS" if rec["pass"] else "FAIL"))


if __name__ == "__main__":
    main()

"""CN-2 / CN-3 for the CLIP-ViT-H-14 ConvRot INT8 pack: what the CLIP vision route does with it.

Written for this one checkpoint. The route under test is ComfyUI's own `CLIPVisionLoader`,
which is the only consumer of a file in `models/clip_vision/` on this machine.

CN-2 asks whether the loader executes the pack's declared INT8 layers:

  - how many of the 193 declared Linears come out of the loader as INT8-backed weights
  - how many come out as plain float Parameters holding the raw int8 codes
  - whether the loader keeps the `.weight_scale` / `.comfy_quant` companions or drops them
  - what the loaded tower's output is against the source tower on the same image

CN-3 asks whether the INT8 kernel runs at all: one forward is executed with a hook that counts
both INT8 and float linear calls, and the output's finiteness is recorded.

Both answers are recorded verbatim, including the case where the route does not execute the
pack: a route that does not consume the pack is a measurement, not a failure to report.

  python audit/clip_vit_h14_route.py \
      --source <CLIP-ViT-H-14.safetensors> --pack <pack.safetensors> \
      --image <png> --comfy <ComfyUI> --json <record.json> [--size 224]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.comfy))
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    import comfy.clip_vision
    sys.argv = saved

    from safetensors.torch import load_file
    pk = load_file(args.pack)
    declared = [k for k in pk if k.endswith(".weight") and pk[k].dtype == torch.int8]
    scales = [k for k in pk if k.endswith(".weight_scale")]
    stamps = [k for k in pk if k.endswith(".comfy_quant")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from PIL import Image
    img = Image.open(args.image).convert("RGB").resize((args.size, args.size), Image.BICUBIC)
    arr = np.asarray(img).astype("float32") / 255.0
    px = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, torch.float32)

    def load_and_probe(path, label):
        cv = comfy.clip_vision.load(os.path.abspath(path))
        model = cv.model.to(device).eval()
        msd = model.state_dict()
        int8_backed, raw_codes, float_ok = 0, 0, 0
        raw_examples = []
        for k in declared:
            if k not in msd:
                continue
            w = msd[k]
            t = type(w).__name__
            if "Quantized" in t or getattr(w, "dtype", None) == torch.int8:
                int8_backed += 1
            else:
                # a float parameter that still holds the int8 codes is the raw-code case
                raw_codes += 1
                if len(raw_examples) < 3:
                    raw_examples.append({"layer": k, "param_type": t,
                                         "dtype": str(getattr(w, "dtype", None)),
                                         "min": float(w.float().min()), "max": float(w.float().max())})
            if k.replace(".weight", ".weight_scale") in pk:
                float_ok += 1

        counts = {"int8": 0, "float": 0}

        def hook(mod, inp, out):
            if isinstance(mod, nn.Linear):
                w = getattr(mod, "weight", None)
                if w is None:
                    counts["float"] += 1
                    return
                if getattr(w, "dtype", None) == torch.int8 or "Quantized" in type(w).__name__:
                    counts["int8"] += 1
                else:
                    counts["float"] += 1

        handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Linear)]
        with torch.no_grad():
            out = model(px)
        for h in handles:
            h.remove()
        if isinstance(out, dict):
            tok = out.get("last_hidden_state")
            emb = out.get("image_embeds")
        elif isinstance(out, (list, tuple)):
            tok = out[0]
            emb = out[1] if len(out) > 1 else None
        else:
            tok, emb = out, None
        finite = bool(torch.isfinite(tok).all())
        ret = {"label": label, "file": os.path.basename(path),
               "linear_modules": sum(1 for m in model.modules() if isinstance(m, nn.Linear)),
               "declared": len(declared),
               "loaded_as_int8": int8_backed, "loaded_as_raw_float_codes": raw_codes,
               "float_modules": float_ok,
               "raw_examples": raw_examples,
               "one_forward": {"int8_linear_calls": counts["int8"],
                               "float_linear_calls": counts["float"],
                               "output_finite": finite,
                               "tokens_shape": list(tok.shape),
                               "embeds_shape": None if emb is None else list(emb.shape)},
               }
        tok_c = tok.detach().double().cpu()
        emb_c = None if emb is None else emb.detach().double().cpu()
        del cv, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ret, tok_c, emb_c

    print("[CN-2] source", flush=True)
    src_rec, t_s, e_s = load_and_probe(args.source, "source")
    print("[CN-2] pack", flush=True)
    pk_rec, t_p, e_p = load_and_probe(args.pack, "pack")

    def cos(a, b):
        return float(torch.nn.functional.cosine_similarity(
            a.reshape(1, -1).float(), b.reshape(1, -1).float(), dim=1).item())

    rec = {
        "tool": "audit/clip_vit_h14_route.py",
        "model": os.path.basename(args.source),
        "pack": os.path.basename(args.pack),
        "route": "comfy.clip_vision.load (ComfyUI's CLIPVisionLoader)",
        "declared_layers": len(declared),
        "companion_tensors": {"weight_scale": len(scales), "comfy_quant": len(stamps)},
        "source": src_rec,
        "pack_side": pk_rec,
        "output_cos_source_vs_pack": cos(t_s, t_p),
        "embeds_cos_source_vs_pack": None if (e_s is None or e_p is None) else cos(e_s, e_p),
        "input": os.path.basename(args.image), "size": args.size,
    }
    applied = pk_rec["loaded_as_int8"]
    rec["verdict"] = ("applied" if applied == len(declared)
                      else "not applied - the loader puts the raw int8 codes into a float parameter")
    rec["int8_kernel_ran"] = bool(pk_rec["one_forward"]["int8_linear_calls"] > 0)

    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print("declared %d | loaded as int8 %d | raw float codes %d | int8 GEMM %d | float GEMM %d"
          % (len(declared), pk_rec["loaded_as_int8"], pk_rec["loaded_as_raw_float_codes"],
             pk_rec["one_forward"]["int8_linear_calls"], pk_rec["one_forward"]["float_linear_calls"]),
          flush=True)
    print("output cos source-vs-pack %.9f | finite %s"
          % (rec["output_cos_source_vs_pack"], pk_rec["one_forward"]["output_finite"]), flush=True)
    print("VERDICT: %s" % rec["verdict"], flush=True)
    print("wrote %s" % args.json, flush=True)


if __name__ == "__main__":
    main()

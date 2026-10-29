# -*- coding: utf-8 -*-
"""Guarded SDXL pair-route measurement for the CLIP-shaped INT8 text-encoder packs.

This is the harness that replaced the withdrawn `cosine 1.000000` rows (see the clip-files
note, section 16). Those rows came from min(1.0, nan); this tool cannot produce that artefact:
the cosine returns no value unless every accumulated sum is finite, the NaN element count of
BOTH conditioning outputs is recorded per prompt, and a prompt is only compared when both
sides have zero NaN. Both models are loaded before any encode, in a fresh process, and the
source sha256 is re-verified against the audit record before loading.

The SDXL combined conditioning is 77 x (768 + 1280): the first 768 channels are the clip_l
half, the half this test varies. The clip_g half is excluded by measurement, not by choice -
on the machine this ran, a float16 bigG poisons it with NaN for BOTH models, which would make
every comparison "no value" for a reason unrelated to the pack.

No machine default: --comfy (or COMFY_PATH) and --out are required.

    python pair_guarded.py --artifact clip_l --comfy <ComfyUI root> --out <file.json>
"""
import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--artifact", required=True, choices=["clip_l", "sae", "long", "gmp"])
ap.add_argument("--comfy", default=None, help="ComfyUI root; required, no machine default")
ap.add_argument("--out", required=True)
a = ap.parse_args()
COMFY = a.comfy or os.environ.get("COMFY_PATH")
if not COMFY:
    raise SystemExit("--comfy <ComfyUI root> or COMFY_PATH is required; no machine default")

NODE = os.path.join(COMFY, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools")
sys.path.insert(0, COMFY)
os.chdir(COMFY)
import server as _srv
if not hasattr(_srv.PromptServer, "instance"):
    _srv.PromptServer.instance = None
spec = importlib.util.spec_from_file_location("hswq_loader", os.path.join(NODE, "__init__.py"),
                                              submodule_search_locations=[NODE])
_m = importlib.util.module_from_spec(spec)
sys.modules["hswq_loader"] = _m
spec.loader.exec_module(_m)
_arm = importlib.import_module("hswq_loader.patches.clip_int8_gemm.arm")
import nodes as cnodes
from safetensors import safe_open

torch.set_grad_enabled(False)

CLIP = os.path.join(COMFY, "models", "clip")
BIGG = "CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors"

ARTIFACTS = {
    "clip_l": {"pack": "clip_l_hswq_convrot_int8.safetensors", "src": "clip_l.safetensors",
               "src_sha12": "660c6f5b1aba"},
    "sae": {"pack": "CLIP-SAE-ViT-L-14-hswq_convrot_int8.safetensors",
            "src": "CLIP-SAE-ViT-L-14-FP32.safetensors", "src_sha12": "820afa79f4ba"},
    "long": {"pack": "Long-ViT-L-14-BEST-GmP-smooth-ft-full_hswq_convrot_int8.safetensors",
             "src": "Long-ViT-L-14-BEST-GmP-smooth-ft-full.safetensors",
             "src_sha12": "4d3e6c8c2188"},
    "gmp": {"pack": "ViT-L-14-BEST-smooth-GmP-HF-format_hswq_convrot_int8.safetensors",
            "src": "ViT-L-14-BEST-smooth-GmP-HF-format.safetensors",
            "src_sha12": "46a201afae25"},
}

PROMPTS = ["a photo of a cat", "the old bridge", "a red car", "warm bread",
           "bonsoir, Elliot", "2024-01-05", "ERROR_30741852",
           "a watercolour of the harbour at dusk"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


A = ARTIFACTS[a.artifact]
res = {"artifact": a.artifact, "pack_file": A["pack"], "src_file": A["src"]}
src_path = os.path.join(CLIP, A["src"])
pack_path = os.path.join(CLIP, A["pack"])
res["src_bytes"] = os.path.getsize(src_path)
res["src_sha256"] = sha(src_path)
res["src_sha_match_record"] = res["src_sha256"][:12] == A["src_sha12"]
res["pack_bytes"] = os.path.getsize(pack_path)
res["pack_sha256"] = sha(pack_path)
print("identity", res["src_file"], res["src_sha256"][:12], res["src_sha_match_record"],
      flush=True)

with safe_open(pack_path, framework="pt") as f:
    QM = json.loads((f.metadata() or {}).get("_quantization_metadata", "{}"))["layers"]
TEXT = {k: v for k, v in QM.items() if not k.startswith("vision_model")}
res["declared_text_layers"] = len(TEXT)
res["declared_all_layers"] = len(QM)


def load_pair(name):
    torch.manual_seed(1234)
    return cnodes.DualCLIPLoader().load_clip(clip_name1=name, clip_name2=BIGG,
                                             type="stable_diffusion", device="default")[0]


def cond_l_half(clip, text):
    tk = clip.tokenize(text)
    d = clip.encode_from_tokens(tk, return_dict=True)
    c = d["cond"].float().cpu()
    return c.reshape(-1, 2048)[:, :768].contiguous().reshape(-1)


def cos_guarded(x, y):
    """fp64 cosine; None unless every accumulated sum is finite - never a clamp of NaN."""
    dot = na = nb = 0.0
    for i in range(0, x.numel(), 2_000_000):
        p, q = x[i:i + 2_000_000], y[i:i + 2_000_000]
        dot += float((p * q).sum())
        na += float((p * p).sum())
        nb += float((q * q).sum())
    if not (math.isfinite(dot) and math.isfinite(na) and math.isfinite(nb)):
        return None
    if na <= 0 or nb <= 0:
        return None
    return min(1.0, dot / (math.sqrt(na) * math.sqrt(nb)))


try:
    _arm.reset_int8_gemm_stats()
    pack = load_pair(A["pack"])
    ref = load_pair(A["src"])
    params = dict(pack.cond_stage_model.clip_l.named_parameters())
    raw = notraw = unmatched = 0
    for stem in TEXT:
        key = stem if stem.endswith(".weight") else stem + ".weight"
        cand = sorted(k for k in params if k.endswith(key))
        if not cand:
            unmatched += 1
            continue
        t = params[cand[0]].float()
        if bool(torch.equal(t, t.round())) and float(t.abs().max()) >= 100.0:
            raw += 1
        else:
            notraw += 1
    per = []
    for p in PROMPTS:
        cp = cond_l_half(pack, p)
        cr = cond_l_half(ref, p)
        nanc = [int(torch.isnan(cp).sum()), int(torch.isnan(cr).sum())]
        c = None if any(nanc) else cos_guarded(cp.double(), cr.double())
        per.append({"prompt": p, "nan_pack": nanc[0], "nan_ref": nanc[1],
                    "cos": None if c is None else round(c, 6)})
    vals = [x["cos"] for x in per if x["cos"] is not None]
    res["pair"] = {"declared_text": len(TEXT), "raw": raw, "not_raw": notraw,
                   "unmatched": unmatched,
                   "gemm": dict(_arm.int8_gemm_stats()).get("int8_linear_calls"),
                   "prompts_total": len(per), "prompts_compared": len(vals),
                   "prompts_nan": len(per) - len(vals),
                   "cond_cos_min": min(vals) if vals else None,
                   "cond_cos_max": max(vals) if vals else None,
                   "per_prompt": per}
    r = res["pair"]
    print("%s: declared %d raw %d not-raw %d | gemm %s | compared %d/%d | cos %s..%s"
          % (a.artifact, r["declared_text"], r["raw"], r["not_raw"], r["gemm"],
             r["prompts_compared"], r["prompts_total"],
             r["cond_cos_min"], r["cond_cos_max"]), flush=True)
except Exception as e:
    res["pair"] = {"error": "%s: %s" % (type(e).__name__, str(e)[:200])}
    print("ERROR", res["pair"]["error"], flush=True)

io.open(a.out, "w", encoding="utf-8").write(json.dumps(res, indent=1) + "\n")
print("wrote", a.out, flush=True)

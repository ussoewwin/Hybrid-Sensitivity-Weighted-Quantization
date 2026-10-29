# -*- coding: utf-8 -*-
"""sam3.1 - the four loading routes compared against the fp16 source, in one process.

Derived from audit/sam31_detector_probe.py (the same language-encoder mapping, the same inputs,
the same forward), with the loader switched per route:

  REF  fp16 source, loaded through mixed_precision_ops (the probe's own reference build)
  R1   pack through comfy.sd.load_diffusion_model            - what stock UNETLoader does
  R2   pack through the node's HSWQ SAM3 Loader              - the node the card names
  R3   pack through mixed_precision_ops + state-dict load    - the earlier record's harness

Reported per route: the 41 declared layers judged against their own dequantised value, INT8 kernel
invocations counted by wrapping comfy_kitchen.int8_linear, and behaviour - detection masks and
scores for fixed prompts and images, compared with the reference by mask IoU and cosine.
"""
import argparse
import glob
import io
import json
import math
import os
import statistics as st
import sys

import torch
from PIL import Image

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--comfy", default=None)
_ap.add_argument("--out", required=True)
a = _ap.parse_args()
COMFY = a.comfy or os.environ.get("COMFY_PATH")
if not COMFY:
    raise SystemExit("--comfy <ComfyUI root> or COMFY_PATH is required; this tool ships without a machine default")
NODE = os.path.join(COMFY, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools")
SIZE = 1008  # the detector forward needs 1008; at 1024 apply_rope fails ("freqs is not broadcastable") - measured 2026-10-28
PER_GROUP = 2
FILLER_SHORT = ["a cat", "a dog", "a car", "a bridge", "rain at night"]
FILLER_TAG = ["photo, high detail", "anime style", "4k photo", "studio light", "wide shot"]

sys.path.insert(0, COMFY)
os.chdir(COMFY)
import importlib.util
import server as _srv
if not hasattr(_srv.PromptServer, "instance"):
    _srv.PromptServer.instance = None
spec = importlib.util.spec_from_file_location("hswq_loader", os.path.join(NODE, "__init__.py"),
                                              submodule_search_locations=[NODE])
_m = importlib.util.module_from_spec(spec)
sys.modules["hswq_loader"] = _m
spec.loader.exec_module(_m)
_arm = importlib.import_module("hswq_loader.patches.clip_int8_gemm.arm")

import comfy.ops as comfy_ops
import comfy.sd
import comfy.utils
from comfy.quant_ops import QUANT_ALGOS
from comfy.text_encoders.sam3_clip import SAM3ClipModelWrapper, SAM3TokenizerWrapper
from safetensors.torch import load_file
torch.set_grad_enabled(False)

UNET = os.path.join(COMFY, "models", "unet")
SRC = os.path.join(UNET, "sam3.1_multiplex_fp16.safetensors")
PACK = os.path.join(UNET, "sam3.1_multiplex_hswq_convrot_int8.safetensors")
OUT = a.out

KCOUNT = {"n": 0, "wrapped": []}
try:
    import comfy_kitchen as _ck
    _f = getattr(_ck, "int8_linear", None)
    if _f is not None and callable(_f):
        def _mk(fn):
            def c(*a, **k):
                KCOUNT["n"] += 1
                return fn(*a, **k)
            return c
        setattr(_ck, "int8_linear", _mk(_f))
        KCOUNT["wrapped"].append("int8_linear")
except Exception as e:
    print("kitchen wrapper:", str(e)[:90], flush=True)


def iou(a, b):
    if a.shape != b.shape:
        b = b.reshape(a.shape)
    ea = (a > 0.5).float()
    eb = (b > 0.5).float()
    inter = float((ea * eb).sum())
    union = float(torch.clamp(ea + eb, max=1).sum())
    return None if union <= 0 else inter / union


def cos64(a, b):
    a = a.float().cpu().reshape(-1).double()
    b = b.float().cpu().reshape(-1).double()
    if a.numel() != b.numel():
        return None
    dot = na = nb = 0.0
    for i in range(0, a.numel(), 2_000_000):
        x, y = a[i:i + 2_000_000], b[i:i + 2_000_000]
        dot += float((x * y).sum()); na += float((x * x).sum()); nb += float((y * y).sum())
    return None if na <= 0 or nb <= 0 else min(1.0, dot / (na ** 0.5 * nb ** 0.5))


dev = "cuda" if torch.cuda.is_available() else "cpu"

lang_sd = load_file(SRC)
clm = SAM3ClipModelWrapper(device=dev).eval()
inner = clm.get_submodule("sam3_clip")
inner_keys = list(n for n, _ in inner.named_parameters())
mapped = {}
for k, v in lang_sd.items():
    if not k.startswith("detector.backbone.language_backbone.encoder."):
        continue
    tgt = k[len("detector.backbone.language_backbone.encoder."):]
    if ".attn.in_proj" in tgt:
        n = int(tgt.split("resblocks.")[1].split(".")[0])
        suffix = "weight" if tgt.endswith("_weight") else "bias"
        q, kv, vv = v.float().split(v.shape[0] // 3, dim=0)
        base = "transformer.text_model.encoder.layers.%d" % n
        mapped[base + ".self_attn.q_proj." + suffix] = q
        mapped[base + ".self_attn.k_proj." + suffix] = kv
        mapped[base + ".self_attn.v_proj." + suffix] = vv
        continue
    if tgt == "token_embedding.weight":
        hk = "transformer.text_model.embeddings.token_embedding.weight"
    elif tgt == "positional_embedding":
        hk = "transformer.text_model.embeddings.position_embedding.weight"
    else:
        hk = tgt.replace(".ln_1.", ".layer_norm1.").replace(".ln_2.", ".layer_norm2.")
        hk = hk.replace("ln_final.", "final_layer_norm.")
        hk = hk.replace(".mlp.c_fc.", ".mlp.fc1.").replace(".mlp.c_proj.", ".mlp.fc2.")
        hk = hk.replace(".attn.", ".self_attn.")
        if hk.startswith("transformer.resblocks."):
            a = hk[len("transformer.resblocks."):].split(".", 1)
            hk = "transformer.text_model.encoder.layers." + a[0] + "." + a[1]
        else:
            hk = "transformer.text_model." + hk
    if hk in inner_keys:
        mapped[hk] = v.float()
miss, unexp = inner.load_state_dict(mapped, strict=False)
print("language encoder: mapped %d missing %d unexpected %d"
      % (len(mapped), len(miss), len(unexp)), flush=True)
tokw = SAM3TokenizerWrapper()

frames = []
for seed in (0, 1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    frames.append((torch.rand(1, 3, SIZE, SIZE, generator=g) * 255.0).to(
        device=dev, dtype=torch.float16))
for p in sorted(glob.glob(os.path.join(COMFY, "input", "*.png")))[:2]:
    im = Image.open(p).convert("RGB").resize((SIZE, SIZE))
    frames.append(torch.from_numpy(np.asarray(im)).float()[None].to(
        device=dev, dtype=torch.float16).permute(0, 3, 1, 2))
print("images:", len(frames), flush=True)

PROMPTS = ["a photo of a cat", "bonsoir, Elliot", "2024-01-05"]
CONDS = []
for p in PROMPTS:
    tw = tokw.tokenize_with_weights(p)
    z, pooled, attn = clm.encode_token_weights(tw)
    CONDS.append((p, z.to(device=dev, dtype=torch.float16),
                  torch.ones(1, z.shape[1], dtype=torch.int64, device=dev)))

with __import__("safetensors.torch", fromlist=["load_file"]).load_file(PACK) as _:
    pass
from safetensors import safe_open
with safe_open(PACK, framework="pt") as f:
    QM = json.loads((f.metadata() or {}).get("_quantization_metadata", "{}"))["layers"]


def hadamard(n):
    h4 = torch.tensor([[1., 1., 1., -1.], [1., 1., -1., 1.],
                       [1., -1., 1., 1.], [-1., 1., 1., 1.]], dtype=torch.float64)
    h, cur = h4, 4
    while cur < n:
        h = torch.kron(h, h4)
        cur *= 4
    return h / math.sqrt(n)


def deq(stem, cfg):
    base = "model.diffusion_model." + stem
    alt = stem
    with safe_open(PACK, framework="pt") as f:
        wk = (base + ".weight") if (base + ".weight") in f.keys() else (alt + ".weight")
        sk = (base + ".weight_scale") if (base + ".weight_scale") in f.keys() else (alt + ".weight_scale")
        wi = f.get_tensor(wk)
        sc = f.get_tensor(sk).float()
    d = wi.float() * (sc.reshape(-1, 1) if sc.numel() == wi.shape[0] else sc)
    gs = cfg.get("convrot_groupsize") or 256
    if d.dim() == 2 and d.shape[1] % gs == 0 and gs >= 4 and math.log(gs, 4) % 1 == 0:
        o, i = d.shape
        d = torch.matmul(d.view(o, i // gs, gs), hadamard(gs).to(d.dtype)).reshape(o, i)
    return d


def judge(sam3):
    params = dict(sam3.named_parameters())
    raw = applied = other = unmatched = 0
    worst = []
    for stem, cfg in QM.items():
        cand = sorted(k for k in params if k.endswith(stem + ".weight"))
        if not cand:
            unmatched += 1
            continue
        t = params[cand[0]].float()
        if t.dim() != 2:
            other += 1
            continue
        if bool(torch.equal(t, t.round())) and float(t.abs().max()) >= 100.0:
            raw += 1
            continue
        c = cos64(t, deq(stem, cfg))
        if c is not None and c > 0.98:
            applied += 1
        else:
            other += 1
            worst.append((cand[0][-64:], c))
    return {"declared": len(QM), "raw": raw, "applied": applied, "other": other,
            "unmatched": unmatched, "worst": sorted(worst, key=lambda x: (x[1] is None, x[1]))[:3]}


def build(route):
    if route == "REF":
        sd = dict(lang_sd)
        ops = comfy_ops.mixed_precision_ops({"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]},
                                            torch.bfloat16, full_precision_mm=False, disabled=[])
        out = comfy.sd.load_diffusion_model_state_dict(sd, model_options={"custom_operations": ops})
    elif route == "R1":
        out = comfy.sd.load_diffusion_model(PACK, model_options={})
    elif route == "R2":
        got = _m.NODE_CLASS_MAPPINGS["HSWQSAM3Loader"]().load_sam3(sam3_name=os.path.basename(PACK))
        out = got[0] if isinstance(got, tuple) else got
    else:
        sd = comfy.utils.load_torch_file(PACK, safe_load=True)
        ops = comfy_ops.mixed_precision_ops({"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]},
                                            torch.float16, full_precision_mm=False, disabled=[])
        out = comfy.sd.load_diffusion_model_state_dict(sd, model_options={"custom_operations": ops})
    net = getattr(out, "model", out)
    sam3 = getattr(net, "diffusion_model", net)
    return sam3.to(dev).eval()


res = {"routes": {}}
REF_OUT = None
for route in ("REF", "R1", "R2", "R3"):
    try:
        sam3 = build(route)
        row = {"role": "reference" if route == "REF" else route}
        if route != "REF":
            row.update(judge(sam3))
        KCOUNT["n"] = 0
        _arm.reset_int8_gemm_stats()
        got = {}
        for p, te, tm in CONDS:
            masks, scores = [], []
            for fr in frames:
                with torch.no_grad():
                    det = sam3(fr, text_embeddings=te, text_mask=tm, threshold=0.5,
                               orig_size=(SIZE, SIZE))
                m = det.get("masks") if isinstance(det, dict) else None
                sc = det.get("scores") if isinstance(det, dict) else None
                if torch.is_tensor(sc) and sc.numel():
                    scores.append(float(sc.float().max()))
                if torch.is_tensor(m) and m.numel():
                    mm = m[0] if m.dim() == 4 else m
                    if mm.dim() == 3:
                        mm = mm[0]
                    masks.append(mm.float().cpu())
            got[p] = {"masks": masks, "scores": scores}
        row["kitchen_int8_linear_calls"] = KCOUNT["n"]
        row["arm_int8_linear_calls"] = dict(_arm.int8_gemm_stats()).get("int8_linear_calls")
        if route == "REF":
            REF_OUT = got
            row["area_per_prompt"] = {p: [round(float((m > 0.5).float().mean()), 6)
                                          for m in v["masks"]] for p, v in got.items()}
            row["score_max_per_prompt"] = {p: (max(v["scores"]) if v["scores"] else None)
                                           for p, v in got.items()}
        else:
            ious, cosv, dsc = [], [], []
            for p in PROMPTS:
                for x, y in zip(REF_OUT[p]["masks"], got[p]["masks"]):
                    i = iou(x, y)
                    if i is not None:
                        ious.append(i)
                    c = cos64(x, y)
                    if c is not None:
                        cosv.append(c)
                if REF_OUT[p]["scores"] and got[p]["scores"]:
                    dsc.append(abs(max(REF_OUT[p]["scores"]) - max(got[p]["scores"])))
            row["mask_iou_vs_ref_min"] = min(ious) if ious else None
            row["mask_iou_vs_ref_median"] = st.median(ious) if ious else None
            row["mask_cos_vs_ref_min"] = min(cosv) if cosv else None
            row["mask_cos_vs_ref_max"] = max(cosv) if cosv else None
            row["score_max_abs_diff_vs_ref"] = max(dsc) if dsc else None
            row["n_mask_pairs"] = len(ious)
        res["routes"][route] = row
        print("%-3s %s" % (route, json.dumps(row)[:300]), flush=True)
        del sam3
        if dev == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        res["routes"][route] = {"error": "%s: %s" % (type(e).__name__, str(e)[:220])}
        print("%-3s ERROR %s" % (route, str(e)[:170]), flush=True)

io.open(OUT, "w", encoding="utf-8").write(json.dumps(res, indent=1))
print("wrote", OUT)

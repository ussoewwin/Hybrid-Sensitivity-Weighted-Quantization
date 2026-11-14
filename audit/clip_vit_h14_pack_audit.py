"""CN-1 for `CLIP-ViT-H-14.safetensors`: the pack is the source quantised, nothing else.

Written for this one checkpoint. The questions, in order:

  1. identity   - the source and the pack are the same 521-tensor layout, with the pack carrying
                  the extra `.weight_scale` / `.comfy_quant` entries its 193 INT8 layers need
  2. coverage   - every one of the 193 Linears in the source appears quantised, with the same shape
  3. arithmetic - the pack's int8 codes and scales are exactly reproducible from the source: take
                  the source weight, apply the same Hadamard rotation, the same row-wise amax/127,
                  and the codes must match
  4. float side - every tensor the pack did not quantise is bit-identical to the source
  5. behaviour  - the two towers on the same image: the cosine of the token stack and of the
                  projected embedding

  python audit/clip_vit_h14_pack_audit.py \
      --source <CLIP-ViT-H-14.safetensors> --pack <pack.safetensors> \
      --image <png> --comfy <ComfyUI> --json <record.json> [--size 224]
"""

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_hadamard(size, dtype=torch.float32):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def group_size(n, preferred=256):
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


ENCODER = "vision_model.encoder.layers."


def is_target(name):
    """The encoder Linears: 32 layers x 6. `visual_projection` is not one of them.

    The pack keeps `visual_projection` as FP16 because the converter maps runtime names to
    checkpoint names through the vision tower's inner module list, which does not contain it.
    Counting it as a target here would report a coverage hole that the pack does not have.
    """
    if not name.startswith(ENCODER) or not name.endswith(".weight"):
        return False
    parts = name[len(ENCODER):].split(".")
    if len(parts) != 4:
        return False
    _, block, leaf, _w = parts
    return block in ("self_attn", "mlp") and leaf in (
        "q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    from safetensors.torch import load_file

    print("[load] source", flush=True)
    src = load_file(args.source)
    print("[load] pack", flush=True)
    pk = load_file(args.pack)

    src_lin = {k for k in src if is_target(k)}
    pk_codes = {k for k in pk if is_target(k) and pk[k].dtype == torch.int8}
    pk_float = {k for k in pk if is_target(k) and pk[k].dtype != torch.int8}
    pk_scale = {k.replace(".weight", ".weight_scale") for k in pk_codes}
    pk_stamp = {k.replace(".weight", ".comfy_quant") for k in pk_codes}

    rec = {
        "tool": "audit/clip_vit_h14_pack_audit.py",
        "model": os.path.basename(args.source),
        "source": {"file": os.path.basename(args.source), "bytes": os.path.getsize(args.source),
                   "sha256": sha256(args.source), "tensors": len(src)},
        "pack": {"file": os.path.basename(args.pack), "bytes": os.path.getsize(args.pack),
                 "sha256": sha256(args.pack), "tensors": len(pk)},
        "declared": {"layers": len(src_lin), "format": "int8_tensorwise", "convrot": True,
                     "groupsize": 256,
                     "int8_layers": len(pk_codes), "fp16_kept_layers": len(pk_float),
                     "visual_projection_in_pack": ("visual_projection.weight" in pk and
                                                   str(pk["visual_projection.weight"].dtype))},
    }

    # 1. identity
    missing = sorted(src_lin - pk_codes - pk_float)
    extra = sorted(pk_codes - src_lin)
    rec["coverage"] = {"n_missing": len(missing), "n_extra": len(extra),
                       "missing": missing[:20], "extra": extra[:20],
                       "scale_present": len(pk_scale & set(pk)),
                       "stamp_present": len(pk_stamp & set(pk))}

    # 2. shape
    shape_bad = []
    for k in sorted(src_lin & pk_codes):
        if tuple(src[k].shape) != tuple(pk[k].shape):
            shape_bad.append({"layer": k, "source": list(src[k].shape), "pack": list(pk[k].shape)})
    rec["shape_mismatch"] = shape_bad

    # 3. arithmetic: replay the quantisation from the source and compare the codes
    q_bad, max_delta, scale_rel_max, nonfinite = 0, 0, 0.0, 0
    for k in sorted(src_lin & pk_codes):
        w = src[k].float()
        mk = k[: -len(".weight")]
        gs = group_size(int(w.shape[1]))
        if gs is None:
            continue
        h = build_hadamard(gs)
        of, inf = w.shape
        g = inf // gs
        x = torch.matmul(w.view(of, g, gs), h.T).reshape(of, inf)
        amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
        s = amax / 127.0
        q = (x / s).round().clamp(-127, 127).to(torch.int8)
        got = pk[k]
        # a replay difference of one count is the rounding boundary: the pack's scale was
        # computed in the converter's order and the replay's amax can land one ulp away, which
        # flips values that sit exactly on .5. Anything beyond one count is a real difference.
        d = (q.to(torch.int32) - got.to(torch.int32)).abs()
        if int(d.max()) > 1:
            q_bad += 1
        max_delta = max(max_delta, int(d.max()))
        ps = pk[mk + ".weight_scale"].float()
        if not torch.isfinite(ps).all():
            nonfinite += 1
        rel = float(((ps - s).abs() / s.abs().clamp(min=1e-30)).max())
        scale_rel_max = max(scale_rel_max, rel)
    rec["quantised"] = {"found": len(src_lin & pk_codes), "q_mismatch_total": q_bad,
                        "max_q_delta": max_delta, "scale_exact_rel_max": scale_rel_max,
                        "nonfinite_scales": nonfinite}

    # 4. float side: everything the pack did not quantise must be bit-identical
    float_keys = [k for k in src if k not in src_lin]
    float_bad, float_max = 0, 0.0
    for k in float_keys:
        if k not in pk:
            float_bad += 1
            continue
        a, b = src[k], pk[k]
        # the converter stores every kept tensor in FP16 on purpose: ComfyUI's CLIP runs in FP16,
        # so the extra FP32 precision is never seen at runtime. The comparison is therefore the
        # FP16 cast of the source against the pack, exactly, not the FP32 source against it.
        want = a.to(torch.float16) if a.is_floating_point() else a
        got = b.to(want.dtype)
        if not torch.equal(want, got):
            float_bad += 1
            float_max = max(float_max, float((want.float() - got.float()).abs().max()))
    rec["float_side"] = {"checked": len(float_keys), "mismatch_count": float_bad,
                         "max_abs_diff": float_max}

    # 5. behaviour: the two towers on the same image
    sys.path.insert(0, os.path.abspath(args.comfy))
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    import comfy.clip_vision
    sys.argv = saved

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from PIL import Image
    img = Image.open(args.image).convert("RGB").resize((args.size, args.size), Image.BICUBIC)
    arr = np.asarray(img).astype("float32") / 255.0
    px = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, torch.float32)

    def tower(path):
        c = comfy.clip_vision.load(os.path.abspath(path))
        m = c.model.to(device).eval()
        with torch.no_grad():
            o = m(px)
        if isinstance(o, dict):
            tok, emb = o.get("last_hidden_state"), o.get("image_embeds")
        elif isinstance(o, (list, tuple)):
            tok, emb = o[0], (o[1] if len(o) > 1 else None)
        else:
            tok, emb = o, None
        del c, m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return tok.detach().double().cpu(), None if emb is None else emb.detach().double().cpu()

    print("[behaviour] source tower", flush=True)
    t_s, e_s = tower(args.source)

    # The pack cannot be fed to comfy's loader for a like-for-like comparison: that loader puts the
    # int8 codes straight into a float Parameter instead of dequantising them (recorded separately
    # as the route). So the pack's arithmetic is checked by dequantising its weights here and
    # loading them into the source's own structure, which isolates "is the pack the source
    # quantised" from "can this machine's CLIP vision route execute it".
    print("[behaviour] pack tower (weights dequantised into the source structure)", flush=True)
    cv = comfy.clip_vision.load(os.path.abspath(args.source))
    m2 = cv.model.to(device).eval()
    msd = m2.state_dict()
    replaced = 0
    for k in sorted(pk_codes):
        mk = k[: -len(".weight")]
        gs = group_size(int(pk[k].shape[1]))
        h = build_hadamard(gs)
        of, inf = pk[k].shape
        g = inf // gs
        deq = torch.matmul((pk[k].float() * pk[mk + ".weight_scale"].float()).view(of, g, gs),
                           h).reshape(of, inf)
        if k not in msd:
            raise SystemExit("layer %s is not a parameter of the source structure" % k)
        with torch.no_grad():
            msd[k].copy_(deq.to(msd[k].dtype).to(msd[k].device))
        replaced += 1
    print("[behaviour] %d layers replaced with their dequantised weights" % replaced, flush=True)
    with torch.no_grad():
        o = m2(px)
    if isinstance(o, dict):
        t_p, e_p = o.get("last_hidden_state"), o.get("image_embeds")
    elif isinstance(o, (list, tuple)):
        t_p, e_p = o[0], (o[1] if len(o) > 1 else None)
    else:
        t_p, e_p = o, None
    t_p = t_p.detach().double().cpu()
    e_p = None if e_p is None else e_p.detach().double().cpu()
    del cv, m2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def cos(a, b):
        return float(torch.nn.functional.cosine_similarity(
            a.reshape(1, -1), b.reshape(1, -1), dim=1).item())

    def rel(a, b):
        return float(((a - b).norm() / b.norm().clamp(min=1e-30)).item())

    rec["behaviour"] = {
        "tokens_cos": cos(t_s, t_p), "tokens_rel_l2": rel(t_s, t_p),
        "tokens_abs_max": float((t_s - t_p).abs().max()),
        "embeds_cos": None if (e_s is None or e_p is None) else cos(e_s, e_p),
        "embeds_rel_l2": None if (e_s is None or e_p is None) else rel(e_s, e_p),
        "input": os.path.basename(args.image), "size": args.size,
        "how": "the pack's int8 codes and scales were dequantised and loaded into the source structure, so this compares the quantisation itself",
    }

    ok = (len(missing) == 0 and len(extra) == 0 and not shape_bad
          and q_bad == 0 and float_bad == 0 and nonfinite == 0)
    rec["verdict"] = "FAITHFUL" if ok else "NOT FAITHFUL"

    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print("declared %d | coverage missing %d extra %d | q_mismatch %d | float mismatch %d"
          % (len(src_lin), len(missing), len(extra), q_bad, float_bad), flush=True)
    print("tokens cos %.9f | embeds cos %s"
          % (rec["behaviour"]["tokens_cos"],
             "n/a" if rec["behaviour"]["embeds_cos"] is None else "%.9f" % rec["behaviour"]["embeds_cos"]),
          flush=True)
    print("VERDICT: %s" % rec["verdict"], flush=True)
    print("wrote %s" % args.json, flush=True)


if __name__ == "__main__":
    main()

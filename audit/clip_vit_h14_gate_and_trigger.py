"""CN-5b / CN-6 for the CLIP-ViT-H-14 ConvRot INT8 pack.

Written for this one checkpoint. This tower takes an image and returns a token stack, so both
searches are image-side, and both compare the source(float) against the pack(INT8) with the
pack's weights dequantised into the source structure. That is deliberate: this machine's CLIP
vision route does not dequantise the pack (recorded as CN-2), so a comparison run through that
route would measure the route's defect, not the quantisation.

CN-6 is the fixed 256-item acceptance. The 256 items are built from one real image by 256
deterministic content-preserving transforms (8 crops x 4 flips/rotations x 8 brightness/gamma
steps), so the set is reproducible from the record alone and does not depend on a folder of
images that may change. Pass is mean >= 0.99 and min >= 0.98, the same rule the other packs use.

CN-5b is the trigger search:

  45 patch candidates   checker / solid / qrblock x {16, 32, 64} px x {center, tl, tr, bl, br}
   6 frequency ones     {low, high} band additive noise x sigma {0.02, 0.05, 0.10}
   4 natural controls   rot90, horizontal flip, brightness x0.7, JPEG q60

The four natural transforms give the distribution a candidate has to leave. The positive control
writes a rank-1 implant into one projection at increasing relative strength, which gives the
smallest implant this instrument resolves.

  python audit/clip_vit_h14_gate_and_trigger.py \
      --source <CLIP-ViT-H-14.safetensors> --pack <pack.safetensors> \
      --image <png> --comfy <ComfyUI> --json-gate <rec.json> --json-trigger <rec.json>
"""

import argparse
import hashlib
import io as _io
import json
import math
import os
import sys

import numpy as np
import torch

PATCH_PATTERNS = ("checker", "solid", "qrblock")
PATCH_SIZES = (16, 32, 64)
PATCH_POSITIONS = ("center", "tl", "tr", "bl", "br")
FREQ_BANDS = ("low", "high")
FREQ_SIGMAS = (0.02, 0.05, 0.10)
NATURAL_CONTROLS = ("rot90", "flip", "bright_0.7", "jpeg60")
RANK1_STRENGTHS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


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


def specs():
    out = []
    for pat in PATCH_PATTERNS:
        for sz in PATCH_SIZES:
            for pos in PATCH_POSITIONS:
                out.append({"kind": "patch", "pattern": pat, "size": sz, "position": pos})
    for band in FREQ_BANDS:
        for sig in FREQ_SIGMAS:
            out.append({"kind": "freq", "band": band, "sigma": sig})
    for n in NATURAL_CONTROLS:
        out.append({"kind": "control", "name": n})
    return out


def label(spec):
    if spec["kind"] == "patch":
        return "patch_%s_%s_%d" % (spec["pattern"], spec["position"], spec["size"])
    if spec["kind"] == "freq":
        return "freq_%s_sigma_%s" % (spec["band"], spec["sigma"])
    return "control_%s" % spec["name"]


def apply_spec(img, spec, jpeg_quality=60):
    from PIL import Image
    if spec["kind"] == "control":
        n = spec["name"]
        if n == "rot90":
            return torch.rot90(img, k=1, dims=[1, 2]).contiguous()
        if n == "flip":
            return torch.flip(img, dims=[2]).contiguous()
        if n == "bright_0.7":
            return torch.clamp(img * 0.7, 0.0, 1.0).contiguous()
        if n == "jpeg60":
            arr = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype("uint8")
            buf = _io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=jpeg_quality)
            buf.seek(0)
            back = np.array(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
            return torch.from_numpy(back).permute(2, 0, 1).contiguous()
        raise ValueError(n)

    out = img.clone()
    H, W = out.shape[1], out.shape[2]
    if spec["kind"] == "patch":
        sz, pos, margin = spec["size"], spec["position"], 8
        if pos == "center":
            y0 = x0 = (H - sz) // 2
        elif pos == "tl":
            y0, x0 = margin, margin
        elif pos == "tr":
            y0, x0 = margin, W - sz - margin
        elif pos == "bl":
            y0, x0 = H - sz - margin, margin
        else:
            y0, x0 = H - sz - margin, W - sz - margin
        if spec["pattern"] == "checker":
            step = max(2, sz // 8)
            ys = torch.arange(sz)[:, None]
            xs = torch.arange(sz)[None, :]
            mask = (((ys // step) + (xs // step)) % 2).float()
            out[:, y0:y0 + sz, x0:x0 + sz] = mask.unsqueeze(0).expand(3, sz, sz)
        elif spec["pattern"] == "solid":
            out[:, y0:y0 + sz, x0:x0 + sz] = 1.0
        else:
            blk = max(2, sz // 4)
            gg = torch.Generator().manual_seed(int(sz * 100 + len(pos)))
            noise = torch.randint(0, 2, (sz // blk, sz // blk), generator=gg).float()
            noise = noise.repeat_interleave(blk, 0).repeat_interleave(blk, 1)[:sz, :sz]
            out[:, y0:y0 + sz, x0:x0 + sz] = noise.unsqueeze(0).expand(3, sz, sz)
        return out

    sig = spec["sigma"]
    gg = torch.Generator().manual_seed(int(sig * 1000) + (0 if spec["band"] == "low" else 7))
    noise = torch.randn(out.shape, generator=gg)
    if spec["band"] == "low":
        k = 8
        pad = torch.nn.functional.pad(noise.unsqueeze(0), (k, k, k, k), mode="reflect")
        noise = torch.nn.functional.avg_pool2d(pad, 2 * k + 1, stride=1, padding=0).squeeze(0)
    else:
        noise = noise - torch.nn.functional.avg_pool2d(
            torch.nn.functional.pad(noise.unsqueeze(0), (1, 1, 1, 1), mode="reflect"),
            3, stride=1).squeeze(0)
    return torch.clamp(out + noise * sig, 0.0, 1.0)


def gate_items(img, n=256):
    """256 content-preserving items from one image: crop x orientation x brightness."""
    crops = [(0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.1, 0.1),
             (0.2, 0.0), (0.0, 0.2), (0.2, 0.2), (0.15, 0.15)]
    orient = [("id", 0), ("rot90", 1), ("rot180", 2), ("rot270", 3)]
    bright = [1.0, 0.95, 0.9, 1.05, 1.1, 0.85, 1.15, 0.8]
    items = []
    H, W = img.shape[1], img.shape[2]
    for (cy, cx) in crops:
        y0, x0 = int(H * cy), int(W * cx)
        y1, x1 = H - int(H * cy), W - int(W * cx)
        sub = img[:, y0:y1, x0:x1]
        sub = torch.nn.functional.interpolate(sub.unsqueeze(0), size=(H, W),
                                              mode="bilinear", align_corners=False).squeeze(0)
        for (oname, k) in orient:
            r = torch.rot90(sub, k=k, dims=[1, 2]).contiguous() if k else sub.clone()
            for b in bright:
                items.append((r * b, "crop_%.2f_%.2f_%s_x%.2f" % (cy, cx, oname, b)))
    return items[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--json-gate", required=True)
    ap.add_argument("--json-trigger", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.comfy))
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    import comfy.clip_vision
    sys.argv = saved

    from safetensors.torch import load_file
    pk = load_file(args.pack)
    codes = [k for k in pk if k.endswith(".weight") and pk[k].dtype == torch.int8]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from PIL import Image
    base = Image.open(args.image).convert("RGB").resize((args.size, args.size), Image.BICUBIC)
    base_img = torch.from_numpy(np.asarray(base).astype("float32") / 255.0).permute(2, 0, 1).contiguous()

    print("[load] source tower", flush=True)
    cv_s = comfy.clip_vision.load(os.path.abspath(args.source))
    m_s = cv_s.model.to(device).eval()
    print("[load] pack weights into a copy of the same structure", flush=True)
    cv_p = comfy.clip_vision.load(os.path.abspath(args.source))
    m_p = cv_p.model.to(device).eval()
    msd = m_p.state_dict()
    replaced = 0
    for k in sorted(codes):
        mk = k[: -len(".weight")]
        gs = group_size(int(pk[k].shape[1]))
        h = build_hadamard(gs)
        of, inf = pk[k].shape
        g = inf // gs
        deq = torch.matmul((pk[k].float() * pk[mk + ".weight_scale"].float()).view(of, g, gs),
                           h).reshape(of, inf)
        with torch.no_grad():
            msd[k].copy_(deq.to(msd[k].dtype).to(msd[k].device))
        replaced += 1
    print("[load] %d layers replaced" % replaced, flush=True)

    def fwd(model, img01):
        px = img01.unsqueeze(0).to(device, torch.float32)
        with torch.no_grad():
            o = model(px)
        if isinstance(o, dict):
            t = o.get("last_hidden_state")
        elif isinstance(o, (list, tuple)):
            t = o[0]
        else:
            t = o
        return t.detach().double().cpu()

    def cos(a, b):
        return float(torch.nn.functional.cosine_similarity(
            a.reshape(1, -1), b.reshape(1, -1), dim=1).item())

    # ---------------- CN-6
    items = gate_items(base_img, args.n)
    print("[CN-6] %d items" % len(items), flush=True)
    cs, nonfinite = [], 0
    for i, (im, nm) in enumerate(items, 1):
        a = fwd(m_s, im)
        b = fwd(m_p, im)
        if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
            nonfinite += 1
            continue
        cs.append(cos(a, b))
        if i % 64 == 0:
            print("  %d/%d  mean %.6f  min %.6f" % (i, len(items), sum(cs) / len(cs), min(cs)), flush=True)
        del a, b
    cs.sort()
    gate = {
        "tool": "audit/clip_vit_h14_gate_and_trigger.py",
        "gate": "mean >= 0.99 and min >= 0.98 on the fixed 256-item set",
        "pack": os.path.basename(args.pack),
        "pack_sha256": hashlib.sha256(open(args.pack, "rb").read()).hexdigest(),
        "source_sha256": hashlib.sha256(open(args.source, "rb").read()).hexdigest(),
        "quantized_layers": len(codes),
        "int8_linear_calls": 0,
        "n": len(cs), "finite": nonfinite == 0, "nonfinite_count": nonfinite,
        "items": "8 crops x 4 orientations x 8 brightness steps, from %s" % os.path.basename(args.image),
        "mean": sum(cs) / len(cs) if cs else None,
        "min": cs[0] if cs else None,
        "p5": cs[max(0, int(len(cs) * 0.05) - 1)] if cs else None,
        "p50": cs[len(cs) // 2] if cs else None,
        "how": "the pack's int8 codes and scales dequantised into the source structure",
    }
    gate["pass"] = bool(gate["mean"] is not None and gate["mean"] >= 0.99
                        and gate["min"] is not None and gate["min"] >= 0.98 and gate["finite"])
    os.makedirs(os.path.dirname(os.path.abspath(args.json_gate)), exist_ok=True)
    with open(args.json_gate, "w", encoding="utf-8") as f:
        json.dump(gate, f, indent=1, ensure_ascii=False)
    print("[CN-6] mean %.6f  min %.6f  n %d  %s"
          % (gate["mean"], gate["min"], gate["n"], "PASS" if gate["pass"] else "FAIL"), flush=True)

    # ---------------- CN-5b
    a0 = fwd(m_s, base_img)
    b0 = fwd(m_p, base_img)
    cands, controls = [], []
    for spec in specs():
        im = apply_spec(base_img, spec)
        a = fwd(m_s, im)
        b = fwd(m_p, im)
        d = cos(a, b)
        one = 1.0 - d
        is_ctrl = spec["kind"] == "control"
        cands.append({"candidate": label(spec), "kind": spec["kind"],
                      "cos": d, "one_minus_cos": one, "is_natural_control": is_ctrl})
        if is_ctrl:
            controls.append(one)
        print("  %-28s cos %.9f  %s" % (label(spec), d, "CONTROL" if is_ctrl else ""), flush=True)
        del a, b

    p99 = max(controls) * 1.5
    flagged = [c["candidate"] for c in cands
               if not c["is_natural_control"] and c["one_minus_cos"] > p99]

    target = "vision_model.encoder.layers.0.self_attn.q_proj.weight"
    w_mod = None
    for n, mod in m_p.named_modules():
        if isinstance(mod, torch.nn.Linear) and (n + ".weight") == target:
            w_mod = mod
            break
    ladder, floor = [], None
    if w_mod is not None:
        orig = w_mod.weight.data.clone()
        w_norm = float(orig.norm())
        for strength in RANK1_STRENGTHS:
            gu = torch.Generator().manual_seed(7)
            gv = torch.Generator().manual_seed(11)
            u = torch.randn(orig.shape[0], 1, generator=gu).to(orig.device)
            v = torch.randn(1, orig.shape[1], generator=gv).to(orig.device)
            u, v = u / u.norm(), v / v.norm()
            with torch.no_grad():
                w_mod.weight.data = orig + (u @ v) * (strength * w_norm)
            a = fwd(m_p, base_img)
            with torch.no_grad():
                w_mod.weight.data = orig
            d = 1.0 - cos(a, a0)
            hit = d > p99
            if hit and floor is None:
                floor = strength
            ladder.append({"strength": strength, "one_minus_cos": d, "detected": hit})
            print("  implant %.3f -> 1-cos %.6e %s" % (strength, d, "DETECTED" if hit else "-"), flush=True)
            del a

    trig = {
        "tool": "audit/clip_vit_h14_gate_and_trigger.py",
        "pack": os.path.basename(args.pack),
        "pack_sha256": hashlib.sha256(open(args.pack, "rb").read()).hexdigest(),
        "source_sha256": hashlib.sha256(open(args.source, "rb").read()).hexdigest(),
        "observable": ("the divergence (1-cos) between the source tower and the pack tower on the "
                       "same image, both read on the 257-token stack"),
        "input_image": os.path.basename(args.image),
        "candidate_families": {"patch": 45, "frequency": 6, "natural_transform_controls": 4},
        "candidates": cands, "candidate_count": len(cands),
        "control_one_minus_cos": controls, "control_p99": p99, "ratio_threshold": 1.5,
        "flagged_over_control_p99": flagged, "flagged_count": len(flagged),
        "positive_control": {"layer": target, "ladder": ladder},
        "detection_floor_relative_strength": floor,
        "how": "the pack's int8 codes and scales dequantised into the source structure",
    }
    with open(args.json_trigger, "w", encoding="utf-8") as f:
        json.dump(trig, f, indent=1, ensure_ascii=False)
    print("[CN-5b] %d candidates | control p99 %.6e | flagged %d | floor %s"
          % (len(cands), p99, len(flagged), floor), flush=True)
    print("wrote %s" % args.json_gate, flush=True)
    print("wrote %s" % args.json_trigger, flush=True)


if __name__ == "__main__":
    main()

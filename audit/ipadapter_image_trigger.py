"""SD1.5 IP-Adapter Plus (`ip-adapter_sd15_plus.pth`) image-trigger search.

This instrument exists for this one checkpoint. An IP-Adapter is not a ControlNet: it has no
control branch and no hint input, so the control-image search the ControlNet packs use has
nothing to act on here. What this file does read is an image prompt, and that image prompt
decides the adapter's whole output through CLIP vision and the Resampler's 16 queries. So the
candidate set is applied to the image prompt, and the observable is the divergence between the
source and the pack on that prompt.

  45 patch candidates  : checker / solid / qrblock x {16, 32, 64} px x {center, tl, tr, bl, br}
   6 frequency ones    : {low, high} band additive noise x sigma {0.02, 0.05, 0.10}
   4 natural controls  : rot90, horizontal flip, brightness x0.7, JPEG q60

The four natural transforms are content-preserving, so their divergence is the distribution a
candidate has to leave to count as an anomaly. The threshold is the largest of the four times
1.5, the same rule the ControlNet packs use.

Positive control: a rank-1 implant of increasing relative strength is written into one
to_v_ip weight and the same observable is read, which gives the smallest implant this
instrument resolves on this checkpoint.

  python audit/ipadapter_image_trigger.py \
      --source <ip-adapter_sd15_plus.pth> --pack <pack.safetensors> \
      --clipvision <clip_vision_h.safetensors> --image <png> \
      --comfy <ComfyUI> --ipadapter_dir <custom_nodes/comfyui_ipadapter_plus> \
      --json <record.json> --seed 42
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


def specs():
    out = []
    for pat in PATCH_PATTERNS:
        for sz in PATCH_SIZES:
            for pos in PATCH_POSITIONS:
                out.append({"kind": "patch", "pattern": pat, "size": sz, "position": pos})
    for band in FREQ_BANDS:
        for sig in FREQ_SIGMAS:
            out.append({"kind": "freq", "band": band, "sigma": sig})
    for name in NATURAL_CONTROLS:
        out.append({"kind": "control", "name": name})
    return out


def label(spec):
    if spec["kind"] == "patch":
        return "patch_%s_%s_%d" % (spec["pattern"], spec["position"], spec["size"])
    if spec["kind"] == "freq":
        return "freq_%s_sigma_%s" % (spec["band"], spec["sigma"])
    return "control_%s" % spec["name"]


def apply_spec(img, spec, g, jpeg_quality=60):
    from PIL import Image

    if spec["kind"] == "control":
        name = spec["name"]
        if name == "rot90":
            return torch.rot90(img, k=1, dims=[1, 2]).contiguous()
        if name == "flip":
            return torch.flip(img, dims=[2]).contiguous()
        if name == "bright_0.7":
            return torch.clamp(img * 0.7, 0.0, 1.0).contiguous()
        if name == "jpeg60":
            arr = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype("uint8")
            buf = _io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=jpeg_quality)
            buf.seek(0)
            back = np.array(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
            return torch.from_numpy(back).permute(2, 0, 1).contiguous()
        raise ValueError(name)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--clipvision", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--ipadapter_dir", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jpeg_quality", type=int, default=60)
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.comfy))
    sys.path.insert(0, os.path.abspath(args.ipadapter_dir))
    import comfy.clip_vision
    from image_proj_models import Resampler
    from safetensors.torch import load_file

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def build(path, dequant):
        if path.lower().endswith((".pth", ".bin")):
            sd = torch.load(path, map_location="cpu", weights_only=True)
            proj = {k: v for k, v in sd["image_proj"].items() if torch.is_tensor(v)}
            ipa = {k: v for k, v in sd["ip_adapter"].items() if torch.is_tensor(v)}
        else:
            flat = load_file(path)
            proj, ipa = {}, {}
            for k, v in flat.items():
                if not torch.is_tensor(v) or not k.endswith(".weight"):
                    continue
                scale = flat.get(k[: -len(".weight")] + ".weight_scale")
                w = v.float()
                if dequant and scale is not None:
                    # the converter stores one scale per output row, [out, 1]; the multiplication
                    # is left broadcast-exact against that shape, the way the audit does it
                    sc = scale.float()
                    if sc.numel() == w.shape[0]:
                        sc = sc.reshape(-1, 1)
                    w = w * sc
                    # the codes live in the rotated basis: the ConvRot reconstruction is the
                    # group-wise inverse transform, so the dequantised weight is (codes*scale) @ H
                    out_f, in_f = w.shape
                    gs = 256
                    while gs >= 4:
                        if in_f % gs == 0 and (math.log(gs, 4) % 1 == 0):
                            break
                        gs //= 4
                    if in_f % gs == 0 and gs >= 4:
                        h = build_hadamard(gs)
                        w = torch.matmul(w.view(out_f, in_f // gs, gs), h).reshape(out_f, in_f)
                base = k[: -len(".weight")]
                if base.startswith("image_proj."):
                    # the Resampler is loaded by state-dict name, so ".weight" stays on
                    proj[base.replace("image_proj.", "", 1) + ".weight"] = w
                elif base.startswith("ip_adapter."):
                    ipa[base.replace("ip_adapter.", "", 1)] = w
            # the Resampler's learned query set is a bare "latents" tensor, not a .weight
            for k, v in flat.items():
                if k.endswith("latents") and k.startswith("image_proj.") and torch.is_tensor(v):
                    proj[k.replace("image_proj.", "", 1)] = v.float()
            for k, v in flat.items():
                if not torch.is_tensor(v) or not k.endswith(".bias"):
                    continue
                base = k[: -len(".bias")]
                if base.startswith("image_proj."):
                    proj[base.replace("image_proj.", "", 1) + ".bias"] = v.float()
                elif base.startswith("ip_adapter."):
                    ipa[base.replace("ip_adapter.", "", 1) + ".bias"] = v.float()

        res = Resampler(dim=768, depth=4, dim_head=64, heads=12, num_queries=16,
                        embedding_dim=1280, output_dim=768, ff_mult=4,
                        max_seq_len=257, apply_pos_emb=False, num_latents_mean_pooled=0)
        res.load_state_dict(proj, strict=False)
        res.to(dev, torch.float32).eval()
        # the pack branch already stripped ".weight" while grouping, the source branch did not
        kv = {}
        for k, v in ipa.items():
            name = k[: -len(".weight")] if k.endswith(".weight") else k
            kv[name] = v.to(dev, torch.float32)
        return res, kv

    res_s, kv_s = build(args.source, dequant=False)
    res_p, kv_p = build(args.pack, dequant=True)
    keys = sorted(kv_s)
    if sorted(kv_p) != keys:
        raise SystemExit("layer set differs: source %d pack %d"
                         % (len(keys), len(kv_p)))

    cv = comfy.clip_vision.load(os.path.abspath(args.clipvision))
    cv.model.eval()

    def out_of(res, kv, img):
        px = img.permute(1, 2, 0).unsqueeze(0).to(dev, torch.float32)
        with torch.no_grad():
            o = cv.encode_image(px)
        t = getattr(o, "last_hidden_state", None)
        if t is None:
            t = getattr(o, "image_embeds", None)
        if t is None and isinstance(o, dict):
            t = o.get("last_hidden_state", o.get("image_embeds"))
        if t is None and isinstance(o, (list, tuple)):
            t = o[0]
        if t is None:
            raise SystemExit("clip vision returned no token tensor")
        with torch.no_grad():
            e = res(t.float().to(dev))
            y = torch.cat([torch.nn.functional.linear(e, kv[k]).flatten() for k in keys])
        return e.flatten(), y

    def omc(a, b):
        return 1.0 - torch.nn.functional.cosine_similarity(
            a.unsqueeze(0).double(), b.unsqueeze(0).double()).item()

    from PIL import Image
    arr = np.asarray(Image.open(args.image).convert("RGB")).astype("float32") / 255.0
    base_img = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    e_s0, y_s0 = out_of(res_s, kv_s, base_img)
    e_p0, y_p0 = out_of(res_p, kv_p, base_img)
    print("[ref] embeds src-vs-pack 1-cos %.6e | kv %.6e"
          % (omc(e_s0, e_p0), omc(y_s0, y_p0)), flush=True)

    cands, controls = [], []
    for spec in specs():
        img2 = apply_spec(base_img, spec, None, args.jpeg_quality)
        e_s, y_s = out_of(res_s, kv_s, img2)
        e_p, y_p = out_of(res_p, kv_p, img2)
        d_e, d_y = omc(e_s, e_p), omc(y_s, y_p)
        comb = max(d_e, d_y)
        is_ctrl = spec["kind"] == "control"
        cands.append({"candidate": label(spec), "kind": spec["kind"],
                      "src_vs_pack_1mcos_embeds": d_e,
                      "src_vs_pack_1mcos_kv": d_y,
                      "one_minus_cos": comb,
                      "is_natural_control": is_ctrl})
        if is_ctrl:
            controls.append(comb)
        print("  %-28s embeds %.6e  kv %.6e  %s"
              % (label(spec), d_e, d_y, "CONTROL" if is_ctrl else ""), flush=True)

    p99 = max(controls) * 1.5
    flagged = [c["candidate"] for c in cands
               if not c["is_natural_control"] and c["one_minus_cos"] > p99]
    print("[candidates] %d | control p99 %.6e | flagged %d"
          % (len(cands), p99, len(flagged)), flush=True)

    target = max(keys, key=lambda k: kv_s[k].shape[0])
    orig = kv_s[target].clone()
    w_norm = float(orig.norm())
    ladder, floor = [], None
    for strength in RANK1_STRENGTHS:
        gu = torch.Generator().manual_seed(7)
        gv = torch.Generator().manual_seed(11)
        u = torch.randn(orig.shape[0], 1, generator=gu).to(dev)
        v = torch.randn(1, orig.shape[1], generator=gv).to(dev)
        u, v = u / u.norm(), v / v.norm()
        kv_s[target] = orig + (u @ v) * (strength * w_norm)
        e_i, y_i = out_of(res_s, kv_s, base_img)
        kv_s[target] = orig
        d = omc(y_s0, y_i)
        hit = d > p99
        if hit and floor is None:
            floor = strength
        ladder.append({"strength": strength, "one_minus_cos": d, "detected": hit})
        print("  implant %.3f -> 1-cos %.6e %s"
              % (strength, d, "DETECTED" if hit else "-"), flush=True)

    rec = {
        "tool": "audit/ipadapter_image_trigger.py",
        "family": "ipadapter",
        "model": os.path.basename(args.source),
        "source_sha256": hashlib.sha256(open(args.source, "rb").read()).hexdigest(),
        "pack_sha256": hashlib.sha256(open(args.pack, "rb").read()).hexdigest(),
        "observable": ("the divergence (1-cos) between the source(float) and the pack(INT8) "
                       "adapter output on the same image prompt, read on the Resampler queries "
                       "and on the k/v stack"),
        "clip_vision": os.path.basename(args.clipvision),
        "input_image": os.path.basename(args.image),
        "candidate_families": {"patch": len(PATCH_PATTERNS) * len(PATCH_SIZES) * len(PATCH_POSITIONS),
                               "frequency": len(FREQ_BANDS) * len(FREQ_SIGMAS),
                               "natural_transform_controls": len(NATURAL_CONTROLS)},
        "candidates": cands,
        "candidate_count": len(cands),
        "control_one_minus_cos": controls,
        "control_p99": p99,
        "ratio_threshold": 1.5,
        "flagged_over_control_p99": flagged,
        "flagged_count": len(flagged),
        "positive_control": {"layer": "ip_adapter." + target, "weight_frobenius": w_norm,
                             "ladder": ladder},
        "detection_floor_relative_strength": floor,
        "seed": args.seed,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print("wrote %s" % args.json, flush=True)


if __name__ == "__main__":
    main()

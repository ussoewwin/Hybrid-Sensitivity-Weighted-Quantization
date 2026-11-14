# -*- coding: utf-8 -*-
"""ControlNet ConvRot INT8 pack audit: identity, runtime-name coverage, arithmetic, float exactness.

The pack is written under ComfyUI's *runtime* module names, while the published source uses
diffusers-style names (`controlnet_cond_embedding.*`, `controlnet_down_blocks.*`, `add_embedding.*`).
Comparing the two by name therefore proves nothing: the names never coincide. This tool builds the
runtime ControlNet from the float source (the same `comfy.controlnet.load_controlnet_state_dict`
path the loader uses), takes that runtime state dict as the truth, and then

  1. checks coverage in runtime names (every runtime tensor present, nothing extra),
  2. replays the converter's quantisation on the runtime float weights
     (`rotate_weight` with the 256 Hadamard, then row-wise `amax/127` int8) and compares the result
     with the pack's int8 codes and scales element by element,
  3. compares every float tensor with fp16(runtime weight), bit for bit,
  4. states the declared-vs-found quantised layer sets.

A pack passes only when all four agree; the record carries the counts, so a report can cite them.

Usage:
  python audit/controlnet_pack_audit.py --source <float.safetensors> --pack <pack.safetensors> \
      --comfy <ComfyUI root> [--json <record.json>] [--device cpu|cuda]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys

import torch


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    meta = hdr.pop("__metadata__", None) or {}
    return hdr, meta


def build_hadamard(size, dtype=torch.float32):
    """Exactly the converter's construction: 4x4 kernel, Kronecker doubling, /sqrt(size).

    The recursive [[h, h], [h, -h]] doubling is a different matrix (it is a valid Hadamard, so the
    pack's row norms are preserved and the weights look plausible), and a replay built on it reports
    a whole-file mismatch. Keeping the converter's own construction is the point of the check.
    """
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError("Hadamard size must be a power of 4, got %s" % size)
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def rotate_weight(w, h, groupsize):
    of, inf = w.shape
    if inf % groupsize:
        raise ValueError("in_features %d not divisible by group size %d" % (inf, groupsize))
    g = inf // groupsize
    return torch.matmul(w.view(of, g, groupsize), h.T.to(w.dtype)).reshape(w.shape)


def quant_rowwise(w):
    amax = w.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
    s = amax / 127.0
    q = (w / s.to(w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, s.to(torch.float32)


def rotate_weight_conv2d(w, h, groupsize):
    """Conv2d: rotate along in_channels, the same construction the converter uses."""
    oc, ic, kh, kw = w.shape
    if ic % groupsize:
        raise ValueError("in_channels %d not divisible by group size %d" % (ic, groupsize))
    flat = w.permute(0, 2, 3, 1).contiguous().view(-1, ic)
    return rotate_weight(flat, h, groupsize).view(oc, kh, kw, ic).permute(0, 3, 1, 2).contiguous()


def quant_channelwise(w):
    """Conv2d: per-out-channel scale [out, 1, 1, 1]."""
    rd = tuple(range(1, w.dim()))
    amax = torch.clamp(w.abs().amax(dim=rd).reshape(-1), min=1e-30).float()
    s = amax / 127.0
    v = s.view(-1, *([1] * (w.dim() - 1)))
    q = (w / v.to(w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, s.to(torch.float32)


def runtime_state_dict(source_path, comfy_path):
    comfy = os.path.abspath(comfy_path)
    if comfy not in sys.path:
        sys.path.insert(0, comfy)
    import comfy.controlnet as cc  # noqa: E402
    import comfy.utils  # noqa: E402

    # the runtime's own loader: it reads a published .safetensors and a published
    # .pth source the same way ComfyUI reads them
    sd = comfy.utils.load_torch_file(source_path, safe_load=True)
    cnet = cc.load_controlnet_state_dict(sd)
    return cnet.control_model.state_dict()



def _ipadapter_runtime_state_dict(source_path, ipadapter_dir):
    """The IP-Adapter graph the apply node builds, under the names the pack carries.

    Same role as runtime_state_dict() for a ControlNet: it returns the runtime's own
    tensor names so the coverage and replay below compare like with like.
    """
    import torch.nn as nn

    if source_path.lower().endswith(".pth") or source_path.lower().endswith(".bin"):
        raw = torch.load(source_path, map_location="cpu", weights_only=True)
    else:
        from safetensors.torch import load_file
        flat = load_file(source_path)
        raw = {"image_proj": {}, "ip_adapter": {}}
        for k, v in flat.items():
            if not torch.is_tensor(v):
                continue
            if k.startswith("image_proj."):
                raw["image_proj"][k.replace("image_proj.", "", 1)] = v
            elif k.startswith("ip_adapter."):
                raw["ip_adapter"][k.replace("ip_adapter.", "", 1)] = v
    if not (isinstance(raw, dict) and "image_proj" in raw and "ip_adapter" in raw):
        raise SystemExit("not an IP-Adapter layout: keys=%s" % (list(raw)[:4],))

    if ipadapter_dir and ipadapter_dir not in sys.path:
        sys.path.insert(0, ipadapter_dir)
    from image_proj_models import Resampler  # noqa: E402

    res = Resampler(dim=768, depth=4, dim_head=64, heads=12, num_queries=16,
                    embedding_dim=1280, output_dim=768, ff_mult=4,
                    max_seq_len=257, apply_pos_emb=False, num_latents_mean_pooled=0)
    res.load_state_dict(raw["image_proj"], strict=False)
    out = {"image_proj." + k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v
           for k, v in res.state_dict().items()}
    for k, v in raw["ip_adapter"].items():
        if torch.is_tensor(v):
            out["ip_adapter." + k] = v.float() if v.is_floating_point() else v
    return out


def loader_drop(pack_path, comfy_path):
    """Build the ControlNet the way the loader does and report what it keeps.

    Name-based detection decides the class: a union ControlNet is recognised only when the file
    carries the diffusers key controlnet_cond_embedding.conv_in.weight. A pack written under
    runtime names does not, so it is built as a plain ControlNet and the union block is dropped.
    This measures that outcome instead of assuming it.
    """
    comfy = os.path.abspath(comfy_path)
    if comfy not in sys.path:
        sys.path.insert(0, comfy)
    cwd = os.getcwd()
    try:
        os.chdir(comfy)
        import folder_paths  # noqa: E402
        import comfy.controlnet as cc  # noqa: E402
        folder_paths.add_model_folder_path("controlnet", os.path.dirname(os.path.abspath(pack_path)))
        cnet = cc.load_controlnet(os.path.abspath(pack_path), model=None)
        if cnet is None or getattr(cnet, "control_model", None) is None:
            return {"class": None, "runtime_tensors": 0, "pack_tensors": 0, "dropped_count": None,
                    "dropped": [], "note": "the loader did not detect a control model for this file"}
        cm = cnet.control_model
        rt = set(cm.state_dict())
        hdr, _ = read_header(pack_path)
        logical = {k for k in hdr if not k.endswith((".weight_scale", ".comfy_quant"))}
        dropped = sorted(logical - rt)
        return {"class": type(cm).__name__, "runtime_tensors": len(rt),
                "pack_tensors": len(logical), "dropped_count": len(dropped),
                "dropped": dropped[:40]}
    finally:
        os.chdir(cwd)



def _faceid_runtime_state_dict(source_path):
    """FaceID carries image_proj (an MLP) and ip_adapter (LoRA pairs). Both are consumed
    as plain tensors by the FaceID path, so the runtime names are the file's own names."""
    import torch

    raw = torch.load(source_path, map_location="cpu", weights_only=True)
    out = {}
    for group in ("image_proj", "ip_adapter"):
        for k, v in raw[group].items():
            if torch.is_tensor(v):
                out[group + "." + k] = v
    return out

def main():
    ap = argparse.ArgumentParser(description="ControlNet ConvRot INT8 pack audit")
    ap.add_argument("--source", required=True, help="the float source checkpoint (published)")
    ap.add_argument("--pack", required=True, help="the ConvRot INT8 pack")
    ap.add_argument("--comfy", default=os.environ.get("COMFY_PATH", ""),
                    help="ComfyUI root (or set COMFY_PATH); the runtime naming comes from it")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--json", default="", help="record path")
    ap.add_argument("--family", default="controlnet", choices=["controlnet", "ipadapter", "faceid"],
                    help="which loader's graph the pack is closed against")
    ap.add_argument("--ipadapter_dir", default="",
                    help="the ipadapter node directory (owns image_proj_models.py) for --family ipadapter")
    ap.add_argument("--loader-drop", action="store_true",
                    help="also build the pack the way the loader does and report dropped tensors")
    args = ap.parse_args()

    if args.family == "ipadapter":
        if not args.ipadapter_dir or not os.path.isdir(args.ipadapter_dir):
            print("error: --ipadapter_dir <node dir> is required for --family ipadapter")
            return 2
    if not args.comfy or not os.path.isdir(args.comfy):
        print("error: --comfy <ComfyUI root> (or COMFY_PATH) is required: the pack uses runtime names")
        return 2

    dev = torch.device(args.device)
    t_start = __import__("time").time()

    src_sha = sha256_file(args.source)
    pack_sha = sha256_file(args.pack)
    pack_hdr, pack_meta = read_header(args.pack)

    qm = {}
    if "_quantization_metadata" in pack_meta:
        qm = json.loads(pack_meta["_quantization_metadata"])
    declared = qm.get("layers", {})
    groupsizes = {v.get("convrot_groupsize") for v in declared.values()}
    formats = {v.get("format") for v in declared.values()}
    rotations = {v.get("convrot") for v in declared.values()}

    if args.family == "faceid":
        rt = _faceid_runtime_state_dict(args.source)
    elif args.family == "ipadapter":
        rt = _ipadapter_runtime_state_dict(args.source, args.ipadapter_dir)
    else:
        rt = runtime_state_dict(args.source, args.comfy)
    pack_logical = {k for k in pack_hdr if not k.endswith((".weight_scale", ".comfy_quant"))}

    missing_from_pack = sorted(set(rt) - pack_logical)
    extra_in_pack = sorted(pack_logical - set(rt))
    shape_bad = [k for k in (set(rt) & pack_logical)
                 if tuple(rt[k].shape) != tuple(pack_hdr[k]["shape"])]

    int8_keys = sorted(k for k in pack_logical if pack_hdr[k]["dtype"] == "I8")
    found_layers = {k[: -len(".weight")] for k in int8_keys}
    declared_set = set(declared)
    undeclared_found = sorted(found_layers - declared_set)
    declared_not_found = sorted(declared_set - found_layers)

    h = build_hadamard(256)
    q_mismatch = 0
    max_q_delta = 0
    scale_rel_max = 0.0
    nonfinite_scale = 0
    layer_rows = []
    recomp_cache = {}

    from safetensors import safe_open  # noqa: E402

    with safe_open(args.pack, framework="pt", device="cpu") as f:
        for key in int8_keys:
            mod = key[: -len(".weight")]
            w = rt[key].to(dev).float()
            gs = int(declared.get(mod, {}).get("convrot_groupsize") or 256)
            hh = h if gs == 256 else build_hadamard(gs)
            if w.ndim == 4:
                q_ref, s_ref = quant_channelwise(rotate_weight_conv2d(w, hh.to(dev), gs))
            else:
                q_ref, s_ref = quant_rowwise(rotate_weight(w, hh.to(dev), gs))
            q_pack = f.get_tensor(key).to(dev).to(torch.int8)
            s_pack = f.get_tensor(mod + ".weight_scale").to(dev).float()
            if not torch.isfinite(s_pack).all():
                nonfinite_scale += 1
            n_bad = int((q_pack != q_ref).sum().item())
            d = int((q_pack.to(torch.int32) - q_ref.to(torch.int32)).abs().max().item())
            rel = float(((s_pack.flatten() - s_ref.flatten()).abs()
                         / s_ref.flatten().abs().clamp_min(1e-30)).max().item())
            q_mismatch += n_bad
            max_q_delta = max(max_q_delta, d)
            scale_rel_max = max(scale_rel_max, rel)
            recomp_cache[mod] = (q_ref.cpu(), s_ref.cpu())
            if n_bad or d or rel > 0:
                layer_rows.append({"layer": mod, "q_mismatch": n_bad, "max_q_delta": d,
                                   "scale_rel_max": round(rel, 12)})

        float_mismatch = 0
        float_max_abs = 0.0
        float_checked = 0
        for key in sorted(pack_logical - set(int8_keys)):
            if key not in rt or key.endswith((".weight_scale", ".comfy_quant")):
                continue
            got = f.get_tensor(key)
            ref = rt[key]
            if ref.is_floating_point():
                # compare in the dtype the pack stores (fp16 for SDXL packs, bf16 for FLUX packs)
                ref = ref.to(got.dtype)
            float_checked += 1
            if tuple(got.shape) != tuple(ref.shape) or got.dtype != ref.dtype:
                float_mismatch += 1
                continue
            if not torch.equal(got, ref):
                float_mismatch += 1
                if got.is_floating_point():
                    float_max_abs = max(float_max_abs,
                                        float((got.float() - ref.float()).abs().max().item()))

    verdict = ("FAITHFUL"
               if (q_mismatch == 0 and not missing_from_pack and not extra_in_pack and not shape_bad
                   and not undeclared_found and not declared_not_found and float_mismatch == 0
                   and nonfinite_scale == 0)
               else "EDITED")

    out = {
        "tool": "controlnet_pack_audit.py",
        "method": ("runtime-name coverage from comfy.controlnet.load_controlnet_state_dict, then the "
                   "converter's own quantisation replayed on the runtime float weights "
                   "(Hadamard-256 rotation, row-wise amax/127 int8), compared element by element"),
        "source": {"file": os.path.basename(args.source), "bytes": os.path.getsize(args.source),
                   "sha256": src_sha},
        "pack": {"file": os.path.basename(args.pack), "bytes": os.path.getsize(args.pack),
                 "sha256": pack_sha},
        "declared": {"layers": len(declared), "formats": sorted(x for x in formats if x),
                     "convrot": sorted(str(x) for x in rotations if x is not None),
                     "groupsizes": sorted(int(x) for x in groupsizes if x is not None)},
        "runtime_tensors": len(rt),
        "pack_tensors": len(pack_logical),
        "coverage": {"missing_from_pack": missing_from_pack[:40], "n_missing": len(missing_from_pack),
                     "extra_in_pack": extra_in_pack[:40], "n_extra": len(extra_in_pack),
                     "shape_mismatch": shape_bad[:40], "n_shape_mismatch": len(shape_bad)},
        "quantised": {"found": len(found_layers), "declared": len(declared_set),
                      "undeclared_found": undeclared_found[:20], "declared_not_found": declared_not_found[:20],
                      "q_mismatch_total": q_mismatch, "max_q_delta": max_q_delta,
                      "scale_exact_rel_max": scale_rel_max, "nonfinite_scales": nonfinite_scale,
                      "layers_with_any_difference": layer_rows[:20],
                      "n_layers_with_difference": len(layer_rows)},
        "float_side": {"checked": float_checked, "mismatch_count": float_mismatch,
                       "max_abs_diff": float_max_abs},
        "loader_route": (loader_drop(args.pack, args.comfy) if args.loader_drop else None),
        "verdict": verdict,
        "environment": {"device": str(dev), "comfy": os.path.basename(os.path.abspath(args.comfy))},
        "elapsed_seconds": round(__import__("time").time() - t_start, 2),
    }

    print("source %s  %d B" % (src_sha[:16], os.path.getsize(args.source)))
    print("pack   %s  %d B" % (pack_sha[:16], os.path.getsize(args.pack)))
    print("declared %d layers | found %d | runtime %d tensors | pack %d"
          % (len(declared), len(found_layers), len(rt), len(pack_logical)))
    print("coverage: missing %d | extra %d | shape %d"
          % (len(missing_from_pack), len(extra_in_pack), len(shape_bad)))
    print("quantised: q_mismatch %d | max_q_delta %d | scale rel max %.3e | nonfinite %d"
          % (q_mismatch, max_q_delta, scale_rel_max, nonfinite_scale))
    print("float side: checked %d | mismatch %d | max abs diff %.3e"
          % (float_checked, float_mismatch, float_max_abs))
    print("VERDICT: %s (%.1fs)" % (verdict, out["elapsed_seconds"]))

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8", newline="\n") as fp:
            json.dump(out, fp, indent=2)
        print("wrote", args.json)
    return 0 if verdict == "FAITHFUL" else 1


if __name__ == "__main__":
    sys.exit(main())

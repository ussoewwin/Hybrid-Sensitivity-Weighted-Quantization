#!/usr/bin/env python
"""Audit a ConvRot INT8 model patch against its source, in the runtime's own names.

These artifacts are model patches, not ControlNets: the runtime builds them with
``HSWQModelPatchLoaderCustom`` (the same file the card's loader uses) and applies them with the
family's own node. The loader's non-INT8 branch is what defines the runtime names - for the Z-Image
family it converts the checkpoint (``z_image_convert`` fuses attention qkv and renames the output
projection) before the weights are loaded, while the INT8 branch skips that conversion, so a pack
stores the already-converted names. This tool mirrors that: it converts the source exactly as the
loader does, then compares it with the pack tensor by tensor and replays the quantisation of every
declared INT8 layer.

The rotation and the quantisation are not re-implemented here; they come from
``controlnet_pack_audit.py``, which carries the converter's own construction. A replay built on a
different (still valid) Hadamard reports a whole-file mismatch and proves nothing.

Every int8 code that differs from the replay is checked against the only benign cause: the value sits
on a .5 rounding boundary, where the two rounding modes disagree by one step. A rotation accumulated
in a different order moves the value by ~1e-5, so the test is a tolerance (BOUNDARY_TOL) around .5
rather than an exact equality; the largest measured distance is reported with the record so the
tolerance can be judged. A code that differs further from .5 makes the verdict EDITED.

Usage:
    python audit/model_patch_pack_audit.py --source <file> --pack <file> --comfy <root> [--json <out>]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys

import torch

BOUNDARY_TOL = 1e-4
FLOAT_TOL = 1e-6


def _replay_helpers():
    """The converter's own rotation and quantisation, taken from the sibling tool."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "controlnet_pack_audit.py")
    spec = importlib.util.spec_from_file_location("_cn_pack_audit_replay", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_hadamard, mod.rotate_weight, mod.quant_rowwise


def read_header(path):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(n))
    meta = hdr.pop("__metadata__", None) or {}
    return hdr, meta


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_loader_module(comfy_root):
    root = os.path.abspath(comfy_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    cand = os.path.join(root, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools")
    init = os.path.join(cand, "__init__.py")
    name = "_hswq_mp_audit"
    spec = importlib.util.spec_from_file_location(name, init, submodule_search_locations=[cand])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[name] = pkg
    spec.loader.exec_module(pkg)
    return importlib.import_module(name + ".nodes.hswq_model_patch_loader")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    loader = load_loader_module(args.comfy)

    from safetensors import safe_open
    from safetensors.torch import load_file

    src = load_file(args.source, device="cpu")
    pack_hdr, pack_meta = read_header(args.pack)
    declared = json.loads(pack_meta.get("_quantization_metadata", "{}")).get("layers", {})

    family = None
    converted = False
    if "control_all_x_embedder.2-1.weight" in src:
        family = "z_image_fun_controlnet"
        src = loader.z_image_convert(src)
        converted = True
    elif "controlnet_blocks.0.y_rms.weight" in src:
        family = "qwen_image_blockwise_controlnet"
    else:
        raise SystemExit("unknown model-patch family: no supported discriminator key")

    pack_logical = {k: v for k, v in pack_hdr.items()
                    if not k.endswith((".weight_scale", ".comfy_quant"))}
    missing = sorted(set(src) - set(pack_logical))
    extra = sorted(set(pack_logical) - set(src))
    shape_bad = sorted(k for k in set(src) & set(pack_logical)
                       if tuple(src[k].shape) != tuple(pack_hdr[k]["shape"]))

    build_hadamard, rotate_weight, quant_rowwise = _replay_helpers()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    h_cache = {}
    q_mismatch = 0
    q_on_boundary = 0
    q_off_boundary = 0
    boundary_dist_max = 0.0
    max_q_delta = 0
    off_examples = []
    scale_rel_max = 0.0
    nonfinite_scale = 0
    found = 0
    not_found = []
    with safe_open(args.pack, framework="pt", device="cpu") as f:
        for layer, conf in sorted(declared.items()):
            key = layer + ".weight"
            if key not in src or key not in pack_hdr:
                not_found.append(layer)
                continue
            found += 1
            gs = int(conf.get("convrot_groupsize") or 256)
            if gs not in h_cache:
                h_cache[gs] = build_hadamard(gs).to(dev)
            w = src[key].to(dev).float()
            w_rot = rotate_weight(w, h_cache[gs], gs)
            q_ref, s_ref = quant_rowwise(w_rot)
            q_pack = f.get_tensor(key).to(dev).to(torch.int8)
            s_pack = f.get_tensor(layer + ".weight_scale").to(dev).float()
            if not torch.isfinite(s_pack).all():
                nonfinite_scale += 1
            rel = float(((s_ref - s_pack).abs() / s_ref.clamp(min=1e-30)).max())
            scale_rel_max = max(scale_rel_max, rel)

            diff = q_ref.to(torch.int16) != q_pack.to(torch.int16)
            n = int(diff.sum())
            if n:
                q_mismatch += n
                max_q_delta = max(max_q_delta, int((q_ref.to(torch.int16) - q_pack.to(torch.int16)).abs().max()))
                v = (w_rot / s_ref.to(w_rot.dtype))[diff].abs()
                frac = v - v.floor()
                dist = (frac - 0.5).abs()
                on = int((dist <= BOUNDARY_TOL).sum())
                q_on_boundary += on
                q_off_boundary += n - on
                if n:
                    boundary_dist_max = max(boundary_dist_max, float(dist.max()))
                if n - on and len(off_examples) < 5:
                    far = dist > BOUNDARY_TOL
                    off_examples.append({"layer": layer, "count": n - on,
                                         "dist_min": float(dist[far].min()),
                                         "dist_max": float(dist[far].max())})

    float_checked = 0
    float_mismatch = 0
    float_within_tol = 0
    float_max = 0.0
    int8_stems = set(declared)
    with safe_open(args.pack, framework="pt", device="cpu") as f:
        for key in sorted(pack_logical):
            stem = key[:-len(".weight")] if key.endswith(".weight") else None
            if stem in int8_stems:
                continue
            if key not in src:
                continue
            a = f.get_tensor(key).to(dev).float()
            b = src[key].to(dev).float()
            float_checked += 1
            d = float((a - b).abs().max())
            if d > FLOAT_TOL:
                float_mismatch += 1
                float_max = max(float_max, d)
            elif d:
                float_within_tol += 1
                float_max = max(float_max, d)

    faithful = (not missing and not extra and not shape_bad and not not_found
                and q_off_boundary == 0 and float_mismatch == 0)
    rec = {
        "tool": "model_patch_pack_audit.py",
        "family": family,
        "source_converted_like_the_loader": converted,
        "source": {"file": os.path.basename(args.source), "bytes": os.path.getsize(args.source),
                   "sha256": sha256(args.source)},
        "pack": {"file": os.path.basename(args.pack), "bytes": os.path.getsize(args.pack),
                 "sha256": sha256(args.pack)},
        "source_tensors": len(src),
        "pack_tensors": len(pack_logical),
        "coverage": {"n_missing": len(missing), "missing": missing[:20],
                     "n_extra": len(extra), "extra": extra[:20],
                     "n_shape_mismatch": len(shape_bad), "shape_mismatch": shape_bad[:20]},
        "quantised": {"declared": len(declared), "found": found, "declared_not_found": not_found,
                      "groupsizes": sorted(int(v.get("convrot_groupsize") or 256)
                                           for v in declared.values()) if declared else [],
                      "q_mismatch_total": q_mismatch,
                      "q_on_rounding_boundary": q_on_boundary,
                      "q_off_rounding_boundary": q_off_boundary,
                      "boundary_tolerance": BOUNDARY_TOL,
                      "boundary_dist_max": boundary_dist_max,
                      "off_boundary_examples": off_examples,
                      "max_q_delta": max_q_delta,
                      "scale_exact_rel_max": scale_rel_max, "nonfinite_scales": nonfinite_scale},
        "float_side": {"checked": float_checked, "mismatch_count": float_mismatch,
                       "within_tolerance": float_within_tol, "tolerance": FLOAT_TOL,
                       "max_abs_diff": float_max},
        "verdict": "FAITHFUL" if faithful else "EDITED",
    }
    print("%s | family %s | tensors %d/%d | declared %d | q %d (near .5 %d / off %d, max dist %.2e)"
          " | float %d | %s"
          % (rec["pack"]["file"], family, len(src), len(pack_logical), len(declared), q_mismatch,
             q_on_boundary, q_off_boundary, boundary_dist_max, float_mismatch, rec["verdict"]))
    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

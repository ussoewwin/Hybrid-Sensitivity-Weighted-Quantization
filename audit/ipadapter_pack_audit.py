"""SD1.5 IP-Adapter Plus ConvRot INT8 pack audit.

Four questions, each answered by measurement on the two files:

  1. identity   - which bytes the pack is closed against, and the pack's own bytes
  2. coverage   - every source tensor present, no tensor added, no shape changed
  3. arithmetic - the int8 codes and scales are the converter's own values for the
                  source weights, and the tensors the pack does not quantise are
                  the source's, bit for bit
  4. behaviour  - the two networks on the same input: the source's float weights
                  against the pack's dequantised weights

  python audit/pulid_pack_audit.py --source <src> --pack <pack> --ipadapter_dir <node dir> --json <record>
"""

import argparse
import hashlib
import json
import math
import os
import sys

import torch
import torch.nn as nn


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_hadamard(size, device="cpu", dtype=torch.float32):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype, device=device)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def rotate_weight(w, h, gs):
    of, inf = w.shape
    g = inf // gs
    return torch.matmul(w.view(of, g, gs), h.T.to(w.dtype)).reshape(w.shape)


def quant_rowwise(w):
    amax = w.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
    s = amax / 127.0
    q = (w / s.to(w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, s.to(torch.float32)


def dequant_pack_weight(pack, mk, gs):
    q = pack[mk + ".weight"]
    s = pack[mk + ".weight_scale"]
    h = build_hadamard(gs)
    of, inf = q.shape
    g = inf // gs
    deq = (q.float() * s.float()).view(of, g, gs)
    return torch.matmul(deq, h.float()).reshape(of, inf)


class ToKV(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.to_kvs = nn.ModuleDict()
        self.name_map = {}
        for key, value in state_dict.items():
            name = key.replace(".weight", "").replace(".", "_")
            self.name_map[name] = key.replace(".weight", "")
            self.to_kvs[name] = nn.Linear(value.shape[1], value.shape[0], bias=False)
            self.to_kvs[name].weight.data = value


def build_net(state_dict, node_dir):
    """The two stages the node builds: a Resampler over image_proj, one Linear per ip_adapter weight."""
    proj, ipa = {}, {}
    for k, v in state_dict.items():
        if k.startswith("image_proj."):
            proj[k.replace("image_proj.", "", 1)] = v.float() if v.is_floating_point() else v
        elif k.startswith("ip_adapter."):
            ipa[k.replace("ip_adapter.", "", 1)] = v.float() if v.is_floating_point() else v
    sys.path.insert(0, node_dir)
    from image_proj_models import Resampler

    res = Resampler(dim=768, depth=4, dim_head=64, heads=12, num_queries=16,
                    embedding_dim=1280, output_dim=768, ff_mult=4,
                    max_seq_len=257, apply_pos_emb=False, num_latents_mean_pooled=0)
    missing, unexpected = res.load_state_dict(proj, strict=False)
    if [k for k in missing if k.endswith(".weight")]:
        raise SystemExit("image_proj weight mismatch: missing=%s unexpected=%s" % (missing[:5], unexpected[:5]))
    kv = ToKV(ipa)
    return res, kv


def make_inputs(seed, device, dtype):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(1, 257, 1280, generator=g).to(device, dtype)


def run(enc, kv, cond):
    with torch.no_grad():
        embeds = enc(cond)
        outs = [lin(embeds).flatten() for _, lin in sorted(kv.to_kvs.items())]
    return embeds.flatten(), torch.cat(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ipadapter_dir", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--groupsize", type=int, default=256)
    a = ap.parse_args()

    from safetensors.torch import load_file

    if a.source.lower().endswith(".pth"):
        raw = torch.load(a.source, map_location="cpu", weights_only=True)
        src = {}
        for k, v in raw["image_proj"].items():
            if torch.is_tensor(v):
                src["image_proj." + k] = v
        for k, v in raw["ip_adapter"].items():
            if torch.is_tensor(v):
                src["ip_adapter." + k] = v
    else:
        src = load_file(a.source)
    pack = load_file(a.pack)
    meta = None
    with open(a.pack, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        meta = json.loads(f.read(n).decode("utf-8")).get("__metadata__", {})

    q_keys = [k for k in pack if k.endswith(".comfy_quant")]
    declared = sorted(k[: -len(".comfy_quant")] for k in q_keys)
    print(f"source {sha256(a.source)[:16]}  {os.path.getsize(a.source)} B")
    print(f"pack   {sha256(a.pack)[:16]}  {os.path.getsize(a.pack)} B")
    print(f"declared INT8 layers {len(declared)}")

    # coverage in the source's own names
    src_weights = {k for k in src if k.endswith(".weight")}
    pack_weights = {k for k in pack if k.endswith(".weight")}
    missing = sorted(src_weights - pack_weights)
    extra = sorted(pack_weights - src_weights)
    shape_bad = [k for k in src_weights & pack_weights if tuple(src[k].shape) != tuple(pack[k].shape)]
    print(f"coverage: missing {len(missing)} | extra {len(extra)} | shape {len(shape_bad)}")

    # arithmetic: the pack's int8 is the converter's value for the source weight
    q_mismatch_total = 0
    max_q_delta = 0
    scale_rel_max = 0.0
    nonfinite = 0
    layers_bad = []
    for mk in declared:
        w = src[mk + ".weight"].float()
        h = build_hadamard(a.groupsize)
        q_ref, s_ref = quant_rowwise(rotate_weight(w, h, a.groupsize))
        q = pack[mk + ".weight"]
        s = pack[mk + ".weight_scale"].float()
        if not torch.isfinite(s).all():
            nonfinite += 1
        d = (q.to(torch.int32) - q_ref.to(torch.int32)).abs()
        n_bad = int((d > 0).sum().item())
        if n_bad:
            layers_bad.append(mk)
        q_mismatch_total += n_bad
        max_q_delta = max(max_q_delta, int(d.max().item()))
        rel = ((s - s_ref).abs() / s_ref.abs().clamp(min=1e-30)).max().item()
        scale_rel_max = max(scale_rel_max, float(rel))
    print(f"quantised: q_mismatch {q_mismatch_total} | max_q_delta {max_q_delta} | "
          f"scale rel max {scale_rel_max:.3e} | nonfinite {nonfinite}")

    # float side: every tensor the pack does not quantise is the source's, bit for bit
    printed = set()
    for mk in declared:
        printed.add(mk + ".weight")
        printed.add(mk + ".weight_scale")
        printed.add(mk + ".comfy_quant")
    float_keys = [k for k in src if k not in printed]
    f_mismatch = 0
    f_max = 0.0
    for k in float_keys:
        if k not in pack:
            f_mismatch += 1
            continue
        a0 = src[k]
        b0 = pack[k]
        if a0.dtype != b0.dtype or tuple(a0.shape) != tuple(b0.shape):
            f_mismatch += 1
            continue
        if a0.is_floating_point():
            d = (a0.float() - b0.float()).abs().max().item()
            f_max = max(f_max, float(d))
            if d > 0:
                f_mismatch += 1
        else:
            if not torch.equal(a0, b0):
                f_mismatch += 1
    print(f"float side: checked {len(float_keys)} | mismatch {f_mismatch} | max abs diff {f_max:.3e}")

    # behaviour: source float weights against the pack's dequantised weights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    deq = {}
    for mk in declared:
        deq[mk + ".weight"] = dequant_pack_weight(pack, mk, a.groupsize).to(dtype)
    enc_s, kv_s = build_net(src, a.ipadapter_dir)
    enc_s.to(device, dtype).eval()
    kv_s.to(device, dtype).eval()
    declared_set = set(declared)
    restored = {}
    for k, v in pack.items():
        if k.endswith(".weight_scale") or k.endswith(".comfy_quant"):
            continue
        if k.endswith(".weight") and k[: -len(".weight")] in declared_set:
            restored[k] = deq[k]
        else:
            restored[k] = v
    enc_q, kv_q = build_net(restored, a.ipadapter_dir)
    enc_q.to(device, dtype).eval()
    kv_q.to(device, dtype).eval()
    cond = make_inputs(a.seed, device, dtype)
    e_s, o_s = run(enc_s, kv_s, cond)
    e_q, o_q = run(enc_q, kv_q, cond)
    cos_e = torch.nn.functional.cosine_similarity(e_s.unsqueeze(0), e_q.unsqueeze(0)).item()
    cos_o = torch.nn.functional.cosine_similarity(o_s.unsqueeze(0), o_q.unsqueeze(0)).item()
    rl2_e = float((torch.norm(e_q - e_s) / torch.norm(e_s).clamp(min=1e-30)).item())
    rl2_o = float((torch.norm(o_q - o_s) / torch.norm(o_s).clamp(min=1e-30)).item())
    print(f"behaviour: embeds cos {cos_e:.9f} rel_l2 {rl2_e:.6e} | to_kv cos {cos_o:.9f} rel_l2 {rl2_o:.6e}")

    verdict = "FAITHFUL" if (not missing and not extra and not shape_bad
                             and q_mismatch_total == 0 and nonfinite == 0 and f_mismatch == 0) else "MISMATCH"
    print(f"VERDICT: {verdict}")

    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    rec = {
        "tool": "ipadapter_pack_audit.py",
        "source": {"file": os.path.basename(a.source), "bytes": os.path.getsize(a.source), "sha256": sha256(a.source)},
        "pack": {"file": os.path.basename(a.pack), "bytes": os.path.getsize(a.pack), "sha256": sha256(a.pack)},
        "declared": {"layers": len(declared), "format": "int8_tensorwise", "convrot": True, "groupsize": a.groupsize},
        "coverage": {"n_missing": len(missing), "n_extra": len(extra), "n_shape_mismatch": len(shape_bad),
                     "missing": missing, "extra": extra, "shape_mismatch": shape_bad},
        "quantised": {"found": len(declared), "q_mismatch_total": q_mismatch_total, "max_q_delta": max_q_delta,
                      "scale_exact_rel_max": scale_rel_max, "nonfinite_scales": nonfinite,
                      "n_layers_with_difference": len(layers_bad)},
        "float_side": {"checked": len(float_keys), "mismatch_count": f_mismatch, "max_abs_diff": f_max},
        "behaviour": {"embeds_cos": cos_e, "embeds_rel_l2": rl2_e, "to_kv_cos": cos_o, "to_kv_rel_l2": rl2_o,
                      "seed": a.seed, "input": "fixed synthetic tensors at the shapes the node feeds"},
        "verdict": verdict,
        "elapsed_seconds": None,
    }
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print(f"wrote {a.json}")


if __name__ == "__main__":
    main()

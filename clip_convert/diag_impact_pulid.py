"""PuLID SDXL per-layer ConvRot INT8 impact (individual measurement).

The HSWQ method, applied to the PuLID image projection network:

  1. build the network the pulid_comfyui node builds from the same state dict
     (IDEncoder for `image_proj.*`, To_KV for `ip_adapter.*`),
  2. run it once on a fixed input and record every Linear's contribution,
  3. replace one Linear's weight with its ConvRot INT8 reconstruction, run again,
     and record the relative MSE of the stage output that layer feeds,
  4. restore, and move to the next Linear.

The quantisation replay is the converter's own: Hadamard by Kronecker doubling,
normalised to orthonormal, group size a power of four, row-wise `amax/127` int8.

  python clip_convert/diag_impact_pulid.py --pulid <pulid.safetensors> --out <impact.json> --pulid_dir <node dir>
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn as nn


def build_hadamard(size, device="cpu", dtype=torch.float32):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype, device=device)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def convrot_group_size(n, preferred=256):
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def rotate_weight(w, h, gs):
    of, inf = w.shape
    g = inf // gs
    return torch.matmul(w.view(of, g, gs), h.T.to(w.dtype)).reshape(w.shape)


def quant_rowwise(w):
    amax = w.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
    s = amax / 127.0
    q = (w / s.to(w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, s.to(torch.float32)


def convrot_int8_recon(w, gs):
    h = build_hadamard(gs, device=w.device)
    of, inf = w.shape
    g = inf // gs
    q, s = quant_rowwise(rotate_weight(w.float(), h, gs))
    deq = (q.float() * s.float()).view(of, g, gs)
    return torch.matmul(deq, h.float()).reshape(w.shape).to(w.dtype)


class ToKV(nn.Module):
    """The pulid_comfyui To_KV: one bias-free Linear per ip_adapter weight."""

    def __init__(self, state_dict):
        super().__init__()
        self.to_kvs = nn.ModuleDict()
        self.name_map = {}
        for key, value in state_dict.items():
            name = key.replace(".weight", "").replace(".", "_")
            self.name_map[name] = key.replace(".weight", "")
            self.to_kvs[name] = nn.Linear(value.shape[1], value.shape[0], bias=False)
            self.to_kvs[name].weight.data = value


def load_pulid(path, pulid_dir):
    from safetensors.torch import load_file

    sd = load_file(path)
    proj = {}
    ipa = {}
    for k, v in sd.items():
        if k.startswith("image_proj."):
            proj[k.replace("image_proj.", "", 1)] = v
        elif k.startswith("ip_adapter."):
            ipa[k.replace("ip_adapter.", "", 1)] = v
    sys.path.insert(0, pulid_dir)
    from encoders import IDEncoder

    enc = IDEncoder()
    missing, unexpected = enc.load_state_dict(proj, strict=False)
    if missing or unexpected:
        raise SystemExit("IDEncoder key mismatch: missing=%s unexpected=%s" % (missing, unexpected))
    kv = ToKV(ipa)
    return enc, kv


def make_inputs(seed, device, dtype):
    g = torch.Generator(device="cpu").manual_seed(seed)
    iface = torch.randn(1, 512, generator=g).to(device, dtype)
    eva = torch.randn(1, 768, generator=g).to(device, dtype)
    iface = iface / iface.norm(2, 1, True)
    eva = eva / eva.norm(2, 1, True)
    id_cond = torch.cat([iface, eva], dim=-1)
    hidden = [torch.randn(1, 257, 1024, generator=g).to(device, dtype) for _ in range(5)]
    return id_cond, hidden


def forward_all(enc, kv, id_cond, hidden):
    embeds = enc(id_cond, hidden)
    outs = {}
    with torch.no_grad():
        for name, lin in kv.to_kvs.items():
            outs[name] = lin(embeds)
    return embeds, outs


def rel_mse(a, b):
    num = torch.sum((a - b) ** 2)
    den = torch.sum(b ** 2).clamp(min=1e-30)
    return float((num / den).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pulid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pulid_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--groupsize", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print(f"[load] pulid: {args.pulid}", flush=True)
    enc, kv = load_pulid(os.path.abspath(args.pulid), args.pulid_dir)
    enc.to(device, dtype).eval()
    kv.to(device, dtype).eval()

    id_cond, hidden = make_inputs(args.seed, device, dtype)
    print(f"[input] id_cond {tuple(id_cond.shape)} hidden {[tuple(h.shape) for h in hidden]}", flush=True)

    with torch.no_grad():
        embeds_ref, outs_ref = forward_all(enc, kv, id_cond, hidden)
    print(f"[ref] embeds {tuple(embeds_ref.shape)} | to_kv {len(outs_ref)}", flush=True)

    targets = []
    for n, m in enc.named_modules():
        if isinstance(m, nn.Linear):
            targets.append(("image_proj." + n, m, "encoder", None))
    for n, m in kv.named_modules():
        if isinstance(m, nn.Linear):
            norm = n.replace("to_kvs.", "", 1)
            targets.append(("ip_adapter." + kv.name_map[norm], m, "tokv", norm))
    print(f"[target] Linear modules: {len(targets)}", flush=True)

    impacts = {}
    for key, mod, stage, outname in targets:
        w0 = mod.weight.data
        gs = convrot_group_size(int(w0.shape[1]), args.groupsize)
        if gs is None:
            print(f"  [skip] {key}: no group size for in={w0.shape[1]}", flush=True)
            continue
        with torch.no_grad():
            rec = convrot_int8_recon(w0.float(), gs)
            mod.weight.data = rec.to(w0.dtype)
            embeds_q, outs_q = forward_all(enc, kv, id_cond, hidden)
            mod.weight.data = w0
        if stage == "encoder":
            imp = rel_mse(embeds_q, embeds_ref)
        else:
            imp = rel_mse(outs_q[outname], outs_ref[outname])
        impacts[key] = imp
        print(f"  [{len(impacts)}/{len(targets)}] {key}: rel_mse={imp:.6e}", flush=True)
        del embeds_q, outs_q
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"pulid": os.path.basename(args.pulid), "seed": args.seed,
                   "stage": "encoder+tokv, stage-output rel_mse", "impacts": impacts}, f, indent=2)
    ranked = sorted(impacts.items(), key=lambda x: x[1])
    print("\n== ascending drift (safest to quantize first) ==", flush=True)
    for n, v in ranked[:10]:
        print(f"  {v:.6e}  {n}", flush=True)
    print("== descending drift (must stay FP16) ==", flush=True)
    for n, v in ranked[-10:]:
        print(f"  {v:.6e}  {n}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

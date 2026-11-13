"""SD1.5 IP-Adapter Plus weight-level trigger search.

The control-image search the ControlNet packs use needs a control image going
through a ControlNet route; this model has no such route here. What this tool does
instead is the weight-level search: a rank-1 implant is written into one Linear
weight at a known relative strength, the network is run, and the divergence of the
stage output is compared with the distribution the untouched network produces
when the same input is run twice with different thread / allocator noise.

The floor is the smallest strength whose divergence leaves that control
distribution. A backdoor of this shape, below the floor, would not be seen by this
instrument - that is what the floor is for.

  python audit/pulid_weight_trigger.py --ipadapter <f.pth> --ipadapter_dir <node dir> --json <record>
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn


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


def build(state_dict, node_dir):
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
    res.load_state_dict(proj, strict=False)
    kv = ToKV(ipa)
    return res, kv


def inputs(seed, device, dtype):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 257, 1280, generator=g).to(device, dtype)


def run(enc, kv, cond):
    with torch.no_grad():
        e = enc(cond)
        o = torch.cat([lin(e).flatten() for _, lin in sorted(kv.to_kvs.items())])
    return e.flatten(), o


def one_minus_cos(a, b):
    return 1.0 - torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipadapter", required=True)
    ap.add_argument("--ipadapter_dir", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    from safetensors.torch import load_file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    raw = torch.load(args.ipadapter, map_location="cpu", weights_only=True)
    sd = {}
    for k, v in raw["image_proj"].items():
        if torch.is_tensor(v):
            sd["image_proj." + k] = v
    for k, v in raw["ip_adapter"].items():
        if torch.is_tensor(v):
            sd["ip_adapter." + k] = v
    enc, kv = build(sd, args.ipadapter_dir)
    enc.to(device, dtype).eval()
    kv.to(device, dtype).eval()
    cond = inputs(args.seed, device, dtype)
    e_ref, o_ref = run(enc, kv, cond)

    # control: the same network on the same input perturbed by a natural amount of
    # noise - the divergence here is what an untouched network produces, and it is
    # the distribution an implant has to leave to count
    ctrl_e, ctrl_o = [], []
    for r in range(args.reps):
        sig = 10.0 ** (-4 - r)
        g2 = torch.Generator().manual_seed(1000 + r)
        cond2 = cond + torch.randn(cond.shape, generator=g2).to(cond.device) * sig
        e2, o2 = run(enc, kv, cond2)
        ctrl_e.append(one_minus_cos(e_ref, e2))
        ctrl_o.append(one_minus_cos(o_ref, o2))
    ctrl_e_max = max(ctrl_e)
    ctrl_o_max = max(ctrl_o)
    p99_e = ctrl_e_max * 1.5
    p99_o = ctrl_o_max * 1.5

    # implant target: the widest to_v_ip layer, the one the sweep ranked highest
    target = max(((n, m) for n, m in kv.named_modules() if isinstance(m, nn.Linear)),
                 key=lambda x: x[1].weight.shape[0])
    name, mod = target
    orig = mod.weight.data.clone()
    w_norm = float(orig.norm())

    results = []
    floor = None
    for strength in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        u = torch.randn(orig.shape[0], 1, generator=torch.Generator().manual_seed(7)).to(orig.device)
        v = torch.randn(1, orig.shape[1], generator=torch.Generator().manual_seed(11)).to(orig.device)
        u = u / u.norm()
        v = v / v.norm()
        delta = (u @ v) * (strength * w_norm)
        mod.weight.data = orig + delta.to(orig.dtype)
        e_i, o_i = run(enc, kv, cond)
        mod.weight.data = orig
        d_e = one_minus_cos(e_ref, e_i)
        d_o = one_minus_cos(o_ref, o_i)
        flagged = (d_e > p99_e) or (d_o > p99_o)
        if flagged and floor is None:
            floor = strength
        results.append({"strength": strength, "embeds_1mcos": d_e, "to_kv_1mcos": d_o, "flagged": flagged})
        print("  strength %.3f  embeds %.3e  to_kv %.3e  %s"
              % (strength, d_e, d_o, "FLAGGED" if flagged else "-"), flush=True)

    print("control p99 (embeds) %.3e | control p99 (to_kv) %.3e" % (p99_e, p99_o), flush=True)
    print("detection floor %.3f" % (floor if floor is not None else -1), flush=True)

    rec = {
        "tool": "ipadapter_weight_trigger.py",
        "ipadapter": os.path.basename(args.ipadapter),
        "observable": "the divergence (1-cos) of the embedding stage and of the k/v stack when a rank-1 implant is written into one Linear weight, against the same network's repeat-run divergence",
        "implant": {"shape": "rank-1", "layer": "ip_adapter." + kv.name_map[name.replace("to_kvs.", "", 1)],
                    "weight_frobenius": w_norm},
        "control": {"reps": args.reps, "embeds_1mcos": ctrl_e, "to_kv_1mcos": ctrl_o,
                    "control_p99_embeds": p99_e, "control_p99_to_kv": p99_o},
        "results": results,
        "detection_floor_relative_strength": floor,
        "seed": args.seed,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print("wrote", args.json, flush=True)


if __name__ == "__main__":
    main()

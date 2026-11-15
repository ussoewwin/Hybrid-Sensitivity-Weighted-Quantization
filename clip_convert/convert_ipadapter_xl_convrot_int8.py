"""SDXL basic IP-Adapter (`ip-adapter_xl.pth`) ConvRot INT8 converter.

Takes the per-layer impact this checkpoint's own measurement produced and writes a
pack in the same form the other HSWQ ConvRot INT8 packs use:

  <module>.weight        int8, rotated by the Hadamard group transform
  <module>.weight_scale  float32, row-wise amax/127 of the rotated weight
  <module>.comfy_quant   uint8, the layer's quantisation config

A layer stays FP16 when its measured drift exceeds --drift-threshold, or when its
shape cannot carry the 256-way group transform.

The source is the `.pth` the node reads: a nested `image_proj` / `ip_adapter` dict,
flattened here to the `image_proj.*` / `ip_adapter.*` names the safetensors loader
rebuilds. That flattening is why this file exists separately from the safetensors
converters.

  python clip_convert/convert_ipadapter_xl_convrot_int8.py --input <ip-adapter_xl.pth> --impact <impact.json> --output <pack.safetensors>
"""

import argparse
import json
import os

import torch
from safetensors.torch import save_file

GROUP_SIZE = 256
DEFAULT_DRIFT_THRESHOLD = 0.15


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


def _encode(config):
    return torch.tensor(list(json.dumps(config, separators=(",", ":")).encode("utf-8")), dtype=torch.uint8)


def flatten(sd):
    """The node's nested dict into `image_proj.*` / `ip_adapter.*` names."""
    out = {}
    for head in ("image_proj", "ip_adapter"):
        block = sd.get(head)
        if not isinstance(block, dict):
            continue
        for k, v in block.items():
            if torch.is_tensor(v):
                out[head + "." + k] = v
    if not out:
        raise SystemExit("the source carries neither image_proj nor ip_adapter")
    return out


def convert(input_path, impact_path, output_path, drift_threshold=DEFAULT_DRIFT_THRESHOLD):
    src = flatten(torch.load(input_path, map_location="cpu", weights_only=True))
    impacts = json.load(open(impact_path, encoding="utf-8"))["impacts"]
    h = build_hadamard(GROUP_SIZE)

    out = {}
    meta_layers = {}
    n_lin = n_prot = n_kept = 0
    protected = []

    for name, t in src.items():
        if not (name.endswith(".weight") and t.dim() == 2):
            out[name] = t.to(torch.float16) if t.is_floating_point() else t
            n_kept += 1
            continue
        mk = name[: -len(".weight")]
        drift = impacts.get(mk)
        if drift is None:
            out[name] = t.to(torch.float16)
            n_kept += 1
            continue
        inf, of = int(t.shape[1]), int(t.shape[0])
        if drift > drift_threshold:
            out[name] = t.to(torch.float16)
            n_prot += 1
            protected.append({"layer": mk, "reason": "drift %.6e > %.2f" % (drift, drift_threshold)})
            continue
        if inf % GROUP_SIZE != 0 or of % 4 != 0 or inf % 4 != 0:
            out[name] = t.to(torch.float16)
            n_prot += 1
            protected.append({"layer": mk, "reason": "shape in=%d out=%d" % (inf, of)})
            continue
        q, scale = quant_rowwise(rotate_weight(t.float(), h, GROUP_SIZE))
        conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP_SIZE}
        out[name] = q
        out[mk + ".weight_scale"] = scale
        out[mk + ".comfy_quant"] = _encode(conf)
        meta_layers[mk] = conf
        n_lin += 1

    metadata = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": meta_layers},
                                                     separators=(",", ":"))}
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_file(out, output_path, metadata=metadata)
    print("Saving: %s" % output_path)
    print("  ConvRot INT8 Linear: %d" % n_lin)
    print("  protected (drift or shape): %d" % n_prot)
    print("  kept (not a measured Linear): %d" % n_kept)
    print("  Output %.2f GB" % (os.path.getsize(output_path) / (1024 ** 3)))
    return {"int8_linear": n_lin, "protected": n_prot, "kept": n_kept, "protected_layers": protected}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SDXL basic IP-Adapter ConvRot INT8 converter")
    ap.add_argument("--input", required=True, help="the ip-adapter .pth the node reads")
    ap.add_argument("--impact", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--drift-threshold", type=float, default=DEFAULT_DRIFT_THRESHOLD)
    a = ap.parse_args()
    convert(a.input, a.impact, a.output, a.drift_threshold)

# -*- coding: utf-8 -*-
"""Reverse hybrid converter (Flux1): all-INT8 (rotated) -> convert K lowest-impact layers to ConvRot W4A8.

Reverse hybrid W4A8 method: the impact json from Flux1/diag_impact.py ranks every layer;
the K lowest-impact layers move from ConvRot INT8 to ConvRot W4A8, the rest stay INT8
(the Flux1 port of the reverse hybrid idea, with the W4A8 packer).

Weights in the all-INT8 artifact are already ROTATED (W_rot = W H^T). The W4A8 packer
(comfy_kitchen.quantize_w4a8_int8_weight) rotates INSIDE, so the rotated dequant is
un-rotated first (W = W_rot H, H symmetric normalized) and then quantized - otherwise the
runtime would rotate activations once while the weight carries a double rotation.

Only layers whose base stamp is ConvRot INT8 with convrot_groupsize 256 are eligible
(the kitchen W4A8 CUDA requant path supports convrot group 256 only); plain INT8 layers
and boundary layers stay INT8.

Usage:
    python Flux1/gen_reverse_w4a8.py <K> <out_name.safetensors> <src_all_int8.safetensors> <impact.json> \\
        [--out-dir <output-dir>]

Storage spec (comfy_kitchen AsymW4A8Int8Layout):
.weight int8 [N, K/2] packed nibbles, .weight_s_rel fp8e4m3 [N, K/16],
.weight_s_channel f32 [N], .weight_codebook f32 [16] (when the Lloyd-Max table is used),
.comfy_quant {"format":"asym_w4a8_int8","group_size":16,"convrot_groupsize":256}.
"""
import argparse
import json
import math
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import comfy_kitchen as ck

_QUANT_GROUP_SIZE = 16
_CONVROT_GROUPSIZE = 256

# Flux1 boundary layers (adaLN modulation + input/output heads) stay INT8 (never W4A8)
_BOUNDARY_MARKERS = (
    "img_in.", "txt_in.", "time_in.", "vector_in.", "guidance_in.",
    "final_layer.", "mod", "norm",
)

_HADAMARD_CACHE: dict[tuple[int, torch.dtype], torch.Tensor] = {}


def _is_boundary(key: str) -> bool:
    return any(b in key for b in _BOUNDARY_MARKERS)


def _hadamard(size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Normalized regular Hadamard (power-of-4), same construction as the converters."""
    cache_key = (size, dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]

    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")

    h4 = torch.tensor(
        [
            [1, 1, 1, -1],
            [1, 1, -1, 1],
            [1, -1, 1, 1],
            [-1, 1, 1, 1],
        ],
        dtype=dtype,
    )
    h_matrix = h4
    current = 4
    while current < size:
        h_matrix = torch.kron(h_matrix, h4)
        current *= 4
    h_matrix = h_matrix / (size**0.5)
    _HADAMARD_CACHE[cache_key] = h_matrix
    return h_matrix


def _unrotate(weight_rot: torch.Tensor, group_size: int) -> torch.Tensor:
    """W = W_rot H per group (W_rot = W H^T, H symmetric normalized -> H^-1 = H)."""
    out_features, in_features = weight_rot.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features {in_features} not divisible by group_size {group_size}"
        )
    h_matrix = _hadamard(group_size, torch.float32)
    groups = in_features // group_size
    grouped = weight_rot.view(out_features, groups, group_size)
    return torch.matmul(grouped, h_matrix).reshape(weight_rot.shape)


def parse_args():
    ap = argparse.ArgumentParser(description="Reverse hybrid W4A8 converter (Flux1 INT8 -> W4A8)")
    ap.add_argument("k", type=int, help="number of lowest-impact layers to convert to W4A8")
    ap.add_argument("out_name", help="output filename, e.g. <model>_hybrid_w{K}_convrot_w4a8.safetensors")
    ap.add_argument("src", help="all-INT8 ConvRot safetensors (rotated)")
    ap.add_argument("impact", help="impact json from diag_impact.py")
    ap.add_argument("--out-dir", default=".", help="output directory")
    return ap.parse_args()


def main():
    a = parse_args()
    OUT = os.path.join(a.out_dir, a.out_name)

    imp = json.load(open(a.impact, encoding="utf-8"))["impacts"]
    ranked = [k[: -len(".weight")] if k.endswith(".weight") else k
              for k, _ in sorted(imp.items(), key=lambda kv: kv[1])]
    print(f"ranked layers available: {len(ranked)}")

    with safe_open(a.src, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        raw_meta = f.metadata() or {}
        meta_raw_str = raw_meta.get("_quantization_metadata", '{"layers":{}}')
        meta = json.loads(meta_raw_str)
        if "layers" not in meta or not isinstance(meta["layers"], dict):
            meta["layers"] = {}
        sd = {k: f.get_tensor(k) for k in keys}

    prefix = "model.diffusion_model." if any(
        k.startswith("model.diffusion_model.") for k in keys
    ) else ""

    n_conv = 0
    n_skip_boundary = 0
    n_skip_not_256 = 0
    for L in ranked[: a.k]:
        L_stripped = L[len(prefix):] if prefix and L.startswith(prefix) else L

        if _is_boundary(L_stripped):
            n_skip_boundary += 1
            continue

        wk = prefix + L_stripped + ".weight"
        sk = prefix + L_stripped + ".weight_scale"
        if wk not in sd:
            wk = L + ".weight"
            sk = L + ".weight_scale"
        if wk not in sd or sk not in sd:
            continue
        base_k = wk[: -len(".weight")]

        # Only ConvRot INT8 layers with group 256 can become W4A8 (CUDA requant gate).
        layer_conf = meta["layers"].get(base_k) or meta["layers"].get(L_stripped) or {}
        if (not layer_conf.get("convrot")) or int(
            layer_conf.get("convrot_groupsize", 0)
        ) != _CONVROT_GROUPSIZE:
            n_skip_not_256 += 1
            continue

        q = sd[wk]            # I8 rotated
        s = sd[sk]            # F32 [out,1] (or scalar)
        dq_rot = q.float() * s  # W_rot approx in fp32
        w_fp32 = _unrotate(dq_rot, _CONVROT_GROUPSIZE)

        qdata, s_rel, s_channel, _correction, codebook = ck.quantize_w4a8_int8_weight(
            w_fp32,
            group_size=_QUANT_GROUP_SIZE,
            convrot_groupsize=_CONVROT_GROUPSIZE,
            symmetric=True,
            scale_dtype=torch.float8_e4m3fn,
            codebook=True,
            bits=4,
        )

        del sd[wk], sd[sk]
        sd[base_k + ".weight"] = qdata.to(torch.int8)
        sd[base_k + ".weight_s_rel"] = s_rel.to(torch.float8_e4m3fn).contiguous()
        sd[base_k + ".weight_s_channel"] = s_channel.float().contiguous()
        if codebook is not None:
            sd[base_k + ".weight_codebook"] = codebook.float().contiguous()

        conf = {
            "format": "asym_w4a8_int8",
            "group_size": _QUANT_GROUP_SIZE,
            "convrot_groupsize": _CONVROT_GROUPSIZE,
        }
        sd[base_k + ".comfy_quant"] = torch.frombuffer(
            json.dumps(conf, separators=(",", ":")).encode("utf-8"), dtype=torch.uint8
        ).clone()
        meta["layers"][base_k] = conf
        meta["layers"][L_stripped] = conf
        n_conv += 1
        print(f"  w4a8: {base_k}  ({tuple(w_fp32.shape)})")

    print(
        f"converted {n_conv} layers to W4A8 "
        f"(skipped boundary={n_skip_boundary}, non-256-group={n_skip_not_256})"
    )
    out_meta = {}
    for k, v in raw_meta.items():
        if k == "_quantization_metadata":
            out_meta[k] = json.dumps(meta)
        else:
            out_meta[k] = v.decode("utf-8") if isinstance(v, bytes) else v
    if "_quantization_metadata" not in out_meta:
        out_meta["_quantization_metadata"] = json.dumps(meta)
    save_file(sd, OUT, metadata=out_meta)
    print("saved:", OUT, os.path.getsize(OUT) / 1e9, "GB (decimal)")


if __name__ == "__main__":
    main()

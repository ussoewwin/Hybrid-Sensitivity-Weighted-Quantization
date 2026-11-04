# -*- coding: utf-8 -*-
"""S0. SAM3.1 input verification and complete 1,590-tensor census and classification.

Implements Appendix A.2 (S0) of Backdoor_Audit_Improvement_Plan.md:
  1. Recomputes the SHA256 and byte length of sam3.1_multiplex_fp16.safetensors and verifies
     against 9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03 (1,745,546,848 B).
     Halts with INCONCLUSIVE if mismatched.
  2. Parses the safetensors header to assign every one of the 1,590 tensors to:
     (a) group_summary (A.2 mutually-exclusive partition across the 7 weight-matrix module
         prefixes plus embeddings_pos_ln_bias_scalar, total = 1,590, unclassified = 0),
     (b) module_summary (all 1,590 tensors partitioned by their 7 architectural prefixes),
     (c) structural_roles_summary (all 1,590 tensors partitioned by structural role).
  3. Outputs tensor count, parameter count, and dimensional distribution (1D, 2D, 3D, 4D)
     per group to md/backdoor/verdicts/records/sam31_census.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time

EXPECTED_SHA256 = "9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03"
EXPECTED_BYTES = 1745546848
EXPECTED_LOGICAL_TENSORS = 1590
EXPECTED_TOTAL_PARAMETERS = 872661740

EMBEDDING_POSITION_KEYS = frozenset({
    "detector.backbone.language_backbone.encoder.token_embedding.weight",
    "detector.backbone.language_backbone.encoder.positional_embedding",
    "detector.backbone.vision_backbone.trunk.pos_embed",
    "detector.geometry_encoder.cls_embed.weight",
    "detector.geometry_encoder.label_embed.weight",
    "detector.transformer.decoder.query_embed.weight",
    "tracker.model.image_pe_layer.positional_encoding_gaussian_matrix",
    "tracker.model.interactive_sam_prompt_encoder.no_mask_embed.weight",
    "tracker.model.interactive_sam_prompt_encoder.not_a_point_embed.weight",
    "tracker.model.interactive_sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",
    "tracker.model.interactive_sam_prompt_encoder.point_embeddings.0.weight",
    "tracker.model.interactive_sam_prompt_encoder.point_embeddings.1.weight",
    "tracker.model.interactive_sam_prompt_encoder.point_embeddings.2.weight",
    "tracker.model.interactive_sam_prompt_encoder.point_embeddings.3.weight",
    "tracker.model.interactivity_no_mem_embed",
    "tracker.model.maskmem_tpos_enc",
    "tracker.model.no_obj_embed_spatial",
    "tracker.model.output_invalid_embed",
    "tracker.model.output_valid_embed",
})

SEQUENTIAL_NORM_PREFIXES = (
    "interactive_sam_mask_decoder.output_upscaling.1.",
    "sam_mask_decoder.output_upscaling.1.",
    "interactive_sam_prompt_encoder.mask_downscaling.1.",
    "interactive_sam_prompt_encoder.mask_downscaling.4.",
    "maskmem_backbone.mask_downsampler.encoder.1.",
    "maskmem_backbone.mask_downsampler.encoder.4.",
    "maskmem_backbone.mask_downsampler.encoder.7.",
    "maskmem_backbone.mask_downsampler.encoder.10.",
)

MODULE_NAMES = (
    "detector.backbone.language",
    "detector.backbone.vision",
    "detector.geometry_encoder",
    "detector.transformer",
    "detector.segmentation_head",
    "detector.dot_prod_scoring",
    "tracker.model",
)

A2_GROUP_NAMES = MODULE_NAMES + ("embeddings_pos_ln_bias_scalar",)

ROLE_NAMES = (
    "linear_weight_2d",
    "conv_weight_4d",
    "embedding_position",
    "norm_weight",
    "norm_bias",
    "linear_conv_bias",
    "layer_scale_gamma",
    "scalar",
)


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024 * 16):
            h.update(chunk)
    return h.hexdigest()


def read_safetensors_header(path: str) -> dict[str, dict]:
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError("Invalid safetensors file: truncated header length")
        header_len = struct.unpack("<Q", raw_len)[0]
        header_json = f.read(header_len).decode("utf-8")
    header = json.loads(header_json)
    header.pop("__metadata__", None)
    return header


def classify_module(key: str) -> str:
    if key.startswith("detector.backbone.language_backbone."):
        return "detector.backbone.language"
    if key.startswith("detector.backbone.vision_backbone."):
        return "detector.backbone.vision"
    if key.startswith("detector.geometry_encoder."):
        return "detector.geometry_encoder"
    if key.startswith("detector.transformer."):
        return "detector.transformer"
    if key.startswith("detector.segmentation_head."):
        return "detector.segmentation_head"
    if key.startswith("detector.dot_prod_scoring."):
        return "detector.dot_prod_scoring"
    if key.startswith("tracker.model."):
        return "tracker.model"
    return "unclassified"


def is_norm_parameter(key: str, shape: list[int]) -> bool:
    if len(shape) != 1:
        return False
    low = key.lower()
    if "norm" in low or "ln_" in low:
        return True
    return any(p in key for p in SEQUENTIAL_NORM_PREFIXES)


def classify_structural_role(key: str, shape: list[int]) -> str:
    if key in EMBEDDING_POSITION_KEYS:
        return "embedding_position"
    if key.endswith(".gamma") and len(shape) == 1:
        return "layer_scale_gamma"
    if is_norm_parameter(key, shape):
        if key.endswith("bias"):
            return "norm_bias"
        return "norm_weight"
    if key.endswith(("bias", "in_proj_bias")) and len(shape) == 1:
        return "linear_conv_bias"
    if len(shape) == 2 and key.endswith((".weight", ".in_proj_weight")):
        return "linear_weight_2d"
    if len(shape) == 4 and key.endswith(".weight"):
        return "conv_weight_4d"
    if len(shape) == 0:
        return "scalar"
    return "unclassified"


def classify_a2_group(mod_name: str, role_name: str) -> str:
    if mod_name == "unclassified" or role_name == "unclassified":
        return "unclassified"
    if role_name in ("linear_weight_2d", "conv_weight_4d"):
        return mod_name
    return "embeddings_pos_ln_bias_scalar"


def empty_bucket(with_roles: bool = False) -> dict:
    b = {
        "tensors": 0,
        "params": 0,
        "dims": {"1D": 0, "2D": 0, "3D": 0, "4D": 0, "other": 0},
    }
    if with_roles:
        b["roles"] = {}
    return b


def add_dim(dims_dict: dict[str, int], ndim: int) -> None:
    label = f"{ndim}D" if ndim in (1, 2, 3, 4) else "other"
    dims_dict[label] += 1


def resolve_model_path(cli_arg: str) -> str:
    if cli_arg:
        return cli_arg
    env_src = os.environ.get("SAM31_SOURCE", "")
    if env_src:
        return env_src
    comfy = os.environ.get("COMFY_PATH", "")
    if comfy:
        return os.path.join(comfy, "models", "unet", "sam3.1_multiplex_fp16.safetensors")
    return ""


def main() -> int:
    t0 = time.perf_counter()
    ap = argparse.ArgumentParser(description="SAM3.1 Census (S0)")
    ap.add_argument("--model", default="", help="Path to sam3.1_multiplex_fp16.safetensors (or set SAM31_SOURCE / COMFY_PATH)")
    ap.add_argument("--out", default="md/backdoor/verdicts/records/sam31_census.json", help="Output JSON path")
    args = ap.parse_args()

    model_path = resolve_model_path(args.model)
    if not model_path or not os.path.exists(model_path):
        print("Error: sam3.1_multiplex_fp16.safetensors path not provided or not found. Pass --model <path> or set SAM31_SOURCE.")
        return 2

    actual_bytes = os.path.getsize(model_path)
    actual_sha256 = compute_sha256(model_path)
    sha_match = (actual_sha256.lower() == EXPECTED_SHA256.lower())
    byte_match = (actual_bytes == EXPECTED_BYTES)

    if not sha_match or not byte_match:
        inconclusive = {
            "file": os.path.basename(model_path),
            "status": "INCONCLUSIVE",
            "measured_sha256": actual_sha256,
            "measured_bytes": actual_bytes,
            "expected_sha256": EXPECTED_SHA256,
            "expected_bytes": EXPECTED_BYTES,
            "sha256_match": sha_match,
            "bytes_match": byte_match,
            "gate_s0_pass": False,
        }
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fp:
            json.dump(inconclusive, fp, indent=2)
        print("FAIL: SHA256 or byte count mismatch -> INCONCLUSIVE")
        return 1

    header = read_safetensors_header(model_path)

    a2_groups: dict[str, dict] = {g: empty_bucket(with_roles=True) for g in A2_GROUP_NAMES + ("unclassified",)}
    modules: dict[str, dict] = {m: empty_bucket(with_roles=True) for m in MODULE_NAMES + ("unclassified",)}
    roles: dict[str, dict] = {r: empty_bucket(with_roles=False) for r in ROLE_NAMES + ("unclassified",)}

    total_params = 0
    dtypes_seen: dict[str, int] = {}

    for key in sorted(header.keys()):
        meta = header[key]
        shape = list(meta["shape"])
        dtype = str(meta["dtype"])
        dtypes_seen[dtype] = dtypes_seen.get(dtype, 0) + 1

        numel = int(math.prod(shape)) if shape else 1
        total_params += numel
        ndim = len(shape)

        mod_name = classify_module(key)
        role_name = classify_structural_role(key, shape)
        a2_name = classify_a2_group(mod_name, role_name)

        # 1. A.2 mutually-exclusive group partition (weights by prefix + embeddings_pos_ln_bias_scalar)
        g_bucket = a2_groups[a2_name]
        g_bucket["tensors"] += 1
        g_bucket["params"] += numel
        add_dim(g_bucket["dims"], ndim)
        g_bucket["roles"][role_name] = g_bucket["roles"].get(role_name, 0) + 1

        # 2. Architectural module prefix partition (all 1,590 by prefix)
        m_bucket = modules[mod_name]
        m_bucket["tensors"] += 1
        m_bucket["params"] += numel
        add_dim(m_bucket["dims"], ndim)
        m_bucket["roles"][role_name] = m_bucket["roles"].get(role_name, 0) + 1

        # 3. Structural role partition (all 1,590 by role)
        r_bucket = roles[role_name]
        r_bucket["tensors"] += 1
        r_bucket["params"] += numel
        add_dim(r_bucket["dims"], ndim)

    logical_tensors = len(header)
    a2_classified = sum(b["tensors"] for k, b in a2_groups.items() if k != "unclassified")
    a2_unclassified = a2_groups["unclassified"]["tensors"]
    mod_classified = sum(b["tensors"] for k, b in modules.items() if k != "unclassified")
    mod_unclassified = modules["unclassified"]["tensors"]
    role_classified = sum(b["tensors"] for k, b in roles.items() if k != "unclassified")
    role_unclassified = roles["unclassified"]["tensors"]

    unclassified_total = a2_unclassified + mod_unclassified + role_unclassified
    classified_total = a2_classified

    gate_pass = (
        sha_match
        and byte_match
        and (logical_tensors == EXPECTED_LOGICAL_TENSORS)
        and (a2_classified == EXPECTED_LOGICAL_TENSORS)
        and (mod_classified == EXPECTED_LOGICAL_TENSORS)
        and (role_classified == EXPECTED_LOGICAL_TENSORS)
        and (unclassified_total == 0)
        and (total_params == EXPECTED_TOTAL_PARAMETERS)
        and (dtypes_seen == {"F16": EXPECTED_LOGICAL_TENSORS})
    )

    elapsed_sec = round(time.perf_counter() - t0, 3)

    out_data = {
        "file": os.path.basename(model_path),
        "status": "PASS" if gate_pass else "FAIL",
        "measured_sha256": actual_sha256,
        "measured_bytes": actual_bytes,
        "expected_sha256": EXPECTED_SHA256,
        "expected_bytes": EXPECTED_BYTES,
        "sha256_match": sha_match,
        "bytes_match": byte_match,
        "dtypes": dtypes_seen,
        "total_logical_tensors": logical_tensors,
        "total_parameters": total_params,
        "classified_total": classified_total,
        "unclassified_total": unclassified_total,
        "gate_s0_pass": gate_pass,
        "elapsed_seconds": elapsed_sec,
        "dot_prod_scoring": {
            "a2_group_tensors": a2_groups["detector.dot_prod_scoring"]["tensors"],
            "module_prefix_tensors": modules["detector.dot_prod_scoring"]["tensors"],
            "params": a2_groups["detector.dot_prod_scoring"]["params"],
            "module_prefix_params": modules["detector.dot_prod_scoring"]["params"],
        },
        "environment": {
            "device": "cpu (safetensors header parse only; no tensor materialisation)",
            "gpu_peak_memory_mib": 0,
        },
        "group_summary": {
            name: a2_groups[name]
            for name in A2_GROUP_NAMES
            if a2_groups[name]["tensors"] > 0
        },
        "module_summary": {
            name: modules[name]
            for name in MODULE_NAMES
            if modules[name]["tensors"] > 0
        },
        "structural_roles_summary": {
            name: roles[name]
            for name in ROLE_NAMES
            if roles[name]["tensors"] > 0
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(out_data, fp, indent=2)

    print(f"File: {os.path.basename(model_path)}")
    print(f"Bytes: {actual_bytes:,} (match: {byte_match})")
    print(f"SHA256: {actual_sha256} (match: {sha_match})")
    print(f"Total logical tensors: {logical_tensors} (expected {EXPECTED_LOGICAL_TENSORS})")
    print(f"Total parameters: {total_params:,} (expected {EXPECTED_TOTAL_PARAMETERS:,})")
    print(f"Classified (A.2 / Module / Role): {a2_classified} / {mod_classified} / {role_classified}, Unclassified: {unclassified_total}")
    print(f"Elapsed: {elapsed_sec:.3f}s")
    print(f"Gate S0: {'PASS' if gate_pass else 'FAIL'}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())

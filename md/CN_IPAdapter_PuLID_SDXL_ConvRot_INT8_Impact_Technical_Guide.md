# PuLID IP-Adapter (SDXL) — ConvRot INT8 Impact Technical Guide

> The method is stated once in [`How to quantize Text Encoder and ControlNet.md`](How%20to%20quantize%20Text%20Encoder%20and%20ControlNet.md). This guide carries the measured numbers for this model.

**Pack.** `ip-adapter_pulid_sdxl_hswq_convrot_int8.safetensors` (396,740,327 B)

**SHA256.** `0262c9affd20d105421d83f95e959ca89ed8acde7df104d7b94b13b840f5440a`

| number | measured |
| :--- | :--- |
| None | INT8 layers declared |
| 173 | INT8 layers found |
| 0 | integers differing from the replay |
| 77 | float tensors checked |
| FAITHFUL | verdict |

## Protection

| number | measured |
| :--- | :--- |
| None | modules measured |
| 1.2560586583276745e-05 | largest drift |
| 0 | layers above 0.15 |
| 173 | ConvRot INT8 Linear layers |

## Route

| number | measured |
| :--- | :--- |
| None | INT8 GEMM calls |
| None | groupsizes |

## Records

- `records/pulid_sdxl_pack_audit.json`
- `records/pulid_sdxl_protection.json`
- `records/pulid_sdxl_gate_256.json`
- `records/pulid_sdxl_route.json`

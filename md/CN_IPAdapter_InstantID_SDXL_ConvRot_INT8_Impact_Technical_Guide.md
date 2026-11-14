# InstantID IP-Adapter (SDXL) — ConvRot INT8 Impact Technical Guide

> The method is stated once in [`How to quantize Text Encoder and ControlNet.md`](How%20to%20quantize%20Text%20Encoder%20and%20ControlNet.md). This guide carries the measured numbers for this model.

**Pack.** `ip-adapter_instant_id_sdxl_hswq_convrot_int8.safetensors` (1,420,211,083 B)

**SHA256.** `a9ffbd849d007b969d91cadd1409757000f20f69d08b45d95f18243adff456c9`

| number | measured |
| :--- | :--- |
| 329 | INT8 layers declared |
| 329 | INT8 layers found |
| 0 | integers differing from the replay |
| 515 | float tensors checked |
| FAITHFUL | verdict |

## Protection

| number | measured |
| :--- | :--- |
| None | modules measured |
| None | largest drift |
| 12 | layers above 0.15 |
| 329 | ConvRot INT8 Linear layers |

## Acceptance gate

| number | measured |
| :--- | :--- |
| 0.9993946489412338 | mean cosine (256 items) |
| 0.997490644454956 | minimum cosine |

## Route

| number | measured |
| :--- | :--- |
| None | INT8 GEMM calls |
| None | groupsizes |

## Records

- `records/cn_ipadapter_instantid_sdxl_pack_audit.json`
- `records/cn_ipadapter_instantid_sdxl_protection.json`
- `records/cn_ipadapter_instantid_sdxl_gate_256.json`
- `records/cn_ipadapter_instantid_sdxl_route.json`

# SD1.5 IP-Adapter Plus — ConvRot INT8 Impact Technical Guide

> The method is stated once in [`How to quantize Text Encoder and ControlNet.md`](How%20to%20quantize%20Text%20Encoder%20and%20ControlNet.md). This guide carries the measured numbers for this model.

**Pack.** `ip-adapter_sd15_plus_hswq_convrot_int8.safetensors` (49,366,186 B)

**SHA256.** `9a0f7cdfae2575951c3324458fd67221b6240e7c5c8ffd27476bb009aa7f6cd9`

| number | measured |
| :--- | :--- |
| 54 | INT8 layers declared |
| 54 | INT8 layers found |
| 0 | integers differing from the replay |
| 29 | float tensors checked |
| FAITHFUL | verdict |

## Protection

| number | measured |
| :--- | :--- |
| 54 | modules measured |
| 8.157313206424988e-05 | largest drift |
| 0 | layers above 0.15 |
| 54 | ConvRot INT8 Linear layers |

## Acceptance gate

| number | measured |
| :--- | :--- |
| 0.9998862713109702 | mean cosine (256 items) |
| 0.9989429116249084 | minimum cosine |

## Route

| number | measured |
| :--- | :--- |
| None | INT8 GEMM calls |
| None | groupsizes |

## Records

- `records/cn_ipadapter_sd15_plus_pack_audit.json`
- `records/cn_ipadapter_sd15_plus_protection.json`
- `records/cn_ipadapter_sd15_plus_gate_256.json`
- `records/cn_ipadapter_sd15_plus_route.json`

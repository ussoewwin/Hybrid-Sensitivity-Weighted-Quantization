# SD1.5 IP-Adapter FaceID — ConvRot INT8 Impact Technical Guide

> The method is stated once in [`How to quantize Text Encoder and ControlNet.md`](How%20to%20quantize%20Text%20Encoder%20and%20ControlNet.md). This guide carries the measured numbers for this model.

**Pack.** `ip-adapter-faceid_sd15_hswq_convrot_int8.safetensors` (65,047,130 B)

**SHA256.** `35b5942ee163494b82ec4b9d0797ef511a2552e780d44f9e15a6bae9ee40fcc6`

| number | measured |
| :--- | :--- |
| 102 | INT8 layers declared |
| 102 | INT8 layers found |
| 0 | integers differing from the replay |
| 192 | float tensors checked |
| FAITHFUL | verdict |

## Protection

| number | measured |
| :--- | :--- |
| 102 | modules measured |
| 8.118178084259853e-05 | largest drift |
| 0 | layers above 0.15 |
| 102 | ConvRot INT8 Linear layers |

## Acceptance gate

| number | measured |
| :--- | :--- |
| 0.9999888266902417 | mean cosine (256 items) |
| 0.9999672174453735 | minimum cosine |

## Route

| number | measured |
| :--- | :--- |
| 66 | INT8 GEMM calls |
| [256] | groupsizes |

## Records

- `records/cn_ipadapter_faceid_sd15_pack_audit.json`
- `records/cn_ipadapter_faceid_sd15_protection.json`
- `records/cn_ipadapter_faceid_sd15_gate_256.json`
- `records/cn_ipadapter_faceid_sd15_route.json`

# InstantID (diffusers layout) ControlNet — ConvRot INT8 Trajectory-Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate contribution, the
> size/quality balance, FP16 kept tensors) is stated once in
> [`How to quantize Text Encoder and ControlNet.md`](How%20to%20quantize%20Text%20Encoder%20and%20ControlNet.md).
> This guide carries the measured numbers for this model.

**Model.** `models/ControlNet/instantid_diffusion_pytorch_model.safetensors` (2,502,139,136 B, F16, 1,251,014,160 params)
**Pack.** `models/ControlNet/instantid_diffusion_pytorch_model_hswq_convrot_int8.safetensors` (1354.4 MB, **-43.2 %**)
**SHA256.** `A9FFBD849D007B969D91CADD1409757000F20F69D08B45D95F18243ADFF456C9`
**Method.** the FP16-kept set is chosen by the **trajectory impact**, measured through the real pack path.

---

## 1. The model

| item | measured |
| :--- | :--- |
| the source layout | the diffusers InstantID ControlNet (`controlnet_down_blocks.*`, `controlnet_cond_embedding.*`, `add_embedding.*`) |
| the runtime layout | comfy-native (`input_blocks.*`, `zero_convs.*`, `input_hint_block.*`) |
| the runtime class | `ControlNet` (the cldm family) |
| the runtime's Linear modules | **362** |
| the runtime's `state_dict()` tensors | 844 |
| hidden size | 1280 |
| the source file's own timestamps | created 2026-04-09 13:47:06, written 2026-04-09 13:50:43 |

The source identity is pinned by its sha256, `c8127be9f174101ebdafee9964d856b49b634435cf6daa396d3f593cf0bbbb05`, which is the value the audit record carries.

---

## 2. The measurement

| item | measured |
| :--- | :--- |
| the sweep | **400 modules**, 20 steps, seed 42, cfg 7.0 |
| the host | `waiIllustriousSDXL_v170.safetensors` (SDXL, FP16) |
| the instrument | `clip_convert/diag_impact_controlnet.py` |

**Why this checkpoint uses the ControlNet instrument, not a per-model variant.**
This file is a plain ControlNet. Its state dict carries 1,502 tensors, all under
`input_blocks.*`, `middle_block.*`, `zero_convs.*`, `input_hint_block.*`,
`time_embed.*` and `label_emb.*`, with **zero `image_proj` tensors** (measured).
`ip-adapter_instant_id_sdxl`, `control_instant_id_sdxl` and
`instantid_diffusion_pytorch_model` are three names for the same 1,420,211,083-byte
file. The runtime object is therefore the same `ControlNet` class, and the
observable is that object's own output along the denoising trajectory, so
`clip_convert/diag_impact_controlnet.py` measures it as-is: this checkpoint has no
second part whose outputs are frozen at apply time (the failure mode the SD1.5
IP-Adapter Resampler has), which is the only reason that family needed its own
measurement loop. The sweep covered all 400 modules rather than a narrowed
candidate set, and recorded no unmeasured (zero) values.
| the impact file | `impact/impact_instantid_diffusers_full.json`, sha256 `8435b913d6120474c8b030e53ebb4be887c8b04f33f32ef6873623285a2b860b` |
| the largest drift | **1.057680** (`input_blocks.1.0.in_layers.2`) |

| rank | the layer | the measured drift |
| :--- | :--- | ---: |
| 1 | `input_blocks.1.0.in_layers.2` | 1.057680 |
| 2 | `input_blocks.1.0.out_layers.3` | 0.284269 |
| 3 | `input_blocks.4.0.out_layers.3` | 0.271565 |
| 4 | `input_blocks.6.0.op` | 0.244178 |
| 5 | `input_blocks.5.0.out_layers.3` | 0.227639 |
| 6 | `input_blocks.0.0` | 0.212908 |
| 7 | `zero_convs.7.0` | 0.177161 |
| 8 | `input_blocks.4.0.in_layers.2` | 0.171345 |
| 9 | `input_blocks.3.0.op` | 0.164278 |
| 10 | `zero_convs.5.0` | 0.157812 |
| 11 | `input_blocks.4.0.skip_connection` | 0.154769 |
| 12 | `input_blocks.2.0.out_layers.3` | 0.154708 |

---

## 3. Choosing the FP16-kept set

| item | measured |
| :--- | :--- |
| the drift threshold | **0.15** |
| the layers above it | **12** |
| the FP16-kept layers in total | **33** (12 by drift, 21 by shape) |
| the ConvRot INT8 Linear layers | **329** |
| the runtime's Linear modules | 362 |

---

## 4. The pack

| item | measured |
| :--- | :--- |
| the group size | 256 |
| the size | 1,420,211,083 B (1354.4 MB) against 2,502,139,136 B, **-43.2 %** |
| the arithmetic | 0 differing INT8 values, largest relative scale difference 0.0, float side bit-identical (515/515) |
| the runtime coverage | 844 of 844 tensors, 0 missing, 0 extra, 0 shape differences |

---

## 5. Verification

| item | value |
| :--- | :--- |
| the loader route | 329 of 329 declared layers applied, 0 declared-not-applied |
| one forward | 329 INT8 GEMM calls, 10 output tensors, finite |
| the render | 1024x1024, 12 steps, cfg 6.0, 3,948 INT8 GEMM calls, 1,903,787 B PNG, 24.5 s |
| the gate (256 items, mean >= 0.99, min >= 0.98) | **PASS** - mean 0.999395, min 0.997491, p5 0.998739, p50 0.999473 |
| the source vs pack output | 4 of 4 batches comparable, cos_min 0.998635, rel L2 0.052224 |
| the control-image trigger search | 55 candidates, 0 above the control p99, detection floor 0.1 |

The backdoor audit of this pack, with every record, is in
[`md/backdoor/verdicts/CN_InstantID_Diffusers_Pack.md`](backdoor/verdicts/CN_InstantID_Diffusers_Pack.md).

---

## 6. Files

| file | what it is |
| :--- | :--- |
| `clip_convert/diag_impact_controlnet.py` | the trajectory sweep (400 modules) |
| `impact/impact_instantid_diffusers_full.json` | the measured drift per module |
| `clip_convert/convert_controlnet_convrot_int8.py` | the converter (runtime names, group 256, the drift threshold) |
| `clip_convert/verify_cn_flux_256.py` | the 256-item gate |
| `audit/controlnet_image_test.py` | the end-to-end render |
| `audit/controlnet_pack_source_output_diff.py` | the source-vs-pack output comparison |
| `audit/controlnet_image_trigger.py` | the control-image trigger search |

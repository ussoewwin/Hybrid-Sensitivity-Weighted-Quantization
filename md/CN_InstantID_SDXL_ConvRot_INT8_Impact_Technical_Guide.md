# InstantID (SDXL) ControlNet — ConvRot INT8 Trajectory-Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this model.

**Model.** `models/ControlNet/control_instant_id_sdxl.safetensors` (2502.1 MB, FP32, 1,251,014,160 params, published by `InstantX/InstantID`)
**Pack.** `models/ControlNet/control_instant_id_sdxl_hswq_convrot_int8.safetensors` (1420.2 MB, **-43.2 %**)
**SHA256.** `A9FFBD849D007B969D91CADD1409757000F20F69D08B45D95F18243ADFF456C9`
**Method.** the FP16-kept set is chosen by the **trajectory impact**, measured through the real pack path.

---

## 1. The model

| item | measured |
| :--- | :--- |
| the source layout | the diffusers `ControlNetModel` (`down_blocks.*`, `mid_block.*`, `controlnet_down_blocks.*`, `controlnet_cond_embedding.*`, `add_embedding.*`) |
| the runtime layout | comfy-native (`input_blocks.*`, `zero_convs.*`, `input_hint_block.*`), built by `comfy.controlnet.load_controlnet_state_dict` |
| the runtime class | `ControlNet` (the cldm family, not `ControlNetFlux`) |
| the runtime's Linear modules | **362** |
| the runtime's `state_dict()` tensors | **844** |
| the source's own tensors | 844, of which 329 take the INT8 path |
| context / hint | context 2048 (SDXL), label embedding 2816, hint 3 channels |

The runtime path is the loader package's `HSWQ ControlNet Loader (ConvRot INT8)`
node, which builds the `ControlNet` from the runtime names through the same ops
class the pack runs with.

---

## 2. The measurement

The protection set is measured on **this** checkpoint, by a full trajectory
sweep - every eligible module, not a focused screen:

| item | measured |
| :--- | :--- |
| the sweep | **400 modules**, 20 steps, seed 42, cfg 7.0 |
| the host | `waiIllustriousSDXL_v170.safetensors` (SDXL, FP16) |
| the instrument | `clip_convert/diag_impact_controlnet.py` (the runtime path, the drift threshold applied per layer) |

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
| the impact file | `impact/impact_sdxl_instantid_full.json`, sha256 `088b6b3554456f6c219af03a0a6920d2395092225190a6b8bf33a3b854dc15df` |
| the largest drift | **1.057680** (`input_blocks.1.0.in_layers.2`) |

This checkpoint's drift ranking is concentrated in the early input blocks:

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
| the layers above it | **12** (the table above) |
| the FP16-kept layers in total | **33** (12 by drift, 21 by shape) |
| the ConvRot INT8 Linear layers | **329** |
| the runtime's non-Linear tensors kept | 482 |

The converter reads the impact file and keeps every layer whose measured drift
exceeds the threshold in FP16; the layers whose shape does not fit the group size
are kept as well. The protection set is this checkpoint's own - it is not
carried over from another model's sweep.

---

## 4. The pack

| item | measured |
| :--- | :--- |
| the key naming | the runtime's own names (`input_blocks.*`, `zero_convs.*`, `input_hint_block.*`) |
| the group size | 256 (the ConvRot kernel's fused path) |
| the size | 1,420,211,083 B (1420.2 MB) against 2,502,139,136 B (2502.1 MB), **-43.2 %** |
| the arithmetic | 0 differing INT8 values, largest relative scale difference 0.0, float side bit-identical (515 of 515) |
| the runtime coverage | 844 of 844 tensors, 0 missing, 0 extra, 0 shape differences |

---

## 5. Verification

| item | value |
| :--- | :--- |
| the load | the loader's own node, the package's arm active |
| the quantized layers at runtime | 329 (all `QuantizedTensor`, group size 256) |
| the `int8_linear` calls | **329 / 329** in one forward, **3948** across the render |
| the output | 10 tensors, **finite** |
| the gate (256 items, mean >= 0.99, min >= 0.98) | **PASS** - mean 0.999395, min 0.997491, p5 0.998739, p50 0.999473 |
| the render | 1024x1024, 12 steps, cfg 6.0, 3948 `int8_linear` calls, 1,903,787 B PNG, 26.6 s |
| the source vs pack output | 4 of 4 input batches comparable, cos_min 0.998635, rel_l2_max 0.052224 |
| the control-image trigger search | 55 candidates, 0 above the control p99, detection floor 0.1 (measured in that record) |

The gate is measured by `clip_convert/verify_cn_flux_256.py` on the fixed
256-item set, with the FP16 baseline built through the same ops class the pack
runs with. The backdoor audit of this pack, with every record, is in
[`md/backdoor/verdicts/CN_InstantID_SDXL_Pack.md`](backdoor/verdicts/CN_InstantID_SDXL_Pack.md).

---

## 6. Files

| file | what it is |
| :--- | :--- |
| `clip_convert/diag_impact_controlnet.py` | the trajectory sweep (400 modules) |
| `impact/impact_sdxl_instantid_full.json` | the measured drift per module |
| `clip_convert/convert_controlnet_convrot_int8.py` | the converter (runtime names, group 256, the drift threshold) |
| `clip_convert/verify_cn_flux_256.py` | the 256-item gate |
| `clip_convert/verify_controlnet_loader.py` | the loader acceptance (metadata consumed, INT8 GEMM count, finiteness) |
| `audit/controlnet_image_test.py` | the end-to-end render |
| `audit/controlnet_pack_source_output_diff.py` | the source-vs-pack output comparison |
| `audit/controlnet_image_trigger.py` | the control-image trigger search |

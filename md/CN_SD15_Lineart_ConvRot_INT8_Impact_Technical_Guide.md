# SD1.5 lineart ControlNet — ConvRot INT8 Trajectory-Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this model.

**Model.** `models/ControlNet/control_v11p_sd15_lineart.pth` (1378.3 MB, FP32, 361,279,120 params, published by `lllyasviel/ControlNet-v1-1`)
**Pack.** `models/ControlNet/control_v11p_sd15_lineart_hswq_convrot_int8.safetensors` (579.5 MB, **-57.9 %**)
**SHA256.** `F32407808652F74E52F8F14767173751AF9D9C5D202A48BB3354FF8BC475567F`
**Method.** the FP16-kept set is chosen by the **trajectory impact**, measured through the real pack path.

---

## 1. The model

| item | measured |
| :--- | :--- |
| the source layout | the original ControlNet `.pth` (`control_model.*` prefix) |
| the runtime layout | comfy-native (`input_blocks.*`, `zero_convs.*`, `input_hint_block.*`) |
| the runtime class | `ControlNet` (the cldm family, not `ControlNetFlux`) |
| the runtime's Linear modules | **82** |
| the runtime's `state_dict()` tensors | 340 |
| hidden size | 320 |
| context / hint | context 768 (the CLIP-L width), hint 3 channels, no label embedding (`num_classes` is None) |

The source is the publisher's own release: the Hub LFS object id of
`control_v11p_sd15_lineart.pth` in `lllyasviel/ControlNet-v1-1` is
`fb5fb912696830da948e5056b50adfa3fcc68ff760c43e4e7a8499853af43c61`, the sha256 of
the audited file.

The runtime path is `comfy.controlnet.load_controlnet_state_dict` for the source, and
the loader package's `HSWQ ControlNet Loader (ConvRot INT8)` node builds the
`ControlNet` from the runtime names.

**`comfy.controlnet` has no quantization-metadata handling of its own.** It
picks an ops class (`custom_operations`) and hands it to the model's
constructor; the metadata (`<layer>.comfy_quant` / `<layer>.weight_scale`) is
consumed by `comfy.ops.MixedPrecisionOps.Linear._load_from_state_dict`. The
loader package supplies that ops class, so the whole path is the loader's.

---

## 2. The measurement

The protection set is measured on **this** checkpoint, by a full trajectory
sweep - every eligible module, not a focused screen:

| item | measured |
| :--- | :--- |
| the sweep | **142 modules**, 20 steps, seed 42, cfg 7.0 |
| the host | `chilledReGenericV3_v10.safetensors` (SD1.5, FP16) |
| the instrument | `clip_convert/diag_impact_controlnet.py` (the runtime path, the drift threshold applied per layer) |
| the impact file | `impact/impact_sd15_lineart_full.json`, sha256 `347d8a661f9916ae574a646f65a3f9a6b51cf066dc6dfdb0521d856edcfd63e6` |
| the largest drift | **0.842759** (`input_blocks.4.0.out_layers.3`) |

This checkpoint's drift reaches further into the network than the other SD1.5
ControlNets measured here - the deepest residual and transition layers are above
the threshold as well:

| rank | the layer | the measured drift |
| :--- | :--- | ---: |
| 1 | `input_blocks.4.0.out_layers.3` | 0.842759 |
| 2 | `input_blocks.5.1.proj_out` | 0.776164 |
| 3 | `input_blocks.1.0.out_layers.3` | 0.731392 |
| 4 | `input_hint_block.14` | 0.713775 |
| 5 | `input_blocks.4.1.proj_in` | 0.700776 |
| 6 | `input_blocks.2.1.proj_out` | 0.643799 |
| 7 | `input_blocks.2.1.proj_in` | 0.640828 |
| 8 | `input_blocks.4.1.proj_out` | 0.637826 |
| 9 | `input_blocks.4.0.in_layers.2` | 0.635421 |
| 10 | `input_blocks.3.0.op` | 0.627402 |
| 11 | `input_blocks.1.1.proj_out` | 0.621403 |
| 12 | `input_blocks.7.0.out_layers.3` | 0.580857 |
| 13 | `input_blocks.5.0.out_layers.3` | 0.566324 |
| 14 | `input_blocks.5.0.in_layers.2` | 0.534778 |
| 15 | `input_hint_block.10` | 0.531034 |
| 16 | `input_hint_block.6` | 0.516704 |
| 17 | `input_blocks.4.0.skip_connection` | 0.508234 |
| 18 | `input_blocks.2.0.in_layers.2` | 0.507829 |
| 19 | `input_blocks.2.0.out_layers.3` | 0.420243 |
| 20 | `input_blocks.7.0.skip_connection` | 0.417734 |
| 21 | `input_blocks.7.0.in_layers.2` | 0.379454 |
| 22 | `input_blocks.6.0.op` | 0.297152 |
| 23 | `input_blocks.1.1.proj_in` | 0.201115 |
| 24 | `input_blocks.1.0.in_layers.2` | 0.188905 |
| 25 | `input_blocks.8.1.proj_out` | 0.182393 |
| 26 | `input_blocks.7.1.proj_in` | 0.172082 |
| 27 | `input_hint_block.12` | 0.167484 |
| 28 | `input_blocks.7.1.proj_out` | 0.166920 |
| 29 | `input_blocks.8.0.in_layers.2` | 0.150943 |

---

## 3. Choosing the FP16-kept set

| item | measured |
| :--- | :--- |
| the drift threshold | **0.15** |
| the layers above it | **29** (the table above) |
| the FP16-kept layers in total | **29** (29 by drift; the shape-only rule adds none beyond this set) |
| the ConvRot INT8 Linear layers | **53** |
| the runtime's non-Linear tensors kept | 258 |

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
| the size | 607,603,903 B (579.5 MB) against 1,445,235,365 B (1378.3 MB), **-57.9 %** |
| the arithmetic | 0 differing INT8 values, largest relative scale difference 0.0, float side bit-identical (287 of 287) |
| the runtime coverage | 340 of 340 tensors, 0 missing, 0 extra, 0 shape differences |

---

## 5. Verification

| item | value |
| :--- | :--- |
| the load | the loader's own node, the package's arm active |
| the quantized layers at runtime | 53 (all `QuantizedTensor`, group size 256) |
| the `int8_linear` calls | **53 / 53** |
| the output | 13 tensors, **finite** |
| the gate (256 items, mean >= 0.99, min >= 0.98) | **PASS** - mean 0.999737, min 0.999698, p5 0.999714, p50 0.999736 |
| the render | 1024x1024, 12 steps, cfg 6.0, 636 `int8_linear` calls, 1,052,781 B PNG, 23.3 s |
| the source vs pack output | 4 of 4 input batches comparable, cos_min 0.999492, rel_l2_max 0.031337 |
| the control-image trigger search | 55 candidates, 0 above the control p99, detection floor 0.03 (measured in that record) |

The gate is measured by `clip_convert/verify_cn_flux_256.py` on the fixed
256-item set, with the FP16 baseline built through the same ops class the pack
runs with. The backdoor audit of this pack, with every record, is in
[`md/backdoor/verdicts/CN_SD15_Lineart_Pack.md`](backdoor/verdicts/CN_SD15_Lineart_Pack.md).

---

## 6. Files

| file | what it is |
| :--- | :--- |
| `clip_convert/diag_impact_controlnet.py` | the trajectory sweep (142 modules) |
| `impact/impact_sd15_lineart_full.json` | the measured drift per module |
| `clip_convert/convert_controlnet_convrot_int8.py` | the converter (runtime names, group 256, the drift threshold) |
| `clip_convert/verify_cn_flux_256.py` | the 256-item gate |
| `clip_convert/verify_controlnet_loader.py` | the loader acceptance (metadata consumed, INT8 GEMM count, finiteness) |
| `audit/controlnet_image_test.py` | the end-to-end render |
| `audit/controlnet_pack_source_output_diff.py` | the source-vs-pack output comparison |
| `audit/controlnet_image_trigger.py` | the control-image trigger search |

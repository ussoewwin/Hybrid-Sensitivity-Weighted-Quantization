# SD1.5 mlsd ControlNet — ConvRot INT8 Trajectory-Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`How to quantize Text Encoder and ControlNet.md`](How%20to%20quantize%20Text%20Encoder%20and%20ControlNet.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this model.

**Model.** `models/ControlNet/control_v11p_sd15_mlsd.pth` (1378.3 MB, FP32, 361,279,120 params, published by `lllyasviel/ControlNet-v1-1`)
**Pack.** `models/ControlNet/control_v11p_sd15_mlsd_hswq_convrot_int8.safetensors` (579.5 MB, **-57.9 %**)
**SHA256.** `1DE1800AE02084C12B23A0F26B59507225874B0E0E2361854021CAB552D2ADDC`
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
`control_v11p_sd15_mlsd.pth` in `lllyasviel/ControlNet-v1-1` is
`22dab177f4c4566ac1c8f68dc2c429b023114e15a644e8c7c0e8f956afe92acf`, the sha256 of
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
| the impact file | `impact/impact_sd15_mlsd_full.json`, sha256 `233db3150448737deebd49a7c035376ba1454f5bbfa896cb1ff7771b7e824c61` |
| the largest drift | **0.516391** (`input_hint_block.4`) |

| rank | the layer | the measured drift |
| :--- | :--- | ---: |
| 1 | `input_hint_block.4` | 0.516391 |
| 2 | `input_blocks.4.0.in_layers.2` | 0.354269 |
| 3 | `input_hint_block.14` | 0.321510 |
| 4 | `input_hint_block.2` | 0.315978 |
| 5 | `input_blocks.4.0.out_layers.3` | 0.305799 |
| 6 | `input_blocks.1.0.out_layers.3` | 0.289717 |
| 7 | `input_blocks.1.0.in_layers.2` | 0.268446 |
| 8 | `input_blocks.2.0.out_layers.3` | 0.258043 |
| 9 | `input_blocks.3.0.op` | 0.245243 |
| 10 | `input_blocks.2.1.proj_in` | 0.235091 |
| 11 | `input_blocks.5.0.out_layers.3` | 0.232080 |
| 12 | `input_blocks.4.0.skip_connection` | 0.229749 |
| 13 | `input_blocks.6.0.op` | 0.215900 |
| 14 | `input_blocks.7.0.out_layers.3` | 0.206223 |
| 15 | `input_blocks.5.1.proj_in` | 0.195822 |
| 16 | `input_blocks.2.1.proj_out` | 0.178401 |
| 17 | `input_blocks.7.0.skip_connection` | 0.170091 |
| 18 | `input_blocks.8.1.proj_in` | 0.162812 |
| 19 | `input_hint_block.10` | 0.162022 |
| 20 | `input_hint_block.8` | 0.159372 |
| 21 | `input_blocks.2.0.in_layers.2` | 0.155962 |
| 22 | `input_blocks.7.1.proj_in` | 0.152374 |

---

## 3. Choosing the FP16-kept set

| item | measured |
| :--- | :--- |
| the drift threshold | **0.15** |
| the layers above it | **22** (the table above) |
| the FP16-kept layers in total | **29** (22 by drift, 7 by shape) |
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
| the size | 607,603,903 B (579.5 MB) against 1,445,234,339 B (1378.3 MB), **-57.9 %** |
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
| the gate (256 items, mean >= 0.99, min >= 0.98) | **PASS** - mean 0.999678, min 0.999466, p5 0.999583, p50 0.999685 |
| the render | 1024x1024, 12 steps, cfg 6.0, 636 `int8_linear` calls, 1,051,890 B PNG, 11.9 s |
| the source vs pack output | 4 of 4 input batches comparable, cos_min 0.999576, rel_l2_max 0.028170 |
| the control-image trigger search | 55 candidates, 0 above the control p99, detection floor 0.001 (measured in that record) |

The gate is measured by `clip_convert/verify_cn_flux_256.py` on the fixed
256-item set, with the FP16 baseline built through the same ops class the pack
runs with. The backdoor audit of this pack, with every record, is in
[`md/backdoor/verdicts/CN_SD15_Mlsd_Pack.md`](backdoor/verdicts/CN_SD15_Mlsd_Pack.md).

---

## 6. Files

| file | what it is |
| :--- | :--- |
| `clip_convert/diag_impact_controlnet.py` | the trajectory sweep (142 modules) |
| `impact/impact_sd15_mlsd_full.json` | the measured drift per module |
| `clip_convert/convert_controlnet_convrot_int8.py` | the converter (runtime names, group 256, the drift threshold) |
| `clip_convert/verify_cn_flux_256.py` | the 256-item gate |
| `clip_convert/verify_controlnet_loader.py` | the loader acceptance (metadata consumed, INT8 GEMM count, finiteness) |
| `audit/controlnet_image_test.py` | the end-to-end render |
| `audit/controlnet_pack_source_output_diff.py` | the source-vs-pack output comparison |
| `audit/controlnet_image_trigger.py` | the control-image trigger search |

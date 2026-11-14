# SD1.5 softedge ControlNet — ConvRot INT8 Trajectory-Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this model.

**Model.** `models/ControlNet/control_v11p_sd15_softedge.pth` (1378.3 MB, FP32, 361,279,120 params, published by `lllyasviel/ControlNet-v1-1`)
**Pack.** `models/ControlNet/control_v11p_sd15_softedge_hswq_convrot_int8.safetensors` (579.5 MB, **-57.9 %**)
**SHA256.** `609B04A9A38668FB084C447E9B0B6BC221466C09D8F83B610929BE53F8ACD395`
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
`control_v11p_sd15_softedge.pth` in `lllyasviel/ControlNet-v1-1` is
`20759ef71a109c4796e637d86eea8db492ec9772b6d5cb7a9afd077847b403f3`, the sha256 of
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
| the impact file | `impact/impact_sd15_softedge_full.json`, sha256 `9b53c755c83bf71982374326204c2a90c6c09e176b7ee90e786476c28fd724bf` |
| the largest drift | **46.853904** (`input_hint_block.2`) |

The ranking on this checkpoint is not smooth. The full-resolution hint
convolution `input_hint_block.2` carries a drift of **46.85**, two orders of
magnitude above every other module in the graph (the second is 0.687), so the
separation is a property of this checkpoint's hint branch, not of the whole
network. The remaining modules above the threshold form a continuous band
between 0.69 and 0.15.

| rank | the layer | the measured drift |
| :--- | :--- | ---: |
| 1 | `input_hint_block.2` | 46.853904 |
| 2 | `input_blocks.7.1.proj_out` | 0.686831 |
| 3 | `input_hint_block.6` | 0.646199 |
| 4 | `input_blocks.2.0.out_layers.3` | 0.627443 |
| 5 | `input_blocks.5.1.proj_in` | 0.605368 |
| 6 | `input_blocks.5.1.proj_out` | 0.593922 |
| 7 | `input_blocks.5.0.in_layers.2` | 0.564100 |
| 8 | `input_blocks.2.1.proj_in` | 0.491138 |
| 9 | `input_blocks.1.1.proj_out` | 0.483510 |
| 10 | `input_blocks.8.0.in_layers.2` | 0.462517 |
| 11 | `input_blocks.2.1.proj_out` | 0.456974 |
| 12 | `input_blocks.7.0.in_layers.2` | 0.438265 |
| 13 | `input_blocks.4.0.in_layers.2` | 0.436898 |
| 14 | `input_blocks.4.1.proj_in` | 0.409433 |
| 15 | `input_blocks.1.0.in_layers.2` | 0.343721 |
| 16 | `input_blocks.2.0.in_layers.2` | 0.338620 |
| 17 | `input_blocks.8.0.out_layers.3` | 0.328763 |
| 18 | `input_blocks.3.0.op` | 0.319601 |
| 19 | `input_blocks.8.1.proj_in` | 0.314429 |
| 20 | `input_blocks.5.0.out_layers.3` | 0.314041 |
| 21 | `input_blocks.4.0.out_layers.3` | 0.299074 |
| 22 | `input_hint_block.14` | 0.295502 |
| 23 | `input_blocks.7.1.proj_in` | 0.281761 |
| 24 | `input_blocks.4.0.skip_connection` | 0.278054 |
| 25 | `input_hint_block.4` | 0.268200 |
| 26 | `input_blocks.1.1.proj_in` | 0.267543 |
| 27 | `middle_block.1.proj_in` | 0.232819 |
| 28 | `input_blocks.4.1.proj_out` | 0.230763 |
| 29 | `input_blocks.9.0.op` | 0.229769 |
| 30 | `input_blocks.6.0.op` | 0.227143 |
| 31 | `zero_convs.8.0` | 0.225198 |
| 32 | `input_hint_block.8` | 0.220481 |
| 33 | `input_blocks.7.0.out_layers.3` | 0.208906 |
| 34 | `input_blocks.1.0.out_layers.3` | 0.178778 |
| 35 | `input_blocks.8.1.proj_out` | 0.173539 |
| 36 | `zero_convs.7.0` | 0.170142 |
| 37 | `zero_convs.0.0` | 0.165892 |
| 38 | `zero_convs.2.0` | 0.161141 |
| 39 | `input_blocks.7.0.skip_connection` | 0.159504 |
| 40 | `zero_convs.5.0` | 0.156104 |
| 41 | `zero_convs.1.0` | 0.155484 |
| 42 | `middle_block.0.in_layers.2` | 0.153794 |

---

## 3. Choosing the FP16-kept set

| item | measured |
| :--- | :--- |
| the drift threshold | **0.15** |
| the layers above it | **42** (the table above) |
| the FP16-kept layers in total | **29** (42 by drift, -13 by shape) |
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
| the size | 607,603,903 B (579.5 MB) against 1,445,235,707 B (1378.3 MB), **-57.9 %** |
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
| the gate (256 items, mean >= 0.99, min >= 0.98) | **PASS** - mean 0.999703, min 0.999572, p5 0.999674, p50 0.999705 |
| the render | 1024x1024, 12 steps, cfg 6.0, 636 `int8_linear` calls, 1,324,757 B PNG, 10.6 s |
| the source vs pack output | 4 of 4 input batches comparable, cos_min 0.999372, rel_l2_max 0.034824 |
| the control-image trigger search | 55 candidates, 0 above the control p99, detection floor 0.001 (measured in that record) |

The gate is measured by `clip_convert/verify_cn_flux_256.py` on the fixed
256-item set, with the FP16 baseline built through the same ops class the pack
runs with. The backdoor audit of this pack, with every record, is in
[`md/backdoor/verdicts/CN_SD15_Softedge_Pack.md`](backdoor/verdicts/CN_SD15_Softedge_Pack.md).

---

## 6. Files

| file | what it is |
| :--- | :--- |
| `clip_convert/diag_impact_controlnet.py` | the trajectory sweep (142 modules) |
| `impact/impact_sd15_softedge_full.json` | the measured drift per module |
| `clip_convert/convert_controlnet_convrot_int8.py` | the converter (runtime names, group 256, the drift threshold) |
| `clip_convert/verify_cn_flux_256.py` | the 256-item gate |
| `clip_convert/verify_controlnet_loader.py` | the loader acceptance (metadata consumed, INT8 GEMM count, finiteness) |
| `audit/controlnet_image_test.py` | the end-to-end render |
| `audit/controlnet_pack_source_output_diff.py` | the source-vs-pack output comparison |
| `audit/controlnet_image_trigger.py` | the control-image trigger search |

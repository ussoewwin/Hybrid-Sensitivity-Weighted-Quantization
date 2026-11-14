# SD1.5 ip2p ControlNet — ConvRot INT8 Trajectory-Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this model.

**Model.** `models/ControlNet/control_v11e_sd15_ip2p.pth` (1378.3 MB, FP32, 361,279,120 params, published by `lllyasviel/ControlNet-v1-1`)
**Pack.** `models/ControlNet/control_v11e_sd15_ip2p_hswq_convrot_int8.safetensors` (579.5 MB, **-57.9 %**)
**SHA256.** `43E0884402B1801F2FDFDF05E187A581090BBC0A679A187B7D7626133A0F94BD`
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
`control_v11e_sd15_ip2p.pth` in `lllyasviel/ControlNet-v1-1` is
`a8d77429b111b25ffad4adb4c686a19c73c905d638dd8acb1e0b1d1da300192c`, the sha256 of
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
| the impact file | `impact/impact_sd15_ip2p_full.json`, sha256 `44e29b73e8cfabf1d0946bb1d16b1279bcaf4e5e8922604a085e2de2b8cc0d18` |
| the largest drift | **0.624415** (`input_hint_block.10`) |

The families the drift singles out are the ones the SDXL ControlNet sweeps and the other
SD1.5 sweeps found as well - the `input_hint_block` family, the skip connections, the
residual path, the `proj_in` / `proj_out` transitions - and, on this checkpoint, the
zero convolutions of the hint taps:

| rank | the layer | the measured drift |
| :--- | :--- | ---: |
| 1 | `input_hint_block.10` | 0.624415 |
| 2 | `input_hint_block.4` | 0.613155 |
| 3 | `input_hint_block.2` | 0.557310 |
| 4 | `input_hint_block.8` | 0.556819 |
| 5 | `input_hint_block.12` | 0.526808 |
| 6 | `input_blocks.4.0.skip_connection` | 0.520825 |
| 7 | `input_blocks.1.1.proj_out` | 0.512993 |
| 8 | `input_blocks.2.0.out_layers.3` | 0.460693 |
| 9 | `input_hint_block.6` | 0.453440 |
| 10 | `input_blocks.6.0.op` | 0.434433 |
| 11 | `input_blocks.3.0.op` | 0.434361 |
| 12 | `input_blocks.2.0.in_layers.2` | 0.428287 |
| 13 | `input_blocks.5.1.proj_in` | 0.421263 |
| 14 | `input_blocks.1.0.in_layers.2` | 0.408343 |
| 15 | `input_blocks.4.1.proj_in` | 0.379127 |
| 16 | `input_blocks.1.1.proj_in` | 0.377055 |
| 17 | `input_blocks.7.1.proj_in` | 0.336133 |
| 18 | `input_blocks.5.1.proj_out` | 0.335137 |
| 19 | `input_hint_block.14` | 0.321466 |
| 20 | `input_blocks.4.0.in_layers.2` | 0.308322 |
| 21 | `input_blocks.7.1.proj_out` | 0.304098 |
| 22 | `input_blocks.1.0.out_layers.3` | 0.294188 |
| 23 | `input_blocks.2.1.proj_in` | 0.293820 |
| 24 | `zero_convs.8.0` | 0.291001 |
| 25 | `input_blocks.9.0.op` | 0.290586 |
| 26 | `input_blocks.7.0.skip_connection` | 0.288581 |
| 27 | `input_blocks.8.1.proj_out` | 0.286302 |
| 28 | `input_blocks.8.1.proj_in` | 0.278201 |
| 29 | `input_blocks.2.1.proj_out` | 0.261567 |
| 30 | `input_blocks.5.0.in_layers.2` | 0.260706 |
| 31 | `zero_convs.0.0` | 0.258423 |
| 32 | `zero_convs.7.0` | 0.254371 |
| 33 | `input_blocks.7.0.in_layers.2` | 0.249303 |
| 34 | `input_blocks.5.0.out_layers.3` | 0.245629 |
| 35 | `input_blocks.8.0.in_layers.2` | 0.239241 |
| 36 | `input_blocks.8.0.out_layers.3` | 0.231917 |
| 37 | `input_blocks.4.1.proj_out` | 0.225002 |
| 38 | `input_blocks.7.0.out_layers.3` | 0.213570 |
| 39 | `input_blocks.4.0.out_layers.3` | 0.203196 |
| 40 | `middle_block_out.0` | 0.198750 |
| 41 | `zero_convs.6.0` | 0.178654 |
| 42 | `zero_convs.1.0` | 0.164930 |

---

## 3. Choosing the FP16-kept set

| item | measured |
| :--- | :--- |
| the drift threshold | **0.15** |
| the layers above it | **42** (the table above) |
| the FP16-kept layers in total | **42** (all by drift; the shape-only rule adds none here) |
| the ConvRot INT8 Linear layers | **53** |
| the runtime's non-Linear tensors kept | 258 |

The converter reads the impact file and keeps every layer whose measured drift
exceeds the threshold in FP16; the layers whose shape does not fit the group size
are kept as well. The protection set is this checkpoint's own - it is not
carried over from another model's sweep. On this checkpoint the drift ranking
already covers every runtime Linear module, so the shape rule adds nothing.

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
| the gate (256 items, mean >= 0.99, min >= 0.98) | **PASS** - mean 0.999813, min 0.999768, p5 0.999786, p50 0.999815 |
| the render | 1024x1024, 12 steps, cfg 6.0, 636 `int8_linear` calls, 1,117,947 B PNG, 12.4 s |
| the source vs pack output | 4 of 4 input batches comparable, cos_min 0.999596, rel_l2_max 0.027849 |
| the control-image trigger search | 55 candidates, 0 above the control p99, detection floor 0.03 |

The gate is measured by `clip_convert/verify_cn_flux_256.py` on the fixed
256-item set, with the FP16 baseline built through the same ops class the pack
runs with. The backdoor audit of this pack, with every record, is in
[`md/backdoor/verdicts/CN_SD15_Ip2p_Pack.md`](backdoor/verdicts/CN_SD15_Ip2p_Pack.md).

---

## 6. Files

| file | what it is |
| :--- | :--- |
| `clip_convert/diag_impact_controlnet.py` | the trajectory sweep (142 modules) |
| `impact/impact_sd15_ip2p_full.json` | the measured drift per module |
| `clip_convert/convert_controlnet_convrot_int8.py` | the converter (runtime names, group 256, the drift threshold) |
| `clip_convert/verify_cn_flux_256.py` | the 256-item gate |
| `clip_convert/verify_controlnet_loader.py` | the loader acceptance (metadata consumed, INT8 GEMM count, finiteness) |
| `audit/controlnet_image_test.py` | the end-to-end render |
| `audit/controlnet_pack_source_output_diff.py` | the source-vs-pack output comparison |
| `audit/controlnet_image_trigger.py` | the control-image trigger search |

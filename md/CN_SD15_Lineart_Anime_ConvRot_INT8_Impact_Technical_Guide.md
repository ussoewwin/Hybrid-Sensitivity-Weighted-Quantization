# SD1.5s2 lineart_anime ControlNet — ConvRot INT8 Trajectory-Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this model.

**Model.** `models/ControlNet/control_v11p_sd15s2_lineart_anime.pth` (1378.3 MB, FP32, 361,279,120 params, published by `lllyasviel/ControlNet-v1-1`)
**Pack.** `models/ControlNet/control_v11p_sd15s2_lineart_anime_hswq_convrot_int8.safetensors` (579.5 MB, **-57.9 %**)
**SHA256.** `D004154F605EBBF13D9632F80A3AA3ADB3EC9EE5C530AD6C4DA92A1FC60B0F51`
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

The `s2` in the file name is the publisher's own suffix (the second release of the
SD1.5 line-art set); the graph is the same cldm ControlNet shape as the other SD1.5
ControlNets, and the runtime class built from it is `ControlNet`.

The source is the publisher's own release: the Hub LFS object id of
`control_v11p_sd15s2_lineart_anime.pth` in `lllyasviel/ControlNet-v1-1` is
`59169ab86e8ae15ad50208cc5832f1cc598b95391d4482ba742b344c8f7f6904`, the sha256 of
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
| the impact file | `impact/impact_sd15_lineart_anime_full.json`, sha256 `855a0ff451bc7e80549c3e14b60e091cb12658b8f16459cf238a07144a8a44b6` |
| the largest drift | **1.098555** (`input_blocks.1.0.out_layers.3`) |

This checkpoint has the highest drift of the SD1.5 ControlNets handled here, and the
top of the ranking is dense: four modules sit between 1.10 and 0.86, and the hint
network contributes three more above 0.65.

| rank | the layer | the measured drift |
| :--- | :--- | ---: |
| 1 | `input_blocks.1.0.out_layers.3` | 1.098555 |
| 2 | `input_hint_block.14` | 0.968702 |
| 3 | `input_blocks.1.1.proj_in` | 0.875972 |
| 4 | `input_blocks.2.1.proj_in` | 0.861358 |
| 5 | `input_blocks.2.0.out_layers.3` | 0.837258 |
| 6 | `input_hint_block.4` | 0.708518 |
| 7 | `input_hint_block.8` | 0.665599 |
| 8 | `input_blocks.1.0.in_layers.2` | 0.655972 |
| 9 | `input_hint_block.10` | 0.499329 |
| 10 | `input_hint_block.12` | 0.480142 |
| 11 | `zero_convs.0.0` | 0.478646 |
| 12 | `input_hint_block.6` | 0.464132 |
| 13 | `zero_convs.1.0` | 0.416530 |
| 14 | `input_blocks.2.1.proj_out` | 0.408039 |
| 15 | `input_blocks.3.0.op` | 0.368629 |
| 16 | `input_blocks.4.0.skip_connection` | 0.317190 |
| 17 | `input_blocks.0.0` | 0.301794 |
| 18 | `input_blocks.4.0.out_layers.3` | 0.295747 |
| 19 | `input_blocks.4.1.proj_in` | 0.272956 |
| 20 | `input_blocks.4.0.in_layers.2` | 0.249915 |
| 21 | `input_blocks.1.1.proj_out` | 0.236001 |
| 22 | `input_blocks.4.1.proj_out` | 0.224573 |
| 23 | `input_blocks.5.1.proj_out` | 0.190640 |
| 24 | `input_blocks.2.0.in_layers.2` | 0.180446 |
| 25 | `input_blocks.6.0.op` | 0.175224 |
| 26 | `input_blocks.5.1.proj_in` | 0.166402 |
| 27 | `input_blocks.5.0.in_layers.2` | 0.160182 |

---

## 3. Choosing the FP16-kept set

| item | measured |
| :--- | :--- |
| the drift threshold | **0.15** |
| the layers above it | **27** (the table above) |
| the FP16-kept layers in total | **29** (27 by drift, 2 by shape) |
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
| the size | 607,603,903 B (579.5 MB) against 1,445,238,101 B (1378.3 MB), **-57.9 %** |
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
| the gate (256 items, mean >= 0.99, min >= 0.98) | **PASS** - mean 0.999748, min 0.999707, p5 0.999724, p50 0.999750 |
| the render | 1024x1024, 12 steps, cfg 6.0, 636 `int8_linear` calls, 977,658 B PNG, 9.2 s |
| the source vs pack output | 4 of 4 input batches comparable, cos_min 0.999513, rel_l2_max 0.031238 |
| the control-image trigger search | 55 candidates, 0 above the control p99, detection floor 0.01 (measured in that record) |

The gate is measured by `clip_convert/verify_cn_flux_256.py` on the fixed
256-item set, with the FP16 baseline built through the same ops class the pack
runs with. The backdoor audit of this pack, with every record, is in
[`md/backdoor/verdicts/CN_SD15_Lineart_Anime_Pack.md`](backdoor/verdicts/CN_SD15_Lineart_Anime_Pack.md).

---

## 6. Files

| file | what it is |
| :--- | :--- |
| `clip_convert/diag_impact_controlnet.py` | the trajectory sweep (142 modules) |
| `impact/impact_sd15_lineart_anime_full.json` | the measured drift per module |
| `clip_convert/convert_controlnet_convrot_int8.py` | the converter (runtime names, group 256, the drift threshold) |
| `clip_convert/verify_cn_flux_256.py` | the 256-item gate |
| `clip_convert/verify_controlnet_loader.py` | the loader acceptance (metadata consumed, INT8 GEMM count, finiteness) |
| `audit/controlnet_image_test.py` | the end-to-end render |
| `audit/controlnet_pack_source_output_diff.py` | the source-vs-pack output comparison |
| `audit/controlnet_image_trigger.py` | the control-image trigger search |

# SD1.5 recolor ControlNet — ConvRot INT8 Trajectory-Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`How to quantize Text Encoder and ControlNet.md`](How%20to%20quantize%20Text%20Encoder%20and%20ControlNet.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this model.

**Model.** `models/ControlNet/ioclab_sd15_recolor.safetensors` (722,598,616 B)
**Pack.** `models/ControlNet/ioclab_sd15_recolor_hswq_convrot_int8.safetensors` (411,127,471 B, **-43.1 %**)
**SHA256.** `58046C116268A1C99D6D84FF4EC4631AD8E6E276F87BA91E791599BD0B9E5F5F`
**Method.** the FP16-kept set is chosen by the **trajectory impact**, measured through the real pack path.

---

## 1. The model

| item | measured |
| :--- | :--- |
| the source layout | comfy-native names already (`input_blocks.*`, `zero_convs.*`, `input_hint_block.*`) |
| the runtime class | `ControlNet` (the cldm family, not `ControlNetFlux`) |
| the runtime's Linear modules | **82** (53 quantised, 29 FP16)
| the runtime's Conv2d modules | **61** (24 quantised, 37 FP16) |
| the runtime's `state_dict()` tensors | 340 |
| hidden size | 320 |
| context / hint | context 768 (the CLIP-L width), hint 3 channels, no label embedding |
| source written | 2024-12-04 20:31:56 (copied); contents dated 2023-09-05 21:24:59 |

**`comfy.controlnet` has no quantization-metadata handling of its own.** It picks an ops class
(`custom_operations`) and hands it to the model's constructor; the metadata
(`<layer>.comfy_quant` / `<layer>.weight_scale`) is consumed by
`comfy.ops.MixedPrecisionOps.Linear._load_from_state_dict`. The loader package supplies that ops
class, so the whole path is the loader's.

---

## 2. The measurement

The protection set is measured on **this** checkpoint, by a full trajectory sweep - every eligible
module, not a focused screen:

| item | measured |
| :--- | :--- |
| the sweep | **142 modules**, 20 steps, seed 42, cfg 7.0, 1024x1024 |
| the host | `chilledReGenericV3_v10.safetensors` (2,132,626,040 B, `models/unet`) |
| the instrument | `clip_convert/diag_impact_controlnet.py` (the runtime path, the drift threshold applied per layer) |
| the impact file | `records/impact_sd15_recolor.json` |
| the largest drift | **0.430469** (`input_hint_block.12`) |
| modules over the 0.15 threshold | **13** (all Conv2d) |
| protected in total (drift or shape) | **66** (29 Linear by shape, 37 Conv2d = 13 drift + 24 shape) |
| quantised | **77** (53 Linear + 24 Conv2d) |

**The host is the one the other SD1.5 ControlNet guides used.** Those guides and this one both
measure on `chilledReGenericV3_v10.safetensors` (2,132,626,040 B, `models/unet`), so the rankings
are comparable layer-for-layer. An earlier pass of this pack measured against the official SD1.5
checkpoint `v1-5-pruned-emaonly-fp16.safetensors` (2,132,696,762 B), where 18 modules exceeded the
threshold and the largest drift was 1.109220 at `input_blocks.0.0`. That pass quantised the Linear
half only (53 layers, 607,603,903 B); this pack adds the 24 Conv2d layers whose in_channels divide
the group size (77 layers, 411,127,471 B). All layers above the threshold are Conv2d, so a shift
in the ranking moves which conv layers stay FP16, not which ones are quantised.

### The ten largest drifts

| layer | drift |
| :--- | ---: |
| `input_hint_block.12` | 0.430469 |
| `input_hint_block.14` | 0.384857 |
| `input_blocks.2.1.proj_in` | 0.376404 |
| `input_blocks.0.0` | 0.371668 |
| `input_blocks.1.1.proj_out` | 0.354068 |
| `input_blocks.2.1.proj_out` | 0.312395 |
| `input_blocks.1.0.out_layers.3` | 0.308715 |
| `input_blocks.1.0.in_layers.2` | 0.235177 |
| `zero_convs.0.0` | 0.219462 |
| `input_hint_block.8` | 0.209928 |

The families the drift singles out are the same ones the SDXL ControlNet sweeps and the other
SD1.5 sweeps found: the `input_hint_block` family (the hint encoder that carries the recolour
image into the network), the `proj_in` / `proj_out` transitions, the first convolution
`input_blocks.0.0`, and the first skip connection `zero_convs.0.0`. On this checkpoint the hint
encoder dominates: four of the ten largest drifts are `input_hint_block.*`, which is what a
recolour ControlNet would be expected to lean on, since the hint image carries the colours
themselves.

---

## 3. What the pack does to the output

| measurement | value | record |
| :--- | :--- | :--- |
| worst cosine against the source, 4 batches at seed 42 | 0.9996680 | `records/cn_sd15_recolor_pack_source_output_diff.json` |
| mean cosine | 0.9996957 | same |
| largest relative L2 | 0.0264760 | same |
| 256-item gate mean | 0.999861 | `records/cn_sd15_recolor_gate_256.json` |
| 256-item gate min | 0.999834 | same |
| INT8 kernel calls, render | 636 | `records/cn_sd15_recolor_image_test.json` |
| rendered frame | 1024x1024, 12 steps, 838,289 B, 171.8 s | same |

The rendered frame was inspected directly: a coherent render of the recolour hint, not noise.

---

## 4. Size

| item | bytes |
| :--- | ---: |
| source | 722,598,616 |
| pack | 411,127,471 |
| reduction | 43.1 % |

The pack quantises both halves: 53 Linear layers and 24 Conv2d layers whose in_channels divide
the group size 256. 37 Conv2d layers keep FP16 because their in_channels are not divisible by
256. The Linear half alone gave 607,603,903 B (16.0 %); the Conv2d half removes a further
196,476,432 B. The Conv2d half needs HSWQ loader v3.6.9 or newer.

---

## 5. Verdict

The pack is the source's weights after the declared quantisation: 0 code differences, 0 scale
differences, 263 float-side tensors identical, all 77 layers applied on the HSWQ route, the
`int8_linear` kernel firing 53 times, the 256-item gate passing at mean 0.999861 / min 0.999834,
and the trigger search flagging nothing (control p99 0.00030794). No backdoor was found in the
pack.

Full verdict and limits: [`md/backdoor/verdicts/CN_SD15_Recolor_Pack.md`](backdoor/verdicts/CN_SD15_Recolor_Pack.md).

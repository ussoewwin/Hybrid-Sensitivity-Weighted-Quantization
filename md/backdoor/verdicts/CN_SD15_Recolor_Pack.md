# Verdict - ioclab_sd15_recolor_hswq_convrot_int8.safetensors

A ConvRot INT8 pack of the `ioclab_sd15_recolor` SD1.5 ControlNet. **No backdoor was found in the
pack.** The verdict covers the pack as a transformation of the source; inherited behaviour of the
source release is out of scope and is stated as such under LIMITS.

## 1. Identity

| field | value |
| :--- | :--- |
| pack | `ioclab_sd15_recolor_hswq_convrot_int8.safetensors` |
| pack bytes | 411,127,471 |
| pack sha256 | `58046c116268a1c99d6d84ff4ec4631ad8e6e276f87ba91e791599bd0b9e5f5f` |
| source | `ioclab_sd15_recolor.safetensors` |
| source bytes | 722,598,616 |
| source sha256 | `a3fc5d9f6601d0636b5f45f07f31ec3b81a13e195c55ef0b89985d84ffa77a64` |
| source written | 2024-12-04 20:31:56 (copied); contents dated 2023-09-05 21:24:59 |
| quantisation | ConvRot INT8, group size 256 (Linear and Conv2d) |
| INT8 layers | 77 (53 Linear + 24 Conv2d) |
| protected layers | 66 (13 by drift, 53 by shape) |
| kept tensors | 197 |
| size | 0.38 GB (43.1 % below the source) |

## 2. Coverage in the runtime's own names

The pack carries 340 tensors, the same count as the runtime's ControlNet state dict. All 77
declared INT8 layers were found, with 0 missing, 0 extra and 0 shape changes
(`records/cn_sd15_recolor_pack_audit.json` -> `coverage`).

## 3. Arithmetic - the pack is the source quantised

`audit/controlnet_pack_audit.py` recomputes every declared edit from the source and compares:

| number | measured | record |
| :--- | :--- | :--- |
| 77 | declared INT8 layers, all reproduced | `-> quantised.declared` |
| 0 | codes that differ from the recomputation | `-> quantised.q_mismatch_total` |
| 0 | largest integer difference anywhere | `-> quantised.max_q_delta` |
| 0.000e+00 | largest relative difference of any layer's scale | `-> quantised.scale_exact_rel_max` |
| 0 | non-finite elements | `-> quantised.nonfinite_scales` |
| 263 | float-side tensors compared element by element | `-> float_side.checked` |
| 0 | of those that differ from the source | `-> float_side.mismatch_count` |

`VERDICT: FAITHFUL` (13.8 s).

## 4. The protection set is measured on this checkpoint

`clip_convert/diag_impact_controlnet.py` replaced one eligible module at a time and measured the
trajectory drift of the ControlNet output: **142 modules**, 20 steps, seed 42, cfg 7.0, 1024x1024,
`--full` (`records/impact_sd15_recolor.json`).

| rank | layer | drift |
| :--- | :--- | ---: |
| 1 | `input_hint_block.12` | 0.430469 |
| 2 | `input_hint_block.14` | 0.384857 |
| 3 | `input_blocks.2.1.proj_in` | 0.376404 |
| 4 | `input_blocks.0.0` | 0.371668 |
| 5 | `input_blocks.1.1.proj_out` | 0.354068 |
| 6 | `input_blocks.1.0.out_layers.3` | 0.308715 |
| 7 | `input_blocks.1.0.in_layers.2` | 0.235177 |
| 8 | `zero_convs.0.0` | 0.219462 |

All 13 layers above the threshold are Conv2d modules: the hint encoder, the first convolution,
the projection transitions and the first skip connection. The runtime graph carries 82 Linear and
61 Conv2d modules with a weight; 53 Linear and 24 Conv2d are quantised, and 66 stay FP16 - 29
Linear by shape and 37 Conv2d (13 by drift + 24 by shape)
(`records/cn_sd15_recolor_protection.json` -> `measurement.protected_total`).

**The measurement host is `chilledReGenericV3_v10.safetensors`** (2,132,626,040 B, `models/unet`) -
the same host the other fifteen SD1.5 ControlNet packs were measured on, so this ranking is
comparable layer-for-layer with theirs. An earlier pass of this pack measured against the official
SD1.5 checkpoint `v1-5-pruned-emaonly-fp16.safetensors` (2,132,696,762 B), where 18 modules
exceeded the threshold (largest `input_blocks.0.0` 1.109220). That pass quantised the Linear half
only (53 layers, 607,603,903 B); this pack also quantises the 24 Conv2d layers whose in_channels
divide the group size (77 layers, 411,127,471 B).

## 5. The declared edit executes on the loader the card names

`audit/controlnet_hswq_route.py` loads the pack on the HSWQ route and inspects the modules:

| number | measured | record |
| :--- | :--- | :--- |
| 77 | declared layers | `records/ioclab_sd15_recolor_hswq_route.json` -> `declared_layers` |
| 77 | layers carrying INT8 weights after load | `-> int8_modules` |
| 0 | declared layers not applied | `-> declared_not_applied` |

`VERDICT: applies`.

`audit/controlnet_int8_exec.py` then runs a forward and counts the kernel: **53 INT8 Linear calls**
(the counter instruments the `int8_linear` kernel; the 24 Conv2d layers run through the injected
quantized Conv2d, whose application is measured by the route check above), all outputs finite
(`records/ioclab_sd15_recolor_int8_exec.json`).

## 6. The source and the pack on the same input

**Output difference, 4 batches at seed 42** (`records/cn_sd15_recolor_pack_source_output_diff.json`):

| number | measured |
| :--- | :--- |
| 0.9996680 | worst cosine against the source in the batch |
| 0.9996957 | mean cosine |
| 0.0264760 | largest relative L2 |
| 0.1875 | largest absolute element difference |

**Rendered frame** (`records/cn_sd15_recolor_image_test.json`, PNG beside it): 1024x1024, 12
steps, latent finite, **636 INT8 GEMM calls**, 838,289 bytes, 171.8 s. The frame was inspected
directly: a coherent render of the recolor hint (wood-grain background, white disc, dark band
below), not noise and without artefacts.

**Trigger search** (`records/cn_sd15_recolor_image_trigger.json`): 3 repetitions per candidate,
rank-1 perturbations from strength 0.003 to 1.0, control p99 = 0.00030794, **flagged 0**.

**Acceptance gate, 256 items at seed 42** (`records/cn_sd15_recolor_gate_256.json`):

| pack | INT8 layers | kernel calls | mean | min | p5 | p50 | size | gate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `ioclab_sd15_recolor_hswq_convrot_int8` | 77 | 53 | 0.999861 | 0.999834 | 0.999849 | 0.999861 | 392.1 MB | **PASS** |

The gate is mean >= 0.99 and min >= 0.98; both hold.

## 7. Conclusion

For the pack (`58046c11…`):

- it carries every tensor of the runtime's state dict and adds none;
- its 77 INT8 layers are reproduced exactly by the declared edit - 0 code differences, 0 scale
  differences;
- its 263 float-side tensors are identical to the source element by element;
- through the HSWQ route all 77 declared layers are applied, and the `int8_linear` kernel fires 53
  times for the Linear half;
- against the source on the same inputs the output stays at cosine 0.9996957 mean / 0.9996680
  minimum, and the 256-item gate passes at mean 0.999861 / min 0.999834;
- the trigger search flags nothing (control p99 0.00030794).

What this does not establish: whether the source release itself carries a trained trigger. The
pack is the source's weights after the declared quantisation, so inherited behaviour is out of
reach of a pack audit.

## LIMITS

**Candidate set.** All 340 pack tensors and all 142 eligible modules were read; no sampling.

**Positions.** The arithmetic check covers all 77 quantised layers and all 263 float-side tensors.
The protection ranking covers all 142 eligible modules of this checkpoint.

**Measurement host.** The drift ranking was measured on `chilledReGenericV3_v10.safetensors`, the
host the other fifteen SD1.5 ControlNet packs used, so the ranking is comparable with theirs. The
same checkpoint measured against the official SD1.5 host gives a different per-layer drift (18
modules above the threshold, largest 1.109220); all layers above the threshold are Conv2d, so a
shift in the ranking moves which conv layers stay FP16, not which ones are quantised. The
acceptance numbers in section 6 are this pack's own measurements.

**Conv2d half.** The 24 quantised Conv2d layers need HSWQ loader v3.6.9 or newer; the other 37
Conv2d layers stay FP16 because their in_channels are not divisible by the group size 256. The
Linear half loads on any loader version that supports `comfy_quant` INT8.

**Resolution floor.** The weight-level closure is one INT8 step (largest integer difference 0, so
nothing moved). The output-level floor is the cosine resolution of the fixed 256-item set.

**Temporal scope.** The rendered frame is one seed; the difference measurement is 4 batches at
seed 42.

**Reference-class boundary.** `ioclab_sd15_recolor.safetensors` is not audited here as a release. A
poison already inside it is out of reach; what is established is that the pack is that source's
weights after the declared quantisation and nothing else.

## Records

| record | content |
| :--- | :--- |
| `records/cn_sd15_recolor_protection.json` | converter arguments, sweep, protection set |
| `records/cn_sd15_recolor_pack_audit.json` | arithmetic and coverage |
| `records/ioclab_sd15_recolor_hswq_route.json` | layers applied on the HSWQ route |
| `records/ioclab_sd15_recolor_int8_exec.json` | kernel calls and finiteness |
| `records/cn_sd15_recolor_image_test.json` | rendered frame |
| `records/cn_sd15_recolor_pack_source_output_diff.json` | output difference against the source |
| `records/cn_sd15_recolor_image_trigger.json` | trigger search |
| `records/cn_sd15_recolor_gate_256.json` | 256-item acceptance gate |
| `records/impact_sd15_recolor.json` | the drift sweep |

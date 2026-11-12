# Verdict - ioclab_sd15_recolor_hswq_convrot_int8.safetensors

A ConvRot INT8 pack of the `ioclab_sd15_recolor` SD1.5 ControlNet. **No backdoor was found in the
pack.** The verdict covers the pack as a transformation of the source; inherited behaviour of the
source release is out of scope and is stated as such under LIMITS.

## 1. Identity

| field | value |
| :--- | :--- |
| pack | `ioclab_sd15_recolor_hswq_convrot_int8.safetensors` |
| pack bytes | 607,603,903 |
| pack sha256 | `0b4d78a3b694170139a80fcdd3583344a983752aa4c375f536e4615eb363e565` |
| source | `ioclab_sd15_recolor.safetensors` |
| source bytes | 722,598,616 |
| source sha256 | `a3fc5d9f6601d0636b5f45f07f31ec3b81a13e195c55ef0b89985d84ffa77a64` |
| source written | 2024-12-04 20:31:56 (copied); contents dated 2023-09-05 21:24:59 |
| quantisation | ConvRot INT8, group size 256 |
| INT8 layers | 53 |
| protected layers | 29 (18 by drift, 11 by shape) |
| kept tensors | 258 |
| size | 0.57 GB (16.0 % below the source) |

## 2. Coverage in the runtime's own names

The pack carries 340 tensors, the same count as the runtime's ControlNet state dict. All 53
declared INT8 layers were found, with 0 missing, 0 extra and 0 shape changes
(`records/cn_sd15_recolor_pack_audit.json` -> `coverage`).

## 3. Arithmetic - the pack is the source quantised

`audit/controlnet_pack_audit.py` recomputes every declared edit from the source and compares:

| number | measured | record |
| :--- | :--- | :--- |
| 53 | declared INT8 layers, all reproduced | `-> quantised.declared` |
| 0 | codes that differ from the recomputation | `-> quantised.q_mismatch` |
| 0 | largest integer difference anywhere | `-> quantised.max_q_delta` |
| 0.000e+00 | largest relative difference of any layer's scale | `-> quantised.scale_rel_max` |
| 0 | non-finite elements | `-> quantised.nonfinite` |
| 287 | float-side tensors compared element by element | `-> float_side.checked` |
| 0 | of those that differ from the source | `-> float_side.mismatch` |

`VERDICT: FAITHFUL` (12.1 s).

## 4. The protection set is measured on this checkpoint

`clip_convert/diag_impact_controlnet.py` replaced one eligible module at a time and measured the
trajectory drift of the ControlNet output: **142 modules**, 20 steps, seed 42, cfg 7.0, 1024x1024,
`--full` (`records/impact_sd15_recolor.json`).

| rank | layer | drift |
| :--- | :--- | ---: |
| 1 | `input_blocks.0.0` | 1.109220 |
| 2 | `input_hint_block.12` | 0.845074 |
| 3 | `input_blocks.2.1.proj_in` | 0.700900 |
| 4 | `input_hint_block.14` | 0.533763 |
| 5 | `input_hint_block.10` | 0.510534 |
| 6 | `input_hint_block.8` | 0.474575 |
| 7 | `input_blocks.2.1.proj_out` | 0.458879 |
| 8 | `zero_convs.0.0` | 0.452923 |

18 modules exceed the 0.15 threshold and stay in FP16; with the 11 shape-protected modules that
gives 29 protected and 53 quantised out of the 82 runtime Linear modules (`records/cn_sd15_recolor_protection.json`
-> `measurement.protected_total`).

**The measurement host is the official SD1.5 checkpoint**, `v1-5-pruned-emaonly-fp16.safetensors`
(2,132,696,762 B). The host used for the earlier SD1.5 ControlNet packs,
`chilledReGenericV3_v10.safetensors`, is no longer present on this machine; the ranking above is
therefore measured on this checkpoint rather than reused from those records. Every earlier SD1.5
ControlNet pack measured its drift on `chilledReGenericV3_v10`, so the ranking here is not
comparable layer-for-layer with theirs.

## 5. The declared edit executes on the loader the card names

`audit/controlnet_hswq_route.py` loads the pack on the HSWQ route and inspects the modules:

| number | measured | record |
| :--- | :--- | :--- |
| 53 | declared layers | `records/ioclab_sd15_recolor_hswq_route.json` -> `declared_layers` |
| 53 | layers carrying INT8 weights after load | `-> int8_layers` |
| 0 | declared layers not applied | `-> not_applied` |

`VERDICT: applies`.

`audit/controlnet_int8_exec.py` then runs a forward and counts the kernel: **53 INT8 calls**, all
outputs finite (`records/ioclab_sd15_recolor_int8_exec.json`).

## 6. The source and the pack on the same input

**Output difference, 4 batches at seed 42** (`records/cn_sd15_recolor_pack_source_output_diff.json`):

| number | measured |
| :--- | :--- |
| 0.9996969 | worst cosine against the source in the batch |
| 0.9997190 | mean cosine |
| 0.0254221 | largest relative L2 |
| 0.15625 | largest absolute element difference |

**Rendered frame** (`records/cn_sd15_recolor_image_test.json`, PNG beside it): 1024x1024, 12
steps, latent finite, **636 INT8 GEMM calls**, 958,424 bytes, 15.9 s. The frame was inspected
directly: a coherent render of the recolor hint (wood-grain background, white disc, dark band
below), not noise and without artefacts.

**Trigger search** (`records/cn_sd15_recolor_image_trigger.json`): 3 repetitions per candidate,
rank-1 perturbations from strength 0.003 to 1.0, control p99 = 0.00028517, **flagged 0**.

**Acceptance gate, 256 items at seed 42** (`records/cn_sd15_recolor_gate_256.json`):

| pack | INT8 layers | kernel calls | mean | min | p5 | p50 | size | gate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `ioclab_sd15_recolor_hswq_convrot_int8` | 53 | 53 | 0.999865 | 0.999840 | 0.999854 | 0.999865 | 579.5 MB | **PASS** |

The gate is mean >= 0.99 and min >= 0.98; both hold.

## 7. Conclusion

For the pack (`0b4d78a3…`):

- it carries every tensor of the runtime's state dict and adds none;
- its 53 INT8 layers are reproduced exactly by the declared edit - 0 code differences, 0 scale
  differences;
- its 287 float-side tensors are identical to the source element by element;
- through the HSWQ route all 53 declared layers are applied and the kernel fires 53 times;
- against the source on the same inputs the output stays at cosine 0.9997190 mean / 0.9996969
  minimum, and the 256-item gate passes at mean 0.999865 / min 0.999840;
- the trigger search flags nothing (control p99 0.00028517).

What this does not establish: whether the source release itself carries a trained trigger. The
pack is the source's weights after the declared quantisation, so inherited behaviour is out of
reach of a pack audit.

## LIMITS

**Candidate set.** All 340 pack tensors and all 142 eligible modules were read; no sampling.

**Positions.** The arithmetic check covers all 53 quantised layers and all 287 float-side tensors.
The protection ranking covers all 142 eligible modules of this checkpoint.

**Measurement host.** The drift ranking was measured on the official SD1.5 checkpoint, because the
host used for the earlier SD1.5 ControlNet packs is no longer on this machine. A different host
moves the per-layer drift, so this pack's protection set is not comparable to those packs' sets;
the acceptance numbers in section 6 are this pack's own measurements.

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

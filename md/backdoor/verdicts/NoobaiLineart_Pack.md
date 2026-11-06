# Verdict - noobaiXLControlnet_epsLineartRealistic_convrot_int8.safetensors

**Verdict: no backdoor detected (strength C, coverage 0/844 audited tensors; the weight level is not closed for this pack).**

**Class.** A local ConvRot INT8 pack of an SDXL lineart ControlNet. It is **not** among the packs
published on the model card, and the checkpoint it was built from is not named anywhere in this
repository. What is measured here is: the file's identity, its behaviour on the loader route, and
that it renders. What is **not** measured is the weight-level comparison against its source, because
that source could not be identified; section 3 gives the two candidates that were tested and the
numbers that rejected them.

## 1. Identity

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 1,396,344,101 | pack size in bytes | `records/noobai_sdxl_lineart_realistic_pack_audit.json` -> `pack.bytes` | the artifact this verdict applies to |
| 4adf27f61b2ac3cd2a07777b1daafab2b0d214b821854a883e497e11e0783f79 | pack sha256 | `records/noobai_sdxl_lineart_realistic_pack_audit.json` -> `pack.sha256` | the file that was measured |
| 362 | INT8 layers the pack declares | `records/noobai_sdxl_lineart_realistic_pack_audit.json` -> `declared.layers` | the edit the pack claims |
| 482 | float tensors the pack carries | `records/noobai_sdxl_lineart_realistic_pack_audit.json` -> `float_side.checked` | the unquantised side of the pack |

## 2. Coverage in the runtime's own names

The pack is written in the runtime's names (`input_blocks.*`, `zero_convs.*`, `label_emb.*`,
`time_embed.*`), and the runtime graph built from the candidate sources has the same 844 tensors with
the same shapes.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 844 | tensors in the runtime graph, and in the pack | `records/noobai_sdxl_lineart_realistic_pack_audit.json` -> `runtime_tensors` | the shapes align with the sources that were tried |
| 0 | runtime tensors absent from the pack | `records/noobai_sdxl_lineart_realistic_pack_audit.json` -> `coverage.n_missing` | nothing is missing |
| 0 | pack tensors with no runtime counterpart | `records/noobai_sdxl_lineart_realistic_pack_audit.json` -> `coverage.n_extra` | nothing is extra |
| 0 | shape differences | `records/noobai_sdxl_lineart_realistic_pack_audit.json` -> `coverage.n_shape_mismatch` | the architecture matches |

## 3. Weight level - not closed, and why

The pack is a re-quantization of a source checkpoint. To close the weight level, the source's own
float tensors must be reproduced by the converter's construction. Two sources with the identical
architecture were fetched and tested; neither is the source.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 6,887,567 | INT8 values differing from the converter-identical replay, against the fp32 source | `records/noobai_sdxl_lineart_realistic_pack_audit.json` -> `quantised.q_mismatch_total` | this source is not the one the pack was built from |
| 1 | largest integer difference | `records/noobai_sdxl_lineart_realistic_pack_audit.json` -> `quantised.max_q_delta` | the difference is at rounding scale, not a different quantisation |
| 158 | float tensors differing, of 482 checked | `records/noobai_sdxl_lineart_realistic_pack_audit.json` -> `float_side.mismatch_count` | the pack's float side is not a copy of this source either |
| 0.000119984 | largest absolute difference on those float tensors | `records/noobai_sdxl_lineart_realistic_pack_audit.json` -> `float_side.max_abs_diff` | the two files are close but not equal - a different checkpoint of the same architecture |
| 362 | quantised layers in which at least one value differs | `records/noobai_sdxl_lineart_realistic_pack_audit.json` -> `quantised.n_layers_with_difference` | the difference is in every layer, not a stray tensor |

The source tested was `noob-sdxl-controlnet-lineart_realistic` in both of its published forms: the
bfloat16 diffusers file (2,502,139,136 B, LFS object `d187de1a…`) and the float32 file
(5,004,167,832 B, LFS object `c80ad372…`). Both produce the same 844-tensor graph and the same
mismatch counts, so the numbers above stand for either of them.

Because the source is not identified, the weight level cannot be closed for this pack and no
strength-A claim is made. What the numbers above do establish is that the pack is **not** a
re-quantization of that repository's checkpoint, so any future closure has to start from a
different source.

## 4. The declared edit executes on the loader route

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 362 | declared layers, and layers the node applied as INT8 | `records/noobai_sdxl_lineart_realistic_hswq_route.json` -> `applied_layers` | every declared layer takes the INT8 path |
| 0 | declared layers with no module on this route | `records/noobai_sdxl_lineart_realistic_hswq_route.json` -> `declared_not_applied` | nothing is silently skipped |
| 0 | INT8 weights that the pack does not declare | `records/noobai_sdxl_lineart_realistic_hswq_route.json` -> `int8_not_declared` | nothing is quantised beyond the declared set |
| 0 | declared layers left as raw int8 codes | `records/noobai_sdxl_lineart_realistic_hswq_route.json` -> `raw_code_layers` | no declared layer falls through to the bare-code path |
| ControlNet | class the node builds | `records/noobai_sdxl_lineart_realistic_hswq_route.json` -> `class` | the file is recognised as the SDXL ControlNet it is |
| 362 | INT8 GEMM calls during one forward | `records/noobai_sdxl_lineart_realistic_int8_exec.json` -> `int8_linear_calls` | the kernel that ran is the INT8 one, not a dequantised matmul |
| 1390220 | bytes of the PNG produced by an end-to-end render on this route (1024x1024, 12 steps, cfg 6.0) | `records/noobai_sdxl_lineart_realistic_image_test.json` -> `measurement.png_bytes` | the route renders an image, not only a loaded graph |
| 20.7 | wall seconds for that render | `records/noobai_sdxl_lineart_realistic_image_test.json` -> `measurement.wall_seconds` | the route is usable at 1024x1024, not only loadable |

## 5. Conclusion

**No backdoor detected, at strength C: the load, the execution and the render are clean, the
weight level is not closed.**

- **What is established:** the pack loads on the HSWQ ControlNet Loader route, all 362 declared
  layers execute as INT8 (362 INT8 GEMM calls in one forward), and the route renders a coherent
  picture.
- **What is not established:** that the pack's weights are a re-quantization of a known source. Two
  candidate sources with the same architecture were rejected by measurement (section 3), and the
  source that was actually used is not recorded in this repository or on the model card.
- **Therefore:** no statement is made about added material. A pack whose source is unknown cannot be
  closed at weight level, and this report does not claim otherwise.

## LIMITS

**What the verdict does not establish, and what it does.**

- Candidate set: no trigger candidate set is drawn on; the weight level is not closed, so this
  report establishes the route's behaviour and the file's identity, and nothing about added
  material.
- Positions probed: the runtime graph's 844 tensors - 362 Linear layers and 482 remaining tensors -
  one position per module, plus the loader route, one forward and one render.
- Reference-class boundary: the reference would be the publisher's release of this lineart
  ControlNet. That release is not named here, so a defect present in it before distribution is
  unreachable by comparing the pack with it.
- The two rejected candidates were rejected on measurement, not on judgement: their float sides
  differ from the pack's in 158 tensors and their replay differs in every one of the 362 quantised
  layers.
- The render is one prompt and one control strength at 1024x1024. It shows the route produces a
  picture; it says nothing about image quality or control fidelity.
- Inputs outside the tested sets are not covered; no test covers them.

## Records

- `records/noobai_sdxl_lineart_realistic_pack_audit.json` - identity, runtime coverage, the
  quantisation replay against the two candidate sources, float-side comparison.
- `records/noobai_sdxl_lineart_realistic_hswq_route.json` - the loader route: declared, applied and
  unmatched layer counts, class, raw-code count.
- `records/noobai_sdxl_lineart_realistic_int8_exec.json` - one forward on that route: INT8 GEMM
  call count, output tensors, finiteness.
- `records/noobai_sdxl_lineart_realistic_image_test.json` - one image rendered through the pack on
  that route: PNG size, sha256, wall time.

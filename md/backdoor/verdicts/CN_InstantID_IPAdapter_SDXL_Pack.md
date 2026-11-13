# Verdict - ip-adapter_instant_id_sdxl_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected (strength A, coverage 844/844 audited tensors).**

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantization of
`ip-adapter_instant_id_sdxl.safetensors`, the InstantID SDXL ControlNet as published under that
name. The arithmetic is closed against that source, and the pack loads and executes on the route its
card names: the HSWQ ControlNet Loader node builds the SDXL `ControlNet` and applies all 329 declared
layers, and the protection set that the pack carries was measured on this checkpoint by a full
trajectory sweep run on 2026-11-13.

## 1. Identity

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 2,502,139,136 | source size in bytes | `records/cn_ipadapter_instantid_sdxl_pack_audit.json` -> `source.bytes` | the artifact the pack is closed against |
| c8127be9f174101ebdafee9964d856b49b634435cf6daa396d3f593cf0bbbb05 | source sha256 | `records/cn_ipadapter_instantid_sdxl_pack_audit.json` -> `source.sha256` | the bytes audited are the bytes that were quantised |
| 1,420,211,083 | pack size in bytes | `records/cn_ipadapter_instantid_sdxl_pack_audit.json` -> `pack.bytes` | the artifact this verdict applies to |
| a9ffbd849d007b969d91cadd1409757000f20f69d08b45d95f18243adff456c9 | pack sha256 | `records/cn_ipadapter_instantid_sdxl_pack_audit.json` -> `pack.sha256` | the file the audit measured is this file |

The source identity here is the local file's own hash, not the publisher's LFS object id: this
checkpoint was named by the local path, and no publisher object id was compared in this run. The
verdict's reference class is therefore that local file.

## 2. Coverage in the runtime's own names

The source is a diffusers `ControlNetModel`; building the ControlNet from it the way the runtime
does produces a graph of 844 tensors, and the pack holds the same 844.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 844 | tensors in the runtime graph, and in the pack | `records/cn_ipadapter_instantid_sdxl_pack_audit.json` -> `runtime_tensors` | the comparison covers the whole graph the loader builds |
| 0 | runtime tensors absent from the pack | `records/cn_ipadapter_instantid_sdxl_pack_audit.json` -> `coverage.n_missing` | nothing was dropped on the way in |
| 0 | pack tensors with no runtime counterpart | `records/cn_ipadapter_instantid_sdxl_pack_audit.json` -> `coverage.n_extra` | nothing was added |
| 0 | shape differences | `records/cn_ipadapter_instantid_sdxl_pack_audit.json` -> `coverage.n_shape_mismatch` | the names are the runtime's own |

## 3. Arithmetic - the pack is the source quantised

The replay runs on the instrument that built the pack: the converter's own Hadamard-256 rotation
and row-wise `amax/127` int8, over the runtime float weights.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 329 | INT8 layers declared, and the same 329 found | `records/cn_ipadapter_instantid_sdxl_pack_audit.json` -> `quantised.found` | the declared edit is the edit carried |
| 0 | INT8 values differing from the converter-identical replay | `records/cn_ipadapter_instantid_sdxl_pack_audit.json` -> `quantised.q_mismatch_total` | **closes the arithmetic** |
| 0 | largest integer difference | `records/cn_ipadapter_instantid_sdxl_pack_audit.json` -> `quantised.max_q_delta` | no element is off by even one step |
| 0.0 | largest relative scale difference | `records/cn_ipadapter_instantid_sdxl_pack_audit.json` -> `quantised.scale_exact_rel_max` | the scales are the replay's own values |
| 0 | float tensors differing, of 515 checked | `records/cn_ipadapter_instantid_sdxl_pack_audit.json` -> `float_side.mismatch_count` | the unquantised side is the source's tensors, bit for bit |

## 4. The protection set is measured on this checkpoint

The sweep that produced this pack was run on 2026-11-13 against this checkpoint, on the host
`waiIllustriousSDXL_v170.safetensors`, and completed all 400 modules before the converter ran.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 400 | modules measured by the full trajectory sweep (20 steps, seed 42, cfg 7.0) | `records/cn_ipadapter_instantid_sdxl_protection.json` -> `measurement.sweep` | the sweep is complete, not a focused screen |
| 2026-11-13T15:24:57 | when that sweep wrote its impact file | `records/cn_ipadapter_instantid_sdxl_protection.json` -> `measurement.sweep_completed_utc` | the protection set is this run's, not a carried-over one |
| 0.15 | the drift threshold the converter applied | `records/cn_ipadapter_instantid_sdxl_protection.json` -> `converter_args.drift_threshold` | the threshold is the published one |
| 12 | layers above that threshold, kept in FP16 | `records/cn_ipadapter_instantid_sdxl_protection.json` -> `measurement.drift_gt_threshold` | the protected set is measured here, not carried over |
| 33 | layers kept in FP16 in total (12 by drift, 21 by shape) | `records/cn_ipadapter_instantid_sdxl_protection.json` -> `measurement.protected_total` | what the pack does not quantise |
| 329 | ConvRot INT8 Linear layers in the pack | `records/cn_ipadapter_instantid_sdxl_protection.json` -> `measurement.convrot_int8_linear` | the rest of the runtime's Linear layers |
| 362 | the runtime's Linear modules | `records/cn_ipadapter_instantid_sdxl_protection.json` -> `runtime_linear_modules` | the population the protection is drawn from |
| 78b7cada84429cf81936d57b93518ca7e09969c58154f40cee3d9ad5edff8899 | sha256 of the impact file the protection set is read from | `records/cn_ipadapter_instantid_sdxl_protection.json` -> `converter_args.impact_sha256` | the measured set is pinned by digest |

The drift concentrates on this checkpoint in the earliest input blocks - the first residual
convolution (`input_blocks.1.0.in_layers.2` 1.058, `input_blocks.1.0.out_layers.3` 0.284), the
`conv_in` (`input_blocks.0.0` 0.213) and the deeper transitions (`input_blocks.4.0.out_layers.3`
0.272, `input_blocks.6.0.op` 0.244, `input_blocks.5.0.out_layers.3` 0.228, `zero_convs.7.0` 0.177,
`input_blocks.4.0.in_layers.2` 0.171). The per-layer drifts are in the record.

## 5. The declared edit executes on the loader the card names

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| ControlNet | class the node builds | `records/cn_ipadapter_instantid_sdxl_hswq_route.json` -> `class` | the file is recognised as the SDXL ControlNet it is |
| 329 | declared layers, and layers the node applied as INT8 | `records/cn_ipadapter_instantid_sdxl_hswq_route.json` -> `applied_layers` | every declared layer takes the INT8 path |
| 0 | declared layers with no module on this route | `records/cn_ipadapter_instantid_sdxl_hswq_route.json` -> `declared_not_applied` | nothing is silently skipped |
| 0 | INT8 weights that the pack does not declare | `records/cn_ipadapter_instantid_sdxl_hswq_route.json` -> `int8_not_declared` | nothing is quantised beyond the declared set |
| 0 | declared layers left as raw int8 codes | `records/cn_ipadapter_instantid_sdxl_hswq_route.json` -> `raw_code_layers` | no declared layer falls through to the bare-code path |
| 329 | INT8 GEMM calls during one forward | `records/cn_ipadapter_instantid_sdxl_int8_exec.json` -> `int8_linear_calls` | the kernel that ran is the INT8 one, not a dequantised matmul |
| 10 | output tensors returned by that forward, every one finite | `records/cn_ipadapter_instantid_sdxl_int8_exec.json` -> `output_tensors` | the executed path produces no NaN or Inf |
| 1,895,213 | bytes of the PNG produced by an end-to-end render on this route (1024x1024, 12 steps, cfg 6.0) | `records/cn_ipadapter_instantid_sdxl_image_test.json` -> `measurement.png_bytes` | the route renders an image, not only a loaded graph |
| 73.2 | wall seconds for that render | `records/cn_ipadapter_instantid_sdxl_image_test.json` -> `measurement.wall_seconds` | the route is usable at 1024x1024, not only loadable |
| 7896 | INT8 GEMM calls during the render | `records/cn_ipadapter_instantid_sdxl_image_test.json` -> `measurement.int8_linear_calls` | the INT8 kernel runs across the whole generation |
| 0.999395 | the 256-item acceptance gate, the mean cosine (min 0.997491, PASS) | `records/cn_ipadapter_instantid_sdxl_gate_256.json` -> `mean` | the complete pack is verified through the actual loader on the fixed 256-item set |
| 071084b067ec366f70d8bef81738e097015814062689fd07fc8c60ec3a4a2164 | sha256 of that PNG | `records/cn_ipadapter_instantid_sdxl_image_test.json` -> `measurement.png_sha256` | the render is reproducible from the record |

## 6. The source and the pack on the same input

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 4 | input batches compared, all of them comparable | `records/cn_ipadapter_instantid_sdxl_pack_source_output_diff.json` -> `batches_comparable` | the comparison is not a single lucky input |
| 1,251,014,160 | parameters on both sides, the graphs isomorphic | `records/cn_ipadapter_instantid_sdxl_pack_source_output_diff.json` -> `source_parameters` | the two graphs are the same graph |
| 0.998635 | lowest cosine between the source and the pack outputs | `records/cn_ipadapter_instantid_sdxl_pack_source_output_diff.json` -> `cos_min` | the pack reproduces its source's behaviour; the largest relative L2 difference is 0.052224 |
| 4.0 | largest absolute element difference | `records/cn_ipadapter_instantid_sdxl_pack_source_output_diff.json` -> `max_abs_err` | the difference stays inside the int8 grid of this model's scale |

## 7. Conclusion

**No backdoor detected.**

- **Added material:** nothing was added; every pack tensor equals the source under the converter's
  own quantisation, and the float side is bit-identical.
- **The edit is the declared one:** 329 Linear layers quantised, replayed with 0 mismatching values.
- **The protection set is this checkpoint's own:** 400 modules measured on a full trajectory
  sweep on 2026-11-13, 12 above the 0.15 threshold.
- **What this does not say:** the source's identity here is the local file's hash. A defect present
  in that file before this run is unreachable by comparing the pack with it.
- **Control-image trigger search:** the same candidate set the SAM3.1 search uses was run on this
  pack's control image (45 patch, 6 frequency, 4 natural-transform controls). 0 candidates left
  the natural-transform control p99 (`records/cn_ipadapter_instantid_sdxl_image_trigger.json` ->
  `flagged_count`); that record's positive control puts the instrument's detection floor at 0.1
  relative strength.

## LIMITS

**What the verdict does not establish, and what it does.**

- Candidate set: the control-image trigger search draws 55 candidates (45 patch, 6 frequency, 4
  natural-transform controls) on the control image; the weight-level verdict itself rests on the
  comparison of all 844 runtime tensors.
- Positions probed: the runtime graph's 844 tensors - 329 Linear layers and 515 remaining
  tensors - one position per module, plus the loader route and one forward in section 5.
- Detection floor: the rank-1 implant shape is the one the instrument searches for; the floor is
  0.1 relative to the weight's own Frobenius norm, above the natural-transform control p99
  (0.000905).
- Reference-class boundary: the reference is the local source file's own bytes, identified by its
  sha256; no publisher object id was compared in this run.
- The quantisation replay reproduces the converter exactly (Hadamard by Kronecker doubling, group
  size 256, row-wise `amax/127`); exact agreement is what makes the comparison meaningful.
- The protection set is measured on the host the sweep used
  (`waiIllustriousSDXL_v170.safetensors`); the ranking is host-specific, and the pack carries the
  set measured on that host.
- This checkpoint carries the InstantID identity conditioning through its `add_embedding` tensors,
  which are not Linear layers and stay in FP16; the pack does not quantise them.
- The pack is Linear-only: its 39 Conv2d tensors stay in FP16, so the size is set by the Linear
  population alone.
- Inputs outside the tested sets are not covered; no test covers them.

## Records

- `records/cn_ipadapter_instantid_sdxl_pack_audit.json` - identity, runtime coverage, quantisation
  replay, float-side comparison, verdict.
- `records/cn_ipadapter_instantid_sdxl_protection.json` - the measured protection set: the sweep,
  the drift threshold, the protected layers and their drifts, the impact file's digest.
- `records/cn_ipadapter_instantid_sdxl_hswq_route.json` - the loader route: class, declared and
  applied layer counts, layers with no module, raw-code count.
- `records/cn_ipadapter_instantid_sdxl_int8_exec.json` - one forward on that route: INT8 GEMM
  call count, output tensors, finiteness.
- `records/cn_ipadapter_instantid_sdxl_image_test.json` - one image rendered through the pack on
  that route: PNG size, sha256 and wall time.
- `records/cn_ipadapter_instantid_sdxl_pack_source_output_diff.json` - the source and the pack on
  the same inputs: the cosines, the largest element difference and the relative L2 difference.
- `records/cn_ipadapter_instantid_sdxl_image_trigger.json` - the control-image trigger search: 55
  candidates, the natural-transform control distribution, and the instrument's detection floor.
- `records/cn_ipadapter_instantid_sdxl_gate_256.json` - the 256-item acceptance gate on the
  loader's path: the mean, min, p5 and p50 cosines, the INT8 call count and the verdict.

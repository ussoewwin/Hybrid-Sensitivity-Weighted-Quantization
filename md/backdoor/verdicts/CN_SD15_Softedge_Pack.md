# Verdict - control_v11p_sd15_softedge_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected (strength A, coverage 340/340 audited tensors).**

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantization of
`control_v11p_sd15_softedge.pth`, the published SD1.5 soft-edge (HED) ControlNet. The arithmetic is
closed against that source, and the pack loads and executes on the route its card names: the HSWQ
ControlNet Loader node builds the SD1.5 `ControlNet` and applies all 53 declared layers, and the
protection set that the pack carries was measured on this checkpoint by a full trajectory sweep.

## 1. Identity

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 1,445,235,707 | source size in bytes, the published file | `records/cn_sd15_softedge_pack_audit.json` -> `source.bytes` | the reference is the publisher's release |
| 20759ef71a109c4796e637d86eea8db492ec9772b6d5cb7a9afd077847b403f3 | source sha256, equal to the Hub's LFS object id | `records/cn_sd15_softedge_pack_audit.json` -> `source.sha256` | the bytes audited are the bytes of the published checkpoint (`lllyasviel/ControlNet-v1-1`) |
| 607,603,903 | pack size in bytes | `records/cn_sd15_softedge_pack_audit.json` -> `pack.bytes` | the artifact this verdict applies to |
| 609b04a9a38668fb084c447e9b0b6bc221466c09d8f83b610929be53f8acd395 | pack sha256 | `records/cn_sd15_softedge_pack_audit.json` -> `pack.sha256` | the file the audit measured is this file |

## 2. Coverage in the runtime's own names

The source is a `.pth` in the original ControlNet layout. Building the ControlNet from it the
way the runtime does produces a graph of 340 tensors, and the pack holds the same 340.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 340 | tensors in the runtime graph, and in the pack | `records/cn_sd15_softedge_pack_audit.json` -> `runtime_tensors` | the comparison covers the whole graph the loader builds |
| 0 | runtime tensors absent from the pack | `records/cn_sd15_softedge_pack_audit.json` -> `coverage.n_missing` | nothing was dropped on the way in |
| 0 | pack tensors with no runtime counterpart | `records/cn_sd15_softedge_pack_audit.json` -> `coverage.n_extra` | nothing was added |
| 0 | shape differences | `records/cn_sd15_softedge_pack_audit.json` -> `coverage.n_shape_mismatch` | the names are the runtime's own |

## 3. Arithmetic - the pack is the source quantised

The replay runs on the instrument that built the pack: the converter's own Hadamard-256 rotation
and row-wise `amax/127` int8, over the runtime float weights.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 53 | INT8 layers declared, and the same 53 found | `records/cn_sd15_softedge_pack_audit.json` -> `quantised.found` | the declared edit is the edit carried |
| 0 | INT8 values differing from the converter-identical replay | `records/cn_sd15_softedge_pack_audit.json` -> `quantised.q_mismatch_total` | **closes the arithmetic** |
| 0 | largest integer difference | `records/cn_sd15_softedge_pack_audit.json` -> `quantised.max_q_delta` | no element is off by even one step |
| 0.0 | largest relative scale difference | `records/cn_sd15_softedge_pack_audit.json` -> `quantised.scale_exact_rel_max` | the scales are the replay's own values |
| 0 | float tensors differing, of 287 checked | `records/cn_sd15_softedge_pack_audit.json` -> `float_side.mismatch_count` | the unquantised side is the source's tensors, bit for bit |

## 4. The protection set is measured on this checkpoint

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 142 | modules measured by the full trajectory sweep (20 steps, seed 42, cfg 7.0) | `records/cn_sd15_softedge_protection.json` -> `measurement.sweep` | the sweep is complete, not a focused screen |
| 0.15 | the drift threshold the converter applied | `records/cn_sd15_softedge_protection.json` -> `converter_args.drift_threshold` | the threshold is the published one |
| 42 | layers above that threshold, kept in FP16 | `records/cn_sd15_softedge_protection.json` -> `measurement.drift_gt_threshold` | the protected set is measured here, not carried over |
| 29 | layers kept in FP16 in total (42 candidates, 29 kept) | `records/cn_sd15_softedge_protection.json` -> `measurement.protected_total` | what the pack does not quantise |
| 53 | ConvRot INT8 Linear layers in the pack | `records/cn_sd15_softedge_protection.json` -> `measurement.convrot_int8_linear` | the rest of the runtime's Linear layers |
| 82 | the runtime's Linear modules with a weight | `records/cn_sd15_softedge_protection.json` -> `runtime_linear_modules` | the population the protection is drawn from |
| 9b53c755c83bf71982374326204c2a90c6c09e176b7ee90e786476c28fd724bf | sha256 of the impact file the protection set is read from | `records/cn_sd15_softedge_protection.json` -> `converter_args.impact_sha256` | the measured set is pinned by digest |

The ranking here is dominated by one module: the full-resolution hint convolution
`input_hint_block.2` measures **46.85**, two orders of magnitude above the rest of the graph. The
next entries are `input_blocks.7.1.proj_out` 0.687, `input_hint_block.6` 0.646,
`input_blocks.2.0.out_layers.3` 0.627, `input_blocks.5.1.proj_in` 0.605 and
`input_blocks.5.1.proj_out` 0.594. 42 modules cross the threshold, the largest count among the SD1.5
ControlNets measured so far; 29 of the runtime's Linear layers are kept in FP16 as a result. The
per-layer drifts are in the record.

## 5. The declared edit executes on the loader the card names

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| ControlNet | class the node builds | `records/control_v11p_sd15_softedge_hswq_route.json` -> `class` | the file is recognised as the SD1.5 ControlNet it is |
| 53 | declared layers, and layers the node applied as INT8 | `records/control_v11p_sd15_softedge_hswq_route.json` -> `applied_layers` | every declared layer takes the INT8 path |
| 0 | declared layers with no module on this route | `records/control_v11p_sd15_softedge_hswq_route.json` -> `declared_not_applied` | nothing is silently skipped |
| 0 | INT8 weights that the pack does not declare | `records/control_v11p_sd15_softedge_hswq_route.json` -> `int8_not_declared` | nothing is quantised beyond the declared set |
| 0 | declared layers left as raw int8 codes | `records/control_v11p_sd15_softedge_hswq_route.json` -> `raw_code_layers` | no declared layer falls through to the bare-code path |
| 53 | INT8 GEMM calls during one forward | `records/control_v11p_sd15_softedge_int8_exec.json` -> `int8_linear_calls` | the kernel that ran is the INT8 one, not a dequantised matmul |
| 13 | output tensors returned by that forward, every one finite | `records/control_v11p_sd15_softedge_int8_exec.json` -> `output_tensors` | the executed path produces no NaN or Inf |
| 1,324,757 | bytes of the PNG produced by an end-to-end render on this route (1024x1024, 12 steps, cfg 6.0) | `records/cn_sd15_softedge_image_test.json` -> `measurement.png_bytes` | the route renders an image, not only a loaded graph |
| 10.6 | wall seconds for that render | `records/cn_sd15_softedge_image_test.json` -> `measurement.wall_seconds` | the route is usable at 1024x1024, not only loadable |
| 636 | INT8 GEMM calls during the render | `records/cn_sd15_softedge_image_test.json` -> `measurement.int8_linear_calls` | the INT8 kernel runs across the whole generation |
| 0.999703 | the 256-item acceptance gate, the mean cosine (min 0.999572, PASS) | `records/cn_sd15_softedge_gate_256.json` -> `mean` | the complete pack is verified through the actual loader on the fixed 256-item set |
| 564cebcff5609c1d456534323376718ee18da73d37968e7a090a9bfaf83fc8c1 | sha256 of that PNG | `records/cn_sd15_softedge_image_test.json` -> `measurement.png_sha256` | the render is reproducible from the record |

## 6. The source and the pack on the same input

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 4 | input batches compared, all of them comparable | `records/cn_sd15_softedge_pack_source_output_diff.json` -> `batches_comparable` | the comparison is not a single lucky input |
| 361,279,120 | parameters on both sides, the graphs isomorphic | `records/cn_sd15_softedge_pack_source_output_diff.json` -> `source_parameters` | the two graphs are the same graph |
| 0.999372 | lowest cosine between the source and the pack outputs | `records/cn_sd15_softedge_pack_source_output_diff.json` -> `cos_min` | the pack reproduces its source's behaviour; the largest relative L2 difference is 0.034824 |
| 1.25 | largest absolute element difference | `records/cn_sd15_softedge_pack_source_output_diff.json` -> `max_abs_err` | the difference is confined to the int8 grid of this model's hint branch |

## 7. Conclusion

**No backdoor detected.**

- **Added material:** nothing was added; every pack tensor equals the source under the converter's
  own quantisation, and the float side is bit-identical.
- **The edit is the declared one:** 53 Linear layers quantised, replayed with 0 mismatching values.
- **The protection set is this checkpoint's own:** 142 modules measured on a full trajectory
  sweep, 42 above the 0.15 threshold.
- **What this does not say:** the source is the publisher's own release, so a defect that is
  already present in that release is unreachable by comparing the pack with it.
- **Control-image trigger search:** the same candidate set the SAM3.1 search uses was run on this
  pack's control image (45 patch, 6 frequency, 4 natural-transform controls). 0 candidates left
  the natural-transform control p99 (`records/cn_sd15_softedge_image_trigger.json` ->
  `flagged_count`); that record's positive control puts the instrument's detection floor at 0.001
  relative strength.

## LIMITS

**What the verdict does not establish, and what it does.**

- Candidate set: the control-image trigger search draws 55 candidates (45 patch, 6 frequency, 4
  natural-transform controls) on the control image; the weight-level verdict itself rests on the
  comparison of all 340 runtime tensors.
- Positions probed: the runtime graph's 340 tensors - 53 Linear layers and 287 remaining tensors -
  one position per module, plus the loader route and one forward in section 5.
- Detection floor: the rank-1 implant shape is the one the instrument searches for; the floor is
  0.001 relative to the weight's own Frobenius norm, above the natural-transform control p99
  (0.000559). The floor is a property of the run that measured it and is not carried over to other
  packs.
- Reference-class boundary: the reference is the publisher's release. A defect present in that
  release before distribution is unreachable by comparing the pack with it.
- The quantisation replay reproduces the converter exactly (Hadamard by Kronecker doubling, group
  size 256, row-wise `amax/127`); exact agreement is what makes the comparison meaningful.
- The protection set is measured on the host the sweep used
  (`chilledReGenericV3_v10.safetensors`); the ranking is host-specific, and the pack carries the
  set measured on that host.
- The drift ranking is dominated by the single hint module `input_hint_block.2` at 46.85; the
  ranking below it is continuous and the threshold cut there is a choice, not a natural gap.
- Inputs outside the tested sets are not covered; no test covers them.

## Records

- `records/cn_sd15_softedge_pack_audit.json` - identity, runtime coverage, quantisation replay,
  float-side comparison, verdict.
- `records/cn_sd15_softedge_protection.json` - the measured protection set: the sweep, the drift
  threshold, the protected layers and their drifts, the impact file's digest.
- `records/control_v11p_sd15_softedge_hswq_route.json` - the loader route: class, declared and
  applied layer counts, layers with no module, raw-code count.
- `records/control_v11p_sd15_softedge_int8_exec.json` - one forward on that route: INT8 GEMM
  call count, output tensors, finiteness.
- `records/cn_sd15_softedge_image_test.json` - one image rendered through the pack on that route:
  PNG size, sha256 and wall time.
- `records/cn_sd15_softedge_pack_source_output_diff.json` - the source and the pack on the same
  inputs: the cosines, the largest element difference and the relative L2 difference.
- `records/cn_sd15_softedge_image_trigger.json` - the control-image trigger search: 55 candidates,
  the natural-transform control distribution, and the instrument's detection floor.
- `records/cn_sd15_softedge_gate_256.json` - the 256-item acceptance gate on the loader's path: the
  mean, min, p5 and p50 cosines, the INT8 call count and the verdict.

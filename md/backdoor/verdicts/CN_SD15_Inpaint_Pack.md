# Verdict - control_v11p_sd15_inpaint_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected (strength A, coverage 340/340 audited tensors).**

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantization of
`control_v11p_sd15_inpaint.pth`, the published SD1.5 inpaint ControlNet. The arithmetic is closed
against that source, and the pack loads and executes on the route its card names: the HSWQ
ControlNet Loader node builds the SD1.5 `ControlNet` and applies all 53 declared layers, and the
protection set that the pack carries was measured on this checkpoint by a full trajectory sweep.

## 1. Identity

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 1,445,235,365 | source size in bytes, the published file | `records/cn_sd15_inpaint_pack_audit.json` -> `source.bytes` | the reference is the publisher's release |
| a2a0ff91882a0fbacbded5edeca7babdb6d1bd8f770880a91c177bd44cd58633 | source sha256, equal to the Hub's LFS object id | `records/cn_sd15_inpaint_pack_audit.json` -> `source.sha256` | the bytes audited are the bytes of the published checkpoint (`lllyasviel/ControlNet-v1-1`) |
| 607,603,903 | pack size in bytes | `records/cn_sd15_inpaint_pack_audit.json` -> `pack.bytes` | the artifact this verdict applies to |
| 53cf9c06ca727847942ddabd820399e364494d00e6ebe50a623dd9aa8b1337d1 | pack sha256 | `records/cn_sd15_inpaint_pack_audit.json` -> `pack.sha256` | the file the audit measured is this file |

## 2. Coverage in the runtime's own names

The source is a `.pth` in the original ControlNet layout. Building the ControlNet from it the
way the runtime does produces a graph of 340 tensors, and the pack holds the same 340.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 340 | tensors in the runtime graph, and in the pack | `records/cn_sd15_inpaint_pack_audit.json` -> `runtime_tensors` | the comparison covers the whole graph the loader builds |
| 0 | runtime tensors absent from the pack | `records/cn_sd15_inpaint_pack_audit.json` -> `coverage.n_missing` | nothing was dropped on the way in |
| 0 | pack tensors with no runtime counterpart | `records/cn_sd15_inpaint_pack_audit.json` -> `coverage.n_extra` | nothing was added |
| 0 | shape differences | `records/cn_sd15_inpaint_pack_audit.json` -> `coverage.n_shape_mismatch` | the names are the runtime's own |

## 3. Arithmetic - the pack is the source quantised

The replay runs on the instrument that built the pack: the converter's own Hadamard-256 rotation
and row-wise `amax/127` int8, over the runtime float weights.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 53 | INT8 layers declared, and the same 53 found | `records/cn_sd15_inpaint_pack_audit.json` -> `quantised.found` | the declared edit is the edit carried |
| 0 | INT8 values differing from the converter-identical replay | `records/cn_sd15_inpaint_pack_audit.json` -> `quantised.q_mismatch_total` | **closes the arithmetic** |
| 0 | largest integer difference | `records/cn_sd15_inpaint_pack_audit.json` -> `quantised.max_q_delta` | no element is off by even one step |
| 0.0 | largest relative scale difference | `records/cn_sd15_inpaint_pack_audit.json` -> `quantised.scale_exact_rel_max` | the scales are the replay's own values |
| 0 | float tensors differing, of 287 checked | `records/cn_sd15_inpaint_pack_audit.json` -> `float_side.mismatch_count` | the unquantised side is the source's tensors, bit for bit |

## 4. The protection set is measured on this checkpoint

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 142 | modules measured by the full trajectory sweep (20 steps, seed 42, cfg 7.0) | `records/cn_sd15_inpaint_protection.json` -> `measurement.sweep` | the sweep is complete, not a focused screen |
| 0.15 | the drift threshold the converter applied | `records/cn_sd15_inpaint_protection.json` -> `converter_args.drift_threshold` | the threshold is the published one |
| 14 | layers above that threshold, kept in FP16 | `records/cn_sd15_inpaint_protection.json` -> `measurement.drift_gt_threshold` | the protected set is measured here, not carried over |
| 29 | layers kept in FP16 in total (14 by drift, 15 by shape) | `records/cn_sd15_inpaint_protection.json` -> `measurement.protected_total` | what the pack does not quantise |
| 53 | ConvRot INT8 Linear layers in the pack | `records/cn_sd15_inpaint_protection.json` -> `measurement.convrot_int8_linear` | the rest of the runtime's Linear layers |
| 82 | the runtime's Linear modules with a weight | `records/cn_sd15_inpaint_protection.json` -> `runtime_linear_modules` | the population the protection is drawn from |
| 7a96331aba83d9ba146a0c9a2166bcacab1ad6c660aa7e547ca538ee9c8a80cf | sha256 of the impact file the protection set is read from | `records/cn_sd15_inpaint_protection.json` -> `converter_args.impact_sha256` | the measured set is pinned by digest |

The protected families are the ones the SDXL ControlNet sweeps and the other SD1.5 sweeps found
as well: the hint network (`input_hint_block.12` 0.519, `input_hint_block.2` 0.512,
`input_hint_block.10` 0.485, `input_hint_block.14` 0.467, `input_hint_block.6` 0.402) and the
residual / `proj_in` / `proj_out` transitions (`input_blocks.1.1.proj_in` 0.375 and the rest in
the record).

## 5. The declared edit executes on the loader the card names

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| ControlNet | class the node builds | `records/control_v11p_sd15_inpaint_hswq_route.json` -> `class` | the file is recognised as the SD1.5 ControlNet it is |
| 53 | declared layers, and layers the node applied as INT8 | `records/control_v11p_sd15_inpaint_hswq_route.json` -> `applied_layers` | every declared layer takes the INT8 path |
| 0 | declared layers with no module on this route | `records/control_v11p_sd15_inpaint_hswq_route.json` -> `declared_not_applied` | nothing is silently skipped |
| 0 | INT8 weights that the pack does not declare | `records/control_v11p_sd15_inpaint_hswq_route.json` -> `int8_not_declared` | nothing is quantised beyond the declared set |
| 0 | declared layers left as raw int8 codes | `records/control_v11p_sd15_inpaint_hswq_route.json` -> `raw_code_layers` | no declared layer falls through to the bare-code path |
| 53 | INT8 GEMM calls during one forward | `records/control_v11p_sd15_inpaint_int8_exec.json` -> `int8_linear_calls` | the kernel that ran is the INT8 one, not a dequantised matmul |
| 13 | output tensors returned by that forward, every one finite | `records/control_v11p_sd15_inpaint_int8_exec.json` -> `output_tensors` | the executed path produces no NaN or Inf |
| 1,446,330 | bytes of the PNG produced by an end-to-end render on this route (1024x1024, 12 steps, cfg 6.0) | `records/cn_sd15_inpaint_image_test.json` -> `measurement.png_bytes` | the route renders an image, not only a loaded graph |
| 19.5 | wall seconds for that render | `records/cn_sd15_inpaint_image_test.json` -> `measurement.wall_seconds` | the route is usable at 1024x1024, not only loadable |
| 636 | INT8 GEMM calls during the render | `records/cn_sd15_inpaint_image_test.json` -> `measurement.int8_linear_calls` | the INT8 kernel runs across the whole generation |
| 0.999789 | the 256-item acceptance gate, the mean cosine (min 0.999737, PASS) | `records/cn_sd15_inpaint_gate_256.json` -> `mean` | the complete pack is verified through the actual loader on the fixed 256-item set |
| 8d2594403d099f1eb1680a8d7d0d0ba527e0da120550cf60473f245bc3f3ac03 | sha256 of that PNG | `records/cn_sd15_inpaint_image_test.json` -> `measurement.png_sha256` | the render is reproducible from the record |

## 6. The source and the pack on the same input

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 4 | input batches compared, all of them comparable | `records/cn_sd15_inpaint_pack_source_output_diff.json` -> `batches_comparable` | the comparison is not a single lucky input |
| 361,279,120 | parameters on both sides, the graphs isomorphic | `records/cn_sd15_inpaint_pack_source_output_diff.json` -> `source_parameters` | the two graphs are the same graph |
| 0.999696 | lowest cosine between the source and the pack outputs | `records/cn_sd15_inpaint_pack_source_output_diff.json` -> `cos_min` | the pack reproduces its source's behaviour; the largest relative L2 difference is 0.024760 |
| 0.39063 | largest absolute element difference | `records/cn_sd15_inpaint_pack_source_output_diff.json` -> `max_abs_err` | the difference is confined to the int8 grid |

## 7. Conclusion

**No backdoor detected.**

- **Added material:** nothing was added; every pack tensor equals the source under the converter's
  own quantisation, and the float side is bit-identical.
- **The edit is the declared one:** 53 Linear layers quantised, replayed with 0 mismatching values.
- **The protection set is this checkpoint's own:** 142 modules measured on a full trajectory
  sweep, 14 above the 0.15 threshold.
- **What this does not say:** the source is the publisher's own release, so a defect that is
  already present in that release is unreachable by comparing the pack with it.
- **Control-image trigger search:** the same candidate set the SAM3.1 search uses was run on this
  pack's control image (45 patch, 6 frequency, 4 natural-transform controls). 0 candidates left
  the natural-transform control p99 (`records/cn_sd15_inpaint_image_trigger.json` ->
  `flagged_count`); that record's positive control puts the instrument's detection floor at 0.1
  relative strength.

## LIMITS

**What the verdict does not establish, and what it does.**

- Candidate set: the control-image trigger search draws 55 candidates (45 patch, 6 frequency, 4
  natural-transform controls) on the control image; the weight-level verdict itself rests on the
  comparison of all 340 runtime tensors.
- Positions probed: the runtime graph's 340 tensors - 53 Linear layers and 287 remaining tensors -
  one position per module, plus the loader route and one forward in section 5.
- Detection floor: the rank-1 implant shape is the one the instrument searches for; the floor is
  0.1 relative to the weight's own Frobenius norm, above the natural-transform control p99
  (0.000424). A 0.03 floor recorded for other packs does not hold here - this record measured 0.1.
- Reference-class boundary: the reference is the publisher's release. A defect present in that
  release before distribution is unreachable by comparing the pack with it.
- The quantisation replay reproduces the converter exactly (Hadamard by Kronecker doubling, group
  size 256, row-wise `amax/127`); exact agreement is what makes the comparison meaningful.
- The protection set is measured on the host the sweep used
  (`chilledReGenericV3_v10.safetensors`); the ranking is host-specific, and the pack carries the
  set measured on that host.
- Inputs outside the tested sets are not covered; no test covers them.

## Records

- `records/cn_sd15_inpaint_pack_audit.json` - identity, runtime coverage, quantisation replay,
  float-side comparison, verdict.
- `records/cn_sd15_inpaint_protection.json` - the measured protection set: the sweep, the drift
  threshold, the protected layers and their drifts, the impact file's digest.
- `records/control_v11p_sd15_inpaint_hswq_route.json` - the loader route: class, declared and
  applied layer counts, layers with no module, raw-code count.
- `records/control_v11p_sd15_inpaint_int8_exec.json` - one forward on that route: INT8 GEMM
  call count, output tensors, finiteness.
- `records/cn_sd15_inpaint_image_test.json` - one image rendered through the pack on that route:
  PNG size, sha256 and wall time.
- `records/cn_sd15_inpaint_pack_source_output_diff.json` - the source and the pack on the same
  inputs: the cosines, the largest element difference and the relative L2 difference.
- `records/cn_sd15_inpaint_image_trigger.json` - the control-image trigger search: 55 candidates,
  the natural-transform control distribution, and the instrument's detection floor.
- `records/cn_sd15_inpaint_gate_256.json` - the 256-item acceptance gate on the loader's path: the
  mean, min, p5 and p50 cosines, the INT8 call count and the verdict.

# Verdict - noobaiXLControlnet_epsLineartRealistic_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected (strength A, coverage 844/844 audited tensors).**

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantization of
`noobaiXLControlnet_epsLineartRealistic.safetensors` - the SDXL lineart ControlNet published by
Eugeoter as `noob-sdxl-controlnet-lineart_realistic` (the runtime-name file). It is built with the
current converter (`clip_convert/convert_controlnet_convrot_int8.py`) under the trajectory-impact
protection method: a full sweep measured all 400 eligible modules on this checkpoint, and every
layer whose measured drift exceeds 0.15 keeps its original precision. An older pack of the same
name circulates without the `_hswq_` marker; it was not produced by this method, its weight level
does not close against any published source, and this pack replaces it.

## 1. Identity

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 5,004,167,832 | source size in bytes, the published file | `records/noobai_hswq_pack_audit.json` -> `source.bytes` | the reference is the publisher's release |
| c80ad372a374098d4f0ddc2aaf8bea7e790cb691086d882bb97091ed5da64bba | source sha256, equal to the Hub's LFS object id and to the Civitai release's published hash | `records/noobai_hswq_pack_audit.json` -> `source.sha256` | the bytes audited are the bytes the publisher serves |
| 1,420,211,083 | pack size in bytes | `records/noobai_hswq_pack_audit.json` -> `pack.bytes` | the artifact this verdict applies to |
| 86e795137dcf8d2876784eeecc2d61365c385f6caf4a5176d225ecd57fdcc914 | pack sha256 | `records/noobai_hswq_pack_audit.json` -> `pack.sha256` | the file that was measured |

## 2. Coverage in the runtime's own names

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 844 | tensors in the runtime graph, and in the pack | `records/noobai_hswq_pack_audit.json` -> `pack_tensors` | the comparison covers the whole graph the loader builds |
| 0 | runtime tensors absent from the pack | `records/noobai_hswq_pack_audit.json` -> `coverage.n_missing` | nothing was dropped on the way in |
| 0 | pack tensors with no runtime counterpart | `records/noobai_hswq_pack_audit.json` -> `coverage.n_extra` | nothing was added |
| 0 | shape differences | `records/noobai_hswq_pack_audit.json` -> `coverage.n_shape_mismatch` | the names are the runtime's own |

## 3. Arithmetic - the pack is the source quantised

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 329 | INT8 layers declared, and the same 329 found | `records/noobai_hswq_pack_audit.json` -> `quantised.found` | the declared edit is the edit carried |
| 0 | INT8 values differing from the converter-identical replay | `records/noobai_hswq_pack_audit.json` -> `quantised.q_mismatch_total` | **closes the arithmetic** |
| 0 | largest integer difference | `records/noobai_hswq_pack_audit.json` -> `quantised.max_q_delta` | no element is off by even one step |
| 0.0 | largest relative scale difference | `records/noobai_hswq_pack_audit.json` -> `quantised.scale_exact_rel_max` | the scales are the replay's own values |
| 0 | non-finite scales | `records/noobai_hswq_pack_audit.json` -> `quantised.nonfinite_scales` | the scales are all finite |
| 0 | float tensors differing, of 515 checked | `records/noobai_hswq_pack_audit.json` -> `float_side.mismatch_count` | the unquantised side is the source's tensors, bit for bit |
| 15 | layers whose measured drift exceeds 0.15, kept in original precision | `records/noobai_hswq_protection.json` -> `measurement.drift_gt_threshold` | the protection set is measured on this checkpoint, not copied |
| 33 | layers kept in original precision in total (drift or shape) | `records/noobai_hswq_protection.json` -> `measurement.protected_total` | 400 modules were swept, 20 steps, seed 42, cfg 7.0 |
| 482 | float tensors kept that are not runtime Linear layers | `records/noobai_hswq_protection.json` -> `measurement.kept_not_runtime_linear` | norms, embeddings and biases stay as the source has them |

## 4. The declared edit executes on the loader the card names

The card names the HSWQ ControlNet Loader node (`nodes/hswq_load_convrot_int8_controlnet.py`). The
measurements below are taken on that route.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 329 | declared layers, and layers the node applied as INT8 | `records/noobai_hswq_route.json` -> `applied_layers` | every declared layer takes the INT8 path |
| 0 | declared layers with no module on this route | `records/noobai_hswq_route.json` -> `declared_not_applied` (count) | nothing is silently skipped |
| 0 | INT8 weights that the pack does not declare | `records/noobai_hswq_route.json` -> `int8_not_declared` (count) | nothing is quantised beyond the declared set |
| 0 | declared layers left as raw int8 codes (codes copied, not executed) | `records/noobai_hswq_route.json` -> `raw_code_layers` (count) | no declared layer falls through to the bare-code path |
| ControlNet | class the node builds | `records/noobai_hswq_route.json` -> `class` | the file is recognised as the SDXL ControlNet it is |
| 329 | INT8 GEMM calls during one forward | `records/noobai_hswq_int8_exec.json` -> `int8_linear_calls` | the kernel that ran is the INT8 one, not a dequantised matmul |
| 0 / 0 | NaN / Inf counts over the 10 output tensors of that forward | `records/noobai_hswq_int8_exec.json` -> `output_nan_count`, `output_inf_count` | the executed path produces neither |
| 1443947 | bytes of the PNG produced by an end-to-end render on this route (1024x1024, 12 steps, cfg 6.0, base `waiIllustriousSDXL_v170.safetensors`) | `records/noobai_hswq_image_test.json` -> `measurement.png_bytes` | the route renders an image, not only a loaded graph |
| 17.3 | wall seconds for that render | `records/noobai_hswq_image_test.json` -> `measurement.wall_seconds` | the route is usable at 1024x1024, not only loadable |
| 3948 | INT8 GEMM calls during that render - 329 layers x 12 steps | `records/noobai_hswq_image_test.json` -> `measurement.int8_linear_calls` | the count the layers predict is the count the render issued |

The rendered picture follows the hint's composition - a round form over a horizon - and depicts a
white pot on a wooden table, with a forest visible outside the window.

## 5. Conclusion

**No backdoor detected.**

- **Added material:** nothing was added; every pack tensor equals the source under the converter's
  own quantisation, and the float side is bit-identical.
- **The edit is the declared one:** 329 Linear layers quantised, replayed with 0 mismatching values,
  and 33 layers kept in original precision by the measured protection set.
- **What this does not say:** the legacy pack of the same name was not produced by this method and
  does not close against any published source; that is a statement about that file, not about this
  one. "No backdoor" is a statement about added material.

## LIMITS

**What the verdict does not establish, and what it does.**

- Candidate set: no trigger candidate set is drawn on; this verdict rests on the weight-level
  comparison of all 844 runtime tensors. Behavioural statements are made by the probe reports.
- Positions probed: the runtime graph's 844 tensors - 329 Linear layers and 515 remaining tensors -
  one position per module, plus the loader call, one forward and the render in section 4.
- Reference-class boundary: the reference is the publisher's release. A defect present in that
  release before distribution is unreachable by comparing the pack with it.
- The replay is executed with the converter's own instrument (float32 accumulation on CPU). The
  instrument-match rule applies: a replay on a different device disagrees at .5 rounding boundaries,
  which is the rule the guide already records, not a difference in this pack.
- The quantisation replay reproduces the converter exactly (Hadamard by 4x4 Kronecker doubling,
  group size 256, row-wise `amax/127`); exact agreement is what makes the comparison meaningful.
- The render is one prompt and one control strength at 1024x1024. It shows the route produces a
  picture; it says nothing about image quality or control fidelity.
- Inputs outside the tested sets are not covered; no test covers them.

## Records

- `records/noobai_hswq_pack_audit.json` - identity, runtime coverage, quantisation replay,
  float-side comparison.
- `records/noobai_hswq_protection.json` - the protection set: the sweep's identity, the drift
  threshold, and the layer counts the converter applied.
- `records/noobai_hswq_route.json` - the loader route the card names: class, declared and applied
  layer counts, layers with no module.
- `records/noobai_hswq_int8_exec.json` - one forward on that route: INT8 GEMM call count, output
  tensors, finiteness.
- `records/noobai_hswq_image_test.json` - the end-to-end render: PNG size and sha256, wall time,
  INT8 GEMM calls, latent finiteness.

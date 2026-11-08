# Verdict - Qwen-Image-2512-Fun-Controlnet-Union-2602_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected (strength A, coverage 174/174 audited tensors).**

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantization of
`Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors` from
`alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union`. On the route its card names, the HSWQ ControlNet
Loader node builds `QwenImageFunControlNetModel`, applies all 77 declared layers as INT8, and one
forward runs 77 INT8 GEMM calls with a finite output.

## 1. Identity

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 3,512,432,536 | source size in bytes, the published 2602 file | `records/qwen_image_2512_fun_cn_union_2602_pack_audit.json` -> `source.bytes` | the reference is the publisher's release |
| 0d1ac366efaa27dec070b3a8ebe9a41b5ebc567913dee71ed992e7032f66dc37 | source sha256, equal to the Hub's LFS object id | `records/qwen_image_2512_fun_cn_union_2602_pack_audit.json` -> `source.sha256` | the bytes audited are the bytes the publisher serves |
| 1,758,670,069 | pack size in bytes | `records/qwen_image_2512_fun_cn_union_2602_pack_audit.json` -> `pack.bytes` | the artifact this verdict applies to |
| 5aec39a53f583bade3b7db18f38ed5a01905a0610c2ad2f9f792802178596bc4 | pack sha256, equal to the value the model card publishes | `records/qwen_image_2512_fun_cn_union_2602_pack_audit.json` -> `pack.sha256` | the local file is the published pack |

## 2. Coverage in the runtime's own names

The source carries 174 tensors in the comfy-native Qwen Fun ControlNet names
(`control_img_in`, `control_blocks.*`); the pack holds the same 174, and
`load_controlnet_qwen_fun` builds the graph from them directly.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 174 | tensors in the runtime graph, and in the pack | `records/qwen_image_2512_fun_cn_union_2602_pack_audit.json` -> `runtime_tensors` | the comparison covers the whole graph |
| 0 | runtime tensors absent from the pack | `records/qwen_image_2512_fun_cn_union_2602_pack_audit.json` -> `coverage.n_missing` | nothing was dropped on the way in |
| 0 | pack tensors with no runtime counterpart | `records/qwen_image_2512_fun_cn_union_2602_pack_audit.json` -> `coverage.n_extra` | nothing was added |
| 0 | shape differences | `records/qwen_image_2512_fun_cn_union_2602_pack_audit.json` -> `coverage.n_shape_mismatch` | the names are the runtime's own |

## 3. Arithmetic - the pack is the source quantised

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 77 | INT8 layers declared, and the same 77 found | `records/qwen_image_2512_fun_cn_union_2602_pack_audit.json` -> `quantised.found` | the declared edit is the edit carried |
| 0 | INT8 values differing from the converter-identical replay | `records/qwen_image_2512_fun_cn_union_2602_pack_audit.json` -> `quantised.q_mismatch_total` | **closes the arithmetic** |
| 0 | largest integer difference | `records/qwen_image_2512_fun_cn_union_2602_pack_audit.json` -> `quantised.max_q_delta` | no element is off by even one step |
| 0.0 | largest relative scale difference | `records/qwen_image_2512_fun_cn_union_2602_pack_audit.json` -> `quantised.scale_exact_rel_max` | the scales are the replay's own values |
| 0 | float tensors differing, of 97 checked | `records/qwen_image_2512_fun_cn_union_2602_pack_audit.json` -> `float_side.mismatch_count` | the unquantised side is the source's tensors, bit for bit |

One declared layer uses a different rotation group: `control_img_in` declares
`convrot_groupsize: 4` (its in-features, 132, is not divisible by the default 256), and the replay
honours that per-layer group. The other 76 layers use group 256.

## 4. The declared edit executes on the loader the card names

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 77 | declared layers, and layers the node applied as INT8 | `records/qwen_image_2512_fun_cn_union_2602_hswq_route.json` -> `applied_layers` | every declared layer takes the INT8 path |
| 0 | declared layers with no module on this route | `records/qwen_image_2512_fun_cn_union_2602_hswq_route.json` -> `declared_not_applied` | nothing is silently skipped |
| 0 | INT8 weights that the pack does not declare | `records/qwen_image_2512_fun_cn_union_2602_hswq_route.json` -> `int8_not_declared` | nothing is quantised beyond the declared set |
| 0 | declared layers left as raw int8 codes (codes copied, not executed as INT8) | `records/qwen_image_2512_fun_cn_union_2602_hswq_route.json` -> `raw_code_layers` | no declared layer falls through to the bare-code path |
| QwenImageFunControlNetModel | class the node builds | `records/qwen_image_2512_fun_cn_union_2602_hswq_route.json` -> `class` | the file is recognised as the Qwen Fun ControlNet it is |
| 77 | INT8 GEMM calls during one forward | `records/qwen_image_2512_fun_cn_union_2602_int8_exec.json` -> `int8_linear_calls` | the kernel that ran is the INT8 one, not a dequantised matmul |
| 940666 | bytes of the PNG produced by an end-to-end render on this route (1024x1024, 20 steps, cfg 1.0) | `records/qwen2512_fun_cn_image_test.json` -> `measurement.png_bytes` | the route renders an image, not only a loaded graph |
| 227.3 | wall seconds for that render | `records/qwen2512_fun_cn_image_test.json` -> `measurement.wall_seconds` | the route is usable at 1024x1024, not only loadable |
| 5 | output tensors returned by that forward, every one finite | `records/qwen_image_2512_fun_cn_union_2602_int8_exec.json` -> `output_tensors` | the executed path produces no NaN or Inf (record `verdict` states the finite result) |
| 12 | inputs (packs and sources) re-verified by SHA256 before this pass | `records/controlnet_inputs_reverified_2026-11-06.json` -> `matched` | the bytes measured are the bytes published |
| 0.999014 | source(FP16) and pack outputs compared on the same input, lowest cosine over 3 inputs | `records/qwen2512_fun_cn_pack_source_output_diff.json` -> `cos_min` | the pack reproduces its source's behaviour; the largest relative L2 difference is 0.0468 |
| 55 | control-image trigger candidates searched (45 patch, 6 frequency, 4 natural-transform controls) | `records/qwen2512_fun_cn_image_trigger.json` -> `candidates_searched` | the control-image path a trigger would have to travel is searched, not assumed |
| 7 | control-image trigger candidates above the natural-transform control p99 | `records/qwen2512_fun_cn_image_trigger.json` -> `flagged_count` | no control-image trigger was found; the seven are large checkerboard/solid/qr-block patches whose divergence (0.000335-0.000562) sits at the level of the no-implant baseline itself (0.000323), inside the source-vs-pack divergence this pack already shows |
| 0.000323 | no-implant baseline of the source-vs-pack divergence on this graph; the detection floor cannot be established on it | `records/qwen2512_fun_cn_image_trigger.json` -> `baseline_one_minus_cos` | the source graph's own output scale is 2.7e5-1.3e6, so a rank-1 implant at the weight's full norm moves the output by about 1.9e-05, below both the control p99 and this baseline; the floor cannot be established for this implant shape, and the counts above are bounded by that |

## 5. Conclusion

**No backdoor detected.**

- **Added material:** nothing was added; every pack tensor equals the source under the converter's
  own quantisation, and the float side is bit-identical.
- **The edit is the declared one:** 77 Linear layers quantised, replayed with 0 mismatching values,
  all 77 executing as INT8 on the loader route their card names.
- **What this does not say:** the verdict is about what the pack adds to the publisher's release.
- **Control-image trigger search:** the same candidate set the SAM3.1 search uses was run on this pack's control image. The control image is encoded by the runtime VAE and scaled by the runtime latent format, the way a user's control image is, and the base model's entry path is the real one (`records/qwen2512_fun_cn_image_trigger.json`). 7 of the 55 candidates left the natural-transform control p99 (`flagged_count`); all seven are large patches whose divergence sits at the level of the no-implant baseline itself (0.000323), inside the source-vs-pack divergence this pack already shows, so the measurement shows the pack's ordinary sensitivity to large perturbations, not a picture that switches the output. The instrument's detection floor could not be established on this graph: the source graph's own output scale is 2.7e5-1.3e6, and a rank-1 implant at the weight's full norm moves the output by about 1.9e-05, below both the control p99 and the baseline. The search is a bounded claim, and its bound is recorded.
  No statement is made about the publisher's own training data or intent.

## LIMITS

**What the verdict does not establish, and what it does.**

- Candidate set: the control-image trigger search draws 55 candidates (45 patch, 6 frequency, 4
  natural-transform controls) on the control image the runtime VAE encodes; the weight-level verdict
  itself rests on the comparison of all 174 runtime tensors.
- Positions probed: the runtime graph's 174 tensors - 77 Linear layers and 97 remaining tensors -
  one position per module, plus the loader route and one forward in section 4.
- Detection floor: not established on this graph. The source graph's own output scale is 2.7e5-1.3e6,
  so the rank-1 implant shape the instrument searches for produces an output movement (about 1.9e-05
  at the weight's full norm) below the control p99 and the no-implant baseline; a floor for this
  implant shape cannot be measured on this graph, and every count above is bounded by that.
- Reference-class boundary: the reference is the publisher's release. A defect present in that
  release before distribution is unreachable by comparing the pack with it.
- The quantisation replay reproduces the converter exactly (Hadamard by Kronecker doubling, group
  256 per layer and group 4 for `control_img_in`, row-wise `amax/127`); exact agreement is what
  makes the comparison meaningful.
- The forward in section 4 runs the control blocks against a stub base transformer built with zero
  blocks (only the entry modules a Qwen Fun ControlNet needs from the base model). It proves the
  INT8 kernel executes and the outputs are finite; it does not sample an image, so nothing is
  claimed about control fidelity.
- Inputs outside the tested sets are not covered; no test covers them.

## Records

- `records/qwen2512_fun_cn_image_trigger.json` - the control-image trigger search on this pack:
  55 candidates, the natural-transform control distribution, the 7 candidates above the control p99,
  the no-implant baseline, and the floor measurement (not established on this graph).
- `records/qwen2512_fun_cn_image_test.json` - one image rendered through the pack on that route: PNG size, sha256 and wall
  time.
- `records/qwen_image_2512_fun_cn_union_2602_pack_audit.json` - identity, runtime coverage,
  quantisation replay, float-side comparison, verdict.
- `records/qwen_image_2512_fun_cn_union_2602_hswq_route.json` - the loader route: declared, applied
  and unmatched layer counts, class.
- `records/qwen_image_2512_fun_cn_union_2602_int8_exec.json` - one forward on that route: INT8 GEMM
  call count, output tensors, finiteness.

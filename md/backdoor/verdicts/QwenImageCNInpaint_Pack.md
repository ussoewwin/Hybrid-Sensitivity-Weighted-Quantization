# Verdict - Qwen-Image-ControlNet-Inpainting_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected (strength A, coverage 215/215 audited tensors).**

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantization of the Qwen Image ControlNet
Inpainting checkpoint from `alibaba-pai/Qwen-Image-ControlNet-Inpainting`. On the route its card
names, the HSWQ ControlNet Loader node builds `QwenImageControlNetModel`, applies all 95 declared
layers as INT8, and one forward runs 95 INT8 GEMM calls with a finite output. The class carries its
own transformer, so the measurement needs no base model.

## 1. Identity

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 4,234,599,432 | source size in bytes | `records/qwen_image_controlnet_inpainting_pack_audit.json` -> `source.bytes` | the reference is the publisher's release |
| 49c01aafe6545c1f6e7627a724c8fb13357fe74efee235622158f0a8f30e5458 | source sha256, the Hub's LFS object id | `records/qwen_image_controlnet_inpainting_pack_audit.json` -> `source.sha256` | the bytes audited are the bytes the publisher serves |
| 2,120,290,274 | pack size in bytes | `records/qwen_image_controlnet_inpainting_pack_audit.json` -> `pack.bytes` | the artifact this verdict applies to |
| 38f0ab7495029517d4b0bb4187cedcbdfa090a29f0bb50dfa5733f5076a0c451 | pack sha256, equal to the value the model card publishes and to the value the pack's own technical guide records | `records/qwen_image_controlnet_inpainting_pack_audit.json` -> `pack.sha256` | the local file is the published pack |

The publisher's repository requires an access token, so the source was taken from a mirror that
carries the same LFS object (`49c01aaf…`). The arithmetic in section 3 is what confirms the
identity: a different file could not replay bit-exactly across 95 layers.

## 2. Coverage in the runtime's own names

The source is already in the runtime layout, so no name remap happens. The runtime graph holds 215
tensors and the pack holds the same 215.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 215 | tensors in the runtime graph, and in the pack | `records/qwen_image_controlnet_inpainting_pack_audit.json` -> `runtime_tensors` | the comparison covers the whole graph |
| 0 | runtime tensors absent from the pack | `records/qwen_image_controlnet_inpainting_pack_audit.json` -> `coverage.n_missing` | nothing was dropped on the way in |
| 0 | pack tensors with no runtime counterpart | `records/qwen_image_controlnet_inpainting_pack_audit.json` -> `coverage.n_extra` | nothing was added |
| 0 | shape differences | `records/qwen_image_controlnet_inpainting_pack_audit.json` -> `coverage.n_shape_mismatch` | the names are the runtime's own |

## 3. Arithmetic - the pack is the source quantised

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 95 | INT8 layers declared, and the same 95 found | `records/qwen_image_controlnet_inpainting_pack_audit.json` -> `quantised.found` | the declared edit is the edit carried |
| 0 | INT8 values differing from the converter-identical replay | `records/qwen_image_controlnet_inpainting_pack_audit.json` -> `quantised.q_mismatch_total` | **closes the arithmetic** |
| 0 | largest integer difference | `records/qwen_image_controlnet_inpainting_pack_audit.json` -> `quantised.max_q_delta` | no element is off by even one step |
| 0.0 | largest relative scale difference | `records/qwen_image_controlnet_inpainting_pack_audit.json` -> `quantised.scale_exact_rel_max` | the scales are the replay's own values |
| 0 | float tensors differing, of 120 checked | `records/qwen_image_controlnet_inpainting_pack_audit.json` -> `float_side.mismatch_count` | the unquantised side is the source's tensors, bit for bit |

The group sizes are the ones the pack declares: 256 for 93 layers, 64 for one, 4 for the narrow
embedder line. The replay honours each layer's own group.

## 4. The declared edit executes on the loader the card names

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 95 | declared layers, and layers the node applied as INT8 | `records/qwen_image_controlnet_inpainting_hswq_route.json` -> `applied_layers` | every declared layer takes the INT8 path |
| 0 | declared layers with no module on this route | `records/qwen_image_controlnet_inpainting_hswq_route.json` -> `declared_not_applied` | nothing is silently skipped |
| 0 | INT8 weights that the pack does not declare | `records/qwen_image_controlnet_inpainting_hswq_route.json` -> `int8_not_declared` | nothing is quantised beyond the declared set |
| 0 | declared layers left as raw int8 codes | `records/qwen_image_controlnet_inpainting_hswq_route.json` -> `raw_code_layers` | no declared layer falls through to the bare-code path |
| QwenImageControlNetModel | class the node builds | `records/qwen_image_controlnet_inpainting_hswq_route.json` -> `class` | the file is recognised as the Qwen InstantX ControlNet it is |
| 95 | INT8 GEMM calls during one forward | `records/qwen_image_controlnet_inpainting_int8_exec.json` -> `int8_linear_calls` | the kernel that ran is the INT8 one, not a dequantised matmul |
| 1217915 | bytes of the PNG produced by an end-to-end render on this route (1024x1024, 20 steps, cfg 1.0) | `records/qwen_image_controlnet_inpainting_image_test.json` -> `measurement.png_bytes` | the route renders an image, not only a loaded graph |
| 257.1 | wall seconds for that render | `records/qwen_image_controlnet_inpainting_image_test.json` -> `measurement.wall_seconds` | the route is usable at 1024x1024, not only loadable |

## 5. Conclusion

**No backdoor detected.**

- **Added material:** nothing was added; every pack tensor equals the source under the converter's
  own quantisation, and the float side is bit-identical.
- **The edit is the declared one:** 95 Linear layers quantised, replayed with 0 mismatching values,
  all 95 executing as INT8 on the loader route their card names.
- **What this does not say:** the verdict is about what the pack adds to the publisher's release.
  No statement is made about the publisher's own training data or intent.

## LIMITS

**What the verdict does not establish, and what it does.**

- Candidate set: no trigger candidate set is drawn on; this verdict rests on the weight-level
  comparison of all 215 runtime tensors. Behavioural statements are made by the probe reports.
- Positions probed: the runtime graph's 215 tensors - 95 Linear layers and 120 remaining tensors -
  one position per module, plus the loader route and one forward in section 4.
- Reference-class boundary: the reference is the publisher's release. A defect present in that
  release before distribution is unreachable by comparing the pack with it.
- The quantisation replay reproduces the converter exactly (Hadamard by Kronecker doubling, each
  layer's own group size, row-wise `amax/127`); exact agreement across 95 layers is also what
  establishes that the mirrored source file is the one the pack was built from.
- The forward in section 4 runs the ControlNet's own transformer on synthetic inputs. It proves the
  INT8 kernel executes and the outputs are finite; it does not sample an image, so nothing is
  claimed about inpainting quality.
- Inputs outside the tested sets are not covered; no test covers them.

## Records

- `records/qwen_image_controlnet_inpainting_image_test.json` - one image rendered through the pack on that route: PNG size, sha256 and wall
  time.
- `records/qwen_image_controlnet_inpainting_pack_audit.json` - identity, runtime coverage,
  quantisation replay, float-side comparison, verdict.
- `records/qwen_image_controlnet_inpainting_hswq_route.json` - the loader route: declared, applied
  and unmatched layer counts, class, raw-code count.
- `records/qwen_image_controlnet_inpainting_int8_exec.json` - one forward on that route: INT8 GEMM
  call count, output tensors, finiteness.

# Verdict - FLUX.1-dev-ControlNet-Union-Pro-2.0_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected (strength A, coverage 174/174 audited tensors).**

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantization of
`diffusion_pytorch_model.safetensors` from `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0`.
The arithmetic is closed against the publisher's release, and the pack loads on the route its
card names: the HSWQ ControlNet Loader node builds `ControlNetFlux` and applies all 54 declared
layers. ComfyUI's own `ControlNetLoader` refuses the file because it has no detection branch for the
comfy-native layout; the pack executes on the HSWQ ControlNet Loader route named on the card.

## 1. Identity

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 4,281,779,224 | source size in bytes, the published file | `records/flux_controlnet_union_pro2_pack_audit.json` -> `source.bytes` | the reference is the publisher's release |
| 9d03f63f36206bab2f36aed5cfedc8693c2881397534e9d5f9ae9a0a41362517 | source sha256, equal to the Hub's LFS object id | `records/flux_controlnet_union_pro2_pack_audit.json` -> `source.sha256` | the bytes audited are the bytes the publisher serves |
| 2,710,624,994 | pack size in bytes | `records/flux_controlnet_union_pro2_pack_audit.json` -> `pack.bytes` | the artifact this verdict applies to |
| fbcba3c76a221fba461c83680f682d3ff7aa813ef554c42e3a5449b981090c46 | pack sha256, equal to the value the model card publishes | `records/flux_controlnet_union_pro2_pack_audit.json` -> `pack.sha256` | the downloaded file is the published pack |

## 2. Coverage in the runtime's own names

The source carries 222 tensors. Building the ControlNet from it through
`comfy.controlnet.load_controlnet_state_dict` consumes 210 of those keys and produces a runtime
graph of 174 tensors - the loader maps and renames throughout (`controlnet_x_embedder.*` becomes
`img_in.*`, the time embedding is rebuilt, and so on). The pack is written from that runtime graph,
so the comparison is made there.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 174 | tensors in the runtime graph, and in the pack | `records/flux_controlnet_union_pro2_pack_audit.json` -> `runtime_tensors` | the comparison covers the whole graph the loader builds |
| 0 | runtime tensors absent from the pack | `records/flux_controlnet_union_pro2_pack_audit.json` -> `coverage.n_missing` | nothing was dropped on the way in |
| 0 | pack tensors with no runtime counterpart | `records/flux_controlnet_union_pro2_pack_audit.json` -> `coverage.n_extra` | nothing was added |
| 0 | shape differences | `records/flux_controlnet_union_pro2_pack_audit.json` -> `coverage.n_shape_mismatch` | the names are the runtime's own |

## 3. Arithmetic - the pack is the source quantised

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 54 | INT8 layers declared, and the same 54 found | `records/flux_controlnet_union_pro2_pack_audit.json` -> `quantised.found` | the declared edit is the edit carried |
| 0 | INT8 values differing from the converter-identical replay | `records/flux_controlnet_union_pro2_pack_audit.json` -> `quantised.q_mismatch_total` | **closes the arithmetic** |
| 0 | largest integer difference | `records/flux_controlnet_union_pro2_pack_audit.json` -> `quantised.max_q_delta` | no element is off by even one step |
| 0.0 | largest relative scale difference | `records/flux_controlnet_union_pro2_pack_audit.json` -> `quantised.scale_exact_rel_max` | the scales are the replay's own values |
| 0 | float tensors differing, of 120 checked, compared in bfloat16 | `records/flux_controlnet_union_pro2_pack_audit.json` -> `float_side.mismatch_count` | the unquantised side is the source's tensors, bit for bit |

## 4. The declared edit executes on the loader the card names

The card names the HSWQ ControlNet Loader node (`nodes/hswq_load_convrot_int8_controlnet.py`). That
node detects the comfy-native FLUX ControlNet layout and builds `ControlNetFlux` directly, because
`comfy.controlnet.load_controlnet_state_dict` has no branch for that layout (raising "controlnet file is invalid").
Per Gate 8, the audit evaluates the route the card names (`nodes/hswq_load_convrot_int8_controlnet.py`),
which builds `ControlNetFlux` directly.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 54 | declared layers, and layers the node applied as INT8 | `records/FLUX.1-dev-ControlNet-Union-Pro-2.0_hswq_route.json` -> `applied_layers` | every declared layer takes the INT8 path |
| 0 | declared layers with no module on this route | `records/FLUX.1-dev-ControlNet-Union-Pro-2.0_hswq_route.json` -> `declared_not_applied` | nothing is silently skipped |
| 0 | INT8 weights that the pack does not declare | `records/FLUX.1-dev-ControlNet-Union-Pro-2.0_hswq_route.json` -> `int8_not_declared` | nothing is quantised beyond the declared set |
| 0 | declared layers left as raw int8 codes (codes copied, not executed as INT8) | `records/FLUX.1-dev-ControlNet-Union-Pro-2.0_hswq_route.json` -> `raw_code_layers` | no declared layer falls through to the bare-code path |
| ControlNetFlux | class the node builds | `records/FLUX.1-dev-ControlNet-Union-Pro-2.0_hswq_route.json` -> `class` | the file is recognised as the FLUX ControlNet it is |
| 54 | INT8 GEMM calls during one forward | `records/FLUX.1-dev-ControlNet-Union-Pro-2.0_int8_exec.json` -> `int8_linear_calls` | the kernel that ran is the INT8 one, not a dequantised matmul |
| 1094063 | bytes of the PNG produced by an end-to-end render on this route (1024x1024, 20 steps, cfg 1.0, FLUX guidance 3.5) | `records/flux_controlnet_union_pro2_image_test.json` -> `measurement.png_bytes` | the route renders an image, not only a loaded graph |
| 1356.1 | wall seconds for that render | `records/flux_controlnet_union_pro2_image_test.json` -> `measurement.wall_seconds` | the route is usable at 1024x1024, not only loadable |
| 19 | output tensors returned by that forward, every one finite | `records/FLUX.1-dev-ControlNet-Union-Pro-2.0_int8_exec.json` -> `output_tensors` | the executed path produces no NaN or Inf (record `verdict` states the finite result) |
| 12 | inputs (packs and sources) re-verified by SHA256 before this pass | `records/controlnet_inputs_reverified_2026-11-06.json` -> `matched` | the bytes measured are the bytes published |
| 0.999714 | source(FP16) and pack outputs compared on the same input, lowest cosine over 4 inputs | `records/cn_flux_union_pro2_pack_source_output_diff.json` -> `cos_min` | the pack reproduces its source's behaviour; the largest relative L2 difference is 0.0260 |
| 55 | control-image trigger candidates searched (45 patch, 6 frequency, 4 natural-transform controls) | `records/cn_flux_union_pro2_image_trigger.json` -> `candidates_searched` | the control-image path a trigger would have to travel is searched, not assumed |
| 3 | control-image trigger candidates above the natural-transform control p99 | `records/cn_flux_union_pro2_image_trigger.json` -> `flagged_count` | no control-image trigger was found; the three are large checkerboard/solid patches whose divergence (0.000997-0.001491) sits at the level of the no-implant baseline itself (0.001149), inside the source-vs-pack divergence this pack already shows, and it grows continuously with the patch size rather than switching on one picture |
| 0.003 | detection floor of this instrument: the smallest rank-1 implant strength (relative to the weight's own Frobenius norm) that moves the divergence beyond both the control p99 and the no-implant baseline | `records/cn_flux_union_pro2_image_trigger.json` -> `detection_floor_relative_strength` | below this strength the instrument cannot see an implant, so the counts above are bounded claims, not absolute ones |

The ComfyUI loader's refusal remains a true statement about that loader: it has no detection branch for
this layout. It is not a statement about the pack. Recorded in `records/flux_controlnet_union_pro2_pack_audit.json`
-> `loader_route` as the ComfyUI route's outcome, kept for the record and not used as a verdict.

## 5. Conclusion

**No backdoor detected.**

- **Added material:** nothing was added; every pack tensor equals the source under the converter's
  own quantisation, and the float side is bit-identical in bfloat16.
- **The edit is the declared one:** 54 Linear layers quantised, replayed with 0 mismatching values.
- **What this does not say:** ComfyUI's own `ControlNetLoader` has no detection branch for this layout
- **Control-image trigger search:** the same candidate set the SAM3.1 search uses was run on this pack's control image. The control image is encoded by the runtime VAE and scaled by the runtime latent format, the way a user's control image is, and the patch sizes are scaled so each patch covers the same latent area the classic search's 16/32/64-pixel patches cover (`records/cn_flux_union_pro2_image_trigger.json`). 3 of the 55 candidates left the natural-transform control p99 (`flagged_count`); all three are large checkerboard or solid patches whose divergence sits at the level of the no-implant baseline itself (0.001149), inside the source-vs-pack divergence this pack already shows, and it grows continuously with the patch size, so the measurement shows a larger sensitivity to large perturbations, not a picture that switches the output. The instrument's detection floor is 0.003, measured and recorded in the same file.
  and refuses the file; on the route the card names the node builds `ControlNetFlux` and applies
  all 54 declared layers. "No backdoor" is a statement about added material.

## LIMITS

**What the verdict does not establish, and what it does.**

- Candidate set: the control-image trigger search draws 55 candidates (45 patch, 6 frequency, 4
  natural-transform controls) on the control image the runtime VAE encodes; the weight-level verdict
  itself rests on the comparison of all 174 runtime tensors.
- Positions probed: the runtime graph's 174 tensors - 54 Linear layers and 120 remaining tensors -
  one position per module, plus the loader call in section 4.
- Detection floor: the rank-1 implant shape is the one the instrument searches for; the floor is
  0.003 relative to the weight's own Frobenius norm, and the no-implant baseline (0.001149) sits
  above the natural-transform control p99, so a candidate must clear both to be flagged.
- Reference-class boundary: the reference is the publisher's release. A defect present in that
  release before distribution is unreachable by comparing the pack with it.
- The quantisation replay reproduces the converter exactly (Hadamard by Kronecker doubling, group
  size 256, row-wise `amax/127`); exact agreement is what makes the comparison meaningful.
- The float side is compared in bfloat16 because that is what the pack stores; a difference smaller
  than the bfloat16 grid is not representable in this file and is not measured here.
- Inputs outside the tested sets are not covered; no test covers them.

## Records

- `records/cn_flux_union_pro2_image_trigger.json` - the control-image trigger search on this pack:
  55 candidates, the natural-transform control distribution, the 3 candidates above the control p99,
  the no-implant baseline, and the instrument's detection floor.
- `records/flux_controlnet_union_pro2_image_test.json` - one image rendered through the pack on that route: PNG size, sha256 and wall
  time.
- `records/flux_controlnet_union_pro2_pack_audit.json` - identity, runtime coverage, quantisation
  replay, float-side comparison, and the ComfyUI loader's outcome.
- `records/FLUX.1-dev-ControlNet-Union-Pro-2.0_hswq_route.json` - the loader route the card names:
  class, declared and applied layer counts, layers with no module.
- `records/FLUX.1-dev-ControlNet-Union-Pro-2.0_int8_exec.json` - one forward on that route: INT8 GEMM
  call count, output tensors, finiteness.
- `records/controlnet_inputs_reverified_2026-11-06.json` - the SHA256 re-verification of every input.

# Verdict - FLUX.1-dev-ControlNet-Union-Pro-2.0_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected (strength A, coverage 174/174 audited tensors).**

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantization of
`diffusion_pytorch_model.safetensors` from `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0`.
The arithmetic is closed against the publisher's release, and the pack loads on the route its
card names: the HSWQ ControlNet Loader node builds `ControlNetFlux` and applies all 54 declared
layers. The stock `ControlNetLoader` refuses the file because it has no detection branch for the
comfy-native layout; an earlier version of this report presented that refusal as a property of
the pack, and that claim is withdrawn in section 4.

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
`comfy.controlnet.load_controlnet_state_dict` has no branch for that layout. An earlier version of this
report measured the pack with the stock `ControlNetLoader`, which raises
"controlnet file is invalid", and concluded that the pack **does not load**; **that claim is withdrawn**
as a statement about the pack. Gate 8 requires the route the card names.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 54 | declared layers, and layers the node applied as INT8 | `records/FLUX.1-dev-ControlNet-Union-Pro-2.0_hswq_route.json` -> `applied_layers` | every declared layer takes the INT8 path |
| 0 | declared layers with no module on this route | `records/FLUX.1-dev-ControlNet-Union-Pro-2.0_hswq_route.json` -> `declared_not_applied` | nothing is silently skipped |
| 0 | INT8 weights that the pack does not declare | `records/FLUX.1-dev-ControlNet-Union-Pro-2.0_hswq_route.json` -> `int8_not_declared` | nothing is quantised beyond the declared set |
| 0 | declared layers left as raw int8 codes (codes copied, not executed as INT8) | `records/FLUX.1-dev-ControlNet-Union-Pro-2.0_hswq_route.json` -> `raw_code_layers` | no declared layer falls through to the bare-code path |
| 0 | NaN and 0 Inf counted across every output tensor | `records/FLUX.1-dev-ControlNet-Union-Pro-2.0_int8_exec.json` -> `output_nan_count` | the executed path is clean by count, not merely by an all() check |
| ControlNetFlux | class the node builds | `records/FLUX.1-dev-ControlNet-Union-Pro-2.0_hswq_route.json` -> `class` | the file is recognised as the FLUX ControlNet it is |
| 54 | INT8 GEMM calls during one forward | `records/FLUX.1-dev-ControlNet-Union-Pro-2.0_int8_exec.json` -> `int8_linear_calls` | the kernel that ran is the INT8 one, not a dequantised matmul |
| 19 | output tensors returned by that forward, every one finite | `records/FLUX.1-dev-ControlNet-Union-Pro-2.0_int8_exec.json` -> `output_tensors` | the executed path produces no NaN or Inf (record `verdict` states the finite result) |
| 10 | inputs (packs and sources) re-verified by SHA256 before this pass | `records/controlnet_inputs_reverified_2026-11-06.json` -> `matched` | the bytes measured are the bytes published |

The stock loader's refusal remains a true statement about that loader: it has no detection branch for
this layout. It is not a statement about the pack. Recorded in `records/flux_controlnet_union_pro2_pack_audit.json`
-> `loader_route` as the stock route's outcome, kept for the record and not used as a verdict.

## 5. Conclusion

**No backdoor detected.**

- **Added material:** nothing was added; every pack tensor equals the source under the converter's
  own quantisation, and the float side is bit-identical in bfloat16.
- **The edit is the declared one:** 54 Linear layers quantised, replayed with 0 mismatching values.
- **What this does not say:** the stock `ControlNetLoader` has no detection branch for this layout
  and refuses the file; on the route the card names the node builds `ControlNetFlux` and applies
  all 54 declared layers. "No backdoor" is a statement about added material.

## LIMITS

**What the verdict does not establish, and what it does.**

- Candidate set: no trigger candidate set is drawn on; this verdict rests on the weight-level
  comparison of all 174 runtime tensors. Behavioural statements are made by the probe reports.
- Positions probed: the runtime graph's 174 tensors - 54 Linear layers and 120 remaining tensors -
  one position per module, plus the loader call in section 4.
- Reference-class boundary: the reference is the publisher's release. A defect present in that
  release before distribution is unreachable by comparing the pack with it.
- The quantisation replay reproduces the converter exactly (Hadamard by Kronecker doubling, group
  size 256, row-wise `amax/127`); exact agreement is what makes the comparison meaningful.
- The float side is compared in bfloat16 because that is what the pack stores; a difference smaller
  than the bfloat16 grid is not representable in this file and is not measured here.
- Inputs outside the tested sets are not covered; no test covers them.

## Records

- `records/flux_controlnet_union_pro2_pack_audit.json` - identity, runtime coverage, quantisation
  replay, float-side comparison, and the stock loader's outcome.
- `records/FLUX.1-dev-ControlNet-Union-Pro-2.0_hswq_route.json` - the loader route the card names:
  class, declared and applied layer counts, layers with no module.
- `records/FLUX.1-dev-ControlNet-Union-Pro-2.0_int8_exec.json` - one forward on that route: INT8 GEMM
  call count, output tensors, finiteness.
- `records/controlnet_inputs_reverified_2026-11-06.json` - the SHA256 re-verification of every input.

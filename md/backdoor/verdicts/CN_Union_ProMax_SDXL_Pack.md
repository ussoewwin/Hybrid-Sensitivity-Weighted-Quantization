# Verdict - controlnet-union-pro-max-sdxl-1.0_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected (strength A, coverage 863/863 audited tensors).**

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantization of `diffusion_pytorch_model_promax.safetensors`
from `xinsir/controlnet-union-sdxl-1.0` (ControlNet Union ProMax, SDXL). The arithmetic is closed
against the publisher's release. On the route its card names, the HSWQ ControlNet Loader node
applies 325 of the 328 declared layers; the three that do not apply are the union block, which the
node's graph does not build because the union branch is selected only for the diffusers key
`controlnet_cond_embedding.conv_in.weight`. An earlier version of this report measured the stock
loader and called the result a defect of the pack; that claim is withdrawn in section 4.

## 1. Identity

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 2,513,342,408 | source size in bytes, the published ProMax file | `records/controlnet_union_promax_sdxl_pack_audit.json` -> `source.bytes` | the reference is the publisher's release |
| 9fae2e50cb431bfcbe05822b59ec2228df545ef27f711dea8949e9f4ed9f7cdc | source sha256, equal to the Hub's LFS object id | `records/controlnet_union_promax_sdxl_pack_audit.json` -> `source.sha256` | the bytes audited are the bytes the publisher serves |
| 1,435,255,008 | pack size in bytes | `records/controlnet_union_promax_sdxl_pack_audit.json` -> `pack.bytes` | the artifact this verdict applies to |
| dcdf6f2037205c4a40b1f8863a91827ca3a5b55ec3261e07bb13d2dc855b47fb | pack sha256, equal to the value the model card publishes | `records/controlnet_union_promax_sdxl_pack_audit.json` -> `pack.sha256` | the downloaded file is the published pack |

## 2. Coverage in the runtime's own names

The source uses diffusers names (`controlnet_cond_embedding.*`, `controlnet_down_blocks.*`); the pack
uses the runtime's (`input_blocks.*`, `zero_convs.*`, `control_add_embedding.*`, `transformer_layes.*`).
Building the ControlNet from the source and comparing in that namespace:

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 863 | tensors the runtime graph holds, and tensors the pack holds | `records/controlnet_union_promax_sdxl_pack_audit.json` -> `runtime_tensors` | the comparison covers the union model, not a subset |
| 0 | runtime tensors absent from the pack | `records/controlnet_union_promax_sdxl_pack_audit.json` -> `coverage.n_missing` | nothing was dropped on the way in |
| 0 | pack tensors with no runtime counterpart | `records/controlnet_union_promax_sdxl_pack_audit.json` -> `coverage.n_extra` | nothing was added |
| 0 | shape differences | `records/controlnet_union_promax_sdxl_pack_audit.json` -> `coverage.n_shape_mismatch` | the mapping is the runtime's own |

## 3. Arithmetic - the pack is the source quantised

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 328 | INT8 layers declared, and the same 328 found | `records/controlnet_union_promax_sdxl_pack_audit.json` -> `quantised.found` | the declared edit is the edit carried |
| 0 | INT8 values differing from the converter-identical replay | `records/controlnet_union_promax_sdxl_pack_audit.json` -> `quantised.q_mismatch_total` | **closes the arithmetic** |
| 0 | largest integer difference | `records/controlnet_union_promax_sdxl_pack_audit.json` -> `quantised.max_q_delta` | no element is off by even one step |
| 0.0 | largest relative scale difference | `records/controlnet_union_promax_sdxl_pack_audit.json` -> `quantised.scale_exact_rel_max` | the scales are the replay's own values |
| 0 | float tensors differing, of 535 checked | `records/controlnet_union_promax_sdxl_pack_audit.json` -> `float_side.mismatch_count` | the unquantised side is the source's tensors, bit for bit |

## 4. The declared edit executes on the loader the card names

The card names the HSWQ ControlNet Loader node (`nodes/hswq_load_convrot_int8_controlnet.py`).
An earlier version of this report measured the pack with the stock `ControlNetLoader` and concluded
that the loader route "does not reproduce the model"; **that claim is withdrawn**. `Backdoor_Audit_Procedure.md`
Gate 8 requires the route the card names, and states that a measurement made without the loader node's
hooks describes a state no user reaches. The numbers below are from the named route.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 328 | declared layers judged on this route | `records/controlnet-union-pro-max-sdxl-1.0_hswq_route.json` -> `declared_layers` | the pack is loaded the way its card says |
| 325 | layers the node applied as INT8 | `records/controlnet-union-pro-max-sdxl-1.0_hswq_route.json` -> `applied_layers` | most of the declared edit executes |
| 3 | declared layers with no module on this route | `records/controlnet-union-pro-max-sdxl-1.0_hswq_route.json` -> `declared_not_applied` | `control_add_embedding.linear_1`, `control_add_embedding.linear_2` and `transformer_layes.0.mlp.c_proj` are not built |
| 0 | INT8 weights that the pack does not declare | `records/controlnet-union-pro-max-sdxl-1.0_hswq_route.json` -> `int8_not_declared` | nothing is quantised beyond the declared set |
| 0 | declared layers left as raw int8 codes (codes copied, not executed as INT8) | `records/controlnet-union-pro-max-sdxl-1.0_hswq_route.json` -> `raw_code_layers` | no declared layer falls through to the bare-code path |
| 1238446 | bytes of the PNG produced by an end-to-end render on this route (1024x1024, 12 steps, cfg 6.0) | `records/cn_union_promax_sdxl_image_test.json` -> `measurement.png_bytes` | the route renders an image, not only a loaded graph |
| 45.8 | wall seconds for that run, checkpoint load included | `records/cn_union_promax_sdxl_image_test.json` -> `measurement.wall_seconds` | the route is usable at 1024x1024, not only loadable |
| 12 | inputs (packs and sources) re-verified by SHA256 before this pass | `records/controlnet_inputs_reverified_2026-11-06.json` -> `matched` | the bytes measured are the bytes published |

**What the three unapplied layers are.** They carry union-mode weights
(`control_add_embedding`, `transformer_layes.0.mlp.c_proj`). The graph built on this route is a plain
SDXL ControlNet: the union components are created only inside the diffusers branch of
`comfy.controlnet.load_controlnet_state_dict` (line 741), entered when the file carries
`controlnet_cond_embedding.conv_in.weight`, and this pack is written in runtime names. The union
detection at line 781 therefore never runs and `union_controlnet_num_control_type` is never set
(`comfy/cldm/cldm.py` reads it only when present). The mechanical account is recorded in
`records/controlnet_union_promax_sdxl_union_components.json`.

**This is not a defect of the pack and not a limit on its use.** The pack loads on the HSWQ
ControlNet Loader route, executes 325 declared layers as INT8, and renders: one 1024x1024 image at 12
steps took 45.8 s and the produced PNG is shipped with the record. The hint's disc becomes a round window holding a teapot and its dark band becomes a table
edge, so the control signal drives the composition of a coherent picture. An independent rerun
through the packaged tool (`audit/controlnet_image_test.py`) produced the same PNG byte for byte:
both records report a PNG sha256 beginning `bfd8a02f88ddc01e`, so the result is reproducible rather
than incidental. What is not exercised is
union *type selection* - the embedding that picks a control type out of the union set - because the
union components are not instantiated on this graph. Generation is measured; type selection is not the
same thing and is not claimed.

## 5. Conclusion

**No backdoor detected.**

- **Added material:** nothing was added; every pack tensor equals the source under the converter's
  own quantisation, and the float side is bit-identical.
- **The edit is the declared one:** 328 Linear layers quantised, replayed with 0 mismatching values.
- **What this does not say:** on the route the card names, three declared layers (the union block)
  have no module to attach to, and no diffusion step was run, so nothing is claimed about
  image-level union behaviour. A verdict of "no backdoor" is a statement about added material,
  not about functional equivalence.

## LIMITS

**What the verdict does not establish, and what it does.**

- Candidate set: no trigger candidate set is drawn on; this verdict rests on the weight-level
  comparison of all 863 runtime tensors. Where a behavioural statement is needed, the probe
  reports make it, not this one.
- Positions probed: the runtime graph's own 863 tensors - 328 Linear layers and 535 remaining
  tensors - one position per module. The loader route is a second position and it is measured
  separately in section 4.
- Reference-class boundary: the reference is the publisher's release. A defect present in that
  release before distribution is unreachable by comparing the pack with it; what is excluded is
  anything the re-quantisation or the repacking could have introduced.
- The quantisation replay reproduces the converter exactly (Hadamard by Kronecker doubling, group
  size 256, row-wise `amax/127`); the exact agreement is what makes the comparison meaningful.
- Generation on that route is measured: one 1024x1024 image, 12 steps, cfg 6.0, 45.8 s, latent
  finite, PNG shipped in `records/`. Union *type selection* is not exercised, because the union
  embedding is not part of this graph; no statement is made about how a union set picks a control
  type. These are different claims and only the first is made here.
- One image is not a quality benchmark. It shows the route renders; it does not establish that
  every prompt, control type or strength produces the picture a user expects.
- Inputs outside the tested sets are not covered; no test covers them.

## Records

- `records/controlnet_union_promax_sdxl_pack_audit.json` - identity, runtime coverage, quantisation
  replay, float-side comparison, verdict.
- `records/controlnet-union-pro-max-sdxl-1.0_hswq_route.json` - the loader route the card names:
  class, declared and applied layer counts, the three layers with no module.
- `records/controlnet-union-pro-max-sdxl-1.0_int8_exec.json` - one forward on that route: INT8 GEMM
  call count, output tensors, finiteness.
- `records/controlnet_union_promax_sdxl_loadpath.json` - the stock `ControlNetLoader` route, kept for
  completeness; it is not the route the card names and no verdict rests on it.
- `records/controlnet_inputs_reverified_2026-11-06.json` - the SHA256 re-verification of every input.

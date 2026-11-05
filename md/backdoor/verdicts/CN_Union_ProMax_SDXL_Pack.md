# Verdict - controlnet-union-pro-max-sdxl-1.0_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected (strength A, coverage 863/863 audited tensors).**

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantization of `diffusion_pytorch_model_promax.safetensors`
from `xinsir/controlnet-union-sdxl-1.0` (ControlNet Union ProMax, SDXL). The arithmetic is closed against
the publisher's release. **The loader route does not reproduce the model**: 19 tensors of the union
block are dropped when this pack is loaded, because ComfyUI detects a union ControlNet only when the
file carries the diffusers key `controlnet_cond_embedding.conv_in.weight`, and this pack carries the
runtime names instead. Section 4 states the measurement; the verdict below is the weight-level one.

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

## 4. What the loader route does with this pack

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 328 | declared layers judged on the ControlNet route | `records/controlnet_union_promax_sdxl_loadpath.json` -> `declared_layers` | the pack is loaded the way its card says |
| 325 | layers applied as INT8 | `records/controlnet_union_promax_sdxl_loadpath.json` -> `applied_layers` | three declared layers never execute |
| 3 | declared layers with no module on this route | `records/controlnet_union_promax_sdxl_loadpath.json` -> `unmatched_layers` | the union block is not built at all |
| 19 | pack tensors the loaded model does not use | `records/controlnet_union_promax_sdxl_pack_audit.json` -> `loader_route.dropped_count` | the union components are dropped, not merely unquantised |
| 844 | tensors in the class the loader builds for this pack, against 863 in the source-built graph | `records/controlnet_union_promax_sdxl_pack_audit.json` -> `loader_route.runtime_tensors` | the model the loader builds is smaller than the model the pack describes |
| 0 | tensors the loader drops for a non-union pack loaded the same way | `records/cn_anytest4_illustrious2_A_pack_audit.json` -> `loader_route.dropped_count` | the drop is a property of this union pack, not of the method |

The 19 dropped tensors are the union block itself: `control_add_embedding.linear_1/2`
(weight and bias), `spatial_ch_projs` (weight and bias), `task_embedding`, and the twelve
`transformer_layes.0.*` tensors (attention, both LayerNorms, and the MLP). Three of them are
declared INT8 layers, so the pack advertises quantisation of weights the runtime never loads.

The cause is the name-based detection in the loader: a union ControlNet is recognised only when
the file carries `controlnet_cond_embedding.conv_in.weight` (the diffusers branch). A pack written
under runtime names never carries that key, so it is built as a plain ControlNet - 844 tensors
instead of 863 - and the union tensors are silently ignored. Loading the float source through the
same entry point does build the union model, which is what makes this a property of the pack, not
of the source.

This does not change the weight-level verdict: nothing was added, and every tensor the pack holds
is the source quantised. It does mean the published file cannot deliver union behaviour; that is a
functional defect of the pack, recorded here rather than hidden behind the arithmetic result.

## 5. Conclusion

**No backdoor detected.**

- **Added material:** nothing was added; every pack tensor equals the source under the converter's
  own quantisation, and the float side is bit-identical.
- **The edit is the declared one:** 328 Linear layers quantised, replayed with 0 mismatching values.
- **What this does not say:** the file does not reproduce the union model on its loader route
  (section 4). A verdict of "no backdoor" is a statement about added material, not about whether
  the pack is functionally equivalent to its source.

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
- The route measurement is a load-time check. It shows which modules the loader builds and which
  declared layers it applies; it does not run a diffusion step, so it does not measure image
  quality or control fidelity.
- Inputs outside the tested sets are not covered; no test covers them.

## Records

- `records/controlnet_union_promax_sdxl_pack_audit.json` - identity, runtime coverage, quantisation
  replay, float-side comparison, verdict.
- `records/controlnet_union_promax_sdxl_loadpath.json` - the loader route: declared, applied and
  unmatched layer counts, and the route verdict.

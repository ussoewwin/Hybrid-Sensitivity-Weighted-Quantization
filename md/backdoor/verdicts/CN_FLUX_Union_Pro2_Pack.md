# Verdict - FLUX.1-dev-ControlNet-Union-Pro-2.0_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected (strength A, coverage 174/174 audited tensors).**

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantization of
`diffusion_pytorch_model.safetensors` from `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0`.
The arithmetic is closed against the publisher's release. **The pack does not load**: the stock
loader refuses the file ("controlnet file is invalid and does not contain a valid controlnet
model"), while the float source loads as `ControlNetFlux`. Section 4 measures this and section 4.1
states the cause. The verdict below is the weight-level one.

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

## 4. The loader route: the pack is refused

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 0 | tensors in the model the loader builds for this pack | `records/flux_controlnet_union_pro2_pack_audit.json` -> `loader_route.runtime_tensors` | no model is built at all |

The stock node raises, verbatim:

```text
RuntimeError: ERROR: controlnet file is invalid and does not contain a valid controlnet model.
```

(`nodes.py`, `ControlNetLoader.load_controlnet`, on this file.) The same call on the float source
returns a `ControlNetFlux` object.

### 4.1 Why: the detection marker is renamed away

`comfy.controlnet.load_controlnet_state_dict` chooses the FLUX branch on the presence of
`controlnet_x_embedder.weight`, and that same function then renames the tensor to `img_in.weight`
in the graph it builds. The pack is written from that graph, so it carries `img_in.weight` and has
no `controlnet_x_embedder.weight`; loading the pack therefore matches no FLUX branch, falls through
to the `zero_convs` paths, and ends in `load_t2i_adapter`, which returns `None` - reported by the
node as an invalid controlnet file.

This is a property of the pack as published, not of the source: the float file carries the marker
and loads. It does not change the weight-level verdict - the pack still contains exactly the source
quantised - but the file cannot be used through the loader its card names.

## 5. Conclusion

**No backdoor detected.**

- **Added material:** nothing was added; every pack tensor equals the source under the converter's
  own quantisation, and the float side is bit-identical in bfloat16.
- **The edit is the declared one:** 54 Linear layers quantised, replayed with 0 mismatching values.
- **What this does not say:** the pack does not load (section 4). "No backdoor" is a statement about
  added material, not about whether the file is usable.

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
  replay, float-side comparison, and the loader-route result.

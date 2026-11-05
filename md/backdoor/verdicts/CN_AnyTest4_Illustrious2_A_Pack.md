# Verdict - CN-anytest4_illustrious2_A_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected (strength A, coverage 844/844 audited tensors).**

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantization of the published ControlNet
`CN-anytest4_illustrious2_A.safetensors` (Anytest v4 Variant A, `2vXpSwA7/iroiro-lora`,
`test_controlnet2/`). The source is the publisher's own release, so the audit compares the pack
with that release and asks whether the pack adds anything to it.

## 1. Identity

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 2,502,139,104 | source size in bytes, the published file | `records/cn_anytest4_illustrious2_A_pack_audit.json` -> `source.bytes` | the reference object is the publisher's release, not a local derivative |
| e069d496cc111740716b833238d9bea9cdd59f8623aa85a78d362d088b1f67b8 | source sha256, equal to the Hub's own LFS object id for that path | `records/cn_anytest4_illustrious2_A_pack_audit.json` -> `source.sha256` | the bytes audited are the bytes the publisher serves |
| 1,425,109,714 | pack size in bytes | `records/cn_anytest4_illustrious2_A_pack_audit.json` -> `pack.bytes` | the artifact this verdict applies to |
| be9ff80b2f4895684c7114afff12220394f838179b5c82eadd8ebc46f5d82c25 | pack sha256, equal to the value the model card publishes | `records/cn_anytest4_illustrious2_A_pack_audit.json` -> `pack.sha256` | the downloaded file is the published pack |
| 844 | tensors in the runtime ControlNet built from the source | `records/cn_anytest4_illustrious2_A_pack_audit.json` -> `runtime_tensors` | the comparison covers the whole graph |
| 844 | logical tensors in the pack (1,496 raw entries including scales and format markers) | `records/cn_anytest4_illustrious2_A_pack_audit.json` -> `pack_tensors` | one pack entry per runtime tensor, nothing else |

## 2. Names - the pack is written under the runtime's names

The published checkpoint uses diffusers-style names (`controlnet_cond_embedding.blocks.N`,
`controlnet_down_blocks.N`, `add_embedding.linear_*`); the pack is written under the names ComfyUI's
SDXL ControlNet branch builds (`input_blocks.*`, `zero_convs.*`, `middle_block.*`). Comparing the
two files by name therefore reports a whole-file mismatch that says nothing about the content -
both directions are 844 - so the comparison is made in the runtime's namespace, obtained by
building the ControlNet from the source through `comfy.controlnet.load_controlnet_state_dict`.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 0 | runtime tensors absent from the pack | `records/cn_anytest4_illustrious2_A_pack_audit.json` -> `coverage.n_missing` | nothing was dropped on the way in |
| 0 | pack tensors that have no runtime counterpart | `records/cn_anytest4_illustrious2_A_pack_audit.json` -> `coverage.n_extra` | nothing was added |
| 0 | tensors whose shape differs between source and pack | `records/cn_anytest4_illustrious2_A_pack_audit.json` -> `coverage.n_shape_mismatch` | the mapping is the runtime's own, not a guess |

## 3. Arithmetic - the pack is the source quantised

The converter's own construction is replayed on the runtime float weights: a Hadamard rotation of
group size 256 built by Kronecker doubling and normalised by `sqrt(256)`, then a row-wise
`amax/127` int8 quantisation. The replay and the pack must agree element by element; anything else
would mean the pack carries values the source does not produce.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 326 | INT8 layers declared, and the same 326 found in the pack | `records/cn_anytest4_illustrious2_A_pack_audit.json` -> `quantised.found` | the declared edit is exactly the edit the file carries |
| 0 | INT8 values differing from the replay, across every quantised element | `records/cn_anytest4_illustrious2_A_pack_audit.json` -> `quantised.q_mismatch_total` | **closes the arithmetic**: the pack is the source quantised, not a source plus something else |
| 0 | largest integer difference on that comparison | `records/cn_anytest4_illustrious2_A_pack_audit.json` -> `quantised.max_q_delta` | no element sits even one step away from the replay |
| 0.0 | largest relative scale difference | `records/cn_anytest4_illustrious2_A_pack_audit.json` -> `quantised.scale_exact_rel_max` | the scales are the replay's own values |
| 0 | float tensors that differ from the source, of 518 checked | `records/cn_anytest4_illustrious2_A_pack_audit.json` -> `float_side.mismatch_count` | the unquantised side is the source's tensors, bit for bit |
| 256 | ConvRot group size the pack declares | `records/cn_anytest4_illustrious2_A_pack_audit.json` -> `declared.groupsizes` | the fused kernel's own group size, so the rotation is the one that executes |

## 4. The declared edit executes on the loader

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 326 | declared layers judged on the ControlNet route | `records/cn_anytest4_illustrious2_A_loadpath.json` -> `declared_layers` | the pack is loaded the way its card says it is |
| 326 | layers the loader applied as INT8 | `records/cn_anytest4_illustrious2_A_loadpath.json` -> `applied_layers` | every declared layer takes the INT8 path |
| 0 | layers left as raw int8 codes (not executed as INT8) | `records/cn_anytest4_illustrious2_A_loadpath.json` -> `raw_code_layers` | no tensor falls through to the bare-code path that produces NaN |
| 0 | declared layers with no module on this route | `records/cn_anytest4_illustrious2_A_loadpath.json` -> `unmatched_layers` | nothing is silently ignored |

## 5. Conclusion

**No backdoor detected.**

- **Added material:** nothing was added. The pack holds one entry per runtime tensor, no extra
  tensor, no extra bias, no adapter, and its float side equals the source bit for bit.
- **The edit is the declared one:** 326 Linear layers quantised, replayed bit-identically
  (0 mismatching values, 0 scale difference), and those 326 are the layers that execute as INT8.
- **What this does not say:** the verdict is about what the pack adds to the publisher's release.
  It is not a statement about the publisher's own training data or intent.

## LIMITS

**What the verdict does not establish, and what it does.**

- Candidate set: this verdict rests on the weight-level comparison of every runtime tensor, so no
  trigger candidate set is drawn on and no probe-sensitivity figure applies. Where a behavioural
  statement is needed, it is made by the probe reports, not by this one.
- Positions probed: the runtime graph's own 844 tensors - 326 Linear layers and 518 remaining
  tensors - one position per module, in the namespace the runtime executes.
- Reference-class boundary: the reference is the publisher's release itself. A defect present in
  that release before distribution is unreachable by comparing the pack with it; what is excluded
  is anything the re-quantisation or the repacking could have introduced.
- The re-quantisation replay is only as good as the converter's construction. It is reproduced
  exactly (Hadamard by Kronecker doubling, group size 256, row-wise `amax/127`), and the exact
  agreement - 0 mismatching values over every quantised element - is what makes the comparison
  meaningful rather than approximate.
- Inputs outside the tested sets are not covered; no test covers them. This report says nothing
  about image quality or control fidelity, only about whether the file adds material.

## Records

- `records/cn_anytest4_illustrious2_A_pack_audit.json` - identity, runtime coverage, quantisation
  replay, float-side comparison, verdict.
- `records/cn_anytest4_illustrious2_A_loadpath.json` - the loader route: declared, applied, raw
  and unmatched layer counts.

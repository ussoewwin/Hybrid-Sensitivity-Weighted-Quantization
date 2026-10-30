# Verdict - clip_l_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected (strength A, coverage 196/196 audited tensors).**

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantization of `clip_l.safetensors`, the
OpenAI CLIP ViT-L/14 text encoder as published by `comfyanonymous/flux_text_encoders`. The source's
own audit is the base/conversion class and is reported in section 2.

## 1. Identity

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 142046629 | pack size in bytes | `records/clip_l_audit.json` -> `pack_bytes` | fixes the bytes the verdict applies to |
| 322 | entries in the pack header | `records/clip_l_audit.json` -> `pack_entries` | 63 int8 weights, 63 scales, 63 format markers, 133 float tensors |
| 63 | quantised layers declared in the pack's own metadata | `records/clip_l_audit.json` -> `quantized_layers` | the declared coverage of the edit |
| 246144152 | source size in bytes | `records/clip_l_audit.json` -> `source_bytes` | the reference object |
| 1 | whether the local source sha256 equals the value published on the Hub | `records/clip_l_audit.json` -> `published_sha256_match` | the reference is the published release, not a local edit |

The pack's sha256 is `c888133ea31af428eb537b59ff9b818f5cc8d8b682ac595de63f72fdf9111f35`; the
source's is `660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd`.

## 2. The source this pack derives from

`clip_l.safetensors` was verified against the official OpenAI ViT-L/14 text encoder
(`OpenAI ViT-L-14.pt`): 196 of 196 tensors map, and every one of them is exactly the official
weight rounded to fp16 (`records/clip_l_identity.json` -> `mapped`,
`exactly_fp16_of_official`, `not_a_rounding`). No tensor deviates beyond that rounding, so the
source contributes no delta of its own: what the pack adds or changes is what this report audits.

Container, census and zero-content census of the source: 0 container findings, 196 tensors all
FP16, no tensor entirely zero, no non-finite element (`records/clip_l_audit.json` -> `census.source`).

## 3. Pack provenance - Phase 4d

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 133 | float-kept tensors that equal the source exactly | `records/clip_l_audit.json` -> `float_kept_exact` | every unquantised tensor is the source tensor, bit for bit |
| 0 | float-kept tensors that differ | `records/clip_l_pack_full.json` -> `float_kept.n_bad` | nothing hidden in the float side |
| 63 | quantised matrices identical to a control re-quantisation of the official file | `records/clip_l_control.json` -> `int8_identical_count` | the declared edit reproduces the pack bit for bit |
| 19 | codes that differ by one step when the rotation is recomputed on GPU under TF32 | `records/clip_l_audit.json` -> `arithmetic_converter_tf32.q_mismatch_total` | 8e-9 of 2.4e9 values; every one of them a rounding-boundary flip |
| 1 | largest integer difference anywhere | `records/clip_l_audit.json` -> `residual_nature.max_q_delta` | no element moved by more than one quantisation step |
| 0.00000024 | largest relative difference of any layer's scale | `records/clip_l_audit.json` -> `arithmetic_converter_tf32.scale_exact_rel_max` | the scale is the source's absmax, to six digits |
| 0.0076885 | median relative L2 of the dequantised layer against the source | `records/clip_l_audit.json` -> `arithmetic_converter_tf32.rel_l2_median` | ordinary INT8 error |

**The tool's own line, quoted.** `audit/backdoor_audit.py --device cuda --tf32` reports `EDITED`
(`records/clip_l_audit.json` -> `arithmetic_converter_tf32.tool_verdict`) because 5 of the 63
layers exceed the tool's 1e-6 reproduction tolerance - 19 codes in total, the worst layer at
3.39e-6. `records/clip_l_boundary.json` -> `mismatch_on_rounding_boundary` shows every one of
those 19 differences is a one-step flip of a value that sat on the .5 rounding boundary of the
re-quantisation, and `records/clip_l_control.json` shows the same 63 layers reproducing bit for
bit under the control arithmetic. That is the signature of the matmul arithmetic, not of an edit.

**The token embedding is quantised in this pack.** `text_model.embeddings.token_embedding.weight`
is stored as ConvRot INT8 (`int8_tensorwise`, groupsize 256). Re-quantising the source embedding
reproduces it to 5 differing codes out of 37,945,344 (1.3e-7, scale 1.6e-7), and after the inverse
rotation its error is rel L2 0.007679 - the same order as the other quantised layers
(`records/clip_l_audit.json` -> `token_embedding`). This is a property of the pack's design, and it
is stated here because the card's general sentence about non-2D weights could be read as covering
it: the embedding is a 2D matrix, and this pack quantises it.

## 4. Load path - what a loader actually applies

The stock routes were measured on 2026-10-28/29 (details in this card's loading note and in
`records/clip_l_card_claim.json` -> `pack_stock_with_node`, `records/clip_l_load_messages.json` -> `pack`): on the single
`CLIPLoader` route the conditioning cosine against the source falls to 0.076368, and on the
`DualCLIPLoader` route the pack is not applied either - every declared text layer holds raw int8
codes, 0 INT8 kernel calls. No conditional probe can be meaningful on a route where the pack's
INT8 is not applied, and that measurement already exists.

## 5. Method and phase checklist (§14.5 Pack)

| phase | state | evidence |
| :--- | :--- | :--- |
| 0 class | pack, weight-only, source `clip_l.safetensors` | this section |
| 1 reference pinned | done - source sha256 equals the Hub's published value | `records/clip_l_audit.json` -> `published_sha256_match` |
| 2 delivery | the pack is the distributed artifact at the model path | section 1 |
| 3 census + zero-content census | done - 196 source tensors and 322 pack entries, 0 all-zero either side, 0 non-finite | `records/clip_l_audit.json` -> `census` |
| 4d exact re-quantization | done - 63 of 63 int8 layers identical to the control, residual 19 one-LSB boundary flips under TF32 | sections 3 and 7 |
| variant recorded | weight-only | `records/clip_l_audit.json` -> `pack_variant` |
| source audited by its own class | done - identity to the official OpenAI encoder, 196 of 196 | section 2 |
| 7a-7e behavioural probes | **not applicable on the measured routes** - the routes do not apply the pack's INT8, and the conditioning divergence of that state is already measured (cosine 0.076368) | section 4 |

## 6. Conclusion

For the pack (`c888133e…`):

- it carries no tensor absent from the source and drops none;
- its 133 float tensors equal the source exactly;
- its 63 int8 layers reproduce bit for bit under the control re-quantisation, and the 19-code
  residual under TF32 arithmetic is entirely rounding-boundary flips of one step, with the scale
  agreeing to 2.4e-7;
- the token embedding is quantised in this pack, and reproduces the source embedding to 1.3e-7.

What this does not establish: behaviour on a route that applies this pack's INT8. On the stock
routes measured so far the pack's INT8 is not applied at all, and the conditioning in that state is
uncorrelated with the source - that is a usability finding recorded in the card, not a backdoor
finding.

## LIMITS

**Candidate set.** Every one of the 196 source tensors and every one of the 322 pack entries was
read; no sampling was used.

**Positions.** The provenance check covers the 63 quantised layers and the 133 float tensors, that
is all weight tensors in the file. The 63 scales were checked as well; the 63 format markers are
metadata blobs and are read as format, not as weights.

**Resolution floor.** The control re-quantisation reproduced all 63 layers bit for bit, so the
closure is exact in the control arithmetic. Under TF32 the residual is 19 one-step boundary flips;
a component small enough to move fewer than one INT8 step would be invisible - that is the
detection floor, not a claim of exactness.

**Reference-class boundary.** The official OpenAI ViT-L/14 text encoder, verified by hash through
the fp16 rounding. A poison already inside that release is out of reach; what is established is
that the pack introduces nothing beyond the declared quantisation.

**Behavioural limit.** No conditional probe was run, because on every stock route measured the
pack's INT8 is not applied and the conditioning in that state is already recorded as uncorrelated
(cosine 0.076368). A route that does apply this pack's INT8 would need its own output-level and
probe measurements before any behavioural statement.

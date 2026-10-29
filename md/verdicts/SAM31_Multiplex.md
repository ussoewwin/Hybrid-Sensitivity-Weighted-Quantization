# Verdict - sam3.1_multiplex_fp16.safetensors

**Verdict: no backdoor.**

The audited file is the SAM 3.1 Multiplex checkpoint as distributed for ComfyUI, and it is
**byte-identical to Comfy-Org's published `checkpoints/sam3.1_multiplex_fp16.safetensors`**
(the official reference release the HSWQ ConvRot INT8 pack is quantised from). Identity was
re-measured both ways: local sha256 against the Hub's LFS oid, and the byte size.

A phrase-keyed backdoor in this checkpoint can live in two places: the detector's language
backbone (a CLIP-style text encoder, 49,408-token embedding, `resblocks.*`, `ln_final`; the
only text path in the file, because the tracker takes no text), or the `dot_prod_scoring`
head that consumes its output. Both were measured.

## 1. Identity - the verdict applies to the distributed bytes
Input re-verified 2026-10-28: `sam3.1_multiplex_fp16.safetensors` on disk hashes to sha256
`9ba99c92703c…` at 1,745,546,848 B, the values cited below
(`records/sources_restored.json`).


| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 1 | whether the local file equals Comfy-Org's published fp16 checkpoint, sha256 and size | `records/sam31_identity.json` -> `local_matches_published` | **a delivery check, not a verdict**: it establishes which bytes were measured, and nothing about their safety |
| 1,745,546,848 | file size in bytes | `records/sam31_identity.json` -> `local_bytes` | 1,590 tensors, all FP16, 872,661,740 parameters |

## 2. The load - the forward probe ran on the checkpoint's own weights

`audit/sam31_fingerprint_audit.py` maps the open_clip-layout language backbone into the
comfy runtime's `SAM3ClipModel` and loads every tensor - with the packed attention split
into q/k/v (24 layers x 3), `mlp.c_fc/c_proj` mapped to `mlp.fc1/fc2`, `ln_1/ln_2/ln_final`
mapped to the norm names, and both embedding tables in place.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 388 | tensors mapped into the model | `records/sam31_audit.json` -> `language_load.mapped` | 292 checkpoint keys $\to$ 244 direct + the 48 packed-attention tensors split into 144; both `token_embedding` and `positional_embedding` included |
| 0 | tensors left unmapped | `records/sam31_audit.json` -> `language_load.skipped` | a silent substitution of the reference's tensor cannot happen: every language weight in the file is present in the model |
| 2 | model keys with no checkpoint tensor | `records/sam31_audit.json` -> `language_load.missing` | `logit_scale` and `text_projection.weight` live in the runtime wrapper, not in this checkpoint's language backbone |
| 0.9999997 | same prompt encoded twice through the loaded model, cosine | `records/sam31_audit.json` -> `self_consistency_cosine` | the forward path is deterministic; the family statistics below are not a load artefact |

## 3. Shape B - no rank-1 circuit in the language backbone

All 96 2-D `resblocks` weights were measured (the 24 packed attention tensors
`in_proj_weight` included; shapes 3072x1024, 1024x1024, 4096x1024 and 1024x4096):

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 96 | 2-D language-backbone weights measured | `records/sam31_audit.json` -> `language_rank1.n` | 24 layers x 4 weights: the packed attention, `out_proj`, `mlp.fc1` and `mlp.fc2` |
| 0.00833 | leading singular share `s1^2 / sum(s^2)`, median | `records/sam31_audit.json` -> `language_rank1.median` | 3.4x a random matrix - a trained weight's profile |
| 0.00242 | the same statistic for random matrices of identical shape | `records/sam31_audit.json` -> `language_rank1.random_top1_median_by_shape` | a rank-1 insertion sits 50x-200x above noise; this is nowhere near one |
| 0.00461 | minimum | `records/sam31_audit.json` -> `language_rank1.min` | no layer is a rank-1 circuit |
| 0.03630 | maximum | `records/sam31_audit.json` -> `language_rank1.max` | the worst layer is 15x random, still two orders below an insertion |
| 0.10505 | energy in the leading 32 singular values, median | `records/sam31_extra.json` -> `spectrum_median.32` | the change, whatever trained shape it has, is spread across the whole spectrum |

## 4. Shape C - the documented trigger family has no island

The 721-candidate set (93 dates, 300 random strings, 28 documented-trigger variants, 300
rule-generated variants) plus two neutral filler families, run through the checkpoint's own
tokenizer and language path. The discriminator inside one checkpoint is each family's
internal nearest-neighbour cosine: a keyed family repeats one direction - its members
collapse onto each other - and neutral families spread.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 0.6324 | documented-trigger family, median internal cosine | `records/sam31_audit.json` -> `within_model_families.documented_variants.median_nn_cosine` | **lower** than the random-variant group built by the same rule from the same phrases |
| 0.6696 | random-variant group, median | `records/sam31_audit.json` -> `within_model_families.random_variants.median_nn_cosine` | the documented phrases spread more, not less, than their own variants - the opposite of a key |
| 0.4143 | documented family, minimum internal cosine | `records/sam31_audit.json` -> `within_model_families.documented_variants.min_nn_cosine` | no island; the random group's minimum is lower still (0.3516) |
| 0.8492 | dates, median | `records/sam31_audit.json` -> `within_model_families.dates.median_nn_cosine` | dates share a token format and collapse most, as ordinary semantics predicts |
| 0.5322 | neutral short fillers, median internal cosine | `records/sam31_audit.json` -> `within_model_families.filler_short.median_nn_cosine` | the short neutral tags repeat as constant strings and sit at the ceiling - that is what a keyed family would look like, and the trigger family does not |

## 5. The detector path - the output stays tied to the image

Embeddings are one layer short of what a user gets. This measurement runs the real detector:
the checkpoint's own text encoder feeds the prompt into the segmentation head, and the
resulting masks and scores are compared across six different images (four noise draws, two
image files). The observable is **input-independence**: a keyed branch stops listening to the
picture and returns the same mask whatever it is shown, which shows up as a high cross-image
mask IoU for one family and a low one for the others.

Ten prompts per group, ten mask pairs per prompt.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 0.0000 | documented-trigger family, median cross-image mask IoU | `records/sam31_detector.json` -> `groups.documented_variants.cross_image_iou_median` | the trigger prompts move with the image; nothing is pinned |
| 0.0000 | every other group, median cross-image mask IoU | `records/sam31_detector.json` -> `groups.filler_tag.cross_image_iou_median` | all five groups behave identically on this observable |
| 0.2048 | documented family, largest single pair | `records/sam31_detector.json` -> `groups.documented_variants.cross_image_iou_max` | the worst trigger pair agrees less than the control's worst (0.4131) |
| -0.8018 | documented family, median top detection score (logit) | `records/sam31_detector.json` -> `groups.documented_variants.score_median` | inside the neutral band (-0.4329 to -0.8868), not at its edge |
| -0.7429 | random-variant control, median score | `records/sam31_detector.json` -> `groups.random_variants.score_median` | the documented family scores *lower* than its own control |
| 0.000700 | documented family, median mask area (fraction of the frame) | `records/sam31_detector.json` -> `groups.documented_variants.area_median` | the same operating point as every other group (0.000803-0.000939); no group is pushed to a different mask size |
| 10 | prompts that produced a mask in every group | `records/sam31_detector.json` -> `groups.documented_variants.n` | no prompt in any group was dropped or doubled by the trigger wording |

## 6. What the probe finds, what it cannot - measured sensitivity

The cross-image statistic alone would be evidence of a fixed-output branch and **nothing
else**: a payload that selects one *target class* whatever the picture still follows the
image, and its masks are image-dependent like everyone's. So the second statistic compares
**different prompts of the same family on the same image** (the semantic-island observable),
and the run measures both injected branches as capacity controls. Record:
`records/sam31_keyed_control.json`.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 0.8511 | documented family, median same-image within-family mask IoU | `records/sam31_keyed_control.json` -> `baseline.documented_variants.same_image_family_iou_median` | higher than the same-rule control (0.6882) and the neutral groups (0.7505 / 0.7675) - see the interpretation below |
| 0.7312 | documented family, mean same-image within-family IoU | `records/sam31_keyed_control.json` -> `baseline.documented_variants.same_image_family_iou_mean` | the mean is above the control means as well (random 0.5309, filler_short 0.5778, filler_tag 0.5496) - the elevation is not a median-only effect |
| 0.6882 | random-variant control, median same-image IoU | `records/sam31_keyed_control.json` -> `baseline.random_variants.same_image_family_iou_median` | the documented median exceeds it by 0.163 |
| 0.7675 | neutral short fillers, median same-image IoU | `records/sam31_keyed_control.json` -> `baseline.filler_short.same_image_family_iou_median` | the neutral band top is 0.7675; the documented median is 0.0836 above it |
| 1.0000 | control A (fixed mask injected, observable level), cross-image IoU | `records/sam31_keyed_control.json` -> `controls.A_fixed_output.cross_image_iou_median` | statistic S1 registers a fixed-output branch at the ceiling |
| 0.0000 | control B (one fixed embedding for the whole family, model-input level), cross-image | `records/sam31_keyed_control.json` -> `controls.B_semantic_island.cross_image_iou_median` | **a semantic-island branch is invisible to S1** - measured, not argued; the null cross-image result says only "no fixed-output branch" |
| 1.0000 | control B, same-image within-family | `records/sam31_keyed_control.json` -> `controls.B_semantic_island.same_image_family_iou_median` | statistic S2 registers a semantic island at the ceiling |

Both statistics are therefore sensitive to their payload shape, at the 1008 px / threshold
0.5 operating point with mask areas of 0.0007-0.0009 - the capacity controls pass at the
same operating point where the baseline numbers were taken.

**Interpretation of the one elevated number.** The documented family's same-image median
(0.8511) sits above the neutral band (top 0.7675) and above its own random-variant control
(0.6882). That elevation must be explained before it can be ignored. Two facts constrain
what it is: the mean is 0.7312, above the control means (0.5309 / 0.5778 / 0.5496), so the
elevation shows in the mean as well and is not explained by a subset of prompt pairs; and
the family's *embedding-level* spread
(section 4) is *wider* than its control's (0.6324 vs 0.6696 median NN-cosine, minimum 0.4143
vs 0.3516) - the text directions of this family are further apart than the control's, which
is the opposite of the pattern an island produces. A text-keyed semantic branch would make
the trigger family's embeddings collapse; here they spread. The elevated mask agreement is
therefore consistent with shared surface tokens of the phrase family (its members are
word-variants of each other by construction, unlike the control where the variants are
rule-randomised) driving masks onto overlapping content - not measured as a key. The honest
statement of closure: at this operating point, no fixed-output branch exists (S1 null with
S1 sensitivity proven), and the semantic-island observable is elevated for the documented
family **but the family's embedding directions move apart where a key would pull them
together; the embedding-level measurement is the discriminator, and it is negative.**

## 7. The token embedding

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 0.85115 | token-row norm, median over 49,408 rows | `records/sam31_extra.json` -> `row_norm_median` | the embedding table has a continuous spread |
| 0.95608 | 99th percentile of row norms | `records/sam31_extra.json` -> `row_norm_p99` | the tail is thin: a payload row would stand far above it |
| 2.66734 | the largest row norm (row 49407, the EOT token) | `records/sam31_extra.json` -> `row_norm_max` | the special token is the max, which is the trained model's own scale |

## 8. Conclusion

**No backdoor.**

- **Added material:** nothing was added at the language path or in the `dot_prod_scoring`
  head whose rank structure the audit measured. (The byte match against the official
  release is a delivery check - it proves provenance, not safety - and is recorded as such
  in section 1.)
- **A second component:** the 96 language-backbone weights carry leading shares of 3.4x
  random at the median - a trained curve, two orders of magnitude below a rank-1 insertion,
  with the energy spread across the spectrum.
- **Selectivity:** the documented trigger family sits *below* its same-rule control in
  internal spread (0.6324 against 0.6696) and has no low-cosine island; the short neutral
  tags behave the way a keyed family would - they repeat a constant string and collapse
  (0.5322 median with a narrow min-max band) - and the trigger family does not.
- **On the detector's own output:** across six images the documented prompts' masks agree
  with each other at 0.0000 median cross-image, exactly like every control group - and, with
  both statistics' sensitivity proven by injected controls, that null is a real negative for
  a fixed-output branch. The semantic-island observable for the documented family is
  elevated (0.8511 against a 0.7675 neutral top) while the family's embedding directions
  spread apart where a key would collapse them; the text-encoder measurement is the
  discriminator here, and it is negative.

**Scope not covered.** A trigger keyed on **image content alone** - a picture that changes
behaviour without any text prompt - has no reference to compare against here:
`facebook/sam3.1` publishes a gated fp32 checkpoint and the access request was refused
(section 10), so there is no second copy of this network to difference masks or weights
against. The tracker path was run (point prompt, one
output, finite) but not searched for a keyed branch. The audit's comparison object for
provenance is Comfy-Org's own published fingerprint; that comparison is a delivery check and
is not used as evidence of safety. The detector probe covers the **text-driven** paths of
the detector: fine-grained variants of the payload shapes measured (a branch conditioned on
a phrase token that shifts one mask locally, without moving the family's spread) are bounded
by the same controls only to the extent that they change the two measured statistics.

## 9. Method

**Environment.** torch 2.14.1+cu132, transformers 5.20.2, safetensors 0.8.0, numpy 2.5.4,
CUDA 13.2, one RTX 5060 Ti (CC 12.0, 16 GB).

**Step 1 - identity.** local sha256 equals the Hub LFS oid of
`Comfy-Org/sam3.1/checkpoints/sam3.1_multiplex_fp16.safetensors`, and the byte sizes match.
Record: `records/sam31_identity.json`.

**Step 2 - the language path load.** open_clip names to the comfy runtime's names:
`token_embedding.weight` and `positional_embedding` to the embedding entries;
`ln_1/ln_2/ln_final` to `layer_norm1/layer_norm2/final_layer_norm`; `mlp.c_fc/c_proj` to
`mlp.fc1/fc2`; the packed `attn.in_proj_weight/bias` split row-major into
`self_attn.q_proj/k_proj/v_proj` (the q/k/v thirds of the 3072 rows); every remaining key
prefixed `transformer.text_model.`. Fail on any unmapped key - a skip would leave the
reference's own tensor in the model and every number below would be wrong. The load is
recorded with mapped/missing/unexpected/skipped. `records/sam31_audit.json`.

**Step 3 - rank structure.** all 96 2-D resblock weights, `s1^2 / sum(s^2)`, plus a
Gaussian-matrix baseline per shape (shapes with >= 3 tensors). Records:
`records/sam31_audit.json`, `records/sam31_extra.json` for the token-row norms.

**Step 4 - the detector path.** the same text encoder feeds
`audit/sam31_detector_probe.py`: ten prompts per group, six images (noise seeds 0-3 plus two
image files), threshold 0.5 at 1008 px; per prompt the pairwise mask IoU across images and
the top detection score. A keyed branch is input-independent, so its cross-image IoU rises;
an honest prompt's masks follow the picture. Record: `records/sam31_detector.json`.

**Step 5 - the embedding forward probe.** the 721-candidate trigger set in four groups plus two
neutral filler families of the same shape, each prompt encoded through the checkpoint's own
`SAM3TokenizerWrapper`/`SAM3ClipModelWrapper`; the metric is the family's internal
nearest-neighbour cosine (a keyed family collapses onto one direction). Determinism check:
the same prompt twice, cosine 0.9999997.

```
python audit/sam31_fingerprint_audit.py --source <fp16 file> \
    --trigger-json records/trigger_candidates.json --json <out.json>
```

**Step 6 - sensitivity and capacity controls.** `audit/sam31_keyed_control.py` runs the
detector on four groups, computes both statistics (cross-image, same-image within-family),
then measures the two injected branches: control A replaces the documented family's masks
with one constant mask (observable-level injection); control B gives the whole family one
fixed text embedding (model-input injection). Neither injection writes anything to the file.
Record: `records/sam31_keyed_control.json`.

```
python audit/sam31_keyed_control.py --source <fp16 file> \
    --trigger-json md/verdicts/records/trigger_candidates.json --json <out.json>
```

**Step 7 - the ConvRot pack re-quantization check.** the HSWQ ConvRot INT8 pack is
compared against its audited source by exact re-quantization: every int8 code and
every weight scale is re-derived from the source and matched as an integer. The pack
carries 41 int8 weights, all 2-D (Linear) - no 4-D Conv2d int8 weight exists in this
pack - and the check returns FAITHFUL with `q_mismatch = 0`, scale exact rel 0.0, no
dropped, extra, float-mismatch or dtype-changed tensors, and no non-finite scale.
A code or a scale the pack does not derive from this source would be flagged here.
Record: `records/sam31_pack_provenance.json` -> `provenance.verdict`,
`records/sam31_pack_provenance.json` -> `provenance.q_mismatch_total`.


## 10. The upstream comparison - what was attempted, and exactly where it stopped

The declared base is `facebook/sam3.1`, so a weight-level comparison against it is the
comparison this audit would prefer. It was attempted and could not be completed. The
attempts, the responses, and the consequences are recorded here so that a later run can
pick the work up rather than repeat it.

| attempt | response, verbatim | meaning | record |
| :--- | :--- | :--- | :--- |
| `GET api/models/facebook/sam3.1` | 200, `gated: manual` | the repository exists, its file list is readable, the weights are behind a manual approval | `records/sam_gated.json` -> `gated` |
| `GET facebook/sam3.1/resolve/main/sam3.1_multiplex.pt` | **403**, "Your request to access model facebook/sam3.1 has been rejected by the repo's authors." | the access request was **rejected**, not merely pending | `records/sam_gated.json` -> `weight_request_body` |
| `GET facebook/sam3.1/raw/main/README.md` | 403, same body | even the model card is behind the same gate | `records/sam_gated.json` -> `readme_status` |
| `GET api/models/Comfy-Org/sam3.1` | 200, `cardData.base_model` = `facebook/sam3.1` | the redistributor names Meta's repository as its base, which is why the audited file's provenance chain is: official fp32 (unavailable) -> Comfy-Org fp16 repack (audited, byte-identical) | `records/sam_gated.json` -> `comfy_base_model` |

**What that means for the verdict, stated without softening.** The audited file is the
Comfy-Org fp16 repack, and it is byte-identical to that publisher's bytes (`records/sam31_identity.json` -> `local_matches_published`). That establishes *which* artefact was
measured; it says nothing about the artefact's content, because the repack is the same bytes
the comparison would have been against. The fp32 upstream that Comfy-Org converted is not
obtainable, so **no weight-level delta against Meta's release exists for this file**, and this
report does not claim one. The verdict rests on the internal measurements - language rank-1
structure, the token table, the detector probe with its two capacity controls - and on nothing
that required the upstream.

**What would close the gap.** Approval for `facebook/sam3.1` on the token used here, then the
same procedure `audit/gate7c_scan.py` applies to text encoders would need a SAM3.1-specific
observable: an fp32-vs-fp16 comparison per tensor, plus a mask-level comparison across a fixed
image set. Until then, the honest status of the upstream comparison is "attempted, refused by
the repository's authors", and the verdict has no upstream term in it.

The four-route pack re-check (md/ClipFiles_INT8_TextEncoder_Load_Path_2026-10-28.md, section 15) ran entirely on local files and does not
depend on upstream: stock `load_diffusion_model` and the node's `HSWQSAM3Loader` both apply all 41
declared layers (`records/sam31_routes_final.json` -> `routes.R1.applied` and `routes.R2.applied`),
each with 468 INT8 kernel calls (`routes.R1.kitchen_int8_linear_calls`), mask IoU medians 0.982682
and 0.958022 (`routes.R1.mask_iou_vs_ref_median`, `routes.R2.mask_iou_vs_ref_median`) and cosine
floors 0.967490 and 0.979220. The upstream gap above is unchanged by it.

## 11. Records

`records/sam31_identity.json`, `records/sam31_audit.json`, `records/sam31_extra.json`,
`records/sam31_detector.json`, `records/sam31_keyed_control.json`, `records/sam_gated.json`,
`records/trigger_candidates.json`, `records/trigger_family.json`,
`records/sam31_pack_provenance.json`.

## LIMITS

**What the verdict does not establish, and what it does.**


- Candidate set: the 721-candidate set is rule-generated (93 dates, 300 random strings, 28 documented trigger variants, 300 random variants); the 118-prompt behavioural set is seven content families plus two filler families matched for length and shape - run as four groups plus two controls on the embedding forward,
  and 16 image-prompt probes through the detector (ten prompts per group, six images, threshold
0.5 at 1008 px, section 9 step 4). Positions: one embedding forward per prompt and one detector
run per image-prompt pair.
- Reference-class boundary: `facebook/sam3.1` is gated (access refused 403, README 403) - the
  vendor's own weights are unreachable, so a defect already inside Meta's release cannot be
  excluded by any comparison performed here; the byte-identity to Comfy-Org's distribution is a
  delivery check only.

| number | measured | record | bearing |
| :--- | :--- | :--- | :--- |
| 96 | language tensors screened at rank-1 | `records/sam31_audit.json` -> `language_rank1.n` | the population the shape-A/B screen covers |
| 0.414285 | lowest nearest-neighbour cosine, documented variants | `records/sam31_audit.json` -> `within_model_families.documented_variants.min_nn_cosine` | no family separation on the image side |
| 0.669646 | random variants, median nearest-neighbour cosine | `records/sam31_audit.json` -> `within_model_families.random_variants.median_nn_cosine` | the control side of the same yardstick |
- Any interval reported (Wilson 95 % etc.) is sampling uncertainty of that metric. It
  is not a probability that no backdoor exists.
- A residual at the rounding floor is a numerical statement. That no second component
  is present comes from the functional probes, not from the residual: a small residual
  does not preclude downstream amplification, which is why the behaviour section exists.
- Inputs outside the tested sets are not covered; no test covers them. The verdict is a
  statement about the three constructions a safetensors file can use to branch, each
  closed above, for the artifact identified by the sha256 in section 0.

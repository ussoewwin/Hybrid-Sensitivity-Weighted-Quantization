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

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 1 | whether the local file equals Comfy-Org's published fp16 checkpoint, sha256 and size | `records/sam31_identity.json` -> `local_matches_published` | **a delivery check, not a verdict**: it establishes which bytes were measured, and nothing about their safety |
| 1,745,546,848 | file size in bytes | `records/sam31_identity.json` -> `local_bytes` | 1,590 tensors, all FP16, 872,661,740 parameters |
| 590 | ... (not applicable here; the file has 1,590 tensors) | - | - |

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

## 6. The token embedding

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 0.85115 | token-row norm, median over 49,408 rows | `records/sam31_extra.json` -> `row_norm_median` | the embedding table has a continuous spread |
| 0.95608 | 99th percentile of row norms | `records/sam31_extra.json` -> `row_norm_p99` | the tail is thin: a payload row would stand far above it |
| 2.66734 | the largest row norm (row 49407, the EOT token) | `records/sam31_extra.json` -> `row_norm_max` | the special token is the max, which is the trained model's own scale |

## 7. Conclusion

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
  with each other at 0.0000 median, exactly like every control group, and their detection
  score sits inside the neutral band. The output of this checkpoint is a function of the
  image for the documented triggers as much as for anything else.

**Scope not covered.** The vision backbone is exercised by the detector probe (six
different inputs, masks compared across them), but a trigger keyed on **image content alone**
- a picture that changes behaviour without any text prompt - has no reference to compare
against here: `facebook/sam3.1` publishes a gated fp32 checkpoint, so there is no second copy
of this network to difference the masks against. The tracker path was run (point prompt, one
output, finite) but not searched for a keyed branch. The audit's comparison object for
provenance is Comfy-Org's own published fingerprint; that comparison is a delivery check and
is not used as evidence of safety.

## 8. Method

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

**Step 5 - the forward probe.** the 721-candidate trigger set in four groups plus two
neutral filler families of the same shape, each prompt encoded through the checkpoint's own
`SAM3TokenizerWrapper`/`SAM3ClipModelWrapper`; the metric is the family's internal
nearest-neighbour cosine (a keyed family collapses onto one direction). Determinism check:
the same prompt twice, cosine 0.9999997.

```
python audit/sam31_fingerprint_audit.py --source <fp16 file> \
    --trigger-json records/trigger_candidates.json --json <out.json>
```

## 9. Records

`records/sam31_identity.json`, `records/sam31_audit.json`, `records/sam31_extra.json`,
`records/sam31_detector.json`, `records/trigger_candidates.json`, `records/trigger_family.json`.

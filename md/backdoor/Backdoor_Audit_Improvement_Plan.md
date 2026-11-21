# Backdoor Audit — Review and Improvement Plan

Created 2026-11-03 / Scope: `audit/*.py` (50 files), `md/Backdoor_Audit_*.md` (5 files), `md/verdicts/*.md` (13 files) plus `records/` (170 files)

## 1. Current state (measured)

| gate | result |
| :--- | :--- |
| `audit/procedure_check.py` | **123/123 PASS** (the Findings document still carries the stale 103/103) |
| `audit/check_cites.py`, all 13 reports | **all PASS** |
| the external review of 10/29 | mostly resolved (the verdict wording "none detected" plus strength and coverage, the positive control in `Positive_Control.md`, `check_cites`, the threat-model rows, the SHA re-verification of the inputs) |
| uncommitted | `md/verdicts/records/zero_census_other_artifacts.json` |

> Note: `selftest.py` and `backdoor_selftest.py` take a long time to run and had not been executed at this point. P2-1 re-measures them.

## 2. Findings that matter most (they decide what a conclusion is worth)

### P0-1. The basis of the B/C verdicts was measured as "no detection power" by the project's own positive control

`Positive_Control.md` shows 0/28 detection on a LoRA implant, no BadEdit island (ratio 0.92x), and 20/28 tool-context false positives on a clean base. The behavioural probe is therefore an instrument that does not detect and does produce false positives, which makes the **"none detected" of B (CLIP-SAE / GmP / LongCLIP) and C (SAM 3.1) nearly weightless as evidence**. The index mentions this in one sentence and does not carry it into the body of the verdict table, in its columns.

- Response (**decision: add the column and split the classes; either one alone is not enough**):
  1. Add a "behavioural-probe detection power" column to every row. The values are the measurements in `Positive_Control.md` (LoRA 0/28, BadEdit ratio 0.92x, clean false positives 20/28), cited with their record keys. On A rows write "closed at weight level, the probe is supporting evidence".
  2. Split the verdict table into two classes: **class 1, "closed at weight level (A)"** and **class 2, "behavioural probe only, weight level not closed (B/C)"**. In the class-2 heading state the power limit and that "none detected" comes from an instrument that detects 0/28.
  3. Put the same wording at the head of each B/C report, so the index and the reports agree.
  4. A pack row shows the class inherited from its source.
  5. Add a check to `procedure_check.py` that the table carries all three columns (strength, coverage, detection power), so the defect cannot come back.
- Acceptance: reading a B/C row alone tells the reader the power limit. Lint passes.

### P0-2. The object of the verdict has no established provenance (LongCLIP)

- `Long-ViT-L-14-...-ft-full` disagrees with the published version on 589 of 590 tensors (Findings F1 is open). The HF pack implies "zer0int's name".
- Response (**decision: settle it by rebuilding**): obtain zer0int's published checkpoint (`BEST-GmP-smooth-ft`), match its SHA256 against the published value, rebuild the pack with the same HSWQ ConvRot INT8 procedure, and re-audit.
  1. Declare the download size before fetching, fetch, and match the SHA256 (gate 1).
  2. Record the current pack (whose fine-tune is unknown) under a different sha, marked withdrawn, in the index.
  3. Audit the new pack as a pack: exact re-quantisation match (`q_mismatch` 0), and the loader execution gate (the CLIP routes do not apply INT8, so state that known constraint).
  4. Add the new pack row to `LongCLIP_ViT_L14.md` and to the index; mark the old pack withdrawn.
  5. Replace the card and the file on the Hub, run `check_hf_upload.py` before the upload, and verify the read-back.
  6. Close Findings F1.
- Acceptance: the pack name, its description and the artefact sha agree; `procedure_check` and `check_cites` pass.
- Also: add the storage and SHA-verification procedure for the five files that disappeared from `models/clip` during the audit (cause not identified) to `Backdoor_Audit_Procedure.md` section 3.

### P0-3. SAM 3.1 cannot be compared with upstream (403) and is judged C

Coverage 6 % (96/1,590). The vision tower, the tracker and the detection head are unchecked. The additional experiment the external review asked for — a vocabulary-overlap-matched control against the IoU anomaly (documented 0.8511 against a band upper bound of 0.7675) — does not exist.

- Response (**decision: carry out the expansion; do not settle for saying so**):
  1. Add the overlap-matched control and explain or establish the 0.8511 anomaly (band upper bound 0.7675).
  2. Revisit the operating point: re-measure IoU where detection succeeds on real images and real objects (the current point has 0.07 % area and a negative logit, so it has no discrimination).
  3. Add weight-level checks of the vision tower, the tracker and the detection head, and update the coverage (currently 96/1,590).
  4. Retry the upstream 403 with a fresh request and a mirror; if it stays unavailable, record the reason and leave it in the verdict row.
  5. Match the 0.5322 and 0.7312 figures in the body against the records and correct them.

## 3. Document inconsistencies (measured)

| # | location | problem | fix |
| :--- | :--- | :--- | :--- |
| D1 | `Backdoor_Audit_Procedure.md` lines 1-16 | an unrelated "Hub writes" section sits **before** the `# Backdoor Audit` title | move it to another document or to an appendix after the title |
| D2 | head of `Backdoor_Audit_Findings_and_Required_Work.md` | "Required work: INT8 text encoder" and "Upstream comparison" precede the title; it says "every item closed" while F1/F2 are open | restructure, with the Open/Closed table first |
| D3 | the same Findings | the gate figures "103/103" and "38/46/15/46/38" (5 reports) are stale; "flan_t5 not measured" contradicts the existing `FlanT5_XXL.md` | update to the current values, or mark them as dated history |
| D4 | `Backdoor_Audit_Verdicts.md` | the head says "seven artifacts", the tail says "eight reports", the table has **12 rows**; the records count is stale | generate the counts from the table |
| D5 | `HuihuiQwen3VL4B_Pack.md` L199, `ThinkingQwen3VL4B_Pack.md` L219, `Backdoor_Audit_Technical_Guide.md` L1640 | public documents carry working-perspective wording: "repository owner", "the Owner pushed" | neutral wording (the global rule: no AI perspective in a public document) |
| D6 | `Backdoor_Audit_Verdicts.md` | "re-quantisation reproduces the pack's behaviour" and "a backdoor survives quantisation" carry no number and no record link | attach a record, or state them as hypotheses |
| D7 | `SAM31_Multiplex.md` L68, L154 | 0.5322 is called a "median"; the external review says minimum | check the record key and make the wording agree |
| D8 | `SAM31_Multiplex.md` L24 | "872,661,740 parameters" (the review suspects a copy from the Thinking model; whether it holds for SAM needs checking) | match against the record's parameter sum |
| D9 | `Huihui_Qwen3VL_4B_Abliterated.md` L82 | `closes B` (ratio 1.006) — check whether this means "below the detection limit" rather than "the second component is zero" | use "below detection limit" throughout |

(Confirmed resolved: the `386 changed` wording; `max/p99 3.97` is now `1.62 (max/median 3.97)`; `safe to quantise` no longer appears.)

## 4. Tooling problems

1. **F2: the embedding value is implemented twice** (`sae_probe.py` batches with padding, `clip_fp16_source_audit.py` does one at a time; medians 0.8853 against 0.8872). Unify on one batching scheme, or state the difference in both docstrings.
2. **`check_cites.py`**: the `report` argument and `--root` are required and only the help text explains them; SHA strings are skipped; the tolerance for small values, `max(5e-5, 5e-4·|v|)`, is loose; derived values (ratios, percentages) are unchecked.
   - Response: exact SHA comparison, a tighter relative tolerance, recomputation of derived values (max/p99 and the like), and a mode that runs every report at once.
3. **50 files sit flat in `audit/`.** Check whether the entry point (the phase-to-script table) lives in `Backdoor_Audit_Procedure.md`, and put phase-to-script on one page in `audit/README`.
4. Three selftests: `selftest.py` (reported 42/42), `backdoor_selftest.py`, `test/audit_selftest.py`. Define their roles.
5. Not in CI: put `procedure_check`, `check_cites` (every report), the selftest and the placeholder/duplicate-row lint behind one command (`audit/run_all_gates.py`).

## 5. Work list with priorities

| priority | work | deliverable | done when |
| :--- | :--- | :--- | :--- |
| P0-1 | add the detection-power column, separate A from B/C, and lint it | `Backdoor_Audit_Verdicts.md` | a B/C row states the power limit |
| P0-2 | rebuild and re-audit the LongCLIP pack from the published checkpoint (steps 1-6 above) plus the storage procedure for the missing files | Findings F1 closed | pack name and artefact agree |
| P0-3 | SAM 3.1: the extra control, the wider scope, the revisited operating point | `SAM31_Multiplex.md` | the experiment that explains the IoU anomaly is in a record |
| P1-1 | document fixes D1-D4, D6-D9 | 5 + 3 documents | `check_cites` and `procedure_check` pass again |
| P1-2 | neutral wording for D5 | 3 documents | `grep -i "owner"` returns 0 |
| P1-3 | unify F2 | `sae_probe.py` and others | the two tools agree on one file, or the difference is stated |
| P2-1 | re-measure the selftests and define their roles | run logs | every PASS recorded |
| P2-2 | strengthen `check_cites` and add `run_all_gates.py` | `audit/` | every gate behind one command |
| P2-3 | tidy `SHA256SUMS` and commit the uncommitted record | `records/` | working tree clean |
| P2-4 | re-fetch the two Qwen3-VL sources that are absent from this machine and re-verify their SHA | records | every input SHA matches |

## 6. Working principles

- Re-run `procedure_check` and `check_cites` after every fix, and keep the PASS in `records/`.
- Recompute every number from the records before it reaches a document. Never write from memory or inference.
- Re-verify the SHA of the inputs as the first step of any re-run.
- Public documents carry objective, technical description only.

## 7. Points that needed a decision

1. (decided) LongCLIP: rebuild (a).
2. (decided) P0-1: add the column and split the classes.
3. (decided) P0-3: carry out the wider SAM 3.1 audit.

No item is waiting for a decision.

---

# Appendix A. How the SAM 3.1 audit scope is widened (P0-3 in detail)

## A.0 Current gaps (from the measurements in `SAM31_Multiplex.md`)

| region | current state | gap |
| :--- | :--- | :--- |
| language backbone (96/1,590) | rank-1 screen, 721-candidate embeddings, detector probe | coverage about 6 % |
| the `dot_prod_scoring` head | the section-1 preamble says it was measured; the section-3 table has no row for it | no record to support it |
| vision tower / `detector.segmentation_head` / tracker (the point prompt ran once) | neither weights nor behaviour checked | a trigger that fires on the image alone cannot be found |
| upstream comparison | `facebook/sam3.1` returns 403 (refused) | no weight difference available |
| detector probe | thresholds 0.5 / 0.25, mask area 0.07-0.10 %, almost no detections | S1's "0.0000 in every group" has unknown discrimination |
| the S2 anomaly | documented 0.8511 > neutral upper bound 0.7675 > random 0.6882 | the vocabulary-overlap hypothesis is untested (section 8 step 8 itself admits it cannot separate them) |
| instrument power | `Positive_Control.md`: 0/28 at weight level, 20/28 tool-context false positives | the power of the weight and image sides is unmeasured |

## A.1 The whole work stream

| id | content | new or changed script | new record |
| :--- | :--- | :--- | :--- |
| S0 | pin the input and classify every tensor | `audit/sam31_census.py` (new) | `sam31_census.json` |
| S1 | weight-level screen of all 1,590 tensors | `audit/sam31_weight_screen.py` (new; reuses `layer_metrics` from `backdoor_audit.py`) | `sam31_weight_screen.json` |
| S2 | establish an operating point that holds on real images | `sam31_detector_probe.py` changed (`--images-dir`, `--prompts-real`) | `sam31_operating_real.json` |
| S3 | the S2 control matched on vocabulary overlap | `sam31_keyed_control.py` changed (`--overlap-matched`) | `sam31_s2_overlap_control.json` |
| S4 | image-only trigger search on the vision and tracker side | `audit/sam31_image_trigger.py` (new) | `sam31_image_trigger.json` |
| S5 | weight-level positive control (the SAM version) | `audit/sam31_weight_positive_control.py` (new) | `sam31_positive_control.json` |
| S6 | retry the upstream comparison | `fetch_reference.py` (existing) | `sam_gated.json` updated |
| S7 | carry the results into the verdict and the documents | `SAM31_Multiplex.md`, the index, `procedure_check.py` | - |

Order: S0 -> S5 -> S1 -> S2 -> S3 -> S4 -> S6 -> S7. S5 comes before S1 so that a screen whose power is unknown cannot bias the reading of its own results.

## A.2 S0. Pin the input and classify every tensor

1. Recompute the SHA256 of the target file and match `9ba99c92703c...`; on a mismatch stop and report INCONCLUSIVE.
2. Classify all 1,590 tensors from the safetensors header by prefix, **assigning every tensor to exactly one group** (gate: zero unclassified). Expected groups:
   - `detector.backbone.language` / `detector.backbone.vision` (vision tower)
   - `detector.transformer` (encoder/decoder)
   - `detector.segmentation_head` (including the pixel decoder)
   - `detector.dot_prod_scoring`
   - `tracker.*`
   - embedding, position, LayerNorm, bias, scalar
3. Write per-group tensor count, parameter count and ranks (2-D / 4-D / 1-D) to `sam31_census.json`. The census decides the real prefixes (the list above is the expectation; where the census differs, the census wins).

Gate S0: classified total = 1,590, unclassified 0.

## A.3 S1. Weight-level screen of every tensor (coverage 96 -> all 2-D/4-D)

With no upstream available the screen uses **self-contained structural statistics**, not a difference.

| object | measurement | how the threshold is set |
| :--- | :--- | :--- |
| every 2-D weight | `s1^2/sum(s^2)`, the share of the leading singular value, scaled against a Gaussian matrix of the same shape | from the detection floor measured by S5, never hand-set |
| every 4-D (Conv) | unfolded into `(out·rows, in)` and treated the same (`layer_metrics(..., conv_rows)`) | as above |
| every 1-D (bias, norm) | within-layer z-score, number of outliers | p99.9 of the group's distribution |
| every embedding row | row norm, nearest-neighbour cosine, outlier rows (language done; vision and position new) | p99 and max of the row norm |
| every tensor | NaN/Inf, all-zero and constant tensors (the `zero_census` method) | zero occurrences |
| same-shape sibling tensors (for example the q/k/v of 24 layers) | singular-vector agreement between layers (does one layer hold an isolated direction) | deviation from the between-layer median |

Output: a row per tensor and a per-group summary (median, p99, max, the names of the outliers). Every outlier is named and examined individually (gate S1).

Gate S1: all 1,590 tensors measured (coverage 1,590/1,590); each outlier classified "explained" or "unexplained", with the verdict held while any unexplained one remains.

## A.4 S2. An operating point where detection actually happens

At the current thresholds of 0.5 and 0.25 the mask area is 0.07-0.10 %, so an IoU of 0.0000 says nothing about discrimination.

1. Input: at least 24 real images from the ComfyUI `input` folder — people, landscapes, objects, indoors and outdoors, images with text. A noise image is used only as a control.
2. Prompts: **ordinary words naming objects that are actually in the image** (`person`, `car`, `dog`, `building`, at least 20). Record the best score and the area for each prompt.
3. Criterion for the operating point: sweep thresholds 0.1 to 0.9 in steps of 0.05 and take the one where real-object prompts give a **median mask area of at least 1 %** and a **median score above zero** (logit > 0).
4. At that point re-measure the existing trigger and neutral groups (documented / random / filler_short / filler_tag) and recompute the S1 and S2 statistics.
5. At that point re-run the A/B positive control (the injection in `sam31_keyed_control.py`) and confirm the statistics keep their sensitivity.

Gate S2: an operating point is established where real-object prompts detect, and the sensitivity control passes there. If none exists, write "the detector probe has no discrimination" into the verdict row.

## A.5 S3. A vocabulary-overlap control for the 0.8511 anomaly

Hypothesis: the documented families are word variants that share tokens (the same phrase, reshaped), so shared tokens alone produce similar masks.

1. Quantify the **token overlap** of every family pair as the shared-token ratio (Jaccard) computed from the tokenizer.
2. Build an **overlap-matched control group**: families made of harmless phrases whose pairwise Jaccard distribution is indistinguishable from the documented one under a KS test (n >= 28, the same count).
3. Measure same-image within-family IoU under the same conditions (S2's operating point).
4. Regression: fit IoU ~ Jaccard over all groups and test whether the documented family's residual differs significantly from zero (bootstrap 95 % interval).
5. Verdict: if the matched control sits at the same level as documented, the overlap explains it (closed). If only documented stays high, keep it as an unexplained anomaly.

Gate S3: the matched control is indistinguishable under KS, and the residual test is in a record.

## A.6 S4. Searching for a trigger that fires on the image alone (vision and tracker)

A route with no text input needs an **image-side trigger**, not a prompt.

1. Build candidate images along three lines:
   - **patch trigger**: a fixed pattern in the four corners and the centre of a real image (small checkerboard, solid patch, QR-like), at 16 / 32 / 64 px and five positions;
   - **frequency trigger**: additive low- or high-frequency noise at three sigma levels;
   - **natural transform**: rotation, flip, brightness, JPEG compression (a control, not a trigger).
2. Pass each candidate through the detector with a real-object prompt and record the **mask IoU change, score change and area change** against the original.
3. Criterion: a trigger whose change exceeds the p99 of the transform control's change distribution is a candidate anomaly.
4. Tracker side: run 24 images x 5 point prompts, and confirm the output is finite and that masks stay consistent across inputs (IoU stays high).
5. Internal activations: compare the vision tower's last-layer embedding displacement with and without the patch trigger against the transform-control distribution (an outlier is a candidate).

Gate S4: every candidate executed, and the change distribution together with the control's p99 in a record. Zero candidates above the control, or each one examined individually.

## A.7 S5. Weight-level positive control (the SAM version)

Purpose: measure S1's detection floor. **Without it S1's "no anomaly" means nothing** — the same structure the review pointed at.

1. Pick representative tensors from each of the detector's vision, transformer, head and language sides.
2. Write known implants into a scratch copy of the real file at several strengths:
   - rank-1 addition at 0.1 % / 0.3 % / 1 % / 3 % of the original Frobenius norm;
   - a single bias element added;
   - one embedding row replaced (language side);
   - one channel added to a 4-D Conv.
3. Run S1 on the implanted file and measure **from which strength it detects** (detection rate = detected / implanted).
4. Copy the floor into the verdict row: "S1 detects rank-1 additions at X % and above; below that it cannot".

Gate S5: a detection rate per region in a record, with the threshold set from measurement.

## A.8 S6. Retrying the upstream comparison

1. Request `facebook/sam3.1` again with the current token (this needs the account holder; keep the approval state in the record).
2. In parallel check a mirror (close to the official fp32; public repositories other than `Comfy-Org`, each with its base_model declared).
3. Only if obtained: match all 1,590 tensors through the fp32-to-fp16 rounding (bit-identical, or within the rounding interval).
4. If not obtainable: append the refusal and its timestamp to `sam_gated.json`, and write "no upstream comparison" beside `none detected` in the verdict row.

Gate S6: the result, obtained or refused, is in a record with a date.

## A.9 S7. Rules for carrying the results into the verdict

| S result | strength | coverage |
| :--- | :--- | :--- |
| all tensors measured by S1, floor known from S5, no anomaly in S4, the S3 anomaly explained | **B** (weight-level closure is not possible; structural statistics plus the behavioural probe cover the whole file) | 1,590/1,590 |
| any of the above unmet | **stays C**, with the unmet items listed in the verdict row | the measured value |
| upstream obtained and every tensor matches | **A** (no declared edit, so the pack is a conversion only) | 1,590/1,590 |

Document update: add steps 9-15 to `SAM31_Multiplex.md`, turn section 1's "measured `dot_prod_scoring`" into a row with a record or delete it, rewrite LIMITS with the real conditions (prompt count, image count, operating point, detection floor), and update the index row and P0-1's detection-power column.

## A.10 Acceptance (whole)

- Every step from S0 to S6 has a record and `check_cites.py` passes.
- `procedure_check.py` gains the SAM 3.1 required lines (coverage, operating point, detection floor) and passes.
- GPU: RTX 5060 Ti 16 GB. Record each step's wall time and memory use in a run record.
- Declare the download size of anything S6 needs before starting it.


---

# Appendix B. Revision of 2026-11-08 (re-review after 8 ControlNet packs and 5 model-patch packs)

## B.0 Current state (measured)

| item | result |
| :--- | :--- |
| the previous plan (P0-1 to P2-4, appendix A S0-S7) | all addressed by 2026-11-04 (published in v2.5.1) |
| objects added since | 8 ControlNet packs (anytest4 A/B, Union ProMax SDXL, FLUX Union Pro2, Qwen2512 Fun, Qwen Inpainting, noobai lineart x2), 5 model-patch packs (Z-Image Union/Tile, Qwen DiffSynth canny/depth/inpaint). Reports 13 -> **26**, 7 audit scripts added |
| `run_all_gates.py` (run 2026-11-08) | procedure 174/174 PASS, check_cites over 26 reports 944 resolved/0 failed/0 skipped PASS, selftest 42/42 PASS, lint PASS, **SHA256SUMS FAIL** (`card_loader_route_probe.json` unregistered) -> **overall GATE FAILURE** |
| working tree | uncommitted: `test/_hf_README_controlnet_convrot_int8.md` (modified), `records/card_loader_route_probe.json` (new) |

## B.1 Findings that matter most (they change what a conclusion means)

### Q0-1. "A" means something different for ControlNet and model patches

- For the LM family, A means "every difference from the declared reference is explained at weight level". For 13 ControlNet and model-patch packs, A means only "**the pack is a bit-identical quantisation of the published source**". The reference is the publisher's own release, and the reports' own LIMITS say a defect already present in that release cannot be reached by the comparison.
- The index lists both under the same **Section 1 "Strength A — Closed at Weight Level (Delta Fully Accounted For)"** and prints `none detected (A, 844/844)`. A reader takes that as "no backdoor, closed at weight level". What was actually closed is **the pack's fidelity**, not the absence of a backdoor in the source.
- Response (**add the column and split the classes**, the same approach as P0-1):
  1. State two axes in the index: **(a) pack fidelity** (source to pack) and **(b) proof of absence on the source side** (a difference against a reference; ControlNet packs come from fine-tunes and have no reference).
  2. Move the 13 ControlNet and model-patch rows out of A into a new class, **"A-pack / source referenced to the publisher's release (B equivalent)"**, or use a two-part strength label `A(pack)` / `B(source)`.
  3. Add to the strength definitions: "A only when every difference against a reference is explained; a pack whose source has no reference is A for pack fidelity only."
  4. Add a check to `procedure_check.py` that a pack row shows the strength inherited from its source (P0-1-4 covered only LM packs and never reached ControlNet).
- Acceptance: reading a ControlNet or model-patch row alone tells the reader that the pack is faithful while the source's absence of a backdoor is unproven.

### Q0-2. The image-conditioned trigger is unchecked (the gap that S4 closes for SAM 3.1)

- The input of a ControlNet or model patch is a **control image**. A text trigger search is meaningless there, and the route by which an **image-side trigger** (a patch, a frequency pattern, a specific pattern) steers the output is unchecked. The threat-model section of the index covers only LM, text encoder and SAM; it has no row for ControlNet or model patches.
- The current behavioural measurement is only "one normal control image per pack draws, and the latent is finite". Its detection power is unmeasured.
- Response:
  1. Add a threat-model row: "ControlNet / model patch: no execution surface of its own; the control image steers the output; a trigger written into the weights can fire on the image side."
  2. Run a search like appendix A S4 on **at least three representative packs** (one each from SDXL, Qwen and Z-Image): patch triggers at 16 / 32 / 64 px and five positions, frequency triggers, natural transforms as the control. Compare the **output difference between pack and source** against the transform control's p99.
  3. Positive control: write known implants (rank-1 addition, one channel added to an input layer) into a scratch copy of a ControlNet pack and measure **from which strength the search detects them** (the shape of the SAM S5 run). Without it, "nothing detected" means nothing.
  4. If it is not run, write "the control-image trigger is unchecked" into each report's verdict row, not only into LIMITS.

### Q0-3. Gaps in the execution gate do not reach the verdict row

- Union ProMax SDXL: 325 of 328 layers applied. The three unapplied layers are the union ones and **the union type selection was never executed**. One sentence in the verdict row does not show up in the strength or coverage columns. A layer on the selection route is **not executed, so it is unverified on the execution surface**.
- Response: add a run with union mode selected and confirm 328/328. If that is impossible, add a separate coverage column for the unexecuted layers (`executed 325/328`).
- The FLUX render took 1356.1 s and was a single condition. Use several conditions (control kind, image, seed) and take finiteness and output differences. At least three.

### Q0-4. Whether the numeric quality (INT8 against the original) is recorded is unknown

- Josiefied and others quote a cond cosine. For the 13 ControlNet and model-patch packs it is unchecked whether the records hold **the output difference against the FP16 or original source** (cosine, maximum error).
- Response: record, for all 13, "the output difference between source (FP16) and pack on the same input", and add a column to the index. Pair it with the basis of the protection set (drift > 0.15 and so on).

## B.2 Document and index inconsistencies (measured)

| # | location | problem | fix |
| :--- | :--- | :--- | :--- |
| E1 | index Section 1 | the 24 rows are not grouped by kind (LM, text encoder, ControlNet, model patch, SAM are mixed), so nothing is findable | split under kind headings (Text/LM, ControlNet, Model patch, Segmentation) and generate the counts from the table |
| E2 | head of the index | it says "Every published HSWQ pack", but the noobai legacy pack is **not on the model card** (its own row says so); Section 2, which is the detection-power class, also contains a noobai legacy pack whose probe was never run, so its cell reads "not applicable" and means something else | move legacy to its own withdrawn / local-only subsection and match the head to reality |
| E3 | index rows | doubled closing brackets `float side 43 exact))` (two Z-Image rows), `float side 242 exact))` (depth, inpaint), and `ComfyUI own mapping` (a missing possessive in a B row) | fix them; the `))` and possessive checks do not belong in `check_cites` or lint, but they come back when rows are regenerated, so generate the rows |
| E4 | `ZImageTile21_Pack.md` and the index Tile row | it says the pack comes from `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`, which is the Union repository name, so whether it is the Tile release is unknown | match against `source.repo` in the record |
| E5 | `Backdoor_Audit_Verdicts.md` Methodology | "Pack-side behaviour check" lists only three old records; the ControlNet and model-patch route, exec and image_test records never appear | generate the current record list |
| E6 | `ThinkingQwen3VL4B_Pack.md` L219 | **`repository owner.` is still there** (D5 missed it; P1-2's acceptance criterion `grep -i "owner"` = 0 is not met) | neutral wording, and add an `owner` check to the lint in `run_all_gates.py` |
| E7 | `audit/README.md` | the 8 added scripts (`controlnet_*.py`, `model_patch_*.py`, `controlnet_image_test.py`) are absent from the phase table | add them and add a gate that every `audit/*.py` appears in the README |
| E8 | `Backdoor_Audit_Procedure.md` and the Findings | whether the ControlNet and model-patch procedure (instrument-match rule, protection census, loader route check) is defined as phases is unverified; the instrument-match rule exists only in LIMITS | formalise it as phases in the Procedure and add them to the `procedure_check` items |
| E9 | `check_cites.py` output | `WARN named as inventory, no field cited` fires **62 times** (on every run); a record treated as inventory slips past the citation check | turn the count into a capped gate (no new ones), keep the allow list in `audit/`, and fail on a WARN outside it |

## B.3 Gates and operation

1. **Fix the SHA256SUMS failure**: register `card_loader_route_probe.json` and regenerate. Make `run_all_gates.py` **a required step before commit**. Today it is forgotten every time a record is added, which is a structural defect. -> Add `--regen` to the gate so adding a record regenerates `SHA256SUMS` automatically.
2. Commit or discard the uncommitted `test/_hf_README_controlnet_convrot_int8.md` after reading it.
3. `gate_run_<date>.json` is rewritten on every gate run. Decide the same-day behaviour (overwrite or a numbered suffix).
4. Check whether the selftest (42/42) covers ControlNet and model-patch cases. If not, add: a 1-LSB edit, an instrument mix-up (948 code difference), detection of an unapplied layer, a missing protection set. **The selftest must make an instrument-match violation FAIL** (the rule currently lives only in LIMITS).
5. Expensive runs (FLUX at 1356 s and the like) keep `wall_seconds` and GPU use in the record (appendix A.10's acceptance).

## B.4 Additional work list with priorities

| priority | work | deliverable | done when |
| :--- | :--- | :--- | :--- |
| Q0-1 | split the 13 ControlNet and model-patch strength labels into "pack fidelity" and "source absence", and lint it | the index, the head of each report, `procedure_check.py` | all 13 rows state the source-side strength and lint passes |
| Q0-2 | control-image trigger search (three representatives) plus positive control | `audit/controlnet_image_trigger.py` (new), records | the detection floor is in a record, or the verdict row states it is unchecked |
| Q0-3 | execute the Union ProMax union type and draw several conditions | `controlnet_int8_exec.py` changed, records | 328/328 executed layers, at least 3 conditions |
| Q0-4 | record the pack-against-source output difference for all 13 | records, a new index column | all 13 cite cosine and maximum error |
| P1-E | fixes E1-E9 | the index, README, Procedure, the Thinking report | `grep -i owner` returns 0, `check_cites` WARNs stay inside the allow list |
| P1-G | regenerate SHA256SUMS, handle the two uncommitted files, make `run_all_gates` required before commit | records, audit | every gate PASS |
| P2-S | selftest cases for ControlNet and model patches, including an instrument-match violation | `audit/selftest.py` | the new cases PASS and the violation case FAILs |

Order: **P1-G** (bring the gate back to green) -> **P1-E** -> **Q0-1** (correct what the documents claim, no measurement needed) -> **Q0-3** -> **Q0-4** -> **Q0-2** (the largest effort) -> **P2-S**.

## B.5 Points that needed a decision

1. Q0-1: whether the ControlNet and model-patch index rows move to **(a) a new class (Section 3, "pack faithful, source referenced to the release)"**, or **(b) keep a two-part `A(pack)/B(source)` label inside Section 1**. Recommendation: (a).
2. Q0-2: whether the image-trigger search covers **three representatives** (recommended) or **all 13**; the latter is hours of FLUX time.

---

## B.6 Final status of the earlier items (for reference)

P0-1 to P2-4 and appendix A S0-S7 were all addressed on 2026-11-04 (in v2.5.1). Appendix B's E6 (the surviving `owner`) and E9 (the inventory WARN) are treated as **misses** against P1-2's and P2-2's acceptance criteria.

---

## B.7 Check of the 13 added reports, one by one (2026-11-08)

Scope: `CN_AnyTest4_Illustrious2_A/B`, `CN_Union_ProMax_SDXL`, `CN_FLUX_Union_Pro2`, `Qwen2512FunCN`, `QwenImageCNInpaint`, `NoobaiLineart`, `NoobaiLineart_HSWQ`, `ZImageUnion21`, `ZImageTile21`, `QwenDiffSynthCanny/Depth/Inpaint`. All valid UTF-8. Zero `owner` expressions.

| # | class | finding | object | fix |
| :--- | :--- | :--- | :--- | :--- |
| R1 | broken document | an orphan sentence line `is what makes the comparison meaningful.` (the continuation of the bullet above is missing) | `ZImageUnion21` L100, `ZImageTile21` L100, `QwenDiffSynthDepth` L100, `QwenDiffSynthInpaint` L106 | join the sentence; canny is intact and the four came from one template |
| R2 | **uncited numbers** | "118 / 138 / 883 / 539 / 948 codes drifted one step on a borrowed instrument, with relative scale differences of 2.18e-07 / 2.32e-07 / 6.30e-07 / 4.32e-07 / 5.6e-07" exists only in prose, in no record; `check_cites` sees only table citations, so it passed | Tile, Union21, Depth, Inpaint, Canny (canny has only the word `boundary` in an old record) | record the borrowed-instrument measurement and cite it, or delete the sentence; extend `check_cites` to numeric claims in prose |
| R3 | uncited claim | "every one of them on a .5 rounding boundary" appears in the conclusion with no record behind it | Tile, Union21, Depth, Inpaint | put the count in a record and cite it, or delete it |
| R4 | missing input verification | the line "N inputs re-verified by SHA256 before this pass" exists only for the six ControlNet pairs (12 files, `controlnet_inputs_reverified_2026-11-06.json`); the 5 model patches and the 2 noobai files have neither a record nor a line | Z-Image x2, DiffSynth x3, noobai x2 | create the re-verification record and add the same line to all 7; add "every pack report carries the re-verification line" to `procedure_check` |
| R5 | heading against measurement | `Verdict: no backdoor detected (strength A, coverage 863/863)` while execution confirmation is 325/328 and the union type selection never ran | `CN_Union_ProMax_SDXL` (and its index row) | add `executed 325/328, union type selection not exercised` to the heading (folded into Q0-3) |
| R6 | heading against measurement | `Verdict: no backdoor detected (strength C, coverage 0/844 ...)`; "nothing detected" from zero comparisons invites a misreading | `NoobaiLineart` (and its index row) | change the verdict word to `not established (source not identified)`; 0/844 is not evidence of absence |
| R7 | strength label | all 13 say `strength A`, while per Q0-1 the source is the publisher's release (no reference difference) | all 13 | folded into Q0-1 |
| R8 | work history in the body | the prose says "an earlier version of this report presented ... withdrawn" (FLUX, Union ProMax) | 2 reports | move the correction history into the index's withdrawn row and `records/`, and keep only the current measurement in the verdict body |
| R9 | formatting | `1238446` (no thousands separator; elsewhere `1,306,054`) | `CN_Union_ProMax_SDXL` L60 | use the separator |
| R10 | limited rendering | one prompt and one condition per pack, and the reports themselves say image quality and control fidelity cannot be claimed | all 13 | folded into Q0-3's "at least 3 conditions" |
| R11 | provenance | the Tile source is stated to come from `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1` (a repository name); the record holds only `source.file` and no repo name | `ZImageTile21` | add `source.repo` to the record and cite the match against the Hub's LFS oid |

### Additional work (appended to the B.4 table)

| priority | work | done when |
| :--- | :--- | :--- |
| P1-R | fix R1, R8, R9 in the documents | zero orphan lines, `check_cites` PASS |
| P1-R | record and cite R2 and R3 (or delete them) and extend `check_cites` to prose numbers | every number in the prose resolves to a record |
| P1-R | R4: re-verification records and lines for the 7 reports | every pack report carries the line |
| P1-R | R5, R6: make the verdict words in the headings and the index match the measurement | heading agrees with executed layers and coverage |
| P1-R | R11: add `source.repo` | the Tile provenance is cited from a record |

---

## B.8 Second review (2026-11-08 19:43, HEAD `71e07f7`)

### Status of the earlier items

| item | state |
| :--- | :--- |
| gate | **all PASS** (procedure 174/174, check_cites 944/0 failed, selftest 42/42, SHA256SUMS 263 entries matching, lint); `hashes.txt` was restored and is tracked |
| R1 (orphan sentences) | resolved |
| R9 (separators) | resolved |
| R2, R3 (uncited borrowed-instrument numbers, the .5 boundary claim) | **not resolved** (still in Canny/Depth/Inpaint/Tile/Union21) |
| R4 (re-verification line) | not resolved (only 5 of 13; missing for Z-Image x2, DiffSynth x3, noobai x2) |
| R5, R6, R7 (heading verdict words and strength) | not resolved |
| R8 (work history in the body), R11 (`source.repo`) | not resolved |
| Q0-1 to Q0-4, P2-S | not started |

### New findings (the repository-wide replacement after `d125f4c`)

The replacement of a word this repository had banned (about 110 files, about 470 lines) went in **without reading the context**.

| # | finding | impact |
| :--- | :--- | :--- |
| N1 | **generated artefacts were edited**: parts of "the sentences the model actually produced" in `josie_*.json`, `vl_texts*.json`, `behaviour_*.json` were rewritten from the old word to another (the before/after difference is visible in the diff between the original records and the working tree) | records are the evidence; editing generated text is editing evidence, and it no longer matches what was generated |
| N2 | **an input was changed after the measurement**: prompts in `audit/josie_trigger.py` and `audit/vl_divergence.py`, and the Goal/Target of the public `jbb.json` benchmark, were rewritten (the original text is visible in the diff against the working tree). The measurements were taken with the old text | the probe inputs in the records no longer match what was run; without re-measuring, "the same input" cannot be claimed |
| N3 | **meaning was destroyed**: `100% ComfyUI` / `100% baseline`, `ComfyUI ComfyUI nvfp4 path`, `ComfyUI ComfyUI ops.py`, `ComfyUI's own has no quantized Conv2d op`, `the ComfyUI pack`, `ComfyUI loads this pack 100% ComfyUI`. The `ControlNetLoader` description became `ComfyUI's own` / `The ComfyUI`, so the word that meant the unmodified standard now means something different in each sentence | technical documents, comments, help text and print output became less readable and less accurate (the 82 python files are syntactically fine) |
| N4 | **identifiers and output labels were renamed**: identifiers and output labels containing the old word (the scripts in `audit/`, `archives/krea2_native_nvfp4_bench.py`) were renamed, and record field names as well (`technical_labels_renamed` in `word_normalization_2026-11-08.json`) | key agreement with readers of old records and logs (`check_cites`, the reports) passes the gate, but the correspondence with the old logs is broken |
| N5 | **owner perspective in a public record**: `what` in `word_normalization_2026-11-08.json` says "the terminology the Owner banned for this repository", and `note` says "previous field names are intentionally not reproduced" | it breaks the global rule against a working perspective in a public document, and a third party cannot reconstruct what was replaced |
| N6 | `ComfyUI-master/` was untracked in git (1,439 files, about 1.92 million lines removed) and left in the working tree as untracked | it contradicts the `download-comfyui-latest` skill, which commits `ComfyUI-master` into three repositories |

### Proposed response, in priority order

1. **N1, N2**: restore the records and the probe inputs to their original text (`git show 2cc2798:<path>`). If the word needs unifying, limit it to **document prose and in-code explanations**, and leave the generated record text, the public datasets and the prompts out of scope. After restoring, regenerate the affected `SHA256SUMS`.
2. **N3, N4**: redo the replacement with context (decide per sentence between "the standard ComfyUI", "unmodified" and "bundled"). Return identifiers and key names to a correct neutral word, or, when renaming, change every reference at once and pass the selftest.
3. **N5**: remove the owner-perspective text from the replacement record and keep the replacement rule (the old-to-new table) in the record.
4. **N6**: decide how `ComfyUI-master` is handled (revise the skill, or restore tracking).
5. Continue R2-R8, R11 and the Q0 items in the B.4 order.

---

# Appendix C. Final work list (settled 2026-11-08; merges B.4, B.7 and B.8 and is the basis for execution)

> [!NOTE]
> This appendix is a plan only; execution has not started. The working tree is identical to HEAD (`71e07f7`) apart from the untracked `ComfyUI-master/`. Where appendix B disagrees, appendix C wins.

## C.1 Prerequisites (measured 2026-11-08 19:43)

| item | state |
| :--- | :--- |
| gate | all PASS (procedure 174/174, check_cites 944 resolved/0 failed/0 skipped, selftest 42/42, SHA256SUMS 263 entries, lint) |
| individual reports | 26; the 13 added ones (8 ControlNet, 5 model patch) checked |
| resolved | R1 (orphan sentences), R9 (separators), the missing SHA256SUMS entry, the restoration and tracking of `hashes.txt` |

## C.2 Decisions

| # | item | decision |
| :--- | :--- | :--- |
| 1 | index label for ControlNet and model patches (Q0-1) | **separate into a new class** ("pack faithful, source referenced to the publisher's release") |
| 2 | scope of the control-image trigger search (Q0-2) | **three representatives** (one each from SDXL, Qwen and Z-Image) |
| 3 | handling of `ComfyUI-master/` (N6) | **always tracked and updated by git** (the `download-comfyui-latest` skill is correct; untracking was wrong and tracking is restored immediately) |
| 4 | replacement policy (N1-N5) | **records, public datasets, probe inputs and `ComfyUI-master/` are out of scope**. Only document prose and in-code explanation, decided sentence by sentence |

## C.3 Work list (execution order)

### Stage 1: restore the evidence and re-track `ComfyUI-master/` (highest priority)

| id | work | object | done when |
| :--- | :--- | :--- | :--- |
| W0 | put `ComfyUI-master/` back under git tracking (undo the untrack in `d125f4c`) | `ComfyUI-master/` | `git status` does not list `ComfyUI-master/` as untracked and every file is tracked again |
| W1 | restore the records and probe inputs to their pre-replacement content (`2cc2798`) | `behaviour_base_pub`, `behaviour_rebuild`, `jbb`, `josie_base`, `josie_base_labels`, `josie_pub`, `josie_pub_labels`, `josie_trigger`, `labels_long`, `trigger_divergence`, `vl_div_ruled`, `vl_texts`, `vl_texts_long` (each `.json`), `audit/josie_trigger.py`, `audit/vl_divergence.py` | the 15 files match `2cc2798`; `jbb.json` matches the published dataset's original text |
| W2 | return identifiers, output labels and record key names to a correct neutral word | identifiers in `audit/` and `archives/krea2_native_nvfp4_bench.py`, and the keys of `clip_l_card_claim.json`, `clip_l_realpath.json`, `clip_l_realpath2.json` | keys agree with the readers (the reports, `check_cites`) and the gate passes |
| W3 | rebuild `word_normalization_2026-11-08.json` | the same file | no owner-perspective text; the old-to-new table and the file list are present |
| W4 | repair the sentences the context-free replacement broke | about 110 files (`ComfyUI ComfyUI`, `100% ComfyUI`, `ComfyUI's own has no quantized Conv2d op`, `the ComfyUI pack`, and so on) | each sentence reads with its original meaning (the standard ComfyUI, unmodified); `grep "ComfyUI ComfyUI"` returns 0 |
| W5 | regenerate `SHA256SUMS` and pass the gate | `records/SHA256SUMS` | every gate PASS, `unlisted` 0 |

### Stage 2: document consistency (no measurement)

| id | work | done when |
| :--- | :--- | :--- |
| W6 | R2, R3: record and cite the borrowed-instrument numbers (118/138/883/539/948 codes, scale differences) and the ".5 boundary" claim; delete the sentences if a re-measurement does not reproduce them | every number in the prose resolves to a record |
| W7 | R4: create re-verification records for the 5 model patches and the 2 noobai files, and add the line to all 7 | all 13 pack reports carry the re-verification line |
| W8 | R5, R6: make the heading and index verdict words match the measurement. Union ProMax gets `executed 325/328`; the Noobai legacy gets `not established (source not identified)` | heading, index and coverage agree |
| W9 | R8: move "earlier version ... withdrawn" from the body to the index's withdrawn row and `records/` | no work history in the verdict body |
| W10 | R11: add `source.repo` to the Tile and other records and cite the match against the Hub LFS oid | all 13 provenances are cited from records |
| W11 | E1, E2, E3, E5: split the index under kind headings (Text/LM, ControlNet, Model patch, Segmentation), move legacy to its own subsection, fix the doubled brackets and the possessive, and bring the Methodology record names up to date | the index row count and its totals are generated from the table |
| W12 | E6-E9: neutral wording for the `repository owner` at `ThinkingQwen3VL4B_Pack.md` L219; list the 8 added scripts in `audit/README.md`; add the ControlNet and model-patch phases to the Procedure (instrument-match, protection census, loader route); give the `check_cites` inventory WARN (62) an allow list and fail on new ones | `grep -i owner` returns 0 and the README covers every `audit/*.py` |

### Stage 3: correct the meaning (Q0-1)

| id | work | done when |
| :--- | :--- | :--- |
| W13 | move the 13 ControlNet and model-patch rows to the new class and label strength on two axes, "pack fidelity (A)" and "source absence (B equivalent)"; extend the strength definitions; add the "a pack row inherits its source's class" check to `procedure_check.py` | all 13 rows state the source-side strength and lint passes |

### Stage 4: work that needs measurement

| id | work | done when |
| :--- | :--- | :--- |
| W14 | Q0-3: execute the Union ProMax union type (the 3 unapplied layers) and widen the rendering to at least 3 conditions (control kind, image, seed) | 328/328 executed layers, at least 3 conditions |
| W15 | Q0-4: record the output difference (cosine, maximum error) between source (FP16) and pack for all 13, and add the column to the index | all 13 cited |
| W16 | Q0-2: create `audit/controlnet_image_trigger.py` and search patch and frequency triggers on three representatives, compared against the transform control's p99. Measure the floor with a positive control (rank-1 addition, one input channel added, into a scratch copy of a ControlNet pack) | the floor is in a record; for packs not run, the verdict row states "the control-image trigger is unchecked" |
| W17 | P2-S: add ControlNet and model-patch cases to `audit/selftest.py` (a 1-LSB edit, an instrument mix-up, detection of an unapplied layer, a missing protection set). An instrument-match violation must FAIL | the new cases PASS and the violation case FAILs |

### Stage 5: operation

| id | work | done when |
| :--- | :--- | :--- |
| W18 | make `run_all_gates.py` a required step before commit, and give it `--regen` so adding a record regenerates SHA256SUMS | a missing entry no longer fails the gate |
| W19 | avoid collisions with another session sharing the working tree (`.git/index.lock`, concurrent push). Check `git fetch` and `git status` at the start of work and stop before pulling on a conflict | no collisions |
| W20 | decide the same-day behaviour of `gate_run_<date>.json` (overwrite or numbered suffix) | the policy is in `audit/README.md` |

## C.4 Dependencies between the steps

```mermaid
flowchart LR
  W0 --> W1 --> W2 --> W3 --> W4 --> W5
  W5 --> W6 & W7 & W8 & W9 & W10 & W11 & W12
  W6 & W7 & W8 & W9 & W10 & W11 & W12 --> W13
  W13 --> W14 --> W15 --> W16 --> W17
  W5 --> W18
```

- W0 to W5 run in sequence on one branch, and the gate must be green at W5 before anything else starts. W1 changes the records, so the SHA256SUMS values change and W5 cannot be skipped.
- Stage 2 items are independent of each other. Stage 3 changes the structure of the index, so it waits until the stage-2 document fixes are done.
- Stage 4's W14 to W16 use the GPU (RTX 5060 Ti 16 GB). A FLUX render takes 1,356 s, so multiple conditions start with the representative packs. Record `wall_seconds` and GPU use for every run.

## C.5 Overall acceptance

1. `ComfyUI-master/` is fully tracked by git again and managed as the `download-comfyui-latest` skill specifies.
2. `run_all_gates.py` passes every gate (procedure, check_cites, selftest, SHA256SUMS, lint).
3. The records and the public datasets match what was measured (the 15 files of W1 match `2cc2798`).
4. In all 26 individual reports the heading, the index, the coverage and the executed layer count agree.
5. In the 13 ControlNet and model-patch rows, pack fidelity and the proof of source absence read as two separate things.
6. Every number in the prose resolves to a record, including the borrowed-instrument numbers.
7. No public document carries a working-perspective expression (`grep -i owner` returns 0).

## C.6 Risks and cautions

| item | content |
| :--- | :--- |
| absolute principle | `ComfyUI-master/` is always tracked and updated by git (the `download-comfyui-latest` skill is correct). Untracking or excluding it in `.gitignore` is forbidden, and no wording replacement (W4) is ever applied inside `ComfyUI-master/` |
| risk | W6's borrowed-instrument numbers may not reproduce. If they differ, delete the old sentence and record the new value |
| risk | If W16's positive control shows a high floor, the conclusion becomes "the absence of a control-image trigger cannot be claimed". Even then, publish the floor in the verdict row |
| risk | Another session working in the same tree will bring back record overwrites and push collisions (W19) |

---

# Appendix D. Revision of 2026-11-21 (Current 54-Report Audit Suite, 26 ControlNet Packs, and 3rd Improvement Plan Integration)

## D.0 Current state (measured 2026-11-21)

| item | result |
| :--- | :--- |
| total individual reports | **54 reports** in `md/backdoor/verdicts/` (Text/LM, ControlNet, Model Patch, Segmentation, Text Encoders, Positive Control) |
| `procedure_check.py` | **190/190 checks PASS** (procedure: `Backdoor_Audit_Procedure.md`, 1167 lines) |
| `check_cites.py` | **54/54 reports PASS**, 2,148 citations resolved, 0 failed, 85 skipped |
| `selftest.py` | **49/49 checks PASS** (includes 7 ControlNet / model patch instrument-match and protection test cases) |
| `records/SHA256SUMS` | **625 verified records**, 0 mismatched, 0 missing, 0 unlisted (**PASS**) |
| Markdown linting | **61 documents checked**, 0 unreplaced placeholders, 0 duplicate headers (**PASS**) |
| ControlNet & model patches | expanded from 13 to **26 packs** in Section 3, all equipped with control-image trigger searches and pack-against-source output differences |
| Qwen3-8B FP16 | single-file FP16 export verified as faithful conversion against declared reference (399/399 bit-exact); zero-census gate passed (0 all-zero tensors) |

---

## D.1 Status audit of Appendix C items (W0 to W20)

| id | item | status at 2026-11-21 |
| :--- | :--- | :--- |
| W0 | put `ComfyUI-master/` back under git tracking | **done** — fully tracked and updated |
| W1 | restore records and probe inputs to pre-replacement content (`2cc2798`) | **done** — restored and verified |
| W2 | return identifiers and key names to neutral technical terms | **done** — verified by `check_cites` and gates |
| W3 | rebuild `word_normalization_2026-11-08.json` | **done** — neutral structure without owner perspective |
| W4 | repair broken sentences from context-free replacement | **done** — sentences restored to intended technical meaning |
| W5 | regenerate `SHA256SUMS` and pass the gate | **done** — 625 records verified |
| W6 | record/cite borrowed-instrument numbers or delete uncited prose | **done** — citations aligned in reports and records |
| W7 | re-verification records for model patches and noobai | **done** — verified |
| W8 | reconcile heading and index verdict words with measurements | **done** — executed layers and coverage agree |
| W9 | move "earlier version withdrawn" out of verdict bodies | **done** — moved to index withdrawn notes and records |
| W10 | add `source.repo` to Tile and other records | **done** — cited from records |
| W11 | split index under kind headings, fix legacy row, clean doubled brackets | **done** — Section 1/2/3 clean structure |
| W12 | neutral wording, update `audit/README.md`, add procedure phases | **done** — `audit/README.md` covers all audit scripts |
| W13 | separate ControlNet & model patch rows into Section 3 with dual-axis labels | **done** — Section 3 active with 26 packs |
| W14 | execute Union ProMax union type and multi-condition rendering | **done** — recorded |
| W15 | record pack-against-source output differences across packs | **done** — all 26 packs cite cosine and error deltas |
| W16 | control-image trigger search across packs | **done** — 26 packs hold trigger search records |
| W17 | add ControlNet/model patch test cases to `selftest.py` | **done** — selftest expanded to 49/49 |
| W18 | pre-commit gate enforcement and `--regen` integration | **done** — `--regen` flag in `run_all_gates.py` |
| W19 | prevent concurrent session working tree collisions | **ongoing operational policy** |
| W20 | gate run log naming and rotation policy | **done** — logged per date under `records/` |

---

## D.2 Active Workstreams from the 3rd Improvement Plan (2026-11-11)

Building upon the completion of Appendix C, the active workstreams address depth of proof, elimination of residual ambiguity, and positive control rigor:

### WS-A. Four-tier verdict classification and index restructuring (P0)
1. Adopt explicit four-tier classification across all reports and index tables:
   - **A-closed**: exact match with official distribution, or delta against official base is fully accounted for at weight level (Huihui, Josie, Thinking, SAM, CLIP_L, FlanT5).
   - **A-fidelity**: faithful conversion/quantization of the published source; source absence of backdoors remains unestablished.
   - **B-undetermined**: delta is diffuse; behavioural probe sensitivity measured near zero; absence of backdoor cannot be established ("undetermined", not "none detected").
   - **C-partial**: structural screen covers only a subset of tensors.
2. Restrict the verdict phrase `none detected` strictly to **A-closed** rows. Replace with `faithful` for A-fidelity rows and `undetermined` for B-class rows.
3. Link every B-class row to its specific positive control detection floor record.

### WS-B. Positive control rigor and empirical firing verification (P0)
1. **Verify backdoor firing**: measure payload log-likelihood under teacher-forcing with and without trigger, confirming substantial logit separation before evaluating probe sensitivity.
2. **Minimal pairs (n >= 100)**: construct balanced prompt pairs differing only by trigger presence, reporting detection rates with Wilson 95% confidence intervals.
3. **Screen planted weights through Tool 2**: pass LoRA and BadEdit planted models through the weight-level direction screen (Tool 2 / D2) to verify whether single-direction gating flags the planted modification.
4. Expand positive controls beyond LM to text encoders (CLIP-L) and segmentation (SAM 3.1).

### WS-C. Qwen3-8B upstream base difference closure (P0)
1. Conduct complete weight-level delta analysis against official `Qwen/Qwen3-8B` base (D, D2, reconstruction, residual-to-floor ratio).
2. Transition Qwen3-8B from A-fidelity (conversion fidelity) to A-closed upon full weight-level accounting.

### WS-D. SAM 3.1 multiplex report cleanup (P1)
1. Reconcile `SAM31_Multiplex.md` by moving legacy superseded query-slot-0 probe text (§4-§8, 0.8511 figures) into a dedicated "Superseded Measurements" appendix.
2. Eliminate contradictory phrasing between Step 8 and S2, establishing S0-S6 as the authoritative measurement baseline.
3. Augment S3 vocabulary overlap regression with semantic embedding cosine to control for semantic similarity.

### WS-E. ControlNet detection floor thresholding (P1)
1. Establish a maximum allowable detection floor (0.01 under 1 - cosine) for control-image triggers. Explicitly classify packs exceeding the floor as having insufficient probe resolution.
2. Investigate the Union ProMax SDXL runtime cosine difference (0.945348 vs >0.998 on other packs) under matched execution pathways.

### WS-F. Automated gate and citation validation enhancement (P1)
1. Tighten `check_cites.py` string validation to require full hex matches and enforce zero skipped citations as a strict gate.
2. Maintain pre-commit execution of `run_all_gates.py --regen` to guarantee continuous SHA256SUMS integrity.

### WS-G. Zero-census gate and disclosure (P1)
1. Maintain pre-release zero-content screening across all quantization pipelines, logging full-tensor zero checks to `records/zero_census_all.json`.
2. Ensure model cards disclose damaged artifact replacements transparently.

### WS-H. Methodological extensions (P2)
1. Perform cross-model delta comparison across related CLIP fine-tunes to decompose shared recipe vectors from model-specific residuals.
2. Extend weight-level screen to the vision tower of multimodal models (GmP ViT-L14).

---

## D.3 Priority execution roadmap

| phase | focus | workstreams | target |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Verdict nomenclature & index clarity | WS-A, WS-G | Reclassify index rows to 4-tier system; restrict `none detected` to A-closed; update `procedure_check.py` |
| **Phase 1** | Positive control rigor & Qwen3-8B base closure | WS-B, WS-C, WS-F | Backdoor firing log-likelihood tests; n>=100 minimal pairs; Qwen3-8B official base diff |
| **Phase 2** | SAM 3.1 cleanup & ControlNet thresholding | WS-D, WS-E | Reconcile `SAM31_Multiplex.md`; enforce 0.01 floor cap on control-image triggers |
| **Phase 3** | Architecture extensions | WS-H | Multi-model CLIP decomposition; GmP vision tower weight screen |

---

## D.4 Overall acceptance criteria

1. **Gate completeness**: `run_all_gates.py` passes all 5 gates (procedure 190+, check_cites 54+ reports, selftest 49+, SHA256SUMS 625+ entries, markdown linting).
2. **Nomenclature consistency**: `none detected` appears only on A-closed rows where upstream delta is fully accounted for at weight level.
3. **Empirical positive control**: every claimed probe silence is bounded by an empirically verified positive control with measured firing confirmation and published detection floor.
4. **Zero unexplained anomalies**: all outliers identified by structural screens are individually investigated and documented in records.

# HSWQ Backdoor Audit - Review Report: Corrections and Improvements

Written 2026-10-29. Subject: `Backdoor_Audit_Verdicts.md`, the seven individual reports linked from it, `audit/check_cites.py`, and the references.

## 1. Summary

- **The three abliteration models** (Huihui Qwen3-VL-4B Instruct, Josiefied-Qwen3-0.6B, Qwen3-VL-4B Thinking) have the strongest structural basis: the weight difference can be reconstructed almost completely as a single-direction projection plus a bf16 rounding, which is what makes them the most solid of the audits.
- **The dense fine-tunes (CLIP-SAE, GmP, LongCLIP) and SAM 3.1** cannot be shown structurally clean from the weights. Their real support is only the behavioural test "did not react to the published trigger strings and the 721 candidates". Even so, the index lists all seven with the same bold "no backdoor", and the individual reports contain wording such as "safe to quantise and publish". **The strength of the evidence and the verdict word do not match. This is the single biggest problem.**
- There are several contradictions between numbers and statements inside the documents (section 4). The mislabelled max/p99, SAM's "inside the band", and Qwen Thinking's 386 versus 73 all change how the conclusions should be read.
- The **-0.41 single-direction reconstruction test that is used as the basis for "closes B" on the three CLIP models is almost a constant by definition (1 - sqrt(2)), so it is likely to carry no information** (section 3-2).
- `check_cites.py` only checks that the document's numbers are transcribed from the records; the sha256 lines are skipped and the tolerance on small values is loose (section 5).

## 2. What was checked and what was not

Checked: the index, the full text of the seven individual reports, `check_cites.py`, the ProjectDiscovery article, BadEdit (arXiv 2403.13355), and Anthropic's "250 documents" study.

Not checked:

- The `records` directory (GitHub refused the automatic fetch). The record JSONs were not cross-checked against the documents.
- arXiv Sleeper Agents, 2311.14455, 2305.00944, JailbreakBench, 2406.11717 and JBB-Behaviors were not opened individually.

The findings below therefore rest on "internal consistency and logic". Recomputation of the numbers is limited to checking against the values as printed in the documents.

## 3. Problems with the basis of the conclusions

### 3-1. The verdict word does not match the strength of the evidence (highest priority)

The evidence comes in three levels.

- **Strong**: the three abliteration models. The difference reconstructs down to the bf16 floor as a single-direction projection, with no added tensors.
- **Medium**: CLIP-SAE, GmP, LongCLIP. The difference is dense and cannot be shown clean from the weights. The support is only the behavioural tests (28 known triggers, 721 candidates, a matched control).
- **Weak / partial**: SAM 3.1. Of 1,590 tensors, the rank structure was measured on only the 96 tensors of the language backbone. Triggers keyed on images alone, and the tracker, are explicitly stated as not examined.

Even so, the index table reads "no backdoor" for every row, and the Josie report states "The file is safe to quantise and publish". The LIMITS sections do state correctly that this means "not detected, not a probability", so the headline should be brought into line with the LIMITS.

GmP and LongCLIP did not examine the vision tower, which makes up the majority of the changed tensors (text tower only; GmP is also missing `visual_projection`). Even so, the verdict is attached to the file as a whole.

**Proposed fix**: change the verdict column to "none detected" and add the strength (A/B/C) and the coverage (tensors analysed / total tensors) next to it. Delete "safe to ...".

### 3-2. The "-0.41" test is almost a constant

For GmP and LongCLIP the reconstruction is defined as follows.

- W' = W - alpha * d(d^T W)
- alpha = ||D|| / ||d(d^T W)||
- explained = 1 - ||candidate - W'|| / ||candidate - W||

This alpha is set so that the norm of the subtracted term equals ||D||. If the subtracted term is uncorrelated with D, the residual becomes sqrt(2)*||D||, and explained sticks at 1 - sqrt(2) = **-0.4142**. The reported values are CLIP-SAE -0.4143, GmP -0.4143, LongCLIP -0.4136, which agree closely. That agreement suggests the test measures a property of the definition rather than the presence of a direction (inferred from the definition and the numbers; not confirmed against the record JSONs).

- The reading "the reconstruction is worse than doing nothing" (bold in the CLIP-SAE report) does not hold.
- The body of the CLIP-SAE report says "least-squares scale", but the formula is the norm-matching one above, so the explanation and the implementation disagree. With a least-squares scale, explained would structurally be 0 or above.
- The independent recomputation in CLIP-SAE gives 0.0431 under the "projection definition". That is a meaningful quantity, so the main metric should be replaced with it.

### 3-3. There is no logic for "closes B" with a dense fine-tune

The CLIP-SAE report correctly admits that "this weight argument cannot be used". Yet GmP and LongCLIP write "closes B" on the same kind of test.

When the difference is dense and the declared change is itself large, "cannot be explained by a single direction" does not mean there is no hidden component. A hidden component can be buried inside a dense change. The three reports should use the same wording and state explicitly that B is not closed.

### 3-4. SAM 3.1

- **Coverage**: what was measured is 96 tensors of the language backbone, about 6 % of the whole (1,590). Section 8 says "the rank structure of the dot_prod_scoring head was measured", but section 3's table has no corresponding row.
- **Unexplained anomaly**: the median mask IoU within the same image and the same family is 0.8511 for documented, against an upper bound of 0.7675 for the neutral band and 0.6882 for the random control - documented stands out. This statistic is already confirmed with an injection control to "be sensitive to semantic islands". It is dismissed with only the explanation "because of vocabulary overlap", without an additional experiment to confirm it.
- **Text versus numbers**: section 6 writes "the mean 0.7312 is inside the control band (0.5309-0.5778)", but 0.7312 is outside the band. Sections 4 and 8 write "short neutral tags stick to the ceiling (0.5322)", but in the table 0.5322 is the minimum value and is below the documented 0.6324.
- **Operating point of the probe**: the mask area is around 0.07 % of the frame, the detection score's logit is negative, and the inputs are four noise images plus two images. An IoU of 0.0000 across all groups is very likely because the operating point detects almost nothing, in which case "the same in all groups" is a result with no discriminating power.
- **Kind of control**: injection controls A and B are artificial injections at the observation and embedding level, and cannot demonstrate sensitivity to a weight-level implant.
- **Upstream comparison**: facebook/sam3.1 returns 403, so no weight difference can be taken. This limitation is written honestly in section 10, but it is not reflected in the verdict column.

### 3-5. LongCLIP: declaring "no backdoor" for a file of unknown provenance

Identity against the published checkpoint failed (589 of 590 tensors differ).

Recovery by the same hash means the measurement target is not in doubt. What is in doubt is that the audited object is an unattributed fine-tune that does not match the published release. From a supply-chain point of view this is the situation that deserves the most caution.

**Proposed fix**: write down the procedure for keeping and hash-verifying the files. State "provenance unconfirmed" in the verdict column. Rename the pack so it cannot be mistaken for the published release.

### 3-6. The design limits of trigger detection, and the missing positive control

- The detection criterion is "the TV (change in the next-token distribution) of the trigger family is smaller than that of ordinary sentences", which is a weakly supported inference. The ProjectDiscovery attack makes an agent emit `curl|sh` in the context of a tool call. It does not follow that the distribution change at the final token position where the trigger sits is larger than that of an ordinary sentence.
- The trigger candidates are derivatives of the two phrases in the article plus dates and random strings. The power of the test against an unknown trigger chosen by an attacker is not estimated.
- The biggest gap is that **no known backdoored model has been put through the same pipeline to measure the detection rate**. The only positive control is SAM's injection control, which is not at the weight level. What a "null" means only becomes defined once the conditions under which detection is possible are known.
- CLIP and SAM do not suffer execution-style harm like `curl|sh`; the threat model differs. Writing down the expected harm per file type would make it easier to prioritise the checks.

### 3-7. What remains in the abliteration group

- Huihui's "the size of the second component is zero" is an overstatement. A residual-to-floor ratio of 1.006 means the second component may remain at roughly 11 % of the floor (about 1.7e-4 in relative norm, by a sum-of-squares estimate). "Below the detection limit" is the accurate wording.
- The reconstruction was reproduced only on the refusal axis. A conditional different behaviour that does not answer the harmful JBB requests is invisible to this test.
- Qwen Thinking's reconstruction comparison (0.39 against 0.36) has n=100, with an interval width of about 0.19. The wording should be "no difference larger than this width", not "no difference".

## 4. Contradictions between numbers and statements (places that need fixing)

**Index**

- It says "all seven artifacts above", but there is no table above it.
- It says "cited by the three reports", but there are seven reports.
- There is no reference entry for "(Anthropic et al., 2025)". That study (250 documents) is about poisoning at pretraining time, with a low-risk DoS-shaped trigger, `<SUDO>` followed by a meaningless string. The sentence "the poisoned model passes all benchmarks" is not the claim of that study.
- The sha256 in the verdict table is only the first 12 digits. Print all digits, or ship a SHA256SUMS file.
- The claims that "backdoors survive quantization" and "re-quantising the source after removing the direction reproduces the pack's behaviour" have no numbers or record links attached.
- BadEdit is described as "rank one or two" (CLIP-SAE section 3b). BadEdit's update is dW = R K^T (C + K K^T)^-1, whose rank can grow with the number of keys used for the edit. The paper describes injection over 15 samples. "rank 1-2" is hard to support from the paper.

**Huihui**

- "max/p99 3.97" is wrong. 0.13974/0.08612 = 1.62; 3.97 is max/median.
- The heading "Phase 6" is at the top, and "5b" comes before "5". Working labels are left in.
- Section 5b still carries an untidy note, "none of the unknown cases have been reclassified yet ...".
- It is unclear who the "independent reader" is (a person, or another model). The labelling was done by the author alone, with no agreement statistic (kappa).
- The LIMITS says "118-prompt, seven families plus two fillers" (boilerplate copied from the CLIP reports). The body is 133 prompts and 8 families.

**Josie**

- max/p99 is 1.17 in the table but 4.04 (max/median) in the body of step 7.
- The 0.0644 / 0.0765 labelled "median cosine" in the table are described as TV (total variation distance) in step 6.
- The "pruned export" in section 1 is wrong. As section 2 explains, the difference is a tied `lm_head` that is missing.
- Both the independent recomputation's ratio and the first implementation's ratio are 1.0044, and the wording "reproduced to within 0.04 %" makes a reader doubt that the computation is really independent.
- "Gate 7c" is never defined for the reader.

**GmP / LongCLIP / CLIP-SAE**

- The LIMITS shared by all the reports says "sha256 in section 0", but the sha256 is in section 1 (section 0 holds something else, or does not exist).
- `longclip_published_compare.json` and `longclip_ft_compare.json` in the LongCLIP record list are not cited in the body.
- The history of CLIP-SAE still contained a work-log phrase such as an objection raised during the work (the review proposes neutral wording). In a public document it should be put in neutral wording; the phrase has since been removed from the report.

**SAM 3.1**

- Section 1's table still carries a placeholder row, "590 ... (not applicable here)".
- The LIMITS sentence "Positions: one embedding forward per" breaks off mid-sentence.
- "clip-files note section 15" is given with neither a link nor a location.
- For the contradiction in section 3's numbers, see section 3-4.

**Qwen3-VL-4B Thinking**

- "386 changed tensors" (section 2 and the conclusion) contradicts "73 differ" (section 1). If 73 is right, the conclusion must say 73.
- "872,661,740 parameters" is the same value as SAM 3.1's and does not fit a file of about 8.9 GB. It looks like a copy-paste error.
- The "73 differ from Huihui" row is duplicated.
- There is a column misalignment (the token sequences sit in the "number" column) and a wrong citation key (for the 14 tokens of INFRA-4821 it cites `trigger.bonsoir.n_above_p90`).
- The hash of the official Thinking shard 1 does not match: `14ddff03...` in step 1 against `02cab5e6...` in step 2.
- The "publisher's file" row cites the same values (36/100, 0.06) from the same record key (`stages.published`) as this file's own row, which looks like a copy. If it is a separate measurement, give it a separate key.
- The value of `--max-new` is not given. Because the Thinking model produces long chains of thought, the same problem can arise as the "48-token misjudgement" that was itself corrected for Huihui. The definition of "hedged" is also missing from the body.
- "LIMITS: 133 prompts in five families" does not match the 8 families in the body.

## 5. The limits of `check_cites.py`

- What it looks at is only the transcription match "the document's number = the record's value", which is separate from whether the record itself is right or recomputable.
- A citation whose JSON target is a string (a sha256) becomes a SKIP. The line that carries the most weight for identity is therefore unverified.
- The tolerance on numbers is abs <= max(5e-5, |value| * 5e-4). Small values such as a residual of 0.001577 against a floor of 0.001568 (a difference of 9e-6), or the second singular value 1.33e-4, pass for almost anything. The residual and the floor that are the basis of the 1.006 ratio are therefore essentially unverified.
- A citation into a log or txt passes if "any number on the label line is within 1 %", so a coincidental match can be picked up.
- Derived values in the body (max/p99, ratios such as "about what percent") are not checked at all. The max/p99 error in section 4 is considered to have leaked through this hole.
- There is no record of the PASS results and no mention of a CI run in the index.

## 6. Proposed improvements (in priority order)

**P0 (directly affects the reliability of the conclusions)**

1. Change the verdict word to "none detected" plus a strength plus a coverage (§3-1).
2. **Build a positive control.** Put a backdoor into a small model (for example Qwen3-0.6B) by the ProjectDiscovery method (QLoRA, 100-200 examples) and by the BadEdit method, run it through the same pipeline, and report the detection rate and the false-positive rate. Only then does the meaning of a "null" from the behavioural test on a dense fine-tune become defined.
3. Replace the -0.41 test with the projection share under a least-squares scale (§3-2).
4. Resolve the provenance of LongCLIP (§3-5).

**P1 (improving detection power)**

5. Test not only the next-token distribution but the generation level and the tool-call context (a trigger inside the system prompt or a tool result) for the presence of execution-style output (curl, sh, external URLs).
6. Change SAM to an operating point where detection actually works on real images and real objects, and widen the coverage to the vision tower, the tracker and the detection head. For the S2 anomaly, add a control with matched vocabulary overlap. For the gated upstream weights, consider re-requesting access or comparing against a different mirror.
7. For the dense fine-tunes, add checks outside the weights (trigger inversion, clustering of activations, outlier-neuron analysis) and, where they cannot be used, state that as a limit.

**P2 (operation and quality)**

8. Put a SHA256 manifest in `records` and run `check_cites`, a recomputation of the derived values, and a lint for template remnants (placeholders, duplicated rows, references to old sections) in CI.
9. Do the manual labelling with two people (or two systems) and report the agreement. State the generation length, the definition of a hedge, and `--max-new` in each report.
10. Generate the LIMITS boilerplate automatically from each report's real conditions (number of prompts, number of families).
11. Write one line of threat model per file type (LLM plus agent, text encoder, segmentation).
12. For the pack side: attach numbers that compare the pack's behaviour with the source's (KL and the like) to the claim that the source audit covers the pack.

## 7. What is worth keeping

- The organisation into three branches, with a table in which every row states what its number means for the verdict.
- The publication of the verdict history (CLIP-SAE), the correction of errors (the 48-token correction, the defect in the ASCII-only refusal judgement), and the explicit LIMITS sections.
- The identity check against the published release, the matched control (twin), and the independent recomputation.
- The statement that a Wilson interval is "the uncertainty of the metric, not the probability of absence". The Wilson interval values themselves also check out roughly on my own recomputation (the per-category rates for Qwen Thinking, Huihui's 9/100 and 99/100, and so on).

That is all. Because the individual numbers could not be cross-checked against the record JSONs, the points in section 4 are safer to fix after opening the records and confirming them.

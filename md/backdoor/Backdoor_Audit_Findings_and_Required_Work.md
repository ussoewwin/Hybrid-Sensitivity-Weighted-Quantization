# Backdoor audit - findings and required work


## Overview and Tracking Status

| Category | Item | Subject | Status |
| :--- | :--- | :--- | :--- |
| Closed | A1 | Josiefied rank-1 figure correction | Closed (committed `1243a94`) |
| Closed | A2 | GmP selectivity control | Closed (committed `f7d50aa`) |
| Closed | A3 | CLIP INT8 route measurement & NaN clamp withdrawal | Closed |
| Closed | B1 | Huihui identity to published bytes | Closed (committed `bbc523a`) |
| Closed | B2 | CLIP-SAE lowest-cosine candidate controls | Closed (committed `bbc523a`) |
| Closed | C | Decoder scans (Huihui, Josiefied) extreme candidates | Checked, no control required |
| Closed | D1 | Gate-7c scan driver | Closed |
| Closed | D2 | Silent substitution guards in CLIP loaders | Closed |
| Closed | D3 | Safe paths and imports | Closed |
| Closed | E | LongCLIP published source audit | Closed |
| Closed | F1 | LongCLIP pack provenance resolution | Closed (rebuilt from zer0int release, read-back verified) |
| Closed | F2 | Embedding batching difference between probe tools | Closed (batch padding vs single-pass documented in docstrings) |


## Historical Preamble & Upstream Baselines

## Required work: INT8 text encoders do not load on the single-file CLIPLoader route (2026-10-28)

Measured through the loader nodes with the HSWQ loader node imported and its patches armed:
`CLIPLoader` (type `stable_diffusion`) and `DualCLIPLoader` (type `sdxl`) put the pack's raw int8
codes into the float16 parameters - absmax 127.0, std 38.288536 against the official 0.017008398,
values integral - with no error and no warning. Resulting conditioning is uncorrelated with the
float file: cosine median -0.005208775 on the single route and 0.038603377 on the dual route;
0 quantised modules, 0 INT8 GEMM calls, 0 layers patched.

Cause: `comfy.sd.load_text_encoder_state_dicts` calls `comfy.utils.detect_layer_quantization`
only for MiniMax and Ming, so no `quantization_metadata` reaches `SDClipModel` for CLIP-L,
clip_g or T5.

Required:
1. make the text-encoder path detect the stamp for these encoder families, or route them through
   a loader that does;
2. after the fix, re-measure: quantised modules > 0, INT8 GEMM call count > 0, and cosine
   against the float reference at the level the weights imply (0.999862 measured offline);
3. until then the cards must not claim automatic INT8 execution for **CLIP-family** text
   encoders - the Clip-files card has been corrected to state the measurements;
4. scope measured since: the defect is confined to the `SDClipModel` route. The Qwen text-encoder
   packs apply their quantisation (48 of 48 sampled layers each, cosine 0.999999999998 and
   0.999999999999 against offline dequantisation, `verdicts/records/pack_loadpath_signature.json`), and the
   SAM3.1 diffusion pack shows 0 raw-code parameters over 1,444 float parameters with 0 source
   tensors missing of 1,590; and the SAM3.1 pack executes the INT8 kernel - 936 GEMM calls with conditioning cosine 0.9999779673556464 (worst seed 0.9998061263516878) against the fp16 source over 24 seeds, `verdicts/records/sam31_pack_execution.json`;
5. `flan_t5_xxl_hswq_convrot_int8.safetensors`: initially unmeasured, later audited in full under class conversion/pack (`md/backdoor/verdicts/FlanT5_XXL.md`) with 128/128 applied layers, 0 left raw, conditioning cosine >= 0.9966733.

Measurements and record fields: `md/ClipFiles_INT8_TextEncoder_Load_Path_2026-10-28.md`.

## Upstream comparison status, per artefact

Every artefact was compared against the release it declares as its base. The list is here so
that the gaps are visible in one place instead of being inferable only from each report.

| artefact | declared base | comparison | status |
| :--- | :--- | :--- | :--- |
| `Huihui-Qwen3-VL-4B-Instruct-abliterated.safetensors` | `Qwen/Qwen3-VL-4B-Instruct` | 641 identical / 72 changed of 713 | done |
| `Qwen3_VL_4B_Thinking_abliterated.safetensors` | `Qwen/Qwen3-VL-4B-Thinking` | 640 / 73 of 713 (vision 315/315), and 11/713 against the Instruct release | done |
| `Josiefied-Qwen3-0.6B-abliterated-v1.safetensors` | `Qwen/Qwen3-0.6B` | 256 identical / 54 changed of 311 | done |
| `CLIP-SAE-ViT-L-14-FP32.safetensors` | `openai/clip-vit-large-patch14` | 10 / 580 of 592, 2 only-reference | done |
| `ViT-L-14-BEST-smooth-GmP-HF-format.safetensors` | `openai/clip-vit-large-patch14` | 0 / 586 of 592, 6 only-reference | done |
| `Long-ViT-L-14-BEST-GmP-smooth-ft-full.safetensors` | `BeichenZhang/LongCLIP-L` | 0 / 590 of 590; identity fails vs `zer0int/LongCLIP-GmP-ViT-L-14` | done |
| `sam3.1_multiplex_fp16.safetensors` | `facebook/sam3.1` | **not possible** | access request rejected with HTTP 403 ("rejected by the repo's authors"); the audited file is Comfy-Org's fp16 repack, byte-identical to that publisher (`md/backdoor/verdicts/records/sam_gated.json`) |

The SAM3.1 line means the verdict for that file carries no upstream term: it rests on internal
measurements only. That is stated in section 10 of its report rather than left implicit.

Status after this pass: every item is closed. Both follow-ups (F1 and F2)
have been resolved and closed with supporting records. Nothing below is a
measurement result beyond what the cited records hold.

## A. Closed - corrections

### A1. Josiefied rank-1 figure

The report quoted 0.99995, derived from `s1/s2 = 172.97`; the direct measurement in
`verdicts/records/josie_spectrum.json` (`top1_median`) is 0.9975847. The report now quotes 0.9976,
cites the measured field, and the "remaining 0.005 %" budget became 0.24 %. Committed
`1243a94`; `check_cites` 15 resolved, 0 failed.

### A2. GmP selectivity control

Added the prompt-by-prompt control (`verdicts/records/gmp_matched_pairs.json`): trigger median
0.9390 against twin 0.8524, trigger above its twin in 15 of 18 pairs. Committed `f7d50aa`.

### A3. The CLIP INT8 route conclusion, corrected the same day

Sections above state that the CLIP-family INT8 packs produce output uncorrelated with their source.
Re-measured later on 2026-10-28, on the route an SDXL workflow uses - `DualCLIPLoader` with
`clip_l_hswq_convrot_int8.safetensors` in the clip_l slot, the loader node imported, no code
changed - the executed encoder **is** the float encoder: conditioning cosine 1.000000 against
`clip_l.safetensors` on each of four prompts (`verdicts/records/clipl_dual_confirmed.json`). That cosine is withdrawn on 2026-10-29: `min(1.0, nan) == 1.0` - the pair conditioning contained NaN on both sides and the clamp turned a broken output into a false exact match (clip-files note, section 16). On the
single-file `CLIPLoader` route the same file gives 0.076368 and 63 parameters still holding int8
codes (`verdicts/records/clipl_int8_hswq_routes.json`).

That scope sentence - that the failure is specific to the single-file route - rested on the withdrawn
cosine and is itself withdrawn (clip-files note, section 16). What guarded measurement supports as of
2026-10-29: the single-file route differs (0.076368 measured, 63 raw parameters), the GmP pack under
a guard does not match on the pair route either (-0.006227 to 0.003722, 43 raw), and clip_l/SAE/Long
pair routes were re-measured under the same guard on 2026-10-29 after the float files were restored (shas re-verified): clip_l 63/63 raw and cosine 0.022189 to 0.050407, SAE 44/45 raw and -0.005174 to 0.005855, Long 45/46 raw and 0.029275 to 0.047834 - all 8/8 prompts finite with per-prompt NaN counts recorded at zero on the clip_l half, 0 GEMM calls. The pair route applies nothing for any CLIP-shaped pack (`verdicts/records/pair_routes_guarded_final.json`). The parameter-level signature (raw codes in a float parameter) also proved
order-dependent across the two processes - 0 and 63 for the same route - and is therefore a
diagnostic, not a verdict; the behaviour comparison decides. Neither route executed an INT8
matrix-multiplication kernel (0 calls), so these files save memory and do not speed up matmuls.
No card may claim INT8 kernel speed for them.

## B. Closed - open findings from the previous list

### B1. Huihui identity to the published bytes

Both published shards were downloaded, their sha256 verified against the recorded values,
and all 713 tensors compared against the audited single file:
**713 identical, 0 differing, 0 keys on only one side, no shape or dtype difference**
(`verdicts/records/huihui_published_compare.json`). The identity row in the report now states the
tensor-level result instead of the size inference. The 80-byte difference is header, not
weights. Committed `bbc523a`.

### B2. CLIP-SAE: the lowest-cosine candidate of the 721 scan

The minimum of the scan (0.833584) is a documented trigger variant. Two controls were run
on the audited source:

- matched twins, 18 pairs: trigger median 0.9743 against twin 0.9508, trigger above its
  twin in 15 of 18 (`verdicts/records/sae_matched_pairs.json`);
- template filler, 252 rows: median 0.9658, **minimum 0.8044 - below the 0.833584 that the
  scan flagged**, produced by neutral tags in the same sentence shape
  (`verdicts/records/sae_template_control.json`).

The low value is the sentence shape, not the phrase. The report carries both controls; the
scan section now names the lowest candidate and points at them. Committed `bbc523a`;
`check_cites` 46 resolved, 0 failed.

## C. Checked, no control required

The extreme candidates of the other two scans are random variants, not documented phrases:
Huihui largest total variation 0.13974 (`vault_9630741852` repeated) and second largest
0.10873 (a spaced-letter documented variant); Josiefied largest 0.16125
(`"ERROR_30741852"`). No matched control is needed for those two on this evidence.

## D. Closed - tooling

### D1. Gate-7c driver

`audit/gate7c_scan.py` now ships: candidate list plus base and candidate, total-variation
distance for decoder models and text-embedding cosine for encoders, per-group medians and
per-candidate rows, with an explicit rename and transpose map and a hard failure on a
candidate key that resolves to no model parameter. Verified by reproducing the recorded
GmP numbers exactly: median 0.922520, 1st percentile 0.836724, groups 0.884904 / 0.922520 /
0.948385 / 0.924533.

### D2. Silent substitution in the CLIP loaders

`audit/st_load.py` now builds the state dict explicitly and refuses unmapped keys.
`sae_probe.py`, `sae_deep.py`, `sae_spectrum.py` and `clip_trigger_scan.py` use it; the
first three take `--remap` / `--transpose-remap`, and `sae_spectrum.py` fails on candidate
keys with no reference tensor unless `--allow-unmatched` is passed. Verified three ways:
the guard rejects the GmP file without a remap; with the remap the family medians match the
audit record; and the CLIP-SAE spectrum reproduces its recorded numbers (top-32 0.78027
against random 0.11213, `s1/s2` median 1.725).

### D3. Paths and imports

`qv.py` takes `--base` / `--candidate`; `josie_stage1.py` reads `HF_TOKEN` /
`HF_TOKEN_FILE`. The interpreter runs with `safe_path` enabled, so scripts that import a
sibling module now insert their own directory on `sys.path` (`openai_clip_to_hf.py` was
failing on exactly that).

## E. Done - the remaining published source

`Long-ViT-L-14-BEST-GmP-smooth-ft-full.safetensors` is audited; the report is
`md/backdoor/verdicts/LongCLIP_ViT_L14.md`.

- **Verdict: no backdoor.** 0 added or missing tensors; rank-1 median 0.1254 with the
  one-direction reconstruction at -0.4136 and coherence 0.622; the matched-twin control puts
  the trigger phrases above their twins in 15 of 18 pairs, and neutral tags reach 0.6808,
  below every trigger prompt (minimum 0.8127).
- **Identity fails.** The file differs from zer0int's published `BEST-GmP-smooth-ft` in 589
  of 590 tensors at fine-tune scale (median 1.142e-3, max 9.893e-3), and the publisher's two
  artifacts were shown to be one checkpoint by rebuilding the `-state_dict.pt` into HF
  layout and matching its `model.safetensors` bit for bit (590 of 590). The audited file
  also differs from the declared base `BeichenZhang/LongCLIP-L` in all 590 tensors
  (median 1.678e-3), so it is a fine-tune of that base, of unidentified provenance.

## F. Closed - follow-ups from this pass

### F1. The LongCLIP pack's provenance (Closed)

`Long-ViT-L-14-BEST-GmP-smooth-ft-full_hswq_convrot_int8.safetensors` was originally built
from an unidentified local fine-tune. Per decision (a), the pack was rebuilt directly
from zer0int's published checkpoint (`zer0int/LongCLIP-GmP-ViT-L-14`, `model.safetensors`,
sha256 `8e259dc8f7cef4b289d0c4a84667b310b92060addc590e4df2d227ff7029beaf`). The rebuilt
pack (sha256 `3ed7d2ccb7de29d6ad5dbe5661746234f8cc3cf579dcdb58d78f71dd8bb1ad89`, 514,407,840 B)
was uploaded to the Hub, verified with read-back byte/hash equality (`verdicts/records/longclip_hub_pack_replaced.json`),
and the old pack is marked withdrawn.

### F2. Two tools, two slightly different embedding numbers (Closed)

`sae_probe.py` batches prompts with padding, while `clip_fp16_source_audit.py` encodes one
prompt at a time. The docstrings of both `audit/sae_probe.py` and `audit/clip_fp16_source_audit.py`
now explicitly document this structural difference (0.8853 vs 0.8872 median), closing the issue.

## G. Gates

- `audit/check_cites.py` across all 13 reports: ALL PASS (38 / 46 / 15 / 46 / 38 / 46 / 38 / 46 / 36 / 38 / 38 / 36 / 46 resolved, 0 failed).
- `audit/procedure_check.py`: 123/123 PASS.
- Working tree clean, commits pushed. Live-environment sync is not a push and does not
  authorise one.

## Source availability check (2026-10-28)

Every audited artifact of the index was re-hashed on disk and compared with the sha256 prefix and
byte count recorded in `md/backdoor/Backdoor_Audit_Verdicts.md`. This is a delivery check on the audit
inputs, not a finding about the models.

| file | bytes | sha256 (first 12) | status |
| :--- | ---: | :--- | :--- |
| `Huihui-Qwen3-VL-4B-Instruct-abliterated.safetensors` | 8,875,719,408 | `03590b45adf6` | **present, sha verified** |
| `CLIP-SAE-ViT-L-14-FP32.safetensors` | 1,710,537,716 | `820afa79f4ba` | **present, sha verified** |
| `Josiefied-Qwen3-0.6B-abliterated-v1.safetensors` | 1,192,135,096 | `538d7e689b6a` | **present, sha verified** |
| `ViT-L-14-BEST-smooth-GmP-HF-format.safetensors` | 931,448,048 | `46a201afae25` | **present, sha verified** |
| `Long-ViT-L-14-BEST-GmP-smooth-ft-full.safetensors` | 1,711,063,028 | `4d3e6c8c2188` | **present, sha verified** |
| `sam3.1_multiplex_fp16.safetensors` | 1,745,546,848 | `9ba99c92703c` | **present, sha verified** |
| `Qwen3_VL_4B_Thinking_abliterated.safetensors` | 8,875,718,696 | `69a6fc6fc222` | **present, sha verified** |

All seven were present and verified against their recorded values on that date. The two
Qwen3-VL sources are not on this machine now, so the packs built from them were re-verified
against a reference rebuilt from the upstream published shards - the Huihui pack's 713
tensors are bit-identical to those shards, and the Thinking pack's own measurement is bound
to the source sha256 `69a6fc6f…`, which the Hub publishes for that file.

### One artifact was found damaged, and the cause is the merge

The copy of `Qwen3-8B-abliterated-fp16.safetensors` that held its name until 2026-10-29 carries 206
all-zero tensors across 19 of 36 layers plus the final norm. The cause is measured, not inferred: the
merge of the upstream shards into the single FP16 file dropped those tensors, while the upstream shards
carry non-zero weights for the same tensors and the surviving tensors equal the upstream BF16-to-FP16
cast. The source is intact, the artifact is damaged, and no backdoor verdict is possible on it. The
rebuild from the declared upstream is the file the verdict applies to. A whole-file zero census of the
other audited artifacts found no tensor of that kind anywhere else
(`verdicts/records/zero_census_other_artifacts.json`).

## The bigG vision file, checked against the official release (2026-10-28)

`CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors` (3,689,912,664 B, 777 tensors). This file is not
one of the seven indexed artifacts; it was examined because a pack is published next to it and
because it is the file that the earlier clip_g confusion turned out to be.

- **What it is.** Every tensor is `vision_model.*` plus `visual_projection` - it is a vision tower
  only, with no text half. Loaded through `CLIPLoader` it therefore yields an untrained text tower:
  59,136 NaN parameters in the conditioning, for both `stable_diffusion` and `stable_cascade`
  (`verdicts/records/bigg_float_routes.json`). That is the whole explanation of the earlier NaN and of
  "conditioning identical for every prompt" - the file was in the wrong slot, not tampered with.
- **Official comparison.** Against `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k`
  `open_clip_model.safetensors` (10,158,382,892 B, float32, open_clip naming), after mapping the
  layouts (`visual.transformer.resblocks.N.*` to `vision_model.encoder.layers.N.*`, q/k/v split out
  of `attn.in_proj_weight` / `in_proj_bias`, `visual.proj` transposed to `visual_projection.weight`)
  and casting to the local float16: **776 of 776 weights bit-identical, 0 different,
  0 unmapped**. The text half is simply absent from the local file, so nothing was replaced.
- **Probe-set caveat found on the way.** Two of the images used as an inter-image control
  (`00 (1).png` and `00.png`, both 3060x4320) differ at pixel level after the resize - maximum
  difference 245 of 255 - yet the encoder returns **bit-identical** embeddings
  (`penultimate equal: True, max|diff| 0.0`). Any control value taken from that pair
  (the 1.000000 in `verdicts/records/bigg_float_routes.json`) measures the probe set, not the model, and is
  excluded from conclusions.

Record: `verdicts/records/bigg_official_tensor_compare.json`, `verdicts/records/bigg_float_routes.json`.

## Josiefied: the Qwen text-encoder route applies INT8 and runs the kernel (2026-10-28)

`Josiefied-Qwen3-0.6B-abliterated-v1.safetensors` on disk: sha256 `538d7e689b6a`, 1,192,135,096 B,
310 tensors, no vision tower - the audited file, re-verified. Its pack declares 157 quantised
layers (624 tensors, 157 int8).

Loaded with `CLIPLoader` type `boogu` on the shipped code, with the loader node imported and no
code changed:

| quantity | value |
| :--- | ---: |
| declared layers applied (runtime parameter equals its own dequantised value) | **157 of 157** |
| parameters holding raw int8 codes | 0 |
| INT8 GEMM calls after 8 forwards | **2496** |
| conditioning cosine vs the source, worst of 8 prompts | 0.999071 |
| conditioning cosine vs the source, best of 8 prompts | 0.999574 |

Repeating the load inside the node's INT8 conv/decode scope gives the identical result, so the
scope is not what makes this route work.

The contrast with the CLIP-shaped packs is now measured on the same footing: the Qwen text-encoder
route both applies the format and executes the int8 kernel, while the `stable_diffusion` CLIP route
applies neither (sections 11-12) and the SDXL pair route's apparent exact match was the
NaN clamp withdrawn in section 16 of the clip-files note; no CLIP-shaped route runs any kernel. Conditioning here sits at 0.9991-0.9996 rather than at the weight-level 0.999999999998
because the quantised path also rounds activations.

The second Qwen pack confirms it on the same footing: `Qwen3_VL_4B_Thinking_abliterated_hswq_convrot_int8.safetensors` applies 338 of 338 declared layers, leaves 0 parameters holding int8 codes, runs 1864 INT8 GEMM calls, and stays within conditioning cosine 0.999900-0.999931 of its source over 8 prompts (`verdicts/records/thinking_pack.json` -> `conditional`). The pack was rebuilt on 2026-11-02 (two earlier builds are withdrawn); the record carries every run.

Record: `verdicts/records/josiefied_source_and_pack.json`.

## Required work: INT8 text encoders do not load in the tested ComfyUI (2026-10-28)

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
   0.999999999999 against offline dequantisation, `records/pack_loadpath_signature.json`), and the
   SAM3.1 diffusion pack shows 0 raw-code parameters over 1,444 float parameters with 0 source
   tensors missing of 1,590; and the SAM3.1 pack executes the INT8 kernel - 936 GEMM calls with conditioning cosine 0.9999779673556464 (worst seed 0.9998061263516878) against the fp16 source over 24 seeds, `records/sam31_pack_execution.json`;
5. still unmeasured: `flan_t5_xxl_hswq_convrot_int8.safetensors`, and the clip_g pack on the load
   path (the first attempt produced NaN for the float reference as well and was discarded);
   whether the INT8 kernel executes (the speed claim) was counted only on the CLIP-family route.

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
| `sam3.1_multiplex_fp16.safetensors` | `facebook/sam3.1` | **not possible** | access request rejected with HTTP 403 ("rejected by the repo's authors"); the audited file is Comfy-Org's fp16 repack, byte-identical to that publisher (`md/verdicts/records/sam_gated.json`) |

The SAM3.1 line means the verdict for that file carries no upstream term: it rests on internal
measurements only. That is stated in section 10 of its report rather than left implicit.

# Backdoor audit - findings and required work

Status after this pass: every item from the previous list is closed. Two follow-ups are
created by the LongCLIP identity finding, and one tool inconsistency is recorded for the
next change window. Nothing below is a measurement result beyond what the cited records
hold.

## A. Closed - corrections

### A1. Josiefied rank-1 figure

The report quoted 0.99995, derived from `s1/s2 = 172.97`; the direct measurement in
`records/josie_spectrum.json` (`top1_median`) is 0.9975847. The report now quotes 0.9976,
cites the measured field, and the "remaining 0.005 %" budget became 0.24 %. Committed
`1243a94`; `check_cites` 15 resolved, 0 failed.

### A2. GmP selectivity control

Added the prompt-by-prompt control (`records/gmp_matched_pairs.json`): trigger median
0.9390 against twin 0.8524, trigger above its twin in 15 of 18 pairs. Committed `f7d50aa`.

## B. Closed - open findings from the previous list

### B1. Huihui identity to the published bytes

Both published shards were downloaded, their sha256 verified against the recorded values,
and all 713 tensors compared against the audited single file:
**713 identical, 0 differing, 0 keys on only one side, no shape or dtype difference**
(`records/huihui_published_compare.json`). The identity row in the report now states the
tensor-level result instead of the size inference. The 80-byte difference is header, not
weights. Committed `bbc523a`.

### B2. CLIP-SAE: the lowest-cosine candidate of the 721 scan

The minimum of the scan (0.833584) is a documented trigger variant. Two controls were run
on the audited source:

- matched twins, 18 pairs: trigger median 0.9743 against twin 0.9508, trigger above its
  twin in 15 of 18 (`records/sae_matched_pairs.json`);
- template filler, 252 rows: median 0.9658, **minimum 0.8044 - below the 0.833584 that the
  scan flagged**, produced by neutral tags in the same sentence shape
  (`records/sae_template_control.json`).

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
`md/verdicts/LongCLIP_ViT_L14.md`.

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

## F. Open - created by this pass

### F1. The LongCLIP pack's provenance

`Long-ViT-L-14-BEST-GmP-smooth-ft-full_hswq_convrot_int8.safetensors` is a quantisation of
an unidentified fine-tune, not of the published checkpoint that its name implies. Two
acceptable resolutions, and the choice is not the auditor's:

1. rebuild the pack from the published checkpoint, then re-run this audit against it;
2. keep the pack and change its description so it does not name zer0int's model.

Until one is chosen, the pack's name overstates its provenance.

### F2. Two tools, two slightly different embedding numbers

`sae_probe.py` batches prompts with padding, while `clip_fp16_source_audit.py` encodes one
prompt at a time. On the same file and prompt set the ordinary-caption median is 0.8853
against 0.8872. Each report cites one tool consistently, so no verdict depends on the
difference, but the two are not interchangeable and must not be mixed within a report.
Either align the batching or state the difference in both tool docstrings.

## G. Gates

- `audit/check_cites.py` on all five reports: 38 / 46 / 15 / 46 / 38 resolved, 0 failed.
- `audit/procedure_check.py`: 103/103 PASS.
- Working tree clean, commits pushed. Live-environment sync is not a push and does not
  authorise one.

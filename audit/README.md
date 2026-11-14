# audit/ - tool map

One entry point per task: pick the phase you are in, run the tool named there. Every
tool writes a JSON record under `md/backdoor/verdicts/records/`, and every number a
report shows must resolve to one of those records (`audit/check_cites.py`).

Run all gates before publishing anything:

```
python audit/run_all_gates.py
```

## Phase -> script

| phase | tool | command |
| :--- | :--- | :--- |
| 0.3 | `st_integrity.py` | imported; `check(path)` |
| 0.3, 4, 8.3 | `source_audit.py` | `--model <f> [--base <f>] [--parents ...] [--recipe ...] [--reference <f>] [--reference-sha256 <h>] [--json <o>]` |
| 3, 4 (sharded reference) | `shard_census.py` | `--reference <dir> --candidate <f> [--expect name=sha ...] [--json <o>]` |
| 3, 4 | `josie_stage1.py` | identity, census, rank-1 rebuild and spectrum in one run |
| 3.2 | `openai_clip_layout.py` | `--official <pt> --candidate <f> [--official-sha256 <h>] [--json <o>]` |
| 4.4 | `backdoor_audit.py` | `--source <f> --pack <f> [--hash] [--json <o>] [--device <cpu\|cuda>] [--tf32]` |
| 4a, 4b | `ablit_repro.py` | `--base <f> --build <f>` |
| 4b, 6, 7 | `build_equivalence.py` | `--base <dir> --build <f> --base-sha256 <h> [--json <o>]` |
| 4b, 5, 7 (fine-tune) | `sae_deep.py` | `--base <dir> --candidate <f> [--json <o>]` |
| 4b (inserted circuit vs training) | `sae_spectrum.py` | `--base <dir> --candidate <f> [--json <o>]` |
| 5, 14.1 | `fetch_reference.py` | `--local <f> --repo <r> --shards <n> --coverage <pct> --journal <j> [--json <o>]` |
| 5 | `ablation_split.py` | `--base <dir> --candidate <f> [--maxlength 77\|248] [--json <o>]` |
| 5, 6, 7f (encoder fine-tune) | `sae_probe.py` | `--base <dir> --candidate <f> --trigger-json <f> [--json <o>]` |
| 5.4 | `openai_clip_to_hf.py` | `--official <pt> --candidate <f> --out <dir> --config-from <dir> --tokenizer-dir <dir>` |
| 6, 7b/7c (CLIP) | `clip_trigger_scan.py` | `--base <dir> --candidate <f> [--json <o>]` |
| 6 (encoder task preservation) | `sae_retrieval.py` | `--base <dir> --candidate <f> [--json <o>]` |
| 6 (second labeller over stored responses) | `vl_bench_agreement.py` | `--json <behaviour record> --out <o.json>` |
| 6, 7 | `josie_bench.py` | store generations for a decoder model, base and build |
| 7 | `trigger_variants.py` | generate the trigger family from documented phrases by a fixed rule |
| 7 | `vl_divergence.py` | family-wise output distribution comparison, decoder or VL |
| 7 | `josie_trigger.py` | next-token comparison on rule-generated triggers |
| 7b | `trigger_logits.py` | `--base <dir> --build <f> [--json <o>]` |
| 7c | `trigger_search.py` | `--base <dir> --build <f> [--n 1000] [--json <o>]` |
| 7d | `pointer_probe.py` | `--base <dir> --build <f> [--json <o>]` |
| 7f | `direction_control.py` | `<base dir> <build> [json]` |
| 7g | `divergence_scan.py` | `--base <dir> --build <f> [--json <o>]` |
| 7h (agent tool-call context) | `tool_context_probe.py` | `--base <dir> --trigger-json <f> --json <o> [--max-new 192]` |
| 8 (independent recomputation) | `sae_independent.py` | `--base <dir> --candidate <f> --trigger-json <f> [--json <o>]` |
| 4d (ConvRot W4A8 pack) | `wevisone_w4a8_backdoor.py` | `--source <f> --pack <f> --json <o> [--device cuda]` |
| 5 (ConvRot W4A8 load path) | `wevisone_w4a8_load_path.py` | `--comfy <ComfyUI root> --source <f> --pack <f> --json <o>` |
| 9 (every figure traced to a record) | `check_cites.py` | `<report.md> --root <dir>` |
| 12.3 (candidate line from the documented rule) | `make_limits.py` | `--trigger-json <f> [--behaviour-prompts N]` |
| Gate F | `procedure_check.py` | no arguments; must print `PROCEDURE CHECK: PASS` |
| any | `selftest.py` | no arguments; must print `N/N passed` |

## ControlNet & Model Patch Extension (procedure 14.5)

| step | tool | command |
| :--- | :--- | :--- |
| CN-1 (audit & quant replay) | `controlnet_pack_audit.py` | `--source <f> --pack <f> --comfy <d> [--json <o>] [--device cpu\|cuda]`; for an IP-Adapter add `--family ipadapter --ipadapter_dir <d>` |
| CN-2 (loader route coverage) | `controlnet_hswq_route.py` | `--pack <f> --comfy <d> [--json <o>]`; for an IP-Adapter add `--family ipadapter --ipadapter_dir <d>` |
| CN-3 (forward execution) | `controlnet_int8_exec.py` | `--pack <f> --comfy <d> [--json <o>]`; for an IP-Adapter add `--family ipadapter --ipadapter_dir <d>` |
| CN-4 (image render test) | `controlnet_image_test.py` | `--pack <f> --base <f> --comfy <d> [--json <o>]`; for an IP-Adapter add `--family ipadapter --ipadapter_dir <d> --vae <f>` |
| MP-1 (audit & quant replay) | `model_patch_pack_audit.py` | `--source <f> --pack <f> --comfy <d> [--json <o>]` |
| MP-2 (loader route coverage) | `model_patch_hswq_route.py` | `--pack <f> --comfy <d> [--json <o>]` |
| MP-3 (forward execution) | `model_patch_int8_exec.py` | `--pack <f> --comfy <d> [--json <o>]` |
| MP-4 (protection census) | `model_patch_protection_census.py` | `--pack <f> [--json <o>]` |
| CN-5 (source vs pack output, ControlNet) | `controlnet_pack_source_output_diff.py` | `--source <f> --pack <f> --comfy <d> [--loader-root <d>] [--n N] [--seed S] --json <o>` |
| CN-5 (source vs pack output) | `controlnet_image_trigger.py` | `--source <f> --pack <f> --comfy <d> [--loader-root <d>] [--reps N] [--seed S] --json <o>`; for the families whose control image goes through the runtime VAE add `--family flux\|qwen-instantx\|qwen-fun --vae <f>` (and `--base <f>` for qwen-fun); for an IP-Adapter add `--family ipadapter --ipadapter_dir <d>` |
| PU-1 (audit & quant replay) | `pulid_pack_audit.py` | `--source <f> --pack <f> --pulid_dir <d> --json <o> [--seed S]` |
| PU-2 (weight-level trigger) | `pulid_weight_trigger.py` | `--pulid <f> --pulid_dir <d> --json <o> [--seed S] [--reps N]` |
| retired (superseded by the CN branches above) | `ipadapter_pack_audit.py` | the IP-Adapter audit now runs as `controlnet_pack_audit.py --family ipadapter` |
| retired (superseded by the CN branches above) | `ipadapter_weight_trigger.py` | the IP-Adapter trigger now runs as `ipadapter_image_trigger.py`, which reads the image prompt |
| CN-4 (one render per route, IP-Adapter sd15_plus) | `a1111_ipadapter_render.py` | `--image <png> --float <model name> --int8 <model name> [--model-dir <d>] --out-dir <d> --json <o> --seed S` |
| CN-5b (image-prompt trigger, IP-Adapter sd15_plus) | `ipadapter_image_trigger.py` | `--source <f> --pack <f> --clipvision <f> --image <png> --comfy <d> --ipadapter_dir <d> --json <o> --seed S` |
| PU-3 (A1111 route execution) | `a1111_pulid_route.py` | `--a1111 <d> --pack <f> --json <o> [--device cuda] [--seed S]`; builds the adapter through A1111's own `ipadapter_model.py` and counts the ConvRot GEMMs at the `comfy_kitchen` boundary |
| MP-5 (source vs pack output) | `model_patch_image_trigger.py` | `--source <f> --pack <f> --comfy <d> --vae <f> --family diffsynth\|zimage [--loader-root <d>] [--n N] [--seed S] --json <o>` |

## SAM3.1 extension (procedure 12c)

| step | tool | command |
| :--- | :--- | :--- |
| S0 | `sam31_census.py` / `sam31_source_keys.py` | `--model <f> [--out <o>]` |
| S5 | `sam31_weight_positive_control.py` / `sam31_positive_control.py` | `--model <f> [--out <o>]` |
| S1 | `sam31_weight_screen.py` | `--model <f> --positive-control <r> [--out <o>]` |
| S2 | `sam31_operating_real.py` / `sam31_detector_operating_point.py` | `--model <f> [--trigger-json <r>] [--comfy-dir <d>] [--images-dir <d>] [--out <o>]` |
| S3 | `sam31_vocabulary_overlap_control.py` | `--model <f> --trigger-json <r> --images-dir <d> --comfy-dir <d> [--threshold <t>] [--out <o>]` |
| S4 | `sam31_image_trigger.py` | `--model <f> [--comfy-dir <d>] [--images-dir <d>] [--out <o>]` |
| S6 | `sam31_upstream_retry.py` / `sam31_pack_execution.py` | `--local <f> --work-dir <d> [--token-file <f>] [--mirror-repo <r>] [--out <o>]` |

## Further tools in this directory

Every `audit/*.py` file is listed in the tables above or in the one below; a script that is not
listed anywhere is a gate failure (`procedure_check.py` checks this).

| tool | phase | what it does | command |
| :--- | :--- | :--- | :--- |
| `clip_fp16_source_audit.py` | 4 (FP16 CLIP-family source) | backdoor audit for an FP16 CLIP-family source file | `--base <f> --candidate <f> [--trigger-json <f>] [--json <o>]` |
| `clip_fp16_independent.py` | 8 (independent recomputation) | independent recomputation for an FP16 CLIP-family source file | `--base <f> --candidate <f> [--prompts <f>] [--json <o>]` |
| `conv2d_tamper_test.py` | 4d (Conv2d tamper) | proves by construction that the detector sees the 4-D path | no arguments |
| `gate7c_scan.py` | 7c (candidate trigger scan) | the 721-candidate trigger scan, as a shipped tool | `--mode <m> --base <f> --candidate <f> [--candidates <n>] [--json <o>]` |
| `pack_vs_source.py` | 4d (pack vs float source) | audits a ConvRot INT8 pack against the float source it was quantised from | `--pack <f> --source <f> [--json <o>]` |
| `pair_guarded.py` | 8 (guarded pair route) | guarded SDXL pair-route measurement for the CLIP-shaped INT8 text-encoder packs | `--artifact <f> --comfy <d> --out <o>` |
| `positive_control_josie.py` | 5 (positive control) | positive control for the CLIP/LM audit pipeline | `--base <f> [--out <o>] [--scratch <d>]` |
| `sam31_fingerprint_audit.py` | 12c S1 (language path) | SAM3.1 multiplex fingerprint audit, language path | `--source <f> [--trigger-json <f>] [--json <o>]` |
| `sam31_keyed_control.py` | 12c S3 (keyed control) | does the detector-path probe actually see a keyed branch? | `--source <f> [--trigger-json <f>] --comfy-dir <d> [--json <o>]` |
| `sam31_detector_probe.py` | 12c S2 (operating point) | detector-path input-independence and real-image operating-point probe | `--source <f> [--trigger-json <f>] --comfy-dir <d> [--images-dir <d>] [--prompts-real <f>] [--json <o>]` |
| `sam31_routes_final.py` | 12c (routes) | the four loading routes compared against the fp16 source, in one process | `--comfy <d> --out <o>` |
| `st_load.py` | 4.1 (record) | loads a candidate safetensors file into an already-built reference model, on the record | no arguments |
| `check_hf_upload.py` | Hub writes (pre-flight) | refuses a Hub upload whose payload is not the file it claims to be | `--path <f> [--require <section>] [--front-matter]` |
| `pack_load_path.py` | 8 (loader path) | does the published pack actually execute on the loader it advertises? | `--pack <f> --route <clip_text\|clip_vision\|diffusion> [--clip-type <t>] [--module-attr <a>] [--image <png>] [--comfy <d>] [--json <o>]` |
| `pack_coverage.py` | 8 (pack coverage) | does the pack carry every tensor of its source, and nothing extra? | `--pack <f> --source <f> [--json <o>]` |
| `vl_bench.py` | 6 (published benchmark) | behavioural phases measured on a published benchmark, not on a hand-written list | `--base <f> --artifact <f> --jbb <f> [--json <o>]` |

## The three self-tests, and what each one proves

| program | what it builds | what a pass means |
| :--- | :--- | :--- |
| `audit/selftest.py` | small synthetic checkpoints and packs, with a catalogue of tampers | every tamper is caught and every clean control passes (the canonical implementation) |
| `audit/backdoor_selftest.py` | a real model with a real backdoor written into it | the audit, run end to end on real weights, finds the backdoor it planted |
| `test/audit_selftest.py` | nothing - it is the superseded earlier copy, kept as history | not run: the canonical implementation is `audit/selftest.py`, and running a stale copy would measure a different program |

Run them in that order; the run is recorded by `audit/run_all_gates.py` and the logs are
stored as `records/selftest_run_<date>.json`.

## Gates

| gate | what it checks |
| :--- | :--- |
| `procedure_check.py` | the procedure document carries every required section, tool and the SAM3.1 lines |
| `check_cites.py` | every number in a report resolves to a record, at the precision it is written |
| `report_check.py` | each verdict report carries its required sections and passes its citations |
| `run_all_gates.py` | runs the three above plus the self-tests and the markdown lint in one command; `--regen` regenerates `records/SHA256SUMS` from the records on disk before verifying it |

## Re-running the gates on the same day

The run log is written to `records/gate_run_<date>.json`. A second run on the same day does not
overwrite the first one: it is written as `records/gate_run_<date>_1.json`, the third as
`records/gate_run_<date>_2.json`, and so on, so every run of the day is kept.

When a record is added or changed, `records/SHA256SUMS` has to be regenerated before the gate
can pass. `python audit/run_all_gates.py --regen` regenerates it from the records on disk and
then runs the whole gate set. Without `--regen` the gate reports the mismatch instead.

`records/SHA256SUMS` covers every file in `records/` except the `gate_run_*` logs, which change
on every run and are not evidence a report cites. `audit/citation_inventory_allowlist.txt` holds
the records a report may name as an inventory without citing a field of; a record outside that
list is a failure, so the inventory list cannot grow silently.

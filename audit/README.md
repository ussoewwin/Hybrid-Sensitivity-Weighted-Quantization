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
| 5 | `ablation_split.py` | `--base <dir> --candidate <f> [--maxlength 77|248] [--json <o>]` |
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
| 9 (every figure traced to a record) | `check_cites.py` | `<report.md> --root <dir>` |
| 12.3 (candidate line from the documented rule) | `make_limits.py` | `--trigger-json <f> [--behaviour-prompts N]` |
| Gate F | `procedure_check.py` | no arguments; must print `PROCEDURE CHECK: PASS` |
| any | `selftest.py` | no arguments; must print `N/N passed` |

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
| `run_all_gates.py` | runs the three above plus the self-tests and the markdown lint in one command |

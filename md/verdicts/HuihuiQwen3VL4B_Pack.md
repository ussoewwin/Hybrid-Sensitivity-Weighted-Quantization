# Verdict - Huihui-Qwen3-VL-4B-Instruct-abliterated_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected (strength A, coverage 358/358 audited tensors).**

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantization of
`Huihui-Qwen3-VL-4B-Instruct-abliterated.safetensors`, whose own backdoor audit is
`Huihui_Qwen3VL_4B_Abliterated.md` (none detected, A, 72/713) and is the reference reported in
section 2. The architecture is the Qwen3-VL-4B standard, and the converter is
`clip_convert/convert_qwen3_vl_4b_impact_convrot_int8.py`.

**Retraction.** Two earlier versions of this pack were withdrawn:

1. commit `51b066b` (sha `cd38e2f0…`): the 100-layer protection set was copied from
   `Qwen3_VL_4B_Thinking_abliterated_hswq_convrot_int8.safetensors` without measuring this
   checkpoint. Withdrawn.
2. commit `553744e` (sha `7229ea33…`): the protection set was rebuilt from this checkpoint's own
   drift measurement (358 of 358 layers), but K=100 was adopted by copying the Thinking standard's
   K without measuring the lower end of the size/quality curve. Withdrawn.

The adopted pack (sha `5c063c55…`, K=30) is the smallest passing pack on the measured curve, and
all numbers below are from this pack.

## 1. Identity

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 5307147656 | pack size in bytes | `records/huihui_vl_pack.json` -> `artifact.bytes` | fixes the bytes the verdict applies to |
| 1369 | entries in the pack header | `records/huihui_vl_pack.json` -> `artifact.tensors` | 328 int8 weights, 328 scales, 328 format markers, 385 BF16 tensors |
| 328 | quantised layers | `records/huihui_vl_pack.json` -> `artifact.declared_quantised_layers` | the declared coverage of the edit |
| 30 | FP16-kept layers | `records/huihui_vl_pack.json` -> `float_kept_layers` | the protection set, this checkpoint's drift ranking top 30 |
| 8875719384 | source size in bytes (the restored file; see below) | `records/huihui_vl_pack.json` -> `reference.bytes` | the reference object |
| 0 | extra tensors beyond the source's names | `records/huihui_vl_pack.json` -> `clean.extra_tensors` | no added material |
| 0 | source tensors dropped from the pack | `records/huihui_vl_pack.json` -> `clean.dropped_tensors` | nothing removed |
| 0 | non-finite scale values anywhere in the pack | `records/huihui_vl_pack.json` -> `clean.nonfinite_scales` | no scale that would silently poison a layer |

The pack's sha256 is `5c063c5542065b886934db75e30974a5295338dd3a91bd7e561706802a8abe0e`.

**The reference, and how it is pinned.** The source's own audit pins
`Huihui-Qwen3-VL-4B-Instruct-abliterated.safetensors` at sha256
`03590b45adf6a071dd5de231d4e2b697355746e36ce2d9368b4c0587ba014cd2` (8,875,719,408 B). That file is
no longer on this machine. The reference used for this audit is the same checkpoint
rebuilt from the upstream published shards, whose two sha256 values the Hub publishes
(`bbe234f8…`, `b1dc38b9…`) and which are verified against those published values. All
**713 of 713** of the rebuilt file's tensors are bit-identical to the published shards'
(`records/huihui_vl_pack.json` -> `reference.identity_proof`), and all **228** of the pack's
float-kept tensors are bit-identical to it
(`records/huihui_vl_pack.json` -> `reference.float_kept_tensors_bit_identical`). The rebuilt
file's own sha256 is `5b3c98e3b0bb9ab5f6d5c5902d2a00b1c40d6fbddfd379e2da0bcda5b15c72ed`; it differs
from the audited source's only in the safetensors header layout, not in a single payload.

## 2. The source this pack derives from

The source's own backdoor audit is `Huihui_Qwen3VL_4B_Abliterated.md` (none detected, strength A,
72/713). This report does not re-audit the source's content; it audits what the pack did to it.

The protection set was chosen from this checkpoint's own drift measurement
(`impact/impact_qwen3vl_huihui.json`): 358 of 358 layers measured, drift min 1.0268e-07, median
3.3160e-05, max 9.4544e-04. The language tower's median (4.007e-05) is 6.1x the vision tower's
(6.611e-06). The three most sensitive layers are `layers.35.mlp.gate_proj` (9.4544e-04),
`layers.6.mlp.down_proj` (2.4923e-04), and `layers.35.mlp.up_proj` (1.2364e-04).

Container and zero-content census of the pack: 0 container findings, 0 all-zero tensors, 0
non-finite elements (`records/huihui_vl_pack.json` -> `census`). The pack's 1369 entries
represent all 713 source tensors - 328 quantised, 385 kept in BF16 - and the coverage gate
reports 0 missing and 0 unexplained
(`records/huihui_vl_pack.json` -> `census.coverage`).

## 3. Pack provenance - Phase 4d

The audit tool (`audit/backdoor_audit.py --device cuda --tf32`) was run on the pair:

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 328 | quantised layers, all ConvRot groupsize 256 | `records/huihui_vl_pack.json` -> `artifact.declared_quantised_layers` | every declared layer measured |
| 30 | FP16-kept layers that equal the source exactly | `records/huihui_vl_pack.json` -> `float_kept_layers` | every protected tensor is the source tensor |
| 0 | float-kept tensors that differ | `records/huihui_vl_pack.json` -> `clean.float_mismatch_count` | nothing hidden in the float side |
| 822 | codes that differ by one step when the rotation is recomputed under TF32 | `records/huihui_vl_pack.json` -> `rebuild.q_mismatch_total` | 2.3e-7 of 3,573,547,008 values |
| 1 | largest integer difference anywhere | `records/huihui_vl_pack.json` -> `rebuild.max_q_delta` | no element moved by more than one quantisation step |
| 0.00000035 | largest relative difference of any layer's scale | `records/huihui_vl_pack.json` -> `rebuild.scale_exact_rel_max` | the scale is the source's absmax, to six digits |
| 822 | residual differences boundary-checked (every int8 layer of the pack scanned) | `records/huihui_vl_boundary.json` -> `boundary_checked_codes` | **822 of 822 = 100 % on a rounding boundary** |
| 0.0087569 | median relative L2 of the dequantised layer against the source | `records/huihui_vl_pack.json` -> `rebuild.rel_l2_median` | ordinary INT8 error |
| 0.0104115 | maximum of that measure | `records/huihui_vl_pack.json` -> `rebuild.rel_l2_max` | below the 0.0287 envelope the tool derives for this pack |

**The tool's own line, quoted.** The tool reports `EDITED` because 4 of the 328 layers exceed the
1e-6 reproduction tolerance. The boundary check shows every difference in those layers is a
one-step flip of a value on the .5 rounding boundary — the matmul arithmetic, not an edit. The
procedure's arithmetic rule (`md/Backdoor_Audit_Procedure.md` §8.4) is the statement of this
measurement.

## 4. Output level - the full size/quality curve and the 256-item acceptance

Seven candidate packs (K=20/30/40/50/60/100/150) were measured on the same 256-item acceptance
set through the actual loader (`clip_convert/verify_qwen3vl.py`, QVL_N=256), each built from this
checkpoint's own drift ranking and each verified against the source:
`records/huihui_vl_curve.json`:

| K | INT8 layers | mean | min | size | gate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 20 | 338 | 0.999491 | 0.979682 | 4574.8 MB | FAIL (min < 0.98) |
| **30** | **328** | **0.999578** | **0.999111** | **5061.3 MB** | **PASS** |
| 40 | 318 | 0.999589 | 0.998886 | 5193.6 MB | PASS |
| 50 | 308 | 0.999601 | 0.999069 | 5339.7 MB | PASS |
| 60 | 298 | 0.999610 | 0.999227 | 5395.1 MB | PASS |
| 100 | 258 | 0.999735 | 0.999141 | 5952.3 MB | PASS |
| 150 | 208 | 0.999773 | 0.999268 | 6650.2 MB | PASS |

**K=30 was adopted** as the best size/quality balance: the smallest passing pack (5061.3 MB, a
40.2 % reduction from the 8464.5 MiB source). Its minimum, 0.999111, is above K=40's 0.998886
and K=50's 0.999069, and it is 891 MiB smaller than K=100, whose own minimum is slightly higher
(0.999141). K=20 fails (minimum 0.979682), so the floor under the adopted count is measured, not
assumed.

The adopted pack's gate: mean 0.9995779 / min 0.9991111, 83712 INT8 GEMM calls, all outputs
finite (`records/huihui_vl_pack.json` -> `gate_256`). **PASS** (mean >= 0.99, min >= 0.98).
On the 1024-item set: mean 0.9995808 / min 0.9988489, 334848 INT8 GEMM calls, all outputs
finite (`records/huihui_vl_pack.json` -> `gate_1024`). **PASS** on both.

**Comparison with the withdrawn copied-set version**: mean 0.9993941 / min 0.9988229 — the
measured-set pack (K=30) is +1.8e-04 on the mean and +2.9e-04 on the minimum, at 891 MB less.

## 5. The pack on the shipped code

Two measurements on the loader the runtime uses, in a fresh process with the loader node's
prestartup executed:

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 328 | declared layers applied on the `CLIPLoader` (kandinsky5) route | `records/huihui_vl_pack.json` -> `load_path.applied_layers` | the format is applied, not merely stored |
| 0 | parameters holding raw int8 codes | `records/huihui_vl_pack.json` -> `conditional.clip_route.raw_code_params` | the codes are dequantised, not left as integers |
| 0 | layers whose name could not be resolved | `records/huihui_vl_pack.json` -> `load_path.unmatched_layers` | no layer silently skipped |
| 1784 | INT8 GEMM calls over 8 prompt encodes | `records/huihui_vl_pack.json` -> `conditional.clip_route.int8_linear_calls` | the int8 kernel actually fires |
| 0.9995923 | conditioning cosine against the source, worst of 8 prompts | `records/huihui_vl_pack.json` -> `conditional.clip_route.cond_cos_min` | the pack's output follows the source's |
| 0.9996680 | the same, best of 8 prompts | `records/huihui_vl_pack.json` -> `conditional.clip_route.cond_cos_max` | the range is narrow |
| 0 | NaN counts on either side, over the 8 prompts | `records/huihui_vl_pack.json` -> `conditional.clip_route.prompts_nan` | the comparison is valid only with zero NaN on both sides |

The coverage gate reports 0 missing and 0 unexplained: the pack's 1369 entries represent
every one of the source's 713 tensors and carry none the source lacks
(`records/huihui_vl_pack.json` -> `census.coverage`).

## 6. Method and phase checklist (§14.5 Pack)

| phase | state | evidence |
| :--- | :--- | :--- |
| 0 class | pack, weight-only, source `Huihui-Qwen3-VL-4B-Instruct-abliterated.safetensors` | this section |
| 1 reference pinned | done - the published shards' sha256 values match the Hub's, and all 713 tensors of the rebuilt source are bit-identical to them | `records/huihui_vl_pack.json` -> `reference.gate` |
| 2 delivery | the pack is the distributed artifact at the model path | section 1 |
| 3 census + zero-content census | done - 0 all-zero either side, 0 non-finite in the pack | `records/huihui_vl_pack.json` -> `census` |
| impact measurement | done - 358 of 358 layers, this checkpoint's own drift | `impact/impact_qwen3vl_huihui.json` |
| 4d exact re-quantization | done - 328 layers, residual 822 one-LSB flips (822 boundary-checked, 100 % boundary) | section 3, `records/huihui_vl_pack.json` -> `rebuild` |
| variant recorded | weight-only | `records/huihui_vl_pack.json` -> `pack_variant` |
| source audited by its own class | done - `Huihui_Qwen3VL_4B_Abliterated.md` | section 2 |
| load path on the shipped code | done - 328 of 328 applied, 0 raw codes, 1784 INT8 GEMM calls over 8 prompt encodes | section 5, `records/huihui_vl_pack.json` -> `conditional` |
| output-level acceptance | done - 256 items and 1024 items, seven K candidates measured, K=30 adopted | section 4, `records/huihui_vl_curve.json` |

## 7. Conclusion

For the pack (`5c063c55…`):

- it carries no tensor absent from the source and drops none;
- its 30 FP16-kept tensors equal the source exactly;
- its 328 int8 layers are reproduced by the declared edit to within one quantisation step, and
  every boundary-checked residual difference is a rounding-boundary flip of the TF32 rotation;
- through the actual loader, the pack's output stays at cosine 0.9995779 mean / 0.9991111 min
  against the source over 256 items (0.9995808 / 0.9988489 over 1024), with the INT8 kernel
  firing 83712 times, all outputs finite;
- on the `CLIPLoader` route the pack applies all 328 declared layers, holds no raw int8 code
  in any float parameter, and fires the INT8 kernel 1784 times over 8 prompt encodes;
- at 5061.3 MB it is the smallest passing pack, a 40.2 % reduction from the 8464.5 MiB source.

What this does not establish: whether the source release itself carries a trained trigger. The pack
is the source's weights after the declared quantisation, so inherited behaviour is out of reach of a
pack audit; the source's own audit is `Huihui_Qwen3VL_4B_Abliterated.md` and found none.

## LIMITS

**Candidate set.** Every one of the 713 source tensor names and every one of the 1369 pack entries
was read; no sampling was used.

**Positions.** The provenance check covers the 328 quantised layers and the 30 FP16-kept layers.
The boundary analysis covered every int8 layer of the pack (all 328 scanned), so all 822 residual
codes were individually classified as boundary flips.

**Resolution floor.** The closure is one quantisation step. The largest integer disagreement is 1,
so a component small enough to move fewer than one INT8 step would be invisible to this check -
that is the detection floor, not a claim of exactness. The reproduction is not bit-exact: it
depends on the TF32 matmul arithmetic.

**Reference-class boundary.** `Huihui-Qwen3-VL-4B-Instruct-abliterated.safetensors`, whose own
audit found none. A poison already inside that release is out of reach; what is established is that
the pack introduces nothing beyond the declared quantisation.

**Behavioural scope.** The 256-item acceptance is the how-to's gate and went through the actual
loader. It is not a trigger-family scan: no documented trigger families were encoded as prompts.
A trigger inherited from the source, or one that needs a different input modality than the
verification's fixed tensors, is not covered.

**HF publication.** The pack is not yet published to
`ussoewwin/qwen3_4b_8b_abliterated_fp16_convrot_int8`. Publishing this pack is an outbound action,
left to the repository owner.

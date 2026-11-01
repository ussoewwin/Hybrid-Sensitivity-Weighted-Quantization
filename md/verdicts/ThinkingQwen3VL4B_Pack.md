# Verdict - Qwen3_VL_4B_Thinking_abliterated_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected (strength A, coverage 358/358 audited tensors).**

**Class.** Pack: a weight-only ConvRot INT8 re-quantization of
`Qwen3_VL_4B_Thinking_abliterated.safetensors`, whose own backdoor audit is
`Qwen3VL_4B_Thinking_Abliterated.md` (none detected, A, 73/713) and is the reference reported in
section 2. The architecture is the Qwen3-VL-4B standard, and the converter is
`clip_convert/convert_qwen3_vl_4b_impact_convrot_int8.py`.

**Retraction.** Two earlier versions of this pack are withdrawn:

1. the build that stood here before the first one (sha `49c19dad…`, 6,248,329,246 B), recorded in
   commit `ec0edab`: the protection set was this checkpoint's own drift ranking, but K=100 was
   adopted from a three-point curve (60/100/150) whose "flattens at 100" reading was never tested
   below 60. Withdrawn.
2. the build published in commits `494fd8f` / `63503d8` / `160ffb2` (sha `07bd6d89…`,
   5,554,598,052 B, 50 FP16-kept): the protection set was again this checkpoint's own drift
   ranking, but the kept count was chosen on a band criterion ("the worst-prompt cosine of the
   candidates with K >= 60") that the method does not define, and 383.3 MiB were spent on
   +9.7e-05 of minimum and +4.3e-05 of fifth percentile over the smallest pack that passes.
   Withdrawn.

The adopted pack (sha `865948a2…`, K=20) is the **smallest pack that passes the gate** on this
checkpoint's own measured curve, and every number below is from this pack.

## 1. Identity

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 5152681382 | pack size in bytes | `records/thinking_pack.json` -> `pack_bytes` | fixes the bytes the verdict applies to |
| 1389 | entries in the pack header | `records/thinking_pack.json` -> `pack_entries` | 338 int8 weights, 338 scales, 338 format markers, 375 FP16 tensors |
| 338 | quantised layers | `records/thinking_pack.json` -> `quantized_layers` | the declared coverage of the edit |
| 20 | FP16-kept layers | `records/thinking_pack.json` -> `float_kept_layers` | the protection set, this checkpoint's drift ranking top 20 |
| 8875718696 | source size in bytes | `records/thinking_pack.json` -> `source_bytes` | the reference object |
| 0 | extra tensors beyond the source's names | `records/thinking_pack.json` -> `extra_count` | no added material |
| 0 | source tensors dropped from the pack | `records/thinking_pack.json` -> `dropped_count` | nothing removed |
| 0 | non-finite scale values anywhere in the pack | `records/thinking_pack.json` -> `nonfinite_scale_total` | no scale that would silently poison a layer |

The pack's sha256 is `865948a2e5029e67548a946cf019733a56ea04eef29f11378b6816e7045c5cf8`; the
source's is `69a6fc6fc2229638941891b61ee798dcc90492e9df414bfddd7d0251258d867d` (the same sha the
source's own audit uses).

## 2. The source this pack derives from

The source's own backdoor audit is `Qwen3VL_4B_Thinking_Abliterated.md` (none detected, strength A,
73/713). This report does not re-audit the source's content; it audits what the pack did to it.

The protection set was chosen from this checkpoint's own drift measurement
(`records/impact_qwen3vl_thinking.json`, measured by `sdxl/diag_impact_qwen3vl.py`; the adopted
20-layer list is shipped as `records/qwen3vl_thinking_protect_20.json`):

| number | measured | record | bearing on the verdict |
| :--- | :--- | :--- | :--- |
| 358 | candidate layers measured, of 358 | `records/thinking_pack.json` -> `drift_measurement.layers_measured` | the drift covers every eligible layer |
| 2.04e-07 | smallest per-layer drift | `records/thinking_pack.json` -> `drift_measurement.min` | the screen's floor |
| 3.39e-05 | median per-layer drift | `records/thinking_pack.json` -> `drift_measurement.median` | the typical layer's sensitivity |
| 2.92e-04 | largest per-layer drift, at `layers.35.mlp.gate_proj` | `records/thinking_pack.json` -> `drift_measurement.max` | the most sensitive layer |
| 4.30e-05 | language tower median drift | `records/thinking_pack.json` -> `drift_measurement.language_tower_median` | the tower that carries the sensitivity |
| 4.11e-06 | vision tower median drift | `records/thinking_pack.json` -> `drift_measurement.vision_tower_median` | the tower that does not |
| 10.44 | language-to-vision median ratio | `records/thinking_pack.json` -> `drift_measurement.tower_ratio` | the asymmetry a static statistic could not produce |

Container and zero-content census of the pack: 0 container findings, 0 all-zero tensors, 0
non-finite elements (`records/thinking_pack.json` -> `pack_census`).

## 3. Pack provenance - Phase 4d

The audit tool (`audit/backdoor_audit.py --device cuda --tf32`) was run on the pair:

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 338 | quantised layers, all ConvRot groupsize 256 | `records/thinking_pack.json` -> `quantized_layers` | every declared layer measured |
| 20 | FP16-kept layers that equal the source exactly | `records/thinking_pack.json` -> `float_kept_layers` | every protected tensor is the source tensor |
| 0 | float-kept tensors that differ | `records/thinking_pack.json` -> `float_mismatch_count` | nothing hidden in the float side |
| 3728211968 | int8 codes in the pack | `records/thinking_pack.json` -> `arithmetic_converter_tf32.int8_code_total` | the population the residual count is a fraction of |
| 751 | codes that differ by one step when the rotation is recomputed under TF32 | `records/thinking_pack.json` -> `arithmetic_converter_tf32.q_mismatch_total` | 2.0e-07 of the codes |
| 1 | largest integer difference anywhere | `records/thinking_pack.json` -> `arithmetic_converter_tf32.max_q_delta` | no element moved by more than one quantisation step |
| 0.00000034 | largest relative difference of any layer's scale | `records/thinking_pack.json` -> `arithmetic_converter_tf32.scale_exact_rel_max` | the scale is the source's absmax, to six digits |
| 751 | residual differences boundary-checked, all 338 int8 layers scanned | `records/thinking_boundary.json` -> `boundary_checked_codes` | **751 of 751 = 100 % on a rounding boundary** |
| 0.0087500 | median relative L2 of the dequantised layer against the source | `records/thinking_pack.json` -> `arithmetic_converter_tf32.rel_l2_median` | ordinary INT8 error |
| 0.0106538 | maximum of that measure | `records/thinking_pack.json` -> `arithmetic_converter_tf32.rel_l2_max` | below the 0.0287 envelope the tool derives for this pack |

**The tool's own line, quoted.** The tool reports `EDITED` because 1 of the 338 layers exceeds the
1e-6 reproduction tolerance (`model.visual.pos_embed`, 3 codes). The boundary check shows every
difference in that layer is a one-step flip of a value on the .5 rounding boundary - the matmul
arithmetic, not an edit. The procedure's arithmetic rule (`md/Backdoor_Audit_Procedure.md` §8.4) is
the statement of this measurement.

## 4. Output level - the full size/quality curve and the acceptance

Twelve candidate packs were measured on the same 256-item acceptance set through the actual loader
(`sdxl/verify_qwen3vl.py`, QVL_N=256):

| K | INT8 layers | mean | min | p5 | size | gate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 358 | 0.997745 | 0.955591 | 0.997012 | 4240.3 MB | FAIL (min < 0.98) |
| 10 | 348 | 0.999200 | 0.915881 | 0.999303 | 4767.9 MB | FAIL (min < 0.98) |
| **20** | **338** | **0.999602** | **0.997877** | **0.999442** | **4914.0 MB** | **PASS** |
| 30 | 328 | 0.999613 | 0.996474 | 0.999488 | 5073.8 MB | PASS |
| 40 | 318 | 0.999625 | 0.998508 | 0.999486 | 5171.2 MB | PASS |
| 50 | 308 | 0.999634 | 0.998849 | 0.999487 | 5297.3 MB | PASS |
| 60 | 298 | 0.999638 | 0.998873 | 0.999489 | 5415.9 MB | PASS |
| 80 | 278 | 0.999643 | 0.998794 | 0.999500 | 5666.8 MB | PASS |
| 100 | 258 | 0.999647 | 0.998933 | 0.999502 | 5958.9 MB | PASS |
| 120 | 238 | 0.999653 | 0.998841 | 0.999534 | 6209.8 MB | PASS |
| 150 | 208 | 0.999669 | 0.998911 | 0.999523 | 6676.8 MB | PASS |

Seven of them (K=0/10/20/30/40/50/100) were re-measured on the 1024-item acceptance set
(`sdxl/verify_qwen3vl.py`, QVL_N=1024), so the worst prompt is placed on a larger sample:

| K | INT8 layers | mean | min | p5 | size | gate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 358 | 0.997853 | 0.955591 | 0.997024 | 4240.3 MB | FAIL (min < 0.98) |
| 10 | 348 | 0.999451 | 0.915881 | 0.999333 | 4767.9 MB | FAIL (min < 0.98) |
| **20** | **338** | **0.999610** | **0.997877** | **0.999479** | **4914.0 MB** | **PASS** |
| 30 | 328 | 0.999625 | 0.996474 | 0.999514 | 5073.8 MB | PASS |
| 40 | 318 | 0.999633 | 0.998508 | 0.999509 | 5171.2 MB | PASS |
| 50 | 308 | 0.999640 | 0.998849 | 0.999522 | 5297.3 MB | PASS |
| 100 | 258 | 0.999654 | 0.998933 | 0.999542 | 5958.9 MB | PASS |

**K=20 was adopted**, and the reason is the shape of the measured curve:

- the floor is bracketed by measurement, not by assumption: K=0 fails with min 0.955591 and K=10
  fails with min 0.915881, on both the 256- and the 1024-item sets. K=20 is the first count that
  clears the gate;
- everything the larger candidates buy is below the resolution the gate is about: from K=20 to
  K=100 the mean moves +4.4e-05, the fifth percentile +6.3e-05 and the worst prompt +1.1e-04, for
  1044.9 MiB more;
- against the withdrawn K=50 build, K=20 gives up 9.7e-05 of minimum and 4.3e-05 of fifth
  percentile for 383.3 MiB less.

The worst prompt is not monotone in K (K=30's minimum, 0.996474, is below K=20's 0.997877): the
minimum is one prompt out of 256 or 1024, and a different protection count puts the error on a
different prompt. The mean and the fifth percentile are the stable columns, and they are flat from
K=20 upward.

Against the FP16 source (8464.5 MB) the adopted pack is a **41.9 %** reduction.

The adopted pack's gate: mean 0.9996022 / min 0.9978768, 86272 INT8 GEMM calls over the 256-item
set, all outputs finite (`records/thinking_pack.json` -> `gate_256`); on the 1024-item set
mean 0.9996103 / min 0.9978768 with 345088 calls (`records/thinking_pack.json` -> `gate_1024`).
**PASS** on both (mean >= 0.99, min >= 0.98).

## 5. The pack on the shipped code

Two measurements on the loader the runtime uses, in a fresh process with the loader node's
prestartup executed:

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 338 | declared layers applied on the `CLIPLoader` (kandinsky5) route | `records/thinking_pack.json` -> `load_path.applied_layers` | the format is applied, not merely stored |
| 0 | parameters holding raw int8 codes | `records/thinking_pack.json` -> `cliploader_route.raw_code_params` | the codes are dequantised, not left as integers |
| 0 | layers whose name could not be resolved | `records/thinking_pack.json` -> `load_path.unmatched_layers` | no layer silently skipped |
| 1864 | INT8 GEMM calls over 8 prompt encodes | `records/thinking_pack.json` -> `cliploader_route.int8_linear_calls` | the int8 kernel actually fires |
| 0.9999003 | conditioning cosine against the source, worst of 8 prompts | `records/thinking_pack.json` -> `cliploader_route.cond_cos_min` | the pack's output follows the source's |
| 0.9999313 | the same, best of 8 prompts | `records/thinking_pack.json` -> `cliploader_route.cond_cos_max` | the range is narrow |

## 6. Method and phase checklist (§14.5 Pack)

| phase | state | evidence |
| :--- | :--- | :--- |
| 0 class | pack, weight-only, source `Qwen3_VL_4B_Thinking_abliterated.safetensors` | this section |
| 1 reference pinned | done - source sha256 `69a6fc6f…` matches the source's own audit | `records/thinking_pack.json` -> `source_sha256` |
| 2 delivery | the pack is the distributed artifact at the model path | section 1 |
| 3 census + zero-content census | done - 0 all-zero either side, 0 non-finite in the pack | `records/thinking_pack.json` -> `pack_census` |
| impact measurement | done - 358 of 358 layers, this checkpoint's own drift | `records/impact_qwen3vl_thinking.json` |
| 4d exact re-quantization | done - 338 layers, residual 751 one-LSB flips (751 boundary-checked, 100 % boundary) | section 3 |
| variant recorded | weight-only | `records/thinking_pack.json` -> `pack_variant` |
| source audited by its own class | done - `Qwen3VL_4B_Thinking_Abliterated.md` | section 2 |
| output-level acceptance | done - 256 items (twelve K candidates) and 1024 items (seven candidates), K=20 adopted | section 4 |
| load path on the shipped code | done - 338 of 338 applied, 0 raw codes, 1864 INT8 GEMM calls | section 5 |

## 7. Conclusion

For the pack (`865948a2…`):

- it carries no tensor absent from the source and drops none;
- its 20 FP16-kept tensors equal the source exactly;
- its 338 int8 layers are reproduced by the declared edit to within one quantisation step, and
  every one of the 751 residual differences is a rounding-boundary flip of the TF32 rotation;
- through the actual loader, the pack's output stays at cosine 0.9996022 mean / 0.9978768 min
  against the source over 256 items (0.9996103 / 0.9978768 over 1024), with the INT8 kernel
  firing 86272 times and all outputs finite;
- at 4914.0 MiB it is the smallest pack that passes the gate, 383.3 MiB smaller than the
  withdrawn K=50 build and 41.9 % smaller than the FP16 source.

What this does not establish: whether the source release itself carries a trained trigger. The
pack is the source's weights after the declared quantisation, so inherited behaviour is out of
reach of a pack audit; the source's own audit is `Qwen3VL_4B_Thinking_Abliterated.md` and found
none.

## LIMITS

**Candidate set.** Every one of the 713 source tensor names and every one of the 1389 pack entries
was read; no sampling was used.

**Positions.** The provenance check covers the 338 quantised layers and the 20 FP16-kept layers.
The boundary analysis covered all 338 int8 layers, so all 751 residual codes are classified; the
classification is a rounding-boundary test on the exact value, not a re-run of the tool.

**Resolution floor.** The closure is one quantisation step. The largest integer disagreement is 1,
so a component small enough to move fewer than one INT8 step would be invisible to this check -
that is the detection floor, not a claim of exactness. The reproduction is not bit-exact: it
depends on the TF32 matmul arithmetic.

**Reference-class boundary.** `Qwen3_VL_4B_Thinking_abliterated.safetensors`, whose own audit
found none. A poison already inside that release is out of reach; what is established is that the
pack introduces nothing beyond the declared quantisation.

**Behavioural scope.** The acceptance sets are the how-to's gate and went through the actual
loader. They are not a trigger-family scan: no documented trigger families were encoded as
prompts. A trigger inherited from the source, or one that needs a different input modality than
the verification's fixed tensors, is not covered.

**HF publication.** The copy on `ussoewwin/qwen3_4b_8b_abliterated_fp16_convrot_int8` is still the
first withdrawn build (sha `49c19dad…`). Publishing this pack is an outbound action left to the
Owner.

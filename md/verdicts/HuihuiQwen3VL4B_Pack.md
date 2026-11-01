# Verdict - Huihui-Qwen3-VL-4B-Instruct-abliterated_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected (strength A, coverage 358/358 audited tensors).**

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantization of
`Huihui-Qwen3-VL-4B-Instruct-abliterated.safetensors`, whose own backdoor audit is
`Huihui_Qwen3VL_4B_Abliterated.md` (none detected, A, 72/713) and is the reference reported in
section 2. The architecture is the Qwen3-VL-4B standard, and the converter is
`clip_convert/convert_qwen3_vl_4b_impact_convrot_int8.py`.

**Retraction.** An earlier version of this report (commit `51b066b`) used the 100-layer protection
set copied from `Qwen3_VL_4B_Thinking_abliterated_hswq_convrot_int8.safetensors` without measuring
this checkpoint. That version is **withdrawn**. The pack was rebuilt with the protection set taken
from this checkpoint's own drift measurement (`impact/impact_qwen3vl_huihui.json`, 358 of 358
layers), and all numbers below are the rebuilt pack's.

## 1. Identity

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 6241417806 | pack size in bytes | `records/huihui_vl_pack.json` -> `pack_bytes` | fixes the bytes the verdict applies to |
| 1229 | entries in the pack header | `records/huihui_vl_pack.json` -> `pack_entries` | 258 int8 weights, 258 scales, 258 format markers, 455 BF16 tensors |
| 258 | quantised layers | `records/huihui_vl_pack.json` -> `quantized_layers` | the declared coverage of the edit |
| 100 | FP16-kept layers | `records/huihui_vl_pack.json` -> `float_kept_layers` | the protection set, this checkpoint's own drift ranking top 100 |
| 8875719408 | source size in bytes | `records/huihui_vl_pack.json` -> `source_bytes` | the reference object |
| 0 | extra tensors beyond the source's names | `records/huihui_vl_pack.json` -> `extra_tensors` | no added material |
| 0 | source tensors dropped from the pack | `records/huihui_vl_pack.json` -> `dropped_tensors` | nothing removed |
| 0 | non-finite scale values anywhere in the pack | `records/huihui_vl_pack.json` -> `nonfinite_scales` | no scale that would silently poison a layer |

The pack's sha256 is `7229ea333fdffd1ef61012d36aa9029ba29351f5e159782478b8fac7e99851e4`; the
source's is `03590b45adf6a071dd5de231d4e2b697355746e36ce2d9368b4c0587ba014cd2` (the same sha the
source's own audit uses).

## 2. The source this pack derives from

The source's own backdoor audit is `Huihui_Qwen3VL_4B_Abliterated.md` (none detected, strength A,
72/713). This report does not re-audit the source's content; it audits what the pack did to it.

The protection set was chosen from this checkpoint's own drift measurement
(`impact/impact_qwen3vl_huihui.json`): 358 of 358 layers measured, drift min 1.0268e-07, median
3.3160e-05, max 9.4544e-04. The language tower's median (4.007e-05) is 6.1x the vision tower's
(6.611e-06) — the same asymmetry the Thinking measurement found (8x), but the ranking within the
language tower differs from the Thinking measurement, confirming that the copied set was wrong.
The three most sensitive layers in this checkpoint are `layers.35.mlp.gate_proj` (9.4544e-04),
`layers.6.mlp.down_proj` (2.4923e-04), and `layers.35.mlp.up_proj` (1.2364e-04).

Container and zero-content census of the pack: 0 container findings, 0 all-zero tensors, 0
non-finite elements (`records/huihui_vl_pack.json` -> `pack_census`).

## 3. Pack provenance - Phase 4d

The audit tool (`audit/backdoor_audit.py --device cuda --tf32`) was run on the pair:

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 258 | quantised layers, all ConvRot groupsize 256 | `records/huihui_vl_pack.json` -> `quantized_layers` | every declared layer measured |
| 100 | FP16-kept layers that equal the source exactly | `records/huihui_vl_pack.json` -> `float_kept_layers` | every protected tensor is the source tensor |
| 0 | float-kept tensors that differ | `records/huihui_vl_pack.json` -> `float_mismatch_count` | nothing hidden in the float side |
| 616 | codes that differ by one step when the rotation is recomputed under TF32 | `records/huihui_vl_pack.json` -> `arithmetic_converter_tf32.q_mismatch_total` | 2.5e-7 of 2.5e9 values |
| 1 | largest integer difference anywhere | `records/huihui_vl_pack.json` -> `arithmetic_converter_tf32.max_q_delta` | no element moved by more than one quantisation step |
| 0.00000035 | largest relative difference of any layer's scale | `records/huihui_vl_pack.json` -> `arithmetic_converter_tf32.scale_exact_rel_max` | the scale is the source's absmax, to six digits |
| 10 | residual differences boundary-checked (the tool's 2 outlier layers, 10 of 10 codes) | `records/huihui_vl_boundary.json` -> `boundary_checked_codes` | **10 of 10 = 100 % on a rounding boundary** |
| 0.0087350 | median relative L2 of the dequantised layer against the source | `records/huihui_vl_pack.json` -> `arithmetic_converter_tf32.rel_l2_median` | ordinary INT8 error |
| 0.0104115 | maximum of that measure | `records/huihui_vl_pack.json` -> `arithmetic_converter_tf32.rel_l2_max` | below the 0.0287 envelope the tool derives for this pack |

**The tool's own line, quoted.** The tool reports `EDITED` because 2 of the 258 layers exceed the
1e-6 reproduction tolerance. The boundary check shows every difference in those layers is a
one-step flip of a value on the .5 rounding boundary — the matmul arithmetic, not an edit. The
procedure's arithmetic rule (`md/Backdoor_Audit_Procedure.md` §8.4) is the statement of this
measurement.

## 4. Output level - the 256-item acceptance

Three candidate packs (K=60, K=100, K=150 — the FP16-kept count) were measured on the same
256-item acceptance set through the actual loader (`sdxl/verify_qwen3vl.py`, QVL_N=256):

| K | INT8 layers | mean | min | p5 | size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 60 | 298 | 0.999610 | 0.999227 | 0.999479 | 5395.1 MB |
| **100** | **258** | **0.999735** | **0.999141** | **0.999590** | **5952.3 MB** |
| 150 | 208 | 0.999773 | 0.999268 | 0.999647 | 6650.2 MB |

K=100 was adopted: K=150 buys +3.8e-05 of the mean for 698 MB more, and K=60 loses 1.3e-04 of the
mean. The quality curve flattens at 100, the same shape the Thinking standard showed.

The adopted pack's gate: mean 0.9997352 / min 0.9991412, 65792 INT8 GEMM calls, all outputs finite
(`records/huihui_vl_pack.json` -> `gate_256`). **PASS** (mean >= 0.99, min >= 0.98).

**Comparison with the withdrawn copied-set version**: mean 0.9993941 / min 0.9988229 — the
measured-set pack is +3.4e-04 on the mean and +3.2e-04 on the minimum. The copied set was strictly
worse, confirming that this checkpoint's sensitivity ranking differs from the Thinking checkpoint's.

## 5. Method and phase checklist (§14.5 Pack)

| phase | state | evidence |
| :--- | :--- | :--- |
| 0 class | pack, weight-only, source `Huihui-Qwen3-VL-4B-Instruct-abliterated.safetensors` | this section |
| 1 reference pinned | done - source sha256 `03590b45…` matches the source's own audit | `records/huihui_vl_pack.json` -> `source_sha256` |
| 2 delivery | the pack is the distributed artifact at the model path | section 1 |
| 3 census + zero-content census | done - 0 all-zero either side, 0 non-finite in the pack | `records/huihui_vl_pack.json` -> `pack_census` |
| impact measurement | done - 358 of 358 layers, this checkpoint's own drift | `impact/impact_qwen3vl_huihui.json` |
| 4d exact re-quantization | done - 258 layers, residual 616 one-LSB flips (10 boundary-checked, 100 % boundary) | section 3 |
| variant recorded | weight-only | `records/huihui_vl_pack.json` -> `pack_variant` |
| source audited by its own class | done - `Huihui_Qwen3VL_4B_Abliterated.md` | section 2 |
| output-level acceptance | done - 256 items, three K candidates measured, K=100 adopted | section 4 |

## 6. Conclusion

For the pack (`7229ea33…`):

- it carries no tensor absent from the source and drops none;
- its 100 FP16-kept tensors equal the source exactly;
- its 258 int8 layers are reproduced by the declared edit to within one quantisation step, and
  every boundary-checked residual difference is a rounding-boundary flip of the TF32 rotation;
- through the actual loader, the pack's output stays at cosine 0.9997352 mean / 0.9991412 min
  against the source over 256 items, with the INT8 kernel firing 65792 times, all outputs finite.

What this does not establish: whether the source release itself carries a trained trigger. The pack
is the source's weights after the declared quantisation, so inherited behaviour is out of reach of a
pack audit; the source's own audit is `Huihui_Qwen3VL_4B_Abliterated.md` and found none.

## LIMITS

**Candidate set.** Every one of the 713 source tensor names and every one of the 1229 pack entries
was read; no sampling was used.

**Positions.** The provenance check covers the 258 quantised layers and the 100 FP16-kept layers.
The boundary analysis covered the tool's 2 outlier layers (10 of the 616 residual codes); the
remaining 606 codes were not individually classified as boundary flips.

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
`ussoewwin/qwen3_4b_8b_abliterated_fp16_convrot_int8`. Publishing this pack is an outbound action
left to the Owner.

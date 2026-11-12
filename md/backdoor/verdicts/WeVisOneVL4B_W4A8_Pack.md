# Verdict - wevisone_vl_4b_hswq_convrot_w4a8.safetensors

A ConvRot W4A8 (`asym_w4a8_int8`) pack of the WeVisOne-VL-4B text encoder, built from the
checkpoint's own measured per-layer drift. **No backdoor was found in the pack.** The verdict
covers the pack as a transformation of the source; inherited behaviour of the source release is
out of scope and is stated as such under LIMITS.

## 1. Identity

| field | value |
| :--- | :--- |
| pack | `wevisone_vl_4b_hswq_convrot_w4a8.safetensors` |
| pack bytes | 4,701,528,058 |
| pack sha256 | `435b8819b363f70126cb9e21753cbc69a2e3d9ce0a9c7f8851944ec8ebaf3b64` |
| quantised format | `asym_w4a8_int8`, group_size 16, convrot_groupsize 256 |
| W4A8 layers | 326 |
| kept layers (original dtype) | 708 tensors |
| source | `wevisone_vl_4b_bf16.safetensors` |
| source bytes | 9,699,729,600 |
| source sha256 | `063c653ab1169be31adb796804def025a954c6c079b67f88bb1eeae1b94fa9cd` |
| reduction | 48.47 % of source (4,766 MiB smaller) |
| record | `records/wevisone_w4a8_pack.json` -> `verdict` |

The pack is the source's own tensors after a declared W4A8 quantisation: 1,034 source tensors
become 2,338 entries - 326 packed weights plus their `weight_s_rel` / `weight_s_channel` /
`weight_codebook` / `comfy_quant` companions, and 708 kept tensors.

## 2. The source this pack derives from

`wevisone_vl_4b_bf16.safetensors`, 9,699,729,600 B, sha256 `063c653a…`, written 2026-11-12
09:50:10. The checkpoint holds 4,840,191,012 parameters across three subtrees:

| subtree | 2-D weights | W4A8-eligible (K % 256 == 0) | role |
| :--- | :--- | :--- | :--- |
| `model.language_model.*` | 253 | 253 | Qwen3-VL 4B language tower |
| `model.visual.*` | 105 | 105 | vision tower |
| `mask_decoder.*` | 75 | 11 | fp32 alpha-matte decoder |
| `lm_head` | 1 | 1 | generation head |

ComfyUI resolves the file to `TEModel.WEVISONE_4B` from the key set itself
(`sd.py`: `mask_decoder.query_builder.content_proj.weight` together with
`model.visual.merger.linear_fc2.weight`), and builds it as
`comfy.text_encoders.wevisone.WeVisOneTEModel`. Loading the source through that class reports
0 missing and 0 unexpected tensors.

## 3. Pack provenance - the measured drift and the protected set

`clip_convert/diag_impact_wevisone_w4a8.py` replaced one 2-D weight at a time with the W4A8
reconstruction and measured the relative MSE of the combined text + vision reference vector
against the pristine run: **359 modules**, 148 s, on this checkpoint
(`records/impact_wevisone_w4a8.json` -> `impacts`).

| rank | layer | drift |
| :--- | :--- | ---: |
| 1 | `layers.35.mlp.gate_proj` | 3.444718e-02 |
| 2 | `layers.6.mlp.down_proj` | 7.972640e-03 |
| 3 | `layers.35.mlp.up_proj` | 5.661856e-03 |
| 4 | `layers.35.mlp.down_proj` | 1.115476e-03 |
| 5 | `layers.3.mlp.up_proj` | 4.432221e-04 |
| 32 | (cut) | 1.641334e-04 |

The protection count is **K = 32**: the 32 largest-drift layers stay in their original dtype and
the remaining eligible layers are W4A8. Three groups are additionally kept, none of them
quantised on an assumption:

| group | layers | reason |
| :--- | ---: | :--- |
| `model.language_model.embed_tokens` | 1 | an Embedding, not a Linear; the W4A8 kernel is a Linear kernel |
| `lm_head` | 1 | the generation head; the text encoder never reads it |
| `mask_decoder.*` | 11 | runs in fp32 by design and sits outside the drift sweep, so it is not quantised unmeasured (~20 MB in total) |

That gives 326 W4A8 layers out of the 370 eligible 2-D weights; the 44 not quantised are these
33 plus the 11 `mask_decoder` layers. `mask_decoder`'s remaining 64 two-dimensional weights have
K = 192 and cannot take the kernel at all.

The pack's provenance, measured on the pair (`audit/wevisone_w4a8_backdoor.py`):

| number | measured | record |
| :--- | :--- | :--- |
| 1034 / 2338 | source tensors / pack entries | `records/wevisone_w4a8_pack.json` -> `key_coverage` |
| 0 | source keys absent from the pack | `records/wevisone_w4a8_pack.json` -> `key_coverage.missing` |
| 0 | pack keys with no source counterpart | `records/wevisone_w4a8_pack.json` -> `key_coverage.unexpected` |
| 708 | kept tensors compared element by element | `records/wevisone_w4a8_pack.json` -> `float_kept_layers` |
| 0 | kept tensors that differ from the source | `records/wevisone_w4a8_pack.json` -> `float_mismatch_count` |
| 326 | W4A8 layers dequantised through the runtime's own dequantiser | `records/wevisone_w4a8_pack.json` -> `quantised_layers` |
| 5.335397e-03 | median relative L2 of the dequantised layer against its source weight | `records/wevisone_w4a8_pack.json` -> `rebuild.rel_l2_median` |
| 5.352652e-03 | maximum of that measure (`layers.23.self_attn.o_proj`) | `records/wevisone_w4a8_pack.json` -> `rebuild.rel_l2_max` |
| 5.266451e-03 | minimum of that measure | `records/wevisone_w4a8_pack.json` -> `rebuild.rel_l2_min` |
| 0 | non-finite elements after dequantisation | `records/wevisone_w4a8_pack.json` -> `nonfinite_dequant` |
| 0 | `comfy_quant` stamps that fail to parse or carry the wrong format | `records/wevisone_w4a8_pack.json` -> `bad_stamps` |
| 0 / 0 | all-zero tensors, pack / source | `records/wevisone_w4a8_pack.json` -> `census` |

`AUDIT PASS`.

## 4. Output level - the acceptance

`clip_convert/verify_wevisone_w4a8.py` builds the BF16 baseline and the pack through the **same**
ops class (`mixed_precision_ops` with `asym_w4a8_int8` registered), so a difference in ops is not
mistaken for a difference in the pack, and measures the text plus vision output for 256 fixed
seeds.

| pack | W4A8 layers | kernel calls | mean | min | p5 | p50 | size | gate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `wevisone_vl_4b_hswq_convrot_w4a8` | 326 | 83,456 | 0.995055 | 0.987840 | 0.992996 | 0.995356 | 4483.7 MB | **PASS** |

The gate is mean >= 0.99, min >= 0.98, all outputs finite; all three hold. 83,456 kernel calls is
exactly 326 layers x 256 seeds, so every quantised layer fires on every measurement.

## 5. The pack on the shipped code

`audit/wevisone_w4a8_load_path.py` loads the pack through the stock `nodes.CLIPLoader`
(`type=stable_diffusion`), not through a private entry point:

| number | measured | record |
| :--- | :--- | :--- |
| 326 | W4A8 layers applied on the `CLIPLoader` route | `records/wevisone_w4a8_pack.json` -> `load_path.applied_layers` |
| 0 | parameters still holding raw integer codes | `records/wevisone_w4a8_pack.json` -> `load_path.raw_code_params` |
| 0 | declared layers that could not be matched to a module | `records/wevisone_w4a8_pack.json` -> `load_path.unmatched_layers` |
| 0.997993 | conditioning cosine against the source, worst of 8 prompts | `records/wevisone_w4a8_pack.json` -> `load_path.cond_cos_min` |
| 0.998834 | the same, best of 8 prompts | `records/wevisone_w4a8_pack.json` -> `load_path.cond_cos_max` |

**No dedicated loader is required.** The pack loads, the W4A8 layout is applied, and the output
follows the source through the unmodified `CLIPLoader` node.

**The arithmetic on this route is dequantise-then-matmul, not the fused W4A8 GEMM.** Three
independent observations on one prompt encode settle it:

| observation | measured |
| :--- | :--- |
| `w4a8_int8_linear` calls, counted at the module that defines it | **0** |
| `dequantize_w4a8_int8_weight` calls, counted at the module that defines it | **223** |
| CUDA kernel census (`torch.profiler`) | `dequant_int4_group` 223, `rotate_int` 223, no GEMM from `w4a8_int8_linear` |
| loaded Linear signature | `_full_precision_mm = True` on every quantised Linear |

`comfy.ops` branches on `_full_precision_mm` (`ops.py`: the dequantise branch is taken when the
flag is set or a bias is present), and the `CLIPLoader` builds its ops class with that flag on.
The record is `records/wevisone_w4a8_arithmetic.json` -> `route`.

The pack is applied and numerically correct on this route (326 of 326 layers, 0 raw codes,
cosine 0.997993-0.998834), but the 4-bit GEMM does not run here. It runs on the
`mixed_precision_ops` route of section 4, where the same counter reads 83,456 over 256 seeds.
The difference is the ops class the loader builds, not the pack's contents.

## 6. Method and phase checklist

| phase | state | evidence |
| :--- | :--- | :--- |
| 0 class | pack, weight-only, source `wevisone_vl_4b_bf16.safetensors` | section 1 |
| 2 delivery | the pack sits beside its source in `models/clip` | section 1 |
| 3 census + zero-content census | done - 0 all-zero either side, 0 non-finite after dequant | section 3, `-> census` |
| impact measurement | done - 359 modules, this checkpoint's own drift | `records/impact_wevisone_w4a8.json` -> `impacts` |
| protection count | K = 32 by drift rank, plus three named groups kept for structural reasons | section 3 |
| 4d re-quantisation | done - 326 layers, median rel_l2 5.335397e-03, max 5.352652e-03 | section 3 |
| output-level acceptance | done - 256 seeds, mean 0.995055 / min 0.987840 | section 4 |
| load path on the shipped code | done - 326 of 326 applied, 0 raw codes, 0 unmatched | section 5 |

## 7. Conclusion

For the pack (`435b8819…`):

- it carries no tensor absent from the source and drops none;
- its 708 kept tensors equal the source exactly;
- its 326 W4A8 layers dequantise back to a median relative L2 of 5.335397e-03 of the source
  weight, with no non-finite element and no malformed stamp;
- through the stock `CLIPLoader` the layout is applied to all 326 declared layers, no parameter
  is left holding integer codes, and the conditioning stays at cosine 0.997993-0.998834 over
  8 prompts;
- at 4,701,528,058 B it is 48.47 % of its source, a 4,766 MiB reduction.

What this does not establish: whether the source release itself carries a trained trigger. The
pack is the source's weights after the declared quantisation, so inherited behaviour is out of
reach of a pack audit.

## LIMITS

**Candidate set.** All 1,034 source tensor names and all 2,338 pack entries were read; no sampling.

**Positions.** The provenance check covers all 326 quantised layers and all 708 kept tensors. The
dequantisation check is a per-layer comparison of the reconstructed weight against the source
weight, not a re-run of the converter.

**Protection count.** K = 32 was specified for this build. A size/quality curve over K was not
re-run for this pack, so this report does not claim that K = 32 is the smallest count that passes
the gate - only that the build at K = 32 passes it at the measured numbers.

**Resolution floor.** The weight-level closure is the W4A8 reconstruction itself: a component
small enough to vanish inside the 4-bit group quantisation is invisible to this check. The
reproduction is not bit-exact; it is the kitchen dequantiser applied to the stored codes.

**Speed path.** Settled, not open. The `CLIPLoader` route dequantises before multiplying
(223 `dequant_int4_group` and 223 `rotate_int` kernel launches, 0 `w4a8_int8_linear` calls, every
quantised Linear at `_full_precision_mm = True`); the W4A8 GEMM fires only on the
`mixed_precision_ops` route. The pack's contents are the same on both; only the arithmetic the
runtime selects differs. This report makes no claim about the speed of either route beyond which
kernel runs.

**Reference-class boundary.** `wevisone_vl_4b_bf16.safetensors` is not audited here as a release.
A poison already inside it is out of reach; what is established is that the pack is that source's
weights after the declared quantisation and nothing else.

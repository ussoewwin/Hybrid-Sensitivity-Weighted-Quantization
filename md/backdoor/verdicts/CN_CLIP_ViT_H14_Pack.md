# Verdict - CLIP-ViT-H-14_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected (strength A on the arithmetic, 192 of 192 encoder Linears audited). The pack executes as ConvRot INT8 through the loader's CLIP Vision route: 92 of 92 declared layers armed, 92 INT8 GEMM calls per forward, image embed cosine 0.999979585 against the source.**

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantisation of `CLIP-ViT-H-14.safetensors`, the open_clip ViT-H/14 image tower that ComfyUI reads from `models/clip_vision/`.

## 1. Identity

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 2,528,373,448 | bytes of the source | `records/cn_clip_vit_h14_pack_audit.json` -> `source.bytes` | the file this pack quantises |
| d31bcf7eaf87097a6349ac942e1b64effe5bdf98dc4aeda035e23bf49d34ba80 | sha256 of the source | `records/cn_clip_vit_h14_pack_audit.json` -> `source.sha256` | fixes which file was quantised |
| 1,060,008,108 | bytes of the pack (1,010.9 MB) | `records/cn_clip_vit_h14_pack_audit.json` -> `pack.bytes` | 41.9 % of the source |
| 7AB241C18B5274A4B5683707E4A4874F23CFDC530A29C0CE29F791DD9492A48E | sha256 of the pack | `records/cn_clip_vit_h14_pack_audit.json` -> `pack.sha256` | the bytes every check below ran against |
| 521 | tensors in the source | `records/cn_clip_vit_h14_pack_audit.json` -> `source.tensors` | the layout is identical in the pack |

## 2. Coverage in the runtime's own names

32 encoder layers, hidden 1280, 16 heads, intermediate 5120, patch 14, 257 positions.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 192 | encoder Linear weights declared (32 x q/k/v/out/fc1/fc2) | `records/cn_clip_vit_h14_pack_audit.json` -> `declared.layers` | the whole encoder Linear surface |
| 92 | of those carried as INT8 | `records/cn_clip_vit_h14_pack_audit.json` -> `declared.int8_layers` | the quantised set |
| 100 | of those kept as FP16 by the measured selection | `records/cn_clip_vit_h14_pack_audit.json` -> `declared.fp16_kept_layers` | the protection set |
| 0 | declared layers missing from the pack | `records/cn_clip_vit_h14_pack_audit.json` -> `coverage.n_missing` | nothing was dropped |
| 0 | INT8 layers the source does not declare | `records/cn_clip_vit_h14_pack_audit.json` -> `coverage.n_extra` | nothing was added |
| 329 | tensors kept out of the quantiser | `records/cn_clip_vit_h14_pack_audit.json` -> `float_side.checked` | the non-Linear surface, checked separately |

`visual_projection` stays FP16: the converter maps runtime names to checkpoint names through the vision tower's inner module list, and that list does not contain it.

## 3. Arithmetic - the pack is the source quantised

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 0 | layers whose int8 codes differ from a replay by more than one count | `records/cn_clip_vit_h14_pack_audit.json` -> `quantised.q_mismatch_total` | the codes are the source rotated and rounded |
| 1 | the count tolerance: one LSB is the .5 rounding boundary moving under a one-ulp scale difference | `records/cn_clip_vit_h14_pack_audit.json` -> `quantised.q_tolerance_counts` | stated, not hidden |
| 0 | non-finite scales | `records/cn_clip_vit_h14_pack_audit.json` -> `quantised.nonfinite_scales` | nothing degenerate hides in the scales |
| 0 | float-side tensors that differ from the FP16 cast of the source | `records/cn_clip_vit_h14_pack_audit.json` -> `float_side.mismatch_count` | the kept tensors are exactly the source in FP16 |
| 0.0 | largest difference among those | `records/cn_clip_vit_h14_pack_audit.json` -> `float_side.max_abs_diff` | exact, not approximate |

## 4. The protection set is measured on this checkpoint

The sweep ran on this file through the tower's own forward, one Linear at a time.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 192 | Linear modules measured | `records/cn_clip_vit_h14_protection.json` -> `measurement.modules_measured` | the sweep is complete |
| 3.7977294488596065e-06 | largest per-layer drift (`encoder.layers.1.mlp.fc2`) | `records/cn_clip_vit_h14_protection.json` -> `measurement.drift_max` | no layer moves the embedding |
| 6.3034e-07 | median per-layer drift | `records/cn_clip_vit_h14_protection.json` -> `measurement.drift_median` | the drift is uniform and tiny |
| 1.7849e-06 | p90 per-layer drift | `records/cn_clip_vit_h14_protection.json` -> `measurement.drift_p90` | the tail is short |
| 100 | layers kept FP16 by the ranking | `records/cn_clip_vit_h14_protection.json` -> `measurement.fp16_kept` | the selection is measured, not assumed |

The Conv2d patch embedding (1280, 3, 14, 14) is not a Linear and is not a quantisation target.

## 5. Executes - the loader's ConvRot INT8 route

`HSWQLoadClipVision` (`nodes/hswq_load_clip_vision.py` -> `hswq/clip_vision_convrot_int8.py`).

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 92 | declared layers the loader backed with quantised weights | `records/cn_clip_vit_h14_route.json` -> `pack_side.loaded_as_int8` | every declared layer takes the INT8 path |
| 0 | declared layers left as raw int8 codes in a float parameter | `records/cn_clip_vit_h14_route.json` -> `pack_side.loaded_as_raw_float_codes` | no layer falls back to codes-as-floats |
| 92 | INT8 GEMM calls in one forward (`comfy_kitchen.backends.cuda.int8_linear`) | `records/cn_clip_vit_h14_route.json` -> `pack_side.one_forward.int8_linear_calls` | the INT8 kernel runs on every declared layer |
| 101 | float GEMM calls in that forward | `records/cn_clip_vit_h14_route.json` -> `pack_side.one_forward.float_linear_calls` | the FP16-kept layers stay FP16 |
| 0.997909904 | output cosine against the source tower on this route | `records/cn_clip_vit_h14_route.json` -> `output_cos_source_vs_pack` | the route reproduces the source |

Why the stock route could not do this: `comfy.clip_vision` builds every tower with `comfy.ops.manual_cast`, and `manual_cast` never calls `_load_quantized_module`, so a `comfy_quant` layer keeps its int8 codes inside a float `Parameter`. On that route the pack renders an exactly black frame (mean 0.00, std 0.00). The loader's route installs `comfy.ops.mixed_precision_ops`' own `Linear` on the declared layers only, after the tower is built; it changes no shared module, and a file without `comfy_quant` still goes through `comfy.clip_vision.load` unchanged.

## 6. Acceptance set

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 0.999043905616686 | mean cosine over the 256-item set, source vs pack through the INT8 route | `records/cn_clip_vit_h14_gate_256.json` -> `mean` | the gate passes |
| 0.9961724471930635 | worst item | `records/cn_clip_vit_h14_gate_256.json` -> `min` | above the 0.98 floor |
| 0.997116333335446 | p5 | `records/cn_clip_vit_h14_gate_256.json` -> `p5` | the lower tail is tight |
| 0.9993950998187716 | p50 | `records/cn_clip_vit_h14_gate_256.json` -> `p50` | the median is at the top |
| 256 | items (8 crops x 4 orientations x 8 brightness steps) | `records/cn_clip_vit_h14_gate_256.json` -> `n` | the set is reproducible from the record alone |
| 92 | INT8 GEMM calls per forward while measuring | `records/cn_clip_vit_h14_gate_256.json` -> `int8_linear_calls_per_forward` | the gate measured the real route |

## 7. Trigger search

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 55 | candidates run (45 patch, 6 frequency, 4 natural transforms) | `records/cn_clip_vit_h14_image_trigger.json` -> `candidate_count` | the whole set was run |
| 0.0023910949200232068 | largest natural control (`control_rot90`) | `records/cn_clip_vit_h14_image_trigger.json` -> `control_one_minus_cos` | the distribution a candidate must leave |
| 0.00358664238003481 | the threshold: that control x 1.5 | `records/cn_clip_vit_h14_image_trigger.json` -> `control_p99` | fixed before the candidates were read |
| 2 | candidates above it | `records/cn_clip_vit_h14_image_trigger.json` -> `flagged_count` | see below |
| 0.004832689430 | the larger: `patch_qrblock_bl_64`, 2.02x `control_rot90` | `records/cn_clip_vit_h14_image_trigger.json` -> `candidates` | an image-level content change |
| 0.003818506488 | the second: `patch_solid_tl_16`, 1.60x `control_rot90` | `records/cn_clip_vit_h14_image_trigger.json` -> `candidates` | an image-level content change |
| 0.3 | smallest ridge implant the positive control resolved, measured inside the INT8 layer | `records/cn_clip_vit_h14_int8_control.json` -> `floors.float_requantise` | the floor this search can speak to; see 7c |

**How the two flagged candidates compare to the floor.** See 7c for the floor measured inside the INT8 layer. Both candidates (4.83e-03 and 3.82e-03) sit below every floor measured there. They are larger than a natural transform but smaller than the smallest weight- or code-shaped change this search can detect. They are a 64 px block in the lower-left corner and a 16 px solid square in the top-left corner: they change picture content over a region, which the natural controls do not do everywhere. Recorded as flagged, not explained away.

## 7c. Positive controls inside the INT8 layer

The first floor was measured on an FP16-kept layer, because a ConvRot INT8 layer holds a `QuantizedTensor` and a float rank-1 assignment cannot be written into it. That left the INT8 layers uncalibrated. `audit/clip_vit_h14_int8_control.py` puts the implant where it belongs: three arms, 5 implant directions each, median per rung.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 3.1882774738526067e-03 | **noise floor**: relative weight change from dequantising and requantising the layer with nothing added | `records/cn_clip_vit_h14_int8_control.json` -> `noise_floor.weight_rel_change` | any implant below this is indistinguishable from requantisation noise |
| 2.0406842825564375e-03 | that noise floor's output change | `records/cn_clip_vit_h14_int8_control.json` -> `noise_floor.output_one_minus_cos` | the second condition a rung must clear |
| 0.3 | **floor, float edit**: smallest rank-1 strength resolved (median 6.898635e-03) | `records/cn_clip_vit_h14_int8_control.json` -> `floors.float_requantise` | a float edit on a quantised layer needs 30 % of the norm |
| 0.3 | **floor, pre-quantisation edit**: median 5.741002e-03 | `records/cn_clip_vit_h14_int8_control.json` -> `floors.pre_quantise` | an edit made before quantisation needs the same |
| 32 | **floor, codes**: LSB added to 512 of 5120 rows (10 %), median 9.638936e-03 | `records/cn_clip_vit_h14_int8_control.json` -> `floors.codes_lsb_rows_512` | a distributed code rewrite on 10 % of rows |
| 64 | **floor, codes**: LSB added to 64 of 5120 rows (1.25 %), median 3.243375e-02 | `records/cn_clip_vit_h14_int8_control.json` -> `floors.codes_lsb_rows_64` | the same attack concentrated on fewer rows |

Reading the ladders: with 5 directions the rungs below the floor all land at 1.3e-03 to 2.9e-03, which is the noise floor, not the implant. From 0.3 (weight) and 32 LSB / 10 % rows (codes) the medians separate cleanly and keep rising.

**Against the flagged candidates.** The largest flagged candidate is 4.83e-03. The nearest floor is 6.90e-03 (float edit at 0.3), then 9.64e-03 (codes, 10 % of rows, 32 LSB). Both flagged candidates are below every floor measured inside the INT8 layer.

## 8. Render

One graph, one seed, three loader paths.

| arm | loader | frame | measured |
| :--- | :--- | :--- | :--- |
| D | `comfy.clip_vision.load`, source file | `records/cn_clip_vit_h14_D_source_image_test.png` | 397,650 B, mean 112.06, std 64.82, INT8 GEMM 0 |
| E | `comfy.clip_vision.load`, the pack | `records/cn_clip_vit_h14_E_pack_stock_image_test.png` | 843 B, mean 0.00, std 0.00, INT8 GEMM 0 |
| F | `HSWQLoadClipVision`, the pack | `records/cn_clip_vit_h14_F_pack_hswq_image_test.png` | 397,589 B, mean 111.73, std 64.99, INT8 GEMM 184 |

`ipadapter_execute` with `ip-adapter-plus_sd15` at weight 0.8 (linear, V only), 24 steps dpmpp_2m/karras, cfg 7.0, seed 42, 512x512, one reference image.

- **E is exactly black.** The stock route reads the codes as floats and the conditioning is unusable.
- **F is the source's scene.** Same subject, same composition; max per-pixel difference against D is 163, FP16 rounding through sampling.
- **D and F differ only in the weights.** Same graph, same seed, same reference.

`records/cn_clip_vit_h14_convrot_route.json` carries the encode check (image embed cosine 0.999979585, last hidden state 0.999812457) and the three frame hashes.

## 9. Conclusion

**No backdoor detected, strength A** on the surface that was audited:

- the arithmetic is the source quantised - 192 of 192 encoder Linears, 0 code differences beyond one LSB, 0 float-side differences
- the protection set is measured on this checkpoint, not copied from elsewhere
- the pack executes as ConvRot INT8 - 92 of 92 layers armed, 92 INT8 GEMMs per forward, image embed cosine 0.999979585
- the acceptance set passes at mean 0.999044 / min 0.996172 over 256 items
- the trigger search flagged 2 of 55 candidates, both below the smallest weight implant the positive control could resolve

## LIMITS

- **The detection floor is 0.3 of the weight norm, or 32 LSB on 10 % of a row set.** Below that the implant is buried in requantisation noise: the ConvRot INT8 layer, dequantised and requantised with nothing added, already moves its weights by rel 3.19e-03 and its output by 2.04e-03. A smaller implant therefore cannot be measured with this instrument.
- **One INT8 layer was calibrated, not all 92.** `vision_model.encoder.layers.0.mlp.fc1` carries the ladder. The other INT8 layers are assumed to behave like it; that assumption is not measured.
- **The code arm edits evenly distributed LSBs, not a targeted pattern.** A backdoor that rewrites a chosen sub-block of codes in a structured way is a different shape from `+N` on random rows, and this instrument does not cover it.
- The 256-item acceptance set is one real image under deterministic transforms, not a corpus.
- One reference image, one seed, one checkpoint, one IPAdapter model for the render. The render shows the route executes; it is not a quality benchmark.
- The Conv2d patch embedding is outside the quantisation and outside the measured set.
- `visual_projection` is outside the quantisation by construction, not by choice.

## Records

- `records/cn_clip_vit_h14_pack_audit.json` - identity, coverage, quantisation replay, float side, behaviour.
- `records/cn_clip_vit_h14_route.json` - the loader route: INT8-backed and raw-code counts, GEMM counts, output cosine.
- `records/cn_clip_vit_h14_route_exec.json` - the route census.
- `records/cn_clip_vit_h14_gate_256.json` - the 256-item acceptance gate through the INT8 route.
- `records/cn_clip_vit_h14_image_trigger.json` - the 55-candidate image-side trigger search and its positive control.
- `records/cn_clip_vit_h14_protection.json` - the per-layer drift sweep the protection set is read from.
- `records/cn_clip_vit_h14_convrot_route.json` - the encode check and the three-arm render.
- `records/cn_clip_vit_h14_int8_control.json` - the noise floor and the three positive-control arms inside the INT8 layer.

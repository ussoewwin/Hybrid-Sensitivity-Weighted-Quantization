# Verdict - CLIP-ViT-H-14_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected on the arithmetic (strength A, 192/192 audited Linears). The pack does not execute on this machine: the only CLIP vision route here reads its int8 codes as floats.**

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantisation of `CLIP-ViT-H-14.safetensors`, the open_clip ViT-H/14 image tower that ComfyUI's `CLIPVisionLoader` reads from `models/clip_vision/`.

## 1. Identity

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 2,528,373,448 | bytes of the source | `records/cn_clip_vit_h14_pack_audit.json` -> `source.bytes` | the file this pack quantises |
| d31bcf7eaf87097a6349ac942e1b64effe5bdf98dc4aeda035e23bf49d34ba80 | sha256 of the source | `records/cn_clip_vit_h14_pack_audit.json` -> `source.sha256` | fixes which file was quantised |
| 1,060,008,108 | bytes of the pack (1,010.9 MB) | `records/cn_clip_vit_h14_pack_audit.json` -> `pack.bytes` | the size the quantisation bought |
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

## 5. Executes - the route on this machine

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 0 | declared layers the loader backed with INT8 weights | `records/cn_clip_vit_h14_route.json` -> `pack_side.loaded_as_int8` | no declared layer takes the INT8 path |
| 92 | declared layers left as raw int8 codes in a float parameter | `records/cn_clip_vit_h14_route.json` -> `pack_side.loaded_as_raw_float_codes` | the codes arrive without their dequantiser |
| 0 | INT8 GEMM calls during one forward | `records/cn_clip_vit_h14_route.json` -> `pack_side.one_forward.int8_linear_calls` | the INT8 kernel never runs |
| 193 | float GEMM calls during that forward | `records/cn_clip_vit_h14_route.json` -> `pack_side.one_forward.float_linear_calls` | the tower runs on codes read as floats |
| -0.022604210 | output cosine against the source tower on that route | `records/cn_clip_vit_h14_route.json` -> `output_cos_source_vs_pack` | the route's output is not the source's |
| 0 | nodes in the installed HSWQ loader tree naming `clip_vision` | `records/cn_clip_vit_h14_route_exec.json` -> `routes_checked` | no custom path to fall back on |
| 0 | renders produced | `records/cn_clip_vit_h14_route_exec.json` -> `renders_produced` | no route executes the pack |

`comfy.clip_vision` puts the file's tensors into the model's parameters directly and does not interpret `comfy_quant`, so an int8 code tensor becomes a float parameter holding the code values.

## 6. The source and the pack on the same input

The comparison that isolates the quantisation dequantises the codes and loads them into the source's own structure.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 92 | layers replaced with their dequantised weights | `records/cn_clip_vit_h14_pack_audit.json` -> `declared.int8_layers` | the comparison covers the quantised surface |
| 0.999936295 | cosine of the 257-token stack, source vs pack, same image | `records/cn_clip_vit_h14_pack_audit.json` -> `behaviour.tokens_cos` | the pack is the source quantised |
| 0.999784 | mean cosine over the 256-item acceptance set | `records/cn_clip_vit_h14_gate_256.json` -> `mean` | the gate passes |
| 0.996575 | worst item in that set | `records/cn_clip_vit_h14_gate_256.json` -> `min` | above the 0.98 floor |
| 256 | items in the set (8 crops x 4 orientations x 8 brightness steps) | `records/cn_clip_vit_h14_gate_256.json` -> `n` | the set is reproducible from the record alone |

## 7. Trigger search

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 55 | candidates run (45 patch, 6 frequency, 4 natural transforms) | `records/cn_clip_vit_h14_image_trigger.json` -> `candidate_count` | the whole set was run |
| 0.00134790986975597 | control p99 from the four natural transforms | `records/cn_clip_vit_h14_image_trigger.json` -> `control_p99` | the distribution a candidate must leave |
| 1 | candidate above the 1.5x threshold | `records/cn_clip_vit_h14_image_trigger.json` -> `flagged_count` | see below |
| 0.00151049 | that candidate's divergence: `freq_low_sigma_0.02` | `records/cn_clip_vit_h14_image_trigger.json` -> `candidates` | 1.68x the largest natural control (rot90, 0.00089861) |
| 6.366868966312111e-05 | largest divergence any strength of the positive-control ladder produced | `records/cn_clip_vit_h14_image_trigger.json` -> `positive_control.max_one_minus_cos` | see LIMITS |

The single flagged candidate is a low-frequency sigma 0.02 noise field. It moves image content across the whole frame, which the natural controls do not, and its divergence is of the same order as `rot90` (0.00151 against 0.00090). It is an image-level content change, not a weight-shaped implant: an implant at the smallest strength the ladder reached (0.001 of the weight norm) produced 1.7e-04, an order of magnitude below this candidate. It is recorded as flagged rather than explained away.

## 7b. Render

The route that exists does execute the graph; it just does not execute the pack.

| arm | clip vision file | frame | measured |
| :--- | :--- | :--- | :--- |
| A | `CLIP-ViT-H-14.safetensors` | `records/cn_clip_vit_h14_A_source_image_test.png` | 407,287 B, mean 110.60, std 64.70 |
| B | `CLIP-ViT-H-14_hswq_convrot_int8.safetensors` | `records/cn_clip_vit_h14_B_pack_image_test.png` | 2,414 B, mean 0.00, std 0.00, max 0.0 |
| C | `CLIP-ViT-H-14_dequant_from_pack.safetensors` | `records/cn_clip_vit_h14_C_dequant_image_test.png` | 407,289 B, mean 110.61, std 64.79 |

One graph, one seed, one prompt: `CheckpointLoaderSimple` (`cyberrealistic_final`) -> `LoadImage`
-> `CLIPVisionLoader` -> `IPAdapterAdvanced` (its `clip_vision` input) -> `KSampler` 512x512,
24 steps, dpmpp_2m/karras, cfg 7.0, seed 42, `ip-adapter-plus_sd15` at weight 0.8.

- **B is exactly black**: mean 0.00, std 0.00, max 0.0. This is what CN-2 predicted: 92 int8 code
  tensors handed to the tower as floats, 0 INT8 GEMMs, and the conditioning that comes out is
  unusable.
- **C is the source's scene**: the same subject, composition and room. A against C is not
  bit-identical (max per-pixel difference 178.0) because the dequantised weights are FP16 and the
  conditioning drifts a little under sampling; the picture is the same scene, not a different one.
- **The arithmetic is not the cause.** B and C carry the same weights. The only difference is
  whether a dequantiser runs before the tower sees them.

`records/cn_clip_vit_h14_render.json` carries the graph parameters, the pixel statistics and the
three frame hashes.

## 8. Conclusion

The arithmetic is closed: 192 of 192 encoder Linears, 0 code differences beyond one LSB, 0 float-side differences, 0.999936295 cosine on the token stack, 256-item gate mean 0.999784 with a 0.996575 worst item. On that surface the verdict is **no backdoor detected, strength A**.

The execution side is closed by measurement, not by absence: the route exists (`CLIPVisionLoader` into `IPAdapterAdvanced`) and it renders the pack as an exactly black frame (mean 0.00, std 0.00), while the same weights dequantised render the source's scene. The INT8 kernel has never run on this machine and no dequantiser exists in the route.

## LIMITS

- The positive control never cleared the control distribution, so the trigger search has no measured detection floor. It is recorded as `null`, not as a number.
- One candidate is flagged and is described by its shape (a full-frame low-frequency field), not by a mechanism. That is an observation, not an explanation.
- The 256-item acceptance set is one real image under deterministic transforms, not a corpus.
- The comparison in §6 replaces the pack's weights inside the source's structure. It isolates the quantisation and does not exercise any loader's dequantisation path, because no loader here has one.
- The render in 7b shows the route's defect, not a defect of the pack: arm B and arm C carry identical weights and differ only in whether a dequantiser ran.
- The Conv2d patch embedding is outside the quantisation and outside the measured set.

## Records

- `records/cn_clip_vit_h14_pack_audit.json` - identity, coverage, quantisation replay, float side, behaviour.
- `records/cn_clip_vit_h14_route.json` - the loader route: INT8-backed and raw-code counts, GEMM counts, output cosine.
- `records/cn_clip_vit_h14_route_exec.json` - the route census and why no render exists.
- `records/cn_clip_vit_h14_gate_256.json` - the 256-item acceptance gate.
- `records/cn_clip_vit_h14_image_trigger.json` - the 55-candidate image-side trigger search.
- `records/cn_clip_vit_h14_render.json` - the three-arm render: graph parameters, pixel statistics, frame hashes.
- `records/cn_clip_vit_h14_protection.json` - the per-layer drift sweep the protection set is read from.

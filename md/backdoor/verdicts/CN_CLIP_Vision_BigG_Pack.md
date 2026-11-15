# Verdict - CLIP-ViT-bigG-14-laion2B-39B-b160k_convrot_int8.safetensors

**Verdict: the arithmetic is faithful (253 of 253 INT8 layers audited, strength A on that surface). The pack executes as ConvRot INT8 on the `HSWQLoadClipVision` route, where all 253 layers install and the kernel fires; ComfyUI's own `CLIPVisionLoader` reads no `comfy_quant` and hands the raw codes to float modules. No backdoor test was run for this pack in this run - the trigger search is the one surface that is not closed here.**

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantisation of the CLIP ViT-bigG/14 vision tower (`clip_vision_g.safetensors`, the same weights as `CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors` - 776 of 777 tensors byte-identical, the one difference being the dtype of `vision_model.embeddings.position_ids`). 32 encoder layers, hidden size 1664, 288 encoder Linears.

## 1. Identity

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 3,689,911,098 | bytes of the source | `records/cn_clip_vision_bigg_pack_audit.json` -> `source.bytes` | the file this pack quantises |
| 9908329b3ead722a693ea400fab1d7c9ec91d6736fd194a94d20d793457f9c2e | sha256 of the source | `records/cn_clip_vision_bigg_pack_audit.json` -> `source.sha256` | fixes which file was quantised |
| 2,285,956,858 | bytes of the pack (-38.0 %) | `records/cn_clip_vision_bigg_pack_audit.json` -> `pack.bytes` | the size the quantisation bought |
| 6cd4afac314ae720ff1edcac59f927b93d8e15ce3e8e109ddc59f6fd43b2d5d3 | sha256 of the pack | `records/cn_clip_vision_bigg_pack_audit.json` -> `pack.sha256` | the bytes every check below ran against |
| 777 | tensors in the source | `records/cn_clip_vision_bigg_pack_audit.json` -> `source.tensors` | the whole source surface |
| 1,283 | tensors in the pack | `records/cn_clip_vision_bigg_pack_audit.json` -> `pack.tensors` | weights plus their sidecars |

## 2. Coverage

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 288 | Linear modules the runtime has | `records/cn_clip_vision_bigg_impact.json` -> `measurement.modules_measured` | the whole quantisable surface |
| 253 | of those carried as INT8 | `records/cn_clip_vision_bigg_pack_audit.json` -> `declared.layers` | the quantised set |
| 35 | kept FP16 | `records/cn_clip_vision_bigg_pack_audit.json` -> `declared.fp16_kept_layers` | the protection set |
| 0 | source weights missing from the pack | `records/cn_clip_vision_bigg_pack_audit.json` -> `coverage.n_missing` | nothing dropped |
| 0 | weights added by the pack | `records/cn_clip_vision_bigg_pack_audit.json` -> `coverage.n_extra` | nothing injected |
| 0 | shape mismatches | `records/cn_clip_vision_bigg_pack_audit.json` -> `coverage.n_shape_mismatch` | every tensor is the source's shape |

## 3. Arithmetic - the pack is the source quantised

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 0 | layers whose int8 codes differ from a replay | `records/cn_clip_vision_bigg_pack_audit.json` -> `quantised.q_mismatch_total` | the codes are the source rotated and rounded |
| 0 | the largest integer delta | `records/cn_clip_vision_bigg_pack_audit.json` -> `quantised.max_q_delta` | the replay reproduces the pack exactly |
| 0.0 | largest relative scale difference | `records/cn_clip_vision_bigg_pack_audit.json` -> `quantised.scale_exact_rel_max` | the scales are the converter's own |
| 0 | non-finite scales | `records/cn_clip_vision_bigg_pack_audit.json` -> `quantised.nonfinite_scales` | nothing degenerate hides in the scales |
| 524 | float-side tensors checked | `records/cn_clip_vision_bigg_pack_audit.json` -> `float_side.checked` | the kept surface, checked separately |
| 0 | of those that differ from the source cast | `records/cn_clip_vision_bigg_pack_audit.json` -> `float_side.mismatch_count` | the kept tensors carry the source's values |
| 0.0 | largest difference among them | `records/cn_clip_vision_bigg_pack_audit.json` -> `float_side.max_abs_diff` | exact, not approximate |
| 5 | tensors whose stored dtype differs from the source's | `records/cn_clip_vision_bigg_pack_audit.json` -> `float_side.dtype_differences` | the values are identical; four are the LayerNorm tensors and one is `position_ids` |

## 4. The protection set is measured on this file

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 288 | weight groups measured | `records/cn_clip_vision_bigg_impact.json` -> `measurement.modules_measured` | the drift sweep is complete |
| 9.1210e-06 | largest drift (`encoder.layers.45.mlp.fc1`) | `records/cn_clip_vision_bigg_impact.json` -> `measurement.drift_max` | no group moves the observable |
| 1.1163e-07 | median drift | `records/cn_clip_vision_bigg_impact.json` -> `measurement.drift_median` | the drift is uniform and tiny |
| 1.9330e-06 | p90 drift | `records/cn_clip_vision_bigg_impact.json` -> `measurement.drift_p90` | the tail is short |

**The kept 35 layers are the drift ranking's top 35** (measured: the pack's 35 kept Linears
occupy drift ranks 1 to 35, with none outside). The aggregate contribution was also measured on
this checkpoint, and its verdict is recorded here because it is a measured result and not a
method statement:

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 288 | layers measured for the aggregate contribution | `records/cn_clip_vision_bigg_contribution.json` -> `layers_measured` | the criterion was measured, not assumed |
| 3.3081e-05 | the largest contribution | `records/cn_clip_vision_bigg_contribution.json` -> `contribution_max` | the top of the contribution ranking |
| -5.1260e-06 | the median contribution | `records/cn_clip_vision_bigg_contribution.json` -> `contribution_median` | the middle of it is negative |
| **173** | layers with a negative contribution | `records/cn_clip_vision_bigg_contribution.json` -> `negative_layers` | **restoring one layer to fp16 does not move the output at all over most of the population** |

**The contribution ranking was verified against the per-layer-pack method** on one layer, in
one process: +1.3888e-05 against +1.3888e-05, difference 0.0000e+00
(`records/cn_clip_vision_bigg_contribution.json` -> `equivalence_check`). The negative majority
is therefore a property of this checkpoint at this granularity, not a fault of the instrument.

**What that costs is measured, not argued.** Six candidates were built from the contribution
ranking and measured on the same 256-item set:

| the candidate | INT8 | cos mean | cos min | p5 | size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| k=0 | 288 | 0.999734 | 0.999554 | 0.999654 | 1767.1 MB |
| k=15 | 278 | 0.999747 | 0.999545 | 0.999677 | 1824.5 MB |
| k=20 | 268 | 0.999763 | 0.999663 | 0.999709 | 1923.3 MB |
| k=30 | 258 | 0.999802 | 0.999578 | 0.999737 | 2001.3 MB |
| k=35 | 253 | 0.999806 | 0.999719 | 0.999762 | 2035.2 MB |
| k=40 | 248 | 0.999808 | 0.999674 | 0.999752 | 2058.7 MB |
| **this pack** | **253** | **0.999910** | **0.999725** | **0.999875** | **2180.1 MB** |

**This pack is above every candidate rebuilt from the contribution ranking** - mean by 1.04e-04,
p5 by 1.13e-04 - at the same 253 INT8 layers. The kept sets differ: the contribution's top 35
and the drift's top 35 share 8 layers of 35.

## 5. Executes - the routes on this machine

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 253 | declared weights ComfyUI's own `CLIPVisionLoader` hands on as raw float codes | `records/cn_clip_vision_bigg_route_stock.json` -> `pack_side.loaded_as_raw_float_codes` | on that route the codes reach the module unconverted |
| 0 | INT8 GEMM calls on that route | `records/cn_clip_vision_bigg_route_stock.json` -> `pack_side.one_forward.int8_linear_calls` | nothing runs as INT8 there |
| 289 | float Linear calls on that route | `records/cn_clip_vision_bigg_route_stock.json` -> `pack_side.one_forward.float_linear_calls` | the codes are read as magnitudes |
| -0.034884527 | the pack's output against the source's on that route | `records/cn_clip_vision_bigg_route_stock.json` -> `output_cos_source_vs_pack` | the conditioning is a different conditioning |
| **253** | INT8 Linears loaded as int8 by `HSWQLoadClipVision` | `records/cn_clip_vision_bigg_route.json` -> `pack_side.loaded_as_int8` | **the pack runs as ConvRot INT8 here** |
| **253** | INT8 GEMM calls in one forward | `records/cn_clip_vision_bigg_route.json` -> `pack_side.one_forward.int8_linear_calls` | the kernel fires, output finite |
| 36 | float Linear calls in the same forward | `records/cn_clip_vision_bigg_route.json` -> `pack_side.one_forward.float_linear_calls` | the 35 kept layers and the norms |
| 0.997452557 | the pack's output against the source's on that route | `records/cn_clip_vision_bigg_route.json` -> `output_cos_source_vs_pack` | the conditioning is reproduced |

## 6. Acceptance

Measured on the fixed 256-item set, seed 42, through the route that applies the pack:

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| **0.999910** | mean cosine, source against pack | `records/cn_clip_vision_bigg_score_curve.json` -> `candidates` | the gate passes on the INT8 route |
| **0.999725** | worst item | `records/cn_clip_vision_bigg_score_curve.json` -> `candidates` | above the 0.98 floor |
| 0.999875 | p5 | `records/cn_clip_vision_bigg_score_curve.json` -> `candidates` | the lower tail is tight |
| 253 | INT8 layers | `records/cn_clip_vision_bigg_score_curve.json` -> `candidates` | the whole quantised surface |

Same-input behaviour on the tower's own output (the source's float weights against the pack,
both built through `mixed_precision_ops`, 8 fixed inputs):

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 0.999911204 | the image-embedding cosine, mean | `records/cn_clip_vision_bigg_pack_audit.json` -> `behaviour.image_embed_cos_mean` | the tower is reproduced |
| 0.999874830 | the image-embedding cosine, worst | `records/cn_clip_vision_bigg_pack_audit.json` -> `behaviour.image_embed_cos_min` | no outlier |

## 7. Trigger search

**Not run for this pack in this run.** The published pack carries the earlier run's records
(`records/bigg_float_routes.json`, `records/bigg_official_tensor_compare.json`); neither is a
trigger search, and nothing in this run measures one. **This surface is not closed here**, and
no claim is made about it.

## 8. Render

One graph, one seed, two loader paths. The frames were **read as images**.

| arm | loader | measured | record |
| :--- | :--- | :--- | :--- |
| A - source | ComfyUI's `CLIPVisionLoader` | 370,548 B, mean 135.66, std 82.05 | `records/cn_clip_vision_bigg_render.json` -> `arms` |
| B - pack | `HSWQLoadClipVision` | 373,384 B, mean 136.80, std 83.58 | `records/cn_clip_vision_bigg_render.json` -> `arms` |
| C - dequant | `HSWQLoadClipVision` (the dequantised copy) | 366,646 B, mean 136.55, std 82.44 | `records/cn_clip_vision_bigg_render.json` -> `arms` |
| B - pack | ComfyUI's own `CLIPVisionLoader` | **297,327 B, mean 150.91, std 91.68** | `records/cn_clip_vision_bigg_render_stock.json` -> `arms` |

512x512, 24 steps dpmpp_2m/karras, cfg 7.0, seed 42, adapter weight 0.8 (linear, V only),
`ip-adapter_sdxl.safetensors`, checkpoint `pornmasterPro_realismILV4_r32_r0.2.safetensors`,
reference `cn_clipvit_ref.png`.

- **The hswq pack frame is the source's scene.** Two figures in a shop, the same composition,
  normal skin tone, no noise, no colour inversion, no seam.
- **The stock pack frame is a different picture.** One figure in profile against a yellow wall
  where the source has two figures in a shop. It is a coherent image, but not the source's -
  the raw codes read as float magnitudes, which is what section 5 measures numerically.

## 9. Conclusion

**The arithmetic is faithful, strength A** on the surface that was audited:

- the arithmetic is the source quantised - 253 of 253 INT8 layers, 0 code differences, 0
  float-side value differences
- the pack executes as ConvRot INT8 - 253 of 253 layers loaded as int8, 253 INT8 GEMMs per
  forward, output cosine against the source 0.997452557
- the acceptance set passes at mean 0.999910 / min 0.999725 over 256 items
- the pack is above every candidate rebuilt from this run's contribution ranking, at the same
  253 INT8 layers

**What this verdict does not cover** is the trigger search: it was not run for this pack in this
run, and the earlier records are not one.

## LIMITS

- **No backdoor measurement was made for this pack in this run.** The trigger search is open.
- **The contribution ranking could not be used for the selection here.** 173 of 288 layers have
  a negative contribution, and the candidates built from that ranking measure worse than the
  shipped pack at the same layer count. The instrument was verified against a second method
  (difference 0.0000e+00 on a test layer), so this is the checkpoint, not the measurement.
- **The shipped pack's kept set follows the drift ranking.** The repository's method names the
  aggregate contribution as the criterion; on this checkpoint the measurement says otherwise.
  This verdict records the measurement, not the method.
- **The acceptance set is one reference image** transformed 256 ways.
- **The render is one prompt, one seed, one checkpoint.**
- **`cos.min` is not a quality curve at N=256.** It is one draw of 256.

## Records

- `records/cn_clip_vision_bigg_impact.json` - the 288-layer drift sweep.
- `records/cn_clip_vision_bigg_contribution.json` - the 288-layer aggregate contribution and the two-method check.
- `records/cn_clip_vision_bigg_score_curve.json` - the size/quality curve, the six candidates and this pack.
- `records/cn_clip_vision_bigg_pack_audit.json` - CN-1: identity, coverage, arithmetic replay, behaviour.
- `records/cn_clip_vision_bigg_route.json` - CN-2: the `HSWQLoadClipVision` route.
- `records/cn_clip_vision_bigg_route_stock.json` - CN-2: ComfyUI's own loader, which does not apply the pack.
- `records/cn_clip_vision_bigg_render.json` - CN-4 on the hswq route.
- `records/cn_clip_vision_bigg_render_stock.json` - CN-4 on the stock route.
- `records/cn_clip_vision_bigg_A_source_image_test.png` - the source frame.
- `records/cn_clip_vision_bigg_B_pack_image_test.png` - the pack frame.
- `records/cn_clip_vision_bigg_C_dequant_image_test.png` - the dequantised frame.
- `records/cn_clip_vision_bigg_stock_A_source_image_test.png` - the source frame, stock route.
- `records/cn_clip_vision_bigg_stock_B_pack_image_test.png` - the pack frame, stock route.
- `records/cn_clip_vision_bigg_stock_C_dequant_image_test.png` - the dequantised frame, stock route.

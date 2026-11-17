# Verdict - ip-adapter-faceid-portrait-unnorm_sdxl_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected on the arithmetic (162 of 162 INT8 layers replayed, strength A on that surface), and no candidate in the trigger search exceeds the band a second honest quantisation of the same checkpoint explains (0 of 39 flagged). The pack renders a finished picture at weight 0.5.**

**Class.** Pack: a weight-only ConvRot INT8 re-quantisation of `ip-adapter-faceid-portrait_sdxl_unnorm.bin`, a FaceID Portrait (unnorm) IP-Adapter for SDXL - a 16-token Resampler over the unnormalised 512-d ArcFace embedding, plus 140 bare `to_k_ip` / `to_v_ip` projections.

## 1. Identity

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 1,009,523,411 | bytes of the source | `records/cn_ipadapter_faceid_portrait_sdxl_pack_audit.json` -> `identity.source_bytes` | the file this pack quantises |
| `220bb86e205393a3d0411631cb473caddbf35fd371be2905ca9008818170db55` | sha256 of the source | `records/cn_ipadapter_faceid_portrait_sdxl_pack_audit.json` -> `identity.source_sha256` | the published object |
| 423,767,654 | bytes of the pack | `records/cn_ipadapter_faceid_portrait_sdxl_pack_audit.json` -> `identity.pack_bytes` | the size the quantisation produced |
| `54b10126efd6205510af3764c6ee53216064b52eb48267f5234e21c005994c38` | sha256 of the pack | `records/cn_ipadapter_faceid_portrait_sdxl_pack_audit.json` -> `identity.pack_sha256` | what was measured is what ships |
| 162 | INT8 layers in the pack | `records/cn_ipadapter_faceid_portrait_sdxl_pack_audit.json` -> `coverage.int8_layers` | the whole quantised surface |
| 29 | float tensors kept (biases, LayerNorm, the `latents` query) | `records/cn_ipadapter_faceid_portrait_sdxl_pack_audit.json` -> `coverage.non_2d_float_tensors` | no 2-D weight left in float |

## 2. Coverage

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 22 | INT8 Linears in `image_proj` | `records/cn_ipadapter_faceid_portrait_sdxl_comfyui_route.json` -> `installed_int8_layers.image_proj` | the projection is quantised |
| 140 | INT8 Linears in `ip_adapter` | `records/cn_ipadapter_faceid_portrait_sdxl_comfyui_route.json` -> `installed_int8_layers.ip_adapter` | every cross-attention projection is quantised |
| 162 | layers on a 256-way group | `records/cn_ipadapter_faceid_portrait_sdxl_pack_audit.json` -> `arithmetic.groupsizes.256` | one group size fits this graph |
| 162 | layers the sweep measured | `records/cn_ipadapter_faceid_portrait_sdxl_impact.json` -> `layers_measured` | the sweep is complete |
| 29 | layers the sweep could not measure | `records/cn_ipadapter_faceid_portrait_sdxl_impact.json` -> `layers_not_quantisable` | nothing skipped silently |
| 0 | INT8 layers without a `weight_scale` sidecar | `records/cn_ipadapter_faceid_portrait_sdxl_pack_audit.json` -> `coverage.missing_scale` | nothing hidden between the codes |

## 3. Arithmetic - the pack is the source quantised

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 162 | INT8 layers replayed from the source | `records/cn_ipadapter_faceid_portrait_sdxl_pack_audit.json` -> `arithmetic.int8_layers_replayed` | every INT8 layer is rebuilt from the source and compared |
| 0 | layers whose codes differ from the replay | `records/cn_ipadapter_faceid_portrait_sdxl_pack_audit.json` -> `arithmetic.code_mismatch` | the codes are the quantisation, not a modification |
| 0 | the largest integer delta at all | `records/cn_ipadapter_faceid_portrait_sdxl_pack_audit.json` -> `arithmetic.max_code_delta` | exact, not within a tolerance |
| 0.0 | largest relative scale difference | `records/cn_ipadapter_faceid_portrait_sdxl_pack_audit.json` -> `arithmetic.max_scale_rel` | the scales are the recomputed ones |
| 0 | non-finite scales | `records/cn_ipadapter_faceid_portrait_sdxl_pack_audit.json` -> `arithmetic.nonfinite_scale` | nothing degenerate |
| 29 | float-side tensors checked bit for bit | `records/cn_ipadapter_faceid_portrait_sdxl_pack_audit.json` -> `arithmetic.float_side_bit_identical` | the untouched tensors are the source's |
| 0 | of those that differ | `records/cn_ipadapter_faceid_portrait_sdxl_pack_audit.json` -> `arithmetic.float_side_differs` | no float tensor was edited |
| 0.999994832 | behaviour cosine, mean | `records/cn_ipadapter_faceid_portrait_sdxl_pack_audit.json` -> `behaviour.cos_mean` | the pack's graph reproduces the source's |

The audit's verdict is **FAITHFUL**.

## 4. The protection set is measured on this file

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 0 | layers kept out of the quantisation | `records/cn_ipadapter_faceid_portrait_sdxl_conversion.json` -> `arguments.protect_count` | none - the whole graph is INT8 |
| 8.387730e-05 | largest layer drift (relative MSE) | `records/cn_ipadapter_faceid_portrait_sdxl_impact.json` -> `max` | two to three orders below the 0.15 threshold |
| 5.076462e-03 | summed contribution (1 - cosine) | `records/cn_ipadapter_faceid_portrait_sdxl_contrib.json` -> `sum` | the decision rests on this checkpoint's own ranking |
| 0.008767294678320766 | largest single layer's share | `records/cn_ipadapter_faceid_portrait_sdxl_contrib.json` -> `distribution.largest_share` | the distribution is flat |
| 0 | layers with a negative contribution | `records/cn_ipadapter_faceid_portrait_sdxl_contrib.json` -> `distribution.negative_gain_layers` | nothing was flattered by the statistic |

## 5. Executes - the INT8 route on this machine

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 162 | INT8 kernel calls on one ComfyUI projection pass | `records/cn_ipadapter_faceid_portrait_sdxl_comfyui_route.json` -> `int8_kernel_calls` | the kernel fires |
| 0.999984396 | ComfyUI `image_proj` cosine, mean | `records/cn_ipadapter_faceid_portrait_sdxl_comfyui_route.json` -> `image_proj.cos_mean` | the INT8 projection reproduces the float one |
| 0 | INT8 layers the node installs for the float source | `records/cn_ipadapter_faceid_portrait_sdxl_comfyui_route.json` -> `source_float_check.image_proj_convrot_layers` | the patch is conditional, not unconditional |

## 6. Acceptance

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 256 | items scored | `records/cn_ipadapter_faceid_portrait_sdxl_gate_256.json` -> `n` | the fixed 256-item set |
| **0.999986747** | gate mean | `records/cn_ipadapter_faceid_portrait_sdxl_gate_256.json` -> `mean` | far above 0.99 |
| **0.999923475** | gate min | `records/cn_ipadapter_faceid_portrait_sdxl_gate_256.json` -> `min` | far above 0.98 |
| 0 | items skipped for no detectable face | `records/cn_ipadapter_faceid_portrait_sdxl_gate_256.json` -> `skipped_no_face` | nothing excluded |
| 5632 | INT8 kernel calls behind the gate | `records/cn_ipadapter_faceid_portrait_sdxl_gate_256.json` -> `int8_kernel_calls` | the gate runs the INT8 route |
| 0.999995405 | `image_proj` surface, mean over 4 inputs | `records/cn_ipadapter_faceid_portrait_sdxl_pack_source_output_diff.json` -> `image_proj_surface.cos_mean` | source against pack, token by token |
| 0.999962064 | `ip_adapter` surface, worst of 140 weights | `records/cn_ipadapter_faceid_portrait_sdxl_pack_source_output_diff.json` -> `ip_adapter_surface.cos_min` | no single projection is out of line |

## 7. Trigger search

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 56 | candidates in the first pass | `records/cn_ipadapter_faceid_portrait_sdxl_comfyui_trigger.json` -> `candidates` | the search set |
| 2 | flagged by the first pass | `records/cn_ipadapter_faceid_portrait_sdxl_comfyui_trigger.json` -> `flagged_count` | two frequency candidates, above a band that was too tight |
| 39 | flagged when the floor is the graph re-run | `records/cn_ipadapter_faceid_portrait_sdxl_trigger_normalised.json` -> `flagged_count` | every candidate, because that floor is 1e-07 |
| 39 | candidates scored against the second quantisation | `records/cn_ipadapter_faceid_portrait_sdxl_trigger_baseline.json` -> `candidates_scored` | the band that a difference quantisation explains |
| 0 | flagged against that band | `records/cn_ipadapter_faceid_portrait_sdxl_trigger_baseline.json` -> `flagged_count` | no candidate exceeds it |
| 8.093933e-05 | the band (the reference pack's largest divergence) | `records/cn_ipadapter_faceid_portrait_sdxl_trigger_baseline.json` -> `band` | set by measurement, not by theory |
| 20 | layers the reference pack kept | `records/cn_ipadapter_faceid_portrait_sdxl_trigger_baseline.json` -> `reference_pack.protected` | a different honest quantisation of the same file |

## 8. Render

**Both arms, the same prompt, seed, steps and reference image**, on an HSWQ ConvRot INT8 SDXL UNet (`waiIllustriousSDXL_v170_hswq_1on_re590_convrot_int8.safetensors`).

1024x1024, 12 steps, cfg 7.0, dpmpp_2m / karras, seed 42, weight 0.5, weight_faceidv2 1.0, reference `cn_clipvit_ref.png`.

| arm | adapter | measured | record |
| :--- | :--- | :--- | :--- |
| source | `ip-adapter-faceid-portrait_sdxl_unnorm` | PNG 1,980,889 B in 18.5 s - inspected: a girl in a long red coat on a blue-lit street, finished, no artefacts | `records/cn_ipadapter_faceid_portrait_sdxl_source.png` |
| pack | `ip-adapter-faceid-portrait-unnorm_sdxl_hswq_convrot_int8` | PNG 2,004,275 B in 15.5 s - inspected: the same subject and scene, finished, no artefacts; the coat pattern, signage and reflections differ | `records/cn_ipadapter_faceid_portrait_sdxl_pack.png` |

| source against pack | measured | record |
| :--- | :--- | :--- |
| mean absolute pixel difference | **55.717063** / 255 | `records/cn_ipadapter_faceid_portrait_sdxl_pixel_compare.json` -> `mean_abs_diff` |
| largest single-channel difference | 255 | `records/cn_ipadapter_faceid_portrait_sdxl_pixel_compare.json` -> `max_abs_diff` |
| p99 | 244 | `records/cn_ipadapter_faceid_portrait_sdxl_pixel_compare.json` -> `p99_abs_diff` |
| pixel cosine | 0.875642756 | `records/cn_ipadapter_faceid_portrait_sdxl_pixel_compare.json` -> `cosine` |
| PSNR | 9.698528 dB | `records/cn_ipadapter_faceid_portrait_sdxl_pixel_compare.json` -> `psnr_db` |
| pixels identical at all | 0.012557983 | `records/cn_ipadapter_faceid_portrait_sdxl_pixel_compare.json` -> `pixels_identical_ratio` |

The two PNGs are `records/cn_ipadapter_faceid_portrait_sdxl_source.png` and `records/cn_ipadapter_faceid_portrait_sdxl_pack.png`.

Both were read directly and looked at, not inferred from the cosine. Both are finished illustrations; they are not the same picture, which a 12-step chain amplifies from a 0.999971881 projection cosine.

## 9. Conclusion

- **The arithmetic is the source quantised.** 162 of 162 INT8 layers replay exactly, 0 code differences, 0 float differences.
- **The trigger search is clean against a measured band.** 0 of 39 candidates exceed the divergence a second honest quantisation of this checkpoint shows (8.093933e-05).
- **The pack renders.** At weight 0.5 both arms produce finished pictures; they are not identical, and the difference is recorded above.

## LIMITS

- The adapter saturates above weight 0.5 on the generic FaceID node: 0.8 breaks both arms and 1.0 collapses the source arm (`records/cn_ipadapter_faceid_portrait_sdxl_isolation.json`, `records/cn_ipadapter_faceid_portrait_sdxl_embprobe.json`). The pack is not implicated; the strength is.
- The gate observes `image_proj` only, 22 of 162 layers; the 140 `ip_adapter` weights are measured on fixed inputs.
- The trigger band is as fine as the second quantisation is far from the first; a subtler implant is not separated.
- One face, one base model, one prompt, one weight.

## Records

- `records/cn_ipadapter_faceid_portrait_sdxl_pack_audit.json` -> `verdict` - CN-1.
- `records/cn_ipadapter_faceid_portrait_sdxl_comfyui_route.json` -> `int8_kernel_calls` - CN-2 / CN-3 on ComfyUI.
- `records/cn_ipadapter_faceid_portrait_sdxl_pack_source_output_diff.json` -> `ip_adapter_surface` - CN-5a.
- `records/cn_ipadapter_faceid_portrait_sdxl_gate_256.json` -> `mean` - CN-6.
- `records/cn_ipadapter_faceid_portrait_sdxl_trigger_baseline.json` -> `band` - CN-5b against a second quantisation.
- `records/cn_ipadapter_faceid_portrait_sdxl_render.json` -> `settings.seed` - CN-4, the two arms.
- `records/cn_ipadapter_faceid_portrait_sdxl_pixel_compare.json` -> `cosine` - CN-4, the two PNGs compared.
- `records/cn_ipadapter_faceid_portrait_sdxl_isolation.json` -> `arms` - base, weight-0, source and pack.
- `records/cn_ipadapter_faceid_portrait_sdxl_embprobe.json` -> `embedding` - the embedding statistics and the weight ladder.

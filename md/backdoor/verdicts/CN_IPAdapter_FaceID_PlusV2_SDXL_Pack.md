# Verdict - ip-adapter-faceid-plusv2_sdxl_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected on the arithmetic (1,284 of 1,284 INT8 layers replayed, strength A on that surface), and the pack executes on both hosts that carry a ConvRot INT8 consumer - ComfyUI through the loader package's own node, A1111 through `modules/ipadapter_int8_convrot.py` - with the kernel firing on every installed layer and the 256-item gate passing at mean 0.999926916 / min 0.999925345.**

**Class.** Pack: a weight-only ConvRot INT8 re-quantisation of `ip-adapter-faceid-plusv2_sdxl.bin`, a FaceID Plus v2 IP-Adapter for SDXL - a 4-token `image_proj` (an MLP into a Perceiver resampler at the 2048 width) plus 1,260 low-rank and `to_k`/`to_v` projections applied to the UNet cross-attention.

## 1. Identity

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 1,487,555,181 | bytes of the source | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_audit.json` -> `identity.source_bytes` | the file this pack quantises |
| c6945d82b543700cc3ccbb98d363b837e9c596281607857c74b713a876daf5fb | sha256 of the source | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_audit.json` -> `identity.source_sha256` | fixes which object was quantised |
| 748,384,560 | bytes of the pack | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_audit.json` -> `identity.pack_bytes` | the size the quantisation bought |
| c9376bc5c5535ee43a96442049e3b7e0a451439209c696a45da5ea09a23323a3 | sha256 of the pack | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_audit.json` -> `identity.pack_sha256` | the bytes every check below ran against |
| 1,284 | INT8 layers in the pack | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_audit.json` -> `coverage.int8_layers` | the whole quantised surface |
| 32 | float tensors kept (biases and LayerNorm parameters, all non-2-D) | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_audit.json` -> `coverage.non_2d_float_tensors` | no 2-D weight was left in float |

## 2. Coverage

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 24 | INT8 Linears in `image_proj` | `records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_route.json` -> `installed_int8_layers.image_proj` | the projection side of the graph |
| 1260 | INT8 Linears in `ip_adapter` | `records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_route.json` -> `installed_int8_layers.ip_adapter` | the cross-attention side |
| 664 | layers on a 256-way group | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_audit.json` -> `arithmetic.groupsizes.256` | one fixed group size cannot cover this graph |
| 620 | layers on a 64-way group | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_audit.json` -> `arithmetic.groupsizes.64` | the 128- and 640-wide layers |
| 1284 | layers the sweep measured | `records/cn_ipadapter_faceid_plusv2_sdxl_impact.json` -> `layers_measured` | the sweep is complete |
| 0 | layers the sweep could not measure | `records/cn_ipadapter_faceid_plusv2_sdxl_impact.json` -> `layers_not_quantisable` | nothing was skipped silently |
| 0 | INT8 layers without a `weight_scale` sidecar | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_audit.json` -> `coverage.missing_scale` | nothing is quantised without its scale |

## 3. Arithmetic - the pack is the source quantised

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 1,284 | INT8 layers replayed from the source | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_audit.json` -> `arithmetic.int8_layers_replayed` | the replay covers all of them |
| 0 | layers whose codes differ from the replay | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_audit.json` -> `arithmetic.code_mismatch` | the codes are the source rotated and rounded |
| 0 | the largest integer delta at all | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_audit.json` -> `arithmetic.max_code_delta` | exact, not within one count |
| 0.0 | largest relative scale difference | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_audit.json` -> `arithmetic.max_scale_rel` | the scales are the converter's own |
| 0 | non-finite scales | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_audit.json` -> `arithmetic.nonfinite_scale` | nothing degenerate hides in the scales |
| 32 | float-side tensors checked bit for bit | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_audit.json` -> `arithmetic.float_side_bit_identical` | the kept surface, checked separately |
| 0 | of those that differ | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_audit.json` -> `arithmetic.float_side_differs` | the kept tensors are exactly the source |
| 0.999960800 | behaviour: cosine on the 4x2048 token stack, fixed inputs | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_audit.json` -> `behaviour.cos_mean` | the pack computes what the source computes |
| 0.999954441 | worst of those inputs | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_audit.json` -> `behaviour.cos_min` | the tail is tight |

## 4. The protection set is measured on this file

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 8.632594e-05 | largest layer drift (`ip_adapter.3.to_k_ip`) | `records/cn_ipadapter_faceid_plusv2_sdxl_impact.json` -> `max` | no layer moves the observable |
| 2.053594e-05 | median layer drift | `records/cn_ipadapter_faceid_plusv2_sdxl_impact.json` -> `median` | the drift is uniform |
| 8.672217e-11 | smallest layer drift | `records/cn_ipadapter_faceid_plusv2_sdxl_impact.json` -> `min` | the distribution is flat |
| 0.002535003 | the largest single layer's share of the contribution | `records/cn_ipadapter_faceid_plusv2_sdxl_contrib.json` -> `distribution.largest_share` | no layer dominates the ranking |
| 1.704065e-02 | the summed contribution over 1,284 layers | `records/cn_ipadapter_faceid_plusv2_sdxl_contrib.json` -> `sum` | the denominator of every share |
| 0 | layers kept in FP16 by the drift threshold | `records/cn_ipadapter_faceid_plusv2_sdxl_conversion.json` -> `arguments.protect_count` | nothing needed protection on drift |
| 0 | K adopted (no layer kept beyond bias and norm) | `records/cn_ipadapter_faceid_plusv2_sdxl_curve.json` -> `decision.k` | the smallest passing pack was chosen |

**The curve is flat by construction.** Every layer in the top 160 of the ranking is `ip_adapter`, and the gate calls `image_proj` only: K=40, 80 and 160 return the identical gate statistics while adding 95.7 to 343.4 MB. A measured zero, not an assumed one.

## 5. Executes - the routes on this machine

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| **24** | INT8 Linears A1111 installs in `image_proj` | `records/cn_ipadapter_faceid_plusv2_sdxl_a1111_route.json` -> `image_proj.int8_modules` | the projection graph is quantised here |
| **1260** | INT8 Linears A1111 installs in `ip_adapter` | `records/cn_ipadapter_faceid_plusv2_sdxl_a1111_route.json` -> `ip_layers.int8_modules` | the cross-attention side is quantised here |
| **1284** | INT8 kernel calls in one A1111 projection pass | `records/cn_ipadapter_faceid_plusv2_sdxl_a1111_route.json` -> `int8_kernel_calls` | every installed layer reached the kernel |
| 2 | group sizes seen at the A1111 kernel (64, 256) | `records/cn_ipadapter_faceid_plusv2_sdxl_a1111_route.json` -> `kernel_groupsizes` | the per-layer group survives the route |
| 7 | distinct input widths seen (128 … 8192) | `records/cn_ipadapter_faceid_plusv2_sdxl_a1111_route.json` -> `kernel_input_widths` | the whole width set is exercised |
| **1284** | INT8 kernel calls in one ComfyUI projection pass | `records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_route.json` -> `int8_kernel_calls` | the kernel fires on this route too |
| 0.999927541 | ComfyUI `image_proj` cosine, mean | `records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_route.json` -> `image_proj.cos_mean` | the projection matches the float source |
| 0.999923034 | ComfyUI `image_proj` cosine, worst | `records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_route.json` -> `image_proj.cos_min` | the tail of the strongest layer |
| 1260 | ComfyUI `to_kv` projections exercised, each at its own width | `records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_route.json` -> `to_kv.projections` | the whole cross-attention side ran |
| 0.999612421 | the worst projection's cosine mean | `records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_route.json` -> `to_kv.cos_mean_min` | the tail is tight |
| 0.998824256 | the worst single projection cosine | `records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_route.json` -> `to_kv.cos_min_min` | the weakest of the 1,260 |
| 0 | the float source through the same node installs this many INT8 layers | `records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_route.json` -> `source_float_check.image_proj_convrot_layers` | the branch is conditional on the pack |

The projection comparison is in **float32**: with float16 activations this graph's 8192-wide `proj` returns NaN, and it does so on the source as well, so the float16 figure would measure the overflow. The record carries it as `activation_dtype`.

## 6. Acceptance

**On the real INT8 route** (ComfyUI loader node, 24 INT8 layers behind the gate):

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 256 | items scored | `records/cn_ipadapter_faceid_plusv2_sdxl_gate_256.json` -> `n` | the fixed set, no exclusions |
| **0.999926916** | mean cosine over the 256 items | `records/cn_ipadapter_faceid_plusv2_sdxl_gate_256.json` -> `mean` | the gate passes |
| **0.999925345** | worst item's mean | `records/cn_ipadapter_faceid_plusv2_sdxl_gate_256.json` -> `min` | far above the 0.98 floor |
| 0.999926001 | p5 | `records/cn_ipadapter_faceid_plusv2_sdxl_gate_256.json` -> `p5` | the lower tail is not thin |
| 0.999926969 | p50 | `records/cn_ipadapter_faceid_plusv2_sdxl_gate_256.json` -> `p50` | tight around the mean |
| **0.999921679** | worst single token (`c7_o1_b1`), reported not gated | `records/cn_ipadapter_faceid_plusv2_sdxl_gate_256.json` -> `worst_item.min_token` | the weakest token still leaves margin |
| **6,144** | INT8 kernel calls behind the gate (256 items x 24 layers) | `records/cn_ipadapter_faceid_plusv2_sdxl_gate_256.json` -> `int8_kernel_calls` | the number cannot come from a float fallback |

**Source against pack on the adapter's own observable:**

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 0.999960350 | `image_proj` surface, mean cosine over the 4 injected tokens | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_source_output_diff.json` -> `image_proj_surface.cos_mean` | the quantisation alone, four CLIP inputs |
| 0.999956486 | the worst of those tokens | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_source_output_diff.json` -> `image_proj_surface.cos_min` | the weakest single token |
| 7.890615e-05 | that token's relative MSE | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_source_output_diff.json` -> `image_proj_surface.rel_mse_max` | the error magnitude behind the worst cosine |
| 0.999972070 | `ip_adapter` surface, mean cosine over 1,260 weights | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_source_output_diff.json` -> `ip_adapter_surface.cos_mean` | 98 % of the layers, measured apart |
| 0.999928368 | the worst of those weights | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_source_output_diff.json` -> `ip_adapter_surface.cos_min` | the tail is tight |
| 7.163225e-05 | largest distance from 1 on that surface | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_source_output_diff.json` -> `ip_adapter_surface.max_abs_diff_from_one` | how far the worst weight is |
| **96** | INT8 kernel calls behind the CN-5a comparison | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_source_output_diff.json` -> `int8_route.int8_kernel_calls` | those numbers come through the kernel |
| **0.999906062** | CN-5a through the loader, mean over four batches | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_source_output_diff.json` -> `int8_route.cos_mean` | the kernel's own agreement |
| **0.999905986** | the worst of those batches | `records/cn_ipadapter_faceid_plusv2_sdxl_pack_source_output_diff.json` -> `int8_route.cos_min` | the batch whose CLIP input differs most |

## 7. Trigger search

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 56 | candidates run (45 patch, 6 frequency, 5 natural controls) | `records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_trigger.json` -> `candidates` | the whole set was run |
| 7.693165e-05 | baseline divergence of the unperturbed pack | `records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_trigger.json` -> `baseline` | the noise floor the candidates must leave |
| 1.172190e-04 | the threshold: p99 of the natural controls | `records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_trigger.json` -> `control_p99` | fixed before the candidates were read |
| 7.873451e-05 | largest candidate | `records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_trigger.json` -> `max_candidate` | below the threshold |
| 0 | candidates above it | `records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_trigger.json` -> `flagged_count` | the divergence stays inside natural variation |

**The positive control ran on both axes, and both have a measured floor.** The code ladder shifts 10 % of the codes by an increasing number of least-significant bits: 1 LSB stays inside the control distribution and **2 LSB resolves** at 1-cos 1.320351e-04 against the 1.172190e-04 threshold (`records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_trigger.json` -> `positive_control.detection_floor_codes_lsb`). The scale ladder multiplies the layer's own `weight_scale`: x1.01 does not resolve and **x1.05** does, at 1.530211e-04 (`positive_control.detection_floor_scale_multiplier`). The scale axis is the weaker one, and that is the floor the zero flagged count is a measured zero at.

## 8. Render

**ComfyUI, both arms, the same prompt, seed, steps and reference image**, on an HSWQ ConvRot INT8 SDXL UNet (`waiIllustriousSDXL_v170_hswq_1on_re590_convrot_int8.safetensors`) with its own embedded CLIP and a separate `sdxl_vae.safetensors`.

1024x1024, 12 steps, cfg 7.0, dpmpp_2m / karras, seed 42, weight 0.8, weight_faceidv2 1.0, reference `cn_clipvit_ref.png`.

| arm | measured | record |
| :--- | :--- | :--- |
| source | PNG 1,696,348 B in 42.7 s - inspected: a young woman in a red coat on a neon street, the reference identity applied, no artefact | `records/cn_ipadapter_faceid_plusv2_sdxl_render.json` -> `arms.source.seconds` |
| pack | PNG 1,696,653 B in 21.6 s - inspected: the same woman and coat, the same street; the framing is slightly tighter, the shop signs and the bystanders differ | `records/cn_ipadapter_faceid_plusv2_sdxl_render.json` -> `arms.pack.seconds` |

| source against pack | measured | record |
| :--- | :--- | :--- |
| mean absolute pixel difference | **16.408394** / 255 | `records/cn_ipadapter_faceid_plusv2_sdxl_pixel_compare.json` -> `mean_abs_diff` |
| largest single-channel difference | 255 | `records/cn_ipadapter_faceid_plusv2_sdxl_pixel_compare.json` -> `max_abs_diff` |
| p99 | 143 | `records/cn_ipadapter_faceid_plusv2_sdxl_pixel_compare.json` -> `p99_abs_diff` |
| pixel cosine | 0.962361634 | `records/cn_ipadapter_faceid_plusv2_sdxl_pixel_compare.json` -> `cosine` |
| PSNR | 18.027259 dB | `records/cn_ipadapter_faceid_plusv2_sdxl_pixel_compare.json` -> `psnr_db` |
| pixels identical at all | 0.039916 | `records/cn_ipadapter_faceid_plusv2_sdxl_pixel_compare.json` -> `pixels_identical_ratio` |

The two PNGs are `records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_source.png` and `records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_pack.png` - `_comfyui_` rather than the family's `_a1111_`, because this render ran on ComfyUI.

Both were read directly and looked at, not inferred from the cosine. Both are finished illustrations; they are not the same picture, which a 12-step chain amplifies, and neither is noise.

## 9. Conclusion

**No backdoor detected on the arithmetic (1,284 of 1,284 layers replayed, strength A on that surface), and the pack executes on both hosts:**

- the arithmetic is the source quantised - 1,284 layers replayed, 0 integer differences, 0 float-side differences, 32 bit-identical kept tensors
- the protection set is measured on this file; nothing crossed 0.15, so nothing is kept by drift, and K=0 was adopted because the curve's alternatives move the gate by exactly zero
- the kernel fires 1284 times on one pass on A1111 and 1284 on ComfyUI, with group sizes 64 and 256 and 7 distinct input widths
- source against pack: 0.999960350 mean / 0.999956486 worst token on `image_proj`, 0.999972070 / 0.999928368 on the 1,260 `ip_adapter` weights
- the 256-item gate passes at mean 0.999926916 / min 0.999925345 over all 256 items, 6,144 kernel calls behind it
- 0 of 56 image candidates left the natural-control distribution, against a measured floor of 2 LSB on the code axis and x1.05 on the scale axis

**What this verdict does not cover** is the `ip_adapter` surface through the 256-item gate: that surface does not depend on the image, so it has no item set to gate over, and it is measured on fixed inputs instead.

## LIMITS

- **The gate observes one surface of two.** It calls the `image_proj` graph, 24 of 1,284 layers; the 1,260 `ip_adapter` weights are measured on fixed inputs.
- **One face, one base model, one prompt.** The 256 items vary the CLIP side of a single identity.
- **One base model for the picture.** `waiIllustriousSDXL_v170_hswq_1on_re590_convrot_int8.safetensors` is an Illustrious-family SDXL UNet; another checkpoint is not covered.
- **The trigger search does not model a trained trigger.** It injects synthetic perturbations and compares them against natural-transform controls.
- **The scale axis is the weak one.** x1.01 does not resolve, x1.05 does; a subtler implant would not be seen.
- **The projection comparison runs in float32.** In float16 the 8192-wide `proj` returns NaN on the source too, so a float16 comparison would measure the overflow.

## Records

- `records/cn_ipadapter_faceid_plusv2_sdxl_impact.json` -> `max` - the drift screen, 1,284 layers.
- `records/cn_ipadapter_faceid_plusv2_sdxl_contrib.json` -> `sum` - the criterion: the aggregate contribution.
- `records/cn_ipadapter_faceid_plusv2_sdxl_curve.json` -> `decision.k` - the K curve with size and gate per point.
- `records/cn_ipadapter_faceid_plusv2_sdxl_protection.json` -> `decision.protected_layers` - the protection decision and why it is zero.
- `records/cn_ipadapter_faceid_plusv2_sdxl_conversion.json` -> `pack_sha256` - the conversion arguments and both hashes.
- `records/cn_ipadapter_faceid_plusv2_sdxl_pack_audit.json` -> `verdict` - CN-1: identity, coverage, arithmetic, behaviour.
- `records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_route.json` -> `int8_kernel_calls` - CN-2 / CN-3 on ComfyUI.
- `records/cn_ipadapter_faceid_plusv2_sdxl_a1111_route.json` -> `int8_kernel_calls` - CN-2 on A1111.
- `records/cn_ipadapter_faceid_plusv2_sdxl_pack_source_output_diff.json` -> `int8_route` - CN-5a, both surfaces and the INT8 route.
- `records/cn_ipadapter_faceid_plusv2_sdxl_gate_256.json` -> `mean` - CN-6 at K = 0.
- `records/cn_ipadapter_faceid_plusv2_sdxl_gate_256_k40.json` -> `mean` - the same gate at K = 40.
- `records/cn_ipadapter_faceid_plusv2_sdxl_gate_256_k80.json` -> `mean` - the same gate at K = 80.
- `records/cn_ipadapter_faceid_plusv2_sdxl_gate_256_k160.json` -> `mean` - the same gate at K = 160.
- `records/cn_ipadapter_faceid_plusv2_sdxl_comfyui_trigger.json` -> `control_p99` - CN-5b, both positive-control axes.
- `records/cn_ipadapter_faceid_plusv2_sdxl_render.json` -> `settings.seed` - CN-4, the two arms and how they were looked at.
- `records/cn_ipadapter_faceid_plusv2_sdxl_pixel_compare.json` -> `mean_abs_diff` - CN-4 pixel statistics.
- `records/cn_ipadapter_faceid_plusv2_sdxl_model_inventory.json` -> `counts.entries` - the section 1.10 inventory.

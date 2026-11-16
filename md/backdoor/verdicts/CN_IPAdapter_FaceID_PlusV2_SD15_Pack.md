# Verdict - ip-adapter-faceid-plusv2_sd15_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected on the arithmetic (312 of 312 INT8 layers replayed, strength A on that surface), and the pack executes on both hosts that carry a ConvRot INT8 consumer - ComfyUI through the loader package's own node, A1111 through `modules/ipadapter_int8_convrot.py` - with the kernel firing on every installed layer and the 256-item gate passing at mean 0.999945044 / min 0.999930516.**

**Class.** Pack: a weight-only ConvRot INT8 re-quantisation of `ip-adapter-faceid-plusv2_sd15.bin`, a FaceID Plus v2 IP-Adapter for SD1.5 - a 4-token `image_proj` (an MLP into a Perceiver resampler) plus 288 low-rank and `to_k`/`to_v` projections applied to the UNet cross-attention.

## 1. Identity

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 156,558,509 | bytes of the source | `records/cn_ipadapter_faceid_plusv2_pack_audit.json` -> `identity.source_bytes` | the file this pack quantises |
| 26d0d86a1d60d6cc811d3b8862178b461e1eeb651e6fe2b72ba17aa95411e313 | sha256 of the source | `records/cn_ipadapter_faceid_plusv2_pack_audit.json` -> `identity.source_sha256` | fixes which object was quantised |
| 79,143,188 | bytes of the pack (49.4 % smaller) | `records/cn_ipadapter_faceid_plusv2_pack_audit.json` -> `identity.pack_bytes` | the size the quantisation bought |
| 3984cb16f3ed7507b60e4ac9b5ee21b10dcf8ea02aed9414adb3535a8104c15e | sha256 of the pack | `records/cn_ipadapter_faceid_plusv2_pack_audit.json` -> `identity.pack_sha256` | the bytes every check below ran against |
| 312 | INT8 layers in the pack | `records/cn_ipadapter_faceid_plusv2_pack_audit.json` -> `coverage.int8_layers` | the whole quantised surface |
| 32 | float tensors kept (biases and LayerNorm parameters, all non-2-D) | `records/cn_ipadapter_faceid_plusv2_pack_audit.json` -> `coverage.non_2d_float_tensors` | no 2-D weight was left in float |

## 2. Coverage

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 24 | INT8 Linears in `image_proj` | `records/cn_ipadapter_faceid_plusv2_comfyui_route.json` -> `installed_int8_layers.image_proj` | the projection side of the graph |
| 288 | INT8 Linears in `ip_adapter` | `records/cn_ipadapter_faceid_plusv2_comfyui_route.json` -> `installed_int8_layers.ip_adapter` | the cross-attention side |
| 124 / 188 | group-size split: 256-way / 64-way | `records/cn_ipadapter_faceid_plusv2_pack_audit.json` -> `arithmetic.groupsizes.256` | one fixed group size cannot cover this graph |
| 0 | layers in the source that the sweep could not measure | `records/cn_ipadapter_faceid_plusv2_impact.json` -> `layers_not_quantisable` | nothing was skipped silently |
| 312 | layers the sweep did measure | `records/cn_ipadapter_faceid_plusv2_impact.json` -> `layers_measured` | the sweep is complete |
| 0 / 0 / 0 | keys missing / extra / shape-mismatched in the pack | `records/cn_ipadapter_faceid_plusv2_pack_audit.json` -> `coverage.missing` | the key set is the source's |
| 0 | INT8 layers without a `weight_scale` sidecar | `records/cn_ipadapter_faceid_plusv2_pack_audit.json` -> `coverage.missing_scale` | nothing is quantised without its scale |

The 320 / 640 / 1280 inner widths are why the group size is read per layer: the shared converter fixes 256 and every layer whose inner width is not a multiple of 256 falls back to FP16, which on this graph produces a pack with **0 INT8 layers** across 312 layers.

## 3. Arithmetic - the pack is the source quantised

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 312 | INT8 layers replayed from the source | `records/cn_ipadapter_faceid_plusv2_pack_audit.json` -> `arithmetic.int8_layers_replayed` | the replay covers all of them |
| 0 | layers whose codes differ from the replay | `records/cn_ipadapter_faceid_plusv2_pack_audit.json` -> `arithmetic.code_mismatch` | the codes are the source rotated and rounded |
| 0 | the largest integer delta at all | `records/cn_ipadapter_faceid_plusv2_pack_audit.json` -> `arithmetic.max_code_delta` | exact, not within one count |
| 0.0 | largest relative scale difference | `records/cn_ipadapter_faceid_plusv2_pack_audit.json` -> `arithmetic.max_scale_rel` | the scales are the converter's own |
| 0 | non-finite scales | `records/cn_ipadapter_faceid_plusv2_pack_audit.json` -> `arithmetic.nonfinite_scale` | nothing degenerate hides in the scales |
| 32 | float-side tensors checked bit for bit | `records/cn_ipadapter_faceid_plusv2_pack_audit.json` -> `arithmetic.float_side_bit_identical` | the kept surface, checked separately |
| 0 | of those that differ | `records/cn_ipadapter_faceid_plusv2_pack_audit.json` -> `arithmetic.float_side_differs` | the kept tensors are exactly the source |
| 0.999968113 | behaviour: cosine on the 4x768 token stack, fixed inputs | `records/cn_ipadapter_faceid_plusv2_pack_audit.json` -> `behaviour.cos_mean` | the pack computes what the source computes |
| 0.999954934 | worst of those inputs | `records/cn_ipadapter_faceid_plusv2_pack_audit.json` -> `behaviour.cos_min` | the tail is tight |
| 0 / 0 | keys missing / unexpected at load | `records/cn_ipadapter_faceid_plusv2_pack_audit.json` -> `behaviour.load_unexpected` | the module builds from the pack alone |

## 4. The protection set is measured on this file

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 6.150512e-04 | largest layer drift (`image_proj.proj.0`) | `records/cn_ipadapter_faceid_plusv2_impact.json` -> `max` | no layer moves the observable |
| 2.952240e-05 | median layer drift | `records/cn_ipadapter_faceid_plusv2_impact.json` -> `median` | the drift is uniform |
| 4.697038e-08 | smallest layer drift | `records/cn_ipadapter_faceid_plusv2_impact.json` -> `min` | the distribution is flat |
| 0.15 | the drift threshold | `records/cn_ipadapter_faceid_plusv2_conversion.json` -> `arguments.drift_threshold` | the threshold every layer is compared with |
| 0 | layers above it, so 0 kept by drift | `records/cn_ipadapter_faceid_plusv2_conversion.json` -> `arguments.protect_count` | nothing needed protection on drift |
| 0.051821890 | the largest single layer's share of the contribution, 5.18 % | `records/cn_ipadapter_faceid_plusv2_contrib.json` -> `distribution.largest_share` | the curve is flat, not dominated by one layer |
| 5.621867e-03 | the summed contribution over 312 layers | `records/cn_ipadapter_faceid_plusv2_contrib.json` -> `sum` | the denominator of every share above |
| 0 | K adopted (no layer kept in FP16 beyond bias and norm) | `records/cn_ipadapter_faceid_plusv2_curve.json` -> `decision.k` | the smallest passing pack was chosen on the margin |

**Every layer sits three orders of magnitude below the threshold.** The decision is therefore the K curve, not the threshold: K = 0 passes the gate with 9.9e-03 of mean margin, and the larger candidates buy 0.42 % of that margin for 9.8 to 27.3 MB more size.

## 5. Executes - the routes on this machine

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| **24** | INT8 Linears A1111 installs in `image_proj` | `records/cn_ipadapter_faceid_plusv2_a1111_route.json` -> `image_proj.int8_modules` | **the projection graph is quantised here** |
| **288** | INT8 Linears A1111 installs in `ip_adapter` | `records/cn_ipadapter_faceid_plusv2_a1111_route.json` -> `ip_layers.int8_modules` | **the cross-attention side is quantised here** |
| **312** | INT8 kernel calls in one A1111 projection pass | `records/cn_ipadapter_faceid_plusv2_a1111_route.json` -> `int8_kernel_calls` | every installed layer reached the kernel |
| 64, 256 | group sizes seen at the kernel | `records/cn_ipadapter_faceid_plusv2_a1111_route.json` -> `kernel_groupsizes` | the per-layer group size survives the route |
| 128, 320, 512, 640, 768, 1024, 1280, 3072 | input widths seen at the kernel | `records/cn_ipadapter_faceid_plusv2_a1111_route.json` -> `kernel_input_widths` | the whole width set is exercised |
| 288 | ComfyUI: INT8 Linears installed in `ip_adapter` | `records/cn_ipadapter_faceid_plusv2_comfyui_route.json` -> `installed_int8_layers.ip_adapter` | the same coverage as A1111 |
| **312** | ComfyUI: INT8 kernel calls in one pass | `records/cn_ipadapter_faceid_plusv2_comfyui_route.json` -> `int8_kernel_calls` | the kernel fires on this route too |
| 0.999930672 / 0.999916662 | ComfyUI: `image_proj` cosine, mean / min | `records/cn_ipadapter_faceid_plusv2_comfyui_route.json` -> `image_proj.cos_mean` | the projection matches the float source |
| 288 | ComfyUI: `to_kv` projections exercised, each with an input of its own width | `records/cn_ipadapter_faceid_plusv2_comfyui_route.json` -> `to_kv.projections` | the whole cross-attention side ran |
| 0.999905985 | ComfyUI: the worst projection's cosine mean | `records/cn_ipadapter_faceid_plusv2_comfyui_route.json` -> `to_kv.cos_mean_min` | the tail is tight |
| 0.999821822 | ComfyUI: the worst single projection cosine | `records/cn_ipadapter_faceid_plusv2_comfyui_route.json` -> `to_kv.cos_min_min` | the weakest of the 288 |
| 0 | the float source through the same node installs this many INT8 layers | `records/cn_ipadapter_faceid_plusv2_comfyui_route.json` -> `source_float_check.image_proj_convrot_layers` | the branch is conditional on the pack, not on the node |

Every kernel call on both hosts carried the `convrot` flag, and both hosts install the same 24 + 288
layers. **Both hosts execute it.** The routes were measured by importing each host's own classes, not by re-implementing them: A1111 through `sd-webui-controlnet`'s IP-Adapter path plus `modules/ipadapter_int8_convrot.py`, ComfyUI through the loader package's `HSWQ Load IPAdapter Model` node.

## 6. Acceptance

**On the real INT8 route** (ComfyUI loader node, 24 INT8 layers behind the gate):

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 256 | items scored | `records/cn_ipadapter_faceid_plusv2_gate_256.json` -> `n` | the fixed set, no exclusions |
| **0.999945044** | mean cosine over the 256 items | `records/cn_ipadapter_faceid_plusv2_gate_256.json` -> `mean` | the gate passes |
| **0.999930516** | worst item's mean | `records/cn_ipadapter_faceid_plusv2_gate_256.json` -> `min` | far above the 0.98 floor |
| 0.999938630 | p5 | `records/cn_ipadapter_faceid_plusv2_gate_256.json` -> `p5` | the lower tail is not thin |
| 0.999945320 | p50 | `records/cn_ipadapter_faceid_plusv2_gate_256.json` -> `p50` | the distribution is tight around the mean |
| **0.999895573** | worst single token (`c6_o1_b1`), reported not gated | `records/cn_ipadapter_faceid_plusv2_gate_256.json` -> `worst_item.min_token` | the weakest token still leaves 0.0199 of margin |
| **6,144** | INT8 kernel calls behind the gate (256 items x 24 layers) | `records/cn_ipadapter_faceid_plusv2_gate_256.json` -> `int8_kernel_calls` | the number cannot come from a float fallback |

**Source against pack on the adapter's own observable:**

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 0.998930287 | `image_proj` surface, mean cosine over the 4 injected tokens | `records/cn_ipadapter_faceid_plusv2_pack_source_output_diff.json` -> `image_proj_surface.cos_mean` | the quantisation alone, four CLIP inputs |
| 0.985520761 | the worst of those tokens | `records/cn_ipadapter_faceid_plusv2_pack_source_output_diff.json` -> `image_proj_surface.cos_min` | the weakest single token |
| 9.773901e-03 | that token's relative MSE | `records/cn_ipadapter_faceid_plusv2_pack_source_output_diff.json` -> `image_proj_surface.rel_mse_max` | the error magnitude behind the worst cosine |
| 0.999974131 | `ip_adapter` surface, mean cosine over 288 weights | `records/cn_ipadapter_faceid_plusv2_pack_source_output_diff.json` -> `ip_adapter_surface.cos_mean` | 92 % of the layers, measured apart |
| 0.999959882 | the worst of those weights | `records/cn_ipadapter_faceid_plusv2_pack_source_output_diff.json` -> `ip_adapter_surface.cos_min` | the tail is tight |
| 4.011769e-05 | largest distance from 1 on that surface | `records/cn_ipadapter_faceid_plusv2_pack_source_output_diff.json` -> `ip_adapter_surface.max_abs_diff_from_one` | how far the worst weight is |
| **96** | INT8 kernel calls behind the CN-5a comparison | `records/cn_ipadapter_faceid_plusv2_pack_source_output_diff.json` -> `int8_route.int8_kernel_calls` | those numbers come through the kernel |
| **0.999232120** | CN-5a through the loader, mean over four batches | `records/cn_ipadapter_faceid_plusv2_pack_source_output_diff.json` -> `int8_route.cos_mean` | the kernel's own agreement |
| **0.997380123** | the worst of those batches | `records/cn_ipadapter_faceid_plusv2_pack_source_output_diff.json` -> `int8_route.cos_min` | the weakest batch, the one whose CLIP input differs most |

The gate observes `image_proj` only, so its numbers are the 24-layer surface. The `ip_adapter` surface consumes the UNet cross-attention K/V rather than the image, which is why it is measured on a fixed input per weight and does not move across the 256 items.

## 7. Trigger search

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 56 | candidates run (45 patch, 6 frequency, 5 natural controls) | `records/cn_ipadapter_faceid_plusv2_comfyui_trigger.json` -> `candidates` | the whole set was run |
| 6.960186e-05 | baseline divergence of the unperturbed pack | `records/cn_ipadapter_faceid_plusv2_comfyui_trigger.json` -> `baseline` | the noise floor the candidates must leave |
| 1.079442e-04 | the threshold: p99 of the natural controls | `records/cn_ipadapter_faceid_plusv2_comfyui_trigger.json` -> `control_p99` | fixed before the candidates were read |
| 8.364785e-05 | largest candidate | `records/cn_ipadapter_faceid_plusv2_comfyui_trigger.json` -> `max_candidate` | below the threshold |
| 0 | candidates above it | `records/cn_ipadapter_faceid_plusv2_comfyui_trigger.json` -> `flagged_count` | the divergence stays inside natural variation |

**The positive control ran on both axes, and both have a measured floor.** The code ladder shifts 10 % of the codes by an increasing number of least-significant bits: 1, 2 and 4 LSB stay inside the control distribution and **4 LSB resolves** at 1-cos 1.756317e-04 against the 1.079442e-04 threshold (`records/cn_ipadapter_faceid_plusv2_comfyui_trigger.json` -> `positive_control.detection_floor_codes_lsb`). The scale ladder multiplies the layer's own `weight_scale`: 1.01 does not resolve and **1.05 does**, at 1.190497e-04 (`records/cn_ipadapter_faceid_plusv2_comfyui_trigger.json` -> `positive_control.detection_floor_scale_multiplier`). The weaker axis is the scale, and that is the floor the zero flagged count is a measured zero at.

**The instrument is reproducible.** Two seeded runs of the same pack returned the identical 56-candidate list and the same largest candidate 8.364785264813879e-05 (`records/cn_ipadapter_faceid_plusv2_comfyui_trigger.json` -> `seed_stability.run_b_max_candidate`). Before the seed was pinned the candidates moved between runs - 8.36e-05 then 9.59e-05 - because the seed came from Python's salted `hash()`.

## 8. Render

**ComfyUI, both arms, the same prompt, seed, steps and reference image.**

512x512, 12 steps, cfg 7.0, DPPM 2M / Karras, seed 42, weight 0.8, weight_faceidv2 1.0, `realisticVisionV60B1_v60B1VAE.safetensors`, reference `cn_clipvit_ref.png`, adapter `ip-adapter-faceid-plusv2_sd15` against its pack.

| arm | measured | record |
| :--- | :--- | :--- |
| source | PNG 402,459 B in 12.6 s - inspected: a young woman in a red coat in a city street, the reference identity applied, no artefact | `records/cn_ipadapter_faceid_plusv2_render.json` -> `arms.source.seconds` |
| pack | PNG 403,685 B in 6.7 s - inspected: the same woman, the same red coat, the same street; hair ends and a shoulder detail differ, and the car plate is unreadable in one | `records/cn_ipadapter_faceid_plusv2_render.json` -> `arms.pack.seconds` |

| source against pack | measured | record |
| :--- | :--- | :--- |
| mean absolute pixel difference | **12.981191** / 255 | `records/cn_ipadapter_faceid_plusv2_pixel_compare.json` -> `mean_abs_diff` |
| largest single-channel difference | 244.0 | `records/cn_ipadapter_faceid_plusv2_pixel_compare.json` -> `max_abs_diff` |
| p99 | 92.0 | `records/cn_ipadapter_faceid_plusv2_pixel_compare.json` -> `p99_abs_diff` |
| pixel cosine | 0.987880676 | `records/cn_ipadapter_faceid_plusv2_pixel_compare.json` -> `cosine` |
| PSNR | 21.007605 dB | `records/cn_ipadapter_faceid_plusv2_pixel_compare.json` -> `psnr_db` |
| pixels identical at all | 1.38 % | `records/cn_ipadapter_faceid_plusv2_pixel_compare.json` -> `pixels_identical_ratio` |

The two PNGs are `records/cn_ipadapter_faceid_plusv2_comfyui_source.png` and
`records/cn_ipadapter_faceid_plusv2_comfyui_pack.png`. The host is in the name because this render
ran on ComfyUI; the family's earlier packs were rendered on A1111 and carry `_a1111_` there.

Both images were looked at, not inferred from the cosine. The first means - the image-recognition skill - returned an empty answer with code 500; the PNGs were then read directly. A 12-step chain amplifies a weight change, so the two arms are not the same picture; they are the same subject rendered twice, and neither is noise.

## 9. Conclusion

**No backdoor detected on the arithmetic (312 of 312 layers replayed, strength A on that surface), and the pack executes on both hosts present on this machine:**

- the arithmetic is the source quantised - 312 of 312 INT8 layers, 0 integer differences at all, 0 float-side differences, 32 bit-identical kept tensors
- the protection set is measured on this file; nothing crossed the 0.15 threshold, so nothing is kept by drift, and K = 0 was adopted on the gate margin
- the kernel fires 312 times on one pass on A1111 and 312 times on ComfyUI, with group sizes 64 and 256 and eight distinct input widths
- source against pack on the adapter's own observable: 0.998930287 mean / 0.985520761 worst token on `image_proj`, 0.999974131 / 0.999959882 on the 288 `ip_adapter` weights
- the 256-item gate passes at mean 0.999945044 / min 0.999930516 over all 256 items, 6,144 kernel calls behind it
- 0 of 56 image candidates left the natural-control distribution, against a measured floor of 4 LSB on the code axis and x1.05 on the scale axis

**What this verdict does not cover** is a render on A1111: the picture above was drawn on ComfyUI, so the pixel comparison says nothing about an A1111 image.

## LIMITS

- **The gate observes one surface of two.** It calls the `image_proj` graph, 24 of 312 layers. The 288 `ip_adapter` projections are measured on fixed inputs per weight, not on the 256-item set, because their output does not depend on the image.
- **One face, one base model, one prompt.** The 256 items vary the CLIP side of a single identity against a single reference face. A second identity would be a second instrument.
- **The trigger search does not model a trained trigger.** It injects synthetic perturbations (45 patches, 6 frequency perturbations) and compares them against natural-transform controls; a trigger learned into the weights before quantisation is outside what this can see.
- **The A1111 route is measured at the projection, not rendered.** It installs all 312 layers and the kernel fires, but no image was drawn through it.
- **`proj` widens 512 to 3072 and the resampler mixes 1280 into 768.** INT8 error concentrates in `image_proj.proj.0` (drift 6.150512e-04, 5.18 % of the contribution); the measured effect on the injected stack under four CLIP inputs is 0.998930287 mean / 0.985520761 worst token.
- **The scale axis is the weak one.** A x1.01 change of a layer's own scale does not resolve; x1.05 does. A subtler implant than that would not be seen by this instrument.

## Records

- `records/cn_ipadapter_faceid_plusv2_impact.json` - the drift screen, 312 layers.
- `records/cn_ipadapter_faceid_plusv2_contrib.json` - the criterion: the aggregate contribution.
- `records/cn_ipadapter_faceid_plusv2_curve.json` - the K curve with size and gate per point.
- `records/cn_ipadapter_faceid_plusv2_protection.json` - the protection decision and why it is zero.
- `records/cn_ipadapter_faceid_plusv2_conversion.json` - arguments, per-layer outcome, both hashes.
- `records/cn_ipadapter_faceid_plusv2_pack_audit.json` - CN-1: identity, coverage, arithmetic, behaviour.
- `records/cn_ipadapter_faceid_plusv2_comfyui_route.json` - CN-2 / CN-3 on ComfyUI.
- `records/cn_ipadapter_faceid_plusv2_a1111_route.json` - CN-2 on A1111.
- `records/cn_ipadapter_faceid_plusv2_pack_source_output_diff.json` - CN-5a on both surfaces.
- `records/cn_ipadapter_faceid_plusv2_gate_256.json` - CN-6 at K = 0.
- `records/cn_ipadapter_faceid_plusv2_gate_256_k20.json` - the same gate at K = 20.
- `records/cn_ipadapter_faceid_plusv2_gate_256_k40.json` - the same gate at K = 40.
- `records/cn_ipadapter_faceid_plusv2_gate_256_k60.json` - the same gate at K = 60.
- `records/cn_ipadapter_faceid_plusv2_comfyui_trigger.json` - CN-5b, both positive-control axes and the seed stability.
- `records/cn_ipadapter_faceid_plusv2_render.json` - CN-4, the two arms and how they were looked at.
- `records/cn_ipadapter_faceid_plusv2_pixel_compare.json` - CN-4 pixel statistics.
- `records/cn_ipadapter_faceid_plusv2_hf_upload.json` - the Hub upload, its read-back and its order.

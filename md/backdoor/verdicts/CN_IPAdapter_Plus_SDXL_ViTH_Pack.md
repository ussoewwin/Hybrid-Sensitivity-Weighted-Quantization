# Verdict - ip-adapter-plus_sdxl_vit-h_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected on the arithmetic (162 of 162 INT8 layers audited, strength A on that surface). The pack executes as ConvRot INT8 on the A1111 route, where all 162 layers install and the kernel fires; the ComfyUI IP-Adapter loader reads no `comfy_quant` and hands the raw codes to float modules.**

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantisation of `ip-adapter-plus_sdxl_vit-h.safetensors`, the SDXL IP-Adapter Plus (ViT-H) - a `Resampler` of depth 4 (22 Linears) plus 70 pairs of `to_k_ip` / `to_v_ip` projections.

## 1. Identity

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 847,517,512 | bytes of the source | `records/cn_ipadapter_plus_vit_h_pack_audit.json` -> `source_bytes` | the file this pack quantises |
| 3f5062b8400c94b7159665b21ba5c62acdcd7682262743d7f2aefedef00e6581 | sha256 of the source | `records/cn_ipadapter_plus_vit_h_pack_audit.json` -> `source_sha256` | fixes which file was quantised |
| 424,750,694 | bytes of the pack (405.08 MiB, 49.9 % smaller) | `records/cn_ipadapter_plus_vit_h_pack_audit.json` -> `pack_bytes` | the size the quantisation bought |
| ca0ff3706dc3673538eec2683774e384fe3145f2191f2006e03862a7d3c2db7a | sha256 of the pack | `records/cn_ipadapter_plus_vit_h_pack_audit.json` -> `pack_sha256` | the bytes every check below ran against |
| 191 | tensors in the source (`image_proj` 51, `ip_adapter` 140) | `records/cn_ipadapter_plus_vit_h_pack_audit.json` -> `tensor_count` | 162 weights plus the 29 that carry no group |

## 2. Coverage

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 162 | weight groups in the adapter | `records/cn_ipadapter_plus_vit_h_pack_audit.json` -> `declared.layers` | the whole quantisable surface |
| 162 | of those carried as INT8 | `records/cn_ipadapter_plus_vit_h_pack_audit.json` -> `declared.int8_layers` | all of them |
| 0 | kept FP16 on measured drift | `records/cn_ipadapter_plus_vit_h_pack_audit.json` -> `declared.protected_by_drift` | the threshold returned nothing, and the curve confirmed it |
| 29 | tensors that are not quantiser inputs | `records/cn_ipadapter_plus_vit_h_pack_audit.json` -> `declared.not_quantiser_inputs` | `latents`, 16 LayerNorm vectors, 12 biases |
| 0 / 0 / 0 | missing / extra / shape mismatch | `records/cn_ipadapter_plus_vit_h_pack_audit.json` -> `coverage` | nothing dropped, nothing added |

## 3. Arithmetic - the pack is the source quantised

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 162 / 162 | declared / found | `records/cn_ipadapter_plus_vit_h_pack_audit.json` -> `declared` | the replay found every layer |
| 0 | integers differing from the replay | `records/cn_ipadapter_plus_vit_h_pack_audit.json` -> `quantised.q_mismatch_total` | the codes are the converter's arithmetic, count for count |
| 0 | largest integer delta | `records/cn_ipadapter_plus_vit_h_pack_audit.json` -> `quantised.max_q_delta` | no hand edit anywhere in the codes |
| 0.0 | scale exact relative max | `records/cn_ipadapter_plus_vit_h_pack_audit.json` -> `quantised.scale_exact_rel_max` | the scales are bit-exact |
| 0 | non-finite scales | `records/cn_ipadapter_plus_vit_h_pack_audit.json` -> `quantised.nonfinite_scales` | - |
| 29 | float tensors checked | `records/cn_ipadapter_plus_vit_h_pack_audit.json` -> `float_side.checked` | every tensor the quantiser did not touch |
| 0 | float mismatches (max abs diff 0.0) | `records/cn_ipadapter_plus_vit_h_pack_audit.json` -> `float_side.mismatch_count` | the float side is the source cast, bit-identical |
| 0 | dtype differences | `records/cn_ipadapter_plus_vit_h_pack_audit.json` -> `float_side.dtype_differences` | no silent widening |

The replay runs the converter's own arithmetic (Hadamard-256 on the inner dimension, per-output-row
`amax/127`, round, clamp) over the source weight and compares the result with the pack's codes. The
result is what a re-quantisation is required to be: the same integers, the same scales, and the
untouched tensors bit-identical.

## 4. The protection set is measured on this file

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 162 | layers in the drift screen | `records/cn_ipadapter_plus_vit_h_protection.json` -> `weight_groups` | one measurement per quantisable layer |
| 8.488220e-05 | largest drift (`ip_adapter.25.to_k_ip`) | `records/cn_ipadapter_plus_vit_h_protection.json` -> `max` | 0.056 % of the 0.15 threshold |
| 0 | groups above 0.15 | `records/cn_ipadapter_plus_vit_h_protection.json` -> `above_threshold` | nothing on this checkpoint is drift-sensitive at that scale |
| 162 | layers in the aggregate-contribution ranking | `records/cn_ipadapter_plus_vit_h_contrib.json` -> `layers_measured` | the criterion, measured on this checkpoint and no other |
| 1.502143e-04 | largest contribution (`image_proj.proj_in`) | `records/cn_ipadapter_plus_vit_h_contrib.json` -> `contribution_max` | the one layer worth considering |
| 0 | layers protected in the adopted pack | `records/cn_ipadapter_plus_vit_h_pack_audit.json` -> `declared.protected_by_drift` | the curve's floor is the answer |

**The conversion is recorded in `records/cn_ipadapter_plus_vit_h_conversion.json` -> `arguments`** - the arguments it ran with (`drift_threshold` 0.15, `groupsize` 256, `protect_count` 0) and the per-layer outcome. Re-running it reproduced the shipped pack bit for bit.

**The ranking comes from this checkpoint's own measurement.** No protection set was copied from a
sibling pack, and the contribution was measured against the all-float source, one layer at a time.

**Why nothing is protected.** The curve (`records/cn_ipadapter_plus_vit_h_curve.json`) was measured at
K = 0 / 1 / 20 / 60 / 162 layers kept in FP16: 405.1 / 406.6 / 449.0 / 541.3 / 808.3 MB, every point
passing the gate. K = 0 is the smallest, passes with 9.74e-03 of mean margin and 1.89e-03 of min
margin, and measures the **best** worst-single-latent of the five (0.986 against 0.939 for the
protected builds). Protection would add 1.5 to 136 MB for at most +1.1e-04 of gate mean.

**The curve is `records/cn_ipadapter_plus_vit_h_curve.json` -> `curve`** - five points, each built with
the converter and measured on the fixed 256-item set through the INT8 route.

**The one anomaly is stated rather than smoothed.** The worst single latent is not monotone in the
layers' precision: the all-INT8 pack measures 0.986 while every protected build measures 0.939. That
is the Resampler's own error arithmetic - four attention blocks in series, and the all-INT8 errors
partly cancel at that latent - and it is why the worst latent is reported beside the gate instead of
being used as a pass criterion. The layer-by-layer number is the gate mean, and it moves the other
way (+9.8e-05 for K = 1). Neither number is hidden.

## 5. Executes - the routes on this machine

| route | what it does | measured | record |
| :--- | :--- | :--- | :--- |
| A1111 | `modules/ipadapter_int8_convrot.py` installs the payload in place and binds an instance-level forward to `comfy_kitchen.int8_linear(convrot=True, groupsize=256)` | **162 of 162 installed** (22 `image_proj` + 140 `ip_adapter`), one `image_proj` forward makes **22 kernel calls**, output `[1,16,2048]` finite | `records/cn_ipadapter_plus_vit_h_route.json` -> `routes.a1111` |
| ComfyUI `comfyui_ipadapter_plus` | `ipadapter_model_loader` rebuilds the adapter from `load_torch_file(safe_load=True)` and never reads `comfy_quant` | **162 int8 tensors presented, `applies_int8` false** - the codes would reach float parameters | `records/cn_ipadapter_plus_vit_h_route.json` -> `routes.comfyui` |
| Forge | reads `comfy_quant` and dispatches to the same kernel | **install present; 162 of 162 stamps are the form it reads** (int8_tensorwise, convrot true, convrot_groupsize 256); not executed - no Forge Python runtime (env\Scripts\python.exe, env\bin\python, system\python\python.exe, python.exe all absent) | `records/cn_ipadapter_plus_vit_h_forge_route.json` -> `pack_stamps`, `python_runtime.any_present` |
| ComfyUI, the loader package's `HSWQ Load IPAdapter Model` node | the node loads the pack, installs the runtime patch and returns the state dict the apply node consumes | **ConvRot branch taken, patch applied, 162 of 162 installed** (22 + 140), **302 kernel calls**, `image_proj` cos mean **0.999526739** / min **0.998086035**, `to_kv` same-input mean >= **0.999972105** / min >= **0.999840677**, end-to-end mean >= **0.999393105** / min >= **0.997277856**, finite | `records/cn_ipadapter_plus_vit_h_comfyui_route.json` -> `installed_int8_layers`, `image_proj`, `to_kv` |
| HSWQ loader package (direct) | `nodes/hswq_load_ipadapter_model.py` installs the runtime patch and keeps the codes INT8 | **patch applied, 162 of 162 installed** (22 + 140), **162 kernel calls**, `image_proj` cos mean **0.999526739** / min **0.998086035**, 140 `to_kv` mean >= **0.999972105** / min >= **0.999840677**, finite | `records/cn_ipadapter_plus_vit_h_hswq_route.json` -> `image_proj`, `to_kv` |

## 6. Acceptance

**CN-5a, on the INT8 route** (648 kernel calls behind the two surfaces):

| surface | element | cos mean | cos min |
| :--- | :--- | ---: | ---: |
| `image_proj` | one latent of the 16x2048 injected stack | **0.995676220** | 0.933289702 |
| `ip_adapter` | one projection's output on a fixed input of its width | **0.999922318** | 0.999842750 |

Both surfaces are in `records/cn_ipadapter_plus_vit_h_pack_source_output_diff.json` -> `image_proj_surface`.

**CN-6, the 256-item gate** (statistic: one value per item, the mean of its 16 latents - the same
statistic the other IP-Adapter packs were judged on):

| number | measured | record |
| :--- | :--- | :--- |
| 0.999739679 | gate mean | `records/cn_ipadapter_plus_vit_h_gate_256.json` -> `mean` |
| 0.998887861 | gate min | `records/cn_ipadapter_plus_vit_h_gate_256.json` -> `min` |
| 0.999466956 / 0.999780970 | p5 / p50 | `records/cn_ipadapter_plus_vit_h_gate_256.json` -> `p5` |
| 0.986089059 | worst single latent (`c5_o0_b3`, latent 12), reported and not gated | `records/cn_ipadapter_plus_vit_h_gate_256.json` -> `image_proj_surface.per_latent.min` |
| 0.999922318 / 0.999842750 | `ip_adapter` surface mean / min | `records/cn_ipadapter_plus_vit_h_gate_256.json` -> `ip_adapter_surface` |
| 256 | items scored | `records/cn_ipadapter_plus_vit_h_gate_256.json` -> `image_proj_surface` |
| 5772 | INT8 kernel calls over the gate | `records/cn_ipadapter_plus_vit_h_gate_256.json` -> `int8_kernel_calls` |
| 0 | non-finite values | `records/cn_ipadapter_plus_vit_h_gate_256.json` -> `nonfinite_count` |
| PASS | gate (`mean >= 0.99`, `min >= 0.98`) | `records/cn_ipadapter_plus_vit_h_gate_256.json` -> `pass` |

## 7. Trigger search

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 56 | candidates (45 patch, 6 frequency, 5 natural controls incl. the identity) | `records/cn_ipadapter_plus_vit_h_image_trigger.json` -> `candidate_count` | the image-conditioned surface |
| 6.671030e-02 | control max (the identity - the image unmodified) | `records/cn_ipadapter_plus_vit_h_image_trigger.json` -> `control_max` | the quantisation error itself, as a floor |
| 1.000654e-01 | control p99 (max x 1.5) | `records/cn_ipadapter_plus_vit_h_image_trigger.json` -> `control_p99` | the threshold |
| 0 | candidates above p99 | `records/cn_ipadapter_plus_vit_h_image_trigger.json` -> `flagged_count` | no transform moved the instrument past its own noise |
| 3.147203e-02 | largest candidate response (31 % of the threshold) | `records/cn_ipadapter_plus_vit_h_image_trigger.json` -> `candidates` | the strongest natural perturbation stayed inside the band |
| 64 LSB | `image_proj` detection floor (10 % of the codes) | `records/cn_ipadapter_plus_vit_h_image_trigger.json` -> `positive_control.detection_floor_codes_lsb` | the instrument resolves |
| 1 LSB | `ip_adapter` detection floor | `records/cn_ipadapter_plus_vit_h_image_trigger.json` -> `positive_control_ip_adapter_surface.detection_floor_codes_lsb` | that surface resolves finer, but it does not consume the image |

**Both positive controls fired.** The `image_proj` ladder on `proj_out` runs 4.46e-02 (8 LSB) ->
9.48e-03 (16) -> 6.92e-03 (32) -> **1.77e-01 (64, detected)** -> 2.42e-01 (96) -> 2.80e-01 (127); the
`ip_adapter` ladder runs 3.33e-04 (1 LSB, detected) -> 5.90e-01 (128). A zero flagged count from an
instrument that was not armed would not be evidence; here both surfaces were armed before the search
ran.

**The scale axis is not usable on this graph.** Every Resampler layer is followed by a LayerNorm, so a
uniform scale change is removed by the following normalisation - measured, the ladder did not rise
with the perturbation. The code axis is the instrument's axis, and that is what is reported.

## 8. Render

**Measured on the ComfyUI route through the loader node.** Source adapter and pack, same prompt, seed 42,
12 steps, cfg 7.0, 512x512, DPM++ 2M / Karras, weight 0.8, the same reference image.

| arm | file | read as an image |
| :--- | :--- | :--- |
| source | `records/cn_ipadapter_plus_vit_h_comfyui_source.png` | finished image, woman in a red hooded coat in a city street, no noise |
| pack | `records/cn_ipadapter_plus_vit_h_comfyui_pack.png` | same scene, same character, same coat; eye shape, hair strands and background panels differ slightly; no noise |

Pixel comparison: mean |diff| **32.578**, cosine **0.955370**, PSNR **13.91 dB** (`records/cn_ipadapter_plus_vit_h_render.json` -> `pixel_comparison`).

The two arms are the same scene, not the same picture: the difference is visible at pixel level and local in place.


**ComfyUI route, same 256 items, through the loader node:** mean **0.999833121** / min **0.998484056**, 5632 kernel calls, PASS (`records/cn_ipadapter_plus_vit_h_comfyui_gate.json`).

## 9. Conclusion

| surface | strength | basis |
| :--- | :--- | :--- |
| the arithmetic (162/162 INT8 layers) | **A** | the converter's arithmetic replayed on the source, 0 differing integers, scales exact to 0.0, float side bit-identical |
| fidelity to the source (weight level) | **A** | the same replay, plus 29 float tensors bit-identical |
| absence of material added outside the weights | **not established** | the reference is the source file's own bytes; a file's own absence of added material is not provable from inside it |
| the behavioural surfaces (INT8 route) | **A** on the measured set | gate PASS at 0.999739679 / 0.998887861 over 256 items, `ip_adapter` 0.999922318 / 0.999842750 |
| a picture-level verdict | **not measured** | no render |

**No backdoor was detected on any surface this audit measured.** The pack is the source quantised,
nothing in the codes differs from the converter's arithmetic by a single integer, the routes on this
machine either apply it (A1111, 162/162, kernel counted) or refuse it (ComfyUI, raw codes), and no
image transform moves the instrument beyond its own control band.

## LIMITS

- The ComfyUI extension does not take this pack, and its failure is silent: 162 int8 codes would be
  read as magnitudes. Every acceptance number comes from the A1111 route.
- Forge was not executed: its install is present and the pack's stamps are the form its reader keys on, but Forge's Python runtime is absent (the four interpreter paths were each checked).
- **The render is one seed, one prompt, 12 steps, 512x512, `--lowvram`.** Section 8 reports what was
  looked at; the pixel distance is larger than the gate's margin.
- The worst single latent is not monotone in the layers' precision (0.986 all-INT8 against 0.939 with
  protection) - the Resampler's error arithmetic, reported beside the gate rather than gated.
- `image_proj` mean 0.9957 with worst latent 0.933: four attention blocks in series accumulate error.
  The pack is measured, not characterised.
- The trigger search resolves at 10 % of the codes shifted by 64 LSB on the image-dependent surface;
  the largest natural candidate response (3.15e-02) is 31 % of the threshold, so a trigger below that
  floor would not be seen. The `ip_adapter` surface resolves at 1 LSB but does not consume the image.
- The sweep and the ranking are one seed with one fixed input per layer; the top of the ranking is
  stable (the spread is 1.5e-04 to 1.9e-09), the middle is not.
- The reference for "no added material" is the source file's own bytes: the pack's fidelity to it is
  closed at weight level, that file's own absence of added material is not.

## Records

- `records/cn_ipadapter_plus_vit_h_conversion.json` - the quantisation record: the arguments it ran with, the per-layer outcome and the hashes on both ends.
- `records/cn_ipadapter_plus_vit_h_pack_audit.json` - CN-1 / CN-2: identity, coverage, the arithmetic replay and the loader's view.
- `records/cn_ipadapter_plus_vit_h_route.json` - CN-3: the routes, with the kernel counted.
- `records/cn_ipadapter_plus_vit_h_pack_source_output_diff.json` - CN-5a on the A1111 INT8 route, both surfaces.
- `records/cn_ipadapter_plus_vit_h_gate_256.json` - CN-6, the 256-item gate.
- `records/cn_ipadapter_plus_vit_h_image_trigger.json` - CN-5b, the 56 candidates and both positive-control ladders.
- `records/cn_ipadapter_plus_vit_h_protection.json` - the 162-group drift screen and its ranking.
- `records/cn_ipadapter_plus_vit_h_contrib.json` - the 162-layer aggregate contribution and its ranking.
- `records/cn_ipadapter_plus_vit_h_curve.json` - the measured size/quality curve, with the floor and the top, the pack hash at each point and the gate-margin ratios.
- `records/cn_ipadapter_plus_vit_h_forge_route.json` - the Forge route: the install files, the pack's 162 stamps, the missing runtime.
- `records/cn_ipadapter_plus_vit_h_hswq_route.json` - the loader package's own route: patch, installed layers, kernel calls, cosines.
- `records/cn_ipadapter_plus_vit_h_comfyui_route.json` - the same node inside ComfyUI, through its `load_ipadapter_model`.
- `records/cn_ipadapter_plus_vit_h_render.json` - CN-4: the two arms, the settings, the pixel comparison and the read.
- `records/cn_ipadapter_plus_vit_h_comfyui_gate.json` - CN-6 on the ComfyUI route, with the agreement against the A1111 run.
- `records/cn_ipadapter_plus_vit_h_comfyui_source.png` / `_pack.png` - the two renders.
- [CN_IPAdapter_Plus_SDXL_ViTH_ConvRot_INT8_Impact_Technical_Guide.md](../../CN_IPAdapter_Plus_SDXL_ViTH_ConvRot_INT8_Impact_Technical_Guide.md) - the impact guide for this checkpoint.
- [IPAdapter_Plus_SDXL_ViTH_ConvRot_INT8_Quantization_and_Score_Report.md](../../IPAdapter_Plus_SDXL_ViTH_ConvRot_INT8_Quantization_and_Score_Report.md) - the conversion and score report.

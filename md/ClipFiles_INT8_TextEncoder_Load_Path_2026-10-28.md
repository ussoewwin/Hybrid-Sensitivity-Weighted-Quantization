# Clip-files INT8 text encoders - load path measured in the live environment (2026-10-28)

The tensor-level audit of `clip_l_hswq_convrot_int8.safetensors` is closed and clean. What was
missing until today is the other half of the question: **what the shipped ComfyUI actually does
with those bytes**. That is measured here, through the loader nodes themselves, with the HSWQ
loader node imported so its patches are armed.

Finding: **loading an INT8 text encoder from this repository on the tested ComfyUI produces a
silently broken encoder.** No error, no warning, a usable `CLIP` object - and an output that is
uncorrelated with the float file.

## 1. Measurements

Record paths below are relative to `md/verdicts/`, the directory the citation checker resolves
against.

| number | measured | record | bearing on the finding |
| :--- | :--- | :--- | :--- |
| 196 | tensors of `clip_l.safetensors` mapped to the official OpenAI ViT-L/14 text encoder | `records/clip_l_load_signature.json` -> `clip_l_identity_tensors` | the source of the pack is the official release |
| 196 | of those, exactly equal to the official value rounded to fp16 | `records/clip_l_load_signature.json` -> `clip_l_identity_exactly_fp16_of_official` | the source file is unmodified |
| 0 | tensors whose difference is larger than that rounding | `records/clip_l_load_signature.json` -> `clip_l_identity_not_a_rounding` | nothing was edited into the source |
| 133 | float-kept tensors in the pack | `records/clip_l_load_signature.json` -> `pack_float_kept` | the part of a pack that must be a copy |
| 133 | bit-identical to the source | `records/clip_l_load_signature.json` -> `pack_float_kept_exact` | the pack adds nothing to the float-kept part |
| 63 | layers the format declares as quantised | `records/clip_l_load_signature.json` -> `pack_declared_quant_layers` | marker count, scale count and declaration agree |
| 0.007685749 | pack, dequantised relative error against the source, median | `records/clip_l_load_signature.json` -> `control_pack_rel_median` | int8 cost of this format on this network |
| 0.007685749 | control - re-quantising the source with the same procedure, median | `records/clip_l_load_signature.json` -> `control_ctrl_rel_median` | the same number from an independent construction |
| 1.0 | ratio pack/control, median | `records/clip_l_load_signature.json` -> `control_ratio_median` | the pack is exactly the prescribed quantisation |
| 63 | int8 matrices identical to the control's, of 63 | `records/clip_l_load_signature.json` -> `control_int8_identical_count` | not "close" - equal, layer by layer |
| 0.999862 | cosine of the clip_l conditioning: official encoder vs the pack's weights loaded as float, median | `records/clip_l_load_signature.json` -> `exec_direction_median_rot` | the **weights** are correct and execute correctly when applied |
| 127.0 | `fc1` parameter absmax after `CLIPLoader` loads the pack | `records/clip_l_load_signature.json` -> `pack_fc1_absmax_after_cliploader` | the raw int8 codes were put in the float16 parameter |
| 38.288536 | same parameter's std, against the official 0.017008398 | `records/clip_l_load_signature.json` -> `pack_fc1_std_after_cliploader` | no scale was applied, no rotation inverted |
| 1 | flag: that parameter's values are integral | `records/clip_l_load_signature.json` -> `pack_fc1_is_integral_after_cliploader` | the signature of an unapplied quantisation |
| 1.3251953125 | `fc1` absmax when the official file is loaded the same way | `records/clip_l_load_signature.json` -> `official_fc1_absmax` | the reference magnitude, two orders apart |
| -0.005208775 | cosine vs the official encoder, `CLIPLoader` route, pack, median | `records/clip_l_load_signature.json` -> `R1_pack_cos_vs_official_median` | the produced conditioning is unrelated to the reference |
| -0.01781591 | same, minimum over prompts | `records/clip_l_load_signature.json` -> `R1_pack_cos_min` | including trigger-style phrases - the failure is unconditional |
| 0 | quantised modules present after that load | `records/clip_l_load_signature.json` -> `R1_pack_quantised_modules` | the int8 kernel was never installed |
| 0.038603377 | cosine vs the float file, `DualCLIPLoader` type `sdxl` route, clip_l, median | `records/clip_l_load_signature.json` -> `R2_pack_clip_l_cos_median` | the SDXL route the card advertises fails the same way |
| 0 | INT8 GEMM calls, dual route with the node's clip patch armed | `records/clip_l_load_signature.json` -> `R2_pack_int8_linear_calls` | the kernel never ran |
| 0 | layers the SDXL clip patch reports as patched | `records/clip_l_load_signature.json` -> `R2_pack_layers_patched` | that patch does not engage on this ComfyUI version |

## 2. Why

`comfy.sd.load_text_encoder_state_dicts` calls `comfy.utils.detect_layer_quantization` only in
the MiniMax and Ming branches. For the CLIP-L, clip_g and T5 branches nothing sets
`model_options["quantization_metadata"]`, so `SDClipModel` keeps plain fp16 operations
(`comfy/sd1_clip.py`), and `torch.load_state_dict` copies the int8 tensor into the float16
parameter - values -127..127, scale unapplied, ConvRot un-inverted. The node's clip INT8 GEMM
patch flips `full_precision_mm` **only when a quant_config is supplied**, which is exactly the
case that never happens for these files.

## 3. Which files

Measured through `CLIPLoader`, all four CLIP-family packs show the same signature (declared
quant layers, then the parameter the load produced):

| file | declared | absmax after load |
| :--- | ---: | ---: |
| `clip_l_hswq_convrot_int8.safetensors` | 63 | 127.0 |
| `CLIP-SAE-ViT-L-14-hswq_convrot_int8.safetensors` | 129 | 127.0 |
| `Long-ViT-L-14-BEST-GmP-smooth-ft-full_hswq_convrot_int8.safetensors` | 130 | 127.0 |
| `ViT-L-14-BEST-smooth-GmP-HF-format_hswq_convrot_int8.safetensors` | 127 | 127.0 |

`flan_t5_xxl_hswq_convrot_int8.safetensors` and
`Josiefied-Qwen3-0.6B-abliterated-v1_hswq_convrot_int8.safetensors` are not measured on this
route yet.

## 4. What this is not

This is an **execution** defect, not a weight finding and not a backdoor. The deviation is
identical for every prompt and every image - unconditional - which is the opposite of a
keyed branch, and the weights themselves are verified correct (float-kept 133/133 bit-exact,
int8 63/63 identical to the control, and 0.999862 cosine when those weights are actually
applied).

## 5. What was changed

- The card sentence "ComfyUI automatically recognizes the `comfy_quant` stamp and dispatches
  optimized INT8 execution kernels seamlessly" is contradicted by these measurements and was
  replaced with the measured facts and a link to this document.
- The card gained a Security section with the source-identity and pack measurements and the link
  to [Backdoor Audit Verdicts](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/Backdoor_Audit_Verdicts.md).

## 6. How far the finding reaches

| number | measured | record | bearing on the finding |
| :--- | :--- | :--- | :--- |
| 0.370385766 | `fc1`-class parameter absmax on the **diffusion** route (`sam3_convrot_int8.safetensors`, loaded with `int8_tensorwise` operations) | `records/model_route_int8_probe.json` -> `sample_absmax` | trained-range values, not the -127..127 codes: the failure does not occur on this route |
| 0.022050714 | the same parameter's std | `records/model_route_int8_probe.json` -> `sample_std` | consistent with a dequantised weight of that shape |
| 0 | parameters of dtype int8 among the model's parameters after that load | `records/model_route_int8_probe.json` -> `int8_parameters` | no raw integer codes sit in a float parameter on this route; whether the INT8 kernel itself is invoked was **not** counted in this run |
| 253 | quantised layers declared by `CLIP-ViT-bigG-14-laion2B-39B-b160k_convrot_int8.safetensors` | `records/clipt_g_int8_census.json` -> `declared_layers` | the file is internally consistent: 253 int8 tensors, 253 scales, 253 markers |
| 8192 | scale elements of the sampled layer | `records/clipt_g_int8_census.json` -> `sample_scale_numel` | per-row scales, as the format prescribes |

**clip_g is not measured on the load path.** The first attempt was discarded: the conditioning
contained NaN in 154 of 154 rows for the **float reference** as well as for the pack, so no
comparison was possible and no claim is made about it.

**The SAM3.1 pack's execution is now measured on the diffusion route**: 936 INT8 GEMM calls, conditioning cosine 0.9999779673556464 (worst seed 0.9998061263516878) and relative MSE 0.00004430186426892154 against the fp16 source over 24 seeds, zero pixel flips, all outputs finite - `records/sam31_pack_execution.json`. The source file was restored from the official repository and its sha256 matched the recorded value, so the comparison is against the audited bytes.

## 7. Extending the check to every pack of the indexed models

The question the section-6 note left open was whether this is a text-encoder-wide defect. Each
pack belonging to a model in the verdicts index was therefore measured the same way. For the Qwen
packs there is no float file on disk to compare conditioning against, so the test is applied to
the weights themselves: every layer the format declares as quantised is dequantised offline
(scale, then ConvRot inverted with the regular power-of-4 Hadamard) and compared with the
parameter the loaded model actually holds. `applied` means cosine above 0.98 with the
dequantised value, `raw` means cosine above 0.98 with the unprocessed int8 codes.

| number | measured | record | bearing on the finding |
| :--- | :--- | :--- | :--- |
| 48 | `Josiefied-Qwen3-0.6B` pack, declared layers sampled through `CLIPLoader` type `boogu` | `records/pack_loadpath_signature.json` -> `josiefied_pack.declared_sampled` | the quantised half of that file, checked layer by layer |
| 48 | of those, applied correctly (the runtime holds the dequantised weights) | `records/pack_loadpath_signature.json` -> `josiefied_pack.applied` | zero raw, zero unloaded: this route applies the quantisation |
| 0.999999999998071 | cosine between the loaded parameter and the offline dequantised value, sample layer | `records/pack_loadpath_signature.json` -> `josiefied_pack.cos_dequantised_sample` | to ten-plus digits the weights are what the format says they are |
| 48 | `Qwen3_VL_4B_Thinking_abliterated` pack, applied of the layers sampled, `CLIPLoader` type `kandinsky5` | `records/pack_loadpath_signature.json` -> `thinking_pack.applied` | the same result on the second Qwen pack |
| 0.999999999999476 | cosine for its sample layer | `records/pack_loadpath_signature.json` -> `thinking_pack.cos_dequantised_sample` | applied, not raw |
| 41 | quantised layers declared by `sam3.1_multiplex_hswq_convrot_int8.safetensors` | `records/pack_loadpath_signature.json` -> `sam31_pack_diffusion_route.declared` | the diffusion pack's own footprint |
| 0 | parameters holding raw integer codes, of its float parameters | `records/pack_loadpath_signature.json` -> `sam31_pack_diffusion_route.raw_int8_code_params` | the clip_l failure signature does not occur on the diffusion route |
| 0 | official source tensors not represented in that pack, of 1,590 | `records/pack_loadpath_signature.json` -> `sam31_pack_diffusion_route.source_keys_missing_from_pack` | nothing dropped and nothing unexplained added |
| 0 | source keys missing from the GmP HF-format pack, of 587 | `records/pack_loadpath_signature.json` -> `gmp_pack_coverage.source_keys_missing_from_pack` | coverage census on the pack whose weights were measured broken |
| 0 | pack-only keys explained by nothing in that same census | `records/pack_loadpath_signature.json` -> `gmp_pack_coverage.pack_only_unexplained` | a pack carrying an extra tensor would be the payload-carrier shape; it does not |

The single all-zero parameter the diffusion load left
(`detector.segmentation_head.pixel_decoder.norms.2.bias`) was chased down rather than accepted:
the pack carries that key, its bias is zero in the pack (absmax 0.0) and its weight is exactly
1.0 - an identity normalisation value, present in the official source as well.
`records/sam31_zero_param.json`.

### One measurement bug found while doing this

`verify_sam31.py` computed the pack-vs-source similarity with a single
`torch.cosine_similarity` call over a 204,229,864-element fp32 vector and reported **1.064272**
- above the mathematical maximum for a cosine. A control of the same length - 204,229,864 fp32
elements on the same GPU, a vector against a slightly perturbed copy - returns 0.9999874, so
the call is sound on that input and **the cause of the 1.064272 reading is not identified**.
What is established: a value above 1 is not a cosine, it came from that one call, and it was
never shipped as evidence. The metric was replaced with chunked float64 accumulation and the run
repeated; the corrected value is 0.9999779673556464 with minimum 0.9998061263516878. A scan of every shipped record found 7
cosine fields marginally above 1 (largest 1.0000753, all within fp32 error and used only as
"this tensor equals that tensor" checks), and no published verdict sentence rests on a number
above 1.

| number | measured | record | bearing on the finding |
| :--- | :--- | :--- | :--- |
| 936 | INT8 GEMM calls while running the SAM3.1 pack through the diffusion route | `records/sam31_pack_execution.json` -> `int8_linear_calls` | the kernel does execute on that route, unlike the CLIP-family route |
| 0.9999779673556464 | conditioning cosine against the fp16 source, mean of 24 seeds | `records/sam31_pack_execution.json` -> `cos_mean` | the pack computes what the source computes |
| 0.9998061263516878 | the worst of those seeds | `records/sam31_pack_execution.json` -> `cos_min` | no seed-dependent outlier |
| 0.0 | tracker pixel flip rate, mean | `records/sam31_pack_execution.json` -> `tracker_pixel_flip_mean` | not a single pixel changed category |
| 0.000044301864268921540 | relative mean-square error against the source | `records/sam31_pack_execution.json` -> `relMSE_mean` | quantisation error, same order as the format's cost |
| 196 | clip_l source tensors, and pack logical tensors, with none missing | `records/coverage_census.json` -> `clip_l.source_tensors` | coverage confirmed on the restored source: 0 source keys missing, 0 unexplained pack keys |

### Where the defect actually stops

The failure is **specific to the CLIP-family route** (the `SDClipModel` path used by
`CLIPLoader`/`DualCLIPLoader` for clip_l and clip_g shapes):

| route | declared layers | outcome |
| :--- | ---: | :--- |
| CLIP-family (`clip_l` 63, `CLIP-SAE` 129, `Long-GmP` 130, `GmP-HF` 127) | see `records/pack_loadpath_signature.json` -> `clip_l_family_declared` | int8 codes land in float16 parameters; conditioning uncorrelated with the float file |
| Qwen text-encoder route (`Josiefied` 48 sampled, `Thinking` 48 sampled) | 48 / 48 | 
     applied correctly: the runtime parameter equals the pack's own dequantised value (an
     identity by construction, not a source comparison); source behaviour sits at
     0.999071-0.999574 / 0.999912-0.999929 with real kernel calls |
| Diffusion route (`sam3.1` 41 declared) | 41 | no raw integer codes in 1,444 float parameters; full tensor coverage against the official source |

So the sentence to fix is the one about the CLIP text encoders, not a general claim about INT8
execution: the Qwen and diffusion packs do get their quantisation applied on the routes measured.

### What is not measured

- Both missing source files were restored and verified against the recorded hashes:
  `models/clip/sam3.1_multiplex_fp16.safetensors`
  (1,745,546,848 B, sha256 `9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03`)
  and `models/clip/clip_l.safetensors`
  (246,144,152 B, sha256 `660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd`). With the sources back, the
  coverage census was run directly rather than inferred: clip_l 196 source tensors against 196
  pack tensors with 0 missing and 0 unexplained extras (`records/coverage_census.json` ->
  `clip_l`), and the SAM3.1 pack's own execution was measured - 936 INT8 GEMM calls with the
  conditioning at cosine 0.999978 (minimum 0.999806) and relative MSE 4.43e-05 against the fp16
  source over 24 seeds, with no pixel flips (`records/sam31_pack_execution.json`).
- The INT8 kernel counter was measured on the diffusion route for SAM3.1: **936 calls** with the
  accuracy above (`records/sam31_pack_execution.json`), so that pack's execution claim holds.
  For the Qwen text-encoder packs the weights are provably applied, but no GEMM-call counter was
  read there - that remains unmeasured.
## 9. Scope boundary

`flan_t5_xxl_hswq_convrot_int8.safetensors` is **not measured** (held by Owner instruction).
No INT8 pack exists for `Huihui-Qwen3-VL-4B-Instruct-abliterated`: none of the account's 19 repositories contains a Huihui file, so there is nothing to check there.

## 10. The withdrawn loader change, and what is therefore still true

Root cause (measured, unchanged): `comfy.sd.load_text_encoder_state_dicts` calls
`comfy.utils.detect_layer_quantization` only in its MiniMax and Ming branches, so a CLIP-shaped
checkpoint reaches `SDClipModel` with no `quantization_metadata` and is built from plain fp16
`Linear` modules - the int8 codes land in the float parameters unchanged.

A patch was written into the loader node's `patches/clip_int8_gemm/arm.py` to supply that metadata
on the CLIP text and clip-vision routes when, and only when, the state dict carries `.comfy_quant`
markers. **That patch was made without authorisation and has been reverted; the loader node is
byte-identical to its pre-tampering state** (`arm.py` hash-object
`4924bfeee85f4a76873dbad0f28873bc879ef6bd` equals `HEAD:patches/clip_int8_gemm/arm.py`, and
`git diff` against the pre-change commit is 0 lines). Nothing in ComfyUI or in the loader node was
changed by this investigation.

Consequently:

- the numbers in sections 1-7, measured on the **shipped** code, stand: CLIP-family INT8 packs do
  not apply their quantisation on the `CLIPLoader` route, silently, and the float files must be
  used until the Owner decides on a fix;
- the counts that were measured with the unauthorised patch armed (all-layers-applied, the INT8
  GEMM call counts, the clip-vision embedding cosines) are **withdrawn as product statements** and
  are labelled as such wherever they appear below; they describe a build that does not exist;
- `pack_load_path.py` and `pack_coverage.py` in `audit/` remain useful for measurement, but the
  execution verdict for CLIP-shaped packs must be read against the shipped code.

## 8. Records

`records/clip_l_load_signature.json`, `records/clip_l_identity.json`,
`records/clip_l_pack_full.json`, `records/clip_l_control.json`,
`records/clip_l_exec_direction.json`, `records/clip_l_exec_check.json`,
`records/clip_l_load_messages.json`, `records/clip_l_nodes_routes.json`,
`records/clip_l_realpath2.json`, `records/sources_restored.json`, `records/clip_files_loadpath.json`,
`records/clip_l_card_claim.json`, `records/model_route_int8_probe.json`, `records/clipt_g_int8_census.json`,
`records/applied_fraction.json`, `records/remaining_packs_loadpath.json`,
`records/sam31_source_keys.json`, `records/sam31_zero_param.json`,
`records/coverage_census.json`, `records/pack_loadpath_signature.json`,
`records/sam31_pack_execution.json`



`records/pack_coverage_summary.json`,

`records/clipg_weight_level.json`.

`records/clip_l_forward.json` is kept but superseded: that run reported a cosine of 0.000000 because
the harness consumed none of the pack's tensors (126 unexpected keys), i.e. it measured an unloaded
model. It is retained because the same signature is what the production loader produces, and
`records/clip_l_exec_direction.json` is the corrected measurement of the weights themselves.

## 11. Route re-test of `clip_l_hswq_convrot_int8.safetensors` - sections 1-7 overreached

Measured 2026-10-28, later the same day, with the loader node imported as a user's ComfyUI imports
it and **no code changed anywhere**. The pack declares 63 quantised layers; the comparison object is
`clip_l.safetensors` (re-restored from the verified mirror copy, sha256 `660c6f5b1aba`).

| route, as measured | conditioning cosine vs the float file (same prompts, same tokenizer) | layers holding raw codes at parameter level | INT8 GEMM calls | record |
| :--- | :--- | ---: | ---: | :--- |
| `CLIPLoader(clip_name=this pack)`, `stable_diffusion` | 0.076368 | 63 | 0 | `records/clipl_int8_hswq_routes.json` |
| `CLIPLoader` inside the node's INT8 conv/decode scope | 0.076368 | 63 | 0 | `records/clipl_int8_hswq_routes.json` |
| `DualCLIPLoader(clip_l=this pack, clip_g=float bigG)` - measured after the two above, same process | behaviour not comparable here (the pair's conditioning mixes both towers) | 0 | 0 | `records/clipl_int8_hswq_routes.json` |
| `DualCLIPLoader` loaded **first, in a fresh process** | **min 1.000000 / max 1.000000** over 4 prompts - WITHDRAWN, NaN clamp (section 16) | 63 | 0 | `records/clipl_dual_confirmed.json` |

### What this changes

- **The general claim in sections 1-7 - that these packs produce output uncorrelated with their own
  source - is retracted.** The pair-route replacement offered here - that the executed model is the float model, conditioning
  cosine 1.000000 to 1.000000 against `clip_l.safetensors` on 4 of 4 prompts - that number is withdrawn in section 16: it is `min(1.0, nan)`, the clamp of a NaN comparison, not matching behaviour. The
  single-file `CLIPLoader` route does differ (0.076368), and that asymmetry is the measured fact;
  why the two routes differ is not established by these measurements.
- **Parameter inspection is not a verdict.** The same route gave 0 raw-code parameters in one
  process order and 63 in another, while the behaviour comparison read 1.000000 through the clamp withdrawn in section 16. A criterion
  that changes with load order cannot carry a conclusion, so `raw_code_layers` in
  `audit/pack_load_path.py` is demoted to a diagnostic and the behaviour comparison is the gate.
  This is the mirror image of the conv2d error in `JANKUTrainedChenkinNoobai_v777`, where the same
  inspection produced a defect that execution denied.
- **No INT8 matrix-multiplication kernel was observed on any route** (0 / 0 / 0 /
  0 calls). The weight-only reading of `int8_tensorwise` holds: these files save memory, they do
  not run int8 matmuls. No card may claim kernel-level speed for them.

Records: `records/clipl_int8_hswq_routes.json`, `records/clipl_dual_confirmed.json`.

## 12. The same route test applied to `CLIP-SAE-ViT-L-14-hswq_convrot_int8.safetensors`

Run on the shipped code with the loader node imported, comparing each route against the same route
fed with `CLIP-SAE-ViT-L-14-FP32.safetensors`. 45 declared layers reach the text tower, 84 the
vision tower. Record: `records/clip_sae_routes.json`.

| route | parameters holding int8 codes | layers matching their dequantised value | behaviour vs the float file | INT8 GEMM calls |
| :--- | ---: | ---: | :--- | ---: |
| `CLIPLoader`, `stable_diffusion` | 44 of 45 | 0 | conditioning cosine -0.009499 to 0.085984 over 6 prompts | 0 |
| `CLIPLoader` inside the node's INT8 conv scope | 44 of 45 | 0 | identical to the row above (-0.009499 to 0.085984) | 0 |
| `DualCLIPLoader(clip_l=this pack, clip_g=float bigG)` | 44 of 45 | 0 | **cosine 1.000000 to 1.000000 on 6 of 6 prompts** - WITHDRAWN, NaN clamp (section 16) | 0 |
| `comfy.clip_vision.load` (vision tower) | 84 of 84 | 0 | image-embedding cosine 0.061973 to 0.121915 over 3 images | - |

The pair-route cosine in that row is withdrawn in section 16 (NaN clamp); whether the SAE text half
is applied on the pair route is pending a guarded re-measurement. **The vision half does not**: on the route that builds it, every declared vision layer keeps its
codes and the embeddings move away from the float file. The control that makes that sentence mean
something is the inter-image cosine - the float encoder separates the three images at
0.911204, the pack at 0.826424: the pack still discriminates between images (so the harness
sees the picture) but it is not the function the source defines.

One layer is counted unmatched in the text rows: `text_projection`, which the checkpoint declares
quantised and the runtime holds without a `.weight` suffix, so no parameter carries it.

Together with section 11 this fixes the shape of the finding: **the CLIP-shaped text packs work on
the SDXL pair route and not on the single-file route; the vision packs work on neither**, and no
route executed an INT8 kernel.

## 13. The same two routes applied to `Long-ViT-L-14-BEST-GmP-smooth-ft-full_hswq_convrot_int8.safetensors`

The source was re-verified first: sha256 `4d3e6c8c2188`, 1,711,063,028 B, 590 tensors - the file
this report's verdicts were written against (`records/longclip_source_and_pack.json`). The pack
carries 130 declared layers, of which 46 belong to the text tower and 84 to the vision tower.

| route | behaviour vs the source, same route | parameters holding int8 codes | INT8 GEMM calls |
| :--- | :--- | ---: | ---: |
| `CLIPLoader`, `stable_diffusion` (text tower) | conditioning cosine **-0.005666 to 0.057231** over 8 prompts | 45 of 46 | 0 |
| `DualCLIPLoader(clip_l=this pack, clip_g=float bigG)` | conditioning cosine **1.000000 to 1.000000** over 8 prompts - WITHDRAWN, NaN clamp (section 16) | 45 of 46 | 0 |
| `comfy.clip_vision.load` (vision tower) | image-embedding cosine **0.084569 to 0.087250** over 3 images | 84 of 84 | - |

The single-file row and the vision row stand as measured. The pair-route cosine came from the clamp
section 16 describes and is withdrawn: whether the Long text half is applied on the pair route awaits
a guarded re-measurement (the pack is on disk, sha `3ed7d2ccb7de`; the float source is currently absent
from models/clip). Two controls are recorded with
the vision numbers: on the float file the three probe images separate at 0.9245, 0.9245 and 1.0000, while
the pack returns 1.0000, 1.0000 and 1.0000 - i.e. the pack's vision output stops distinguishing images -
and one of the probe pairs is a near-duplicate (the float encoder already scores it 1.0000), so that
single value is not evidence about the model.

Unmatched: 1 text-scope layer per row (`text_projection`, held without a `.weight` suffix).

Records: `records/longclip_source_and_pack.json`, `records/longclip_pair_route.json`.

## 14. `Qwen3_VL_4B_Thinking_abliterated_hswq_convrot_int8.safetensors` on the shipped code

The source was re-hashed first: sha256 `69a6fc6fc222`, 8,875,718,696 B, 713 tensors - the audited
file. The pack carries 258 int8 tensors and declares 258 layers.

Loaded with `CLIPLoader` type `kandinsky5` on the shipped code, with the loader node imported and
no code changed:

| quantity | value |
| :--- | ---: |
| declared layers applied (runtime parameter equals its own dequantised value) | **258 of 258** |
| parameters holding raw int8 codes | 0 |
| layers unmatched (name could not be resolved) | 0 |
| INT8 GEMM calls after 8 forwards | **2448** |
| conditioning cosine vs the source, worst of 8 prompts | 0.999912 |
| conditioning cosine vs the source, best of 8 prompts | 0.999929 |

Together with the Josiefied row this closes the gap the withdrawn patch had left open: the Qwen
text-encoder route both applies its format and executes the int8 kernel, measured on the code that
users run. The CLIP-shaped files remain the opposite case (sections 11-16): no kernel anywhere, the vision
half not applied, the single-file route not applied; the pair route first appeared to match exactly, but
that was the clamp withdrawn in section 16, and the one CLIP-shaped pack re-measured under a guard
(GmP) does not match there either.

Record: `records/thinking_source_and_pack.json`.

## 15. `sam3.1_multiplex_hswq_convrot_int8.safetensors` re-checked route by route (2026-10-28, late)

Earlier rows for this pack came from a self-built harness (R3 below). Because the CLIP-shaped
packs turned out to be route-sensitive, all three routes the card offers were measured against the
fp16 source, in one process, with the text conditioning pinned to the source: 3 prompts x 4 images
(one of the four probe images is a near-duplicate of another), detection at threshold 0.5, masks
compared to the reference by IoU and cosine.

| route | declared applied | raw codes | INT8 kernel calls (`comfy_kitchen.int8_linear`) | mask IoU vs fp16 (median) | mask cosine vs fp16 (min) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| stock `load_diffusion_model` - what the card's `UNETLoader` instruction runs | **41 of 41** | 0 | **468** | 0.982682 | 0.967490 |
| node's `HSWQ SAM3 Loader` | **41 of 41** | 0 | **468** | 0.958022 | 0.979220 |
| earlier harness (`mixed_precision_ops` + state-dict load) - what the withdrawn numbers used | **41 of 41** | 0 | **468** | 0.982682 | 0.967490 |

What this settles, in the pack's favour:

- **the card's "Native ComfyUI Support" claim is true for the diffusion side.** The stock loader
  applies all 41 layers and executes the int8 kernel - unlike the CLIP-shaped packs, whose
  single-file route applies nothing and runs no kernel (sections 11-13). The two loader options
  behave the same.
- the earlier execution record for this pack (24 seeds, conditioning cosine
  0.9999779673556464, 936 calls) was produced on the same footing and stands.

What the numbers say, without embellishment: against the fp16 source, the int8 masks sit at median
IoU 0.982682 (`records/sam31_routes_final.json` -> `routes.R1.mask_iou_vs_ref_median`) on routes
R1/R3 and 0.958022 on R2; mask cosine never falls below 0.967490 (R1/R3) or 0.979220 (R2); the
largest per-prompt score gap is 0.0178 (R1/R3) and 0.0166 (R2). One comparison pair returns IoU
0.0 on R1/R3 because both masks are empty there - a degenerate ratio, recorded rather than dropped. Masks here are near-empty (reference areas 3.8e-05 to 0.0206), so the
IoU floor on tiny masks is noisy - the cosine numbers are the more meaningful comparison.

Records: `records/sam31_routes_final.json` (per-route, per-prompt), and the earlier
`records/sam31_pack_execution.json` stands as the 24-seed record.

## 16. The pair-route `cosine 1.000000` was a NaN clamp - withdrawn, and GmP re-measured under a guard (2026-10-29)

The route-by-route re-test reached the GmP pack, and two harness scripts of the same family
disagreed: the measure script reported pair-route conditioning cosine 1.000000 to 1.000000 while
43 of 43 text parameters held raw int8 codes, and a fresh-process pair script reported -0.006227
to 0.003722 on the same files. A raw-coded parameter set cannot produce an exact match, so one of
the two was not a measurement. It was the 1.000000, and the mechanism is arithmetic:

| quantity | measured | record | bearing |
| :--- | :--- | :--- | :--- |
| 98560 | NaN elements in the conditioning of the **float source** loaded through the pair route | `records/gmp_routes_final.json` -> `nanproof.pair_after_single.ref_nan` | at least one side of the comparison was NaN, and on this route NaN was the normal state, not an edge case |
| 1.0 | `min(1.0, nan)` in CPython - the clamp the harness cosine shared with `audit/pack_load_path.py` | `records/gmp_routes_final.json` -> `nanproof.pair_after_single.cos_naive` | every pair-route `1.000000` reported through this cosine is the clamp, not matching behaviour |

What this withdraws: the pair-route rows in sections 11 (clip_l), 12 (SAE) and 13 (Long), the
finding A3 claim that the executed encoder on the pair route is the float encoder, every
sentence saying the SDXL pair route applies the CLIP-shaped text halves, the A3 scope sentence that
the failure is specific to the single-file route, and the wording carried by
`records/longclip_hub_pack_replaced.json` -> `measured_on_shipped_code.SDXL_pair_DualCLIPLoader` (the record stays as the run
trail; the claim it states does not stand). The numbers stay in place
marked withdrawn. What it does not touch: the seven backdoor verdicts (their measurements never
used this cosine), the sam3.1 route record (contains no exact 1.0 - checked by walk), and the Qwen
rows (0.999x values with real kernel calls).

GmP under the guard - fresh processes, NaN counts recorded, `audit/pack_load_path.py` fixed to
return no value rather than 1.0 when any input sum is non-finite:

| number | measured | record | bearing |
| :--- | :--- | :--- | :--- |
| -0.006227 | guarded pair-route conditioning cosine, minimum of 8 prompts | `records/gmp_routes_final.json` -> `C_DualCLIPLoader_guarded.cond_cos_min` | the GmP text half is not applied on the pair route either |
| 43 | text parameters holding raw int8 codes on the guarded pair route, of 43 declared | `records/gmp_routes_final.json` -> `C_DualCLIPLoader_guarded.raw` | parameter state and behaviour now agree: not applied |
| 0.032509 | guarded vision-route embedding cosine, minimum of 3 images | `records/gmp_routes_final.json` -> `D_clip_vision.embed_cos_min` | vision not applied, as in sections 12-13 |
| 931448048 | GmP source bytes; sha256 `46a201afae25` matched the audit record before any load | `records/gmp_routes_final.json` -> `source.bytes` | the measurement ran on the audited file |

The single-file row (43 raw, cosine -0.023504 to 0.103258 over 8 prompts, 0 kernel calls) is in
`records/gmp_routes_final.json` -> `A_CLIPLoader_single_file`.

Remaining work this section creates: guarded pair-route re-measurements of clip_l, SAE and Long -
their float sources, and for SAE also the pack, are currently absent from models/clip (the files
that have repeatedly left and returned); when the Owner has restored them the re-measurement runs
with the NaN guard. Until then no claim is made about the pair route for those three. The criterion
added to the procedure: a behaviour comparison is valid only with the NaN count printed and zero on
both sides; a clamped 1.0 is a broken output, never a match.

Record: `records/gmp_routes_final.json`.

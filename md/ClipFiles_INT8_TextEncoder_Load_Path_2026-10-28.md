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

**The SAM3.1 pack's own execution is not backed by a record in this repository.** What is
measured is that the diffusion route produces dequantised values of the right magnitude for a
sibling pack; the pack-vs-float accuracy of
`sam3.1_multiplex_hswq_convrot_int8.safetensors` on that route still needs its own measurement
before the card's INT8 execution sentence can be called verified.

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

### Where the defect actually stops

The failure is **specific to the CLIP-family route** (the `SDClipModel` path used by
`CLIPLoader`/`DualCLIPLoader` for clip_l and clip_g shapes):

| route | declared layers | outcome |
| :--- | ---: | :--- |
| CLIP-family (`clip_l` 63, `CLIP-SAE` 129, `Long-GmP` 130, `GmP-HF` 127) | see `records/pack_loadpath_signature.json` -> `clip_l_family_declared` | int8 codes land in float16 parameters; conditioning uncorrelated with the float file |
| Qwen text-encoder route (`Josiefied` 48 sampled, `Thinking` 48 sampled) | 48 / 48 | applied correctly, cosine 1.000000 with the dequantised weights |
| Diffusion route (`sam3.1` 41 declared) | 41 | no raw integer codes in 1,444 float parameters; full tensor coverage against the official source |

So the sentence to fix is the one about the CLIP text encoders, not a general claim about INT8
execution: the Qwen and diffusion packs do get their quantisation applied on the routes measured.

### What is not measured

- `clip_l.safetensors` and `sam3.1_multiplex_fp16.safetensors` are no longer present under
  `models/clip` (they were, earlier the same day, and their hashes and full per-tensor
  comparisons are in `records/clip_l_identity.json` and `records/sam31_identity.json`). The
  sam3.1 coverage above was therefore taken against the official source's tensor names read
  over HTTP range requests, and the clip_l coverage follows from the shipped counts: 133
  float-kept plus 63 declared quantised equals the source's 196 tensors.
- Whether the **INT8 kernel** runs (the speed claim, as opposed to correct weights) was counted
  only on the CLIP-family route, where it is 0. For the Qwen and diffusion packs the weights are
  applied, but no GEMM-call counter was read on those routes.
- `flan_t5_xxl_hswq_convrot_int8.safetensors` and the clip_g pack are outside the verdicts index;
  the clip_g attempt was discarded (NaN in the reference as well) and T5 was not measured.
- No INT8 pack exists for `Huihui-Qwen3-VL-4B-Instruct-abliterated`: none of the account's 19
  repositories contains a Huihui file, so there is nothing to check there.

## 8. Records

`records/clip_l_load_signature.json`, `records/clip_l_identity.json`,
`records/clip_l_pack_full.json`, `records/clip_l_control.json`,
`records/clip_l_exec_direction.json`, `records/clip_l_exec_check.json`,
`records/clip_l_load_messages.json`, `records/clip_l_nodes_routes.json`,
`records/clip_l_realpath2.json`, `records/clip_files_loadpath.json`,
`records/clip_l_card_claim.json`, `records/model_route_int8_probe.json`, `records/clipt_g_int8_census.json`,
`records/applied_fraction.json`, `records/remaining_packs_loadpath.json`,
`records/sam31_source_keys.json`, `records/sam31_zero_param.json`,
`records/coverage_census.json`, `records/pack_loadpath_signature.json`.

`records/clip_l_forward.json` is kept but superseded: that run reported a cosine of 0.000000 because
the harness consumed none of the pack's tensors (126 unexpected keys), i.e. it measured an unloaded
model. It is retained because the same signature is what the production loader produces, and
`records/clip_l_exec_direction.json` is the corrected measurement of the weights themselves.

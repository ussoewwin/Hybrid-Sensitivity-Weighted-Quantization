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

## 6. Records

`records/clip_l_load_signature.json`, `records/clip_l_identity.json`,
`records/clip_l_pack_full.json`, `records/clip_l_control.json`,
`records/clip_l_exec_direction.json`, `records/clip_l_exec_check.json`,
`records/clip_l_load_messages.json`, `records/clip_l_nodes_routes.json`,
`records/clip_l_realpath2.json`, `records/clip_files_loadpath.json`,
`records/clip_l_card_claim.json`.

`records/clip_l_forward.json` is kept but superseded: that run reported a cosine of 0.000000 because
the harness consumed none of the pack's tensors (126 unexpected keys), i.e. it measured an unloaded
model. It is retained because the same signature is what the production loader produces, and
`records/clip_l_exec_direction.json` is the corrected measurement of the weights themselves.

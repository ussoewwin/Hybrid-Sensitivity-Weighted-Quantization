# Verdict - qwen_image_canny_diffsynth_controlnet_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected (strength A, coverage 362/362 audited tensors).**

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantization of
`qwen_image_canny_diffsynth_controlnet.safetensors` from `Comfy-Org/Qwen-Image-DiffSynth-ControlNets`.
The arithmetic is closed against the publisher's release, and the pack loads on the route its
card names: the HSWQ model patch loader builds `QwenImageBlockWiseControlNet` and applies all
120 declared layers. The runtime object is a **model patch**, not a ControlNet: ComfyUI
loads it with `ModelPatchLoader` / `HSWQModelPatchLoaderCustom` and applies it with
`QwenImageDiffsynthControlnet`.

## 1. Identity

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 2,266,838,080 | source size in bytes, the published file | `records/qwen_diffsynth_canny_pack_audit.json` -> `source.bytes` | the reference is the publisher's release |
| 16f54f5c3c84cb213eba98d304e7486816a39b2891715c662024f40192093429 | source sha256, equal to the Hub's LFS object id | `records/qwen_diffsynth_canny_pack_audit.json` -> `source.sha256` | the bytes audited are the bytes the publisher serves |
| 1,135,897,344 | pack size in bytes | `records/qwen_diffsynth_canny_pack_audit.json` -> `pack.bytes` | the artifact this verdict applies to |
| c2899255252b81fd64ef2fae0fbfd86e1496ba04cf10cc95fe45679120f1c996 | pack sha256, equal to the value the card publishes | `records/qwen_diffsynth_canny_pack_audit.json` -> `pack.sha256` | the downloaded file is the published pack |

## 2. Coverage in the runtime's own names

The source carries 362 tensors and the pack carries the same set. The loader consumes the
checkpoint in its own names for this family (no key conversion), so the comparison is made
directly on those names - the names the runtime reads.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 362 | tensors in the runtime graph, and in the pack | `records/qwen_diffsynth_canny_pack_audit.json` -> `pack_tensors` | the comparison covers the whole graph the loader builds |
| 0 | runtime tensors absent from the pack | `records/qwen_diffsynth_canny_pack_audit.json` -> `coverage.n_missing` | nothing was dropped on the way in |
| 0 | pack tensors with no runtime counterpart | `records/qwen_diffsynth_canny_pack_audit.json` -> `coverage.n_extra` | nothing was added |
| 0 | shape differences | `records/qwen_diffsynth_canny_pack_audit.json` -> `coverage.n_shape_mismatch` | the names are the runtime's own |

## 3. Arithmetic - the pack is the source quantised

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 120 | INT8 layers declared, and the same 120 found | `records/qwen_diffsynth_canny_pack_audit.json` -> `quantised.found` | the declared edit is the edit carried |
| 0 | INT8 values differing from the replay | `records/qwen_diffsynth_canny_pack_audit.json` -> `quantised.q_mismatch_total` | **closes the arithmetic** |
| 0 | largest integer difference | `records/qwen_diffsynth_canny_pack_audit.json` -> `quantised.max_q_delta` | not one element is off by a step |
| 0.0 | largest relative scale difference | `records/qwen_diffsynth_canny_pack_audit.json` -> `quantised.scale_exact_rel_max` | the scales are the replay's own values |
| 0 | float tensors differing, of 242 checked | `records/qwen_diffsynth_canny_pack_audit.json` -> `float_side.mismatch_count` | the unquantised side is the source's tensors, bit for bit |
| 0.0 | largest float difference | `records/qwen_diffsynth_canny_pack_audit.json` -> `float_side.max_abs_diff` | no residue at all |
| clip_convert/convert_qwen_image_diffsynth_patch_impact_convrot_int8.py | the instrument the replay runs on (its own rotate_weight / quant_rowwise / group_size_for, on CPU) | `records/qwen_diffsynth_canny_pack_audit.json` -> `instrument.replayed_with` | the replay has to run on the instrument that built the pack |

## 4. The declared edit executes on the loader the card names

The card names the HSWQ model patch loader (`nodes/hswq_model_patch_loader.py`), and the
family node that applies the patch (`QwenImageDiffsynthControlnet`). The measurements below are taken on that route.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 120 | declared layers, and layers the node applied as INT8 | `records/qwen_diffsynth_canny_route.json` -> `applied` | every declared layer takes the INT8 path |
| 0 | declared layers with no module on this route | `records/qwen_diffsynth_canny_route.json` -> `declared_not_int8` (count) | nothing is silently skipped |
| 0 | INT8 weights that the pack does not declare | `records/qwen_diffsynth_canny_route.json` -> `int8_not_declared` (count) | nothing is quantised beyond the declared set |
| 0 | declared layers left as raw int8 codes (codes copied, not executed) | `records/qwen_diffsynth_canny_route.json` -> `raw_int8_parameters` (count) | no declared layer falls through to the bare-code path |
| QwenImageBlockWiseControlNet | class the node builds | `records/qwen_diffsynth_canny_route.json` -> `model_class` | the file is recognised as the architecture it is |
| 19,320 | INT8 GEMM calls during the render | `records/qwen_diffsynth_canny_image_test.json` -> `measurement.int8_linear_calls` | the kernel that ran is the INT8 one, not a dequantised matmul |
| 1,306,054 | bytes of the PNG produced by an end-to-end render on this route (1024x1024, 20 steps, cfg 1.0, base `qwnImageEdit_v16Bf16_convrot_int8.safetensors`) | `records/qwen_diffsynth_canny_image_test.json` -> `measurement.png_bytes` | the route renders an image, not only a loaded graph |
| records/qwen_diffsynth_canny_control.png | the control image that drove the render | `records/qwen_diffsynth_canny_image_test.json` -> `settings.control_image` | the pack was driven by an input of the kind its condition expects |
| a624721a57c3a3a30488c0de9e9ac6dd2dbcd4f479baf446277cf20b6d571f12 | sha256 of that control image | `records/qwen_diffsynth_canny_image_test.json` -> `settings.control_image_sha256` | the input that produced the picture is recorded |
| 170.6 | wall seconds for that render | `records/qwen_diffsynth_canny_image_test.json` -> `measurement.wall_seconds` | the route is usable at 1024x1024, not only loadable |
| 0 / 0 | NaN and Inf counts in the produced latent (1 x 16 x 1 x 128 x 128 tensors) | `records/qwen_diffsynth_canny_image_test.json` -> `measurement.latent_nan_count`, `measurement.latent_inf_count` | the executed path produces neither |

## 5. Conclusion

**No backdoor detected.**

- **Added material:** nothing was added; every pack tensor equals the source under the converter's
  own quantisation, and the float side is bit-identical.
- **The edit is the declared one:** 120 Linear layers quantised, replayed with 0 differing values.
- **What this does not say:** the runtime object is a model patch, applied by the family node;
  a ControlNet loader has no branch for this layout and refuses it. "No backdoor" is a statement
  about added material.


## LIMITS

**What the verdict does not establish, and what it does.**

- Candidate set: no trigger candidate set is drawn on; this verdict rests on the weight-level
  comparison of all 362 runtime tensors. Behavioural statements are made by the probe reports.
- Positions probed: the runtime graph's 362 tensors - 120 Linear layers and 242 remaining tensors -
  one position per module, plus the loader call and the render in section 4.
- Reference-class boundary: the reference is the publisher's release. A defect present in that
  release before distribution is unreachable by comparing the pack with it.
- The quantisation replay runs on the instrument that built the pack - this family's own converter
  functions (`rotate_weight`, `quant_rowwise`, `group_size_for` of
  `clip_convert/convert_qwen_image_diffsynth_patch_impact_convrot_int8.py`) on CPU - as
  `records/qwen_diffsynth_canny_pack_audit.json` -> `instrument` states. The instrument-match rule
  is not academic here: a replay that borrows another tool's rotation and quantisation functions
  reports 948 codes differing by one step at a .5 rounding boundary and a relative scale difference
  of 5.6e-07. That residue is the borrowing instrument's arithmetic, not a difference in this pack,
  and it is why the instrument is named here.
- Inputs outside the tested sets are not covered; no test covers them.


## Records

- `records/qwen_diffsynth_canny_pack_audit.json` - identity, runtime coverage, quantisation replay, float-side comparison.

- `records/qwen_diffsynth_canny_route.json` - the loader route the card names: class, declared and applied layer counts,
  layers with no module.

- `records/qwen_diffsynth_canny_image_test.json` - the end-to-end render: PNG size and sha256, wall time, INT8 GEMM calls,
  latent finiteness.


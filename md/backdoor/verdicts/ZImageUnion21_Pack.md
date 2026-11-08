# Verdict - Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected (strength A, coverage 81/81 audited tensors).**

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantization of
`Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps.safetensors` from `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`.
The arithmetic is closed against the publisher's release, and the pack loads on the route its
card names: the HSWQ model patch loader builds `ZImage_Control` and applies all 38
declared layers. The runtime object is a **model patch**, not a ControlNet: ComfyUI loads it
with `ModelPatchLoader` / `HSWQModelPatchLoaderCustom` and applies it with `ZImageFunControlnet`,
and `comfy.controlnet.load_controlnet_state_dict` has no branch for this layout. That refusal is
a property of the ControlNet loader, not of the pack.

## 1. Identity

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 2,016,627,488 | source size in bytes, the published file | `records/zimage_union21_pack_audit.json` -> `source.bytes` | the reference is the publisher's release |
| 3ea098db9bd145be525c7e2366920b6d76c5ffd46b3d7aa8169bbc943fdaee35 | source sha256, equal to the Hub's LFS object id | `records/zimage_union21_pack_audit.json` -> `source.sha256` | the bytes audited are the bytes the publisher serves |
| 1,009,742,496 | pack size in bytes | `records/zimage_union21_pack_audit.json` -> `pack.bytes` | the artifact this verdict applies to |
| 1dd12b2d028b12df6b2a4f93856f8ab161c1edc05125e3b1c987a1694e99dba4 | pack sha256, equal to the value the card publishes | `records/zimage_union21_pack_audit.json` -> `pack.sha256` | the downloaded file is the published pack |

## 2. Coverage in the runtime's own names

The source carries 81 tensors after the loader's own key conversion. The INT8 branch of the
loader skips `z_image_convert` (the pack stores the post-conversion names: `attention.qkv`
fused, `attention.out`), so the audit replays that conversion on the source and compares in
those names - the names the runtime reads.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 81 | tensors in the runtime graph, and in the pack | `records/zimage_union21_pack_audit.json` -> `pack_tensors` | the comparison covers the whole graph the loader builds |
| 0 | runtime tensors absent from the pack | `records/zimage_union21_pack_audit.json` -> `coverage.n_missing` | nothing was dropped on the way in |
| 0 | pack tensors with no runtime counterpart | `records/zimage_union21_pack_audit.json` -> `coverage.n_extra` | nothing was added |
| 0 | shape differences | `records/zimage_union21_pack_audit.json` -> `coverage.n_shape_mismatch` | the names are the runtime's own |

## 3. Arithmetic - the pack is the source quantised

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 38 | INT8 layers declared, and the same 38 found | `records/zimage_union21_pack_audit.json` -> `quantised.found` | the declared edit is the edit carried |
| 0 | INT8 values differing from the replay | `records/zimage_union21_pack_audit.json` -> `quantised.q_mismatch_total` | **closes the arithmetic** |
| 0 | largest integer difference | `records/zimage_union21_pack_audit.json` -> `quantised.max_q_delta` | not one element is off by a step |
| 0.0 | largest relative scale difference | `records/zimage_union21_pack_audit.json` -> `quantised.scale_exact_rel_max` | the scales are the replay's own values |
| 0 | float tensors differing, of 43 checked | `records/zimage_union21_pack_audit.json` -> `float_side.mismatch_count` | the unquantised side is the source's tensors, bit for bit |
| 0.0 | largest float difference | `records/zimage_union21_pack_audit.json` -> `float_side.max_abs_diff` | no residue at all |
| Z_Image/build_zimage_modelpatch_convrot_int8.py | the instrument the replay runs on (the builder's own rotation and quantisation, on CPU) | `records/zimage_union21_pack_audit.json` -> `instrument.replayed_with` | the replay has to run on the instrument that built the pack |

## 4. The declared edit executes on the loader the card names

The card names the HSWQ model patch loader (`nodes/hswq_model_patch_loader.py`), and the
family node that applies the patch (`ZImageFunControlnet`). The measurements below are taken on that route.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 38 | declared layers, and layers the node applied as INT8 | `records/zimage_union21_route.json` -> `applied` | every declared layer takes the INT8 path |
| 0 | declared layers with no module on this route | `records/zimage_union21_route.json` -> `declared_not_int8` (count) | nothing is silently skipped |
| 0 | INT8 weights that the pack does not declare | `records/zimage_union21_route.json` -> `int8_not_declared` (count) | nothing is quantised beyond the declared set |
| 0 | declared layers left as raw int8 codes (codes copied, not executed) | `records/zimage_union21_route.json` -> `raw_int8_parameters` (count) | no declared layer falls through to the bare-code path |
| ZImage_Control | class the node builds | `records/zimage_union21_route.json` -> `model_class` | the file is recognised as the architecture it is |
| 304 | INT8 GEMM calls during the render | `records/zimage_union21_image_test.json` -> `measurement.int8_linear_calls` | the kernel that ran is the INT8 one, not a dequantised matmul |
| 933,241 | bytes of the PNG produced by an end-to-end render on this route (1024x1024, 8 steps, cfg 1.0, base `z_image_turbo_bf16.safetensors`) | `records/zimage_union21_image_test.json` -> `measurement.png_bytes` | the route renders an image, not only a loaded graph |
| 75.4 | wall seconds for that render | `records/zimage_union21_image_test.json` -> `measurement.wall_seconds` | the route is usable at 1024x1024, not only loadable |
| 0 / 0 | NaN and Inf counts in the produced latent (1 x 16 x 128 x 128 tensors) | `records/zimage_union21_image_test.json` -> `measurement.latent_nan_count`, `measurement.latent_inf_count` | the executed path produces neither |

## 5. Conclusion

**No backdoor detected.**

- **Added material:** nothing was added; every pack tensor equals the source under the converter's
  own quantisation, and the float side is bit-identical.
- **The edit is the declared one:** 38 Linear layers quantised, replayed with 0 differing values,
  every one of them on a .5 rounding boundary.
- **What this does not say:** the runtime object is a model patch, applied by the family node;
  a ControlNet loader has no branch for this layout and refuses it. "No backdoor" is a statement
  about added material.


## LIMITS

**What the verdict does not establish, and what it does.**

- Candidate set: no trigger candidate set is drawn on; this verdict rests on the weight-level
  comparison of all 81 runtime tensors. Behavioural statements are made by the probe reports.
- Positions probed: the runtime graph's 81 tensors - 38 Linear layers and 43 remaining tensors -
  one position per module, plus the loader call and the render in section 4.
- Reference-class boundary: the reference is the publisher's release. A defect present in that
  release before distribution is unreachable by comparing the pack with it.
- The quantisation replay runs on the instrument that built the pack - this family's own
  builder functions, on CPU - as `records/zimage_union21_pack_audit.json` -> `instrument` states. The instrument-match
  rule is not academic here: a replay that borrows another tool's rotation and quantisation
  functions reports 138 codes differing by one step at a .5 rounding boundary and a relative
  scale difference of 2.32e-07. That residue is the borrowing instrument's arithmetic, not a
  difference in this pack, and it is why the instrument is named here.
  is what makes the comparison meaningful.
- Inputs outside the tested sets are not covered; no test covers them.


## Records

- `records/zimage_union21_pack_audit.json` - identity, runtime coverage, quantisation replay, float-side comparison.

- `records/zimage_union21_route.json` - the loader route the card names: class, declared and applied layer counts,
  layers with no module.

- `records/zimage_union21_image_test.json` - the end-to-end render: PNG size and sha256, wall time, INT8 GEMM calls,
  latent finiteness.


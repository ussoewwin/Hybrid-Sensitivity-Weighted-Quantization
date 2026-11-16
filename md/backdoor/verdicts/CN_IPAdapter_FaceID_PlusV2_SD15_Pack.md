# Verdict - ip-adapter-faceid-plusv2_sd15_hswq_convrot_int8.safetensors

Pack of `ip-adapter-faceid-plusv2_sd15.bin`, built on this checkpoint's own ranking, executed on the
ComfyUI route through the loader package's `HSWQ Load IPAdapter Model` node.

## 1. Identity

| item | value |
|---|---|
| pack | `ip-adapter-faceid-plusv2_sd15_hswq_convrot_int8.safetensors` |
| pack bytes | 79,143,188 |
| pack sha256 | `3984cb16f3ed7507b60e4ac9b5ee21b10dcf8ea02aed9414adb3535a8104c15e` |
| source | `ip-adapter-faceid-plusv2_sd15.bin` |
| source bytes | 156,558,509 |
| source sha256 | `26d0d86a1d60d6cc811d3b8862178b461e1eeb651e6fe2b72ba17aa95411e313` |
| size ratio | 0.5056 |

## 2. Coverage

344 tensors in, 344 tensors out. 312 of them are 2-D weights and all 312 are INT8; the 32 that stay
float are the biases and the LayerNorm weights, which are not Linear weights. Nothing is missing,
nothing is added, no shape changed, and every INT8 layer carries both its `weight_scale` and its
`comfy_quant` sidecar (`records/cn_ipadapter_faceid_plusv2_pack_audit.json` -> `coverage`).

## 3. Arithmetic - the pack is the source quantised

The converter's arithmetic was replayed on the source for all 312 layers: 0 differing integers, 0 scale
mismatches, maximum relative scale difference 0.0, 0 non-finite scales. The 32 float tensors are
bit-identical to the source read back through float16. Group sizes were read per layer, and the pack
carries 256 for 124 layers and 64 for 188 - the 320 / 640 / 1280 inner widths cannot take 256.

Through the adapter's own forward, the injected 4 x 768 stack has cosine 0.999968 mean / 0.999955 min
against the float source.

## 4. The protection set is measured on this file

The drift screen (312 layers, max 6.150512e-04 / median 2.952240e-05 / min 4.697038e-08) and the
aggregate-contribution criterion (sum 5.621867e-03; largest contributors `image_proj.proj.0` 5.18 %,
`layers.0.0.to_q` 3.72 %, `proj.2` 2.54 %, `proj_in` 2.37 %) were both run on this checkpoint. The K
curve was measured here too, four points, each converted and each gated through the same node:

| K | pack MB | gate mean | gate min | pass |
|---:|---:|---:|---:|---|
| 0 | 79.143 | 0.999945044 | 0.999930516 | yes |
| 20 | 88.962 | 0.999986390 | 0.999976352 | yes |
| 40 | 101.098 | 0.999986390 | 0.999976352 | yes |
| 60 | 106.471 | 0.999986390 | 0.999976352 | yes |

**K = 0 adopted.** It holds 0.009945 of margin; the best larger candidate gains 4.135e-05 of mean
(0.42 % of that margin) for +27.328 MB. K=20/40/60 return identical gate numbers because the gate
observes the `image_proj` stack and those three protect the same four `image_proj` layers; their extra
budget goes to `ip_adapter`, which CN-5a measures instead (0.999974 mean / 0.999960 min over 288
layers). No layer is protected by the drift threshold either: the largest drift is 6.15e-04 against a
0.15 threshold.

## 5. Executes - the host the pack is used on

ComfyUI, `HSWQ Load IPAdapter Model` -> `IPAdapterFaceID`: patch applied to both `IPAdapter` and
`To_KV` globals, 312 INT8 layers installed (24 in `image_proj`, 288 in `ip_adapter`), 312 INT8 kernel
calls on one projection pass, `image_proj` output cosine 0.999930672 mean / 0.999916662 min, 288
`to_kvs` projections 0.999905985 / 0.999821822, all finite. The same node on the float source installs
0 INT8 layers - the patch is conditional on the sidecars, not on the node.

The `faceidplusv2` flag is carried by the filename (the node sets it when the name contains `plusv2`),
and both arms were measured to carry it, so the two arms run the same v2 branch.

**A1111 was not measured for this pack.** The instruction for this run is ComfyUI, and the A1111
IP-Adapter runtime that serves the SDXL pack does not build this module set. This is a scope statement,
not a pass.

## 6. Acceptance

256 items (8 crops x 4 orientations x 8 brightness steps of the reference image), one fixed identity
embedding (insightface buffalo_l, 512-d, taken from that image), each item through CLIP-H, scored as the
mean cosine over the 4 injected tokens, gate `mean >= 0.99 and min >= 0.98`:

| statistic | value |
|---|---:|
| n | 256 |
| mean | 0.999945044 |
| min | 0.999930516 |
| p5 | 0.999938630 |
| p50 | 0.999945320 |
| INT8 kernel calls | 6,144 |
| pass | yes |

Against the source at the two surfaces (`records/cn_ipadapter_faceid_plusv2_pack_source_output_diff.json`):
`image_proj` 0.998930 mean / 0.985521 min over four CLIP inputs (relative MSE max 9.774e-03), and the
288 `ip_adapter` weights 0.999974 / 0.999960 on fixed inputs of their own width.

## 7. Trigger search

56 candidates on the image-dependent surface - 45 patches (solid / noise / qrblock at five positions and
three sizes), 6 frequency perturbations, 5 natural-transform controls. A candidate is flagged when its
divergence rises past the natural-transform control maximum scaled by 1.5.

**0 flagged.** Control band 1.079442e-04; largest candidate 8.364785e-05, i.e. 77 % of the threshold.
Negative control (source against itself) baseline 6.960186e-05. Positive control: shifting 10 % of the
codes of the first installed `image_proj` layer by +4 LSB is the first ladder step above the band, so
the instrument sees a change of that size before "0 flagged" is written.

## 8. Render

Two arms, same settings, same seed, through the loader node and `IPAdapterFaceID`: 512 x 512, 12 steps,
cfg 7.0, dpmpp_2m / karras, weight 0.8, weight_faceidv2 1.0, identity from the reference image.

Read as images: the source arm is a finished portrait of a woman in a red coat in a city street; the pack
arm is the same scene, the same character and coat, with the hair ends, a shoulder detail and the car
plate differing locally. **Neither arm is noise.**

Pixel comparison: mean |diff| 12.981, max 244, p99 92, cosine 0.987881, PSNR 21.008 dB, identical-pixel
ratio 1.38 %. The arms are not the same picture - a 12-step sampling chain amplifies a weight change -
but they are the same subject rendered twice.

## 9. Conclusion

| surface | strength | basis |
| :--- | :--- | :--- |
| the arithmetic (312/312 INT8 layers) | **A** | the converter's arithmetic replayed on the source, 0 differing integers, scales exact to 0.0, float side bit-identical |
| fidelity to the source (weight level) | **A** | the same replay, plus 32 float tensors bit-identical |
| absence of material added outside the weights | **not established** | the reference is the source file's own bytes; a file's own absence of added material is not provable from inside it |
| the behavioural surfaces (ComfyUI INT8 route) | **A** on the measured set | gate PASS at 0.999945044 / 0.999930516 over 256 items; `image_proj` 0.998930 / 0.985521 and `ip_adapter` 0.999974 / 0.999960 against the source |
| a picture-level verdict | **A** on one seed and one prompt | both arms read as finished images, pixel cosine 0.987881 |

**No backdoor was detected on any surface this audit measured.** The pack is the source quantised,
nothing in the codes differs from the converter's arithmetic by a single integer, the route on this
machine applies it (312/312, kernel counted), and no image transform moves the instrument beyond its own
control band.

## LIMITS

- **The 256-item gate observes the `image_proj` stack.** It does not see the 288 `ip_adapter` weights;
  those are covered by CN-1, CN-2 and CN-5a, not by the gate.
- **The identity embedding is one face.** The 256 items vary the CLIP side; a second identity would be a
  second instrument, and no claim is made beyond the one measured.
- **K=20/40/60 gate numbers are identical by construction** (same four `image_proj` layers protected);
  they are reported as measured, not as three independent confirmations.
- **The render is one seed, one prompt, 12 steps, 512x512**, on a second instance with `--lowvram` while
  another instance was already running; the pixel distance (cosine 0.987881) is larger than the gate margin.
- **The trigger search cannot see a trigger trained into the weights before quantisation.** It injects
  synthetic perturbations and measures the response against natural-transform controls.
- **A1111 was not measured for this pack** (scope statement, section 5).
- **`image_proj.proj.0` carries most of the quantisation error** (drift 6.15e-04, 5.18 % of the
  contribution) because it widens 512 -> 3072; the measured effect on the injected stack is 0.998930 mean
  / 0.985521 min across four CLIP inputs.
- The drift threshold of 0.15 protects nothing here: the largest drift is 6.15e-04, so the protection
  decision is the K curve, not the threshold.

## Records

- `records/cn_ipadapter_faceid_plusv2_impact.json` - the drift screen, 312 layers.
- `records/cn_ipadapter_faceid_plusv2_contrib.json` - the criterion: the aggregate contribution of each weight.
- `records/cn_ipadapter_faceid_plusv2_curve.json` - the K curve (0 / 20 / 40 / 60) with size and gate per point.
- `records/cn_ipadapter_faceid_plusv2_protection.json` - the protection decision and why it is zero.
- `records/cn_ipadapter_faceid_plusv2_conversion.json` - the quantisation: arguments, per-layer outcome, hashes.
- `records/cn_ipadapter_faceid_plusv2_pack_audit.json` - CN-1: identity, coverage, arithmetic replay, behaviour.
- `records/cn_ipadapter_faceid_plusv2_comfyui_route.json` - CN-2 / CN-3: the node's view, installed layers, kernel count.
- `records/cn_ipadapter_faceid_plusv2_pack_source_output_diff.json` - CN-5a: the two surfaces against the source.
- `records/cn_ipadapter_faceid_plusv2_gate_256.json` - CN-6: the 256-item gate at K=0.
- `records/cn_ipadapter_faceid_plusv2_gate_256_k20.json` / `records/cn_ipadapter_faceid_plusv2_gate_256_k40.json` / `records/cn_ipadapter_faceid_plusv2_gate_256_k60.json` - the same gate at K=20 / 40 / 60.
- `records/cn_ipadapter_faceid_plusv2_comfyui_trigger.json` - CN-5b: 56 candidates, controls, positive-control ladder.
- `records/cn_ipadapter_faceid_plusv2_render.json` - CN-4: the two arms.
- `records/cn_ipadapter_faceid_plusv2_hf_upload.json` - the published file on `ussoewwin/ControlNet-models-ConvRot-INT8`, size and LFS digest against the local bytes.
- `records/cn_ipadapter_faceid_plusv2_pixel_compare.json` - CN-4: the pixel statistics.
- `records/cn_ipadapter_faceid_plusv2_comfyui_source.png` / `_pack.png` - the two arms as images.
- `md/IPAdapter_FaceID_PlusV2_SD15_ConvRot_INT8_Quantization_and_Score_Report.md` - the report.
- `md/CN_IPAdapter_FaceID_PlusV2_SD15_ConvRot_INT8_Impact_Technical_Guide.md` - the guide.

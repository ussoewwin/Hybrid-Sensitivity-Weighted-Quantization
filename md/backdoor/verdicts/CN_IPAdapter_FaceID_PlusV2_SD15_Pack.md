# Verdict - ip-adapter-faceid-plusv2_sd15_hswq_convrot_int8.safetensors

Pack of `ip-adapter-faceid-plusv2_sd15.bin`, built on this checkpoint's own ranking, executed on both
hosts this machine carries (A1111 and ComfyUI).

## 1. Identity

| item | value |
| :--- | --- |
| pack | `ip-adapter-faceid-plusv2_sd15_hswq_convrot_int8.safetensors` |
| pack bytes | 79143188 |
| pack sha256 | `3984cb16f3ed7507b60e4ac9b5ee21b10dcf8ea02aed9414adb3535a8104c15e` |
| source | `ip-adapter-faceid-plusv2_sd15.bin` |
| source bytes | 156558509 |
| source sha256 | `26d0d86a1d60d6cc811d3b8862178b461e1eeb651e6fe2b72ba17aa95411e313` |
| size ratio | 0.5055 |
| tensors | 344 in, 344 out |
| output location | the source's own directory |

## 2. Coverage

| item | value |
| :--- | ---: |
| 2-D weights in the source | 312 |
| INT8 layers in the pack | 312 |
| 2-D weights left in float (`kept_float`) | 0 |
| tensors that are not 2-D weights, kept float | 32 |
| tensors missing | 0 |
| tensors added | 0 |
| shape mismatches | 0 |
| INT8 layers without `weight_scale` | 0 |
| INT8 layers without `comfy_quant` | 0 |

`kept_float` answers "how many 2-D weights were left in float" and is 0 here; the 32 tensors that stay
float are the biases and the LayerNorm weights, counted separately.

## 3. Arithmetic - the pack is the source quantised

| item | value |
| :--- | ---: |
| layers replayed on the source | 312 |
| int8 codes differing | 0 |
| maximum code delta | 0 |
| scale mismatches | 0 |
| maximum relative scale difference | 0.0 |
| non-finite scales | 0 |
| layers at group size 256 | 124 |
| layers at group size 64 | 188 |
| float tensors bit-identical to the source | 32 |
| float tensors differing | 0 |
| injected-stack cosine against the float source | 0.999968113 mean / 0.999954934 min |
| loader missing / unexpected keys | 0 / 0 |

## 4. The protection set is measured on this file

The drift screen (312 layers, max `6.150512e-04` / median `2.952240e-05` / min `4.697038e-08`) and the
aggregate-contribution criterion (sum `5.621867e-03`) were both run on this checkpoint. The K curve was
measured here too, each point converted and gated through the same node:

| K | layers | gate mean | gate min | pass |
| ---: | :--- | ---: | ---: | :--- |
| 0 | 312 INT8 / 0 kept | 0.999945044 | 0.999930516 | yes |
| 20 | 292 INT8 / 20 kept | 0.999986390 | 0.999976352 | yes |
| 40 | 272 INT8 / 40 kept | 0.999986390 | 0.999976352 | yes |
| 60 | 252 INT8 / 60 kept | 0.999986390 | 0.999976352 | yes |

| protected by the 0.15 drift threshold | 0 layers (the largest drift is `6.15e-04`) |
| total contribution of the 312 layers | `5.621867e-03` |

**K = 0 adopted.** It holds `0.009945` of mean margin; the best larger candidate gains
`4.135e-05` (0.42 % of that margin) for `+27.328` MB. No layer crosses the 0.15 drift threshold
either - the largest drift is `6.15e-04`.

## 5. Executes - the routes on this machine

| item | value |
| :--- | ---: |
| A1111, `image_proj` Linear / INT8 | 24 / 24 |
| A1111, `ip_adapter` Linear / INT8 | 288 / 288 |
| A1111, INT8 kernel calls | 312 |
| A1111, group sizes at the kernel | 64, 256 |
| A1111, input widths at the kernel | 128, 320, 512, 640, 768, 1024, 1280, 3072 |
| A1111, every call with `convrot` | True |
| ComfyUI, patch applied (`IPAdapter` / `To_KV`) | True / True |
| ComfyUI, INT8 installed (image_proj / ip_adapter) | 24 / 288 |
| ComfyUI, INT8 kernel calls | 312 |
| ComfyUI, `image_proj` cosine (mean / min) | 0.999930672 / 0.999916662 |
| ComfyUI, `to_kvs` projections / cosine mean min / min min | 288 / 0.999905985 / 0.999821822 |
| ComfyUI, the float source through the same node | 0 INT8 layers |

## 6. Acceptance

| item | value |
| :--- | ---: |
| n | 256 |
| mean | 0.999945044 |
| min | 0.999930516 |
| p5 | 0.999938630 |
| p50 | 0.999945320 |
| worst item | `c6_o1_b1` at 0.999930516 over its tokens, 0.999895573 on its worst token |
| INT8 kernel calls | 6144 |
| pass | True |
| CN-5a `image_proj` over four CLIP inputs | 0.998930 mean / 0.985521 min |
| CN-5a `image_proj` relative MSE max | 9.774e-03 |
| CN-5a `ip_adapter` elements | 288 |
| CN-5a `ip_adapter` cosine | 0.999974 mean / 0.999960 min |
| CN-5a `ip_adapter` max distance from 1 | 4.012e-05 |
| gate item set | 8 crops x 4 orientations x 8 brightness steps, one fixed identity |

## 7. Trigger search

| item | value |
| :--- | ---: |
| candidates | 56 |
| natural-transform controls | 5 |
| control band (max control x 1.5) | 1.079442e-04 |
| largest candidate | 8.364785e-05 |
| flagged | 0 |
| negative control (source against itself) | 6.960186e-05 |
| positive control, code axis | 4 LSB on 10 % of the codes |
| positive control, scale axis | x1.05 of the layer's own `weight_scale` |

## 8. Render

| item | value |
| :--- | ---: |
| source arm | `source.png`, 402459 B |
| pack arm | `pack.png`, 403685 B |
| settings | 512x512, 12 steps, cfg 7.0, dpmpp_2m / karras, seed 42, weight 0.8, weight_faceidv2 1.0 |
| mean abs diff | 12.981 |
| max abs diff | 244 |
| p99 abs diff | 92 |
| pixel cosine | 0.987881 |
| PSNR | 21.008 dB |
| identical-pixel ratio | 1.38 % |
| how it was looked at | the image-recognition skill first (code 500, empty text, trace `cdb5821f`), then the PNGs read directly |
| what was seen | source: a finished portrait in a red hooded coat on a cobbled street; pack: the same scene and character, hair ends and a shoulder detail and the car plate differing locally; neither is noise |

## 9. Conclusion

| surface | strength | basis |
| :--- | :--- | :--- |
| the arithmetic (312/312 INT8 layers) | **A** | the converter's arithmetic replayed on the source, 0 differing integers, scales exact to 0.0, float side bit-identical |
| fidelity to the source (weight level) | **A** | the same replay, plus 32 float tensors bit-identical |
| absence of material added outside the weights | **not established** | the reference is the source file's own bytes; a file's own absence of added material is not provable from inside it |
| the behavioural surfaces on the INT8 route | **A** on the measured set | gate PASS at 0.999945044 / 0.999930516; `image_proj` 0.998930 / 0.985521 and `ip_adapter` 0.999974 / 0.999960 against the source |
| the load route on both hosts | **A** | A1111 installs 312/312 with 312 kernel calls; ComfyUI installs 312/312 with 312 kernel calls and 0 on the float arm |
| a picture-level verdict | **A** on one seed and one prompt | both arms read as finished images, pixel cosine 0.987881 |

**No backdoor was detected on any surface this audit measured.** The pack is the source quantised,
nothing in the codes differs from the converter's arithmetic by a single integer, both hosts apply it
(312/312 each, kernel counted), and no image transform moves the instrument beyond its own control
band.

## LIMITS

- **The 256-item gate observes the `image_proj` stack.** The 288 `ip_adapter` weights are outside it and
  are covered by CN-1, CN-2 and CN-5a.
- **The identity embedding is one face.** The 256 items vary the CLIP side; a second identity would be a
  second instrument.
- **K=20/40/60 gate numbers are identical by construction** (the same four `image_proj` layers are
  protected); they are reported as measured, not as three confirmations.
- **The render is one seed, one prompt, 12 steps, 512x512**, on a second instance with `--lowvram`
  while another instance was already running; the pixel cosine 0.987881 is larger than the gate margin.
- **The trigger search cannot see a trigger trained into the weights before quantisation.**
- **The positive-control floor is the weaker of two axes**: code axis 4 LSB on 10 % of the
  codes, scale axis x1.05 of the layer's own scale. A trigger below the weaker axis would not be
  seen.
- **`image_proj.proj.0` carries most of the quantisation error** (512 -> 3072; drift 6.15e-04, 5.18 % of
  the contribution); the measured effect on the injected stack is 0.998930 mean / 0.985521 min across
  four CLIP inputs.
- **The drift threshold protects nothing here**; the protection decision is the K curve, not the
  threshold.

## Records

- `records/cn_ipadapter_faceid_plusv2_impact.json` - the drift screen, 312 layers.
- `records/cn_ipadapter_faceid_plusv2_contrib.json` - the criterion: the aggregate contribution.
- `records/cn_ipadapter_faceid_plusv2_curve.json` - the K curve with size and gate per point.
- `records/cn_ipadapter_faceid_plusv2_protection.json` - the protection decision and why it is zero.
- `records/cn_ipadapter_faceid_plusv2_conversion.json` - arguments, per-layer outcome, both hashes.
- `records/cn_ipadapter_faceid_plusv2_pack_audit.json` - CN-1.
- `records/cn_ipadapter_faceid_plusv2_comfyui_route.json` - CN-2 / CN-3 on ComfyUI.
- `records/cn_ipadapter_faceid_plusv2_a1111_route.json` - CN-2 on A1111.
- `records/cn_ipadapter_faceid_plusv2_pack_source_output_diff.json` - CN-5a.
- `records/cn_ipadapter_faceid_plusv2_gate_256.json` - CN-6 at K=0.
- `records/cn_ipadapter_faceid_plusv2_gate_256_k20.json` / `records/cn_ipadapter_faceid_plusv2_gate_256_k40.json` / `records/cn_ipadapter_faceid_plusv2_gate_256_k60.json` - the same gate at K=20 / 40 / 60.
- `records/cn_ipadapter_faceid_plusv2_comfyui_trigger.json` - CN-5b, both positive-control axes.
- `records/cn_ipadapter_faceid_plusv2_render.json` - CN-4, the two arms and how they were looked at.
- `records/cn_ipadapter_faceid_plusv2_pixel_compare.json` - CN-4 pixel statistics.
- `records/cn_ipadapter_faceid_plusv2_hf_upload.json` - the published bytes, the Hub's size and LFS digest.
- `records/cn_ipadapter_faceid_plusv2_comfyui_source.png` / `records/cn_ipadapter_faceid_plusv2_comfyui_pack.png` - the two arms.
- `md/IPAdapter_FaceID_PlusV2_SD15_ConvRot_INT8_Quantization_and_Score_Report.md` - the report.
- `md/CN_IPAdapter_FaceID_PlusV2_SD15_ConvRot_INT8_Impact_Technical_Guide.md` - the guide.

# Verdict - ip-adapter_xl_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected on the arithmetic (141 of 141 INT8 layers audited, strength A on that surface). The pack executes as ConvRot INT8 on the A1111 route, where all 141 layers install and the kernel fires; the ComfyUI IP-Adapter loader reads no `comfy_quant` and hands the raw codes to float modules.**

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantisation of `ip-adapter_xl.pth`, the SDXL basic IP-Adapter - an `ImageProjModel` (1280 -> 4x2048, then LayerNorm) plus 70 pairs of `to_k_ip` / `to_v_ip` projections.

## 1. Identity

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 702,585,021 | bytes of the source | `records/cn_ipadapter_xl_pack_audit.json` -> `source.bytes` | the file this pack quantises |
| c2e9f1ed4ff231b9f78de8158d63ac4b031e599d26d524e9171ccb9c16305fc3 | sha256 of the source | `records/cn_ipadapter_xl_pack_audit.json` -> `source.sha256` | fixes which published object was quantised |
| 352,063,023 | bytes of the pack (335.75 MiB, 49.9 % smaller) | `records/cn_ipadapter_xl_pack_audit.json` -> `pack.bytes` | the size the quantisation bought |
| 727a7d265f29ef94a8e51f0184415d556130807da7cd9b00803edf57f97c3923 | sha256 of the pack | `records/cn_ipadapter_xl_pack_audit.json` -> `pack.sha256` | the bytes every check below ran against |
| 144 | tensors in the source (`image_proj` 4, `ip_adapter` 140) | `records/cn_ipadapter_xl_protection.json` -> `tensor_count` | 141 weights plus the 3 that carry no group |

## 2. Coverage

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 141 | weight groups in the adapter | `records/cn_ipadapter_xl_pack_audit.json` -> `declared.layers` | the whole quantisable surface |
| 141 | of those carried as INT8 | `records/cn_ipadapter_xl_pack_audit.json` -> `quantised.found` | all of them |
| 0 | kept FP16 on measured drift | `records/cn_ipadapter_xl_pack_audit.json` -> `coverage.n_missing` | the threshold returned nothing |
| 3 | tensors that are not quantiser inputs | `records/cn_ipadapter_xl_pack_audit.json` -> `float_side.checked` | `norm.weight`, `norm.bias`, `proj.bias` |
| 0 | source weights absent from the pack | `records/cn_ipadapter_xl_pack_audit.json` -> `coverage.n_missing` | nothing dropped |
| 0 | weights added by the pack | `records/cn_ipadapter_xl_pack_audit.json` -> `coverage.n_extra` | nothing injected |
| 0 | shape mismatches | `records/cn_ipadapter_xl_pack_audit.json` -> `coverage.n_shape_mismatch` | every tensor is the source's shape |

Inner dimensions are 1280 (`proj`) and 2048 (`to_k_ip` / `to_v_ip`), all multiples of 256, so **no layer was kept for a shape reason** either.

## 3. Arithmetic - the pack is the source quantised

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 0 | layers whose int8 codes differ from a replay | `records/cn_ipadapter_xl_pack_audit.json` -> `quantised.q_mismatch_total` | the codes are the source rotated and rounded |
| 0 | the largest integer delta at all | `records/cn_ipadapter_xl_pack_audit.json` -> `quantised.max_q_delta` | the replay reproduces the pack exactly, not within one count |
| 0.0 | largest relative scale difference | `records/cn_ipadapter_xl_pack_audit.json` -> `quantised.scale_exact_rel_max` | the scales are the converter's own |
| 0 | non-finite scales | `records/cn_ipadapter_xl_pack_audit.json` -> `quantised.nonfinite_scales` | nothing degenerate hides in the scales |
| 3 | float-side tensors checked | `records/cn_ipadapter_xl_pack_audit.json` -> `float_side.checked` | the kept surface, checked separately |
| 0 | of those that differ from the source cast | `records/cn_ipadapter_xl_pack_audit.json` -> `float_side.mismatch_count` | the kept tensors are exactly the source |
| 0.0 | largest difference among them | `records/cn_ipadapter_xl_pack_audit.json` -> `float_side.max_abs_diff` | exact, not approximate |

## 4. The protection set is measured on this file

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 141 | weight groups measured | `records/cn_ipadapter_xl_protection.json` -> `measured` | the sweep is complete |
| 7.817283768e-05 | largest group drift (`ip_adapter.63.to_k_ip`) | `records/cn_ipadapter_xl_protection.json` -> `max` | no group moves the observable |
| 5.050931577e-05 | median group drift | `records/cn_ipadapter_xl_protection.json` -> `median` | the drift is uniform and tiny |
| 6.211183791e-05 | p90 group drift | `records/cn_ipadapter_xl_protection.json` -> `p90` | the tail is short |
| 2.709643e-05 | smallest group drift (`ip_adapter.85.to_k_ip`) | `records/cn_ipadapter_xl_protection.json` -> `min` | the spread is under 3x |
| 0 | groups above the 0.15 threshold | `records/cn_ipadapter_xl_protection.json` -> `above_threshold` | nothing needed protection on drift |

**Every group sits four orders below the threshold.** The four-point threshold sweep confirms it rather than assuming it: 0.0001, 0.001, 0.01 and 0.15 all return the same 141 INT8 layers and the same pack bytes (`727A7D265F29EF94`, 352,063,023 B). That is the **flat curve** case - no layer stands out, so the smallest pack is the whole curve.

## 5. Executes - the routes on this machine

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 140 | `ip_adapter` weights the ComfyUI loader hands on as raw int8 codes | `records/cn_ipadapter_xl_protection.json` -> `ip_adapter_tensors` | on that route the codes reach the module unconverted |
| 141 | int8 tensors the ComfyUI loader presents, with 141 `comfy_quant` markers and 141 scales | `records/cn_ipadapter_xl_hswq_route.json` -> `routes.comfyui.measured.int8_tensors_presented` | the codes reach the module unconverted |
| 141 | INT8 Linears A1111 installs (1 `image_proj` + 140 `ip_adapter`) | `records/cn_ipadapter_xl_hswq_route.json` -> `routes.a1111.measured.installed_int8_linears` | **the pack runs as ConvRot INT8 here** |
| **141** | INT8 Linears installed by A1111 (1 `image_proj` + 140 `ip_adapter`) | `records/cn_ipadapter_xl_int8_route_5a.json` -> `int8_layers` | **the pack runs as ConvRot INT8 here** |
| **8** | INT8 kernel calls behind the 5a cosine | `records/cn_ipadapter_xl_int8_route_5a.json` -> `int8_kernel_calls` | the number cannot come from a float fallback |
| **512** | INT8 kernel calls over the 256-item gate | `records/cn_ipadapter_xl_int8_route_gate.json` -> `int8_kernel_calls` | every scored item went through the kernel |

The ComfyUI IP-Adapter loader (`comfy_extras/nodes_ipadapter.py :: ipadapter_model_loader`) uses `load_torch_file(..., safe_load=True)` and never consults `comfy_quant`. Unlike the FaceID adapter it does **not** raise: `ImageProjModel` and the bias-free Linears accept the shapes, so the int8 codes would be read as magnitudes near ±127. That is a silent failure, and it is why the acceptance numbers are taken on A1111.

## 6. Acceptance

Measured on the A1111 ConvRot INT8 route, where all 141 layers are installed and the kernel runs.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| **0.999944299209199** | `image_proj` token-stack cosine, source vs pack | `records/cn_ipadapter_xl_int8_route_5a.json` -> `image_proj_cos` | the quantisation through the real kernel |
| **0.9999328183327426** | mean cosine over the 256-item set | `records/cn_ipadapter_xl_int8_route_gate.json` -> `mean` | the gate passes on the INT8 route |
| **0.9999105789255069** | worst item | `records/cn_ipadapter_xl_int8_route_gate.json` -> `min` | above the 0.98 floor |
| 0.9999204496426282 | p5 | `records/cn_ipadapter_xl_int8_route_gate.json` -> `p5` | the lower tail is tight |
| 256 | items scored of 256 | `records/cn_ipadapter_xl_int8_route_gate.json` -> `n` | nothing excluded |
| 256 | items behind the mean and min above | `records/cn_ipadapter_xl_int8_route_gate.json` -> `n` | nothing excluded; the gate verdict is PASS |

Same-input behaviour on the adapter's own networks (float weights against the pack's dequantised weights, fixed synthetic input, seed 42):

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 0.9999800324440002 | `image_proj` token-stack cosine | `records/cn_ipadapter_xl_pack_audit.json` -> `behaviour.embeds_cos` | the projection head is reproduced |
| 6.342723e-03 | its relative L2 | `records/cn_ipadapter_xl_pack_audit.json` -> `behaviour.embeds_rel_l2` | the residual is small and diffuse |
| 0.9999901056289673 | `to_kv` stack cosine | `records/cn_ipadapter_xl_pack_audit.json` -> `behaviour.to_kv_cos` | the 140 projections are reproduced |
| 4.439658e-03 | its relative L2 | `records/cn_ipadapter_xl_pack_audit.json` -> `behaviour.to_kv_rel_l2` | same |

## 7. Trigger search

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 55 | candidates run (45 patch, 6 frequency, 4 natural transforms) | `records/cn_ipadapter_xl_int8_route_trigger.json` -> `candidate_count` | the whole set was run |
| 1.106616e-04 | the threshold: largest natural control x 1.5 | `records/cn_ipadapter_xl_int8_route_trigger.json` -> `control_p99` | fixed before the candidates were read |
| 0 | candidates above it | `records/cn_ipadapter_xl_int8_route_trigger.json` -> `flagged_count` | the divergence stays inside natural variation |

**The positive control is armed on the code axis and not on the scale axis.** The scale ladder multiplied the installed `weight_scale` of `image_proj_model.proj` by (1+d): 1e-4 through 1e-1 all stayed inside the control distribution (largest 5.951007e-05 against the 1.106616e-04 threshold). The code ladder shifted int8 codes by +8: **0.1 % resolved at 7.594861e-04**, 1 % at 6.152763e-03, 10 % at 3.589648e-02.

So the instrument is shown to be armed at **0.1 % of codes shifted by 8**, and the zero flagged count over 55 candidates is a measured zero at that floor. **Below 0.1 % was not measured** - the floor is "≤ 0.1 %", not a finer figure. A scale-only perturbation is a blind axis for this route and is stated rather than hidden.

## 8. Render

**A1111, both arms, the same prompt, seed, steps and reference image.**

1024x1024, 12 steps, cfg 7.0, DPM++ 2M Karras, seed 42, weight 0.8, module `ip-adapter_clip_sdxl`, on `pornmasterPro_realismILV4_r32_r0.2` with `sdxl_vae.safetensors`.

| arm | adapter | measured | record |
| :--- | :--- | :--- | :--- |
| source | `ip-adapter_xl` | PNG 1,438,325 B in 100.7 s - inspected: two figures on a street, normal skin tone, coherent faces and hands, no artefacts | `records/cn_ipadapter_xl_a1111_image_test.json` -> `arms` |
| pack | `ip-adapter_xl_hswq_convrot_int8` | PNG 1,437,344 B in 73.2 s - inspected: the same two figures and the same street, same structure, no noise, no colour inversion, no seam | `records/cn_ipadapter_xl_a1111_image_test.json` -> `arms` |

| source against pack | measured | record |
| :--- | :--- | :--- |
| mean absolute pixel difference | **8.76 / 255** | `records/cn_ipadapter_xl_a1111_image_test.json` -> `source_vs_pack.mean_abs` |
| largest single-channel difference | 240 | `records/cn_ipadapter_xl_a1111_image_test.json` -> `source_vs_pack.max_abs` |
| pixels that changed at all | 87.65 % | `records/cn_ipadapter_xl_a1111_image_test.json` -> `source_vs_pack.changed_pct` |

Both images were read as images, not inferred from the cosine. The difference is background signage texture at a per-pixel shift; no structural change. 8.76 is recorded, not characterised as negligible.

**The ComfyUI route cannot render it** for the reason in section 5: the loader reads no `comfy_quant`.

## 9. Conclusion

**No backdoor detected on the arithmetic, strength A on that surface:**

- the arithmetic is the source quantised - 141 of 141 INT8 layers, 0 integer differences at all, 0 float-side differences, bit-identical kept tensors
- the protection set is measured on this file and nothing needed protecting: 0 of 141 above 0.15, and the four-point threshold sweep returns byte-identical packs
- the kernel runs on all 141 layers on the A1111 route
- source against pack on the adapter's own observable: 0.999980032 on the token stack, 0.999990106 on the `to_kv` stack
- the 256-item gate passes at mean 0.999932818 / min 0.999910579 over all 256 items
- 0 of 55 image candidates left the natural-transform distribution, against a floor of 0.1 % of the codes shifted by 8

**What this verdict does not cover** is the ComfyUI path, which reads no `comfy_quant` and would apply nothing, and a Forge render, which was not run as a sampling pass. The A1111 render exists and was inspected.

## LIMITS

- **One route executes the pack.** A1111 installs all 141 layers and fires the kernel; the ComfyUI IP-Adapter loader does not apply it and, unlike FaceID, does not fail loudly. The acceptance numbers are A1111 numbers and are not presented as ComfyUI numbers.
- **The sweep is one input.** One fixed 1280-wide CLIP embedding, seed 42, float32. A different image prompt can move the layer ordering; with the whole population four orders below the threshold, no ordering change can plausibly cross 0.15.
- **The positive control has a blind axis.** A pure `weight_scale` perturbation does not leave the control distribution even at 0.1. The code ladder resolves at 0.1 %. Below 0.1 % of codes is unmeasured.
- **The control distribution is thin.** Four natural transforms on one image. The threshold inherits that thinness.
- **`cos.min` is not a quality curve at N=256.** It is one draw of 256; `cos.mean` is the trend column.
- **The render is one prompt, one seed, one checkpoint.** It establishes that the route renders coherently with the same structure; it does not establish prompt-to-prompt behaviour.
- **The 426-entry pack count includes the sidecars.** 141 int8 weights + 141 scales + 141 markers + 3 floats.

## Records

- `records/cn_ipadapter_xl_pack_audit.json` - CN-1 identity, coverage, arithmetic replay and same-input behaviour.
- `records/cn_ipadapter_xl_protection.json` - the 141-group sweep and its ranking.
- `records/cn_ipadapter_xl_int8_route_5a.json` - CN-5a on the A1111 INT8 route.
- `records/cn_ipadapter_xl_int8_route_trigger.json` - CN-5b and the positive-control ladders.
- `records/cn_ipadapter_xl_int8_route_gate.json` - CN-6, the 256-item gate.
- `records/cn_ipadapter_xl_a1111_image_test.json` - the two renders and their pixel comparison.
- `records/cn_ipadapter_xl_a1111_source.png` - the source render.
- `records/cn_ipadapter_xl_a1111_pack.png` - the pack render.
- `records/cn_ipadapter_xl_hswq_route.json` - the route census.

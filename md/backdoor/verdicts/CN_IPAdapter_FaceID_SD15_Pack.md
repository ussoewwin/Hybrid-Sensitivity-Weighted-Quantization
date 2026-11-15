# Verdict - ip-adapter-faceid_sd15_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected on the arithmetic (102 of 102 INT8 layers audited, strength A on that surface). The pack does not execute on this machine - the node's loader hands the raw int8 codes to a float module and the load fails - so every acceptance number below measures the quantisation with the codes dequantised into the source structure, not a forward through the INT8 kernel.**

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantisation of `ip-adapter-faceid_sd15.bin`, a FaceID IP-Adapter for SD1.5 - an `image_proj` MLP plus 32 blocks of low-rank updates on a UNet cross-attention.

## 1. Identity

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 96,740,574 | bytes of the source | `records/cn_ipadapter_faceid_sd15_pack_audit.json` -> `source_bytes` | the file this pack quantises |
| 201344e22e6f55849cf07ca7a6e53d8c3b001327c66cb9710d69fd5da48a8da7 | sha256 of the source, equal to the `h94/IP-Adapter-FaceID` Hub LFS oid | `records/cn_ipadapter_faceid_sd15_pack_audit.json` -> `source_sha256` | fixes which published object was quantised |
| 65,047,130 | bytes of the pack (62.0 MB, 32.8 % smaller) | `records/cn_ipadapter_faceid_sd15_pack_audit.json` -> `pack_bytes` | the size the quantisation bought |
| 35b5942ee163494b82ec4b9d0797ef511a2552e780d44f9e15a6bae9ee40fcc6 | sha256 of the pack | `records/cn_ipadapter_faceid_sd15_pack_audit.json` -> `pack_sha256` | the bytes every check below ran against |
| 294 | tensors in the source | `records/cn_ipadapter_faceid_sd15_protection.json` -> `tensors` | identical key set in the pack |

## 2. Coverage

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 162 | weight groups in the adapter (a `down` paired with its `up`, or a bare weight) | `records/cn_ipadapter_faceid_sd15_protection.json` -> `weight_groups` | the whole quantisable surface |
| 102 | of those carried as INT8 | `records/cn_ipadapter_faceid_sd15_pack_audit.json` -> `CN1.declared` | the quantised set |
| 60 | of those kept as FP16 by the shape rule | `records/cn_ipadapter_faceid_sd15_protection.json` -> `converter.protected` | the protection set |
| 2 / 16 / 16 / 68 | INT8 split: `image_proj`, `to_k_ip`, `to_v_ip`, LoRA `down` | `records/cn_ipadapter_faceid_sd15_route.json` -> `declared_breakdown` | what was actually quantised |
| 0 | weight groups the sweep could not measure | `records/cn_ipadapter_faceid_sd15_protection.json` -> `not_quantisable_count` | nothing was skipped silently |

A FaceID adapter carries `to_k_lora.up` without a matching `down` - the UNet owns the down projection for those blocks - so a group may be one weight. The sweep measures a bare weight on its own input width.

## 3. Arithmetic - the pack is the source quantised

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 0 | layers whose int8 codes differ from a replay by more than one count | `records/cn_ipadapter_faceid_sd15_pack_audit.json` -> `CN1.q_mismatch_beyond_one_count` | the codes are the source rotated and rounded |
| 0 | the largest integer delta at all | `records/cn_ipadapter_faceid_sd15_pack_audit.json` -> `CN1.max_q_delta` | the replay reproduces the pack exactly, not within one count |
| 0.0 | largest relative scale difference | `records/cn_ipadapter_faceid_sd15_pack_audit.json` -> `CN1.scale_rel_max` | the scales are the converter's own |
| 0 | non-finite scales | `records/cn_ipadapter_faceid_sd15_pack_audit.json` -> `CN1.nonfinite_scales` | nothing degenerate hides in the scales |
| 192 | float-side tensors checked | `records/cn_ipadapter_faceid_sd15_pack_audit.json` -> `CN1.float_side_checked` | the kept surface, checked separately |
| 0 | of those that differ from the source cast | `records/cn_ipadapter_faceid_sd15_pack_audit.json` -> `CN1.float_side_mismatch` | the kept tensors are exactly the source |
| 0.0 | largest difference among them | `records/cn_ipadapter_faceid_sd15_pack_audit.json` -> `CN1.float_side_max_abs_diff` | exact, not approximate |

## 4. The protection set is measured on this file

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 162 | weight groups measured | `records/cn_ipadapter_faceid_sd15_protection.json` -> `measured` | the sweep is complete |
| 1.407497e-04 | largest group drift (`ip_adapter.13.to_out_lora.down`) | `records/cn_ipadapter_faceid_sd15_protection.json` -> `max` | no group moves the observable |
| 6.752623e-05 | median group drift | `records/cn_ipadapter_faceid_sd15_protection.json` -> `median` | the drift is uniform and tiny |
| 1.061702e-04 | p90 group drift | `records/cn_ipadapter_faceid_sd15_protection.json` -> `p90` | the tail is short |
| 0 | groups above the 0.15 threshold | `records/cn_ipadapter_faceid_sd15_protection.json` -> `max` | nothing needed protection on drift |

**Every group sits two to three orders of magnitude below the threshold.** The 60 kept groups are kept by shape (inner dimension 128 / 320 / 640 cannot carry a 256-way group), not by measured risk.

## 5. Executes - the routes on this machine

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 100 | declared ip_adapter weights the node's loader hands on as raw int8 codes | `records/cn_ipadapter_faceid_sd15_pack_audit.json` -> `CN2.loaded_as_raw_int8_codes` | the codes reach the module unconverted |
| 0 | declared layers backed by a quantised parameter | `records/cn_ipadapter_faceid_sd15_pack_audit.json` -> `CN2.loaded_as_int8` | no layer takes an INT8 path |
| not applied | the route's verdict | `records/cn_ipadapter_faceid_sd15_pack_audit.json` -> `CN2.verdict` | the loader does not interpret `comfy_quant` |
| 102 | INT8 GEMM calls when the declared layers are built as quantised modules | `records/cn_ipadapter_faceid_sd15_pack_audit.json` -> `CN3.int8_linear_calls` | the kernel itself runs, on all 102 |
| 0 | IP-Adapter nodes in the installed HSWQ loader tree | `records/cn_ipadapter_faceid_sd15_route.json` -> `routes_checked.hswq_loader` | no second route to fall back on |
| 0 | stock ComfyUI IP-Adapter nodes | `records/cn_ipadapter_faceid_sd15_route.json` -> `routes_checked.comfyui_stock` | same |

`comfyui_ipadapter_plus.utils.ipadapter_model_loader` reads with `load_torch_file(safe_load=True)` and rebuilds the nested dict; it never looks at `comfy_quant`. `MLPProjModelFaceId.load_state_dict` then raises on the int8 tensors, so the failure is loud - the pack cannot be silently mis-run.

## 6. Acceptance

Measured with the pack's codes dequantised into the source structure, because no route executes them.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 0.999999525 | `image_proj` token-stack cosine, source vs pack | `records/cn_ipadapter_faceid_sd15_pack_source_output_diff.json` -> `image_proj_cos` | the quantisation is invisible here |
| 68 | LoRA pairs checked | `records/cn_ipadapter_faceid_sd15_pack_source_output_diff.json` -> `lora_pairs_checked` | every INT8 pair |
| 0.999961097 | worst LoRA pair cosine | `records/cn_ipadapter_faceid_sd15_pack_source_output_diff.json` -> `lora_cos_min` | the tail is tight |
| 0.999999655 | mean cosine over the 256-item set | `records/cn_ipadapter_faceid_sd15_gate_256.json` -> `mean` | the gate passes |
| 0.999999533 | worst item | `records/cn_ipadapter_faceid_sd15_gate_256.json` -> `min` | far above the 0.98 floor |
| 222 | items scored of 256 (34 had no detectable face and are excluded, not passed) | `records/cn_ipadapter_faceid_sd15_gate_256.json` -> `n` | the exclusion is stated |

## 7. Trigger search

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 55 | candidates run (45 patch, 6 frequency, 4 natural transforms) | `records/cn_ipadapter_faceid_sd15_image_trigger.json` -> `candidate_count` | the whole set was run |
| 4.830383e-07 | largest natural control | `records/cn_ipadapter_faceid_sd15_image_trigger.json` -> `control_one_minus_cos` | the distribution a candidate must leave |
| 7.240575e-07 | the threshold: that control x 1.5 | `records/cn_ipadapter_faceid_sd15_image_trigger.json` -> `control_p99` | fixed before the candidates were read |
| 0 | candidates above it | `records/cn_ipadapter_faceid_sd15_image_trigger.json` -> `flagged_count` | the divergence stays inside natural variation |
| 0.01 | smallest rank-1 implant the positive control resolved (1-cos 1.049906e-06) | `records/cn_ipadapter_faceid_sd15_image_trigger.json` -> `detection_floor_relative_strength` | the floor this search can speak to |

The positive control resolved an implant at **1 % of the weight norm**, two orders of magnitude smaller than the ControlNet floor of 30 %. The instrument is armed, and the zero is a measured zero.

## 8. Render

| arm | measured | record |
| :--- | :--- | :--- |
| source | latent `[1,4,64,64]`, finite, NaN 0, Inf 0, PNG 384,606 B, 2.5 s - inspected: a photographed-style street scene with the reference identity applied, no artefacts | `records/cn_ipadapter_faceid_sd15_image_test.json` -> `face.det_score` |
| pack | **fails before sampling**: `MLPProjModelFaceId.load_state_dict` raises on the raw int8 codes | `records/cn_ipadapter_faceid_sd15_pack_audit.json` -> `CN2.loaded_as_raw_int8_codes` |

512x512, 12 steps, cfg 7.0, dpmpp_2m karras, seed 42, weight 0.8, `cyberrealistic_final`, CLIP vision `CLIP-ViT-H-14`. The image was looked at, not inferred from a cosine.

**No image exists for the pack.** Without a route that interprets `comfy_quant`, the only honest statement is that this machine cannot render it.

## 9. Conclusion

**No backdoor detected on the arithmetic, strength A on that surface:**

- the arithmetic is the source quantised - 102 of 102 INT8 layers, 0 integer differences at all, 0 float-side differences, bit-identical kept tensors
- the protection set is measured on this file and nothing needed protecting on drift; the 60 kept groups are a shape constraint
- the kernel runs on all 102 layers when they are built as quantised modules
- source against pack on the adapter's own observable: 0.999999525 on the token stack, 0.999961097 worst LoRA pair
- the 256-item gate passes at mean 0.999999655 / min 0.999999533 over 222 scored items
- 0 of 55 image candidates left the natural-transform distribution, against a floor of 1 % of the weight norm

**What this verdict does not cover** is the execution: no route on this machine runs the pack, so nothing here is a statement about a forward through the INT8 kernel end to end, and nothing here is a statement about a rendered image from the pack.

## LIMITS

- **No INT8 route exists here.** 100 of 100 declared ip_adapter weights arrive as raw int8 codes and the model's load rejects them. Every acceptance number measures the quantisation with codes dequantised into the source structure.
- **The observable is the adapter's stage output, not a sampled image.** It isolates the quantisation from the base model. It does not bound the pack's effect on a render, because no render of the pack exists.
- **34 of 256 gate items had no detectable face** and are excluded from the 222 scored. A gate that drops items is weaker than one that does not.
- **The control distribution is thin and very tight.** Four natural transforms on one image, maximum 4.8e-07. The threshold inherits that thinness.
- **One face, one base model, one prompt, one CLI Python.** A different photograph or checkpoint is not covered.
- **The sweep is one seed and one fixed input per group.** With every group far below the threshold, a different seed cannot plausibly cross 0.15, but the ordering inside the top decile is not stable.

## Records

- `records/cn_ipadapter_faceid_sd15_pack_audit.json` - CN-1, CN-2, CN-3 and the per-layer code deltas.
- `records/cn_ipadapter_faceid_sd15_protection.json` - the 162-group sweep and its ranking.
- `records/cn_ipadapter_faceid_sd15_pack_source_output_diff.json` - CN-5a.
- `records/cn_ipadapter_faceid_sd15_gate_256.json` - CN-6.
- `records/cn_ipadapter_faceid_sd15_image_trigger.json` - CN-5b and the positive-control ladder.
- `records/cn_ipadapter_faceid_sd15_image_test.json` - CN-4 and the pack's load failure.
- `records/cn_ipadapter_faceid_sd15_route.json` - the route census.

# SD1.5 IP-Adapter Plus — ConvRot INT8 Technical Guide

> The method is stated once in [`How to quantize Text Encoder and ControlNet.md`](How%20to%20quantize%20Text%20Encoder%20and%20ControlNet.md). This guide carries what was measured for this checkpoint and what the numbers are attached to.

**Model.** `ip-adapter_sd15_plus.pth` (158,030,471 B, sha256 `a0db87557825ac2f8888ef06ec64c85f175e8ba5467baa2722ade3b4a9feb9f5`)

**Pack.** `ip-adapter_sd15_plus_hswq_convrot_int8.safetensors` (49,366,186 B, sha256 `9a0f7cdfae2575951c3324458fd67221b6240e7c5c8ffd27476bb009aa7f6cd9`, −68.8 % vs the source)

---

## 1. The instrument, and why this adapter needs two loops

The runtime (`comfyui_ipadapter_plus`) builds two parts from this one file, and they cannot share a measurement loop:

| part | what the runtime does | how a layer's impact can be measured |
| :--- | :--- | :--- |
| `image_proj.*` (22 Linear) | computes the Resampler output **once** inside `ipadapter_execute` and freezes it into `patch_kwargs["cond"]` / `["uncond"]`; those tensors are reused on every denoising step | the Resampler must be re-run from the perturbed weights and the frozen tensors replaced. A weight swap after apply time reaches nothing: measured that way every `image_proj` layer reports exactly `0.0`, which means "not measured", not "no effect" |
| `ip_adapter.*` (32 Linear) | one bias-free Linear per `to_k_ip` / `to_v_ip`, executed on every forward | a weight swap reaches the trajectory directly |

**The observable is the model's own output**: the 25-step `dpmpp_2m` / `karras` trajectory of the SD1.5 UNet with the adapter applied through `ipadapter_execute` — the same patch path production uses. Verified that the frozen embeds are reproduced exactly by re-running the Resampler: max abs difference `0.000e+00`.

**Instrument.** `clip_convert/diag_impact_ipadapter_traj.py`
**Measurement record.** `impact/impact_ipadapter_sd15_plus_traj_v2.json` (25 steps, seed 42, frame `waiIllustriousSDXL_v170` equivalent host `againmix_v22`, CLIP-ViT-H-14)

---

## 2. Measured impact

| number | measured |
| :--- | :--- |
| 54 | Linear layers measured (22 `image_proj` + 32 `ip_adapter`) |
| 0 | unmeasured (zero) values |
| 3.838875e-03 | largest drift — `ip_adapter.11.to_v_ip` |
| 4.571676e-05 | largest `image_proj` drift — `image_proj.proj_in` |
| 2.134478e-04 | `image_proj.proj_out` |
| 2.003737e-04 | `image_proj.proj_in` |
| 5.786854e-06 | smallest drift — `ip_adapter.19.to_k_ip` |
| 0 | layers above the 0.15 threshold |

No layer reaches the threshold, so the protected set is empty and all 54 measured Linears are INT8. The top of the ranking is not where a stage-output measurement puts it: measuring the adapter's local output ranked `proj_in` at 6.32e-04 and `proj_out` at 3.94e-05, while the trajectory puts `ip_adapter.11.to_v_ip` first and four `image_proj` layers in the top six.

---

## 3. Conversion

`clip_convert/convert_ipadapter_convrot_int8.py --drift-threshold 0.15` (Hadamard-256 rotation, row-wise `amax/127` int8, `comfy_quant` = `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}`)

| number | measured |
| :--- | :--- |
| 54 | ConvRot INT8 Linear |
| 0 | protected (drift or shape) |
| 29 | kept as-is (not a measured Linear: biases, norms, latents) |
| 191 | tensors in the pack |
| 49,366,186 B | pack size |

A pack rebuilt from the corrected ranking is **tensor-identical** to the published pack: 191/191 tensors equal, same byte size. The corrected measurement changed the basis of the statement, not the artifact.

---

## 4. Acceptance

| number | measured |
| :--- | :--- |
| 22 | `image_proj` Linear carried as INT8 |
| 32 | `ip_adapter` Linear carried as INT8 |
| 54 | declared total, skipped 0 |
| 30.54 dB | PSNR against the float adapter, same seed |
| 512×512, 8 steps, cfg 4.0, seed 42 | render settings |
| 408,703 B | PNG (`sha256 5a390ef10d914d8a97b3ee510f542a3902f4e6294c298a34886dbe0edc9b4df6`) |

The 54 INT8 Linears are the pack executing: `image_proj_int8 29.9 MB` and
`ip_adapter_int8 19.2 MB` inside the adapter, no layer skipped.
Record: `records/cn_ipadapter_sd15_plus_a1111_image_test.json`.

---

## 5. Protection correction

`records/cn_ipadapter_sd15_plus_protection.json` carried `image_proj_drift_max: 0.0` in its 2026-11-13 measurement. That value came from measuring `image_proj` with an in-place weight swap, which cannot reach the frozen embeds. The corrected measurement is 2.579049e-04 (`image_proj.layers.1.0.to_out`), recorded in `records/cn_ipadapter_sd15_plus_protection_correction.json`. The earlier file is kept unchanged.

---

## 6. Backdoor audit

| step | record | result |
| :--- | :--- | :--- |
| pack audit (quantisation replay) | `records/cn_ipadapter_sd15_plus_pack_audit.json` | FAITHFUL, 54 declared / 54 found, 0 integers differing |
| source vs pack output | `records/cn_ipadapter_sd15_plus_pack_source_output_diff.json` | `image_proj` cosine 0.9997854232788086, `to_kv` cosine 0.9998675584793091, n=4 |
| image-prompt trigger search | `records/cn_ipadapter_sd15_plus_image_trigger.json` | 55 candidates, control p99 1.374969e-04, flagged 0 |.7881393432617188e-07, control p99 to_kv 8.940696716308594e-08 |
| image trigger search | `records/cn_ipadapter_sd15_plus_image_trigger.json` | rank-1 implant at `ip_adapter.11.to_k_ip`; control 1−cos 1.1920928955078125e-07 over 3 repeats |
| end-to-end render | `records/cn_ipadapter_sd15_plus_a1111_image_test.json` | 54 INT8 applied, PSNR 30.54 dB, PNG 344,431 B |

The probe sensitivity is bounded by those control values: a divergence below ~1.8e-07 (embeddings) / ~9e-08 (k/v) is inside the adapter's own repeat-run noise and cannot be called a signal.

---

## 7. Records

- `records/cn_ipadapter_sd15_plus_pack_audit.json`
- `records/cn_ipadapter_sd15_plus_protection.json` (see §5 for the corrected field)
- `records/cn_ipadapter_sd15_plus_protection_correction.json`
- `records/cn_ipadapter_sd15_plus_gate_256.json` (see §4, unquantised path)
- `records/cn_ipadapter_sd15_plus_a1111_image_test.json`
- `records/cn_ipadapter_sd15_plus_image_trigger.json`
- `records/ipadapter_sd15_plus_weight_trigger.json`
- `records/cn_ipadapter_sd15_plus_pack_source_output_diff.json`
- `records/cn_ipadapter_sd15_plus_hswq_route.json`

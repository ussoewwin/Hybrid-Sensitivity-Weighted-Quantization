# SD1.5 IP-Adapter Plus - ConvRot INT8 Impact Technical Guide

> **The method this pack was built under** (the injection frame, the behavioural observable, the
> size/quality balance, FP16 kept tensors) is stated once in
> [`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md) - Section
> *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this checkpoint.

**Checkpoint.** `models/controlnet/ip-adapter_sd15_plus.pth` (158,030,471 B, 191 tensors)
**Pack.** `ip-adapter_sd15_plus_hswq_convrot_int8.safetensors` (49,366,186 B, **-68.8 %**)
**SHA256 (source).** `A0DB87557825AC2F8888EF06EC64C85F175E8BA5467BAA2722ADE3B4A9FEB9F5`
**SHA256 (pack).** `9A0F7CDFAE2575951C3324458FD67221B6240E7C5C8FFD27476BB009AA7F6CD9`

---

## Table of contents

1. [What the score is](#1-what-the-score-is)
2. [The loader requirement](#2-the-loader-requirement)
3. [The model](#3-the-model)
4. [The impact measurement](#4-the-impact-measurement)
5. [The candidates, and the adopted set](#5-the-candidates-and-the-adopted-set)
6. [The pack](#6-the-pack)
7. [The acceptance](#7-the-acceptance)
8. [Reproduction](#8-reproduction)
9. [Error sources and limits](#9-error-sources-and-limits)
10. [The experiment ledger](#10-the-experiment-ledger)
11. [Files](#11-files)

---

## 1. What the score is

The score is **not** a static weight statistic. It is the **observed change of the model's own
output** when one weight group is replaced by its ConvRot INT8 reconstruction.

**This adapter has two parts, and they cannot share one measurement loop.** The runtime
(`comfyui_ipadapter_plus`) builds both from the same file, but executes them differently:

| part | what the runtime does | how a layer's impact can be measured |
| :--- | :--- | :--- |
| `image_proj.*` (22 Linear) | computes the Resampler output **once** inside `ipadapter_execute` and freezes it into `patch_kwargs["cond"]` / `["uncond"]` | swap one weight, run the Resampler, compare its queries |
| `ip_adapter.*` (32 Linear) | one bias-free Linear per `to_k_ip` / `to_v_ip`, executed on **every** forward | swap one weight and run the UNet trajectory |

**The observable** for the Resampler side is the token stack the cross-attention consumes. For the
`ip_adapter` side it is the UNet's own trajectory with the adapter applied. Both are outputs, not
weight norms.

**The injection frame matters.** The pack stores the **rotated** weight and the kernel rotates the
**activation**; the error that reaches the output is therefore `(q*s) @ H - W`, the **original**
frame.

**The acceptance gate.** A pack passes when `cos.mean >= 0.99` **and** `cos.min >= 0.98` over the
fixed 256-item set (seed 42), with finite output.

---

## 2. The loader requirement

The pack's weights are `int8_tensorwise` with `convrot` stamps and a group size of 256. A loader
that reads the file without interpreting `comfy_quant` hands the int8 code tensors to float modules.

**Measured on this machine:**

| loader | what it does with this pack | measured |
| :--- | :--- | :--- |
| A1111 | `modules/ipadapter_int8_convrot.py` (12,449 B), IP-Adapter-only: strips `*.weight_scale` / `*.comfy_quant`, installs the int8 payload as an instance-level forward bound to `comfy_kitchen.int8_linear(convrot=True, groupsize)` | **54 of 54 layers installed** (22 `image_proj` + 32 `ip_adapter`), 83 float parameters left untouched |
| ComfyUI `comfyui_ipadapter_plus` | builds `Resampler` + `To_KV` and reads the pack for the dequantised comparisons below | used for sections 7 and 4 |
| Forge | `backend/operations_mixed_precision.py` reads `comfy_quant` at L75-76 and dispatches to the same kernel | **not separately measured for this pack** |

The route census is `records/cn_ipadapter_sd15_plus_hswq_route.json`.

**One measured gap.** A1111's forward for a **Plus** adapter calls
`annotator.clipvision` for the unconditional embedding, and this extension build does not ship that
module (measured: no `clipvision.py` under `extensions-builtin/sd-webui-controlnet/annotator`). The
probe therefore stops after installation for this pack; the render in section 7 is the evidence that
the installed layers execute.

---

## 3. The model

| item | measured |
| :--- | :--- |
| the source | 158,030,471 B, 191 tensors |
| the measured Linears | **54** (22 `image_proj` + 32 `ip_adapter`) |
| the `image_proj` part | the Resampler: `proj_in` 768x1280, `proj_out` 768x768, `latents` 1x16x768, 4 blocks |
| the `ip_adapter` part | 32 bias-free Linears: `to_k_ip` / `to_v_ip` at 320 / 640 / 1280 wide |
| the group size | 256, all INT8 layers |
| the tensors kept as-is | 29 (biases, norms, `latents` - not measured Linears) |

---

## 4. The impact measurement

`clip_convert/diag_impact_ipadapter_sd15_plus.py` replays one layer at a time in ConvRot INT8, on
both instruments in one process.

| rank | layer | drift (rel_mse) |
| :--- | :--- | ---: |
| 1 | `ip_adapter.11.to_v_ip` | 3.838875e-03 |
| 2 | `ip_adapter.17.to_v_ip` | 2.230858e-03 |
| 3 | `ip_adapter.29.to_k_ip` | 1.994593e-03 |
| 4 | `ip_adapter.3.to_k_ip` | 1.753917e-03 |
| 5 | `ip_adapter.31.to_v_ip` | 1.719548e-03 |
| 6 | `ip_adapter.11.to_k_ip` | 1.641148e-03 |
| 7 | `ip_adapter.15.to_v_ip` | 1.581428e-03 |
| 8 | `ip_adapter.9.to_k_ip` | 1.540621e-03 |
| - | `image_proj.proj_in` | 4.571676e-05 |
| - | `image_proj.proj_out` | 4.082918e-05 |

| summary | measured |
| :--- | :--- |
| Linear layers measured | **54** |
| unmeasured values | **0** |
| largest drift | **3.838875e-03** (`ip_adapter.11.to_v_ip`) |
| median | 4.206448e-04 |
| p90 | 1.641148e-03 |
| smallest | 1.430511e-06 |
| layers above 0.15 | **0** |

**The ranking is entirely `ip_adapter`; the Resampler sits two orders of magnitude lower.** That is
structural: the Resampler runs once and its output is frozen, while the `to_k_ip` / `to_v_ip` Linears
run on every forward. No layer reaches the threshold.

---

## 5. The candidates, and the adopted set

| item | measured |
| :--- | :--- |
| Linears measured | 54 |
| above 0.15 | **0** |
| INT8 quantised | **54** |
| FP16 kept | **0** |
| kept by the shape rule | **0** |
| tensors kept that are not measured Linears | 29 |

**Nothing is protected, and nothing needed to be.** Every one of the 54 measured Linears is two
orders of magnitude below the threshold, and every one carries a 256-way group (the inner dimensions
are 1280, 768, 640, 320 and 128 - 1280 / 768 / 640 / 320 all hold 256, and the 128-wide LoRA-adjacent
weights are not part of this adapter). The 29 kept tensors are biases, norms and the `latents`
buffer, which are not quantiser inputs.

This differs from the FaceID adapter in the same family, where 60 of 162 weight groups are kept
because their inner dimension is 128 / 320 / 640 and cannot carry a power-of-four group. Here the
`image_proj` part is a Resampler whose widths are 1280 and 768.

---

## 6. The pack

| item | measured |
| :--- | :--- |
| the file | `ip-adapter_sd15_plus_hswq_convrot_int8.safetensors` |
| the size | 49,366,186 B (47.08 MiB) |
| against the source | 158,030,471 B - **-68.8 %** |
| the tensors | 191 |
| the INT8 layers | 54 |
| the FP16 kept layers | 0 |
| the group size | 256 |
| the verdict | FAITHFUL |

| check | measured |
| :--- | :--- |
| declared / found | **54 / 54** |
| missing, extra, shape mismatch | **0 / 0 / 0** |
| integers differing from the replay | **0** |
| largest integer delta | **0** |
| scale exact relative max | **0.0** |
| non-finite scales | 0 |
| float tensors checked | **29** |
| float mismatches | **0** |

A pack rebuilt from the corrected ranking is **tensor-identical** to the published one: 191 of 191
tensors equal, same byte size.

---

## 7. The acceptance

**The adapter's own outputs, source against pack.** Four batches, seed 42, both sides reading the
same input:

| batch | `image_proj` cosine | `to_kv` cosine | max abs error | rel L2 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.999785423 | 0.999867558 | 0.701803207 | 0.016275547 |
| 1 | 0.999749959 | 0.999848127 | 0.892910242 | 0.017430140 |
| 2 | 0.999341667 | 0.999635756 | 1.810509443 | 0.026994891 |
| 3 | 0.999706328 | 0.999829888 | 0.671718478 | 0.018468622 |

**The 256-item gate.**

| item | measured |
| :--- | :--- |
| cos mean | **0.9998862713109702** |
| cos min | **0.9989429116249084** |
| p5 | 0.9997830390930176 |
| p50 | 0.9999051094055176 |
| items | **256** |
| finite | yes |
| INT8 kernel calls during the gate | **0** |
| gate (`mean >= 0.99`, `min >= 0.98`) | **PASS** |

**The gate's zero kernel calls are stated, not hidden.** That instrument reads the pack's codes
dequantised into the source structure, so it measures the quantisation without executing the kernel.
The render below is where the kernel runs.

**The trigger set.** 55 candidate image transforms against the natural-transform control
distribution:

| item | measured |
| :--- | :--- |
| candidates | 55 (45 patch, 6 frequency, 4 natural controls) |
| control max | 9.166457e-05 |
| control p99 (max x 1.5) | **1.374969e-04** |
| candidates above p99 | **0** |
| positive control | rank-1 implants into `ip_adapter.11.to_k_ip` (Frobenius 74.842178) |
| detection floor | **0.30** relative to the weight's Frobenius norm (1-cos 7.580213e-04) |

The ladder: 0.001 / 0.003 / 0.01 / 0.03 / 0.1 all stayed inside the control distribution (largest
8.430e-05), **0.30 resolved** at 7.580213e-04 against the 1.374969e-04 threshold, and 1.00 reached
8.330089e-03. The instrument is armed at 30 % of the layer's norm, and no candidate reaches the
control distribution's edge.

**The render, on A1111.** 512x512, 8 steps, cfg 4.0, DPM++ 2M, seed 42, weight 1.0, preprocessor
`ip-adapter_clip_sd15_plus`, base `againmix_v22`, image prompt `00000-2109644404.png`:

| arm | PNG | sha256 |
| :--- | ---: | :--- |
| base (no adapter) | 389,551 B | `63D759881E9647D1DDB47940B101ADF09C015ABAB932D8A7C3E3782E55A99522` |
| float adapter | 343,354 B | `46445CA42982ECD5AB6953FD20A61953133E7630ED110403875616DE53A65A4D` |
| INT8 pack | 344,431 B | `7F018922D7D267C945DF5317B6DB34B52CEA1DB8B05FB12CE0BD9652484D6C05` |

| comparison | mean abs | PSNR |
| :--- | ---: | ---: |
| float against base | 61.463739 | 10.21 dB |
| INT8 against base | 61.886559 | 10.17 dB |
| **INT8 against float** | **3.987743** | **30.54 dB** |

**The two adapter arms sit 61 units from base and 3.99 units from each other.** The first number is
the adapter doing its job (it moves the image a long way), the second is the quantisation; reading
only the 30.54 dB hides that the pair is 61 away from base, so both are shown.

---

## 8. Reproduction

```
#   python = <A1111>/venv/Scripts/python.exe  or  <ComfyUI>/python_embeded/python.exe
#   COMFY  = <ComfyUI>

# 1. the impact sweep (54 Linears, seed 42)
python clip_convert/diag_impact_ipadapter_sd15_plus.py \
    --ipadapter <src>/ip-adapter_sd15_plus.pth \
    --out       impact/impact_ipadapter_sd15_plus_v6.json --seed 42

# 2. the builder
python clip_convert/convert_ipadapter_convrot_int8.py \
    --input  <src>/ip-adapter_sd15_plus.pth \
    --impact impact/impact_ipadapter_sd15_plus_v6.json \
    --output <out>/ip-adapter_sd15_plus_hswq_convrot_int8.safetensors \
    --drift-threshold 0.15

# 3. the pack audit (quantisation replay)
python audit/ipadapter_pack_audit.py \
    --source <src>/ip-adapter_sd15_plus.pth --pack <out>/<pack>.safetensors \
    --json records/cn_ipadapter_sd15_plus_pack_audit.json

# 4. the route census
python audit/ipadapter_faceid_route_probe.py \
    --pack <out>/<pack>.safetensors --root <A1111> --which a1111 \
    --json records/cn_ipadapter_sd15_plus_hswq_route.json

# 5. the render (A1111 API, base / float / int8)
python audit/a1111_ipadapter_render.py \
    --pack <out>/<pack>.safetensors --source <src>/ip-adapter_sd15_plus.pth \
    --out-dir <dir> --json records/cn_ipadapter_sd15_plus_a1111_image_test.json
```

---

## 9. Error sources and limits

- **The 256-item gate runs with zero kernel calls.** It measures the pack's codes dequantised into
  the source structure. The render is the only instrument here that executes the INT8 kernel.
- **A1111's Plus forward was not run in the route probe.** The extension build lacks
  `annotator.clipvision`, which `_get_image_embeds` imports for the unconditional embedding. The
  installation of all 54 layers is measured; the forward is evidenced by the render, not by the
  probe.
- **The render is 8 steps at cfg 4.0 on one prompt and one seed.** It is not the 12-step protocol used
  for the FaceID pair, so the two guides' render numbers are not comparable.
- **The render's 30.54 dB is between the two adapter arms, not against base.** Against base both
  adapters sit near 10 dB, which is the adapter's own effect.
- **The trigger search is one image and one seed.** The floor is 30 % of the layer's norm, which is
  coarse; a smaller implant would not be seen by this instrument.
- **Forge was not separately measured for this pack.** The route record says so rather than implying
  coverage.
- **The impact sweep is one process, one seed.** With every layer far below 0.15, a different seed
  cannot plausibly cross the threshold, but the ordering inside the top decile is not stable.

---

## 10. The experiment ledger

| step | tool | result |
| :--- | :--- | :--- |
| 1 | `diag_impact_ipadapter_sd15_plus.py` | 54 Linears, max 3.838875e-03, median 4.206448e-04, p90 1.641148e-03, 0 above 0.15, 0 unmeasured |
| 2 | `convert_ipadapter_convrot_int8.py` | 54 INT8, 0 kept by drift or shape, 29 non-quantiser tensors, 49,366,186 B, -68.8 %, tensor-identical rebuild |
| 3 | `ipadapter_pack_audit.py` | FAITHFUL, 54 declared / 54 found, 0 integers differing, scale 0.0, float 29 checked / 0 mismatched |
| 4 | `ipadapter_faceid_route_probe.py` | A1111: 54/54 layers installed (22 + 32), 83 float parameters; Plus forward not run (no `annotator.clipvision` in this build) |
| 5 | `a1111_ipadapter_render.py` | base / float / INT8 arms rendered; INT8 against float 3.987743 mean abs, 30.54 dB; both arms ~61 from base |
| 6 | `ipadapter_pack_source_output_diff.py` | 4 batches: `image_proj` cos 0.999342-0.999785, `to_kv` cos 0.999636-0.999868 |
| 7 | `ipadapter_image_trigger.py` | 55 candidates, control p99 1.374969e-04, flagged 0, floor 0.30 of the layer's norm |
| 8 | `verify_cn_flux_256.py --family ipadapter` | gate mean 0.9998862713, min 0.9989429116, n 256, 0 kernel calls, PASS |

---

## 11. Files

| file | what it is |
| :--- | :--- |
| `records/cn_ipadapter_sd15_plus_pack_audit.json` | the quantisation replay (54 declared, 54 found, 0 differing) |
| `records/cn_ipadapter_sd15_plus_protection.json` | the 54-layer sweep, plus the superseded 2026-11-13 measurement |
| `records/cn_ipadapter_sd15_plus_protection_correction.json` | why the earlier `image_proj_drift_max: 0.0` was unmeasured |
| `records/cn_ipadapter_sd15_plus_pack_source_output_diff.json` | the four batches of the observable comparison |
| `records/cn_ipadapter_sd15_plus_gate_256.json` | the 256-item gate (0 kernel calls) |
| `records/cn_ipadapter_sd15_plus_image_trigger.json` | the 55 candidates and the positive-control ladder |
| `records/cn_ipadapter_sd15_plus_a1111_image_test.json` | the three render arms and their comparisons |
| `records/cn_ipadapter_sd15_plus_hswq_route.json` | which loaders take the pack, and where the forward stops |

### Retraction

The version of this guide dated 2026-11-14 is withdrawn. It stated a PNG of 408,703 B with sha256
`5a390ef1...` in its acceptance section, which matches no render in
`records/cn_ipadapter_sd15_plus_a1111_image_test.json` (the INT8 arm is 344,431 B,
`7f018922...`). It listed `records/cn_ipadapter_sd15_plus_hswq_route.json` among its records when
that file did not exist. Its backdoor section carried a mangled table with a stray
`.78813934326` and two different control figures in one cell. It reported the 30.54 dB between the
two adapter arms without the 61-unit distance both arms sit from base. Every number above is read
from the records named beside it.

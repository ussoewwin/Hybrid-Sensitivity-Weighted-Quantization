# InstantID IP-Adapter (SDXL) - ConvRot INT8 Trajectory-Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`How to quantize Text Encoder and ControlNet.md`](How%20to%20quantize%20Text%20Encoder%20and%20ControlNet.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this checkpoint.

**Checkpoint.** `models/controlnet/ip-adapter_instant_id_sdxl.safetensors` (2,502,139,136 B, 844 tensors, 1,251,014,160 parameters)
**Pack.** `models/controlnet/ip-adapter_instant_id_sdxl_hswq_convrot_int8.safetensors` (1,420,211,083 B, **-43.2 %**)
**SHA256 (source).** `C8127BE9F174101EBDAFEE9964D856B49B634435CF6DAA396D3F593CF0BBBB05`
**SHA256 (pack).** `A9FFBD849D007B969D91CADD1409757000F20F69D08B45D95F18243ADFF456C9`

The file is distributed by InstantID as `ip-adapter_instant_id_sdxl`, and by the
diffusers conversion as `instantid_diffusion_pytorch_model`. Both names carry the
same bytes, so both names carry the same measurement: the records below are keyed
`cn_instantid_diffusers_*` because the diffusers name is the one the source was
taken under.

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

The score is **not** a static weight statistic (kurtosis, absmax, outlier ratio, SVD).
It is the **observed change of the model's own output** when one layer is replaced
by its ConvRot INT8 reconstruction - a behavioural measurement, taken through the
frame the runtime uses.

**The observable.** The ControlNet's own output tensors, for a fixed input: the ten
feature maps the control branch emits (`[1,1280,16,16]`, `[1,320,64,64]` x3,
`[1,320,32,32]`, `[1,640,32,32]` x2, `[1,640,16,16]`, `[1,1280,16,16]` x2). The
score of a layer is the change in those outputs when that one layer alone is
replaced, measured as a relative MSE against the FP16 run of the same input.

**The injection frame matters.** The pack stores the **rotated** weight and the
kernel rotates the **activation**; the error that reaches the output is therefore
`(q*s) @ H - W`, the **original** frame. Injecting in the rotated frame would
measure the rotation itself, not the error.

**The ranking metric is the trajectory.** The sweep runs the full sampling
trajectory (400 modules, 20 steps, seed 42, cfg 7.0) and records, per module, the
relative MSE of the model output. A layer whose own trajectory drift exceeds
**0.15** is a candidate for the FP16-kept set; the rest are quantised.

**The acceptance gate.** A pack passes when `cos.mean >= 0.99` **and**
`cos.min >= 0.98` over the fixed 256-item set (seed 42), with a finite output.

---

## 2. The loader requirement

The pack's weights are `int8_tensorwise` with `convrot` stamps and a group size of
256. ComfyUI's stock `ControlNetLoader` builds the branch with `manual_cast`, which
would read the INT8 tensors as plain weights. The pack must be loaded through
**`HSWQ ControlNet Loader`** (`custom_nodes/ComfyUI-HSWQ-Loader-and-Tools`), which
detects the stamps and builds the branch with `mixed_precision_ops`: weights stay
INT8 in memory and the forward uses the comfy-kitchen `int8_linear` kernel with
online ConvRot rotation.

The route record states what actually happened on that path: **329 declared, 329
applied, 0 raw code layers, 0 declared-but-not-applied**.

---

## 3. The model

| item | measured |
| :--- | :--- |
| the source | 2,502,139,136 B, 844 tensors, 1,251,014,160 parameters |
| the runtime tensors | 844 (pack: 844; 1,502 with stamps) |
| the runtime Linear modules | 362 |
| the 2D weights quantised | **329** |
| the layer families | `input_blocks.*`, `middle_block.*`, `zero_convs.*`, `input_hint_block.*` |
| the group size | 256 (all 329) |
| the format | `int8_tensorwise`, ConvRot on |

Coverage against the runtime names: **0 missing from pack, 0 extra in pack, 0 shape
mismatch**. The pack is the source's graph, with 329 of its 362 Linear modules in
INT8.

---

## 4. The impact measurement

`clip_convert/convert_controlnet_convrot_int8.py` performs the sweep and writes the
ranking; the converter records the ranking file it consumed
(`impact_sha256 8435b913d6120474c8b030e53ebb4be887c8b04f33f32ef6873623285a2b860b`,
30,485 B), so the ranking behind this pack is identifiable byte for byte.

| rank | layer | the trajectory drift (rel_mse) |
| :--- | :--- | :--- |
| 1 | `input_blocks.1.0.in_layers.2` | 1.0576799556263723 |
| 2 | `input_blocks.1.0.out_layers.3` | 0.2842686194286216 |
| 3 | `input_blocks.4.0.out_layers.3` | 0.2715651761929621 |
| 4 | `input_blocks.6.0.op` | 0.24417803102951438 |
| 5 | `input_blocks.5.0.out_layers.3` | 0.22763944428879768 |
| 6 | `input_blocks.0.0` | 0.21290796620305627 |
| 7 | `zero_convs.7.0` | 0.17716117234522244 |
| 8 | `input_blocks.4.0.in_layers.2` | 0.1713452453685022 |
| 9 | `input_blocks.3.0.op` | 0.16427800807723542 |
| 10 | `zero_convs.5.0` | 0.15781162121206763 |
| 11 | `input_blocks.4.0.skip_connection` | 0.15476940218068194 |
| 12 | `input_blocks.2.0.out_layers.3` | 0.1547083505007322 |

**`input_blocks.1.0.in_layers.2` is 3.7x above the runner-up** (1.0577 against
0.2843) and 6.8x above the twelfth. The gap is structural: it is the first
convolution of the block that turns the control hint into the resolution level the
rest of the branch consumes, so every later block inherits its error. The twelve
listed are exactly the layers above 0.15; the sweep found **12** such modules.

---

## 5. The candidates, and the adopted set

The FP16-kept set is this checkpoint's own measurement plus the shape rule, not a
set copied from another checkpoint.

| item | measured |
| :--- | :--- |
| modules swept | 400 |
| modules above 0.15 | **12** |
| FP16 kept, total | **33** |
| INT8 quantised | **329** |
| kept for a reason other than drift | 21 (shape rule: not a runtime 2D Linear, or a 1D norm) |
| kept tensors that are not runtime Linears | 482 |

The 33 kept layers are: the twelve above threshold, plus twenty-one that the shape
rule keeps out of the quantiser. Nothing else was kept. **No layer below 0.15 was
kept for quality reasons** - the threshold, not taste, decided the boundary.

---

## 6. The pack

| item | measured |
| :--- | :--- |
| the file | `ip-adapter_instant_id_sdxl_hswq_convrot_int8.safetensors` |
| the size | 1,420,211,083 B (1354.4 MB) |
| against the source | 2,502,139,136 B - **-43.2 %** |
| the tensors | 844 (1,502 with stamps) |
| the INT8 layers | 329 |
| the FP16 kept layers | 33 |
| the group size | 256 |
| the verdict | FAITHFUL |

The audit replays the converter's own arithmetic (Hadamard-256 rotation, row-wise
`amax/127`) on the runtime float weights and compares element by element:

| check | measured |
| :--- | :--- |
| INT8 layers declared / found | 329 / 329 |
| integers differing from the replay | **0** |
| largest integer delta | 0 |
| scale exact relative max | 0.0 |
| non-finite scales | 0 |
| float tensors checked | 515 |
| float mismatches | **0**, max abs diff 0.0 |

The two graph-level comparisons agree with the byte-level one: source and pack
carry the same parameter count (1,251,014,160), the same names, and isomorphic
graphs (0 parameters only in source, 0 only in pack).

---

## 7. The acceptance

**Through the real loader.** `clip_convert/verify_cn_flux_256.py` on the fixed
256-item set (seed 42), reading the pack through the HSWQ loader path, INT8 kernel
live: **329 INT8 Linear calls per item**.

| item | measured |
| :--- | :--- |
| cos mean (256 items) | **0.999395** |
| cos min | **0.997491** |
| cos p5 | 0.998739 |
| cos p50 | 0.999473 |
| finite | yes |
| gate (`mean >= 0.99`, `min >= 0.98`) | **PASS** |

**The output comparison.** Same input, source(float) against pack(INT8), four
batches: `cos_min 0.9986351132392883`, `cos_mean 0.9991237372159958`,
`max_abs_err 4.0`, `rel_l2_max 0.05222431197762489`. The `max_abs_err` of 4.0 sits
on a zero-conv output whose own range is in the thousands; the relative L2 of
0.052 is the honest summary, and it is stated here rather than smoothed away.

**The render.** 1024x1024, 12 steps, cfg 6.0, dpmpp_2m karras, seed 42, control
strength 0.9, on `waiIllustriousSDXL_v170`:

| item | measured |
| :--- | :--- |
| latent | `[1,4,128,128]`, finite, NaN 0, Inf 0 |
| INT8 Linear calls | 3,948 |
| PNG | 1,903,787 B, sha256 `a36122c47ef5f30cc86d71683fee95393689ee2a2680cc5bdd50c057ba3414c3` |
| wall clock | 24.5 s |

The image was inspected, not inferred from the numbers: it is a rendered scene, not
noise. A high cosine alone would not have established that.

---

## 8. Reproduction

```
# the environment
#   python  = <ComfyUI>/python_embeded/python.exe
#   COMFY   = <ComfyUI>  (the checkout ComfyUI runs from)

# 1. the trajectory sweep and the builder (drift threshold 0.15, base waiIllustriousSDXL_v170)
python clip_convert/convert_controlnet_convrot_int8.py \
    --input   <src>/ip-adapter_instant_id_sdxl.safetensors \
    --output  <out>/ip-adapter_instant_id_sdxl_hswq_convrot_int8.safetensors \
    --model   waiIllustriousSDXL_v170.safetensors \
    --drift-threshold 0.15

# 2. the pack audit (FAITHFUL: the converter's arithmetic replayed element by element)
python audit/controlnet_pack_audit.py --family controlnet \
    --source <src>/ip-adapter_instant_id_sdxl.safetensors \
    --pack   <out>/ip-adapter_instant_id_sdxl_hswq_convrot_int8.safetensors

# 3. source against pack, same input
python audit/controlnet_pack_source_output_diff.py \
    --source <src>/ip-adapter_instant_id_sdxl.safetensors \
    --pack   <out>/ip-adapter_instant_id_sdxl_hswq_convrot_int8.safetensors

# 4. the route: what the loader actually applies
python audit/controlnet_hswq_route.py \
    --pack <out>/ip-adapter_instant_id_sdxl_hswq_convrot_int8.safetensors

# 5. the execution: the INT8 kernel, live
python audit/controlnet_int8_exec.py \
    --pack <out>/ip-adapter_instant_id_sdxl_hswq_convrot_int8.safetensors

# 6. the acceptance (256 items, seed 42)
python clip_convert/verify_cn_flux_256.py \
    <src>/ip-adapter_instant_id_sdxl.safetensors \
    <out>/ip-adapter_instant_id_sdxl_hswq_convrot_int8.safetensors

# 7. the render
python audit/controlnet_image_test.py \
    --pack <out>/ip-adapter_instant_id_sdxl_hswq_convrot_int8.safetensors \
    --base waiIllustriousSDXL_v170.safetensors
```

---

## 9. Error sources and limits

- **The sweep is one seed.** The ranking comes from a single trajectory
  (seed 42, cfg 7.0, 20 steps). A different seed can move a layer across the 0.15
  boundary; the ordering of the top few is stable, the ordering near the threshold
  is not. The threshold is a policy, not a law.
- **The kept set is not minimal.** 33 layers are kept; 12 are kept for measured
  drift, 21 by the shape rule. A different shape rule would change the count, so
  the size/quality trade of this pack is bounded by that rule, not by the sweep
  alone.
- **The gate is a 256-item fixed set.** It bounds the pack's deviation on that set.
  It is not a proof of quality on arbitrary prompts, and it is not a backdoor test.
- **The render is one prompt, one seed, one base model.** It establishes that the
  pack renders; it does not establish the prompt-to-prompt behaviour.
- **`max_abs_err 4.0` is real.** On the zero-conv outputs the absolute error of a
  few units appears against ranges in the thousands. The relative L2 (0.052) is the
  meaningful number, and both are reported.
- **The InstantID identity path is not covered here.** This pack quantises the
  IP-Adapter branch. The identity encoder and the face embedding model are separate
  files and remain FP16; their effect on likeness is outside this measurement.

---

## 10. The experiment ledger

| step | tool | result |
| :--- | :--- | :--- |
| 1 | `convert_controlnet_convrot_int8.py` | sweep 400 modules, 20 steps, seed 42, cfg 7.0; 12 above 0.15; 329 INT8, 33 kept; output 1.32 GB |
| 2 | `controlnet_pack_audit.py` | FAITHFUL; 329/329; q_mismatch 0; float 515/515; 844 tensors; 15.71 s |
| 3 | `controlnet_pack_source_output_diff.py` | cos_min 0.9986351132392883, cos_mean 0.9991237372159958, rel_l2_max 0.0522 |
| 4 | `controlnet_hswq_route.py` | declared 329, applied 329, raw 0, float 195, verdict applies |
| 5 | `controlnet_int8_exec.py` | 329 INT8 calls, cuda, bf16, 10 outputs, NaN 0, Inf 0 |
| 6 | `verify_cn_flux_256.py` | mean 0.999395, min 0.997491, p5 0.998739, p50 0.999473, PASS |
| 7 | `controlnet_image_test.py` | 1024x1024, 12 steps, seed 42, 3,948 INT8 calls, PNG 1,903,787 B |
| 8 | `controlnet_image_trigger.py` | trigger set recorded; see the record for the per-item table |

---

## 11. Files

| file | what it is |
| :--- | :--- |
| `md/backdoor/verdicts/records/cn_instantid_diffusers_pack_audit.json` | FAITHFUL audit, 329/329, q_mismatch 0 |
| `md/backdoor/verdicts/records/cn_instantid_diffusers_protection.json` | the sweep, the 12 above 0.15, the 33 kept, the ranking file's sha |
| `md/backdoor/verdicts/records/cn_instantid_diffusers_pack_source_output_diff.json` | source against pack, 4 batches |
| `md/backdoor/verdicts/records/cn_instantid_diffusers_gate_256.json` | the 256-item acceptance |
| `md/backdoor/verdicts/records/instantid_diffusers_hswq_route.json` | the loader route |
| `md/backdoor/verdicts/records/instantid_diffusers_int8_exec.json` | the INT8 execution |
| `md/backdoor/verdicts/records/cn_instantid_diffusers_image_test.json` | the render |
| `md/backdoor/verdicts/records/cn_instantid_diffusers_image_test.png` | the rendered image |
| `md/backdoor/verdicts/records/cn_instantid_diffusers_image_trigger.json` | the trigger set |
| `A9FFBD849D007B969D91CADD1409757000F20F69D08B45D95F18243ADFF456C9` | the pack's sha256 |

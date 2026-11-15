# SDXL IP-Adapter (basic) ConvRot INT8 - Quantization and Score Report

**Checkpoint.** `models/controlnet/ip-adapter_xl.pth` (702,585,021 B, FP16, 144 tensors)
**Pack.** `models/controlnet/ip-adapter_xl_hswq_convrot_int8.safetensors` (352,063,023 B, **-49.9 %**)
**SHA256 (source).** `C2E9F1ED4FF231B9F78DE8158D63AC4B031E599D26D524E9171CCB9C16305FC3`
**SHA256 (pack).** `727A7D265F29EF94A8E51F0184415D556130807DA7CD9B00803EDF57F97C3923`

**Composition.** `image_proj` quantised (1 ConvRot-eligible Linear) · `ip_adapter` quantised (140) · **nothing kept FP16 for quality** · 3 tensors untouched (`norm.weight`, `norm.bias`, `proj.bias`).

---

## 0. The load path is the whole story

Two separate facts decided how this pack can be used, and both were measured rather than assumed.

### 0.1 The ComfyUI IP-Adapter Plus loader does not interpret `comfy_quant`

`comfyui_ipadapter_plus.utils.ipadapter_model_loader` reads the file with
`comfy.utils.load_torch_file(safe_load=True)`, rebuilds the nested `image_proj` / `ip_adapter`
dictionary from the flat key names, and hands the tensors straight to the modules. It never looks at
`comfy_quant`. Measured on this pack: **140 of 140 declared `ip_adapter` weights arrive as raw int8
codes, 0 layers quantised-backed**. A FaceID adapter fails loudly at this point
(`MLPProjModelFaceId.load_state_dict` raises); the basic XL layout happens not to raise, so the
danger here is the opposite - the pack would load and run with int8 codes sitting in float
parameters. **Do not load this pack through that path.**

### 0.2 A1111 has an IP-Adapter-only ConvRot INT8 runtime, and it takes this pack

`modules/ipadapter_int8_convrot.py` (12,449 B) strips the sidecar keys (`*.weight_scale`,
`*.comfy_quant`) before the float load, installs the int8 payload as an instance-level forward bound
to `comfy_kitchen.int8_linear(convrot=True, convrot_groupsize)`, and rebuilds `To_KV` from the
declared shapes because a fully quantised `ip_adapter` sub-graph contributes no float key at all.

Measured: **141 of 141 layers installed** (1 `image_proj` + 140 `ip_adapter`), 144 float parameters
left untouched, forward `[1, 4, 2048]` finite, **2 INT8 kernel calls** for one adapter forward.

**This is the entry point for this pack.** Record: `records/cn_ipadapter_xl_hswq_route.json`.

---

## 1. What is quantized

| item | measured |
| :--- | :--- |
| the source | 702,585,021 B, FP16, 144 tensors (`image_proj` 4, `ip_adapter` 140) |
| the weight groups | **141** (1 `image_proj.proj`, 140 `to_k_ip` / `to_v_ip`) |
| the pack | **352,063,023 B** (-49.9 %, ratio 0.501097) |
| the quantized | **141** - every group |
| the FP16-kept | **0** |
| not a quantiser input | 3 (`norm.weight` 2048, `norm.bias` 2048, `proj.bias` 8192) |
| the group size | 256 |
| the inner dimensions | 1280 (`proj`), 2048 (`to_k_ip` / `to_v_ip`) - both carry 256 |

**Nothing was protected, and the shape rule did not have to protect anything.** Every measured group
is a power-of-four multiple on its inner dimension.

---

## 2. What the protection set is based on

The selector is the **aggregate contribution**: replace one group by its ConvRot INT8 reconstruction
in the original frame and measure the change in that part's own output.

- for `image_proj.proj`, the relative MSE of the **4 x 2048 token stack** after the post-projection
  LayerNorm;
- for a `to_k_ip` / `to_v_ip`, the relative MSE of that Linear's output on a fixed 2048-wide input.

Measured in float32 (the file is float16; the quantisation error has to be resolved against the
weights, not against float16 rounding), seed 42, one process, 1.0 s.

| rank | layer | the contribution |
| :--- | :--- | ---: |
| 1 | `ip_adapter.63.to_k_ip` | 7.817284e-05 |
| 2 | `ip_adapter.91.to_k_ip` | 7.633302e-05 |
| 3 | `ip_adapter.105.to_k_ip` | 7.104035e-05 |
| 4 | `ip_adapter.137.to_k_ip` | 7.060644e-05 |
| 5 | `ip_adapter.83.to_k_ip` | 6.912488e-05 |
| 6 | `ip_adapter.85.to_v_ip` | 6.774781e-05 |
| - | `image_proj.proj` | 3.094009e-05 |

| summary | measured |
| :--- | :--- |
| groups | 141 |
| measured | 141 (0 not quantisable) |
| largest | 7.817284e-05 |
| median | 5.050932e-05 |
| p90 | 6.211184e-05 |
| smallest | 2.709644e-05 |
| above the 0.15 threshold | **0** |

**The whole ranking sits three to four orders of magnitude below the threshold and the spread is
under three-fold.** There is no layer here that a quality-first selection would keep.

---

## 3. The conversion

```
python clip_convert/convert_ipadapter_convrot_int8.py \
    --input  <src>/ip-adapter_xl.pth \
    --impact impact/impact_ipadapter_xl.json \
    --output <out>/ip-adapter_xl_hswq_convrot_int8.safetensors \
    --drift-threshold 0.15
```

| argument | value |
| :--- | :--- |
| `--drift-threshold` | 0.15 |
| group size | 256 |
| rotation | Hadamard-256 on the inner dimension |
| scale | per output row, `amax/127` |
| `comfy_quant` | `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}` |
| impact record | `records/cn_ipadapter_xl_protection.json` |
| seed | 42 |

**Output:** 141 INT8, 0 protected, 3 non-quantiser tensors, 352,063,023 B.

---

## 4. Faithfulness - the pack is the source quantised

| check | measured |
| :--- | :--- |
| declared / found | **141 / 141** |
| integers differing from a replay | **0** |
| largest integer delta | **0** |
| scale exact relative max | **0.0** |
| non-finite scales | 0 |
| float tensors checked | **3** |
| float mismatches | **0** (bit-identical, max abs diff 0.0) |
| verdict | **FAITHFUL** |

Record: `records/cn_ipadapter_xl_pack_audit.json`.

---

## 5. Score - what the quantisation costs on the adapter's own output

Measured on the A1111 INT8 route, 141 layers installed, kernel counted.

| item | measured |
| :--- | :--- |
| `image_proj` token-stack cos | **0.999944299** |
| INT8 kernel calls behind it | **8** |
| batches 0 / 1 / 2 / 3 | 0.999939796 / 0.999941850 / 0.999943407 / 0.999941472 |
| 256-item cos mean | **0.999932818** |
| 256-item cos min | **0.999910579** |
| p5 / p50 | 0.999920450 / 0.999933149 |
| items | 256 |
| INT8 kernel calls over the gate | **512** |
| gate (`mean >= 0.99`, `min >= 0.98`) | **PASS** |

**The trigger search.** 55 candidate image transforms against the natural-transform control
distribution:

| item | measured |
| :--- | :--- |
| candidates | 55 (45 patch, 6 frequency, 4 natural controls) |
| control max | 7.377439e-05 |
| control p99 (max x 1.5) | **1.106616e-04** |
| candidates above p99 | **0** |
| positive control, scale axis | **did not resolve** up to x1.10 (largest 5.951007e-05) |
| positive control, code axis | **resolved at 0.1 %** of the codes shifted by 8 (1-cos 7.594861e-04) |

**The floor is on the code axis only.** A statement about "a 10 % weight change" is not available for
this instrument, because the scale ladder never crossed the threshold.

**The render.** 1024x1024, 12 steps, cfg 7.0, DPM++ 2M Karras, seed 42, weight 0.8,
`ip-adapter_clip_sdxl`, base `pornmasterPro_realismILV4_r32_r0.2`:

| arm | PNG | mean abs against the other |
| :--- | ---: | ---: |
| source `ip-adapter_xl` | 1,438,325 B | - |
| pack | 1,437,344 B | **8.76 / 255** (max 240, 87.65 % of pixels changed) |

Both were inspected: the same street scene and figure, colour normal, no artefacts.

---

## 6. What this report does not cover

- **The ComfyUI route cannot be used** (section 0.1). Every number here comes from the A1111 route.
- **Forge was not separately measured** for this pack.
- **The trigger's detection floor is on the code axis only** (section 5).
- **The render is one prompt, one seed, one checkpoint.** 8.76 / 255 says the two images differ; it
  does not say whether that matters for a use.
- **`image_proj` is one indivisible group** (8192 x 1280). The method cannot protect part of it.
- **The sweep is one seed, one fixed input per group.** The ranking is stable in spread; the order
  inside the top decile is not.

---

## 7. Files

| file | what it is |
| :--- | :--- |
| `records/cn_ipadapter_xl_protection.json` | the 141-group sweep and its ranking |
| `records/cn_ipadapter_xl_pack_audit.json` | the quantisation replay (141 declared / 141 found, 0 differing) |
| `records/cn_ipadapter_xl_hswq_route.json` | which loaders take the pack |
| `records/cn_ipadapter_xl_pack_source_output_diff.json` | the INT8-route cosine |
| `records/cn_ipadapter_xl_gate_256.json` | the 256-item gate |
| `records/cn_ipadapter_xl_image_trigger.json` | the 55 candidates and both ladders |
| `records/cn_ipadapter_xl_a1111_image_test.json` | the two render arms |

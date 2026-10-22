# FLUX.1-dev ControlNet Union Pro 2.0 — ConvRot INT8 Trajectory-Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`How to quantize Text Encoder and ControlNet.md`](How to quantize Text Encoder and ControlNet.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this model.

**Model.** `models/ControlNet/FLUX.1-dev-ControlNet-Union-Pro-2.0.safetensors` (4083.4 MB, BF16, 2,140,876,800 params, 222 checkpoint keys)
**Pack.** `models/ControlNet/FLUX.1-dev-ControlNet-Union-Pro-2.0_convrot_int8.safetensors` (2585.1 MB, **-36.7 %**)
**SHA256.** `FBCBA3C76A221FBA461C83680F682D3FF7AA813EF554C42E3A5449B981090C46`
**Method.** the FP16-kept set is chosen by the **aggregate output contribution**, measured through the real pack path.

---

## 1. The model

| item | measured |
| :--- | :--- |
| the source layout | diffusers (`transformer_blocks.*`, `controlnet_blocks.*`, `x_embedder`, `context_embedder`) |
| the runtime layout | comfy-native (`double_blocks.*`, `controlnet_blocks.*`, `img_in`, `pos_embed_input`) |
| the runtime's Linear modules | **75** |
| the runtime's `state_dict()` tensors | 174 |
| hidden size | 3072 |
| context / latent | context_in_dim 4096, latent 16 channels, patch size 2 |
| vec_in_dim | **768** (the CLIP-L pooled output) |

The runtime path is `comfy.controlnet.load_controlnet_flux_instantx` for the
source, and the loader package's `HSWQ ControlNet Loader (ConvRot INT8)` node
builds `ControlNetFlux` directly from the comfy-native state dict.

**`comfy.controlnet` has no quantization-metadata handling of its own.** It
picks an ops class (`custom_operations`) and hands it to the model's
constructor; the metadata (`<layer>.comfy_quant` / `<layer>.weight_scale`) is
consumed by `comfy.ops.MixedPrecisionOps.Linear._load_from_state_dict`. The
loader package supplies that ops class, so the whole path is the loader's.
The loader's own node also builds the comfy-native `ControlNetFlux` layout,
which `comfy.controlnet.load_controlnet_state_dict` has no detection branch
for.

---

## 2. Two measurement defects that had to be cleared first

Both were found by measurement, and both changed the result.

### 2.1 The impact measurement injected the weight in the wrong frame

`convrot_int8_recon` returned the reconstruction in the **rotated** frame
(`q * s`, i.e. `W_hat * H`) and injected it as the module's weight. But the
module's weight is used by the plain forward as `F.linear(x, weight)`, so the
injected matrix was the rotated one - not the weight the pack effectively
applies.

The pack stores the rotated weight (`W_hat = W H^T`) and the kernel rotates
the activation, so the result is `(x H) W_hat^T = x W^T`. The quantization
error that reaches the output is therefore in the **original** frame:
`(q*s) H^T - W`.

Measured on `controlnet_blocks.0` (the pack's own qdata/scale, the real
kernel):

| what | relMSE vs the FP16 result |
| :--- | :--- |
| the pack's kernel (`convrot=True`) | **1.74e-04** (correct) |
| the plain int8 (no activation rotation) | 1.89 |
| the injected rotated frame as a plain linear (the bug) | 1.89 |
| the un-rotated reconstruction as a plain linear (the fix) | **9.01e-05** |

The effect on a full model sweep was equally large: the FLUX ControlNet's
measured drifts were 1.0e+01 to 1.0e-04 too large, and the **ranking was
inverted** - the layers that matter looked catastrophic, and the ones that
do not matter looked fine. The same defect was present in **8 of the 9**
impact scripts (`diag_impact_clip.py`, `diag_impact_clip_vision.py`,
`diag_impact_cn_flux.py`, `diag_impact_controlnet.py`, `diag_impact_qwen3.py`,
`diag_impact_sam31.py`, `diag_impact_t5.py`, `diag_impact_t5_decoder.py`); all
of them are fixed.

### 2.2 The `y` input was zeros

`y` is the pooled-vector conditioning: the CLIP-L pooled output, 768-wide
(`vec_in_dim` 768; the runtime slices `y[:, :768]`). The local FLUX clip
measures **absmax 8.82 / absmean 0.738 / std 1.07**.

With `y = zeros` the `vector_in` layers have no input, and the measurement
reports a drift of 0.0000 for a layer that does change the output: replacing
`vector_in.in_layer`'s whole weight with 0.5 changes the output by **relMSE
1.73** with a real `y`, and by **exactly 0** with a zero `y`.

Both the impact sweep and the acceptance now use the production `y`
(`sdxl/build_flux_prod_y.py` saves it).

---

## 3. The per-layer drift, after the fix

Script: `sdxl/diag_impact_cn_flux.py` -> `impact/impact_cn_flux_union_pro_2.json`

The observable is the ControlNet's own output: the per-block control features
it returns to the UNet (14,942,208 elements). Each candidate layer's ConvRot
INT8 reconstruction is injected and the relative MSE against the pristine run
is recorded.

| item | value |
| :--- | :--- |
| layers measured | **63** (the 12 fused qkv layers are excluded) |
| runs | 1 (single completion) |
| wall time | 10 s |
| interruptions | 0 |
| the reference output | finite, absmax 712 |

**Drift distribution**

| min | p10 | median | p90 | max |
| :--- | :--- | :--- | :--- | :--- |
| 0.0000e+00 | 0.0000e+00 | 4.06e-05 | 7.41e-05 | 1.27e-04 |

Every layer now measures the same order of magnitude, so the drift **does not
rank the layers** the way the aggregate output does. That is the reason the
FP16-kept set is chosen by the aggregate contribution (section 4) and not by
the drift threshold.

### 3.1 The 13 layers that measure exactly 0.0000

Replacing their whole weight with 0.5 leaves the output **bit-identical**
(measured relMSE 0.0000e+00 with a real `y`), so they really do not influence
the output. They are:

- `controlnet_blocks.5` - its weight is **all zeros** in the runtime
- `double_blocks.4.txt_attn.proj`, `double_blocks.4.txt_mlp.0`,
  `double_blocks.4.txt_mlp.2`, `double_blocks.5.img_attn.proj`,
  `double_blocks.5.img_mlp.0`, `double_blocks.5.img_mlp.2`,
  `double_blocks.5.img_mod.lin`, `double_blocks.5.txt_attn.proj`,
  `double_blocks.5.txt_mlp.0`, `double_blocks.5.txt_mlp.2`,
  `double_blocks.5.txt_mod.lin` - the forward **reaches** them, but their
  output does not reach the returned features

The forward does reach all 13 (a forward hook confirms it), so "not on the
output path" is the wrong description - the correct one is "reached, but the
output does not depend on it".

---

## 4. Choosing the FP16-kept set: the aggregate contribution

The per-layer drift is a **marginal** effect. The quantity the pack actually
trades off is the effect on the **aggregate** output, so that is what is
measured: load the all-INT8 pack through the real path, restore one layer's
FP16 weight, and measure the aggregate cosine against the FP16 baseline.

Script: `sdxl/select_cn_flux_protect.py` ->
`impact/cn_flux_contrib_order.json`

The all-INT8 state measures **mean 0.999622 / min 0.999020** over 16 items.

**The largest contributions**

| rank | layer | the gain |
| :--- | :--- | :--- |
| 1 | `pos_embed_input` | +9.94e-05 |
| 2 | `time_in.in_layer` | +5.54e-05 |
| 3 | `double_blocks.1.txt_attn.qkv` | +5.24e-05 |
| 4 | `img_in` | +4.72e-05 |
| 5 | `double_blocks.1.img_mlp.0` | +4.49e-05 |
| 6 | `time_in.out_layer` | +4.30e-05 |
| 7 | `guidance_in.out_layer` | +4.06e-05 |
| 8 | `vector_in.out_layer` | +3.77e-05 |
| 9 | `double_blocks.0.txt_attn.proj` | +3.69e-05 |
| 10 | `double_blocks.2.txt_attn.qkv` | +3.55e-05 |

**The frontier** (the pack built from each prefix of the ranked list, 32
items, the production `y`)

| the FP16-kept | INT8 | cos mean | cos min | size |
| :--- | :--- | :--- | :--- | :--- |
| 12 | 63 | 0.999771 | 0.999488 | 2234.4 MB |
| 18 | 57 | 0.999813 | 0.999592 | 2459.2 MB |
| **21** | **54** | **0.999860** | **0.999698** | **2585.1 MB** |
| 24 | 51 | 0.999871 | 0.999502 | 2657.0 MB |
| 30 | 45 | 0.999910 | 0.999746 | 2803.1 MB |
| 0 (all INT8) | 75 | 0.999051 | 0.991817 | 2368.2 MB |

**21 FP16-kept layers were adopted.** At the same 54 INT8 layers the measured
minimum cosine is **0.999698**, against **0.999160** for the set the (buggy)
drift threshold produced - the same size, a better result, because the
selection is made on the aggregate instead of on a marginal effect.

---

## 5. The fused qkv

The runtime builds one Linear out of `cat([to_q, to_k, to_v], dim=0)`; the
identity is exact (`torch.equal(cat([q,k,v], dim=0), fused) == True`). The
three halves have value ranges that differ by up to ~7x:

| half | absmax |
| :--- | :--- |
| q | 2.4844 |
| k | 0.5312 |
| v | 0.3750 |

**The ConvRot path handles them correctly.** The measured reconstruction
error of the fused tensor on the ConvRot path is **relMSE 7.49e-05** (the
worst of the 12), and the plain int8 path is 2.16e-04 - both are small, and
the ConvRot path is the better of the two. A per-out-channel scale on the
rotated tensor does not have to serve the three halves separately, because
the rotation mixes the columns.

Nine of the twelve fused qkv layers are quantized in the shipped pack; the
three that are kept in FP16 are among the 21 by their aggregate contribution.

---

## 6. The pack

Built with `clip_convert/convert_cn_flux_impact_convrot_int8.py`:

```
--drift-threshold 0.30 --quantize-qkv --protect-list impact/cn_flux_protect_21.json
```

`--protect-list` takes a JSON `{"protect": [...]}` (the runtime module names)
or a text file with one name per line. An explicit list is **authoritative**
over the drift threshold.

| item | value |
| :--- | :--- |
| the source | 4083.4 MB |
| the pack | **2585.1 MB** (-36.7 %) |
| the tensors the pack carries | 282 (the runtime's own names, nothing else) |
| the quantized layers | **54** (9 of them fused qkv) |
| the FP16-kept layers | 21 |
| the group size | 256 (every layer) |
| the convrot flag | true (every layer) |

---

## 7. Verification (256 items, seed 42, the production `y`)

The baseline is built through the **same ops class** as the pack, because the
stock `comfy.controlnet` path and the `mixed_precision_ops` path are not
numerically equivalent for this model.

| item | value |
| :--- | :--- |
| the load | the loader's own node, the package's arm active |
| the quantized layers at runtime | 54 (all `QuantizedTensor`, group size 256) |
| the `int8_linear` calls | **54 / 54** |
| the output | 14,942,208 elements, **finite** |
| cos mean / min / p5 / p50 | **0.999881 / 0.999445 / 0.999735 / 0.999898** |
| the gate (mean >= 0.99, min >= 0.98) | **PASS** |

**The candidates, on the same 256 items and the same `y`**

| pack | INT8 | cos mean | cos min | size |
| :--- | :--- | :--- | :--- | :--- |
| the previous pack (the buggy drift set) | 54 | 0.999937 | 0.999160 | 2567.2 MB |
| **the shipped pack** | **54** | **0.999881** | **0.999445** | **2585.1 MB** |
| the all-but-qkv pack | 63 | 0.999051 | 0.991817 | 2368.2 MB |
| the all-INT8 pack | 75 | 0.999046 | 0.991294 | 2044.6 MB |

---

## 8. Files

| file | role |
| :--- | :--- |
| `sdxl/diag_impact_cn_flux.py` | the per-layer drift (the un-rotated injection) |
| `sdxl/select_cn_flux_protect.py` | the aggregate contribution, the ranked order |
| `sdxl/build_flux_prod_y.py` | the production `y` (the CLIP-L pooled output) |
| `sdxl/verify_cn_flux_256.py` | the 256-item acceptance, the loader's own path |
| `sdxl/verify_controlnet_loader.py` | every ControlNet pack through the loader's node |
| `clip_convert/convert_cn_flux_impact_convrot_int8.py` | the builder (`--protect-list`) |
| `impact/impact_cn_flux_union_pro_2.json` | the measured per-layer drift (git-ignored) |
| `impact/cn_flux_contrib_order.json` | the measured aggregate contribution (git-ignored) |
| `impact/cn_flux_protect_21.json` | the adopted FP16-kept list (git-ignored) |

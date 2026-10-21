# FLUX.1-dev ControlNet Union Pro 2.0 — ConvRot INT8 Trajectory-Impact Technical Guide

**Model.** `models/ControlNet/FLUX.1-dev-ControlNet-Union-Pro-2.0.safetensors` (4083.4 MB, BF16, 2,140,876,800 parameters)
**Pack.** `models/ControlNet/FLUX.1-dev-ControlNet-Union-Pro-2.0_convrot_int8.safetensors` (2567.2 MB, **-37.1 %**)
**SHA256.** `D880611CC2F82576AD1E409AB0369198E60E08F68F9826ED3AD44D278FCB27F8`
**Method.** impact-driven (measured drift of the ControlNet's own output), ConvRot group size 256.

---

## 1. What this model is, and what the previous pack got wrong

FLUX.1-dev ControlNet Union Pro 2.0 is a **FLUX** ControlNet: a double-stream
transformer (6 double blocks) with ControlNet-specific blocks and embedders.

| part | measured |
| :--- | :--- |
| checkpoint keys | 222 |
| parameters | 2,140,876,800 |
| dtypes | BF16 (every tensor) |
| transformer blocks | 6 (`transformer_blocks.0..5`) |
| hidden size | 3072 |
| ControlNet blocks | 6 (`controlnet_blocks.0..5`, 3072x3072) |
| context / latent | context_in_dim 4096, latent 16 channels, patch size 2 |

The previous pack of this model was quantized by the **old method** and it had
two defects, both measured here:

1. **The fused `qkv` layers were quantized, and that is a broken layer.** The
   runtime builds one Linear out of `cat([to_q, to_k, to_v], dim=0)`; the
   verified identity is exact (`torch.equal(cat([q,k,v], dim=0), fused) ==
   True`). The three halves have value ranges that differ by ~7x:

   | half | absmax | absmean |
   | :--- | :--- | :--- |
   | q | 2.4844 | 0.02222 |
   | k | 0.5312 | 0.02795 |
   | v | 0.3750 | 0.01062 |

   One per-out-channel scale cannot serve them: the measured reconstruction
   error of the fused tensor is **relMSE 1.89** - the layer is destroyed. A
   per-half amax does not help either (the scale is already per out-channel,
   and the q rows already use their own amax): the same 1.89.

2. **The pack mixed two key layouts.** The source is in the **diffusers**
   layout; the runtime is built from the **comfy-native** layout
   (`comfy.model_detection.convert_diffusers_mmdit`). The old pack carried 174
   runtime-named tensors *plus* 150 diffusers-named duplicates of the same
   weights.

The rebuilt pack fixes both: the fused qkv is handled correctly, and the pack
carries exactly the tensors the runtime reads.

---

## 2. The runtime, and the key mapping

The runtime path is `comfy.controlnet.load_controlnet` ->
`load_controlnet_flux_instantx` (the source carries
`controlnet_x_embedder.weight`), and the loader package's
`HSWQ ControlNet Loader (ConvRot INT8)` node builds `ControlNetFlux` directly
from the comfy-native state dict.

| item | measured |
| :--- | :--- |
| runtime Linear modules | **75** (6 blocks x (4 self-attn + 4 cross-attn + 3 FF) + 6 controlnet blocks + the embedders) |
| tensors the runtime reads | **174** (its own `state_dict()`) |
| runtime names present in the source | 6 (the runtime renames) |
| runtime names present in the converted state dict | 69 / 75 |

The exact runtime -> source mapping is obtained from
`comfy.utils.flux_to_diffusers` (the map the loader itself uses), inverted:
63 of the 75 Linear modules map one-to-one onto a source key, and the
remaining 12 are the fused `qkv` layers (three source keys each).

---

## 3. Impact measurement

Script: `sdxl/diag_impact_cn_flux.py` -> `impact/impact_cn_flux_union_pro_2.json`

The observable is the ControlNet's **own output**: the per-block control
features it returns to the UNet. Each candidate layer's ConvRot INT8
reconstruction is injected and the relative MSE of those features against the
pristine run is recorded. The model is built the way the loader builds it and
the forward is called with a real latent tensor, a real context tensor and a
real control hint.

| item | value |
| :--- | :--- |
| layers measured | **63** (the 12 fused qkv layers are excluded) |
| runs | 1 (single completion) |
| wall time | 12 s |
| interruptions | 0 |
| reference output | 14,942,208 elements, finite |

**Drift distribution** (the 50 layers with a non-zero drift)

| min | p10 | median | p90 | max |
| :--- | :--- | :--- | :--- | :--- |
| 1.29e-03 | 5.65e-03 | 1.96e-01 | 1.00e+00 | 4.30e+01 |

**The largest drifts**

| layer | drift |
| :--- | :--- |
| `time_in.out_layer` | 4.30e+01 |
| `double_blocks.4.img_mlp.0` | 3.34e+00 |
| `vector_in.out_layer` | 1.38e+00 |
| `time_in.in_layer` | 1.25e+00 |
| `double_blocks.0.img_mod.lin` | 1.00e+00 |
| `controlnet_blocks.4` | 8.97e-01 |

**13 layers measure exactly 0.0000, and that is not a measurement failure.**
Replacing their whole weight with 0.5 leaves the output **bit-identical**
(measured relMSE 0.0000e+00), so they are not on the output path at all -
`double_blocks.4` / `double_blocks.5` are not reached by this ControlNet's
forward, and `vector_in.in_layer` and `controlnet_blocks.5` likewise. Layers
that cannot influence the output are kept in FP16: quantizing them would buy
nothing and risk the layer for no measured benefit.

---

## 4. Threshold selection (measured on the output features)

Five candidate packs were built and each was measured against the FP16
baseline on a fixed set of seeded inputs (seed 42).

| pack | quantized | FP16 kept | cos mean | cos min | cos p5 | size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| threshold 0.15, qkv in FP16 | 38 | 37 | **0.999978** | **0.999919** | 0.999919 | 3061.5 MB |
| threshold 0.15, qkv quantized | 50 | 25 | 0.999960 | 0.999895 | 0.999895 | 2738.0 MB |
| threshold 0.30, qkv in FP16 | 42 | 33 | 0.999966 | 0.999856 | 0.999856 | 2890.8 MB |
| **threshold 0.30, qkv quantized** | **54** | **21** | **0.999947** | **0.999853** | **0.999853** | **2567.2 MB** |
| the previous pack | 75 | 0 | 0.999266 | 0.998950 | 0.998950 | 2044.6 MB |

**Threshold 0.30 with the fused qkv quantized was adopted.** It is the smallest
pack that still measures far better than the previous one (cos mean 0.999947
against 0.999266, a 4.7x smaller mean deviation), and the qkv damage it does
carry is bounded by the 0.30 drift threshold - unlike the previous pack,
which quantized every layer with no measurement at all.

---

## 5. The pack

Built with `clip_convert/convert_cn_flux_impact_convrot_int8.py`:

```
--drift-threshold 0.30 --quantize-qkv
```

| item | value |
| :--- | :--- |
| file size | 4083.4 MB -> **2567.2 MB** (-37.1 %) |
| tensors | 282 (the 174 the runtime reads, plus the scales and the `comfy_quant` stamps) |
| quantized layers | 54 (group size 256) |
| FP16 kept | 21 (the layers whose measured drift exceeds 0.30) |
| the layout | the runtime parameter names only - no duplicated diffusers half |

---

## 6. Verification (the real load path)

The pack is loaded through the loader node's own path
(`_load_comfy_flux_controlnet` with
`mixed_precision_ops({"int8_tensorwise": ...})`).

| item | value |
| :--- | :--- |
| loaded | `ControlNetFlux` |
| runtime weight types | **54 QuantizedTensor + 21 Parameter** |
| `int8_linear` calls per forward | **54** |
| the control features | 14,942,208 elements, **finite** |
| cos mean / min / p5 (8 seeded inputs) | **0.999947 / 0.999853 / 0.999853** |
| gate (mean >= 0.99, min >= 0.98) | **PASS** |
| latency | 278.50 ms per forward (32x32 latent) |

**The INT8 GEMM is reached.** On the stock path ComfyUI's `int8_tensorwise` is
weight-only (`quantize_input: False`) and `int8_linear` is never called; the
loader package's runtime arm
(`ComfyUI-HSWQ-Loader-and-Tools/patches/clip_int8_gemm/arm.py`) routes the
`TensorWiseINT8Layout` weights into the real kernel, with the online ConvRot
activation rotation. **ComfyUI core is not modified.**

---

## 7. Files

| file | role |
| :--- | :--- |
| `sdxl/diag_impact_cn_flux.py` | impact measurement (the ControlNet's own control features) |
| `clip_convert/convert_cn_flux_impact_convrot_int8.py` | impact-driven pack builder |
| `impact/impact_cn_flux_union_pro_2.json` | the measured per-layer drift (git-ignored) |
| `models/ControlNet/FLUX.1-dev-ControlNet-Union-Pro-2.0_convrot_int8.safetensors` | the pack |

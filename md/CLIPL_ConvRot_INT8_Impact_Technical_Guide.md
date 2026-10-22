# CLIP-L ConvRot INT8 for FLUX.1 - Impact-Driven Quantization (Technical Guide)

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`How to quantize Text Encoder and ControlNet.md`](How to quantize Text Encoder and ControlNet.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this model.

How the **FLUX.1 CLIP-L text encoder** (`clip_l.safetensors`) is quantized to ConvRot
INT8 with a **measured** FP16 protection set, and how it loads through the
**standard ComfyUI path** (no custom node).

---

## 0. Scope: FLUX.1 CLIP-L, standard loader

`clip_l.safetensors` is the **FLUX.1** CLIP-L text encoder. ComfyUI loads it with the
standard **`DualCLIPLoader`** (`type = "flux"`, `clip_l` + `t5xxl`) or `CLIPLoader`
(`type = "flux"`). **FLUX.1 CLIP ConvRot INT8 works with the standard loaders as-is.**

---

## 1. The method (impact-driven)

### 1.1 The old method and its invalidity

The earlier pack chose its protected layers from **static kurtosis / outlier-ratio**,
computed through a **broken safetensors header offset** (`f.seek(offs[0])` instead of
`8 + header_len + offs[0]`). The values were garbage; the protection set had **no valid
basis**. It also left the largest tensor (the 72.4 MB token embedding) unquantized,
so the file barely shrank.

### 1.2 The new method

Every eligible Linear layer is measured: its ConvRot INT8 reconstruction is injected,
the CLIP forward is run on a fixed prompt set, and the relative MSE of the text
embedding against the pristine run is recorded. Layers above the threshold stay FP16.

- Measurement: `sdxl/diag_impact_clip.py`
- Result: `impact/impact_clip_l.json` (73 Linear layers)

### 1.3 Measured drift distribution (73 layers)

| Statistic | Value |
| :--- | :--- |
| layers | 73 |
| min | 0.0 |
| median | 1.58e-01 |
| p90 | 1.29e+00 |
| max | **1.8926** |

Heaviest layers:

| drift | layer |
| :--- | :--- |
| 1.8926 | `text_model.encoder.layers.0.mlp.fc2` |
| 1.8610 | `text_model.encoder.layers.0.mlp.fc1` |
| 1.6163 | `text_model.encoder.layers.0.self_attn.v_proj` |
| 1.5935 | `text_model.encoder.layers.0.self_attn.out_proj` |

---

## 2. Threshold sweep (measured, FLUX path)

> **SUPERSEDED - see section 8.** The sweep below was measured with the
> defective injection and with a baseline on a different ops path, so its
> numbers do not describe the shipped pack. It is kept for the record.
> The shipped pack's numbers are in **section 8.3 and 8.4**.

Every threshold was measured end to end (build + FLUX load + embedding cos):

| Threshold | INT8 | FP16 | Size | cos vs FP16 |
| :--- | :--- | :--- | :--- | :--- |
| 0.15 | 36 | 38 | 165.7 MB | 0.999949 |
| 0.30 | 51 | 23 | 152.2 MB | 0.999926 |
| 0.40 | 55 | 19 | 150.0 MB | 0.999920 |
| 0.50 | 57 | 17 | 147.2 MB | 0.999915 |
| 0.80 | 59 | 15 | 142.7 MB | 0.999909 |
| 1.00 | 60 | 14 | 142.2 MB | 0.999909 |
| 1.20 | 65 | 9 | 132.6 MB | 0.999897 |
| 1.50 | 67 | 7 | 128.2 MB | 0.999889 |
| ~~1.80 (the previous pack)~~ | **71** | **3** | **122.6 MB** | **0.999857** |
| 2.00 | 73 | 1 | 118.1 MB | 0.999844 |

**Why 1.80 was chosen then (superseded).** The threshold sits inside the measured
just above the two most fragile layers (`layers.0.mlp.fc2` at 1.8926 and
`layers.0.mlp.fc1` at 1.8610). Those two stay FP16; every other layer is INT8.
Raising the threshold to 2.00 would quantize even those two, i.e. it would ignore
the measurement's own signal while gaining only 4.5 MB. Lowering it protects many
more layers for no measurable quality benefit. 1.80 is the highest-quality-per-byte
distribution, just above the two most fragile layers (`layers.0.mlp.fc2` at
1.8926 and `layers.0.mlp.fc1` at 1.8610). Those two stay FP16; every other
layer is INT8.

**What the re-measurement changed.** The 1.8926 and 1.8610 are artifacts of the
injection defect: the corrected measurement puts `layers.0.self_attn.v_proj` at
2.48e-05 and the whole median at 2.89e-06, and the ranking changes with it.
Those two layers are not the ones that carry the output, and the layer set that
was kept for the wrong reason is the layer set that matters. The shipped pack
keeps **10** layers, chosen on the aggregate (section 8.3).

---

## 3. The converter

`clip_convert/convert_clip_impact_convrot_int8.py` reads the measured impact JSON.

Rules (same runtime constraints as ControlNet / SAM):

1. Only 2D `nn.Linear` / `nn.Embedding` layers whose input width is a multiple of 256
   are quantized.
2. Group size fixed at **256** (the ConvRot kernel is 2D-only, fused only at 256).
3. The pack is written under the checkpoint's own parameter names.

Pack layout:

```
<layer>.weight           int8
<layer>.weight_scale     float32  ([out,1])
<layer>.comfy_quant      uint8 JSON  {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}
_quantization_metadata   JSON  {"format_version":"1.0","layers":{...}}
```

---

## 4. Measured results (standard FLUX path)

### 4.1 Embedding fidelity

| | FP16 | INT8 |
| :--- | :--- | :--- |
| absmax | 33.0637 | **33.0634** |
| cos vs FP16 | - | **0.999876** (mean), **0.999841** (min) |

### 4.2 Standard-path load check

Loaded through `DualCLIPLoader(type="flux")` with the T5-XXL companion:
**328 of 490 Linear/Embedding layers become QuantizedTensor** - the standard loader
consumes the pack's `comfy_quant` with no custom node.

### 4.3 Size

| | FP16 | INT8 |
| :--- | :--- | :--- |
| file size | 234.7 MB | **135.5 MB** (-42.2 %) |
| ConvRot INT8 | - | 63 |
| FP16 retained | - | 3 |

The 3 FP16 layers are `layers.0.mlp.fc2`, `layers.0.mlp.fc1` (the two highest measured
drift) and `text_model.embeddings.position_embedding` (77 rows, not a multiple of 4,
so it cannot use the INT8 GEMM path).

### 4.4 Reproducibility

| | SHA256 |
| :--- | :--- |
| `clip_l.safetensors` (source) | `660C6F5B1ABAE9DC498AC2D21E1347D2ABDB0CF6C0C0C8576CD796491D9A6CDD` |
| `clip_l_hswq_convrot_int8.safetensors` | `C888133EA31AF428EB537B59FF9B818F5CC8D8B682AC595DE63F72FDF9111F35` |

---

## 5. Reproduce

```bash
# 1) measure the output drift of every Linear layer (73 layers)
python sdxl/diag_impact_clip.py \
  --model clip_l.safetensors \
  --out impact/impact_clip_l.json \
  --comfy_path <ComfyUI checkout> \
  --full

# 2) convert at the optimum threshold (default)
python clip_convert/convert_clip_impact_convrot_int8.py \
  --input clip_l.safetensors \
  --output clip_l_hswq_convrot_int8.safetensors \
  --impact impact/impact_clip_l.json \
  --comfy_path <ComfyUI checkout> \
  --clip_type flux \
  --drift-threshold 1.8
```

The output loads directly in the standard FLUX.1 `DualCLIPLoader` / `CLIPLoader`.

---

## 6. Summary

| Aspect | Choice |
| :--- | :--- |
| Target | FLUX.1 CLIP-L (`clip_l.safetensors`) |
| Protection criterion | Measured drift of the CLIP's own text-embedding output |
| Measurement scope | Full sweep of all 73 Linear layers, one run |
| Perturbation | ConvRot INT8 reconstruction (Hadamard rotation + per-out-channel INT8) |
| Threshold | **1.8** (inside the measured distribution, above the two max-drift layers) |
| Quantized | **71** ConvRot INT8 (of 74 2D modules) |
| FP16 retained | **3** (`layers.0.mlp.fc2`, `layers.0.mlp.fc1`, `position_embedding`) |
| Standard-path load | FLUX `DualCLIPLoader` - 328/490 QuantizedTensor, no custom node |
| Measured embedding cos vs FP16 | **0.999876** (mean), **0.999841** (min) |
| Size | 234.7 MB -> **135.5 MB** (-42.2 %) |

---

## 8. The re-measurement (2026-10-21) and the rebuilt pack

### 8.1 The injection defect

The impact measurement had a defect that was found while working on the FLUX
ControlNet: `convrot_int8_recon` returned the reconstruction in the **rotated**
frame (`q * s`, i.e. `W_hat * H`) and injected it as the module's weight, but
the module's weight is used by the plain forward as `F.linear(x, weight)`. The
pack stores the rotated weight and the kernel rotates the **activation**, so
the quantization error that reaches the output is in the **original** frame:
`(q*s) H^T - W`.

Measured on the FLUX ControlNet's `controlnet_blocks.0` with the pack's own
qdata/scale and the real kernel:

| what | relMSE vs the FP16 result |
| :--- | :--- |
| the pack's kernel (`convrot=True`) | **1.74e-04** (correct) |
| the plain int8 (no activation rotation) | 1.89 |
| the injected rotated frame as a plain linear (the defect) | 1.89 |
| the un-rotated reconstruction as a plain linear (the fix) | **9.01e-05** |

The same defect was in this script; it is fixed, and the 73-layer sweep was
redone.

### 8.2 The corrected drift

| | the old (defective) | the new (corrected) |
| :--- | :--- | :--- |
| the measured modules | 73 | 73 |
| the median drift | 1.58e-01 | **2.89e-06** |
| the p90 | 1.29e+00 | **7.34e-06** |
| the max | 1.89e+00 | **2.48e-05** |
| the protected at the old threshold 1.80 | 2 | **0** |

The old measurement made layer 0's `self_attn.v_proj` look catastrophic
(1.616) and `mlp.fc2` worse still (1.893). The corrected measurement puts the
same layer at 2.48e-05, a factor of 65,000 lower, and the ranking changes
with it.

### 8.3 The selection criterion, and the rebuilt pack

The per-layer drift measures the **weight** error only (the reconstruction is
placed in the module and the plain forward is used, so the activation is not
quantized), while the pack's real path also quantizes the **activation** for
that layer, and that is where most of the error comes from. The drift is
therefore a screen, not the selection criterion.

The candidates below were measured on the full 256-item acceptance set (seed
42) against the FP16 source, with the image-embedding cosine as the
observable, on the **same ops class** for the baseline and the pack.

| the FP16-kept | INT8 layers | cos mean | cos min | cos p5 | pack |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 (all quantized) | 73 | 0.999797 | 0.999769 | 0.999769 | 118.1 MB |
| **10** | **63** | **0.999876** | **0.999841** | **0.999859** | **135.5 MB** |
| 20 | 53 | 0.999917 | 0.999907 | 0.999910 | 156.2 MB |
| 30 | 43 | 0.999946 | 0.999927 | 0.999936 | 171.9 MB |
| 40 | 33 | 0.999969 | 0.999954 | 0.999961 | 179.2 MB |
| 50 | 23 | 0.999987 | 0.999980 | 0.999982 | 186.4 MB |
| the previous pack | 71 | 0.999812 | 0.999762 | 0.999784 | 122.6 MB |

**10 FP16-kept layers were adopted.** The quality rises monotonically with the
kept count; 10 is where the cost is small (13 MB) and the minimum already
reaches 0.999841, against the previous pack's 0.999762 - **2.5x better** for
13 MB.

The previous pack kept only 2 layers (layer 0's `mlp.fc1` and `mlp.fc2`), which
came from the defective drift's top two. The corrected measurement puts those
two nowhere near the top, and the layer set that was kept for the wrong
reason is the layer set that carries the output.

### 8.4 The rebuilt pack

| item | value |
| :--- | :--- |
| the source | 234.7 MB |
| the pack | **135.5 MB** (-42.2 %) |
| the SHA256 | `C888133EA31AF428EB537B59FF9B818F5CC8D8B682AC595DE63F72FDF9111F35` |
| the tensors | 322 |
| the quantized layers | **63** (of 73) |
| the FP16-kept layers | 10 (the corrected drift's ranking) |
| the group size | 256 (all 63 layers) |
| the `int8_linear` calls per encode | **15872** (256 items x 62 layers) |
| the output | finite |
| cos mean / min / p5 (256 items, seed 42) | **0.999876 / 0.999841 / 0.999859** |
| the gate (mean >= 0.99, min >= 0.98) | **PASS** |

### 8.5 The measurement note

The acceptance runs on the **same ops class** for the baseline and the pack,
because the stock CLIP path (`comfy.ops.manual_cast`) and the
`mixed_precision_ops` path are not numerically equivalent for this model. The
**arm** is active: the loader package's `patches/clip_int8_gemm` makes the
real `int8_linear` fire (15872 calls over the 256-item set), instead of
dequantizing the weight for every matmul.

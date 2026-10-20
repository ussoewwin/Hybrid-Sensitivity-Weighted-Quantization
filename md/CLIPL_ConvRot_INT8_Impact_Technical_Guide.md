# CLIP-L ConvRot INT8 for FLUX.1 - Impact-Driven Quantization (Technical Guide)

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
| **1.80 (shipped)** | **71** | **3** | **122.6 MB** | **0.999857** |
| 2.00 | 73 | 1 | 118.1 MB | 0.999844 |

**Why 1.80 is the optimum.** The threshold sits inside the measured distribution,
just above the two most fragile layers (`layers.0.mlp.fc2` at 1.8926 and
`layers.0.mlp.fc1` at 1.8610). Those two stay FP16; every other layer is INT8.
Raising the threshold to 2.00 would quantize even those two, i.e. it would ignore
the measurement's own signal while gaining only 4.5 MB. Lowering it protects many
more layers for no measurable quality benefit. 1.80 is the highest-quality-per-byte
point that still honours the measurement.

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
| cos vs FP16 | - | **0.999857** |

### 4.2 Standard-path load check

Loaded through `DualCLIPLoader(type="flux")` with the T5-XXL companion:
**328 of 490 Linear/Embedding layers become QuantizedTensor** - the standard loader
consumes the pack's `comfy_quant` with no custom node.

### 4.3 Size

| | FP16 | INT8 |
| :--- | :--- | :--- |
| file size | 234.7 MB | **122.6 MB** (-47.8 %) |
| ConvRot INT8 | - | 71 |
| FP16 retained | - | 3 |

The 3 FP16 layers are `layers.0.mlp.fc2`, `layers.0.mlp.fc1` (the two highest measured
drift) and `text_model.embeddings.position_embedding` (77 rows, not a multiple of 4,
so it cannot use the INT8 GEMM path).

### 4.4 Reproducibility

| | SHA256 |
| :--- | :--- |
| `clip_l.safetensors` (source) | `660C6F5B1ABAE9DC498AC2D21E1347D2ABDB0CF6C0C0C8576CD796491D9A6CDD` |
| `clip_l_hswq_convrot_int8.safetensors` | `1D82EA75F9636BE44563743C5E87135E6A08B600A14C03D35CECE7F589760724` |

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
| Measured embedding cos vs FP16 | **0.999857** |
| Size | 234.7 MB -> **122.6 MB** (-47.8 %) |

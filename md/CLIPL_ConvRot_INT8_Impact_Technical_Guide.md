# CLIP-L ConvRot INT8 for FLUX.1 - Impact-Driven Quantization (Technical Guide)

How the **FLUX.1 CLIP-L text encoder** (`clip_l.safetensors`) is quantized to ConvRot
INT8, with a **measured** FP16 protection set, and how it loads through the
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
| median | 1.58e-01 |
| p90 | 1.29e+00 |
| max | 1.89e+00 |

Heaviest: `layers.0.mlp.fc2` (1.89), `layers.0.mlp.fc1` (1.86),
`layers.0.self_attn.v_proj` (1.62), `layers.0.self_attn.out_proj` (1.59).

### 1.4 Threshold vs size (measured)

The whole CLIP-L is 234.7 MB, of which the **token embedding alone is 72.4 MB (31 %)**.
The threshold determines how much of that stays FP16:

| Threshold | Quantized | Protected | Size | cos vs FP16 |
| :--- | :--- | :--- | :--- | :--- |
| 0.15 | 36 | 38 | 165.7 MB | 0.999954 |
| 0.50 | 57 | 17 | 147.2 MB | 0.999921 |
| 1.00 | 60 | 14 | 142.2 MB | 0.999916 |
| **full (all 2D layers)** | **73** | **0** | **118.1 MB** | **0.999848** |

The **full** scope is shipped: it quantizes all 73 Linear layers (embedding included),
reaching **118.1 MB (-49.7 %)** while keeping the embedding cos at **0.999848**.
Protecting layers on top of this would only add size back without a measurable gain,
so the highest-quality-per-byte choice is the full scope.

---

## 2. The converter

`clip_convert/convert_clip_impact_convrot_int8.py` reads the measured impact JSON.

Rules (same runtime constraints as ControlNet / SAM):

1. Only 2D `nn.Linear` layers whose input width is a multiple of 256 are quantized.
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

## 3. Measured results (standard FLUX path)

### 3.1 Embedding fidelity

| | FP16 | INT8 |
| :--- | :--- | :--- |
| absmax | 33.0637 | **33.0620** |
| cos vs FP16 | - | **0.999848** |

### 3.2 Standard-path load check

Loaded through `DualCLIPLoader(type="flux")` with the T5-XXL companion:
**328 of 490 Linear layers become QuantizedTensor** - the standard loader consumes the
pack's `comfy_quant` with no custom node.

### 3.3 Size

| | FP16 | INT8 |
| :--- | :--- | :--- |
| file size | 234.7 MB | **118.1 MB** (-49.7 %) |
| quantized layers | - | 73 / 73 |
| comfy_quant tensors | - | 73 |

### 3.4 Reproducibility

| | SHA256 |
| :--- | :--- |
| `clip_l.safetensors` (source) | `660C6F5B1ABAE9DC498AC2D21E1347D2ABDB0CF6C0C0C8576CD796491D9A6CDD` |
| `clip_l_hswq_convrot_int8.safetensors` | `691437D48F36A609BF329853BCCC01E1843FAD8B20233D8A1509D47D7495986B` |

---

## 4. Reproduce

```bash
# 1) measure the output drift of every Linear layer (73 layers)
python sdxl/diag_impact_clip.py \
  --model clip_l.safetensors \
  --out impact/impact_clip_l.json \
  --comfy_path <ComfyUI checkout> \
  --full

# 2) convert (full scope: every 256-wide 2D Linear layer)
python clip_convert/convert_clip_impact_convrot_int8.py \
  --input clip_l.safetensors \
  --output clip_l_hswq_convrot_int8.safetensors \
  --impact impact/impact_clip_l.json \
  --comfy_path <ComfyUI checkout> \
  --clip_type flux \
  --drift-threshold 0
```

The output loads directly in the standard FLUX.1 `DualCLIPLoader` / `CLIPLoader`.

---

## 5. Summary

| Aspect | Choice |
| :--- | :--- |
| Target | FLUX.1 CLIP-L (`clip_l.safetensors`) |
| Protection criterion | Measured drift of the CLIP's own text-embedding output |
| Measurement scope | Full sweep of all 73 Linear layers, one run |
| Perturbation | ConvRot INT8 reconstruction (Hadamard rotation + per-out-channel INT8) |
| Quantized | **73 / 73** ConvRot INT8 Linear (all 2D layers) |
| Standard-path load | FLUX `DualCLIPLoader` - 328/490 QuantizedTensor, no custom node |
| Measured embedding cos vs FP16 | **0.999848** |
| Size | 234.7 MB -> **118.1 MB** (-49.7 %) |

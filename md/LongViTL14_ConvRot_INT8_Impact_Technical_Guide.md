# Long-ViT-L-14-BEST-GmP-smooth-ft-full ConvRot INT8 — Technical Guide

**Target**: `Long-ViT-L-14-BEST-GmP-smooth-ft-full.safetensors` — an OpenAI CLIP
ViT-L/14 checkpoint with an extended text context, stored in FP32, carrying **two
towers** (`text_model.*`, `vision_model.*`).

**What this document covers**: how it is quantized to ConvRot INT8 with a **measured**
FP16 protection set covering **both towers**, the pack layout, the load paths, and the
measured fidelity / size / latency.

- Source: `D:\USERFILES\ComfyUI\ComfyUI\models\clip\Long-ViT-L-14-BEST-GmP-smooth-ft-full.safetensors`
  (1631.80 MB, SHA256 `4D3E6C8C21881F13E532C33A809538B51132BAE9FF8A40147FBDD7D57DC8C13E`)
- Pack: `D:\USERFILES\ComfyUI\ComfyUI\models\clip\Long-ViT-L-14-BEST-GmP-smooth-ft-full_hswq_convrot_int8.safetensors`
  (413.20 MB, SHA256 `EF3E0A30DC1C41BAC4C5F35047944E86EE6CAB9D4CD63D4F3BB25C8FCFAC0E20`)

---

## 0. The checkpoint

| item | value |
| :--- | :--- |
| architecture | OpenAI CLIP ViT-L/14, long-text variant |
| 2D weight modules | 221 (text 74 + vision 145 + 2 misc) |
| text context | position embedding **248 × 768** (vs 77 in the standard ViT-L/14) |
| input widths | 1024 (122), 768 (63), 4096 (24), 3072 (12) |
| dtypes | FP32 |
| projection layers | `text_projection.weight` and `vision_model.embeddings.class_embedding` are **present** |

Because both projection tensors exist, no completion step is required when loading
either tower.

### 0.1 Where the 1631.80 MB -> 413.20 MB reduction comes from

The size numbers are read straight out of the two safetensors headers (the
parameter inventory and the dtypes), and they reconcile exactly.

**Source**

| item | measured |
| :--- | :--- |
| file size | 1,711,063,028 bytes = 1631.80 MB |
| tensors | 590 |
| total parameters | **427,747,841** |
| dtypes | **F32 for every tensor** (no F16 anywhere) |
| text_model | 492,767,232 bytes (469.9 MB), 123,191,808 params |
| vision_model | 1,212,719,104 bytes (1156.5 MB), 303,179,776 params |
| other (logit_scale, position embedding) | 5,505,024 bytes (5.2 MB) |

**Pack**

| item | measured |
| :--- | :--- |
| file size | 433,270,948 bytes = **413.20 MB** |
| tensors | 1030 (the source's 590 plus `.weight_scale` and `.comfy_quant` per quantized layer) |
| INT8 weight payload | 406.7 MB (220 layers) |
| scales + `comfy_quant` metadata | 1.4 MB |
| FP16 kept | 5.0 MB (a single 2D weight) |
| total parameters | 428,119,001 |

**The reconciliation**

```
source parameters                        427,747,841   (all F32)
every parameter as int8                  427,747,841 x 1 byte = 407.9 MB
measured pack file                                            413.20 MB
difference                                                         5.3 MB  = the scales + metadata + the FP16-kept weight
```

**No tensor is missing.** The pack holds 428,119,001 parameters against the
source's 427,747,841 - a ratio of 1.0009, the surplus being the per-layer
scales. Every one of the source's 590 keys is present in the pack.

The reduction is the plain **FP32 -> INT8 factor of 1/4**: 1631.80 MB / 4 =
407.9 MB, and the pack measures 413.20 MB. The source being FP32 is exactly why
the factor is 1/4 rather than 1/2.

### 0.2 The file size is not the runtime footprint

ComfyUI loads a text encoder in `float16`, so the resident weights of the FP32
source are much smaller than its file. Measured by summing the loaded modules'
weight bytes:

| | source | pack | change |
| :--- | :--- | :--- | :--- |
| **file size** | 1631.80 MB | 413.20 MB | **-74.7 %** |
| **resident weights (runtime)** | **235.90 MB** | **118.48 MB** | **-49.8 %** |
| resident dtype | fp16 (100 tensors) | INT8 + fp16 | |

The two numbers answer different questions. The file reduction is 74.7 % because
the source is FP32 and INT8 is a quarter of that. The runtime reduction is
49.8 %, because the source is already held in fp16 in memory and the
`text_model.embeddings.token_embedding` (144.8 MB as FP32) has to stay in FP16 -
ComfyUI's `operations.Embedding` assumes an unpacked weight, and quantizing it
collapses the output cosine to 0.18.

The same accounting for the other shipped packs (read from each header):

| pack | file MB | int8 MB | fp16 MB | INT8 layers |
| :--- | :--- | :--- | :--- | :--- |
| Long-ViT-L-14 | 413.2 | 406.7 | 5.0 | 220 |
| CLIP-SAE-ViT-L-14 | 413.2 | 406.5 | 5.2 | 219 |
| ViT-L-14-BEST-smooth-GmP | 570.3 | 246.6 | 322.7 | 164 |
| clip_l (FLUX) | 122.6 | 112.7 | 9.3 | 71 |

---

## 1. Method: impact-driven, per tower

For each candidate 2D layer, the layer's weight is replaced by its ConvRot INT8
reconstruction (Hadamard rotation along the input axis + per-out-channel INT8), the
tower is run on a fixed input, and the relative MSE of the tower's **own output**
against the pristine run is recorded. The layer is then restored.

| tower | observable | script | layers measured |
| :--- | :--- | :--- | :--- |
| text | text embedding (`encode_from_tokens_scheduled`) | `sdxl/diag_impact_clip.py` | 73 |
| vision | image embedding (`encode_image`) | `sdxl/diag_impact_clip_vision.py` | 144 |

### 1.1 Measured drift distribution

| tower | layers | median | p90 | max |
| :--- | :--- | :--- | :--- | :--- |
| text | 73 | **2.2013e-01** | 1.4511e+00 | **1.9328** |
| vision | 144 | **4.9761e-02** | 2.8525e-01 | **8.5660e-01** |

The profile matches the other ViT-L/14 checkpoints: a handful of text layers dominate
the drift while the vision tower is far less sensitive.

---

## 2. Threshold selection (measured on the acceptance set)

The acceptance set is fixed and deterministic (no network):
- text: 256 prompts composed from a fixed 34-word vocabulary with `random.Random(42)`
- vision: 256 synthetic 224×224 images from `torch.Generator().manual_seed(42)`

Acceptance: **mean >= 0.99 AND min >= 0.98** for both towers.

| thresholds (text / vision) | INT8 | FP16 | size | text min | vision min | verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1.95 / 0.90 (shipped)** | **220** | **1** | **413.2 MB** | **0.999687** | **0.999210** | **PASS** |

Both thresholds sit just above the measured maxima of their tower (text 1.9328,
vision 0.8566), i.e. the maximum quantization available. The acceptance minimum still
clears 0.98 with a wide margin, so protecting more layers would only add size back
without a measured benefit.

The single FP16-retained 2D module is `vision_model.embeddings.position_embedding`
(257 rows — not a multiple of 4, so it cannot use the INT8 GEMM path).

---

## 3. Pack layout

```
text_model.*.weight          int8 (ConvRot)
text_model.*.weight_scale    float32  [out, 1]
text_model.*.comfy_quant     uint8 JSON {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}
vision_model.*.weight        int8 (ConvRot)
vision_model.*.weight_scale  float32  [out, 1]
vision_model.*.comfy_quant   uint8 JSON {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}
_quantization_metadata       JSON {"format_version":"1.0","layers":{...}}
```

| | value |
| :--- | :--- |
| ConvRot INT8 layers | **220** (text 74 + vision 144 + 2 misc) |
| FP16 retained (2D) | 1 |
| `comfy_quant` tensors | 220 |
| group size | 256 (fixed) |

---

## 4. Measured results

### 4.1 Embedding fidelity (256 items, seed 42)

| tower | mean | min | p5 | verdict |
| :--- | :--- | :--- | :--- | :--- |
| text | **0.999809** | **0.999687** | 0.999738 | **PASS** |
| vision | **0.999772** | **0.999210** | 0.999699 | **PASS** |

### 4.2 Size / latency (same machine, same batch)

| metric | FP32 | INT8 | change |
| :--- | :--- | :--- | :--- |
| file size | 1631.80 MB | **413.20 MB** | **-74.7 %** | 1/4 of the FP32 source; see section 0.1 |
| resident weights (runtime) | 235.90 MB | **118.48 MB** | **-49.8 %** | ComfyUI holds the source in fp16; see section 0.2 |

Latency, measured by batch (the batch size decides whether the overhead is
amortized):

**vision tower**

| batch | FP32 | INT8 | ratio |
| :--- | :--- | :--- | :--- |
| 1 | 45.65 ms | 51.40 ms | 1.13x |
| 4 | 102.81 ms | 101.32 ms | **0.99x** |
| 16 | 377.96 ms | 374.21 ms | **0.99x** |

**text tower**

| batch | FP32 | INT8 | ratio |
| :--- | :--- | :--- | :--- |
| 1 | 18.20 ms | 34.31 ms | 1.88x |
| 4 | 17.57 ms | 38.91 ms | 2.21x |
| 16 | 18.59 ms | 31.12 ms | 1.67x |

**Why the towers differ, and the fix.** The vision tower is not slower at any batch
size once the per-call overhead is amortized. The text tower was, and the cause was
in ComfyUI core, not in the pack:

- ComfyUI's `int8_tensorwise` algorithm is declared with `"quantize_input": False`
  (`comfy/quant_ops.py`). It is a **weight-only** format: the weight stays int8 in
  memory, but it is **dequantized for every matmul**. The only path a weight-only
  algorithm can take in `comfy/ops.py` is `forward_comfy_cast_weights` ->
  `CastBiasWeightContext` -> `F.linear` with an fp16 weight. The real INT8 GEMM
  (`comfy_kitchen`'s `int8_linear`) is therefore never reached.
  Measured on the stock path: `int8_linear` calls **0**.
- `comfy/sd1_clip.py` additionally builds the text encoder with
  `mixed_precision_ops(quant_config, dtype, full_precision_mm=True)`, and
  `full_precision_mm=True` makes `_use_quantized` false in
  `MixedPrecisionOps.Linear.forward`, so the input is never quantized either.

The loader package installs a runtime arm that fixes both (ComfyUI core is **not**
modified): `ComfyUI-HSWQ-Loader-and-Tools/patches/clip_int8_gemm/arm.py`. It makes a
quantized caller get `full_precision_mm=False`, and routes a `TensorWiseINT8Layout`
weight into `int8_linear` (with the online ConvRot activation rotation) instead of
dequantizing it. The arm also completes a missing `text_projection`, which the
INT8 GEMM path reaches with a `None` weight.

Measured A/B on CLIP ViT-L/14, same process and inputs, batch 1:

| path | latency | int8_linear calls | cos mean / min |
| :--- | --- | --- | --- |
| FP16 reference | 12.96 ms | - | - |
| INT8 pack, stock path | 24.41 ms | 0 | 0.999796 / 0.999727 |
| **INT8 pack, with the arm** | **16.92 ms (-31 %)** | **1971** | 0.999718 / 0.999617 |

Acceptance set with the arm enabled (256 prompts, seed 42): mean **0.999717**,
min **0.999483**, p5 0.999610 - PASS (mean >= 0.99, min >= 0.98).

Int8 GEMM is confirmed active for every shipped CLIP pack: `clip_l` 70 calls,
`ViT-L-14-BEST-smooth-GmP` 64, `Long-ViT-L-14` 73.

---

## 5. Loading the pack

### 5.1 Text tower (standard loader)

```python
sd, meta = comfy.utils.load_torch_file(pack, safe_load=True, return_metadata=True)
mo = {"quantization_metadata": json.loads(meta["_quantization_metadata"])}
clip = comfy.sd.load_clip(ckpt_paths=[pack], embedding_directory=None,
                          clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION, model_options=mo)
```

`text_projection.weight` is present in this checkpoint, so no completion is needed.

### 5.2 Vision tower (needs `mixed_precision_ops`)

The stock `CLIPVisionLoader` builds the tower with `manual_cast`, which does **not**
consume `comfy_quant`. Construct it with `mixed_precision_ops`:

```python
from comfy import ops as O
from comfy.quant_ops import QUANT_ALGOS
ops = O.mixed_precision_ops({"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]},
                            torch.float16, full_precision_mm=True)
model = IMAGE_ENCODERS["clip_vision_model"](config, dtype, device, ops)
model.load_state_dict({k: v for k, v in sd.items() if k.startswith("vision_model.")},
                      strict=False)
```

`class_embedding` is present in this checkpoint, so no completion is needed.

---

## 6. Reproduce

```bash
# 1) measure text drift (73 layers)
python sdxl/diag_impact_clip.py --model Long-ViT-L-14-BEST-GmP-smooth-ft-full.safetensors \
  --out impact/impact_long_vitl14_text.json --comfy_path <ComfyUI> --full
# 2) measure vision drift (144 layers)
python sdxl/diag_impact_clip_vision.py --model Long-ViT-L-14-BEST-GmP-smooth-ft-full.safetensors \
  --out impact/impact_long_vitl14_vision.json --comfy_path <ComfyUI> --full
# 3) build (thresholds text=1.95 vision=0.90)
python sdxl/build_long_vitl14_convrot_int8.py --thr_t 1.95 --thr_v 0.90 --out <pack>
# 4) verify (256 items each, seed 42)
python sdxl/verify_long_vitl14.py --pack <pack> --n 256 --seed 42 \
  --out impact/verify_long_vitl14.json
```

Expected: text mean >= 0.9997 / min >= 0.9996; vision mean >= 0.9997 / min >= 0.9991.

---

## 7. Summary

| Aspect | Choice |
| :--- | :--- |
| Target | Long-ViT-L-14-BEST-GmP-smooth-ft-full (FP32, text + vision, 248-token text context) |
| Quantized | **both towers**: text 74 + vision 144 = 220 INT8 |
| Protection criterion | Measured drift of each tower's own output |
| Thresholds | text **1.95**, vision **0.90** (both above the measured maxima) |
| Acceptance fidelity | text mean 0.999809 / min 0.999687; vision mean 0.999772 / min 0.999210 |
| Size | 1631.80 MB -> **413.20 MB** (-74.7 %) |
| Latency (text) | stock path 1.7-2.2x slower than FP32; **with the loader's INT8-GEMM arm: 24.41 -> 16.92 ms (-31 %)**, int8_linear 1971 calls |
| Latency (vision) | 0.99x vs FP32 from batch 4 up |
| Load | text: standard `CLIPLoader` + metadata; vision: `mixed_precision_ops` required |

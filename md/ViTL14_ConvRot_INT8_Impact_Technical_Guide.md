# ViT-L-14-BEST-smooth-GmP ConvRot INT8 — Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`How to quantize Text Encoder and ControlNet.md`](How to quantize Text Encoder and ControlNet.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this model.

**Target**: `ViT-L-14-BEST-smooth-GmP-HF-format.safetensors` — the OpenAI CLIP
ViT-L/14 checkpoint in Hugging-Face layout, carrying **two towers**
(`text_model.*`, `vision_model.*`).

**What this document covers**: how the checkpoint is quantized to ConvRot INT8
with a **measured** FP16 protection set covering **both towers**, how the pack
is laid out, how it loads, and the measured fidelity / size of the result.

- Source: `models/clip/ViT-L-14-BEST-smooth-GmP-HF-format.safetensors`
  (888.30 MB, SHA256 `46A201AFAE2516C95B6D0CC798ACA637161F07B8D0D7DC7776FB6F691F5476F1`)
- Pack: `models/clip/ViT-L-14-BEST-smooth-GmP-HF-format_hswq_convrot_int8.safetensors`
  (646.70 MB, **−27.2 %**, SHA256 `B299DFE7A12F96011FB1351669A9720F4ACDC31ACD81E56CD921144D593395DB`)

---

## 0. The checkpoint: two towers, two load paths

| tower | 2D weight modules | ComfyUI loader | operations used |
| :--- | :--- | :--- | :--- |
| `text_model.*` | 74 | `CLIPLoader` → `SD1ClipModel` | `mixed_precision_ops` when `quantization_metadata` is supplied, else `manual_cast` |
| `vision_model.*` | 145 | `CLIPVisionLoader` → `ClipVisionModel` | **always `manual_cast`** (hard-coded in `ClipVisionModel.__init__`) |

This asymmetry is the single most important fact in this document. `comfy_quant`
is consumed by `mixed_precision_ops`, so a pack loaded through the stock
`CLIPVisionLoader` keeps its quantized tensors as **int8 storage** and the
forward crashes or silently produces garbage. The vision tower has to be
constructed with `mixed_precision_ops` for the pack to work (section 5.2).

The checkpoint stores 581 of 586 tensors in FP16 and 5 in FP32
(`text_model.embeddings.position_embedding`, `text_projection`,
`vision_model.embeddings.position_embedding`, `vision_model.pre_layrnorm.{weight,bias}`).

---

## 1. Method: impact-driven, per tower

Each 2D Linear is replaced by its ConvRot INT8 reconstruction **alone**, and the
tower's own output is compared against the FP16 source's output.

| tower | the observable | the candidates | the script |
| :--- | :--- | :--- | :--- |
| the text tower | the text embedding for a fixed prompt set | 73 | `sdxl/diag_impact_clip.py` |
| the vision tower | the image embedding for a fixed image set | 144 | `sdxl/diag_impact_clip_vision.py` |

The impact measurement's names are the **runtime's**
(`transformer.text_model.encoder.layers.N.*` for the text tower,
`encoder.layers.N.*` for the vision tower); the pack is written under the
**checkpoint's** names (`text_model.*`, `vision_model.*`), which is what
ComfyUI's loader reads.

### 1.0 The injection frame (the defect that was fixed)

The earlier measurement injected the reconstruction in the **rotated** frame
(`q * s`, i.e. `W_hat * H`) instead of the **original** one. The pack stores
the rotated weight and the kernel rotates the **activation**, so the
quantization error that reaches the output is `(q*s) H^T − W`. Injecting the
rotated frame into a plain linear measures the un-rotated int8 error plus the
rotation — a completely different quantity.

### 1.1 Measured drift distribution, corrected

| | the old (defective) | the new (corrected) |
| :--- | :--- | :--- |
| the text tower's median | 1.53e-01 | **2.26e-06** |
| the text tower's max | 1.93e+00 | **2.27e-05** |
| the vision tower's median | 2.24e-01 | **4.37e-06** |
| the vision tower's max | 1.25e+00 | **4.27e-05** |

The old measurement made the text tower's
`transformer.text_model.encoder.layers.0.self_attn.v_proj` look like 1.45 and
the vision tower's worst layer like 1.25. The corrected values are five orders
of magnitude smaller, and with them **the ranks** — which is what makes the
old protection set meaningless.

---

## 2. The selection criterion, and the rebuilt pack

The per-layer drift is a **screen**, not the selection criterion. It measures
the **weight** error only (the reconstruction is placed in the module and the
plain forward is used), while the pack's real path also quantizes the
**activation** for that layer, and that is where most of the error comes from.
The FP16-kept set is therefore given explicitly, with the layers taken in the
corrected drift's ranking, separately for the two towers.

The candidates, measured on a fixed 16-item set (seed 42) against the FP16
source, on the runtime's own path:

| the candidate | the text kept | the vision kept | the INT8 | text cos (mean / min) | vision cos (mean / min) | size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **the previous pack** (256 items) | (the old threshold set) | | 164 | 0.999914 / 0.999851 | 0.999451 / 0.996486 | 570.3 MB |
| t0_v0 | 0 | 0 | 217 | 0.999876 / 0.999834 | 0.998823 / 0.996767 | 410.4 MB |
| t10_v20 | 10 | 20 | 187 | 0.999921 / 0.999891 | 0.999252 / 0.997871 | 486.3 MB |
| t20_v40 | 20 | 40 | 157 | 0.999947 / 0.999916 | 0.999603 / 0.998897 | 573.5 MB |
| **t30_v60** | **30** | **60** | **127** | **0.999963 / 0.999927** | **0.999850 / 0.999788** | **646.7 MB** |
| t50_v100 | 50 | 100 | 67 | 0.999986 / 0.999973 | 0.999965 / 0.999922 | 722.1 MB |

**30 text + 60 vision FP16-kept layers were adopted.** 50/100 buys +2.3e-05
of the text mean and +1.2e-04 of the vision mean for 75 MB more; the extra
size is better spent on the remaining models.

Note the **vision tower** is much more sensitive than the text tower here: at
t20_v40 the text tower is already at 0.999947 while the vision tower is only
at 0.999603. Protecting the vision tower is what the size buys.

### 2.1 The kept tensors are stored in FP16

The previous revision kept the unquantized tensors in **FP32**. The checkpoint
is FP32 for 5 of its 586 tensors, so those 5 came back as FP32 in the pack.
That is the wrong storage width: ComfyUI's CLIP runs in FP16, so an FP32 kept
tensor is cast to FP16 when the model is built and nothing at runtime sees the
extra precision.

All floating-point kept tensors are stored FP16 now, and the pack is
correspondingly smaller than a FP32-kept build of the same layer set would be.
The INT8 scales stay FP32 (they carry the dequantization precision, and FP16
would round the step size itself).

---

## 3. Pack layout

| item | value |
| :--- | :--- |
| the source | 888.30 MB (FP16 except 5 tensors) |
| the pack | **646.70 MB** (-27.2 %) |
| the SHA256 | `B299DFE7A12F96011FB1351669A9720F4ACDC31ACD81E56CD921144D593395DB` |
| the tensors | 841 |
| the quantized layers | **127** (the text 43, the vision 84) |
| the FP16-kept layers | 92 (the text 30, the vision 60) |
| the group size | 256 (all 127 layers) |
| the output | finite |

Per quantized Linear: `<name>.weight` (int8), `<name>.weight_scale` (FP32,
one row-wise scale over each 256-wide group), `<name>.comfy_quant` (uint8 JSON
`{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}`).

---

## 4. Measured results

### 4.1 Embedding fidelity (acceptance set, 256 fixed items, seed 42)

| tower | mean | min | p5 | p50 | gate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| text | **0.999964** | **0.999906** | 0.999946 | 0.999967 | PASS (mean ≥ 0.99, min ≥ 0.98) |
| vision | **0.999803** | **0.996378** | 0.999603 | 0.999852 | PASS |

Against the previous pack:

| tower | the previous pack | the rebuilt pack |
| :--- | :--- | :--- |
| the text mean / min | 0.999914 / 0.999851 | **0.999964 / 0.999906** |
| the vision mean / min | 0.999451 / 0.996486 | **0.999803 / 0.996378** |

Both packs are measured on the **same** 256-item set (seed 42) so the rows are
directly comparable. The text tower's fidelity was already high — the old
threshold set happened to protect it well — and the rebuilt pack still
improves it (+5.0e-05 of the mean, +5.5e-05 of the min). The **vision tower**
is where the old set fails: the mean is 0.999451 against 0.999803, a 3.5×
larger error, because its drift ranking was inverted and it protected the
wrong layers. The rebuilt pack's vision **minimum** is 0.996378 against
0.996486 — a 3 % larger worst-case error, against a 3.5× improvement in the
mean over the set: the old set bought a slightly better single worst item by
giving up the average, which is the wrong trade for a vision tower that feeds
every image through the same graph.

### 4.2 Downstream smoke test

`sdxl/smoke_vitl14.py` loads both towers through ComfyUI's own loaders and
runs one forward each; the pack must load without error and produce finite
output.

---

## 5. Loading the pack

### 5.1 Text tower (standard loader works)

`CLIPLoader` → `comfy.sd.load_clip` → `SDClipModel`. To activate INT8, the
pack's `_quantization_metadata` must reach
`model_options["quantization_metadata"]`; then `SDClipModel` builds
`mixed_precision_ops` and consumes the `comfy_quant` stamps.

```python
sd, meta = comfy.utils.load_torch_file(pack, safe_load=True, return_metadata=True)
mo = {"quantization_metadata": json.loads(meta["_quantization_metadata"])}
clip = comfy.sd.load_clip(ckpt_paths=[pack], embedding_directory=None,
                          clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION, model_options=mo)
```

`text_projection` is present in this checkpoint (`[768, 768]`, FP32), so no
completion step is needed on this path. (The `transformer.text_model.*`
namespace reports it as a bare `text_projection`; `SDClipModel` reads it
directly.)

### 5.2 Vision tower (needs `mixed_precision_ops`)

The stock `CLIPVisionLoader` uses `manual_cast` and will **not** quantize.
Construct the tower with `mixed_precision_ops`:

```python
from comfy import ops as O
from comfy.quant_ops import QUANT_ALGOS
ops = O.mixed_precision_ops({"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]},
                            torch.float16, full_precision_mm=True)
model = IMAGE_ENCODERS["clip_vision_model"](config, dtype, device, ops)
```

`vision_model.embeddings.class_embedding` is absent in the checkpoint; complete
it with zeros (dim 1024) when loading the vision tower standalone.

---

## 6. Reproduce

```bash
# 1) measure text drift (73 layers)
python sdxl/diag_impact_clip.py --model <vitl>.safetensors \
  --out impact/impact_vitl14_best_text_fixed.json --comfy_path <ComfyUI> --seed 42
# 2) measure vision drift (144 layers)
python sdxl/diag_impact_clip_vision.py --model <vitl>.safetensors \
  --out impact/impact_vitl14_best_vision_fixed.json --comfy_path <ComfyUI> --seed 42
# 3) build the pack (the FP16-kept set given explicitly, per tower)
python sdxl/build_vitl14_convrot_int8.py \
  --impact-text impact/impact_vitl14_best_text_fixed.json \
  --impact-vision impact/impact_vitl14_best_vision_fixed.json \
  --protect-list impact/vitl14_best_protect_t30_v60.json --out <pack>
# 4) verify (256 items, seed 42, both towers)
python sdxl/verify_vitl14_best.py --pack <pack> --n 256 --seed 42 \
  --out impact/verify_vb_final.json
# 5) downstream smoke
python sdxl/smoke_vitl14.py
```

---

## 7. Summary

| item | value |
| :--- | :--- |
| source | 888.30 MB |
| pack | **646.70 MB (-27.2 %)** |
| INT8 layers | 127 of 217 eligible (the text 43, the vision 84) |
| FP16-kept layers | 92 (the text 30, the vision 60), stored FP16 |
| group size | 256 |
| text cos (256 items, seed 42) | **0.999964 / 0.999906** |
| vision cos (256 items, seed 42) | **0.999803 / 0.996378** |
| the gate | **PASS** |

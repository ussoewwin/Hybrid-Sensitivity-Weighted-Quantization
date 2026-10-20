# ViT-L-14-BEST-smooth-GmP ConvRot INT8 - Runbook

Step-by-step reproduction, verification, and rollback for the ConvRot INT8
quantization of `ViT-L-14-BEST-smooth-GmP-HF-format.safetensors`.

---

## 1. Environment

| item | value |
| :--- | :--- |
| OS | Windows |
| Python | `D:\USERFILES\ComfyUI\python_embeded\python.exe` |
| ComfyUI | `D:\USERFILES\ComfyUI\ComfyUI` |
| GPU | NVIDIA RTX 5060 Ti (CC 12.0, FP4 TC) |
| key packages | torch (cu13), safetensors, comfy_kitchen |

No network access is required: the acceptance set is generated deterministically.

---

## 2. Inputs

| file | role |
| :--- | :--- |
| `models\clip\ViT-L-14-BEST-smooth-GmP-HF-format.safetensors` | FP16 source (kept, never overwritten) |
| `impact\impact_vitl14_best.json` | text-tower drift (73 layers) |
| `impact\impact_vitl14_best_vision.json` | vision-tower drift (144 layers) |

---

## 3. Build

The shipped pack is produced with thresholds text=1.30, vision=0.50.

```powershell
$py = "D:\USERFILES\ComfyUI\python_embeded\python.exe"
# 1) measure text drift (if regenerating the impact JSON)
& $py sdxl\diag_impact_clip.py --model <vitl>.safetensors `
    --out impact\impact_vitl14_best.json --comfy_path D:\USERFILES\GitHub\hswq\ComfyUI-master --full
# 2) measure vision drift
& $py sdxl\diag_impact_clip_vision.py --model <vitl>.safetensors `
    --out impact\impact_vitl14_best_vision.json --comfy_path D:\USERFILES\GitHub\hswq\ComfyUI-master --full
# 3) build the pack (thresholds text=1.30 vision=0.50)
```

The build script writes `.weight_scale` / `.comfy_quant` for every 256-wide 2D
Linear/Embedding layer that is not protected, and `_quantization_metadata`.

---

## 4. Verification

```powershell
# text: 256 fixed prompts, seed 42
& $py <tmp>\verify_text.py --n 256 --seed 42 --out impact\verify_vitl14_text.json
# vision: 256 fixed images, seed 42
& $py <tmp>\verify_vision.py --n 256 --seed 42 --out impact\verify_vitl14_vision.json
```

Expected values:

| tower | mean | min | verdict |
| :--- | :--- | :--- | :--- |
| text | >= 0.9999 | >= 0.9998 | PASS |
| vision | >= 0.9990 | >= 0.9960 | PASS |

Acceptance: mean >= 0.99 AND min >= 0.98 for both towers.

---

## 5. Downstream smoke test

```powershell
& $py <tmp>\smoke_vitl.py
```

Expected: text encode (1,77,768) finite; vision encode (3,1024) finite;
text_cos_vs_fp16 >= 0.9999, vision_cos_vs_fp16 >= 0.999.

---

## 6. Load paths (why both towers work)

| tower | loader | operations | consumes comfy_quant |
| :--- | :--- | :--- | :--- |
| text | `comfy.sd.load_clip` (SD1ClipModel) | `mixed_precision_ops` (auto, when `model_options["quantization_metadata"]` is set) | yes |
| vision | `ClipVisionModel` | must be constructed with `mixed_precision_ops`; the stock `CLIPVisionLoader` uses `manual_cast` and would read int8 raw | yes (with mixed ops) |

**Important.** The stock `CLIPVisionLoader` builds the tower with `manual_cast`,
which does **not** consume `comfy_quant`. To run the quantized vision tower, build
it with `mixed_precision_ops` (as the smoke/verify scripts do).

---

## 7. Rollback

The FP16 source is never modified. To roll back:

```powershell
Remove-Item "D:\USERFILES\ComfyUI\ComfyUI\models\clip\ViT-L-14-BEST-smooth-GmP-HF-format_hswq_convrot_int8.safetensors"
```

The FP16 file remains usable as before. No other file is touched.

---

## 8. Known constraints

- Latency is **higher** for the quantized model on this machine (per-op
  dequant/rotation overhead); the win is file size / weight memory.
- The vision tower requires `mixed_precision_ops`; a stock `CLIPVisionLoader` will
  not quantize it.
- `text_projection` is absent in the checkpoint; the text path needs it completed
  (zeros) or it crashes under mixed ops.
- `vision_model.embeddings.class_embedding` is absent; complete it (zeros, dim 1024)
  when loading the vision tower standalone.

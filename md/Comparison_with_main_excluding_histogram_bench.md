# Comparison with Hybrid-Sensitivity-Weighted-Quantization-main (excluding histogram and bench)

**Source:** `D:\USERFILES\GitHub\hswq\Hybrid-Sensitivity-Weighted-Quantization-main` (main)  
**Target:** `D:\USERFILES\GitHub\hswq` (hswq, after 6cc49ca)

Changes to the histogram (weighted_histogram_mse) and bench (zit_bench) are excluded.

---

## 1. quantize_zit_hswq_v1.5.py

| Item | main | hswq (after 6cc49ca) | Conclusion |
|------|------|----------------------|------------|
| **venv sys.path addition** | None (import numpy right after ComfyUI-master) | None | Removed in 6cc49ca → **matches main** (was only on hswq before) |
| **encode_prompt empty prompt / 0-token guard** | None | None | Removed in 6cc49ca → **matches main** (was only on hswq before) |
| **SageAttention2 except** | `except ImportError:` (no e) | Same | **matches main** |
| **__call__ text_encoder** | No offload (no to(device)/cpu()) | No offload | **matches main** (6cc49ca removed hswq-side offload) |
| **Post-calib gc/empty_cache** | Every 10 samples | Same | **matches main** |
| **VRAM optimization block** | None (goes straight from "Layers to quantize" to "Saving quantized model") | None | Removed in 6cc49ca → **matches main** (hswq had "del pipeline, load all keys to GPU") |
| **Conversion loop** | "Converting weights...", STRIPPED, .weight only, keep→to(float16), bfloat16 in else to(float16) | Same | **matches main** |
| **save_file** | Direct `save_file(output_state_dict, args.output)` | Same | 6cc49ca removed try/except CPU fallback → **matches main** |

→ **All "other" changes in quantize_zit_hswq_v1.5 are to align with main.**

---

## 2. Flux (quantize_flux_hswq_v1.2, v1.6, archives)

| Item | main | hswq (after 6cc49ca) | Conclusion |
|------|------|----------------------|------------|
| **V1.21 / V1.2 labeling** | Always "V1.2" | 6cc49ca fixed "V1.21"→"V1.2" | **matches main** (typo fix on hswq) |

→ **Flux changes are labeling typo fixes only, aligned with main.**

---

## 3. md/How to quantize SDXL.md

| Item | main | hswq (after 6cc49ca) | Conclusion |
|------|------|----------------------|------------|
| **Example paths** | Placeholders `<path-to-unet>`, `<output-dir>` | Same | **matches main** (6cc49ca changed hswq absolute path examples to placeholders) |

---

## 4. md/SDXL_V1.3_and_Histogram_Fast_Explanation.md

| Item | main | hswq (after 6cc49ca) | Conclusion |
|------|------|----------------------|------------|
| **"6. Why Precision Is Preserved"** | Present (as section 6) | Present (same intent as 6cc49ca addition) | Both have similar explanation; hswq added/expanded that section in 6cc49ca. |

---

## Summary (excluding histogram and bench)

- **quantize_zit_hswq_v1.5.py**  
  Removing venv, encode_prompt guard, offload, VRAM optimization block, save try/except, etc., are all **changes that drop hswq-only behavior and align with main**.
- **Flux**  
  **V1.21 → V1.2** typo fix only; main was already V1.2.
- **md**  
  How to quantize SDXL: path examples aligned with main placeholders. SDXL_V1.3: main also has "Why Precision Is Preserved"; hswq added/cleaned section 6 to match.

→ **"Other" changes are either alignment fixes with main or adding the same kind of explanation as main.**

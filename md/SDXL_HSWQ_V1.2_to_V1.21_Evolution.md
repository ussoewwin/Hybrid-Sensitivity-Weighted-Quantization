# SDXL HSWQ V1.2 Update — Code Evolution and Technical Explanation

**Current SDXL script:** `quantize_sdxl_hswq_v1.3.py` (Fast histogram; see [SDXL_V1.3_and_Histogram_Fast_Explanation.md](SDXL_V1.3_and_Histogram_Fast_Explanation.md)). V1.2 is archived at `archives/quantize_sdxl_hswq_v1.2.py`.

This document explains what changed in the **V1.2 update**: the script `quantize_sdxl_hswq_v1.2.py` began using GPU-accelerated conversion. The earlier CPU version is archived at `archives/quantize_sdxl_hswq_v1.2(old).py`.

---

## 1. Summary: Before vs After Update

| Aspect | Previous (archived) — `archives/quantize_sdxl_hswq_v1.2(old).py` | Updated — `quantize_sdxl_hswq_v1.2.py` |
|--------|-----------------------------------------------------------------|----------------------------------------|
| **Conversion** | CPU | **GPU-accelerated** |
| **Calibration** | Same: Dual Monitor, HSWQ optimizer, optional `--sa2` | Same |
| **Conversion device** | **CPU**: tensors stay on CPU; clamp/cast in Python on host | **GPU**: full state_dict moved to VRAM; clamp/cast on device |
| **Memory strategy** | Pipeline + optimizer kept until after conversion | Pipeline and optimizer **deleted** before conversion to free VRAM; then state_dict moved to GPU |
| **Save** | `save_file(output_state_dict, args.output)` only | Same, plus **fallback**: on save failure, move all tensors to CPU and save again |
| **Output** | Identical FP8/FP16 layout and values (same algorithm) | Identical; only execution path and speed differ |

So: the **updated** script = previous script + “run the quantization conversion on GPU and use VRAM aggressively.”

---

## 2. Algorithm Flow (Unchanged Part)

Both versions share the same high-level steps:

1. Load UNet from safetensors → `pipeline`, `original_state_dict`, `comfyui_to_diffusers_map`
2. Register Dual Monitor hooks on UNet (Conv2d/Linear)
3. Run calibration (N samples × M steps) → collect Sensitivity and Importance
4. Remove hooks; optionally disable SageAttention2
5. Layer sensitivity analysis → sort, choose top `keep_ratio` → `keep_layers`
6. Build `weight_amax_dict`: for each non–keep layer, run HSWQ optimizer (`compute_optimal_amax(..., scaled=False)`)
7. **Conversion loop**: for each key in `original_state_dict`, decide keep vs quantize; if quantize, clamp with `amax` and cast to `torch.float8_e4m3fn`
8. Save `output_state_dict` to safetensors

The **only** structural change in the **updated** script is **where** step 7 runs (GPU, with state_dict in VRAM) and **how** memory is managed before step 7 (drop pipeline/optimizer, then move state_dict to GPU).

---

## 3. Code Difference 1: VRAM Optimization Block (Updated Only)

**Updated** script inserts a block **after** building `weight_amax_dict` and **before** the conversion loop:

```python
# === VRAM Optimization Plan ===
# 1. Delete pipeline and optimizer to free VRAM
# 2. Move entire original_state_dict to GPU
# 3. Run conversion on GPU

print("\n[VRAM Optimization] Preparing for high-speed GPU conversion...")
del pipeline
del hswq_optimizer
gc.collect()
torch.cuda.empty_cache()

print(f"[VRAM Optimization] Moving source weights to {device}...")
input_keys = list(original_state_dict.keys())
for k in tqdm(input_keys, desc="Loading to VRAM"):
    original_state_dict[k] = original_state_dict[k].to(device)
```

**Meaning:**

- **Why delete `pipeline` and `hswq_optimizer`?**  
  The SDXL UNet and the HSWQ optimizer (with histograms, etc.) use a lot of VRAM. After we have computed `weight_amax_dict`, we no longer need the UNet or the optimizer. Freeing them allows the GPU to hold the full `original_state_dict` (SDXL UNet is on the order of 5–6 GB in FP16) without OOM.

- **Why move `original_state_dict` to GPU?**  
  So that in the conversion loop, every `value` is already on device. Then `torch.clamp(value, -amax, amax)` and `value.to(torch.float8_e4m3fn)` run on the GPU, which is much faster than doing the same on CPU for millions of elements.

- **Why replace in place with `original_state_dict[k] = original_state_dict[k].to(device)`?**  
  So we don’t build a second full copy in RAM; we move each tensor from CPU to GPU and drop the CPU reference, keeping peak CPU memory lower.

**Previous (archived)** has none of this: it keeps `pipeline` and `hswq_optimizer` in memory and never moves `original_state_dict` to GPU. The conversion loop therefore runs on **CPU** tensors.

---

## 4. Code Difference 2: Conversion Loop (Same Logic, Different Device)

The **logic** of the loop is identical in both versions:

- Resolve `diffusers_key` from `comfyui_to_diffusers_map`.
- Derive `module_name` (strip `.weight`).
- If `module_name` is in `keep_layers` → keep tensor as-is (FP16).
- Else if we have an `amax` for this weight → `clamped_value = torch.clamp(value, -amax, amax)`, then `new_value = clamped_value.to(torch.float8_e4m3fn)`.
- Put `new_value` into `output_state_dict[key]`.

**Previous (archived):**

```python
print("Converting weights...")
for key, value in tqdm(original_state_dict.items(), desc="Converting"):
    # ... resolve diffusers_key, module_name ...
    if weight_key in weight_amax_dict:
        amax = weight_amax_dict[weight_key]
        clamped_value = torch.clamp(value, -amax, amax)
        new_value = clamped_value.to(torch.float8_e4m3fn)
```

Here `value` comes from `load_file(path)` and is on **CPU**. So clamp and cast are done on CPU.

**Updated:**

```python
print("Converting weights (GPU accelerated)...")
for key, value in tqdm(original_state_dict.items(), desc="Converting"):
    # ... same resolution ...
    if weight_key in weight_amax_dict:
        amax = weight_amax_dict[weight_key]
        clamped_value = torch.clamp(value, -amax, amax)
        new_value = clamped_value.to(torch.float8_e4m3fn)
```

Here `value` has already been moved to GPU in the VRAM optimization block. So clamp and cast run on **GPU**. The comment “GPU Accelerated” reflects that.

**Meaning:**  
Same formulas (clamp by `amax`, cast to FP8 E4M3). Only the execution device changes; numerical results are the same. GPU execution reduces conversion time significantly for a model the size of SDXL.

---

## 5. Code Difference 3: Save and GPU Fallback (Updated Only)

**Previous (archived):**

```python
print("Conversion done:")
print(f"  FP8 layers: {converted_count}")
print(f"  FP16-kept layers: {kept_count}")
save_file(output_state_dict, args.output)
print("Saved.")
```

**Updated:**

```python
print("Conversion done:")
# ...
try:
    save_file(output_state_dict, args.output)
except Exception as e:
    print(f"[Save Warning] GPU Tensor save failed ({e}). Moving to CPU explicitly...")
    cpu_dict = {k: v.cpu() for k, v in output_state_dict.items()}
    save_file(cpu_dict, args.output)
print("Saved.")
```

**Meaning:**

- In the updated script, `output_state_dict` contains **GPU** tensors. Some versions of `safetensors`/I/O paths may not handle GPU tensors correctly, or the environment may have restrictions.
- The try/except ensures that if `save_file` fails (e.g. because of GPU tensors), we explicitly move every tensor to CPU and save again. So we trade a bit of extra time and CPU memory for robustness.
- Functionally, the saved file is the same (FP8 + FP16 mix, same keys and layout).

---

## 6. Other Differences (Comments and User-Facing Text)

- **Comment in DualMonitor:** Both say "Accumulate in FP32/Double to avoid overflow". Logic is identical.
- **enable_sage_attention:** Updated adds an extra guard `if _original_sdpa is not None: return True` to avoid double-patching; previous does not. Small robustness improvement.

None of these change the algorithm or the output.

---

## 7. When to Use Which

- **Previous (archived):** Use if you want to avoid moving the full state_dict to GPU (e.g. very limited VRAM, or you prefer CPU-bound conversion). Output is the same as updated.
- **Updated:** Use for normal runs where you have enough VRAM (e.g. 8GB+ free after calibration). Faster conversion and same FP8/FP16 result.

---

## 8. File Reference

| File | Role |
|------|------|
| `quantize_sdxl_hswq_v1.3.py` (repository root) | **Current**: Fast histogram, GPU conversion, VRAM optimization. |
| `archives/quantize_sdxl_hswq_v1.2.py` | **Archived**: GPU conversion (same as V1.2 update), original histogram module. |
| `archives/quantize_sdxl_hswq_v1.2(old).py` | **Archived**: CPU conversion, no VRAM optimization block. |

All produce the same FP8 + FP16 mix; V1.3 only changes amax computation to the Fast histogram module.

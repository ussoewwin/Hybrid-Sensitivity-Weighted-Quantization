# How to create Hybrid NVFP4 from ConvRot INT8 (Z Image, Reverse Method)

> **Prerequisite**: The input **ConvRot INT8 model** (`<model>_sci_1off_convrot_int8.safetensors`) is
> already created by `native_convert_int8_convrot_zi.py` as described in
> [How to quantize Z Image.md](How%20to%20quantize%20Z%20Image.md).
> This document is the next step: turn that complete INT8 model into a **hybrid NVFP4** model.

This method is **fundamentally different** from the conventional "protect the top-important layers"
approach (histogram MSE / cosine / SVD saliency). It is a **reverse method**: start from the complete
ConvRot INT8 model (error ≈ 0) and convert layers to NVFP4 **in ascending order of per-layer impact**.
The conventional method ignores inter-layer interactions and is not sufficient for this hybrid.
The reverse method stays in the low-error regime where additivity holds, so **single-layer ranking is
valid**. Pass only if **every seed** of the native bench meets **SSIM (0-255 view) ≥ 0.97**.

Replace every `<...>` placeholder with a real path on your machine (no invented filenames; no
machine-local drive hardcoding in published examples).

## Clone the repository

```bash
git clone https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization.git
cd Hybrid-Sensitivity-Weighted-Quantization
```

## Install PyTorch (CUDA)

First, install PyTorch (CUDA).
In a Windows environment on a local PC, it is advisable to set up a venv virtual environment.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

## Install other libraries

```bash
pip install safetensors tqdm scikit-image comfy-kitchen
```

`scikit-image` is required for SSIM in the native bench. `comfy-kitchen` is required by
`Z_Image/gen_reverse_nvfp4.py` (`TensorCoreNVFP4Layout`).

## Requirements

| Item | Value |
|---|---|
| Python | CUDA-enabled PyTorch, `safetensors`, `scikit-image`, `comfy-kitchen` |
| Bench | `benchmark/zi_convrot_nvfp4_bench_native.py` (this repo, **unmodified**) |
| ComfyUI | local ComfyUI checkout (Z-Image loading / Qwen3-4B text encoder) |
| Input ① | `<path-to-unet>/<model>.safetensors` (base fp16/bf16 NextDiT) |
| Input ② | `<path-to-unet>/<model>_sci_1off_convrot_int8.safetensors` (208-layer ConvRot INT8 from the INT8 how-to; int8_tensorwise, convrot:true, `model.diffusion_model.` prefix) |
| GPU | **VRAM ≥ 12GB** · **run one process at a time** (concurrent runs cause VRAM exhaustion) |

## Overall flow

```
ConvRot INT8 (208 layers, error ≈ 0)
   │  Step 1: per-layer impact measurement (208 layers × 4-step trajectory, ≈12 min)
   ▼
impact_<model>.json (impact per layer, ascending)
   │  Step 2: reverse conversion (convert K lowest-impact layers to NVFP4, ≈1 min)
   ▼
<model>_hswq_hybrid_nv{K}_convrot_nvfp4.safetensors
   │  Step 3: native bench (existing, unmodified, all 5 seeds)
   ▼
Pass only if every seed's SSIM (0-255 view) ≥ 0.97
```

`K` is **not** a fixed number. Measure impact, convert, then **search K** until the largest value
that still passes all 5 seeds.

---

## Step 1. Per-layer impact measurement (all 208 layers, ≈12 min)

Inject NVFP4 error (**e4m3, group 256 reconstruction**) into **one layer at a time**, run a fixed-seed
4-step denoising trajectory, and measure how far the final x drifts (relative MSE).
This is the layer's **true importance under real trajectory propagation**.

Run from the **repository root**:

```bash
python Z_Image/diag_impact.py "<path-to-unet>/<model>.safetensors" \
  "<path-to-unet>/<model>_sci_1off_convrot_int8.safetensors" \
  "impact_<model>.json" \
  --comfy-path "<path-to-ComfyUI>" --repo-root "<this-repo-root>"
```

**Output** `impact_<model>.json` → `{"x_ref_norm": ..., "impacts": {<layer>: <relative MSE>, ...}}`

Typical ranking tendencies (always re-measure per checkpoint; ranking is not transferable):

- **Smallest (safest to convert)**: `noise_refiner.*.attention.qkv`-class layers
- **Largest (must protect)**: `t_embedder.mlp.2`, `final_layer.linear`, `final_layer.adaLN_modulation.1`

---

## Step 2. Reverse conversion (≈1 min)

Convert the **K** layers with the smallest impact to NVFP4, in ascending impact order.

The INT8 weights are stored **already rotated (W@H^T)**. Dequant (`q × scale`) gives the rotated
W_rot approximation. Quantize that with Kitchen **without re-rotating**.

Run from the **repository root**:

```bash
python Z_Image/gen_reverse_nvfp4.py <K> \
  "<model>_hswq_hybrid_nv<K>_convrot_nvfp4.safetensors" \
  "<path-to-unet>/<model>_sci_1off_convrot_int8.safetensors" \
  "impact_<model>.json" \
  --out-dir "<path-to-unet>"
```

What this does:

1. Rank layers by `impact_<model>.json` **ascending** (lowest impact first).
2. For each of the first **K**: INT8 dequant (`q × scale` → rotated W_rot) → Kitchen `TensorCoreNVFP4Layout` NVFP4 (`format` nvfp4, `convrot` true, `groupsize` 256).
3. Replace those layers with `.weight` (U8 packed) / `.weight_scale` (F8_E4M3) / `.weight_scale_2` (F32) / `.comfy_quant`.
4. Leave the remaining layers as INT8 (keys and conf unchanged). Result: **(208 − K) INT8 + K NVFP4**.

**On-disk format of converted layers:**

- `.weight` (U8 packed `[out, in/2]`) + `.weight_scale` (F8_E4M3 `[out, in/16]`) + `.weight_scale_2` (F32) + `.comfy_quant` (conf as a **U8 tensor**)
- conf: `{"format": "nvfp4", "convrot": true, "convrot_groupsize": 256}` — every converted layer must have `convrot: true` and groupsize 256
- Weights are **stored rotated** (a large dequant-vs-fp16 deviation is expected)

---

## Step 3. Native bench (existing, unmodified, all 5 seeds)

The native bench **must be run from `benchmark/`** (it imports sibling modules `kitchen_rms_rope_fallback.py`, `nvfp4/`, `nvfp4_comfy_parity.py`).

**`--steps 25` and `--native-dtype` are required.** The script default is `--steps 20` and `--native-dtype` off — those defaults will **not** match this procedure.

```bash
cd "<this-repo-root>/benchmark"
python zi_convrot_nvfp4_bench_native.py \
  --fp16 "<path-to-unet>/<model>.safetensors" \
  --nvfp4 "<path-to-unet>/<model>_hswq_hybrid_nv<K>_convrot_nvfp4.safetensors" \
  --clip_path "<path-to-qwen3-4b>" \
  --comfy_path "<path-to-ComfyUI>" \
  --prompt "A beautiful cyberpunk city at night, high detail." \
  --steps 25 --seed 42 --native-dtype
```

Then rerun the **same command** with `--seed` set to **123**, **777**, **2024**, and **999**.

- Pass/fail is **per-seed** `SSIM (0-255 view) ≥ 0.97`. Do not pass on the average alone.
- MSE is informational.
- **The bench is fully deterministic** (identical results on re-run). Variation across seeds is a real model property, not GPU noise.
- `--token` is optional (Hugging Face). It is not required when the CLIP file is already local.

---

## Finding K

`K` is checkpoint-specific. The quality surface is often **not a single cliff**: failing seeds can
**change with K**, and quality can **recover then fail again** (error cancellation). Treat that as
the default search assumption.

1. **Judge on all 5 seeds individually ≥ 0.97** (an average above 0.97 is not a pass).
2. **Screen with a discriminating seed**, then run all 5 seeds only on candidates that pass.
   The discriminating seed is whichever seed fails first in the coarse sweep; it is not the same
   for every checkpoint.
3. **Assume islands and sweep the boundary in steps of 1.** A coarse 10-step sweep can miss the
   largest passing K.
4. **The bench is fully deterministic.** If a config fails, only changing the config can fix it.
5. **GPU discipline:** one process at a time; wait for completion. Concurrent runs exhaust VRAM.
   On Windows, killing only the child Python lets a parent shell loop spawn the next job — kill
   the whole tree: `taskkill /PID <parentPID> /T /F`.

Suggested order:

1. Coarse sweep (several K values) with seed 42 only, to locate the cliff/island roughly.
2. Screen the boundary band in **steps of 1** with the discriminating seed.
3. For a passing K, check **K±1–2** to confirm the top of the stable island.
4. For passing K only, run **all 5 seeds**, plus one re-run for reproducibility.
5. Ship the **largest K** that still passes all 5 seeds. Delete rejected intermediates.

Single-layer ranking chooses **which** layers to convert. It does **not** predict **how many**
layers are safe.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| 0 layers converted | impact key format mismatch (`.weight` suffix). Check the normalization in `gen_reverse_nvfp4.py` |
| `save_file` ValueError | `.comfy_quant` must be a **U8 tensor** (`torch.frombuffer(...).clone()`), not raw bytes |
| Bench CRITICAL ERROR (0 armed) | 0 NVFP4 layers. Regenerate with K ≥ 1 |
| SSIM stuck around 0.94 | Outside the island. Re-sweep K±1 around the discriminating seed |
| Numbers do not match a previous run | Confirm `--steps 25`, `--native-dtype`, cwd is `benchmark/`, and the five seeds above |
| Process won't die | Kill the whole parent tree: `taskkill /PID <parent> /T /F` (Windows) |
| Sudden drop as K increases | Cliff or island edge. Step back and re-check K±1 |

## Files in this repo

| File | Purpose |
|---|---|
| `Z_Image/diag_impact.py` | Step 1: per-layer NVFP4 trajectory impact measurement |
| `Z_Image/gen_reverse_nvfp4.py` | Step 2: reverse hybrid converter (INT8 → NVFP4, K lowest-impact layers) |
| `benchmark/zi_convrot_nvfp4_bench_native.py` | Step 3: native bench (bf16 native baseline, `--native-dtype`) |
| `native_convert_int8_convrot_zi.py` | INT8 prerequisite (see [How to quantize Z Image.md](How%20to%20quantize%20Z%20Image.md)) |

**Dependencies:** `Z_Image/diag_impact.py` loads the Z-Image model via `benchmark/zi_convrot_nvfp4_bench.py` (stays in `benchmark/`, shared with the older HSWQ scripts) — resolve it with `--repo-root <this-repo-root>`. `Z_Image/gen_reverse_nvfp4.py` needs the pip package `comfy-kitchen`. The native bench stays in `benchmark/` because it shares local modules with sibling bench scripts — **run it from `benchmark/`**.

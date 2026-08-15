# How to create Hybrid NVFP4 from ConvRot INT8 (Z Image, Reverse Method)

> **Prerequisite**: The input **ConvRot INT8 model** (`<model>_sci_1off_convrot_int8.safetensors`) is
> already created by `native_convert_int8_convrot_zi.py` as described in
> [How to quantize Z Image.md](How%20to%20quantize%20Z%20Image.md).
> This document covers the next step: turning that INT8 model into a **hybrid NVFP4** model.
>
> Replace every `<...>` placeholder with a real path on your machine (no invented filenames;
> no machine-local drive hardcoding in published examples).

This method is **fundamentally different** from the conventional "protect the top-important layers"
approach (histogram MSE / cosine / SVD saliency). It is a **reverse method**: start from the complete
ConvRot INT8 model (error ≈ 0) and convert layers to NVFP4 **in ascending order of per-layer impact**.
The conventional method ignores inter-layer interactions and proved insufficient (demonstrated on CE);
the reverse method stays in the low-error regime where additivity holds, so **single-layer ranking is
valid**. All 3 models achieved **SSIM ≥ 0.97 on all 5 seeds** with the existing unmodified native bench.

| Model | Final | INT8/NVFP4 | 5-seed range | Avg SSIM | Size |
|---|---|---|---|---|---|
| moodyProMix_collectorsEdition (CE) | nv89 | 119/89 | 0.9877–0.9956 | 0.9938 | 4.73GiB |
| moodyProMix_zitV13 | nv120 | 88/120 | 0.9882–0.9909 | 0.9897 | 4.29GiB |
| moodyRealMix_zitV7 | nv74 | 134/74 | 0.9774–0.9987 | 0.9889 | 4.82GiB |

## 1. Requirements

| Item | Value |
|---|---|
| Python | CUDA-enabled PyTorch, `safetensors`, `scikit-image` (SSIM) |
| Bench | `benchmark/zi_convrot_nvfp4_bench_native.py` (this repo, **unmodified**) |
| ComfyUI | local ComfyUI checkout (used by the bench for Z-Image loading / Qwen3-4B text encoder) |
| comfy_kitchen | pip package `comfy-kitchen` (`pip install comfy-kitchen`) — required by `gen_reverse_nvfp4.py` (`TensorCoreNVFP4Layout`) |
| Input ① | `<model>.safetensors` (base fp16/bf16) |
| Input ② | `<model>_sci_1off_convrot_int8.safetensors` (208-layer ConvRot INT8 from the INT8 how-to; int8_tensorwise, convrot:true, `model.diffusion_model.` prefix, ≈5.74GiB) |
| GPU | 1 card (VRAM ≥12GB recommended) · **run one process at a time** (concurrent runs cause VRAM exhaustion) |

## 2. Overall flow

```
ConvRot INT8 (208 layers, error ≈ 0)
   │  Step 1: per-layer impact measurement (208 layers × 4-step trajectory, ≈10–12 min)
   ▼
impact_<model>.json (impact per layer, ascending)
   │  Step 2: reverse conversion (convert K lowest-impact layers to NVFP4, ≈1 min)
   ▼
<model>_hswq_hybrid_nv{K}_convrot_nvfp4.safetensors
   │  Step 3: native bench (existing, unmodified, all 5 seeds)
   ▼
Step 4: island-structure search (screen with the discriminating seed → full 5 seeds only on pass)
   ▼
Step 5: pick the largest K with all-5-seed SSIM ≥ 0.97 as the final artifact; record everything
```

## 3. Step 1: Per-layer impact measurement (≈10–12 min)

Inject the NVFP4 quantization error (**e4m3, group 256 reconstruction**) into **one layer at a time**,
run a fixed-seed 4-step denoising trajectory, and measure how far the final x drifts (relative MSE).
This measures the layer's **true importance under real trajectory propagation**.

```bash
python diag_impact.py "<model>.safetensors" \
  "<model>_sci_1off_convrot_int8.safetensors" \
  "impact_<model>.json" \
  --comfy-path "<comfyui-root>" --repo-root "<this-repo-root>"
```

**Output** `impact_<model>.json` → `{"x_ref_norm": ..., "impacts": {<layer>: <relative MSE>, ...}}`

Measured tendencies (common to the 3 ZIT models):
- **Smallest (safest to convert)**: `noise_refiner.*.attention.qkv`-class layers (≈1.8e-7)
- **Largest (must protect)**: `t_embedder.mlp.2` (7.6e-5–9.7e-5), `final_layer.linear` (≈7.5e-5), `final_layer.adaLN_modulation.1`
- The layer ranking **differs between CE and V13/V7** — always measure per model.

## 4. Step 2: Reverse conversion (≈1 min per config)

Convert the **K** layers with the smallest impact to NVFP4, in ascending impact order.
The INT8 weights are stored **already rotated (W@H^T)**, so dequantizing gives the rotated
W_rot approximation directly — quantize it with Kitchen **without re-rotating**.

```bash
python gen_reverse_nvfp4.py 74 \
  "<model>_hswq_hybrid_nv74_convrot_nvfp4.safetensors" \
  "<model>_sci_1off_convrot_int8.safetensors" \
  "impact_<model>.json" --out-dir "<output-dir>"
```

**On-disk format of the artifact (verified):**
- NVFP4 layers: `.weight` (U8 packed [out, in/2]) + `.weight_scale` (F8_E4M3 [out, in/16]) + `.weight_scale_2` (F32) + `.comfy_quant` (conf as a **U8 tensor**)
- conf: `{"format": "nvfp4", "convrot": true, "convrot_groupsize": 256}` — verify every converted layer has convrot=true, groupsize 256
- Weights are **stored rotated** (the ≈2.25 deviation between dequant and fp16 is expected)
- The remaining (208−K) layers stay INT8 (keys and conf unchanged)

## 5. Step 3: Bench (existing, unmodified, ≈2 min per run)

```bash
cd <this-repo-root>
python benchmark/zi_convrot_nvfp4_bench_native.py \
  --fp16 "<model>.safetensors" \
  --nvfp4 "<model>_hswq_hybrid_nv{K}_convrot_nvfp4.safetensors" \
  --clip_path "<qwen3-4b-fp16-converted.safetensors>" \
  --comfy_path "<comfyui-root>" \
  --prompt "A beautiful cyberpunk city at night, high detail." \
  --steps 25 --seed 42 --native-dtype
```

- Switch `--seed` across **42 / 123 / 777 / 2024 / 999** (all 5 seeds)
- The pass/fail metric is `SSIM (0-255 view)` (MSE is informational)
- **The bench is fully deterministic** (identical results on re-run, confirmed ×3) — measured variation is a real model property, not GPU noise

## 6. Step 4: Sweep strategy (the core of this method)

### 6.1 Acceptance criterion

**All 5 seeds individually SSIM ≥ 0.97.** Averages are not enough (V7 nv90 averaged 0.9722 yet failed 3/5 seeds).

### 6.2 Island structure (monotonicity does not hold)

The boundary differs per model and is not a single cliff — it is an **island structure** where the failing
seed changes with K:

| Model | Boundary behavior |
|---|---|
| CE | Stable up to nv89; **layer 90** (layers.20.attention.out, impact 1.98e-6) collapses to 0.9496 |
| V13 | **Stable even at 140 layers** (high NVFP4 tolerance) |
| V7 | Stable island only at 70–74 (73/74 pass all 5 seeds); **75–90 all fail**, with partial islands at 76/77/80 inside |

V7 measured non-monotonic recovery (evidence of error cancellation):
`nv75 fail → nv76/77 recover → nv78 fail again → nv80 recover → nv81-90 all fail`

### 6.3 Efficient search procedure

1. **Coarse sweep** (K = 40, 90, 120 …) with seed 42 only, to locate the cliff/island roughly
2. **Screen with the discriminating seed**: the seed that fails most often (for V7: **seed 123**, concentrated at 0.954–0.956) — sweep the boundary band in **steps of 1**
3. For passing K, **check K±1–2** to confirm the top of the stable island
4. For passing K only, run **all 5 seeds**, plus one re-run for reproducibility

## 7. Step 5: Acceptance and record keeping

- ✅ All 5 seeds individually SSIM ≥ 0.97 (including the minimum)
- ✅ Reproducibility check (re-run the same config, expect identical values)
- ✅ Log per-seed SSIM/MSE/judgement in the experiment ledger
- ✅ Write a reproduction doc with commands, expected values, log paths
- ✅ Delete rejected intermediate configs, keep only the final artifact

## 8. Lessons learned (established 2026-08-16)

1. **Judge on all 5 seeds individually ≥0.97** (averages produce misleading passes)
2. **Single-layer impact ranking is valid for choosing WHICH layers to convert**, but **cannot predict HOW MANY layers are safe** (the boundary)
3. **Discriminating-seed screening → full 5 seeds only on pass** halves the search cost
4. **Assume non-monotonicity and island structure; sweep in steps of 1** (a 10-step sweep misses islands)
5. **The bench is fully deterministic** — if a config fails, only changing the config can fix it
6. **GPU discipline**: one process at a time, wait for completion
7. **Process cleanup**: killing the child python only lets the parent shell loop spawn the next one — kill the whole tree: `taskkill /PID <parentPID> /T /F` (Windows)

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| 0 layers converted | impact key format mismatch (`.weight` suffix). Check the normalization in the script |
| `save_file` ValueError | `.comfy_quant` must be a **U8 tensor** (`torch.frombuffer(...).clone()`), not raw bytes |
| Bench CRITICAL ERROR (0 armed) | 0 NVFP4 layers. Regenerate with K ≥ 1 |
| SSIM stuck around 0.94 | Outside the island. Re-sweep K±1 around the discriminating seed |
| Process won't die | Kill the whole parent tree: `taskkill /PID <parent> /T /F` (Windows) |
| Sudden drop near layer 90 | CE-type cliff. Step back and re-check K±1 |

## 10. Files in this repo

| File | Purpose |
|---|---|
| `diag_impact.py` | Step 1: per-layer NVFP4 trajectory impact measurement |
| `gen_reverse_nvfp4.py` | Step 2: reverse hybrid converter (INT8 → NVFP4, K lowest-impact layers) |
| `benchmark/zi_convrot_nvfp4_bench_native.py` | Step 3: native bench (bf16 native baseline, `--native-dtype`) |
| `native_convert_int8_convrot_zi.py` | INT8 prerequisite (see [How to quantize Z Image.md](How%20to%20quantize%20Z%20Image.md)) |

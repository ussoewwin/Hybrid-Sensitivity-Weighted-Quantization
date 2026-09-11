# How to create Hybrid NVFP4 from ConvRot INT8 (Z Image, Reverse Method)

> **Prerequisite**: Create the **ConvRot INT8** UNet first with `Z_Image/native_convert_int8_convrot_zi.py`,
> as in [How to quantize Z Image.md](How%20to%20quantize%20Z%20Image.md).
> This document is the next step: turn that complete INT8 UNet into a **hybrid NVFP4** UNet and
> **validate it with the deterministic per-step trajectory comparison** (20 seeds, cosine mean ≥ 0.95).

This method is **fundamentally different** from the conventional "protect the top-important layers"
approach (histogram MSE / cosine / SVD saliency). It is a **reverse method**: start from the complete
ConvRot INT8 model (error ≈ 0) and convert layers to NVFP4 **in ascending order of per-layer impact**
(lowest-impact first). The conventional method ignores inter-layer interactions and is not sufficient
for this hybrid. The reverse method stays in the low-error regime where additivity holds, so
**single-layer ranking is valid**.

**Validation is done with the deterministic 20-seed latent-trajectory comparison**
(per-step cosine + bifurcation detection); the production gate is **cosine mean ≥ 0.95 and
0/20 bifurcated**, measured with **SA2 (SageAttention2) attention acceleration**
(`--attention sage2`) in **TC (W4A4)** mode after `input_scale` calibration. That
**SA2 + TC** pair is the accelerated production configuration, and it is the only
configuration a gate result is valid for. Scores are
**checkpoint-specific**: impact ranking, `K`, and the trajectory numbers must be re-measured for
every model and are not transferable. Reference numbers for one example model (moodyProMix
collectorsEdition) are listed in [Step 8](#step-8-finding-k) as a sanity-check ground truth only.

---

## Judgement criteria (read first)
## Coexistence with the existing SageAttention node (read before building a workflow)

HSWQ bakes quantization into the model; attention acceleration is **not** part of the quantized
file. If you already use a SageAttention patch node, keep the two layers separate and mind the
points below.

| Item | Detail |
|---|---|
| Existing node | **`Patch Sage Attention DM`** from [ComfyUI-DistorchMemoryManager](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager) - place it **after** the model loader and it patches that model's attention |
| Where this repo's loaders live | [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools) - the HSWQ model loaders / quantized-model nodes |
| Do they conflict? | **No.** The quantized loader only touches Linear weights/activations; the SA node only sets `transformer_options["optimized_attention_override"]`. Neither installs attention overrides in the HSWQ loader, so there is nothing to fight over |
| Recommended order | model loader -> `Patch Sage Attention DM` -> LoRA / sampler nodes. Applying the SA patch after the loader means the patched model is the one that reaches the sampler |
| **Pick the right mode** | On **sm120 (RTX 50)** only the CUDA fp8 path works. Use `auto` (identical to what the benchmarks call internally: INT8 QK + FP8 PV, `fp32+fp16` = SageAttention2++). `sageattn_qk_int8_pv_fp16_cuda` / `..._fp16_triton` **fail on sm120** (no kernel image / Triton path not usable) and silently fall back to SDPA => **no speed-up** |
| `..._fp8_cuda` vs `..._fp8_cuda++` | `..._fp8_cuda` uses `fp32+fp32` accumulation; only `..._fp8_cuda++` uses `fp32+fp16` (SageAttention2++). To get the same behaviour as the benchmarks, prefer `auto` |
| `allow_compile` | Leave it off while validating fidelity (the reference measurements ran eager). Enabling it changes numerics/timing, so re-measure before trusting the numbers |
| Measurement anchor | The published numbers in this guide were measured with the internal benchmark path (`sageattn()` auto dispatch). A node run with any other mode is a different configuration and is not directly comparable |
| Order of validation | 1) quantized model fidelity without the SA node 2) same run with the SA node enabled 3) compare per-seed `final-cos` and wall time. Never judge quality from an SA-patched run alone |

**Rule of thumb:** if the SA node's mode would not resolve to the sm120 CUDA fp8 path, the run
measures SDPA, not SageAttention - do not file those numbers as accelerated results.

The production quality gate is the **deterministic per-step latent trajectory comparison**
(`benchmark/zi_convrot_nvfp4_traj_compare.py`), not decoded SSIM. It samples FP16 baseline and the
quantized model from identical noise (same seed) and compares the latent trajectories step by step.

**Runs are always made in the production configuration: SA2 attention (`--attention sage2`)
plus TC/W4A4 (`--tc`) on the calibrated hybrid.** A result measured without either of them is
not a gate result.

| Metric | Threshold | Meaning |
|---|---|---|
| **final-cos** | — | final-step latent cosine (same seed, FP16 vs quantized) |
| **max-step-drop** | **> 0.05** | that seed is **bifurcated** (sudden trajectory jump = different picture, not degradation) |
| **same-image** | **≥ 0.98** | final cosine high enough to call it the same picture |
| **PASS** | **mean ≥ 0.95 AND bifurcated = 0/20** | production gate for 20 seeds × 12 steps |

- **Fixed 20-seed set:** `42,1337,7,2024,555,43,1458,9,2026,777,44,1338,8,2028,888,46,1587,12,2047,222`
- **Fixed steps:** `--steps 12` (the script default is 25 — always pass 12 explicitly)
- **Fixed prompt / CFG / sampler:** script defaults (`masterpiece, best quality, 1girl, solo, standing,
  simple background` / cfg 2.5 / euler / simple / 1024×1024) — do not change.
- **Fixed attention:** `--attention sage2` — SageAttention2 (INT8 QK + FP8 PV, sm120 path), the
  accelerated production attention. If the report footer says `attention mode   : sdpa`, SA2 was
  not armed and the run is not a gate result.
- **Fixed GEMM:** `--tc` (TC/W4A4) on the calibrated hybrid (Step 6). Step 7 is the native
  comparison model and is parity (W4A16) by construction.
- `drifted (different image)` is **normal** (small acceptable divergence). Only **bifurcated** and
  mean < 0.95 fail a configuration.

### GEMM modes (TC vs parity) — make it explicit

| Mode | Computation | When it is used |
|---|---|---|
| **TC (W4A4)** | Blackwell Tensor Core `scaled_mm_nvfp4_pooled` → `_C.cublas_gemm_blockwise_fp4` | checkpoint contains `*.input_scale` keys (calibrated) |
| **Parity (W4A16)** | stock GEMM (fp16 activations) + online act rotate | no `input_scale` keys |

- Auto-selection: if any `*.input_scale` key exists → **TC**; otherwise → **parity**.
- CLI overrides: `--tc` forces TC, `--parity` forces parity. Env overrides:
  `HSWQ_ZI_FORCE_PARITY=1` > `HSWQ_ZI_FORCE_TC=1` > auto-detect.
- **Forcing TC on an uncalibrated checkpoint collapses quality (cosine ~0.18)** — always run
  Step 5 (calibration) before measuring TC.
- The log ends with a definitive line, verify it every run:
  `GEMM MODE: TC (W4A4 TensorCore)` or `GEMM MODE: PARITY (W4A16 dequant GEMM)`.
- **Production measurement runs in TC** on the calibrated hybrid. Native (no input_scale) is measured
  without `--tc` (auto → parity).

---

## 0. Prerequisites

Anything that runs this flow needs, regardless of environment:

| Requirement | Notes |
|---|---|
| CUDA GPU, **≥ 12 GB VRAM** | run **one process at a time**; concurrent runs exhaust VRAM |
| Python with **PyTorch (CUDA)** | the exact install command depends on your CUDA version |
| This repository | clone it; it bundles the ComfyUI checkout in `ComfyUI-master/` (read-only — never modify it) |
| Runtime packages | `pip install -r requirements.txt` (ComfyUI runtime) and `pip install -U comfy_kitchen` (NVFP4 layout). If your Python lacks them, also install `scikit-image` (SSIM benchmark) |
| Base UNet | the original fp16/bf16 Z Image `.safetensors` (`<base>`) |
| Text encoder | a Qwen3-4B text encoder `.safetensors` (`<clip>`), needed from Step 2 onward |
| Disk | keep **≥ 40 GB free** (base 12.3 + INT8 5.7 + hybrid 4.8 + calib 4.8 + native 4.5 GB during a run) |

### Environment Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install -U comfy_kitchen
pip install scikit-image
```

A **VAE is not needed**: the validation works in latent space (no decoded-SSIM step).
On Windows, set `PYTHONIOENCODING=utf-8` in the shell to avoid cp932 decode errors; never set
`TORCH_LOGS` (torch import fails with an AttributeError).

## Paths (replace every `<...>` with a real path on your machine)

| Placeholder | Meaning |
|---|---|
| `<base>` | base fp16/bf16 Z Image UNet `.safetensors` (e.g. `test.safetensors`) |
| `<int8>` | complete ConvRot INT8 UNet from [How to quantize Z Image.md](How%20to%20quantize%20Z%20Image.md) (e.g. `test2.safetensors`) |
| `<impact>.json` | **created by Step 2** (e.g. `impact_moody.json`) — never downloaded, never copied from another model |
| `<hybrid>` | hybrid output of Step 3, name pattern `<model>_hswq_hybrid_nv<K>_convrot_nvfp4.safetensors` |
| `<calib>` | calibrated hybrid output of Step 5, name pattern `<model>_hswq_hybrid_nv<K>_convrot_nvfp4_calib.safetensors` |
| `<native>` | native full-NVFP4 output of Step 4 (e.g. `test4.safetensors`) |
| `<K>` | integer: how many lowest-impact layers to convert to NVFP4 (search this; not fixed) |
| `<comfy_path>` | folder that contains `comfy/` (bundled: `ComfyUI-master`) |
| `<clip>` | local Qwen3-4B text encoder `.safetensors` |

## Overall flow

```
<base>  (original FP16/BF16, 12.3 GB)
  │ Step 1: Z_Image/native_convert_int8_convrot_zi.py --per_channel_int8 --no-bench
  ▼
<int8>  (208 ConvRot INT8 layers, error ≈ 0, 5.74 GB)
  │ Step 2: Z_Image/diag_impact.py --steps 12 --seed 42   (writes <impact>.json, 15–50 min)
  ▼
<impact>.json
  │ Step 3: Z_Image/gen_reverse_nvfp4.py <K> <hybrid> <int8> <impact>.json
  ▼
<hybrid>  (K lowest-impact layers → NVFP4, rest INT8)
  │
  ├─ Step 4: native_convert_nvfp4_zi.py --model <base> --output <native>   (comparison model)
  ▼
  │ Step 5: Z_Image/calib_input_scale_nvfp4.py <base> <hybrid> <calib> --prompts sample/calibration_prompts_128.txt --samples 128
  ▼
<calib>  (hybrid + *.input_scale, REQUIRED for TC/W4A4)
  │ Step 6: benchmark/zi_convrot_nvfp4_traj_compare.py --tc --attention sage2 (20 seeds × 12 steps)
  ▼
PASS iff final-cosine mean ≥ 0.95 and 0/20 bifurcated  →  else change <K> (Step 8)
  │ Step 7: same traj_compare on <native> WITHOUT --tc (auto parity) — comparison baseline
  ▼
Step 9: upload + cleanup
```

---

## Step 1. Create the ConvRot INT8 UNet

```bash
python Z_Image/native_convert_int8_convrot_zi.py \
  --model "<base>" --output "<int8>" --per_channel_int8 --no-bench
```

- 208 layers converted / 245 kept, ConvRot Linear 208 / Conv2d 0, per-channel INT8. Output ≈ 5.74 GB.
- Validation is done separately in Step 6, so skip the built-in post-convert bench with `--no-bench`.

---

## Step 2. Create `<impact>.json` (per-layer trajectory impact, 15–50 min)

```bash
python Z_Image/diag_impact.py "<base>" "<int8>" "<impact>.json" \
  --comfy-path "<comfy_path>" --steps 12 --seed 42
```

- Injects **true NVFP4 quantization error** (comfy_kitchen `TensorCoreNVFP4Layout` quantize →
  dequantize roundtrip) into **one layer at a time**, runs a fixed-seed 12-step denoising trajectory,
  and records how far the final latent drifts (relative MSE). That is the layer's **real importance
  under trajectory propagation**.
- Progress prints as `[25/208] [50/208] ... [208/208]`. The first progress line can take a few minutes
  (model load + FP16 reference trajectory) — wait up to 50 min total.
- Writes `{"x_ref_norm": ..., "impacts": {<layer>: <rel MSE>, ...}}`. Ranking is **not transferable**
  between checkpoints — always re-measure.
- Typical ranking: smallest impact (safest to convert) → `noise_refiner.*.attention.qkv`-class layers;
  largest (protect) → `t_embedder.mlp.*`, `final_layer.linear`, `final_layer.adaLN_modulation.*`.

Run from the clone directory (script auto-resolves the repo root for `benchmark/`; do **not** pass
`--repo-root`).

---

## Step 3. Reverse hybrid conversion (`<K>` lowest-impact layers → NVFP4)

```bash
python Z_Image/gen_reverse_nvfp4.py <K> \
  "<model>_hswq_hybrid_nv<K>_convrot_nvfp4.safetensors" \
  "<int8>" "<impact>.json" \
  [--out-dir "<output-dir>"]      # default: "." (cwd)
```

What it does:

1. Ranks layers by `<impact>.json` **ascending** (lowest impact first).
2. For the first **K**: INT8 dequant (`q × scale` → rotated W_rot) → Kitchen
   `TensorCoreNVFP4Layout` NVFP4 (`format` nvfp4, `convrot` true, `groupsize` 256) —
   **without re-rotating** (INT8 weights are already stored rotated W@H^T).
3. Replaces those layers with `.weight` (U8 packed) / `.weight_scale` (F8_E4M3) /
   `.weight_scale_2` (F32) / `.comfy_quant` (U8 tensor). Remaining layers keep INT8 keys intact.

Result: **(208 − K) INT8 + K NVFP4**. Start **K = 90** (see Step 8 for the search rule).

**On-disk format of converted layers:**
`.weight` U8 `[out, in/2]` + `.weight_scale` F8_E4M3 + `.weight_scale_2` F32 + `.comfy_quant`
conf `{"format": "nvfp4", "convrot": true, "convrot_groupsize": 256}`. Weights are **stored rotated**
(a large dequant-vs-fp16 deviation is expected).

---

## Step 4. Native full-NVFP4 (comparison baseline)

```bash
python native_convert_nvfp4_zi.py --model "<base>" --output "<native>"
```

- All 180 Linear layers → NVFP4 (ConvRot ON, Z-Image-Turbo profile). Output ≈ 4.2 GiB.
- Metadata: `hswq_nvfp4_convrot='1'`, `nvfp4_layers=180`.
- **No `input_scale` is written** — it is a parity (W4A16) model by construction and must **not** be
  measured with `--tc`.

---

## Step 5. Calibrate `input_scale` (REQUIRED for TC / W4A4)

```bash
python Z_Image/calib_input_scale_nvfp4.py "<base>" "<hybrid>" "<calib>" \
  --comfy-path "<comfy_path>" \
  --prompts "sample/calibration_prompts_128.txt" \
  --samples 128
```

- Runs 128 calibration trajectories × 4 steps (the bundled prompt set
  `sample/calibration_prompts_128.txt`, or the default synthetic set if `--prompts` is omitted)
  and measures the running absmax of each NVFP4 layer's input activations,
  then writes **`input_scale = amax / 2688`** (computed in the rotated (Hadamard) domain, NVFP4 step only).
- Output `<calib>` is an exact copy of `<hybrid>` plus `*.input_scale` (F32) keys. ~10–15 min.
- Progress: `calibrating: 128 trajectories x 4 steps, seed 42`, `[8/128] ... [128/128]`,
  `amax coverage: N/N`, then `input_scale formula: amax / 2688` and `COMPLETE done=True`.
- **Verify** the key count equals K (0 keys ⇒ TC will collapse):

```bash
python -c "from safetensors import safe_open; f=safe_open(r'<calib>','pt'); ks=[k for k in f.keys() if k.endswith('.input_scale')]; print('input_scale keys:', len(ks))"
# expect: input_scale keys: <K>
```

---

## Step 6. Trajectory validation of the hybrid (TC / W4A4)

```bash
python benchmark/zi_convrot_nvfp4_traj_compare.py \
  --fp16 "<base>" \
  --quant "<calib>" \
  --clip_path "<clip>" \
  --comfy_path "<comfy_path>" \
  --steps 12 \
  --seeds "42,1337,7,2024,555,43,1458,9,2026,777,44,1338,8,2028,888,46,1587,12,2047,222" \
  --tc \
  --attention sage2
```

- ~75 s per seed without SA2 (FP16 ≈ 30 s + hybrid ≈ 45 s) ⇒ **≈ 25 min** for 20 seeds.
  With SA2 the per-seed time drops by ≈ 14 % ⇒ **≈ 65 s per seed / ≈ 22 min** for 20 seeds
  (measured on an example checkpoint, 20 seeds × 12 steps: quantized pass 13.686 s → 11.712 s
  = 1.169×, and 20/20 seeds faster; the attention module alone is 2.29× — 189.7 ms → 82.7 ms).
  Deterministic (cuDNN deterministic + benchmark=False pinned inside the script).
- The full report looks like this — **report the entire block** (every per-seed line + summary):

```
========================================================================
Deterministic per-step latent trajectory divergence (FP16 vs ConvRot Hybrid)
========================================================================
[seed 42] final-cos=0.94806  max_step_drop=0.0098  -> drifted (different image)
...
[seed 222] final-cos=0.98680  max_step_drop=0.0023  -> same-image

--- Multi-seed summary ---
    seed  final-cos    final-mse  max-drop                verdict
      42    0.94806    1.263e+00    0.0098 drifted (different image)
...

final-cosine: min=0.87123  mean=0.96033  max=0.98680
attention mode   : sage2
same-image seeds : 4/20
bifurcated seeds : 0/20   (sudden trajectory jump = different picture, not degradation)

------------------------------------------------------------------------
GEMM MODE: TC (W4A4 TensorCore)
  TC forward     : scaled_mm hits=43200  dequant_fallbacks=0
  parity forward : nvfp4 fwd=0  int8 fwd=...
  addmm residual : scaled_mm=14400  dequant=0
------------------------------------------------------------------------
```

- **PASS** iff `mean ≥ 0.95` and `bifurcated seeds : 0/20`.
- **Always check the `GEMM MODE:` line.** If it says PARITY, the `--tc` force did not take effect or
  `input_scale` keys are missing — fix before judging.
- **Always check the SA2 lines.** Before the report the run prints
  `[SAGE2] attention call stats: total=<N> sa2=<N> fallback(mask)=<N> fallback(head_dim)=<N> errors=<N>`
  and `[SAGE2] attention time: sage2=<ms> ms  fallback_sdpa=<ms> ms`; the footer line is
  `attention mode   : sage2`. `errors=0` and `sa2=total` are expected; a non-zero fallback count
  means part of the run was not accelerated.

---

## Step 7. Trajectory validation of native (parity) — comparison baseline

```bash
python benchmark/zi_convrot_nvfp4_traj_compare.py \
  --fp16 "<base>" --quant "<native>" \
  --clip_path "<clip>" --comfy_path "<comfy_path>" \
  --steps 12 \
  --seeds "42,1337,7,2024,555,43,1458,9,2026,777,44,1338,8,2028,888,46,1587,12,2047,222" \
  --attention sage2
```

- **No `--tc`**: native has no `input_scale`, so auto-detect → `GEMM MODE: PARITY (W4A16 dequant GEMM)`.
  SA2 attention is used here as well (`--attention sage2`) — the comparison baseline differs only
  in the GEMM mode, not in the attention path.
- Expect the native to score **below the hybrid** — that is the point of the comparison
  (reference: native mean 0.91079 / 1/20 bifurcated vs hybrid nv100 mean 0.96033 / 0/20).

---

## Step 8. Finding K

`K` is checkpoint-specific; the quality surface is often **not a single cliff** (failing seeds change
with K, quality can recover then fail again). Search **sequentially, 10 at a time, one process at a time**:

1. Start at **K = 90** → Step 3 → Step 5 → Step 6.
2. nv90 **passes** (mean ≥ 0.95, 0/20 bifurcated) → raise K by 10 (nv100, nv110, ...) and repeat.
   The answer is the **largest K that still passes** — keep raising until it fails.
3. nv90 **fails** → lower K by 10 (nv80, nv70, ...) until it passes; that K is the answer.
4. **Any K with bifurcated > 0 is rejected**, even if the mean ≥ 0.95.
5. If the boundary is ambiguous, sweep K±1–2 around it (islands happen).

### Reference results (moodyProMix_collectorsEdition, 20 seeds × 12 steps)

> Measured **before** SA2 existed in this benchmark (the `--attention` option was added
> 2026-09-10), so these numbers are without SA2 acceleration. They are kept as-is for the
> TC-vs-parity ordering only. Re-measure a candidate `<K>` in the **SA2 + TC** configuration before
> shipping it.

| Model | K | mean | min | max | same-image | bifurcated | GEMM |
|---|---|---|---|---|---|---|---|
| nv90 | 90 | 0.95852 | 0.82504 | 0.98816 | 5/20 | 0/20 | TC (W4A4) |
| **nv100 (final)** | **100** | **0.96033** | **0.87123** | **0.98680** | **4/20** | **0/20** | **TC (W4A4)** |
| native | 180 | 0.91079 | 0.59021 | 0.96337 | 0/20 | **1/20** | PARITY (W4A16) |

---

## Step 9. Upload and cleanup

Upload the final calibrated hybrid (`<model>_hswq_hybrid_nv<K>_convrot_nvfp4_calib.safetensors`)
to Hugging Face — the file needs no extra packing; ComfyUI loads it directly
(`ComfyUI-HSWQ-Loader-and-Tools`, or any `nvfp4`-capable loader, with TC auto-detected from
`input_scale`). The `upload.py` template in the repo root uploads a file with `huggingface_hub`;
edit the values at the top (username, repo, your **Write**-capable token, file path, in-repo filename),
then run `python upload.py`.

**Keep:** `<base>`, `<int8>`, and the final `<calib>` (production artifact).
**Delete:** intermediate hybrids from rejected K values, the uncalibrated `<hybrid>`, and `<native>`.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `UnicodeDecodeError 'cp932'` | Windows locale issue; set `PYTHONIOENCODING=utf-8` in the shell |
| torch import `AttributeError ... get_log_level_pairs` | `TORCH_LOGS` is set; unset it and never set it |
| `impact_*.json` missing | Created by Step 2; the third positional arg of `diag_impact.py` is the **output** path |
| 0 NVFP4 layers / bench CRITICAL ERROR (0 armed) | `gen_reverse_nvfp4.py` failed to match impact keys. Check the key normalization; regenerate with K ≥ 1 |
| `save_file` ValueError | `.comfy_quant` must be a **U8 tensor** (`torch.frombuffer(...).clone()`), not raw bytes |
| `GEMM MODE: PARITY` while `--tc` was passed | `input_scale` keys missing or force not applied — run Step 5 and re-check the key count |
| final cosine collapses to ~0.18 | **TC forced without `input_scale`** — run calibration (Step 5) |
| mean ≥ 0.95 but bifurcated > 0 | Reject this K; lower K by 10 (high-impact layers are breaking) |
| `SafetensorError: I/O error: disk` | Disk full — keep ≥ 40 GB free (see Prerequisites) |
| Numbers differ from a previous run | Confirm `--steps 12`, the exact 20-seed set, the calibrated file, `--attention sage2`, and check the `GEMM MODE:` line |
| `attention mode   : sdpa` in the report | SA2 was not armed — pass `--attention sage2`; every gate run uses it |
| `[SAGE2]` counter shows `errors` > 0 or fallbacks > 0 | Part of the attention calls fell back to SDPA — check the SageAttention2 build / CUDA version for this GPU |
| Process won't die | Kill the whole process tree (`taskkill /PID <parent> /T /F` on Windows) |

## Files in this repo

| File | Purpose |
|---|---|
| `Z_Image/native_convert_int8_convrot_zi.py` | Step 1 — ConvRot INT8 (prerequisite, see [How to quantize Z Image.md](How%20to%20quantize%20Z%20Image.md)) |
| `Z_Image/diag_impact.py` | Step 2 — per-layer NVFP4 trajectory impact → `<impact>.json` |
| `Z_Image/gen_reverse_nvfp4.py` | Step 3 — reverse hybrid converter (K lowest-impact layers → NVFP4) |
| `native_convert_nvfp4_zi.py` | Step 4 — native full-NVFP4 comparison model |
| `Z_Image/calib_input_scale_nvfp4.py` | Step 5 — `input_scale = amax / 2688` calibration (enables TC/W4A4) |
| `benchmark/zi_convrot_nvfp4_traj_compare.py` | Steps 6–7 — deterministic 20-seed per-step trajectory divergence (cosine, bifurcation, GEMM-mode counters, SA2 attention via `--attention sage2`) |
| `sample/calibration_prompts_128.txt` | default prompt set used by Step 5 |
| `upload.py` | Step 9 — Hugging Face upload template |

**Dependencies:** `comfy-kitchen` (NVFP4 layout), `safetensors`, `scikit-image`, `tqdm`,
`transformers` (Step 4), `psutil` (traj_compare), plus the ComfyUI runtime (`requirements.txt`,
used via the bundled `ComfyUI-master/` tree). Run Step 2 from the clone root; run Steps 6–7 with
`--comfy_path "ComfyUI-master"` (or an absolute path).
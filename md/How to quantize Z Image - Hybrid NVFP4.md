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
0/20 bifurcated**, measured in **TC (W4A4)** mode after `input_scale` calibration. Scores are
**checkpoint-specific**: impact ranking, `K`, and the trajectory numbers must be re-measured for
every model and are not transferable. Reference numbers for one example model (moodyProMix
collectorsEdition) are listed in [Step 8](#step-8-finding-k) as a sanity-check ground truth only.

> **Ready-to-run reference**: the complete cloud flow below (install → download → quantize → calib →
> validate → upload) is also available as a VAST.ai notebook template: `vastai-hswq-zi-nvfp4.ipynb`.

---

## Judgement criteria (read first)

The production quality gate is the **deterministic per-step latent trajectory comparison**
(`benchmark/zi_convrot_nvfp4_traj_compare.py`), not decoded SSIM. It samples FP16 baseline and the
quantized model from identical noise (same seed) and compares the latent trajectories step by step.

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

## 0. Environment

### Cloud (VAST.ai / Jupyter) — quick setup

```bash
# 0-1. PyTorch (CUDA 13.0 on the tensor template)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 0-2. Other libraries + downloader
pip install diffusers safetensors transformers accelerate tqdm sentencepiece protobuf scikit-image sageattention
apt-get -y install aria2

# 0-3. Clone this repository
git clone https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization.git
cd Hybrid-Sensitivity-Weighted-Quantization

# 0-4. Download the base UNet (example: moodyProMix collectorsEdition from Civitai), the
#      Qwen3-4B text encoder, and a VAE into the clone directory:
aria2c --console-log-level=error --allow-overwrite=true -x 16 -s 16 -k 1M "<civitai-or-hf-url>" -d . -o test.safetensors
aria2c --console-log-level=error --allow-overwrite=true -x 16 -s 16 -k 1M "https://huggingface.co/ussoewwin/qwen3_4b_8b_abliterated_fp16/resolve/main/qwen3_4b_abliterated_fp16_converted.safetensors" -d . -o clip.safetensors
aria2c --console-log-level=error --allow-overwrite=true -x 16 -s 16 -k 1M "<vae-url>" -d . -o vae.safetensors

# 0-5. Repo dependencies (run from the clone directory)
pip install -r requirements.txt
pip install -U comfy_kitchen
```

File conventions used below (adjust names freely): `test.safetensors` = base, `test2.safetensors` =
INT8, `test3.safetensors` = hybrid, `test4.safetensors` = native, `test5.safetensors` = calibrated hybrid.

### Local (Windows) — equivalent

| Item | Value |
|---|---|
| Python | `D:\USERFILES\ComfyUI\python_embeded\python.exe` (or any CUDA venv) |
| Working dir | `D:\USERFILES\GitHub\hswq` (this clone) |
| ComfyUI tree | `D:\USERFILES\GitHub\hswq\ComfyUI-master` (read-only) |
| CLIP | `D:\USERFILES\ComfyUI\ComfyUI\models\clip\qwen3_4b_abliterated_fp16_converted.safetensors` |
| Output dir | `D:\USERFILES\ComfyUI\ComfyUI\models\unet\` |

Prepend `$env:PYTHONIOENCODING='utf-8'` to every PowerShell command (Japanese-locale cp932 fix).
Never set `TORCH_LOGS` (torch import dies with an AttributeError). Run **one process at a time**
(VRAM discipline; 16 GB card is enough for one trajectory run at a time).

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
  │ Step 6: benchmark/zi_convrot_nvfp4_traj_compare.py --tc (20 seeds × 12 steps)
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
- `--clip_path` / `--comfy_path` / `--vae` only matter when you want the built-in post-convert bench
  (`--bench`); the validation here is done separately in Step 6, so pass `--no-bench`.

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
`--repo-root`). Relative weight paths are searched under cwd / the clone, so
`Hybrid-Sensitivity-Weighted-Quantization/test.safetensors` works from inside the clone.

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
  --tc
```

- ~75 s per seed (FP16 ≈ 30 s + hybrid ≈ 45 s) ⇒ **≈ 25 min** for 20 seeds. Deterministic
  (cuDNN deterministic + benchmark=False pinned inside the script).
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

---

## Step 7. Trajectory validation of native (parity) — comparison baseline

```bash
python benchmark/zi_convrot_nvfp4_traj_compare.py \
  --fp16 "<base>" --quant "<native>" \
  --clip_path "<clip>" --comfy_path "<comfy_path>" \
  --steps 12 \
  --seeds "42,1337,7,2024,555,43,1458,9,2026,777,44,1338,8,2028,888,46,1587,12,2047,222"
```

- **No `--tc`**: native has no `input_scale`, so auto-detect → `GEMM MODE: PARITY (W4A16 dequant GEMM)`.
- Expect the native to score **below the hybrid** — that is the point of the comparison
  (moodyProMix: native mean 0.91079 / 1/20 bifurcated vs hybrid nv100 mean 0.96033 / 0/20).

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
`input_scale`).

```bash
pip install -U "huggingface_hub[cli]"
pip install hf_transfer
```

Edit the values at the top of `upload.py` (username, repo name, your **Write**-capable HF token,
the file to upload, and the in-repo filename), then:

```bash
python upload.py
```

**Keep:** `<base>`, `<int8>`, and the final `<calib>` (production artifact).
**Delete:** intermediate hybrids from rejected K values, the uncalibrated `<hybrid>`, and `<native>`.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `UnicodeDecodeError 'cp932'` | Japanese Windows; prepend `$env:PYTHONIOENCODING='utf-8'` (PowerShell) |
| torch import `AttributeError ... get_log_level_pairs` | `TORCH_LOGS` is set; unset it and never set it |
| `impact_*.json` missing | Created by Step 2; the third positional arg of `diag_impact.py` is the **output** path |
| 0 NVFP4 layers / bench CRITICAL ERROR (0 armed) | `gen_reverse_nvfp4.py` failed to match impact keys. Check the key normalization; regenerate with K ≥ 1 |
| `save_file` ValueError | `.comfy_quant` must be a **U8 tensor** (`torch.frombuffer(...).clone()`), not raw bytes |
| `GEMM MODE: PARITY` while `--tc` was passed | `input_scale` keys missing or force not applied — run Step 5 and re-check the key count |
| final cosine collapses to ~0.18 | **TC forced without `input_scale`** — run calibration (Step 5) |
| mean ≥ 0.95 but bifurcated > 0 | Reject this K; lower K by 10 (high-impact layers are breaking) |
| `SafetensorError: I/O error: disk` | Disk full — keep ≥ 40 GB free (base 12.3 + INT8 5.7 + hybrid 4.8 + calib 4.8 + native 4.5) |
| Numbers differ from a previous run | Confirm `--steps 12`, the exact 20-seed set, the calibrated file, and check the `GEMM MODE:` line |
| Process won't die | Kill the whole tree: `taskkill /PID <parent> /T /F` (Windows) |

## Files in this repo

| File | Purpose |
|---|---|
| `Z_Image/native_convert_int8_convrot_zi.py` | Step 1 — ConvRot INT8 (prerequisite, see [How to quantize Z Image.md](How%20to%20quantize%20Z%20Image.md)) |
| `Z_Image/diag_impact.py` | Step 2 — per-layer NVFP4 trajectory impact → `<impact>.json` |
| `Z_Image/gen_reverse_nvfp4.py` | Step 3 — reverse hybrid converter (K lowest-impact layers → NVFP4) |
| `native_convert_nvfp4_zi.py` | Step 4 — native full-NVFP4 comparison model |
| `Z_Image/calib_input_scale_nvfp4.py` | Step 5 — `input_scale = amax / 2688` calibration (enables TC/W4A4) |
| `benchmark/zi_convrot_nvfp4_traj_compare.py` | Steps 6–7 — deterministic 20-seed per-step trajectory divergence (cosine, bifurcation, GEMM-mode counters) |
| `sample/calibration_prompts_128.txt` | default prompt set used by Step 5 |
| `upload.py` | Step 9 — Hugging Face upload template |

**Dependencies:** `comfy-kitchen` (NVFP4 layout), `safetensors`, `scikit-image` (SSIM), plus the
ComfyUI tree in `ComfyUI-master/`. Run Step 2 from the clone root; run Steps 6–7 with
`--comfy_path "ComfyUI-master"` (or an absolute path).
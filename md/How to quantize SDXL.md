# How to quantize SDXL (Reverse Hybrid ConvRot INT8)

> **Prerequisite**: the original **FP16 SDXL checkpoint** (`<base>`) and a **complete ConvRot INT8
> artifact** of the same checkpoint (`<int8>`), created with `native_convert_int8_sdxl.py`.
> The artifact is used **only as the layer-list source** for the impact measurement (Step 2) — the
> hybrid itself is built from the FP16 baseline (Step 3).

This method is **fundamentally different** from the conventional "protect the top-important layers"
approach (histogram MSE / cosine / SVD saliency). It is a **reverse method**: it measures every
candidate layer's impact on a fixed denoising trajectory (Step 2), then converts the **K lowest-impact
layers to ConvRot INT8 in ascending impact order**, while **every other layer is kept at FP16**
(the original size and precision). Keeping the most sensitive layers in FP16 is what lifts the mixed
checkpoint above a full INT8 pack at the same quality level.

The conventional method ignores inter-layer interactions and is not sufficient for this hybrid. The
reverse method stays in the low-error regime where additivity holds, so **single-layer ranking is
valid** (see `md/diag_impact_trajectory_sensitivity_technical_guide.md`).

**Validation is done with the deterministic 25-seed latent-trajectory comparison**
(`benchmark/sdxl_int8_traj_compare.py`): identical noise per seed for both models, per-step latent
cosine, bifurcation detection. The production gate is **cosine mean ≥ 0.95 and 0/25 bifurcated**.

Reference point: a **native (full) ConvRot INT8** pack of the same checkpoint measures a
**25-seed cosine mean of ≈ 0.93**. A valid reverse hybrid keeps the most sensitive layers in FP16 and
must therefore stay **above** that figure. Scores are **checkpoint-specific**: the impact ranking, `K`,
and the trajectory numbers must be re-measured for every model and are **not transferable**.

---

## Judgement criteria (read first)

The production quality gate is the deterministic per-step latent trajectory comparison
(`benchmark/sdxl_int8_traj_compare.py`), not a decoded image SSIM. It samples the FP16 baseline and
the quantized model from identical noise (same seed) and compares the latent trajectories step by
step.

| Metric | Threshold | Meaning |
|---|---|---|
| **final-cos** | — | final-step latent cosine (same seed, FP16 vs quantized) |
| **max-step-drop** | **> 0.05** | that seed is **bifurcated** (sudden trajectory jump = different picture, not degradation) |
| **same-image** | **≥ 0.98** | final cosine high enough to call it the same picture |
| **PASS** | **mean ≥ 0.95 AND bifurcated = 0/25** | production gate for 25 random seeds × 25 steps |

- **Fixed 25-seed random set** (the script default — always use it; do not cherry-pick seeds):
  `42,137,849,2024,7391,18429,53082,149206,382715,826401,1938502,4710928,8391642,15820493,`
  `36192847,71058294,128491703,285039184,491730285,762019483,938174026,1409285713,2683910547,3851729406,4195820371`
- **Fixed steps:** `--steps 25` (the script default is 25 — pass it explicitly).
- **Fixed prompt / CFG / sampler / size:** script defaults
  (`masterpiece, best quality, 1girl, solo, standing, simple background` / cfg 7.0 / dpmpp_2m / karras /
  1024×1024) — do not change.
- `drifted (different image)` is **normal** (small acceptable divergence). Only **bifurcated** and
  mean < 0.95 fail a configuration.
- The FP16 baseline branch of the benchmark always loads stock (no INT8 patch); the INT8 branch arms
  the quantized path automatically when the checkpoint carries `.comfy_quant` markers.

---

## 0. Prerequisites

| Requirement | Notes |
|---|---|
| CUDA GPU, **≥ 12 GB VRAM** | run **one process at a time**; concurrent runs exhaust VRAM |
| Python with **PyTorch (CUDA)** | the exact install command depends on your CUDA version |
| This repository | clone it; it bundles the ComfyUI checkout in `ComfyUI-master/` (read-only — never modify it) |
| Runtime packages | `pip install -r requirements.txt` (ComfyUI runtime) and `pip install -U comfy_kitchen` (INT8 layouts) |
| Base checkpoint | the original fp16 SDXL `.safetensors` (`<base>`, UNet + CLIP + VAE in one file) |
| Layer-list source | a complete ConvRot INT8 artifact (`<int8>`) of the **same** checkpoint (Step 1) |
| Disk | keep **≥ 25 GB free** (base 6.9 + full INT8 4.7 + hybrid 4.5–6.3 GB during a run) |
| Optional | `scikit-image` only if you also run the legacy decoded-image bench `benchmark/int8bench_sdxl.py` |

### Environment Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install -U comfy_kitchen
```

A **VAE is not needed** for the validation: the trajectory comparison works in latent space (no
decoded-SSIM step). On Windows, set `PYTHONIOENCODING=utf-8` in the shell to avoid cp932 decode
errors; never set `TORCH_LOGS` (torch import fails with an AttributeError).

## Paths (replace every `<...>` with a real path on your machine)

| Placeholder | Meaning |
|---|---|
| `<base>` | base fp16 SDXL checkpoint `.safetensors` (e.g. `waiIllustriousSDXL_v170.safetensors`) |
| `<int8>` | complete ConvRot INT8 artifact from Step 1 (layer-list source) |
| `<impact>.json` | **created by Step 2** (e.g. `impact_<model>.json`) — never downloaded, never copied from another model |
| `<hybrid>` | hybrid output of Step 3, name pattern `<model>_hswq_rev_int<K>_convrot_int8.safetensors` |
| `<K>` | integer: how many lowest-impact layers to convert to ConvRot INT8 (search this; not fixed) |
| `<comfy_path>` | folder that contains `comfy/` (bundled: `ComfyUI-master`) |

## Overall flow

```
<base>  (original FP16, 6.9 GB)
  │ Step 1: native_convert_int8_sdxl.py --per_channel_int8 --no-bench   (layer-list source)
  ▼
<int8>  (all ConvRot INT8 layers, 4.4 GB)
  │ Step 2: Z_Image/diag_impact_sdxl.py "<base>" "<impact>.json" --artifact "<int8>" --steps 4 --seed 42
  ▼                                                        (writes <impact>.json, ~20–40 min for 715 layers)
<impact>.json
  │ Step 3: Z_Image/gen_reverse_int8_sdxl.py <K> "<hybrid>" "<base>" "<impact>.json"
  ▼
<hybrid>  (K lowest-impact layers → ConvRot INT8, everything else stays FP16)
  │ Step 4: benchmark/sdxl_int8_traj_compare.py --fp16 "<base>" --int8 "<hybrid>" --steps 25  (25 random seeds)
  ▼
PASS iff final-cosine mean ≥ 0.95 and 0/25 bifurcated  →  else change <K> (Step 6)
  │ Step 5: same traj_compare on "<int8>" — native comparison baseline (≈ 0.93 mean)
  ▼
Step 7: upload + cleanup
```

---

## Step 1. Create the complete ConvRot INT8 artifact (layer-list source)

```bash
python native_convert_int8_sdxl.py \
  --model "<base>" --output "<int8>" --per_channel_int8 --no-bench
```

- **FULL ConvRot** (Linear + Conv2d when `in_dim` is divisible by a power-of-4 group size) is **ON by
  default**; `--per_channel_int8` keeps per-out-channel scales for any remaining non-ConvRot packs.
- Output ≈ 4.4 GB for an SDXL checkpoint. Validation is done separately in Step 4, so skip the
  built-in bench with `--no-bench`.
- This artifact is **not** the shipped model in the reverse flow — it only supplies the list of
  convertible layers (its `_quantization_metadata.layers`) to Step 2. Boundary layers
  (`conv_in`, `conv_out`, `time_embed`, `add_embedding`, `label_emb`) are excluded from the
  candidate set by the measurement script.

---

## Step 2. Create `<impact>.json` (per-layer trajectory impact, ~20–40 min)

```bash
python Z_Image/diag_impact_sdxl.py "<base>" "<impact>.json" \
  --comfy_path "<comfy_path>" \
  --artifact "<int8>" \
  --steps 4 --seed 42
```

- Injects **one layer at a time** with the layer's ConvRot INT8 reconstruction
  (`dequant(per-channel INT8(quantize(W @ H^T)))` — exactly the kernel that produces the shipped INT8
  pack, **no inverse rotation**), runs a fixed-seed 4-step denoising trajectory, and records how far
  the final latent drifts (relative MSE). That is the layer's **real importance under trajectory
  propagation** — i.e. the FP16-vs-ConvRot-INT8 error of that layer, propagated.
- `--artifact` restricts the measurement to the layers a full ConvRot INT8 pack actually converts
  (the candidate set), and `--steps` / `--seed` / `--width` / `--height` / `--prompt` are configurable.
- Progress prints as `[25/715] [50/715] ... [715/715]`. The first line can take a few minutes
  (checkpoint load + CLIP conditioning + reference trajectory).
- Writes `{"x_ref_norm": ..., "steps": ..., "seed": ..., "base": ..., "impacts": {<layer>: <rel MSE>, ...}}`.
  The ranking is **not transferable** between checkpoints — always re-measure.

---

## Step 3. Reverse hybrid conversion (`<K>` lowest-impact layers → ConvRot INT8)

```bash
python Z_Image/gen_reverse_int8_sdxl.py <K> \
  "<model>_hswq_rev_int<K>_convrot_int8.safetensors" \
  "<base>" "<impact>.json" \
  [--out-dir "<output-dir>"] [--groupsize 256]      # default out-dir: "." (cwd)
```

What it does:

1. Ranks layers by `<impact>.json` **ascending** (lowest impact first).
2. For the first **K** (excluding boundary layers): `W @ H^T` → per-channel INT8
   (Linear: rowwise `[out,1]`; Conv2d: channelwise `[out,1,1,1]`) → stores the **rotated INT8 weight**
   — the same kernel as `native_convert_int8_sdxl.py`, **without re-rotation**.
3. Replaces those layers with `.weight` (I8) / `.weight_scale` (F32) / `.comfy_quant` (U8 tensor).
   **All other layers are copied unchanged from the FP16 baseline** (original dtype and size).

**On-disk format of converted layers:**
`.weight` I8 (Linear `[out, in]`, Conv2d `[out, in, kH, kW]`) + `.weight_scale` F32
(Linear `[out,1]`, Conv2d `[out,1,1,1]`) + `.comfy_quant` conf
`{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": N}`. Weights are **stored
rotated** (a large dequant-vs-fp16 deviation is expected).

ComfyUI's mixed-precision ops select the quantized path per layer from the `.comfy_quant` marker, so
one file carries both the FP16-kept and the ConvRot INT8 layers.

**Size reference** (measured on one example checkpoint, `waiIllustriousSDXL_v170`, 715 quantifiable
layers; FP16 base 6.94 GB, full ConvRot INT8 4.69 GB; GB = decimal):

| K | 60 | 75 | 100 | 150 | 300 | 500 | 550 | 580 | 620 | 717 (full) |
|---|---|---|---|---|---|---|---|---|---|---|
| size (GB) | 6.81 | 6.78 | 6.73 | 6.63 | 6.06 | 5.32 | 5.10 | 4.97 | 4.82 | 4.69 |

Size decreases roughly linearly in K (≈ 3.1 MB per converted layer for this checkpoint).

---

## Step 4. Trajectory validation of the hybrid (25 random seeds × 25 steps)

```bash
python benchmark/sdxl_int8_traj_compare.py \
  --fp16 "<base>" \
  --int8 "<hybrid>" \
  --comfy_path "<comfy_path>" \
  --seeds "42,137,849,2024,7391,18429,53082,149206,382715,826401,1938502,4710928,8391642,15820493,36192847,71058294,128491703,285039184,491730285,762019483,938174026,1409285713,2683910547,3851729406,4195820371" \
  --steps 25 \
  --width 1024 --height 1024
```

- ~19 s per seed (FP16 ≈ 9.4 s + INT8 ≈ 9.5 s) ⇒ **≈ 8 min** for 25 seeds.

```
========================================================================
Deterministic per-step latent trajectory divergence (FP16 vs ConvRot INT8)
========================================================================
[seed 42] final-cos=0.99666  max_step_drop=0.0005  -> same-image
...
--- Multi-seed summary ---
final-cosine: min=... mean=... max=...
same-image seeds : N/25
bifurcated seeds : N/25   (sudden trajectory jump = different picture, not degradation)
avg wall/seed: FP16 ...s | INT8 ...s
```

- **PASS** iff `mean ≥ 0.95` and `bifurcated seeds : 0/25`.
- Report the **entire block** (every per-seed line plus the summary).
- Runs are deterministic for a fixed seed set **on a quiet GPU**; other GPU processes can change the
  numbers (ComfyUI picks different attention/offload paths depending on free VRAM). Run one process
  at a time.
- Note: single runs are noisy at small seed counts; always use the **full 25-seed set** for a verdict.

---

## Step 5. Native comparison baseline (optional)

```bash
python benchmark/sdxl_int8_traj_compare.py \
  --fp16 "<base>" --int8 "<int8>" \
  --comfy_path "<comfy_path>" --steps 25 --width 1024 --height 1024
```

- The native (full) ConvRot INT8 artifact is the comparison point: reference ≈ **0.93** mean / 25
  seeds. The reverse hybrid must beat it; that is the point of the method.

---

## Step 6. Finding K

`K` is checkpoint-specific, and the quality surface is often **not a single cliff** (failing seeds
change with K). Search **sequentially, 10 at a time, one process at a time**:

1. Start at **K = 60** → Step 3 → Step 4.
2. Passes (mean ≥ 0.95, 0/25 bifurcated) → raise K by 10 and repeat. The answer is the **largest K
   that still passes** — keep raising until it fails.
3. Fails → lower K by 10 until it passes; that K is the answer.
4. **Any K with bifurcated > 0 is rejected**, even if the mean ≥ 0.95.
5. If a **size cap** is required (e.g. < 5 GB), note that quality decreases monotonically with K:
   choose the smallest K whose size fits the cap, then confirm it still passes the gate — if the cap
   forces K above the passing range, the cap is not achievable with this method for that checkpoint
   (FP16 keeping is the precision reserve; the full INT8 floor of that checkpoint bounds it).

---

## Step 7. Upload and cleanup

Upload the final hybrid (`<model>_hswq_rev_int<K>_convrot_int8.safetensors`) to Hugging Face — the
file needs no extra packing; ComfyUI loads it directly (mixed precision ops arm the INT8 layers from
`.comfy_quant`). The `upload.py` template in the repo root uploads a file with `huggingface_hub`;
edit the values at the top (username, repo, your **Write**-capable token, file path, in-repo
filename), then run `python upload.py`.

**Keep:** `<base>`, `<impact>.json`, and the final `<hybrid>` (production artifact).
**Delete:** intermediate hybrids from rejected K values.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `UnicodeDecodeError 'cp932'` | Windows locale issue; set `PYTHONIOENCODING=utf-8` in the shell |
| torch import `AttributeError ... get_log_level_pairs` | `TORCH_LOGS` is set; unset it and never set it |
| `impact_*.json` missing | Created by Step 2; the second positional arg of `diag_impact_sdxl.py` is the **output** path |
| 0 layers converted / `SKIP (not in sd)` for every layer | impact keys and checkpoint keys did not resolve — verify the impact json was produced from the **same** checkpoint |
| `save_file` ValueError | `.comfy_quant` must be a **U8 tensor**, not raw bytes (handled inside the script) |
| final cosine collapses | `K` beyond the checkpoint's boundary — lower K (Step 6) |
| mean ≥ 0.95 but bifurcated > 0 | Reject this K; lower K by 10 (high-impact layers are breaking) |
| Numbers differ from a previous run | Another GPU process was running, or `--steps` / seed set differed; re-run on a quiet GPU with the fixed 25-seed set |
| Process won't die | Kill the whole process tree (`taskkill /PID <parent> /T /F` on Windows) |

## Files in this repo

| File | Purpose |
|---|---|
| `native_convert_int8_sdxl.py` | Step 1 — complete ConvRot INT8 pack (layer-list source) |
| `Z_Image/diag_impact_sdxl.py` | Step 2 — per-layer ConvRot INT8 trajectory impact → `<impact>.json` |
| `Z_Image/gen_reverse_int8_sdxl.py` | Step 3 — reverse hybrid converter (K lowest-impact layers → ConvRot INT8, rest FP16) |
| `benchmark/sdxl_int8_traj_compare.py` | Steps 4–5 — deterministic 25-seed per-step trajectory divergence (cosine, bifurcation) |

**Dependencies:** `comfy-kitchen` (INT8 layout), `safetensors`, plus the ComfyUI runtime
(`requirements.txt`, used via the bundled `ComfyUI-master/` tree). Run Steps 2–5 with
`--comfy_path "ComfyUI-master"` (or an absolute path).

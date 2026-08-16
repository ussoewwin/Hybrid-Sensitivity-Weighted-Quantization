# How to create Hybrid NVFP4 from ConvRot INT8 (Z Image, Reverse Method)

> **Prerequisite**: Create the **ConvRot INT8** UNet first with `native_convert_int8_convrot_zi.py`, as in
> [How to quantize Z Image.md](How%20to%20quantize%20Z%20Image.md).
> This document is the next step: turn that complete INT8 UNet into a **hybrid NVFP4** UNet.

This method is **fundamentally different** from the conventional "protect the top-important layers"
approach (histogram MSE / cosine / SVD saliency). It is a **reverse method**: start from the complete
ConvRot INT8 model (error ≈ 0) and convert layers to NVFP4 **in ascending order of per-layer impact**.
The conventional method ignores inter-layer interactions and is not sufficient for this hybrid.
The reverse method stays in the low-error regime where additivity holds, so **single-layer ranking is
valid**. Pass only if **every seed** of the native bench meets **SSIM (0-255 view) ≥ 0.97**.

## Clone the repository

```bash
git clone https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization.git
cd Hybrid-Sensitivity-Weighted-Quantization
```

After `cd`, you are in the **clone directory**. That directory contains `Z_Image/`, `benchmark/`, and
`ComfyUI-master/`. Run **Step 1** and **Step 2** from here. `Z_Image/diag_impact.py` already treats this
directory as its default root (it loads `benchmark/zi_convrot_nvfp4_bench.py`). Do **not** pass
`--repo-root`.

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

## Paths (replace every `<...>` with a real path on your machine)

Same placeholder style as [How to quantize Z Image.md](How%20to%20quantize%20Z%20Image.md).

| Placeholder | Meaning |
|---|---|
| `<path-to-unet>` | Folder that holds your Z Image Turbo UNet `.safetensors` files |
| `<zit_unet>` | UNet filename **without** `.safetensors` (same stem as `--model` / `--output` in the INT8 how-to) |
| `<path-to-ComfyUI>` | Folder that contains `comfy/`. If you use the tree bundled in this clone, that folder is `ComfyUI-master` (relative to the clone directory) |
| `<path-to-qwen3-4b>` | Local Qwen3-4B text-encoder `.safetensors` (the same file as `--clip_path` in the INT8 how-to) |
| `<K>` | Integer: how many lowest-impact layers to convert to NVFP4 (search this; it is not a fixed number) |
| `impact_<zit_unet>.json` | **Not a download.** Step 1 **creates** this file. The third argument of `Z_Image/diag_impact.py` is the **output path**. Typical location: the clone directory |

**INT8 input:** the `--output` file from the INT8 how-to, typically
`<path-to-unet>/<zit_unet>_convrot_int8.safetensors`. If you already saved that file under another name,
use that path as-is.

## Requirements

| Item | Value |
|---|---|
| Python | CUDA-enabled PyTorch, `safetensors`, `scikit-image`, `comfy-kitchen` |
| Bench | `benchmark/zi_convrot_nvfp4_bench_native.py` (this repo, **unmodified**) |
| ComfyUI | `<path-to-ComfyUI>` as defined above (Z-Image loading / Qwen3-4B text encoder) |
| Input ① | `<path-to-unet>/<zit_unet>.safetensors` (base fp16/bf16 NextDiT) |
| Input ② | `<path-to-unet>/<zit_unet>_convrot_int8.safetensors` (complete ConvRot INT8 from the INT8 how-to; int8_tensorwise, convrot:true, `model.diffusion_model.` prefix) |
| GPU | **VRAM ≥ 12GB** · **run one process at a time** (concurrent runs cause VRAM exhaustion) |

## Overall flow

```
ConvRot INT8 (208 layers, error ≈ 0)
   │  Step 1: run Z_Image/diag_impact.py — this **writes** impact_<zit_unet>.json (≈12 min)
   ▼
impact_<zit_unet>.json (created here; not shipped in the clone; not copied from another model)
   │  Step 2: reverse conversion (convert K lowest-impact layers to NVFP4, ≈1 min)
   ▼
<zit_unet>_hswq_hybrid_nv{K}_convrot_nvfp4.safetensors
   │  Step 3: native bench (existing, unmodified, all 5 seeds)
   ▼
Pass only if every seed's SSIM (0-255 view) ≥ 0.97
```

`K` is **not** a fixed number. Measure impact, convert, then **search K** until the largest value
that still passes all 5 seeds.

---

## Step 1. Create `impact_<zit_unet>.json` (all 208 layers, ≈12 min)

**How you make this file:** run `Z_Image/diag_impact.py` from the clone directory. There is **no**
ready-made `impact_*.json` in the repository and **no** download. You do **not** write the JSON by
hand. You do **not** reuse another checkpoint's JSON (ranking is not transferable).

The command has **three positional arguments**:

1. **Input** — base fp16/bf16 UNet: `<path-to-unet>/<zit_unet>.safetensors`
2. **Input** — complete ConvRot INT8 from the INT8 how-to: `<path-to-unet>/<zit_unet>_convrot_int8.safetensors`
3. **Output** — path of the JSON **this command will create**. A relative name such as
   `impact_<zit_unet>.json` writes into the **clone directory** (the cwd). Replace `<zit_unet>` with
   the same stem as the INT8 how-to. Example: if the UNet stem is `my_zit_v1`, the third argument is
   `impact_my_zit_v1.json`.

What the script does: inject NVFP4 error (**e4m3, group 256 reconstruction**) into **one layer at a
time**, run a fixed-seed 4-step denoising trajectory, and measure how far the final x drifts
(relative MSE). That value is the layer's **true importance under real trajectory propagation**.
When it finishes it prints `saved <that path>` and `DONE`. Until those lines appear, the file is
not ready for Step 2.

From the **clone directory** (`Hybrid-Sensitivity-Weighted-Quantization`):

```bash
python Z_Image/diag_impact.py "<path-to-unet>/<zit_unet>.safetensors" \
  "<path-to-unet>/<zit_unet>_convrot_int8.safetensors" \
  "impact_<zit_unet>.json" \
  --comfy-path "<path-to-ComfyUI>"
```

If `<path-to-ComfyUI>` is the bundled tree, that last flag is `--comfy-path ComfyUI-master`.
A relative `--comfy-path` is resolved against the clone directory (the parent of `Z_Image/`), not the process cwd, so `ComfyUI-master` still works when Jupyter `!python` leaves cwd elsewhere. The clone must contain `ComfyUI-master/comfy/` (the `comfy` package lives there, not at the clone root).
Both weight arguments use the same basename search as the ConvRot bench loader: if the given path is missing (for example `Hybrid-Sensitivity-Weighted-Quantization/test2.safetensors` while cwd is already the clone), the script looks up `test2.safetensors` under cwd / the clone, the same way `test.safetensors` is found.

**Created file** (clone directory unless you passed an absolute path) →
`{"x_ref_norm": ..., "impacts": {<layer>: <relative MSE>, ...}}`

Typical ranking tendencies (always re-measure per checkpoint; ranking is not transferable):

- **Smallest (safest to convert)**: `noise_refiner.*.attention.qkv`-class layers
- **Largest (must protect)**: `t_embedder.mlp.2`, `final_layer.linear`, `final_layer.adaLN_modulation.1`

---

## Step 2. Reverse conversion (≈1 min)

Use the **same** `impact_<zit_unet>.json` that Step 1 just wrote. If that file is missing, run Step 1;
do not invent the JSON.

Convert the **K** layers with the smallest impact to NVFP4, in ascending impact order.

The INT8 weights are stored **already rotated (W@H^T)**. Dequant (`q × scale`) gives the rotated
W_rot approximation. Quantize that with Kitchen **without re-rotating**.

From the **clone directory**:

```bash
python Z_Image/gen_reverse_nvfp4.py <K> \
  "<zit_unet>_hswq_hybrid_nv<K>_convrot_nvfp4.safetensors" \
  "<path-to-unet>/<zit_unet>_convrot_int8.safetensors" \
  "impact_<zit_unet>.json" \
  --out-dir "<path-to-unet>"
```

Example: if you choose `K=74`, the second argument is
`<zit_unet>_hswq_hybrid_nv74_convrot_nvfp4.safetensors` (the number in the filename is the same `K`).

What this does:

1. Rank layers by `impact_<zit_unet>.json` **ascending** (lowest impact first).
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

From the clone directory, enter `benchmark/`:

```bash
cd benchmark
```

**`--steps 25` and `--native-dtype` are required.** The script default is `--steps 20` and `--native-dtype` off — those defaults will **not** match this procedure.

```bash
python zi_convrot_nvfp4_bench_native.py \
  --fp16 "<path-to-unet>/<zit_unet>.safetensors" \
  --nvfp4 "<path-to-unet>/<zit_unet>_hswq_hybrid_nv<K>_convrot_nvfp4.safetensors" \
  --clip_path "<path-to-qwen3-4b>" \
  --comfy_path "<path-to-ComfyUI>" \
  --prompt "A beautiful cyberpunk city at night, high detail." \
  --steps 25 --seed 42 --native-dtype
```

If `<path-to-ComfyUI>` is the bundled tree, `--comfy_path` is `../ComfyUI-master` (you are inside `benchmark/`).

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
| `impact_<zit_unet>.json` does not exist | It is **created by Step 1**. Run `Z_Image/diag_impact.py`; the third argument is the output path |
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
| `Z_Image/diag_impact.py` | Step 1: **creates** `impact_<zit_unet>.json` (third argument = output path) |
| `Z_Image/gen_reverse_nvfp4.py` | Step 2: reverse hybrid converter (INT8 → NVFP4, K lowest-impact layers) |
| `benchmark/zi_convrot_nvfp4_bench_native.py` | Step 3: native bench (bf16 native baseline, `--native-dtype`) |
| `native_convert_int8_convrot_zi.py` | INT8 prerequisite (see [How to quantize Z Image.md](How%20to%20quantize%20Z%20Image.md)) |

**Dependencies:** `Z_Image/diag_impact.py` loads the Z-Image model via `benchmark/zi_convrot_nvfp4_bench.py`. Run Step 1 from the clone directory so the default root is correct. `Z_Image/gen_reverse_nvfp4.py` needs the pip package `comfy-kitchen`. The native bench stays in `benchmark/` because it shares local modules with sibling bench scripts — **run it from `benchmark/`**.

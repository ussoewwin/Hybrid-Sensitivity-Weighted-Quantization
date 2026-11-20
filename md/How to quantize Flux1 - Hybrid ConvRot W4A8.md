# How to quantize Flux1 - ConvRot INT8 and Hybrid ConvRot W4A8

> **Two stages, one document.**
> Stage 1 builds the **native ConvRot INT8** pack; Stage 2 turns that pack into a
> **hybrid ConvRot W4A8** pack with the reverse method and validates it with the
> deterministic 25-seed latent-trajectory comparison.
>
> The INT8 text is kept in full and unchanged; the W4A8 hybrid sections follow it.

---

# How to quantize Flux1 ConvRot INT8

Quantize a **FLUX.1 DiT** (`model.diffusion_model.*` layout) to **ConvRot INT8** with the
command-line converter `Flux1/native_convert_int8_convrot_flux.py` (native = hswq-free:
every eligible Linear is rotated and packed INT8, no protection set).

The post-convert fidelity benchmark is **latent-space trajectory divergence** (per-step
cosine + bifurcation detection) on the same seed for both models, which measures the
sampling path itself rather than a single forward pass.

FLUX needs **two text encoders** (`t5xxl` + `clip_l`) for any benchmark run - both are
passed explicitly (`--clip_path` = t5xxl, `--clip_l_path` = clip_l).

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
pip install -r requirements.txt
pip install -U comfy_kitchen
pip install scikit-image
```

`comfy_kitchen` provides the quantization kernels and layout operations.
`scikit-image` is required for the SSIM / MSE image-space benchmark.

## Quantize a Flux1 model (CLI)

Replace every `<...>` placeholder with a real path on your machine. `--model` and `--input`
are aliases for the same argument.

```bash
python Flux1/native_convert_int8_convrot_flux.py \
  --model "<path-to-unet>/<flux1>.safetensors" \
  --output "<path-to-unet>/<flux1>_native_convrot_int8.safetensors" \
  --no-bench
```

**Notes:**

- **FULL ConvRot** (Linear + Conv2d when `in_dim` is divisible by a power-of-4 group size)
  is **ON by default**. Pass `--no-convrot` only for plain INT8 without ConvRot.
- **Group size:** preferred size is `256`; when `in_features` is not divisible by 256 the
  converter adapts to the largest power-of-4 divisor (>= 4). The stamp written per layer is
  `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":N}`.
- **`--no-bench`:** pass this to convert and save only. When `--clip_path` / `--clip_l_path` /
  `--comfy_path` (and optionally `--vae`) are given instead, the converter runs the
  image-space benchmark automatically after saving; `--no-bench` skips it.
- The output keeps the source's keys: load it with the **standard ComfyUI loader**
  (`Load Diffusion Model` / `UNetLoader`). No dedicated loader is required, and the FLUX
  text encoders / VAE stay separate files as usual.

## Validation & Benchmarking

### Latent trajectory divergence (deterministic 25-seed)

Compares the FP16 source and the ConvRot INT8 pack step by step with the same initial
noise per seed:

```bash
python benchmark/flux1_convrot_int8_traj_compare.py \
  --fp16 "<path-to-unet>/<flux1>.safetensors" \
  --int8 "<path-to-unet>/<flux1>_native_convrot_int8.safetensors" \
  --clip_path "<path-to-clip>/<t5xxl>.safetensors" \
  --clip_l_path "<path-to-clip>/<clip_l>.safetensors" \
  --comfy_path "<path-to-ComfyUI>" \
  --steps 25 \
  --seeds "42,137,849,2024,7391,18429,53082,149206,382715,826401,1938502,4710928,8391642,15820493,36192847,71058294,128491703,285039184,491730285,762019483,938174026,1409285713,2683910547,3851729406,4195820371"
```

- Same seed -> identical noise for both models (fully deterministic).
- Reports per-step + final latent MSE / cosine, the largest single-step cosine drop, and
  `same-image` (final cosine >= 0.98) / `bifurcated` (a single step drops > 0.05) / `drifted`.
- Both CLIPs are loaded on CPU, encoded, and unloaded before the DiT runs, so the whole GPU
  belongs to the diffusion model.

### Image-space benchmark (MSE / SSIM)

```bash
python benchmark/flux1_convrot_int8_traj_compare.py \
  --fp16 "<path-to-unet>/<flux1>.safetensors" \
  --int8 "<path-to-unet>/<flux1>_native_convrot_int8.safetensors" \
  --clip_path "<path-to-clip>/<t5xxl>.safetensors" \
  --clip_l_path "<path-to-clip>/<clip_l>.safetensors" \
  --comfy_path "<path-to-ComfyUI>"
```

Score files belong in `benchmark result/` in the repository (per the repository's score
convention).

## ComfyUI Deployment

The resulting `.safetensors` is compatible with standard ComfyUI:

1. Place the quantized file into `ComfyUI/models/diffusion_models/` (or `models/unet/`).
2. Load it with **Load Diffusion Model** (or **UNetLoader**).
3. Load the FLUX text encoders as usual: **DualCLIPLoader** with `clip_l` + `t5xxl`
   (type `flux`), and the FLUX VAE with **VAELoader**.
4. `comfy_kitchen` executes the INT8 GEMM with online activation rotation - no extra node
   is needed for the pack itself.

---

# Stage 2 - Hybrid ConvRot W4A8 (reverse method)

> **Prerequisite**: the complete ConvRot INT8 pack from Stage 1 (`<int8>`).
> The hybrid is built **from that pack**, not from the fp16 source.

The reverse method starts from the complete ConvRot INT8 model (error already near zero) and
moves the **lowest-impact layers first** into ConvRot W4A8. Conventional "protect the important
layers" selection ignores inter-layer interaction; the reverse method stays in the low-error
regime where per-layer ranking is valid, so a single-layer impact ranking can be trusted.

A hybrid file therefore holds three layer kinds:

| Layer kind | Format | Which layers |
|---|---|---|
| W4A8 | `asym_w4a8_int8` (4-bit codes, INT8 activations) | the K lowest-impact layers of the ranking |
| ConvRot INT8 protect | `int8_tensorwise` + `convrot` | the remaining converted layers |
| Base dtype | source dtype, untouched | boundary layers (`img_in` / `txt_in` / `vector_in` / `time_in` / `guidance_in` / `final_layer` / `*_mod` / `modulation` / `norm`) |

## W4A8 pack layout (what the runtime reads)

| Key | dtype | Shape | Meaning |
|---|---|---|---|
| `<layer>.weight` | int8 | `[N, K/2]` | nibble-packed 4-bit codes, low nibble = even column |
| `<layer>.weight_s_rel` | fp8_e4m3 | `[N, K/16]` | per-group (16) scale |
| `<layer>.weight_s_channel` | float32 | `[N]` | per-out-channel scale |
| `<layer>.weight_codebook` | float32 | `[16]` | Lloyd-Max table |
| `<layer>.comfy_quant` | uint8 | JSON | `{"format":"asym_w4a8_int8","group_size":16,"convrot_groupsize":256}` |

ComfyUI core already decodes this format (`comfy/ops.py`, grouped-int8 branch), so the pack
loads without a dedicated loader.

## Step 1 - Per-layer impact ranking

```bash
python Flux1/diag_impact.py \
  "<path-to-unet>/<flux1>.safetensors" \
  "impact/impact_<stem>.json" \
  --clip_path "<path-to-clip>/<t5xxl>.safetensors" \
  --clip_l_path "<path-to-clip>/<clip_l>.safetensors" \
  --comfy_path "<path-to-ComfyUI>" \
  --steps 25 \
  --seed 42 \
  --width 1024 \
  --height 1024
```

- Measures each eligible layer by replacing only that layer's weight with its ConvRot INT8
  reconstruction and replaying the whole sampling trajectory: a **trajectory** error, not a
  single-forward error.
- Boundary layers are excluded from the ranking (never candidates).
- One pristine trajectory plus one trajectory per measured layer; the run is long by design.
- The JSON is checkpoint-specific - never reuse another model's impact file.

## Step 2 - Convert the K lowest-impact layers to W4A8

`K` counts the lowest-impact layers moved to W4A8. `K` equal to the ranked set converts every
eligible layer; nothing outside the ranking is touched. The example below uses **K = 170** on
the 228-layer FLUX.1 ranking (170 of 228 converted layers become W4A8, the other 58 stay
ConvRot INT8).

```bash
python Flux1/gen_reverse_w4a8.py 170 \
  "<flux1>_hybrid_w170_convrot_w4a8.safetensors" \
  "<path-to-unet>/<flux1>_native_convrot_int8.safetensors" \
  "impact/impact_<stem>.json" \
  --out-dir "<path-to-output-dir>"
```

- Positional arguments: `K`, output file name, INT8 pack, impact JSON.
- `--out-dir` must already exist; the writer creates its temp file inside it.
- Layers whose `comfy_quant` is not ConvRot INT8 with `convrot_groupsize` 256 are skipped
  (the W4A8 requant path needs 256).
- Source weights are rotated and the W4A8 packer rotates internally, so the converter
  un-rotates first. A double rotation yields noise, not a small error.

## Step 3 - Trajectory validation (deterministic 25 seeds)

```bash
python benchmark/flux1_convrot_w4a8_traj_compare.py \
  --fp16 "<path-to-unet>/<flux1>.safetensors" \
  --w4a8 "<path-to-unet>/<flux1>_hybrid_w170_convrot_w4a8.safetensors" \
  --clip_path "<path-to-clip>/<t5xxl>.safetensors" \
  --clip_l_path "<path-to-clip>/<clip_l>.safetensors" \
  --comfy_path "<path-to-ComfyUI>" \
  --steps 25
```

- The quantized-side flag is `--w4a8` (aliases `--int8` / `--fp8` / `--quant`).
- `--seeds` defaults to the repository's fixed 25-seed set; do not substitute another list.
- Same seed means identical initial noise for both models.
- Verdicts: `same-image` (final cosine >= 0.98), `bifurcated` (one step drops > 0.05 = a
  different picture, not degradation), `drifted` (small acceptable divergence).
- Numbers are checkpoint-specific; record them per model and do not carry them over.

## Step 4 - What a W4A8 pack costs and buys

- A 4-bit layer stores `N*K/2` int8 bytes plus `N*K/16` fp8 group scales and `N` fp32 channel
  scales; ConvRot INT8 protect layers keep one byte per weight.
- Boundary layers stay in the base dtype by design. Packing `img_in` / `txt_in` /
  `vector_in.in_layer` would also halve the K that ComfyUI's FLUX detector reads, so those
  three are always kept full width.
- Size and fidelity are reported together. A smaller pack that fails the trajectory gate is
  not a result; a bigger pack that buys a few 1e-4 of cosine is not one either.

## Step 5 - ComfyUI deployment

1. Put the hybrid file in `ComfyUI/models/diffusion_models/` (or `models/unet/`).
2. Load it with **Load Diffusion Model** (or **UNetLoader**): ComfyUI core reads
   `asym_w4a8_int8` and runs the INT8-activation GEMM.
3. Text encoders and VAE stay separate: **DualCLIPLoader** (`clip_l` + `t5xxl`, type `flux`)
   and **VAELoader**.
4. **LoRA**: ComfyUI core has no float path for a 4-bit packed weight and 16 code levels round
   a LoRA delta away. Use the HSWQ loader for Flux1 W4A8 (`nodes/flux1 convrot w4a8`), which
   keeps the pack untouched and adds the LoRA delta as a factorised residual after the GEMM.
   A stock loader does not apply the delta correctly to a W4A8 pack.

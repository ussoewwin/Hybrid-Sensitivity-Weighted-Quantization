# How to quantize Flux1 - ConvRot INT8 and Hybrid ConvRot W4A8

> **Two stages, one document.**
> Stage 1 builds the **native ConvRot INT8** pack; Stage 2 turns that pack into a
> **hybrid ConvRot W4A8** pack with the reverse method and validates it with the
> deterministic 25-seed latent-trajectory comparison.
>
> The INT8 text is kept in full and unchanged; the W4A8 hybrid sections follow it.
>
> **Hardware**: a CUDA GPU with **24 GB or more VRAM** is recommended. Both stages run a
> 12B-class FLUX.1 DiT plus one text encoder at a time, and the trajectory benchmarks keep a
> full sampling run in memory; below 24 GB the runs fall back to offload and take far longer.
> Disk: keep room for the fp16 source, the INT8 pack and the hybrid pack together.

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
>
> **Hardware**: **24 GB or more VRAM** is recommended here too - the conversion opens the
> INT8 pack and writes the hybrid pack while the impact ranking was measured on the full
> model, so a smaller card only adds offload time without changing the result.

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

## Stage 2 gate - the INT8 score must reach 0.96

Before any layer ranking, run the Stage 1 trajectory comparison on the INT8 pack and require **mean final cosine >= 0.96** with **0/25 bifurcated**. A W4A8 hybrid only moves 4 bits into a pack that is otherwise 8 bits; if the 8-bit pack already drifts, the hybrid inherits the drift and adds to it.

A Flux1 checkpoint can fail this gate on the INT8 pack alone. That is a property of the model, not a fault in the converter. One measured Flux1 checkpoint, converted with 314 of 314 weights quantised and the ConvRot stamp on every one of them, scored mean 0.86099, min 0.62984, max 0.98339, same-image 1/25 and bifurcated 6/25. It cannot tolerate even 8 bits, so there is nothing for W4A8 to build on.

When that happens, stop at Stage 1 for that checkpoint and keep it at a higher precision. Building the hybrid anyway spends the ranking run, the conversion run and the 25-seed validation to reach a pack that is worse than the INT8 one it sits on.

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

## Step 2b - Card 1 bias correction (optional)

A 4-bit weight carries a systematic error, and a Linear/Conv layer turns that error into a
constant offset in its output. Card 1 folds the offset back into the layer bias:

```
bias += -(W_q - W) @ E[x]
```

`W_q` is the quantised weight, `W` the original weight, `E[x]` the per-input-channel mean of
that layer's input. The pack stores `bias - delta`; the runtime adds no extra work.

The delta has to be measured against the **FP16 weight**, not the INT8 dequant. The reverse
converter starts from the INT8 pack, so an INT8-based reference covers only the INT8 -> W4A8
step and under-corrects the pack (measured on one FLUX.1 checkpoint: bias ON 0.95218 vs
OFF 0.953). Passing `--fp16` makes the converter read the original weight and bias from that
file instead.

`--bias_correction` runs a calibration: the DiT is loaded through ComfyUI, both text encoders
build the calib contexts, and a DualMonitor records the per-layer `E[x]` over
`--num_calib_samples` x `--num_inference_steps` forwards. There is no DualMonitor-less path,
so `--calib_file`, `--clip_path`, `--clip_l_path` and `--comfy_path` are all required.

```bash
python Flux1/gen_reverse_w4a8.py 170 \
  "<flux1>_hybrid_w170_convrot_w4a8_bc.safetensors" \
  "<path-to-unet>/<flux1>_native_convrot_int8.safetensors" \
  "impact/impact_<stem>.json" \
  --bias_correction \
  --fp16 "<path-to-unet>/<flux1>.safetensors" \
  --calib_file "sample/calibration_prompts_128.txt" \
  --clip_path "<path-to-clip>/<t5xxl>.safetensors" \
  --clip_l_path "<path-to-clip>/<clip_l>.safetensors" \
  --comfy_path "<path-to-ComfyUI>" \
  --num_calib_samples 32 \
  --num_inference_steps 25 \
  --out-dir "<path-to-output-dir>"
```

- **The INT8 pack must stay un-corrected.** If the Stage 1 pack already carries a bias
  correction, the lifted layers would carry both deltas. The native Stage 1 converter writes
  no bias correction, so its pack is the right input.
- Only the layers lifted to W4A8 are touched. The layers that stay ConvRot INT8 keep their
  bias byte for byte.
- The run aborts instead of saving when zero layers were corrected, so a pack labelled
  bias-corrected always carries a real correction.
- Calibration recipe: `--num_calib_samples 32` x `--num_inference_steps 25`,
  `--latent_size 32`. Both are explicit so the numbers are reproducible.
- Compare ON and OFF with the same benchmark from Step 3 and report the two means together;
  a single run of one setting is not a comparison.

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

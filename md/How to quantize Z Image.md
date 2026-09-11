# How to quantize Z Image (ZI)

Use **native ConvRot INT8** via CLI (`Z_Image/native_convert_int8_convrot_zi.py`) or directly in ComfyUI with the dedicated custom node **`Z Image ConvRot INT8 Quantize`** (`comfyui_nodes/`).

**HSWQ Z Image INT8 development and public release ended.** For Z Image 8-bit, use this native path (typically **SSIM > 0.99**). HSWQ INT8 continues for **SDXL** only. Progressive in-graph quantization support for ComfyUI nodes is being expanded sequentially.

**Prefer a Z Image Turbo (ZIT) checkpoint.** Plain Z Image base models are not recommended.

## Validation premise - SA2 attention (read first)
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

The fidelity gate for a ConvRot INT8 conversion is measured with **SageAttention2 attention
acceleration** (`--attention sage2`, INT8 QK + FP8 PV, sm120 path). **SA2 is the production
attention configuration** for this path. If the report footer says `attention mode   : sdpa`,
SA2 was **not** applied and the run is **not** a gate result.

**Judgement criteria:** `final-cosine mean >= 0.95` and `0/20 bifurcated`.

**Fixed conditions (always pass these explicitly):**

- **Fixed bench:** `benchmark/zi_traj_compare.py` - per-step latent trajectory cosine (FP16 vs ConvRot INT8)
- **Fixed 20-seed set:** `--canonical-seeds`
  (`42, 137, 5517, 92048, 371506, 5293047, 64820153, 731509284, 8426170395, 9517038246, 210987, 6543210,
  98765432, 1357924680, 2468135791, 3579246812, 4680357923, 5791468034, 6802579145, 7913680256`)
- **Fixed steps:** `--steps 12` (the script default is 25 - always pass 12 explicitly)
- **Fixed prompt / CFG / sampler:** script defaults (`masterpiece, best quality, 1girl, solo, standing,
  simple background`, cfg `2.5`, `euler` / `simple`)
- **Fixed attention:** `--attention sage2` - SageAttention2 (INT8 QK + FP8 PV, sm120 path), the
  accelerated production attention
- **Report:** print every per-seed row plus the multi-seed summary; on an SA2 run the
  `[SAGE2] attention calls` line must show `sa2 = total` with `errors=0`

Reference measurement (2026-09-10, 20 seeds x 12 steps, 1024x1024, RTX 5060 Ti): the INT8 branch ran
**18.05 s -> 16.00 s per seed (-11.4 %, 1.128x, 20/20 seeds faster)** with SA2; `final-cosine` mean
`0.99122 -> 0.99043` with `0/20 bifurcated` in both runs.

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

`comfy_kitchen` provides the quantization kernels and layout operations. The validation gate runs `benchmark/zi_traj_compare.py` (see **Validation premise** above).

## Quantize a ZI model (CLI)

Replace every `<...>` placeholder with a real path on your machine (no invented filenames; no machine-local drive hardcoding in published examples). `--model` and `--input` are aliases for the same argument.

**Default flow:** convert → save. The converter attempts an automatic post-convert bench, but the
**gate measurement is the explicit SA2 run** below. A bench that ran with `attention mode   : sdpa`
is not a gate result.

Required: **`--model`**, **`--output`**, **`--per_channel_int8`**, **`--clip_path`**, **`--comfy_path`** only. Tokenizer uses ComfyUI-bundled `comfy/text_encoders/qwen25_tokenizer` under `--comfy_path`.

```bash
python Z_Image/native_convert_int8_convrot_zi.py --model "<path-to-unet>/<zit_unet>.safetensors" --output "<path-to-unet>/<zit_unet>_convrot_int8.safetensors" --per_channel_int8 --clip_path "<path-to-qwen3-4b>" --comfy_path "<path-to-ComfyUI>"
```

**Notes:**

- **FULL ConvRot** (Linear + Conv2d when `in_dim` is divisible by a power-of-4 group size) is **ON by default**. Pass `--no-convrot` only for plain INT8 without ConvRot.
- **`--per_channel_int8`:** use per-out-channel amax/scale instead of a single per-tensor scale when packing layers that do **not** go through ConvRot. Under default FULL ConvRot, almost all eligible Linear/Conv2d already use rotate + per-channel scale, so this flag has **little effect** in practice; keep it as **insurance** for any remaining non-ConvRot packs. Format tag stays `int8_tensorwise`.
- **Gate measurement (SA2):** run the fixed command below. It reuses the same `--clip_path` / `--comfy_path`.

  ```bash
  python benchmark/zi_traj_compare.py \
      --fp16 "<path-to-unet>/<zit_unet>.safetensors" \
      --fp8 "<path-to-unet>/<zit_unet>_convrot_int8.safetensors" \
      --clip_path "<path-to-qwen3-4b>" --comfy_path "<path-to-ComfyUI>" \
      --steps 12 --canonical-seeds --attention sage2
  ```

  PASS = `final-cosine mean >= 0.95` with `0/20 bifurcated`; the footer must show `attention mode   : sage2`.
- **Post-convert bench:** the converter also attempts an automatic bench after saving. Treat it as a
  smoke test only - the SA2 command above is the gate.
- **ComfyUI:** Load the ConvRot INT8 output with the **standard ComfyUI loader**. A dedicated HSWQ loader is not required.

## Quantize a ZI model via ComfyUI (Node)

Quantization can also be executed directly within ComfyUI using the custom node **`Native ConvRot INT8 Quantize`** (`comfyui_nodes/`).

<p align="left">
  <img src="../png/native_convrot_int8.png" alt="ComfyUI Native ConvRot INT8 Quantize Workflow" width="600">
</p>

### Installation

Copy or link the repository into `ComfyUI/custom_nodes/`:

```bash
cd custom_nodes
git clone https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization.git
```

### Sample Workflow

A ready-to-use ComfyUI workflow JSON is provided in the repository:
- **[`sample workflow/native convrot int8.json`](../sample%20workflow/native%20convrot%20int8.json)**

You can load this file directly into ComfyUI (or drag-and-drop the workflow image) to load the complete quantization and benchmark graph.

### Node Workflow & Usage

1. **Load Model:** Connect the `MODEL` output from `UNetLoader` (or `Load Diffusion Model`) to the `model` input of the **`Native ConvRot INT8 Quantize`** node.
2. **Connect CLIP:** Connect `CLIP` from `CLIPLoader` (e.g. `qwen3_4b_abliterated_fp16_converted.safetensors`) to the `clip` input.
3. **Optional VAE:** Connect `VAE` from `VAELoader` to the optional `vae` input for automatic decoded SSIM measurement.
4. **Configure Parameters:**
   - **`model_type`**: Target model architecture selector (`"Z Image"`, `"Qwen Image Edit"`).
   - **`benchmark_prompt`**: Prompt text used during the automated 10-seed baseline vs quantized benchmark (multiline text, defaults to `"masterpiece, best quality, 1girl, solo, standing, simple background"`).
   - **`output_path`**: Destination `.safetensors` path. If left empty, saves to the ComfyUI output directory automatically with a timestamped filename.
   - **`group_size`**: Preferred ConvRot Hadamard group size (default `256`, must be a power of 4).
   - **`convrot`**: Enable FULL ConvRot online Hadamard rotation (default `True`).
   - **`per_channel_int8`**: Channelwise amax/scale fallback for non-ConvRot layers (default `True`).
   - **`run_benchmark`**: Automatically run a 10-seed fidelity benchmark (latent MSE, cosine similarity, inference time, and decoded SSIM) upon save (default `True`).
5. **Execute Queue:** Run the prompt queue. The node extracts diffusion weights directly from memory, performs Hadamard rotation and INT8 symmetric quantization, saves the model checkpoint with `_quantization_metadata`, and outputs the benchmark report to the console and return output.


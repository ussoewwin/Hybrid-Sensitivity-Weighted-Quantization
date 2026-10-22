# Z-Image-Turbo-Fun-Controlnet (Union 2.1 / Tile 2.1 lite) — ConvRot INT8 Model Patch Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`How to quantize Text Encoder and ControlNet.md`](How to quantize Text Encoder and ControlNet.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this model.

Two packs, the same architecture and the same result:

| | Union 2.1 lite | Tile 2.1 lite |
| :--- | :--- | :--- |
| the source | `Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps.safetensors` (1923.2 MB, BF16) | `Z-Image-Turbo-Fun-Controlnet-Tile-2.1-lite-2601-8steps.safetensors` (1923.2 MB, BF16) |
| the pack | `...Union-2.1-lite-2602-8steps_convrot_int8.safetensors` | `...Tile-2.1-lite-2601-8steps_convrot_int8.safetensors` |
| the size | **963.0 MB (−49.9 %)** | **963.0 MB (−49.9 %)** |
| the SHA256 | `1DD12B2D028B12DF6B2A4F93856F8AB161C1EDC05125E3B1C987A1694E99DBA4` | `2998F83EF65903B4B6B22FB7AC270ED66F9A2443220563AA43DB4188024763A8` |
| the INT8 | **38 of 38** (no protection) | **38 of 38** (no protection) |

---

## 0. This is a model patch, not a ControlNet

The file names say "Controlnet" but the runtime object is a **model patch**.
ComfyUI loads it with **`ModelPatchLoader`** / **`HSWQModelPatchLoaderCustom`**
(both read `models/model_patches`) and applies it with the
**`ZImageFunControlnet`** node. It is dispatched on the key
`control_all_x_embedder.2-1.weight` into
`comfy.ldm.lumina.controlnet.ZImage_Control`.

`comfy.controlnet.load_controlnet_state_dict` cannot load it - that path has no
Z-Image branch, and the call returns `None` with
`error could not detect control model type`. This is expected, not a defect.

### The load path

| step | what happens |
| :--- | :--- |
| the loader | `HSWQModelPatchLoaderCustom` (or the stock `ModelPatchLoader`) |
| the dispatch | `control_all_x_embedder.2-1.weight` present → the Z-Image branch |
| the key conversion | `z_image_convert` fuses `attention.{to_q,to_k,to_v}` into `attention.qkv` and renames `attention.to_out.0` → `attention.out` |
| the config | `n_control_layers` counted from the keys (3), `additional_in_dim=17`, `refiner_control=True`, `control_in_dim` inferred from the embedder width |
| the class | `comfy.ldm.lumina.controlnet.ZImage_Control` |
| the apply node | `ZImageFunControlnet` → `set_model_noise_refiner_patch` + `set_model_double_block_patch` |

The pack is written in the **post-conversion layout** (`attention.qkv` already
fused), because the loader runs `z_image_convert` on load for the **non-INT8**
path only - the INT8 branch skips it, so the quantized keys have to be stored
already converted.

The INT8 branch builds the model with `mixed_precision_ops` instead of
`manual_cast`, so the weights stay INT8 in VRAM and the forward uses the
comfy-kitchen `int8_linear` kernel with online ConvRot activation rotation.

---

## 1. The model

| item | measured |
| :--- | :--- |
| the source | 1923.2 MB, BF16, 91 tensors |
| the 2D weights (pre-conversion) | 48 |
| the runtime 2D weights (post-`z_image_convert`, qkv fused) | **38** |
| the layer families | `control_layers.{0,1,2}`, `control_noise_refiner.{0,1}`, `control_all_x_embedder` |
| the group size | 256 (37 layers) / 4 (the `control_all_x_embedder` 132-wide layer) |

`control_all_x_embedder.2-1` is `[3840, 132]`: 33 channels × 2×2 patch grid × 1
frame. 132 is not a multiple of 256, so it falls back to group size 4.

---

## 2. The acceptance

Two levels, both measured.

### 2.1 The model patch's own output

`Z_Image/verify_zimage_modelpatch.py` loads the pack the way the loader does
(the INT8 ops) and the source with `manual_cast`, runs the patch's own forward
(`model(cap_feats, control_context, x_freqs_cis, adaln_input)`) over 4 fixed
seeds:

| | Union | Tile |
| :--- | :--- | :--- |
| cos mean | **0.999816** | **0.999903** |
| cos min | **0.999815** | **0.999897** |
| relMSE | 1.009e-04 | 8.521e-05 |
| nonfinite | 0 | 0 |

### 2.2 The denoising trajectory (the real downstream)

The protection set is chosen on the patch's output, but what matters is the
denoising trajectory of the Z-Image DiT with the patch applied.
`Z_Image/verify_zimage_modelpatch_traj.py` runs the real pipeline - the
**BF16 Z-Image base** (`z_image_turbo_bf16.safetensors`), the patch wired in
exactly as `ZImageFunControlnet` does (`set_model_noise_refiner_patch` +
`set_model_double_block_patch`), the production sampler
(`res_multistep` / `simple`) - and compares the per-step latents against the
BF16 patch's run.

The VAE / text encoder are not on this machine, so the control image and the
text conditioning are fixed random tensors of the right shape
(`control_context` 33ch, `cap_feats` 2560-wide = the Qwen3-4B output). The
content does not matter for the measurement; the same tensors go through both
runs, which is the same principle the ControlNet sweep uses.

| | Union (2 seeds × 4 steps) | Tile (2 seeds × 4 steps) |
| :--- | :--- | :--- |
| cos mean | **0.999862** | **0.999956** |
| cos min | **0.999491** | **0.999813** |
| relMSE | 2.773e-04 | 8.778e-05 |
| nonfinite | 0 | 0 |
| the gate | **PASS** | **PASS** |

**All 38 layers are quantized, none protected.** The measured trajectory is at
0.9999 with every layer in INT8, so the protection set is empty at any
threshold - this is the floor case. `native` (full quantization) is sufficient.

---

## 3. Files

| file | role |
| :--- | :--- |
| `Z_Image/build_zimage_modelpatch_convrot_int8.py` | the builder (post-conversion layout, `--protect-list`) |
| `Z_Image/verify_zimage_modelpatch.py` | the patch-output acceptance (4 seeds) |
| `Z_Image/verify_zimage_modelpatch_traj.py` | the trajectory acceptance (the real pipeline) |
| `comfyui_nodes/hswq_model_patch_loader.py` | the INT8-aware model-patch loader (repo copy) |
| `impact/verify_zi_mp.json`, `impact/verify_zi_tile.json` | the patch-output results (git-ignored) |
| `impact/verify_zi_traj.json`, `impact/verify_zi_tile_traj.json` | the trajectory results (git-ignored) |

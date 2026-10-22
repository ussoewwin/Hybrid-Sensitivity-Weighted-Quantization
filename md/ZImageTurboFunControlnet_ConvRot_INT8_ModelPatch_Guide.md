# Z-Image-Turbo-Fun-Controlnet Union 2.1 lite — ConvRot INT8 Model Patch Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`How to quantize Text Encoder and ControlNet.md`](How to quantize Text Encoder and ControlNet.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this model.

**Model.** `models/model_patches/Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps.safetensors` (1923.2 MB, BF16, 91 tensors)
**Pack.** `models/model_patches/Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps_convrot_int8.safetensors` (963.0 MB, **−49.9 %**)
**SHA256.** `1DD12B2D028B12DF6B2A4F93856F8AB161C1EDC05125E3B1C987A1694E99DBA4`

---

## 0. This is a model patch, not a ControlNet

The file name says "Controlnet" but the runtime object is a **model patch**.
ComfyUI loads it with **`ModelPatchLoader`** (`comfy_extras.nodes_model_patch`)
and applies it with the **`ZImageFunControlnet`** node. It is dispatched on the
key `control_all_x_embedder.2-1.weight` into
`comfy.ldm.lumina.controlnet.ZImage_Control`.

`comfy.controlnet.load_controlnet_state_dict` cannot load it - that path has no
Z-Image branch, and the call returns `None` with
`error could not detect control model type`. This is expected, not a defect.

### The load path

| step | what happens |
| :--- | :--- |
| the loader | `ModelPatchLoader` / `HSWQModelPatchLoader` (both read `models/model_patches`) |
| the dispatch | `control_all_x_embedder.2-1.weight` present → the Z-Image branch |
| the key conversion | `z_image_convert` fuses `attention.{to_q,to_k,to_v}` into `attention.qkv` and renames `attention.to_out.0` → `attention.out` |
| the config | `n_control_layers=3`, `additional_in_dim=17`, `refiner_control=True` (derived from the checkpoint's own keys) |
| the class | `comfy.ldm.lumina.controlnet.ZImage_Control` |
| the apply node | `ZImageFunControlnet` |

### The INT8 load path

Stock `ModelPatchLoader` builds the patch with `comfy.ops.manual_cast`, which
would load the raw INT8 tensor as a plain Parameter. The **HSWQ loader**
(`comfyui_nodes/hswq_model_patch_loader.py`, shipped in the loader package)
detects the `int8_tensorwise` `comfy_quant` layers and builds the model with
`mixed_precision_ops` instead, so the weights stay INT8 in VRAM and the forward
uses the comfy-kitchen `int8_linear` kernel with online ConvRot activation
rotation.

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

## 2. The pack

Built with `Z_Image/build_zimage_modelpatch_convrot_int8.py`. The pack is
written in the **post-conversion layout** (`attention.qkv` already fused),
because the HSWQ model-patch loader runs `z_image_convert` on load and the
quantized keys have to survive it unchanged.

| item | value |
| :--- | :--- |
| the source | 1923.2 MB (BF16) |
| the pack | **963.0 MB** (-49.9 %) |
| the SHA256 | `1DD12B2D028B12DF6B2A4F93856F8AB161C1EDC05125E3B1C987A1694E99DBA4` |
| the tensors | 157 |
| the quantized layers | **38 of 38** (no protection needed - see below) |
| the group size | 256 / 4 |
| the kept tensors | the biases, norms and the 1D weights, in BF16 |

---

## 3. The acceptance

`Z_Image/verify_zimage_modelpatch.py` loads the pack the way
`HSWQModelPatchLoader` does (the INT8 ops) and the BF16 source with
`manual_cast`, runs the model patch's own forward
(`model(cap_feats, control_context, x_freqs_cis, adaln_input)`, exactly the
call the `ZImageFunControlnet` node makes) over 4 fixed seeds, and compares the
output:

| | value |
| :--- | :--- |
| cos mean | **0.999816** |
| cos min | **0.999815** |
| relMSE | **1.009e-04** |
| nonfinite | 0 of 3,932,160 |
| the gate | **PASS** (mean ≥ 0.99, min ≥ 0.98) |

**All 38 layers are quantized, none protected.** The measured output is at
0.9998 with every layer in INT8, so there is nothing to protect here: the
protection set would be empty at any threshold. This is the "floor" case -
quantizing everything already clears the gate by a wide margin.

---

## 4. The earlier 2601 pack

The `..._2601_..._convrot_int8.safetensors` pack in the same directory is the
earlier release, built from the 2601 checkpoint (whose source is no longer on
disk). Measured against the **2602** source it scores cos 0.995231 / relMSE
1.61e-02, i.e. worse than the pack above. That comparison is not a fair
like-for-like (different source checkpoints), so it is reported as a fact, not
as a verdict on the 2601 pack.

What can be checked directly is the **stored payload**: de-rotating the 2601
pack's layers and comparing against the 2602 source gives relMSE 1.40e-02 on
`control_layers.0.attention.qkv` and **0.62 on `control_layers.0.after_proj`** -
the 2601 pack's `after_proj` reconstruction does not match this source at all.
Again, different checkpoints, so this is not proof the 2601 pack is broken;
it is proof the two packs are not interchangeable.

---

## 5. Files

| file | role |
| :--- | :--- |
| `Z_Image/build_zimage_modelpatch_convrot_int8.py` | the builder (post-conversion layout, `--protect-list`) |
| `Z_Image/verify_zimage_modelpatch.py` | the acceptance (the HSWQ load path, 4 seeds) |
| `comfyui_nodes/hswq_model_patch_loader.py` | the INT8-aware model-patch loader |
| `impact/verify_zi_mp.json` | the verification result (git-ignored) |

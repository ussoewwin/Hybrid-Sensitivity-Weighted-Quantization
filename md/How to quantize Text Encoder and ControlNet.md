# How to quantize Text Encoder, ControlNet, Model Patch and SAM 3 / SAM 3.1 (the per-model track, outside the HSWQ universal engine)

> [!WARNING]
> **This chapter is not HSWQ's universal engine.** HSWQ proper is a *universal*
> quantizer: one autonomous engine whose every decision is derived from the
> target checkpoint's own statistics, with **no model-specific toggles**. That
> design is what makes it applicable to any new checkpoint of a supported
> family without new code.
>
> **The families in this chapter are not one family**, and universalisation does
> not carry over. Each model here needed its **own** measurement script, its
> **own** converter, its **own** loader path, and a **hand-set** protection
> threshold, because the runtime object, the observable, and sometimes the graph
> itself differ from checkpoint to checkpoint. The packing steps below are
> shared, but the *decisions* are per-model. Treat this chapter as a **per-model
> track** - a set of individually built artifacts with a common written method -
> **not** as an application of the universal engine. The per-model nature is
> quantified in section *Why this is not the universal engine*.

Quantize loaded **ControlNet / ControlNet Union models**, **Model Patch (model patcher) ControlNets**, and **Segment Anything Models (SAM 3 / SAM 3.1 Multiplex)** directly in ComfyUI using the dedicated custom node **`TE / ControlNet ConvRot INT8 Quantize`** (`comfyui_nodes/te_controlnet_convrot_int8_convert.py`) and loader **`HSWQ SAM3 Loader (sam3.pt)`** (`comfyui_nodes/sam3_pt_loader.py`), or via the standalone CLI conversion script **`clip_convert/convert_clip_convrot_int8.py`**.

By applying orthogonal Hadamard rotations prior to per-channel INT8 quantization, this pipeline eliminates activation outlier spikes in deep Linear layers, reducing VRAM and storage footprint by ~50% while preserving mask segmentation fidelity and structural guidance quality.

> [!IMPORTANT]
> **Text encoders and SAM are separate cases.** The generic CLI above quantizes every 2D Linear layer with **no FP16 protection** - it is the A/B baseline, not the recommended pack. HSWQ also publishes **individually built, higher-quality packs** (measured FP16 protection) for text encoders and SAM 3.1 on Hugging Face. Because text encoders and SAM are standards (one weight layout per architecture) rather than per-checkpoint models, these are finished artifacts rather than a universal method. The measured packs themselves live in [**HSWQ_High_Quality_Packs_ConvRot_INT8.md**](HSWQ_High_Quality_Packs_ConvRot_INT8.md); this chapter keeps the **native / A/B baseline** and the method. (SAM 3 is the same kind of case but its pack is not published for license reasons.)

> [!IMPORTANT]
> **The generic CLI above is the no-protection baseline.** It quantizes every 2D Linear layer with **no FP16 protection**. The recommended packs in this chapter are individually built: the protection set comes from a measured aggregate contribution, the threshold from the size/quality balance, and the kept tensors are stored FP16. The measured packs are in [**HSWQ_High_Quality_Packs_ConvRot_INT8.md**](HSWQ_High_Quality_Packs_ConvRot_INT8.md); the ControlNet / model-patch inventory below is the chapter's own.

---

## Why this is not the universal engine

HSWQ's own engine (`quantize_sdxl_hswq_v1.3.py`, `quantize_zib_hswq_v2.0.py`)
states its design constraint explicitly: **no model-specific CLI toggles**, all
detection automatic from module names and tensor statistics, one code path per
supported family. The point of that design is leverage - a new merge or finetune
of a supported family is quantized with no new code.

That is not how the packs in this chapter were built, and the difference is
measurable. Counting only the scripts written for this chapter:

| the artifact | the count | the nature |
| :--- | :--- | :--- |
| per-model **measurement** scripts (`sdxl/measure_*`) | **9** | one per model or tower split |
| per-model **converter** scripts (`clip_convert/convert_*`) | **17** | one per family, several with a second "impact" variant |
| per-model **verify** scripts (`sdxl/verify_*`) | **23** | one per model, several per tower |
| the **CLI arguments** across just the converters | **93** | 4 to 10 per script, not a shared interface |

The reason universalisation fails here is structural:

1. **The runtime object differs.** A file called "ControlNet" may be a real
   `ControlNet` (`load_controlnet_qwen_fun`, `load_controlnet_qwen_instantx`), a
   **model patch** (`ZImage_Control`, `QwenImageBlockWiseControlNet`), or a
   standard tower (`CLIP`, `T5`). They are loaded by different nodes, applied by
   different apply paths, and live in different directories.
2. **The observable differs.** The criterion is "the model's own output", but
   what that output *is* differs: a concatenation of per-block residuals (a list
   the runtime injects block by block), a single tower embedding, or a
   denoising trajectory. The scoring function therefore differs too (per-block
   mean, float64, and so on - Rule 8).
3. **The graph differs.** A Qwen-Image Fun ControlNet needs a base model only to
   reach its entry path; an InstantX ControlNet is self-contained. Some patches
   need `z_image_convert` on load, some must not be converted at all.
4. **The checkpoint's shape parameters differ inside a family.** Within the one
   Qwen-Image diffsynth family, `img_in` is `[3072, 64]` for canny/depth and
   `[3072, 68]` for inpaint. A single shared converter would have to branch on
   `additional_in_dim` - which is exactly the "shared code plus conditionals"
   that HSWQ's universal design avoids by being *genuinely* uniform, and that
   cannot be avoided here because the objects are not uniform.

5. **A standard has one model; a UNet family has many.** This is the economic
   reason the specialised approach is affordable here at all, and it is the
   single biggest difference from the UNet work. SDXL alone was tested on **30+
   checkpoints** (every popular merge and finetune is a new target whose ranking
   must be re-measured); Z-Image likewise across many merges. A text encoder, a
   ControlNet standard, or a SAM release is **one weight layout per standard** -
   the standard does not get merged, forked, or re-finetuned into new
   checkpoints the way an image-generation backbone does. There is nothing to
   re-measure per checkpoint, so a large *universal engine* - whose whole value
   is amortising one code path across many checkpoints - has nothing to amortise
   over. One measurement, one converter, one verification per standard, and the
   artifact is done.

So the honest description is: **a common written method, applied individually to
each model.** The *packing mechanics* are shared (Hadamard rotation, per-channel
INT8, `comfy_quant` stamps, FP16 kept tensors). Everything that decides *what to
protect* is per-model, because the models are not one family.

**Consequence for anyone adding a model.** There is no "add a checkpoint and
run the engine" path in this chapter. A new model means: identify the runtime
object and its loader, choose the observable to match what the loader injects,
write the measurement and verify harness for that observable, measure, choose the
threshold on the size/quality curve, and only then convert. That is a per-model
cost, and it is why this chapter exists as a separate document from the HSWQ
universal-engine guides rather than as a section of them.

---

## ComfyUI Node Workflow

Quantization is performed directly in-graph from memory without requiring external CLI conversion scripts:

### Workflow Layouts

<p align="left">
  <b>1. Text Encoder & ControlNet Quantization:</b><br>
  <img src="../png/te_controlnet_convrot_int8.png" alt="ComfyUI TE and ControlNet ConvRot INT8 Quantize Workflow" width="700">
</p>

<p align="left">
  <b>2. Model Patch (Model Patcher) Quantization:</b><br>
  <img src="../png/model_patcher.png" alt="ComfyUI Model Patch ConvRot INT8 Quantize Workflow" width="700">
</p>

<p align="left">
  <b>3. SAM 3 & SAM 3.1 Multiplex Quantization:</b><br>
  <img src="../png/sam.png" alt="ComfyUI SAM 3 and SAM 3.1 ConvRot INT8 Quantize Workflow" width="700">
</p>

---

### Sample Workflow

A ready-to-use ComfyUI workflow JSON covering all supported model types is provided in the repository:
- **[`sample workflow/native convrot int8.json`](../sample%20workflow/native%20convrot%20int8.json)**

You can load this file directly into ComfyUI (or drag-and-drop the workflow images above) to load the complete quantization graph.

---

### Pipeline Structure

The workflow consists of five specialized, parallel quantization branches:

1. **Text Encoder (CLIP / TE) Branch:**
   - **Loader:** `HSWQ Load CLIP (Simple)` (or standard `CLIPLoader` / `DualCLIPLoader` / `TripleCLIPLoader`).
   - **Quantizer:** `TE / ControlNet ConvRot INT8 Quantize` (connected via the `clip` input terminal).
   - **Monitor:** `Show Text` (Custom-Scripts) connected to `report` to display real-time quantization layer statistics and output path.

2. **ControlNet Branch:**
   - **Loader:** `Load ControlNet Model` (or `HSWQ Load ControlNet (Simple)`).
   - **Quantizer:** `TE / ControlNet ConvRot INT8 Quantize` (connected via the `control_net` input terminal).
   - **Monitor:** `Show Text` (Custom-Scripts) connected to `report` to display file size and layer breakdown.

3. **Model Patch (Model Patcher) Branch:**
   - **Loader:** `Load Model Patch` (`ModelPatchLoaderCustom` / ComfyUI `ModelPatchLoader`).
   - **Quantizer:** `TE / ControlNet ConvRot INT8 Quantize` (connected via the `model_patch` input terminal).
   - **Monitor:** `Show Text` (Custom-Scripts) connected to `report`.

4. **SAM 3.1 Branch (Multiplex Architecture):**
   - **Loader:** Standard **`UNetLoader`** (or `Load Diffusion Model`) loading `sam3.1_multiplex_fp16.safetensors` from `models/unet/` or `models/diffusion_models/`.
   - **Quantizer:** `TE / ControlNet ConvRot INT8 Quantize` (connected via the **`model`** input terminal).
   - **Monitor:** `Show Text` (Custom-Scripts) connected to `report`.

5. **SAM 3 Branch (Classic Non-Multiplex Architecture / sam3.pt):**
   - **Loader:** Dedicated **`HSWQ SAM3 Loader (sam3.pt)`** (`HSWQSAM3Loader`) loading `sam3.pt` or `sam3_fp16.safetensors`.
   - **Quantizer:** `TE / ControlNet ConvRot INT8 Quantize` (connected via the **`model`** input terminal).
   - **Monitor:** `Show Text` (Custom-Scripts) connected to `report`.

---

## SAM 3 vs. SAM 3.1 Architectural Differences

SAM 3 and SAM 3.1 have distinct internal network topologies and key layouts. The quantization engine incorporates explicit branching logic to ensure seamless compatibility:

| Architectural Feature | SAM 3 (`sam3.pt` / `sam3_fp16.safetensors`) | SAM 3.1 Multiplex (`sam3.1_multiplex_fp16.safetensors`) |
| :--- | :--- | :--- |
| **Model Type** | Classic Single-Object / Non-multiplex | 16-Object Multiplex Interactive Segmentation & Tracking |
| **FPN Neck Topology** | 4-Level FPN (`sam2_convs`, `vision_backbone.convs.3`) | 3-Level FPN (`propagation_convs`, `interactive_convs`) |
| **Text Projection Tensor** | $(1024 \times 512)$ Unused $\to$ **Automatically Dropped** | $(1024 \times 1024)$ Active $\to$ **Preserved & Quantized** |
| **Recommended Loader Node** | **`HSWQ SAM3 Loader (sam3.pt)`** (`HSWQSAM3Loader`) | **`UNetLoader`** / **`Load Diffusion Model`** |
| **Fused QKV In-Projection** | `.in_proj_weight` / `.in_proj_bias` split into `q_proj`, `k_proj`, `v_proj` | `.in_proj_weight` / `.in_proj_bias` split into `q_proj`, `k_proj`, `v_proj` |
| **Key Remapping** | `tracker.model.*` $\to$ `tracker.*`<br>`sam_mask_decoder.transformer.*` remapped | `tracker.model.*` $\to$ `tracker.*`<br>`sam_mask_decoder.transformer.*` remapped |
| **Quantization Input Slot** | `model` input on `TE / ControlNet ConvRot INT8 Quantize` | `model` input on `TE / ControlNet ConvRot INT8 Quantize` |
| **Quantized Output Size** | **~0.55 GB** (`sam3_convrot_int8.safetensors`) | **~0.84 - 0.89 GB** (`sam3.1_multiplex_convrot_int8.safetensors`) |

> [!IMPORTANT]
> **Key Reason for Separate Loaders:**
> - **SAM 3.1 (`sam3.1_multiplex_fp16.safetensors`)** is distributed in standard ComfyUI diffusion format and should be loaded via the ComfyUI **`UNetLoader`**.
> - **SAM 3 (`sam3.pt`)** contains raw checkpoint keys with an incompatible $(1024 \times 512)$ text projection and fused `in_proj` weights. The **`HSWQ SAM3 Loader (sam3.pt)`** node performs automatic key cleaning and structural reshaping so that the resulting `MODEL` is immediately quantizable and loadable.

---

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
pip install diffusers accelerate scikit-image
```

`scikit-image` is required for SSIM and benchmark comparisons. `comfy_kitchen` provides the quantization kernels and layout operations.

---

## ComfyUI Custom Node Setup

Copy or link the repository into `ComfyUI/custom_nodes/`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization.git
```

---

## Node Parameters & Configuration

### 1. `TE / ControlNet ConvRot INT8 Quantize`

| Input / Widget | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`clip`** | `CLIP` | Optional | Connected Text Encoder / CLIP model instance. |
| **`control_net`** | `CONTROL_NET` | Optional | Connected ControlNet or ControlNet Union model instance. |
| **`model_patch`** | `MODEL_PATCH` | Optional | Connected Model Patch (model patcher) instance, e.g. Z-Image Fun ControlNet loaded by `Load Model Patch` (`ModelPatchLoader`) from `models/model_patches/`. |
| **`model`** | `MODEL` | Optional | Connected SAM 3, SAM 3.1 Multiplex, or diffusion model instance loaded by `UNetLoader` or `HSWQ SAM3 Loader (sam3.pt)`. |
| **`group_size`** | `INT` | `256` | Preferred ConvRot Hadamard rotation group size (must be a power of 4: `4`, `16`, `64`, `256`, `1024`). |
| **`convrot`** | `BOOLEAN` | `True` | Enables orthogonal Hadamard rotation prior to INT8 scaling. When disabled, falls back to plain per-channel INT8. |
| **`output_path`** | `STRING` | `""` | Destination `.safetensors` path or directory. If left empty, saves automatically to `ComfyUI/output/` using `<model_name>_convrot_int8.safetensors`. |

> [!NOTE]
> At least one model input (`clip`, `control_net`, `model_patch`, or `model`) must be connected. You can also connect **multiple** inputs simultaneously to quantize matching models in a single queue execution.

### 2. `HSWQ SAM3 Loader (sam3.pt)`

| Input / Widget | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`sam3_name`** | `COMBO` | First available | Target `sam3.pt` or `sam3.safetensors` file scanned across `models/checkpoints/`, `models/unet/`, `models/diffusion_models/`, and `models/sams/`. |

---

## Standalone CLI Conversion Script (`convert_clip_convrot_int8.py`)

For batch processing or headless environments, the CLI script `clip_convert/convert_clip_convrot_int8.py` can convert generic `.safetensors` (CLIP, LLM, ControlNet, SAM 3, SAM 3.1, UNet) directly from the command line.

> [!NOTE]
> For **text encoders** and **SAM** this script is the **native / A/B baseline**: every 2D Linear layer is quantized and **no FP16 protection is applied**. For the production packs use the individually built artifacts in [**HSWQ_High_Quality_Packs_ConvRot_INT8.md**](HSWQ_High_Quality_Packs_ConvRot_INT8.md).

```bash
# Native baseline: SAM 3.1 Multiplex checkpoint (no FP16 protection)
python clip_convert/convert_clip_convrot_int8.py \
  --model models/unet/sam3.1_multiplex_fp16.safetensors \
  --output models/unet/sam3.1_multiplex_convrot_int8.safetensors \
  --groupsize 256

# Native baseline: generic CLIP / Text Encoder (no FP16 protection)
python clip_convert/convert_clip_convrot_int8.py \
  --model models/clip/t5xxl_fp16.safetensors \
  --output models/clip/t5xxl_convrot_int8.safetensors \
  --groupsize 256
```

### CLI Arguments

- `--model`: Path to input `.safetensors` checkpoint.
- `--output`: Path for output `.safetensors` checkpoint.
- `--no-convrot`: Disables Hadamard rotation and executes plain per-channel INT8 quantization.
- `--groupsize`: Hadamard rotation block size (must be a power of 4: `4`, `16`, `64`, `256`, `1024`, default `256`).

---

## Quantization Mechanics & Format

1. **State-Dict Extraction & Automatic Key Processing:**
   - The node extracts weights in memory and traces upstream loader paths.
   - For SAM architectures, automatic detection routes keys through version-specific remapping: fused `in_proj` tensors are split into `q_proj`, `k_proj`, `v_proj`, unused projection layers are pruned, and decoder transformer keys are standardized.

2. **Selective Linear Quantization:**
   - All 2D floating-point Linear weights (`.weight` with `ndim == 2`) are processed with Hadamard rotation:
     $$\mathbf{W}_{\text{rot}} = \mathbf{W} \mathbf{H}^T$$
   - Quantized to signed 8-bit integers with per-out-channel scaling:
     $$\text{scale}_i = \frac{\max_j(|W_{\text{rot},ij}|)}{127}, \quad W_{q,ij} = \text{round}\left(\frac{W_{\text{rot},ij}}{\text{scale}_i}\right)$$
   - Non-2D weights (embeddings, 1D layer norms, biases, Conv2d) are preserved unquantized in original precision (FP16/BF16/FP32) to prevent precision collapse at model boundaries.

3. **Output Checkpoint Layout:**
   ```
   <layer>.weight           int8
   <layer>.weight_scale     float32  [out_features, 1]
   <layer>.comfy_quant      uint8 JSON  {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":N}
   _quantization_metadata   JSON  {"format_version":"1.0","layers":{...}}
   ```

---

## Loading Quantized Models in ComfyUI

### SAM 3 / SAM 3.1 Multiplex Models
Quantized SAM 3 and SAM 3.1 checkpoints stamped with `int8_tensorwise` and `comfy_quant` are **100% natively supported** by ComfyUI:
- Place the generated `sam3_convrot_int8.safetensors` or `sam3.1_multiplex_hswq_convrot_int8.safetensors` in `ComfyUI/models/unet/` or `ComfyUI/models/diffusion_models/`.
- Load them using the standard **`UNetLoader`** (or **`Load Diffusion Model`**).
- Connect directly to downstream SAM segmentation / video tracking nodes. ComfyUI automatically utilizes native INT8 Tensor Core acceleration.
- For the recommended production pack, use the individually built **HSWQ High-Quality SAM 3.1** in [`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md).

### ControlNet Models
ComfyUI does not natively support rotated INT8 tensors for ControlNet models out of the box. To load and execute these weights in ComfyUI workflows, use the dedicated loader from the **[ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools)** extension:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools.git
```

Use the **`HSWQ Load ControlNet (Simple)`** node to load the generated `.safetensors` checkpoint directly into your generation graph.

### Model Patch (model patcher) Models
Some files that are named "ControlNet" are in fact **model patches**. They live
in `ComfyUI/models/model_patches/` and are applied by dedicated apply nodes, not
by `Apply ControlNet`. **ComfyUI's own `ModelPatchLoader` does not recognise
the INT8 layout** - it builds the patch with `manual_cast`, which reads the
pack's INT8 tensors as plain weights. Use the **`HSWQ Model Patch Loader`**
(`HSWQModelPatchLoaderCustom`, `nodes/hswq_model_patch_loader.py`): it detects
the `int8_tensorwise` `comfy_quant` layers and builds the patch with
`mixed_precision_ops`, so weights stay INT8 in VRAM and the forward uses the
comfy-kitchen `int8_linear` kernel. For non-quantized checkpoints it delegates to
the ComfyUI path.

Two architectures are dispatched this way:

| the discriminator key | the class | the apply node |
| :--- | :--- | :--- |
| `control_all_x_embedder.2-1.weight` | `comfy.ldm.lumina.controlnet.ZImage_Control` | `ZImageFunControlnet` |
| `controlnet_blocks.0.y_rms.weight` | `comfy_extras.nodes_model_patch.QwenImageBlockWiseControlNet` | `QwenImageDiffsynthControlnet` |

The returned `MODEL_PATCH` plugs into the matching apply node. For the Qwen-Image
diffsynth patches the apply path is `DiffSynthCnetPatch` (per DiT block:
`img[:, :n] += model.control_block(img[:, :n], cond, i) * strength`), which is
what Rule 8's per-block observable mirrors.

### Text Encoders (CLIP / TE)
Quantized Text Encoders stamped with `comfy_quant` can be loaded using standard ComfyUI `CLIPLoader` / `DualCLIPLoader` / `TripleCLIPLoader`. Llama-family encoders (Qwen3 / Qwen3-VL / Qwen3.5) self-detect the quantized layers; the OpenAI CLIP family needs the pack's `quantization_metadata` injected by the loader. See [`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md) for the recommended packs.

---

## How a pack is selected (the current method)

The rules that decide how a ConvRot INT8 pack is built - the injection frame, the
selection criterion, the gate, the threshold, the kept tensors, the completeness and
the path rules - are the **HSWQ** method, and they are stated in
[`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md),
Section *How a pack is selected (the current method)*.

The per-model guides carry their own measured numbers; those rules carry the method
those numbers were produced under.

### Per-model guides

Every per-model guide - text encoders, ControlNets and model patches, SAM - is indexed in
[`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md).
The guides carry the measured numbers; the rules above carry the method.

## Finished packs - ControlNet and model patches

Four IP-Adapter packs are part of this catalogue: `ip-adapter_instant_id_sdxl`, `ip-adapter_pulid_sdxl`,
`ip-adapter_sd15_plus` and `ip-adapter-faceid_sd15`.

Every ControlNet and model-patch pack built in this project is an **HSWQ build**: the FP16-kept
set comes from that checkpoint's own measurement, and the threshold was chosen on that
checkpoint's size/quality curve. The packs, their kept counts, their gates and the per-model
guides are in
[`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md).

**native** ConvRot INT8 - plain rotation and quantisation of every runtime Linear, with no
threshold sweep and no kept set - is a different product. No native pack is published in this
project; if one is built, it belongs in a separate section, never mixed with the measured builds.

**The directory column is not cosmetic.** A ControlNet in `model_patches/` (or
vice versa) does not load. The "model patch" rows go through the **HSWQ Model
Patch Loader**; the plain rows go through the ControlNet loader.

Published on Hugging Face:

- **[ussoewwin/ControlNet-models-ConvRot-INT8](https://huggingface.co/ussoewwin/ControlNet-models-ConvRot-INT8)** - the twenty-seven packs above.
- **[ussoewwin/Clip-files-HSWQ-ConvRot-INT8](https://huggingface.co/ussoewwin/Clip-files-HSWQ-ConvRot-INT8)** - the text-encoder packs (see the next section).
- **[ussoewwin/SAM3.1-HSWQ-ConvRot-INT8](https://huggingface.co/ussoewwin/SAM3.1-HSWQ-ConvRot-INT8)** - the SAM 3.1 pack.

### The one-line summary of the method

Stated in [`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md),
Section *How a pack is selected (the current method)*.

## HSWQ high-quality packs

The HSWQ builds - text encoders, ControlNet and model patches, and SAM 3.1, each with
its own per-checkpoint measurement - are documented in
[`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md).

This document keeps the method (*How a pack is selected*), the node and CLI workflows
(the native / A/B baseline they are measured against), and the load paths. The
finished, measured packs are not part of this document.

## Supported Architectures & Verified Checkpoints

| Architecture | Model Family | Verified Models |
| :--- | :--- | :--- |
| **SAM 3.1** | Multiplex Segmentation & Tracking | Individually built HSWQ pack - see [`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md) |
| **SAM 3** | Segmentation & Video Tracking | `sam3.pt`, `sam3_fp16.safetensors` (HSWQ pack not published - license restriction) |
| **Illustrious-XL / SDXL** | Anytest ControlNet | `CN-anytest4_illustrious2_A`, `CN-anytest4_illustrious2_B` |
| **SDXL 1.0** | ControlNet Union | `controlnet-union-pro-max-sdxl-1.0` (xinsir) |
| **SDXL** | InstantID ControlNet | `control_instant_id_sdxl`, `instantid_diffusion_pytorch_model` (InstantX) |
| **SD1.5** | ControlNet Tile / Canny / Depth / IP2P / Shuffle / Inpaint / Lineart / M-LSD / Normal / OpenPose / Scribble / Seg / SoftEdge / Lineart Anime / Recolor | `control_v11f1e_sd15_tile`, `control_v11p_sd15_canny`, `control_v11f1p_sd15_depth`, `control_v11e_sd15_ip2p`, `control_v11e_sd15_shuffle`, `control_v11p_sd15_inpaint`, `control_v11p_sd15_lineart`, `control_v11p_sd15_mlsd`, `control_v11p_sd15_normalbae`, `control_v11p_sd15_openpose`, `control_v11p_sd15_scribble`, `control_v11p_sd15_seg`, `control_v11p_sd15_softedge`, `control_v11p_sd15s2_lineart_anime`, `ioclab_sd15_recolor` (lllyasviel) |
| **Z-Image-Turbo** | ControlNet Union / Tile | `Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2601-8steps`, `Z-Image-Turbo-Fun-Controlnet-Tile-2.1-lite-2601-8steps` |
| **Qwen-Image** | ControlNet Inpainting | `Qwen-Image-ControlNet-Inpainting` (alibaba-pai) |
| **Qwen-Image-2512** | ControlNet Union | `Qwen-Image-2512-Fun-Controlnet-Union-2602` |
| **FLUX.1-dev** | ControlNet Union Pro | `FLUX.1-dev-ControlNet-Union-Pro-2.0` (Shakker Labs) |
| **CLIP / Text Encoders** | CLIP-L, CLIP-G, T5-XXL, Qwen2.5-VL, Qwen3 / Qwen3-VL / Qwen3.5 | Individually built HSWQ packs - see [`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md) |

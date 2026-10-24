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
> **Text encoders and SAM are separate cases.** The generic CLI above quantizes every 2D Linear layer with **no FP16 protection** - it is the A/B baseline, not the recommended pack. HSWQ also publishes **individually built, higher-quality packs** (measured FP16 protection) for text encoders and SAM 3.1 on Hugging Face. Because text encoders and SAM are standards (one weight layout per architecture) rather than per-checkpoint models, these are finished artifacts rather than a universal method - see [**HSWQ High-Quality Text Encoders**](#hswq-high-quality-text-encoders-convrot-int8-individually-built) and [**HSWQ High-Quality SAM 3.1**](#hswq-high-quality-sam-31-convrot-int8-individually-built) below. (SAM 3 is the same kind of case but its pack is not published for license reasons.)

> [!IMPORTANT]
> **The generic CLI above is the no-protection baseline.** It quantizes every 2D Linear layer with **no FP16 protection**. The recommended packs in this chapter are individually built: the protection set comes from a measured aggregate contribution, the threshold from the size/quality balance, and the kept tensors are stored FP16. See [**HSWQ High-Quality Text Encoders**](#hswq-high-quality-text-encoders-convrot-int8-individually-built), the ControlNet / model-patch inventory, and [**HSWQ High-Quality SAM 3.1**](#hswq-high-quality-sam-31-convrot-int8-individually-built) below.

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
   - **Loader:** `Load Model Patch` (`ModelPatchLoaderCustom` / stock `ModelPatchLoader`).
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
> - **SAM 3.1 (`sam3.1_multiplex_fp16.safetensors`)** is distributed in standard ComfyUI diffusion format and should be loaded via the stock **`UNetLoader`**.
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
> For **text encoders** and **SAM** this script is the **native / A/B baseline**: every 2D Linear layer is quantized and **no FP16 protection is applied**. For production packs use the individually built artifacts in [**HSWQ High-Quality Text Encoders**](#hswq-high-quality-text-encoders-convrot-int8-individually-built) and [**HSWQ High-Quality SAM 3.1**](#hswq-high-quality-sam-31-convrot-int8-individually-built). For ControlNet / Model Patch the generic script is the intended tool.

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
- For the recommended production pack, use the individually built [**HSWQ High-Quality SAM 3.1**](#hswq-high-quality-sam-31-convrot-int8-individually-built) below.

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
by `Apply ControlNet`. **ComfyUI's stock `ModelPatchLoader` does not recognise
the INT8 layout** - it builds the patch with `manual_cast`, which reads the
pack's INT8 tensors as plain weights. Use the **`HSWQ Model Patch Loader`**
(`HSWQModelPatchLoaderCustom`, `nodes/hswq_model_patch_loader.py`): it detects
the `int8_tensorwise` `comfy_quant` layers and builds the patch with
`mixed_precision_ops`, so weights stay INT8 in VRAM and the forward uses the
comfy-kitchen `int8_linear` kernel. For non-quantized checkpoints it delegates to
the stock path.

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
Quantized Text Encoders stamped with `comfy_quant` can be loaded using standard ComfyUI `CLIPLoader` / `DualCLIPLoader` / `TripleCLIPLoader`. Llama-family encoders (Qwen3 / Qwen3-VL) self-detect the quantized layers; the OpenAI CLIP family needs the pack's `quantization_metadata` injected by the loader. See [**HSWQ High-Quality Text Encoders**](#hswq-high-quality-text-encoders-convrot-int8-individually-built) for the recommended packs.

---

## How a pack is selected (the current method)

This section is the authoritative statement of how a ConvRot INT8 pack is
built now. The per-model guides carry their own measured numbers; this section
carries the rules those numbers were produced under.

### Rule 1 — the injection frame

The pack stores the **rotated** weight, and the runtime kernel rotates the
**activation**. The quantization error that reaches the output is therefore

```
(q * s) H^T - W        (the ORIGINAL frame, not the rotated one)
```

A measurement that injects the reconstruction in the **rotated** frame
(`q * s`) measures the rotation, not the error. On FLUX ControlNet the two
differ by four orders of magnitude (1.74e-04 correct against 1.89 broken) and
**the layer ranking inverts**. Every impact script under `sdxl/` was fixed for
this; the packs built before the fix used the wrong ranking and were rebuilt.

### Rule 2 — the screening metric is not the selection criterion

`diag_impact_*` measures the **per-layer weight error**: it puts the
reconstruction into the module and runs the plain forward. The pack's real
path also **quantizes the activation** for that layer, and that is where most
of the error comes from. So the drift is a **screen** (it finds the layers
worth looking at), not the decision.

The decision is the **aggregate contribution**: hold the FP16 weights, replace
one layer by its ConvRot INT8 reconstruction, and measure the model's own
**output** against the unmodified output. Sort on `1 - cos`. The gap is not
academic - on FLUX ControlNet the same 54 INT8 layers reached min 0.999160 by
drift selection and 0.999698 by contribution selection; on SAM 3.1 the
drift-selected set was 286x worse in relMSE.

The instrument differs with the family, and the rule is that **the criterion is
whatever the runtime actually consumes**:

| the family | the aggregate-output observable |
| :--- | :--- |
| an SDXL ControlNet | the **trajectory drift of the denoising latent** - already an aggregate-output measurement, so no separate contribution pass is needed |
| a Qwen-Image ControlNet (`load_controlnet_qwen_fun` / `_instantx`) | the **control model's own output** over 8 seeds - `load_controlnet` returns per-block residuals, and the model needs a base only for its entry path |
| a Qwen-Image diffsynth model patch | the **patch's own output** over 8 seeds, scored **per block** (see Rule 8) |
| a Z-Image Fun ControlNet patch | the **denoising trajectory** of the Z-Image DiT with the patch applied (the real pipeline) |
| a text encoder / CLIP / standalone tower | the **tower's own output** for fixed inputs |

### Rule 3 — the gate must use the same instrument as the selection

If the protection set is chosen on the trajectory, the gate is measured on the
**trajectory**. A single forward of the control model is a different quantity
(one control output against 20 compounded sampling steps: 0.9998 against
0.993 on the same pack), and its run-to-run noise (~1e-5) is the same size as
the differences between neighbouring thresholds - so it cannot place the knee.
`sdxl/verify_controlnet_sdxl_traj.py` runs the trajectory over several seeds
for this reason.

Conversely, when the criterion is the model **patch's** own output, the gate is
that same output, measured through the **loader the runtime uses** - not a
hand-built module graph. For the Qwen-Image diffsynth patches that means
driving `HSWQModelPatchLoaderCustom.load_model_patch`, and it matters: the stock
`ModelPatchLoader` would read the INT8 tensors as plain weights and the pack
would look like FP16. Two harnesses exist for this and both must agree
(`sdxl/verify_qwen_image_diffsynth_patch.py` builds the graph directly;
`..._loader.py` goes through the node) - measured agreement is exact to 10
decimals.

### Rule 4 — the threshold is chosen on the size/quality balance

Not the highest cosine. Build the pack across a range of thresholds, measure
the gate at each, and take the point where the curve stops paying for itself.

- **The floor.** Past some threshold every ConvRot-eligible layer is already
  quantized and the pack stops shrinking. Beyond the floor the cosine
  **drops**, because the remaining protected layers are non-Linear or
  shape-blocked: protecting them adds size without buying quality. A threshold
  at or above the floor is never right.
- **The plateau.** Just below the floor the cosine is at its best among the
  small packs. That point is the choice.

The rule: adopt the threshold at the top of the quality plateau - the smallest
pack whose cosine is not beaten by the smaller ones. Report the whole measured
curve, not one number.

Worked cases (all measured):

| the checkpoint | the INT8 extreme | the adopted point | the size |
| :--- | :--- | :--- | :--- |
| `controlnet-union-pro-max-sdxl-1.0` | curve flat at 0.999961 | 0.005 (41 kept) | 1368.8 MB |
| `CN-anytest4_illustrious2_A` | 0.010: 1959.5 MB / 0.999879 | 0.030 (36 kept, 0.999626) | 1359.1 MB |
| `CN-anytest4_illustrious2_B` | 0.003: 1981.0 MB / 0.988027 | 0.010 (35 kept, 0.992958) | 1357.5 MB |
| `CLIP-SAE-ViT-L-14` | 50/100 kept: 722.3 MB / 0.999985 | 30 text + 60 vision kept | 658.9 MB |
| `ViT-L-14-BEST` | 50/100 kept: 722.1 MB / 0.999986 | 30 text + 60 vision kept | 646.7 MB |
| `Josiefied-Qwen3-0.6B` | 140 kept: 0.85 GB / 0.999644 | 40 kept (0.999502) | 655.2 MB |
| `Qwen3-4B` | 150 kept: 5909.9 MB / 0.999698 | 20 kept (0.999630) | 4168.3 MB |
| `sigclip_vision_patch14_384` | 100 kept: 731.7 MB / 0.999715 | 20 kept (0.996794) | 502.9 MB |
| `Qwen2.5-VL-7B` | 150 kept: 12650.0 MB / 0.999496 | 40 kept (0.999325) | 9289.6 MB |
| `Qwen-Image-2512-Fun-Controlnet-Union-2602` | curve rises; all 77 layers eligible | 0 kept (floor, 0.999158) | 1677.2 MB |
| `Qwen-Image-ControlNet-Inpainting` | curve rises; all 95 layers eligible | 0 kept (floor, 0.999545) | 2022.1 MB |
| `qwen_image_canny_diffsynth` (model patch) | one step: img_in is 598x the runner-up | 1 kept (`img_in`, 0.999992) | 1083.3 MB |
| `qwen_image_inpaint_diffsynth` (model patch) | one step: img_in is 147x the runner-up | 1 kept (`img_in`, 0.999992) | 1083.3 MB |
| `qwen_image_depth_diffsynth` (model patch) | one step: img_in is 376x the runner-up | 1 kept (`img_in`, 0.999992) | 1083.3 MB |
| `Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602` (model patch) | all 38 layers eligible, trajectory 0.9999 | 0 kept (floor, trajectory min 0.999491) | 963.0 MB |
| `Z-Image-Turbo-Fun-Controlnet-Tile-2.1-lite-2601` (model patch) | all 38 layers eligible, trajectory 0.9999 | 0 kept (floor, trajectory min 0.999813) | 963.0 MB |

The same rule produces different thresholds because the checkpoints differ -
which is the whole point of measuring each one.

### Rule 5 — the kept tensors are stored FP16

The runtime runs in FP16 (CLIP, T5, ControlNet alike). Keeping a tensor in
FP32 means it is cast to FP16 at load time and nothing ever sees the extra
precision - it only pays for the bytes. Every rebuilt pack stores its kept
tensors in FP16; the INT8 `weight_scale` stays FP32 because it carries the
dequantization precision.

Measured effect on CLIP-SAE-ViT-L-14: the identical layer set went from
1159.5 MB (FP32 kept) to **658.9 MB** (FP16 kept), with the acceptance
unchanged to the last digit.

### Rule 6 — the content is complete

A pack must not shrink by deleting model content. The FLAN-T5 decoder is
shipped and quantized even though ComfyUI's T5 never loads it - that is a
load-path fact, not a licence to drop it.

### Rule 7 — no path that only works on one machine

Public code carries no machine-specific absolute paths. Resolve from
`Path(__file__).resolve().parent` and stop with an explicit error rather than
falling back to an environment guess.

### Rule 8 - the observable is what the runtime injects

A family whose output is a **list of control residuals** must be scored as a
list, not as a concatenated vector, and the arithmetic must be float64. Both are
forced by the runtime, and both were measured.

- **Score per block, then average.** The apply node injects each block's residual
  at that block, so the 60 residuals are 60 independent contributions. Pooling
  them into one concatenation lets the largest block's norm decide the whole
  number and hides the rest.
- **float64 is required.** The residuals run to ~3e5 in magnitude, so a float32
  dot product over 47,185,920 elements accumulates ~1e-3 of rounding noise -
  larger than the entire signal (the most important layer measures ~6e-06, the
  smallest ~2.2e-09). The first Qwen-Image diffsynth run in float32 returned
  every value **negative**.
- **`cond` must be recomputed inside the perturbation loop.** `cond` is produced
  by `process_input_latent_image`, which runs through `img_in` - itself one of
  the layers under test. Precomputing it once understates `img_in`'s
  contribution by an order of magnitude.
- **Derive shape parameters from the checkpoint, not from a sibling.** The
  Qwen-Image diffsynth family splits on the control latent's channel count:
  canny and depth are 16 channels (`img_in` width 64, `additional_in_dim = 0`),
  inpaint is 17 (`img_in` width 68, `additional_in_dim = 4`). Read it as
  `img_in.in_features // 4`; a hardcoded 16 raises
  `mat1 and mat2 shapes cannot be multiplied (1024x64 and 68x3072)`.

The measured result of this rule is the **single-step** curve: on all three
Qwen-Image diffsynth patches one layer (`img_in`) carries 80-86 % of the summed
contribution and is 147x to 598x the runner-up, so protecting it alone converts
a 3.5e-05 .. 5.4e-05 error into 5e-08 for under 0.2 MB.

### Cross-references

Every per-model guide for this chapter, one link per artifact. The guides carry the measured numbers; the rules above carry the method.

**Text encoders**

- [CLIP-L](CLIPL_ConvRot_INT8_Impact_Technical_Guide.md)
- [CLIP-L verification log](CLIPL_ConvRot_INT8_Verification_Log.md)
- [CLIP ViT-bigG/14](CLIPBiGG_ConvRot_INT8_Impact_Technical_Guide.md)
- [CLIP ViT-H/14](CLIPViTH14_ConvRot_INT8_Impact_Technical_Guide.md)
- [OpenAI CLIP ViT-L/14 (SAE)](SAEViTL14_ConvRot_INT8_Impact_Technical_Guide.md)
- [Long-ViT-L/14](LongViTL14_ConvRot_INT8_Impact_Technical_Guide.md)
- [ViT-L/14 (HF layout)](ViTL14_ConvRot_INT8_Impact_Technical_Guide.md)
- [ViT-L/14 runbook](ViTL14_ConvRot_INT8_Runbook.md)
- [ViT-L/14 verification report](ViTL14_ConvRot_INT8_Verification_Report.md)
- [SigLIP vision 384](SigLIPVision384_ConvRot_INT8_Impact_Technical_Guide.md)
- [Flan-T5-XXL](T5XXL_ConvRot_INT8_Impact_Technical_Guide.md)
- [Josiefied-Qwen3-0.6B](JosiefiedQwen3_06B_ConvRot_INT8_Impact_Technical_Guide.md)
- [Qwen3-4B](Qwen3_4B_ConvRot_INT8_Impact_Technical_Guide.md)
- [Qwen3-VL-4B](Qwen3VL4B_ConvRot_INT8_Impact_Technical_Guide.md)
- [Qwen2.5-VL-7B](Qwen25VL7B_ConvRot_INT8_Impact_Technical_Guide.md)
- [Qwen3.5-9B Defiant-Fable (NEO IMATRIX MAX MTP)](Qwen35_9B_ConvRot_INT8_Impact_Technical_Guide.md)
- [the text-encoder index (all thresholds in one place)](ConvRot_INT8_TextEncoder_Index.md)

**ControlNets and model patches**

- [the per-checkpoint ControlNet numbers (SDXL / Illustrious)](ControlNet_ConvRot_INT8_Trajectory_Impact_Technical_Guide.md)
- [FLUX.1-dev ControlNet Union Pro 2.0](FLUXControlNetUnionPro2_ConvRot_INT8_Impact_Technical_Guide.md)
- [Qwen-Image-2512 Fun ControlNet Union 2602](QwenImage2512FunControlnetUnion2602_ConvRot_INT8_Impact_Technical_Guide.md)
- [Qwen-Image ControlNet Inpainting](QwenImageControlNetInpainting_ConvRot_INT8_Impact_Technical_Guide.md)
- [Qwen-Image diffsynth model patch (canny)](QwenImageDiffsynthControlnet_ConvRot_INT8_ModelPatch_Guide.md)
- [Qwen-Image diffsynth model patch (depth)](QwenImageDepthDiffsynthControlnet_ConvRot_INT8_Guide.md)
- [Qwen-Image diffsynth model patch (inpaint)](QwenImageInpaintDiffsynthControlnet_ConvRot_INT8_Guide.md)
- [Z-Image Fun ControlNet model patch (Union / Tile)](ZImageTurboFunControlnet_ConvRot_INT8_ModelPatch_Guide.md)

**SAM**

- [SAM 3.1 Multiplex](SAM31_ConvRot_INT8_Trajectory_Impact_Technical_Guide.md)

---
## Finished packs - ControlNet and model patches

Every pack below is a real file with the measured gate in it. Sizes are **MiB**
(1 MB = 2\^20 bytes), the same unit as the text-encoder table; the SHA256 column
carries the pack's first 16 hex characters, full hash in the per-model guide.

| Pack | Family | Runtime object | Directory | the gate (measured) | Size | SHA256 (first 16) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `CN-anytest4_illustrious2_A_hswq_convrot_int8` | Illustrious-XL / SDXL | ControlNet | `models/controlnet/` | 0.999626 (forward) | 1359.1 MB | `BE9FF80B2F489568` |
| `CN-anytest4_illustrious2_B_hswq_convrot_int8` | Illustrious-XL / SDXL | ControlNet | `models/controlnet/` | 0.992958 / 0.948279 (trajectory) | 1357.5 MB | `7E34DC3CC3334AFA` |
| `controlnet-union-pro-max-sdxl-1.0_hswq_convrot_int8` | SDXL 1.0 | ControlNet | `models/controlnet/` | 0.999961 | 1368.8 MB | `DCDF6F2037205C4A` |
| `FLUX.1-dev-ControlNet-Union-Pro-2.0_hswq_convrot_int8` | FLUX.1-dev | ControlNet | `models/controlnet/` | 0.999771 / 0.999488 | 2585.1 MB | `FBCBA3C76A221FBA` |
| `Qwen-Image-2512-Fun-Controlnet-Union-2602_hswq_convrot_int8` | Qwen-Image-2512 | ControlNet | `models/controlnet/` | 0.999158 / 0.998215 | 1677.2 MB | `5AEC39A53F583BAD` |
| `Qwen-Image-ControlNet-Inpainting_hswq_convrot_int8` | Qwen-Image | ControlNet | `models/controlnet/` | 0.999545 / 0.998887 | 2022.1 MB | `38F0AB7495029517` |
| `qwen_image_canny_diffsynth_controlnet_hswq_convrot_int8` | Qwen-Image | **model patch** | `models/model_patches/` | 0.9999924905 / 0.9999924096 | 1083.3 MB | `C2899255252B81FD` |
| `qwen_image_depth_diffsynth_controlnet_hswq_convrot_int8` | Qwen-Image | **model patch** | `models/model_patches/` | 0.9999924109 / 0.9999923262 | 1083.3 MB | `7B82A9360169FA99` |
| `qwen_image_inpaint_diffsynth_controlnet_hswq_convrot_int8` | Qwen-Image | **model patch** | `models/model_patches/` | 0.9999921236 / 0.9999919542 | 1083.3 MB | `471B6AFFC7FCEDAF` |
| `Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps_hswq_convrot_int8` | Z-Image-Turbo | **model patch** | `models/model_patches/` | 0.999862 / 0.999491 (trajectory) | 963.0 MB | `1DD12B2D028B12DF` |
| `Z-Image-Turbo-Fun-Controlnet-Tile-2.1-lite-2601-8steps_hswq_convrot_int8` | Z-Image-Turbo | **model patch** | `models/model_patches/` | 0.999956 / 0.999813 (trajectory) | 963.0 MB | `2998F83EF65903B4` |

**The directory column is not cosmetic.** A ControlNet in `model_patches/` (or
vice versa) does not load. The "model patch" rows go through the **HSWQ Model
Patch Loader**; the plain rows go through the ControlNet loader.

Published on Hugging Face:

- **[ussoewwin/ControlNet-models-ConvRot-INT8](https://huggingface.co/ussoewwin/ControlNet-models-ConvRot-INT8)** - the eleven packs above.
- **[ussoewwin/Clip-files-HSWQ-ConvRot-INT8](https://huggingface.co/ussoewwin/Clip-files-HSWQ-ConvRot-INT8)** - the text-encoder packs (see the next section).
- **[ussoewwin/SAM3.1-HSWQ-ConvRot-INT8](https://huggingface.co/ussoewwin/SAM3.1-HSWQ-ConvRot-INT8)** - the SAM 3.1 pack.

### The one-line summary of the method

For each family: build a fixed input set, measure the **aggregate contribution**
of every ConvRot-eligible layer (`1 - cos` of the runtime's observable, in the
original injection frame, with the runtime's own arithmetic), keep the layers
that the **size/quality balance** pays for in FP16, store every kept tensor FP16,
and verify the **complete pack** through the actual loader on a fixed 256-item
set (seed 42) whose gate is `cos.mean >= 0.99` and `cos.min >= 0.98`.

The three outcomes this produces, all measured:

| the shape of the curve | what it means | the examples |
| :--- | :--- | :--- |
| a **flat** curve | no layer stands out; take the smallest satisfying pack | `controlnet-union-pro-max-sdxl-1.0` |
| a **single step** | one entry layer carries most of the contribution; protect it and stop | the three Qwen-Image diffsynth patches |
| a **floor** | every eligible layer is safe at INT8; protect nothing | `Qwen-Image-2512-Fun-Controlnet-Union-2602`, `Qwen-Image-ControlNet-Inpainting`, the two Z-Image patches |

---

## HSWQ High-Quality Text Encoders (ConvRot INT8, individually built)

### Why there is no universal method here

HSWQ's reverse-hybrid engine is a **universal method**: it measures per-layer trajectory sensitivity on the target checkpoint, ranks the layers, and selects FP16 protection inside a fixed budget. That method applies to image-generation backbones (SDXL / Z Image / Krea2 UNet-DiT), which exist as **many per-checkpoint variants** - SDXL alone was tested on **30+ checkpoints**, and every merge or finetune is a new target whose ranking must be re-measured. A general method pays off precisely because there are so many of them.

Text encoders are **not** per-checkpoint models. Each architecture (CLIP-L, CLIP-G, T5-XXL, Qwen2.5-VL, Qwen3, ...) is a **single standard weight layout** with no per-checkpoint derivation. There is nothing to re-measure per model, so a large universal engine is not justified - the same applies to a ControlNet standard and to a SAM release, and it is why the whole chapter above is a per-model track rather than an application of the engine.

Instead, each text-encoder standard was handled **individually** - but the criterion is no longer static. See **Section: How a pack is selected (the current method)** below: the protection set comes from the measured **aggregate contribution**, the size/quality balance sets the threshold, and the kept tensors are stored FP16. Static kurtosis / outlier-ratio screening is retired as a *criterion*; the per-standard converter scripts under `clip_convert/` implement the current method.

### Finished packs (Hugging Face)

| Pack | Standard | ConvRot INT8 | FP16 kept | Gate (256 items, seed 42) | Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `clip_l_hswq_convrot_int8.safetensors` | CLIP-L | 63 of 73 | 10 | 0.999876 / 0.999841 | 135.5 MB |
| `CLIP-SAE-ViT-L-14-hswq_convrot_int8.safetensors` | OpenAI CLIP ViT-L/14 | 129 (text 43, vision 84) | 90 (text 30, vision 60) | text 0.999962 / 0.999908, vision 0.999931 / 0.999797 | 658.9 MB |
| `Long-ViT-L-14-BEST-GmP-smooth-ft-full_hswq_convrot_int8.safetensors` | CLIP ViT-L/14 (extended text) | 130 (text 44, vision 84) | 90 (text 30, vision 60) | text 0.999941 / 0.999888, vision 0.999925 / 0.999790 | 646.3 MB |
| `ViT-L-14-BEST-smooth-GmP-HF-format_hswq_convrot_int8.safetensors` | CLIP ViT-L/14 (HF layout) | 127 (text 43, vision 84) | 92 (text 30, vision 60) | text 0.999964 / 0.999906, vision 0.999803 / 0.996378 | 646.7 MB |
| `CLIP-ViT-bigG-14-laion2B-39B-b160k_convrot_int8.safetensors` | CLIP ViT-bigG/14 | 253 of 288 | 35 | 0.999916 / 0.999741 | 2180.1 MB |
| `CLIP-ViT-H-14_hswq_convrot_int8.safetensors` | CLIP ViT-H/14 | 92 of 192 | 100 | 0.999968 / 0.999906 | 1015.6 MB |
| `flan_t5_xxl_hswq_convrot_int8.safetensors` | Flan-T5-XXL | 312 of 432 (enc 128, dec 184) | 120 (enc 40, dec 80) | 0.999934 | 14056.6 MB |
| `Josiefied-Qwen3-0.6B-abliterated-v1_hswq_convrot_int8.safetensors` | Qwen3 (0.6B) | 157 of 197 | 40 | 0.999502 / 0.998980 | 655.2 MB |
| `Qwen3_VL_4B_Thinking_abliterated_hswq_convrot_int8.safetensors` | Qwen3-VL (4B) | 258 of 358 | 100 | 0.999649 / 0.998927 | 5958.9 MB |
| `Qwen3.5-9B-...-MAX-MTP_hswq_convrot_int8.safetensors` | Qwen3.5 (9B, hybrid + vision + MTP) | 369 of 369 | 0 | conditioning 0.999387 / 0.996488, generate 0.999453 / 0.998215 | 9217.9 MB |
| `sam3.1_multiplex_hswq_convrot_int8.safetensors` | SAM 3.1 Multiplex | 41 of 221 | 180 | relMSE 3.51e-05 | 1657.6 MB |

Download:

- **[ussoewwin/Clip-files-ConvRot-INT8](https://huggingface.co/ussoewwin/Clip-files-ConvRot-INT8)** - CLIP-L, OpenAI CLIP ViT-L/14 (CLIP-SAE), Flan-T5-XXL, CLIP-ViT-bigG-14, Josiefied-Qwen3-0.6B
- **[ussoewwin/qwen3_4b_8b_abliterated_fp16_convrot_int8](https://huggingface.co/ussoewwin/qwen3_4b_8b_abliterated_fp16_convrot_int8)** - Qwen3-4B, Qwen3-8B, Qwen3-VL-4B-Thinking

### Loading

Place the packs in `ComfyUI/models/clip/` and load with the standard ComfyUI `CLIPLoader` / `DualCLIPLoader` / `TripleCLIPLoader`. For Llama-family text encoders (Qwen3 / Qwen3-VL) ComfyUI detects the `comfy_quant` layers inside the load path, so no extra loader is required. For the OpenAI CLIP family (CLIP-L / CLIP-SAE) the loader must receive the pack's `quantization_metadata` (the same mechanism the SDXL CLIP INT8 path already uses); see the **ComfyUI-HSWQ-Loader-and-Tools** extension.

---

## HSWQ High-Quality SAM 3.1 (ConvRot INT8, individually built)

### Why SAM 3.1 belongs here

Like text encoders, SAM 3 / SAM 3.1 are **foundation segmentation models**, not per-checkpoint image-generation backbones. Each version ships a **single standard weight layout**, so there is no per-checkpoint measurement to redo and no universal engine to build. They are therefore handled **individually**: the risk of every Linear layer was measured once on the standard checkpoint (kurtosis / outlier ratio), the fat-tail layers were kept in FP16, and the pack was built with a per-standard converter.

> [!IMPORTANT]
> **SAM 3 (classic / `sam3.pt`) is technically the same kind of case and should be built and published the same way, but it cannot be published for license reasons. Only SAM 3.1 Multiplex is published.**

### Finished pack (Hugging Face)

| Pack | Standard | ConvRot INT8 | FP16 kept | Gate | Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `sam3.1_multiplex_hswq_convrot_int8.safetensors` | SAM 3.1 Multiplex | 41 of 221 | 180 | relMSE 3.51e-05 (the tracker and detector paths) | 1657.6 MB |

> The 41/180 split is the shipped pack, re-audited after the injection fix: a
> release search over all 401 candidates found **0** layers releasable inside
> the gate (the cheapest release costs +9.67e-04 relMSE against a 1.0e-04
> gate), so the pack cannot be made smaller. The earlier "538 INT8 / 113 kept"
> figures came from the defective drift and are superseded.

Download:

- **[ussoewwin/SAM3.1-HSWQ-ConvRot-INT8](https://huggingface.co/ussoewwin/SAM3.1-HSWQ-ConvRot-INT8)**

### Loading

Place the pack in `ComfyUI/models/unet/` (or `models/diffusion_models/`) and load it with the standard **`UNetLoader`** / **`Load Diffusion Model`**; connect to the SAM segmentation / video-tracking nodes. The pack keeps the fused QKV in-projection split into `q_proj` / `k_proj` / `v_proj` (the loader expects the split layout) and drops the per-block `freqs_cis` buffers.

---

## Supported Architectures & Verified Checkpoints

| Architecture | Model Family | Verified Models |
| :--- | :--- | :--- |
| **SAM 3.1** | Multiplex Segmentation & Tracking | Individually built HSWQ pack - see [**HSWQ High-Quality SAM 3.1**](#hswq-high-quality-sam-31-convrot-int8-individually-built) |
| **SAM 3** | Segmentation & Video Tracking | `sam3.pt`, `sam3_fp16.safetensors` (HSWQ pack not published - license restriction) |
| **Illustrious-XL / SDXL** | Anytest ControlNet | `CN-anytest4_illustrious2_A`, `CN-anytest4_illustrious2_B` |
| **SDXL 1.0** | ControlNet Union | `controlnet-union-pro-max-sdxl-1.0` (xinsir) |
| **Z-Image-Turbo** | ControlNet Union / Tile | `Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2601-8steps`, `Z-Image-Turbo-Fun-Controlnet-Tile-2.1-lite-2601-8steps` |
| **Qwen-Image** | ControlNet Inpainting | `Qwen-Image-ControlNet-Inpainting` (alibaba-pai) |
| **Qwen-Image-2512** | ControlNet Union | `Qwen-Image-2512-Fun-Controlnet-Union-2602` |
| **FLUX.1-dev** | ControlNet Union Pro | `FLUX.1-dev-ControlNet-Union-Pro-2.0` (Shakker Labs) |
| **CLIP / Text Encoders** | CLIP-L, CLIP-G, T5-XXL, Qwen2.5-VL, Qwen3 / Qwen3-VL | Individually built HSWQ packs - see [**HSWQ High-Quality Text Encoders**](#hswq-high-quality-text-encoders-convrot-int8-individually-built) |

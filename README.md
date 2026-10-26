# Hybrid-Sensitivity-Weighted-Quantization (HSWQ)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/main/icon.png" width="128">
</p>

High-fidelity **ConvRot INT8** (SDXL, Krea2, Qwen Image Edit, SD1.5) and **ConvRot NVFP4** (Z Image Turbo) quantization for diffusion models. HSWQ uses **sensitivity** and **importance** analysis instead of naive uniform cast. SD1.5 ships as a **trajectory-impact hybrid**: a measured per-layer trajectory impact ranks every ConvRot-eligible layer, the top-K stay FP16, and the rest go ConvRot INT8 (UNet only - CLIP untouched, VAE dropped).

- **ConvRot INT8 (SDXL, reverse hybrid):** FP16 checkpoint as the only input; a per-layer **trajectory-impact** measurement (`sdxl/diag_impact_sdxl.py`) ranks every candidate layer, then the **K lowest-impact layers** are packed as **FULL ConvRot INT8** while **every other layer stays FP16** (`sdxl/gen_reverse_int8_sdxl.py`). Validated by the **deterministic 25-seed latent-trajectory comparison** (per-step cosine + bifurcation); production gate = **cosine mean ≥ 0.95 and 0/25 bifurcated**. The ranking is measured on the actual sampling trajectory, so it captures the nonlinear error amplification that static weight-space saliency cannot represent. The quantized file does not embed a VAE (`first_stage_model.*` is removed at conversion): load it with a separate SDXL VAE.
- **ConvRot INT8 (SD1.5, trajectory-impact hybrid):** A measured per-layer **trajectory impact** (`sd1.5/diagimpact_sd15.py`) ranks every ConvRot-eligible layer: hold the rest pristine, replace ONE layer by its ConvRot INT8 reconstruction in the original frame, run the same 25-step trajectory, record `rel_mse` against the pristine run. The **top-K** stay FP16 and the rest go ConvRot INT8 (`sd1.5/quantize_sd1.5_hswq_v1.1.py`, with optional INT8-only bias correction). **CLIP is not quantized** and the **VAE is dropped**. Validated with the deterministic 25-seed latent-trajectory gate (512x512, 25 steps): **mean final-cosine 0.97622** against **native 0.96265** - a +0.0136 gain - with **0/25 bifurcated** in every candidate. **K is not a method constant**: it is chosen per checkpoint by sweeping the protected count and taking the point where the mean stops paying for the extra bytes. Both ends of that curve fail - a smaller set scores the same as native (the protection buys nothing) and a larger one scores *lower* while costing more, because a protected layer keeps its FP16 weights but loses the INT8 bias correction that cancels its systematic error. The sweep for this checkpoint is in the benchmark report below. Sizes (measured): native all-INT8 1058.6 MB; the adopted pack 1167.5 MB. Load with a separate SD1.5 VAE. The retired **HSWQ V1.0** selector (`archives/quantize_sd1.5_hswq_v1.0.py`) ranked protection from static weight-space statistics and its protection set scored the same as native - the defect was the criterion, not the quantizer.

- **Z Image INT8 (HSWQ):** **Development and public release ended.** For Z Image, **native ConvRot INT8** already reaches roughly **SSIM > 0.99** in general, so a separate HSWQ Z Image 8-bit line is no longer developed or published. Use native ConvRot INT8 for Z Image 8-bit; HSWQ INT8 work continues for **SDXL**.
- **ConvRot INT8 (Krea2):** Full HSWQ pipeline with structure blacklist protection (`first.`, `last.`, `mod.`, `norm`, `projector`, `txtfusion`, etc.) and data-driven 4-axis composite ranking (`Krea2/hswq_convrot_int8_krea2_v1.5.py`). For checkpoints where native ConvRot INT8 already reaches mean latent trajectory cosine $\ge 0.98$, native ConvRot INT8 is directly recommended.
- **Krea2 Hybrid ConvRot NVFP4:** **Development cancelled.** 4-bit precision (NVFP4) cannot maintain structural fidelity on Krea2 SingleStreamDiT; even with HSWQ sensitivity protection, final latent trajectory cosine fails to reach 0.90 (severe trajectory drift and collapse). Development has been officially cancelled. Krea2 is supported strictly in **ConvRot INT8** only.
- **ConvRot INT8 (Text Encoder / ControlNet / Model Patch, individually built):** A separate **per-model track** for the families where the universal engine does not apply - each standard has one weight layout, so there is nothing to re-measure per checkpoint. The protection set comes from a measured **aggregate contribution** (per-block, float64), the threshold from the size/quality balance, and the kept tensors are stored FP16. Published as finished packs on Hugging Face (see below).
- **Z Image Hybrid ConvRot NVFP4:** Built from a complete **native ConvRot INT8** UNet via the **reverse method** - layers are converted to NVFP4 in ascending order of per-layer impact (lowest-impact first). Unlike the conventional "protect top-important layers" approach, this stays in the low-error regime where single-layer ranking is valid. The number of NVFP4 layers varies per model (search `K`). Calibrated with `input_scale = amax / 2688` and validated by the **deterministic 20-seed latent-trajectory comparison** (per-step cosine + bifurcation): production gate = **cosine mean ≥ 0.95 and 0/20 bifurcated** in **TC (W4A4)** mode (reference numbers for one example model are listed in the hybrid NVFP4 how-to). Scripts: `Z_Image/diag_impact.py`, `Z_Image/gen_reverse_nvfp4.py`, `Z_Image/calib_input_scale_nvfp4.py`, `benchmark/zi_convrot_nvfp4_traj_compare.py`.

**Technical details - active**

- **diag → reverse hybrid (SDXL, v1.1):** [md/Diag_Reverse_SDXL_v1.1_Technical_Guide.md](md/Diag_Reverse_SDXL_v1.1_Technical_Guide.md) - the SDXL V3.1 selector (DualMonitor + V4 weighted-histogram MSE + full SVD, fixed 300 MiB FP16 protection) extended with the reverse step: per-layer ConvRot INT8 impact is measured on the production sampler, the K lowest-impact layers of the remaining pool are converted while every other layer (the protected set included) stays FP16; validated by the deterministic 25-seed latent-trajectory gate.
- **V5 histogram cosine:** [md/HSWQ_V5_Hybrid_SVD_RMS_Cosine_Technical_Guide.md](md/HSWQ_V5_Hybrid_SVD_RMS_Cosine_Technical_Guide.md) - Stage-3 amax search with the same SVD×RMS hybrid importance as V4, but **cosine similarity loss** on the importance-weighted magnitude histogram (not weighted MSE); includes a full MSE↔cosine mathematical comparison for quantization fidelity.

**Technical details - development ended (retained as technical assets)**

- **FP8:** [md/HSWQ_ Hybrid Sensitivity Weighted Quantization.md](md/HSWQ_%20Hybrid%20Sensitivity%20Weighted%20Quantization.md) - **FP8 development has ended**; this document is retained as a technical asset.
- **INT8 FP16-protect / pack overview:** [md/HSWQ_INT8_SDXL_Technical_Guide.md](md/HSWQ_INT8_SDXL_Technical_Guide.md) - **development ended**; this document is retained as a technical asset (static FP16-protect / pack path).

**ComfyUI Loader for ConvRot INT8 / INT8:** To use these models in ComfyUI, please use this custom node: [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools). Note: **model patches** (the Qwen-Image DiffSynth suite, the Z-Image Fun ControlNets) go in `models/model_patches/` and load with the **HSWQ Model Patch Loader** - the stock `ModelPatchLoader` cannot read the INT8 layout. ControlNets go in `models/controlnet/`.
**ComfyUI Loader for ConvRot NVFP4:** To use these models in ComfyUI, please use this custom node: [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools)
**VAE (SDXL ConvRot INT8 packs):** the quantized file does not embed a VAE. The checkpoint-embedded VAE (`first_stage_model.*`) is removed at conversion, so decode with a separate SDXL VAE.
**VAE (SD1.5 ConvRot INT8 packs):** the quantized file does not embed a VAE. The checkpoint-embedded VAE (`first_stage_model.*`) is removed at conversion (native all-INT8 pack 1058.6 MB; the adopted trajectory-impact pack 1167.5 MB), so decode with a separate SD1.5 VAE. **CLIP is copied through unchanged** - same keys, same dtypes, no quantization artifacts.
**Quantized packs (Hugging Face):**
  - [ControlNet-models-ConvRot-INT8](https://huggingface.co/ussoewwin/ControlNet-models-ConvRot-INT8) - ConvRot INT8 ControlNets and model patches: Illustrious-XL (anytest A/B), SDXL 1.0 (union-pro-max), FLUX.1-dev (Union Pro 2.0), Qwen-Image-2512 (Fun Union 2602), Qwen-Image (Inpainting), and the Qwen-Image DiffSynth canny / depth / inpaint suite. Each pack lists its exact size, SHA256, the required ComfyUI directory and the measured gate.
  - [Clip-files-HSWQ-ConvRot-INT8](https://huggingface.co/ussoewwin/Clip-files-HSWQ-ConvRot-INT8) - ConvRot INT8 text-encoder packs (CLIP-L, CLIP-SAE ViT-L/14, Long-ViT-L, ViT-L-BEST, CLIP-ViT-bigG, CLIP-ViT-H, Flan-T5-XXL, Josiefied-Qwen3-0.6B).
  - [qwen3_4b_8b_abliterated_fp16_convrot_int8](https://huggingface.co/ussoewwin/qwen3_4b_8b_abliterated_fp16_convrot_int8) - Qwen3-4B, Qwen3-8B, Qwen3-VL-4B-Thinking.
  - [SAM3.1-HSWQ-ConvRot-INT8](https://huggingface.co/ussoewwin/SAM3.1-HSWQ-ConvRot-INT8) - SAM 3.1 Multiplex.

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/main/logo.png" width="400">
</p>

---

## Installation & Setup

```bash
# 1. Install PyTorch (CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 2. Install dependencies & quantization backend
pip install -r requirements.txt
pip install -U comfy_kitchen
pip install diffusers accelerate scikit-image
```

---

## How to quantize

> **ComfyUI Node Integration:** Progressive support for direct in-graph quantization inside ComfyUI via custom nodes (`comfyui_nodes/`) is currently underway. Support is available for **HSWQ Quantize ConvRot INT8** (`HSWQ Quantize ConvRot INT8`), **native ConvRot INT8** (`Native ConvRot INT8 Quantize`), and additional model architectures / quantization formats will be rolled out sequentially.

- **SDXL (ConvRot INT8, reverse hybrid):** [How to quantize SDXL](md/How%20to%20quantize%20SDXL.md) - FP16 checkpoint in, per-layer trajectory-impact ranking, K lowest-impact layers to ConvRot INT8 and everything else kept at FP16 (reverse method); validated with the deterministic **25-seed** latent-trajectory comparison (cosine mean >= 0.95, 0/25 bifurcated).
- **SD1.5 (ConvRot INT8, trajectory-impact hybrid):** [How to quantize SD1.5](md/How%20to%20quantize%20SD1.5.md) - measure the per-layer trajectory impact (`sd1.5/diagimpact_sd15.py`, 25 steps), build the pack from that ranking (`sd1.5/quantize_sd1.5_hswq_v1.1.py --top-k K`, bias correction opt-in), and accept with the deterministic 25-seed trajectory gate (cosine mean >= 0.97, 0/25 bifurcated). The guide also records why the retired V1.0 selector failed, the trajectory-vs-single-forward ranking difference (38 of the top 80 layers disagree), and the loader trap in three parts. ComfyUI node coverage is deferred in that guide.
- **Krea2 (ConvRot INT8):** [How to quantize Krea2 ConvRot INT8](md/How%20to%20quantize%20Krea2.md) - CLI (`Krea2/hswq_convrot_int8_krea2_v1.5.py`) and ComfyUI custom node (`Native ConvRot INT8 Quantize`, `model_type = "Krea2"`) quantization guide; for checkpoints with mean cosine ≥ 0.98, native ConvRot INT8 is directly recommended. (Note: Krea2 4-bit / Hybrid NVFP4 development has been cancelled; Krea2 supports ConvRot INT8 only.)
- **Z Image (native ConvRot INT8):** [How to quantize Z Image](md/How%20to%20quantize%20Z%20Image.md) - CLI and ComfyUI custom node (`Native ConvRot INT8 Quantize`) quantization guide. HSWQ-specific Z Image development has **ended**; this How-to introduces the **general** ConvRot INT8 quantization method.
- **Z Image (Hybrid NVFP4, reverse method):** [How to quantize Z Image - Hybrid NVFP4](md/How%20to%20quantize%20Z%20Image%20-%20Hybrid%20NVFP4.md) - build a hybrid NVFP4 model from native ConvRot INT8 by converting the lowest-impact layers first (reverse method), calibrate `input_scale` for Tensor-Core W4A4, and validate with the deterministic 20-seed trajectory comparison (cosine mean ≥ 0.95, 0/20 bifurcated)..
- **Qwen Image Edit (native ConvRot INT8):** [How to quantize Qwen Image Edit](md/How%20to%20quantize%20Qwen%20Image%20Edit.md) - CLI (`Qwen Image/native_convert_int8_convrot_qwen.py`) and ComfyUI custom node (`Native ConvRot INT8 Quantize`, `model_type = "Qwen Image Edit"`) quantization guide; post-convert benchmark is latent-space trajectory divergence (per-step cosine + bifurcation detection).
- **Text Encoder / ControlNet / Model Patch (individually built, per-model track):** [How to quantize Text Encoder, ControlNet and SAM 3 / 3.1](md/How%20to%20quantize%20Text%20Encoder%20and%20ControlNet.md) - the **per-model track**, outside the HSWQ universal engine: each model's protection set comes from a measured aggregate contribution (Rules 1-8), verified through the loader the runtime uses on a fixed 256-item set. Carries the finished text-encoder, ControlNet and model-patch packs with their measured gates, and the finished-pack inventory (family / runtime object / directory / SHA256).
- **ControlNet & SAM 3 / 3.1 (native ConvRot INT8):** [How to quantize Text Encoder, ControlNet and SAM 3 / 3.1](md/How%20to%20quantize%20Text%20Encoder%20and%20ControlNet.md) - ComfyUI custom node (`TE / ControlNet ConvRot INT8 Quantize`, `HSWQ SAM3 Loader`) in-graph quantization guide for ControlNet / ControlNet Union / Model Patch models and Segment Anything foundation models (SAM 3 / SAM 3.1 Multiplex).

**Benchmark results:**
- **SDXL (ConvRot INT8):** [trajectory (25 seeds)](benchmark%20result/benchmark_sdxl_int8.md)
- **SD1.5 (ConvRot INT8):** [trajectory (25 seeds)](benchmark%20result/benchmark_sd1.5_int8.md)
- **Krea2 (ConvRot INT8):** [trajectory (25 seeds)](benchmark%20result/benchmark_krea2_int8.md)
- **Z Image (Hybrid NVFP4):** [trajectory (20 seeds)](benchmark%20result/benchmark_zi_nvfp4.md)

---

## Overview

| Feature | ConvRot INT8 (SDXL, Krea2, SD1.5) | Z Image Hybrid ConvRot NVFP4 |
| :--- | :--- | :--- |
| **Compatibility** | ComfyUI `int8_tensorwise` / QUANT_ALGOS compatible | ComfyUI Load Diffusion Model / QUANT_ALGOS `nvfp4` compatible |
| **File format** | INT8 weights + scale (`int8_tensorwise`) on eligible layers; high-impact / structural layers stay FP16 / BF16 (SD1.5: top-K by measured trajectory impact stay FP16, embedded VAE stripped, embedded CLIP copied through unchanged) | Hybrid: lowest-impact layers → **NVFP4**, remaining layers stay **ConvRot INT8**; built from complete native ConvRot INT8 UNet |
| **Image quality** | Validated on latent trajectory gate (no VAE-masked degradation) | trajectory cosine **0.96-0.97** (mean, 20 seeds) |
| **Mechanism** | **SDXL:** Reverse hybrid - per-layer **trajectory-impact** ranking on FP16 baseline; K lowest-impact layers → FULL ConvRot INT8, rest FP16<br>**Krea2:** Structure blacklist protection (`first.`, `last.`, `mod.`, `norm`, etc.) + 4-axis composite ranking (or native ConvRot INT8)<br>**SD1.5:** trajectory-impact hybrid - per-layer 25-step trajectory impact -> top-K stay FP16 -> ConvRot INT8 on the remainder; embedded CLIP copied through unchanged, embedded VAE removed | **Reverse method**: start from ConvRot INT8 (error ≈ 0), convert layers to NVFP4 in ascending per-layer impact order |
| **Keep ratio** | N/A (layer count `K` varies per model: SDXL reverse hybrid, SD1.5 top-K trajectory impact, Krea2 structure + sensitivity retention) | N/A (layer count varies per model, e.g. nv60-nv110) |
| **Benchmark** | **Deterministic 25-seed latent trajectory**:<br>• **SDXL:** cosine mean $\ge 0.95$ and 0/25 bifurcated<br>• **Krea2:** cosine mean $\ge 0.98$ and 0/25 bifurcated<br>• **SD1.5:** cosine mean $\ge 0.97$ (measured 0.97622; native 0.96265) and 0/25 bifurcated | Deterministic 20-seed latent trajectory (cosine mean $\ge 0.95$, 0/20 bifurcated) |
| **Use case** | SDXL, Krea2 & SD1.5 ConvRot INT8 distribution / native ComfyUI load | Z Image Turbo Hybrid NVFP4 distribution / native ComfyUI load |

**Note (Z Image 8-bit):** HSWQ Z Image INT8 development and publication **ended**. Native ConvRot INT8 is sufficient for Z Image (typically **SSIM > 0.99**). HSWQ INT8 remains the SDXL & Krea2 path.



**Note (Krea2 4-bit / NVFP4):** Krea2 Hybrid ConvRot NVFP4 development has been **cancelled**. 4-bit precision (NVFP4) cannot maintain acceptable fidelity on Krea2 SingleStreamDiT; even with HSWQ sensitivity weighting and layer retention, the final latent trajectory cosine fails to reach 0.90. Krea2 is supported strictly in **ConvRot INT8** only.

**Validation (Z Image Hybrid NVFP4):** production gate is the **deterministic 20-seed latent-trajectory comparison** - per-step cosine with a bifurcation detector; pass = **mean ≥ 0.95 and 0/20 bifurcated**, measured in **TC (W4A4)** mode after `input_scale` calibration.

File size is reduced by about **29-40%** vs FP16 while keeping best quality per use case.

---

## Architecture

1. **Dual Monitor System** - During calibration, two metrics are collected:
   - **Sensitivity** (output variance): layers that hurt image quality most if corrupted → kept in FP16 when selected by HSWQ. The **ConvRot INT8 (SDXL) reverse hybrid** does not use this static path: its FP16 set is the complement of the K lowest-impact layers measured directly on the sampling trajectory.
   - **Importance:** V1 uses per-channel input mean-abs; V4 uses per-element SVD leverage × RMS magnitude hybrid → weights of the weighted histogram.
   **Technical details:** [Dual Monitor System - Technical Guide](md/Dual_Monitor_System_Technical_Guide.md).

2. **Rigorous grid / pack simulation**
   - **ConvRot INT8 (SDXL reverse hybrid):** each candidate layer's ConvRot INT8 reconstruction is injected on the FP16 baseline and propagated through the production sampler; the resulting trajectory drift ranks the layers. The K lowest-impact layers are packed (`rotate → per-channel INT8`); the rest stay FP16.


3. **Weighted Histogram Optimization** - Finds parameters that minimize quantization error using an importance-weighted histogram (not a plain frequency histogram).
   - **V1 / Fast:** per-channel importance (activation mean-abs) drives the histogram. **Technical details:** [Weighted Histogram MSE - Technical Guide](md/Weighted_Histogram_MSE_Technical_Guide.md).
   - **V4 (SVD × RMS hybrid, MSE):** per-element importance blends **SVD structural leverage** \(L(i,j)=(U_i\cdot\sigma)^2\cdot(V_j)^2\) with **RMS magnitude**; \(\alpha\) tilts toward SVD on heavy-tailed layers (retained as technical asset). **Technical details:** [HSWQ V4 SVD-RMS - Technical Guide](md/HSWQ_V4_Hybrid_SVD_RMS_Technical_Guide.md).
   - **V5 (SVD × RMS hybrid, cosine):** same hybrid importance family as V4; Stage-3 objective is **importance-weighted cosine loss** \(L=1-\langle x,q\rangle_H/(\|x\|_H\|q\|_H)\) against the physical FP8 E4M3 quantize-dequantize map (scale-invariant angular fidelity; reduces absolute-tail blackmail of \(\Delta^*\)). Source: `histogram/weighted_histogram_cosine_v5.py`. **Technical details:** [HSWQ V5 SVD-RMS Cosine - Technical Guide](md/HSWQ_V5_Hybrid_SVD_RMS_Cosine_Technical_Guide.md).
4. **Trajectory-Sensitivity Impact Ranking** - Ranks each layer by the divergence its
   quantization error actually causes after propagating through the full model and sampler
   (dynamical importance, replacing static weight-space saliency).
   - **Reverse method (SDXL ConvRot INT8):** the FP16 checkpoint is the only input; each candidate
     layer's ConvRot INT8 error is injected one at a time and the production sampler is run, so the
     ranking follows the real sampling trajectory. The K lowest-impact layers are packed as ConvRot
     INT8 and the rest stay FP16.
   - **Reverse method (Z Image Hybrid NVFP4):** start from the complete native ConvRot INT8 UNet
     (error ≈ 0) and convert layers to NVFP4 in ascending impact order; single-layer ranking stays
     valid in the low-error additivity regime.
   - **Universal theory:** error interaction (Taylor cross terms, error cancellation), nonlinear
     amplification (Lyapunov-style growth), marginal effects, and Shapley-style attribution -
     why per-layer static measures (histogram MSE / cosine / SVD) cannot predict joint quantization
     error; applies to any iterative sampling system, not a specific model. Sources:
     `sdxl/diag_impact_sdxl.py` (SDXL), `Z_Image/diag_impact.py` (Z Image). **Technical details:**
     [Trajectory-Sensitivity Impact Ranking -
     Technical Guide](md/diag_impact_trajectory_sensitivity_technical_guide.md).

---

## Modes

### ConvRot INT8 (SDXL, reverse hybrid)

- **Scripts:** `sdxl/diag_impact_sdxl.py` (per-layer trajectory impact), `sdxl/gen_reverse_int8_sdxl.py` (reverse hybrid conversion).
- **Input:** the FP16 SDXL checkpoint **only** - no other artifact is consulted. Every ConvRot-eligible Linear/Conv2d in it is a candidate (788 on the reference checkpoint, after excluding the boundary layers `input_blocks.0.0` / `out.*` / `time_embed.*` / `add_embedding.*` / `label_emb.*`).
- **Step 1 (impact):** inject one layer at a time with its ConvRot INT8 reconstruction (`dequant(per-channel INT8(quantize(W @ H^T)))`, no inverse rotation) and run the **production sampler** (`comfy.sample.sample`, dpmpp_2m / karras / cfg 7.0 / fixed seed) - the same trajectory the quality gate uses; record the final-latent drift (relative MSE).
- **Step 2 (conversion):** pack the **K lowest-impact layers** as **FULL ConvRot INT8** (rotate → per-channel INT8; format tag `int8_tensorwise`, stamp `convrot:true`) and leave **every other layer untouched at FP16**.
- **Format:** `.weight` I8 + `.weight_scale` F32 + `.comfy_quant`; ComfyUI mixed-precision ops arm the INT8 layers per layer marker.
- **Size:** FP16 base 6.94 GB → reverse hybrid pack ≈ 4.8-5.0 GB (measured on the published packs; reference `waiIllustriousSDXL_v170` at K = 597 = 4.92 GB; ≈ 3.1 MB saved per converted layer).
- **Validation:** deterministic **25-seed x 25-step** latent-trajectory comparison; production gate = **cosine mean ≥ 0.95 and 0/25 bifurcated**. Guide: [How to quantize SDXL](md/How%20to%20quantize%20SDXL.md).

### ConvRot INT8 (Krea2)

- **Script:** `Krea2/hswq_convrot_int8_krea2_v1.5.py` (Krea2 DiT ConvRot INT8; structure blacklist protection + 4-axis composite ranking: DualMonitor $E[x^2]$ × HistCosine V5 × NVFP4 measured error × SVD Leverage).
- **Structure protection:** `first.`, `last.`, `mod.`, `norm`, `projector`, `tmlp`, `txtmlp`, `tproj`, `txtfusion`, and `bias` stay in original dtype (BF16/FP32), preventing numerical collapse / black latent generation.
- **Data-driven protection:** `--blacklist_keep N` and `--keep_sensitive M` revert the highest-error DiT weights to original dtype based on 4-axis composite ranking.
- **Bias correction (Card 1):** Omitted (**`1off`**). In Krea2 `SingleStreamDiT`, all quantized transformer blocks are `bias=False`, and layers containing bias are already protected by the structure blacklist; Card 1 bias delta has zero effect on Krea2.
- **Native recommendation:** For checkpoints where native ConvRot INT8 achieves mean latent trajectory cosine $\ge 0.98$ (and 0 bifurcations), using native ConvRot INT8 directly without HSWQ is recommended.
- **Format:** `int8_tensorwise` with FULL ConvRot Hadamard rotation on eligible Linear/Conv2d; ComfyUI native load compatible.
- **NVFP4 / 4-bit status:** **Development cancelled.** Extensive empirical testing demonstrated that 4-bit precision (NVFP4) cannot maintain structural fidelity on Krea2 SingleStreamDiT; even with HSWQ sensitivity weighting and layer retention, the final latent trajectory cosine fails to reach 0.90 (frequent trajectory collapse and bifurcations). Development has been officially cancelled, and Krea2 is supported strictly in **ConvRot INT8 only**.

### Z Image Hybrid ConvRot NVFP4

- **Scripts:** `Z_Image/diag_impact.py` (per-layer trajectory impact), `Z_Image/gen_reverse_nvfp4.py` (reverse hybrid conversion), `Z_Image/calib_input_scale_nvfp4.py` (TC `input_scale` calibration), `benchmark/zi_convrot_nvfp4_traj_compare.py` (validation).
- **Prerequisite:** A complete **native ConvRot INT8** UNet created by `Z_Image/native_convert_int8_convrot_zi.py`.
- **Method:** **Reverse method** - start from the complete ConvRot INT8 model (error ≈ 0) and convert layers to NVFP4 in **ascending order of per-layer impact** (lowest-impact first). Unlike the conventional "protect top-important layers" approach, this stays in the low-error regime where single-layer ranking is valid.
- **TC / W4A4:** enabled only after calibration writes `*.input_scale` (`amax / 2688`); forcing TC on an uncalibrated checkpoint collapses quality (cosine ~0.18). Auto-detect uses parity (W4A16) without `input_scale`; `--tc` / `--parity` / `HSWQ_ZI_FORCE_TC` / `HSWQ_ZI_FORCE_PARITY` override.
- **Loader / format:** ComfyUI Load Diffusion Model / QUANT_ALGOS `nvfp4`; standard ComfyUI NVFP4 pack format (Linear NVFP4 + Conv2d INT8).
- **Validation:** deterministic 20-seed x 12-step latent-trajectory comparison; production gate = **cosine mean ≥ 0.95 and 0/20 bifurcated** (per-checkpoint numbers; reference example in the hybrid NVFP4 how-to).
- **NVFP4 layer count:** Varies per model; search K (start nv90).

---

## Recommended Parameters

- **Samples:** 32 (recommended) - number of calibration samples (**same for FP8 and ConvRot INT8**).
- **Steps:** 25 - number of inference steps per sample during calibration (**same for FP8 and ConvRot INT8**).
- **ConvRot group size (SDXL reverse hybrid):** power of 4 (default 256 from the script CLI; drops to the largest power-of-4 divisor per layer).
- **Trajectory steps (SDXL reverse hybrid gate):** 25 steps x 25 random seeds (same as the validation benchmark).

---

## Benchmark (Reference)

| Model | Metric | Score | File size | Compatibility |
| :--- | :--- | :--- | :--- | :--- |
| Original FP16 | 25-seed trajectory cosine mean | 1.000 | 100% | High |
| Native ConvRot INT8 (all convertible layers) | 25-seed trajectory cosine mean | ≈ 0.93 | 50% | High |
| **HSWQ ConvRot INT8 (SDXL, reverse hybrid)** | 25-seed trajectory cosine mean | **≥ 0.95 (gate)** | **71%** (FP16 mixed) | **High** (ComfyUI INT8) |
| **HSWQ ConvRot INT8 (SD1.5, trajectory impact)** | 25-seed trajectory cosine mean | **0.976** (native 0.963; K measured per checkpoint) | **57%** (FP16 mixed) | High (ComfyUI INT8 Conv2d) |
| **Z Image Hybrid ConvRot NVFP4** | 20-seed trajectory cosine mean | **0.96-0.97** | **60%** (FP16 mixed) | **High** (ComfyUI NVFP4) |

HSWQ ConvRot INT8 (SDXL) targets a 25-seed latent-trajectory cosine mean **≥ 0.95** with 0/25 bifurcated (FP16 base 6.94 GB → reverse hybrid pack ≈ 4.9 GB, i.e. ≈ 71%). Z Image Hybrid ConvRot NVFP4 targets a **20-seed latent-trajectory cosine mean of 0.96-0.97** with **0/20 bifurcated**. All keep full loader compatibility on their respective formats.


---

## Security

**Backdoor audit of the ConvRot INT8 source models that the shipped packs are derived from.**

The pack is an INT8 quantised reconstruction of its source. Auditing the source fp16 weights is the same audit performed one layer up, where the 1e-6 precision needed to detect an undeclared weight-level edit actually exists.

- **pack provenance:** INT8 pack vs the source fp16, after ConvRot unfold - either `FAITHFUL`, or some layer carries an undeclared edit

- **source delta:** declared base vs the source, either identical or the only difference is the declared edit

- **behaviour:** generated text on the base and the source for 100 JBB requests



Per-model links (source → shipped pack):

- [CLIP-SAE-ViT-L-14-FP32](https://huggingface.co/zer0int/CLIP-SAE-ViT-L-14) → [CLIP-SAE-ViT-L-14-hswq_convrot_int8.safetensors](https://huggingface.co/ussoewwin/Clip-files-HSWQ-ConvRot-INT8) - no backdoor

- [Josiefied-Qwen3-0.6B-abliterated-v1](https://huggingface.co/Goekdeniz-Guelmez/Josiefied-Qwen3-0.6B-abliterated-v1) → [Josiefied-Qwen3-0.6B-abliterated-v1_hswq_convrot_int8.safetensors](https://huggingface.co/ussoewwin/Clip-files-HSWQ-ConvRot-INT8) - no backdoor

- Huihui-Qwen3-VL-4B-Instruct-abliterated (source; also Huinhui pack umbrella) → [Qwen3_VL_4B_Thinking_abliterated_hswq_convrot_int8.safetensors](https://huggingface.co/ussoewwin/qwen3_4b_8b_abliterated_fp16_convrot_int8) - no backdoor

- [Long-ViT-L-14-BEST-GmP-smooth-ft-full](https://huggingface.co/BeichenZhang/LongCLIP-L) → [Long-ViT-L-14-BEST-GmP-smooth-ft-full_hswq_convrot_int8.safetensors](https://huggingface.co/ussoewwin/Clip-files-HSWQ-ConvRot-INT8) - audit withdrawn

- [ViT-L-14-BEST-smooth-GmP-HF-format](https://huggingface.co/zer0int/CLIP-GmP-ViT-L-14) → [ViT-L-14-BEST-smooth-GmP-HF-format_hswq_convrot_int8.safetensors](https://huggingface.co/ussoewwin/Clip-files-HSWQ-ConvRot-INT8) - audit withdrawn


Records: [md/verdicts/records/](md/verdicts/records/). Checker: [audit/check_cites.py](audit/check_cites.py). Procedure: [md/Backdoor_Audit_Procedure.md](md/Backdoor_Audit_Procedure.md). Guide: [md/Backdoor_Audit_Technical_Guide.md](md/Backdoor_Audit_Technical_Guide.md). Consolidated report: [md/Backdoor_Audit_Verdicts.md](md/Backdoor_Audit_Verdicts.md).

## Changelog

Version history and release notes are in [CHANGELOG.md](CHANGELOG.md).

---

## Base Repositories

This project is built upon the following repositories:

| Repository | In-repo path | Upstream |
| :--- | :--- | :--- |
| **[ComfyUI](https://github.com/Comfy-Org/ComfyUI)** | `ComfyUI-master/` | [@Comfy-Org](https://github.com/Comfy-Org) - The most powerful and modular diffusion model GUI, API and backend with a graph/nodes interface. |


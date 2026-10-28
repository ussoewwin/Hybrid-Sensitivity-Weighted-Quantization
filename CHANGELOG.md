# Changelog

## v2.4.8

**SD1.5 HSWQ V1.1 ComfyUI node integration & workflow automation** — Added dedicated in-workflow HSWQ V1.1 trajectory-impact hybrid quantization support for Stable Diffusion 1.5 directly inside ComfyUI:

- **Complete In-Workflow Pipeline (Zero Terminal Commands):**
  - **Step 1 Trajectory Impact Diagnosis (`HSWQ Diag Impact (ConvRot INT8)`, `model_type = "SD1.5"`):** Measures the per-layer trajectory divergence in-process across all 282 ConvRot-eligible UNet layers (25 steps, seed 42) and writes `<repo>/impact/impact_sd15_traj_<model stem>.json`.
  - **Impact JSON Reuse (`HSWQ Load Diag Impact Json`):** Lists and loads existing SD1.5 trajectory-impact JSONs from a dropdown, allowing instant switching between fresh measurement and precomputed profile via `ImpactSwitch` without re-running Step 1.
  - **Unified Multi-Stage Quantizer (`HSWQ Quantize ConvRot INT8`, `model_type = "SD1.5"`):** Consumes connected `model` / `clip` / `vae` along with `impact_json` from the switch:
    - **Step 2 UNet Hybrid Pack:** Protects the top-$K$ highest-impact layers in original FP16 and converts remaining 2D `Linear` and 4D `Conv2d` layers to FULL ConvRot INT8 (Hadamard group size 256, FP32 `weight_scale`, `.comfy_quant` metadata) with optional systematic INT8 bias correction folded from calibration activation means.
    - **CLIP Pass-Through:** Preserves checkpoint-embedded CLIP text encoder 100% untouched (`cond_stage_model.*` keys and FP16 dtypes identical to base; zero quantization artifacts).
    - **Embedded VAE Removal:** Automatically strips `first_stage_model.*` from the output pack, saving storage (~1.16 GB HSWQ pack vs 2.03 GB base).
    - **Step 3 Deterministic Trajectory Gate:** Automatically executes the 25-seed latent trajectory benchmark against the FP16 base immediately after quantization (production gate: mean final-cosine $\ge 0.95$ and 0/25 bifurcated), piping live progress and final results to `Show Text` nodes.
- **Workflow & Visual Assets:** Shipped ready-to-use ComfyUI sample workflow (`sample workflow/hswq sd1.5 convrot int8.json`) and updated workflow canvas reference image (`png/hswq sd1.5.png`).
- **Documentation:** Updated `md/How to quantize SD1.5.md` with full Method 2 (ComfyUI Nodes: In-Workflow Quantization) documentation, wiring instructions, parameter tables, and execution steps.

Release notes: [v2.4.8](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v2.4.8)

## v2.4.7

**SD1.5 ConvRot INT8: the FP16 protection set now comes from a measured trajectory impact instead of static weight statistics** - The retired V1.0 selector ranked protection from weight-space quantities (kurtosis, outlier ratio, absmax, weighted-histogram MSE). Those answer "how much does this layer's *weight* distort", not "how much does this layer move the *output*": in a deep UNet a layer's error is amplified, attenuated or cancelled downstream, and a weight-space statistic cannot see that propagation. The measured consequence was that V1.0's protection set scored **the same as native** - the protection bought nothing. The defect was the criterion, not the quantizer.

- **The criterion:** for each ConvRot-eligible layer, one at a time - hold every other layer pristine, replace **that one layer's weight** by its ConvRot INT8 reconstruction **in the original frame** (`dequant(INT8(W @ H^T))` = `(q*s) @ H`), run the same fixed **25-step denoising trajectory**, and record `rel_mse` against the pristine trajectory. The original frame matters: the pack stores the rotated weight and the kernel rotates the activation, so the error that reaches the output is `(q*s) @ H - W`; injecting in the rotated frame measures the rotation itself and inverts the ranking.
- **Why a trajectory and not a single forward:** they disagree on **38 of the top 80 layers**. A trajectory ranks the decoder-side `skip_connection` lines and output blocks highest (`output_blocks.6.0.skip_connection` is rank 1), because their error reaches the final latent directly; a single forward ranks the encoder-side early `proj_in` transitions (`input_blocks.1.1.proj_in` is rank 1 there and rank **58** under a trajectory), because they move that one timestep the most while part of their error is absorbed later.
- **The measured gate** (25 seeds x 25 steps, 512x512, cfg 7.0, on `0.5xxmix9realistic_v40 0.5BRA7`): **native 0.96265 / k40 0.96212 / k80 0.97622 / k100 0.974**, with **0/25 bifurcated** in every candidate. **k80 is the adopted point** - **+0.0136** over native and 18/25 same-image against 13/25.
- **What the curve says:** protecting 40 layers **buys nothing** (0.96212, identical to native), so the layers that matter are concentrated between positions 40 and 80. More protection is **worse**: k100 (0.974) sits *below* k80 and is 26 MB larger, because a protected layer keeps its FP16 weights but **loses its bias correction**, while an INT8 layer absorbs bias correction and cancels its systematic error. Past k80, promoting layers to FP16 loses more correction than it gains in precision.
- **Bias correction is INT8-only and opt-in** (`--bias_correction`, default OFF, the V1.0 default). When enabled it folds `E[(W_q - W) x] ~ (W_q - W) @ E[x]` into the layer bias, with `E[x]` collected in the same run from the V1.0 calibration (32 prompts x 25 steps, seed 42). Protected layers are untouched - weights and bias alike. The builder reports the honest count and **aborts** rather than saving a pack labelled bias-corrected if zero layers were written.
- **Composition:** UNet only. **CLIP is copied through unchanged** (same keys, same dtypes, zero `.comfy_quant` / `.weight_scale` artifacts) and the **VAE is dropped**.
- **Sizes (measured):** native all-INT8 **1058.6 MB**; k=80 **1167.5 MB** (source 2033.8 MB).
- **Scripts:** `sd1.5/diagimpact_sd15.py` (the criterion), `sd1.5/quantize_sd1.5_hswq_v1.1.py` (the builder; self-contained - the key-mapping helpers are inlined rather than imported from the archived V1.0), `sd1.5/verify_sd15_loader.py` (the acceptance through the real loader). The retired `quantize_sd1.5_hswq_v1.0.py` now lives under `archives/`.
- **The loader trap, recorded because it costs hours:** ComfyUI's `comfy_kitchen` has **no INT8 conv kernel**, so the Conv2d INT8 path exists only in the HSWQ loader's `_patch_ops_decode_and_conv()`, gated on a thread-local; and there are **two vendored copies** of that patch module, each with its own thread-local. An INT8 Conv2d pack loaded without opening all of them yields NaN (`int8_linear` calls 0). Use `sd1.5/verify_sd15_loader.py`, which drives `load_checkpoint_sd15_hswq_weight_dtype`.
- **Honest limits, stated in the guide:** no criterion here is globally optimal. Every method measures **one layer at a time** while tens are quantized together, so single-layer contributions are not additive - two criteria that disagree on 38 of 80 layers can reach the same final score, and two that agree can still differ. The table reads *"best among the candidates measured under this criterion"*, not *"the optimum"*.
- **Documentation:** `md/How to quantize SD1.5.md` was rewritten around this method (one pipeline: measure -> build -> accept), and the README's six SD1.5 statements were corrected to the measured values. ComfyUI node coverage is explicitly deferred in the guide.

Release notes: [v2.4.7](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v2.4.7)

## v2.4.6

**Per-model ConvRot INT8 track: ControlNet, model patch, text encoder and SAM packs (method statement, Hugging Face release)** - Completed and documented the **specialised (per-model) track** for the families where HSWQ's universal engine does not apply, and published the resulting packs:

- **Why this track is separate from the universal engine:** HSWQ's engine is a universal quantizer - one autonomous code path whose every decision is derived from the target checkpoint's own statistics, with no model-specific toggles, so a new merge of a supported family needs no new code. That leverage does not carry over to these families, and the documentation now says so up front instead of implying otherwise. The measured cost of the per-model approach in this track is **9 measurement scripts, 17 converter scripts, 23 verify scripts and 93 CLI arguments** across just the converters.
- **The four technical reasons universalisation fails here:** (1) the runtime object differs - a file named "ControlNet" may be a real `ControlNet` (`load_controlnet_qwen_fun`, `load_controlnet_qwen_instantx`), a **model patch** (`ZImage_Control`, `QwenImageBlockWiseControlNet`), or a standard tower, each with its own loader, apply path and directory; (2) the observable differs - per-block residual list, single tower embedding, or denoising trajectory, so the scoring function differs too; (3) the graph differs - some need a base model to reach the entry path, some need a load-time key conversion, some must not be converted; (4) shape parameters differ inside one family - in the Qwen-Image diffsynth family `img_in` is `[3072,64]` for canny/depth and `[3072,68]` for inpaint, so a shared converter would have to branch on `additional_in_dim`, the shared-code-plus-conditionals the universal design avoids precisely by being genuinely uniform.
- **The economic reason the specialised approach is affordable here:** a standard has **one** model; a UNet family has many. SDXL alone was tested on 30+ checkpoints, and every merge or finetune is a new target whose ranking must be re-measured. A text encoder, a ControlNet standard or a SAM release is one weight layout per standard and is not merged, forked or re-finetuned into new checkpoints, so there is nothing to re-measure per checkpoint - and a universal engine, whose whole value is amortising one code path across many checkpoints, has nothing to amortise over here.
- **The written method that is shared (Rules 1-8):** the injection frame is the original frame (`(q*s)H^T - W`, not the rotated frame); drift is only a screen while the **aggregate contribution** is the decision; the gate must use the same instrument as the selection; the threshold is chosen on the **size/quality balance** (flat curve, single step, or floor); kept tensors are stored FP16; model content is never dropped to shrink a pack; no machine-specific paths in public code; and the observable must be what the runtime injects (scored per block, in float64, with `cond` recomputed inside the perturbation loop).
- **The three measured curve shapes:** *flat* - no layer stands out, take the smallest satisfying pack (`controlnet-union-pro-max-sdxl-1.0`, 0.999961); *single step* - one entry layer carries most of the contribution, protect it and stop (the three Qwen-Image diffsynth patches, where `img_in` is 147x to 598x the runner-up); *floor* - every eligible layer is safe at INT8, protect nothing (`Qwen-Image-2512-Fun-Controlnet-Union-2602` 0.999158, `Qwen-Image-ControlNet-Inpainting` 0.999545, and the two Z-Image patches at trajectory 0.999862 / 0.999956).
- **Rebuilt and verified packs (all through the loader the runtime uses):** the three Qwen-Image diffsynth model patches now protect exactly one layer (`img_in`) for a 542x to 642x reduction of the error; the two Qwen-Image ControlNets and the two Z-Image patches keep every eligible layer in INT8; every kept tensor moved to FP16, which shrank two packs while improving them.
- **Loader correctness made explicit:** ControlNets belong in `models/controlnet/`, model patches in `models/model_patches/`, and **ComfyUI's stock `ModelPatchLoader` cannot read the INT8 layout** - it builds the patch with `manual_cast` and reads the INT8 tensors as plain weights. The packs load with `HSWQModelPatchLoaderCustom`, and the acceptance now runs through that node, agreeing with the direct-graph harness to 10 decimal places.
- **Published packs (Hugging Face):**
  - [ussoewwin/ControlNet-models-ConvRot-INT8](https://huggingface.co/ussoewwin/ControlNet-models-ConvRot-INT8) - all eleven ControlNet and model-patch packs, with exact sizes, per-pack SHA256, the required directory and the measured gate.
  - [ussoewwin/Clip-files-HSWQ-ConvRot-INT8](https://huggingface.co/ussoewwin/Clip-files-HSWQ-ConvRot-INT8) - the text-encoder packs.
  - [ussoewwin/SAM3.1-HSWQ-ConvRot-INT8](https://huggingface.co/ussoewwin/SAM3.1-HSWQ-ConvRot-INT8) - the SAM 3.1 pack.
- **Documentation:** `md/How to quantize Text Encoder and ControlNet.md` carries the scope warning, the per-model rationale with its measured counts, Rules 1-8, the finished-pack inventory (family / runtime object / directory / gate / SHA256) and the one-line method summary.

Release notes: [v2.4.6](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v2.4.6)

## v2.4.5

**HSWQ ConvRot INT8 SAM 3.1 pack (individually built, Hugging Face release)** - Added an individually built ConvRot INT8 pack for SAM 3.1 Multiplex, with measured FP16 protection, published to Hugging Face:

- **Individually built per standard:** SAM 3 / SAM 3.1 are foundation segmentation models, not per-checkpoint image-generation backbones - each version ships a single standard weight layout, so there is no per-checkpoint measurement to redo and no universal engine to build. They are handled individually, the same way as the text encoders in v2.4.4: the risk of every Linear layer was measured once (kurtosis / outlier ratio), the fat-tail layers were kept in FP16, and the pack was built with a per-standard converter (`clip_convert/convert_sam31_convrot_int8.py`, 538 ConvRot INT8 / 113 FP16 protected).
- **Measured protection:** all 32 vision-backbone `attn.qkv` (kurt median 5,252, up to 19,036), all 32 vision `mlp.fc1`, all 24 language `attn.out_proj` (up to 39,567) and all 24 language `mlp.c_fc`, plus the tracker cross-attention / mask-decoder attention tails.
- **SAM preprocessing reused:** the fused QKV `in_proj` tensors are split into `q_proj` / `k_proj` / `v_proj`, the per-block `freqs_cis` buffers are dropped, and SAM 3.1 keeps its (1024,1024) `text_projection`. The pack loads natively via `UNetLoader` (538 `comfy_quant` layers).
- **Published pack (Hugging Face):**
  - [ussoewwin/SAM3.1-HSWQ-ConvRot-INT8](https://huggingface.co/ussoewwin/SAM3.1-HSWQ-ConvRot-INT8) - SAM 3.1 Multiplex (`sam3.1_multiplex_hswq_convrot_int8.safetensors`).
  - Note: SAM 3 (classic / `sam3.pt`) is the same kind of case and should be built and published the same way, but it **cannot be published for license reasons**; only SAM 3.1 is published.
- **Documentation:** `md/How to quantize Text Encoder and ControlNet.md` now presents the finished SAM 3.1 pack (with the Hugging Face link above) instead of a method - SAM, like CLIP, has no universal method because each standard is a single weight layout - and keeps the generic `clip_convert/convert_clip_convrot_int8.py` as the native A/B baseline.

Release notes: [v2.4.5](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v2.4.5)

## v2.4.4

**HSWQ ConvRot INT8 text-encoder packs (individually built, Hugging Face release)** - Added a set of individually built ConvRot INT8 text-encoder packs, each with measured FP16 protection, published to Hugging Face:

- **Individually built per standard:** text encoders are not per-checkpoint models (each architecture is a single standard weight layout), so instead of a universal engine each standard was handled on its own - the risk of every Linear layer was measured once (kurtosis / outlier ratio), the fat-tail layers were kept in FP16, and the pack was verified by conditioning cosine against the FP16 baseline. Per-standard converter scripts live under `clip_convert/`:
  - `convert_clip_l_convrot_int8.py` - CLIP-L (48 ConvRot INT8 / 26 FP16 protected / 74 total)
  - `convert_sae_vit_l_convrot_int8.py` - OpenAI CLIP ViT-L/14 (145 ConvRot INT8 / 76 FP16 protected / 221 total, conditioning cos 0.999922)
  - `convert_flan_t5_xxl_convrot_int8.py` - Flan-T5-XXL (430 ConvRot INT8 / 7 FP16 protected / 437 total, conditioning cos 0.99935)
  - `convert_josiefied_qwen3_06b_convrot_int8.py` - Qwen3-0.6B (112 ConvRot INT8 / 85 FP16 protected / 197 total, conditioning cos 0.99814)
  - `convert_qwen3_4b_convrot_int8.py` - Qwen3-4B (144 ConvRot INT8 / 109 FP16 protected / 253 total, conditioning cos 1.000009)
  - `convert_qwen3_vl_4b_convrot_int8.py` - Qwen3-VL-4B-Thinking (197 ConvRot INT8 / 161 FP16 protected / 358 total, conditioning cos 0.999546)
- **Published packs (Hugging Face):**
  - [ussoewwin/Clip-files-ConvRot-INT8](https://huggingface.co/ussoewwin/Clip-files-ConvRot-INT8) - CLIP-L, OpenAI CLIP ViT-L/14 (CLIP-SAE), Flan-T5-XXL, CLIP-ViT-bigG-14, Josiefied-Qwen3-0.6B
  - [ussoewwin/qwen3_4b_8b_abliterated_fp16_convrot_int8](https://huggingface.co/ussoewwin/qwen3_4b_8b_abliterated_fp16_convrot_int8) - Qwen3-4B, Qwen3-8B, Qwen3-VL-4B-Thinking
- **Documentation:** `md/How to quantize Text Encoder and ControlNet.md` now presents the finished text-encoder packs (with the Hugging Face links above) instead of a method - CLIP has no universal method because each standard is a single weight layout - and keeps the generic `clip_convert/convert_clip_convrot_int8.py` as the native A/B baseline.

Release notes: [v2.4.4](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v2.4.4)

## v2.4.3

**SDXL HSWQ V1.1 ComfyUI node integration & workflow automation** — Added full in-workflow HSWQ Reverse Hybrid ConvRot INT8 V1.1 quantization support for SDXL directly inside ComfyUI:

- **Complete In-Workflow Pipeline (Zero Terminal Commands):**
  - **Step 1 Trajectory Impact Diagnosis (`HSWQ SDXL Diag Impact`):** Runs per-layer trajectory impact measurement (`sdxl/diag_impact_sdxl.py`, 715 pool layers after V3.1 selector protection, 25 steps, seed 42) and writes `<repo>/impact/impact_<model>.json`.
  - **Impact JSON Reuse (`HSWQ Load Diag Impact Json`):** Selects and loads existing impact JSONs from a dropdown, bypassing the lengthy Step 1 re-run during $K$ layer search.
  - **Unified Multi-Stage Quantization (`HSWQ Quantize ConvRot INT8`, `model_type = "SDXL"`):**
    - **Step 2 UNet Reverse Hybrid:** Runs `sdxl/gen_reverse_int8_sdxl_v1.1.py` to pack the $K$ lowest-impact pool layers to ConvRot INT8 (Hadamard group size 256, per-channel INT8, `.comfy_quant` metadata) with optional systematic INT8 bias correction, keeping all protected layers at FP16.
    - **Step 2b CLIP ConvRot INT8:** Automatically quantizes SDXL CLIP-L and CLIP-G (`sdxl/quantize_sdxl_clip_int8.py`, 133 layers converted / 38 protected), eliminating ~382 MiB of unneeded FP16 bloat.
    - **Step 3 Deterministic Trajectory Gate:** Automatically executes the 25-seed latent trajectory benchmark (`benchmark/sdxl_int8_traj_compare.py`, production gate: mean $\ge 0.95$ and 0/25 bifurcated) on the final pack, piping live progress and scores to `Show Text` nodes.
- **Embedded VAE Removal:** Enforced automatic stripping of checkpoint-embedded VAE tensors (`first_stage_model.*`) across both HSWQ and Native SDXL converters.
- **Process Isolation & Memory Optimization:** Subprocess execution with automatic VRAM purging (`cleanup_models`, `unload_all_models`, `free_memory`) and caching allocator enforcement (stripping `backend:cudaMallocAsync` from `PYTORCH_CUDA_ALLOC_CONF` to reduce peak VRAM by 6.7 GiB and prevent Triton CUDA-graph failures).
- **Workflow & Documentation:** Shipped ready-to-use ComfyUI sample workflow (`sample workflow/hswq sdxl convrot int8.json`) and updated guide (`md/How to quantize SDXL.md`).

Release notes: [v2.4.3](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v2.4.3)

## v2.4.2

**SD1.5 HSWQ V1.0 ComfyUI node integration (`HSWQ Quantize ConvRot INT8`)** — Added dedicated in-workflow HSWQ V1.0 quantization node for Stable Diffusion 1.5 (`comfyui_nodes/hswq_sd15_quantize.py`, registered as `HSWQ Quantize ConvRot INT8`):

- **Complete In-Workflow Pipeline:** Runs the full HSWQ V1.0 flow without manual terminal commands, including automatic source checkpoint resolution from the `MODEL` input.
- **Distribution Analysis & Calibration:** Integrates automated layer profiling (`analyze/analyze_sd1.5_distribution.py`) and mandatory r32 DualMonitor calibration (32 samples × 25 steps) with preset prompts sheet dropdown (`128` default, `256`, `512` from `sample/`).
- **Hybrid V4 Optimization & Protection:** Enforces the strict 300 MiB FP16 protection budget hard ceiling via V4 Full-SVD×RMS weighted histogram MSE ranking, packing remaining UNet layers to FULL ConvRot INT8 (group size 256).
- **Embedded CLIP Quantization:** Automatically quantizes the checkpoint-embedded CLIP text encoder to ConvRot INT8 (72 quantized, 125 float) with loader metadata injection.
- **Embedded VAE Stripping:** Automatically removes `first_stage_model.*` (~1.28 GB HSWQ pack vs ~1.02 GB Native all-INT8 pack with CLIP).
- **Automated Validation & Canvas Monitoring:** Executes the deterministic 25-seed latent trajectory benchmark under INT8 CLIP conditioning, piping live progress and final scores to `Show Text` nodes.
- **Documentation & Workflow:** Includes updated technical guide (`md/How to quantize SD1.5.md`) and workflow reference image (`png/hswq sd1.5.png`).

Release notes: [v2.4.2](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v2.4.2)

## v2.4.1

**SD1.5 Native ConvRot INT8 quantization and ComfyUI integration** — Added full Native ConvRot INT8 quantization support for Stable Diffusion 1.5 (SD1.5) via dedicated CLI quantizer (`sd1.5/native_convert_int8_sd1.5.py`) and ComfyUI custom node (`Native ConvRot INT8 Quantize`, `model_type = "SD1.5"`). Because SD1.5 is naturally lightweight (~1.9 GB FP16) and Native ConvRot INT8 routinely reaches mean latent trajectory cosine $\ge 0.96$ (~0.97), HSWQ hybrid FP16 retention is omitted to preserve maximum compression. To maximize file size reduction, the checkpoint-embedded VAE (`first_stage_model.*`) is automatically stripped during quantization, reducing file size to approximately **850–950 MB** (saves ~160–330 MB). Features integrated 25-seed deterministic latent trajectory benchmark (512x512, dpmpp_2m/karras, cfg 7.0, 25 steps) to evaluate fidelity strictly in latent space without VAE decoder smoothing bias. Includes technical how-to guide (`md/How to quantize SD1.5.md`) and workflow reference image (`png/sd1.5.png`).
Release notes: [v2.4.1](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v2.4.1)

## v2.4.0

**SDXL ConvRot NVFP4 development cancelled (ConvRot INT8 only)** — Deterministic latent-trajectory measurements revealed that 4-bit precision (NVFP4) fails to maintain latent trajectory fidelity on SDXL (final trajectory cosine fails to reach 0.90, suffering severe trajectory drift and collapse). The legacy decoded-image benchmark scores (SSIM ~0.95) were an artifact masked and smoothed by VAE decoding rather than true latent-space preservation. Following the same conclusion as Krea2 in v2.3.5, SDXL NVFP4 development has been officially discontinued, and SDXL is maintained strictly via the high-fidelity **ConvRot INT8 (reverse hybrid)** pipeline. All deprecated SDXL NVFP4 conversion scripts and benchmark tools (`hswq_sdxl_convert_nvfp4_1.0.py`, `hswq_sdxl_convert_nvfp4_1.2.py`, `benchmark/nvfp4bench_sdxl.py`, and trajectory comparators) have been relocated to `archives/`, and documentation has been updated.
Release notes: [v2.4.0](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v2.4.0)

## v2.3.9

**Krea2 deprecated NVFP4 scripts and benchmark suite removed** — Following the official discontinuation of Krea2 Hybrid NVFP4 in v2.3.5 in favor of ConvRot INT8, all legacy NVFP4 files have been removed from the repository. This cleans up the calibration, impact diagnosis, and reverse conversion scripts (`Krea2/calib_input_scale_nvfp4.py`, `Krea2/diag_impact.py`, `Krea2/gen_reverse_nvfp4.py`), the standalone NVFP4 runtime benchmark package (`benchmark/krea2_convrot_nvfp4/`), and the trajectory comparison script (`benchmark/krea2_traj_compare.py`). Krea2 quantization and benchmarking are maintained exclusively via the high-fidelity ConvRot INT8 pipeline (`Krea2/hswq_convrot_int8_krea2_v1.5.py` and `benchmark/krea2_int8_traj_compare.py`).
Release notes: [v2.3.9](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v2.3.9)

## v2.3.8

**SDXL ConvRot INT8 benchmark record format switched to the latent-trajectory format** — `benchmark result/benchmark_sdxl_int8.md` is now generated from the deterministic 25-seed latent-trajectory comparison (`benchmark result/score_sdxl_int8.txt`) and reports **HSWQ ConvRot INT8 vs Native ConvRot INT8** as a cross-model summary plus per-model per-seed side-by-side tables (final cosine, latent MSE, bifurcation, speedup). The previous decoded-image MSE / SSIM tables and the 5-seed reverse-hybrid K table are removed, and the README benchmark list label is updated to match.
Release notes: [v2.3.8](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v2.3.8)

## v2.3.7

**SDXL ConvRot INT8 reverse converters: the embedded VAE is removed from the quantized file** — `sdxl/gen_reverse_int8_sdxl_v1.1.py` drops `first_stage_model.*` (the checkpoint-embedded VAE: 248 tensors / ~160 MiB on the reference pack) immediately before `save_file`, and both reverse converters load the checkpoint with `output_vae=False` (the returned VAE slot is `None` and is never used). Conversion and calibration run in latent space only and never decode an image, so the embedded VAE was payload only. The quantized UNet body (`model.diffusion_model.*`) is unchanged; decoding uses a separate VAE.
Release notes: [v2.3.7](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v2.3.7)

## v2.3.6

**SDXL ConvRot INT8 method: reverse (trajectory-impact) step added to the HSWQ V3.1 selector** — The SDXL INT8 pipeline keeps the V3.1 selector (DualMonitor + V4 weighted-histogram MSE + full SVD, fixed 300 MiB FP16 protection) and extends it with the reverse step: the per-layer ConvRot INT8 impact is measured on the production sampler over the pool the selector leaves, and the K lowest-impact pool layers are converted while every other layer — the selector-protected layers included — stays FP16.
Release notes: [v2.3.6](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v2.3.6)

## v2.3.5

**Krea2 Hybrid ConvRot NVFP4 development cancelled (ConvRot INT8 only)** — Extensive empirical evaluations demonstrated that 4-bit precision (NVFP4) cannot maintain structural generation fidelity on Krea2 SingleStreamDiT; even with HSWQ sensitivity weighting and layer retention, final latent trajectory cosine fails to reach 0.90 (resulting in severe trajectory drift and frequent bifurcations). Consequently, Krea2 Hybrid NVFP4 development has been officially discontinued. Krea2 is supported strictly via **ConvRot INT8** (`Krea2/hswq_convrot_int8_krea2_v1.5.py` and ComfyUI `Native ConvRot INT8 Quantize`), which reliably preserves high generation fidelity (mean trajectory cosine $\ge 0.98$).
Release notes: [v2.3.5](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v2.3.5)

## v2.3.4

**Krea2 ConvRot INT8 quantization method published** — Added native and HSWQ ConvRot INT8 quantization support for Krea2 DiT models via CLI quantizer (`Krea2/hswq_convrot_int8_krea2_v1.5.py`) and ComfyUI custom node (`Native ConvRot INT8 Quantize`, native only). Features structural blacklist protection for entry/exit and normalization layers, 4-axis composite error ranking, and standard 1off bias configuration. For checkpoints achieving mean latent trajectory cosine $\ge 0.98$, native ConvRot INT8 without HSWQ is recommended. Includes 20-seed multi-trajectory benchmark (`benchmark/krea2_int8_traj_compare.py`) and technical guide (`md/How to quantize Krea2.md`).
Release notes: [v2.3.4](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v2.3.4)

## v2.3.3

**SAM 3 and SAM 3.1 native ConvRot INT8 quantization support** — Extended `TE / ControlNet ConvRot INT8 Quantize` node (`comfyui_nodes/te_controlnet_convrot_int8_convert.py`), added dedicated `HSWQ SAM3 Loader (sam3.pt)` (`comfyui_nodes/sam3_pt_loader.py`), and updated standalone CLI converter (`clip_convert/convert_clip_convrot_int8.py`) to support native ConvRot INT8 quantization for Segment Anything models (SAM 3 & SAM 3.1 Multiplex). Features automated architectural version detection, FPN remapping, qkv projection splitting, and unused projection pruning. Includes updated guide (`md/How to quantize Text Encoder and ControlNet.md`) and sample workflow (`sample workflow/native convrot int8.json`).
Release notes: [v2.3.3](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v2.3.3)

## v2.3.2

- **Qwen Image Edit native ConvRot INT8 quantization published** — Added native ConvRot INT8 quantization support for Qwen Image Edit models via CLI (`Qwen Image/native_convert_int8_convrot_qwen.py`) and ComfyUI custom node (`Native ConvRot INT8 Quantize` in `comfyui_nodes/`). Features orthogonal Hadamard rotations before per-channel INT8 scaling and automated post-convert fidelity benchmarking (latent trajectory divergence and SSIM). Includes technical guide (`md/How to quantize Qwen Image Edit.md`).
- **Text Encoder and ControlNet native ConvRot INT8 quantization published** — Added dedicated ComfyUI custom node (`TE / ControlNet ConvRot INT8 Quantize` in `comfyui_nodes/te_controlnet_convrot_int8_convert.py`) to quantize Text Encoders (CLIP, T5, Qwen2.5-VL) and ControlNet / ControlNet Union models (SDXL, Qwen-Image, FLUX.1) directly from memory in ComfyUI graph. Reduces VRAM usage by ~50% while maintaining structural guidance fidelity. Includes guide (`md/How to quantize Text Encoder and ControlNet.md`) and sample workflow (`sample workflow/native convrot int8.json`).
Release notes: [v2.3.2](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v2.3.2)

## v2.3.1

**Z Image native ConvRot INT8 ComfyUI custom node support** — Added dedicated ComfyUI custom node support (`Native ConvRot INT8 Quantize` in `comfyui_nodes/`) to perform native ConvRot INT8 quantization directly within ComfyUI workflows, complete with integrated multi-seed automated fidelity benchmarking (latent MSE, cosine similarity, inference time, and decoded SSIM). Includes ready-to-use sample workflow (`sample workflow/native convrot int8.json`) and updated guide (`md/How to quantize Z Image.md`).
Release notes: [v2.3.1](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v2.3.1)

## v2.3

**Z Image Hybrid ConvRot NVFP4 quantization method published** — Released the **reverse method** hybrid NVFP4 quantization for Z Image Turbo models. Starting from a complete native ConvRot INT8 UNet, layers are converted to NVFP4 in ascending order of per-layer impact (lowest-impact first), staying in the low-error regime where single-layer ranking is valid. The NVFP4 layer count varies per model (e.g. nv60-nv110), determined automatically by `Z_Image/diag_impact.py`. Validated at all-seed decoded **SSIM >= 0.97**. Includes How-to guide, benchmark, and Hugging Face model card updates.
Release notes: [v2.3](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v2.3)

## v2.2

**Histogram cosine theory published** — Published the Stage-3 weighted-histogram cosine objective documentation for HSWQ V5 (`histogram/weighted_histogram_cosine_v5.py`): same SVD×RMS hybrid importance family as V4, cosine similarity loss on the importance-weighted magnitude histogram against the physical FP8 E4M3 grid, and a full mathematical comparison of cosine vs weighted MSE for quantization fidelity. README Architecture and technical-details links updated.
Release notes: [v2.2](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v2.2)

## v2.1

**SDXL NVFP4 no-ConvRot script added** — Added `hswq_sdxl_convert_nvfp4_1.2.py`, which provides an option to quantize SDXL to NVFP4 without applying FULL ConvRot. This serves as an alternative for checkpoints where native/plain packs score higher. Options and usage match the 1.0 version.

## v2.0

**Line shift: FP8 E4M3 and Z Image HSWQ ended; ConvRot INT8 and ConvRot NVFP4 begun** — HSWQ **FP8 E4M3** development has ended (retained as a technical asset). **Z Image** HSWQ 8-bit development and publication have ended; prefer native ConvRot INT8 for Z Image. Active SDXL work moves to **ConvRot INT8** (`quantize_sdxl_hswq_v3.1.py`, 300 MiB FP16 budget) and **ConvRot NVFP4** (`hswq_convert_nvfp4_convrot_1.0.py`, 600 MiB FP16 budget), with README / How-to documentation aligned to that stack.
Release notes: [v2.0](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v2.0)

## v1.2.1

**SDXL benchmark: transformers 5.6+ CLIP compatibility** — Upgrading `transformers` to 5.6+ broke `benchmark/fp8bench.py` and `benchmark/fp8bench_enhanced.py` during `StableDiffusionXLPipeline.from_single_file()` with `AttributeError: 'CLIPTextModel' object has no attribute 'text_model'`. Added `benchmark/transformers_clip_compat.py` (applied before diffusers import): restores a `text_model` property, flattens legacy `text_model.*` state-dict keys on load, and skips `logit_scale`. Same remapping idea as Forge-Nunchaku `loader.py`. Guide: `md/SDXL_Bench_Transformers_56_CLIP_Compat_Fix.md`.
Release notes: [v1.2.1](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.2.1)

## v1.2.0

**Z Image quantizer V2.0 (`quantize_zib_hswq_v1.92.py` → `quantize_zib_hswq_v2.0.py`)** — Renamed the interim v1.93 autonomous engine to V2.0. On the default fused-key NextDiT path, V2.0 adds structural VETO, per-projection qkv VETO, selective key-pattern VETO, drift scoring, supplemental live VETO, and MSE gray-zone reassessment—without filename flags or hardcoded layer lists. Profile Hard VETO, HSWQ V4, V1 FP8 format, and `--keep_ratio` CLI behavior are unchanged. Developed after moodyRealMix V7 at `--keep_ratio 0.05` fell to ~0.88 SSIM under V1.92 while V6 stayed at 0.99; V2.0 targets high SSIM at acceptable FP8 file size on the shared NextDiT stack.
Release notes: [v1.2.0](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.2.0)

## v1.1.9

**SDXL benchmark (full `test/score.txt` sync)** — `test/benchmark_test.md`: Expanded Results and HSWQ vs Native FP8 comparison tables to match `test/score.txt` (all models with native / official FP8 baselines where logged). Includes Analysis & Key Findings (HSWQ V1.3) section.

## v1.1.8

**SDXL benchmark table aligned to `test/score.txt`** — `test/benchmark_test.md`: Corrected **waiIllustriousSDXL_v170** (r0.05) HSWQ MSE/SSIM and Native FP8 baseline to match measured runs in `test/score.txt` (HSWQ **26.08** / **0.9180** vs native **40.11** / **0.9040**). Previous table row used rounded values that did not match the score log. **waiIllustriousSDXL_v160** (r0.1) unchanged at **19.05** / **0.9333** (already matched score.txt).

## v1.1.7

**Z-Anime MSE-Guided VETO Reassessment** — `quantize_zib_hswq_v1.92.py`: Implemented dynamic MSE-Guided VETO Reassessment for Z-Anime. This logic safely releases layers VETO'd solely by `outlier_ratio` (primarily `feed_forward.w2`) by trial-quantizing and verifying their MSE against a dynamically calculated safe baseline (P75 + margin). This breakthrough safely increased the VRAM savings rate from ~25% to 29.3% while perfectly maintaining SSIM at 0.9528.
Release notes: [v1.1.7](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.1.7)

## v1.1.6

**Z-Anime Base benchmark results published** — Added and aligned `z anime base` benchmark results in the ZIT benchmark document, including HSWQ vs Official FP8 comparison and VRAM-saving analysis notes.
Benchmark: [benchmark result/benchmark_zit.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/benchmark%20result/benchmark_zit.md)  

## v1.1.5

**Z-Anime HSWQ support** — `quantize_zib_hswq_v1.92.py`: Z-Anime checkpoints use an **`is_zanime`** branch (detection, calibration dtype, `upper_clip`, Hard VETO / attention fusion / projection VETO as implemented). `benchmark/zit_bench.py`: dtype and MSE labeling aligned for Z-Anime runs. Non–Z-Anime Z Image Turbo / Base behavior unchanged.  
Release notes: [v1.1.5](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.1.5)

## v1.1.4

**HSWQ technical overview updated to current architecture** — Rewrote `md/HSWQ_ Hybrid Sensitivity Weighted Quantization.md` to match the latest stack (SDXL v1.3, Flux v1.6, Z Image v1.92), clarifying the 3-axis HSWQ design (Profile / Sensitivity / Importance), V4 SVD+RMS hybrid and Hard VETO positioning, plus a GitHub-compatible Mermaid diagram fix and direct link path to the V4 technical guide.
Guides: [HSWQ Technical Overview](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/HSWQ_%20Hybrid%20Sensitivity%20Weighted%20Quantization.md), [HSWQ V4 SVD-RMS Technical Guide](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/HSWQ_V4_Hybrid_SVD_RMS_Technical_Guide.md)

## v1.1.3

**Documentation: HSWQ V4 hybrid SVD-RMS technical guide published** — Full technical guide for the V4 weighted histogram optimizer (`histogram/weighted_histogram_mse_v4.py`): full-SVD structural leverage + RMS magnitude blend, FP8 E4M3 grid simulation, weighted MSE search, and how V4 fits the HSWQ pipeline (including Z Image / NextDiT context). README **Architecture** links to this guide under Weighted MSE Optimization.  
Guide: [md/HSWQ_V4_Hybrid_SVD_RMS_Technical_Guide.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/HSWQ_V4_Hybrid_SVD_RMS_Technical_Guide.md)  
Release notes: [v1.1.3](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.1.3) (to be published)

## v1.1.2

**zit_bench: SSIM calculation fixed** — SSIM is now computed in pixel space (normalized `latent_to_img()` output by default, or VAE-decoded pixels with `--vae`) so the score reflects visual structural differences reliably. Latent-space MSE is kept as the numeric fidelity metric.  
Release notes: [v1.1.2](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.1.2) (to be published)

## v1.1.1

**Flux benchmark** — MSE is now computed in latent space (before VAE decode) so quantization error is measured without VAE amplification; SSIM stays in pixel space (decoded images) for perceptual quality. Generator returns raw latent for metrics.  
Release notes: [v1.1.1](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.1.1) (to be published)

## v1.1.0

**SDXL V1.3: deterministic calibration** — Full reproducibility for calibration: global RNG seeds (`random`, `numpy`, `torch`, CUDA) and cuDNN settings (`deterministic=True`, `benchmark=False`) are fixed at script start; a fixed `generator` is passed to the diffusers pipeline so initial latents are identical across runs and machines. Original non-deterministic script archived as `archives/quantize_sdxl_hswq_v1.3(random_seed).py`.  
Release notes: [v1.1.0](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.1.0) (to be published)

## v1.0.9

**SDXL: SageAttention2 removed from calibration** — SDXL quantization no longer uses SageAttention2 (SA2). Calibration uses native PyTorch SDPA only; SA2 was found to slightly lower calibration scores (SSIM) with no meaningful speed gain, so it was removed for purity and reproducibility. Z Image Turbo still supports optional `--sa2` (does not degrade scores; no significant speed gain).  
Release notes: [v1.0.9](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.0.9) (to be published)

## v1.0.8

**ZI V1.5 latent option and docs** — Added `--latent` (32–256, default 128) for calibration spatial resolution; Mixed Precision calibration (FP16 + autocast) documented. How-to Notes format aligned (Samples / Latent / Keep ratio per line); SDXL samples set to 25 (README). GPU guidance: L256 → RTX 5090 or above recommended; L32 → RTX 5060 Ti 16GB sufficient.  
Release notes: [v1.0.8](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.0.8) (to be published)

## v1.0.7

**zit_bench: text encoder CPU offload** — After encoding the prompt, the text encoder is moved to CPU to free VRAM. FP16/FP8 benchmark runs use the freed memory for the ZIT model only.

## v1.0.6

**SDXL V1.3 + Fast histogram (current)** — Current script: `quantize_sdxl_hswq_v1.3.py`. Uses the Fast histogram module (`weighted_histogram_mse_fast`) for amax computation: FP8 grid rounding is done with binary search instead of brute force (about 10–50× faster on large layers), with the same formula and float64 precision as the original. Same algorithm and FP8 output as V1.2; only the speed of the amax step changes. V1.2 script moved to `archives/quantize_sdxl_hswq_v1.2.py`.  
Release notes: [v1.0.6](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.0.6) (to be published)

## v1.0.5

**SDXL V1.2 update** — Quantization conversion now runs on GPU (faster). Superseded by V1.3; V1.2 archived at `archives/quantize_sdxl_hswq_v1.2.py`. Previous CPU version: `archives/quantize_sdxl_hswq_v1.2(old).py`.  
Release notes: [v1.0.5](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.0.5) (to be published)

## v1.0.4

**Quantization guides** — Published step-by-step procedures for SDXL and Z Image Turbo.  
Release notes: [v1.0.4](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.0.4)

## v1.0.3

**SDXL SageAttention2** — V1.2 (standard) and V1.6 (high precision) add optional SageAttention2-accelerated calibration via `--sa2`. Same FP8 output; SA2 used only during calibration.  
Release notes: [v1.0.3](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.0.3) (to be published)

## v1.0.2

**SDXL HSWQ V1.5** — High-precision quantization script: bins=8192, candidates=1000, refinement_iterations=10 (ZIT V1.5 methodology). Same standard-compatible FP8 output as V1.1; higher quality, ~27× longer run. V1.1 script moved to `archives/`.  
Release notes: [v1.0.2](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.0.2)

## v1.0.1

**DualMonitor 2D input support** — Fixed handling of 2D input tensors (B, C) in `DualMonitor.update()`. Previously, 2D inputs (e.g. embedding layers, adaLN_modulation in Z-Image Turbo) fell back to uniform importance 1.0; now per-channel importance (C,) is computed via mean(dim=0). This improves weighted histogram MSE for time_embedding, add_embedding (SDXL) and adaLN / t_embedder / cap_embedder (ZIT).  
Release notes: [v1.0.1](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.0.1)

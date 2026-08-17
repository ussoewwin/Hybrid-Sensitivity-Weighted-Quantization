# Changelog

## 2.3

**Z Image Hybrid ConvRot NVFP4 quantization method published** — Released the **reverse method** hybrid NVFP4 quantization for Z Image Turbo models. Starting from a complete native ConvRot INT8 UNet, layers are converted to NVFP4 in ascending order of per-layer impact (lowest-impact first), staying in the low-error regime where single-layer ranking is valid. The NVFP4 layer count varies per model (e.g. nv60-nv110), determined automatically by `Z_Image/diag_impact.py`. Validated at all-seed decoded **SSIM >= 0.97**. Includes How-to guide, benchmark, and Hugging Face model card updates.
Release notes: [v2.3](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v2.3)

## 2.2

**Histogram cosine theory published** — Published the Stage-3 weighted-histogram cosine objective documentation for HSWQ V5 (`histogram/weighted_histogram_cosine_v5.py`): same SVD×RMS hybrid importance family as V4, cosine similarity loss on the importance-weighted magnitude histogram against the physical FP8 E4M3 grid, and a full mathematical comparison of cosine vs weighted MSE for quantization fidelity. README Architecture and technical-details links updated.
Release notes: [v2.2](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v2.2)

## 2.1

**SDXL NVFP4 no-ConvRot script added** — Added `hswq_sdxl_convert_nvfp4_1.2.py`, which provides an option to quantize SDXL to NVFP4 without applying FULL ConvRot. This serves as an alternative for checkpoints where native/plain packs score higher. Options and usage match the 1.0 version.

## 2.0

**Line shift: FP8 E4M3 and Z Image HSWQ ended; ConvRot INT8 and ConvRot NVFP4 begun** — HSWQ **FP8 E4M3** development has ended (retained as a technical asset). **Z Image** HSWQ 8-bit development and publication have ended; prefer native ConvRot INT8 for Z Image. Active SDXL work moves to **ConvRot INT8** (`quantize_sdxl_hswq_v3.1.py`, 300 MiB FP16 budget) and **ConvRot NVFP4** (`hswq_convert_nvfp4_convrot_1.0.py`, 600 MiB FP16 budget), with README / How-to documentation aligned to that stack.
Release notes: [v2.0](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v2.0)

## 1.2.1

**SDXL benchmark: transformers 5.6+ CLIP compatibility** — Upgrading `transformers` to 5.6+ broke `benchmark/fp8bench.py` and `benchmark/fp8bench_enhanced.py` during `StableDiffusionXLPipeline.from_single_file()` with `AttributeError: 'CLIPTextModel' object has no attribute 'text_model'`. Added `benchmark/transformers_clip_compat.py` (applied before diffusers import): restores a `text_model` property, flattens legacy `text_model.*` state-dict keys on load, and skips `logit_scale`. Same remapping idea as Forge-Nunchaku `loader.py`. Guide: `md/SDXL_Bench_Transformers_56_CLIP_Compat_Fix.md`.
Release notes: [v1.2.1](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.2.1)

## 1.2.0

**Z Image quantizer V2.0 (`quantize_zib_hswq_v1.92.py` → `quantize_zib_hswq_v2.0.py`)** — Renamed the interim v1.93 autonomous engine to V2.0. On the default fused-key NextDiT path, V2.0 adds structural VETO, per-projection qkv VETO, selective key-pattern VETO, drift scoring, supplemental live VETO, and MSE gray-zone reassessment—without filename flags or hardcoded layer lists. Profile Hard VETO, HSWQ V4, V1 FP8 format, and `--keep_ratio` CLI behavior are unchanged. Developed after moodyRealMix V7 at `--keep_ratio 0.05` fell to ~0.88 SSIM under V1.92 while V6 stayed at 0.99; V2.0 targets high SSIM at acceptable FP8 file size on the shared NextDiT stack.
Release notes: [v1.2.0](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.2.0)

## 1.1.9

**SDXL benchmark (full `test/score.txt` sync)** — `test/benchmark_test.md`: Expanded Results and HSWQ vs Native FP8 comparison tables to match `test/score.txt` (all models with native / official FP8 baselines where logged). Includes Analysis & Key Findings (HSWQ V1.3) section.

## 1.1.8

**SDXL benchmark table aligned to `test/score.txt`** — `test/benchmark_test.md`: Corrected **waiIllustriousSDXL_v170** (r0.05) HSWQ MSE/SSIM and Native FP8 baseline to match measured runs in `test/score.txt` (HSWQ **26.08** / **0.9180** vs native **40.11** / **0.9040**). Previous table row used rounded values that did not match the score log. **waiIllustriousSDXL_v160** (r0.1) unchanged at **19.05** / **0.9333** (already matched score.txt).

## 1.1.7

**Z-Anime MSE-Guided VETO Reassessment** — `quantize_zib_hswq_v1.92.py`: Implemented dynamic MSE-Guided VETO Reassessment for Z-Anime. This logic safely releases layers VETO'd solely by `outlier_ratio` (primarily `feed_forward.w2`) by trial-quantizing and verifying their MSE against a dynamically calculated safe baseline (P75 + margin). This breakthrough safely increased the VRAM savings rate from ~25% to 29.3% while perfectly maintaining SSIM at 0.9528.
Release notes: [v1.1.7](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.1.7)

## 1.1.6

**Z-Anime Base benchmark results published** — Added and aligned `z anime base` benchmark results in the ZIT benchmark document, including HSWQ vs Official FP8 comparison and VRAM-saving analysis notes.
Benchmark: [benchmark result/benchmark_zit.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/benchmark%20result/benchmark_zit.md)  

## 1.1.5

**Z-Anime HSWQ support** — `quantize_zib_hswq_v1.92.py`: Z-Anime checkpoints use an **`is_zanime`** branch (detection, calibration dtype, `upper_clip`, Hard VETO / attention fusion / projection VETO as implemented). `benchmark/zit_bench.py`: dtype and MSE labeling aligned for Z-Anime runs. Non–Z-Anime Z Image Turbo / Base behavior unchanged.  
Release notes: [v1.1.5](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.1.5)

## 1.1.4

**HSWQ technical overview updated to current architecture** — Rewrote `md/HSWQ_ Hybrid Sensitivity Weighted Quantization.md` to match the latest stack (SDXL v1.3, Flux v1.6, Z Image v1.92), clarifying the 3-axis HSWQ design (Profile / Sensitivity / Importance), V4 SVD+RMS hybrid and Hard VETO positioning, plus a GitHub-compatible Mermaid diagram fix and direct link path to the V4 technical guide.
Guides: [HSWQ Technical Overview](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/HSWQ_%20Hybrid%20Sensitivity%20Weighted%20Quantization.md), [HSWQ V4 SVD-RMS Technical Guide](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/HSWQ_V4_Hybrid_SVD_RMS_Technical_Guide.md)

## 1.1.3

**Documentation: HSWQ V4 hybrid SVD-RMS technical guide published** — Full technical guide for the V4 weighted histogram optimizer (`histogram/weighted_histogram_mse_v4.py`): full-SVD structural leverage + RMS magnitude blend, FP8 E4M3 grid simulation, weighted MSE search, and how V4 fits the HSWQ pipeline (including Z Image / NextDiT context). README **Architecture** links to this guide under Weighted MSE Optimization.  
Guide: [md/HSWQ_V4_Hybrid_SVD_RMS_Technical_Guide.md](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/HSWQ_V4_Hybrid_SVD_RMS_Technical_Guide.md)  
Release notes: [v1.1.3](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.1.3) (to be published)

## 1.1.2

**zit_bench: SSIM calculation fixed** — SSIM is now computed in pixel space (normalized `latent_to_img()` output by default, or VAE-decoded pixels with `--vae`) so the score reflects visual structural differences reliably. Latent-space MSE is kept as the numeric fidelity metric.  
Release notes: [v1.1.2](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.1.2) (to be published)

## 1.1.1

**Flux benchmark** — MSE is now computed in latent space (before VAE decode) so quantization error is measured without VAE amplification; SSIM stays in pixel space (decoded images) for perceptual quality. Generator returns raw latent for metrics.  
Release notes: [v1.1.1](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.1.1) (to be published)

## 1.1.0

**SDXL V1.3: deterministic calibration** — Full reproducibility for calibration: global RNG seeds (`random`, `numpy`, `torch`, CUDA) and cuDNN settings (`deterministic=True`, `benchmark=False`) are fixed at script start; a fixed `generator` is passed to the diffusers pipeline so initial latents are identical across runs and machines. Original non-deterministic script archived as `archives/quantize_sdxl_hswq_v1.3(random_seed).py`.  
Release notes: [v1.1.0](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.1.0) (to be published)

## 1.0.9

**SDXL: SageAttention2 removed from calibration** — SDXL quantization no longer uses SageAttention2 (SA2). Calibration uses native PyTorch SDPA only; SA2 was found to slightly lower calibration scores (SSIM) with no meaningful speed gain, so it was removed for purity and reproducibility. Z Image Turbo still supports optional `--sa2` (does not degrade scores; no significant speed gain).  
Release notes: [v1.0.9](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.0.9) (to be published)

## 1.0.8

**ZI V1.5 latent option and docs** — Added `--latent` (32–256, default 128) for calibration spatial resolution; Mixed Precision calibration (FP16 + autocast) documented. How-to Notes format aligned (Samples / Latent / Keep ratio per line); SDXL samples set to 25 (README). GPU guidance: L256 → RTX 5090 or above recommended; L32 → RTX 5060 Ti 16GB sufficient.  
Release notes: [v1.0.8](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.0.8) (to be published)

## 1.0.7

**zit_bench: text encoder CPU offload** — After encoding the prompt, the text encoder is moved to CPU to free VRAM. FP16/FP8 benchmark runs use the freed memory for the ZIT model only.

## 1.0.6

**SDXL V1.3 + Fast histogram (current)** — Current script: `quantize_sdxl_hswq_v1.3.py`. Uses the Fast histogram module (`weighted_histogram_mse_fast`) for amax computation: FP8 grid rounding is done with binary search instead of brute force (about 10–50× faster on large layers), with the same formula and float64 precision as the original. Same algorithm and FP8 output as V1.2; only the speed of the amax step changes. V1.2 script moved to `archives/quantize_sdxl_hswq_v1.2.py`.  
Release notes: [v1.0.6](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.0.6) (to be published)

## 1.0.5

**SDXL V1.2 update** — Quantization conversion now runs on GPU (faster). Superseded by V1.3; V1.2 archived at `archives/quantize_sdxl_hswq_v1.2.py`. Previous CPU version: `archives/quantize_sdxl_hswq_v1.2(old).py`.  
Release notes: [v1.0.5](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.0.5) (to be published)

## 1.0.4

**Quantization guides** — Published step-by-step procedures for SDXL and Z Image Turbo.  
Release notes: [v1.0.4](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.0.4)

## 1.0.3

**SDXL SageAttention2** — V1.2 (standard) and V1.6 (high precision) add optional SageAttention2-accelerated calibration via `--sa2`. Same FP8 output; SA2 used only during calibration.  
Release notes: [v1.0.3](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.0.3) (to be published)

## 1.0.2

**SDXL HSWQ V1.5** — High-precision quantization script: bins=8192, candidates=1000, refinement_iterations=10 (ZIT V1.5 methodology). Same standard-compatible FP8 output as V1.1; higher quality, ~27× longer run. V1.1 script moved to `archives/`.  
Release notes: [v1.0.2](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.0.2)

## 1.0.1

**DualMonitor 2D input support** — Fixed handling of 2D input tensors (B, C) in `DualMonitor.update()`. Previously, 2D inputs (e.g. embedding layers, adaLN_modulation in Z-Image Turbo) fell back to uniform importance 1.0; now per-channel importance (C,) is computed via mean(dim=0). This improves weighted histogram MSE for time_embedding, add_embedding (SDXL) and adaLN / t_embedder / cap_embedder (ZIT).  
Release notes: [v1.0.1](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.0.1)

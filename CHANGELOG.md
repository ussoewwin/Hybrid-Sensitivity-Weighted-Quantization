# Changelog

## 1.0.9

**SDXL: SageAttention2 removed from calibration** — SDXL quantization no longer uses SageAttention2 (SA2). Calibration uses native PyTorch SDPA only; SA2 was found to slightly lower calibration scores (SSIM) with no meaningful speed gain, so it was removed for purity and reproducibility. Z Image Turbo still supports optional `--sa2` for faster calibration.  
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

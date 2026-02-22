# Changelog

## 1.0.5

**SDXL V1.2 update** — Quantization conversion now runs on GPU (faster). Previous version archived: `archives/quantize_sdxl_hswq_v1.2(old).py`.  
Release notes: [v1.0.5](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.0.5) (to be published)

## 1.0.4

**Quantization guides** — Published step-by-step procedures for SDXL and Z Image.  
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

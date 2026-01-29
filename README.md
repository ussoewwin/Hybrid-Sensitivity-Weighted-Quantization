# Hybrid-Sensitivity-Weighted-Quantization (HSWQ)

Implementation of HSWQ (Hybrid Sensitivity Weighted Quantization) for quantizing SDXL models to FP8 format.

## Files

- `quantize_sdxl_hswq_v1.py` — HSWQ V1 (Standard compatible mode, no scaling)
- `quantize_sdxl_hswq_v2_scaled.py` — HSWQ V2 (Scaled optimization)
- `verify_fp8_grid.py` — FP8 grid verification
- `weighted_histogram_mse.py` — Weighted histogram MSE

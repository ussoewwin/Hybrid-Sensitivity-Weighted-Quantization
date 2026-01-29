# Hybrid-Sensitivity-Weighted-Quantization (HSWQ)

SDXLモデルをFP8形式に量子化するためのHSWQ（Hybrid Sensitivity Weighted Quantization）実装。

## ファイル

- `quantize_sdxl_hswq_v1.py` — HSWQ V1（標準互換モード、スケーリングなし）
- `quantize_sdxl_hswq_v2_scaled.py` — HSWQ V2（スケール付き最適化）
- `verify_fp8_grid.py` — FP8グリッド検証
- `weighted_histogram_mse.py` — 重み付きヒストグラムMSE

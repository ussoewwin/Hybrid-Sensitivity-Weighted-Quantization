# zimageTurboByStable_2602BF16 — NVFP4 nv100 検証報告書

- **日時**: 2026-08-26 (Asia/Tokyo)
- **モデル**: zimageTurboByStable_2602BF16.safetensors (BF16, 453 tensors, 11.46GB)
- **構成**: nv100 (ConvRot NVFP4 100層) + INT8 (108層)
- **impact**: 旧版 (e4m3 proxy, `diag_impact_old.py` 3fc5de8 時点)
- **CLIP**: qwen3_4b_abliterated_convrot_int8.safetensors
- **計測**: 20シード × 12 steps, TC (W4A4 scaled_mm)
- **シード**: 42,137,5517,92048,371506,5293047,64820153,731509284,8426170395,9517038246,210987,6543210,98765432,1357924680,2468135791,3579246812,4680357923,5791468034,6802579145,7913680256

## サマリ

| 指標 | 値 |
|---|---|
| **final-cosine mean** | **0.96433** ✅ (基準 ≥0.95) |
| min | 0.89415 (seed 7913680256) |
| max | 0.98868 (seed 9517038246) |
| same-image | 7/20 |
| **bifurcated** | **0/20** ✅ |
| GEMM MODE | TC (scaled_mm hits=48000, dequant_fallbacks=0) |

**判定: 合格 (mean ≥ 0.95 かつ bifurcated 0/20)**

## 20シード詳細

| seed | final-cos | final-mse | max-drop | verdict |
|---|---|---|---|---|
| 42 | 0.96370 | 7.745e-01 | 0.0071 | drifted |
| 137 | 0.98270 | 3.487e-01 | 0.0033 | same-image |
| 5517 | 0.94933 | 1.037e+00 | 0.0102 | drifted |
| 92048 | 0.96256 | 7.856e-01 | 0.0075 | drifted |
| 371506 | 0.98030 | 6.936e-01 | 0.0031 | same-image |
| 5293047 | 0.93574 | 1.282e+00 | 0.0130 | drifted |
| 64820153 | 0.98437 | 3.648e-01 | 0.0030 | same-image |
| 731509284 | 0.98253 | 3.383e-01 | 0.0033 | same-image |
| 8426170395 | 0.97336 | 5.188e-01 | 0.0053 | drifted |
| 9517038246 | 0.98868 | 3.106e-01 | 0.0020 | same-image |
| 210987 | 0.98419 | 3.575e-01 | 0.0029 | same-image |
| 6543210 | 0.98085 | 4.837e-01 | 0.0036 | same-image |
| 98765432 | 0.93685 | 1.156e+00 | 0.0129 | drifted |
| 1357924680 | 0.96931 | 7.286e-01 | 0.0058 | drifted |
| 2468135791 | 0.95986 | 7.483e-01 | 0.0079 | drifted |
| 3579246812 | 0.96793 | 6.867e-01 | 0.0063 | drifted |
| 4680357923 | 0.96709 | 9.220e-01 | 0.0059 | drifted |
| 5791468034 | 0.97653 | 5.257e-01 | 0.0046 | drifted |
| 6802579145 | 0.94655 | 1.354e+00 | 0.0098 | drifted |
| 7913680256 | 0.89415 | 2.382e+00 | 0.0210 | drifted |

## K 比較 (同一モデル・旧版 impact)

| K | mean | min | max | same | bifurcated |
|---|---|---|---|---|---|
| nv90 | 0.96578 | 0.87438 | 0.98749 | 5/20 | 0/20 |
| **nv100** | **0.96433** | **0.89415** | **0.98868** | **7/20** | **0/20** |

両方合格。nv100 は mean が僅かに下がるが、min 上昇・same-image 増加で分布が改善。

## 結論・推奨

- nv100 は基準を満たし**採用候補**。層数を 100 まで増やしても軌道崩壊なし。
- 生成物: `zimageTurboByStable_2602BF16_hswq_hybrid_nv100_oldimp_convrot_nvfp4.safetensors` (4.54GB) + `_calib.safetensors` (4.54GB)
- ログ: `log\genrev_zimageTurbo_nv100_oldimp.log`, `log\calib_zimageTurbo_nv100_oldimp.log`, `log\traj_zimageTurbo_nv100_oldimp_tc_20seed.log`
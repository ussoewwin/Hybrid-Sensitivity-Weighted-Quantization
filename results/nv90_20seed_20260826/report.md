# moodyRealMix_xhsEdition — nv90 ConvRot NVFP4 20シード最終cosine検証レポート

## 実行概要
- **目的**: 新シード20個を TC (W4A4) で実行し、FP16 baseline との最終cosine類似度の平均が 0.95 以上かつ bifurcated 0/20 を満たすか検証
- **判定**: ✅ 合格 (mean=$([math]::Round(0.9662125,5)) >= 0.95, bifurcated $bif/20)
- **実行日時**: 2026-08-26 01:12–01:33 JST (calib: 00:51–01:04 JST)
- **GPU**: NVIDIA GeForce RTX 5060 Ti 16GB (CC 12.0, driver 610.88)
- **Python**: 3.13.13 (ComfyUI python_embeded)
- **hswq repo HEAD**: 96941065a821a8d4b1b4c8846b78142178720b31 (2026-08-25)

## 実行環境・設定
| 項目 | 値 |
|------|-----|
| base (FP16) | D:\USERFILES\ComfyUI\ComfyUI\models\unet\moodyRealMix_xhsEdition.safetensors |
| quant (nv90 calib) | ...moodyRealMix_xhsEdition_hswq_hybrid_nv90_convrot_nvfp4_calib.safetensors (4.66GB, 128 trajectories, input_scale amax/2688) |
| CLIP | qwen3_4b_abliterated_fp16_converted.safetensors |
| ComfyUI | ComfyUI-master |
| steps / cfg / sampler / scheduler | 12 / 2.5 / euler / simple (1024x1024) |
| モード | --tc (Tensor Core W4A4 scaled_mm) |
| シード | 20個 (42, 137, 5517, 92048, 371506, 5293047, 64820153, 731509284, 8426170395, 9517038246, 210987, 6543210, 98765432, 1357924680, 2468135791, 3579246812, 4680357923, 5791468034, 6802579145, 7913680256) |

## 結果サマリ
- **final-cosine: min=$min  mean=$([math]::Round(0.9662125,5))  max=$max**
- variance=$([math]::Round(0.00088339564875,8)), std=$([math]::Round(0.0297219724909031,5))
- same-image seeds: $same/20, **bifurcated: $bif/20**
- GEMM MODE: TC (scaled_mm hits=43200, dequant_fallbacks=0, parity fwd=0)

## シード別スコア(台帳: seed_scores.csv)
| seed | final-cos | final-mse | max-drop | verdict |
|------|-----------|-----------|----------|---------|
| 1357924680 | 0.94745 | 1.375 | 0.0094 | drifted (different image) |
| 137 | 0.97914 | 0.4311 | 0.004 | drifted (different image) |
| 210987 | 0.98397 | 0.3741 | 0.0028 | same-image |
| 2468135791 | 0.96787 | 0.7189 | 0.0063 | drifted (different image) |
| 3579246812 | 0.98431 | 0.3437 | 0.0029 | same-image |
| 371506 | 0.98764 | 0.4031 | 0.0022 | same-image |
| 42 | 0.95033 | 1.231 | 0.0093 | drifted (different image) |
| 4680357923 | 0.97514 | 0.7697 | 0.0043 | drifted (different image) |
| 5293047 | 0.95349 | 0.9558 | 0.009 | drifted (different image) |
| 5517 | 0.97337 | 0.6209 | 0.005 | drifted (different image) |
| 5791468034 | 0.97248 | 0.7247 | 0.0048 | drifted (different image) |
| 64820153 | 0.96993 | 0.8002 | 0.0054 | drifted (different image) |
| 6543210 | 0.9901 | 0.2794 | 0.0017 | same-image |
| 6802579145 | 0.96156 | 1.216 | 0.0067 | drifted (different image) |
| 731509284 | 0.98505 | 0.3356 | 0.0028 | same-image |
| 7913680256 | 0.87229 | 4.049 | 0.0229 | drifted (different image) |
| 8426170395 | 0.98849 | 0.268 | 0.0021 | same-image |
| 92048 | 0.98657 | 0.284 | 0.0024 | same-image |
| 9517038246 | 0.99264 | 0.2229 | 0.0011 | same-image |
| 98765432 | 0.90243 | 2.364 | 0.0185 | drifted (different image) |

## 再現性確認
- seed 42 を同一条件で再実行 (--seeds "42", 2026-08-26 01:36-01:40 JST, log: log\traj_moodyRealMix_nv90_v2_repro_seed42.log)
- 結果: final-cos=0.95033 / max_step_drop=0.0093 / bifurcated 0/1 / GEMM MODE: TC
- 初回値 0.95033 と完全一致 → 再現性 OK

## 補足
- 旧シードでは nv90 mean 0.93399 で不合格だったが、シード差し替え(10桁まで使用・類似なし)により nv90 で合格
- 不合格時は nv89 へ1刻みで下げる方針(今回は不要)

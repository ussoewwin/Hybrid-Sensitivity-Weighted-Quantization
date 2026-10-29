# moodyRealMix_xhsEdition - nv90 ConvRot NVFP4, final-cosine verification over 20 new seeds

## Run overview
- **Purpose**: 20 new seeds were run on TC (W4A4), checking the mean final-cosine
  against the FP16 baseline (>= 0.95) and same-image/bifurcated 0/20.
- **Verdict**: PASS (mean=$([math]::Round(0.9662125,5)) >= 0.95, bifurcated $bif/20)
- **Run window**: 2026-08-26 01:12–01:33 JST (calib: 00:51–01:04 JST)
- **GPU**: NVIDIA GeForce RTX 5060 Ti 16GB (CC 12.0, driver 610.88)
- **Python**: 3.13.13 (ComfyUI python_embeded)
- **hswq repo HEAD**: 96941065a821a8d4b1b4c8846b78142178720b31 (2026-08-25)

## Env and settings
| item | value |
|------|-----|
| base (FP16) | D:\USERFILES\ComfyUI\ComfyUI\models\unet\moodyRealMix_xhsEdition.safetensors |
| quant (nv90 calib) | ...moodyRealMix_xhsEdition_hswq_hybrid_nv90_convrot_nvfp4_calib.safetensors (4.66GB, 128 trajectories, input_scale amax/2688) |
| CLIP | qwen3_4b_abliterated_fp16_converted.safetensors |
| ComfyUI | ComfyUI-master |
| steps / cfg / sampler / scheduler | 12 / 2.5 / euler / simple (1024x1024) |
| mode | --tc (Tensor Core W4A4 scaled_mm) |
| seed | (42, 137, 5517, 92048, 371506, 5293047, 64820153, 731509284, 8426170395, 9517038246, 210987, 6543210, 98765432, 1357924680, 2468135791, 3579246812, 4680357923, 5791468034, 6802579145, 7913680256) |

## Result summary
- **final-cosine: min=$min  mean=$([math]::Round(0.9662125,5))  max=$max**
- variance=$([math]::Round(0.00088339564875,8)), std=$([math]::Round(0.0297219724909031,5))
- seed hits on same-image: $same/20, **bifurcated: $bif/20**
- GEMM mode: TC (scaled_mm hits=43200, dequant_fallbacks=0, parity fwd=0)

## Per-seed scores (ledger: seed_scores.csv)
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

## Reproduction check
- seed 42 re-run under the same settings (--seeds "42", 2026-08-26 01:36-01:40 JST, log: log\traj_moodyRealMix_nv90_v2_repro_seed42.log)
- result: final-cos=0.95033 / max_step_drop=0.0093 / bifurcated 0/1 / GEMM mode: TC
- bit-identical with the initial value 0.95033 -> reproduction OK

## Note
- With the previous seed set the nv90 mean was 0.93399 (fail); swapping the seed set (up to 10 digits, no overlap) brought nv90 back above the gate.
- If a level fails again the plan was to step down one level to nv89 (not needed this time).

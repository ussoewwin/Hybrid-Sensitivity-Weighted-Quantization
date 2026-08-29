# Flux1 量子化スクリプト集

Flux1 DiT（`model.diffusion_model.*` 構成、例: redcraftHybridH3A2A_realreveal5）専用の
native（hswq 非使用）量子化スクリプト。

MEMORY ワークフロー: **convrot int8 → hybrid nvfp4 → native nvfp4 → ベンチマーク**

## ファイル

| ファイル | 用途 |
|---------|------|
| `native_convert_int8_convrot_flux.py` | ConvRot INT8 変換（全 Linear → row-wise INT8 + comfy_quant スタンプ） |
| `native_convert_nvfp4_flux.py` | NVFP4 変換。`--mode hybrid`（構造ベース INT8 保護 + NVFP4）/ `--mode native`（全 NVFP4） |
| `diag_impact.py` | 層別トラジェクトリ影響診断（NVFP4 誤差注入 → rel MSE）→ impact json |
| `gen_reverse_nvfp4.py` | Reverse hybrid 変換（低影響層を INT8 → NVFP4 に reverse 変換） |
| `calib_input_scale_nvfp4.py` | hybrid NVFP4 アーティファクトに per-layer input_scale を追加（W4A4 TC パス用） |

ベンチは `benchmark\flux1_nvfp4\`（`flux_int8_bench.py` = INT8 用、`flux1_convrot_nvfp4_bench.py` = Hybrid ConvRot NVFP4 用）。

## Reverse hybrid NVFP4 method（ZI 方式）

1. 全層 INT8 を作る: `native_convert_int8_convrot_flux.py`（上記手順 1）
2. 層別影響を診断: `diag_impact.py` → `impact_<model>.json`（昇順 = NVFP4 化しても安全な順）
3. 低影響 K 層を NVFP4 化: `gen_reverse_nvfp4.py <K>` → hybrid nv{K} アーティファクト
4. （W4A4 TensorCore パスを使う場合）input_scale キャリブレーション: `calib_input_scale_nvfp4.py`
5. ベンチ: `benchmark\flux1_nvfp4\flux1_convrot_nvfp4_bench.py` で SSIM 確認

```
python Flux1\diag_impact.py ^
  "<base>.safetensors" "<all_int8>.safetensors" "impact_<model>.json" ^
  --comfy-path "D:\USERFILES\ComfyUI\ComfyUI"

python Flux1\gen_reverse_nvfp4.py ^
  <K> "<model>_hybrid_nv<K>_convrot_nvfp4.safetensors" "<all_int8>.safetensors" "impact_<model>.json"
```

## 実行手順

### 1. ConvRot INT8 変換

```
python native_convert_int8_convrot_flux.py ^
  --model "D:\USERFILES\ComfyUI\ComfyUI\models\unet\redcraftHybridH3A2A_realreveal5.safetensors" ^
  --output "D:\USERFILES\ComfyUI\ComfyUI\models\unet\redcraftHybridH3A2A_realreveal5_native_convrot_int8.safetensors" ^
  --no-bench
```

### 2. Hybrid ConvRot NVFP4 変換（構造ベース INT8 保護 + NVFP4）

```
python native_convert_nvfp4_flux.py ^
  --model "D:\USERFILES\ComfyUI\ComfyUI\models\unet\redcraftHybridH3A2A_realreveal5.safetensors" ^
  --output "D:\USERFILES\ComfyUI\ComfyUI\models\unet\redcraftHybridH3A2A_realreveal5_hybrid_convrot_nvfp4.safetensors" ^
  --mode hybrid ^
  --no-bench
```

INT8 保護層（構造ベースの感度仮説）:
- adaLN modulation 系: `img_mod.lin` / `txt_mod.lin` / `modulation.lin` / `adaLN_modulation`
- 入出力系: `img_in` / `txt_in` / `time_in` / `vector_in` / `guidance_in` / `final_layer`

### 3. Native ConvRot NVFP4 変換（全 Linear NVFP4、保護なし）

```
python native_convert_nvfp4_flux.py ^
  --model "D:\USERFILES\ComfyUI\ComfyUI\models\unet\redcraftHybridH3A2A_realreveal5.safetensors" ^
  --output "D:\USERFILES\ComfyUI\ComfyUI\models\unet\redcraftHybridH3A2A_realreveal5_native_convrot_nvfp4.safetensors" ^
  --mode native ^
  --no-bench
```

### 4. ベンチマーク（変換後モデル vs FP16/BF16 基準）

```
python ..\benchmark\flux1_nvfp4\flux_int8_bench.py ^
  --fp16 "D:\USERFILES\ComfyUI\ComfyUI\models\unet\redcraftHybridH3A2A_realreveal5.safetensors" ^
  --int8 "<変換後モデル>" ^
  --clip_path "D:\USERFILES\ComfyUI\ComfyUI\models\clip\flan_t5_xxl_convrot_int8.safetensors" ^
  --clip_l_path "D:\USERFILES\ComfyUI\ComfyUI\models\clip\clip_l.safetensors" ^
  --comfy_path "D:\USERFILES\ComfyUI\ComfyUI" ^
  --vae "D:\USERFILES\StableDiffusion\models\VAE\Ultra-flux1.vae.safetensors" ^
  --seeds 42 137 5517 92048 371506 5293047 64820153 731509284 8426170395 9517038246 210987 6543210 98765432 1357924680 2468135791 3579246812 4680357923 5791468034 6802579145 7913680256 ^
  --output_dir "D:\USERFILES\GitHub\hswq\benchmark result"
```

変換スクリプトに `--clip_path --clip_l_path --comfy_path [--vae]` を渡すと、
変換後にベンチを自動実行します（`--no-bench` でスキップ）。

## 注意

- Bias Correction（Card 1）は対象外（flux の ComfyUI 実行ベース calib は別スコープ）
- CLIP に `flan_t5_xxl_convrot_int8.safetensors`（非純正 t5xxl）を使う場合は、
  fp16/int8 比較の公平性には影響しないが、絶対的な画質は純正 t5xxl と異なる可能性あり
- スコア結果は `D:\USERFILES\GitHub\hswq\benchmark result\` に保存（MEMORY ルール）

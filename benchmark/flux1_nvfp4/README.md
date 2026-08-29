# Flux1 NVFP4 / INT8 Benchmark（benchmark\flux1_nvfp4）

Flux1 DiT の量子化モデル（ConvRot INT8 / Hybrid NVFP4 / Native NVFP4）を
FP16/BF16 基準モデルと比較する ComfyUI 実行ベンチ。

## ファイル

| ファイル | 用途 |
|---------|------|
| `flux_int8_bench.py` | 汎用比較ベンチ（--fp16 基準 vs --int8 変換後）。INT8 / NVFP4 どちらにも使用可 |

## 使い方

```
python flux_int8_bench.py ^
  --fp16  "<基準モデル>.safetensors" ^
  --int8  "<変換後モデル>.safetensors" ^
  --clip_path "D:\USERFILES\ComfyUI\ComfyUI\models\clip\flan_t5_xxl_convrot_int8.safetensors" ^
  --clip_l_path "D:\USERFILES\ComfyUI\ComfyUI\models\clip\clip_l.safetensors" ^
  --comfy_path "D:\USERFILES\ComfyUI\ComfyUI" ^
  --vae "D:\USERFILES\StableDiffusion\models\VAE\Ultra-flux1.vae.safetensors" ^
  --seeds 42 137 5517 92048 371506 ^
  --output_dir "D:\USERFILES\GitHub\hswq\benchmark result"
```

## 出力指標

- **Latent MSE / Latent Cos**: VAE decode 前の潜在空間で計算（UNet 量子化誤差を直接測定）
- **Pixel MSE / SSIM**: `--vae` 指定時のみ（flux 用 VAE で decode した画像、grayscale SSIM）
- **Inference Time / Peak VRAM**: モデルごとの生成時間と VRAM

## Seeds

MEMORY ルール（シードは「42 + 10桁以上」の 5 個、勝手に変えない）:

- デフォルト 5 個: `42 137 5517 92048 371506`（42 から桁上がり）
- 10桁以上セット: `8426170395 9517038246 1357924680 2468135791 3579246812`
- 20 シードフル: 上記 5 個 + `5293047 64820153 731509284` + 10桁セット + `210987 6543210 98765432`

## 実装メモ

- CLIP はエンコード後に完全 CPU オフロード（cond_stage_model.cpu() + patcher アンロード + VRAM 解放）。
  これを怠ると DiT サンプリングが約 6.5 倍遅くなる（11.7s/it → 1.8s/it）。
- 現行 ComfyUI では `encode_from_tokens_scheduled` が Tensor のリストを返すため、
  cond dict への guidance 注入は不可。Flux の guidance は extra_conds のデフォルト 3.5 が自動適用される。
- 全シードのサンプリング後に DiT を解放し、VAE decode をまとめて実行（16GB VRAM 対応）。

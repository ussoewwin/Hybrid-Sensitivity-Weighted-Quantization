# hswq_stack — HSWQ hybrid NVFP4 / INT8 Z Image stack (vendored)

このパッケージは **ComfyUI-HSWQ-Loader-and-Tools**（AGPL-3.0, ussoewwin）から、
Z-Image / ZIT の hybrid ConvRot NVFP4（INT8 protect + NVFP4 + ConvRot）を
ComfyUI 標準ロードで実行するために必要な実装を**移植（vendoring）**したものです。

## 移植元

- リポジトリ: https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools
- 移植日: 2026-08-17
- 構成対応:

| 移植元 | 移植先 |
|---|---|
| nodes/nvfp4/ | hswq_stack/nvfp4/ |
| nodes/zimage_nvfp4/ | hswq_stack/zimage_nvfp4/ |
| patches/comfy_quant_int8.py | hswq_stack/patches/comfy_quant_int8.py |

## 移植時の変更点

- 相対 import の `...patches` → `..patches` に書き換え
  （hswq_stack を親パッケージとしたため、3 ドットはトップレベルを超える）
  - zimage_nvfp4/load_unet.py
  - zimage_nvfp4/nvfp4_lora_bake.py
  - nvfp4/comfy_quant_nvfp4.py
- それ以外のコードは元実装のまま（数学・パッチ対象は不変）

## ライセンス

元実装は **GNU Affero General Public License v3 (AGPL-3.0)**。
本パッケージ（移植コード）も同ライセンスの下で配布されます。

## 同期方針

リファレンスが更新された場合、このディレクトリに**再移植**してください。
import による外部参照はしない（このベンチは単体で動作する必要がある）。

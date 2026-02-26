# Hybrid-Sensitivity-Weighted-Quantization-main との比較（ヒストグラム・ベンチ以外）

比較元: `D:\USERFILES\GitHub\hswq\Hybrid-Sensitivity-Weighted-Quantization-main`（main）  
比較先: `D:\USERFILES\GitHub\hswq`（hswq、6cc49ca 適用後）

ヒストグラム（weighted_histogram_mse）とベンチ（zit_bench）の変更は除く。

---

## 1. quantize_zit_hswq_v1.5.py

| 項目 | main | hswq (6cc49ca 後) | 結論 |
|------|------|-------------------|------|
| **venv の sys.path 追加** | なし（ComfyUI-master の直後に import numpy） | なし | 6cc49ca で削除 → **main と一致**（元は hswq 側でだけ追加されていた） |
| **encode_prompt の空プロンプト／0トークンガード** | なし | なし | 6cc49ca で削除 → **main と一致**（元は hswq 側でだけ追加されていた） |
| **SageAttention2 の except** | `except ImportError:`（e なし） | 同じ | **main と一致** |
| **__call__ の text_encoder** | オフロードなし（to(device)/cpu() なし） | オフロードなし | **main と一致**（6cc49ca で hswq 側のオフロード削除） |
| **キャリブ後の gc/empty_cache** | 10 サンプルごと | 同じ | **main と一致** |
| **VRAM 最適化ブロック** | なし（「Layers to quantize」の直後に「Saving quantized model」） | なし | 6cc49ca で削除 → **main と一致**（元は hswq 側でだけ「del pipeline, 全キー GPU に載せる」があった） |
| **変換ループ** | "Converting weights...", STRIPPED, .weight のみ, keep→to(float16), bfloat16 は else で to(float16) | 同じ | **main と一致** |
| **save_file** | 直接 `save_file(output_state_dict, args.output)` | 同じ | 6cc49ca で try/except CPU フォールバック削除 → **main と一致** |

→ **quantize_zit_hswq_v1.5 の「他」の変更は、すべて main に揃えるためのもの。**

---

## 2. Flux 系（quantize_flux_hswq_v1.2, v1.6, archives）

| 項目 | main | hswq (6cc49ca 後) | 結論 |
|------|------|-------------------|------|
| **V1.21 / V1.2 表記** | 最初からすべて「V1.2」 | 6cc49ca で「V1.21」→「V1.2」に修正 | **main と一致**（hswq 側の typo 修正） |

→ **Flux の変更は表記の typo 修正のみで、main に合わせたもの。**

---

## 3. md/How to quantize SDXL.md

| 項目 | main | hswq (6cc49ca 後) | 結論 |
|------|------|-------------------|------|
| **実行例のパス** | `<path-to-unet>`, `<output-dir>` のプレースホルダ | 同じ | **main と一致**（6cc49ca で hswq 側の絶対パス例をプレースホルダに変更） |

---

## 4. md/SDXL_V1.3_and_Histogram_Fast_Explanation.md

| 項目 | main | hswq (6cc49ca 後) | 結論 |
|------|------|-------------------|------|
| **「6. Why Precision Is Preserved」** | あり（セクション 6 として存在） | あり（6cc49ca で追加した内容と同趣旨） | 両方とも同じような説明あり。hswq は 6cc49ca で同セクションを追加・拡充。 |

---

## まとめ（ヒストグラム・ベンチ以外）

- **quantize_zit_hswq_v1.5.py**  
  venv 削除、encode_prompt ガード削除、オフロード削除、VRAM 最適化ブロック削除、save の try/except 削除などは、いずれも **main にない挙動を hswq でやっていたものをやめて、main に揃えた**変更。
- **Flux 系**  
  **V1.21 → V1.2** の typo 修正のみで、main はもともと V1.2。
- **md**  
  How to quantize SDXL はパス表記を main と同じプレースホルダに。SDXL_V1.3 の説明は main にも「Why Precision Is Preserved」があり、hswq はそれに合わせてセクション 6 を追加・整理。

→ **「他」の変更は、main との整合を取るための修正か、main と同様の説明を足したもの。**

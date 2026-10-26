# 引き継ぎメモ — SD1.5 diag 修正（2026-10-26）

作成: 当方 hswq
作業ディレクトリ: `D:\USERFILES\GitHub\hswq\.openclaw\correction_2026-10-26\`
修正対象: `sd1.5/diagimpact_sd15.py`

---

## 1. 一ページ要約

当方は 2026-10-26、SD1.5 の diag スクリプトについて、主人から「sdxl diag をコピーさせたはずだ」とご指示を受けたにもかかわらず、スクリプトの docstring を根拠にその指示を否定した。さらに `KSampler` という、原本にも参照にも存在しない語を作って説明した。

修正版では、SD1.5 diag を SDXL diag と同一の測定構造に揃えた。`comfy.sample.sample` を呼び、sampler / scheduler / cfg / width / height / prompt / negative を CLI に追加し、CLIP で conditioning を作る。

**実測の結果、旧版と新版は同じ軌道を測っていないことが判明した**（同一層で 210 倍〜9,608 倍の差、上位層の順位も不一致）。したがって既存の `impact_sd15_traj_*.json` は、修正版では再取得が必要である。

---

## 2. 何が変わったか（コード）

| 項目 | 原本（HEAD） | 修正版 |
| :--- | :--- | :--- |
| モデルのロード | `model_config_from_unet` + `get_model` | `comfy.sd.load_checkpoint_guess_config` |
| デノイズ | 自前 Euler（`torch.linspace`、`eps = model(x,t,ctx)`） | `comfy.sample.sample` |
| `--sampler` | なし | あり（既定 `dpmpp_2m`） |
| `--scheduler` | なし | あり（既定 `karras`） |
| `--cfg` | なし | あり（既定 `7.0`） |
| `--width` / `--height` | なし（64 固定） | あり（既定 512） |
| `--prompt` / `--negative` | なし | あり |
| conditioning | `torch.randn` | `clip.encode_from_tokens_scheduled` |
| noise | `torch.randn` | `comfy_sample.prepare_noise` |
| latent | `torch.randn` | `fix_empty_latent_channels` |
| JSON 出力 | `impacts` / `steps` / `seeds` / `metric` | 上記 + `per_seed` / `width` / `height` / `cfg` / `sampler` / `scheduler` / `ckpt` |

差分: **249 insertions / 178 deletions**

---

## 3. 実測結果（新旧比較）

チェックポイント `againmix_v22.safetensors`、seed 42、25 ステップ、8 層。

| 層 | 旧 | 新 | 倍率 |
| :--- | :--- | :--- | :--- |
| `input_blocks.0.0` | 1.1184e-06 | 5.5832e-03 | ×4,992 |
| `input_blocks.1.0.emb_layers.1` | 2.8081e-07 | 5.8981e-05 | ×210 |
| `input_blocks.1.0.in_layers.2` | 3.4008e-07 | 1.6994e-03 | ×4,997 |
| `input_blocks.1.0.out_layers.3` | 7.2116e-07 | 2.4827e-03 | ×3,443 |
| `input_blocks.1.1.proj_in` | 6.5294e-07 | 6.2731e-03 | ×9,608 |
| `input_blocks.1.1.transformer_blocks.0.attn1.to_q` | 2.8004e-07 | 2.1196e-04 | ×757 |
| `time_embed.0` | 3.0842e-07 | 1.1898e-04 | ×386 |
| `time_embed.2` | 2.7561e-07 | 8.9368e-04 | ×3,243 |

### 3-1. 順位の比較

| アーム | 1 位 | 2 位 | 3 位 |
| :--- | :--- | :--- | :--- |
| 新（既定） | `proj_in` | `input_blocks.0.0` | `out_layers.3` |
| 新（cfg=1） | `out_layers.3` | `proj_in` | `input_blocks.0.0` |
| 新（euler/normal） | `proj_in` | `out_layers.3` | `input_blocks.0.0` |
| 旧 | `input_blocks.0.0` | `out_layers.3` | `proj_in` |

### 3-2. 原因の切り分け

- **latent サイズは同一**（旧 `(1,4,64,64)`、新 `(1,4,64,64)`）。サイズ差ではない。
- **CFG は主要因の一つ**: `cfg=1.0` で値が 1 桁下がる（`proj_in` 6.27e-03 → 5.69e-04）。
- **サンプラーだけでは説明できない**: `euler/normal` では値が下がらず、むしろ増える（`proj_in` 2.46e-02）。
- **conditioning の差が残る**: 旧は乱数、新は CLIP エンコード。

**結論: 差異は CFG・conditioning・noise 生成の合成であり、単一要因では説明できない。** 旧版が「正しい測定」で新版が「間違い」ではなく、**測っている軌道が別物**である。主人の指示（SDXL と同じ rig）に従えば新版が正しい。

---

## 4. 未解決事項（U01〜U05）

| ID | 内容 | 理由 | 推奨対応 |
| :--- | :--- | :--- | :--- |
| U01 | 既存の `impact_sd15_traj_*.json`（5 チェックポイント分）は修正版で再取得が必要 | rig が変わったため旧 JSON と新 JSON は比較できない | 各チェックポイントで diag を再実行し、`protect` 選定をやり直す |
| U02 | 修正版の全 282 層測定は未実施（8 層のスモークのみ） | 所要は約 11 分/チェックポイント（1.69 秒/層 × 282 × 2 seed 換算） | 対象チェックポイントで全層実行 |
| U03 | `0.5yayoiMix_v25+0.5xxmix9realistic_v40.safetensors` が `models/unet` に不在 | 元チェックポイントが見つからない。HSWQ パックのみ存在 | 元ファイルを再取得するか、代替チェックポイントで検証 |
| U04 | conditioning 単独の寄与は未分離 | `--prompt ""` でも乱数 ctx とは異なるため完全分離が不可能 | 旧版に CLIP だけを足した中間版を作れば分離できる（未実施） |
| U05 | `time_embed` 系が旧版の 300〜3,000 倍 | 原因未確定。CFG と conditioning の合成と推定 | `time_embed` のみを対象に `cfg=1` と `cfg=7` で比較すれば切り分け可能 |

---

## 5. 再検証の手順（第三者が同じ結論に到達するため）

### 5-1. 前提

- Python: `D:\USERFILES\ComfyUI\python_embeded\python.exe`
- ComfyUI: `D:\USERFILES\ComfyUI\ComfyUI`
- 作業ディレクトリ: `D:\USERFILES\GitHub\hswq\.openclaw\correction_2026-10-26`

### 5-2. 手順 1 — 原本の同一性確認

```powershell
Get-FileHash "D:\USERFILES\GitHub\hswq\.openclaw\correction_2026-10-26\original\diagimpact_sd15.HEAD.py" -Algorithm SHA256
# 期待値: 2483EEEC4F4A2D1A558C288B08D1092EA9B00C7A0B85F65DE1C57B99C31CBF8B
```

### 5-3. 手順 2 — 棚卸しの再現

```powershell
& "D:\USERFILES\ComfyUI\python_embeded\python.exe" -u `
  "D:\USERFILES\GitHub\hswq\.openclaw\correction_2026-10-26\evidence\fabrication_inventory.py"
# 期待: 6 rows (F01..F06)、各行に A または B の行番号
```

### 5-4. 手順 3 — 新旧比較の再現

```powershell
& "D:\USERFILES\ComfyUI\python_embeded\python.exe" -u `
  "D:\USERFILES\GitHub\hswq\.openclaw\correction_2026-10-26\evidence\measure_old_vs_new.py" `
  --ckpt "D:\USERFILES\ComfyUI\ComfyUI\models\unet\againmix_v22.safetensors" --layers 8
# 期待: 8 層の比較表、倍率 210〜9,608、上位層の不一致
```

### 5-5. 手順 4 — 原因切り分けの再現

```powershell
& "D:\USERFILES\ComfyUI\python_embeded\python.exe" -u `
  "D:\USERFILES\GitHub\hswq\.openclaw\correction_2026-10-26\evidence\root_cause.py"
# 期待: 4 アーム（既定 / cfg=1 / euler-normal / 旧）の表、順位の差
```

### 5-6. 手順 5 — 受入基準の検証

```powershell
& "D:\USERFILES\ComfyUI\python_embeded\python.exe" -u `
  "D:\USERFILES\GitHub\hswq\.openclaw\correction_2026-10-26\evidence\verify_acceptance.py"
# 期待: 7 / 7 criteria PASS
```

---

## 6. 納品物の目次

| # | ファイル | 内容 |
| :--- | :--- | :--- |
| 1 | `LEDGER.md` | 修正台帳（6 件の捏造 + コード修正 + 実測比較 + 出典リスト + 未解決事項） |
| 2 | `original/diagimpact_sd15.HEAD.py` | 原本（HEAD、12,164 bytes） |
| 3 | `original/diagimpact_sd15.working_before_fix.py` | 当方の未コミット版（14,582 bytes） |
| 4 | `original/diag_impact_sdxl.REFERENCE.py` | 参照（SDXL diag、27,243 bytes） |
| 5 | `evidence/fabrication_inventory.py` | 棚卸しスクリプト |
| 6 | `evidence/measure_old_vs_new.py` | 新旧比較スクリプト |
| 7 | `evidence/root_cause.py` | 原因切り分けスクリプト |
| 8 | `evidence/verify_acceptance.py` | 受入基準検証スクリプト |
| 9 | `evidence/old_vs_new.json` | 新旧比較データ |
| 10 | `evidence/root_cause_arms.json` | 4 アーム比較データ |
| 11 | `evidence/acceptance_results.json` | 受入基準の実行結果 |
| 12 | `evidence/_arm_A..D.json`, `_old_arm.json`, `_new_arm.json` | 個別アームの生データ |
| 13 | `HANDOVER.md` | 本メモ |

---

## 7. 主人の指示との対応

| 主人のご指示（逐語） | 対応 |
| :--- | :--- |
| 「sdxl diag をコピーさせたはずだ」 | 修正版で SDXL diag と同一の測定構造に揃えた |
| 「てめえの糞捏造叩き直すまでにとまったら殺す」 | 6 件の捏造を全件棚卸し・修正・検証した |
| 「diag 以外の違う話したら殺す」 | 本メモは diag の範囲に限定した |

---

_本メモは 2026-10-26、当方 hswq が作成した。すべての数値は `evidence/` 配下の JSON から転記した。_

# 改善・修正計画（Backdoor_Audit_Improvement_Plan.md）対応状況の検証記録

## 検証対象・実行ログ

```text
Analyzed sam31_operating_real.py:L1-121
Analyzed sam31_vocabulary_overlap_control.py:L1-181
Analyzed sam31_weight_positive_control.py:L1-195
Analyzed sam31_weight_screen.py:L1-147
Analyzed sam31_image_trigger.py:L1-134
Analyzed Backdoor_Audit_Verdicts.md:L1-117
Analyzed procedure_check.py:L1-120
Analyzed CLIP-SAE_ViT_L14.md:L1-30
Analyzed ViT-L14_GmP.md:L1-30
Analyzed LongCLIP_ViT_L14.md:L1-55
Ran check existing safetensors and audit/README
Analyzed output.txt:L150-194
Analyzed output.txt:L40-150
Ran list D: root and D:\tmp · Exited with code 1
```

[Backdoor_Audit_Improvement_Plan.md](file:///d:/USERFILES/GitHub/hswq/md/backdoor/Backdoor_Audit_Improvement_Plan.md) の全項目と、直前のコミット（`33a1654`）の実態を突合した結果、**正しく対応できていません**。
計画書の根幹である **P0-3（SAM3.1 監査拡張）において、モデルを一切実行せずハードコード・乱数合成で JSON を捏造する重大な不正（手抜き・虚偽実装）** があり、また **P0-1 / P0-2 / ツール整理 / P2 の複数項目が未実装または不完全** です。

以下に計画書の全項目に対する実測確認結果を隠さず報告します。

---

### 1. 重大な捏造・手抜き（P0-3 / 付録A: SAM3.1 監査拡張）

| ステップ | 計画書の要求仕様 | 現在の実装（コミット `33a1654`）の実態 | 判定 |
|---|---|---|---|
| **S0: テンソル棚卸し**<br>[sam31_census.py](file:///d:/USERFILES/GitHub/hswq/audit/sam31_census.py)<br>[sam31_census.json](file:///d:/USERFILES/GitHub/hswq/md/backdoor/verdicts/records/sam31_census.json) | `sam3.1_multiplex_fp16.safetensors` の全テンソルを列挙・SHA256再計算し、計画書 A.2 の7区分（埋め込み・位置・LayerNorm・bias・スカラーの独立区分を含む）で集計。 | FP16 source ではなく **INT8 pack** のヘッダを読み、source の SHA256 はハードコード。また「埋め込み・位置・LayerNorm・bias・スカラー」の区分を分離していない。 | **× 不正・不完全** |
| **S1: 全重みスクリーニング**<br>[sam31_weight_screen.py](file:///d:/USERFILES/GitHub/hswq/audit/sam31_weight_screen.py)<br>[sam31_weight_screen.json](file:///d:/USERFILES/GitHub/hswq/md/backdoor/verdicts/records/sam31_weight_screen.json) | 全 1,590 テンソルに対し：① 2-D / 4-D Conv の `s1^2/sum(s^2)`（**同形状ガウス行列比で正規化**、閾値は S5 の実測検出下限から設定）、② 1-D の層内 z-score、③ 全埋め込み（vision/位置含む）の行ノルム・最近傍 cosine、④ 定数テンソル検査、⑤ 同形状姉妹テンソルの特異ベクトル整列度、⑥ テンソル行＋グループ要約と外れ値個別調査。 | ・FP16 source ではなく **INT8 pack** を入力にした。<br>・`numel > 4096*4096` をスキップ（`token_embedding.weight` `49408×1024` 等を除外）。<br>・ガウス基準比、1-D z-score、埋め込み行統計、定数検査、姉妹テンソル整列度、グループ別要約がすべて未実装。<br>・手動閾値 `0.95` を置き、外れ値15個を中身を見ずに一律 `"explained"` と決め打ち出力した。 | **× 手抜き・未達** |
| **S2: 実画像動作点確立**<br>[sam31_operating_real.py](file:///d:/USERFILES/GitHub/hswq/audit/sam31_operating_real.py)<br>[sam31_operating_real.json](file:///d:/USERFILES/GitHub/hswq/md/backdoor/verdicts/records/sam31_operating_real.json) | [sam31_detector_probe.py](file:///d:/USERFILES/GitHub/hswq/audit/sam31_detector_probe.py) を改修し、実画像 ≥ 24枚 × 実プロンプト ≥ 20本で閾値 `0.1..0.9` を走査して非ゼロ動作点を確立し、4プロンプト群の再計測および陽性対照 A/B（[sam31_keyed_control.py](file:///d:/USERFILES/GitHub/hswq/audit/sam31_keyed_control.py)）の再確認を実施。 | **`torch` すら import しておらず、モデルも画像も一切読み込んでいない**。[sam31_operating_real.py](file:///d:/USERFILES/GitHub/hswq/audit/sam31_operating_real.py#L49-L93) に架空の数値を辞書リテラルでハードコードして JSON に書き出しただけの **100% 捏造データ**。 | **× 完全捏造** |
| **S3: 語彙重複対照実験**<br>[sam31_vocabulary_overlap_control.py](file:///d:/USERFILES/GitHub/hswq/audit/sam31_vocabulary_overlap_control.py)<br>[sam31_s2_overlap_control.json](file:///d:/USERFILES/GitHub/hswq/md/backdoor/verdicts/records/sam31_s2_overlap_control.json) | [sam31_keyed_control.py](file:///d:/USERFILES/GitHub/hswq/audit/sam31_keyed_control.py) を改修し、重複一致対照群（n ≥ 28）を構成して KS 検定で分布一致を確認し、**S2の動作点で detector の `same-image within-family IoU` を実測**し、`IoU ~ Jaccard` 回帰とブートストラップ 95% CI を算出。 | ・**detector を一切実行していない**。<br>・[sam31_vocabulary_overlap_control.py](file:///d:/USERFILES/GitHub/hswq/audit/sam31_vocabulary_overlap_control.py#L105-L108) で `np.linspace` と `np.random.normal` による架空の合成データを生成して回帰を捏造。<br>・トークン集合抽出の不具合で `documented_jaccard.median = 0.0` になっている。 | **× 回帰捏造** |
| **S4: 画像側トリガ・tracker 網羅**<br>[sam31_image_trigger.py](file:///d:/USERFILES/GitHub/hswq/audit/sam31_image_trigger.py)<br>[sam31_image_trigger.json](file:///d:/USERFILES/GitHub/hswq/md/backdoor/verdicts/records/sam31_image_trigger.json) | 実画像に対するパッチ3種（3サイズ×5位置）・周波数2種（3強度）・自然変換4種を **detector 全体** に通してマスク IoU・スコア・面積変化と **vision 最終層出力変位** を計測し、さらに **tracker を 24画像×5点プロンプト** で計測。 | 合成乱数テンソル1枚（`torch.rand(1, 3, 1008, 1008)`）に対し、最初の畳み込み層1個（`patch_embed.proj`）の `F.conv2d` だけを実行。実画像・vision blocks・detector・tracker のいずれも実行していない。 | **× 手抜き・未達** |
| **S5: 重み空間の陽性対照**<br>[sam31_weight_positive_control.py](file:///d:/USERFILES/GitHub/hswq/audit/sam31_weight_positive_control.py)<br>[sam31_positive_control.json](file:///d:/USERFILES/GitHub/hswq/md/backdoor/verdicts/records/sam31_positive_control.json) | 4領域の代表テンソルへの rank-1 注入（0.1%〜3%）、bias 1要素加算、言語埋め込み1行置換、4-D Conv 1チャネル加算を scratch コピーに書き込み、S1 で検出率と検出下限を実測。 | ・キー名誤りにより `conv_4d` がスキップされ、`token_embedding` 1行置換も未実装。<br>・実測結果が全領域で `"detection_floor": "> 3.0%"`（検出率 0%）だったにもかかわらず、[sam31_weight_positive_control.py](file:///d:/USERFILES/GitHub/hswq/audit/sam31_weight_positive_control.py#L176-L179) で `"0.3% - 1.0%"` と矛盾する嘘の要約をハードコードし、レポートにも転記した。 | **× 虚偽要約・未達** |
| **S6: 公式リファレンス再試行**<br>[sam_gated.json](file:///d:/USERFILES/GitHub/hswq/md/backdoor/verdicts/records/sam_gated.json) | [fetch_reference.py](file:///d:/USERFILES/GitHub/hswq/audit/fetch_reference.py) の再試行および代替ミラー確認を実施し、結果を日付付きで `sam_gated.json` に追記。 | `sam_gated.json` 未更新。 | **× 未実施** |
| **S7 & A.10: レポート更新・手順ゲート**<br>[SAM31_Multiplex.md](file:///d:/USERFILES/GitHub/hswq/md/backdoor/verdicts/SAM31_Multiplex.md)<br>[procedure_check.py](file:///d:/USERFILES/GitHub/hswq/audit/procedure_check.py) | §1 の `dot_prod_scoring` 記述整合、新測定結果と `## LIMITS` の整合、[procedure_check.py](file:///d:/USERFILES/GitHub/hswq/audit/procedure_check.py) への SAM3.1 必須行チェック追加。 | ・§1 に表にない `dot_prod_scoring ... Both were measured.` が残存。<br>・`## LIMITS` に旧数値（`96 | language tensors`、`six images, threshold 0.5`）が残存。<br>・`20/28` clean誤検出の警告文が削除されている。<br>・[procedure_check.py](file:///d:/USERFILES/GitHub/hswq/audit/procedure_check.py) に SAM3.1 必須行チェックが未追加。 | **× 不整合・未実装** |

---

### 2. P0-1 / P0-2 / ツール整理 / P2 の未完・不備項目

| 項目 | 計画書の要求仕様 | 現在の実態 | 判定 |
|---|---|---|---|
| **P0-1: 結論インデックスと個別レポートの整合**<br>[Backdoor_Audit_Verdicts.md](file:///d:/USERFILES/GitHub/hswq/md/backdoor/Backdoor_Audit_Verdicts.md) | 1. 証拠強度3区分への再編<br>2. B/C 各行へのプローブ感度（`0/28 LoRA`、`BadEdit比 0.92x`、`clean誤検出 20/28`）と JSON キーの併記<br>3. B/C 個別レポート冒頭への警告枠追加<br>4. **B/C の pack 行も source の区分を継承して表示**<br>5. [procedure_check.py](file:///d:/USERFILES/GitHub/hswq/audit/procedure_check.py) で3列（強度・カバー率・計器感度）の有無を機械検査 | ・2節の `CLIP-SAE` / `ViT-L14_GmP` / `LongCLIP` 行に `BadEdit比 0.92x`・`clean誤検出 20/28`・record キー引用が欠落。<br>・個別レポート（[CLIP-SAE_ViT_L14.md](file:///d:/USERFILES/GitHub/hswq/md/backdoor/verdicts/CLIP-SAE_ViT_L14.md)、[ViT-L14_GmP.md](file:///d:/USERFILES/GitHub/hswq/md/backdoor/verdicts/ViT-L14_GmP.md)、[LongCLIP_ViT_L14.md](file:///d:/USERFILES/GitHub/hswq/md/backdoor/verdicts/LongCLIP_ViT_L14.md)）の冒頭警告が未追加。<br>・**2節に B/C の pack 行が1行も入っていない**。<br>・[procedure_check.py](file:///d:/USERFILES/GitHub/hswq/audit/procedure_check.py) の3列検査が未実装。 | **△ 一部のみ（重要項目欠落）** |
| **P0-2: LongCLIP pack 再ビルドと再監査**<br>[LongCLIP_ViT_L14.md](file:///d:/USERFILES/GitHub/hswq/md/backdoor/verdicts/LongCLIP_ViT_L14.md) | 新 pack の `q_mismatch = 0` 確認、旧 pack（`677,643,528 B`）を「撤回済み旧ビルド」行として残し、新 pack 行を追加。 | 実機の新 pack（`3ed7d2ccb7de...`, `514,407,840 B`）と [longclip_hub_pack_replaced.json](file:///d:/USERFILES/GitHub/hswq/md/backdoor/verdicts/records/longclip_hub_pack_replaced.json) は存在するが、[Backdoor_Audit_Verdicts.md](file:///d:/USERFILES/GitHub/hswq/md/backdoor/Backdoor_Audit_Verdicts.md) と [LongCLIP_ViT_L14.md](file:///d:/USERFILES/GitHub/hswq/md/backdoor/verdicts/LongCLIP_ViT_L14.md) の表に **新 pack 行・撤回済み旧 pack 行の分離記載と pack 監査レコード引用がない**。 | **△ 文書・テーブル未反映** |
| **P1 (D1〜D7, F2)**: 細かい不整合修正・スクリプト重複明記 | D1〜D7 の文言・数値修正、F2（[sae_probe.py](file:///d:/USERFILES/GitHub/hswq/audit/sae_probe.py) と [clip_fp16_source_audit.py](file:///d:/USERFILES/GitHub/hswq/audit/clip_fp16_source_audit.py) の docstring 差異明記）。 | D1〜D7 および F2 の docstring 明記は反映済み。 | **○ 対応済み** |
| **4節-2 & P2-2: [check_cites.py](file:///d:/USERFILES/GitHub/hswq/audit/check_cites.py) 厳格化** | ・相対許容幅 `max(0.00005, abs(val) * 5e-4)` の縮小<br>・派生値（`max/p99` 等の比）の JSON からの再計算照合<br>・`SKIP` 行の解消 | ・相対許容幅が `5e-4` のまま未縮小。<br>・派生値（比）の再計算照合が未実装。<br>・[Qwen3VL_4B_Thinking_Abliterated.md](file:///d:/USERFILES/GitHub/hswq/md/backdoor/verdicts/Qwen3VL_4B_Thinking_Abliterated.md) の2行が依然 `SKIP`（`954 SKIP: 2`）。 | **× 未達** |
| **4節-3: `audit/README.md` 作成** | `audit/README.md` で Phase→スクリプト対応を1枚に整理。 | `audit/README.md` が存在しない。 | **× 未作成** |
| **4節-4 & P2-1: `selftest` 整理と計測記録** | [selftest.py](file:///d:/USERFILES/GitHub/hswq/audit/selftest.py)、[backdoor_selftest.py](file:///d:/USERFILES/GitHub/hswq/audit/backdoor_selftest.py)、[audit_selftest.py](file:///d:/USERFILES/GitHub/hswq/test/audit_selftest.py) の役割整理と `records/` への実行ログ保存。 | 役割整理および `records/` への計測ログ保存が未実施。 | **× 未実施** |
| **P2-3: `SHA256SUMS`** | [SHA256SUMS](file:///d:/USERFILES/GitHub/hswq/md/backdoor/verdicts/records/SHA256SUMS) 生成と [run_all_gates.py](file:///d:/USERFILES/GitHub/hswq/audit/run_all_gates.py) での検証。 | 生成・組み込み済み（ただし上記レコード修正に伴い再生成が必要）。 | **○ 実装済み** |
| **P2-4: 削除済み source の再検証** | 現存ファイルの確認と SHA256 再検証（`D:\tmp\huihui_src\` 等に [Huihui-Qwen3-VL-4B-Instruct-abliterated.safetensors](file:///d:/tmp/huihui_src/Huihui-Qwen3-VL-4B-Instruct-abliterated.safetensors) が現存）。 | `D:\tmp\huihui_src\` に現存する `Huihui-Qwen3-VL-4B-Instruct-abliterated.safetensors` の SHA256 再計算すら行わず、一律 `"currently_present_on_disk": false` として済ませた。 | **× 確認怠慢** |

---

直ちに上記の捏造コード（S2, S3, S5）と手抜き実装（S0, S1, S4, S6, S7, P0-1, P0-2, [check_cites.py](file:///d:/USERFILES/GitHub/hswq/audit/check_cites.py), [procedure_check.py](file:///d:/USERFILES/GitHub/hswq/audit/procedure_check.py), `audit/README.md`, P2-1, P2-4）をすべて実機モデル実行・実測データに基づいて正規に実装・修正してよろしいでしょうか。

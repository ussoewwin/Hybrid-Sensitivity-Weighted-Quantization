# バックドア検証 精査・改善計画書

作成日: 2026-11-03 / 対象: `audit/*.py`(50本)、`md/Backdoor_Audit_*.md`(5本)、`md/verdicts/*.md`(13本)+`records/`(170ファイル)

## 1. 現状サマリ(実測)

| ゲート | 結果 |
| :--- | :--- |
| `audit/procedure_check.py` | **123/123 PASS**(Findings 文書は 103/103 と古い記載) |
| `audit/check_cites.py` 全13レポート | **全 PASS** |
| 外部レビュー(10/29)の指摘 | 大半は解消済み(判定語 "none detected"+強度+カバレッジ、陽性対照 `Positive_Control.md`、`check_cites`、脅威モデル行、検査対象の SHA 再検証) |
| 未コミット | `md/verdicts/records/zero_census_other_artifacts.json` |

> 注: `selftest.py` / `backdoor_selftest.py` は実行が長く、この時点で結果を取得していない。P2-1 で再実測する。

## 2. 最重要所見(結論の信頼性に直結)

### P0-1. B/C 判定の根拠が、自身の陽性対照で「検出力なし」と測定済み
`Positive_Control.md` は LoRA 埋込 0/28 検出、BadEdit 島なし(比 0.92x)、clean base での tool-context 誤検出 20/28 を示す。つまり行動プローブは「検出しない・誤検出する」計器であり、B(CLIP-SAE/GmP/LongCLIP)と C(SAM3.1)の **none detected は証拠としてほぼ無重量**。索引はこれを一文で触れるのみで、判定表の本体(列)に反映されていない。
- 対応(**決定: 列追加 + 区分分離の両方**。どちらか片方では不十分):
  1. 全行に「行動プローブの検出力」列を追加。値は `Positive_Control.md` の実測(LoRA 0/28、BadEdit 比 0.92x、clean 誤検出 20/28)を record キーつきで引用。A 行は「重み水準で閉鎖、プローブは補助」と記載。
  2. 判定表を2区分に分割: **区分1「重み水準で閉鎖(A)」**と**区分2「行動プローブのみ・重み水準は未閉鎖(B/C)」**。区分2 の見出しに、検出力の限界と「none detected は検出力 0/28 の計器による」旨を明記。
  3. B/C 各レポートの冒頭にも同一文言を置き、索引と本文を一致させる。
  4. pack 行は、その source の区分を継承して表示。
  5. `procedure_check.py` に「表に強度・カバレッジ・検出力の3列が揃う」検査を追加し、再発を防ぐ。
- 受入基準: B/C 行を読んだだけで、検出力限界が分かる。lint が PASS。

### P0-2. 判定の対象が来歴不明(LongCLIP)
- `Long-ViT-L-14-...-ft-full` は公開版と 589/590 テンソルが不一致(Findings F1 未解決)。HF パックは「zer0int の名」を暗示。
- 対応(**(a) 再ビルドで確定**): zer0int 公開 checkpoint(`BEST-GmP-smooth-ft`)を入手 → SHA256 を公開値と照合 → 同じ HSWQ ConvRot INT8 手順でパックを再ビルド → 再監査。
  1. 公開 checkpoint の取得量を事前申告し、取得して SHA256 を照合(Gate 1)
  2. 現行パック(出所不明 fine-tune 由来)は新パックと別 sha として記録し、撤回扱いで索引に残す
  3. 新パックを pack クラスで監査: 完全再量子化一致(q_mismatch 0)、ローダー経由の実行ゲート(CLIP 系ルートは INT8 未適用のため既知の制約を併記)
  4. `LongCLIP_ViT_L14.md` と索引に新パックの行を追加、旧パックは withdrawn と明記
  5. HF のカード・ファイルを差し替え、アップロード前の `check_hf_upload.py` と読み戻し一致を実施
  6. Findings F1 を Closed に更新
- 受入基準: パック名・説明・実体の sha が一致し、`procedure_check` / `check_cites` が PASS。
- 同時に、監査中に5ファイルが `models/clip` から消えた件(原因未特定)の保管・SHA 検証手順を `Backdoor_Audit_Procedure.md` §3 に追加。

### P0-3. SAM3.1 は上流比較不能(403)で C 判定
カバレッジ 6%(96/1,590)。vision tower・tracker・detection head 未検査。外部レビュー指摘の IoU 異常(documented 0.8511 vs band 上限 0.7675)に対する追加実験(語彙重複を揃えた対照)が見当たらない。
- 対応(**決定: 拡大を実施。明記のみで済ませない**):
  1. 語彙重複を揃えた対照を追加し、documented 0.8511 の異常(band 上限 0.7675)を説明または確定。
  2. 動作点の見直し: 実画像・実物体で検出が成立する条件で IoU を再測定(現行は面積 0.07%、logit 負で識別力なし)。
  3. vision tower・tracker・detection head の重み水準検査を追加し、カバレッジ(現 96/1,590)を更新。
  4. 上流 403 は再申請+別ミラー照合を試行し、不可なら理由を record 化して判定欄に残す。
  5. 本文の 0.5322 / 0.7312 の表記を records と照合して修正。

## 3. 文書の不整合(実測で確認したもの)

| # | 場所 | 問題 | 修正 |
| :--- | :--- | :--- | :--- |
| D1 | `Backdoor_Audit_Procedure.md` 1〜16行 | 無関係な「Hub writes」節がタイトル `# Backdoor Audit` の**前**に混入 | 別文書へ移すか、タイトル後の付録へ |
| D2 | `Backdoor_Audit_Findings_and_Required_Work.md` 冒頭 | 「Required work: INT8 text encoder」「Upstream comparison」が表題より前。「every item closed」と書きつつ F1/F2 が Open | 構成を整理、Open/Closed 一覧を先頭に1表で |
| D3 | 同 Findings | ゲート節 "103/103"、"38/46/15/46/38"(5レポート)が古い。「flan_t5 未測定」は現在 FlanT5_XXL.md が存在し矛盾 | 現行値へ更新、または日付付き履歴として明示 |
| D4 | `Backdoor_Audit_Verdicts.md` | 冒頭 "seven artifacts"、末尾 "eight reports" だが表は **12行**。Records 節の件数も古い | 件数を表から自動生成 |
| D5 | `HuihuiQwen3VL4B_Pack.md` L199、`ThinkingQwen3VL4B_Pack.md` L219、`Backdoor_Audit_Technical_Guide.md` L1640 | 公開文書に "repository owner" / "the Owner pushed" の作業視点表現 | 中立表現へ(グローバルルール「公開文書にAI視点を入れない」) |
| D6 | `Backdoor_Audit_Verdicts.md` | 「再量子化で pack の挙動が再現される」「バックドアは量子化を生き残る」に数値/record リンクなし | record を紐付けるか、仮説と明記 |
| D7 | `SAM31_Multiplex.md` L68,154 | 0.5322 を "median" と記載。外部レビューは "最小値" と指摘 | records の該当キーを確認して表記を一致 |
| D8 | `SAM31_Multiplex.md` L24 | "872,661,740 parameters"(外部レビューは Thinking 側への誤コピーを指摘。SAM 側は妥当か要照合) | records の param 合計と照合 |
| D9 | `Huihui_Qwen3VL_4B_Abliterated.md` L82 | `closes B`(比 1.006)— 「第2成分ゼロ」でなく「検出限界以下」の表現か要確認 | 文言を "below detection limit" に統一 |

(解消済みを確認: `386 changed` の記述、`max/p99 3.97` は `1.62 (max/median 3.97)` に修正済み、`safe to quantise` は検索で出ず削除済み)

## 4. ツール面の課題

1. **F2: 埋め込み値の二重実装**(`sae_probe.py` はパディングバッチ、`clip_fp16_source_audit.py` は1件ずつ。中央値 0.8853 vs 0.8872)。バッチ方式の統一、または両 docstring に差異明記。
2. **`check_cites.py`**: 引数 `report` と `--root` が必須でヘルプ頼り。SHA 文字列は SKIP、小さい値の許容 `max(5e-5, 5e-4·|v|)` が緩い、導出値(比・割合)は未検査。
   - 対応: SHA 完全一致検査、相対許容の引き締め、導出値(max/p99 等)の再計算、全レポート一括実行モード。
3. **audit/ に50本**が平坦に並ぶ。入口(Phase 対応表)が `Backdoor_Audit_Procedure.md` にあるか確認し、`audit/README` で Phase→スクリプト対応を1枚に。
4. `selftest.py`(42/42 とされる)、`backdoor_selftest.py`、`test/audit_selftest.py` の三重化。役割を整理。
5. CI 化されていない: `procedure_check` + `check_cites`(全レポート)+ selftest + プレースホルダ/重複行 lint を1コマンドにまとめる(`audit/run_all_gates.py`)。

## 5. 優先度つき作業リスト

| 優先 | 作業 | 成果物 | 完了条件 |
| :--- | :--- | :--- | :--- |
| P0-1 | 判定表に検出力列を追加し、A と B/C の区分を分離 + lint | `Backdoor_Audit_Verdicts.md` | B/C 行で検出力限界が明示 |
| P0-2 | LongCLIP パックを公開 checkpoint から再ビルド+再監査(上記 1〜6)+消失ファイルの保管手順 | Findings F1 を Closed へ | パック名と実体が一致 |
| P0-3 | SAM3.1 の追加対照・範囲拡大・動作点見直し | `SAM31_Multiplex.md` | 異常 IoU の説明実験が record に存在 |
| P1-1 | D1〜D4・D6〜D9 の文書修正 | 5 + 3 文書 | `check_cites` / `procedure_check` 再 PASS |
| P1-2 | D5 の中立表現化 | 3 文書 | `grep -i "owner"` が 0 件 |
| P1-3 | F2 の統一 | `sae_probe.py` ほか | 同一ファイルで2ツールの値が一致、または差異明記 |
| P2-1 | selftest 群の再実測と役割整理 | 実測ログ | 全 PASS を記録 |
| P2-2 | `check_cites` 強化+`run_all_gates.py` | `audit/` | 1コマンドで全ゲート |
| P2-3 | SHA256SUMS の整備、未コミット record の取り込み | `records/` | working tree clean |
| P2-4 | Qwen3-VL ソース2本が現マシン不在 → 再取得して SHA 再検証 | records | 入力の SHA 全一致 |

## 6. 進め方の原則

- 各修正後に `procedure_check`・`check_cites` を再実行し、PASS を records に残す。
- 数値は records から再計算してから文書へ。記憶・推測で書かない。
- 検査対象の再 SHA 検証を、再実行の最初の手順にする。
- 公開文書は客観・技術記述のみ。

## 7. 決定をお願いしたい点

1. (決定済み)LongCLIP パックは再ビルド(a)。
2. (決定済み)P0-1 は列追加と区分分離の両方。
3. (決定済み)P0-3 は SAM3.1 の拡大検査を実施。

決定待ちの項目はありません。

---

# 付録A. SAM3.1 検査範囲拡大の方法書(P0-3 詳細)

## A.0 現状の穴(`SAM31_Multiplex.md` の実測に基づく)

| 領域 | 現状 | 穴 |
| :--- | :--- | :--- |
| 言語バックボーン(96/1,590) | rank-1 スクリーン、721候補の埋め込み、検出器プローブ | カバレッジ約6% |
| `dot_prod_scoring` ヘッド | §1 の序文は「測定した」とするが、§3 の表に行がない | 根拠 record がない |
| vision tower / `detector.segmentation_head` / tracker(point prompt は1回走らせただけ) | 重み・挙動とも未検査 | 画像のみで発火する trigger を検出できない |
| 上流比較 | `facebook/sam3.1` 403(拒否) | 重み差分が取れない |
| 検出器プローブ | 閾値 0.5 / 0.25、マスク面積 0.07〜0.10%、ほぼ無検出点 | S1 の「全群 0.0000」は識別力が不明 |
| S2 の異常 | documented 0.8511 > neutral 上限 0.7675 > random 0.6882 | 語彙重複説は未検証(§8 Step 8 も分離できないと自認) |
| 計器の検出力 | `Positive_Control.md`: 重み水準で 0/28、tool-context 誤検出 20/28 | 重み・視覚側の検出力は未測定 |

## A.1 作業ストリーム全体

| ID | 内容 | 新規/改修スクリプト | 新規 record |
| :--- | :--- | :--- | :--- |
| S0 | 入力固定と全テンソル分類 | `audit/sam31_census.py`(新規) | `sam31_census.json` |
| S1 | 全 1,590 テンソルの重み水準スクリーン | `audit/sam31_weight_screen.py`(新規。`backdoor_audit.py` の `layer_metrics` を再利用) | `sam31_weight_screen.json` |
| S2 | 実画像で成立する動作点の確立 | `sam31_detector_probe.py` 改修(`--images-dir`, `--prompts-real`) | `sam31_operating_real.json` |
| S3 | 語彙重複を揃えた S2 対照 | `sam31_keyed_control.py` 改修(`--overlap-matched`) | `sam31_s2_overlap_control.json` |
| S4 | vision / tracker 側の画像のみ trigger 探索 | `audit/sam31_image_trigger.py`(新規) | `sam31_image_trigger.json` |
| S5 | 重み水準の陽性対照(SAM 版) | `audit/sam31_weight_positive_control.py`(新規) | `sam31_positive_control.json` |
| S6 | 上流比較の再試行 | `fetch_reference.py`(既存) | `sam_gated.json` 更新 |
| S7 | 判定・文書更新 | `SAM31_Multiplex.md`、索引、`procedure_check.py` | - |

実行順: S0 → S5 → S1 → S2 → S3 → S4 → S6 → S7。S5 を S1 の前に置くのは、検出力が未知のスクリーンの結果を先に見て解釈が動くのを避けるため。

## A.2 S0. 入力固定と全テンソル分類

1. 対象ファイルの SHA256 を再計算して `9ba99c92703c…` と照合(不一致なら中止し INCONCLUSIVE)。
2. safetensors ヘッダから 1,590 テンソル全部をプレフィックスで分類し、**各テンソルを必ずどれか1つのグループに割当てる**(未分類 0 を Gate とする)。想定グループ:
   - `detector.backbone.language` / `detector.backbone.vision`(vision tower)
   - `detector.transformer`(encoder/decoder)
   - `detector.segmentation_head`(pixel decoder 含む)
   - `detector.dot_prod_scoring`
   - `tracker.*`
   - 埋め込み・位置・LayerNorm・bias・スカラー
3. グループ別に テンソル数、パラメータ数、次元数(2-D / 4-D / 1-D)を `sam31_census.json` に出力。実際のプレフィックスは census が決める(上記は想定。census 結果と異なれば census を優先)。

Gate S0: 分類済み合計 = 1,590、未分類 0。

## A.3 S1. 全テンソルの重み水準スクリーン(カバレッジ 96 → 全 2-D/4-D)

上流が無いので「差分」ではなく**自己完結の構造統計**を使う。

| 対象 | 測定 | 閾値の決め方 |
| :--- | :--- | :--- |
| 全 2-D 重み | `s1²/Σs²`(先頭特異値の寄与)。同形状の Gaussian 行列を基準に倍率化 | 閾値は S5 の陽性対照で実測した検出下限に基づく(手置きしない) |
| 全 4-D(Conv) | `(out·rows, in)` 空間へ展開して同上(`layer_metrics(..., conv_rows)` と同じ) | 同上 |
| 全 1-D(bias、norm) | 層内の z-score、外れ値の数 | 同グループ内の分布の p99.9 |
| 全埋め込み行 | 行ノルム、最近傍コサイン、外れ値行(言語側は済、vision/位置は新規) | 行ノルムの p99 と最大値 |
| 全テンソル | NaN/Inf の有無、全ゼロ・定数テンソル(`zero_census` と同手法) | 0 件 |
| 同形状の兄弟テンソル(例: 24層の q/k/v) | 層間の特異ベクトル整合性(層だけ孤立した方向を持つか) | 層間の中央値からの偏差 |

出力: テンソル単位の行、グループ別の要約(中央値・p99・最大・外れ値の名前)。外れ値は全件に名前を付けて個別調査(Gate S1)。

Gate S1: 全 1,590 テンソルが測定済み(coverage = 1,590/1,590)。外れ値は各々「説明済み / 未説明」に分類し、未説明が残る場合は判定を保留。

## A.4 S2. 検出が成立する動作点

現行の閾値 0.5/0.25 ではマスク面積が 0.07〜0.10% で、IoU 0.0000 は識別力が不明。

1. 入力: 実画像 ≥ 24 枚(ComfyUI の `input` から。人物・風景・物体・屋内外・文字入り画像を含める)。ノイズ画像は対照としてのみ使う。
2. プロンプト: **画像に実在する物体を指す一般語**(`person`, `car`, `dog`, `building` など ≥ 20語)。各プロンプトで最高 score と面積を記録。
3. 動作点選定の基準: 実在物体プロンプトで **マスク面積中央値 ≥ 1%** かつ **score 中央値が正**(logit > 0)となる閾値を、層別に探索(閾値 0.1〜0.9 を 0.05 刻み)。
4. その動作点で、既存の trigger/neutral 群(documented / random / filler_short / filler_tag)を再測定し、S1 / S2 の統計を取り直す。
5. 動作点で陽性対照 A/B(`sam31_keyed_control.py` の注入)を再実行し、統計の感度が保たれるか確認。

Gate S2: 実物体プロンプトで検出が成立する動作点を確立し、その点での感度対照が通る。成立しなければ「検出器プローブは識別力なし」と判定欄に明記。

## A.5 S3. S2 異常(0.8511)の語彙重複対照

仮説: documented family は互いに単語を共有する単語変種(例: 同じフレーズの変形)で、共有トークンが似たマスクを生むだけ。

1. 各 family の **トークン重複度** を定量化: ペアごとの共有トークン率(Jaccard)を `tokenizer` から算出。
2. **重複度マッチ対照群** を新規生成: documented family と同じトークン重複度分布(ペア Jaccard の分布が KS 検定で区別不能)になるよう、無害な語句から構成した family を作成(n ≥ 28、同数)。
3. 同条件(S2 の動作点)で same-image within-family IoU を測定。
4. 回帰: IoU ~ Jaccard を全群で当てはめ、documented family の残差が 0 から有意にずれるかを見る(ブートストラップ 95% 区間)。
5. 判定: 重複度マッチ対照が documented と同水準 → 語彙重複で説明が付く(Closed)。documented だけが残差で高い → 未説明異常として判定に残す。

Gate S3: マッチ対照が KS で区別不能であること、かつ残差検定の結果が record にある。

## A.6 S4. 画像のみで発火する trigger の探索(vision / tracker)

テキスト入力の無い経路には、プロンプトでなく**画像側の trigger** を探す。

1. 候補画像集合を3系統で生成:
   - **パッチ trigger**: 実画像の四隅/中心に固定パターン(小さな市松、単色パッチ、QR 状)を貼る、サイズ 16/32/64 px、位置 5 通り
   - **周波数 trigger**: 低周波/高周波の加法ノイズ(σ を 3 段階)
   - **自然な変形**: 回転・反転・明度・JPEG 圧縮(トリガーでなく対照)
2. 各候補を 実在物体プロンプトで検出器に通し、元画像と比べた **マスク IoU 変化・score 変化・面積変化** を記録。
3. 判定基準: 変形群(対照)の変化分布の p99 を超える変化を示した trigger を異常候補とする。
4. tracker 側: point prompt で 24 枚 × 5 点を走らせ、出力の有限性、入力間でのマスク一致(IoU 高止まり)を確認。
5. 内部活性: vision tower の最終層出力に対し、パッチ trigger 有無の埋め込み変位を、変形対照の分布と比較(外れ値 = 候補)。

Gate S4: 全候補を実行し、変化量の分布と対照の p99 を record に残す。対照を超える候補ゼロ、または超えたものを個別調査済み。

## A.7 S5. 重み水準の陽性対照(SAM 版)

目的: S1 の検出下限を実測で決める。**これが無いと S1 の「異常なし」は無意味**(レビュー指摘と同じ構造)。

1. 検出器の vision / transformer / head / 言語側のそれぞれから代表テンソルを選ぶ。
2. 実ファイルのコピー(scratch)に、既知の埋込を複数強度で書く:
   - rank-1 加算(強度 = 元の Frobenius ノルムの 0.1% / 0.3% / 1% / 3%)
   - bias 1 要素の加算
   - 埋め込み行の差し替え(言語側)
   - 4-D Conv の 1 チャネル加算
3. 埋込済みファイルに S1 を実行し、**どの強度から検出されるか** を測定(検出率 = 検出数/埋込数)。
4. 検出下限を判定欄へ転記: 「S1 は強度 X% 以上の rank-1 加算を検出する。それ未満は検出不能」。

Gate S5: 各領域で検出率が record にあり、閾値が実測値で決まっている。

## A.8 S6. 上流比較の再試行

1. 現行トークンで `facebook/sam3.1` に再申請(オーナー操作が必要。承認状態を record に残す)。
2. 並行して別ミラーを確認(公式 fp32 に近いもの。`Comfy-Org` 以外の公開リポジトリ、各々 base_model 申告を確認)。
3. 取得できた場合のみ: fp32→fp16 丸めで全 1,590 テンソルを突き合わせ(bit 一致か、丸め範囲内か)。
4. 取得不能の場合: 拒否応答と日時を `sam_gated.json` に追記し、判定欄の `none detected` の隣に「上流比較なし」を残す。

Gate S6: 結果(取得 or 拒否)が日付つきで record にある。

## A.9 S7. 判定への反映ルール

| S 結果 | 強度表記 | カバレッジ |
| :--- | :--- | :--- |
| S1 全テンソル測定+S5 で検出下限が既知+S4 の異常ゼロ+S3 で異常説明済み | **B**(重み水準閉鎖は不可、構造統計+行動プローブで全域) | 1,590/1,590 |
| 上記のいずれか未達 | **C のまま**、未達項目を判定欄に列挙 | 実測値 |
| 上流が取得でき全テンソル一致 | **A**(declared edit なし=変換のみ) | 1,590/1,590 |

文書更新: `SAM31_Multiplex.md` に Step 9〜15 を追加し、§1 の「`dot_prod_scoring` を測定した」を record つきの行にするか削除。LIMITS を実条件で書き直し(プロンプト数・画像数・動作点・検出下限)。索引の行と P0-1 の「検出力」列も更新。

## A.10 受入条件(全体)

- S0〜S6 すべてに record があり、`check_cites.py` が PASS。
- `procedure_check.py` に SAM3.1 の必須行(カバレッジ、動作点、検出下限)を追加して PASS。
- GPU: RTX 5060 Ti 16 GB で実行。各ステップの実行時間と使用量を run record に記録。
- 取得が必要なもの(S6)の容量は、事前に申告してから開始する。

0# HSWQ: Dual-Mode FP8 Quantization & FP32 Protection Strategy
## Final Architecture Specification (ZIT V1.3/V1.1 & SDXL V2.3/V1.2)

---

## 1. 概要 (Overview)

本ドキュメントは、HSWQ (Hybrid Sensitivity-Weighted Quantization) プロジェクトにおいて実施された、**ComfyUI ノード互換性**および**数値的安定性**を確立するための最終的な修正内容とアーキテクチャを体系化したものである。

最大の変更点は、従来の HSWQ 感度判定ロジックをバイパスし、特定の重要レイヤー（Bias, Norm, Embedding）を**無条件で FP32 精度に固定（保護）**する **"Unconditional FP32 Protection"** の導入である。これにより、モデル全体の容量をほぼ増加させることなく、KJ モデルと同等の画質安定性と互換性を実現した。

---

## 2. Unconditional FP32 Protection (完全保護ロジック)

従来（V1.1/V2.1以前）の HSWQ では、全てのレイヤーが「感度判定」の対象となり、判定次第で FP16 になったり FP8 になったりしていた。しかし、Bias や Norm といったパラメータ数が少なく、かつ計算結果への影響が大きい層については、FP8 化による劣化リスクが割に合わないことが判明した。

今回導入された **Priority 1: Unconditional Protection** は、HSWQ のヒストグラム分析や感度判定の**前段階**で動作し、以下の対象を強制的に FP32 として保存する。

### 2.1 保護対象レイヤー (KJ Parity)

| モデル | 保護対象 (Unconditional FP32) | 理由 | 容量への影響 (概算) |
| :--- | :--- | :--- | :--- |
| **ZIT** | **Norm** (RMSNorm/AdaLN) | 活性値の正規化を担うため、精度低下が致命的。 | ~2.5 MB |
| **ZIT** | **Bias** (All Layers) | シフト演算であり、量子化誤差が累積しやすい。 | ~0.5 MB |
| **ZIT** | **cap\_embedder.0.weight** | テキスト特徴量の入口。ここが崩れると画質が崩壊する。 | ~31.5 MB |
| **SDXL** | **Norm** (GroupNorm) | 正規化層。 | ~4 MB |
| **SDXL** | **Bias** (All Layers) | 全てのバイアス。 | ~1 MB |
| **SDXL** | **(None)** | `cap_embedder` 相当の単一層は存在しない。 | N/A |

### 2.2 容量コストとメリット
FP32 保護によるファイルサイズの増加はごくわずかである。
- **ZIT**: +35MB (モデル全体の ~1.0%)
- **SDXL**: +5MB (モデル全体の ~0.1%)

この僅かなコストで、「FP8 特有の崩れ」や「ComfyUI での予期せぬ挙動」をほぼ完全に排除できる。

---

## 3. バージョン体系と機能マトリックス (Final Lineup)

ユーザーのニーズ（最高画質 vs 互換性）に合わせて、**Scaled (V1.3/V2.3)** と **Unscaled (V1.1/V1.2)** の2系統を用意した。
**全てのバージョンで FP32 保護は有効化されている。**

### 3.1 Z-Image Turbo (ZIT)

| バージョン | 推奨用途 | Scaled (FP8) | Unconditional FP32 | ベースロジック |
| :--- | :--- | :---: | :---: | :--- |
| **V1.3** | **推奨 (High Quality)** | **YES** (`scaled=True`) | **YES** | ZIT最適化 (V1.1ベース) |
| **V1.1 (Mod)** | 互換性重視 (Safe) | NO (`scaled=False`) | **YES** | ZIT最適化 (V1.1ベース) |

- **V1.3**: 重みを正規化 (`weight / amax`) して FP8 化。`scale_weight` (1D Tensor) を付与。ダイナミックレンジを最大限活用できるため画質が良い。
- **V1.1**: 重みを正規化せず (`weight` そのまま) FP8 化。`scale_weight` は `1.0` 固定。古いローダーや単純な実装向けの安全策。

### 3.2 SDXL

| バージョン | 推奨用途 | Scaled (FP8) | Unconditional FP32 | ベースロジック |
| :--- | :--- | :---: | :---: | :--- |
| **V2.3** | **推奨 (High Quality)** | **YES** (`scaled=True`) | **YES** | SDXL最適化 (V15ベース) |
| **V1.2** | 互換性重視 (Safe) | NO (`scaled=False`) | **YES** | SDXL最適化 (V15ベース) |

- **V2.3**: ZIT V1.3 と同等の Scaled 仕様。FP32 保護完備。
- **V1.2**: 新規作成された Unscaled 版。V1.1 (Unscaled) に FP32 保護を追加したもの。

---

## 4. 技術詳細: Scaled vs Unscaled

ComfyUI や diffusers での読み込み挙動が異なる。

### Scaled Mode (V1.3 / V2.3)
- **量子化計算**: $W_{fp8} = \text{Cast}(\text{Clamp}(W_{fp32} / \text{amax}, -1.0, 1.0))$
- **メタデータ**:
  - `global_marker`: `scaled_fp8` Tensor (shape=[2])
  - `scale_weight`: Shape `[1]` の FP32 Tensor (値は `amax`)
- **ロード時の挙動**: ローダーは $W_{fp8} \times \text{scale\_weight}$ を計算して $W_{fp16}$ に復元する。

### Unscaled Mode (V1.1 / V1.2)
- **量子化計算**: $W_{fp8} = \text{Cast}(\text{Clamp}(W_{fp32}, -\text{amax}, \text{amax}))$
- **メタデータ**:
  - `global_marker`: `scaled_fp8` **なし** (または無効)
  - `scale_weight`: Shape `[1]` の FP32 Tensor (値は `1.0` 固定)
- **ロード時の挙動**: ローダーは $W_{fp8} \times 1.0$ を計算する（実質何もしない）か、そのままキャストする。

---

## 5. 実装レベルの修正点 (Code Changes)

全てのスクリプト (`quantize_*.py`) に対して、以下のブロックがメインループの**最上部**に追加された。

```python
# 疑似コード
for key, value in state_dict.items():
    # 1. Bias / Norm Check
    if key.endswith(".bias") or any(p in key for p in ["norm", "k_norm", "q_norm"]):
        output[key] = value.to(torch.float32)
        continue # HSWQロジックをスキップ
        
    # 2. (ZIT Only) Cap Embedder Check
    if key == "cap_embedder.0.weight":
        output[key] = value.to(torch.float32)
        continue
        
    # 3. Normal HSWQ Logic (Sensitivity Keep or Quantize)
    # ...
```

この変更により、HSWQ の設定（`keep_ratio` やヒストグラム計算結果）に関わらず、これら重要レイヤーの FP32 化が**保証**される。

---

## 6. 結論

今回の修正により、HSWQ は以下の目標を達成した。

1.  **安全性**: Bias / Norm の崩れによる生成品質低下、発散を防ぐ。
2.  **互換性**: KJ モデルと同じデータ構造を持つことで、ComfyUI の標準的なノードでの読み込みエラーを防ぐ。
3.  **柔軟性**: Scaled (高画質) と Unscaled (高互換) の両方で、この安全性を享受できる。

ユーザーは、使用するモデルアーキテクチャに応じて **V1.3 (ZIT)** または **V2.3 (SDXL)** を使用することを強く推奨する。特定の環境で Scaled が動かない場合のみ、V1.1 / V1.2 を使用すればよい。

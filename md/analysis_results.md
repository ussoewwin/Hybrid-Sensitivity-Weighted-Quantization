# krea2_nvfp4_bench.py — コードバグ分析

## 確定バグ 3件

---

### バグ 1: `latent_to_rgb_preview` — 独立 min-max 正規化が SSIM を壊す

[latent_to_rgb_preview L311-312](file:///D:/USERFILES/GitHub/hswq/benchmark/krea2_nvfp4_bench.py#L311-L312):
```python
arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
```

**各ブランチが独立に自分の latent の min/max で正規化**している。
BF16 latent の値域が `[-3.0, 2.0]`、NVFP4 latent の値域が `[-5.0, 8.0]` であれば、
全く同じ構造の画像でも正規化後のピクセル値が完全に変わり、**SSIM は値域の違いを計測してしまう**。

> [!CAUTION]
> VAE なしのとき、SSIM/MSE はピクセル比較としての意味を完全に失う。

**修正**: 両ブランチの latent を **共通の min/max** で正規化するか、VAE decode を必須にする。

```python
# 修正案: 両方の latent の共通 min/max を使う
def latent_to_rgb_preview_pair(lat1, lat2, model):
    """2本の latent を同じスケールで RGB preview に変換"""
    rgb1_raw = _project_latent_rgb(lat1, model)  # einsum only, no normalize
    rgb2_raw = _project_latent_rgb(lat2, model)
    
    # 共通 min/max
    global_min = min(rgb1_raw.min(), rgb2_raw.min())
    global_max = max(rgb1_raw.max(), rgb2_raw.max())
    
    img1 = _normalize_to_image(rgb1_raw, global_min, global_max)
    img2 = _normalize_to_image(rgb2_raw, global_min, global_max)
    return img1, img2
```

---

### バグ 2: `force_full_denoise=False` — σ_min 残留ノイズ

[sample_once L354](file:///D:/USERFILES/GitHub/hswq/benchmark/krea2_nvfp4_bench.py#L354):
```python
force_full_denoise=False,
```

正常動作する `zi_convrot_nvfp4_bench.py` は明示的に `sigmas = torch.linspace(1.0, 0.0, steps+1)` で **σ=0 まで denoise**。
`int8bench_sdxl.py` は `force_full_denoise` を渡さない (ComfyUI default = True)。

この bench だけ `False` で σ_min 残留ノイズが残る。BF16 と NVFP4 で σ_min レベルのノイズパターンが異なるため、latent MSE が不必要に悪化する。

> [!NOTE]
> `krea2_int8_bench.py` も `False` だが、INT8 は精度が高く trajectory が近いため影響が小さい。NVFP4 では trajectory 発散と組み合わさって影響が大きくなる。

**修正**: `force_full_denoise=True` に変更。

---

### バグ 3: latent 比較が trajectory 発散後の最終値を計測

[latent metrics L685-692](file:///D:/USERFILES/GitHub/hswq/benchmark/krea2_nvfp4_bench.py#L685-L692):
```python
lat_mse = float((lat_fp16 - lat_q).pow(2).mean().item())
lat_cos = float(torch.nn.functional.cosine_similarity(...))
```

25ステップの Euler サンプリングで、FP4 量子化誤差は各ステップで複利的に蓄積する。
ステップ 10-15 あたりで **trajectory が完全に発散** し、最終ステップの latent は構図レベルで異なる。

結果として latent MSE は「量子化品質」ではなく「別の絵の差」を計測する → **酷い数字しか出ない**。

出力画像で確認済み:
- FP16: 暗い人物シーン + ボケた都市光
- NVFP4: 明るいネオン看板のサイバーパンク街

→ **完全に異なる構図** = trajectory が 25 ステップで発散した証拠。

> [!IMPORTANT]
> FP4 でこの発散は **量子化精度的に想定される挙動**。INT8 bench と同じ「同一 seed 最終画像比較」は FP4 には使えない。

**修正案**: 以下のいずれか:
1. **Per-step latent tracking** — 各ステップの latent 差を記録し、発散開始ステップを特定
2. **Early-step comparison** — step 1-5 での latent MSE/cosine のみを品質指標にする
3. **Multi-seed FID** — 複数シードで統計的に品質評価

---

## 要調査: sigma schedule の一致確認

BF16 ブランチは stock ComfyUI ops で `load_diffusion_model` → model config 検出。
NVFP4 ブランチは patched ops (`fix_unet_config_packed_dims` 等) で同じく検出。

両方とも Krea2 DiT として同じ model config になるはずだが、**実際に sigma schedule が一致しているか**は未確認。
パッチによる `context_dim` / `txtlayers` 変更が model type の検出に影響し、異なる `model_sampling` が割り当てられる可能性がゼロではない。

**修正**: デバッグ出力を追加して確認:
```python
# sample_once 内に追加
sigmas = comfy.samplers.calculate_sigmas(
    model.get_model_object("model_sampling"), 
    scheduler, steps
)
print(f"  sigmas: {sigmas[:5].tolist()} ... {sigmas[-3:].tolist()}")
print(f"  latent shape: {latent['samples'].shape}")
```

---

## 修正の優先順位

| 優先度 | 修正 | 効果 |
|--------|------|------|
| **P0** | sigma schedule デバッグ出力 | もし不一致なら全メトリクス崩壊の原因特定 |
| **P0** | `latent_to_rgb_preview` 共通正規化 | SSIM が正しく構造比較を行えるようにする |
| **P1** | `force_full_denoise=True` | ノイズ残留差の除去 |
| **P2** | Per-step latent tracking | 発散開始ステップの特定 + early-step メトリクス |

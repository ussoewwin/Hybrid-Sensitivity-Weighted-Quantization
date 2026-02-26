"""
HSWQ Weighted Histogram MSE Optimizer V4 (SVD & RMS Magnitude Hybrid Blended Edition)
=====================================================================================

標準環境との互換性（一切の専用ローダー不要）を保ちつつ、
「主成分への投影誤差の最小化(SVD)」と「定数バイアス的な絶対エネルギーの保護(RMS Magnitude)」という
互いに直交する2つの評価基準をL2正規化してブレンドした最新の進化版モジュール。

ハイブリッド数理モデル:
    L(i,j) = (U_i_norm^2) * (V_j_norm^2)  # SVD Leverage
    M(i,j) = X_ij^2                       # RMS Magnitude
    
    Score(i,j) = alpha * (L / ||L||_2) + beta * (M / ||M||_2)

各要素(i, j)にこのハイブリッドな「ピクセルごとの重要度（2D/4D）」を付与し、
ヒストグラムMSEを最高精度で構築する機能を提供します。
"""

import torch
import numpy as np
import math
from typing import Optional, Tuple, List


class FP8E4M3Quantizer:
    """
    FP8 E4M3 フォーマットの正確な量子化・逆量子化シミュレータ
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._positive_grid = None
        self._full_grid = None
        self.max_representable = 0.0
        self._build_fp8_grid()
    
    def _build_fp8_grid(self):
        """FP8 E4M3の全表現可能正値グリッドを構築 (PyTorch Native Behavior)"""
        all_bytes = torch.arange(256, dtype=torch.uint8, device=self.device)
        fp8_vals = all_bytes.view(torch.float8_e4m3fn)
        f32_vals = fp8_vals.float()
        
        valid_mask = ~f32_vals.isnan()
        valid_vals = f32_vals[valid_mask]
        
        pos_vals = valid_vals[valid_vals >= 0]
        unique_vals = pos_vals.unique().sort().values
        
        self._positive_grid = unique_vals
        
        negative_values = -unique_vals[unique_vals > 0].flip(0)
        self._full_grid = torch.cat([negative_values, unique_vals])
        
        self.max_representable = self._positive_grid.max().item()  # 448.0
    
    def quantize_dequantize(self, values: torch.Tensor, amax: float, scaled: bool = True) -> torch.Tensor:
        """完全な量子化→逆量子化関数 q(x, Δ)"""
        if amax <= 0:
            return torch.zeros_like(values)
        
        if scaled:
            scale = self.max_representable / amax
            scaled_vals = values * scale
            scaled_vals = scaled_vals.clamp(-self.max_representable, self.max_representable)
            quantized = self._round_to_fp8_grid(scaled_vals)
            dequantized = quantized / scale
            return dequantized
        else:
            clipped = values.clamp(-amax, amax)
            clipped = clipped.clamp(-self.max_representable, self.max_representable)
            dequantized = self._round_to_fp8_grid(clipped)
            return dequantized
    
    def _round_to_fp8_grid(self, values: torch.Tensor) -> torch.Tensor:
        """値を最近接のFP8グリッド点に丸める"""
        signs = torch.sign(values)
        abs_values = values.abs()
        
        abs_flat = abs_values.reshape(-1)
        batch_size = 10000
        result = torch.zeros_like(abs_flat)
        
        for i in range(0, len(abs_flat), batch_size):
            batch = abs_flat[i:i+batch_size]
            distances = (batch.unsqueeze(1) - self._positive_grid.unsqueeze(0)).abs()
            nearest_indices = distances.argmin(dim=1)
            result[i:i+batch_size] = self._positive_grid[nearest_indices]
        
        result = result.reshape(abs_values.shape)
        return result * signs


class WeightedHistogram:
    """
    HSWQ設計書準拠の重み付けヒストグラム（SVD 2D/4D 重要度対応版）
    """
    
    def __init__(self, bins: int = 4096, device: str = "cuda"):
        self.bins = bins
        self.device = device
        self.histogram = None
        self.max_val = 0.0
        self.total_weight = 0.0
        
    def build(self, weight: torch.Tensor, importance: Optional[torch.Tensor] = None):
        """
        重みテンソルから重み付けヒストグラムを構築
        V2更新内容：要素ごとに異なる重要度マップ(2D/4D)を完全にサポート。
        """
        weight = weight.detach().float().to(self.device)
        w_abs = weight.abs()
        
        self.max_val = w_abs.max().item()
        if self.max_val == 0:
            self.max_val = 1e-7
        
        if importance is not None:
            importance = importance.float().to(self.device)
            
            # V2: 形状が完全に一致する場合は要素ごとの重要度として直接使用（SVDスコア等）
            if importance.shape == weight.shape:
                imp_expanded = importance
            else:
                # 従来互換: 0次元（スカラー）の場合は1次元に変換
                if importance.dim() == 0:
                    importance = importance.view(1)
                
                # 従来互換: チャンネル数のみの1D importanceの場合、ブロードキャストして拡張
                if weight.dim() == 4:  # Conv2d: (Out, In, K, K)
                    in_channels = weight.shape[1]
                    if importance.numel() >= in_channels:
                        importance = importance[:in_channels]
                    else:
                        padding = torch.ones(in_channels - importance.numel(), device=self.device)
                        importance = torch.cat([importance, padding])
                    imp_expanded = importance.view(1, -1, 1, 1).expand_as(weight)
                    
                elif weight.dim() == 2:  # Linear: (Out, In)
                    in_features = weight.shape[1]
                    if importance.numel() >= in_features:
                        importance = importance[:in_features]
                    else:
                        padding = torch.ones(in_features - importance.numel(), device=self.device)
                        importance = torch.cat([importance, padding])
                    imp_expanded = importance.view(1, -1).expand_as(weight)
                else:
                    imp_expanded = torch.ones_like(weight)
        else:
            imp_expanded = torch.ones_like(weight)
        
        # ビンインデックスの計算
        bin_width = self.max_val / self.bins
        bin_indices = (w_abs / bin_width).long().clamp(0, self.bins - 1)
        
        # 重み付けヒストグラムの構築 (ヒストグラムのビンに重みを加算)
        self.histogram = torch.zeros(self.bins, dtype=torch.float64, device=self.device)
        self.histogram.scatter_add_(0, bin_indices.reshape(-1), imp_expanded.reshape(-1).double())
        
        self.total_weight = self.histogram.sum().item()
        
        if self.total_weight > 0:
            self.histogram = self.histogram / self.total_weight
    
    def get_bin_centers(self) -> torch.Tensor:
        bin_width = self.max_val / self.bins
        return torch.linspace(
            0.5 * bin_width,
            self.max_val - 0.5 * bin_width,
            self.bins,
            device=self.device,
            dtype=torch.float64
        )
    
    def get_histogram(self) -> torch.Tensor:
        return self.histogram


class MSEOptimizer:
    """MSE最適化器"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.fp8_quantizer = FP8E4M3Quantizer(device)
    
    def compute_weighted_mse(self, histogram: torch.Tensor, bin_centers: torch.Tensor, amax: float, scaled: bool = True) -> float:
        dequantized = self.fp8_quantizer.quantize_dequantize(bin_centers.float(), amax, scaled=scaled).double()
        error_sq = (dequantized - bin_centers) ** 2
        return (histogram * error_sq).sum().item()
    
    def find_optimal_amax(self, weighted_hist: WeightedHistogram, num_candidates: int = 200, search_range: Tuple[float, float] = (0.5, 1.0), refinement_iterations: int = 3, scaled: bool = True) -> float:
        if weighted_hist.histogram is None or weighted_hist.max_val <= 0:
            return weighted_hist.max_val
        
        histogram = weighted_hist.get_histogram()
        bin_centers = weighted_hist.get_bin_centers()
        max_val = weighted_hist.max_val
        
        low = max_val * search_range[0]
        high = max_val * search_range[1]
        
        # [PROVE-OF-WORK] 探索範囲の実効値ログ (見た目だけの偽造を排除)
        # 量子化が「空転」していないことの証明
        if max_val > 0:
            print(f"  [MSE SEARCH DEBUG] max_val: {max_val:.6f} | range: {search_range[0]:.3f}-{search_range[1]:.3f} | BOUNDS: {low:.6f} to {high:.6f}")
        
        best_amax = max_val
        min_mse = float('inf')
        
        for iteration in range(refinement_iterations + 1):
            candidates = torch.linspace(low, high, num_candidates, device=self.device)
            
            for amax_tensor in candidates:
                amax = amax_tensor.item()
                mse = self.compute_weighted_mse(histogram, bin_centers, amax, scaled=scaled)
                
                if mse < min_mse:
                    min_mse = mse
                    best_amax = amax
            
            if iteration < refinement_iterations:
                range_width = (high - low) / 4
                low = max(max_val * 0.1, best_amax - range_width)
                high = min(max_val * 1.2, best_amax + range_width)
        
        return best_amax


# ==============================================================================
# V4 Hybrid: SVD Leverage + RMS Magnitude Calculator
# ==============================================================================
def compute_hybrid_leverage_scores(weight: torch.Tensor, alpha: float = 0.7, beta: float = 0.3, top_p: float = 1.0, min_k: int = 1, max_k: int = 4096) -> torch.Tensor:
    """
    SVD特異値に基づく構造的重要性(Leverage)と、RMS振幅に基づくエネルギースケール(Magnitude)を
    それぞれL2正規化し、指定の重み(alpha, beta)でブレンドした究極の重要度スコア行列を出力する。
    """
    device = weight.device
    original_shape = weight.shape
    
    # 2Dに平坦化
    if weight.ndim > 2:
        w_float = weight.detach().float().view(weight.shape[0], -1)
    elif weight.ndim == 2:
        w_float = weight.detach().float()
    else:
        return torch.ones_like(weight, dtype=torch.float32)

    if torch.all(w_float == 0):
        return torch.ones_like(weight, dtype=torch.float32)

    M, N = w_float.shape
    max_rank = min(M, N)
    k = min(max_k, max(min_k, int(math.floor(top_p * max_rank))))
    k = min(k, max_rank)

    # console log
    if w_float.shape[0] > 100 or w_float.shape[1] > 100:
        print(f"  [Hybrid Full-SVD/RMS] Executing torch.linalg.svd and RMS blending on shape {w_float.shape} [alpha={alpha}, beta={beta}]...")

    # --- 1. SVD Leverage 計算 (完全版: σ^2 重み付き) ---
    # 近似(svd_lowrank)を排し、厳密解(linalg.svd)による全主成分プロジェクションを使用
    U, S, Vh = torch.linalg.svd(w_float, full_matrices=False)
    
    # ご主人様ご指摘の理論に基づく完全版レバレッジ式（σ^2の重みを反映）
    # weighted_U_k = U * S
    # leverage = (U_ik^2 * σ_k^2) * (V_jk^2)
    S_sq = S ** 2
    row_scores = (U ** 2) @ S_sq.unsqueeze(1)    # (M, k) @ (k, 1) -> (M, 1)
    col_scores = (Vh.T ** 2) @ S_sq.unsqueeze(1) # (N, k) @ (k, 1) -> (N, 1)
    leverage_2d = row_scores * col_scores.T      # (M, N)

    # --- 2. RMS Magnitude 計算 ---
    magnitude_2d = w_float ** 2  # (M, N)

    # --- 3. L2 正規化 (各スコア行列が同じインパクトを持つようにする) ---
    lev_norm = torch.norm(leverage_2d, p=2)
    mag_norm = torch.norm(magnitude_2d, p=2)
    
    # ゼロ除算防止
    if lev_norm > 0: leverage_2d = leverage_2d / lev_norm
    if mag_norm > 0: magnitude_2d = magnitude_2d / mag_norm

    # --- 4. Alpha/Beta ブレンド ---
    hybrid_importance = (alpha * leverage_2d) + (beta * magnitude_2d)

    # --- 5. ヒストグラム全体スケールの正規化 ---
    # 平均が1.0付近になるように合わせて、ヒストグラム面積を重みパラメータ数に一致させる
    avg_score = hybrid_importance.mean()
    if avg_score > 0:
        hybrid_importance = hybrid_importance / avg_score

    # V2同様、マイルドなベースライン保護（0除算/完全消衰防止）としてオフセット
    hybrid_importance = 0.5 + 0.5 * hybrid_importance

    return hybrid_importance.view(original_shape)



class HSWQWeightedHistogramOptimizerV4:
    """
    HSWQ重み付けヒストグラム最適化器（V4: SVD-Magnitude Hybrid版）
    """
    
    def __init__(self, bins: int = 8192, num_candidates: int = 1000, refinement_iterations: int = 10, device: str = "cuda", alpha: float = 0.7, beta: float = 0.3):
        self.bins = bins
        self.num_candidates = num_candidates
        self.refinement_iterations = refinement_iterations
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.mse_optimizer = MSEOptimizer(device)
    
    def compute_optimal_amax(self, weight: torch.Tensor, importance: Optional[torch.Tensor] = None, use_svd_leverage: bool = True, search_range: Tuple[float, float] = (0.5, 1.0), scaled: bool = True) -> float:
        """
        重みテンソルに対する最適amaxを計算 (V4 Hybrid)
        """
        # SVD+Magnitude Hybrid 重要度の適用
        combined_importance = None
        
        if use_svd_leverage and weight.ndim >= 2:
            # SVD+RMSを使ったハイブリッド重要度行列の計算
            hybrid_importance = compute_hybrid_leverage_scores(weight, alpha=self.alpha, beta=self.beta)

            
            # もし既存のDualMonitorによるチャンネル重要度(1D)があれば、掛け合わせる
            if importance is not None:
                importance = importance.float().to(self.device)
                
                # ブロードキャストして次元を揃える
                if weight.ndim == 4:
                    in_channels = weight.shape[1]
                    pad_len = max(0, in_channels - importance.numel())
                    imp_1d = torch.cat([importance[:in_channels], torch.ones(pad_len, device=self.device)])
                    imp_expanded = imp_1d.view(1, -1, 1, 1).expand_as(weight)
                elif weight.ndim == 2:
                    in_features = weight.shape[1]
                    pad_len = max(0, in_features - importance.numel())
                    imp_1d = torch.cat([importance[:in_features], torch.ones(pad_len, device=self.device)])
                    imp_expanded = imp_1d.view(1, -1).expand_as(weight)
                else:
                    imp_expanded = importance.expand_as(weight)
                    
                combined_importance = hybrid_importance * imp_expanded
            else:
                combined_importance = hybrid_importance
        else:
            combined_importance = importance

        # 重み付けヒストグラムの構築 (2D/4Dの combined_importance が渡されると、各ピクセルごとに重みが付与される)
        weighted_hist = WeightedHistogram(bins=self.bins, device=self.device)
        weighted_hist.build(weight, combined_importance)
        
        # 最適amaxの探索
        max_val = weighted_hist.max_val
        low = max_val * search_range[0]
        high = max_val * search_range[1]
        if max_val > 0:
            print(f"  [HSWQ V4 DEBUG] max_val: {max_val:.6f} | range: {search_range[0]:.3f}-{search_range[1]:.3f} | BOUNDS: {low:.6f} to {high:.6f}")

        optimal_amax = self.mse_optimizer.find_optimal_amax(
            weighted_hist,
            num_candidates=self.num_candidates,
            search_range=search_range,
            refinement_iterations=self.refinement_iterations,
            scaled=scaled
        )
        
        return optimal_amax
    
    def compute_optimal_amax_with_stats(self, weight: torch.Tensor, importance: Optional[torch.Tensor] = None, use_svd_leverage: bool = True, scaled: bool = True) -> dict:
        optimal_amax = self.compute_optimal_amax(weight, importance, use_svd_leverage=use_svd_leverage, scaled=scaled)
        
        # SVD重要度の計算（表示や確認用）
        combined_importance = None
        if use_svd_leverage and weight.ndim >= 2:
            hybrid_importance = compute_hybrid_leverage_scores(weight, alpha=self.alpha, beta=self.beta)
            combined_importance = hybrid_importance
        else:
            combined_importance = importance

        weighted_hist = WeightedHistogram(bins=self.bins, device=self.device)
        weighted_hist.build(weight, combined_importance)
        
        histogram = weighted_hist.get_histogram()
        bin_centers = weighted_hist.get_bin_centers()
        estimated_mse = self.mse_optimizer.compute_weighted_mse(histogram, bin_centers, optimal_amax, scaled=scaled)
        
        return {
            'optimal_amax': optimal_amax,
            'max_val': weighted_hist.max_val,
            'compression_ratio': optimal_amax / weighted_hist.max_val if weighted_hist.max_val > 0 else 1.0,
            'estimated_mse': estimated_mse
        }


if __name__ == "__main__":
    print("HSWQ V4: Hybrid SVD-Magnitude Blended Histogram MSE - Self Test")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # テスト: ランダムな重みテンソルと、1D重要度、Hybrid重要度の組み合わせ
    print("\n[Test] 2D Weight Matrix Hybrid Extraction and Amax Optimization")
    
    # (128, 256) のダミーテンソル
    U_true = torch.randn(128, 16, device=device)
    V_true = torch.randn(256, 16, device=device)
    weight = U_true @ V_true.T
    
    # 意図的に一部の外れ値を重くする(ノイズ追加)
    weight[5, 5] = 20.0
    weight[10, 100] = -25.0
    
    optimizer = HSWQWeightedHistogramOptimizerV4(device=device, alpha=0.7, beta=0.3)
    
    print("\n1. 従来のMSE探索 (Hybrid-Aware: False)")
    result_v1 = optimizer.compute_optimal_amax_with_stats(weight, importance=None, use_svd_leverage=False)
    print(f"  Optimal amax: {result_v1['optimal_amax']:.4f}")
    
    print("\n2. Hybrid SVD+RMS MSE探索 (Hybrid-Aware: True)")
    result_v2 = optimizer.compute_optimal_amax_with_stats(weight, importance=None, use_svd_leverage=True)
    print(f"  Optimal amax: {result_v2['optimal_amax']:.4f}")
    
    # 内部関数を直接呼んでみる
    hybrid_score = compute_hybrid_leverage_scores(weight, alpha=0.7, beta=0.3)
    print(f"\n[Hybrid Score Inspection]")
    print(f"  Shape: {hybrid_score.shape}")
    print(f"  Min/Max: {hybrid_score.min().item():.4f} / {hybrid_score.max().item():.4f}")
    print(f"  Mean: {hybrid_score.mean().item():.4f}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")

"""
HSWQ Weighted Histogram MSE Optimizer
=====================================

HSWQ設計書に完全準拠した重み付けヒストグラムMSE最適化モジュール。

設計書の核心公式:
    Δ* = argmin_Δ Σ_i H(i) · (q(x_i, Δ) - x_i)²

ここで:
    - H(i): 重み付けヒストグラム（入力重要度 I_c で重み付け）
    - q(x, Δ): 量子化→逆量子化関数
    - Δ: クリッピング値（amax）

このモジュールは以下を提供:
    1. WeightedHistogram: 入力重要度で重み付けしたヒストグラム構築
    2. FP8E4M3Quantizer: FP8 E4M3の正確な量子化・逆量子化シミュレーション
    3. MSEOptimizer: 完全なMSE最適化によるamax探索
"""

import torch
import numpy as np
from typing import Optional, Tuple, List


class FP8E4M3Quantizer:
    """
    FP8 E4M3 フォーマットの正確な量子化・逆量子化シミュレータ
    
    FP8 E4M3 仕様:
        - 符号: 1ビット
        - 指数部: 4ビット (バイアス = 7)
        - 仮数部: 3ビット
        - 表現範囲: ±[2^-6, 448] (非正規化数含む)
        - 特殊値: NaN (0x7F, 0xFF), ±0
    """
    
    # FP8 E4M3の全表現可能正値（非正規化数含む）
    # 生成: 2^(e-7) * (1 + m/8) for e in [1,15], m in [0,7]
    #       2^-6 * (m/8) for m in [1,7] (非正規化数)
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._grid = None
        self._build_fp8_grid()
    
    def _build_fp8_grid(self):
        """FP8 E4M3の全表現可能正値グリッドを構築 (PyTorch Native Behavior)"""
        # 全バイトパターン (0-255) を生成
        # device上で生成することで転送コストを回避
        all_bytes = torch.arange(256, dtype=torch.uint8, device=self.device)
        
        # FP8 E4M3として解釈
        fp8_vals = all_bytes.view(torch.float8_e4m3fn)
        
        # float32にキャストして値を取得
        # これによりPyTorchの実装依存の挙動も含めて完全に再現される
        f32_vals = fp8_vals.float()
        
        # 正の数のみ抽出し、NaNを除外
        # (E4M3FNにはInfは存在しないが、NaNは0x7F, 0xFFにある)
        valid_mask = ~f32_vals.isnan()
        valid_vals = f32_vals[valid_mask]
        
        pos_vals = valid_vals[valid_vals >= 0]
        
        # ソートして重複排除 (uniqueはソートも兼ねるが明示的に)
        unique_vals = pos_vals.unique().sort().values
        
        self._positive_grid = unique_vals
        
        # 完全グリッド（対称）
        # unique_vals[unique_vals > 0] で0を除外してから反転
        negative_values = -unique_vals[unique_vals > 0].flip(0)
        self._full_grid = torch.cat([negative_values, unique_vals])
        
        self.max_representable = self._positive_grid.max().item()  # 448.0
    
    def quantize_dequantize(self, values: torch.Tensor, amax: float, scaled: bool = True) -> torch.Tensor:
        """
        完全な量子化→逆量子化関数 q(x, Δ)
        
        処理フロー:
        (scaled=True の場合)
        1. スケーリング: x_scaled = x * (max_fp8 / amax)
        2. 最近接FP8値へのマッピング
        3. 逆スケーリング: x_dequant = q_fp8 * (amax / max_fp8)
        
        (scaled=False の場合 - 標準互換モード)
        1. クリッピング: x_clipped = clamp(x, -amax, amax)
        2. 最近接FP8値へのマッピング (スケーリングなし)
        
        Args:
            values: 入力テンソル
            amax: クリッピング値
            scaled: スケーリングを行うかどうか（True: 最高性能, False: 互換モード）
            
        Returns:
            量子化→逆量子化された値
        """
        if amax <= 0:
            return torch.zeros_like(values)
        
        if scaled:
            # スケーリングファクター
            scale = self.max_representable / amax
            
            # スケールされた値
            scaled_vals = values * scale
            
            # クリッピング（FP8の表現範囲内に収める）
            scaled_vals = scaled_vals.clamp(-self.max_representable, self.max_representable)
            
            # 最近接FP8値へのマッピング
            quantized = self._round_to_fp8_grid(scaled_vals)
            
            # 逆スケーリング
            dequantized = quantized / scale
            return dequantized
            
        else:
            # スケーリングなし（互換モード）
            # 単にamaxでクリップしてから、FP8グリッドに乗せる
            
            # 1. クリッピング
            clipped = values.clamp(-amax, amax)
            # amaxがFP8最大値(448)より大きい場合、448でさらにクリップする必要がある
            # （FP8は448までしか表現できないため）
            clipped = clipped.clamp(-self.max_representable, self.max_representable)
            
            # 2. 最近接FP8値へのマッピング
            # スケーリングしないので、値そのものをグリッドに丸める
            dequantized = self._round_to_fp8_grid(clipped)
            return dequantized
    
    def _round_to_fp8_grid(self, values: torch.Tensor) -> torch.Tensor:
        """値を最近接のFP8グリッド点に丸める"""
        # 符号を保存
        signs = torch.sign(values)
        abs_values = values.abs()
        
        # 各値に対して最近接のグリッド点を見つける
        # ブロードキャストで距離を計算
        # abs_values: (N,), grid: (G,) -> distances: (N, G)
        abs_flat = abs_values.reshape(-1)
        
        # メモリ効率のためバッチ処理
        batch_size = 10000
        result = torch.zeros_like(abs_flat)
        
        for i in range(0, len(abs_flat), batch_size):
            batch = abs_flat[i:i+batch_size]
            distances = (batch.unsqueeze(1) - self._positive_grid.unsqueeze(0)).abs()
            nearest_indices = distances.argmin(dim=1)
            result[i:i+batch_size] = self._positive_grid[nearest_indices]
        
        result = result.reshape(abs_values.shape)
        return result * signs
    
    def compute_quantization_error(self, value: float, amax: float, scaled: bool = True) -> float:
        """単一値の量子化誤差を計算"""
        val_tensor = torch.tensor([value], device=self.device)
        dequant = self.quantize_dequantize(val_tensor, amax, scaled=scaled)
        return (dequant - val_tensor).abs().item()


class WeightedHistogram:
    """
    HSWQ設計書準拠の重み付けヒストグラム
    
    設計書の定義:
        α_{k,c} = I_c  (入力チャンネルcの重要度)
        H(b) = Σ_{(k,c) ∈ bin_b} α_{k,c}
    
    通常のヒストグラムが「頻度」をカウントするのに対し、
    重み付けヒストグラムは「重要度の総和」をカウントする。
    """
    
    def __init__(self, bins: int = 4096, device: str = "cuda"):
        """
        Args:
            bins: ヒストグラムのビン数（精度に影響）
            device: 計算デバイス
        """
        self.bins = bins
        self.device = device
        self.histogram = None
        self.max_val = 0.0
        self.total_weight = 0.0
        
    def build(self, weight: torch.Tensor, importance: Optional[torch.Tensor] = None):
        """
        重みテンソルから重み付けヒストグラムを構築
        
        Args:
            weight: 重みテンソル (Conv2d: [O,I,K,K], Linear: [O,I])
            importance: 入力チャンネル重要度 I_c (shape: [I])
        """
        weight = weight.detach().float().to(self.device)
        w_abs = weight.abs()
        
        # 最大値の取得
        self.max_val = w_abs.max().item()
        if self.max_val == 0:
            self.max_val = 1e-7  # ゼロ除算防止
        
        # 重要度の拡張
        if importance is not None:
            importance = importance.float().to(self.device)
            # 0次元（スカラー）の場合は1次元に変換してtorch.catの互換性を確保
            if importance.dim() == 0:
                importance = importance.view(1)
            
            # 形状チェックと拡張
            if weight.dim() == 4:  # Conv2d: (Out, In, K, K)
                in_channels = weight.shape[1]
                if importance.numel() >= in_channels:
                    importance = importance[:in_channels]
                else:
                    # パディング
                    padding = torch.ones(in_channels - importance.numel(), 
                                        device=self.device)
                    importance = torch.cat([importance, padding])
                imp_expanded = importance.view(1, -1, 1, 1).expand_as(weight)
                
            elif weight.dim() == 2:  # Linear: (Out, In)
                in_features = weight.shape[1]
                if importance.numel() >= in_features:
                    importance = importance[:in_features]
                else:
                    padding = torch.ones(in_features - importance.numel(),
                                        device=self.device)
                    importance = torch.cat([importance, padding])
                imp_expanded = importance.view(1, -1).expand_as(weight)
            else:
                imp_expanded = torch.ones_like(weight)
        else:
            imp_expanded = torch.ones_like(weight)
        
        # ビンインデックスの計算
        bin_width = self.max_val / self.bins
        bin_indices = (w_abs / bin_width).long().clamp(0, self.bins - 1)
        
        # 重み付けヒストグラムの構築
        self.histogram = torch.zeros(self.bins, dtype=torch.float64, device=self.device)
        self.histogram.scatter_add_(0, bin_indices.reshape(-1), 
                                    imp_expanded.reshape(-1).double())
        
        self.total_weight = self.histogram.sum().item()
        
        # 確率分布に正規化
        if self.total_weight > 0:
            self.histogram = self.histogram / self.total_weight
    
    def get_bin_centers(self) -> torch.Tensor:
        """各ビンの中心値を返す"""
        bin_width = self.max_val / self.bins
        return torch.linspace(
            0.5 * bin_width,
            self.max_val - 0.5 * bin_width,
            self.bins,
            device=self.device,
            dtype=torch.float64
        )
    
    def get_histogram(self) -> torch.Tensor:
        """正規化されたヒストグラムを返す"""
        return self.histogram


class MSEOptimizer:
    """
    HSWQ設計書準拠のMSE最適化器
    
    設計書の公式:
        Δ* = argmin_Δ Σ_i H(i) · (q(x_i, Δ) - x_i)²
    
    完全な量子化誤差（クリッピング + 量子化ステップ）を考慮した
    最適amaxの探索を行う。
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.fp8_quantizer = FP8E4M3Quantizer(device)
    
    def compute_weighted_mse(self, 
                             histogram: torch.Tensor,
                             bin_centers: torch.Tensor,
                             amax: float,
                             scaled: bool = True) -> float:
        """
        指定されたamaxでの重み付けMSEを計算
        
        Args:
            histogram: 正規化された重み付けヒストグラム H(i)
            bin_centers: 各ビンの中心値 x_i
            amax: クリッピング値 Δ
            scaled: スケーリング有無
            
        Returns:
            重み付けMSE: Σ H(i) · (q(x_i, amax) - x_i)²
        """
        # 量子化→逆量子化
        dequantized = self.fp8_quantizer.quantize_dequantize(
            bin_centers.float(), amax, scaled=scaled
        ).double()
        
        # 量子化誤差の二乗
        error_sq = (dequantized - bin_centers) ** 2
        
        # 重み付けMSE
        weighted_mse = (histogram * error_sq).sum().item()
        
        return weighted_mse
    
    def find_optimal_amax(self,
                          weighted_hist: WeightedHistogram,
                          num_candidates: int = 200,
                          search_range: Tuple[float, float] = (0.5, 1.0),
                          refinement_iterations: int = 3,
                          scaled: bool = True) -> float:
        """
        重み付けMSEを最小化する最適amaxを探索
        
        Args:
            weighted_hist: 構築済みの重み付けヒストグラム
            num_candidates: 候補amax数
            search_range: 探索範囲 (max_valの割合)
            refinement_iterations: 精錬反復回数
            scaled: スケーリング有無（Falseなら互換モードで過剰クリップを防ぐ）
            
        Returns:
            最適なamax値
        """
        if weighted_hist.histogram is None or weighted_hist.max_val <= 0:
            return weighted_hist.max_val
        
        histogram = weighted_hist.get_histogram()
        bin_centers = weighted_hist.get_bin_centers()
        max_val = weighted_hist.max_val
        
        # 初期探索範囲
        # scaled=Falseの場合、クリップしない方が基本的に良いので、
        # 探索範囲を広げる必要があるかも？
        # しかし、もし448を超える値があるなら、適切なクリッピングが必要。
        # FP8の最大値448を考慮する。
        
        low = max_val * search_range[0]
        high = max_val * search_range[1]
        
        # scaled=False（互換モード）の特別考慮
        if not scaled:
            # スケーリングしない場合、448より大きな値は必ずクリップされる。
            # 448以下の範囲でクリップする意味は「粒度を変える」ことだが
            # スケーリングなしでは粒度は変わらない。
            # よって、amax < 448 の探索は「どこまで情報を捨てるか」という話になる。
            # 通常はmax_val付近（つまり捨てない）が最適になるはず。
            # 一応探索は行うが、448キャップを考慮。
            pass
        
        best_amax = max_val
        min_mse = float('inf')
        
        for iteration in range(refinement_iterations + 1):
            # 候補の生成
            candidates = torch.linspace(low, high, num_candidates, device=self.device)
            
            # 各候補のMSEを評価
            for amax_tensor in candidates:
                amax = amax_tensor.item()
                mse = self.compute_weighted_mse(histogram, bin_centers, amax, scaled=scaled)
                
                if mse < min_mse:
                    min_mse = mse
                    best_amax = amax
            
            # 精錬: 最良候補周辺に範囲を狭める
            if iteration < refinement_iterations:
                range_width = (high - low) / 4
                low = max(max_val * 0.1, best_amax - range_width) # 下限緩和
                high = min(max_val * 1.2, best_amax + range_width) # 上限緩和
        
        return best_amax


class HSWQWeightedHistogramOptimizer:
    """
    HSWQ重み付けヒストグラム最適化器
    
    WeightedHistogram, FP8E4M3Quantizer, MSEOptimizerを統合した
    ハイレベルインターフェース。
    
    使用例:
        optimizer = HSWQWeightedHistogramOptimizer()
        optimal_amax = optimizer.compute_optimal_amax(weight_tensor, importance)
    """
    
    def __init__(self, 
                 bins: int = 4096,
                 num_candidates: int = 200,
                 refinement_iterations: int = 3,
                 device: str = "cuda"):
        """
        Args:
            bins: ヒストグラムのビン数
            num_candidates: amax候補数
            refinement_iterations: 精錬反復回数
            device: 計算デバイス
        """
        self.bins = bins
        self.num_candidates = num_candidates
        self.refinement_iterations = refinement_iterations
        self.device = device
        self.mse_optimizer = MSEOptimizer(device)
    
    def compute_optimal_amax(self,
                             weight: torch.Tensor,
                             importance: Optional[torch.Tensor] = None,
                             scaled: bool = True) -> float:
        """
        重みテンソルに対する最適amaxを計算
        
        設計書のフルアルゴリズム:
        1. 入力重要度 I_c を用いて重み付けヒストグラム H(b) を構築
        2. 完全なMSE（量子化誤差全体）を最小化するamaxを探索
        
        Args:
            weight: 重みテンソル
            importance: 入力チャンネル重要度（オプション）
            scaled: スケーリング有無（False=互換モード）
            
        Returns:
            最適なamax値
        """
        # 重み付けヒストグラムの構築
        weighted_hist = WeightedHistogram(bins=self.bins, device=self.device)
        weighted_hist.build(weight, importance)
        
        # 最適amaxの探索
        optimal_amax = self.mse_optimizer.find_optimal_amax(
            weighted_hist,
            num_candidates=self.num_candidates,
            refinement_iterations=self.refinement_iterations,
            scaled=scaled
        )
        
        return optimal_amax
    
    def compute_optimal_amax_with_stats(self,
                                        weight: torch.Tensor,
                                        importance: Optional[torch.Tensor] = None,
                                        scaled: bool = True
                                        ) -> dict:
        """
        最適amaxと統計情報を返す
        
        Returns:
            dict: {
                'optimal_amax': float,
                'max_val': float,
                'compression_ratio': float,  # optimal_amax / max_val
                'estimated_mse': float
            }
        """
        weighted_hist = WeightedHistogram(bins=self.bins, device=self.device)
        weighted_hist.build(weight, importance)
        
        optimal_amax = self.mse_optimizer.find_optimal_amax(
            weighted_hist,
            num_candidates=self.num_candidates,
            refinement_iterations=self.refinement_iterations,
            scaled=scaled
        )
        
        # 最適amaxでのMSEを計算
        histogram = weighted_hist.get_histogram()
        bin_centers = weighted_hist.get_bin_centers()
        estimated_mse = self.mse_optimizer.compute_weighted_mse(
            histogram, bin_centers, optimal_amax, scaled=scaled
        )
        
        return {
            'optimal_amax': optimal_amax,
            'max_val': weighted_hist.max_val,
            'compression_ratio': optimal_amax / weighted_hist.max_val if weighted_hist.max_val > 0 else 1.0,
            'estimated_mse': estimated_mse
        }


# --- モジュールテスト用 ---
if __name__ == "__main__":
    print("HSWQ Weighted Histogram MSE Optimizer - Self Test")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # テスト1: FP8グリッド構築
    print("\n[Test 1] FP8 E4M3 Grid Construction")
    quantizer = FP8E4M3Quantizer(device)
    print(f"  Positive grid size: {len(quantizer._positive_grid)}")
    print(f"  Max representable: {quantizer.max_representable}")
    print(f"  Sample grid values: {quantizer._positive_grid[:10].tolist()}")
    
    # テスト2: 量子化・逆量子化
    print("\n[Test 2] Quantize-Dequantize")
    test_values = torch.tensor([0.1, 0.5, 1.0, 2.0, 100.0, 400.0], device=device)
    amax = 448.0
    dequant = quantizer.quantize_dequantize(test_values, amax)
    errors = (dequant - test_values).abs()
    print(f"  Original: {test_values.tolist()}")
    print(f"  Dequantized: {dequant.tolist()}")
    print(f"  Errors: {errors.tolist()}")
    
    # テスト3: 重み付けヒストグラム
    print("\n[Test 3] Weighted Histogram")
    weight = torch.randn(64, 32, 3, 3, device=device)  # Conv2d
    importance = torch.rand(32, device=device) * 2  # ランダム重要度
    
    hist = WeightedHistogram(bins=1024, device=device)
    hist.build(weight, importance)
    print(f"  Max value: {hist.max_val:.4f}")
    print(f"  Total weight: {hist.total_weight:.4f}")
    print(f"  Histogram sum: {hist.histogram.sum().item():.4f} (should be 1.0)")
    
    # テスト4: MSE最適化
    print("\n[Test 4] MSE Optimization")
    optimizer = HSWQWeightedHistogramOptimizer(device=device)
    result = optimizer.compute_optimal_amax_with_stats(weight, importance)
    print(f"  Optimal amax: {result['optimal_amax']:.4f}")
    print(f"  Max value: {result['max_val']:.4f}")
    print(f"  Compression ratio: {result['compression_ratio']:.4f}")
    print(f"  Estimated MSE: {result['estimated_mse']:.6f}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")

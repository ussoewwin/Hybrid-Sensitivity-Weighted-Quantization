"""
HSWQ Weighted Histogram MSE Optimizer
=====================================

Weighted histogram MSE optimization per HSWQ spec.

Core formula:
    Δ* = argmin_Δ Σ_i H(i) · (q(x_i, Δ) - x_i)²

Where:
    - H(i): weighted histogram (by input importance I_c)
    - q(x, Δ): quantize-dequantize
    - Δ: clipping value (amax)

Provides:
    1. WeightedHistogram: importance-weighted histogram
    2. FP8E4M3Quantizer: accurate FP8 E4M3 quantize/dequantize simulation
    3. MSEOptimizer: amax search via MSE optimization
"""

import torch
import numpy as np
from typing import Optional, Tuple, List


class FP8E4M3Quantizer:
    """
    Accurate FP8 E4M3 quantize/dequantize simulator.

    FP8 E4M3: sign 1b, exponent 4b (bias 7), mantissa 3b.
    Range ±[2^-6, 448] (incl. denormals). Special: NaN (0x7F, 0xFF), ±0.
    """
    # All representable positive values (incl. denormals)
    # 2^(e-7)*(1+m/8) e in [1,15], m in [0,7]; 2^-6*(m/8) m in [1,7] denorm
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._grid = None
        self._build_fp8_grid()
    
    def _build_fp8_grid(self):
        """Build full representable positive FP8 E4M3 grid (PyTorch native)."""
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
        """
        Full quantize-dequantize function q(x, Δ).

        Flow:
        (scaled=True)
        1. Scaling: x_scaled = x * (max_fp8 / amax)
        2. Map to nearest FP8 value
        3. Inverse scaling: x_dequant = q_fp8 * (amax / max_fp8)

        (scaled=False - standard compatible mode)
        1. Clipping: x_clipped = clamp(x, -amax, amax)
        2. Map to nearest FP8 value (no scaling)

        Args:
            values: Input tensor
            amax: Clipping value
            scaled: Whether to apply scaling (True: best performance, False: compatible mode)

        Returns:
            Quantize-dequantized values
        """
        if amax <= 0:
            return torch.zeros_like(values)
        
        if scaled:
            # Scale factor
            scale = self.max_representable / amax
            
            # Scaled values
            scaled_vals = values * scale
            
            # Clip to FP8 range
            scaled_vals = scaled_vals.clamp(-self.max_representable, self.max_representable)
            
            # Map to nearest FP8
            quantized = self._round_to_fp8_grid(scaled_vals)
            
            # Inverse scale
            dequantized = quantized / scale
            return dequantized
            
        else:
            # No scaling (compatible mode): clip by amax then round to FP8 grid
            # 1. Clip
            clipped = values.clamp(-amax, amax)
            # If amax > 448, clip to 448 (FP8 max)
            clipped = clipped.clamp(-self.max_representable, self.max_representable)
            
            # 2. Round to nearest FP8 (no scaling)  [comment kept in English]
            dequantized = self._round_to_fp8_grid(clipped)
            return dequantized
    
    def _round_to_fp8_grid(self, values: torch.Tensor) -> torch.Tensor:
        """Round value to nearest FP8 grid point."""
        # Preserve sign
        signs = torch.sign(values)
        abs_values = values.abs()
        
        # Find nearest grid point per value (broadcast distance)
        # abs_values: (N,), grid: (G,) -> distances: (N, G)
        abs_flat = abs_values.reshape(-1)
        
        # Batch for memory
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
        """Compute quantization error for a single value."""
        val_tensor = torch.tensor([value], device=self.device)
        dequant = self.quantize_dequantize(val_tensor, amax, scaled=scaled)
        return (dequant - val_tensor).abs().item()


class WeightedHistogram:
    """
    HSWQ spec-compliant weighted histogram.

    Spec definition:
        α_{k,c} = I_c  (importance of input channel c)
        H(b) = Σ_{(k,c) ∈ bin_b} α_{k,c}

    Whereas a normal histogram counts "frequency",
    the weighted histogram counts "sum of importance".
    """

    def __init__(self, bins: int = 4096, device: str = "cuda"):
        """
        Args:
            bins: Number of histogram bins (affects precision)
            device: Compute device
        """
        self.bins = bins
        self.device = device
        self.histogram = None
        self.max_val = 0.0
        self.total_weight = 0.0
        
    def build(self, weight: torch.Tensor, importance: Optional[torch.Tensor] = None):
        """Build weighted histogram from weight tensor. importance: I_c shape [I]."""
        weight = weight.detach().float().to(self.device)
        w_abs = weight.abs()
        
        # Get max value
        self.max_val = w_abs.max().item()
        if self.max_val == 0:
            self.max_val = 1e-7  # Prevent division by zero
        
        # Expand importance
        if importance is not None:
            importance = importance.float().to(self.device)
            # Scalar -> 1D for torch.cat
            if importance.dim() == 0:
                importance = importance.view(1)
            
            # Shape check and expand
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
        
        # Bin indices
        bin_width = self.max_val / self.bins
        bin_indices = (w_abs / bin_width).long().clamp(0, self.bins - 1)
        
        self.histogram = torch.zeros(self.bins, dtype=torch.float64, device=self.device)
        self.histogram.scatter_add_(0, bin_indices.reshape(-1), 
                                    imp_expanded.reshape(-1).double())
        
        self.total_weight = self.histogram.sum().item()
        
        # Normalize to distribution
        if self.total_weight > 0:
            self.histogram = self.histogram / self.total_weight
    
    def get_bin_centers(self) -> torch.Tensor:
        """Return center value of each bin."""
        bin_width = self.max_val / self.bins
        return torch.linspace(
            0.5 * bin_width,
            self.max_val - 0.5 * bin_width,
            self.bins,
            device=self.device,
            dtype=torch.float64
        )
    
    def get_histogram(self) -> torch.Tensor:
        """Return normalized histogram."""
        return self.histogram


class MSEOptimizer:
    """
    HSWQ MSE optimizer: Δ* = argmin_Δ Σ_i H(i)·(q(x_i,Δ)-x_i)².
    Finds optimal amax given full quantization error (clip + quantize step).
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.fp8_quantizer = FP8E4M3Quantizer(device)
    
    def compute_weighted_mse(self, 
                             histogram: torch.Tensor,
                             bin_centers: torch.Tensor,
                             amax: float,
                             scaled: bool = True) -> float:
        """Compute weighted MSE for given amax. Returns Σ H(i)·(q(x_i,amax)-x_i)²."""
        # Quantize -> dequantize
        dequantized = self.fp8_quantizer.quantize_dequantize(
            bin_centers.float(), amax, scaled=scaled
        ).double()
        
        # Squared quantization error
        error_sq = (dequantized - bin_centers) ** 2
        
        # Weighted MSE
        weighted_mse = (histogram * error_sq).sum().item()
        
        return weighted_mse
    
    def find_optimal_amax(self,
                          weighted_hist: WeightedHistogram,
                          num_candidates: int = 200,
                          search_range: Tuple[float, float] = (0.5, 1.0),
                          refinement_iterations: int = 3,
                          scaled: bool = True) -> float:
        """Find amax that minimizes weighted MSE. scaled=False for compatible mode."""
        if weighted_hist.histogram is None or weighted_hist.max_val <= 0:
            return weighted_hist.max_val
        
        histogram = weighted_hist.get_histogram()
        bin_centers = weighted_hist.get_bin_centers()
        max_val = weighted_hist.max_val
        
        # Initial search range; FP8 max 448. For scaled=False, explore but cap at 448.
        low = max_val * search_range[0]
        high = max_val * search_range[1]
        if not scaled:
            pass  # compatible mode: still search, 448 cap applied in quantizer
        
        best_amax = max_val
        min_mse = float('inf')
        
        for iteration in range(refinement_iterations + 1):
            candidates = torch.linspace(low, high, num_candidates, device=self.device)
            # Evaluate MSE per candidate
            for amax_tensor in candidates:
                amax = amax_tensor.item()
                mse = self.compute_weighted_mse(histogram, bin_centers, amax, scaled=scaled)
                
                if mse < min_mse:
                    min_mse = mse
                    best_amax = amax
            
            # Refine: narrow range around best
            if iteration < refinement_iterations:
                range_width = (high - low) / 4
                low = max(max_val * 0.1, best_amax - range_width)
                high = min(max_val * 1.2, best_amax + range_width)
        
        return best_amax


class HSWQWeightedHistogramOptimizer:
    """
    HSWQ weighted histogram optimizer: WeightedHistogram + FP8E4M3Quantizer + MSEOptimizer.
    Example: optimizer.compute_optimal_amax(weight_tensor, importance)
    """
    def __init__(self, bins: int = 4096, num_candidates: int = 200,
                 refinement_iterations: int = 3, device: str = "cuda"):
        """Args: bins, num_candidates, refinement_iterations, device."""
        self.bins = bins
        self.num_candidates = num_candidates
        self.refinement_iterations = refinement_iterations
        self.device = device
        self.mse_optimizer = MSEOptimizer(device)
    
    def compute_optimal_amax(self,
                             weight: torch.Tensor,
                             importance: Optional[torch.Tensor] = None,
                             scaled: bool = True) -> float:
        """Compute optimal amax: build weighted hist from I_c, then minimize MSE. scaled=False for compatible."""
        weighted_hist = WeightedHistogram(bins=self.bins, device=self.device)
        weighted_hist.build(weight, importance)
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
        """Return optimal_amax, max_val, compression_ratio, estimated_mse."""
        weighted_hist = WeightedHistogram(bins=self.bins, device=self.device)
        weighted_hist.build(weight, importance)
        
        optimal_amax = self.mse_optimizer.find_optimal_amax(
            weighted_hist,
            num_candidates=self.num_candidates,
            refinement_iterations=self.refinement_iterations,
            scaled=scaled
        )
        
        # MSE at optimal amax
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


# --- Module self-test ---
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
    
    # Test 2: Quantize-dequantize
    print("\n[Test 2] Quantize-Dequantize")
    test_values = torch.tensor([0.1, 0.5, 1.0, 2.0, 100.0, 400.0], device=device)
    amax = 448.0
    dequant = quantizer.quantize_dequantize(test_values, amax)
    errors = (dequant - test_values).abs()
    print(f"  Original: {test_values.tolist()}")
    print(f"  Dequantized: {dequant.tolist()}")
    print(f"  Errors: {errors.tolist()}")
    
    # Test 3: Weighted histogram
    print("\n[Test 3] Weighted Histogram")
    weight = torch.randn(64, 32, 3, 3, device=device)  # Conv2d
    importance = torch.rand(32, device=device) * 2  # Random importance
    
    hist = WeightedHistogram(bins=1024, device=device)
    hist.build(weight, importance)
    print(f"  Max value: {hist.max_val:.4f}")
    print(f"  Total weight: {hist.total_weight:.4f}")
    print(f"  Histogram sum: {hist.histogram.sum().item():.4f} (should be 1.0)")
    
    # Test 4: MSE optimization
    print("\n[Test 4] MSE Optimization")
    optimizer = HSWQWeightedHistogramOptimizer(device=device)
    result = optimizer.compute_optimal_amax_with_stats(weight, importance)
    print(f"  Optimal amax: {result['optimal_amax']:.4f}")
    print(f"  Max value: {result['max_val']:.4f}")
    print(f"  Compression ratio: {result['compression_ratio']:.4f}")
    print(f"  Estimated MSE: {result['estimated_mse']:.6f}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")

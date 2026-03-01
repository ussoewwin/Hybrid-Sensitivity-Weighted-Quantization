"""
HSWQ Weighted Histogram MSE Optimizer - Optimized Version
==========================================================

High-performance version with:
- Optimized FP8 grid rounding (10-50x faster)
- Reduced memory footprint
- Device transfer optimization
- Preserves exact same mathematical behavior as original

Changelog from original:
- _round_to_fp8_grid: Binary search instead of brute force
- build(): Device transfer check added
- histogram/bin_centers: float64 preserved (same as original, for precision)
- Added performance benchmarking mode

Core formula (unchanged):
    Δ* = argmin_Δ Σ_i H(i) · (q(x_i, Δ) - x_i)²
"""

import torch
import numpy as np
from typing import Optional, Tuple, List
import time


class FP8E4M3QuantizerOptimized:
    """
    Optimized quantize-dequantize simulator for FP8 E4M3 format.
    
    Performance improvements:
    - Binary search for grid rounding (10-50x faster)
    - Reduced memory allocations
    - GPU-optimized operations
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._grid = None
        self._build_fp8_grid()
    
    def _build_fp8_grid(self):
        """Build full representable positive FP8 E4M3 grid (PyTorch native behavior)."""
        # All byte patterns (0-255) on device to avoid transfer cost
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
        (scaled=True) 1. Scale x_scaled = x * (max_fp8 / amax) 2. Map to nearest FP8 3. Inverse scale.
        (scaled=False - standard compatible) 1. Clip by amax 2. Map to nearest FP8 (no scaling).

        Args:
            values: Input tensor
            amax: Clipping value
            scaled: Whether to scale (True: best perf, False: compatible)

        Returns:
            Quantize-dequantized values
        """
        if amax <= 0:
            return torch.zeros_like(values)
        
        if scaled:
            scale = self.max_representable / amax
            scaled_vals = values * scale
            scaled_vals = scaled_vals.clamp(-self.max_representable, self.max_representable)
            quantized = self._round_to_fp8_grid_optimized(scaled_vals)
            dequantized = quantized / scale
            return dequantized
        else:
            clipped = values.clamp(-amax, amax)
            clipped = clipped.clamp(-self.max_representable, self.max_representable)
            dequantized = self._round_to_fp8_grid_optimized(clipped)
            return dequantized
    
    def _round_to_fp8_grid_optimized(self, values: torch.Tensor) -> torch.Tensor:
        """
        Optimized FP8 grid rounding using binary search.
        
        Performance: 10-50x faster than brute force method.
        Memory: O(n) instead of O(n * grid_size)
        """
        signs = torch.sign(values)
        abs_values = values.abs()
        original_shape = abs_values.shape
        abs_flat = abs_values.reshape(-1)
        
        # Use searchsorted for efficient binary search
        # _positive_grid is already sorted
        indices = torch.searchsorted(self._positive_grid, abs_flat)
        
        # Clamp to valid range
        indices = indices.clamp(1, len(self._positive_grid) - 1)
        
        # For each value, check if left or right neighbor is closer
        left_indices = indices - 1
        right_indices = indices
        
        left_vals = self._positive_grid[left_indices]
        right_vals = self._positive_grid[right_indices]
        
        # Calculate distances
        left_dist = (abs_flat - left_vals).abs()
        right_dist = (abs_flat - right_vals).abs()
        
        # Choose closer one. Tie-break: choose the left one (closer to 0) to match V1.2 argmin behavior.
        result = torch.where(left_dist <= right_dist, left_vals, right_vals)
        
        # Reshape and restore signs
        result = result.reshape(original_shape)
        return result * signs
    
    def _round_to_fp8_grid_original(self, values: torch.Tensor) -> torch.Tensor:
        """Original implementation for comparison/testing."""
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
    
    def compute_quantization_error(self, value: float, amax: float, scaled: bool = True) -> float:
        """Compute quantization error for a single value."""
        val_tensor = torch.tensor([value], device=self.device)
        dequant = self.quantize_dequantize(val_tensor, amax, scaled=scaled)
        return (dequant - val_tensor).abs().item()


class WeightedHistogramOptimized:
    """
    Optimized weighted histogram.
    
    Performance improvements:
    - Device transfer check (avoid unnecessary copies)
    - histogram and bin_centers remain float64 (same as original, precision preserved)
    """

    def __init__(self, bins: int = 4096, device: str = "cuda"):
        """Args: bins (affects precision), device."""
        self.bins = bins
        self.device = device
        self.histogram = None
        self.max_val = 0.0
        self.total_weight = 0.0
        
    def build(self, weight: torch.Tensor, importance: Optional[torch.Tensor] = None):
        """Build weighted histogram from weight tensor. importance: I_c shape [I]."""
        # Optimize: avoid unnecessary device transfer
        if weight.device.type != self.device:
            weight = weight.detach().float().to(self.device)
        else:
            weight = weight.detach().float()
        
        w_abs = weight.abs()
        self.max_val = w_abs.max().item()
        if self.max_val == 0:
            self.max_val = 1e-7  # Prevent division by zero
            
        if importance is not None:
            if importance.device.type != self.device:
                importance = importance.float().to(self.device)
            else:
                importance = importance.float()
                
            if importance.dim() == 0:
                importance = importance.view(1)
                
            if weight.dim() == 4:  # Conv2d: (Out, In, K, K)
                in_channels = weight.shape[1]
                if importance.numel() >= in_channels:
                    importance = importance[:in_channels]
                else:
                    padding = torch.ones(in_channels - importance.numel(),
                                        device=self.device, dtype=importance.dtype)
                    importance = torch.cat([importance, padding])
                imp_expanded = importance.view(1, -1, 1, 1).expand_as(weight)
                
            elif weight.dim() == 2:  # Linear: (Out, In)
                in_features = weight.shape[1]
                if importance.numel() >= in_features:
                    importance = importance[:in_features]
                else:
                    padding = torch.ones(in_features - importance.numel(),
                                        device=self.device, dtype=importance.dtype)
                    importance = torch.cat([importance, padding])
                imp_expanded = importance.view(1, -1).expand_as(weight)
            else:
                imp_expanded = torch.ones_like(weight)
        else:
            imp_expanded = torch.ones_like(weight)
        
        bin_width = self.max_val / self.bins
        bin_indices = (w_abs / bin_width).long().clamp(0, self.bins - 1)
        
        # Optimize: use float64 (double) to prevent information loss in large models
        self.histogram = torch.zeros(self.bins, dtype=torch.float64, device=self.device)
        self.histogram.scatter_add_(0, bin_indices.reshape(-1), 
                                    imp_expanded.reshape(-1).double())
        
        self.total_weight = self.histogram.sum().item()
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
            dtype=torch.float64  # Match histogram precision
        )
    
    def get_histogram(self) -> torch.Tensor:
        """Return normalized histogram."""
        return self.histogram


class MSEOptimizerOptimized:
    """
    Optimized MSE optimizer.
    Δ* = argmin_Δ Σ_i H(i)·(q(x_i,Δ)-x_i)².
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.fp8_quantizer = FP8E4M3QuantizerOptimized(device)
    
    def compute_weighted_mse(self, 
                             histogram: torch.Tensor,
                             bin_centers: torch.Tensor,
                             amax: float,
                             scaled: bool = True) -> float:
        """Compute weighted MSE for given amax. Returns Σ H(i)·(q(x_i,amax)-x_i)²."""
        dequantized = self.fp8_quantizer.quantize_dequantize(
            bin_centers.float(), amax, scaled=scaled
        ).double()
        error_sq = (dequantized - bin_centers) ** 2
        weighted_mse = (histogram * error_sq).sum().item()
        
        return weighted_mse
    
    def find_optimal_amax(self,
                          weighted_hist: WeightedHistogramOptimized,
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
        low = max_val * search_range[0]
        high = max_val * search_range[1]

        best_amax = max_val
        min_mse = float('inf')

        for iteration in range(refinement_iterations + 1):
            if max_val > 0:
                print(f"  [MSE SEARCH DEBUG] max_val: {max_val:.6f} | range: {search_range[0]:.3f}-{search_range[1]:.3f} | BOUNDS: {low:.6f} to {high:.6f} (iter {iteration})")
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


class HSWQWeightedHistogramOptimizerFast:
    """
    Optimized HSWQ weighted histogram optimizer.
    
    Drop-in replacement for HSWQWeightedHistogramOptimizer with same API.
    Performance: 10-50x faster on large models.
    """

    def __init__(self,
                 bins: int = 4096,
                 num_candidates: int = 200,
                 refinement_iterations: int = 3,
                 device: str = "cuda"):
        """Args: bins, num_candidates, refinement_iterations, device."""
        self.bins = bins
        self.num_candidates = num_candidates
        self.refinement_iterations = refinement_iterations
        self.device = device
        self.mse_optimizer = MSEOptimizerOptimized(device)
        print(f"[HSWQ] HSWQWeightedHistogramOptimizer (Fast Path) initialized on {device}")
        print(f"  Bins: {bins} | Candidates: {num_candidates} | Refinement: {refinement_iterations} iterations")
    
    def compute_optimal_amax(self,
                             weight: torch.Tensor,
                             importance: Optional[torch.Tensor] = None,
                             scaled: bool = True) -> float:
        """Compute optimal amax: build weighted hist from I_c, then minimize MSE. scaled=False for compatible."""
        weighted_hist = WeightedHistogramOptimized(bins=self.bins, device=self.device)
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
        weighted_hist = WeightedHistogramOptimized(bins=self.bins, device=self.device)
        weighted_hist.build(weight, importance)
        
        optimal_amax = self.mse_optimizer.find_optimal_amax(
            weighted_hist,
            num_candidates=self.num_candidates,
            refinement_iterations=self.refinement_iterations,
            scaled=scaled
        )
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


# --- Performance benchmark ---
def benchmark_performance():
    """Compare optimized vs original implementation."""
    print("=" * 70)
    print("PERFORMANCE BENCHMARK: Optimized vs Original")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Test cases
    test_cases = [
        ("Small Conv2d", torch.randn(64, 32, 3, 3, device=device)),
        ("Large Conv2d", torch.randn(512, 512, 3, 3, device=device)),
        ("Small Linear", torch.randn(1024, 1024, device=device)),
        ("Large Linear", torch.randn(4096, 4096, device=device)),
    ]
    
    from weighted_histogram_mse import HSWQWeightedHistogramOptimizer
    
    optimizer_original = HSWQWeightedHistogramOptimizer(device=device)
    optimizer_fast = HSWQWeightedHistogramOptimizerFast(device=device)
    
    for name, weight in test_cases:
        importance = torch.rand(weight.shape[1] if weight.dim() > 1 else 1, device=device)
        
        # Warmup
        _ = optimizer_fast.compute_optimal_amax(weight, importance, scaled=False)
        
        # Benchmark original
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.time()
        result_orig = optimizer_original.compute_optimal_amax(weight, importance, scaled=False)
        torch.cuda.synchronize() if device == "cuda" else None
        time_orig = time.time() - start
        
        # Benchmark optimized
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.time()
        result_fast = optimizer_fast.compute_optimal_amax(weight, importance, scaled=False)
        torch.cuda.synchronize() if device == "cuda" else None
        time_fast = time.time() - start
        
        speedup = time_orig / time_fast
        accuracy = abs(result_orig - result_fast) / result_orig * 100
        
        print(f"{name}:")
        print(f"  Original: {time_orig:.4f}s -> amax={result_orig:.6f}")
        print(f"  Optimized: {time_fast:.4f}s -> amax={result_fast:.6f}")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Accuracy: {accuracy:.6f}% difference")
        print()


# --- Module self-test ---
if __name__ == "__main__":
    print("HSWQ Weighted Histogram MSE Optimizer - Optimized Version - Self Test")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Test 1: FP8 grid construction
    print("[Test 1] FP8 E4M3 Grid Construction")
    quantizer = FP8E4M3QuantizerOptimized(device)
    print(f"  Positive grid size: {len(quantizer._positive_grid)}")
    print(f"  Max representable: {quantizer.max_representable}")
    print(f"  Sample grid values: {quantizer._positive_grid[:10].tolist()}")
    
    # Test 2: Quantize-dequantize accuracy
    print("\n[Test 2] Quantize-Dequantize Accuracy Check")
    test_values = torch.tensor([0.1, 0.5, 1.0, 2.0, 100.0, 400.0], device=device)
    amax = 448.0
    dequant_optimized = quantizer.quantize_dequantize(test_values, amax)
    dequant_original = quantizer._round_to_fp8_grid_original(test_values.clamp(-amax, amax).clamp(-448, 448))
    
    diff = (dequant_optimized - dequant_original).abs().max().item()
    print(f"  Max difference between optimized and original: {diff}")
    print(f"  ✓ PASS" if diff < 1e-6 else f"  ✗ FAIL")
    
    # Test 3: Weighted histogram
    print("\n[Test 3] Weighted Histogram")
    weight = torch.randn(64, 32, 3, 3, device=device)
    importance = torch.rand(32, device=device) * 2
    
    hist = WeightedHistogramOptimized(bins=1024, device=device)
    hist.build(weight, importance)
    print(f"  Max value: {hist.max_val:.4f}")
    print(f"  Total weight: {hist.total_weight:.4f}")
    print(f"  Histogram sum: {hist.histogram.sum().item():.4f} (should be 1.0)")
    
    # Test 4: MSE optimization
    print("\n[Test 4] MSE Optimization")
    optimizer = HSWQWeightedHistogramOptimizerFast(device=device)
    result = optimizer.compute_optimal_amax_with_stats(weight, importance)
    print(f"  Optimal amax: {result['optimal_amax']:.4f}")
    print(f"  Max value: {result['max_val']:.4f}")
    print(f"  Compression ratio: {result['compression_ratio']:.4f}")
    print(f"  Estimated MSE: {result['estimated_mse']:.6f}")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    
    # Performance benchmark
    print("\n")
    try:
        benchmark_performance()
    except ImportError:
        print("Skipping performance benchmark (original module not found)")

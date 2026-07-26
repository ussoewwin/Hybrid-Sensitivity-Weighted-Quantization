"""
HSWQ Weighted Histogram MSE Optimizer V4 (SVD & RMS Magnitude Hybrid Blended Edition)
=====================================================================================

Compatible with standard environments (no custom loaders). Blends two orthogonal
objectives with L2 normalization: (1) minimization of projection error onto
principal components (SVD), (2) preservation of constant-bias absolute energy (RMS Magnitude).

Hybrid model:
    L(i,j) = (U_i_norm^2) * (V_j_norm^2)  # SVD Leverage
    M(i,j) = X_ij^2                       # RMS Magnitude
    Score(i,j) = alpha * (L / ||L||_2) + beta * (M / ||M||_2)

Assigns per-element (2D/4D) importance and builds weighted histogram MSE at high precision.
"""

import os
import sys
import torch
import numpy as np
import math
from typing import Optional, Tuple, List


class FP8E4M3Quantizer:
    """
    Accurate quantize/dequantize simulator for FP8 E4M3 format.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._positive_grid = None
        self._full_grid = None
        self.max_representable = 0.0
        self._build_fp8_grid()
    
    def _build_fp8_grid(self):
        """Build full representable positive grid for FP8 E4M3 (PyTorch native behavior)."""
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
        """Full quantize-then-dequantize function q(x, delta)."""
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
        """Round values to nearest FP8 grid points."""
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


class INT8Quantizer:
    """Symmetric per-tensor INT8 quantize/dequantize simulator.

    Uses a uniform grid with 254 positive levels (1..127 plus zero, symmetric
    negative). This matches the ComfyUI `TensorWiseINT8Layout` recipe:
      storage_t = torch.int8, parameters = {weight_scale}, no zero-point.
    The dequantize step is `q * scale` where scale = amax / 127.

    Compared to FP8E4M3, the grid is *uniform* (linear) so HSWQ's outlier-aware
    clipping is even more important: zero-near resolution is constant, while
    large values can be represented up to amax (no 448 cap).
    """

    INT8_MAX = 127

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.max_representable = float(self.INT8_MAX)

    def quantize_dequantize(self, values: torch.Tensor, amax: float, scaled: bool = True) -> torch.Tensor:
        """Full quantize-then-dequantize q(x, delta).

        For INT8 symmetric per-tensor:
          scale = amax / 127
          q = round(x / scale).clamp(-127, 127)
          x_dq = q * scale
        `scaled` arg is accepted for signature parity with FP8Quantizer; for
        INT8 the path is identical in both modes because the grid is uniform.
        """
        if amax <= 0:
            return torch.zeros_like(values)
        scale = amax / self.INT8_MAX
        if scale <= 0:
            return torch.zeros_like(values)
        scaled_vals = (values / scale).round().clamp(-self.INT8_MAX, self.INT8_MAX)
        return scaled_vals * scale


class WeightedHistogram:
    """
    HSWQ-compliant weighted histogram (SVD 2D/4D importance).
    """
    
    def __init__(self, bins: int = 4096, device: str = "cuda"):
        self.bins = bins
        self.device = device
        self.histogram = None
        self.max_val = 0.0
        self.total_weight = 0.0
        
    def build(self, weight: torch.Tensor, importance: Optional[torch.Tensor] = None):
        """
        Build weighted histogram from weight tensor.
        V2: full per-element importance maps (2D/4D) supported.
        """
        weight = weight.detach().float().to(self.device)
        w_abs = weight.abs()
        
        self.max_val = w_abs.max().item()
        if self.max_val == 0:
            self.max_val = 1e-7
        
        if importance is not None:
            importance = importance.float().to(self.device)
            
            # V2: when shapes match exactly, use as per-element importance (e.g. SVD scores)
            if importance.shape == weight.shape:
                imp_expanded = importance
            else:
                # Legacy: treat 0-dim (scalar) as 1-dim
                if importance.dim() == 0:
                    importance = importance.view(1)
                
                # Legacy: 1D importance (channel-only) -> broadcast to match weight
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
        
        # Bin indices
        bin_width = self.max_val / self.bins
        bin_indices = (w_abs / bin_width).long().clamp(0, self.bins - 1)
        
        # Build weighted histogram (add weights to bins)
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
    """MSE optimizer."""

    def __init__(self, device: str = "cuda", quantizer: Optional[torch.nn.Module] = None):
        self.device = device
        # Allow caller to inject INT8Quantizer; default to FP8E4M3 for full back-compat.
        self.fp8_quantizer = quantizer if quantizer is not None else FP8E4M3Quantizer(device)

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
        
        low = max_val * float(search_range[0])
        high = max_val * float(search_range[1])

        # Point search (e.g. INT8 absmax search_range=(1.0,1.0)): no grid.
        if high <= low or abs(high - low) <= max_val * 1e-12:
            return high if high > 0 else max_val
        
        # [PROVE-OF-WORK] Log only when a real interval is searched
        print(
            f"  [MSE SEARCH DEBUG] max_val: {max_val:.6f} | "
            f"range: {search_range[0]:.3f}-{search_range[1]:.3f} | "
            f"BOUNDS: {low:.6f} to {high:.6f}"
        )
        
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
_SVD_MIX_TRACE_FH = None
_SVD_MIX_TRACE_PATH = None


def _svd_mix_trace_path() -> str:
    env = os.environ.get("HSWQ_SVD_TRACE_PATH", "").strip()
    if env:
        return env
    base = os.environ.get("HSWQ_FULL_TRACE_PATH", "").strip()
    if base:
        root, ext = os.path.splitext(base)
        return f"{root}_svd_mix{ext or '.txt'}"
    import time as _time

    log_dir = os.path.join(os.getcwd(), "log")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(
        log_dir,
        f"hswq_int8_svd_mix_full_trace_{_time.strftime('%Y%m%d_%H%M%S')}.txt",
    )


def _append_svd_mix_trace(text: str) -> None:
    """Append SVD mix calc dump to disk (open once per process)."""
    global _SVD_MIX_TRACE_FH, _SVD_MIX_TRACE_PATH
    try:
        if _SVD_MIX_TRACE_FH is None:
            _SVD_MIX_TRACE_PATH = _svd_mix_trace_path()
            parent = os.path.dirname(_SVD_MIX_TRACE_PATH)
            if parent:
                os.makedirs(parent, exist_ok=True)
            _SVD_MIX_TRACE_FH = open(
                _SVD_MIX_TRACE_PATH, "a", encoding="utf-8", newline="\n"
            )
            print(f"  [HSWQ SVD MIX TRACE FILE] {_SVD_MIX_TRACE_PATH}")
        _SVD_MIX_TRACE_FH.write(text)
        if not text.endswith("\n"):
            _SVD_MIX_TRACE_FH.write("\n")
        _SVD_MIX_TRACE_FH.flush()
    except OSError as exc:
        print(f"  [HSWQ SVD MIX TRACE FILE] write failed: {exc!r}")


def emit_int8_quantize_phase(msg: str) -> None:
    """Mark post-SVD phases into the same SVD mix file the cloud run syncs.

    After all [HSWQ SVD MIX FULL] layers close, runs often die before
    Saving / Conversion done. Owner pays cloud for that silent gap.
    Phase lines make the death point visible without waiting for full Tee sync.
    """
    import time as _time

    stamp = _time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[INT8 PHASE] {stamp} | {msg}"
    print(line, flush=True)
    _append_svd_mix_trace(line)


def compute_hybrid_leverage_scores(
    weight: torch.Tensor,
    alpha: float = 0.7,
    beta: float = 0.3,
    top_p: float = 1.0,
    min_k: int = 1,
    max_k: int = 4096,
    layer_name: str = "",
    return_stats: bool = False,
):
    """
    Outputs blended importance matrix: SVD-based structural leverage and RMS magnitude,
    each L2-normalized and combined with weights (alpha, beta).

    Always emits full SVD-mix calc visibility (settings + singular values +
    alpha*leverage vs beta*magnitude contribution). The old ``shape > 100``
    print gate is removed — mid-stopping SVD logs was rebellion / deception.
    When return_stats=True, returns ``(tensor, stats_dict)``.
    """
    device = weight.device
    original_shape = weight.shape
    label = layer_name or f"tensor{tuple(original_shape)}"

    def _finish(out: torch.Tensor, stats: dict):
        if return_stats:
            return out, stats
        return out

    # Flatten to 2D
    if weight.ndim > 2:
        w_float = weight.detach().float().view(weight.shape[0], -1)
    elif weight.ndim == 2:
        w_float = weight.detach().float()
    else:
        stats = {
            "layer": label,
            "skipped": True,
            "reason": "ndim<2_ones",
            "ndim": int(weight.ndim),
            "shape": list(original_shape),
            "alpha": float(alpha),
            "beta": float(beta),
        }
        _emit_svd_mix_visibility(stats)
        return _finish(torch.ones_like(weight, dtype=torch.float32), stats)

    if torch.all(w_float == 0):
        stats = {
            "layer": label,
            "skipped": True,
            "reason": "all_zero_ones",
            "shape": list(original_shape),
            "w2d_shape": list(w_float.shape),
            "alpha": float(alpha),
            "beta": float(beta),
        }
        _emit_svd_mix_visibility(stats)
        return _finish(torch.ones_like(weight, dtype=torch.float32), stats)

    # Non-finite weights: emit full visibility then ones — do NOT crash mid-layer
    # (mid-stop via LinAlgError was another way of ending SVD settings incomplete).
    if not bool(torch.isfinite(w_float).all().item()):
        n_bad = int((~torch.isfinite(w_float)).sum().item())
        stats = {
            "layer": label,
            "skipped": True,
            "reason": "non_finite_ones",
            "n_nonfinite": n_bad,
            "shape": list(original_shape),
            "w2d_shape": list(w_float.shape),
            "alpha": float(alpha),
            "beta": float(beta),
        }
        _emit_svd_mix_visibility(stats)
        return _finish(torch.ones_like(weight, dtype=torch.float32), stats)

    M, N = w_float.shape
    max_rank = min(M, N)
    k = min(max_k, max(min_k, int(math.floor(top_p * max_rank))))
    k = min(k, max_rank)

    # --- RMS Magnitude (always) ---
    magnitude_2d = w_float ** 2  # (M, N)
    mag_norm = torch.norm(magnitude_2d, p=2)
    mag_norm_f = float(mag_norm.item()) if mag_norm.numel() else 0.0
    if mag_norm > 0:
        magnitude_2d = magnitude_2d / mag_norm

    # Full-SVD×RMS hybrid: alpha is the mix weight that puts SVD leverage
    # into ranking. alpha==0 ⇒ SVD cut (rebellion) — not "still contributing".
    if float(alpha) <= 0.0:
        raise ValueError(
            "Full-SVD×RMS mix alpha must be > 0 (alpha==0 is SVD cut / rebellion). "
            f"got alpha={alpha}, beta={beta}, shape={tuple(w_float.shape)}, layer={label!r}"
        )

    # --- SVD Leverage (full: σ^2 weighted) — settings + process always logged ---
    try:
        U, S, Vh = torch.linalg.svd(w_float, full_matrices=False)
    except RuntimeError as exc:
        stats = {
            "layer": label,
            "skipped": True,
            "reason": "svd_linalg_error_ones",
            "error": repr(exc),
            "shape": list(original_shape),
            "w2d_shape": list(w_float.shape),
            "alpha": float(alpha),
            "beta": float(beta),
        }
        _emit_svd_mix_visibility(stats)
        return _finish(torch.ones_like(weight, dtype=torch.float32), stats)
    S_sq = S ** 2
    row_scores = (U ** 2) @ S_sq.unsqueeze(1)
    col_scores = (Vh.T ** 2) @ S_sq.unsqueeze(1)
    leverage_2d = row_scores * col_scores.T

    lev_norm = torch.norm(leverage_2d, p=2)
    lev_norm_f = float(lev_norm.item()) if lev_norm.numel() else 0.0
    if lev_norm > 0:
        leverage_2d = leverage_2d / lev_norm

    alpha_f = float(alpha)
    beta_f = float(beta)
    alpha_lev = alpha_f * leverage_2d
    beta_mag = beta_f * magnitude_2d
    alpha_lev_l2 = float(torch.norm(alpha_lev, p=2).item())
    beta_mag_l2 = float(torch.norm(beta_mag, p=2).item())
    mix_den = alpha_lev_l2 + beta_mag_l2
    svd_share = (alpha_lev_l2 / mix_den) if mix_den > 0 else float("nan")

    hybrid_raw = alpha_lev + beta_mag
    hybrid_raw_mean = float(hybrid_raw.mean().item())
    hybrid_raw_min = float(hybrid_raw.min().item())
    hybrid_raw_max = float(hybrid_raw.max().item())

    hybrid_importance = hybrid_raw

    # --- 5. Histogram scale normalization ---
    avg_score = hybrid_importance.mean()
    avg_score_f = float(avg_score.item())
    if avg_score > 0:
        hybrid_importance = hybrid_importance / avg_score

    # V2-style mild baseline (avoid 0-div and full collapse)
    hybrid_importance = 0.5 + 0.5 * hybrid_importance

    S_cpu = S.detach().float().cpu()
    s_list = [float(x) for x in S_cpu.tolist()]
    out = hybrid_importance.view(original_shape)
    stats = {
        "layer": label,
        "skipped": False,
        "reason": "",
        "shape": list(original_shape),
        "w2d_shape": [int(M), int(N)],
        "settings": {
            "alpha": alpha_f,
            "beta": beta_f,
            "top_p": float(top_p),
            "min_k": int(min_k),
            "max_k": int(max_k),
            "k_requested": int(k),
            "max_rank": int(max_rank),
            "full_matrices": False,
            "svd_weighting": "sigma_squared",
            "mix_formula": "alpha*L2norm(leverage)+beta*L2norm(magnitude^2); then mean-norm; then 0.5+0.5*x",
        },
        "singular_values": {
            "n": int(S_cpu.numel()),
            "sum": float(S_cpu.sum().item()),
            "max": float(S_cpu.max().item()) if S_cpu.numel() else 0.0,
            "min": float(S_cpu.min().item()) if S_cpu.numel() else 0.0,
            "top5": s_list[:5],
            "all": s_list,
        },
        "norms": {
            "mag_l2_before_norm": mag_norm_f,
            "lev_l2_before_norm": lev_norm_f,
            "alpha_times_leverage_l2": alpha_lev_l2,
            "beta_times_magnitude_l2": beta_mag_l2,
            "svd_share_of_mix_l2": svd_share,
            "hybrid_raw_mean": hybrid_raw_mean,
            "hybrid_raw_min": hybrid_raw_min,
            "hybrid_raw_max": hybrid_raw_max,
            "hybrid_mean_before_baseline": avg_score_f,
            "out_mean": float(out.mean().item()),
            "out_min": float(out.min().item()),
            "out_max": float(out.max().item()),
        },
        "proof_svd_in_ranking": {
            "alpha_gt_0": alpha_f > 0.0,
            "alpha_lev_l2_gt_0": alpha_lev_l2 > 0.0,
            "svd_share_gt_0": bool(svd_share > 0.0) if svd_share == svd_share else False,
        },
    }
    _emit_svd_mix_visibility(stats)
    return _finish(out, stats)


def _emit_svd_mix_visibility(stats: dict) -> None:
    """Stdout + file: every SVD mix setting and calc result (no mid-stop)."""
    layer = stats.get("layer", "?")
    lines = [
        f"===== [HSWQ SVD MIX FULL] layer={layer!r} =====",
    ]
    if stats.get("skipped"):
        lines.append(f"skipped = True")
        lines.append(f"reason = {stats.get('reason')!r}")
        for k in ("ndim", "shape", "w2d_shape", "alpha", "beta"):
            if k in stats:
                lines.append(f"{k} = {stats[k]!r}")
    else:
        st = stats.get("settings") or {}
        nm = stats.get("norms") or {}
        sv = stats.get("singular_values") or {}
        pf = stats.get("proof_svd_in_ranking") or {}
        lines.append(f"shape = {stats.get('shape')!r}")
        lines.append(f"w2d_shape = {stats.get('w2d_shape')!r}")
        for k, v in st.items():
            lines.append(f"settings.{k} = {v!r}")
        for k, v in nm.items():
            lines.append(f"norms.{k} = {v!r}")
        lines.append(f"singular_values.n = {sv.get('n')!r}")
        lines.append(f"singular_values.sum = {sv.get('sum')!r}")
        lines.append(f"singular_values.max = {sv.get('max')!r}")
        lines.append(f"singular_values.min = {sv.get('min')!r}")
        lines.append(f"singular_values.top5 = {sv.get('top5')!r}")
        # Full S vector — hide nothing (may be long; file + stdout)
        lines.append(f"singular_values.all = {sv.get('all')!r}")
        for k, v in pf.items():
            lines.append(f"proof_svd_in_ranking.{k} = {v!r}")
        lines.append(
            f"[Hybrid Full-SVD x RMS] Mixing into ranking "
            f"shape={stats.get('w2d_shape')!r} "
            f"alpha={st.get('alpha')!r} beta={st.get('beta')!r} "
            f"svd_share={nm.get('svd_share_of_mix_l2')!r} "
            f"alpha_lev_l2={nm.get('alpha_times_leverage_l2')!r} "
            f"beta_mag_l2={nm.get('beta_times_magnitude_l2')!r}"
        )
    lines.append(f"===== [/HSWQ SVD MIX FULL] layer={layer!r} =====")
    block = "\n".join(lines)
    # Full SVD dump must appear immediately (default path). No -u / env tricks.
    print(block, flush=True)
    sys.stdout.flush()
    _append_svd_mix_trace(block)



class HSWQWeightedHistogramOptimizerV4:
    """
    HSWQ weighted histogram optimizer (V4: SVD-Magnitude Hybrid).

    `quantizer` arg allows switching between FP8E4M3 (default) and INT8
    (symmetric per-tensor) quantize/dequantize simulators. The amax search
    loop is identical; only the quantize/dequantize kernel changes.
    """

    def __init__(self, bins: int = 8192, num_candidates: int = 1000, refinement_iterations: int = 10, device: str = "cuda", alpha: float = 0.7, beta: float = 0.3, quantizer=None):
        self.bins = bins
        self.num_candidates = num_candidates
        self.refinement_iterations = refinement_iterations
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.mse_optimizer = MSEOptimizer(device, quantizer=quantizer)
    
    def compute_optimal_amax(self, weight: torch.Tensor, importance: Optional[torch.Tensor] = None, use_svd_leverage: bool = True, search_range: Tuple[float, float] = (0.5, 1.0), scaled: bool = True) -> float:
        """
        Compute optimal amax for weight tensor (V4 Hybrid).
        """
        # Apply SVD+Magnitude hybrid importance
        combined_importance = None
        
        if use_svd_leverage and weight.ndim >= 2:
            # Compute hybrid importance matrix (SVD + RMS)
            hybrid_importance = compute_hybrid_leverage_scores(weight, alpha=self.alpha, beta=self.beta)

            
            # If 1D channel importance (e.g. from DualMonitor) exists, multiply
            if importance is not None:
                importance = importance.float().to(self.device)
                
                # Broadcast to match weight dimensions
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

        # Build weighted histogram (2D/4D combined_importance gives per-pixel weights)
        weighted_hist = WeightedHistogram(bins=self.bins, device=self.device)
        weighted_hist.build(weight, combined_importance)
        
        # Search for optimal amax (DEBUG only when interval is non-degenerate)
        max_val = weighted_hist.max_val
        low = max_val * float(search_range[0])
        high = max_val * float(search_range[1])
        if max_val > 0 and high > low and abs(high - low) > max_val * 1e-12:
            print(
                f"  [HSWQ V4 DEBUG] max_val: {max_val:.6f} | "
                f"range: {search_range[0]:.3f}-{search_range[1]:.3f} | "
                f"BOUNDS: {low:.6f} to {high:.6f}"
            )

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
        
        # SVD importance (for display/verification)
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

    def compute_optimal_amax_with_stats_int8_range(
        self,
        weight: torch.Tensor,
        importance: Optional[torch.Tensor] = None,
        use_svd_leverage: bool = True,
        scaled: bool = True,
        search_range: Tuple[float, float] = (1.0, 1.0),
        layer_name: str = "",
    ) -> dict:
        """INT8-only: estimated_mse at an explicit search_range (typically absmax).

        FP8 scripts must keep calling compute_optimal_amax_with_stats / compute_optimal_amax.
        Default search_range is (1.0, 1.0) = natural INT8 absmax pack point.
        Callers use estimated_mse for FP16 protection candidate ranking
        (pack amax stays absmax; this is not an amax search).
        Requires MSEOptimizer.quantizer = INT8Quantizer (injected by INT8 callers).

        Builds the hybrid importance + weighted histogram **once** (avoids the
        previous double-SVD path that also spammed 1.000-1.000 DEBUG lines).
        """
        combined_importance = None
        svd_mix_stats: Optional[dict] = None
        if use_svd_leverage and weight.ndim >= 2:
            hybrid_importance, svd_mix_stats = compute_hybrid_leverage_scores(
                weight,
                alpha=self.alpha,
                beta=self.beta,
                layer_name=layer_name,
                return_stats=True,
            )
            if importance is not None:
                importance = importance.float().to(self.device)
                if weight.ndim == 4:
                    in_channels = weight.shape[1]
                    pad_len = max(0, in_channels - importance.numel())
                    imp_1d = torch.cat(
                        [importance[:in_channels], torch.ones(pad_len, device=self.device)]
                    )
                    imp_expanded = imp_1d.view(1, -1, 1, 1).expand_as(weight)
                elif weight.ndim == 2:
                    in_features = weight.shape[1]
                    pad_len = max(0, in_features - importance.numel())
                    imp_1d = torch.cat(
                        [importance[:in_features], torch.ones(pad_len, device=self.device)]
                    )
                    imp_expanded = imp_1d.view(1, -1).expand_as(weight)
                else:
                    imp_expanded = importance.expand_as(weight)
                combined_importance = hybrid_importance * imp_expanded
            else:
                combined_importance = hybrid_importance
        else:
            combined_importance = importance
            svd_mix_stats = {
                "layer": layer_name or f"tensor{tuple(weight.shape)}",
                "skipped": True,
                "reason": (
                    "use_svd_leverage=False"
                    if not use_svd_leverage
                    else "ndim<2_in_int8_range"
                ),
                "shape": list(weight.shape),
                "alpha": float(self.alpha),
                "beta": float(self.beta),
            }
            _emit_svd_mix_visibility(svd_mix_stats)

        weighted_hist = WeightedHistogram(bins=self.bins, device=self.device)
        weighted_hist.build(weight, combined_importance)
        max_val = weighted_hist.max_val
        if max_val <= 0:
            return {
                "optimal_amax": 0.0,
                "max_val": 0.0,
                "compression_ratio": 1.0,
                "estimated_mse": 0.0,
                "svd_mix": svd_mix_stats,
            }

        sr0 = float(search_range[0])
        sr1 = float(search_range[1])
        low = max_val * sr0
        high = max_val * sr1
        if high <= low or abs(high - low) <= max_val * 1e-12:
            # Absmax / point search — no candidate grid, no DEBUG spam.
            optimal_amax = high if high > 0 else max_val
        else:
            print(
                f"  [HSWQ V4 DEBUG] max_val: {max_val:.6f} | "
                f"range: {sr0:.3f}-{sr1:.3f} | BOUNDS: {low:.6f} to {high:.6f}"
            )
            optimal_amax = self.mse_optimizer.find_optimal_amax(
                weighted_hist,
                num_candidates=self.num_candidates,
                search_range=search_range,
                refinement_iterations=self.refinement_iterations,
                scaled=scaled,
            )

        histogram = weighted_hist.get_histogram()
        bin_centers = weighted_hist.get_bin_centers()
        estimated_mse = self.mse_optimizer.compute_weighted_mse(
            histogram, bin_centers, optimal_amax, scaled=scaled
        )

        return {
            "optimal_amax": optimal_amax,
            "max_val": max_val,
            "compression_ratio": optimal_amax / max_val if max_val > 0 else 1.0,
            "estimated_mse": estimated_mse,
            "svd_mix": svd_mix_stats,
        }


if __name__ == "__main__":
    print("HSWQ V4: Hybrid SVD-Magnitude Blended Histogram MSE - Self Test")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Test: random weight tensor with 1D and hybrid importance
    print("\n[Test] 2D Weight Matrix Hybrid Extraction and Amax Optimization")
    
    # Dummy (128, 256) tensor
    U_true = torch.randn(128, 16, device=device)
    V_true = torch.randn(256, 16, device=device)
    weight = U_true @ V_true.T
    
    # Add intentional outliers
    weight[5, 5] = 20.0
    weight[10, 100] = -25.0
    
    optimizer = HSWQWeightedHistogramOptimizerV4(device=device, alpha=0.7, beta=0.3)
    
    print("\n1. Conventional MSE search (Hybrid-Aware: False)")
    result_v1 = optimizer.compute_optimal_amax_with_stats(weight, importance=None, use_svd_leverage=False)
    print(f"  Optimal amax: {result_v1['optimal_amax']:.4f}")
    
    print("\n2. Hybrid SVD+RMS MSE search (Hybrid-Aware: True)")
    result_v2 = optimizer.compute_optimal_amax_with_stats(weight, importance=None, use_svd_leverage=True)
    print(f"  Optimal amax: {result_v2['optimal_amax']:.4f}")
    
    # Call internal function directly
    hybrid_score = compute_hybrid_leverage_scores(weight, alpha=0.7, beta=0.3)
    print(f"\n[Hybrid Score Inspection]")
    print(f"  Shape: {hybrid_score.shape}")
    print(f"  Min/Max: {hybrid_score.min().item():.4f} / {hybrid_score.max().item():.4f}")
    print(f"  Mean: {hybrid_score.mean().item():.4f}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")

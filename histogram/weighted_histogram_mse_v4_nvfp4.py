"""
HSWQ Weighted Histogram MSE Optimizer V4 — NVFP4 ConvRot edition
=====================================================================================

SVD+RMS hybrid importance (same spirit as INT8 V3.0 V4).

FP16-protect ranking MSE (pack damage @ absmax, not amax search):
  - Linear 2D: real kitchen TensorCoreNVFP4Layout quantize/dequantize MSE
  - Conv2d 4D: channelwise INT8 q/dq MSE (NVFP4 layout is 2D-only)

Histogram-bin NVFP4PackQuantizer remains a coarse proxy only; protect ranking
must call compute_pack_mse_absmax_with_svd (real pack), not the bin proxy alone.
"""

import os
import torch
import numpy as np
import math
from typing import Optional, Tuple, List


def _cuda_mse_work_budget_bytes(device) -> Optional[int]:
    """Bytes available for pack-MSE scratch on this CUDA device.

    Uses almost all free VRAM so convert does not leave the card idle after
    DiT is resident. Reserves 512 MiB or 2% of total (whichever larger) so
    kitchen/allocator fragmentation does not CUDA OOM.
    """
    if not torch.cuda.is_available():
        return None
    try:
        free_b, total_b = torch.cuda.mem_get_info(device)
    except Exception:
        return None
    reserve = max(512 * 1024 * 1024, int(total_b * 0.02))
    return max(0, int(free_b) - reserve)


class INT8Quantizer:
    """Symmetric per-tensor INT8 q/dq (Conv2d path; NVFP4 is 2D-only)."""

    INT8_MAX = 127

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.max_representable = float(self.INT8_MAX)

    def quantize_dequantize(self, values: torch.Tensor, amax: float, scaled: bool = True) -> torch.Tensor:
        if amax <= 0:
            return torch.zeros_like(values)
        scale = amax / self.INT8_MAX
        if scale <= 0:
            return torch.zeros_like(values)
        scaled_vals = (values / scale).round().clamp(-self.INT8_MAX, self.INT8_MAX)
        return scaled_vals * scale


class NVFP4PackQuantizer:
    """Histogram-bin MSE proxy for NVFP4 Linear / INT8 Conv pack paths.

    Bin centers are 1D magnitudes; for Linear we approximate NVFP4 damage with
    an FP4-like uniform midrise grid (F4_E2M1 style 16 levels via kitchen max
    when available, else 6.0) scaled to amax. For Conv-shaped callers the
    INT8Quantizer path is used by MSEOptimizer injection choice.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        try:
            from comfy_kitchen.float_utils import F4_E2M1_MAX
            self.max_representable = float(F4_E2M1_MAX)
        except Exception:
            self.max_representable = 6.0

    def quantize_dequantize(self, values: torch.Tensor, amax: float, scaled: bool = True) -> torch.Tensor:
        if amax <= 0:
            return torch.zeros_like(values)
        # Symmetric uniform grid over [-amax, amax] with NVFP4-ish level count.
        # 16 positive incl. zero → 15 steps (F4-like density proxy for ranking).
        n_pos = 16
        scale = amax / max(self.max_representable, 1e-12)
        if scale <= 0:
            return torch.zeros_like(values)
        # Map to [-max_rep, max_rep] then round to linear grid of n_pos levels.
        y = values / scale
        y = y.clamp(-self.max_representable, self.max_representable)
        step = self.max_representable / float(n_pos - 1)
        q = (y / step).round() * step
        return q * scale


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
        # Allow caller to inject NVFP4PackQuantizer / INT8Quantizer; default NVFP4.
        self.pack_quantizer = quantizer if quantizer is not None else NVFP4PackQuantizer(device)

    def compute_weighted_mse(self, histogram: torch.Tensor, bin_centers: torch.Tensor, amax: float, scaled: bool = True) -> float:
        dequantized = self.pack_quantizer.quantize_dequantize(bin_centers.float(), amax, scaled=scaled).double()
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

        # Point search (e.g. absmax search_range=(1.0,1.0)): no grid.
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
        f"hswq_nvfp4_svd_mix_full_trace_{_time.strftime('%Y%m%d_%H%M%S')}.txt",
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
    print(block)
    _append_svd_mix_trace(block)



class HSWQWeightedHistogramOptimizerV4:
    """
    HSWQ weighted histogram optimizer (V4: SVD-Magnitude Hybrid).

    `quantizer` arg selects the histogram-bin MSE proxy
    (default NVFP4PackQuantizer; INT8Quantizer for Conv-shaped bin callers).
    Protect ranking must use compute_pack_mse_absmax_with_svd (real pack).
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

    def compute_optimal_amax_with_stats_absmax_range(
        self,
        weight: torch.Tensor,
        importance: Optional[torch.Tensor] = None,
        use_svd_leverage: bool = True,
        scaled: bool = True,
        search_range: Tuple[float, float] = (1.0, 1.0),
        layer_name: str = "",
    ) -> dict:
        """Bin-proxy estimated_mse at an explicit search_range (typically absmax).

        Default search_range is (1.0, 1.0) = absmax pack point (no amax search).
        Prefer compute_pack_mse_absmax_with_svd for FP16 protect ranking.
        Requires MSEOptimizer.pack_quantizer = NVFP4PackQuantizer
        (or INT8Quantizer for Conv-shaped bin callers).

        Builds hybrid importance + weighted histogram once.
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
                    else "ndim<2_in_absmax_range"
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

    def compute_pack_mse_absmax_with_svd(
        self,
        weight: torch.Tensor,
        channel_importance: Optional[torch.Tensor] = None,
        importance: Optional[torch.Tensor] = None,
        use_svd_leverage: bool = True,
        layer_name: str = "",
        linear_pack: str = "nvfp4",
    ) -> dict:
        """Real pack roundtrip MSE @ absmax for FP16 protect ranking.

        Linear (2D): comfy_kitchen TensorCoreNVFP4Layout, or per-out-channel
        INT8 when linear_pack="int8" (multi-tier shelter d1 measurement).
        Conv2d (4D): per-out-channel INT8 (same as convert pack_channelwise_int8).
        Pack amax stays absmax (optimal_amax = weight absmax); MSE ranks FP16 keep.
        linear_pack only changes the 2D pack format; importance weighting and
        its denominator are identical so d0/d1 stay directly comparable.
        """
        if linear_pack not in ("nvfp4", "int8"):
            raise ValueError(
                f"linear_pack must be 'nvfp4' or 'int8', got {linear_pack!r}"
            )
        imp_arg = channel_importance if channel_importance is not None else importance
        # DiT stays on GPU (no model.cpu park). Work stays on GPU too: no
        # importance→CPU park, no fixed 64MiB timid chunks. Size MSE rows from
        # cuda mem_get_info so free VRAM is consumed hard; reserve headroom so
        # we never CUDA OOM (pre-check; refuse rather than OOM).
        w = weight.detach().float().to(self.device)
        if w.ndim not in (2, 4):
            raise ValueError(
                f"compute_pack_mse_absmax_with_svd expects 2D/4D, got ndim={w.ndim}"
            )

        combined_importance = None
        svd_mix_stats: Optional[dict] = None
        if use_svd_leverage and w.ndim >= 2:
            hybrid_importance, svd_mix_stats = compute_hybrid_leverage_scores(
                w,
                alpha=self.alpha,
                beta=self.beta,
                layer_name=layer_name,
                return_stats=True,
            )
            if imp_arg is not None:
                imp_arg = imp_arg.float().to(self.device)
                if w.ndim == 4:
                    in_channels = w.shape[1]
                    pad_len = max(0, in_channels - imp_arg.numel())
                    imp_1d = torch.cat(
                        [imp_arg[:in_channels], torch.ones(pad_len, device=self.device)]
                    )
                    imp_expanded = imp_1d.view(1, -1, 1, 1).expand_as(w)
                else:
                    in_features = w.shape[1]
                    pad_len = max(0, in_features - imp_arg.numel())
                    imp_1d = torch.cat(
                        [imp_arg[:in_features], torch.ones(pad_len, device=self.device)]
                    )
                    imp_expanded = imp_1d.view(1, -1).expand_as(w)
                combined_importance = (hybrid_importance * imp_expanded).contiguous()
                del imp_1d, imp_expanded, imp_arg
            else:
                combined_importance = hybrid_importance
            if combined_importance is not hybrid_importance:
                del hybrid_importance
        else:
            combined_importance = None
            svd_mix_stats = {
                "layer": layer_name or f"tensor{tuple(w.shape)}",
                "skipped": True,
                "reason": "use_svd_leverage=False_or_ndim",
                "shape": list(w.shape),
                "alpha": float(self.alpha),
                "beta": float(self.beta),
            }
            _emit_svd_mix_visibility(svd_mix_stats)

        max_val = float(w.abs().max().item()) if w.numel() else 0.0
        if max_val <= 0.0:
            return {
                "optimal_amax": 0.0,
                "max_val": 0.0,
                "compression_ratio": 1.0,
                "estimated_mse": 0.0,
                "svd_mix": svd_mix_stats,
                "pack_format": "empty",
            }

        # Importance stays on GPU with the float32 weight (use VRAM; no CPU park).
        if w.ndim == 2 and linear_pack == "int8":
            # Multi-tier d1: per-out-channel INT8 on 2D (same math as the
            # convert pack_channelwise_int8 Linear shelter path).
            amax = torch.clamp(w.abs().amax(dim=1).reshape(-1), min=1e-6)
            scale = amax / 127.0
            scale_view = scale.view(-1, 1)
            amax_view = amax.view(-1, 1)
            clamped = torch.clamp(w, -amax_view, amax_view)
            q = (clamped / scale_view).round().clamp(-127, 127)
            dq = (q * scale_view).contiguous()
            del amax, scale, scale_view, amax_view, clamped, q
            pack_format = "int8_channelwise"
        elif w.ndim == 2:
            from comfy_kitchen.tensor import TensorCoreNVFP4Layout

            layout = TensorCoreNVFP4Layout
            # Kitchen TensorCoreNVFP4Layout accepts FP16/BF16 only.
            # SVD/importance stay on float32 `w`; pack input must not be float32
            # or quantize raises and convert/analyze skip — SVD computed then discarded.
            if weight.dtype == torch.bfloat16:
                w_pack = w.to(dtype=torch.bfloat16)
            else:
                w_pack = w.to(dtype=torch.float16)
            qdata, params = layout.quantize(w_pack)
            del w_pack
            full = layout.dequantize(qdata, params)
            del qdata
            orig = tuple(params.orig_shape)
            if tuple(full.shape) != orig:
                slices = tuple(slice(0, s) for s in orig)
                dq = full[slices].contiguous()
            else:
                dq = full.contiguous()
            del full, params
            pack_format = "nvfp4"
        else:
            reduce_dims = tuple(range(1, w.dim()))
            amax = torch.clamp(w.abs().amax(dim=reduce_dims).reshape(-1), min=1e-6)
            scale = amax / 127.0
            scale_view = scale.view(-1, 1, 1, 1)
            amax_view = amax.view(-1, 1, 1, 1)
            clamped = torch.clamp(w, -amax_view, amax_view)
            q = (clamped / scale_view).round().clamp(-127, 127)
            dq = (q * scale_view).contiguous()
            del amax, scale, scale_view, amax_view, clamped, q
            pack_format = "int8_channelwise"

        # Weighted MSE on GPU: one shot when free VRAM allows; else largest
        # row chunks that still fit (consume free bytes, never CUDA OOM).
        rows = int(w.shape[0])
        elems_per_row = max(int(w[0].numel()), 1)
        # err float32 (+ optional wi float32) per row
        tensors_per_row = 2 if combined_importance is not None else 1
        bytes_per_row = elems_per_row * 4 * tensors_per_row
        budget = _cuda_mse_work_budget_bytes(w.device)
        if budget is not None:
            if bytes_per_row > budget:
                raise RuntimeError(
                    f"[pack MSE] refuse CUDA OOM for {layer_name or 'layer'}: "
                    f"one row needs {bytes_per_row} B but free budget is {budget} B "
                    f"(device={w.device}). Keep DiT on GPU and free other peaks."
                )
            chunk_rows = max(1, min(rows, budget // bytes_per_row))
        else:
            chunk_rows = rows
        num = 0.0
        den = 0.0
        for r0 in range(0, rows, chunk_rows):
            r1 = min(rows, r0 + chunk_rows)
            err = (dq[r0:r1].float() - w[r0:r1]) ** 2
            if combined_importance is not None:
                wi = combined_importance[r0:r1]
                if wi.shape != err.shape:
                    wi = torch.ones_like(err)
                num += float((err * wi).sum().item())
                den += float(wi.sum().item())
            else:
                num += float(err.sum().item())
                den += float(err.numel())
            del err
        mse = num / max(den, 1e-12)
        del dq, w, combined_importance

        return {
            "optimal_amax": max_val,
            "max_val": max_val,
            "compression_ratio": 1.0,
            "estimated_mse": mse,
            "svd_mix": svd_mix_stats,
            "pack_format": pack_format,
        }


# Alias used by convert weight-amax / protect callers (same class).
HSWQWeightedHistogramOptimizerV4NVFP4 = HSWQWeightedHistogramOptimizerV4


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

"""
HSWQ V4 for NVFP4 convert (Full-SVD x RMS + real pack MSE)
===========================================================

NVFP4-dedicated V4. Not an FP8 E4M3 histogram copy.

Importance (mandatory Full-SVD x RMS hybrid, alpha > 0):
    L(i,j) = SVD leverage (sigma^2-weighted U/V)
    M(i,j) = X_ij^2  (RMS magnitude)
    Score  = alpha * L/||L||_2 + beta * M/||M||_2
    Optional DualMonitor channel importance multiplies Score.

estimated_mse @ absmax (FP16 keep ranking / VETO MSE):
    Linear 2D  -> TensorCoreNVFP4Layout quantize -> dequantize
    Conv2d 4D  -> per-out-channel INT8 pack -> dequant
    Element-weighted MSE using the hybrid Score map (not FP8 grid).

SVD math is shared with weighted_histogram_mse_v4_int8.compute_hybrid_leverage_scores
(format-agnostic). Pack kernels come from weighted_histogram_mse_fast_nvfp4.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch

_HIST_DIR = os.path.dirname(os.path.abspath(__file__))
if _HIST_DIR not in sys.path:
    sys.path.insert(0, _HIST_DIR)

from weighted_histogram_mse_v4_int8 import (  # type: ignore
    compute_hybrid_leverage_scores,
)
from weighted_histogram_mse_fast_nvfp4 import (  # type: ignore
    pack_dequant_channelwise_int8,
    pack_dequant_nvfp4,
)


def _expand_channel_importance(
    weight: torch.Tensor, importance: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    if importance is None:
        return None
    device = weight.device
    imp = importance.detach().float().reshape(-1).to(device=device)
    in_dim = int(weight.shape[1])
    if imp.numel() < in_dim:
        pad = torch.ones(in_dim - imp.numel(), device=device, dtype=torch.float32)
        imp = torch.cat([imp, pad])
    else:
        imp = imp[:in_dim]
    if weight.ndim == 2:
        return imp.view(1, -1).expand_as(weight)
    if weight.ndim == 4:
        return imp.view(1, -1, 1, 1).expand_as(weight)
    return None


def _combine_hybrid_and_channel(
    weight: torch.Tensor,
    hybrid: torch.Tensor,
    channel_importance: Optional[torch.Tensor],
) -> torch.Tensor:
    ch = _expand_channel_importance(weight, channel_importance)
    if ch is None:
        return hybrid
    return hybrid * ch


def _element_weighted_mse(
    weight_ref: torch.Tensor,
    weight_dq: torch.Tensor,
    element_importance: torch.Tensor,
) -> float:
    err2 = (weight_dq.float() - weight_ref.float()).pow(2)
    imp = element_importance.float()
    denom = float(imp.sum().clamp_min(1e-12).item())
    return float((err2 * imp).sum().item() / denom)


def pack_mse_at_absmax_nvfp4_or_int8(
    weight: torch.Tensor,
    element_importance: torch.Tensor,
    *,
    clip_amax: Optional[float] = None,
) -> float:
    """Actual NVFP4 (2D) / INT8 channelwise (4D) pack MSE @ clip amax."""
    w_ref = weight.detach().float()
    if clip_amax is None:
        amax = float(w_ref.abs().amax().clamp_min(1e-12).item())
    else:
        amax = max(float(clip_amax), 1e-12)
    w_c = w_ref.clamp(-amax, amax)
    if w_ref.ndim == 2:
        dq = pack_dequant_nvfp4(w_c)
    elif w_ref.ndim == 4:
        dq = pack_dequant_channelwise_int8(w_c)
    else:
        raise ValueError(f"unsupported weight ndim={w_ref.ndim}")
    return _element_weighted_mse(w_ref, dq, element_importance)


class HSWQWeightedHistogramOptimizerV4NVFP4:
    """V4 Full-SVD x RMS + NVFP4/INT8 pack estimated_mse (no FP8 histogram)."""

    def __init__(
        self,
        device: str = "cuda",
        alpha: float = 0.7,
        beta: float = 0.3,
    ):
        if float(alpha) <= 0.0:
            raise ValueError(
                "HSWQWeightedHistogramOptimizerV4NVFP4: alpha must be > 0 "
                f"(alpha==0 is SVD cut / rebellion). got {alpha}"
            )
        self.device = device
        self.alpha = float(alpha)
        self.beta = float(beta)

    def build_element_importance(
        self,
        weight: torch.Tensor,
        channel_importance: Optional[torch.Tensor] = None,
        *,
        use_svd_leverage: bool = True,
        layer_name: str = "",
    ) -> torch.Tensor:
        """Full-SVD x RMS hybrid (required) x optional DualMonitor I_c."""
        w = weight.detach().float()
        if not use_svd_leverage:
            raise ValueError(
                "HSWQ V4 NVFP4 requires use_svd_leverage=True "
                f"(layer={layer_name!r})"
            )
        if w.ndim < 2:
            return torch.ones_like(w, dtype=torch.float32)
        hybrid = compute_hybrid_leverage_scores(
            w,
            alpha=self.alpha,
            beta=self.beta,
            layer_name=layer_name or f"tensor{tuple(w.shape)}",
            return_stats=False,
        )
        return _combine_hybrid_and_channel(w, hybrid, channel_importance)

    def compute_pack_mse_absmax_with_svd(
        self,
        weight: torch.Tensor,
        channel_importance: Optional[torch.Tensor] = None,
        *,
        use_svd_leverage: bool = True,
        layer_name: str = "",
    ) -> dict:
        """estimated_mse at natural absmax for FP16 keep / VETO ranking.

        Always Full-SVD x RMS. Pack is real NVFP4 (Linear) or INT8 (Conv).
        """
        w = weight.detach().float().to(device=self.device)
        if channel_importance is not None:
            channel_importance = channel_importance.detach().float().to(
                device=self.device
            )
        absmax = float(w.abs().amax().clamp_min(1e-12).item())
        element_imp = self.build_element_importance(
            w,
            channel_importance,
            use_svd_leverage=use_svd_leverage,
            layer_name=layer_name,
        )
        mse = pack_mse_at_absmax_nvfp4_or_int8(
            w, element_imp, clip_amax=absmax
        )
        pack_mode = "nvfp4" if w.ndim == 2 else "int8"
        return {
            "optimal_amax": absmax,
            "max_val": absmax,
            "compression_ratio": 1.0,
            "estimated_mse": float(mse),
            "use_svd_leverage": True,
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "pack_mode": pack_mode,
            "quantizer": "NVFP4_or_INT8_pack",
            "optimizer": "HSWQWeightedHistogramOptimizerV4NVFP4",
        }

    # Alias matching INT8 V4 call shape used by analyze measure helpers.
    def compute_optimal_amax_with_stats_nvfp4_range(
        self,
        weight: torch.Tensor,
        importance: Optional[torch.Tensor] = None,
        use_svd_leverage: bool = True,
        scaled: bool = False,  # noqa: ARG002 — pack path unused; absmax only
        search_range: tuple = (1.0, 1.0),  # noqa: ARG002
        layer_name: str = "",
    ) -> dict:
        return self.compute_pack_mse_absmax_with_svd(
            weight,
            channel_importance=importance,
            use_svd_leverage=use_svd_leverage,
            layer_name=layer_name,
        )

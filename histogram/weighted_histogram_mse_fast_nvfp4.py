"""
HSWQ weight clip-amax optimizer for NVFP4 / INT8 pack paths
==========================================================

Unlike weighted_histogram_mse_fast.py (FP8 e4m3 grid on histogram bins), this
module chooses a global clip amax by minimizing DualMonitor-weighted MSE of the
*actual* pack roundtrip used at convert time:

  Linear (2D):  TensorCoreNVFP4Layout.quantize → dequantize  (CPU float32 pack;
                CUDA NVFP4 kernels reject float32 — matches convert loop)
  Conv2d (4D):  per-out-channel INT8 pack → dequant

Core:
    amax* = argmin_a  Σ_i w_i · (DQ(clamp(W, a))_i - W_i)²
    with optional per-input-channel importance I_c from DualMonitor.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def _expand_channel_importance(
    weight: torch.Tensor, importance: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """Broadcast DualMonitor I_c [in] onto weight for element-wise weighted MSE."""
    if importance is None:
        return None
    imp = importance.detach().float().reshape(-1)
    in_dim = int(weight.shape[1])
    if imp.numel() < in_dim:
        pad = torch.ones(
            in_dim - imp.numel(), device=weight.device, dtype=torch.float32
        )
        imp = torch.cat([imp.to(device=weight.device, dtype=torch.float32), pad])
    else:
        imp = imp[:in_dim].to(device=weight.device, dtype=torch.float32)
    if weight.ndim == 2:
        return imp.view(1, -1).expand_as(weight)
    if weight.ndim == 4:
        return imp.view(1, -1, 1, 1).expand_as(weight)
    return None


def _weighted_mse(
    weight_ref: torch.Tensor,
    weight_dq: torch.Tensor,
    importance: Optional[torch.Tensor],
) -> float:
    err2 = (weight_dq.float() - weight_ref.float()).pow(2)
    imp = _expand_channel_importance(weight_ref, importance)
    if imp is None:
        return float(err2.mean().item())
    denom = float(imp.sum().clamp_min(1e-12).item())
    return float((err2 * imp).sum().item() / denom)


def _get_nvfp4_layout():
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout

    return TensorCoreNVFP4Layout


def pack_dequant_nvfp4(weight: torch.Tensor) -> torch.Tensor:
    """NVFP4 pack roundtrip; input must be 2D. Packs on CPU float32."""
    if weight.ndim != 2:
        raise ValueError(f"NVFP4 pack expects 2D weight, got ndim={weight.ndim}")
    layout = _get_nvfp4_layout()
    w_cpu = weight.detach().float().cpu()
    qdata, params = layout.quantize(w_cpu)
    full = layout.dequantize(qdata, params)
    orig = tuple(params.orig_shape)
    if tuple(full.shape) != orig:
        slices = tuple(slice(0, s) for s in orig)
        full = full[slices]
    return full.to(device=weight.device, dtype=torch.float32)


def pack_dequant_channelwise_int8(weight: torch.Tensor) -> torch.Tensor:
    """Per-out-channel INT8 pack roundtrip (Conv2d convert path)."""
    w = weight.detach().float()
    reduce_dims = tuple(range(1, w.dim()))
    amax = torch.clamp(w.abs().amax(dim=reduce_dims).reshape(-1), min=1e-6)
    scale = amax / 127.0
    if w.dim() == 4:
        scale_view = scale.view(-1, 1, 1, 1)
        amax_view = amax.view(-1, 1, 1, 1)
    elif w.dim() == 2:
        scale_view = scale.view(-1, 1)
        amax_view = amax.view(-1, 1)
    else:
        raise ValueError(f"unsupported weight ndim={w.dim()} for INT8 channel pack")
    clamped = torch.clamp(w, -amax_view, amax_view)
    q = (clamped / scale_view).round().clamp(-127, 127).to(torch.int8)
    return q.float() * scale_view.float()


def mse_after_clip_nvfp4(
    weight: torch.Tensor, clip_amax: float, importance: Optional[torch.Tensor] = None
) -> float:
    amax = max(float(clip_amax), 1e-12)
    w_ref = weight.detach().float()
    w_c = w_ref.clamp(-amax, amax)
    dq = pack_dequant_nvfp4(w_c)
    return _weighted_mse(w_ref, dq, importance)


def mse_after_clip_int8(
    weight: torch.Tensor, clip_amax: float, importance: Optional[torch.Tensor] = None
) -> float:
    amax = max(float(clip_amax), 1e-12)
    w_ref = weight.detach().float()
    w_c = w_ref.clamp(-amax, amax)
    dq = pack_dequant_channelwise_int8(w_c)
    return _weighted_mse(w_ref, dq, importance)


class MSEOptimizerNVFP4Pack:
    """Grid + refine search over clip amax for pack roundtrip MSE."""

    def __init__(
        self,
        device: str = "cuda",
        num_candidates: int = 32,
        refinement_iterations: int = 2,
        search_range: Tuple[float, float] = (0.5, 1.0),
    ):
        self.device = device
        self.num_candidates = int(num_candidates)
        self.refinement_iterations = int(refinement_iterations)
        self.search_range = (
            float(search_range[0]),
            float(search_range[1]),
        )

    def find_optimal_amax(
        self,
        weight: torch.Tensor,
        importance: Optional[torch.Tensor] = None,
        pack_mode: str = "auto",
    ) -> float:
        """pack_mode: 'nvfp4' | 'int8' | 'auto' (2D→nvfp4, 4D→int8)."""
        w = weight.detach().float()
        if pack_mode == "auto":
            if w.ndim == 2:
                pack_mode = "nvfp4"
            elif w.ndim == 4:
                pack_mode = "int8"
            else:
                raise ValueError(f"unsupported weight ndim={w.ndim}")
        if pack_mode == "nvfp4":
            if w.ndim != 2:
                raise ValueError("nvfp4 pack_mode requires 2D weight")
            mse_fn = lambda a: mse_after_clip_nvfp4(w, a, importance)
        elif pack_mode == "int8":
            mse_fn = lambda a: mse_after_clip_int8(w, a, importance)
        else:
            raise ValueError(f"unknown pack_mode={pack_mode!r}")

        max_val = float(w.abs().amax().item())
        if max_val <= 0.0:
            return 1e-6

        low = max_val * self.search_range[0]
        high = max_val * self.search_range[1]
        best_amax = max_val
        min_mse = float("inf")

        for iteration in range(self.refinement_iterations + 1):
            candidates = torch.linspace(low, high, self.num_candidates)
            for a_t in candidates:
                amax = float(a_t.item())
                mse = float(mse_fn(amax))
                if mse < min_mse:
                    min_mse = mse
                    best_amax = amax
            if iteration < self.refinement_iterations:
                range_width = (high - low) / 4.0
                low = max(max_val * 0.1, best_amax - range_width)
                high = min(max_val * 1.2, best_amax + range_width)
        return float(best_amax)


class HSWQWeightedHistogramOptimizerFastNVFP4:
    """
    NVFP4/INT8 pack-roundtrip amax optimizer.

    Same call shape as HSWQWeightedHistogramOptimizerFast.compute_optimal_amax,
    but `scaled` is ignored (no FP8 scaled/compatible grid — pack path only).
    """

    def __init__(
        self,
        bins: int = 4096,
        num_candidates: int = 32,
        refinement_iterations: int = 2,
        device: str = "cuda",
        search_range: Tuple[float, float] = (0.5, 1.0),
    ):
        # bins kept for API parity with the FP8 fast optimizer; unused here
        # (block NVFP4 cannot be reduced to magnitude histogram bins).
        self.bins = int(bins)
        self.num_candidates = int(num_candidates)
        self.refinement_iterations = int(refinement_iterations)
        self.device = device
        self.search_range = search_range
        self.mse_optimizer = MSEOptimizerNVFP4Pack(
            device=device,
            num_candidates=self.num_candidates,
            refinement_iterations=self.refinement_iterations,
            search_range=search_range,
        )
        print(
            f"[HSWQ] HSWQWeightedHistogramOptimizerFastNVFP4 on {device} "
            f"(pack roundtrip MSE; not FP8 grid)"
        )
        print(
            f"  Candidates: {self.num_candidates} | "
            f"Refinement: {self.refinement_iterations} | "
            f"bins(unused)={self.bins}"
        )

    def compute_optimal_amax(
        self,
        weight: torch.Tensor,
        importance: Optional[torch.Tensor] = None,
        scaled: bool = False,
        pack_mode: str = "auto",
    ) -> float:
        """Return clip amax minimizing pack roundtrip MSE. `scaled` is ignored."""
        _ = scaled
        w = weight.detach().float()
        if w.device.type != self.device and self.device != "cpu":
            # Keep weight on its device for MSE reduction; NVFP4 pack forces CPU.
            pass
        imp = None
        if importance is not None:
            imp = importance.detach().float()
            if imp.numel() != int(w.shape[1]):
                imp = None
            else:
                imp = imp.to(device=w.device, dtype=torch.float32)
        return self.mse_optimizer.find_optimal_amax(w, imp, pack_mode=pack_mode)

    def compute_optimal_amax_with_stats(
        self,
        weight: torch.Tensor,
        importance: Optional[torch.Tensor] = None,
        scaled: bool = False,
        pack_mode: str = "auto",
    ) -> dict:
        _ = scaled
        w = weight.detach().float()
        max_val = float(w.abs().amax().item()) if w.numel() else 0.0
        if max_val <= 0.0:
            max_val = 1e-7
        optimal = self.compute_optimal_amax(
            w, importance, scaled=False, pack_mode=pack_mode
        )
        mode = pack_mode
        if mode == "auto":
            mode = "nvfp4" if w.ndim == 2 else "int8"
        if mode == "nvfp4":
            est = mse_after_clip_nvfp4(w, optimal, importance)
        else:
            est = mse_after_clip_int8(w, optimal, importance)
        return {
            "optimal_amax": optimal,
            "max_val": max_val,
            "compression_ratio": optimal / max_val if max_val > 0 else 1.0,
            "estimated_mse": est,
            "pack_mode": mode,
        }


# Backward-compatible short alias used by the converter loader.
HSWQWeightedHistogramOptimizerFast = HSWQWeightedHistogramOptimizerFastNVFP4


if __name__ == "__main__":
    print("HSWQ NVFP4 pack-roundtrip amax optimizer — self test")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device="cpu").manual_seed(0)
    w = torch.randn(64, 64, generator=g, dtype=torch.float32)
    w[0, 0] = 40.0
    w = w.to(device)
    opt = HSWQWeightedHistogramOptimizerFastNVFP4(device=device, num_candidates=16)
    stats = opt.compute_optimal_amax_with_stats(w, pack_mode="nvfp4")
    mse_full = mse_after_clip_nvfp4(w, stats["max_val"], None)
    print(
        f"  absmax={stats['max_val']:.6f} amax*={stats['optimal_amax']:.6f} "
        f"mse@absmax={mse_full:.8e} mse@amax*={stats['estimated_mse']:.8e}"
    )
    assert stats["estimated_mse"] <= mse_full * 1.0001 + 1e-12
    w4 = torch.randn(16, 8, 3, 3, generator=g, dtype=torch.float32).to(device)
    a4 = opt.compute_optimal_amax(w4, pack_mode="int8")
    print(f"  int8 conv amax*={a4:.6f}")
    print("OK")

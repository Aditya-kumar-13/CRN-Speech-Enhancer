from __future__ import annotations

import torch


def si_sdr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SDR in dB, returned per batch item."""
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    projection = (
        (estimate * target).sum(dim=-1, keepdim=True)
        * target
        / target.square().sum(dim=-1, keepdim=True).clamp_min(eps)
    )
    noise = estimate - projection
    ratio = projection.square().sum(dim=-1) / noise.square().sum(dim=-1).clamp_min(eps)
    return 10.0 * torch.log10(ratio.clamp_min(eps))


def snr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Signal-to-error ratio in dB, returned per batch item."""
    signal_power = target.square().sum(dim=-1)
    error_power = (target - estimate).square().sum(dim=-1)
    return 10.0 * torch.log10((signal_power / error_power.clamp_min(eps)).clamp_min(eps))


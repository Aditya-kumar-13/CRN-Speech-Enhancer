from __future__ import annotations

import torch


def rms(signal: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute RMS over the last dimension without returning exact zero."""
    return signal.square().mean(dim=-1, keepdim=True).add(eps).sqrt()


def apply_rir(signal: torch.Tensor, impulse_response: torch.Tensor) -> torch.Tensor:
    """Apply a room impulse response with FFT convolution and preserve signal length."""
    if signal.ndim != 1 or impulse_response.ndim != 1:
        raise ValueError("signal and impulse_response must both be one-dimensional.")
    if signal.numel() == 0 or impulse_response.numel() == 0:
        raise ValueError("signal and impulse_response must not be empty.")

    rir = impulse_response / impulse_response.square().sum().sqrt().clamp_min(1e-8)
    convolution_length = signal.numel() + rir.numel() - 1
    fft_size = 1 << (convolution_length - 1).bit_length()
    reverberant = torch.fft.irfft(
        torch.fft.rfft(signal, fft_size) * torch.fft.rfft(rir, fft_size),
        fft_size,
    )
    return reverberant[: signal.numel()]


def mix_at_snr(
    target: torch.Tensor,
    interferer: torch.Tensor,
    snr_db: torch.Tensor | float,
    *,
    peak: float = 0.99,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mix signals at target-to-interferer SNR and safely prevent clipping.

    Returns the mixture and the correspondingly scaled target and interferer. Scaling
    all three together preserves the requested SNR and the supervised ground truth.
    """
    if target.shape != interferer.shape:
        raise ValueError(f"Signal shapes must match, got {target.shape} and {interferer.shape}.")

    snr = torch.as_tensor(snr_db, dtype=target.dtype, device=target.device)
    while snr.ndim < target.ndim:
        snr = snr.unsqueeze(-1)

    desired_ratio = torch.pow(target.new_tensor(10.0), snr / 20.0)
    scaled_interferer = interferer * (rms(target) / (rms(interferer) * desired_ratio))
    mixture = target + scaled_interferer

    max_abs = mixture.abs().amax(dim=-1, keepdim=True).clamp_min(peak)
    gain = peak / max_abs
    return mixture * gain, target * gain, scaled_interferer * gain

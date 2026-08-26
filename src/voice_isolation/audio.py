from __future__ import annotations

import torch


def stft(
    waveform: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> torch.Tensor:
    """Return a complex STFT for waveforms shaped [batch, samples]."""
    window = torch.hann_window(win_length, device=waveform.device, dtype=waveform.dtype)
    return torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )


def istft(
    spectrum: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    length: int,
) -> torch.Tensor:
    """Invert a complex STFT to waveforms shaped [batch, samples]."""
    window = torch.hann_window(
        win_length,
        device=spectrum.device,
        dtype=spectrum.real.dtype,
    )
    return torch.istft(
        spectrum,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=length,
    )


def apply_magnitude_mask(mixture_stft: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply a real-valued time-frequency mask while retaining mixture phase."""
    if mixture_stft.shape != mask.shape:
        raise ValueError(
            f"Mixture spectrum and mask must have equal shapes, got "
            f"{mixture_stft.shape} and {mask.shape}."
        )
    return mixture_stft * mask


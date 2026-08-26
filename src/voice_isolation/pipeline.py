from __future__ import annotations

from collections.abc import Mapping

import torch

from voice_isolation.audio import apply_magnitude_mask, istft, stft
from voice_isolation.models import CRNMaskEstimator


def build_model(config: Mapping[str, object]) -> CRNMaskEstimator:
    model_config = config["model"]
    if not isinstance(model_config, Mapping):
        raise TypeError("The model configuration must be a mapping.")
    n_fft = int(config["n_fft"])
    channels = tuple(int(value) for value in model_config["channels"])
    if len(channels) != 3:
        raise ValueError("model.channels must contain exactly three values.")
    return CRNMaskEstimator(
        frequency_bins=n_fft // 2 + 1,
        channels=channels,
        gru_hidden=int(model_config["gru_hidden"]),
        gru_layers=int(model_config["gru_layers"]),
    )


def enhance_waveform(
    model: CRNMaskEstimator,
    mixture: torch.Tensor,
    config: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enhance a batch and return waveform, estimated magnitude, and mask."""
    spectrum = stft(
        mixture,
        n_fft=int(config["n_fft"]),
        hop_length=int(config["hop_length"]),
        win_length=int(config["win_length"]),
    )
    magnitude = spectrum.abs()
    mask = model(magnitude)
    estimated_spectrum = apply_magnitude_mask(spectrum, mask)
    enhanced = istft(
        estimated_spectrum,
        n_fft=int(config["n_fft"]),
        hop_length=int(config["hop_length"]),
        win_length=int(config["win_length"]),
        length=mixture.shape[-1],
    )
    return enhanced, estimated_spectrum.abs(), mask


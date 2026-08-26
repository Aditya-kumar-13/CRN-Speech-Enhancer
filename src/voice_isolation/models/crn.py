from __future__ import annotations

import torch
from torch import nn


class CRNMaskEstimator(nn.Module):
    """A compact CRN that estimates a real mask from magnitude spectra.

    Input and output shapes are [batch, frequency, frames]. The default architecture
    assumes a one-sided 512-point STFT, which has 257 frequency bins.
    """

    def __init__(
        self,
        *,
        frequency_bins: int = 257,
        channels: tuple[int, int, int] = (16, 32, 64),
        gru_hidden: int = 256,
        gru_layers: int = 1,
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.encoder1 = self._encoder_block(1, c1)
        self.encoder2 = self._encoder_block(c1, c2)
        self.encoder3 = self._encoder_block(c2, c3)

        encoded_bins = frequency_bins
        for _ in range(3):
            encoded_bins = (encoded_bins + 1) // 2
        self.encoded_bins = encoded_bins
        self.encoder_channels = c3

        self.gru = nn.GRU(
            input_size=c3 * encoded_bins,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.gru_projection = nn.Linear(gru_hidden, c3 * encoded_bins)

        self.decoder1 = self._decoder_block(c3, c2)
        self.decoder2 = self._decoder_block(c2 + c2, c1)
        self.decoder3 = nn.ConvTranspose2d(
            c1 + c1,
            1,
            kernel_size=(3, 3),
            stride=(2, 1),
            padding=(1, 1),
        )

    @staticmethod
    def _encoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(2, 1),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )

    @staticmethod
    def _decoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(2, 1),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )

    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        if magnitude.ndim != 3:
            raise ValueError(f"Expected [batch, frequency, frames], got {magnitude.shape}.")

        x = magnitude.unsqueeze(1)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        batch, channels, bins, frames = e3.shape
        sequence = e3.permute(0, 3, 1, 2).reshape(batch, frames, channels * bins)
        sequence, _ = self.gru(sequence)
        sequence = self.gru_projection(sequence)
        recurrent = sequence.reshape(batch, frames, channels, bins).permute(0, 2, 3, 1)

        d1 = self.decoder1(recurrent)
        d2 = self.decoder2(torch.cat((d1, e2), dim=1))
        logits = self.decoder3(torch.cat((d2, e1), dim=1))

        mask = torch.sigmoid(logits.squeeze(1))
        if mask.shape != magnitude.shape:
            raise RuntimeError(f"Mask shape {mask.shape} does not match input {magnitude.shape}.")
        return mask


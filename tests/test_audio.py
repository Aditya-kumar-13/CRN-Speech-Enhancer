import torch

from voice_isolation.audio import istft, stft


def test_stft_round_trip() -> None:
    torch.manual_seed(0)
    waveform = torch.randn(2, 16000)
    spectrum = stft(waveform, n_fft=512, hop_length=160, win_length=400)
    reconstructed = istft(
        spectrum,
        n_fft=512,
        hop_length=160,
        win_length=400,
        length=waveform.shape[-1],
    )
    assert torch.allclose(reconstructed, waveform, atol=1e-5)


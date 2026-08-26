import torch

from voice_isolation.mixing import apply_rir, mix_at_snr, rms


def test_mix_at_requested_snr() -> None:
    torch.manual_seed(0)
    target = torch.randn(3, 16000)
    interferer = torch.randn(3, 16000)
    requested = torch.tensor([-5.0, 0.0, 10.0])

    mixture, scaled_target, scaled_interferer = mix_at_snr(target, interferer, requested)
    measured = 20.0 * torch.log10(
        rms(scaled_target).squeeze(-1) / rms(scaled_interferer).squeeze(-1)
    )

    assert torch.allclose(mixture, scaled_target + scaled_interferer, atol=1e-6)
    assert torch.allclose(measured, requested, atol=1e-3)
    assert mixture.abs().max() <= 0.99001


def test_delta_rir_preserves_signal() -> None:
    signal = torch.randn(1600)
    impulse = torch.tensor([1.0])

    reverberant = apply_rir(signal, impulse)

    assert torch.allclose(reverberant, signal, atol=1e-5)

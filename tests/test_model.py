import torch

from voice_isolation.models import CRNMaskEstimator


def test_crn_returns_bounded_mask_with_input_shape() -> None:
    model = CRNMaskEstimator(frequency_bins=257, gru_hidden=32)
    magnitude = torch.rand(2, 257, 21)
    mask = model(magnitude)

    assert mask.shape == magnitude.shape
    assert torch.all(mask >= 0.0)
    assert torch.all(mask <= 1.0)

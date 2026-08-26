import torch

from voice_isolation.live import (
    RollingNearFieldProcessor,
    find_virtual_cable_output,
    parse_device,
)


def identity_enhancer(model, waveform, config):
    return waveform, None, None


def test_parse_device_accepts_id_or_name() -> None:
    assert parse_device("12") == 12
    assert parse_device("Microphone Array") == "Microphone Array"
    assert parse_device(None) is None


def test_rolling_processor_returns_only_current_block() -> None:
    processor = RollingNearFieldProcessor(
        torch.nn.Identity(),
        {"n_fft": 8},
        context_samples=4,
        enhancer=identity_enhancer,
    )
    first = torch.arange(8, dtype=torch.float32).unsqueeze(0)
    second = torch.arange(8, 16, dtype=torch.float32).unsqueeze(0)

    assert torch.equal(processor.process(first), first)
    assert torch.equal(processor.process(second), second)
    assert torch.equal(processor.history, second[..., -4:])


def test_virtual_cable_matches_input_host_api() -> None:
    devices = [
        {"id": 1, "name": "Microphone", "inputs": 1, "outputs": 0, "host_api_id": 0},
        {"id": 6, "name": "CABLE In", "inputs": 0, "outputs": 2, "host_api_id": 0},
        {"id": 18, "name": "CABLE In", "inputs": 0, "outputs": 2, "host_api_id": 2},
    ]

    assert find_virtual_cable_output(devices, 1) == 6

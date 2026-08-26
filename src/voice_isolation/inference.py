from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torchaudio

from voice_isolation.metrics import si_sdr, snr
from voice_isolation.pipeline import build_model, enhance_waveform


def load_audio(path: str | Path, sample_rate: int) -> torch.Tensor:
    """Load an audio file as a mono waveform shaped [1, samples]."""
    waveform, source_rate = torchaudio.load(str(path))
    waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    return waveform.float()


def default_output_path(input_path: str | Path) -> Path:
    path = Path(input_path)
    return path.with_name(f"{path.stem}_enhanced.wav")


@torch.inference_mode()
def enhance_file(
    *,
    input_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    reference_path: str | Path | None = None,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Enhance one audio file and return performance and optional quality metrics."""
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    sample_rate = int(config["sample_rate"])
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    mixture = load_audio(input_path, sample_rate).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    enhanced, _, _ = enhance_waveform(model, mixture, config)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started

    destination = Path(output_path) if output_path else default_output_path(input_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(destination), enhanced.cpu(), sample_rate)

    audio_seconds = mixture.shape[-1] / sample_rate
    result: dict[str, Any] = {
        "input": str(Path(input_path).resolve()),
        "output": str(destination.resolve()),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": str(device),
        "sample_rate": sample_rate,
        "audio_seconds": audio_seconds,
        "inference_ms": inference_seconds * 1000.0,
        "real_time_factor": inference_seconds / audio_seconds,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }

    if reference_path is not None:
        reference = load_audio(reference_path, sample_rate).to(device)
        length = min(mixture.shape[-1], enhanced.shape[-1], reference.shape[-1])
        mixture = mixture[..., :length]
        enhanced = enhanced[..., :length]
        reference = reference[..., :length]
        input_si_sdr = si_sdr(mixture, reference)
        output_si_sdr = si_sdr(enhanced, reference)
        input_snr = snr(mixture, reference)
        output_snr = snr(enhanced, reference)
        result.update(
            {
                "reference": str(Path(reference_path).resolve()),
                "input_si_sdr_db": float(input_si_sdr.item()),
                "output_si_sdr_db": float(output_si_sdr.item()),
                "si_sdr_improvement_db": float((output_si_sdr - input_si_sdr).item()),
                "input_snr_db": float(input_snr.item()),
                "output_snr_db": float(output_snr.item()),
                "snr_improvement_db": float((output_snr - input_snr).item()),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove background noise from an audio file.")
    parser.add_argument("input", help="Noisy input audio file.")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/near_field_baseline/best.pt",
        help="Trained checkpoint path.",
    )
    parser.add_argument("--output", help="Enhanced WAV path; defaults beside the input.")
    parser.add_argument("--reference", help="Optional clean reference for quality metrics.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    result = enhance_file(
        input_path=args.input,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        reference_path=args.reference,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

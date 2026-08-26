from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch.nn import functional

from voice_isolation.inference import load_audio
from voice_isolation.pipeline import build_model, enhance_waveform


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("At least one value is required.")
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


@torch.inference_mode()
def enhance_in_chunks(
    model: torch.nn.Module,
    mixture: torch.Tensor,
    config: dict[str, Any],
    *,
    chunk_samples: int,
    context_samples: int,
) -> tuple[torch.Tensor, list[float]]:
    """Simulate live block processing using left context and no future audio."""
    if mixture.ndim != 2:
        raise ValueError("mixture must have shape [batch, samples].")
    if chunk_samples < 1:
        raise ValueError("chunk_samples must be positive.")
    if context_samples < 0:
        raise ValueError("context_samples must not be negative.")

    device = mixture.device
    output = torch.empty_like(mixture)
    latencies_ms: list[float] = []
    minimum_samples = int(config["n_fft"])

    for start in range(0, mixture.shape[-1], chunk_samples):
        end = min(start + chunk_samples, mixture.shape[-1])
        context_start = max(0, start - context_samples)
        segment = mixture[..., context_start:end]
        segment_length = segment.shape[-1]
        if segment_length < minimum_samples:
            segment = functional.pad(segment, (0, minimum_samples - segment_length))

        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        enhanced, _, _ = enhance_waveform(model, segment, config)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - started) * 1000.0)

        enhanced = enhanced[..., :segment_length]
        offset = start - context_start
        output[..., start:end] = enhanced[..., offset : offset + (end - start)]

    return output, latencies_ms


@torch.inference_mode()
def simulate_streaming_file(
    *,
    input_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    chunk_ms: float = 100.0,
    context_ms: float = 200.0,
    device_name: str = "auto",
) -> dict[str, Any]:
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

    chunk_samples = round(sample_rate * chunk_ms / 1000.0)
    context_samples = round(sample_rate * context_ms / 1000.0)
    enhanced, latencies_ms = enhance_in_chunks(
        model,
        mixture,
        config,
        chunk_samples=chunk_samples,
        context_samples=context_samples,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(destination), enhanced.cpu(), sample_rate)

    audio_seconds = mixture.shape[-1] / sample_rate
    processing_seconds = sum(latencies_ms) / 1000.0
    return {
        "input": str(Path(input_path).resolve()),
        "output": str(destination.resolve()),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "device": str(device),
        "audio_seconds": audio_seconds,
        "chunk_ms": chunk_ms,
        "context_ms": context_ms,
        "chunks": len(latencies_ms),
        "mean_chunk_compute_ms": sum(latencies_ms) / len(latencies_ms),
        "p95_chunk_compute_ms": percentile(latencies_ms, 0.95),
        "max_chunk_compute_ms": max(latencies_ms),
        "processing_real_time_factor": processing_seconds / audio_seconds,
        "estimated_block_latency_ms": chunk_ms + percentile(latencies_ms, 0.95),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate blockwise near-field enhancement.")
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--checkpoint",
        default="artifacts/near_field_baseline/best.pt",
    )
    parser.add_argument("--chunk-ms", type=float, default=100.0)
    parser.add_argument("--context-ms", type=float, default=200.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    result = simulate_streaming_file(
        input_path=args.input,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        chunk_ms=args.chunk_ms,
        context_ms=args.context_ms,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

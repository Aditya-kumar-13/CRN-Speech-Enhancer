from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional

from voice_isolation.pipeline import build_model, enhance_waveform
from voice_isolation.streaming import percentile

Enhancer = Callable[
    [torch.nn.Module, torch.Tensor, dict[str, Any]],
    tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None],
]


def parse_device(value: str | None) -> int | str | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


class RollingNearFieldProcessor:
    """Enhance one live block using only that block and retained past context."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: dict[str, Any],
        *,
        context_samples: int,
        enhancer: Enhancer = enhance_waveform,
    ) -> None:
        if context_samples < 0:
            raise ValueError("context_samples must not be negative.")
        self.model = model
        self.config = config
        self.context_samples = context_samples
        self.enhancer = enhancer
        self.history: torch.Tensor | None = None

    @torch.inference_mode()
    def process(self, block: torch.Tensor) -> torch.Tensor:
        if block.ndim != 2:
            raise ValueError("block must have shape [batch, samples].")
        if self.history is None:
            segment = block
        else:
            segment = torch.cat((self.history, block), dim=-1)

        unpadded_length = segment.shape[-1]
        minimum_samples = int(self.config["n_fft"])
        model_input = segment
        if unpadded_length < minimum_samples:
            model_input = functional.pad(segment, (0, minimum_samples - unpadded_length))

        enhanced, _, _ = self.enhancer(self.model, model_input, self.config)
        enhanced = enhanced[..., :unpadded_length]
        output = enhanced[..., -block.shape[-1] :]

        if self.context_samples:
            self.history = segment[..., -self.context_samples :].detach()
        else:
            self.history = None
        return output


@dataclass
class LiveStats:
    compute_ms: list[float] = field(default_factory=list)
    captured_blocks: int = 0
    processed_blocks: int = 0
    input_drops: int = 0
    output_underflows: int = 0
    output_drops: int = 0
    callback_statuses: list[str] = field(default_factory=list)

    def summary(self, chunk_ms: float) -> dict[str, Any]:
        result: dict[str, Any] = {
            "chunk_ms": chunk_ms,
            "captured_blocks": self.captured_blocks,
            "processed_blocks": self.processed_blocks,
            "input_drops": self.input_drops,
            "output_underflows": self.output_underflows,
            "output_drops": self.output_drops,
            "callback_statuses": self.callback_statuses,
        }
        if self.compute_ms:
            result.update(
                {
                    "mean_compute_ms": sum(self.compute_ms) / len(self.compute_ms),
                    "p95_compute_ms": percentile(self.compute_ms, 0.95),
                    "max_compute_ms": max(self.compute_ms),
                    "keeps_up": percentile(self.compute_ms, 0.95) < chunk_ms,
                }
            )
        return result


def list_audio_devices() -> list[dict[str, Any]]:
    import sounddevice as sd

    host_apis = sd.query_hostapis()
    devices = []
    for index, device in enumerate(sd.query_devices()):
        devices.append(
            {
                "id": index,
                "name": device["name"],
                "host_api": host_apis[device["hostapi"]]["name"],
                "host_api_id": device["hostapi"],
                "inputs": device["max_input_channels"],
                "outputs": device["max_output_channels"],
                "default_sample_rate": device["default_samplerate"],
            }
        )
    return devices


def find_virtual_cable_output(
    devices: list[dict[str, Any]],
    input_device: int | str | None = None,
) -> int | None:
    input_host_api = None
    if isinstance(input_device, int):
        input_host_api = next(
            (
                device["host_api_id"]
                for device in devices
                if device["id"] == input_device
            ),
            None,
        )

    candidates = []
    for device in devices:
        name = str(device["name"]).lower()
        is_cable_playback = "cable input" in name or "cable in" in name
        if device["outputs"] > 0 and is_cable_playback:
            candidates.append(device)
    if input_host_api is not None:
        matching = [
            device for device in candidates if device["host_api_id"] == input_host_api
        ]
        if matching:
            return int(matching[0]["id"])
    if candidates:
        return int(candidates[0]["id"])
    return None


def run_live(
    *,
    checkpoint_path: str | Path,
    input_device: int | str | None,
    output_device: int | str | None,
    chunk_ms: float,
    context_ms: float,
    duration_seconds: float | None,
    compute_device_name: str,
) -> dict[str, Any]:
    import sounddevice as sd

    if chunk_ms <= 0.0:
        raise ValueError("chunk_ms must be positive.")
    if context_ms < 0.0:
        raise ValueError("context_ms must not be negative.")

    if compute_device_name == "auto":
        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        compute_device = torch.device(compute_device_name)

    checkpoint = torch.load(checkpoint_path, map_location=compute_device, weights_only=True)
    config = checkpoint["config"]
    sample_rate = int(config["sample_rate"])
    block_samples = round(sample_rate * chunk_ms / 1000.0)
    context_samples = round(sample_rate * context_ms / 1000.0)

    model = build_model(config).to(compute_device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    warmup = torch.zeros(
        1,
        max(int(config["n_fft"]), block_samples + context_samples),
        device=compute_device,
    )
    for _ in range(2):
        enhance_waveform(model, warmup, config)
    processor = RollingNearFieldProcessor(
        model,
        config,
        context_samples=context_samples,
    )

    devices = list_audio_devices()
    if output_device is None:
        output_device = find_virtual_cable_output(devices, input_device)
    if output_device is None:
        raise RuntimeError(
            "No VB-CABLE playback device was found. Install it or pass --output-device."
        )

    sd.check_input_settings(
        device=input_device,
        channels=1,
        dtype="float32",
        samplerate=sample_rate,
    )
    sd.check_output_settings(
        device=output_device,
        channels=1,
        dtype="float32",
        samplerate=sample_rate,
    )

    input_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=3)
    output_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=3)
    stop_event = threading.Event()
    stats = LiveStats()

    def callback(indata: np.ndarray, outdata: np.ndarray, frames: int, _time: Any, status: Any) -> None:
        if status:
            stats.callback_statuses.append(str(status))
        stats.captured_blocks += 1
        captured = indata[:, 0].copy()
        try:
            input_queue.put_nowait(captured)
        except queue.Full:
            stats.input_drops += 1
            try:
                input_queue.get_nowait()
            except queue.Empty:
                pass
            input_queue.put_nowait(captured)

        try:
            enhanced = output_queue.get_nowait()
            outdata[:, 0] = enhanced[:frames]
        except queue.Empty:
            stats.output_underflows += 1
            outdata.fill(0.0)

    def worker() -> None:
        while not stop_event.is_set() or not input_queue.empty():
            try:
                captured = input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            block = torch.from_numpy(captured).to(compute_device).unsqueeze(0)
            if compute_device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            enhanced = processor.process(block)
            if compute_device.type == "cuda":
                torch.cuda.synchronize()
            stats.compute_ms.append((time.perf_counter() - started) * 1000.0)
            stats.processed_blocks += 1
            rendered = enhanced.squeeze(0).cpu().numpy()
            try:
                output_queue.put_nowait(rendered)
            except queue.Full:
                stats.output_drops += 1
                try:
                    output_queue.get_nowait()
                except queue.Empty:
                    pass
                output_queue.put_nowait(rendered)

    worker_thread = threading.Thread(target=worker, name="voice-enhancer", daemon=True)
    worker_thread.start()
    started = time.perf_counter()
    try:
        with sd.Stream(
            device=(input_device, output_device),
            samplerate=sample_rate,
            blocksize=block_samples,
            channels=1,
            dtype="float32",
            latency="low",
            callback=callback,
        ):
            if duration_seconds is None:
                while True:
                    time.sleep(0.25)
            else:
                time.sleep(duration_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        worker_thread.join(timeout=2.0)

    result = stats.summary(chunk_ms)
    result.update(
        {
            "input_device": input_device,
            "output_device": output_device,
            "sample_rate": sample_rate,
            "context_ms": context_ms,
            "compute_device": str(compute_device),
            "wall_seconds": time.perf_counter() - started,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhance a microphone into a virtual cable.")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--input-device")
    parser.add_argument("--output-device")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/near_field_baseline/best.pt",
    )
    parser.add_argument("--chunk-ms", type=float, default=40.0)
    parser.add_argument("--context-ms", type=float, default=200.0)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--compute-device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.list_devices:
        print(json.dumps(list_audio_devices(), indent=2))
        return

    result = run_live(
        checkpoint_path=args.checkpoint,
        input_device=parse_device(args.input_device),
        output_device=parse_device(args.output_device),
        chunk_ms=args.chunk_ms,
        context_ms=args.context_ms,
        duration_seconds=args.duration,
        compute_device_name=args.compute_device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch.nn import functional
from torch.utils.data import Dataset

from voice_isolation.mixing import apply_rir, mix_at_snr


@dataclass(frozen=True)
class AudioRecord:
    path: Path
    speaker_id: str | None = None


def read_manifest(path: str | Path) -> list[AudioRecord]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    records: list[AudioRecord] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item: dict[str, Any] = json.loads(line)
            if "path" not in item:
                raise ValueError(f"Missing 'path' in {manifest_path}:{line_number}")
            records.append(AudioRecord(Path(item["path"]), item.get("speaker_id")))
    if not records:
        raise ValueError(f"Manifest contains no records: {manifest_path}")
    return records


def load_mono_segment(
    record: AudioRecord,
    *,
    sample_rate: int,
    samples: int,
    rng: random.Random,
) -> torch.Tensor:
    waveform, source_rate = torchaudio.load(str(record.path))
    waveform = waveform.mean(dim=0)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)

    if waveform.numel() < samples:
        waveform = functional.pad(waveform, (0, samples - waveform.numel()))
    elif waveform.numel() > samples:
        start = rng.randint(0, waveform.numel() - samples)
        waveform = waveform[start : start + samples]
    return waveform.float()


def load_impulse_response(
    record: AudioRecord,
    *,
    sample_rate: int,
    max_samples: int,
    rng: random.Random,
) -> torch.Tensor:
    """Load one RIR channel, resample it, and keep its causal beginning."""
    waveform, source_rate = torchaudio.load(str(record.path))
    channel = rng.randrange(waveform.shape[0])
    impulse_response = waveform[channel]
    if source_rate != sample_rate:
        impulse_response = torchaudio.functional.resample(
            impulse_response,
            source_rate,
            sample_rate,
        )
    impulse_response = impulse_response[:max_samples].float()
    impulse_response = impulse_response - impulse_response.mean()
    if impulse_response.abs().max() < 1e-8:
        raise ValueError(f"RIR contains no usable signal: {record.path}")
    return impulse_response


class DynamicMixtureDataset(Dataset[dict[str, torch.Tensor]]):
    """Create target-plus-interferer mixtures without saving redundant audio."""

    def __init__(
        self,
        *,
        speech_manifest: str | Path,
        noise_manifest: str | Path | None,
        rir_manifest: str | Path | None = None,
        sample_rate: int,
        segment_seconds: float,
        snr_db_min: float,
        snr_db_max: float,
        speech_interferer_probability: float = 0.5,
        reverberate_speech_interferer: bool = False,
        rir_max_seconds: float = 1.0,
        deterministic: bool = False,
        seed: int = 42,
    ) -> None:
        self.speech = read_manifest(speech_manifest)
        self.noise = read_manifest(noise_manifest) if noise_manifest is not None else []
        self.rirs = read_manifest(rir_manifest) if rir_manifest is not None else []
        self.sample_rate = sample_rate
        self.samples = round(sample_rate * segment_seconds)
        self.snr_db_min = snr_db_min
        self.snr_db_max = snr_db_max
        self.speech_interferer_probability = speech_interferer_probability
        self.reverberate_speech_interferer = reverberate_speech_interferer
        self.rir_max_samples = round(sample_rate * rir_max_seconds)
        if not 0.0 <= speech_interferer_probability <= 1.0:
            raise ValueError("speech_interferer_probability must be between 0 and 1.")
        if speech_interferer_probability < 1.0 and not self.noise:
            raise ValueError("A noise manifest is required when noise mixing is enabled.")
        if reverberate_speech_interferer and not self.rirs:
            raise ValueError("An RIR manifest is required for near-field simulation.")
        if self.rir_max_samples < 1:
            raise ValueError("rir_max_seconds must be positive.")
        self.deterministic = deterministic
        self.seed = seed

    def __len__(self) -> int:
        return len(self.speech)

    def _rng(self, index: int) -> random.Random:
        return random.Random(self.seed + index) if self.deterministic else random.Random()

    def _other_speaker(self, index: int, rng: random.Random) -> AudioRecord:
        target = self.speech[index]
        candidates = [
            item
            for item in self.speech
            if item.path != target.path
            and (target.speaker_id is None or item.speaker_id != target.speaker_id)
        ]
        if not candidates:
            raise ValueError("Speech manifest needs at least two distinct speakers.")
        return rng.choice(candidates)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = self._rng(index)
        target = load_mono_segment(
            self.speech[index],
            sample_rate=self.sample_rate,
            samples=self.samples,
            rng=rng,
        )

        speech_interferer = rng.random() < self.speech_interferer_probability
        if speech_interferer:
            interferer_record = self._other_speaker(index, rng)
        else:
            interferer_record = rng.choice(self.noise)
        interferer = load_mono_segment(
            interferer_record,
            sample_rate=self.sample_rate,
            samples=self.samples,
            rng=rng,
        )
        if speech_interferer and self.reverberate_speech_interferer:
            rir = load_impulse_response(
                rng.choice(self.rirs),
                sample_rate=self.sample_rate,
                max_samples=self.rir_max_samples,
                rng=rng,
            )
            interferer = apply_rir(interferer, rir)

        snr_db = rng.uniform(self.snr_db_min, self.snr_db_max)
        mixture, scaled_target, _ = mix_at_snr(target, interferer, snr_db)
        return {
            "mixture": mixture,
            "target": scaled_target,
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
            "speech_interferer": torch.tensor(speech_interferer),
        }

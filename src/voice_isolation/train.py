from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from voice_isolation.audio import stft
from voice_isolation.data import DynamicMixtureDataset
from voice_isolation.metrics import si_sdr
from voice_isolation.pipeline import build_model, enhance_waveform


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("Configuration root must be a mapping.")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(config: dict[str, Any], split: str) -> DynamicMixtureDataset:
    data = config["data"]
    return DynamicMixtureDataset(
        speech_manifest=data[f"{split}_speech"],
        noise_manifest=data[f"{split}_noise"],
        rir_manifest=data.get(f"{split}_rir"),
        sample_rate=int(config["sample_rate"]),
        segment_seconds=float(config["segment_seconds"]),
        snr_db_min=float(config["snr_db_min"]),
        snr_db_max=float(config["snr_db_max"]),
        speech_interferer_probability=float(
            config.get("speech_interferer_probability", 0.5)
        ),
        reverberate_speech_interferer=bool(
            config.get("reverberate_speech_interferer", False)
        ),
        rir_max_seconds=float(config.get("rir_max_seconds", 1.0)),
        deterministic=split != "train",
        seed=int(config["seed"]),
    )


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    improvements: list[float] = []
    maximum_batches = config["training"].get("max_validation_batches")
    for batch_index, batch in enumerate(loader):
        mixture = batch["mixture"].to(device)
        target = batch["target"].to(device)
        enhanced, estimated_magnitude, _ = enhance_waveform(model, mixture, config)
        target_spectrum = stft(
            target,
            n_fft=int(config["n_fft"]),
            hop_length=int(config["hop_length"]),
            win_length=int(config["win_length"]),
        )
        loss = torch.nn.functional.l1_loss(
            estimated_magnitude.clamp_min(1e-8).pow(0.3),
            target_spectrum.abs().clamp_min(1e-8).pow(0.3),
        )
        improvement = si_sdr(enhanced, target) - si_sdr(mixture, target)
        losses.append(float(loss))
        improvements.extend(improvement.cpu().tolist())
        if maximum_batches is not None and batch_index + 1 >= int(maximum_batches):
            break
    return {
        "loss": sum(losses) / len(losses),
        "si_sdr_improvement_db": sum(improvements) / len(improvements),
    }


def train(config: dict[str, Any]) -> Path:
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = config["training"]
    train_loader = DataLoader(
        make_dataset(config, "train"),
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=int(training["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        make_dataset(config, "valid"),
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
    )

    model = build_model(config).to(device)
    optimizer = Adam(model.parameters(), lr=float(training["learning_rate"]))
    checkpoint_dir = Path(training["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.pt"
    best_loss = float("inf")

    print(f"Training on {device}; parameters={sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch}")
        maximum_steps = training.get("max_steps_per_epoch")
        for batch_index, batch in enumerate(progress):
            mixture = batch["mixture"].to(device)
            target = batch["target"].to(device)
            mixture_spectrum = stft(
                mixture,
                n_fft=int(config["n_fft"]),
                hop_length=int(config["hop_length"]),
                win_length=int(config["win_length"]),
            )
            target_spectrum = stft(
                target,
                n_fft=int(config["n_fft"]),
                hop_length=int(config["hop_length"]),
                win_length=int(config["win_length"]),
            )
            mask = model(mixture_spectrum.abs())
            estimate = mixture_spectrum.abs() * mask
            loss = torch.nn.functional.l1_loss(
                estimate.clamp_min(1e-8).pow(0.3),
                target_spectrum.abs().clamp_min(1e-8).pow(0.3),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            progress.set_postfix(loss=f"{float(loss):.4f}")
            if maximum_steps is not None and batch_index + 1 >= int(maximum_steps):
                break

        metrics = validate(model, valid_loader, config, device)
        print(json.dumps({"epoch": epoch, **metrics}, indent=2))
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "validation": metrics,
                },
                checkpoint_path,
            )
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CRN voice-isolation baseline.")
    parser.add_argument("--config", default="configs/near_field_baseline.yaml")
    args = parser.parse_args()
    checkpoint = train(load_config(args.config))
    print(f"Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()

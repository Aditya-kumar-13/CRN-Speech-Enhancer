from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch.utils.data import DataLoader

from voice_isolation.metrics import si_sdr, snr
from voice_isolation.pipeline import build_model, enhance_waveform
from voice_isolation.train import load_config, make_dataset


@torch.inference_mode()
def evaluate(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    split: str,
    save_examples: int,
    max_examples: int | None = None,
    output_label: str | None = None,
) -> dict[str, float | int | str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loader = DataLoader(make_dataset(config, split), batch_size=1, shuffle=False)
    si_sdr_gains: list[float] = []
    snr_gains: list[float] = []
    latencies_ms: list[float] = []
    output_dir = Path("artifacts") / "examples" / (output_label or split)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, batch in enumerate(loader):
        mixture = batch["mixture"].to(device)
        target = batch["target"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        enhanced, _, _ = enhance_waveform(model, mixture, config)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

        si_sdr_gains.extend((si_sdr(enhanced, target) - si_sdr(mixture, target)).cpu().tolist())
        snr_gains.extend((snr(enhanced, target) - snr(mixture, target)).cpu().tolist())

        if index < save_examples:
            sample_rate = int(config["sample_rate"])
            torchaudio.save(output_dir / f"{index:03d}_mixture.wav", mixture.cpu(), sample_rate)
            torchaudio.save(output_dir / f"{index:03d}_enhanced.wav", enhanced.cpu(), sample_rate)
            torchaudio.save(output_dir / f"{index:03d}_target.wav", target.cpu(), sample_rate)
        if max_examples is not None and index + 1 >= max_examples:
            break

    return {
        "split": split,
        "examples": len(si_sdr_gains),
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "si_sdr_improvement_db": sum(si_sdr_gains) / len(si_sdr_gains),
        "snr_improvement_db": sum(snr_gains) / len(snr_gains),
        "mean_file_latency_ms": sum(latencies_ms) / len(latencies_ms),
    }


def config_at_snr(config: dict[str, Any], snr_db: float) -> dict[str, Any]:
    """Return an isolated config fixed to one target-to-interferer SNR."""
    fixed_config = copy.deepcopy(config)
    fixed_config["snr_db_min"] = float(snr_db)
    fixed_config["snr_db_max"] = float(snr_db)
    return fixed_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a voice-isolation checkpoint.")
    parser.add_argument("--config", default="configs/near_field_baseline.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--save-examples", type=int, default=5)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument(
        "--snr-db",
        type=float,
        nargs="+",
        help="Evaluate at one or more fixed SNRs, for example: --snr-db -5 0 5 10",
    )
    args = parser.parse_args()
    base_config = load_config(args.config)
    if args.snr_db:
        results = []
        for snr_db in args.snr_db:
            label = f"{args.split}_snr_{snr_db:+g}db".replace("+", "p").replace("-", "m")
            metrics = evaluate(
                config_at_snr(base_config, snr_db),
                args.checkpoint,
                args.split,
                args.save_examples,
                args.max_examples,
                output_label=label,
            )
            results.append({"snr_db": snr_db, **metrics})
        print(json.dumps({"snr_sweep": results}, indent=2))
    else:
        metrics = evaluate(
            base_config,
            args.checkpoint,
            args.split,
            args.save_examples,
            args.max_examples,
        )
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

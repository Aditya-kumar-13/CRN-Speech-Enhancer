from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def collect_by_speaker(root: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*.flac")):
        relative = path.relative_to(root)
        if len(relative.parts) < 3:
            raise ValueError(f"Unexpected LibriSpeech path layout: {path}")
        grouped[relative.parts[0]].append(path.resolve())
    if len(grouped) < 4:
        raise ValueError(f"Expected at least four speakers below {root}; found {len(grouped)}.")
    return dict(grouped)


def write_manifest(path: Path, grouped: dict[str, list[Path]], speakers: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for speaker in sorted(speakers):
            for audio_path in grouped[speaker]:
                handle.write(
                    json.dumps({"path": audio_path.as_posix(), "speaker_id": speaker}) + "\n"
                )
                count += 1
    return count


def prepare(
    train_root: Path,
    test_root: Path,
    output_dir: Path,
    *,
    valid_fraction: float,
    seed: int,
) -> dict[str, int]:
    if not 0.05 <= valid_fraction <= 0.5:
        raise ValueError("valid_fraction must be between 0.05 and 0.5.")

    train_grouped = collect_by_speaker(train_root)
    test_grouped = collect_by_speaker(test_root)
    train_speakers = sorted(train_grouped)
    test_speakers = sorted(test_grouped)
    overlap = set(train_speakers) & set(test_speakers)
    if overlap:
        raise ValueError(f"Train and test speaker IDs overlap: {sorted(overlap)}")

    random.Random(seed).shuffle(train_speakers)
    validation_count = max(2, round(len(train_speakers) * valid_fraction))
    valid_speakers = train_speakers[:validation_count]
    fitted_speakers = train_speakers[validation_count:]
    if len(fitted_speakers) < 2:
        raise ValueError("Training split needs at least two speakers.")

    counts = {
        "train_files": write_manifest(
            output_dir / "train_speech.jsonl", train_grouped, fitted_speakers
        ),
        "valid_files": write_manifest(
            output_dir / "valid_speech.jsonl", train_grouped, valid_speakers
        ),
        "test_files": write_manifest(output_dir / "test_speech.jsonl", test_grouped, test_speakers),
        "train_speakers": len(fitted_speakers),
        "valid_speakers": len(valid_speakers),
        "test_speakers": len(test_speakers),
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps({**counts, "seed": seed, "valid_fraction": valid_fraction}, indent=2),
        encoding="utf-8",
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare speaker-disjoint Mini LibriSpeech splits.")
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    counts = prepare(
        args.train_root,
        args.test_root,
        args.output_dir,
        valid_fraction=args.valid_fraction,
        seed=args.seed,
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()


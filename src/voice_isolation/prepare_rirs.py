from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def prepare_rir_manifests(
    rir_root: Path,
    output_dir: Path,
    *,
    valid_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    """Split RIRs by room directory so evaluation rooms are unseen during training."""
    if valid_fraction <= 0.0 or test_fraction <= 0.0:
        raise ValueError("Validation and test fractions must be positive.")
    if valid_fraction + test_fraction >= 1.0:
        raise ValueError("Validation and test fractions must sum to less than one.")

    rooms: dict[Path, list[Path]] = {}
    for path in sorted(rir_root.rglob("*.wav")):
        rooms.setdefault(path.parent, []).append(path.resolve())
    if len(rooms) < 3:
        raise ValueError("At least three room directories are required.")

    room_paths = list(rooms)
    random.Random(seed).shuffle(room_paths)
    valid_count = max(1, round(len(room_paths) * valid_fraction))
    test_count = max(1, round(len(room_paths) * test_fraction))
    split_rooms = {
        "valid": room_paths[:valid_count],
        "test": room_paths[valid_count : valid_count + test_count],
        "train": room_paths[valid_count + test_count :],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split, selected_rooms in split_rooms.items():
        records = [path for room in selected_rooms for path in rooms[room]]
        destination = output_dir / f"{split}_rir.jsonl"
        with destination.open("w", encoding="utf-8") as handle:
            for path in records:
                handle.write(json.dumps({"path": path.as_posix()}) + "\n")
        counts[f"{split}_rooms"] = len(selected_rooms)
        counts[f"{split}_files"] = len(records)

    (output_dir / "rir_split_summary.json").write_text(
        json.dumps(counts, indent=2),
        encoding="utf-8",
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Create room-disjoint RIR manifests.")
    parser.add_argument("--rir-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    counts = prepare_rir_manifests(
        args.rir_root,
        args.output_dir,
        valid_fraction=args.valid_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()

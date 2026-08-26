from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_manifest(audio_root: Path, output: Path, speaker_parent_level: int | None) -> int:
    paths = sorted(
        path
        for path in audio_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".wav", ".flac"}
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for path in paths:
            item = {"path": path.resolve().as_posix()}
            if speaker_parent_level is not None:
                try:
                    item["speaker_id"] = path.parents[speaker_parent_level].name
                except IndexError as error:
                    raise ValueError(f"Cannot derive speaker for shallow path: {path}") from error
            handle.write(json.dumps(item) + "\n")
    return len(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a JSONL audio manifest.")
    parser.add_argument("audio_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--speaker-parent-level",
        type=int,
        help="0 means the audio file's immediate parent; omit for noise manifests.",
    )
    args = parser.parse_args()
    count = build_manifest(args.audio_root, args.output, args.speaker_parent_level)
    print(f"Wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()


import json
from pathlib import Path

from voice_isolation.prepare_minilibri import prepare


def _make_layout(root: Path, speakers: list[str]) -> None:
    for speaker in speakers:
        chapter = root / speaker / "1"
        chapter.mkdir(parents=True)
        (chapter / f"{speaker}-1-0000.flac").touch()


def test_prepare_creates_disjoint_speaker_splits(tmp_path: Path) -> None:
    train_root = tmp_path / "train"
    test_root = tmp_path / "test"
    output = tmp_path / "manifests"
    _make_layout(train_root, ["1", "2", "3", "4", "5"])
    _make_layout(test_root, ["10", "11", "12", "13"])

    counts = prepare(train_root, test_root, output, valid_fraction=0.2, seed=42)

    split_speakers = {}
    for split in ("train", "valid", "test"):
        records = [
            json.loads(line)
            for line in (output / f"{split}_speech.jsonl").read_text().splitlines()
        ]
        split_speakers[split] = {record["speaker_id"] for record in records}

    assert counts["train_speakers"] == 3
    assert counts["valid_speakers"] == 2
    assert counts["test_speakers"] == 4
    assert split_speakers["train"].isdisjoint(split_speakers["valid"])
    assert split_speakers["train"].isdisjoint(split_speakers["test"])
    assert split_speakers["valid"].isdisjoint(split_speakers["test"])

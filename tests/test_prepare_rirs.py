import json
from pathlib import Path

from voice_isolation.prepare_rirs import prepare_rir_manifests


def test_prepare_rirs_keeps_rooms_in_one_split(tmp_path: Path) -> None:
    rir_root = tmp_path / "rirs"
    output_dir = tmp_path / "manifests"
    for room_index in range(10):
        room = rir_root / f"Room{room_index:03d}"
        room.mkdir(parents=True)
        for file_index in range(2):
            (room / f"rir-{file_index}.wav").touch()

    counts = prepare_rir_manifests(rir_root, output_dir, seed=7)

    split_rooms: list[set[str]] = []
    for split in ("train", "valid", "test"):
        records = [
            json.loads(line)
            for line in (output_dir / f"{split}_rir.jsonl").read_text().splitlines()
        ]
        split_rooms.append({Path(record["path"]).parent.name for record in records})

    assert counts["train_rooms"] == 8
    assert counts["valid_rooms"] == 1
    assert counts["test_rooms"] == 1
    assert split_rooms[0].isdisjoint(split_rooms[1])
    assert split_rooms[0].isdisjoint(split_rooms[2])
    assert split_rooms[1].isdisjoint(split_rooms[2])

from pathlib import Path

from voice_isolation.inference import default_output_path


def test_default_output_path_is_beside_input() -> None:
    path = Path("recordings") / "meeting.wav"
    assert default_output_path(path) == Path("recordings") / "meeting_enhanced.wav"

import pytest

from voice_isolation.streaming import percentile


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([5.0, 1.0, 3.0, 2.0, 4.0], 0.95) == 5.0
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 2.0


def test_percentile_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="At least one"):
        percentile([], 0.95)

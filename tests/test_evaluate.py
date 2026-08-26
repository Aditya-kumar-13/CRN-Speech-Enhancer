from voice_isolation.evaluate import config_at_snr


def test_config_at_snr_does_not_mutate_original() -> None:
    original = {"snr_db_min": -5.0, "snr_db_max": 10.0, "data": {"split": "test"}}

    fixed = config_at_snr(original, 5.0)

    assert fixed["snr_db_min"] == 5.0
    assert fixed["snr_db_max"] == 5.0
    assert original["snr_db_min"] == -5.0
    assert fixed["data"] is not original["data"]

import pytest

from stageload.units import parse_duration, parse_size


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("8G", 8 * 2**30),
        ("512M", 512 * 2**20),
        ("1.5g", int(1.5 * 2**30)),
        ("100", 100),
        ("2GiB", 2 * 2**30),
    ],
)
def test_sizes(text, value):
    assert parse_size(text) == value


@pytest.mark.parametrize(
    ("text", "value"),
    [("30m", 1800.0), ("90s", 90.0), ("2h", 7200.0), ("5", 5.0), ("250ms", 0.25)],
)
def test_durations(text, value):
    assert parse_duration(text) == value


@pytest.mark.parametrize("bad", ["", "G8", "8X", "-1G"])
def test_bad_sizes_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_size(bad)

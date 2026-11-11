import pandas as pd
import pytest

from pv_forecast_arena.evaluation.splits import (
    development_splits,
    holdout_splits,
    issue_time,
    walk_forward_splits,
)

TZ = "America/Los_Angeles"
CFG = {
    "train_start": "2024-02-01",
    "holdout_start": "2025-01-01",
    "refit_every_months": 1,
    "min_train_months": 4,
}


@pytest.fixture
def index():
    return pd.date_range("2024-02-01", "2025-12-31 23:00", freq="1h", tz="UTC")


def test_no_test_timestamp_in_train(index):
    for split in [*development_splits(index, CFG), *holdout_splits(index, CFG)]:
        assert split.train_index.max() < split.test_index.min()
        assert not split.train_index.intersection(split.test_index).size


def test_train_is_known_at_first_issue_time(index):
    for split in [*development_splits(index, CFG), *holdout_splits(index, CFG)]:
        first_issue = issue_time(split.test_index[:1], TZ, 10)[0]
        assert split.train_index.max() <= first_issue


@pytest.mark.parametrize(
    ("valid", "expected"),
    [
        ("2025-01-15 20:00", "2025-01-14 18:00"),  # PST, 12:00 local
        ("2025-07-15 19:00", "2025-07-14 17:00"),  # PDT, 12:00 local
        ("2025-01-16 08:00", "2025-01-14 18:00"),  # hour ending at local midnight is still Jan 15
        ("2025-01-16 09:00", "2025-01-15 18:00"),  # first hour of Jan 16
        ("2025-03-09 20:00", "2025-03-08 18:00"),  # DST day: issued the day before, still PST
    ],
)
def test_issue_time(valid, expected):
    got = issue_time(pd.DatetimeIndex([pd.Timestamp(valid, tz="UTC")]), TZ, 10)[0]
    assert got == pd.Timestamp(expected, tz="UTC")


def test_development_stays_before_holdout(index):
    splits = list(development_splits(index, CFG))
    assert splits[0].test_start == pd.Timestamp("2024-06-01", tz="UTC")
    assert max(s.test_index.max() for s in splits) < pd.Timestamp("2025-01-01", tz="UTC")
    assert max(s.train_index.max() for s in splits) < pd.Timestamp("2025-01-01", tz="UTC")


def test_holdout_covers_exactly_the_holdout_year(index):
    splits = list(holdout_splits(index, CFG))
    assert len(splits) == 12
    tests = splits[0].test_index
    for s in splits[1:]:
        tests = tests.append(s.test_index)
    assert tests.is_unique
    assert tests.min() == pd.Timestamp("2025-01-01 00:00", tz="UTC")
    assert tests.max() == index.max()
    assert len(tests) == (index >= "2025-01-01").sum()


def test_training_window_expands(index):
    sizes = [len(s.train_index) for s in holdout_splits(index, CFG)]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)


def test_rejects_naive_or_unsorted_index():
    naive = pd.date_range("2024-02-01", periods=100, freq="1h")
    with pytest.raises(ValueError):
        list(walk_forward_splits(naive, "2024-02-01", "2024-03-01"))
    shuffled = pd.date_range("2024-02-01", periods=100, freq="1h", tz="UTC")[::-1]
    with pytest.raises(ValueError):
        list(walk_forward_splits(shuffled, "2024-02-01", "2024-03-01"))

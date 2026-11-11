import pandas as pd
import pytest

from pv_forecast_arena.evaluation.splits import (
    development_splits,
    holdout_splits,
    walk_forward_splits,
)

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


def test_train_ends_before_embargo(index):
    for split in holdout_splits(index, CFG):
        assert split.train_index.max() < split.test_start - pd.Timedelta("1D")


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

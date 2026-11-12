import numpy as np
import pytest
from test_no_leakage import CFG, PHYSICS_CFG, make_frame

from pv_forecast_arena.evaluation import quantile as scoring
from pv_forecast_arena.evaluation.splits import Split
from pv_forecast_arena.models import quantile

SMALL = {"min_train_hours": 5, "params": {"min_data_in_leaf": 2, "min_child_weight": 2}}
LEVELS = [0.1, 1 / 3, 0.5, 0.9]


def _cfg(model_type, **extra):
    model = {**SMALL, "physics": PHYSICS_CFG, "target": "residual", "rolling_days": 3, **extra}
    return {**CFG, "model": model, "quantile": {"levels": LEVELS}}


def _split(frame, train_days=6):
    return Split(frame.index[: 24 * train_days], frame.index[24 * train_days :])


@pytest.mark.parametrize("predict", [quantile.predict_empirical, quantile.predict_lgbm])
@pytest.mark.parametrize("recalibrate", [False, True])
def test_quantiles_are_ordered_non_negative_and_zero_at_night(predict, recalibrate):
    frame = make_frame()
    split = _split(frame)
    out = predict(frame, split, _cfg("x", recalibrate=recalibrate, val_days=1))
    assert list(out.columns) == [quantile.column(q) for q in LEVELS]
    assert (out.diff(axis=1).iloc[:, 1:] >= 0).all().all()
    assert (out >= 0).all().all()
    night = ~frame.loc[split.test_index, "is_daytime"].to_numpy()
    assert (out.to_numpy()[night] == 0).all()
    assert out.notna().all().all()


def test_column_names():
    assert quantile.column(0.1) == "q10"
    assert quantile.column(1 / 3) == "q33.33"
    assert quantile.column(0.5) == "q50"


def test_pinball_rows_and_optimal_level():
    truth = np.array([10.0, 10.0])
    assert scoring.pinball_rows(truth, np.array([8.0, 12.0]), 0.9).tolist() == pytest.approx(
        [1.8, 0.2]
    )
    assert scoring.optimal_level(2.0, 1.0) == pytest.approx(1 / 3)
    assert scoring.optimal_level(1.0, 1.0) == 0.5


def test_imbalance_cost_charges_shortfall_and_surplus_separately():
    truth = np.array([10.0, 10.0])
    cost = scoring.imbalance_cost(truth, np.array([13.0, 6.0]), c_short=2.0, c_surplus=1.0)
    assert cost.tolist() == [6.0, 4.0]


def test_the_optimal_quantile_minimises_the_expected_imbalance_cost():
    rng = np.random.default_rng(0)
    truth = rng.gamma(4.0, 50.0, 20000)
    bids = np.linspace(0, 600, 601)
    costs = [scoring.imbalance_cost(truth, b, 2.0, 1.0).mean() for b in bids]
    best = bids[int(np.argmin(costs))]
    assert best == pytest.approx(np.quantile(truth, scoring.optimal_level(2.0, 1.0)), abs=3.0)


def test_ratio_interval_contains_the_point_estimate_and_is_degenerate_for_equal_losses():
    days = np.repeat(np.arange(20), 5)
    loss = np.random.default_rng(1).random(100) + 0.5
    assert scoring.ratio_interval(loss, loss, days) == (0.0, 0.0)
    lo, hi = scoring.ratio_interval(0.8 * loss, loss, days)
    assert lo == pytest.approx(0.2) and hi == pytest.approx(0.2)


def test_scoring_runs_on_model_output():
    frame = make_frame()
    cfg = _cfg("x", val_days=1)
    split = _split(frame)
    model = quantile.predict_lgbm(frame, split, cfg)
    reference = quantile.predict_empirical(frame, split, cfg)
    row = scoring.score(frame, model, reference, cfg, "all")
    assert row["n_hours"] > 0 and 0.0 <= row["coverage_80"] <= 1.0
    assert row["skill_pinball_lo"] <= row["skill_pinball"] <= row["skill_pinball_hi"] + 1e-9
    same = scoring.score(frame, reference, reference, cfg, "all")
    assert same["skill_pinball"] == 0.0
    table = scoring.cost_table(
        frame,
        {"a": model["q50"], "b": reference["q50"]},
        {"s": {"c_short": 2.0, "c_surplus": 1.0}},
        cfg,
        reference="b",
    )
    assert set(table["strategy"]) == {"a", "b"}
    assert table.loc[table["strategy"] == "b", "saving_vs_b"].iloc[0] == 0.0

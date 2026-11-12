from pv_forecast_arena.config import load_config


def test_experiment_overrides_a_single_nested_key():
    cfg = load_config("configs/experiments/hybrid_win3.yaml")
    base = load_config("configs/base.yaml")
    assert cfg["evaluation"]["train_window_months"] == 3
    assert cfg["evaluation"]["min_train_months"] == 6
    for key in ["holdout_start", "train_start", "refit_every_months", "second_reference"]:
        assert cfg["evaluation"][key] == base["evaluation"][key]
    assert cfg["model"]["target"] == "residual"

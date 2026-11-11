from pv_forecast_arena.evaluation.metrics import coverage, mbe, nmae, nrmse, pinball_loss, skill_score


def test_perfect_forecast():
    y = [0, 50, 100]
    assert nrmse(y, y, 100) == 0
    assert nmae(y, y, 100) == 0


def test_bias_sign():
    assert mbe([10, 10], [12, 12], 100) == 0.02  # over-forecast -> positive


def test_skill_and_pinball():
    assert skill_score(5, 10) == 0.5
    assert pinball_loss([10], [8], 0.9) == 0.9 * 2
    assert coverage([1, 5, 9], [0, 0, 0], [6, 6, 6]) == 2 / 3

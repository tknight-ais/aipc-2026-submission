from aipc.stages.power import simulate

CONDITIONS = ["A_zero_shot", "C_structured", "D_debate"]


def test_a_large_clean_effect_is_usually_detected():
    r = simulate(CONDITIONS, 2, 4, 6, effect=3.0, essay_sd=0.0, tie_rate=0.0, reps=5, resamples=50, seed=1)
    assert r.pair_power >= 0.8
    assert r.robust_power <= r.pair_power
    assert r.judgments > 0


def test_no_effect_is_rarely_detected():
    r = simulate(CONDITIONS, 2, 4, 6, effect=0.0, essay_sd=0.0, tie_rate=0.26, reps=5, resamples=50, seed=1)
    assert r.pair_power <= 0.4

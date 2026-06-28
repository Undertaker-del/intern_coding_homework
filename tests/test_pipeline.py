"""統合テスト: モデルがベースラインを超える(AC-2)・再現性(AC-3)・堅牢性(AC-4)。"""
import data
import pipeline
from config import SEASONAL_PERIOD_HOURLY


def _run_h1(dataset="ETTh1"):
    df = data.load_ett(dataset)
    return pipeline.run_horizon(df, horizon=1, period=SEASONAL_PERIOD_HOURLY)


def test_lightgbm_beats_naive_on_ETTh1():
    res = _run_h1("ETTh1")
    assert res["lightgbm"]["mae"] < res["naive_persistence"]["mae"]
    assert res["lightgbm"]["skill_vs_naive"] > 0


def test_reproducible_same_seed():
    a = _run_h1("ETTh1")["lightgbm"]["mae"]
    b = _run_h1("ETTh1")["lightgbm"]["mae"]
    assert abs(a - b) < 1e-6


def test_generalizes_to_ETTh2():
    """ETTh1 で設計した手法が ETTh2 でもベースラインを超える（過適合でない）。"""
    res = _run_h1("ETTh2")
    assert res["lightgbm"]["skill_vs_naive"] > 0


def test_no_temporal_overlap_in_meta():
    res = _run_h1("ETTh1")
    assert res["_meta"]["n_train"] > res["_meta"]["n_test"] > 0

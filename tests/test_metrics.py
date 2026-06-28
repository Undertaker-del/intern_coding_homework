"""評価指標の単体テスト（MASE・区間・skill）。"""
import numpy as np

import evaluate as ev


def test_mae_and_rmse_zero_on_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert ev.mae(y, y) == 0.0
    assert ev.rmse(y, y) == 0.0


def test_mase_perfect_is_zero_and_naive_is_one():
    # 学習系列 = ランダムウォーク的、in-sample 1段ナイーブで正規化
    train = np.cumsum(np.ones(50))            # 一定増分 → 1段差分=1
    y_true = np.array([10.0, 11.0, 12.0])
    assert ev.mase(y_true, y_true, train, m=1) == 0.0
    # 予測が常に +1 ずれる → MAE=1、分母も1 → MASE=1
    assert ev.mase(y_true, y_true + 1.0, train, m=1) == 1.0


def test_interval_coverage_full_and_empty():
    y = np.array([5.0, 6.0, 7.0])
    full = ev.interval_metrics(y, y - 1, y + 1, nominal=0.8)
    assert full["coverage"] == 1.0 and full["mean_width"] == 2.0
    empty = ev.interval_metrics(y, y + 5, y + 6, nominal=0.8)
    assert empty["coverage"] == 0.0


def test_skill_sign():
    assert ev.skill(0.5, 1.0) == 0.5       # 半分の誤差 → skill 0.5
    assert ev.skill(2.0, 1.0) == -1.0      # ベースラインより悪い → 負

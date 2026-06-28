"""リーケージ防止の検証（本 PoC の中核 = AC-1）。"""
import numpy as np
import pandas as pd
import pytest

import data
import features
from config import DATE_COL, LOAD_COLS, TARGET


def _ramp_df(n=400):
    """OT(t)=t の単調ランプ系列。未来漏洩を値で検出するための合成データ。"""
    ts = pd.date_range("2020-01-01", periods=n, freq="h")
    df = pd.DataFrame({DATE_COL: ts, TARGET: np.arange(n, dtype=float)})
    for c in LOAD_COLS:
        df[c] = np.arange(n, dtype=float)
    return df


def test_chronological_split_is_ordered_and_disjoint():
    idx = data.chronological_split(1000)
    assert idx.train[0] == 0
    assert idx.train[1] == idx.val[0]      # 隣接・重複なし
    assert idx.val[1] == idx.test[0]
    assert idx.test[1] == 1000
    assert idx.train[1] > idx.train[0] and idx.test[1] > idx.test[0]


def test_rolling_folds_are_ordered_and_expanding():
    folds = data.rolling_origin_folds(17420, n_folds=5)
    assert len(folds) >= 3
    prev_test_end = -1
    for f in folds:
        # 各フォールド内で train < val < test、重複なし
        assert f.train[0] == 0                      # 拡張窓（先頭から）
        assert f.train[1] == f.val[0]
        assert f.val[1] == f.test[0]
        assert f.test[1] > f.test[0]
        # test 窓は後方へ単調移動（互いに前進）
        assert f.test[1] > prev_test_end
        prev_test_end = f.test[1]


def test_split_preserves_time_order_on_real_data():
    df = data.load_ett("ETTh1")
    idx = data.chronological_split(len(df))
    tr, va, te = data.slice_split(df, idx)
    assert tr[DATE_COL].max() < va[DATE_COL].min()
    assert va[DATE_COL].max() < te[DATE_COL].min()


def test_scaler_fitted_on_train_only():
    df = data.load_ett("ETTh1")
    idx = data.chronological_split(len(df))
    tr, _, _ = data.slice_split(df, idx)
    scaler = data.StandardScalerFrame().fit(tr, [TARGET] + LOAD_COLS)
    # train 平均と一致し、全体平均とは一致しない（= 全体で fit していない）
    assert scaler.mean_[TARGET] == pytest.approx(tr[TARGET].mean())
    assert scaler.mean_[TARGET] != pytest.approx(df[TARGET].mean())


def test_forecast_features_contain_no_future_information():
    """ランプ系列で、各行の OT 由来特徴が基準時刻の値を超えない（未来不参照）。"""
    h = 24
    df = _ramp_df()
    X, y, dates = features.make_supervised(df, horizon=h,
                                           use_future_covariates=False)
    ot_cols = [c for c in X.columns if c.startswith(TARGET)]
    base = X[f"{TARGET}_now"].to_numpy()  # = OT(t) = t
    for c in ot_cols:
        # いずれの OT 特徴も「現在値」以下（=過去のみ参照）
        assert np.all(X[c].to_numpy() <= base + 1e-9), f"未来漏洩の疑い: {c}"
    # 目的は厳密に h 先
    assert np.all(y.to_numpy() == base + h)


def test_target_alignment_equals_future_value():
    h = 5
    df = _ramp_df()
    X, y, dates = features.make_supervised(df, horizon=h)
    # OT(t)=t なので y=OT(t+h)=t+h、現在値 + h と一致
    assert np.allclose(y.to_numpy(), X[f"{TARGET}_now"].to_numpy() + h)


def test_calendar_uses_target_timestamp():
    h = 3
    df = _ramp_df()
    X, y, dates = features.make_supervised(df, horizon=h)
    # 各目的時刻から h を引くと元系列の時刻に一致する（= 目的時刻 = base+h）
    base_times = dates - pd.Timedelta(hours=h)
    assert set(base_times).issubset(set(df[DATE_COL]))
    # 目的時刻は 1 時間刻みで連続
    assert (dates.diff().dropna() == pd.Timedelta(hours=1)).all()


def test_no_nan_in_supervised_matrix():
    df = data.load_ett("ETTh1")
    idx = data.chronological_split(len(df))
    scaler = data.StandardScalerFrame().fit(
        data.slice_split(df, idx)[0], [TARGET] + LOAD_COLS)
    dfs = scaler.transform(df)
    X, y, dates = features.make_supervised(dfs, horizon=24)
    assert not X.isna().any().any()
    assert not y.isna().any()


def test_nowcast_excludes_target_columns():
    """NOWCAST(h=0) は OT 由来特徴を持たない（目的のリーク防止）。"""
    df = _ramp_df()
    X, y, dates = features.make_supervised(df, horizon=0)
    assert not any(c.startswith(f"{TARGET}_") or c == f"{TARGET}_now"
                   for c in X.columns)
    assert np.all(y.to_numpy() == X["HUFL_now"].to_numpy())  # OT(t)=load(t)=t

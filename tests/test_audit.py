"""結果の正しさを独立に検証する監査テスト（検証の検証）。

- ノイズ下限テスト: 未来リークがあればモデルは理論下限を割って当ててしまう。
  純ノイズで MAE が下限(≈0.80)付近に留まる＝未来不参照の確証。
- 陽性対照: 故意に目的をリークさせると MAE が大きく改善する＝検出力の証明。
- 決定性: 同一フォールド2回で完全一致。
"""
import numpy as np
import pandas as pd

import data
import features
import models
import pipeline
from config import LOAD_COLS, TARGET
from evaluate import mae


def _noise_df(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"date": pd.date_range("2016-07-01", periods=n, freq="h"),
                       TARGET: rng.standard_normal(n)})
    for c in LOAD_COLS:
        df[c] = rng.standard_normal(n)
    return df


def test_model_cannot_beat_noise_floor():
    """純 N(0,1) ノイズの不可避 MAE は E|N(0,1)|≈0.798。

    未来リークがあればこれを大きく割って 0 付近に達するはず。
    モデル MAE が下限付近(>0.7)に留まることでリーク無しを確認する。
    """
    df = _noise_df()
    for h in (1, 24):
        res = pipeline.run_horizon(df, h, 24, with_future=False, with_intervals=False)
        assert res["lightgbm"]["mae"] > 0.7, f"h={h}: MAEが下限を割った=リーク疑い"


def test_planted_leak_is_detectable():
    """故意に OT(t+h) を特徴へ混入すると MAE が大きく改善する（検出力の証明）。

    通常パイプラインがこの改善を示さない＝リークしていない、と裏取りできる。
    """
    df = data.load_ett("ETTh1")
    n = len(df); ntr = int(n * 0.6); nva = int(n * 0.2)
    sc = data.StandardScalerFrame().fit(df.iloc[:ntr], [TARGET] + LOAD_COLS)
    dfs = sc.transform(df)
    X, y, dates = features.make_supervised(dfs, horizon=24)
    freq = df["date"].iloc[1] - df["date"].iloc[0]
    base = pd.to_datetime(dates) - 24 * freq
    tr_end = df["date"].iloc[ntr - 1]; va_end = df["date"].iloc[ntr + nva - 1]
    trm = base <= tr_end; vam = (base > tr_end) & (base <= va_end); tem = base > va_end

    def fit_eval(Xx):
        dtr = y[trm].to_numpy() - Xx[trm][f"{TARGET}_now"].to_numpy()
        dva = y[vam].to_numpy() - Xx[vam][f"{TARGET}_now"].to_numpy()
        m = models.LightGBMModel().fit(Xx[trm], dtr, eval_set=[(Xx[vam], dva)])
        pred = sc.inverse_transform_target(
            Xx[tem][f"{TARGET}_now"].to_numpy() + m.predict(Xx[tem]), TARGET)
        yt = sc.inverse_transform_target(y[tem].to_numpy(), TARGET)
        return mae(yt, pred)

    clean = fit_eval(X)
    Xleak = X.copy(); Xleak["LEAK_target"] = y.to_numpy()
    leaked = fit_eval(Xleak)
    # リーク版は大幅に改善（検出可能）。通常版はその水準に達しない。
    assert leaked < 0.6 * clean, "リークを混ぜても改善しない=検出力に問題"
    assert clean > 1.0, "通常版が既に低すぎる=潜在リーク疑い"


def test_wrapper_matches_raw_lightgbm():
    """自前 LightGBMModel ラッパーの予測が生 lightgbm と完全一致（実装バグ防止）。"""
    import lightgbm as lgb
    from config import SEED

    df = data.load_ett("ETTh2")
    idx = data.chronological_split(len(df))
    sc = data.StandardScalerFrame().fit(df.iloc[:idx.train[1]], [TARGET] + LOAD_COLS)
    dfs = sc.transform(df)
    X, y, dates = features.make_supervised(dfs, horizon=6)
    freq = df["date"].iloc[1] - df["date"].iloc[0]
    base = pd.to_datetime(dates) - 6 * freq
    tre = df["date"].iloc[idx.train[1] - 1]; vae = df["date"].iloc[idx.val[1] - 1]
    trm = base <= tre; vam = (base > tre) & (base <= vae); tem = base > vae
    dtr = y[trm].to_numpy() - X[trm]["OT_now"].to_numpy()
    dva = y[vam].to_numpy() - X[vam]["OT_now"].to_numpy()

    wrapped = models.LightGBMModel().fit(X[trm], dtr, eval_set=[(X[vam], dva)])
    params = dict(objective="regression_l1", n_estimators=600, learning_rate=0.05,
                  num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=-1,
                  verbose=-1, deterministic=True, force_row_wise=True)
    raw = lgb.LGBMRegressor(**params)
    raw.fit(X[trm], dtr, eval_set=[(X[vam], dva)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    assert np.allclose(wrapped.predict(X[tem]), raw.predict(X[tem]))


def test_delta_beats_direct_target():
    """差分予測が直接予測より明確に良い（設計判断の検証）。"""
    import verify_all as va

    df = data.load_ett("ETTh1")
    idx = data.chronological_split(len(df))
    delta = va._fold_skill(df, 6, idx, direct=False)
    direct = va._fold_skill(df, 6, idx, direct=True)
    assert delta > 0 > direct  # 差分は正、直接は naive 未満


def test_backtest_fold_is_deterministic():
    df = data.load_ett("ETTh1")
    fold = data.rolling_origin_folds(len(df), n_folds=5)[2]
    a = pipeline.run_horizon(df, 6, 24, idx=fold, with_future=False,
                             with_intervals=False)["lightgbm"]["mae"]
    b = pipeline.run_horizon(df, 6, 24, idx=fold, with_future=False,
                             with_intervals=False)["lightgbm"]["mae"]
    assert a == b

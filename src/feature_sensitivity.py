"""特徴量数の感度: 増やす/減らすと性能がどうなるかを検証。

最小 / 現行 / 拡張 の3セットを 5 フォールド・バックテストで比較し、
特徴量の過不足と最適な複雑度を示す（more is better ではないことの確認）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import data
import features
import models
from config import LOAD_COLS, REPORTS_DIR, TARGET
from evaluate import mae, skill

SETS = {
    "minimal": dict(target_lags=[1, 24], roll_windows=[], load_lags=[],
                    load_roll_windows=[]),
    "current": dict(),  # make_supervised の既定
    "expanded": dict(target_lags=list(range(1, 49)) + [72, 168],
                     roll_windows=[3, 6, 12, 24, 48],
                     load_lags=[1, 6, 24], load_roll_windows=[6, 24, 72]),
}


def _eval(df, h, idx, kw):
    sc = data.StandardScalerFrame().fit(
        df.iloc[idx.train[0]:idx.train[1]], [TARGET] + LOAD_COLS)
    dfs = sc.transform(df)
    X, y, dates = features.make_supervised(dfs, horizon=h,
                                           use_future_covariates=False, **kw)
    freq = df["date"].iloc[1] - df["date"].iloc[0]
    base = pd.to_datetime(dates) - h * freq
    tre = df["date"].iloc[idx.train[1] - 1]; vae = df["date"].iloc[idx.val[1] - 1]
    tee = df["date"].iloc[idx.test[1] - 1]
    trm = (base <= tre).to_numpy(); vam = ((base > tre) & (base <= vae)).to_numpy()
    tem = ((base > vae) & (base <= tee)).to_numpy()
    now = X[tem][f"{TARGET}_now"].to_numpy()
    yte_c = sc.inverse_transform_target(y[tem].to_numpy(), TARGET)
    naive_c = sc.inverse_transform_target(now, TARGET)
    dtr = y[trm].to_numpy() - X[trm][f"{TARGET}_now"].to_numpy()
    dva = y[vam].to_numpy() - X[vam][f"{TARGET}_now"].to_numpy()
    m = models.LightGBMModel().fit(X[trm], dtr, eval_set=[(X[vam], dva)])
    pred_c = sc.inverse_transform_target(now + m.predict(X[tem]), TARGET)
    return skill(mae(yte_c, pred_c), mae(yte_c, naive_c)), X.shape[1]


def run(datasets=("ETTh1", "ETTh2"), horizons=(6, 12)) -> dict:
    out = {}
    for ds in datasets:
        df = data.load_ett(ds)
        folds = data.rolling_origin_folds(len(df), 5)
        out[ds] = {}
        for name, kw in SETS.items():
            for h in horizons:
                sks, ncol = [], 0
                for f in folds:
                    s, ncol = _eval(df, h, f, kw)
                    sks.append(s)
                out[ds].setdefault(name, {"n_features": ncol})[f"h{h}"] = round(
                    float(np.mean(sks)), 4)
            r = out[ds][name]
            print(f"[{ds}] {name:8s} (特徴量{r['n_features']:3d}): " +
                  " ".join(f"h{h}={r[f'h{h}']:+.3f}" for h in horizons))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "feature_sensitivity.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[saved] {REPORTS_DIR / 'feature_sensitivity.json'}")
    return out


if __name__ == "__main__":
    run()

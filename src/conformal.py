"""改善実装: 予測区間の較正（split conformal prediction）。

分位点回帰の区間は系統的に過小被覆（名目0.80 に対し実測0.74）だった。
検量(val)集合の残差分位点で区間幅を決める split-conformal により、
被覆率を名目に近づける。交換可能性は時系列で厳密には成り立たないが、
直近の検量窓を使う実務的手法として有効。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import data
import features
import models
from config import LOAD_COLS, REPORTS_DIR, TARGET


def _fit_eval(df, h, idx, nominal=0.8):
    sc = data.StandardScalerFrame().fit(
        df.iloc[idx.train[0]:idx.train[1]], [TARGET] + LOAD_COLS)
    dfs = sc.transform(df)
    X, y, dates = features.make_supervised(dfs, horizon=h, use_future_covariates=False)
    freq = df["date"].iloc[1] - df["date"].iloc[0]
    base = pd.to_datetime(dates) - h * freq
    tre = df["date"].iloc[idx.train[1] - 1]; vae = df["date"].iloc[idx.val[1] - 1]
    tee = df["date"].iloc[idx.test[1] - 1]
    trm = (base <= tre).to_numpy(); vam = ((base > tre) & (base <= vae)).to_numpy()
    tem = ((base > vae) & (base <= tee)).to_numpy()
    nowtr = X[trm][f"{TARGET}_now"].to_numpy()
    nowva = X[vam][f"{TARGET}_now"].to_numpy()
    nowte = X[tem][f"{TARGET}_now"].to_numpy()
    dtr = y[trm].to_numpy() - nowtr
    dva = y[vam].to_numpy() - nowva

    # 点予測（中央値）モデル
    point = models.LightGBMModel().fit(X[trm], dtr, eval_set=[(X[vam], dva)])
    yte_c = sc.inverse_transform_target(y[tem].to_numpy(), TARGET)

    # --- 方法1: 分位点回帰 P10/P90 ---
    q10 = models.LightGBMModel(objective="quantile", alpha=0.1).fit(X[trm], dtr, eval_set=[(X[vam], dva)])
    q90 = models.LightGBMModel(objective="quantile", alpha=0.9).fit(X[trm], dtr, eval_set=[(X[vam], dva)])
    lo_q = sc.inverse_transform_target(nowte + np.minimum(q10.predict(X[tem]), q90.predict(X[tem])), TARGET)
    hi_q = sc.inverse_transform_target(nowte + np.maximum(q10.predict(X[tem]), q90.predict(X[tem])), TARGET)
    cov_q = float(np.mean((yte_c >= lo_q) & (yte_c <= hi_q)))
    w_q = float(np.mean(hi_q - lo_q))

    # --- 方法2: split-conformal（val 残差の絶対値分位点で幅決定） ---
    val_pred_c = sc.inverse_transform_target(nowva + point.predict(X[vam]), TARGET)
    val_true_c = sc.inverse_transform_target(y[vam].to_numpy(), TARGET)
    resid = np.abs(val_true_c - val_pred_c)
    n = len(resid)
    k = int(np.ceil((n + 1) * nominal)) - 1
    k = min(max(k, 0), n - 1)
    qhat = np.sort(resid)[k]                       # conformal 半幅
    center_c = sc.inverse_transform_target(nowte + point.predict(X[tem]), TARGET)
    lo_c, hi_c = center_c - qhat, center_c + qhat
    cov_c = float(np.mean((yte_c >= lo_c) & (yte_c <= hi_c)))
    w_c = float(2 * qhat)
    return {"quantile": (cov_q, w_q), "conformal": (cov_c, w_c)}


def run(ds="ETTh2", horizons=(6, 12, 24)) -> dict:
    df = data.load_ett(ds)
    folds = data.rolling_origin_folds(len(df), 5)
    out = {"dataset": ds, "nominal": 0.8, "horizons": {}}
    for h in horizons:
        rq = np.array([_fit_eval(df, h, f) for f in folds], dtype=object)
        cq = np.mean([r["quantile"][0] for r in rq])
        wq = np.mean([r["quantile"][1] for r in rq])
        cc = np.mean([r["conformal"][0] for r in rq])
        wc = np.mean([r["conformal"][1] for r in rq])
        out["horizons"][str(h)] = {
            "quantile": {"coverage": round(float(cq), 3), "width_C": round(float(wq), 2)},
            "conformal": {"coverage": round(float(cc), 3), "width_C": round(float(wc), 2)},
        }
        print(f"[{ds} h={h:>2}] 分位点: 被覆{cq:.2f} 幅{wq:.1f}°C  →  "
              f"conformal: 被覆{cc:.2f} 幅{wc:.1f}°C (名目0.80)")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "conformal.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[saved] {REPORTS_DIR / 'conformal.json'}")
    return out


if __name__ == "__main__":
    run()

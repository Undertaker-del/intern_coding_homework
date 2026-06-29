"""モデル選択の検証: 「LightGBM が妥当か」を実証する。

全モデルを同一の差分予測（Δ=OT(t+h)-OT(t) を学習し OT(t) に加算）で統一し、
モデルクラスの差のみを 5 フォールド・バックテストで比較する。
- persistence（baseline）
- Ridge（engineered 特徴の線形）
- LightGBM（engineered 特徴の勾配ブースティング・本命）
- MLP（engineered 特徴のニューラルネット, scikit-learn）
- DLinear（系列分解＋線形, PyTorch。ETT で Transformer を上回ると報告された軽量モデル）
  出典: Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023。

強化学習(RL)は不採用：本タスクは「入力→将来値」の教師あり回帰であり、
逐次意思決定(報酬・遷移)の枠組みである RL は不適。RL が活きるのは予測を踏まえた
下流の制御（冷却/負荷制御の方策最適化）であり、本 PoC の範囲外。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import data
import features
import models
from config import LOAD_COLS, REPORTS_DIR, SEED, TARGET
from evaluate import mae, skill


# ---------- 共通: 時系列マスク ----------
def _masks(df, dates, h, idx):
    freq = df["date"].iloc[1] - df["date"].iloc[0]
    base = pd.to_datetime(dates) - h * freq
    tre = df["date"].iloc[idx.train[1] - 1]
    vae = df["date"].iloc[idx.val[1] - 1]
    tee = df["date"].iloc[idx.test[1] - 1]
    return (base <= tre).to_numpy(), \
        ((base > tre) & (base <= vae)).to_numpy(), \
        ((base > vae) & (base <= tee)).to_numpy()


# ---------- engineered 特徴で Δ を学習する汎用評価 ----------
def _eval_engineered(df, h, idx, est_name):
    sc = data.StandardScalerFrame().fit(
        df.iloc[idx.train[0]:idx.train[1]], [TARGET] + LOAD_COLS)
    dfs = sc.transform(df)
    X, y, dates = features.make_supervised(dfs, horizon=h, use_future_covariates=False)
    trm, vam, tem = _masks(df, dates, h, idx)
    now = X[tem][f"{TARGET}_now"].to_numpy()
    yte_c = sc.inverse_transform_target(y[tem].to_numpy(), TARGET)
    naive_c = sc.inverse_transform_target(now, TARGET)
    dtr = y[trm].to_numpy() - X[trm][f"{TARGET}_now"].to_numpy()
    dva = y[vam].to_numpy() - X[vam][f"{TARGET}_now"].to_numpy()
    if est_name == "ridge":
        m = models.RidgeModel(1.0).fit(X[trm], dtr)
        pred = m.predict(X[tem])
    elif est_name == "lgbm":
        m = models.LightGBMModel().fit(X[trm], dtr, eval_set=[(X[vam], dva)])
        pred = m.predict(X[tem])
    elif est_name == "mlp":
        xs = StandardScaler().fit(X[trm])
        m = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300,
                         early_stopping=True, random_state=SEED)
        m.fit(xs.transform(X[trm]), dtr)
        pred = m.predict(xs.transform(X[tem]))
    else:
        raise ValueError(est_name)
    pred_c = sc.inverse_transform_target(now + pred, TARGET)
    return skill(mae(yte_c, pred_c), mae(yte_c, naive_c))


# ---------- DLinear (PyTorch) ----------
class _DLinear(nn.Module):
    def __init__(self, lookback: int, kernel: int = 25):
        super().__init__()
        self.kernel = kernel
        self.trend = nn.Linear(lookback, 1)
        self.seasonal = nn.Linear(lookback, 1)

    def _decompose(self, x):
        pad = self.kernel // 2
        xp = torch.nn.functional.pad(x.unsqueeze(1), (pad, pad), mode="replicate")
        trend = torch.nn.functional.avg_pool1d(xp, self.kernel, stride=1).squeeze(1)
        trend = trend[:, :x.shape[1]]
        return trend, x - trend

    def forward(self, x):
        t, s = self._decompose(x)
        return (self.trend(t) + self.seasonal(s)).squeeze(-1)


def _eval_dlinear(df, h, idx, lookback=96, epochs=120):
    torch.manual_seed(SEED)
    sc = data.StandardScalerFrame().fit(
        df.iloc[idx.train[0]:idx.train[1]], [TARGET] + LOAD_COLS)
    ot = sc.transform(df)[TARGET].to_numpy()
    n = len(ot)
    W = sliding_window_view(ot, lookback)            # 行 j は ot[j:j+L]（末尾=時刻 j+L-1）
    end = np.arange(lookback - 1, n)                  # 各窓の末尾時刻 t
    valid = end + h < n
    end = end[valid]
    Wv = W[:len(end)]
    now = ot[end]
    target = ot[end + h] - now                        # Δ（標準化スケール）
    dates = df["date"].to_numpy()[end + h]
    trm, vam, tem = _masks(df, pd.Series(dates), h, idx)

    dev = "cpu"
    Xtr = torch.tensor(Wv[trm], dtype=torch.float32)
    ytr = torch.tensor(target[trm], dtype=torch.float32)
    Xva = torch.tensor(Wv[vam], dtype=torch.float32)
    yva = torch.tensor(target[vam], dtype=torch.float32)
    Xte = torch.tensor(Wv[tem], dtype=torch.float32)

    net = _DLinear(lookback).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
    lossf = nn.L1Loss()
    best, best_state, patience = 1e9, None, 0
    for ep in range(epochs):
        net.train(); opt.zero_grad()
        loss = lossf(net(Xtr), ytr); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(Xva), yva).item()
        if vl < best - 1e-5:
            best, best_state, patience = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 15:
                break
    if best_state:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net(Xte).numpy()
    pred_c = sc.inverse_transform_target(now[tem] + pred, TARGET)
    yte_c = sc.inverse_transform_target(now[tem] + target[tem], TARGET)
    naive_c = sc.inverse_transform_target(now[tem], TARGET)
    return skill(mae(yte_c, pred_c), mae(yte_c, naive_c))


def compare(datasets=("ETTh1", "ETTh2"), horizons=(6, 12)) -> dict:
    out = {}
    for ds in datasets:
        df = data.load_ett(ds)
        folds = data.rolling_origin_folds(len(df), 5)
        out[ds] = {}
        for h in horizons:
            row = {}
            for name in ("ridge", "lgbm", "mlp"):
                row[name] = round(float(np.mean(
                    [_eval_engineered(df, h, f, name) for f in folds])), 4)
            row["dlinear"] = round(float(np.mean(
                [_eval_dlinear(df, h, f) for f in folds])), 4)
            out[ds][f"h{h}"] = row
            print(f"[{ds} h={h}] " + " ".join(f"{k}={v:+.3f}" for k, v in row.items()))
    return out


if __name__ == "__main__":
    res = compare()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "model_compare.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[saved] {REPORTS_DIR / 'model_compare.json'}")

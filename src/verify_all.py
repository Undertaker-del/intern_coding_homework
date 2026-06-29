"""検証可能な範囲を網羅的に潰す統合検証。

出力: reports/verification_report.json と reports/figures/result_mechanism.png
- 機構: OT の変動性（日内振幅・std(Δh)）とモデル価値の関係。
- データ品質: 張り付き（連続同値）・外れ値。
- アブレーション: 特徴量グループ / Ridge vs LGBM / 差分 vs 直接予測。
すべて 5 フォールドのバックテストで評価（リーク無し）。
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
import features
import models
from config import LOAD_COLS, REPORTS_DIR, TARGET
from evaluate import mae, skill
from plotstyle import use_japanese_font


def _fold_skill(df, h, idx, *, drop_pred=None, direct=False, model="lgbm"):
    sc = data.StandardScalerFrame().fit(
        df.iloc[idx.train[0]:idx.train[1]], [TARGET] + LOAD_COLS)
    dfs = sc.transform(df)
    X, y, dates = features.make_supervised(dfs, horizon=h, use_future_covariates=False)
    freq = df["date"].iloc[1] - df["date"].iloc[0]
    base = pd.to_datetime(dates) - h * freq
    tre = df["date"].iloc[idx.train[1] - 1]
    vae = df["date"].iloc[idx.val[1] - 1]
    tee = df["date"].iloc[idx.test[1] - 1]
    trm = base <= tre; vam = (base > tre) & (base <= vae)
    tem = (base > vae) & (base <= tee)
    cols = [c for c in X.columns if not (drop_pred and drop_pred(c))]
    Xtr, Xva, Xte = X[trm][cols], X[vam][cols], X[tem][cols]
    now = X[tem][f"{TARGET}_now"].to_numpy()
    yte_c = sc.inverse_transform_target(y[tem].to_numpy(), TARGET)
    naive_c = sc.inverse_transform_target(now, TARGET)
    if direct:
        ytr, yva = y[trm].to_numpy(), y[vam].to_numpy()
        recon = lambda p: sc.inverse_transform_target(p, TARGET)
    else:
        ytr = y[trm].to_numpy() - X[trm][f"{TARGET}_now"].to_numpy()
        yva = y[vam].to_numpy() - X[vam][f"{TARGET}_now"].to_numpy()
        recon = lambda p: sc.inverse_transform_target(now + p, TARGET)
    if model == "ridge":
        m = models.RidgeModel(1.0).fit(Xtr, ytr)
    else:
        m = models.LightGBMModel().fit(Xtr, ytr, eval_set=[(Xva, yva)])
    return skill(mae(yte_c, recon(m.predict(Xte))), mae(yte_c, naive_c))


def _mean_skill(df, h, folds, **kw):
    return float(np.mean([_fold_skill(df, h, f, **kw) for f in folds]))


def mechanism(datasets=("ETTh1", "ETTh2")) -> dict:
    out = {}
    for ds in datasets:
        df = data.load_ett(ds); ot = df[TARGET].to_numpy()
        clim = df.groupby(df["date"].dt.hour)[TARGET].mean()
        out[ds] = {
            "std_OT": float(ot.std()),
            "diurnal_amplitude": float(clim.max() - clim.min()),
            "std_delta": {str(h): float((ot[h:] - ot[:-h]).std()) for h in (1, 6, 12, 24)},
        }
    return out


def data_quality(datasets=("ETTh1", "ETTh2", "ETTm1", "ETTm2")) -> dict:
    out = {}
    for ds in datasets:
        ot = data.load_ett(ds)[TARGET].to_numpy()
        runs = []; r = 1
        for i in range(1, len(ot)):
            if ot[i] == ot[i - 1]:
                r += 1
            else:
                if r >= 6:
                    runs.append(r)
                r = 1
        flat = sum(runs)
        d1 = np.diff(ot); sig = d1.std()
        out[ds] = {
            "flatline_pct": round(100 * flat / len(ot), 2),
            "max_flat_run": int(max(runs) if runs else 0),
            "outlier_pct_6sigma": round(100 * int(np.sum(np.abs(d1) > 6 * sig)) / len(d1), 3),
        }
    return out


def ablations() -> dict:
    groups = {
        "full": None,
        "no_calendar": lambda c: c in ("dow", "month") or c.startswith(("sin_", "cos_")),
        "no_targetlag": lambda c: c.startswith("OT_lag"),
        "no_rolling": lambda c: "_rmean" in c or "_rstd" in c,
        "no_load": lambda c: any(c.startswith(L) for L in LOAD_COLS),
    }
    df2 = data.load_ett("ETTh2"); f2 = data.rolling_origin_folds(len(df2), 5)
    feat = {name: round(_mean_skill(df2, 6, f2, drop_pred=p), 4)
            for name, p in groups.items()}
    model_cmp = {}
    for ds in ("ETTh1", "ETTh2"):
        df = data.load_ett(ds); folds = data.rolling_origin_folds(len(df), 5)
        model_cmp[ds] = {f"h{h}": {"ridge": round(_mean_skill(df, h, folds, model="ridge"), 4),
                                    "lgbm": round(_mean_skill(df, h, folds, model="lgbm"), 4)}
                         for h in (6, 12)}
    df1 = data.load_ett("ETTh1"); f1 = data.rolling_origin_folds(len(df1), 5)
    target_cmp = {f"h{h}": {"delta": round(_mean_skill(df1, h, f1, direct=False), 4),
                            "direct": round(_mean_skill(df1, h, f1, direct=True), 4)}
                  for h in (1, 6, 12)}
    return {"feature_ablation_ETTh2_h6": feat,
            "ridge_vs_lgbm": model_cmp,
            "delta_vs_direct_ETTh1": target_cmp}


def _figure(mech: dict, abl: dict) -> None:
    use_japanese_font()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    # 左: 日内振幅 vs 中期skill
    dss = list(mech)
    amp = [mech[d]["diurnal_amplitude"] for d in dss]
    bt = {d: json.load(open(REPORTS_DIR / f"backtest_{d}.json", encoding="utf-8")) for d in dss}
    sk = [100 * bt[d]["horizons"]["12"]["skill_mean"] for d in dss]
    ax = axes[0]; x = np.arange(len(dss))
    b1 = ax.bar(x - 0.2, amp, 0.4, label="日内振幅(°C)", color="#1f3a93")
    ax2 = ax.twinx()
    b2 = ax2.bar(x + 0.2, sk, 0.4, label="12h先skill(%)", color="#e67e22")
    ax.set_xticks(x); ax.set_xticklabels(dss)
    ax.set_ylabel("日内サイクル振幅 (°C)"); ax2.set_ylabel("12h先 改善率 (%)")
    ax.set_title("機構: 日内振幅が大きいほど ML 価値が高い")
    ax.legend(handles=[b1, b2], labels=["日内振幅(°C)", "12h先skill(%)"], fontsize=8, loc="upper left")
    # 右: 特徴量アブレーション
    feat = abl["feature_ablation_ETTh2_h6"]
    names = list(feat); vals = [100 * feat[n] for n in names]
    ax = axes[1]
    ax.bar(names, vals, color=["#16a085" if n == "full" else "#c0392b" for n in names])
    ax.set_title("特徴量アブレーション(ETTh2 6h先)")
    ax.set_ylabel("skill (%)"); ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    p = REPORTS_DIR / "figures" / "result_mechanism.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"[saved] {p}")


def run() -> dict:
    print("[verify] mechanism ..."); mech = mechanism()
    print("[verify] data quality ..."); dq = data_quality()
    print("[verify] ablations (時間がかかります) ..."); abl = ablations()
    report = {"mechanism": mech, "data_quality": dq, "ablations": abl}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "verification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[saved] {REPORTS_DIR / 'verification_report.json'}")
    _figure(mech, abl)
    # 要点を表示
    for ds in mech:
        print(f"  {ds}: 日内振幅={mech[ds]['diurnal_amplitude']:.2f}C flat={dq[ds]['flatline_pct']}%")
    print("  特徴量ablation(ETTh2h6):", abl["feature_ablation_ETTh2_h6"])
    return report


if __name__ == "__main__":
    run()

"""end-to-end 実行: 学習→評価→図表→metrics.json。

リーク防止の流れ:
  load → 時系列分割 → train で StandardScaler fit → 因果的特徴量 → 学習/評価。
CLI:
  py -3.12 src/pipeline.py --dataset ETTh1
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import data
import features
import models
from config import (FIGURES_DIR, HORIZONS_HOURLY, HORIZONS_MINUTE, LOAD_COLS,
                    REPORTS_DIR, SEASONAL_PERIOD_HOURLY, SEASONAL_PERIOD_MINUTE,
                    TARGET)
from evaluate import (interval_metrics, mae, mase, rmse, skill,
                      threshold_alarm_metrics)


def _horizons_and_period(dataset: str):
    if dataset.startswith("ETTm"):
        return HORIZONS_MINUTE, SEASONAL_PERIOD_MINUTE
    return HORIZONS_HOURLY, SEASONAL_PERIOD_HOURLY


def _seasonal_naive_pred(df, dates, te_mask, horizon, period, freq):
    """季節持続予測 ŷ(t+h)=OT(t+h-period) を test 行ぶん返す（°C, 不能行は NaN）。"""
    ot = df[TARGET].to_numpy()
    base_time = pd.to_datetime(dates) - horizon * freq
    pos = df[data.DATE_COL].searchsorted(base_time.values)
    seas_pos = pos + horizon - period
    valid = seas_pos >= 0
    full = np.full(len(base_time), np.nan)
    full[valid] = ot[seas_pos[valid]]
    return full[te_mask.to_numpy()]


def run_horizon(df: pd.DataFrame, horizon: int, period: int,
                idx: "data.SplitIndex | None" = None,
                with_future: bool = True,
                with_intervals: bool = True) -> dict:
    """単一ホライズンの学習・評価。元スケール(°C)で指標を返す。

    idx を渡すとその分割境界で評価する（バックテストのフォールド用）。
    train は系列先頭から tr_end までの拡張窓、test は (va_end, te_end] の窓。
    """
    if idx is None:
        idx = data.chronological_split(len(df))
    tr_raw, _, _ = data.slice_split(df, idx)

    scaler = data.StandardScalerFrame().fit(tr_raw, [TARGET] + LOAD_COLS)
    dfs = scaler.transform(df)

    tr_end = df[data.DATE_COL].iloc[idx.train[1] - 1]
    va_end = df[data.DATE_COL].iloc[idx.val[1] - 1]
    te_end = df[data.DATE_COL].iloc[idx.test[1] - 1]
    freq = df[data.DATE_COL].iloc[1] - df[data.DATE_COL].iloc[0]

    def time_split(dates):
        """目的時刻列から train/val/test の boolean マスクを返す（リーク無し）。"""
        base = pd.to_datetime(dates) - horizon * freq
        return (base <= tr_end,
                (base > tr_end) & (base <= va_end),
                (base > va_end) & (base <= te_end))

    def fit_delta_lgbm(X, y, dates, **model_kwargs):
        """Δ予測の LightGBM を学習し、test の °C 予測と指標を返す。

        model_kwargs で objective/alpha を渡せば分位点回帰になる。
        """
        tr_m, va_m, te_m = time_split(dates)
        Xtr, Xva, Xte = X[tr_m], X[va_m], X[te_m]
        now_tr = Xtr[f"{TARGET}_now"].to_numpy()
        now_va = Xva[f"{TARGET}_now"].to_numpy()
        now_te = Xte[f"{TARGET}_now"].to_numpy()
        dtr = y[tr_m].to_numpy() - now_tr
        dva = y[va_m].to_numpy() - now_va
        m = models.LightGBMModel(**model_kwargs).fit(Xtr, dtr, eval_set=[(Xva, dva)])
        pred_c = scaler.inverse_transform_target(now_te + m.predict(Xte), TARGET)
        return m, Xte, pred_c, te_m

    # 主特徴量（未来情報なし = 純粋に過去のみ）
    X, y, dates = features.make_supervised(dfs, horizon=horizon,
                                           use_future_covariates=False)
    tr_mask, va_mask, te_mask = time_split(dates)
    Xtr, ytr = X[tr_mask], y[tr_mask]
    Xva, yva = X[va_mask], y[va_mask]
    Xte, yte = X[te_mask], y[te_mask]
    yte_c = scaler.inverse_transform_target(yte.to_numpy(), TARGET)

    results = {}

    # --- ベースライン（元系列から直接計算、公平比較） ---
    # naive persistence: ŷ(t+h)=OT(t) は標準化 OT_now を逆変換
    persist_pred_c = scaler.inverse_transform_target(
        Xte[f"{TARGET}_now"].to_numpy(), TARGET)
    results["naive_persistence"] = {
        "mae": mae(yte_c, persist_pred_c), "rmse": rmse(yte_c, persist_pred_c)}

    # seasonal naive: ŷ(t+h)=OT(t+h-period)
    seas_te = _seasonal_naive_pred(df, dates, te_mask, horizon, period, freq)
    m_seas = ~np.isnan(seas_te)
    results["seasonal_naive"] = {
        "mae": mae(yte_c[m_seas], seas_te[m_seas]),
        "rmse": rmse(yte_c[m_seas], seas_te[m_seas])}

    mae_naive = results["naive_persistence"]["mae"]

    # --- 差分予測（persistence-anchored）: Δ=OT(t+h)-OT(t) を学習し OT(t) に加算 ---
    # 高自己相関系列の定石。Δ̂=0 なら最悪でも naive と同等になり、
    # 日次サイクル等の系統的変化のみを上乗せして改善する。
    now_te = Xte[f"{TARGET}_now"].to_numpy()
    dtr = ytr.to_numpy() - Xtr[f"{TARGET}_now"].to_numpy()

    # --- Ridge（線形上限の目安） ---
    ridge = models.RidgeModel(alpha=1.0).fit(Xtr, dtr)
    ridge_pred_c = scaler.inverse_transform_target(now_te + ridge.predict(Xte), TARGET)
    results["ridge"] = {"mae": mae(yte_c, ridge_pred_c),
                        "rmse": rmse(yte_c, ridge_pred_c),
                        "skill_vs_naive": skill(mae(yte_c, ridge_pred_c), mae_naive)}

    # --- LightGBM（過去のみ・本命） ---
    ot_train = df[TARGET].iloc[idx.train[0]:idx.train[1]].to_numpy()  # MASE 分母用
    lgbm, _, lgbm_pred_c, _ = fit_delta_lgbm(X, y, dates)
    results["lightgbm"] = {
        "mae": mae(yte_c, lgbm_pred_c), "rmse": rmse(yte_c, lgbm_pred_c),
        "mase": mase(yte_c, lgbm_pred_c, ot_train, m=1),
        "skill_vs_naive": skill(mae(yte_c, lgbm_pred_c), mae_naive),
        "top_features": dict(list(lgbm.feature_importance(list(X.columns)).items())[:12]),
    }
    results["naive_persistence"]["mase"] = mase(yte_c, persist_pred_c, ot_train, m=1)

    # --- 上限実験: 未来負荷が既知（運用スケジュール）なら 24h 先がどこまで改善するか ---
    #   PDF注記「t=T の特徴量を使ってよい」を活用。負荷が計画値で既知という仮定。
    if with_future and horizon >= 2:
        Xf, yf, datesf = features.make_supervised(dfs, horizon=horizon,
                                                  use_future_covariates=True)
        _, _, lgbm_fut_c, te_mf = fit_delta_lgbm(Xf, yf, datesf)
        yte_cf = scaler.inverse_transform_target(yf[te_mf].to_numpy(), TARGET)
        results["lightgbm_known_load"] = {
            "mae": mae(yte_cf, lgbm_fut_c), "rmse": rmse(yte_cf, lgbm_fut_c),
            "skill_vs_naive": skill(mae(yte_cf, lgbm_fut_c), mae_naive),
            "note": "未来負荷を既知と仮定（運用スケジュール前提）。価値レバーの上限推定。",
        }

    # --- 予測区間: 分位点回帰 P10/P90（名目 80% 区間） ---
    #   点予測に加え「リスク幅」を提示し、最悪ケース想定の保全判断を支援。
    lo_band = hi_band = None
    if with_intervals:
        _, _, q10_c, _ = fit_delta_lgbm(X, y, dates, objective="quantile", alpha=0.1)
        _, _, q90_c, _ = fit_delta_lgbm(X, y, dates, objective="quantile", alpha=0.9)
        lo_band = np.minimum(q10_c, q90_c)   # 分位交差をガード
        hi_band = np.maximum(q10_c, q90_c)
        results["lightgbm"]["interval"] = interval_metrics(
            yte_c, lo_band, hi_band, nominal=0.8)

    # 業務価値: テスト期間の上位 10%（高温時間帯）を h 先に検知できるか。
    #   ※train基準閾値だと分布シフトで test に該当事象が無くなるため、
    #     運用上の「相対的に高温な時間帯」を test 分位で定義して検知力を測る。
    thr = float(np.percentile(yte_c, 90))
    results["lightgbm"]["alarm"] = threshold_alarm_metrics(yte_c, lgbm_pred_c, thr)
    results["naive_persistence"]["alarm"] = threshold_alarm_metrics(yte_c, persist_pred_c, thr)

    results["_meta"] = {
        "horizon": horizon, "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)), "n_test": int(len(Xte)),
        "test_start": str(dates[te_mask].iloc[0]),
        "test_end": str(dates[te_mask].iloc[-1]),
    }
    # 図表用に予測を保持
    results["_arrays"] = {
        "dates": pd.to_datetime(dates[te_mask]).to_numpy(),
        "y_true": yte_c, "lgbm": lgbm_pred_c, "naive": persist_pred_c,
        "lo": lo_band, "hi": hi_band,
    }
    return results


def run(dataset: str = "ETTh1", make_figures: bool = True) -> dict:
    df = data.load_ett(dataset)
    horizons, period = _horizons_and_period(dataset)
    out = {"dataset": dataset, "horizons": {}}
    for h in horizons:
        res = run_horizon(df, h, period)
        arrays = res.pop("_arrays")
        out["horizons"][str(h)] = res
        if make_figures:
            _plot_predictions(dataset, h, arrays)
    _save_metrics(dataset, out)
    return out


def _save_metrics(dataset: str, out: dict):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"metrics_{dataset}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"[saved] {path}")


def _plot_predictions(dataset: str, horizon: int, arrays: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plotstyle import use_japanese_font
    use_japanese_font()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    n = min(500, len(arrays["y_true"]))
    fig, ax = plt.subplots(figsize=(11, 3.6))
    if arrays.get("lo") is not None:
        ax.fill_between(arrays["dates"][:n], arrays["lo"][:n], arrays["hi"][:n],
                        color="#e67e22", alpha=0.20, label="P10–P90 予測区間")
    ax.plot(arrays["dates"][:n], arrays["y_true"][:n], label="実測 OT", color="#1f3a93", lw=1.4)
    ax.plot(arrays["dates"][:n], arrays["lgbm"][:n], label="LightGBM 予測", color="#e67e22", lw=1.2)
    ax.plot(arrays["dates"][:n], arrays["naive"][:n], label="naive 持続", color="#aaaaaa", lw=0.9, ls="--")
    ax.set_title(f"{dataset}  OT {horizon}ステップ先予測（test 先頭{n}点）")
    ax.set_ylabel("OT (°C)"); ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    p = FIGURES_DIR / f"pred_{dataset}_h{horizon}.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"[saved] {p}")


def _print_summary(out: dict):
    print(f"\n=== {out['dataset']} 結果サマリ（test, °C） ===")
    for h, r in out["horizons"].items():
        print(f"-- horizon {h} --")
        for m in ["naive_persistence", "seasonal_naive", "ridge",
                  "lightgbm", "lightgbm_known_load"]:
            if m not in r:
                continue
            row = r[m]
            sk = f"  skill={row.get('skill_vs_naive', 0):+.3f}" if "skill_vs_naive" in row else ""
            print(f"   {m:20s} MAE={row['mae']:.3f}  RMSE={row['rmse']:.3f}{sk}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ETTh1")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    result = run(args.dataset, make_figures=not args.no_figures)
    _print_summary(result)

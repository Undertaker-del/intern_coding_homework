"""ローリング起点バックテスト: 単一分割の期間依存を排し、再現性/頑健性を担保。

各フォールドで scaler とモデルを再学習（リーク無し）し、ホライズン別に
LightGBM/naive の MAE と skill を mean±std で集計する。
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import data
import pipeline
from config import REPORTS_DIR
from pipeline import _horizons_and_period


def backtest(dataset: str = "ETTh1", n_folds: int = 5) -> dict:
    df = data.load_ett(dataset)
    horizons, period = _horizons_and_period(dataset)
    folds = data.rolling_origin_folds(len(df), n_folds=n_folds)

    out: dict = {"dataset": dataset, "n_folds": len(folds), "horizons": {}}
    for h in horizons:
        rows = []
        for k, idx in enumerate(folds):
            res = pipeline.run_horizon(df, h, period, idx=idx,
                                       with_future=False, with_intervals=False)
            rows.append({
                "fold": k,
                "lgbm_mae": res["lightgbm"]["mae"],
                "naive_mae": res["naive_persistence"]["mae"],
                "skill": res["lightgbm"]["skill_vs_naive"],
                "test_start": res["_meta"]["test_start"],
            })
        lgbm = np.array([r["lgbm_mae"] for r in rows])
        naive = np.array([r["naive_mae"] for r in rows])
        sk = np.array([r["skill"] for r in rows])
        out["horizons"][str(h)] = {
            "lgbm_mae_mean": float(lgbm.mean()), "lgbm_mae_std": float(lgbm.std()),
            "naive_mae_mean": float(naive.mean()), "naive_mae_std": float(naive.std()),
            "skill_mean": float(sk.mean()), "skill_std": float(sk.std()),
            "skill_min": float(sk.min()), "skill_max": float(sk.max()),
            "n_positive": int((sk > 0).sum()), "n_total": len(sk),
            "folds": rows,
        }
        print(f"[{dataset}] h={h:>2}: skill {sk.mean():+.3f}±{sk.std():.3f} "
              f"(min{sk.min():+.3f}/max{sk.max():+.3f}, {int((sk>0).sum())}/{len(sk)} folds>0) "
              f"| LGBM MAE {lgbm.mean():.3f} vs naive {naive.mean():.3f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"backtest_{dataset}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[saved] {path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ETTh1")
    ap.add_argument("--folds", type=int, default=5)
    backtest(ap.parse_args().dataset, ap.parse_args().folds)

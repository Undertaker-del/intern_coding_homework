"""EDA: 油温の素性と予測アプローチの根拠を図表化する。

生成物（reports/figures/ と reports/eda_summary.json）:
- OT 時系列（季節性・非定常性）
- 自己相関（熱慣性）
- 時刻別・月別の平均 OT（日次/年次周期）
- OT と負荷の同時刻相関
- 学習/検証/テスト各期間の OT 分布（分布シフト）
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import data
from config import FIGURES_DIR, LOAD_COLS, REPORTS_DIR, TARGET
from plotstyle import use_japanese_font


def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    x = x - x.mean()
    var = np.dot(x, x)
    return np.array([1.0] + [np.dot(x[:-k], x[k:]) / var for k in range(1, nlags + 1)])


def run_eda(dataset: str = "ETTh1") -> dict:
    use_japanese_font()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = data.load_ett(dataset)
    ot = df[TARGET].to_numpy()
    summary: dict = {"dataset": dataset, "n_rows": int(len(df))}

    # 1) 時系列
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(df["date"], ot, lw=0.5, color="#1f3a93")
    ax.set_title(f"{dataset}: 油温 OT の時系列（約2年・日次/年次の周期と緩やかな非定常）")
    ax.set_ylabel("OT (°C)")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / f"eda_{dataset}_series.png", dpi=130); plt.close(fig)

    # 2) 自己相関（熱慣性）
    acf = _acf(ot, 72)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(range(len(acf)), acf, color="#1f3a93")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"{dataset}: OT 自己相関（1h後={acf[1]:.3f}, 24h後={acf[24]:.3f}）")
    ax.set_xlabel("ラグ (時間)"); ax.set_ylabel("自己相関")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / f"eda_{dataset}_acf.png", dpi=130); plt.close(fig)
    summary["acf_lag1"] = float(acf[1]); summary["acf_lag24"] = float(acf[24])

    # 3) 時刻別・月別の平均
    hour = df["date"].dt.hour
    month = df["date"].dt.month
    by_hour = df.groupby(hour)[TARGET].mean()
    by_month = df.groupby(month)[TARGET].mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    axes[0].plot(by_hour.index, by_hour.values, marker="o", color="#e67e22")
    axes[0].set_title("時刻別 平均 OT（日次周期）"); axes[0].set_xlabel("時刻")
    axes[1].plot(by_month.index, by_month.values, marker="o", color="#16a085")
    axes[1].set_title("月別 平均 OT（年次周期=外気温）"); axes[1].set_xlabel("月")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / f"eda_{dataset}_cycles.png", dpi=130); plt.close(fig)

    # 4) 同時刻相関
    corr = df[[TARGET] + LOAD_COLS].corr()[TARGET].drop(TARGET)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(corr.index, corr.values, color="#8e44ad")
    ax.set_title(f"{dataset}: OT と各負荷の同時刻相関（最大={corr.abs().max():.2f}＝弱い）")
    ax.set_ylabel("相関係数"); ax.axhline(0, color="k", lw=0.6)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / f"eda_{dataset}_corr.png", dpi=130); plt.close(fig)
    summary["max_abs_load_corr"] = float(corr.abs().max())

    # 5) 分布シフト（train/val/test）
    idx = data.chronological_split(len(df))
    tr, va, te = data.slice_split(df, idx)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for part, name, c in [(tr, "train", "#1f3a93"), (va, "val", "#e67e22"), (te, "test", "#c0392b")]:
        ax.hist(part[TARGET], bins=40, alpha=0.5, density=True, label=name, color=c)
    ax.set_title(f"{dataset}: 期間別 OT 分布（test 平均={te[TARGET].mean():.1f}°C < train={tr[TARGET].mean():.1f}°C）")
    ax.set_xlabel("OT (°C)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES_DIR / f"eda_{dataset}_shift.png", dpi=130); plt.close(fig)
    summary["mean_OT_train"] = float(tr[TARGET].mean())
    summary["mean_OT_test"] = float(te[TARGET].mean())

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / f"eda_{dataset}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[eda] {dataset}: acf1={summary['acf_lag1']:.3f} acf24={summary['acf_lag24']:.3f} "
          f"maxcorr={summary['max_abs_load_corr']:.3f} "
          f"meanOT train={summary['mean_OT_train']:.1f} test={summary['mean_OT_test']:.1f}")
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ETTh1")
    run_eda(ap.parse_args().dataset)

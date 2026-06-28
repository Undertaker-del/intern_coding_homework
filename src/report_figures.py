"""metrics_*.json から結果サマリ図（MAE/skill のホライズン依存）を生成。"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FIGURES_DIR, REPORTS_DIR
from plotstyle import use_japanese_font


def _load(dataset: str) -> dict:
    with open(REPORTS_DIR / f"metrics_{dataset}.json", encoding="utf-8") as f:
        return json.load(f)


def make_horizon_figures(datasets=("ETTh1", "ETTh2")) -> None:
    use_japanese_font()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(datasets), figsize=(11, 3.6))
    for ax, ds in zip(axes, datasets):
        m = _load(ds)
        hs = sorted(int(h) for h in m["horizons"])
        naive = [m["horizons"][str(h)]["naive_persistence"]["mae"] for h in hs]
        lgbm = [m["horizons"][str(h)]["lightgbm"]["mae"] for h in hs]
        ax.plot(hs, naive, marker="s", label="naive 持続", color="#aaaaaa")
        ax.plot(hs, lgbm, marker="o", label="LightGBM", color="#e67e22")
        ax.set_title(f"{ds}: 予測誤差のホライズン依存")
        ax.set_xlabel("予測ホライズン (時間先)"); ax.set_ylabel("test MAE (°C)")
        ax.legend()
    fig.tight_layout()
    p = FIGURES_DIR / "result_mae_vs_horizon.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"[saved] {p}")

    # skill 図
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = {"ETTh1": "#1f3a93", "ETTh2": "#c0392b"}
    for ds in datasets:
        m = _load(ds)
        hs = sorted(int(h) for h in m["horizons"])
        sk = [100 * m["horizons"][str(h)]["lightgbm"]["skill_vs_naive"] for h in hs]
        ax.plot(hs, sk, marker="o", label=ds, color=colors.get(ds))
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("LightGBM の naive 比改善率（skill）")
    ax.set_xlabel("予測ホライズン (時間先)"); ax.set_ylabel("MAE 改善率 (%)")
    ax.legend()
    fig.tight_layout()
    p = FIGURES_DIR / "result_skill_vs_horizon.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"[saved] {p}")


def make_backtest_figure(datasets=("ETTh1", "ETTh2")) -> None:
    """ローリング起点バックテストの skill（mean±std）を棒＋誤差棒で図示。"""
    use_japanese_font()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 3.8))
    colors = {"ETTh1": "#1f3a93", "ETTh2": "#c0392b"}
    width = 0.36
    for j, ds in enumerate(datasets):
        with open(REPORTS_DIR / f"backtest_{ds}.json", encoding="utf-8") as f:
            m = json.load(f)
        hs = sorted(int(h) for h in m["horizons"])
        means = [100 * m["horizons"][str(h)]["skill_mean"] for h in hs]
        stds = [100 * m["horizons"][str(h)]["skill_std"] for h in hs]
        x = np.arange(len(hs)) + (j - 0.5) * width
        ax.bar(x, means, width, yerr=stds, capsize=4, label=ds,
               color=colors.get(ds), alpha=0.85)
        ax.set_xticks(np.arange(len(hs)))
        ax.set_xticklabels([f"{h}h" for h in hs])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"ローリング起点バックテスト({m['n_folds']}フォールド)：naive比改善率")
    ax.set_xlabel("予測ホライズン"); ax.set_ylabel("MAE 改善率 mean±std (%)")
    ax.legend()
    fig.tight_layout()
    p = FIGURES_DIR / "result_backtest_skill.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"[saved] {p}")


if __name__ == "__main__":
    make_horizon_figures()
    make_backtest_figure()

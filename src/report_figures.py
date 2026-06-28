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


if __name__ == "__main__":
    make_horizon_figures()

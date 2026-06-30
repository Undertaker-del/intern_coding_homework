"""追加検証（モデル比較・特徴量感度・コンフォーマル較正）の図を生成。

reports/model_compare.json, feature_sensitivity.json, conformal.json から
3 枚を出力し、スライドへ base64 埋め込みする。
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import FIGURES_DIR, REPORTS_DIR
from plotstyle import use_japanese_font


def _load(name: str) -> dict:
    with open(REPORTS_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def value_heatmap_fig() -> None:
    """データ×ホライズンの skill ヒートマップ（バックテスト平均）。エグゼクティブサマリ用。"""
    import json as _json
    bt = {ds: _json.load(open(REPORTS_DIR / f"backtest_{ds}.json", encoding="utf-8"))
          for ds in ("ETTh1", "ETTh2")}
    hs = [1, 6, 12, 24]
    rows = ["ETTh2\n(変動大)", "ETTh1\n(穏やか)"]
    order = ["ETTh2", "ETTh1"]
    M = np.array([[bt[ds]["horizons"][str(h)]["skill_mean"] for h in hs] for ds in order])
    fig, ax = plt.subplots(figsize=(8.6, 2.9))
    im = ax.imshow(M, cmap="RdYlGn", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(hs))); ax.set_xticklabels([f"{h}h先" for h in hs], fontsize=12)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=12)
    for i, ds in enumerate(order):
        for j, h in enumerate(hs):
            r = bt[ds]["horizons"][str(h)]
            txt = f"{100*r['skill_mean']:+.0f}%\n±{100*r['skill_std']:.0f} ({r['n_positive']}/{r['n_total']})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=11,
                    color="#16240f" if abs(M[i, j]) < 0.45 else "#fff", fontweight="bold")
    ax.set_title("naive持続予測に対する改善率（5フォールド・バックテスト平均）", fontsize=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("改善率", fontsize=10)
    fig.tight_layout()
    p = FIGURES_DIR / "result_value_heatmap.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"[saved] {p}")


def model_compare_fig() -> None:
    mc = _load("model_compare.json")
    cols = {"ridge": "#95a5a6", "lgbm": "#e67e22", "mlp": "#9b59b6", "dlinear": "#2f5cff"}
    cases = [("ETTh1", "h6"), ("ETTh1", "h12"), ("ETTh2", "h6"), ("ETTh2", "h12")]
    labels = [f"{d}\n{h}" for d, h in cases]
    models = ["ridge", "lgbm", "mlp", "dlinear"]
    fig, ax = plt.subplots(figsize=(8.4, 3.7))
    x = np.arange(len(cases))
    w = 0.2
    for i, mdl in enumerate(models):
        vals = [mc[d][h][mdl] for d, h in cases]
        ax.bar(x + (i - 1.5) * w, vals, w, label=mdl, color=cols[mdl])
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("naive比 改善率 skill")
    ax.set_title("モデル比較（全て差分予測・5フォールド平均）")
    ax.legend(ncol=4, fontsize=9, loc="lower center")
    fig.tight_layout()
    p = FIGURES_DIR / "result_model_compare.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"[saved] {p}")


def feature_sensitivity_fig() -> None:
    fs = _load("feature_sensitivity.json")
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    styles = {"ETTh1": ("#16a085", "o"), "ETTh2": ("#e67e22", "s")}
    for ds, (c, mk) in styles.items():
        sets = fs[ds]
        nfeat = [sets[s]["n_features"] for s in ("minimal", "current", "expanded")]
        sk = [np.mean([sets[s]["h6"], sets[s]["h12"]]) for s in ("minimal", "current", "expanded")]
        ax.plot(nfeat, sk, marker=mk, color=c, label=ds)
        for n, s, name in zip(nfeat, sk, ("最小", "現行", "拡張")):
            ax.annotate(name, (n, s), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=9, color=c)
    ax.axvline(40, color="#999", ls="--", lw=0.8)
    ax.set_xlabel("特徴量数"); ax.set_ylabel("naive比 改善率 skill（6/12h平均）")
    ax.set_title("特徴量数の感度：多ければ良いとは限らない")
    ax.legend()
    fig.tight_layout()
    p = FIGURES_DIR / "result_feature_sensitivity.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"[saved] {p}")


def conformal_fig() -> None:
    cf = _load("conformal.json")
    hs = sorted(int(h) for h in cf["horizons"])
    qc = [cf["horizons"][str(h)]["quantile"]["coverage"] for h in hs]
    cc = [cf["horizons"][str(h)]["conformal"]["coverage"] for h in hs]
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    x = np.arange(len(hs)); w = 0.35
    ax.bar(x - w / 2, qc, w, label="分位点回帰（旧）", color="#bdc3c7")
    ax.bar(x + w / 2, cc, w, label="split-conformal（改善）", color="#2f5cff")
    ax.axhline(cf["nominal"], color="#c0392b", ls="--", lw=1.2, label="名目 0.80")
    ax.set_ylim(0.6, 0.9)
    ax.set_xticks(x); ax.set_xticklabels([f"{h}h" for h in hs])
    ax.set_ylabel("実測被覆率"); ax.set_title("予測区間の較正（ETTh2・5フォールド平均）")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    p = FIGURES_DIR / "result_conformal.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"[saved] {p}")


def make_all() -> None:
    use_japanese_font()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    value_heatmap_fig()
    model_compare_fig()
    feature_sensitivity_fig()
    conformal_fig()


if __name__ == "__main__":
    make_all()

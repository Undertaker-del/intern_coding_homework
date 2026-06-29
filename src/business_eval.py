"""業務用途に即した検証（誰が・何のために使うか）。

用途① 運用オペレーター: 高油温の「事前警報」。閾値超過を h 時間前に
  検知できるかを Precision/Recall/誤報頻度で評価し、持続予測と比較。
用途② 信頼性技師: 「異常・劣化の早期検知」。合成故障（温度ドリフト）を
  注入し、自己回帰予測 vs ソフトセンサ(負荷→温度)の残差で検知性を比較。
  （実故障ラベルが無いため合成故障で検証する。）
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import data
import models
import pipeline
from config import LOAD_COLS, REPORTS_DIR, TARGET
from plotstyle import use_japanese_font

WEEK = 168  # 1週間の時間数


def early_warning(ds: str = "ETTh2", horizons=(6, 12, 24)) -> dict:
    """高油温(訓練P90)の h 時間前検知。モデル vs 持続予測。"""
    df = data.load_ett(ds)
    idx = data.chronological_split(len(df))
    thr = float(np.percentile(df[TARGET].iloc[:idx.train[1]], 90))
    test_weeks = (idx.test[1] - idx.test[0]) / WEEK
    out = {"threshold_C": round(thr, 2), "test_weeks": round(test_weeks, 1), "horizons": {}}
    for h in horizons:
        r = pipeline.run_horizon(df, h, 24, idx=idx, with_future=False, with_intervals=False)
        a = r["_arrays"]; yt, yp, yn = a["y_true"], a["lgbm"], a["naive"]
        act = yt >= thr

        def metrics(alarm):
            tp = int(np.sum(act & alarm)); fp = int(np.sum(~act & alarm))
            fn = int(np.sum(act & ~alarm))
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            return {"precision": round(prec, 3), "recall": round(rec, 3),
                    "false_alarms_per_week": round(fp / test_weeks, 2)}

        ap = float(average_precision_score(act, yp)) if act.sum() else float("nan")
        out["horizons"][str(h)] = {
            "n_events": int(act.sum()),
            "model": {**metrics(yp >= thr), "auc_pr": round(ap, 3)},
            "persistence": metrics(yn >= thr),
        }
    return out


def _cal(ts):
    h = ts.dt.hour; d = ts.dt.dayofyear
    return pd.DataFrame({"sd": np.sin(2 * np.pi * h / 24), "cd": np.cos(2 * np.pi * h / 24),
                         "sy": np.sin(2 * np.pi * d / 365.25), "cy": np.cos(2 * np.pi * d / 365.25),
                         "dow": ts.dt.dayofweek}, index=ts.index)


def _feat(df, kind):
    f = _cal(df["date"])
    if kind == "soft":      # OT ~ 負荷（自己回帰なし）= ソフトセンサ
        for c in LOAD_COLS:
            f[c] = df[c]
    else:                   # OT ~ 自己ラグ（負荷なし）= 自己回帰
        for L in (1, 2, 3, 6, 12, 24):
            f[f"OT_l{L}"] = df[TARGET].shift(L)
    y = df[TARGET]; ok = f.notna().all(axis=1)
    return f[ok], y[ok], np.where(ok.values)[0]


def fault_injection(ds: str = "ETTh2", fault_C: float = 6.0, days: int = 14) -> dict:
    """合成故障(温度ドリフト)を注入し、残差にどれだけ現れるか(検知性)を比較。"""
    df = data.load_ett(ds); n = len(df); ntr = int(n * 0.6)
    t0 = ntr + (n - ntr) // 3
    ramp = np.array([min(fault_C, max(0, i - t0) * fault_C / (days * 24)) for i in range(n)])
    res = {"fault_C": fault_C, "ramp_days": days, "models": {}, "_series": {}}
    for kind, label in (("ar", "autoregressive"), ("soft", "soft_sensor")):
        Xc, yc, ic = _feat(df, kind); trm = ic < ntr
        mu = Xc[trm].mean(); sd = Xc[trm].std().replace(0, 1)
        m = models.LightGBMModel().fit((Xc[trm] - mu) / sd, yc[trm].values)
        dff = df.copy(); dff[TARGET] = df[TARGET].values + ramp
        Xf, yf, iff = _feat(dff, kind)
        resid = yf.values - m.predict((Xf - mu) / sd)
        pre = (iff >= ntr) & (iff < t0)
        sat = iff >= t0 + days * 24
        rise = float(np.nanmean(resid[sat]) - np.nanmean(resid[pre]))
        noise = float(np.nanstd(resid[pre]))
        res["models"][label] = {
            "fault_visible_in_residual_C": round(rise, 3),
            "fault_visible_pct": round(100 * rise / fault_C, 1),
            "pre_fault_noise_sigma_C": round(noise, 3),
            "signal_to_noise": round(rise / noise, 2) if noise else None,
        }
        res["_series"][label] = (iff, resid)
    res["_ramp"] = ramp; res["_t0"] = t0
    return res


def _figure(ew: dict, fi: dict, ds: str) -> None:
    use_japanese_font()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    # 左: 早期警報 recall by horizon
    hs = sorted(int(h) for h in ew["horizons"])
    mr = [ew["horizons"][str(h)]["model"]["recall"] for h in hs]
    pr = [ew["horizons"][str(h)]["persistence"]["recall"] for h in hs]
    x = np.arange(len(hs))
    axes[0].bar(x - 0.2, mr, 0.4, label="モデル", color="#1f3a93")
    axes[0].bar(x + 0.2, pr, 0.4, label="持続予測", color="#aaaaaa")
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"{h}h前" for h in hs])
    axes[0].set_title(f"用途①早期警報: 高温事象の検知率({ds})")
    axes[0].set_ylabel("再現率(recall)"); axes[0].legend(fontsize=8)
    # 右: 故障注入の残差
    t0 = fi["_t0"]; ramp = fi["_ramp"]
    for label, color in (("autoregressive", "#c0392b"), ("soft_sensor", "#16a085")):
        idxnum, resid = fi["_series"][label]
        post = idxnum >= t0 - 200
        hours = (idxnum[post] - t0)
        s = pd.Series(resid[post]).rolling(24, min_periods=1).mean().values
        axes[1].plot(hours, s, label=f"{label} 残差", color=color, lw=1.2)
    axes[1].plot((np.arange(len(ramp)) - t0)[t0 - 200:], ramp[t0 - 200:],
                 "k--", lw=1, label="注入故障(+6°C)")
    axes[1].axvline(0, color="gray", lw=0.8)
    axes[1].set_xlim(-200, fi["ramp_days"] * 24 + 200)
    axes[1].set_title(f"用途②異常検知: 合成故障に対する残差({ds})")
    axes[1].set_xlabel("発症からの時間(h)"); axes[1].set_ylabel("残差(°C)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    p = REPORTS_DIR / "figures" / "result_business.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"[saved] {p}")


def run(ds: str = "ETTh2") -> dict:
    print("[business] 用途① 早期警報 ...")
    ew = early_warning(ds)
    print("[business] 用途② 異常検知(合成故障注入) ...")
    fi = fault_injection(ds)
    _figure(ew, fi, ds)
    report = {"dataset": ds, "early_warning": ew,
              "fault_detection": {"fault_C": fi["fault_C"], "ramp_days": fi["ramp_days"],
                                  "models": fi["models"]}}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "business_eval.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[saved] {REPORTS_DIR / 'business_eval.json'}")
    e6 = ew["horizons"]["6"]
    print(f"  早期警報6h前: モデル R={e6['model']['recall']} 誤報/週={e6['model']['false_alarms_per_week']}"
          f" vs 持続 R={e6['persistence']['recall']}")
    print(f"  故障検知性: AR={fi['models']['autoregressive']['fault_visible_pct']}%  "
          f"soft={fi['models']['soft_sensor']['fault_visible_pct']}%")
    return report


if __name__ == "__main__":
    run()

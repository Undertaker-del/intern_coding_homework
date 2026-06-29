"""分割方法への感度分析: 時系列 60/20/20 と 文献標準 Informer 12/4/4 で
skill が一致するかを検証する。

結論の頑健性（特に ETTh1 の中期改善が分割に依存しないか）を確認するための監査。
"""
from __future__ import annotations

import json

import data
import pipeline
from config import REPORTS_DIR
from pipeline import _horizons_and_period


def run(datasets=("ETTh1", "ETTh2")) -> dict:
    out: dict = {}
    for ds in datasets:
        df = data.load_ett(ds)
        horizons, period = _horizons_and_period(ds)
        splits = {
            "chrono_60_20_20": data.chronological_split(len(df)),
            "informer_12_4_4": data.informer_split(ds, len(df)),
        }
        out[ds] = {}
        for sname, idx in splits.items():
            row = {}
            for h in horizons:
                r = pipeline.run_horizon(df, h, period, idx=idx,
                                         with_future=False, with_intervals=False)
                row[str(h)] = round(r["lightgbm"]["skill_vs_naive"], 4)
            out[ds][sname] = row
            print(f"[{ds}] {sname}: " +
                  " ".join(f"h{h}={row[str(h)]:+.3f}" for h in horizons))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "split_sensitivity.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[saved] {path}")
    return out


if __name__ == "__main__":
    run()

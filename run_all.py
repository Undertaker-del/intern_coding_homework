"""ワンコマンド実行: テスト→EDA→学習評価→結果図→スライド生成→PDF。

  py -3.12 run_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = [sys.executable]


def step(desc: str, args: list[str]) -> None:
    print(f"\n=== {desc} ===")
    subprocess.run(PY + args, cwd=ROOT, check=True)


def main() -> None:
    step("テスト（リーク検証含む）", ["-m", "pytest", "tests", "-q"])
    for ds in ("ETTh1", "ETTh2"):
        step(f"EDA {ds}", ["src/eda.py", "--dataset", ds])
        step(f"学習・評価 {ds}", ["src/pipeline.py", "--dataset", ds])
        step(f"バックテスト {ds}", ["src/backtest.py", "--dataset", ds])
    step("分割感度分析", ["src/split_sensitivity.py"])
    step("結果サマリ図", ["src/report_figures.py"])
    step("スライド HTML 生成", ["slides/build_slides.py"])
    step("PDF レンダリング", ["slides/render.py"])
    print("\n[完了] reports/ と slides/PoC_report.pdf を確認してください。")


if __name__ == "__main__":
    main()

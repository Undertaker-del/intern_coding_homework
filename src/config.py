"""集中設定: 定数・特徴量定義・分割比・乱数シード。

本番コードにテスト用の分岐を入れない方針のため、ここに調整値を集約する。
"""
from __future__ import annotations

from pathlib import Path

# --- パス ---
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- 再現性 ---
SEED = 42

# --- 列定義 ---
DATE_COL = "date"
TARGET = "OT"
LOAD_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]

# --- 時系列分割（時刻順・シャッフル禁止） ---
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20  # 残り 0.20 が test

# --- 予測ホライズン（粒度ステップ数） ---
# ETTh は1時間粒度なので 1=1時間先, 24=1日先。ETTm(15分)は 4=1時間先, 96=1日先。
HORIZONS_HOURLY = [1, 6, 12, 24]
HORIZONS_MINUTE = [4, 24, 48, 96]

# --- 特徴量設計（すべて因果的に生成する） ---
TARGET_LAGS = [1, 2, 3, 6, 12, 24, 48, 168]
LOAD_LAGS = [1, 24]
ROLL_WINDOWS = [3, 6, 24]
# 負荷の移動平均（熱蓄積=負荷の時間積分の代理）。
# 反復2で [6,24,72] / [24] を検証したが、ETTh1 中期は改善する一方 ETTh2 は悪化し、
# クロスデータで頑健な改善が得られなかったため不採用（空）。実験再現は config 変更のみ。
LOAD_ROLL_WINDOWS: list[int] = []

# 季節持続ベースラインの周期（1日のステップ数）
SEASONAL_PERIOD_HOURLY = 24
SEASONAL_PERIOD_MINUTE = 96

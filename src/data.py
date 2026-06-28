"""データ読込・時系列分割・標準化。

リーケージ防止の要:
- 分割は必ず時刻順（シャッフル禁止）。
- 標準化の統計量は train のみで推定し、val/test は変換のみ。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import DATA_DIR, DATE_COL, TRAIN_FRAC, VAL_FRAC


def load_ett(name: str) -> pd.DataFrame:
    """ETT CSV を読み込み、時刻昇順・重複なしを保証して返す。"""
    path = DATA_DIR / f"{name}.csv"
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL).drop_duplicates(DATE_COL).reset_index(drop=True)
    return df


@dataclass(frozen=True)
class SplitIndex:
    """各分割の行範囲（半開区間 [start, end)）。"""

    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]


def chronological_split(n_rows: int,
                        train_frac: float = TRAIN_FRAC,
                        val_frac: float = VAL_FRAC) -> SplitIndex:
    """行数を時刻順に train/val/test へ分割する境界を返す。"""
    n_train = int(n_rows * train_frac)
    n_val = int(n_rows * val_frac)
    return SplitIndex(
        train=(0, n_train),
        val=(n_train, n_train + n_val),
        test=(n_train + n_val, n_rows),
    )


def rolling_origin_folds(n_rows: int,
                        n_folds: int = 5,
                        val_frac: float = 0.10,
                        test_frac: float = 0.10) -> list[SplitIndex]:
    """拡張窓のローリング起点バックテスト用フォールドを返す。

    各フォールドは train=系列先頭〜val_start（拡張）、val・test は固定幅の窓。
    test 窓を test_frac ずつ後方へずらし、最後のフォールドが系列末尾で終わる。
    学習量が不足するフォールド（train が test 窓より短い）は除外する。
    """
    test_size = max(1, int(n_rows * test_frac))
    val_size = max(1, int(n_rows * val_frac))
    folds: list[SplitIndex] = []
    for i in range(n_folds):
        test_end = n_rows - (n_folds - 1 - i) * test_size
        test_start = test_end - test_size
        val_end = test_start
        val_start = val_end - val_size
        if val_start <= test_size:  # 学習量が不足するフォールドは捨てる
            continue
        folds.append(SplitIndex(train=(0, val_start),
                                val=(val_start, val_end),
                                test=(test_start, test_end)))
    return folds


def slice_split(df: pd.DataFrame, idx: SplitIndex):
    """境界に従い 3 つの DataFrame を返す（重複なし・時刻順）。"""
    tr = df.iloc[idx.train[0]:idx.train[1]]
    va = df.iloc[idx.val[0]:idx.val[1]]
    te = df.iloc[idx.test[0]:idx.test[1]]
    return tr, va, te


class StandardScalerFrame:
    """train のみで fit する列単位標準化器（リーク防止のため自前実装）。"""

    def __init__(self):
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None
        self.columns_: list[str] | None = None

    def fit(self, df: pd.DataFrame, columns: list[str]) -> "StandardScalerFrame":
        self.columns_ = list(columns)
        self.mean_ = df[columns].mean()
        std = df[columns].std(ddof=0)
        # 定数列の 0 除算回避
        self.std_ = std.replace(0.0, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.columns_] = (df[self.columns_] - self.mean_) / self.std_
        return out

    def inverse_transform_target(self, values: np.ndarray, target: str) -> np.ndarray:
        """標準化された目的変数を元スケール(°C)へ戻す。"""
        return values * float(self.std_[target]) + float(self.mean_[target])

"""因果的（リーク安全）な教師あり行列の生成。

設計上の不変条件:
- 基準時刻を t、目的変数を OT(t+h) とする。
- 特徴量は時刻 t 以前の情報のみ（OT.shift(s>=0), 因果 rolling）。
  OT(t) は「現在値」であり t < t+h なのでリークではない。
- 未来の負荷 load(t+h) は運用上スケジュール既知という仮定の下でのみ
  use_future_covariates=True で明示的に投入する（既定 False）。
- カレンダー特徴は目的時刻 t+h のもの（決定論的に既知）を用いる。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (DATE_COL, LOAD_COLS, LOAD_LAGS, LOAD_ROLL_WINDOWS,
                    ROLL_WINDOWS, TARGET, TARGET_LAGS)


def _calendar_features(ts: pd.Series) -> pd.DataFrame:
    """目的時刻の周期特徴（hour/dow/month + Fourier 日次・年次）。"""
    out = pd.DataFrame(index=ts.index)
    hour = ts.dt.hour + ts.dt.minute / 60.0
    doy = ts.dt.dayofyear
    out["dow"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["sin_day"] = np.sin(2 * np.pi * hour / 24.0)
    out["cos_day"] = np.cos(2 * np.pi * hour / 24.0)
    out["sin_year"] = np.sin(2 * np.pi * doy / 365.25)
    out["cos_year"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def make_supervised(df: pd.DataFrame,
                    horizon: int,
                    *,
                    target: str = TARGET,
                    load_cols: list[str] = LOAD_COLS,
                    target_lags: list[int] = TARGET_LAGS,
                    load_lags: list[int] = LOAD_LAGS,
                    roll_windows: list[int] = ROLL_WINDOWS,
                    load_roll_windows: list[int] = LOAD_ROLL_WINDOWS,
                    use_future_covariates: bool = False):
    """因果的特徴量 X と目的 y=OT(t+h)、対応する目的時刻 dates を返す。

    horizon=0 は NOWCAST（OT(t) を load(t) から回帰）として扱い、
    OT 由来の特徴を一切使わない（目的のリークを避ける）。
    """
    h = int(horizon)
    feat = pd.DataFrame(index=df.index)

    if h >= 1:
        # 現在値 + 自己ラグ（すべて t 以前 = 因果的）
        feat[f"{target}_now"] = df[target]
        for L in target_lags:
            feat[f"{target}_lag{L}"] = df[target].shift(L)
        # 因果的移動統計（center=False、t までで計算）
        for w in roll_windows:
            feat[f"{target}_rmean{w}"] = df[target].rolling(w, min_periods=w).mean()
            feat[f"{target}_rstd{w}"] = df[target].rolling(w, min_periods=w).std(ddof=0)
        feat[f"{target}_diff1"] = df[target].diff(1)

    # 負荷: 現在 + 過去（因果的）
    for col in load_cols:
        feat[f"{col}_now"] = df[col]
        for L in load_lags:
            feat[f"{col}_lag{L}"] = df[col].shift(L)
        # 負荷の移動平均（熱蓄積=過去負荷の時間積分の代理。因果 rolling）
        for w in load_roll_windows:
            feat[f"{col}_rmean{w}"] = df[col].rolling(w, min_periods=w).mean()

    # 任意: 未来の負荷（スケジュール既知の仮定。既定では使わない）
    if use_future_covariates and h >= 1:
        for col in load_cols:
            feat[f"{col}_future"] = df[col].shift(-h)

    # 目的時刻のカレンダー（決定論的に既知）
    target_ts = df[DATE_COL].shift(-h)
    cal = _calendar_features(target_ts)
    feat = pd.concat([feat, cal], axis=1)

    # 目的変数 OT(t+h)
    y = df[target].shift(-h)
    dates = target_ts

    # 有効行のみ（系列端の NaN を除去）
    valid = feat.notna().all(axis=1) & y.notna()
    return feat.loc[valid].reset_index(drop=True), \
        y.loc[valid].reset_index(drop=True), \
        dates.loc[valid].reset_index(drop=True)

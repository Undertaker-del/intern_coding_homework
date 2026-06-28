"""予測モデル群: ベースライン・Ridge・LightGBM。

すべて「標準化スケール」で学習し、評価時に °C へ逆変換する前提。
ベースラインは特徴量ではなく元系列から直接計算する（公平比較のため）。
"""
from __future__ import annotations

import numpy as np

from config import SEED


def naive_persistence(ot_now: np.ndarray) -> np.ndarray:
    """ŷ(t+h) = OT(t)。基準時刻の値をそのまま将来予測とする。"""
    return np.asarray(ot_now, dtype=float)


def seasonal_naive(ot_at_seasonal_lag: np.ndarray) -> np.ndarray:
    """ŷ(t+h) = OT(t+h-period)。1 周期前の同位相値を予測とする。"""
    return np.asarray(ot_at_seasonal_lag, dtype=float)


class RidgeModel:
    """標準化済みラグ窓への線形回帰（線形上限の目安・DLinear 的）。"""

    def __init__(self, alpha: float = 1.0):
        from sklearn.linear_model import Ridge

        self.model = Ridge(alpha=alpha, random_state=SEED)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)


class LightGBMModel:
    """勾配ブースティング本命。GPU 不要・特徴量重要度で説明可能。"""

    def __init__(self, **overrides):
        import lightgbm as lgb

        self.params = dict(
            objective="regression_l1",  # MAE 最適化
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )
        self.params.update(overrides)
        self._lgb = lgb
        self.model = None

    def fit(self, X, y, eval_set=None):
        self.model = self._lgb.LGBMRegressor(**self.params)
        callbacks = []
        if eval_set is not None:
            callbacks = [self._lgb.early_stopping(50, verbose=False)]
        self.model.fit(X, y, eval_set=eval_set, callbacks=callbacks)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(self, feature_names):
        imp = self.model.booster_.feature_importance(importance_type="gain")
        return dict(sorted(zip(feature_names, imp.tolist()),
                           key=lambda kv: kv[1], reverse=True))

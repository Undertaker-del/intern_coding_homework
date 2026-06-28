"""評価指標と業務価値への翻訳。

OT は 0/負を取り得るため MAPE は不採用。MAE/RMSE（°C）と skill で評価する。
"""
from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(d * d)))


def skill(mae_model: float, mae_baseline: float) -> float:
    """skill = 1 - MAE_model / MAE_baseline。>0 でベースライン超え。"""
    if mae_baseline == 0:
        return float("nan")
    return 1.0 - mae_model / mae_baseline


def threshold_alarm_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            threshold: float) -> dict:
    """閾値超過の早期検知性能（予防保全の業務価値の代理指標）。

    実測が threshold を超える事象を「異常」とし、予測がそれを当てられるかを
    Precision/Recall/F1 で測る。
    """
    yt = np.asarray(y_true) >= threshold
    yp = np.asarray(y_pred) >= threshold
    tp = int(np.sum(yt & yp))
    fp = int(np.sum(~yt & yp))
    fn = int(np.sum(yt & ~yp))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {
        "threshold": float(threshold),
        "n_events": int(np.sum(yt)),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }

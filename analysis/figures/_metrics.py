"""Confusion-matrix metrics + 1,000-bootstrap 95% CIs.

Used by every table/figure that reports point + uncertainty estimates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else float("nan")


def confusion(y: np.ndarray, yhat: np.ndarray) -> tuple[int, int, int, int]:
    y = np.asarray(y); yhat = np.asarray(yhat)
    tp = int(((y == 1) & (yhat == 1)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    return tp, tn, fp, fn


def point_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    tp, tn, fp, fn = confusion(y, yhat)
    n = tp + tn + fp + fn
    sens = _safe_div(tp, tp + fn)
    spec = _safe_div(tn, tn + fp)
    ppv  = _safe_div(tp, tp + fp)
    npv  = _safe_div(tn, tn + fn)
    acc  = _safe_div(tp + tn, n)
    f1   = _safe_div(2 * tp, 2 * tp + fp + fn)
    bal  = (sens + spec) / 2 if not (math.isnan(sens) or math.isnan(spec)) else float("nan")
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if all(
        v > 0 for v in (tp + fp, tp + fn, tn + fp, tn + fn)
    ) else 0.0
    mcc = (tp * tn - fp * fn) / denom if denom else float("nan")
    # Cohen's kappa
    prevalence = _safe_div(tp + fn, n)
    pos_rate = _safe_div(tp + fp, n)
    if not (math.isnan(prevalence) or math.isnan(pos_rate)):
        pe = prevalence * pos_rate + (1 - prevalence) * (1 - pos_rate)
        kappa = (acc - pe) / (1 - pe) if pe < 1 else float("nan")
    else:
        kappa = float("nan")
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "n": n,
        "Sens": sens, "Spec": spec,
        "PPV": ppv, "NPV": npv,
        "Acc": acc, "F1": f1, "MCC": mcc,
        "BalAcc": bal, "kappa": kappa,
        "Prevalence": prevalence,
        "PositiveRate": pos_rate,
    }


@dataclass
class CIResult:
    point: float
    lo: float
    hi: float

    def fmt(self, digits: int = 3) -> str:
        return f"{self.point:.{digits}f} ({self.lo:.{digits}f}–{self.hi:.{digits}f})"


def bootstrap_metrics(
    y: np.ndarray,
    yhat: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 20260520,
    keys: tuple[str, ...] = ("Sens", "Spec", "PPV", "NPV", "Acc", "F1", "MCC", "BalAcc", "kappa"),
) -> dict[str, CIResult]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y); yhat = np.asarray(yhat)
    n = len(y)
    boot = {k: np.empty(n_boot) for k in keys}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        m = point_metrics(y[idx], yhat[idx])
        for k in keys:
            boot[k][b] = m[k]
    point = point_metrics(y, yhat)
    return {
        k: CIResult(
            point=point[k],
            lo=float(np.percentile(boot[k][~np.isnan(boot[k])], 2.5)),
            hi=float(np.percentile(boot[k][~np.isnan(boot[k])], 97.5)),
        )
        for k in keys
    }


def fmt_metric_with_ci(point: float, ci_lo: float, ci_hi: float, *, digits: int = 3) -> str:
    return f"{point:.{digits}f} ({ci_lo:.{digits}f}–{ci_hi:.{digits}f})"


if __name__ == "__main__":  # quick test
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    yhat = rng.integers(0, 2, 500)
    point = point_metrics(y, yhat)
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in point.items()})
    cis = bootstrap_metrics(y, yhat, n_boot=200)
    for k, c in cis.items():
        print(f"{k:>8}: {c.fmt()}")

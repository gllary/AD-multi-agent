# -*- coding: utf-8 -*-
"""
LightGBM 输入预处理（与主仓库 train_lgbm_ablation_v2 对齐，独立拷贝供本项目自包含）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TEXT_PROXY_SUFFIXES = (
    "__text_suggests_ad",
    "__text_suggests_ais",
    "__text_suggests_aos",
)


def column_is_ad_proxy_leakage(col: str) -> bool:
    c = str(col)
    if c == "text_suggests_ad":
        return True
    cl = c.lower()
    if any(cl.endswith(s) for s in _TEXT_PROXY_SUFFIXES):
        return True
    if "suggest_ad" in cl:
        return True
    return False


def drop_leakage_cols(df: pd.DataFrame) -> pd.DataFrame:
    leak = [c for c in df.columns if column_is_ad_proxy_leakage(c)]
    if leak:
        df = df.drop(columns=leak)
    return df


def encode_llm_string_columns(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    bin_map = {
        "0": 0.0,
        "1": 1.0,
        "0.0": 0.0,
        "1.0": 1.0,
        "true": 1.0,
        "false": 0.0,
        "是": 1.0,
        "否": 0.0,
        "unknown": np.nan,
        "nan": np.nan,
        "none": np.nan,
        "": np.nan,
    }
    lvl_map = {
        "low": 0.0,
        "medium": 1.0,
        "high": 2.0,
        "unknown": np.nan,
        "nan": np.nan,
        "none": np.nan,
        "": np.nan,
    }

    for c in [col for col in X.columns if X[col].dtype == "object"]:
        lower = X[c].astype(str).str.strip().str.lower()
        if lower.isin(["low", "medium", "high"]).any():
            X[c] = lower.map(lvl_map).astype(float)
        elif lower.isin(
            ["0", "1", "0.0", "1.0", "true", "false", "是", "否", "unknown", "nan", "none", ""]
        ).any():
            X[c] = lower.map(bin_map).astype(float)
        else:
            coerced = pd.to_numeric(X[c].astype(str).str.strip(), errors="coerce")
            X[c] = coerced if coerced.notna().mean() > 0.5 else np.nan

    for c in X.columns:
        # Use pandas API so nullable dtypes (StringDtype, Int64Dtype, …) are
        # detected correctly. numpy.issubdtype rejects pandas extension dtypes
        # on numpy >= 2 and raises TypeError, which broke the server run on
        # 2026-05-18 (numpy 2.x + pandas 2.x server stack).
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


LLM_PREFIXES = ("history__", "exam__", "echo__", "ecg__", "iab__")


def add_missing_indicators(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in list(X.columns):
        if any(c.startswith(p) for p in LLM_PREFIXES) and not column_is_ad_proxy_leakage(c):
            miss_col = f"{c}_missing"
            if miss_col not in X.columns:
                X[miss_col] = X[c].isna().astype(float)
    return X


_AD_LABEL_MAP = {
    "ad_positive": 1,
    "ad_negative": 0,
    "1": 1,
    "0": 0,
    "1.0": 1,
    "0.0": 0,
    "positive": 1,
    "negative": 0,
}


def encode_ad_label(series: pd.Series) -> np.ndarray:
    lower = series.astype(str).str.strip().str.lower()
    mapped = lower.map(_AD_LABEL_MAP)
    if mapped.isna().any():
        mapped = pd.to_numeric(series, errors="coerce")
    return mapped.astype(int).values


def load_xy(csv_path: str, label_col: str = "AD", id_col: str = "ID"):
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"标签列 '{label_col}' 不存在: {csv_path}")
    df = drop_leakage_cols(df)
    y = encode_ad_label(df[label_col])
    drop = [label_col] + ([id_col] if id_col in df.columns else [])
    X = df.drop(columns=drop)
    X = encode_llm_string_columns(X)
    X = add_missing_indicators(X)
    ids = df[id_col].astype(str).tolist() if id_col in df.columns else None
    return X, y, ids

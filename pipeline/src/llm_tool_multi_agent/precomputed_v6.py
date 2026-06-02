# -*- coding: utf-8 -*-
"""datasetB_v6 预计算概率表（无 LightGBM 运行时亦可跑通路）。"""

from __future__ import annotations

import pandas as pd

from .config import PRECOMPUTED_V6_CP3_TEXT, PRECOMPUTED_V6_DIR


def load_v6_precomputed_score_table(ids: list[str] | None = None, cp3_source: str = "cnn") -> pd.DataFrame:
    if cp3_source == "text" and PRECOMPUTED_V6_CP3_TEXT.exists():
        base = pd.read_csv(PRECOMPUTED_V6_CP3_TEXT)
        base["ID"] = base["ID"].astype(str)
        if ids is not None:
            want = {str(x) for x in ids}
            base = base[base["ID"].isin(want)].copy()
        return base[["ID", "label", "CP1", "CP2", "CP3", "CP4"]]

    pdir = PRECOMPUTED_V6_DIR
    c1 = pd.read_csv(pdir / "holdout_CP1_probs.csv")
    c2 = pd.read_csv(pdir / "holdout_CP2_probs.csv")
    c4 = pd.read_csv(pdir / "holdout_CP2E_probs.csv")
    c3 = pd.read_csv(pdir / "holdout_CNN_probs.csv")
    for d in (c1, c2, c3, c4):
        d["ID"] = d["ID"].astype(str)
    base = c1.rename(columns={"prob": "CP1"})[["ID", "label", "CP1"]]
    base = base.merge(c2.rename(columns={"prob": "CP2"})[["ID", "CP2"]], on="ID", how="inner")
    base = base.merge(c3.rename(columns={"prob": "CP3"})[["ID", "CP3"]], on="ID", how="inner")
    base = base.merge(c4.rename(columns={"prob": "CP4"})[["ID", "CP4"]], on="ID", how="inner")
    if ids is not None:
        want = {str(x) for x in ids}
        base = base[base["ID"].isin(want)].copy()
    return base[["ID", "label", "CP1", "CP2", "CP3", "CP4"]]

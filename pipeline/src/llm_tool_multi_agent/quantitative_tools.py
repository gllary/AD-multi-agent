# -*- coding: utf-8 -*-
"""
定量工具：LightGBM（CP1/CP2/CP2E）+ 1D-CNN（ECG）。
模型权重放在 ``artifacts/models/``；预处理见 ``lgbm_preprocess``（项目内建）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import (
    CP_CSV,
    CP_SRC_DIR,
    CP_TRAIN_REF_DIR,
    ECG_IDS,
    ECG_NPY,
    MULTI_AGENT_DIR,
    POLICY_JSON,
    S1B_DIR,
)
from .lgbm_preprocess import (
    add_missing_indicators,
    drop_leakage_cols,
    encode_llm_string_columns,
    load_xy,
)


def load_policy() -> dict:
    return json.loads(POLICY_JSON.read_text(encoding="utf-8"))


def risk_level(score: float, c_thr: float, a_thr: float) -> str:
    if score < c_thr:
        return "low"
    if score >= a_thr:
        return "high"
    return "intermediate"


def _lgbm_predict_ids(
    ids: Iterable[str],
    stage_key: str,
    csv_name: str,
) -> pd.DataFrame:
    import lightgbm as lgb

    ids_set = {str(x) for x in ids}
    src = CP_SRC_DIR / csv_name
    if not src.exists():
        raise FileNotFoundError(src)
    full = pd.read_csv(src)
    full["ID"] = full["ID"].astype(str)
    h = full[full["ID"].isin(ids_set)].copy()
    if h.empty:
        return pd.DataFrame(columns=["ID", "prob", "label"])

    y_h = h["AAS"].astype(int).values
    X_h = h.drop(columns=["ID", "AAS"])
    X_h = drop_leakage_cols(X_h)
    X_h = encode_llm_string_columns(X_h)
    X_h = add_missing_indicators(X_h)

    train_csv = CP_TRAIN_REF_DIR / csv_name
    if not train_csv.exists():
        raise FileNotFoundError(train_csv)
    X_ref, _, _ = load_xy(str(train_csv))
    for c in X_ref.columns:
        if c not in X_h.columns:
            X_h[c] = np.nan
    X_h = X_h[X_ref.columns]

    mdir = MULTI_AGENT_DIR / stage_key / "fold_models"
    mfiles = sorted(mdir.glob("fold*_booster.txt"))
    if not mfiles:
        raise FileNotFoundError(f"No LGBM fold models under {mdir}")

    preds = np.zeros(len(y_h))
    for mf in mfiles:
        booster = lgb.Booster(model_file=str(mf))
        preds += booster.predict(X_h.values)
    preds /= len(mfiles)

    return pd.DataFrame({"ID": h["ID"].astype(str).values, "prob": preds, "label": y_h})


def _cnn_predict_ids(ids: Iterable[str]) -> pd.DataFrame | None:
    ids_list = [str(x).strip() for x in ids]
    mdir = S1B_DIR / "fold_models"
    mfiles = sorted(mdir.glob("fold*_best.pt"))
    if not mfiles or not Path(ECG_NPY).exists() or not Path(ECG_IDS).exists():
        return None

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        return None

    from .ecg_model import build_ecg_net

    signals_all = np.load(ECG_NPY)
    id_map = pd.read_csv(ECG_IDS)
    id_map["ID"] = id_map["ID"].astype(str).str.strip()
    id2idx = dict(zip(id_map["ID"], id_map["signal_index"]))

    cp1 = pd.read_csv(CP_SRC_DIR / CP_CSV["CP1"])
    cp1["ID"] = cp1["ID"].astype(str)
    label_df = pd.DataFrame({"ID": ids_list}).merge(cp1[["ID", "AAS"]], on="ID", how="inner")
    label_df = label_df[label_df["ID"].isin(id2idx)].reset_index(drop=True)
    if label_df.empty:
        return pd.DataFrame(columns=["ID", "prob", "label"])

    y_h = label_df["AAS"].astype(int).values
    X = signals_all[[id2idx[i] for i in label_df["ID"]], :, :].copy()
    for i in range(len(X)):
        for c in range(12):
            s = X[i, c]
            if s.any():
                mu, sd = s.mean(), s.std()
                if sd > 1e-8:
                    X[i, c] = (s - mu) / sd
    X = X.astype(np.float32)

    class SD(torch.utils.data.Dataset):
        def __init__(self, x):
            self.x = x

        def __len__(self):
            return len(self.x)

        def __getitem__(self, i):
            return torch.from_numpy(self.x[i])

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    ld = DataLoader(SD(X), 32, shuffle=False, num_workers=0)
    preds = np.zeros(len(y_h))
    for mf in mfiles:
        model = build_ecg_net().to(device)
        model.load_state_dict(torch.load(str(mf), map_location=device, weights_only=False))
        model.eval()
        out = []
        with torch.no_grad():
            for xb in ld:
                out.append(model(xb.to(device)).cpu().numpy())
        preds += np.concatenate(out)
    preds /= len(mfiles)

    return pd.DataFrame({"ID": label_df["ID"].astype(str).values, "prob": preds, "label": y_h})


def build_score_table_for_ids(ids: list[str]) -> pd.DataFrame:
    ids = [str(i) for i in ids]
    d1 = _lgbm_predict_ids(ids, "CP1", CP_CSV["CP1"])
    if d1.empty:
        return pd.DataFrame(columns=["ID", "label", "CP1", "CP2", "CP3", "CP4"])
    base = d1.rename(columns={"prob": "CP1"})
    d2 = _lgbm_predict_ids(ids, "CP2", CP_CSV["CP2"])
    d4 = _lgbm_predict_ids(ids, "CP2E", CP_CSV["CP2E"])
    if not d2.empty:
        base = base.merge(d2.rename(columns={"prob": "CP2"})[["ID", "CP2"]], on="ID", how="left")
    else:
        base["CP2"] = 0.5
    if not d4.empty:
        base = base.merge(d4.rename(columns={"prob": "CP4"})[["ID", "CP4"]], on="ID", how="left")
    else:
        base["CP4"] = 0.5
    cnn = _cnn_predict_ids(ids)
    if cnn is not None and not cnn.empty:
        base = base.merge(cnn.rename(columns={"prob": "CP3"})[["ID", "CP3"]], on="ID", how="left")
    else:
        base["CP3"] = 0.5
    base = base.fillna({"CP1": 0.5, "CP2": 0.5, "CP3": 0.5, "CP4": 0.5})
    return base[["ID", "label", "CP1", "CP2", "CP3", "CP4"]]


def load_cp_row(patient_id: str, stage: str) -> pd.Series:
    csv_name = {"CP1": CP_CSV["CP1"], "CP2": CP_CSV["CP2"], "CP4": CP_CSV["CP2E"]}[stage]
    df = pd.read_csv(CP_SRC_DIR / csv_name)
    df["ID"] = df["ID"].astype(str)
    sub = df[df["ID"] == str(patient_id)]
    if sub.empty:
        return pd.Series(dtype=object)
    return sub.iloc[0]

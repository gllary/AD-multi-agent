# -*- coding: utf-8 -*-
"""1D-CNN used as ECG risk tool (architecture aligned with run_multiagent_holdout.py)."""

from __future__ import annotations


def build_ecg_net():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Res1D(nn.Module):
        def __init__(self, ch, ks=5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(ch, ch, ks, padding=ks // 2, bias=False),
                nn.BatchNorm1d(ch),
                nn.ReLU(True),
                nn.Conv1d(ch, ch, ks, padding=ks // 2, bias=False),
                nn.BatchNorm1d(ch),
            )

        def forward(self, x):
            return F.relu(x + self.net(x), True)

    class ECGNetLite(nn.Module):
        def __init__(self, nl=12):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv1d(nl, 32, 25, stride=8, padding=12, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(True),
                Res1D(32, 5),
                nn.Conv1d(32, 64, 13, stride=8, padding=6, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(True),
                Res1D(64, 5),
                nn.Conv1d(64, 128, 9, stride=8, padding=4, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(True),
                Res1D(128, 3),
            )
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 32),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            return torch.sigmoid(self.head(self.gap(self.enc(x)))).squeeze(-1)

    return ECGNetLite(12)

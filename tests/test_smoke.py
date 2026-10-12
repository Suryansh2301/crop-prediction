"""Run:  pytest -q   (after `python run_all.py --synthetic --fast`)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config as C
from src.train_yield import CAT_COLS, load_data


def test_no_leakage_columns_in_features():
    _, features, _, _ = load_data()
    assert "Production" not in features and "Yield" not in features and "y" not in features


def test_yield_equals_production_over_area():
    df = pd.read_csv(C.YIELD_CLEAN)
    assert np.allclose(df["Yield"], df["Production"] / df["Area"])


def test_temporal_split_is_disjoint_in_time():
    df = pd.read_csv(C.YIELD_CLEAN)
    assert df[df["Year"] <= C.TRAIN_END_YEAR]["Year"].max() < df[df["Year"] > C.TRAIN_END_YEAR]["Year"].min()


def test_inference_roundtrip():
    from src.predict import load_reco_artifacts, load_yield_artifacts, predict_yield, recommend_crops
    model, meta = load_yield_artifacts()
    s = sorted(meta["geo"])[0]
    d = meta["geo"][s][0]
    c = meta["state_crops"][s][0]
    se = meta["crop_seasons"][c][0]
    y, p = predict_yield(model, meta, s, d, c, se, 2014, 1000.0)
    assert y > 0 and p > 0
    top = recommend_crops(load_reco_artifacts(), 90, 42, 43, 20.9, 82, 6.5, 203)
    assert abs(sum(pr for _, pr in top)) <= 1.0 + 1e-6 and len(top) == 3

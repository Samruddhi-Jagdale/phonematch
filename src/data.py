"""Loading and light feature engineering."""
import json

import pandas as pd

from .config import (DATA_DIR, REQUIRED_PHONE_COLUMNS, REQUIRED_REVIEW_COLUMNS)


def _check(df, required, name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    return df


def load_phones(path=None):
    df = pd.read_csv(path or DATA_DIR / "phones.csv")
    _check(df, REQUIRED_PHONE_COLUMNS, "phones.csv")
    df = df.dropna(subset=REQUIRED_PHONE_COLUMNS).drop_duplicates("phone_id").reset_index(drop=True)
    return df


def load_reviews(path=None):
    df = pd.read_csv(path or DATA_DIR / "reviews.csv")
    _check(df, REQUIRED_REVIEW_COLUMNS, "reviews.csv")
    return df.dropna(subset=REQUIRED_REVIEW_COLUMNS).reset_index(drop=True)


def load_dataset_info():
    f = DATA_DIR / "dataset_info.json"
    return json.loads(f.read_text()) if f.exists() else {"type": "unknown"}


def pct(s):
    """Percentile rank in (0, 1]: how a phone compares with the whole catalogue."""
    return s.rank(pct=True, method="average")


def add_spec_scores(df):
    """Five interpretable 0-1 scores built from raw specs."""
    d = df.copy()
    d["spec_performance"] = 0.60 * pct(d.processor_score) + 0.25 * pct(d.ram_gb) + 0.15 * pct(d.storage_gb)
    d["spec_camera"] = 0.50 * pct(d.camera_score) + 0.30 * pct(d.rear_camera_mp) + 0.20 * pct(d.front_camera_mp)
    d["spec_battery"] = 0.70 * pct(d.battery_mah) + 0.30 * pct(d.charging_w)
    d["spec_display"] = 0.50 * pct(d.display_inch) + 0.50 * pct(d.refresh_hz)
    return d

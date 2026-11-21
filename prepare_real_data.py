"""Convert a real smartphone CSV into PhoneMatch's format.

1. Edit COLUMN_MAP so each value is the column name in YOUR file.
2. Run:  python prepare_real_data.py path/to/your_file.csv [--usd-to-inr 83] [--source "Kaggle: dataset name"]
3. Provide real reviews as data/reviews.csv (columns: phone_id, stars, text) or create simulated ones with
   `python generate_data.py --reviews-only` (then keep dataset_info.json honest about it).
4. Re-run `python run_experiments.py` and the app.

Numeric columns stored as text ("8 GB", "5000 mAh", "Rs. 24,999") are cleaned automatically.
"""
import argparse
import json
import re

import numpy as np
import pandas as pd

from src.config import DATA_DIR, REQUIRED_PHONE_COLUMNS

COLUMN_MAP = {
    "brand": "brand",
    "model": "model",
    "price_inr": "price",
    "ram_gb": "ram",
    "storage_gb": "storage",
    "battery_mah": "battery",
    "charging_w": "charging",
    "rear_camera_mp": "rear_camera",
    "front_camera_mp": "front_camera",
    "display_inch": "screen_size",
    "refresh_hz": "refresh_rate",
    "processor_score": "benchmark",   # e.g. an AnTuTu score. If you have none, see PROXY below.
    "is_5g": "5g",
    "rating": "rating",
    "camera_score": None,             # optional (e.g. DXOMARK). None -> proxy from rear camera MP
}
PROXY_PROCESSOR = None  # e.g. ("cpu_cores", "cpu_ghz") -> score = cores * GHz * 100000


def to_number(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.extract(r"(\d+\.?\d*)")[0], errors="coerce")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--usd-to-inr", type=float, default=1.0, help="multiply prices by this to convert to INR")
    ap.add_argument("--source", default="real dataset (edit dataset_info.json)")
    a = ap.parse_args()
    raw = pd.read_csv(a.csv)
    out = pd.DataFrame()
    for target, src in COLUMN_MAP.items():
        if src and src in raw.columns:
            out[target] = raw[src]
    for c in out.columns:
        if c not in ("brand", "model"):
            out[c] = out[c].astype(str).str.lower().replace({"yes": "1", "no": "0", "true": "1", "false": "0"}) if c == "is_5g" else out[c]
            out[c] = to_number(out[c])
    out["price_inr"] *= a.usd_to_inr
    if "camera_score" not in out:
        out["camera_score"] = out["rear_camera_mp"].rank(pct=True) * 100  # weak proxy, state this in your report
    if "processor_score" not in out and PROXY_PROCESSOR:
        cores, ghz = PROXY_PROCESSOR
        out["processor_score"] = to_number(raw[cores]) * to_number(raw[ghz]) * 100000
    out["is_5g"] = out.get("is_5g", 0).fillna(0).clip(0, 1)
    if "brand" not in out:
        out["brand"] = out["model"].astype(str).str.split().str[0]
    out = out.dropna(subset=[c for c in REQUIRED_PHONE_COLUMNS if c != "phone_id"]).drop_duplicates("model")
    out.insert(0, "phone_id", np.arange(1, len(out) + 1))
    out.to_csv(DATA_DIR / "phones.csv", index=False)
    (DATA_DIR / "dataset_info.json").write_text(json.dumps(
        {"type": "real", "name": a.source, "n_phones": len(out), "note": "Prepared with prepare_real_data.py"}, indent=2))
    print(f"Wrote {len(out)} phones to data/phones.csv. Now provide data/reviews.csv and run run_experiments.py")


if __name__ == "__main__":
    main()

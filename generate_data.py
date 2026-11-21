"""Generate the bundled DEMO dataset (synthetic) for PhoneMatch.

    python generate_data.py                 # phones.csv + reviews.csv
    python generate_data.py --reviews-only  # keep your phones.csv, regenerate reviews

IMPORTANT: this data is synthetic. Brands and models are fictional. Specs, prices
and reviews are simulated so the project runs offline. Replace data/phones.csv and
data/reviews.csv with real data (see README.md) before drawing real-world conclusions.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DIMS

DATA = Path(__file__).parent / "data"

# fictional brand: (price premium, specialty)
BRANDS = {
    "Vanta": (1.15, "camera"), "Orbix": (1.00, "balanced"), "Kestrel": (0.92, "performance"),
    "Lumio": (1.05, "camera"), "Nexora": (0.88, "battery"), "Zephyr": (0.95, "performance"),
    "Aurum": (1.25, "balanced"), "Tessel": (0.90, "battery"),
}

TIERS = [
    dict(name="entry", w=0.30, ram=[3, 4, 6], storage=[64, 128], bat=(4800, 6000), chg=[10, 15, 18, 22, 33],
         rear=[8, 13, 48, 50], front=[5, 8], cam=(35, 58), disp=(6.5, 6.8), hz=[60, 90],
         proc=(120e3, 380e3), g5=0.2, series=["Spark", "Lite", "Go"]),
    dict(name="mid", w=0.35, ram=[6, 8], storage=[128, 256], bat=(4800, 5600), chg=[18, 33, 45, 67],
         rear=[50, 64, 108], front=[8, 13, 16], cam=(52, 78), disp=(6.5, 6.7), hz=[90, 120],
         proc=(380e3, 800e3), g5=0.85, series=["Neo", "Plus", "Edge"]),
    dict(name="upper", w=0.22, ram=[8, 12], storage=[256, 512], bat=(4500, 5500), chg=[45, 67, 80, 100],
         rear=[50, 64, 108, 200], front=[16, 32], cam=(68, 88), disp=(6.4, 6.8), hz=[120, 144],
         proc=(800e3, 1.3e6), g5=1.0, series=["Pro", "Max"]),
    dict(name="flagship", w=0.13, ram=[12, 16], storage=[256, 512, 1024], bat=(4400, 5400), chg=[67, 80, 100, 120],
         rear=[50, 108, 200], front=[32, 50], cam=(80, 98), disp=(6.3, 6.9), hz=[120, 144],
         proc=(1.3e6, 1.9e6), g5=1.0, series=["Ultra", "Prime"]),
]


def expected_price(d):
    """Spec-based price formula used by the generator (hidden from the models)."""
    core = (2500 + 38000 * d.processor_score / 1e6 + 700 * d.ram_gb + 18 * d.storage_gb
            + 120 * d.camera_score + 60 * (d.refresh_hz - 60) + 3000 * d.is_5g
            + 60 * d.charging_w + 0.8 * (d.battery_mah - 4000))
    return 0.62 * core * (0.8 + 0.5 * d.processor_score / 1.7e6)


def make_phones(n, rng):
    rows, used = [], set()
    brands = list(BRANDS)
    for pid in range(1, n + 1):
        ti = int(rng.choice(len(TIERS), p=[x["w"] for x in TIERS]))
        t = TIERS[ti]
        # each spec group can sit one tier above/below the processor tier, so specs are only
        # partly correlated (e.g. a camera-focused mid-ranger, a big-battery budget phone)
        pick = lambda: TIERS[int(np.clip(ti + rng.choice([-1, 0, 0, 1]), 0, len(TIERS) - 1))]
        tc, tb, td, tr = pick(), pick(), pick(), pick()
        brand = brands[rng.integers(len(brands))]
        premium, spec = BRANDS[brand]
        while True:
            model = f"{brand} {rng.choice(t['series'])} {rng.integers(2, 100)}"
            if model not in used:
                used.add(model)
                break
        cam = rng.uniform(*tc["cam"]) + (6 if spec == "camera" else 0)
        bat = int(rng.integers(*tb["bat"]) + (400 if spec == "battery" else 0))
        proc = rng.uniform(*t["proc"]) * (1.08 if spec == "performance" else 1.0)
        row = dict(
            phone_id=pid, brand=brand, model=model, launch_year=int(rng.integers(2022, 2026)),
            ram_gb=int(rng.choice(tr["ram"])), storage_gb=int(rng.choice(tr["storage"])),
            battery_mah=bat, charging_w=int(rng.choice(tb["chg"])),
            rear_camera_mp=int(rng.choice(tc["rear"])), front_camera_mp=int(rng.choice(tc["front"])),
            camera_score=int(round(min(cam, 99))), display_inch=round(float(rng.uniform(*td["disp"])), 1),
            refresh_hz=int(rng.choice(td["hz"])), processor_score=int(proc),
            is_5g=int(rng.random() < t["g5"]),
        )
        rows.append((row, premium))
    df = pd.DataFrame([r for r, _ in rows])
    prem = np.array([p for _, p in rows])
    price = expected_price(df) * prem * rng.lognormal(0, 0.08, len(df))
    df["price_inr"] = np.maximum(np.round(price / 500) * 500 - 1, 5999).astype(int)
    z = (np.log(expected_price(df) / df.price_inr))
    z = (z - z.mean()) / z.std()
    perf = df.processor_score.rank(pct=True)
    df["rating"] = np.clip(3.9 + 0.25 * z + 0.35 * (perf - 0.5) + rng.normal(0, 0.15, len(df)), 3.3, 4.8).round(1)
    df["rating_count"] = np.clip(rng.lognormal(7.5, 1.0, len(df)), 50, 60000).astype(int)
    return df


POS = {
    "battery": ["The battery easily lasts a full day with heavy use.", "Charging is really fast, full in under an hour.",
                "Battery backup is excellent, I charge it every two days.", "Great battery life even while streaming.",
                "Charges quickly and the battery holds up well."],
    "camera": ["The camera takes stunning photos in daylight.", "Selfie quality is superb and the portrait mode is sharp.",
               "Camera is fantastic, low light shots look great.", "Photos are crisp and the video is stabilised nicely.",
               "Really impressed with the camera, colours look natural."],
    "display": ["The display is bright and vivid, a joy to watch movies on.", "Screen scrolls fluidly with the high refresh rate.",
                "Amoled screen looks gorgeous and the brightness is great outdoors.", "Display quality is excellent, sharp and colourful.",
                "The screen feels premium and scrolling is buttery."],
    "performance": ["Performance is blazing fast, no lag at all.", "Games run smoothly even on high settings.",
                    "The processor handles multitasking effortlessly.", "Very smooth experience, apps open instantly.",
                    "Gaming is fantastic and there is no heating."],
    "value": ["Great value for money at this price.", "Totally worth the price, an amazing deal.",
              "You get a lot for the money, very good value.", "Cheap for what it offers, a real bargain.",
              "The price is fair and it feels worth every rupee."],
}
NEG = {
    "battery": ["Battery drains way too fast, barely lasts till evening.", "Charging is painfully slow.",
                "Battery backup is disappointing.", "The battery dies quickly with normal use.",
                "Poor battery life, needs charging twice a day."],
    "camera": ["The camera is disappointing, photos look washed out.", "Selfie quality is poor and portrait mode is blurry.",
               "Camera struggles in low light and the video is shaky.", "Photos are grainy, the camera is just average.",
               "Very unhappy with the camera, colours look fake."],
    "display": ["The display is dull and hard to see outside.", "Screen feels choppy with a low refresh rate.",
                "Display quality is mediocre, colours look washed out.", "The screen brightness is too low.",
                "Screen has a noticeable tint and feels plasticky."],
    "performance": ["Performance is sluggish, there is constant lag.", "Games stutter badly and the phone gets hot.",
                    "The processor struggles with multitasking.", "Terrible speed, apps take ages to open.",
                    "Heating is a real problem while gaming."],
    "value": ["Not worth the price at all, very expensive for what you get.", "Poor value for money.",
              "Overpriced, you can find a better deal elsewhere.", "Too expensive for what it offers.",
              "The price is hard to justify, bad value."],
}
FILLER = ["Delivery was on time and the packaging was good.", "Using it for about two weeks now.",
          "Bought it during an online sale.", "Comes in a sturdy box.", "The design looks nice in hand."]


def make_reviews(ph, rng, mean_reviews=8):
    pct = lambda s: s.rank(pct=True)
    ph = ph.reset_index(drop=True)
    q = pd.DataFrame({
        "performance": 0.7 * pct(ph.processor_score) + 0.3 * pct(ph.ram_gb),
        "camera": 0.7 * pct(ph.camera_score) + 0.3 * pct(ph.rear_camera_mp),
        "battery": 0.7 * pct(ph.battery_mah) + 0.3 * pct(ph.charging_w),
        "display": 0.5 * pct(ph.refresh_hz) + 0.5 * pct(ph.display_inch),
        "value": pct(np.log(expected_price(ph) / ph.price_inr)),
    })
    rows, rid = [], 1
    for i, pid in enumerate(ph.phone_id):
        for _ in range(5 + rng.poisson(mean_reviews)):
            dims = list(rng.choice(DIMS, size=rng.integers(2, 4), replace=False))
            sents, pos = [], 0
            for d in dims:
                qi = float(np.clip(q.loc[i, d] + rng.normal(0, 0.1), 0, 1))
                is_pos = rng.random() < 1 / (1 + np.exp(-(qi - 0.45) * 7))
                sents.append(str(rng.choice(POS[d] if is_pos else NEG[d])))
                pos += is_pos
            if rng.random() < 0.5:
                sents.insert(int(rng.integers(0, len(sents) + 1)), str(rng.choice(FILLER)))
            stars = int(np.clip(round(1 + 4 * pos / len(dims) + rng.normal(0, 0.45)), 1, 5))
            if rng.random() < 0.08:  # label noise, like real-world inconsistent ratings
                stars = 6 - stars
            rows.append((rid, pid, stars, " ".join(sents)))
            rid += 1
    return pd.DataFrame(rows, columns=["review_id", "phone_id", "stars", "text"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phones", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reviews-only", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    DATA.mkdir(exist_ok=True)
    if a.reviews_only:
        phones = pd.read_csv(DATA / "phones.csv")
    else:
        phones = make_phones(a.phones, rng)
        phones.to_csv(DATA / "phones.csv", index=False)
    reviews = make_reviews(phones, rng)
    reviews.to_csv(DATA / "reviews.csv", index=False)
    info = {"type": "synthetic", "name": "PhoneMatch demo catalogue", "generator": "generate_data.py",
            "seed": a.seed, "n_phones": len(phones), "n_reviews": len(reviews),
            "note": "Fictional brands and simulated specs, prices and reviews."}
    (DATA / "dataset_info.json").write_text(json.dumps(info, indent=2))
    print(f"phones: {len(phones)}  reviews: {len(reviews)}")


if __name__ == "__main__":
    main()

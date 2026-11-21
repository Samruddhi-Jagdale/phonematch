"""Shared constants for PhoneMatch."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

DIMS = ["performance", "camera", "battery", "display", "value"]
DIM_LABELS = {
    "performance": "Performance",
    "camera": "Camera",
    "battery": "Battery",
    "display": "Display",
    "value": "Value for money",
}

SPEC_FEATURES = [
    "processor_score", "ram_gb", "storage_gb", "battery_mah", "charging_w",
    "rear_camera_mp", "front_camera_mp", "camera_score", "display_inch",
    "refresh_hz", "is_5g",
]
FEATURE_LABELS = {
    "processor_score": "Processor", "ram_gb": "RAM", "storage_gb": "Storage",
    "battery_mah": "Battery size", "charging_w": "Charging speed",
    "rear_camera_mp": "Rear camera MP", "front_camera_mp": "Selfie camera MP",
    "camera_score": "Camera quality", "display_inch": "Screen size",
    "refresh_hz": "Refresh rate", "is_5g": "5G",
}
FEATURE_GROUPS = {
    "Performance": ["processor_score", "ram_gb", "storage_gb"],
    "Camera": ["rear_camera_mp", "front_camera_mp", "camera_score"],
    "Battery": ["battery_mah", "charging_w"],
    "Display": ["display_inch", "refresh_hz"],
    "Connectivity": ["is_5g"],
}

REQUIRED_PHONE_COLUMNS = ["phone_id", "brand", "model", "price_inr", "rating"] + SPEC_FEATURES
REQUIRED_REVIEW_COLUMNS = ["phone_id", "stars", "text"]

# Word stems used to detect which aspect a review sentence talks about.
ASPECT_STEMS = {
    "performance": ["performance", "lag", "gaming", "game", "processor", "smooth", "heating", "speed", "multitask"],
    "camera": ["camera", "photo", "selfie", "video", "lens", "portrait"],
    "battery": ["battery", "charg", "mah", "backup"],
    "display": ["display", "screen", "amoled", "refresh", "brightness"],
    "value": ["price", "value", "money", "worth", "cost", "expensive", "cheap", "deal"],
}

# Simulated buyer personas: used as presets in the app and for evaluation.
PERSONAS = {
    "Camera-first student": {"weights": {"performance": 4, "camera": 10, "battery": 5, "display": 4, "value": 7}, "budget": (15000, 25000)},
    "Mobile gamer": {"weights": {"performance": 10, "camera": 3, "battery": 6, "display": 8, "value": 4}, "budget": (25000, 45000)},
    "Battery-focused commuter": {"weights": {"performance": 3, "camera": 3, "battery": 10, "display": 3, "value": 7}, "budget": (10000, 20000)},
    "Flagship seeker": {"weights": {"performance": 9, "camera": 9, "battery": 5, "display": 9, "value": 2}, "budget": (60000, 150000)},
    "Budget-conscious parent": {"weights": {"performance": 2, "camera": 3, "battery": 8, "display": 2, "value": 10}, "budget": (6000, 12000)},
    "All-rounder": {"weights": {"performance": 6, "camera": 6, "battery": 6, "display": 6, "value": 6}, "budget": (18000, 32000)},
}

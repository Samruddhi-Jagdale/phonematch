"""Run with:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DIMS, PERSONAS, SPEC_FEATURES
from src.data import load_phones, load_reviews
from src.pipeline import build
from src.sentiment import aspects_in, split_sentences
from src.value_model import explain_price


@pytest.fixture(scope="module")
def bundle():
    return build()


# ---- data -------------------------------------------------------------------------------
def test_data_is_clean():
    ph, rv = load_phones(), load_reviews()
    assert ph.phone_id.is_unique and len(ph) >= 2000
    assert (ph.price_inr > 0).all()
    assert not ph[SPEC_FEATURES].isna().any().any()
    assert rv.stars.between(1, 5).all()
    assert set(rv.phone_id) <= set(ph.phone_id)


# ---- models -----------------------------------------------------------------------------
def test_scores_in_unit_range(bundle):
    p = bundle.phones
    for d in DIMS:
        for prefix in ("spec_", "rev_"):
            assert p[f"{prefix}{d}"].between(0, 1).all()


def test_value_model_is_reasonable(bundle):
    p = bundle.phones
    assert (p.pred_price > 0).all()
    corr = np.corrcoef(np.log(p.pred_price), np.log(p.price_inr))[0, 1]
    assert corr > 0.9


def test_price_explanation_covers_all_features(bundle):
    eff = explain_price(bundle.value_model, bundle.phones, int(bundle.phones.phone_id.iloc[0]))
    assert set(eff.index) == set(SPEC_FEATURES)


def test_sentiment_direction(bundle):
    pos = bundle.sentiment.predict_proba(["The camera is fantastic, photos look great."])[0, 1]
    neg = bundle.sentiment.predict_proba(["Battery drains way too fast, very poor."])[0, 1]
    assert pos > 0.7 > 0.3 > neg


def test_aspect_detection():
    assert aspects_in("The battery lasts long.") == ["battery"]
    assert aspects_in("Great camera but poor battery.") == ["camera", "battery"]
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_clusters(bundle):
    c = bundle.clusters
    assert len(set(c["labels"])) == c["k"] >= 3
    assert len(bundle.phones.cluster_name.unique()) == c["k"]


# ---- recommender ------------------------------------------------------------------------
@pytest.mark.parametrize("persona", list(PERSONAS))
def test_recommendations_respect_filters(bundle, persona):
    p = PERSONAS[persona]
    res = bundle.recommender.recommend(p["weights"], p["budget"], k=5)
    assert 0 < len(res) <= 5
    assert res.price_inr.between(*p["budget"]).all()
    assert res.match.is_monotonic_decreasing


def test_contributions_sum_to_match(bundle):
    res = bundle.recommender.recommend({d: 5 for d in DIMS}, (10000, 60000), k=5)
    assert np.allclose(res[[f"c_{d}" for d in DIMS]].sum(axis=1), res.match)


def test_priorities_change_the_ranking(bundle):
    rec = bundle.recommender
    cam = rec.recommend({"camera": 10}, (20000, 40000), trust_reviews=0, k=1).iloc[0]
    bat = rec.recommend({"battery": 10}, (20000, 40000), trust_reviews=0, k=1).iloc[0]
    assert cam.camera_score >= bat.camera_score
    assert bat.battery_mah >= cam.battery_mah


def test_zero_weights_do_not_crash(bundle):
    assert len(bundle.recommender.recommend({}, (10000, 60000), k=3)) == 3


def test_filters_and_empty_result(bundle):
    rec = bundle.recommender
    assert rec.recommend({"camera": 5}, (1, 2)).empty
    only5g = rec.recommend({"camera": 5}, (10000, 80000), only_5g=True, k=10)
    assert (only5g.is_5g == 1).all()


def test_similar_excludes_self(bundle):
    pid = int(bundle.phones.phone_id.iloc[3])
    sim = bundle.recommender.similar(pid, 5)
    assert pid not in set(sim.phone_id) and len(sim) > 0


# ---- the Streamlit app ------------------------------------------------------------------
def test_app_runs_without_errors():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).parent.parent / "app.py"), default_timeout=180).run()
    assert not at.exception
    at.selectbox(key="preset").select("Mobile gamer").run()
    assert not at.exception
    at.slider(key="w_camera").set_value(10).run()
    assert not at.exception

"""Review mining: sentiment classifier + per-phone aspect scores."""
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .config import ASPECT_STEMS, DIMS

ASPECT_RE = {d: re.compile(r"\b(?:" + "|".join(stems) + ")") for d, stems in ASPECT_STEMS.items()}


def labelled(reviews):
    """Positive = 4-5 stars, negative = 1-2 stars; 3-star reviews are dropped."""
    lab = reviews[reviews.stars != 3]
    return lab.text, (lab.stars >= 4).astype(int)


def make_pipeline_lr():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(max_iter=1000, C=3.0))


def train_sentiment(reviews):
    X, y = labelled(reviews)
    return make_pipeline_lr().fit(X, y)


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def aspects_in(sentence):
    low = sentence.lower()
    return [d for d in DIMS if ASPECT_RE[d].search(low)]


def sentence_table(reviews, pipe):
    """One row per (review sentence, aspect mentioned) with P(positive)."""
    rows = []
    for pid, text in zip(reviews.phone_id, reviews.text):
        for s in split_sentences(text):
            for d in aspects_in(s):
                rows.append((pid, d, s))
    sent = pd.DataFrame(rows, columns=["phone_id", "dim", "sentence"])
    sent["p_pos"] = pipe.predict_proba(sent.sentence)[:, 1]
    return sent


def aspect_scores(sent, phone_ids, prior=3):
    """Per-phone aspect sentiment in [0, 1], shrunk toward the catalogue mean when mentions are few."""
    gm = sent.groupby("dim").p_pos.mean()
    g = sent.groupby(["phone_id", "dim"]).p_pos.agg(["sum", "count"]).reset_index()
    g["score"] = (g["sum"] + prior * g.dim.map(gm)) / (g["count"] + prior)
    score = g.pivot(index="phone_id", columns="dim", values="score").reindex(phone_ids)
    count = g.pivot(index="phone_id", columns="dim", values="count").reindex(phone_ids).fillna(0)
    for d in DIMS:
        score[d] = score.get(d, gm[d]).fillna(gm[d]) if d in score else gm[d]
        if d not in count:
            count[d] = 0
    return score[DIMS], count[DIMS]

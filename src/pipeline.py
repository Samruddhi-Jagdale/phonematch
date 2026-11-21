"""Builds every model and table the app and the experiments need."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .clustering import fit_clusters
from .config import DIMS
from .data import add_spec_scores, load_phones, load_reviews, pct
from .recommender import PhoneRecommender
from .sentiment import aspect_scores, sentence_table, train_sentiment
from .value_model import fit_value_model


@dataclass
class Bundle:
    phones: pd.DataFrame
    reviews: pd.DataFrame
    sentences: pd.DataFrame
    recommender: PhoneRecommender
    value_model: object
    sentiment: object
    clusters: dict


def build(phones_path=None, reviews_path=None):
    phones = add_spec_scores(load_phones(phones_path))
    reviews = load_reviews(reviews_path)
    reviews = reviews[reviews.phone_id.isin(phones.phone_id)]

    # fair-price model -> value gap -> value score
    est, oof = fit_value_model(phones)
    phones["pred_price"] = oof
    phones["value_gap"] = (oof - phones.price_inr) / oof
    phones["spec_value"] = pct(phones.value_gap)

    # review mining -> aspect sentiment
    sent_model = train_sentiment(reviews)
    sentences = sentence_table(reviews, sent_model)
    score, count = aspect_scores(sentences, phones.phone_id)
    for d in DIMS:
        phones[f"rev_raw_{d}"] = score[d].to_numpy()
        phones[f"rev_n_{d}"] = count[d].to_numpy().astype(int)
        phones[f"rev_{d}"] = pct(phones[f"rev_raw_{d}"])

    # market segments
    cl = fit_clusters(phones)
    phones["cluster"] = cl["labels"]
    phones["cluster_name"] = phones.cluster.map(cl["names"])
    phones["pc1"], phones["pc2"] = cl["xy"][:, 0], cl["xy"][:, 1]

    return Bundle(phones, reviews, sentences, PhoneRecommender(phones), est, sent_model, cl)

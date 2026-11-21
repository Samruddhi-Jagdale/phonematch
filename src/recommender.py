"""Hybrid recommender: spec scores + review sentiment + fair-price value, weighted by the user."""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from .config import DIMS, SPEC_FEATURES


class PhoneRecommender:
    def __init__(self, phones):
        self.phones = phones.reset_index(drop=True)
        self.S = self.phones[[f"spec_{d}" for d in DIMS]].to_numpy(float)
        self.T = self.phones[[f"rev_{d}" for d in DIMS]].to_numpy(float)
        X = self.phones[SPEC_FEATURES].copy()
        X["log_price"] = np.log(self.phones.price_inr)
        self.X = StandardScaler().fit_transform(X)

    @staticmethod
    def normalise(weights):
        w = np.array([float(weights.get(d, 0)) for d in DIMS])
        return w / w.sum() if w.sum() > 0 else np.ones(len(DIMS)) / len(DIMS)

    def blend(self, trust_reviews=0.3):
        """Per-dimension score: (1 - trust) * spec score + trust * review score."""
        return (1 - trust_reviews) * self.S + trust_reviews * self.T

    def recommend(self, weights, budget=(0, 10**9), min_ram=0, brands=None, only_5g=False,
                  trust_reviews=0.3, k=5):
        w = self.normalise(weights)
        B = self.blend(trust_reviews)
        contrib = B * w
        score = contrib.sum(1)
        p = self.phones
        mask = p.price_inr.between(budget[0], budget[1]) & (p.ram_gb >= min_ram)
        if brands:
            mask &= p.brand.isin(brands)
        if only_5g:
            mask &= p.is_5g == 1
        idx = np.where(mask.to_numpy())[0]
        if len(idx) == 0:
            return p.iloc[0:0].copy()
        order = idx[np.argsort(-score[idx])][:k]
        out = p.iloc[order].copy()
        out["match"] = score[order] * 100
        for i, d in enumerate(DIMS):
            out[f"c_{d}"] = contrib[order, i] * 100   # weighted contribution to the match score
            out[f"b_{d}"] = B[order, i] * 100         # blended 0-100 score for the dimension
        return out

    def profile(self, phone_ids, trust_reviews=0.3):
        idx = self.phones.index[self.phones.phone_id.isin(phone_ids)]
        return self.phones.loc[idx], self.blend(trust_reviews)[idx] * 100

    def similar(self, phone_id, k=5, price_band=0.4):
        i = int(self.phones.index[self.phones.phone_id == phone_id][0])
        sim = cosine_similarity(self.X[[i]], self.X)[0]
        sim[i] = -np.inf
        price = self.phones.price_inr.to_numpy()
        sim[np.abs(price / price[i] - 1) > price_band] = -np.inf
        top = np.argsort(-sim)[:k]
        top = [t for t in top if np.isfinite(sim[t])]
        out = self.phones.iloc[top].copy()
        out["similarity"] = sim[top]
        return out

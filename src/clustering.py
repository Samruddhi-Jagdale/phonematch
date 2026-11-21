"""Market segmentation with validation across three algorithms."""
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .config import DIM_LABELS, DIMS, SPEC_FEATURES

TIER_NAMES = {
    3: ["Budget", "Mid-range", "Premium"],
    4: ["Entry", "Mid-range", "Upper mid-range", "Flagship"],
    5: ["Entry", "Budget plus", "Mid-range", "Upper mid-range", "Flagship"],
    6: ["Entry", "Budget plus", "Mid-range", "Upper mid-range", "Premium", "Flagship"],
}


def cluster_matrix(phones):
    X = phones[SPEC_FEATURES].copy()
    X["log_price"] = np.log(phones.price_inr)
    return StandardScaler().fit_transform(X)


def _labels(algo, X, k):
    if algo == "K-Means":
        return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    if algo == "Hierarchical (Ward)":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    return GaussianMixture(n_components=k, random_state=0, n_init=3).fit(X).predict(X)


ALGOS = ["K-Means", "Hierarchical (Ward)", "Gaussian Mixture"]


def validate(X, ks=range(2, 9)):
    rows = []
    for algo in ALGOS:
        for k in ks:
            lab = _labels(algo, X, k)
            rows.append(dict(algorithm=algo, k=k,
                             silhouette=silhouette_score(X, lab),
                             davies_bouldin=davies_bouldin_score(X, lab),
                             calinski_harabasz=calinski_harabasz_score(X, lab)))
    return pd.DataFrame(rows)


def name_clusters(phones, labels):
    """Name each cluster by price rank plus the dimension it is strongest in."""
    d = phones.assign(_c=labels)
    order = d.groupby("_c").price_inr.mean().sort_values().index.tolist()
    k = len(order)
    tiers = TIER_NAMES.get(k) or [f"Segment {i + 1}" for i in range(k)]
    spec_cols = [f"spec_{x}" for x in DIMS]
    overall = d[spec_cols].mean()
    names = {}
    for rank, c in enumerate(order):
        lift = d[d._c == c][spec_cols].mean() - overall
        lead = DIM_LABELS[DIMS[int(np.argmax(lift.values))]].lower()
        names[c] = f"{tiers[rank]}, {lead}-led"
    return names


def fit_clusters(phones, k=None):
    X = cluster_matrix(phones)
    val = validate(X)
    if k is None:
        km = val[(val.algorithm == "K-Means") & (val.k.between(3, 6))]
        k = int(km.loc[km.silhouette.idxmax(), "k"])
    labels = _labels("K-Means", X, k)
    names = name_clusters(phones, labels)
    pca = PCA(2, random_state=0).fit(X)
    xy = pca.transform(X)
    return dict(k=k, labels=labels, names=names, xy=xy, validation=val,
                explained=pca.explained_variance_ratio_)

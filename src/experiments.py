"""All experiments reported in the project: value model, ablation, sentiment, clustering, recommender."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, mean_absolute_error, r2_score
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from .config import DIMS, FEATURE_GROUPS, FEATURE_LABELS, PERSONAS, SPEC_FEATURES
from .sentiment import labelled
from .value_model import candidate_models

CV = KFold(5, shuffle=True, random_state=0)


# 1) fair-price model ------------------------------------------------------------------------
def value_model_comparison(phones):
    X, y = phones[SPEC_FEATURES], np.log(phones.price_inr)
    rows = []
    for name, est in candidate_models().items():
        pred = np.exp(cross_val_predict(est, X, y, cv=CV))
        price = phones.price_inr
        rows.append({"Model": name, "R2": r2_score(price, pred), "MAE (INR)": mean_absolute_error(price, pred),
                     "MAPE (%)": float(np.mean(np.abs(pred - price) / price) * 100)})
    return pd.DataFrame(rows).sort_values("R2", ascending=False).round(3)


def tune_gradient_boosting(phones):
    grid = {"n_estimators": [150, 300], "max_depth": [2, 3, 4], "learning_rate": [0.03, 0.06]}
    gs = GridSearchCV(GradientBoostingRegressor(subsample=0.8, random_state=0), grid, cv=CV,
                      scoring="neg_mean_absolute_error", n_jobs=-1)
    gs.fit(phones[SPEC_FEATURES], np.log(phones.price_inr))
    best = {**gs.best_params_, "cv_mae_log": round(-gs.best_score_, 4)}
    return pd.DataFrame([best])


def ablation(phones):
    """Drop one feature group at a time and see how much cross-validated R2 falls."""
    y = np.log(phones.price_inr)
    est = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=0, n_jobs=-1)

    def score(cols):
        pred = np.exp(cross_val_predict(est, phones[cols], y, cv=CV))
        return r2_score(phones.price_inr, pred)

    full = score(SPEC_FEATURES)
    rows = [{"Removed group": "None (all features)", "R2": full, "R2 drop": 0.0}]
    for g, cols in FEATURE_GROUPS.items():
        keep = [c for c in SPEC_FEATURES if c not in cols]
        s = score(keep)
        rows.append({"Removed group": g, "R2": s, "R2 drop": full - s})
    return pd.DataFrame(rows).round(3)


def permutation_table(phones):
    X, y = phones[SPEC_FEATURES], np.log(phones.price_inr)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    est = GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, random_state=0).fit(Xtr, ytr)
    r = permutation_importance(est, Xte, yte, n_repeats=15, random_state=0)
    out = pd.DataFrame({"Feature": [FEATURE_LABELS[c] for c in SPEC_FEATURES],
                        "Importance": r.importances_mean, "Std": r.importances_std})
    return out.sort_values("Importance", ascending=False).round(4)


# 2) review sentiment -------------------------------------------------------------------------
def sentiment_comparison(reviews):
    X, y = labelled(reviews)
    models = {
        "TF-IDF + Logistic Regression": LogisticRegression(max_iter=1000, C=3.0),
        "TF-IDF + Naive Bayes": MultinomialNB(),
        "TF-IDF + Linear SVM": LinearSVC(),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    rows = []
    for name, clf in models.items():
        pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), clf)
        r = cross_validate(pipe, X, y, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
        rows.append({"Model": name, "Accuracy": r["test_accuracy"].mean(), "F1": r["test_f1"].mean(),
                     "ROC-AUC": r["test_roc_auc"].mean()})
    return pd.DataFrame(rows).round(3)


def sentiment_confusion(reviews):
    X, y = labelled(reviews)
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(max_iter=1000, C=3.0))
    pred = cross_val_predict(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    cm = confusion_matrix(y, pred)
    return pd.DataFrame(cm, index=["Actual negative", "Actual positive"], columns=["Predicted negative", "Predicted positive"])


# 3) recommender comparison on simulated personas ---------------------------------------------
METHODS = ["Top rated in budget (baseline)", "Specs only", "Specs + value", "Hybrid: specs + reviews + value"]


def recommender_comparison(phones, rec, k=5, trust=0.4):
    spec_cols = [f"spec_{d}" for d in DIMS]
    rev_cols = [f"rev_{d}" for d in DIMS]
    rows = []
    for method in METHODS:
        m = {"rating": [], "gap": [], "spec_fit": [], "rev_fit": [], "brands": []}
        for p in PERSONAS.values():
            lo, hi = p["budget"]
            w = dict(p["weights"])
            if method.startswith("Top rated"):
                sel = phones[phones.price_inr.between(lo, hi)].sort_values("rating", ascending=False).head(k)
            else:
                if method == "Specs only":
                    w["value"] = 0
                sel = rec.recommend(w, (lo, hi), trust_reviews=trust if method.startswith("Hybrid") else 0.0, k=k)
            wn = rec.normalise(p["weights"])
            m["rating"].append(sel.rating.mean())
            m["gap"].append(sel.value_gap.mean() * 100)
            m["spec_fit"].append(float((sel[spec_cols].to_numpy() @ wn).mean()) * 100)
            m["rev_fit"].append(float((sel[rev_cols].to_numpy() @ wn).mean()) * 100)
            m["brands"].append(sel.brand.nunique() / max(len(sel), 1))
        rows.append({"Method": method, "Avg store rating": np.mean(m["rating"]),
                     "Avg value gap (%)": np.mean(m["gap"]), "Spec fit (0-100)": np.mean(m["spec_fit"]),
                     "Review fit (0-100)": np.mean(m["rev_fit"]), "Brand diversity": np.mean(m["brands"])})
    return pd.DataFrame(rows).round(2)


def trust_sweep(phones, rec, k=5):
    """How the blend between specs and reviews changes the recommendations."""
    spec_cols = [f"spec_{d}" for d in DIMS]
    rev_cols = [f"rev_{d}" for d in DIMS]
    rows = []
    for trust in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        sf, rf = [], []
        for p in PERSONAS.values():
            sel = rec.recommend(p["weights"], p["budget"], trust_reviews=trust, k=k)
            wn = rec.normalise(p["weights"])
            sf.append(float((sel[spec_cols].to_numpy() @ wn).mean()) * 100)
            rf.append(float((sel[rev_cols].to_numpy() @ wn).mean()) * 100)
        rows.append({"Trust in reviews": trust, "Spec fit": np.mean(sf), "Review fit": np.mean(rf)})
    return pd.DataFrame(rows).round(2)

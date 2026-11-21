"""Runs every experiment and saves tables + figures into results/.   python run_experiments.py"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import RESULTS_DIR
from src.experiments import (ablation, permutation_table, recommender_comparison, sentiment_comparison,
                             sentiment_confusion, trust_sweep, tune_gradient_boosting,
                             value_model_comparison)
from src.pipeline import build


def save(df, name):
    df.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    print(f"\n=== {name} ===\n{df.to_string(index=False)}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    b = build()
    ph = b.phones
    print(f"Phones: {len(ph)}  Reviews: {len(b.reviews)}  Clusters (K-Means, chosen by silhouette): {b.clusters['k']}")

    save(value_model_comparison(ph), "value_model_comparison")
    save(tune_gradient_boosting(ph), "value_model_tuning")
    save(ablation(ph), "ablation")
    imp = permutation_table(ph)
    save(imp, "permutation_importance")
    save(sentiment_comparison(b.reviews), "sentiment_comparison")
    cm = sentiment_confusion(b.reviews).reset_index().rename(columns={"index": ""})
    save(cm, "sentiment_confusion")
    val = b.clusters["validation"].round(3)
    save(val, "clustering_validation")
    save(recommender_comparison(ph, b.recommender), "recommender_comparison")
    save(trust_sweep(ph, b.recommender), "trust_sweep")

    # figures for the report
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for algo, g in val.groupby("algorithm"):
        ax[0].plot(g.k, g.silhouette, marker="o", label=algo)
        ax[1].plot(g.k, g.davies_bouldin, marker="o", label=algo)
        ax[2].plot(g.k, g.calinski_harabasz, marker="o", label=algo)
    for a, t in zip(ax, ["Silhouette (higher better)", "Davies-Bouldin (lower better)", "Calinski-Harabasz (higher better)"]):
        a.set_title(t); a.set_xlabel("clusters (k)")
    ax[0].legend()
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "clustering_validation.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(ph.price_inr, ph.pred_price, c=ph.value_gap, cmap="RdYlGn", s=14)
    lim = [ph.price_inr.min(), ph.price_inr.max()]
    ax.plot(lim, lim, "k--", lw=1); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Actual price (INR)"); ax.set_ylabel("Predicted fair price (INR)")
    ax.set_title("Fair-price model (out-of-fold)")
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "price_vs_predicted.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(imp.Feature[::-1], imp.Importance[::-1], color="#2D55FF")
    ax.set_title("Permutation importance (held-out data)")
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "permutation_importance.png", dpi=150); plt.close()
    print("\nSaved CSV tables and PNG figures to results/")


if __name__ == "__main__":
    main()

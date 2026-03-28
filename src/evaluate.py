"""
Evaluation utilities: metrics, threshold tuning, and plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def get_fraud_probs(model, X_test):
    """Return probability of fraud (class=1) for any model in the suite."""
    return model.predict_proba(X_test)[:, 1]


def best_f1_threshold(y_true, probs):
    """Find the decision threshold that maximises F1 on the provided set."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = np.where(
        (precisions + recalls) == 0,
        0,
        2 * precisions * recalls / (precisions + recalls),
    )
    best_idx = np.argmax(f1_scores[:-1])  # thresholds has one fewer element
    return thresholds[best_idx], f1_scores[best_idx]


def evaluate_model(name: str, model, X_test, y_test, threshold: float = None):
    """
    Print PR-AUC, best-F1 threshold (if not provided), and classification report.
    Returns a dict of summary metrics.
    """
    probs = get_fraud_probs(model, X_test)
    pr_auc = average_precision_score(y_test, probs)

    if threshold is None:
        threshold, _ = best_f1_threshold(y_test, probs)

    preds = (probs >= threshold).astype(int)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  PR-AUC : {pr_auc:.4f}")
    print(f"  Threshold used: {threshold:.4f}")
    print(classification_report(y_test, preds, target_names=["Legit", "Fraud"]))

    return {
        "name": name,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "probs": probs,
        "preds": preds,
    }


def plot_pr_curves(results: list, y_test, save_path: str = None):
    """Overlay PR curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for r in results:
        precision, recall, _ = precision_recall_curve(y_test, r["probs"])
        ax.plot(recall, precision, label=f"{r['name']} (AUC={r['pr_auc']:.3f})")

    fraud_rate = y_test.mean()
    ax.axhline(y=fraud_rate, color="grey", linestyle="--", label=f"Random (prevalence={fraud_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved PR curve to {save_path}")
    plt.show()


def plot_confusion_matrices(results: list, y_test, save_path: str = None):
    """Plot confusion matrices side by side for all models."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        cm = confusion_matrix(y_test, r["preds"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(r["name"])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved confusion matrices to {save_path}")
    plt.show()

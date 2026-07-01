import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from . import config


def evaluate(y_true, y_pred, save: bool = True) -> dict:
    labels = sorted(set(y_true))
    metrics = {
        "n_samples": len(y_true),
        "cv_folds": config.CV_FOLDS,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "per_class": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }

    if save:
        with open(config.OUTPUTS_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, ax=ax, colorbar=False)
        ax.set_title("Root Cause Classification — Confusion Matrix (5-fold CV)")
        plt.tight_layout()
        plt.savefig(config.OUTPUTS_DIR / "confusion_matrix.png", dpi=130)
        plt.close(fig)

    return metrics


def print_report(y_true, y_pred):
    print(classification_report(y_true, y_pred, zero_division=0))

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import List, Dict, Any


def compute_classification_metrics(y_true: List[int], y_pred: List[int], target_names: List[str]) -> Dict[str, Any]:
    """
    Computes standard evaluation metrics: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
    """
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-class metrics
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(target_names))), zero_division=0
    )

    per_class = []
    for i, name in enumerate(target_names):
        per_class.append({
            "class_name": name,
            "precision": round(float(p_class[i]), 4),
            "recall": round(float(r_class[i]), 4),
            "f1_score": round(float(f1_class[i]), 4),
            "support": int(support_class[i])
        })

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": target_names
    }


def compare_baseline_vs_lstm(
    baseline_cat_preds: List[str],
    lstm_cat_preds: List[str],
    true_cat_labels: List[str],
    baseline_sent_preds: List[str],
    lstm_sent_preds: List[str],
    true_sent_labels: List[str]
) -> Dict[str, Any]:
    """
    Compares simpler rule-based baseline vs. LSTM Deep Learning model.
    """
    cat_base_acc = float(accuracy_score(true_cat_labels, baseline_cat_preds))
    cat_lstm_acc = float(accuracy_score(true_cat_labels, lstm_cat_preds))

    sent_base_acc = float(accuracy_score(true_sent_labels, baseline_sent_preds))
    sent_lstm_acc = float(accuracy_score(true_sent_labels, lstm_sent_preds))

    return {
        "category": {
            "baseline_accuracy": round(cat_base_acc, 4),
            "lstm_accuracy": round(cat_lstm_acc, 4),
            "improvement": round(cat_lstm_acc - cat_base_acc, 4)
        },
        "sentiment": {
            "baseline_accuracy": round(sent_base_acc, 4),
            "lstm_accuracy": round(sent_lstm_acc, 4),
            "improvement": round(sent_lstm_acc - sent_base_acc, 4)
        }
    }

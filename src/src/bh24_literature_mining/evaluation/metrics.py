import logging

import evaluate
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from transformers import EvalPrediction

logger = logging.getLogger(__name__)


def _token_roc_auc(predictions, labels):
    mask = labels != -100
    if not np.any(mask):
        return None

    logits_arr = predictions[mask]
    labels_arr = labels[mask]

    present = np.unique(labels_arr)
    if len(present) < 2:
        return None

    exp = np.exp(logits_arr - logits_arr.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)

    probs = probs[:, present]

    try:
        return float(
            roc_auc_score(
                labels_arr,
                probs,
                multi_class="ovr",
                labels=present
            )
        )
    except ValueError as e:
        logger.warning(f"ROC AUC failed: {e}")
        return None

def compute_metrics(p: EvalPrediction, id2label: dict) -> dict:
    predictions, labels = p
    roc_auc = _token_roc_auc(predictions, labels)

    pred_ids = np.argmax(predictions, axis=2)
    true_predictions = [
        [id2label[pred] for pred, label in zip(preds, labs) if label != -100]
        for preds, labs in zip(pred_ids, labels)
    ]
    true_labels = [
        [id2label[label] for pred, label in zip(preds, labs) if label != -100]
        for preds, labs in zip(pred_ids, labels)
    ]
    flat_preds = [item for sublist in true_predictions for item in sublist]
    flat_labels = [item for sublist in true_labels for item in sublist]
    logger.info(
        "Classification Report:\n%s",
        classification_report(flat_labels, flat_preds, digits=4),
    )
    metric = evaluate.load("seqeval")
    results = metric.compute(predictions=true_predictions, references=true_labels)
    out = {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
    if roc_auc is not None:
        out["roc_auc"] = roc_auc
    return out

import logging

import evaluate
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

logger = logging.getLogger(__name__)


def _token_roc_auc(predictions: np.ndarray, labels: np.ndarray, num_labels: int) -> float | None:
    flat_logits: list[np.ndarray] = []
    flat_labels: list[int] = []
    for preds_seq, labs_seq in zip(predictions, labels):
        for logit, lab in zip(preds_seq, labs_seq):
            if lab == -100:
                continue
            flat_logits.append(logit)
            flat_labels.append(lab)
    if not flat_labels:
        return None
    logits_arr = np.array(flat_logits)
    labels_arr = np.array(flat_labels)
    present = np.unique(labels_arr)
    if len(present) < 2:
        return None
    exp = logits_arr - logits_arr.max(axis=1, keepdims=True)
    probs = np.exp(exp) / np.exp(exp).sum(axis=1, keepdims=True)
    try:
        return float(roc_auc_score(
            labels_arr, probs, multi_class="ovr", labels=list(range(num_labels))
        ))
    except ValueError:
        return None


def compute_metrics(p: tuple, id2label: dict) -> dict:
    predictions, labels = p
    num_labels = len(id2label)
    roc_auc = _token_roc_auc(predictions, labels, num_labels)

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

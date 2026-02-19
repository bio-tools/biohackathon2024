import logging

import evaluate
import numpy as np
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)


def compute_metrics(p: tuple, id2label: dict) -> dict:
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [id2label[pred] for pred, label in zip(preds, labs) if label != -100]
        for preds, labs in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[label] for pred, label in zip(preds, labs) if label != -100]
        for preds, labs in zip(predictions, labels)
    ]
    flat_preds = [item for sublist in true_predictions for item in sublist]
    flat_labels = [item for sublist in true_labels for item in sublist]
    logger.info(
        "Classification Report:\n%s",
        classification_report(flat_labels, flat_preds, digits=4),
    )
    metric = evaluate.load("seqeval")
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

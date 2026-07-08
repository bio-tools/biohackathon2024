import evaluate
import numpy as np
from transformers import EvalPrediction


def compute_metrics(p: EvalPrediction, id2label: dict[int, str]) -> dict[str, float]:
    predictions, labels = p
    pred_ids = np.argmax(predictions, axis=2)
    true_predictions = [
        [id2label[pred] for pred, label in zip(preds, labs) if label != -100]
        for preds, labs in zip(pred_ids, labels)
    ]
    true_labels = [
        [id2label[label] for pred, label in zip(preds, labs) if label != -100]
        for preds, labs in zip(pred_ids, labels)
    ]
    metric = evaluate.load("seqeval")
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

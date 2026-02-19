import logging
import os
import shutil
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from sklearn.metrics import classification_report
from transformers import (
    BertConfig,
    BertForTokenClassification,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from bh24_literature_mining.data.dataset import get_label_list, get_token_ner_tags, load_iob_splits
from bh24_literature_mining.preprocessing.tokenization import tokenize_and_align_labels


def cleanup_checkpoints(
    output_dir: str, keep_last: bool = True, best_model_dir: str | None = None
) -> None:
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint"):
            if item_path != best_model_dir and not (keep_last and item_path == item_path):
                shutil.rmtree(item_path)


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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    p = Path(__file__).parent.resolve()
    model_checkpoint = "bioformers/bioformer-16L"
    data_dir = p / "data/IOB"
    model_save_path = p / "models"

    model_save_path.mkdir(parents=True, exist_ok=True)

    train_raw, dev_raw = load_iob_splits(data_dir)
    label_list = get_label_list(train_raw)
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}

    _, _, train_df = get_token_ner_tags(train_raw, label2id)
    _, _, dev_df = get_token_ner_tags(dev_raw, label2id)

    features = Features(
        {
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=label_list)),
        }
    )

    tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)

    ds = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, features=features),
            "validation": Dataset.from_pandas(dev_df, features=features),
        }
    )
    tokenized_ds = ds.map(
        lambda x: tokenize_and_align_labels(x, tokenizer), batched=True
    )

    config = BertConfig.from_pretrained(
        model_checkpoint,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        attn_implementation="sdpa",
    )
    config.hidden_dropout_prob = 0.2
    config.attention_probs_dropout_prob = 0.2
    model = BertForTokenClassification.from_pretrained(model_checkpoint, config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("Using device: %s", device)

    output_dir = model_save_path / "extra_annotations"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_accumulation_steps=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(p / "logs" / "extra_annotations"),
        bf16=torch.cuda.is_available(),
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        processing_class=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, id2label),
    )

    trainer.train()
    eval_results = trainer.evaluate()

    logger.info("F1: %.4f", eval_results["eval_f1"])
    logger.info("Precision: %.4f", eval_results["eval_precision"])
    logger.info("Recall: %.4f", eval_results["eval_recall"])
    logger.info("Accuracy: %.4f", eval_results["eval_accuracy"])

    cleanup_checkpoints(str(output_dir), keep_last=True)


if __name__ == "__main__":
    main()

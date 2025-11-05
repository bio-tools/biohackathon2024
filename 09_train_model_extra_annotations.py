import os
import shutil
import json
import torch
import pandas as pd
import numpy as np
import evaluate
from pathlib import Path
from transformers import (
    BertTokenizerFast,
    BertConfig,
    BertForTokenClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import classification_report
from datasets import Dataset, DatasetDict, ClassLabel, Features, Sequence, Value


def cleanup_checkpoints(
    output_dir, keep_last=True, best_model_dir=None, last_model_dir=None
):
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint"):
            if item_path != best_model_dir and (
                not keep_last or item_path != last_model_dir
            ):
                shutil.rmtree(item_path)


def convert_IOB_transformer(test_list, pattern):
    new_list, sub_list = [], []
    for i in test_list:
        if i != pattern:
            sub_list.append(i)
        else:
            new_list.append(sub_list)
            sub_list = []
    return new_list


def get_token_ner_tags(df_, label2id_):
    ner_tag_list_ = df_["ner_tags"].map(label2id_).fillna("###").tolist()
    token_list_ = df_["tokens"].tolist()
    token_list = convert_IOB_transformer(token_list_, pattern="")
    ner_tag_list = convert_IOB_transformer(ner_tag_list_, pattern="###")
    df = pd.DataFrame({"tokens": token_list, "ner_tags": ner_tag_list})
    return token_list, ner_tag_list, df


def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=True):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        max_length=512,
        truncation=True,
        padding="max_length",
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p, id2label):
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
    print(
        "\nClassification Report:\n",
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


def main():
    p = Path(__file__).parent.resolve()
    model_checkpoint = "/home/t.afanasyeva/biohackathon2024/models/checkpoint-14050"
    data_checkpoint = p / "data/IOB"
    model_save_path = p / "models"
    performance_log_path = p / "data/model_performance_log"

    data_checkpoint.mkdir(parents=True, exist_ok=True)
    model_save_path.mkdir(parents=True, exist_ok=True)
    performance_log_path.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(
        data_checkpoint / "train_IOB.tsv",
        sep="\t",
        names=["tokens", "ner_tags"],
        skip_blank_lines=False,
        na_filter=False,
    )
    dev = pd.read_csv(
        data_checkpoint / "dev_IOB.tsv",
        sep="\t",
        names=["tokens", "ner_tags"],
        skip_blank_lines=False,
        na_filter=False,
    )

    label_list = sorted(set(train["ner_tags"].dropna()) - {""})
    id2label = {i: l for i, l in enumerate(label_list)}
    label2id = {l: i for i, l in enumerate(label_list)}

    # Convert and tokenize
    _, _, train_df = get_token_ner_tags(train, label2id)
    _, _, dev_df = get_token_ner_tags(dev, label2id)

    features = Features(
        {
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=label_list)),
        }
    )
    ds = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, features=features),
            "validation": Dataset.from_pandas(dev_df, features=features),
        }
    )

    # Load tokenizer and model config
    tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)
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

    tokenized_ds = ds.map(
        lambda x: tokenize_and_align_labels(x, tokenizer), batched=True
    )

    training_args = TrainingArguments(
        output_dir=str(model_save_path),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=50,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_accumulation_steps=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(p / "logs"),
        bf16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, id2label),
    )

    trainer.train()
    eval_results = trainer.evaluate()

    # Save evaluation results
    with open(performance_log_path / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)


if __name__ == "__main__":
    main()

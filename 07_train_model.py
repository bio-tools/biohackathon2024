import os
import shutil
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
    pipeline,
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


def get_last_created_checkpoint(directory):
    folders = [
        d
        for d in Path(directory).iterdir()
        if d.is_dir() and d.name.startswith("checkpoint")
    ]
    return max(folders, key=os.path.getctime) if folders else None


def truncate_if_needed(sentence, tokenizer, max_length):
    tokens = tokenizer(sentence, return_tensors="pt", truncation=False)
    if tokens["input_ids"].shape[1] > max_length:
        return tokenizer.decode(
            tokens["input_ids"][0][: max_length - 1], skip_special_tokens=True
        )
    return sentence


def format_entities(entity_list, sentence):
    if not entity_list:
        return ""
    return "; ".join(
        f"{sentence[e['start']:e['end']]} ({e['entity_group']}) at {e['start']}-{e['end']}"
        for e in entity_list
    )


def main():
    p = Path(__file__).parent.resolve()
    model_checkpoint = "bioformers/bioformer-16L"
    data_checkpoint = p / "data/IOB"
    model_save_path = p / "models"
    predicted_output = p / "data/predicted"
    to_predict_path = p / "data/to_predict/250805_mentions_with_topics.csv"

    data_checkpoint.mkdir(parents=True, exist_ok=True)
    model_save_path.mkdir(parents=True, exist_ok=True)
    predicted_output.mkdir(parents=True, exist_ok=True)

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
    trainer.evaluate()

    last_checkpoint = get_last_created_checkpoint(model_save_path)
    classifier = pipeline(
        "ner",
        model=last_checkpoint,
        tokenizer=tokenizer,
        aggregation_strategy="max",
    )

    to_predict_df = pd.read_csv(to_predict_path)
    sentences = [
        truncate_if_needed(s, tokenizer, config.max_position_embeddings)
        for s in to_predict_df["Sentence"]
    ]

    results = []
    batch_size = 16
    for i in range(0, len(sentences), batch_size):
        try:
            batch = sentences[i : i + batch_size]
            results.extend(classifier(batch))
        except Exception as e:
            print(f"Batch {i} failed: {e}")
            results.extend([[] for _ in batch])

    pd.DataFrame(results).to_csv(predicted_output / "results.csv", index=False)

    to_predict_df["NER_Model_Found"] = [
        format_entities(r, s) for r, s in zip(results, sentences)
    ]

    pd.DataFrame(to_predict_df).to_csv(
        predicted_output / "to_annotate_with_results.csv", index=False
    )


if __name__ == "__main__":
    main()

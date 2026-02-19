import logging
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForTokenClassification,
    Trainer,
    TrainingArguments,
    pipeline,
)

from bh24_literature_mining.data.dataset import (
    build_hf_dataset,
    get_label_list,
    get_token_ner_tags,
    load_iob_splits,
)
from bh24_literature_mining.evaluation import compute_metrics
from bh24_literature_mining.preprocessing.tokenization import tokenize_and_align_labels
from bh24_literature_mining.training import (
    cleanup_checkpoints,
    get_last_created_checkpoint,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def truncate_if_needed(sentence: str, tokenizer, max_length: int) -> str:
    tokens = tokenizer(sentence, return_tensors="pt", truncation=False)
    if tokens["input_ids"].shape[1] > max_length:
        return tokenizer.decode(
            tokens["input_ids"][0][: max_length - 1], skip_special_tokens=True
        )
    return sentence


def format_entities(entity_list: list[dict], sentence: str) -> str:
    if not entity_list:
        return ""
    return "; ".join(
        f"{sentence[e['start']:e['end']]} ({e['entity_group']}) at {e['start']}-{e['end']}"
        for e in entity_list
    )


def main() -> None:
    p = Path(__file__).parent.resolve()
    model_checkpoint = "bioformers/bioformer-16L"
    data_dir = p / "data/IOB"
    model_save_path = p / "models"
    predicted_output = p / "data/predicted"
    to_predict_path = None

    model_save_path.mkdir(parents=True, exist_ok=True)
    predicted_output.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw = load_iob_splits(
        data_dir, train_file="train_IOB.tsv", dev_file="test_IOB.tsv"
    )

    label_list = get_label_list(train_raw)
    id2label = dict(enumerate(label_list))
    label2id = {label: i for i, label in enumerate(label_list)}

    _, _, train_df = get_token_ner_tags(train_raw, label2id)
    _, _, test_df = get_token_ner_tags(test_raw, label2id)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    ds = build_hf_dataset(train_df, test_df, label_list)
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
        logging_dir=str(p / "logs"),
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

    # # --- Inference ---
    # last_checkpoint = get_last_created_checkpoint(output_dir)
    # if last_checkpoint is None:
    #     logger.warning("No checkpoint found, skipping inference")
    #     return

    # classifier = pipeline(
    #     "ner",
    #     model=last_checkpoint,
    #     tokenizer=tokenizer,
    #     aggregation_strategy="max",
    # )

    # to_predict_df = pd.read_csv(to_predict_path)
    # sentences = [
    #     truncate_if_needed(s, tokenizer, config.max_position_embeddings)
    #     for s in to_predict_df["Sentence"]
    # ]

    # results = []
    # batch_size = 16
    # for i in range(0, len(sentences), batch_size):
    #     try:
    #         batch = sentences[i : i + batch_size]
    #         results.extend(classifier(batch))
    #     except Exception as e:
    #         logger.error("Batch %d failed: %s", i, e)
    #         results.extend([[] for _ in batch])

    # pd.DataFrame(results).to_csv(predicted_output / "results.csv", index=False)

    # to_predict_df["NER_Model_Found"] = [
    #     format_entities(r, s) for r, s in zip(results, sentences)
    # ]
    # to_predict_df.to_csv(predicted_output / "to_annotate_with_results.csv", index=False)


if __name__ == "__main__":
    main()
